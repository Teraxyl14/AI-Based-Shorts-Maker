import os
import sys
import json
import time
import asyncio
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.nats_subjects import (
    VIDEO_INGEST, METADATA_AUDIO, METADATA_VISION, METADATA_FUSED,
    VIDEO_RENDER, PIPELINE_STATUS
)

import nats
from nats.js import JetStreamContext

logger = logging.getLogger("AetherIngestWatchdog")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

WATCH_DIR = "/data/input_videos"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv"}


# ─── Hybrid Observer Strategy ────────────────────────────────────────────────
# Primary: native InotifyObserver (captures IN_CLOSE_WRITE on Linux)
# Fallback: PollingObserver with 1s interval + size-stability checks
#           (required for Windows NTFS → WSL2 Docker bind-mount boundary)

def _create_observer():
    """Attempts native inotify observer first, falls back to polling."""
    try:
        from watchdog.observers import Observer
        obs = Observer()
        logger.info("Using native InotifyObserver (IN_CLOSE_WRITE capable).")
        return obs, "inotify"
    except Exception:
        pass

    from watchdog.observers.polling import PollingObserver
    obs = PollingObserver(timeout=1.0)
    logger.info("Falling back to PollingObserver (1s interval) for NTFS bind-mount compatibility.")
    return obs, "polling"


# ─── File System Event Handler ───────────────────────────────────────────────

from watchdog.events import FileSystemEventHandler, FileSystemEvent


class IngestHandler(FileSystemEventHandler):
    """Hybrid handler: uses IN_CLOSE_WRITE when available, size-stability polling otherwise."""

    def __init__(self, loop: asyncio.AbstractEventLoop, nc, js: JetStreamContext, mode: str):
        self.loop = loop
        self.nc = nc
        self.js = js
        self.mode = mode  # "inotify" or "polling"
        self.active_copies: set = set()
        self.published_files: set = set()  # Dedup guard

    def _is_video(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        return ext in ALLOWED_EXTENSIONS

    # ── Primary: IN_CLOSE_WRITE (inotify mode) ──────────────────────────────

    def on_closed(self, event: FileSystemEvent) -> None:
        """Fires on IN_CLOSE_WRITE on Linux — file write is fully complete."""
        if event.is_directory:
            return
        filepath = event.src_path
        if not self._is_video(filepath):
            return
        if filepath in self.published_files:
            return

        logger.info(f"IN_CLOSE_WRITE received: {filepath}. Publishing immediately.")
        self.published_files.add(filepath)
        asyncio.run_coroutine_threadsafe(self.publish_ingest_event(filepath), self.loop)

    # ── Fallback: on_created triggers size-stability polling ─────────────────

    def on_created(self, event: FileSystemEvent) -> None:
        """Fires on file creation. If in polling mode or inotify didn't fire on_closed,
        falls back to size-stability verification."""
        if event.is_directory:
            return
        filepath = event.src_path
        if not self._is_video(filepath):
            return
        if filepath in self.active_copies or filepath in self.published_files:
            return

        if self.mode == "polling":
            # PollingObserver: always use size-stability since no IN_CLOSE_WRITE
            logger.info(f"New file detected (polling): {filepath}. Verifying size stability...")
            self.active_copies.add(filepath)
            asyncio.run_coroutine_threadsafe(self.verify_and_publish(filepath), self.loop)
        else:
            # Inotify mode: schedule a delayed check — if on_closed fires first, this no-ops
            logger.info(f"New file detected (inotify): {filepath}. Waiting for IN_CLOSE_WRITE (5s fallback)...")
            self.active_copies.add(filepath)
            asyncio.run_coroutine_threadsafe(self._inotify_fallback(filepath), self.loop)

    async def _inotify_fallback(self, filepath: str) -> None:
        """If inotify doesn't fire on_closed within 5s (NTFS bind-mount), fall back to polling."""
        try:
            await asyncio.sleep(5.0)
            if filepath in self.published_files:
                return  # on_closed already handled it
            logger.warning(f"No IN_CLOSE_WRITE received for {filepath} after 5s. "
                          f"Falling back to size-stability polling (NTFS bind-mount suspected).")
            await self.verify_and_publish(filepath)
        except Exception as e:
            logger.error(f"Inotify fallback error for {filepath}: {e}")
        finally:
            self.active_copies.discard(filepath)

    async def verify_and_publish(self, filepath: str) -> None:
        """Polls file size until stable for 2 consecutive 1s checks, then publishes."""
        try:
            last_size = -1
            stable_count = 0
            while stable_count < 2:
                await asyncio.sleep(1.0)
                if not os.path.exists(filepath):
                    return  # File deleted mid-copy
                current_size = os.path.getsize(filepath)
                if current_size == last_size and current_size > 0:
                    stable_count += 1
                else:
                    last_size = current_size
                    stable_count = 0

            if filepath not in self.published_files:
                self.published_files.add(filepath)
                await self.publish_ingest_event(filepath)
        except Exception as e:
            logger.error(f"Error verifying copy stability for {filepath}: {e}")
        finally:
            self.active_copies.discard(filepath)

    # ── Move events ──────────────────────────────────────────────────────────

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handles mv/rename into the watch folder."""
        if event.is_directory:
            return
        filepath = event.dest_path
        if not self._is_video(filepath):
            return
        if filepath in self.published_files:
            return

        logger.info(f"File moved in: {filepath}. Publishing immediately.")
        self.published_files.add(filepath)
        asyncio.run_coroutine_threadsafe(self.publish_ingest_event(filepath), self.loop)

    # ── NATS publish ─────────────────────────────────────────────────────────

    async def publish_ingest_event(self, filepath: str) -> None:
        try:
            stat = os.stat(filepath)
            payload = {
                "source_path": filepath,
                "file_size": stat.st_size,
                "timestamp": time.time()
            }

            logger.info(f"Publishing ingest event to NATS for {filepath}...")
            ack = await self.js.publish(VIDEO_INGEST, json.dumps(payload).encode("utf-8"))
            logger.info(f"NATS Publish Success: stream={ack.stream}, sequence={ack.seq}")
        except Exception as e:
            logger.error(f"Failed to publish ingest event for {filepath}: {str(e)}", exc_info=True)
            self.published_files.discard(filepath)  # Allow retry


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    os.makedirs(WATCH_DIR, exist_ok=True)

    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    logger.info(f"Connecting to NATS at {nats_url}...")

    nc = await nats.connect(nats_url)
    js = nc.jetstream()

    # Ensure AETHER_PIPELINE stream exists
    try:
        await js.add_stream(
            name="AETHER_PIPELINE",
            subjects=[VIDEO_INGEST, METADATA_AUDIO, METADATA_VISION,
                      METADATA_FUSED, VIDEO_RENDER, PIPELINE_STATUS]
        )
        logger.info("JetStream Stream 'AETHER_PIPELINE' initialized successfully.")
    except Exception as e:
        logger.info(f"Stream may already exist or info: {e}")

    # Create hybrid observer
    observer, mode = _create_observer()
    loop = asyncio.get_running_loop()
    event_handler = IngestHandler(loop, nc, js, mode)

    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)
    observer.start()
    logger.info(f"Started file watchdog on directory: {WATCH_DIR} (mode={mode})")

    try:
        while True:
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Stopping watchdog observer...")
        observer.stop()
    finally:
        observer.join()
        await nc.close()
        logger.info("NATS connection closed. Exit.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
