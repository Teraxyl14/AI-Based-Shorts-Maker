import os
import sys
import json
import asyncio
import logging
import time
import httpx
import nats
from nats.js.api import ConsumerConfig
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.schemas import (
    WordTimestamp, SceneChange, SpatialEvent,
    FusedTimelineEntry, FusedTimeline
)
from shared.nats_subjects import METADATA_AUDIO, METADATA_VISION, METADATA_FUSED

logger = logging.getLogger("AetherFusionConsumer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Proximity windows for merging events to transcript words
SCENE_CHANGE_WINDOW = 1.0   # seconds: scene changes within ±1s of a word
POSE_WINDOW = 0.5           # seconds: pose events within ±0.5s of a word

BRAIN_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"


class FusionConsumer:
    """Phase 2 Sequential Fusion Join: merges transcription and vision pre-pass streams,
    then feeds the unified text-annotated timeline to the local MoE brain for hook analysis.

    Waits for both metadata.audio and metadata.vision for the same source_path before
    producing a fused metadata.fused output.
    """

    def __init__(self, nats_url: str):
        self.nats_url = nats_url
        self.nc = None
        self.js = None
        self.brain_url = os.getenv("VLLM_BRAIN_URL", "http://vllm-brain:8000").rstrip("/")

        # Pending partial results keyed by source_path
        self.pending_audio: dict = {}   # source_path -> payload
        self.pending_vision: dict = {}  # source_path -> payload

    async def run(self):
        """Connects to NATS and consumes both metadata streams."""
        logger.info(f"Connecting to NATS at {self.nats_url}...")
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()

        # Durable pull subscriptions with hardened max_ack_pending
        self.audio_sub = await self.js.pull_subscribe(
            subject=METADATA_AUDIO,
            durable="fusion_audio",
            config=ConsumerConfig(ack_wait=600.0, max_ack_pending=12)
        )
        self.vision_sub = await self.js.pull_subscribe(
            subject=METADATA_VISION,
            durable="fusion_vision",
            config=ConsumerConfig(ack_wait=600.0, max_ack_pending=12)
        )

        logger.info("Fusion Consumer initialized. Polling for audio and vision metadata...")

        try:
            while True:
                # Poll both streams
                await self._poll_audio()
                await self._poll_vision()

                # Check for any complete pairs
                ready_paths = set(self.pending_audio.keys()) & set(self.pending_vision.keys())
                for source_path in ready_paths:
                    await self._fuse_and_publish(source_path)

                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("Fusion Consumer cancelled, shutting down...")
        finally:
            if self.nc:
                await self.nc.close()

    async def _poll_audio(self):
        """Fetches pending audio metadata messages."""
        try:
            msgs = await self.audio_sub.fetch(batch=1, timeout=1.0)
            for msg in msgs:
                payload = json.loads(msg.data.decode("utf-8"))
                source_path = payload.get("source_path")
                logger.info(f"Received audio metadata for: {source_path}")
                self.pending_audio[source_path] = payload
                await msg.ack()
        except nats.errors.TimeoutError:
            pass

    async def _poll_vision(self):
        """Fetches pending vision metadata messages."""
        try:
            msgs = await self.vision_sub.fetch(batch=1, timeout=1.0)
            for msg in msgs:
                payload = json.loads(msg.data.decode("utf-8"))
                source_path = payload.get("source_path")
                logger.info(f"Received vision metadata for: {source_path}")
                self.pending_vision[source_path] = payload
                await msg.ack()
        except nats.errors.TimeoutError:
            pass

    async def _fuse_and_publish(self, source_path: str):
        """Merges audio + vision data, runs LLM hook analysis, publishes fused timeline."""
        logger.info(f"🔀 Fusing audio + vision metadata for: {source_path}")
        t0 = time.perf_counter()

        audio_data = self.pending_audio.pop(source_path)
        vision_data = self.pending_vision.pop(source_path)

        try:
            # Parse raw data into models
            words = [WordTimestamp(**w) for w in audio_data.get("words", [])]
            scene_changes = [SceneChange(**sc) for sc in vision_data.get("scene_changes", [])]
            spatial_events = [SpatialEvent(**pe) for pe in vision_data.get("spatial_events", [])]
            video_duration = float(vision_data.get("video_duration", 300.0))

            # ── Step 1: Merge into unified timeline entries ──────────────────
            entries = []
            for word in words:
                word_mid = (word.start + word.end) / 2.0

                # Find scene changes within ±SCENE_CHANGE_WINDOW of this word
                nearby_scenes = [
                    sc for sc in scene_changes
                    if abs(sc.timestamp - word_mid) <= SCENE_CHANGE_WINDOW
                ]

                # Find pose events within ±POSE_WINDOW of this word
                nearby_poses = [
                    pe for pe in spatial_events
                    if abs(pe.timestamp - word_mid) <= POSE_WINDOW
                ]

                entries.append(FusedTimelineEntry(
                    timestamp=word.start,
                    word=word.word,
                    word_end=word.end,
                    word_confidence=word.confidence,
                    scene_changes_nearby=nearby_scenes,
                    spatial_events_nearby=nearby_poses
                ))

            logger.info(f"Merged {len(words)} words with {len(scene_changes)} scene changes "
                       f"and {len(spatial_events)} pose events into {len(entries)} timeline entries.")

            # ── Step 2: LLM hook analysis on text-annotated visual timeline ──
            await self._run_hook_analysis(entries, scene_changes, spatial_events)

            # Clear heavy keypoint coordinate data before NATS serialization to prevent MaxPayloadError
            for pe in spatial_events:
                pe.keypoints = []
            for entry in entries:
                for pe in entry.spatial_events_nearby:
                    pe.keypoints = []

            # ── Step 3: Publish fused timeline to NATS ───────────────────────
            fused = FusedTimeline(
                source_path=source_path,
                video_duration=video_duration,
                entries=entries,
                scene_changes=scene_changes,
                spatial_events=spatial_events,
                transcript_words=words
            )

            fused_payload = {
                "source_path": source_path,
                "fused_timeline": fused.model_dump(),
                "timestamp": time.time()
            }

            await self.js.publish(METADATA_FUSED, json.dumps(fused_payload).encode("utf-8"))
            elapsed = time.perf_counter() - t0
            logger.info(f"✅ Published fused timeline to '{METADATA_FUSED}' in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Fusion failed for {source_path}: {e}", exc_info=True)
            # Re-publish a minimal fused timeline so the pipeline doesn't stall
            try:
                words = [WordTimestamp(**w) for w in audio_data.get("words", [])]
                minimal_fused = FusedTimeline(
                    source_path=source_path,
                    video_duration=float(vision_data.get("video_duration", 300.0)),
                    entries=[FusedTimelineEntry(
                        timestamp=w.start, word=w.word, word_end=w.end,
                        word_confidence=w.confidence
                    ) for w in words],
                    transcript_words=words
                )
                fallback_payload = {
                    "source_path": source_path,
                    "fused_timeline": minimal_fused.model_dump(),
                    "timestamp": time.time()
                }
                await self.js.publish(METADATA_FUSED, json.dumps(fallback_payload).encode("utf-8"))
                logger.warning("Published minimal fallback fused timeline (without LLM hook analysis).")
            except Exception as e2:
                logger.error(f"Even fallback fusion publish failed: {e2}")

    async def _run_hook_analysis(self, entries: List[FusedTimelineEntry],
                                  scene_changes: List[SceneChange],
                                  spatial_events: List[SpatialEvent]):
        """Sends windowed timeline segments to the unified MoE brain for hook scoring."""
        logger.info("Running LLM hook analysis on fused timeline segments...")

        # Group entries into 30-second windows for analysis
        window_size = 30.0
        if not entries:
            return

        total_duration = entries[-1].timestamp if entries else 0
        windows = []
        current_window_start = 0.0

        while current_window_start < total_duration:
            window_end = current_window_start + window_size
            window_entries = [e for e in entries
                            if e.timestamp >= current_window_start and e.timestamp < window_end]

            if window_entries:
                # Build context string for this window
                text = " ".join(e.word for e in window_entries if e.word)
                scene_count = sum(1 for e in window_entries for _ in e.scene_changes_nearby)
                gesture_summary = []
                for e in window_entries:
                    for p in e.spatial_events_nearby:
                        if p.action_label and p.action_label != "standing":
                            gesture_summary.append(f"{p.action_label}@{p.timestamp:.1f}s")

                windows.append({
                    "start": current_window_start,
                    "end": window_end,
                    "text": text,
                    "scene_changes": scene_count,
                    "gestures": gesture_summary[:5],  # Limit to avoid token overflow
                    "entries": window_entries
                })

            current_window_start += window_size

        # Analyze each window via the unified brain
        async with httpx.AsyncClient() as client:
            for window in windows:
                try:
                    prompt = (
                        f"Analyze this video segment ({window['start']:.0f}s-{window['end']:.0f}s) "
                        f"for viral short-form potential:\n\n"
                        f"Transcript: \"{window['text']}\"\n"
                        f"Scene changes in window: {window['scene_changes']}\n"
                        f"Notable body language: {', '.join(window['gestures']) if window['gestures'] else 'none detected'}\n\n"
                        f"Rate this segment's hook potential from 0.0 to 1.0 and explain why. "
                        f"Respond with JSON: {{\"arousal_score\": <float>, \"hook_analysis\": \"<reason>\"}}"
                    )

                    payload = {
                        "model": BRAIN_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a viral content analyst. Evaluate video segments for short-form engagement potential based on transcript content, scene dynamics, and body language cues."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 200
                    }

                    response = await client.post(
                        f"{self.brain_url}/v1/chat/completions",
                        json=payload,
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        # Extract JSON
                        if "{" in content:
                            content = content[content.find("{"):content.rfind("}") + 1]
                        parsed = json.loads(content)
                        arousal = float(parsed.get("arousal_score", 0.5))
                        analysis = str(parsed.get("hook_analysis", ""))

                        # Apply scores to the window's entries
                        for entry in window["entries"]:
                            entry.arousal_score = arousal
                            entry.hook_analysis = analysis

                        logger.info(f"Window {window['start']:.0f}-{window['end']:.0f}s: arousal={arousal:.2f}")
                    else:
                        logger.warning(f"Brain returned {response.status_code} for window "
                                      f"{window['start']:.0f}-{window['end']:.0f}s")

                except Exception as e:
                    logger.warning(f"Hook analysis failed for window "
                                  f"{window['start']:.0f}-{window['end']:.0f}s: {e}")


if __name__ == "__main__":
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    consumer = FusionConsumer(nats_url)
    try:
        asyncio.run(consumer.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
