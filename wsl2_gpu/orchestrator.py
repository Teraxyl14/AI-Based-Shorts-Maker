"""
Project Aether V3 — Subprocess-Isolated Pipeline Orchestrator

Replaces the V2 in-process threaded pipeline with subprocess-spawned VRAM
phase isolation. Each pipeline phase runs in a dedicated child process with
its own CUDA context, guaranteeing 100% VRAM reclamation between phases.

Phase 1 (ASR):               Spawned subprocess → Whisper INT8 + CTC Alignment
Phase 2 (Vision & Editorial): Spawned subprocess → Kornia SceneDet + YOLO Pose + VLM Editorial
Phase 3 (Render):            Spawned subprocess → OSQP Trajectory + GPU Atlas + NVDEC/NVENC
"""

import os
import sys
import json
import asyncio
import hashlib
import time
import logging
import gc
import multiprocessing as mp
from typing import Optional, List, Dict, Any

import nats
from nats.js.api import ConsumerConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.schemas import (
    PipelineState, WordTimestamp, SceneChange, SpatialEvent,
    CuttingManifest, ClipManifest, FusedTimeline, FusedTimelineEntry,
    VideoSegment, RenderJob
)
from shared.nats_subjects import (
    VIDEO_INGEST, METADATA_AUDIO, METADATA_VISION, METADATA_FUSED,
    VIDEO_RENDER, PIPELINE_STATUS
)
from wsl2_gpu.vram_orchestrator import (
    SubprocessPhaseRunner,
    run_phase1_asr_worker,
    _run_phase1_asr_worker,
    run_phase2_vlm_editorial_worker,
    run_phase3_render_worker,
)
from wsl2_gpu.extraction_memory import (
    get_excluded_intervals,
    is_interval_excluded,
)

logger = logging.getLogger("AetherOrchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Ensure multiprocessing uses spawn for CUDA isolation
try:
    mp.set_start_method("spawn", force=True)
    logger.info("multiprocessing start method set to 'spawn'")
except RuntimeError:
    pass  # Already set


class Orchestrator:
    """Project Aether V3 — Subprocess-Isolated Pipeline Orchestrator.

    Phase 1 (ASR):               Spawned subprocess (~3.5GB VRAM)
    Phase 2 (Vision & Editorial): Spawned subprocess (~8.5GB VRAM)
    Phase 3 (Render):            Spawned subprocess (~4.2GB VRAM)

    All phases execute strictly sequentially with process-level VRAM isolation.
    """

    def __init__(self, nats_url: str):
        self.nats_url = nats_url
        self.nc = None
        self.js = None
        self.orchestrator_start_time = time.time()

        # Subprocess phase runners
        self.phase1_runner = SubprocessPhaseRunner("phase1_asr")
        self.phase2_runner = SubprocessPhaseRunner("phase2_vlm_editorial")
        self.phase3_runner = SubprocessPhaseRunner("phase3_render")

        # Active serial job tracking
        self.active_state: Optional[PipelineState] = None
        self.active_msg = None  # Original video.ingest message — ACK at the very end
        self.current_job_start_time = None

    def _write_pipeline_status(self, stage: str, status: str = "processing", extra: Optional[Dict[str, Any]] = None):
        try:
            p_path = "/data/pipeline_status.json"
            now = time.time()
            payload = {
                "status": status,
                "current_stage": stage,
                "source_path": getattr(self.active_state, "source_path", None) if self.active_state else None,
                "output_dir": getattr(self.active_state, "output_dir", None) if self.active_state else None,
                "timestamp": now,
                "pipeline_start_time": getattr(self, "current_job_start_time", now),
                "stage_start_time": now,
                **(extra or {})
            }
            with open(p_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write pipeline status file: {e}")

    async def run(self):
        logger.info(f"Connecting to NATS at {self.nats_url}...")

        # Initialize RMM pool if available
        try:
            import rmm
            rmm.reinitialize(pool_allocator=True, initial_pool_size=4 * 1024**3, maximum_pool_size=6 * 1024**3)
            logger.info("RAPIDS Memory Manager (RMM) pool initialized.")
        except ImportError:
            logger.warning("rmm-cu12 not installed. Bounded RAPIDS pool skipped.")
        except Exception as e:
            logger.warning(f"RMM initialization skipped: {e}")

        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()

        # Ensure NATS JetStream stream exists with all subjects
        try:
            await self.js.add_stream(
                name="AETHER_PIPELINE",
                subjects=[VIDEO_INGEST, METADATA_AUDIO, METADATA_VISION,
                          METADATA_FUSED, VIDEO_RENDER, PIPELINE_STATUS]
            )
            logger.info("JetStream Stream 'AETHER_PIPELINE' verified.")
        except Exception as e:
            logger.info(f"Stream configuration info: {e}")

        # Durable pull subscription for video ingest
        self.ingest_sub = await self.js.pull_subscribe(
            subject=VIDEO_INGEST,
            durable="orchestrator_ingest_v3",
            config=ConsumerConfig(ack_wait=1800.0, max_ack_pending=4)
        )

        # Launch inline ingest watchdog as a background task
        watchdog_task = asyncio.create_task(self._run_ingest_watchdog())

        logger.info("Aether V3 Orchestrator initialized. Entering serial execution loop...")

        try:
            while True:
                if self.active_state is None:
                    # Clean up any defunct child processes reparented to PID 1
                    while True:
                        try:
                            wpid, _ = os.waitpid(-1, os.WNOHANG)
                            if wpid <= 0:
                                break
                        except (ChildProcessError, OSError):
                            break
                    await self._pull_next_ingest()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled, shutting down...")
        finally:
            watchdog_task.cancel()
            if self.nc:
                await self.nc.close()

    async def _run_ingest_watchdog(self):
        """Runs the ingest watchdog as an inline background task."""
        try:
            from wsl2_gpu.ingest_watchdog import main as ingest_watchdog_main
            logger.info("Starting inline ingest watchdog...")
            await ingest_watchdog_main()
        except asyncio.CancelledError:
            logger.info("Inline ingest watchdog stopped.")
        except Exception as e:
            logger.error(f"Ingest watchdog crashed: {e}", exc_info=True)

    # ─── Pull Ingest ─────────────────────────────────────────────────────────

    async def _pull_next_ingest(self):
        """Pulls the next video.ingest message and starts the subprocess pipeline."""
        try:
            logger.info("Waiting for next video.ingest message...")
            msgs = await self.ingest_sub.fetch(batch=1, timeout=2.0)
            if msgs:
                await self._start_pipeline_job(msgs[0])
        except nats.errors.TimeoutError:
            pass

    async def _start_pipeline_job(self, msg):
        """Orchestrates the full V3 subprocess-isolated pipeline for one video."""
        try:
            payload = json.loads(msg.data.decode("utf-8"))
            source_path = payload.get("source_path")
            target_count = int(payload.get("target_shorts_count") or os.environ.get("AETHER_TARGET_SHORTS", 4))
            os.environ["AETHER_TARGET_SHORTS"] = str(target_count)

            if not source_path or not os.path.exists(source_path):
                logger.error(f"Ingested file not found: {source_path}. Skipping.")
                await msg.ack()
                return

            video_name_hash = hashlib.md5(source_path.encode()).hexdigest()
            output_dir = f"/data/output_shorts/{video_name_hash}"
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"🚀 Starting V3 subprocess pipeline for: {source_path}")
            logger.info(f"Target Output Folder: {output_dir}")

            self.active_state = PipelineState(
                source_path=source_path,
                video_name_hash=video_name_hash,
                output_dir=output_dir,
                current_stage="phase1_asr"
            )
            self.active_msg = msg
            job_start_time = time.time()
            self.current_job_start_time = job_start_time
            self._write_pipeline_status("phase1_asr", "processing", {"video_name": os.path.basename(source_path)})

            # ═════════════════════════════════════════════════════════════════
            # PHASE 1: ASR Transcription (Spawned Subprocess)
            # ═════════════════════════════════════════════════════════════════
            logger.info("════════════ Phase 1: ASR Transcription [SUBPROCESS] ════════════")
            phase1_result = await asyncio.to_thread(
                self.phase1_runner.run_phase,
                run_phase1_asr_worker,
                (source_path, self.nats_url)
            )

            if phase1_result.get("status") != "success":
                raise RuntimeError(f"Phase 1 failed: {phase1_result.get('error', 'unknown')}")

            words_data = phase1_result.get("words", [])
            logger.info(f"Phase 1 complete: {len(words_data)} word timestamps")

            # Publish audio metadata to NATS for observability
            try:
                audio_payload = {
                    "source_path": source_path,
                    "words": words_data,
                    "timestamp": time.time()
                }
                await self.js.publish(METADATA_AUDIO, json.dumps(audio_payload, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                logger.warning(f"Could not publish audio metadata to NATS (non-fatal): {e}")

            # ═════════════════════════════════════════════════════════════════
            # PHASE 2: Vision + VLM Editorial (Spawned Subprocess)
            # ═════════════════════════════════════════════════════════════════
            logger.info("════════════ Phase 2: Vision & VLM Editorial [SUBPROCESS] ════════════")
            self.active_state.current_stage = "phase2_vlm_editorial"
            self._write_pipeline_status("phase2_vlm_editorial", "processing", {"words_count": len(words_data)})

            phase2_result = await asyncio.to_thread(
                self.phase2_runner.run_phase,
                run_phase2_vlm_editorial_worker,
                (source_path, words_data, self.nats_url)
            )

            if phase2_result.get("status") != "success":
                raise RuntimeError(f"Phase 2 failed: {phase2_result.get('error', 'unknown')}")

            scene_data = phase2_result.get("scene_changes", [])
            spatial_data = phase2_result.get("spatial_events", [])
            editorial_shorts = phase2_result.get("editorial_shorts", [])
            transcript_beats = phase2_result.get("transcript_beats", [])
            video_duration = phase2_result.get("video_duration", 300.0)

            logger.info(
                f"Phase 2 complete: {len(scene_data)} scenes, "
                f"{len(spatial_data)} spatial events, "
                f"{len(transcript_beats)} transcript beats, "
                f"{len(editorial_shorts)} editorial shorts"
            )

            # Publish vision metadata to NATS (observability, downsample if massive)
            try:
                sample_spatial = spatial_data
                if len(spatial_data) > 2000:
                    step = len(spatial_data) // 1000
                    sample_spatial = spatial_data[::max(1, step)]
                vision_payload = {
                    "source_path": source_path,
                    "video_duration": video_duration,
                    "scene_changes": scene_data,
                    "spatial_events": sample_spatial,
                    "timestamp": time.time()
                }
                await self.js.publish(METADATA_VISION, json.dumps(vision_payload).encode("utf-8"))
            except Exception as e:
                logger.warning(f"Could not publish vision metadata to NATS (non-fatal): {e}")

            if len(editorial_shorts) < target_count:
                needed = target_count - len(editorial_shorts)
                logger.warning(f"Only {len(editorial_shorts)}/{target_count} editorial shorts available. Topping up {needed} fallback shorts.")
                fallback_shorts = self._generate_fallback_shorts(video_duration, existing=editorial_shorts, needed=needed, source_path=source_path)
                editorial_shorts.extend(fallback_shorts)

            # ═════════════════════════════════════════════════════════════════
            # PHASE 3: Render (Spawned Subprocess)
            # ═════════════════════════════════════════════════════════════════
            logger.info("════════════ Phase 3: Render [SUBPROCESS] ════════════")
            self.active_state.current_stage = "phase3_render"
            self._write_pipeline_status("phase3_render", "processing", {"shorts_count": len(editorial_shorts)})

            render_manifest = {
                "editorial_shorts": editorial_shorts,
                "words": words_data,
                "spatial_events": spatial_data,
                "scene_changes": scene_data,
                "transcript_beats": transcript_beats,
            }

            # Durable artifact persistence: write Phase 1 & 2 outputs so downstream re-renders never lose data
            try:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "transcript_words.json"), "w", encoding="utf-8") as f:
                    json.dump(words_data, f)
                with open(os.path.join(output_dir, "spatial_events.json"), "w", encoding="utf-8") as f:
                    json.dump(spatial_data, f)
                with open(os.path.join(output_dir, "scene_changes.json"), "w", encoding="utf-8") as f:
                    json.dump(scene_data, f)
                logger.info(f"Persisted Phase 1 & 2 artifacts to {output_dir}")
            except Exception as e:
                logger.warning(f"Failed to persist intermediate artifacts: {e}")

            phase3_result = await asyncio.to_thread(
                self.phase3_runner.run_phase,
                run_phase3_render_worker,
                (source_path, render_manifest, output_dir)
            )

            if phase3_result.get("status") != "success":
                raise RuntimeError(f"Phase 3 failed: {phase3_result.get('error', 'unknown')}")

            render_results = phase3_result.get("render_results", [])
            successful = sum(1 for r in render_results if r.get("success"))
            failed = len(render_results) - successful

            for r in render_results:
                status_payload = {
                    "source_path": source_path,
                    "clip_index": r.get("clip_index"),
                    "status": "success" if r.get("success") else "failed",
                    "error": "; ".join(r.get("errors", []))
                }
                await self.js.publish(PIPELINE_STATUS, json.dumps(status_payload).encode("utf-8"))

            # ═════════════════════════════════════════════════════════════════
            # Finalization & Metadata Persistence
            # ═════════════════════════════════════════════════════════════════
            total_elapsed = time.time() - job_start_time
            logger.info(
                f"🎉 V3 Pipeline completed for: {source_path}\n"
                f"   Rendered {successful}/{len(render_results)} shorts "
                f"({failed} failed) in {total_elapsed:.1f}s\n"
                f"   Output: {output_dir}"
            )

            # Save execution metadata
            metadata = {
                "source_path": source_path,
                "output_dir": output_dir,
                "pipeline_version": "V3-subprocess",
                "total_elapsed_s": round(total_elapsed, 2),
                "editorial_shorts": editorial_shorts,
                "render_results": render_results,
                "phase_history": (
                    self.phase1_runner.history +
                    self.phase2_runner.history +
                    self.phase3_runner.history
                ),
                "words_count": len(words_data),
                "scene_changes_count": len(scene_data),
                "spatial_events_count": len(spatial_data),
            }

            metadata_file = os.path.join(output_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved execution metadata to: {metadata_file}")

            self._write_pipeline_status("completed", "completed", {
                "total_elapsed_s": round(total_elapsed, 2),
                "successful_shorts": successful,
                "failed_shorts": failed
            })

            await msg.ack()
            logger.info("ACKed video.ingest message.")

        except Exception as e:
            logger.error(f"Pipeline job failed: {e}", exc_info=True)
            self._write_pipeline_status("failed", "failed", {"error": str(e)})
            if self.active_msg:
                await self.active_msg.ack()
        finally:
            self.active_state = None
            self.active_msg = None

    def _generate_fallback_shorts(self, video_duration: float,
                                  existing: Optional[List[Dict[str, Any]]] = None,
                                  needed: int = 4,
                                  source_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generates fallback editorial shorts to guarantee at least `needed` shorts (50.0s-60.0s each),
        strictly avoiding previously extracted intervals for multi-batch continuity.
        """
        existing = existing or []
        shorts = []
        start = 60.0
        avail_end = max(start, video_duration - 1.0)
        target_dur = 55.0 if (avail_end - start) >= (needed * 55.0) else max(50.0, min(60.0, (avail_end - start) / float(max(1, needed))))
        slots = max(needed * 2, int((avail_end - start) / target_dur))
        step = (avail_end - start - target_dur) / max(1, slots - 1) if slots > 1 else 0.0

        excluded_intervals = get_excluded_intervals(source_path) if source_path else []

        def collides(s_s: float, s_e: float) -> bool:
            if excluded_intervals and is_interval_excluded(s_s, s_e, excluded_intervals):
                return True
            for ex in existing + shorts:
                if s_s < ex.get("end_time_seconds", 0) and s_e > ex.get("start_time_seconds", 0):
                    return True
            return False

        for step_idx in range(slots + 1):
            if len(shorts) >= needed:
                break
            s_start = start + step_idx * step
            s_end = s_start + target_dur
            if s_end > avail_end:
                s_end = avail_end
                s_start = s_end - target_dur
            if s_start < 60.0 or (s_end - s_start) < 50.0:
                continue
            if not collides(s_start, s_end):
                shorts.append({
                    "start_time_seconds": round(s_start, 2),
                    "duration_seconds": round(s_end - s_start, 2),
                    "end_time_seconds": round(s_end, 2),
                    "title": f"Fallback Short {len(existing) + len(shorts) + 1}",
                    "reasoning": "Auto-generated balanced fallback narrative segment",
                    "framing_mode": "SMART_PORTRAIT_PAN",
                })

        logger.info(f"Generated {len(shorts)} fallback shorts (needed {needed})")
        return shorts


if __name__ == "__main__":
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    orchestrator = Orchestrator(nats_url)
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")