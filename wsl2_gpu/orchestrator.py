import os
import sys
import json
import asyncio
import hashlib
import time
import logging
import subprocess
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
from wsl2_gpu.debate_graph import DebateGraph
from wsl2_gpu.fusion_consumer import FusionConsumer
from wsl2_gpu.ingest_watchdog import main as ingest_watchdog_main
from wsl2_gpu.transcription_spoke import TranscriptionSpoke
from wsl2_gpu.vision_spoke import VisionSpoke
from wsl2_gpu.trajectory_smoother import compute_crop_path
from wsl2_gpu.vram_orchestrator import VRAMOrchestrator
from wsl2_gpu.render_spoke import ThreeThreadRenderPipeline

logger = logging.getLogger("AetherOrchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class Orchestrator:
    """Project Aether V2 — Sequential In-Process Pipeline Orchestrator.

    Phase 1 (ASR):               Whisper-large-v3 INT8 + CTC Alignment (managed by VRAMOrchestrator)
    Phase 2 (Vision & Editorial): Kornia scene detection + YOLO pose tracking + MoE Editorial debate
    Phase 3 (Render):            Global Trajectory Optimization + Premultiplied Subtitle Compositing

    All phases execute strictly sequentially with explicit VRAM lifecycle management.
    """

    def __init__(self, nats_url: str):
        self.nats_url = nats_url
        self.nc = None
        self.js = None
        self.orchestrator_start_time = time.time()

        # Sequential VRAM lifecycle manager
        self.vram = VRAMOrchestrator()

        # Active serial job tracking
        self.active_state: Optional[PipelineState] = None
        self.active_msg = None  # Original video.ingest message — ACK at the very end
        self.debate_graph = DebateGraph()

        # Multi-clip render tracking
        self.pending_renders = 0
        self.completed_renders = 0
        self.render_errors = []

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
            durable="orchestrator_ingest_v2",
            config=ConsumerConfig(ack_wait=900.0, max_ack_pending=12)
        )

        # Launch inline ingest watchdog as a background task
        watchdog_task = asyncio.create_task(self._run_ingest_watchdog())

        logger.info("Aether V2 Orchestrator initialized. Entering serial execution loop...")

        try:
            while True:
                if self.active_state is None:
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
            logger.info("Starting inline ingest watchdog...")
            await ingest_watchdog_main()
        except asyncio.CancelledError:
            logger.info("Inline ingest watchdog stopped.")
        except Exception as e:
            logger.error(f"Ingest watchdog crashed: {e}", exc_info=True)

    # ─── Pull Ingest ─────────────────────────────────────────────────────────

    async def _pull_next_ingest(self):
        """Pulls the next video.ingest message and starts the sequential pipeline."""
        try:
            logger.info("Waiting for next video.ingest message...")
            msgs = await self.ingest_sub.fetch(batch=1, timeout=2.0)
            if msgs:
                await self._start_pipeline_job(msgs[0])
        except nats.errors.TimeoutError:
            pass

    async def _start_pipeline_job(self, msg):
        """Initializes state and orchestrates the full sequential V2 pipeline for one video."""
        try:
            payload = json.loads(msg.data.decode("utf-8"))
            source_path = payload.get("source_path")

            if not source_path or not os.path.exists(source_path):
                logger.error(f"Ingested file not found: {source_path}. Skipping.")
                await msg.ack()
                return

            video_name_hash = hashlib.md5(source_path.encode()).hexdigest()
            output_dir = f"/data/output_shorts/{video_name_hash}"
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"🚀 Starting V2 sequential pipeline for: {source_path}")
            logger.info(f"Target Output Folder: {output_dir}")

            self.active_state = PipelineState(
                source_path=source_path,
                video_name_hash=video_name_hash,
                output_dir=output_dir,
                current_stage="phase1_asr"
            )
            self.active_msg = msg
            self.active_job_start_time = time.time()
            self.pending_renders = 0
            self.completed_renders = 0
            self.render_errors = []

            # ═════════════════════════════════════════════════════════════════
            # PHASE 1: ASR Transcription (Sequential VRAM Lifecycle)
            # ═════════════════════════════════════════════════════════════════
            logger.info("════════════ Phase 1: ASR Transcription ════════════")
            words: List[WordTimestamp] = []
            with self.vram.phase('asr') as ctx:
                spoke = TranscriptionSpoke(self.nats_url)
                words = await spoke.transcribe_file(source_path)
                ctx.register(spoke)

            self.active_state.transcript_words = words
            logger.info(f"Phase 1 complete: Generated {len(words)} word timestamps.")

            # Publish audio metadata to NATS for observability
            audio_payload = {
                "source_path": source_path,
                "words": [w.model_dump() for w in words],
                "timestamp": time.time()
            }
            await self.js.publish(METADATA_AUDIO, json.dumps(audio_payload, ensure_ascii=False).encode("utf-8"))

            # ═════════════════════════════════════════════════════════════════
            # PHASE 2: Vision Pre-Pass + Fusion + Editorial Debate
            # ═════════════════════════════════════════════════════════════════
            logger.info("════════════ Phase 2: Vision & Editorial Debate ════════════")
            self.active_state.current_stage = "phase2_vision_editorial"
            scene_changes: List[SceneChange] = []
            spatial_events: List[SpatialEvent] = []
            video_duration = 300.0

            with self.vram.phase('vision_editorial') as ctx:
                vision_spoke = VisionSpoke(self.nats_url)
                await asyncio.to_thread(vision_spoke._load_models)
                duration, fps, width, height = await asyncio.to_thread(vision_spoke._get_video_info, source_path)
                video_duration = duration

                logger.info(f"Running GPU scene detection and 3-tier spatial tracking for {duration:.1f}s video...")
                scene_changes = await asyncio.to_thread(vision_spoke.detect_scene_changes, source_path, duration, fps)
                spatial_events = await asyncio.to_thread(vision_spoke.detect_spatial_events, source_path, duration, fps)
                ctx.register(vision_spoke)

                # Fuse audio + vision metadata into FusedTimeline
                fusion_consumer = FusionConsumer(self.nats_url)
                entries = []
                for w in words:
                    w_mid = (w.start + w.end) / 2.0
                    nearby_sc = [sc for sc in scene_changes if abs(sc.timestamp - w_mid) <= 1.0]
                    nearby_pe = [pe for pe in spatial_events if abs(pe.timestamp - w_mid) <= 0.5]
                    entries.append(FusedTimelineEntry(
                        timestamp=w.start,
                        word=w.word,
                        word_end=w.end,
                        word_confidence=w.confidence,
                        scene_changes_nearby=nearby_sc,
                        spatial_events_nearby=nearby_pe
                    ))

                # Run LLM hook analysis
                await fusion_consumer._run_hook_analysis(entries, scene_changes, spatial_events)
                ctx.register(fusion_consumer)

                # Clean keypoint data for lighter storage
                for pe in spatial_events:
                    pe.keypoints = []
                for entry in entries:
                    for pe in entry.spatial_events_nearby:
                        pe.keypoints = []

                fused_timeline = FusedTimeline(
                    source_path=source_path,
                    video_duration=video_duration,
                    entries=entries,
                    scene_changes=scene_changes,
                    spatial_events=spatial_events,
                    transcript_words=words
                )
                self.active_state.fused_timeline = fused_timeline

                # Publish vision and fused metadata
                vision_payload = {
                    "source_path": source_path,
                    "video_duration": duration,
                    "scene_changes": [sc.model_dump() for sc in scene_changes],
                    "spatial_events": [pe.model_dump() for pe in spatial_events],
                    "timestamp": time.time()
                }
                await self.js.publish(METADATA_VISION, json.dumps(vision_payload).encode("utf-8"))

                fused_payload = {
                    "source_path": source_path,
                    "fused_timeline": fused_timeline.model_dump(),
                    "timestamp": time.time()
                }
                await self.js.publish(METADATA_FUSED, json.dumps(fused_payload).encode("utf-8"))

                # Run LangGraph Editor-Director Debate Loop
                logger.info("Initiating LangGraph Editor↔Director debate...")
                compiled_graph = self.debate_graph.build_graph()
                result = await compiled_graph.ainvoke(self.active_state.model_dump())

                manifest_data = result.get("cutting_manifest") if isinstance(result, dict) else getattr(result, "cutting_manifest", None)
                if manifest_data:
                    if isinstance(manifest_data, dict):
                        self.active_state.cutting_manifest = CuttingManifest(**manifest_data)
                    else:
                        self.active_state.cutting_manifest = manifest_data
                    num_clips = len(self.active_state.cutting_manifest.clips)
                    logger.info(f"Debate consensus achieved. Cutting manifest contains {num_clips} candidate clips.")
                else:
                    raise RuntimeError("Debate finished without creating a cutting manifest.")

            # ═════════════════════════════════════════════════════════════════
            # PHASE 3: Global Trajectory Optimization + GPU Render
            # ═════════════════════════════════════════════════════════════════
            logger.info("════════════ Phase 3: Trajectory Optimization & Rendering ════════════")
            self.active_state.current_stage = "phase3_render"

            clips = self.active_state.cutting_manifest.clips
            if not clips or len(clips) < 3:
                logger.warning("Fewer than 3 clips in cutting manifest. Ensuring 3 candidate clips of >= 50s.")
                existing_clips = list(clips) if clips else []
                offsets = [0.0, 55.0, 110.0]
                while len(existing_clips) < 3:
                    idx_needed = len(existing_clips)
                    start_t = offsets[idx_needed] if idx_needed < len(offsets) else (idx_needed * 55.0)
                    existing_clips.append(ClipManifest(
                        segments=[VideoSegment(segment_start=start_t, segment_end=start_t + 52.0)],
                        hook_score=0.75, retention_score=0.75,
                        cta_present=True,
                        reasoning=f"Fallback clip {idx_needed + 1} (52s) generated.",
                        caption_text=f"Short {idx_needed + 1}"
                    ))
                clips = existing_clips
                self.active_state.cutting_manifest.clips = clips

            self.pending_renders = len(clips)
            self.completed_renders = 0
            self.render_errors = []

            transcript_words_data = [w.model_dump() for w in words]
            spatial_events_data = [pe.model_dump() for pe in spatial_events]
            scene_cut_frames = [sc.frame_idx for sc in scene_changes]

            with self.vram.phase('render') as ctx:
                for idx, clip in enumerate(clips, start=1):
                    # Compute global trajectory crop path for this specific short clip
                    # Programmatic Duration Safety Clamp (strictly <= 47.5s)
                    raw_segs = clip.segments if hasattr(clip, "segments") else clip.get("segments", [])
                    total_dur = 0.0
                    clip_segs = []
                    for s in raw_segs:
                        s_start = s.get("segment_start", 0.0) if isinstance(s, dict) else s.segment_start
                        s_end = s.get("segment_end", 0.0) if isinstance(s, dict) else s.segment_end
                        dur = s_end - s_start
                        if total_dur + dur > 47.5:
                            remaining = max(0.0, 47.5 - total_dur)
                            if remaining >= 2.0:
                                if isinstance(s, dict):
                                    s["segment_end"] = round(s_start + remaining, 2)
                                else:
                                    s.segment_end = round(s_start + remaining, 2)
                                clip_segs.append(s)
                                total_dur += remaining
                            break
                        else:
                            clip_segs.append(s)
                            total_dur += dur

                    if hasattr(clip, "segments"):
                        clip.segments = clip_segs
                    else:
                        clip["segments"] = clip_segs

                    clip_duration = total_dur
                    total_clip_frames = int(clip_duration * 30.0)

                    focal_target_intervals = []
                    stitched_f_offset = 0
                    for s in clip_segs:
                        s_start = s.get("segment_start", 0.0) if isinstance(s, dict) else getattr(s, "segment_start", 0.0)
                        s_end = s.get("segment_end", 0.0) if isinstance(s, dict) else getattr(s, "segment_end", 0.0)
                        dur = s_end - s_start
                        f_count = int(round(dur * 30.0))
                        f_end = min(total_clip_frames, stitched_f_offset + f_count)
                        ft = s.get("focal_target", "SPEAKER_PRIMARY") if isinstance(s, dict) else getattr(s, "focal_target", "SPEAKER_PRIMARY")
                        focal_target_intervals.append((stitched_f_offset, f_end, str(ft)))
                        stitched_f_offset = f_end

                    if focal_target_intervals and focal_target_intervals[-1][1] < total_clip_frames:
                        last_s, _, last_ft = focal_target_intervals[-1]
                        focal_target_intervals[-1] = (last_s, total_clip_frames, last_ft)

                    logger.info(f"Computing global crop trajectory for clip {idx} ({total_clip_frames} frames, {len(focal_target_intervals)} focal segments)...")
                    crop_path = compute_crop_path(
                        spatial_events=spatial_events_data,
                        scene_changes=[{"frame_idx": fi} for fi in scene_cut_frames],
                        fps=30.0,
                        total_frames=total_clip_frames,
                        crop_ratio=1080.0 / 1920.0,
                        focal_targets=focal_target_intervals
                    )

                    render_job = RenderJob(
                        source_path=source_path,
                        output_dir=output_dir,
                        clip_manifest=clip,
                        clip_index=idx,
                        transcript_words=transcript_words_data,
                        target_resolution=(1080, 1920),
                        target_fps=30,
                        codec="av1_nvenc",
                        spatial_events=spatial_events_data,
                        crop_path_data=crop_path.tolist(),
                        scene_cut_frames=scene_cut_frames
                    )

                    # Execute 3-thread rendering pipeline in-process
                    pipeline = ThreeThreadRenderPipeline(render_job, crop_path=crop_path)
                    success, out_path = await asyncio.to_thread(pipeline.execute)

                    if success:
                        self.completed_renders += 1
                        logger.info(f"✅ Clip {idx}/{self.pending_renders} rendered successfully: {out_path}")
                    else:
                        err = "; ".join(pipeline.errors) or "Render failed."
                        self.render_errors.append(f"Clip {idx}: {err}")
                        self.completed_renders += 1
                        logger.error(f"❌ Clip {idx} failed: {err}")

                    status_payload = {
                        "source_path": source_path,
                        "clip_index": idx,
                        "status": "success" if success else "failed",
                        "error": "" if success else err
                    }
                    await self.js.publish(PIPELINE_STATUS, json.dumps(status_payload).encode("utf-8"))

            # ═════════════════════════════════════════════════════════════════
            # Finalization & Metadata Persistence
            # ═════════════════════════════════════════════════════════════════
            await self._finalize_active_job(success=len(self.render_errors) < self.pending_renders)

        except Exception as e:
            logger.error(f"Pipeline job failed: {e}", exc_info=True)
            if self.active_state:
                self.active_state.error_log.append(f"Pipeline exception: {e}")
            if self.active_msg:
                await self.active_msg.ack()
            self.active_state = None
            self.active_msg = None

    async def _finalize_active_job(self, success: bool):
        """Finalizes the active job: writes metadata.json and ACKs ingest message."""
        if self.active_state is None or self.active_msg is None:
            return

        try:
            self.active_state.pipeline_complete = True
            if success:
                rendered_count = self.completed_renders - len(self.render_errors)
                logger.info(f"🎉 V2 Pipeline successfully completed for: {self.active_state.source_path}")
                logger.info(f"   Rendered {rendered_count} shorts to: {self.active_state.output_dir}")
            else:
                logger.error(f"❌ V2 Pipeline failed for: {self.active_state.source_path}. "
                            f"Errors: {self.active_state.error_log}")

            metadata_file = os.path.join(self.active_state.output_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(self.active_state.model_dump(), f, indent=2, default=str)
            logger.info(f"Saved execution metadata to: {metadata_file}")

            logger.info("ACKing original video.ingest message.")
            await self.active_msg.ack()

        except Exception as e:
            logger.error(f"Failed to finalize active job: {e}")
        finally:
            self.active_state = None
            self.active_msg = None


if __name__ == "__main__":
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    orchestrator = Orchestrator(nats_url)
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")