"""
Project Aether V3 — Subprocess-Spawned VRAM Orchestrator

Guarantees 100% VRAM reclamation between pipeline phases via process-level
isolation using multiprocessing.spawn. Each phase runs in a dedicated child
process with its own CUDA context, and all GPU memory is reclaimed by the
OS when the process terminates.

Lifecycle:
    1. Parent spawns child process for each phase
    2. Child loads models, runs inference, serializes results to Pipe
    3. Child exits → OS reclaims all GPU memory (guaranteed)
    4. Parent runs driver-level sanitization (empty_cache + ipc_collect)
    5. Next phase starts with a clean VRAM slate
"""

import gc
import json
import time
import logging
import threading
import warnings
import multiprocessing as mp
from typing import Optional, Dict, Any, List, Tuple, Set

logger = logging.getLogger("AetherVRAMOrchestrator")

# Hard device-wide VRAM ceiling (R4).
VRAM_BUDGET_MB = 16384
NVML_SAMPLE_INTERVAL_S = 0.1
_TERMINATE_GRACE_S = 5.0

# Phase VRAM budget annotations (informational)
PHASE_BUDGETS = {
    'phase1_asr': {
        'description': 'Faster-Whisper distil-large-v3 INT8 + PyAnnote 3.3 + CTC alignment',
        'estimated_gb': 3.5,
    },
    'phase2_vlm_editorial': {
        'description': 'Kornia GPU SceneDet + YOLO11-Pose TRT + VLM Editorial (vLLM sidecar)',
        'estimated_gb': 8.5,
    },
    'phase3_render': {
        'description': 'NVDEC + OSQP Trajectory + GPU Glyph Atlas + NVENC',
        'estimated_gb': 4.2,
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# PROCESS TREE + TELEMETRY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _snapshot_descendants(pid: int) -> Set[int]:
    """Returns PIDs of all live descendants of `pid` (empty if unavailable)."""
    try:
        import psutil
        return {c.pid for c in psutil.Process(pid).children(recursive=True)}
    except Exception:
        return set()


def _kill_pids(pids: Set[int], grace_s: float = _TERMINATE_GRACE_S) -> List[int]:
    """Terminates then force-kills the given PIDs. Returns PIDs that survived."""
    if not pids:
        return []
    try:
        import psutil
    except ImportError:
        logger.warning("psutil unavailable; cannot reap descendant processes %s", sorted(pids))
        return sorted(pids)
    procs = []
    for pid in pids:
        try:
            procs.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            continue
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace_s)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    _, still = psutil.wait_procs(alive, timeout=grace_s)
    survivors = [p.pid for p in still]
    if survivors:
        logger.error("Processes survived SIGKILL: %s", survivors)
    return survivors


def _kill_process_tree(proc, extra_pids: Optional[Set[int]] = None) -> None:
    """Kills a multiprocessing.Process and every descendant it spawned."""
    pids = _snapshot_descendants(proc.pid) | set(extra_pids or ())
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=_TERMINATE_GRACE_S)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=_TERMINATE_GRACE_S)
    reaped = _kill_pids(pids)
    if pids:
        logger.info("Reaped %d descendant process(es); survivors=%s", len(pids), reaped)


class PeakVRAMSampler:
    """Background thread sampling device-wide GPU memory via NVML every 100 ms.

    `peak_mb` is None when NVML is unavailable (CPU-only hosts).
    """

    def __init__(self, interval_s: float = NVML_SAMPLE_INTERVAL_S, device_index: int = 0):
        self._interval = interval_s
        self._device_index = device_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handle = None
        self.peak_mb: Optional[float] = None
        self.samples = 0

    def _init_nvml(self) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
            self._nvml = pynvml
            return True
        except Exception as e:
            logger.debug(f"NVML unavailable: {e}")
            return False

    def _sample(self) -> None:
        used_mb = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / (1024 ** 2)
        self.samples += 1
        if self.peak_mb is None or used_mb > self.peak_mb:
            self.peak_mb = used_mb

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception as e:
                logger.debug(f"NVML sample failed: {e}")
            self._stop.wait(self._interval)

    def start(self) -> "PeakVRAMSampler":
        if self._init_nvml():
            self._sample()
            self._thread = threading.Thread(target=self._loop, name="aether-nvml", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> Optional[float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            try:
                self._sample()
            except Exception:
                pass
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return self.peak_mb


def _worker_cuda_cleanup() -> None:
    """Child-side GPU teardown before exit: sync, drop caches, collect garbage."""
    gc.collect()
    try:
        import sys
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        logging.getLogger("AetherWorkerCleanup").warning(f"CUDA cleanup warning: {e}")
    gc.collect()


class SubprocessPhaseRunner:
    """Manages subprocess lifecycle for a single pipeline phase.

    Handles:
        - Process spawning via an explicit 'spawn' multiprocessing context
        - Result transfer via Pipe
        - Exit code assertion
        - Recursive process-tree teardown (no orphaned ffmpeg / grandchildren)
        - 100 ms NVML peak-VRAM telemetry with a hard 16,384 MB budget check
        - Driver-level VRAM sanitization after process termination
    """

    def __init__(self, phase_name: str, vram_budget_mb: float = VRAM_BUDGET_MB):
        self.phase_name = phase_name
        self.vram_budget_mb = vram_budget_mb
        self._ctx = mp.get_context("spawn")
        self._history: List[Dict[str, Any]] = []

    def run_phase(self, target_func, args: tuple,
                  timeout: float = 1800.0) -> Any:
        """Runs a phase function in an isolated subprocess.

        Args:
            target_func: The worker function to execute. Must accept
                (*args, result_pipe) as its arguments and write the
                JSON-serializable result to result_pipe.
            args: Arguments to pass to target_func (before the pipe).
            timeout: Maximum time to wait for the process (seconds).

        Returns:
            The deserialized result from the worker.

        Raises:
            RuntimeError: If the subprocess fails, times out, or the
                device-wide peak VRAM exceeds the budget.
        """
        budget = PHASE_BUDGETS.get(self.phase_name, {})
        desc = budget.get('description', 'Unknown')
        est_gb = budget.get('estimated_gb', 0)

        parent_conn, child_conn = self._ctx.Pipe(duplex=False)

        logger.info(
            f"═══ SUBPROCESS PHASE '{self.phase_name}' SPAWN ═══ "
            f"({desc}, est. {est_gb:.1f}GB)"
        )
        phase_start = time.time()
        sampler = PeakVRAMSampler().start()

        proc = self._ctx.Process(
            target=target_func,
            args=(*args, child_conn),
            name=f"aether-{self.phase_name}"
        )
        try:
            proc.start()
            child_conn.close()  # Parent doesn't write to child conn
            result = self._await_result(proc, parent_conn, timeout)
        finally:
            parent_conn.close()
            if proc.is_alive() or proc.exitcode is None:
                _kill_process_tree(proc)
            peak_mb = sampler.stop()

        elapsed = time.time() - phase_start

        if proc.exitcode != 0:
            raise RuntimeError(
                f"Phase '{self.phase_name}' subprocess exited with code {proc.exitcode} "
                f"after {elapsed:.1f}s"
            )

        self._sanitize_vram()

        peak_str = f"{peak_mb:.0f}MB" if peak_mb is not None else "n/a"
        self._history.append({
            'phase': self.phase_name,
            'elapsed_s': round(elapsed, 2),
            'exitcode': proc.exitcode,
            'peak_vram_mb': round(peak_mb, 1) if peak_mb is not None else None,
            'nvml_samples': sampler.samples,
        })

        if peak_mb is not None and peak_mb > self.vram_budget_mb:
            raise RuntimeError(
                f"Phase '{self.phase_name}' peak VRAM {peak_mb:.0f}MB exceeded "
                f"budget {self.vram_budget_mb:.0f}MB"
            )

        logger.info(
            f"═══ SUBPROCESS PHASE '{self.phase_name}' COMPLETE ═══ "
            f"(ran {elapsed:.1f}s, exitcode=0, peak VRAM {peak_str}, VRAM sanitized)"
        )
        return result

    def _await_result(self, proc, parent_conn, timeout: float) -> Any:
        """Waits for the worker result, then for exit; reaps the tree on timeout."""
        result = None
        try:
            if parent_conn.poll(timeout):
                result = parent_conn.recv()
            elif proc.is_alive():
                logger.error(f"Phase '{self.phase_name}' timed out after {timeout}s")
                _kill_process_tree(proc)
                raise RuntimeError(f"Phase '{self.phase_name}' timed out after {timeout}s")
        except EOFError:
            logger.warning(f"Phase '{self.phase_name}' pipe closed without sending result")

        # Snapshot descendants while the worker is still our child, so any
        # ffmpeg / helper processes it leaves behind can be reaped after exit.
        lingering = _snapshot_descendants(proc.pid) if proc.is_alive() else set()
        proc.join(timeout=30)
        if proc.is_alive():
            logger.error(f"Phase '{self.phase_name}' did not exit after sending result; killing tree")
            _kill_process_tree(proc, extra_pids=lingering)
        else:
            _kill_pids(lingering)
        return result

    def _sanitize_vram(self):
        """Driver-level VRAM sanitization after subprocess termination.

        Even though the child process's CUDA context is destroyed on exit,
        the parent process may hold cached allocator state. This ensures
        the parent's view of GPU memory is clean.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()  # Second pass for cyclic refs
                torch.cuda.empty_cache()

                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.info(
                    f"Post-sanitization: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved"
                )
        except Exception as e:
            logger.warning(f"VRAM sanitization warning: {e}")

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE WORKER FUNCTIONS (run in spawned child processes)
# ═════════════════════════════════════════════════════════════════════════════

def run_phase1_asr_worker(video_path: str, nats_url: str, result_pipe,
                          task: str = "transcribe",
                          language: Optional[str] = None):
    """Phase 1 Worker: ASR Transcription (~3.5GB peak VRAM).

    Loads Faster-Whisper distil-large-v3 INT8, transcribes the video,
    and sends word timestamps back via pipe.
    Strictly enforces task="transcribe" and language=None for native autodetect.
    """
    assert task == "transcribe", f"STRICTLY FORBID task='{task}', must be 'transcribe'"
    assert language is None, f"Strictly enforce language=None for autodetect, got '{language}'"

    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    logger_w = logging.getLogger("Phase1-ASR-Worker")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        from wsl2_gpu.transcription_spoke import TranscriptionSpoke
        import asyncio

        logger_w.info(f"Phase 1 ASR worker started for: {video_path} (task={task}, language={language})")

        spoke = TranscriptionSpoke(nats_url)
        words = asyncio.run(spoke.transcribe_file(video_path, task=task, language=language))

        # Serialize to JSON-safe dicts
        words_data = [w.model_dump() if hasattr(w, 'model_dump') else w for w in words]

        logger_w.info(f"Phase 1 complete: {len(words_data)} word timestamps (native transcribe)")
        result_pipe.send({"words": words_data, "status": "success"})

    except Exception as e:
        logger_w.error(f"Phase 1 ASR worker failed: {e}", exc_info=True)
        result_pipe.send({"words": [], "status": "error", "error": str(e)})
    finally:
        _worker_cuda_cleanup()
        result_pipe.close()


_run_phase1_asr_worker = run_phase1_asr_worker


def run_phase2_vlm_editorial_worker(video_path: str, words_data: list,
                                     nats_url: str, result_pipe):
    """Phase 2 Worker: Vision + Editorial Engine (~8.5GB peak VRAM).

    Runs GPU scene detection, YOLO pose tracking, and VLM editorial selection.
    """
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    logger_w = logging.getLogger("Phase2-VLM-Worker")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        import asyncio
        from wsl2_gpu.vision_spoke import VisionSpoke
        from wsl2_gpu.editorial_engine import EditorialEngine
        from shared.schemas import WordTimestamp

        logger_w.info(f"Phase 2 VLM editorial worker started for: {video_path}")

        # Step 1: Vision pre-pass (scene detection + pose tracking)
        vision_spoke = VisionSpoke(nats_url)
        vision_spoke._load_models()
        duration, fps, width, height = vision_spoke._get_video_info(video_path)

        logger_w.info(f"Running GPU scene detection + pose tracking ({duration:.1f}s video)...")
        scene_changes = vision_spoke.detect_scene_changes(video_path, duration, fps)
        spatial_events = vision_spoke.detect_spatial_events(video_path, duration, fps)

        # Serialize scene changes and spatial events
        scene_data = [sc.model_dump() if hasattr(sc, 'model_dump') else sc for sc in scene_changes]
        spatial_data = [pe.model_dump() if hasattr(pe, 'model_dump') else pe for pe in spatial_events]
        # Preserve keypoints for Phase 3 active speaker gesture and facial motion gating

        # Step 2: VLM editorial selection via midpoint keyframes
        from wsl2_gpu.fusion_consumer import extract_transcript_beats
        full_beats = extract_transcript_beats(words_data, spatial_data)

        # Enforce minimum 50s duration & narrative multi-segment bounds
        min_dur = float(os.getenv("AETHER_MIN_SHORT_DURATION", "50.0"))
        max_dur = float(os.getenv("AETHER_MAX_SHORT_DURATION", "90.0"))
        target_dur = float(os.getenv("AETHER_TARGET_SHORT_DURATION", "60.0"))
        min_start = float(os.getenv("AETHER_MIN_START_TIME", "20.0"))

        editorial = EditorialEngine(
            min_duration=min_dur,
            max_duration=max_dur,
            target_duration=target_dur,
            min_start_time=min_start
        )
        shorts = asyncio.run(editorial.select_shorts(
            video_path=video_path,
            scene_changes=scene_data,
            transcript_words=words_data,
            video_duration=duration,
            fps=fps,
            spatial_events=spatial_data
        ))

        # Convert ShortCandidates to serializable dicts
        shorts_data = []
        for s in shorts:
            s_beats = []
            if hasattr(s, "beats") and s.beats:
                s_beats = [b.model_dump() if hasattr(b, "model_dump") else b for b in s.beats]
            shorts_data.append({
                "start_time_seconds": s.start_time_seconds,
                "duration_seconds": s.duration_seconds,
                "end_time_seconds": s.end_time_seconds,
                "segments": getattr(s, "segments", None),
                "title": s.title,
                "reasoning": s.reasoning,
                "framing_mode": s.framing_mode,
                "beats": s_beats,
            })

        beats_data = [b.model_dump() if hasattr(b, "model_dump") else b for b in full_beats]

        result = {
            "status": "success",
            "scene_changes": scene_data,
            "spatial_events": spatial_data,
            "transcript_beats": beats_data,
            "editorial_shorts": shorts_data,
            "video_duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
        }

        logger_w.info(f"Phase 2 complete: {len(scene_data)} scenes, "
                      f"{len(spatial_data)} spatial events, "
                      f"{len(beats_data)} transcript beats, "
                      f"{len(shorts_data)} editorial shorts")
        result_pipe.send(result)

    except Exception as e:
        logger_w.error(f"Phase 2 VLM editorial worker failed: {e}", exc_info=True)
        result_pipe.send({
            "status": "error", "error": str(e),
            "scene_changes": [], "spatial_events": [],
            "editorial_shorts": [],
        })
    finally:
        _worker_cuda_cleanup()
        result_pipe.close()


def run_phase3_render_worker(video_path: str, render_manifest: dict,
                              output_dir: str, result_pipe):
    """Phase 3 Worker: Render Pipeline (~4.2GB peak VRAM).

    Uses OSQP trajectory solver, GPU glyph atlas compositor,
    and NVDEC/NVENC (with FFmpeg fallback) for rendering.
    """
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    logger_w = logging.getLogger("Phase3-Render-Worker")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        import asyncio
        from shared.schemas import RenderJob, ClipManifest, VideoSegment
        from wsl2_gpu.render_spoke import ThreeThreadRenderPipeline
        from wsl2_gpu.trajectory_solver import solve_trajectory
        from wsl2_gpu.engagement_evaluator import EngagementEvaluator

        logger_w.info(f"Phase 3 render worker started for: {video_path}")

        shorts = render_manifest.get("editorial_shorts", [])
        words_data = render_manifest.get("words", [])
        spatial_data = render_manifest.get("spatial_events", [])
        scene_data = render_manifest.get("scene_changes", [])
        scene_cut_frames = [sc.get("frame_idx", 0) for sc in scene_data]
        manifest_beats = render_manifest.get("transcript_beats", [])

        evaluator = EngagementEvaluator()

        results = []

        for idx, short in enumerate(shorts, start=1):
            start_t = float(short["start_time_seconds"])
            end_t = float(short["end_time_seconds"])
            duration_t = end_t - start_t
            framing = short.get("framing_mode", "SMART_PORTRAIT_PAN")
            fps = 30.0
            total_frames = int(round(duration_t * fps))

            logger_w.info(
                f"Rendering short {idx}: [{start_t:.1f}s → {end_t:.1f}s] "
                f"({duration_t:.1f}s, {total_frames} frames) mode={framing}"
            )

            # ── Temporal Coordinate Normalization ──
            # Normalize all spatial event timestamps, YOLO keypoints, and scene cuts relative to short window:
            # t_rel = t_abs - short.start_time_seconds
            norm_spatial_data = []
            for pe in spatial_data:
                ts = float(pe.get("timestamp", 0.0) if isinstance(pe, dict) else getattr(pe, "timestamp", 0.0))
                if start_t <= ts <= end_t:
                    t_rel = ts - start_t
                    rel_frame = int(round(t_rel * fps))
                    if isinstance(pe, dict):
                        new_pe = pe.copy()
                        new_pe["timestamp"] = t_rel
                        new_pe["frame_idx"] = rel_frame
                    else:
                        import copy
                        new_pe = copy.copy(pe)
                        new_pe.timestamp = t_rel
                        if hasattr(new_pe, "frame_idx"):
                            new_pe.frame_idx = rel_frame
                    norm_spatial_data.append(new_pe)

            norm_scene_data = []
            norm_scene_cut_frames = []
            for sc in scene_data:
                ts = float(sc.get("timestamp", 0.0) if isinstance(sc, dict) else getattr(sc, "timestamp", 0.0))
                if start_t <= ts <= end_t:
                    t_rel = ts - start_t
                    rel_frame = int(round(t_rel * fps))
                    if isinstance(sc, dict):
                        new_sc = sc.copy()
                        new_sc["timestamp"] = t_rel
                        new_sc["frame_idx"] = rel_frame
                    else:
                        import copy
                        new_sc = copy.copy(sc)
                        new_sc.timestamp = t_rel
                        if hasattr(new_sc, "frame_idx"):
                            new_sc.frame_idx = rel_frame
                    norm_scene_data.append(new_sc)
                    norm_scene_cut_frames.append(rel_frame)

            # Filter transcript words relevant to this short window
            short_words = [
                w for w in words_data
                if (w.get("end", 0.0) if isinstance(w, dict) else getattr(w, "end", 0.0)) >= start_t - 0.5 and
                   (w.get("start", 0.0) if isinstance(w, dict) else getattr(w, "start", 0.0)) <= end_t + 0.5
            ]

            # Extract relevant transcript beats for this short
            short_beats = short.get("beats", [])
            if not short_beats and manifest_beats:
                short_beats = [
                    b for b in manifest_beats
                    if (b.get("start_time", 0.0) if isinstance(b, dict) else getattr(b, "start_time", 0.0)) < end_t and
                       (b.get("end_time", 0.0) if isinstance(b, dict) else getattr(b, "end_time", 0.0)) > start_t
                ]

            # ── Visual Engagement Evaluation & Autonomous Self-Correction Loop ──
            eval_report, corrected_segments, editorial_cuts, focal_targets = evaluator.evaluate_and_correct(
                short_dict_or_candidate=short,
                words=short_words if short_words else words_data,
                spatial_events=norm_spatial_data,
                transcript_beats=short_beats,
                total_duration=duration_t,
            )

            if eval_report.self_correction_applied:
                logger_w.info(
                    f"EngagementEvaluator self-correction applied for Short {idx}: "
                    f"score={eval_report.overall_engagement_score:.2f}, "
                    f"remedial_actions={len(eval_report.remedial_actions)}, "
                    f"segments={len(corrected_segments)}, "
                    f"cuts={len(editorial_cuts)}"
                )

            # Build clip manifest with corrected segments and updated retention/engagement score
            clip = ClipManifest(
                title=short.get("title", f"Short {idx}"),
                segments=corrected_segments,
                hook_score=short.get("hook_score", 0.80),
                retention_score=max(0.80, eval_report.overall_engagement_score),
                cta_present=True,
                reasoning=short.get("reasoning", ""),
                caption_text=short.get("title", ""),
            )

            # Compute trajectory with focal targets and editorial cuts
            crop_path = solve_trajectory(
                spatial_events=norm_spatial_data,
                scene_changes=norm_scene_data,
                fps=fps,
                total_frames=total_frames,
                clip_start_s=0.0,
                focal_targets=focal_targets,
                editorial_cuts=editorial_cuts,
            )
            # Ensure trajectory array is strictly sliced within [0, total_frames]
            crop_path = crop_path[:total_frames]

            # Merge scene cuts and editorial cuts for render job
            editorial_cut_frames = [int(round(c * fps)) for c in editorial_cuts]
            all_cut_frames = sorted(set(norm_scene_cut_frames + editorial_cut_frames))

            # Build render job
            render_job = RenderJob(
                source_path=video_path,
                output_dir=output_dir,
                clip_manifest=clip,
                clip_index=idx,
                transcript_words=short_words if short_words else words_data,
                target_resolution=(1080, 1920),
                target_fps=int(fps),
                codec="av1_nvenc",
                spatial_events=norm_spatial_data,
                crop_path_data=crop_path.tolist(),
                scene_cut_frames=all_cut_frames,
            )

            # Execute render pipeline
            pipeline = ThreeThreadRenderPipeline(render_job, crop_path=crop_path)
            success, out_path = pipeline.execute()

            results.append({
                "clip_index": idx,
                "success": success,
                "output_path": out_path,
                "errors": pipeline.errors,
            })

            if success:
                logger_w.info(f"✅ Short {idx} rendered: {out_path}")
            else:
                logger_w.error(f"❌ Short {idx} failed: {pipeline.errors}")

        result_pipe.send({
            "status": "success",
            "render_results": results,
        })

    except Exception as e:
        logger_w.error(f"Phase 3 render worker failed: {e}", exc_info=True)
        result_pipe.send({
            "status": "error", "error": str(e),
            "render_results": [],
        })
    finally:
        _worker_cuda_cleanup()
        result_pipe.close()
