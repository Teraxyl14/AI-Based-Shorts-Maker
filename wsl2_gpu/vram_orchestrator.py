"""
Project Aether V2 — Phase-Sequential VRAM Lifecycle Orchestrator

Manages GPU memory across pipeline phases using context managers that enforce
explicit model loading/unloading and CUDA cache clearing between phases.
Prevents VRAM fragmentation, OOM exceptions, and CUDA context duplication.
"""

import gc
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

logger = logging.getLogger("AetherVRAMOrchestrator")

# Phase VRAM budget annotations (informational — not enforced as hard limits)
PHASE_BUDGETS = {
    'asr': {
        'description': 'Whisper INT8 + Forced Alignment',
        'estimated_gb': 3.9,
    },
    'vision_editorial': {
        'description': 'YOLO-Pose TRT + VLM Manifest',
        'estimated_gb': 8.5,
    },
    'render': {
        'description': 'NVDEC → Trajectory Crop → Subtitle Compositor → NVENC AV1/H.264',
        'estimated_gb': 5.5,
    },
}


class VRAMOrchestrator:
    """Phase-sequential VRAM lifecycle manager.

    Provides context managers that govern model loading/unloading across
    the three pipeline phases. Ensures each phase starts with a clean
    CUDA memory state.

    Usage:
        vram = VRAMOrchestrator()

        with vram.phase('asr') as ctx:
            model = load_whisper()
            result = model.transcribe(...)
            ctx.register(model)    # Will be deleted on phase exit

        with vram.phase('vision_editorial') as ctx:
            yolo = load_yolo()
            ctx.register(yolo)
            ...

        with vram.phase('render') as ctx:
            ...
    """

    def __init__(self):
        self._current_phase: Optional[str] = None
        self._phase_history: list = []
        self._registered_objects: list = []

    def _get_gpu_memory_stats(self) -> Dict[str, float]:
        """Returns current GPU memory usage in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                return {
                    'allocated_gb': round(allocated, 2),
                    'reserved_gb': round(reserved, 2),
                    'total_gb': round(total, 2),
                    'free_gb': round(total - allocated, 2),
                }
        except Exception:
            pass
        return {'allocated_gb': 0, 'reserved_gb': 0, 'total_gb': 0, 'free_gb': 0}

    def _purge_vram(self):
        """Aggressively frees all GPU memory."""
        import torch

        # Delete all registered objects
        for obj in self._registered_objects:
            try:
                del obj
            except Exception:
                pass
        self._registered_objects.clear()

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Second GC pass to catch cyclic references
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def phase(self, name: str):
        """Context manager for a VRAM lifecycle phase.

        On entry: logs phase start, records memory baseline.
        On exit: deletes registered objects, runs gc.collect() + empty_cache().

        Args:
            name: Phase name ('asr', 'vision_editorial', 'render').
        """
        if self._current_phase is not None:
            raise RuntimeError(
                f"Cannot enter phase '{name}' while phase '{self._current_phase}' is active. "
                f"Phases must be strictly sequential."
            )

        budget = PHASE_BUDGETS.get(name, {})
        desc = budget.get('description', 'Unknown phase')
        est_gb = budget.get('estimated_gb', 0)

        self._current_phase = name
        self._registered_objects.clear()
        phase_start = time.time()

        mem_before = self._get_gpu_memory_stats()
        logger.info(
            f"═══ VRAM Phase '{name}' ENTER ═══ "
            f"({desc}, est. {est_gb:.1f}GB) | "
            f"GPU: {mem_before['allocated_gb']:.2f}GB allocated, "
            f"{mem_before['free_gb']:.2f}GB free"
        )

        ctx = PhaseContext(self)
        try:
            yield ctx
        finally:
            elapsed = time.time() - phase_start
            mem_during = self._get_gpu_memory_stats()
            logger.info(
                f"═══ VRAM Phase '{name}' EXIT ═══ "
                f"(ran {elapsed:.1f}s) | "
                f"Peak allocated: {mem_during['allocated_gb']:.2f}GB | "
                f"Purging {len(self._registered_objects)} registered objects..."
            )

            self._purge_vram()

            mem_after = self._get_gpu_memory_stats()
            freed = mem_during['allocated_gb'] - mem_after['allocated_gb']
            logger.info(
                f"═══ VRAM Phase '{name}' PURGED ═══ "
                f"Freed {freed:.2f}GB | "
                f"Post-purge: {mem_after['allocated_gb']:.2f}GB allocated, "
                f"{mem_after['free_gb']:.2f}GB free"
            )

            self._phase_history.append({
                'name': name,
                'elapsed_s': round(elapsed, 2),
                'peak_gb': mem_during['allocated_gb'],
                'freed_gb': round(freed, 2),
            })

            self._current_phase = None

    def get_phase_report(self) -> list:
        """Returns a summary of all completed phases."""
        return list(self._phase_history)


class PhaseContext:
    """Helper object yielded by VRAMOrchestrator.phase() context manager."""

    def __init__(self, orchestrator: VRAMOrchestrator):
        self._orchestrator = orchestrator

    def register(self, *objects):
        """Registers objects to be deleted when the phase exits.

        Use this to register loaded models, large tensors, or any GPU-resident
        objects that should be freed at the end of the phase.
        """
        self._orchestrator._registered_objects.extend(objects)

    @property
    def phase_name(self) -> str:
        return self._orchestrator._current_phase or "unknown"
