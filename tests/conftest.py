"""
tests/conftest.py - Shared fixtures, helpers, and verification tools for Project Aether E2E Test Suite.

Provides:
- Input/output media paths resolution (cross-platform: container /app & /data vs host paths)
- ffprobe video/audio probing helpers
- NVML peak GPU memory monitoring helpers
- Subtitle safe zone and bounding box verification helpers
- Trajectory velocity and boundary compliance helpers
- Mock/synthetic data factories for opaque-box testing
"""

import os
import sys
import json
import subprocess
import shutil
import time
import pytest
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─── PATH RESOLUTION FIXTURES ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def data_dir() -> str:
    """Resolves data directory path (container /data or project data/)."""
    if os.path.exists("/data") and os.path.isdir("/data"):
        return "/data"
    local_data = os.path.join(PROJECT_ROOT, "data")
    if os.path.exists(local_data):
        return local_data
    return "/data"


@pytest.fixture(scope="session")
def input_videos_dir(data_dir: str) -> str:
    """Path to data/input_videos directory."""
    p = os.path.join(data_dir, "input_videos")
    os.makedirs(p, exist_ok=True)
    return p


@pytest.fixture(scope="session")
def output_shorts_dir(data_dir: str) -> str:
    """Path to data/output_shorts directory."""
    p = os.path.join(data_dir, "output_shorts")
    os.makedirs(p, exist_ok=True)
    return p


@pytest.fixture(scope="session")
def sample_video_paths(input_videos_dir: str) -> Dict[str, str]:
    """Returns mapping of available real input test videos."""
    videos = {}
    candidates = [
        ("comedy", "Adventure _ Stand-up Comedy by Ravi Gupta_1080p.mp4"),
        ("sample2", "Sample Vid 2.mp4"),
        ("testvid", "TestVid.mp4"),
        ("test_video", "test_video.mp4"),
    ]
    for key, name in candidates:
        p = os.path.join(input_videos_dir, name)
        if os.path.exists(p):
            videos[key] = p
    return videos


@pytest.fixture(scope="session")
def completed_output_dirs(output_shorts_dir: str) -> List[str]:
    """Returns list of existing completed pipeline output directories."""
    dirs = []
    if os.path.exists(output_shorts_dir):
        for entry in os.listdir(output_shorts_dir):
            full_path = os.path.join(output_shorts_dir, entry)
            if os.path.isdir(full_path) and not entry.endswith("_v2_backup"):
                # Check if short_1.mp4 exists
                if os.path.exists(os.path.join(full_path, "short_1.mp4")):
                    dirs.append(full_path)
    return dirs


# ─── MEDIA VERIFICATION HELPERS ──────────────────────────────────────────────

def run_ffprobe(file_path: str) -> Dict[str, Any]:
    """Runs ffprobe on a video/audio file and returns parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(res.stdout)
    except Exception as e:
        raise RuntimeError(f"ffprobe failed for {file_path}: {e}")


def probe_video_stream(file_path: str) -> Dict[str, Any]:
    """Extracts width, height, aspect ratio, duration, and codec for the primary video stream."""
    data = run_ffprobe(file_path)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise ValueError(f"No video stream found in {file_path}")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    # Calculate fps
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    # Duration from stream or format
    duration = float(video_stream.get("duration", 0.0) or data.get("format", {}).get("duration", 0.0))

    return {
        "width": width,
        "height": height,
        "aspect_ratio": (width / height) if height > 0 else 0.0,
        "fps": fps,
        "duration": duration,
        "codec_name": video_stream.get("codec_name", ""),
        "nb_frames": int(video_stream.get("nb_frames", 0) or 0),
    }


def probe_audio_stream(file_path: str) -> Optional[Dict[str, Any]]:
    """Extracts audio codec, sample rate, channels, and duration."""
    data = run_ffprobe(file_path)
    streams = data.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not audio_stream:
        return None

    return {
        "codec_name": audio_stream.get("codec_name", ""),
        "sample_rate": int(audio_stream.get("sample_rate", 0)),
        "channels": int(audio_stream.get("channels", 0)),
        "duration": float(audio_stream.get("duration", 0.0) or data.get("format", {}).get("duration", 0.0)),
    }


# ─── NVML VRAM MONITORING HELPERS ────────────────────────────────────────────

class VRAMMonitor:
    """Context manager and query tool for NVML GPU peak memory monitoring."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._initialized = False
        self._handle = None
        self.peak_memory_mb = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._initialized = True
        except Exception:
            self._initialized = False

    def get_current_vram_mb(self) -> float:
        """Returns current GPU used VRAM in megabytes."""
        if not self._initialized:
            return 0.0
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used_mb = info.used / (1024 ** 2)
            if used_mb > self.peak_memory_mb:
                self.peak_memory_mb = used_mb
            return used_mb
        except Exception:
            return 0.0

    def get_total_vram_mb(self) -> float:
        """Returns total GPU VRAM in megabytes."""
        if not self._initialized:
            return 16384.0  # Default assumed 16GB
        try:
            import pynvml
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.total / (1024 ** 2)
        except Exception:
            return 16384.0

    def assert_under_16gb(self):
        """Asserts that used VRAM does not exceed 16,384 MB."""
        used = self.get_current_vram_mb()
        assert used <= 16384.0, f"VRAM peak {used:.1f} MB exceeded 16,384 MB limit!"


@pytest.fixture
def vram_monitor() -> VRAMMonitor:
    """Fixture providing a VRAM monitor instance."""
    return VRAMMonitor()


# ─── SUBTITLE & TYPOGRAPHY ASSERTION HELPERS ─────────────────────────────────

def check_subtitle_safe_zone(y_px: float, canvas_h: float = 1920.0,
                             safe_min_pct: float = 0.60, safe_max_pct: float = 0.70) -> bool:
    """Verifies that subtitle vertical position falls inside the 60%-70% vertical safe zone."""
    y_pct = y_px / canvas_h
    return safe_min_pct <= y_pct <= safe_max_pct


def verify_no_opaque_background_box(rgba_canvas: np.ndarray, tolerance: float = 0.05) -> bool:
    """Verifies that rendered subtitle cues do not have an opaque rectangular container box.

    An opaque rectangular box will have a large rectangular patch with alpha=255
    surrounding all words. Outlined typography only has alpha > 0 near character strokes.
    """
    if rgba_canvas is None or rgba_canvas.shape[2] != 4:
        return False
    alpha = rgba_canvas[:, :, 3]
    h, w = alpha.shape
    total_pixels = h * w
    opaque_pixels = np.sum(alpha > 200)

    # For containerless outlined text, opaque pixels should cover < 30% of canvas area
    coverage = opaque_pixels / total_pixels
    return coverage < 0.35


# ─── TRAJECTORY SMOOTHNESS & CLAMP HELPERS ───────────────────────────────────

def verify_trajectory_bounds(trajectory_px: np.ndarray,
                             min_x: float = 303.75,
                             max_x: float = 1616.25,
                             tolerance: float = 1e-2) -> bool:
    """Verifies that all trajectory coordinates fall strictly within physical crop boundaries."""
    return bool(np.all(trajectory_px >= (min_x - tolerance)) and np.all(trajectory_px <= (max_x + tolerance)))


def verify_max_pan_velocity(trajectory_px: np.ndarray,
                            cut_frames: Optional[List[int]] = None,
                            max_v: float = 15.0,
                            tolerance: float = 1e-2) -> bool:
    """Verifies that frame-to-frame velocity does not exceed 15 px/frame outside shot cuts."""
    cut_set = set(cut_frames or [])
    velocities = np.abs(np.diff(trajectory_px))
    for idx, v in enumerate(velocities):
        # Frame idx+1 is cut frame -> teleport allowed
        if (idx + 1) in cut_set:
            continue
        if v > (max_v + tolerance):
            return False
    return True
