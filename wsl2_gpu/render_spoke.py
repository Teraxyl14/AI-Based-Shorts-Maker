import os
import sys
import json
import math
import asyncio
import logging
import subprocess
import time
import tempfile
import threading
import queue
import numpy as np
import nats
from nats.js.api import ConsumerConfig
from typing import List, Optional, Tuple, Any, Dict

import torch
try:
    import kornia
except ImportError:
    kornia = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.schemas import RenderJob
from shared.nats_subjects import VIDEO_RENDER, PIPELINE_STATUS
from wsl2_gpu.trajectory_solver import solve_trajectory
from wsl2_gpu.trajectory_smoother import compute_crop_path, smooth_trajectory
from wsl2_gpu.subtitle_compositor import SubtitleCompositor
from wsl2_gpu.gpu_compositor import (
    render_dual_pane_split_screen,
    render_full_frame_topic_closeup,
    calculate_dynamic_topic_crop_2d
)

logger = logging.getLogger("AetherRenderSpoke")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ─── Constants ───────────────────────────────────────────────────────────────
RING_BUFFER_SIZE = 8  # Pre-allocated frame slots in the ring buffer


# ─── GPU-Resident 2-Mode Universal Framing Helpers ──────────────────────────────────────

def render_smart_portrait_pan(frame: torch.Tensor, cx_norm: float,
                              target_w: int = 1080, target_h: int = 1920,
                              face_bbox: Optional[Tuple[float, float, float, float]] = None) -> torch.Tensor:
    """MODE 1 — SMART_PORTRAIT_PAN (Full-Height 9:16 Vertical Crop with Adaptive Headroom):
    - When face is a tight close-up (face_h >= 0.25 or face_w >= 0.15 of source frame):
      * Widen crop by 25% (cw = 759px instead of 607px).
      * Scale foreground to full 1080px width (1080 x 1536).
      * Position in center vertically, leaving slim 10% (192px) blurred bands top and bottom.
      * Result: Face is comfortably zoomed out, revealing natural neck, shoulders, and headroom!
    - When face is medium/wide or unmeasured:
      * Crop full source height (ch = 1.0, H = 1080) and width cw = 607.5px (9:16 ratio).
      * 100% full-bleed vertical 9:16 crop with zero blur!
    """
    B, C, src_h, src_w = frame.shape

    is_close_up = False
    if face_bbox is not None and len(face_bbox) >= 4:
        f_w = float(face_bbox[2])
        f_h = float(face_bbox[3])
        if f_h >= 0.25 or f_w >= 0.15:
            is_close_up = True

    if is_close_up:
        # 25% widened crop for comfortable headroom and shoulder framing
        crop_w = min(float(src_w), float(src_h) * (float(target_w) / float(target_h)) * 1.25)
        crop_h = float(src_h)
        x_center = cx_norm * float(src_w)
        x0_val = x_center - (crop_w / 2.0)
        x0 = max(0, min(int(src_w - crop_w), int(round(x0_val))))
        x1 = int(round(x0 + crop_w))
        cropped = frame[:, :, 0:src_h, x0:x1]

        # Scale foreground to span full width
        fg_w = target_w
        fg_h = int(round(float(crop_h) * (float(target_w) / float(crop_w))))
        layer2 = torch.nn.functional.interpolate(
            cropped, size=(fg_h, fg_w), mode='bilinear', align_corners=False
        )

        # Background: soft Gaussian blurred cover
        scale_cover = max(float(target_w) / float(src_w), float(target_h) / float(src_h))
        bg_w = int(round(src_w * scale_cover))
        bg_h = int(round(src_h * scale_cover))
        bg_scaled = torch.nn.functional.interpolate(frame, size=(bg_h, bg_w), mode='bilinear', align_corners=False)
        start_x_bg = (bg_w - target_w) // 2
        start_y_bg = (bg_h - target_h) // 2
        bg_cropped = bg_scaled[:, :, start_y_bg:start_y_bg + target_h, start_x_bg:start_x_bg + target_w]
        bg_down = torch.nn.functional.interpolate(bg_cropped, size=(target_h // 4, target_w // 4), mode='bilinear', align_corners=False)
        if kornia is not None:
            bg_blurred_down = kornia.filters.gaussian_blur2d(bg_down, kernel_size=(31, 31), sigma=(12.0, 12.0), border_type='replicate')
        else:
            bg_blurred_down = torch.nn.functional.avg_pool2d(bg_down, kernel_size=7, stride=1, padding=3, count_include_pad=False)
        bg_blurred = torch.nn.functional.interpolate(bg_blurred_down, size=(target_h, target_w), mode='bilinear', align_corners=False)
        canvas = bg_blurred * 0.85

        start_y_fg = max(0, (target_h - fg_h) // 2)
        end_y_fg = min(target_h, start_y_fg + fg_h)
        actual_fg_h = end_y_fg - start_y_fg
        canvas[:, :, start_y_fg:end_y_fg, :] = layer2[:, :, :actual_fg_h, :]
        return canvas
    else:
        crop_h = float(src_h)
        crop_w = float(src_h) * (float(target_w) / float(target_h))  # 607.5px for 1080p

        x_center = cx_norm * float(src_w)
        x0_val = x_center - (crop_w / 2.0)
        x0 = max(0, min(int(src_w - crop_w), int(round(x0_val))))
        x1 = int(round(x0 + crop_w))
        y0 = 0
        y1 = src_h

        cropped = frame[:, :, y0:y1, x0:x1]
        resized = torch.nn.functional.interpolate(
            cropped, size=(target_h, target_w), mode='bilinear', align_corners=False
        )
        return resized

# Legacy alias for backward compatibility
render_speaker_solo = render_smart_portrait_pan
render_full_frame_speaker_tight = render_smart_portrait_pan


def render_canvas_graphic_fit(frame: torch.Tensor,
                              target_w: int = 1080, target_h: int = 1920,
                              dynamic_remap: bool = False,
                              focal_target: Optional[str] = None,
                              speaker_cx: float = 0.5,
                              topic_cx: float = 0.5,
                              topic_cy: float = 0.5,
                              layout_mode: Optional[str] = None) -> torch.Tensor:
    """MODE 2 — CANVAS_GRAPHIC_FIT / ZERO-LETTERBOX DYNAMIC COMPOSITING:
    - Background: Scale the 16:9 source to cover 1080 x 1920, downsample 4x,
      apply Gaussian blur (sigma=10.0, k=31), and darken by 45% (bg * 0.55).
    - Foreground: Scale the uncropped 16:9 frame to 1080px width (1080 x 607.5)
      and paste centered vertically at Y = 656px.
    - Zero-Letterbox Dynamic Remap: When dynamic_remap is True, eliminates unstyled
      16:9 blurred letterboxing, dynamically rendering as either:
      * FULL_FRAME_TOPIC_CLOSEUP (Mode 2) centered directly on the topic / screen
      * DUAL_PANE_SPLIT_SCREEN (Mode 3) with styled 3-4px #CCFF00 neon glowing divider.
    Returns: [1, 3, target_h, target_w] float32 tensor in [0, 1].
    """
    if dynamic_remap:
        target_str = str(focal_target or layout_mode or "").upper().strip()
        if "SPLIT" in target_str or target_str == "DUAL_PANE_SPLIT_SCREEN":
            return render_dual_pane_split_screen(
                frame=frame,
                speaker_cx=speaker_cx,
                topic_cx=topic_cx,
                topic_cy=topic_cy,
                target_w=target_w,
                target_h=target_h
            )
        elif "TOPIC_CLOSEUP" in target_str or "CLOSEUP" in target_str:
            return render_full_frame_topic_closeup(
                frame=frame,
                cx_norm=topic_cx,
                cy_norm=topic_cy,
                target_w=target_w,
                target_h=target_h
            )

    B, C, src_h, src_w = frame.shape

    # Layer 1: Blurred, darkened cover background with cover scale safeguard
    scale_cover = max(float(target_w) / float(src_w), float(target_h) / float(src_h))
    bg_w = int(round(src_w * scale_cover))
    bg_h = int(round(src_h * scale_cover))
    bg_scaled = torch.nn.functional.interpolate(frame, size=(bg_h, bg_w), mode='bilinear', align_corners=False)

    start_x = (bg_w - target_w) // 2
    start_y_bg = (bg_h - target_h) // 2
    bg_cropped = bg_scaled[:, :, start_y_bg:start_y_bg + target_h, start_x:start_x + target_w]

    bg_down = torch.nn.functional.interpolate(bg_cropped, size=(target_h // 4, target_w // 4), mode='bilinear', align_corners=False)
    if kornia is not None:
        bg_blurred_down = kornia.filters.gaussian_blur2d(bg_down, kernel_size=(31, 31), sigma=(10.0, 10.0), border_type='replicate')
    else:
        bg_blurred_down = torch.nn.functional.avg_pool2d(bg_down, kernel_size=7, stride=1, padding=3, count_include_pad=False)
    bg_blurred = torch.nn.functional.interpolate(bg_blurred_down, size=(target_h, target_w), mode='bilinear', align_corners=False)
    layer1 = bg_blurred * 0.55


    # Layer 2: Full-fidelity uncropped 16:9 foreground
    fg_w = target_w
    fg_h = int(round(float(src_h) * (float(target_w) / float(src_w))))  # 608px
    layer2 = torch.nn.functional.interpolate(frame, size=(fg_h, fg_w), mode='bilinear', align_corners=False)

    start_y = (target_h - fg_h) // 2  # (1920 - 608) // 2 = 656px
    processed = layer1.clone()
    processed[:, :, start_y:start_y + fg_h, :] = layer2

    return processed

# Legacy aliases for backward compatibility
render_content_fit = render_canvas_graphic_fit


def render_canvas_uncertainty_fit(frame: torch.Tensor,
                                  target_w: int = 1080, target_h: int = 1920,
                                  zoom_factor: float = 1.0) -> torch.Tensor:
    """UNCERTAINTY FALLBACK FRAMING:
    Triggered ONLY when the system cannot identify what is being talked about with high confidence
    (e.g., ambiguous scene, dark/off monitor cutaway, low-entropy visual content).

    - Background (Layer 1): Full 16:9 frame scaled to cover 1080x1920, downsampled 4x,
      Gaussian blurred (sigma=15.0, k=31), and darkened by 50% (bg * 0.50).
    - Foreground (Layer 2): Full uncropped 16:9 frame scaled to 100% width (1080 x 608 px),
      centered vertically (Y = 656px). Zero side margins or floating mini-screen!
    """
    B, C, src_h, src_w = frame.shape

    # Layer 1: Blurred, darkened cover background
    scale_cover = max(float(target_w) / float(src_w), float(target_h) / float(src_h))
    bg_w = int(round(src_w * scale_cover))
    bg_h = int(round(src_h * scale_cover))
    bg_scaled = torch.nn.functional.interpolate(frame, size=(bg_h, bg_w), mode='bilinear', align_corners=False)

    start_x_bg = (bg_w - target_w) // 2
    start_y_bg = (bg_h - target_h) // 2
    bg_cropped = bg_scaled[:, :, start_y_bg:start_y_bg + target_h, start_x_bg:start_x_bg + target_w]

    bg_down = torch.nn.functional.interpolate(bg_cropped, size=(target_h // 4, target_w // 4), mode='bilinear', align_corners=False)
    if kornia is not None:
        bg_blurred_down = kornia.filters.gaussian_blur2d(bg_down, kernel_size=(31, 31), sigma=(15.0, 15.0), border_type='replicate')
    else:
        bg_blurred_down = torch.nn.functional.avg_pool2d(bg_down, kernel_size=9, stride=1, padding=4, count_include_pad=False)
    bg_blurred = torch.nn.functional.interpolate(bg_blurred_down, size=(target_h, target_w), mode='bilinear', align_corners=False)
    layer1 = bg_blurred * 0.85

    # Layer 2: 100% width uncropped 16:9 foreground (zero side margins)
    fg_w = target_w
    fg_h = int(round(float(src_h) * (float(fg_w) / float(src_w))))  # 608px
    layer2 = torch.nn.functional.interpolate(frame, size=(fg_h, fg_w), mode='bilinear', align_corners=False)

    start_y_fg = (target_h - fg_h) // 2  # 656px

    processed = layer1.clone()
    processed[:, :, start_y_fg:start_y_fg + fg_h, :] = layer2

    return processed


def render_broll_pillarbox(source_tensor: torch.Tensor, target_canvas_shape: tuple) -> torch.Tensor:
    """Backward-compatible wrapper for CANVAS_GRAPHIC_FIT layout helper."""
    tgt_h, tgt_w = target_canvas_shape
    canvas = render_canvas_graphic_fit(source_tensor, target_w=tgt_w, target_h=tgt_h)
    return (canvas.squeeze(0).permute(1, 2, 0) * 255.0).byte()


# ─── Shot-Level Adaptive Framing Helpers ────────────────────────────────────

HUMAN_DETECTION_LABELS = frozenset({
    "face", "head", "face_detection", "person", "torso",
    "upper_body", "shoulders", "pose", "human", "body", "upper_torso"
})

SCREEN_DETECTION_LABELS = frozenset({
    "monitor", "screen", "laptop", "tv", "display", "tablet", "screencast"
})


def classify_shot_framing(shot_start: float, shot_end: float,
                          spatial_events: List[Any],
                          sampling_interval: float = 0.5) -> str:
    """Classifies an individual shot into SHOT_PORTRAIT_PAN, SHOT_GRAPHIC_FIT, or CANVAS_UNCERTAINTY_FIT.

    Directive Rules:
    - High-Confidence Display / Object: If an active screen, monitor, software display, or product is prominent:
      Classify shot as 'SHOT_GRAPHIC_FIT' to display full widescreen UI.
    - High-Confidence Speaker: If human face or torso is detected:
      Classify shot as 'SHOT_PORTRAIT_PAN' (centered speaker portrait).
    - Uncertainty Fallback: GATED STRICTLY to true unresolvable edge cases (e.g. black/dark scene, unparseable noise).
      NEVER used freely for standard video footage.
    """
    shot_dur = shot_end - shot_start
    if shot_dur <= 0.0:
        return "SHOT_PORTRAIT_PAN"

    step = min(sampling_interval, max(0.1, shot_dur / 5.0))
    sample_times = np.arange(shot_start, shot_end, step)
    if len(sample_times) == 0:
        sample_times = np.array([shot_start])

    tolerance = step * 0.75
    human_present_count = 0
    screen_present_count = 0
    low_confidence_count = 0

    for st in sample_times:
        has_human = False
        has_screen = False
        max_conf = 0.0
        for pe in spatial_events:
            ts = pe.get("timestamp", 0.0) if isinstance(pe, dict) else getattr(pe, "timestamp", 0.0)
            if abs(ts - st) <= tolerance:
                conf = float(pe.get("confidence", 0.0) if isinstance(pe, dict) else getattr(pe, "confidence", 0.0))
                lbl = str(pe.get("label", "") if isinstance(pe, dict) else getattr(pe, "label", "")).lower().strip()
                max_conf = max(max_conf, conf)
                if conf >= 0.45:
                    if lbl in HUMAN_DETECTION_LABELS:
                        has_human = True
                    if lbl in SCREEN_DETECTION_LABELS:
                        has_screen = True
        if has_human:
            human_present_count += 1
        if has_screen:
            screen_present_count += 1
        if max_conf < 0.20:
            low_confidence_count += 1

    total_samples = float(len(sample_times))
    human_ratio = human_present_count / total_samples
    screen_ratio = screen_present_count / total_samples
    uncertain_ratio = low_confidence_count / total_samples

    # 1. High-confidence active screen / monitor UI
    if screen_ratio >= 0.25 and screen_ratio >= (human_ratio * 0.7):
        return "SHOT_GRAPHIC_FIT"

    # 2. High-confidence speaker portrait
    if human_ratio >= 0.15:
        return "SHOT_PORTRAIT_PAN"

    # 3. If any screen/display detected
    if screen_ratio > 0.0:
        return "SHOT_GRAPHIC_FIT"

    # 4. Screencast, software UI, gameplay, map, or graphic display (no human present):
    # Route directly to complete 9:16 vertical ratio (zero letterbox) via SHOT_GRAPHIC_FIT
    if human_ratio == 0.0:
        return "SHOT_GRAPHIC_FIT"

    # Default to portrait pan for standard footage
    return "SHOT_PORTRAIT_PAN"


MIN_SHOT_DURATION = 2.0  # Minimum shot duration in seconds to eliminate jitter and split-second flashes


def build_shot_framing_schedule(total_duration: float,
                                mapped_cuts: List[Dict[str, Any]],
                                spatial_events: List[Any],
                                min_shot_duration: float = MIN_SHOT_DURATION) -> List[Tuple[float, float, str]]:
    """Builds a smooth, jitter-free per-shot framing schedule across the clip timeline.

    Enforces:
    1. Minimum Shot Dwell Time (min_shot_duration = 2.0s):
       Adjacent raw scene cuts separated by < min_shot_duration are pruned to prevent rapid micro-shots.
    2. Temporal Hysteresis Debouncing:
       If shot i is Mode A, shot i+1 is Mode B with duration < 2.5s, and shot i+2 returns to Mode A,
       absorbs shot i+1 into Mode A to eliminate 1-frame or split-second visual flashes.
    3. Consecutive Shot Collapsing:
       Adjacent shots sharing the same framing mode are merged into single continuous intervals.
    """
    raw_cuts = sorted(set([0.0] + [
        float(c.get("timestamp", 0.0) if isinstance(c, dict) else getattr(c, "timestamp", 0.0))
        for c in mapped_cuts
        if 0.0 < float(c.get("timestamp", 0.0) if isinstance(c, dict) else getattr(c, "timestamp", 0.0)) < total_duration
    ] + [total_duration]))

    # Step 1: Prune micro-cuts separated by < min_shot_duration
    filtered_cuts = [0.0]
    for cut in raw_cuts[1:-1]:
        if (cut - filtered_cuts[-1]) >= min_shot_duration and (total_duration - cut) >= 1.0:
            filtered_cuts.append(cut)
    filtered_cuts.append(total_duration)

    # Step 2: Classify initial shot intervals
    initial_schedule = []
    for i in range(len(filtered_cuts) - 1):
        s_start = filtered_cuts[i]
        s_end = filtered_cuts[i + 1]
        mode = classify_shot_framing(s_start, s_end, spatial_events)
        initial_schedule.append([s_start, s_end, mode])

    if not initial_schedule:
        return [(0.0, total_duration, "SHOT_PORTRAIT_PAN")]

    # Step 3: Temporal Hysteresis Debouncing (sandwich elimination)
    # If A -> B -> A and B duration < 2.5s, absorb B into A
    if len(initial_schedule) >= 3:
        for i in range(1, len(initial_schedule) - 1):
            prev_mode = initial_schedule[i - 1][2]
            curr_mode = initial_schedule[i][2]
            next_mode = initial_schedule[i + 1][2]
            curr_dur = initial_schedule[i][1] - initial_schedule[i][0]
            if prev_mode == next_mode and curr_mode != prev_mode and curr_dur < 2.5:
                initial_schedule[i][2] = prev_mode

    # Step 4: Collapse consecutive shots with identical mode
    collapsed: List[Tuple[float, float, str]] = []
    for s_start, s_end, mode in initial_schedule:
        if not collapsed:
            collapsed.append((s_start, s_end, mode))
        else:
            prev_start, prev_end, prev_mode = collapsed[-1]
            if prev_mode == mode:
                collapsed[-1] = (prev_start, s_end, mode)
            else:
                collapsed.append((s_start, s_end, mode))

    return collapsed


# ─── Event Stub for CPU execution ───────────────────────────────────────────

class CUDAEventStub:
    def record(self):
        pass
    def synchronize(self):
        pass


# ─── Three-Thread Render Pipeline (V2 Global Trajectory) ─────────────────────

class ThreeThreadRenderPipeline:
    """Multi-threaded V2 render architecture:

    Thread 1 (Decode):    Extracts clip segment + decodes frames via ffmpeg pipe
    Thread 2 (Inference): Offline global trajectory crop + premultiplied subtitle overlay
    Thread 3 (Encode):    Encodes final frames via ffmpeg pipe (AV1 NVENC or libx264 fallback)

    Threads communicate over pre-allocated GPU-resident ring buffers using explicit
    CUDA streams and Event synchronization.
    """
    def __init__(self, job: RenderJob, crop_path: Optional[np.ndarray] = None):
        self.job = job
        self.clip = job.clip_manifest
        self.idx = job.clip_index
        self.crop_path = crop_path

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Communication queues between threads
        self.decode_to_inference: queue.Queue = queue.Queue(maxsize=RING_BUFFER_SIZE)
        self.inference_to_encode: queue.Queue = queue.Queue(maxsize=RING_BUFFER_SIZE)

        # Thread completion flags
        self.decode_done = threading.Event()
        self.inference_done = threading.Event()
        self.encode_done = threading.Event()
        self.stop_event = threading.Event()
        self.empty_slots = None

        # Pre-allocated ring buffer slots
        self.decode_slots = []
        self.render_slots = []
        self.decode_events = []
        self.inference_events = []

        self.errors: List[str] = []
        self.frame_count = 0
        self.subtitle_compositor: Optional[SubtitleCompositor] = None
        self.shot_framing_schedule: List[Tuple[float, float, str]] = []

    def _allocate_buffers(self, src_w: int, src_h: int, target_w: int, target_h: int):
        """Pre-allocates tensor memory pools using RMM and DLPack zero-copy."""
        logger.info(f"[Clip {self.idx}] Allocating ring buffer on device: {self.device}")
        
        self.decode_slots = []
        self.render_slots = []

        try:
            import cupy as cp
            import rmm
            use_rmm = (self.device.type == "cuda")
        except ImportError:
            use_rmm = False

        for _ in range(RING_BUFFER_SIZE):
            if use_rmm:
                with cp.cuda.Device(self.device.index or 0):
                    arr_dec = cp.zeros((src_h, src_w, 3), dtype=cp.uint8)
                    self.decode_slots.append(torch.from_dlpack(arr_dec))
                    
                    arr_ren = cp.zeros((target_h, target_w, 3), dtype=cp.uint8)
                    self.render_slots.append(torch.from_dlpack(arr_ren))
            else:
                self.decode_slots.append(torch.zeros((src_h, src_w, 3), dtype=torch.uint8, device=self.device))
                self.render_slots.append(torch.zeros((target_h, target_w, 3), dtype=torch.uint8, device=self.device))

        if self.device.type == "cuda":
            self.decode_events = [torch.cuda.Event(enable_timing=False) for _ in range(RING_BUFFER_SIZE)]
            self.inference_events = [torch.cuda.Event(enable_timing=False) for _ in range(RING_BUFFER_SIZE)]
        else:
            self.decode_events = [CUDAEventStub() for _ in range(RING_BUFFER_SIZE)]
            self.inference_events = [CUDAEventStub() for _ in range(RING_BUFFER_SIZE)]

    def _map_original_to_stitched(self, original_ts: float, segments: List[Any]) -> Optional[float]:
        stitched_offset = 0.0
        total_dur = sum(
            (seg.segment_end - seg.segment_start) if hasattr(seg, "segment_end")
            else (seg.get("segment_end", 0.0) - seg.get("segment_start", 0.0))
            for seg in segments
        )
        # If timestamp is already relative / normalized to the clip [0, total_dur]
        if 0.0 <= original_ts <= total_dur + 0.1 and segments:
            first_s = getattr(segments[0], "segment_start", 0.0) if hasattr(segments[0], "segment_start") else segments[0].get("segment_start", 0.0)
            if original_ts < first_s:
                return original_ts  # Already normalized!

        for seg in segments:
            s_start = seg.segment_start if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
            s_end = seg.segment_end if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
            dur = s_end - s_start
            if s_start <= original_ts <= s_end:
                return stitched_offset + (original_ts - s_start)
            stitched_offset += dur
        return None

    def execute(self) -> Tuple[bool, str]:
        """Runs the three-thread pipeline. Returns (success, output_path)."""
        output_path = os.path.join(self.job.output_dir, f"short_{self.idx}.mp4")
        
        segs = self.clip.segments
        segs_str = ", ".join([f"{s.segment_start:.1f}s-{s.segment_end:.1f}s" for s in segs])
        logger.info(f"[Clip {self.idx}] Starting V2 render pipeline with segments: {segs_str} → {output_path}")

        with tempfile.TemporaryDirectory(dir=self.job.output_dir) as tmpdir:
            # Step 1: Get source video dimensions directly
            src_w, src_h = self._probe_dimensions(self.job.source_path)
            if src_w == 0 or src_h == 0:
                self.errors.append("Failed to probe video dimensions")
                return False, output_path

            # Step 2: Check AV1 NVENC hardware encoder support
            codec_to_use = "av1_nvenc"
            if not self._check_av1_nvenc_support():
                logger.warning(
                    f"av1_nvenc initialization not supported or restricted for clip {self.idx}. "
                    f"Using libx264 software encoding."
                )
                codec_to_use = "libx264"

            # Step 3: Extract and stitch raw clip audio segments to WAV
            stitched_audio_path = os.path.join(tmpdir, f"audio_{self.idx}.wav")
            if not self._extract_stitched_audio(stitched_audio_path):
                return False, output_path

            # Step 4: Calculate total frames and compute global crop trajectory if not provided
            total_duration = sum([
                (s.get("segment_end", 0.0) - s.get("segment_start", 0.0)) if isinstance(s, dict)
                else (s.segment_end - s.segment_start)
                for s in segs
            ])
            total_frames = int(round(total_duration * float(self.job.target_fps)))
            
            # Map spatial events and cuts to the stitched timeline
            mapped_events = []
            spatial_events = getattr(self.job, "spatial_events", [])
            logger.info(f"[Clip {self.idx}] Received {len(spatial_events)} spatial events for clip {self.idx}")

            for pe in spatial_events:
                ts = pe.get("timestamp", 0.0) if isinstance(pe, dict) else getattr(pe, "timestamp", 0.0)
                stitched_ts = self._map_original_to_stitched(ts, segs)
                if stitched_ts is not None:
                    if isinstance(pe, dict):
                        new_pe = pe.copy()
                        new_pe["timestamp"] = stitched_ts
                        mapped_events.append(new_pe)
                    else:
                        import copy
                        new_pe = copy.copy(pe)
                        new_pe.timestamp = stitched_ts
                        mapped_events.append(new_pe)

            self.mapped_spatial_events = mapped_events
            
            mapped_cuts = []
            stitched_offset = 0.0
            for seg in segs:
                s_start = seg.segment_start if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
                s_end = seg.segment_end if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
                dur = s_end - s_start
                if stitched_offset > 0:
                    mapped_cuts.append({"timestamp": stitched_offset, "frame_idx": int(stitched_offset * self.job.target_fps)})
                stitched_offset += dur
                
            for c in getattr(self.job, "scene_cut_frames", []):
                orig_ts = c / float(self.job.target_fps)
                stitched_ts = self._map_original_to_stitched(orig_ts, segs)
                if stitched_ts is not None:
                    mapped_cuts.append({"timestamp": stitched_ts, "frame_idx": int(stitched_ts * self.job.target_fps)})

            # Step 4b: Build Shot-Level Adaptive Framing Schedule
            self.shot_framing_schedule = build_shot_framing_schedule(
                total_duration=total_duration,
                mapped_cuts=mapped_cuts,
                spatial_events=mapped_events
            )
            logger.info(f"[Clip {self.idx}] Shot-Level Adaptive Framing: {len(self.shot_framing_schedule)} shots: {self.shot_framing_schedule}")

            # Build per-segment focal target intervals for trajectory optimization
            focal_target_intervals = []
            stitched_f_offset = 0
            for seg in segs:
                s_start = seg.segment_start if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
                s_end = seg.segment_end if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
                dur = s_end - s_start
                f_count = int(round(dur * float(self.job.target_fps)))
                f_end = min(total_frames, stitched_f_offset + f_count)
                ft = getattr(seg, "focal_target", "SPEAKER_PRIMARY") if hasattr(seg, "focal_target") else seg.get("focal_target", "SPEAKER_PRIMARY")
                focal_target_intervals.append((stitched_f_offset, f_end, str(ft)))
                stitched_f_offset = f_end

            if focal_target_intervals and focal_target_intervals[-1][1] < total_frames:
                last_s, _, last_ft = focal_target_intervals[-1]
                focal_target_intervals[-1] = (last_s, total_frames, last_ft)

            logger.info(f"[Clip {self.idx}] Script-Grounded Semantic Framing: {len(focal_target_intervals)} segments: {focal_target_intervals}")

            if self.crop_path is None:
                if getattr(self.job, "crop_path_data", None) and len(self.job.crop_path_data) > 0:
                    self.crop_path = np.array(self.job.crop_path_data, dtype=np.float64)
                    logger.info(f"[Clip {self.idx}] Ingested pre-computed crop path of shape {self.crop_path.shape}")
                else:
                    logger.info(f"[Clip {self.idx}] Computing global OSQP trajectory path for {total_frames} frames...")
                    self.crop_path = solve_trajectory(
                        spatial_events=mapped_events,
                        scene_changes=mapped_cuts,
                        fps=float(self.job.target_fps),
                        total_frames=total_frames,
                        clip_start_s=0.0,
                        focal_targets=focal_target_intervals
                    )


            target_w, target_h = self.job.target_resolution

            # Step 5: Initialize SubtitleCompositor with CPU word texture atlas
            self.subtitle_compositor = SubtitleCompositor()
            if self.job.transcript_words:
                self.subtitle_compositor.build_word_atlas(self.job.transcript_words, target_w=target_w)

            # Step 6: Pre-allocate buffers based on source/target resolutions
            self._allocate_buffers(src_w, src_h, target_w, target_h)

            # Step 7: Spawn and start threads
            self.stop_event.clear()
            self.empty_slots = threading.Semaphore(RING_BUFFER_SIZE)

            t_decode = threading.Thread(
                target=self._decode_thread,
                args=(self.job.source_path, src_w, src_h),
                name=f"Decode-Clip-{self.idx}"
            )
            t_inference = threading.Thread(
                target=self._inference_thread,
                args=(src_w, src_h, target_w, target_h),
                name=f"Inference-Clip-{self.idx}"
            )
            t_encode = threading.Thread(
                target=self._encode_thread,
                args=(stitched_audio_path, output_path, target_w, target_h, codec_to_use),
                name=f"Encode-Clip-{self.idx}"
            )

            logger.info(f"[Clip {self.idx}] Spawning OS threads for decode, inference, encode...")
            t_decode.start()
            t_inference.start()
            t_encode.start()

            # Wait for execution to finish
            while not self.stop_event.is_set():
                t_decode.join(timeout=0.1)
                t_inference.join(timeout=0.1)
                t_encode.join(timeout=0.1)
                if not (t_decode.is_alive() or t_inference.is_alive() or t_encode.is_alive()):
                    break

            # Cleanup / Join all threads safely
            self.stop_event.set()
            t_decode.join()
            t_inference.join()
            t_encode.join()

            success = len(self.errors) == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000
            if success:
                # Step 8: Generate thumbnail image at 25% duration
                self._generate_thumbnail(self.job.source_path)

            return success, output_path

    def _extract_stitched_audio(self, output_wav_path: str) -> bool:
        """Extracts and concatenates the audio segments using zero-crossing alignment."""
        try:
            segments = self.clip.segments
            if not segments:
                raise ValueError("No segments found in clip manifest")

            tmp_full_wav = output_wav_path.replace(".wav", "_full.wav")
            cmd_extract = [
                "ffmpeg", "-y", "-i", self.job.source_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
                tmp_full_wav
            ]
            logger.info(f"[Clip {self.idx}] Extracting full audio to: {tmp_full_wav}")
            subprocess.run(cmd_extract, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            import wave

            with wave.open(tmp_full_wav, "rb") as wf:
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                audio_data = np.frombuffer(wf.readframes(nframes), dtype=np.int16)

            stitched_audio = []
            target_fps = self.job.target_fps
            
            def find_zero_crossing(sample_idx, window=2000):
                start_idx = max(0, sample_idx - window)
                end_idx = min(len(audio_data)-1, sample_idx + window)
                if start_idx >= end_idx:
                    return sample_idx
                window_data = audio_data[start_idx:end_idx]
                zero_crossings = np.where(np.diff(np.signbit(window_data)))[0]
                if len(zero_crossings) == 0:
                    return sample_idx
                closest = zero_crossings[np.argmin(np.abs(zero_crossings - window))]
                return start_idx + closest

            for seg in segments:
                start_time = getattr(seg, "segment_start", 0.0) if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
                end_time = getattr(seg, "segment_end", 0.0) if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
                
                start_frame = int(start_time * target_fps)
                end_frame = int(end_time * target_fps)
                start_sample = int(start_frame * (48000 / target_fps))
                end_sample = int(end_frame * (48000 / target_fps))
                
                start_zc = find_zero_crossing(start_sample)
                end_zc = find_zero_crossing(end_sample)
                
                stitched_audio.append(audio_data[start_zc:end_zc])
            
            final_audio = np.concatenate(stitched_audio) if stitched_audio else np.array([], dtype=np.int16)
            
            with wave.open(output_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(framerate)
                wf.writeframes(final_audio.tobytes())
                
            if os.path.exists(tmp_full_wav):
                os.remove(tmp_full_wav)
                
            logger.info(f"[Clip {self.idx}] Audio extracted and stitched successfully.")
            return True
        except Exception as e:
            logger.error(f"[Clip {self.idx}] Audio extraction/stitching failed: {e}")
            self.errors.append(f"Audio extract/stitch failed: {e}")
            return False

    def _probe_dimensions(self, video_path: str) -> Tuple[int, int]:
        """Queries width and height of the video file."""
        try:
            probe_cmd = [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            dims = result.stdout.strip().split(",")
            return int(dims[0]), int(dims[1])
        except Exception as e:
            logger.error(f"[Clip {self.idx}] ffprobe failed: {e}")
            return 0, 0

    def _check_av1_nvenc_support(self) -> bool:
        """Verifies if av1_nvenc is available and working on the host GPU."""
        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.1",
                "-c:v", "av1_nvenc",
                "-f", "null", "-"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            return res.returncode == 0
        except Exception:
            return False

    # ─── Thread 1: Decode Thread ─────────────────────────────────────────────

    def _decode_thread(self, source_path: str, src_w: int, src_h: int):
        try:
            slot = 0
            # Identify continuous intervals across segments to eliminate FFmpeg restarts and 1-frame keyframe seek flashes
            continuous_intervals = []
            for seg in self.clip.segments:
                seg_start = seg.segment_start if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
                seg_end = seg.segment_end if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
                if not continuous_intervals:
                    continuous_intervals.append([seg_start, seg_end])
                else:
                    last = continuous_intervals[-1]
                    if abs(seg_start - last[1]) <= 0.05:
                        last[1] = max(last[1], seg_end)
                    else:
                        continuous_intervals.append([seg_start, seg_end])

            for int_start, int_end in continuous_intervals:
                cmd = [
                    "ffmpeg", "-y",
                    "-hwaccel", "nvdec",
                    "-extra_hw_frames", "2",
                    "-ss", f"{int_start:.3f}",
                    "-to", f"{int_end:.3f}",
                    "-i", source_path,
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "pipe:1"
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                frame_bytes = src_w * src_h * 3
                
                while not self.stop_event.is_set():
                    data = proc.stdout.read(frame_bytes)
                    if not data or len(data) < frame_bytes:
                        break

                    # Block until a slot is free in the ring buffer
                    acquired = False
                    while not self.stop_event.is_set():
                        if self.empty_slots.acquire(timeout=0.1):
                            acquired = True
                            break
                    if not acquired or self.stop_event.is_set():
                        break

                    arr = np.frombuffer(data, dtype=np.uint8).reshape(src_h, src_w, 3)
                    cpu_tensor = torch.from_numpy(arr)

                    # Copy to GPU slot
                    self.decode_slots[slot].copy_(cpu_tensor, non_blocking=True)

                    # Record completion event
                    self.decode_events[slot].record()

                    # Pass index to inference thread
                    self.decode_to_inference.put(slot)

                    slot = (slot + 1) % RING_BUFFER_SIZE
                    self.frame_count += 1

                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                if self.stop_event.is_set():
                    break

        except Exception as e:
            logger.critical(f"HARDWARE TRACE - Decode thread failed: {e}", exc_info=True)
            self.errors.append(f"HARDWARE TRACE - Decode thread error: {e}")
            self.stop_event.set()
        finally:
            self.decode_to_inference.put(None)  # Sentinel
            self.decode_done.set()

    # ─── Thread 2: Inference Thread (Global Trajectory + Premultiplied Alpha) ──

    def _inference_thread(self, src_w: int, src_h: int, target_w: int, target_h: int):
        try:
            render_stream = torch.cuda.Stream() if self.device.type == "cuda" else None
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device.index if self.device.index is not None else 0)

            # Static base crop calculation
            base_crop_w = float(src_h) * (float(target_w) / float(target_h))
            base_crop_h = float(src_h)
            half_w = base_crop_w / 2.0
            half_h = base_crop_h / 2.0

            # Destination transform points (target 1080x1920 canvas)
            dst_pts = torch.tensor([
                [0.0, 0.0],
                [float(target_w), 0.0],
                [float(target_w), float(target_h)],
                [0.0, float(target_h)]
            ], dtype=torch.float32, device=self.device).unsqueeze(0)

            frame_idx = 0
            target_fps = float(self.job.target_fps)

            while not self.stop_event.is_set():
                try:
                    slot = self.decode_to_inference.get(timeout=0.2)
                except queue.Empty:
                    if self.decode_done.is_set() and self.decode_to_inference.empty():
                        break
                    continue
                if slot is None:
                    break

                if render_stream:
                    render_stream.wait_event(self.decode_events[slot])
                    torch.cuda.set_stream(render_stream)

                stitched_time = frame_idx / target_fps
                
                # Map stitched time back to original video timeline for subtitle alignment
                original_time = self.clip.segments[0].segment_start if hasattr(self.clip.segments[0], "segment_start") else self.clip.segments[0].get("segment_start", 0.0)
                seg_offset = 0.0
                current_seg = self.clip.segments[0]
                for seg in self.clip.segments:
                    seg_s = seg.segment_start if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
                    seg_e = seg.segment_end if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
                    dur = seg_e - seg_s
                    if seg_offset <= stitched_time <= seg_offset + dur + 0.05:
                        original_time = seg_s + (stitched_time - seg_offset)
                        current_seg = seg
                        break
                    seg_offset += dur

                # Get pre-computed smoothed crop center from global trajectory optimizer
                if self.crop_path is not None and len(self.crop_path) > 0:
                    idx_clamped = min(frame_idx, len(self.crop_path) - 1)
                    cx_norm = float(self.crop_path[idx_clamped, 0])
                    cy_norm = float(self.crop_path[idx_clamped, 1])
                    w_crop_norm = float(self.crop_path[idx_clamped, 2]) if self.crop_path.shape[1] >= 4 else None
                    h_crop_norm = float(self.crop_path[idx_clamped, 3]) if self.crop_path.shape[1] >= 4 else None
                else:
                    cx_norm, cy_norm, w_crop_norm, h_crop_norm = 0.5, 0.5, None, None

                # Source frame as float [1, C, H, W] in [0, 1]
                frame = self.decode_slots[slot].permute(2, 0, 1).unsqueeze(0).float() / 255.0

                # Extract topic and speaker coordinates from grounded_target
                target_cx = cx_norm
                target_cy = cy_norm if cy_norm is not None else 0.5
                spk_cx = cx_norm

                gt = getattr(current_seg, "grounded_target", None) if hasattr(current_seg, "grounded_target") else current_seg.get("grounded_target", None)
                if gt is not None:
                    p_bbox = getattr(gt, "primary_bbox", None) if not isinstance(gt, dict) else gt.get("primary_bbox")
                    if p_bbox and len(p_bbox) >= 2:
                        target_cx = float(p_bbox[0])
                        target_cy = float(p_bbox[1])
                    s_bbox = getattr(gt, "secondary_bbox", None) if not isinstance(gt, dict) else gt.get("secondary_bbox")
                    if s_bbox and len(s_bbox) >= 2:
                        spk_cx = float(s_bbox[0])

                split_cfg = getattr(current_seg, "split_screen_config", None) if hasattr(current_seg, "split_screen_config") else current_seg.get("split_screen_config", None)

                # Prioritize Segment Layout Mode from Autonomous Self-Correction Loop
                raw_layout = getattr(current_seg, "layout_mode", None) if hasattr(current_seg, "layout_mode") else current_seg.get("layout_mode", None)
                if hasattr(raw_layout, "value"):
                    raw_val = raw_layout.value
                else:
                    raw_val = str(raw_layout) if raw_layout is not None else None
                norm_layout = str(raw_val).upper().strip() if raw_val else None

                # Determine Shot-Level Adaptive Framing Mode
                shot_mode = None
                if self.shot_framing_schedule:
                    for s_start, s_end, s_mode in self.shot_framing_schedule:
                        if s_start <= stitched_time < s_end:
                            shot_mode = s_mode
                            break

                # Query active face bounding box for comfortable headroom
                current_face_bbox = None
                if hasattr(self, "mapped_spatial_events") and self.mapped_spatial_events:
                    for pe in self.mapped_spatial_events:
                        ts = pe.get("timestamp", 0.0) if isinstance(pe, dict) else getattr(pe, "timestamp", 0.0)
                        if abs(ts - stitched_time) <= 0.35:
                            lbl = str(pe.get("label", "") if isinstance(pe, dict) else getattr(pe, "label", "")).lower().strip()
                            if lbl in HUMAN_DETECTION_LABELS or lbl in ("face", "head", "person", "upper_body"):
                                bbox = pe.get("bbox", None) if isinstance(pe, dict) else getattr(pe, "bbox", None)
                                if bbox and len(bbox) >= 4:
                                    current_face_bbox = bbox
                                    break

                # Self-Aware Universal Framing Hierarchy:
                # 1. Uncertainty Fallback: Emergency fallback strictly for true unresolvable dark noise / corrupted frames
                if shot_mode in ("CANVAS_UNCERTAINTY_FIT", "SHOT_UNCERTAINTY_FIT") or (norm_layout and ("UNCERTAINTY" in norm_layout or "CANVAS_UNCERTAINTY_FIT" in norm_layout)):
                    rendered_tensor = render_canvas_uncertainty_fit(
                        frame=frame,
                        target_w=target_w,
                        target_h=target_h
                    )
                # 2. Universal Monitor & Graphic Display: Complete 9:16 vertical crop centered on demonstrated UI/screen
                elif shot_mode in ("SHOT_GRAPHIC_FIT", "CANVAS_GRAPHIC_FIT") or (norm_layout and ("FOCAL_DISPLAY" in norm_layout or "CANVAS_GRAPHIC_FIT" in norm_layout or "CONTENT_FIT" in norm_layout or norm_layout in ("BROLL", "GRAPHIC", "FULL_SCREEN", "B_ROLL") or getattr(current_seg, "focal_target", None) == "FOCAL_DISPLAY")):
                    rendered_tensor = render_full_frame_topic_closeup(
                        frame=frame,
                        cx_norm=target_cx if target_cx is not None else 0.5,
                        cy_norm=target_cy if target_cy is not None else 0.5,
                        w_crop_norm=w_crop_norm,
                        h_crop_norm=h_crop_norm,
                        target_w=target_w,
                        target_h=target_h
                    )
                # 3. Dual-Pane Split Screen: Strictly requires verified distinct entities with >= 0.28 separation.
                #    If no verified separation, never duplicate the screen; route to 9:16 topic or speaker portrait.
                elif norm_layout and ("SPLIT" in norm_layout or norm_layout == "DUAL_PANE_SPLIT_SCREEN"):
                    is_screen_focal = getattr(current_seg, "focal_target", None) in ("FOCAL_DISPLAY", "SCREEN")
                    if abs(spk_cx - target_cx) >= 0.28 and not is_screen_focal:
                        rendered_tensor = render_dual_pane_split_screen(
                            frame=frame,
                            speaker_cx=spk_cx,
                            topic_cx=target_cx,
                            topic_cy=target_cy,
                            target_w=target_w,
                            target_h=target_h,
                            config=split_cfg
                        )
                    elif shot_mode == "SHOT_PORTRAIT_PAN":
                        rendered_tensor = render_smart_portrait_pan(
                            frame=frame,
                            cx_norm=cx_norm,
                            target_w=target_w,
                            target_h=target_h,
                            face_bbox=current_face_bbox
                        )
                    else:
                        rendered_tensor = render_full_frame_topic_closeup(
                            frame=frame,
                            cx_norm=target_cx,
                            cy_norm=target_cy,
                            w_crop_norm=w_crop_norm,
                            h_crop_norm=h_crop_norm,
                            target_w=target_w,
                            target_h=target_h
                        )
                # 4. High-Confidence Human Speaker Portrait: Centered horizontally with OSQP trajectory & adaptive headroom
                elif shot_mode == "SHOT_PORTRAIT_PAN" or (norm_layout and ("SPEAKER" in norm_layout or norm_layout in ("FULL_FRAME_SPEAKER_TIGHT", "SMART_PORTRAIT_PAN", "SHOT_PORTRAIT_PAN"))):
                    rendered_tensor = render_smart_portrait_pan(
                        frame=frame,
                        cx_norm=cx_norm,
                        target_w=target_w,
                        target_h=target_h,
                        face_bbox=current_face_bbox
                    )
                # 5. Non-Screen Physical Object Close-up
                elif norm_layout and ("TOPIC" in norm_layout or "CLOSEUP" in norm_layout or norm_layout in ("FULL_FRAME_TOPIC_CLOSEUP", "TOPIC_FOCAL")):
                    rendered_tensor = render_full_frame_topic_closeup(
                        frame=frame,
                        cx_norm=target_cx,
                        cy_norm=target_cy,
                        w_crop_norm=w_crop_norm,
                        h_crop_norm=h_crop_norm,
                        target_w=target_w,
                        target_h=target_h
                    )
                # 6. Default Fallback
                else:
                    rendered_tensor = render_smart_portrait_pan(
                        frame=frame,
                        cx_norm=cx_norm,
                        target_w=target_w,
                        target_h=target_h,
                        face_bbox=current_face_bbox
                    )
                # Uniform Subtitle Anchor: Y = 65% (Y ≈ 1248px)
                sub_baseline_y = float(target_h) * 0.65

                processed = (rendered_tensor.squeeze(0).permute(1, 2, 0) * 255.0).byte()

                # Subtitle Compositor (Premultiplied Alpha Porter-Duff Over)
                if self.subtitle_compositor and self.job.transcript_words:
                    frame_float = processed.unsqueeze(0).float() / 255.0
                    w_words = self.job.transcript_words
                    w_first = w_words[0]
                    w_min = float(w_first.get("start", 0.0) if isinstance(w_first, dict) else getattr(w_first, "start", 0.0))
                    first_seg_s = getattr(self.clip.segments[0], "segment_start", 0.0) if hasattr(self.clip.segments[0], "segment_start") else self.clip.segments[0].get("segment_start", 0.0)

                    # Accurately detect whether word timestamps are in absolute video time or 0-based
                    if w_min >= (first_seg_s - 5.0) and w_min > 5.0:
                        lookup_time = original_time
                    else:
                        lookup_time = stitched_time

                    processed_with_subs = self.subtitle_compositor.composite_frame(
                        frame=frame_float,
                        timestamp=lookup_time,
                        words=self.job.transcript_words,
                        target_w=target_w,
                        target_h=target_h,
                        baseline_y=sub_baseline_y
                    )
                    processed_final = (processed_with_subs.squeeze(0) * 255.0).byte().clone()
                else:
                    processed_final = processed

                # Write to target render slot
                self.render_slots[slot].copy_(processed_final, non_blocking=True)
                
                # Record inference completion event
                self.inference_events[slot].record()
                self.inference_to_encode.put(slot)

                frame_idx += 1
                
        except Exception as e:
            logger.critical(f"HARDWARE TRACE - Inference thread failed: {e}", exc_info=True)
            self.errors.append(f"HARDWARE TRACE - Inference thread error: {e}")
            self.stop_event.set()
        finally:
            self.inference_to_encode.put(None)  # Sentinel
            for _ in range(RING_BUFFER_SIZE):
                try:
                    self.empty_slots.release()
                except ValueError:
                    pass
            self.inference_done.set()

    # ─── Thread 3: Encode Thread ─────────────────────────────────────────────

    def _encode_thread(self, wav_path: str, output_path: str, target_w: int, target_h: int, codec_to_use: str):
        try:
            total_duration = sum([
                (s.get("segment_end", 0.0) - s.get("segment_start", 0.0)) if isinstance(s, dict)
                else (s.segment_end - s.segment_start)
                for s in self.clip.segments
            ])
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{target_w}x{target_h}", "-r", str(self.job.target_fps),
                "-i", "pipe:0",
                "-i", wav_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", codec_to_use
            ]

            if codec_to_use == "av1_nvenc":
                cmd += ["-preset", "p7", "-tune", "hq", "-b:v", "4M", "-maxrate", "6M", "-bufsize", "8M"]
            else:
                cmd += ["-preset", "medium", "-crf", "20"]

            if total_duration > 0:
                cmd += ["-t", f"{total_duration:.3f}"]

            cmd += [
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                output_path
            ]

            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            while not self.stop_event.is_set():
                try:
                    slot = self.inference_to_encode.get(timeout=0.2)
                except queue.Empty:
                    if self.inference_done.is_set() and self.inference_to_encode.empty():
                        break
                    continue
                if slot is None:
                    break

                # Wait for inference processing to finish on GPU
                self.inference_events[slot].synchronize()

                # Copy processed vertical frame to host CPU and write to pipe
                cpu_bytes = self.render_slots[slot].cpu().numpy().tobytes()
                try:
                    proc.stdin.write(cpu_bytes)
                except (BrokenPipeError, IOError):
                    self.stop_event.set()
                    self.empty_slots.release()
                    break

                # Free the slot back to decode thread
                self.empty_slots.release()

            try:
                proc.stdin.close()
            except (BrokenPipeError, IOError):
                pass
            stderr_output = proc.stderr.read().decode('utf-8', errors='ignore')
            proc.wait()

            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg encoding failed with returncode {proc.returncode}: {stderr_output}")

        except Exception as e:
            logger.critical(f"HARDWARE TRACE - Encode thread failed: {e}", exc_info=True)
            self.errors.append(f"HARDWARE TRACE - Encode thread error: {e}")
            self.stop_event.set()
        finally:
            self.stop_event.set()
            for _ in range(RING_BUFFER_SIZE):
                try:
                    self.empty_slots.release()
                except ValueError:
                    pass
            self.encode_done.set()

    def _generate_thumbnail(self, source_path: str):
        """Extracts a thumbnail frame at 25% clip duration mapped to original timeline."""
        try:
            duration = sum([
                (s.get("segment_end", 0.0) - s.get("segment_start", 0.0)) if isinstance(s, dict)
                else (s.segment_end - s.segment_start)
                for s in self.clip.segments
            ])
            thumb_time_stitched = duration * 0.25
            
            # Map stitched offset to original source video timeline
            thumb_time_original = self.clip.segments[0].segment_start if not isinstance(self.clip.segments[0], dict) else self.clip.segments[0].get("segment_start", 0.0)
            current_offset = 0.0
            for seg in self.clip.segments:
                seg_start = seg.segment_start if not isinstance(seg, dict) else seg.get("segment_start", 0.0)
                seg_end = seg.segment_end if not isinstance(seg, dict) else seg.get("segment_end", 0.0)
                dur = seg_end - seg_start
                if current_offset <= thumb_time_stitched <= current_offset + dur:
                    thumb_time_original = seg_start + (thumb_time_stitched - current_offset)
                    break
                current_offset += dur

            thumb_path = os.path.join(self.job.output_dir, f"thumbnail_{self.idx}.jpg")

            thumb_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{thumb_time_original:.3f}",
                "-i", source_path,
                "-vframes", "1",
                "-q:v", "2",
                thumb_path
            ]
            subprocess.run(thumb_cmd, check=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(thumb_path):
                logger.info(f"[Clip {self.idx}] Thumbnail generated: {thumb_path}")
            else:
                logger.warning(f"[Clip {self.idx}] Thumbnail generation failed silently.")
        except Exception as e:
            logger.warning(f"[Clip {self.idx}] Thumbnail generation error: {e}")


# ─── Render Spoke ────────────────────────────────────────────────────────────

class RenderSpoke:
    """NATS consumer that renders short-form clips using the three-thread GPU pipeline."""

    def __init__(self, nats_url: str):
        self.nats_url = nats_url
        self.nc = None
        self.js = None

    async def run(self):
        """Connects to NATS JetStream and listens for rendering jobs."""
        logger.info(f"Connecting to NATS at {self.nats_url}...")
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()

        # Durable pull subscription with hardened config
        sub = await self.js.subscribe(
            subject=VIDEO_RENDER,
            durable="render_spoke",
            config=ConsumerConfig(ack_wait=600.0, max_ack_pending=12)
        )

        logger.info(f"Subscribed to '{VIDEO_RENDER}' as durable subscription. Awaiting jobs...")

        try:
            async for msg in sub.messages:
                try:
                    payload = json.loads(msg.data.decode("utf-8"))
                    job = RenderJob(**payload)

                    # Execute the three-thread render pipeline
                    pipeline = ThreeThreadRenderPipeline(job)
                    success, output_path = await asyncio.to_thread(pipeline.execute)

                    # Publish status report
                    status_payload = {
                        "source_path": job.source_path,
                        "clip_index": job.clip_index,
                        "status": "success" if success else "failed",
                        "error": "" if success else "; ".join(pipeline.errors) or "Render failed."
                    }

                    await self.js.publish(PIPELINE_STATUS,
                                         json.dumps(status_payload).encode("utf-8"))
                    logger.info(f"Published render status for clip {job.clip_index}: "
                               f"{status_payload['status']}")

                    await msg.ack()
                except Exception as e:
                    logger.error(f"Error processing render job: {str(e)}", exc_info=True)
                    await msg.nak()
        except asyncio.CancelledError:
            logger.info("Render Spoke cancelled, closing...")
        finally:
            if self.nc:
                await self.nc.close()


if __name__ == "__main__":
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    spoke = RenderSpoke(nats_url)
    try:
        asyncio.run(spoke.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
