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
from wsl2_gpu.trajectory_smoother import compute_crop_path, smooth_trajectory
from wsl2_gpu.subtitle_compositor import SubtitleCompositor

logger = logging.getLogger("AetherRenderSpoke")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ─── Constants ───────────────────────────────────────────────────────────────
RING_BUFFER_SIZE = 8  # Pre-allocated frame slots in the ring buffer


# ─── GPU-Resident Layout Helpers ─────────────────────────────────────────────

# ─── GPU-Resident Multi-Layout Framing Helpers ──────────────────────────────

def render_speaker_solo(frame: torch.Tensor, cx_norm: float,
                        target_w: int = 1080, target_h: int = 1920) -> torch.Tensor:
    """BRANCH A — SPEAKER_SOLO (Full 9:16 Portrait):
    - Crop full source height (ch = 1.0, H = 1080) and width cw = 0.3164 (607.5px).
    - Center horizontally on the smoothed speaker POI (cx_norm).
    - Scale to 1080 x 1920 via bilinear interpolation.
    Returns: [1, 3, target_h, target_w] float32 tensor in [0, 1].
    """
    B, C, src_h, src_w = frame.shape
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


def render_split_stack(frame: torch.Tensor, cx_norm: float, cy_norm: float,
                       speaker_cx: Optional[float] = None,
                       speaker_cy: Optional[float] = None,
                       focal_target: str = "FOCAL_DISPLAY",
                       target_w: int = 1080, target_h: int = 1920) -> torch.Tensor:
    """BRANCH B — SPLIT_STACK (Vertical Stack 50/50):
    - Canvas is divided into Top (Y: 0 -> 960) and Bottom (Y: 960 -> 1920).
    - Top Pane (Speaker):
      * Crop a region centered on the primary speaker (speaker_cx, speaker_cy).
      * Resize to fill 1080 x 960 (crop-to-fill).
    - Bottom Pane (Content / Screen / Interaction):
      * If target is widescreen monitor/gameplay/UI: scale 16:9 source to fit width
        (1080px wide by 607.5px high) centered vertically inside bottom 960px pane
        over a darkened blurred background.
      * If target is localized object/hands: crop region around hands/object and scale to fill 1080 x 960.
    - Place a subtle 2px dark divider line at Y = 960px.
    Returns: [1, 3, target_h, target_w] float32 tensor in [0, 1].
    """
    B, C, src_h, src_w = frame.shape
    split_y = target_h // 2  # 960
    canvas = torch.zeros((B, C, target_h, target_w), device=frame.device, dtype=frame.dtype)

    # ── Top Pane: Primary Speaker (Y: 0 -> split_y, W: target_w) ──
    spk_cx = speaker_cx if speaker_cx is not None else cx_norm
    if speaker_cy is not None and 0.05 <= speaker_cy <= 0.95:
        spk_cy = speaker_cy
    elif 0.10 <= cy_norm <= 0.90:
        spk_cy = cy_norm
    else:
        spk_cy = 0.40

    pane_h = split_y  # 960
    pane_w = target_w  # 1080
    crop_h_top = int(round(0.56 * src_h))
    crop_w_top = int(round(crop_h_top * (float(pane_w) / float(pane_h))))

    top_x0 = max(0, min(src_w - crop_w_top, int(round(spk_cx * src_w - crop_w_top / 2.0))))
    top_x1 = top_x0 + crop_w_top
    top_y0 = max(0, min(src_h - crop_h_top, int(round(spk_cy * src_h - crop_h_top * 0.35))))
    top_y1 = top_y0 + crop_h_top

    top_crop = frame[:, :, top_y0:top_y1, top_x0:top_x1]
    top_resized = torch.nn.functional.interpolate(
        top_crop, size=(pane_h, pane_w), mode='bilinear', align_corners=False
    )
    canvas[:, :, 0:split_y, :] = top_resized

    # ── Bottom Pane: Content / Screen / Interaction (Y: split_y -> target_h, W: target_w) ──
    ft_upper = str(focal_target).upper()
    if ft_upper == "HELD_OBJECT":
        # Localized object/hands crop
        crop_h_bot = int(round(0.56 * src_h))
        crop_w_bot = int(round(crop_h_bot * (float(pane_w) / float(pane_h))))
        bot_x0 = max(0, min(src_w - crop_w_bot, int(round(cx_norm * src_w - crop_w_bot / 2.0))))
        bot_x1 = bot_x0 + crop_w_bot
        bot_y0 = max(0, min(src_h - crop_h_bot, int(round(cy_norm * src_h - crop_h_bot / 2.0))))
        bot_y1 = bot_y0 + crop_h_bot
        bot_crop = frame[:, :, bot_y0:bot_y1, bot_x0:bot_x1]
        bot_resized = torch.nn.functional.interpolate(
            bot_crop, size=(pane_h, pane_w), mode='bilinear', align_corners=False
        )
        canvas[:, :, split_y:target_h, :] = bot_resized
    else:
        # Widescreen monitor / UI / gameplay: fit 1080 width over darkened blurred background
        # Background: downsampled 4x Gaussian blur, darkened to 40%
        bg_down = torch.nn.functional.interpolate(frame, size=(pane_h // 4, pane_w // 4), mode='bilinear', align_corners=False)
        if kornia is not None:
            bg_blurred_down = kornia.filters.gaussian_blur2d(bg_down, kernel_size=(25, 25), sigma=(8.0, 8.0))
        else:
            bg_blurred_down = bg_down
        bg_bot = torch.nn.functional.interpolate(bg_blurred_down, size=(pane_h, pane_w), mode='bilinear', align_corners=False) * 0.40
        canvas[:, :, split_y:target_h, :] = bg_bot

        # Foreground: uncropped 16:9 frame scaled to width 1080 (height = 608px)
        fg_w = pane_w  # 1080
        fg_h = int(round(float(src_h) * (float(pane_w) / float(src_w))))  # 608px
        fg_bot = torch.nn.functional.interpolate(frame, size=(fg_h, fg_w), mode='bilinear', align_corners=False)

        # Center vertically inside bottom pane
        margin_y = (pane_h - fg_h) // 2  # (960 - 608) // 2 = 176px
        bot_y0 = split_y + margin_y      # 960 + 176 = 1136
        bot_y1 = bot_y0 + fg_h           # 1136 + 608 = 1744
        canvas[:, :, bot_y0:bot_y1, :] = fg_bot

    # ── Subtle 2px Dark Divider Line at Y = 959..961 ──
    div_y0 = max(0, split_y - 1)
    div_y1 = min(target_h, split_y + 1)
    canvas[:, :, div_y0:div_y1, :] = 0.08

    return canvas


def render_content_fit(frame: torch.Tensor,
                       target_w: int = 1080, target_h: int = 1920) -> torch.Tensor:
    """BRANCH C — CONTENT_FIT (Full Graphic Preservation):
    - Background: Scale the 16:9 source to cover 1080 x 1920, downsample 4x,
      apply Gaussian blur (sigma=8.0), and darken by 40% (bg * 0.60).
    - Foreground: Scale the uncropped 16:9 frame to 1080px width (1080 x 607.5)
      and paste centered vertically at Y = 656px.
    - Ensures 100% of text on slides, Wikipedia, or wide documents remains readable.
    Returns: [1, 3, target_h, target_w] float32 tensor in [0, 1].
    """
    B, C, src_h, src_w = frame.shape

    # Layer 1: Blurred, darkened cover background
    scale_factor = float(target_h) / float(src_h)
    bg_w = int(src_w * scale_factor)
    bg_scaled = torch.nn.functional.interpolate(frame, size=(target_h, bg_w), mode='bilinear', align_corners=False)

    start_x = (bg_w - target_w) // 2
    bg_cropped = bg_scaled[:, :, :, start_x:start_x + target_w]

    bg_down = torch.nn.functional.interpolate(bg_cropped, size=(target_h // 4, target_w // 4), mode='bilinear', align_corners=False)
    if kornia is not None:
        bg_blurred_down = kornia.filters.gaussian_blur2d(bg_down, kernel_size=(25, 25), sigma=(8.0, 8.0))
    else:
        bg_blurred_down = bg_down
    bg_blurred = torch.nn.functional.interpolate(bg_blurred_down, size=(target_h, target_w), mode='bilinear', align_corners=False)
    layer1 = bg_blurred * 0.60

    # Layer 2: Full-fidelity uncropped 16:9 foreground
    fg_w = target_w
    fg_h = int(round(float(src_h) * (float(target_w) / float(src_w))))  # 608px
    layer2 = torch.nn.functional.interpolate(frame, size=(fg_h, fg_w), mode='bilinear', align_corners=False)

    start_y = (target_h - fg_h) // 2  # (1920 - 608) // 2 = 656px
    processed = layer1.clone()
    processed[:, :, start_y:start_y + fg_h, :] = layer2

    return processed


def render_broll_pillarbox(source_tensor: torch.Tensor, target_canvas_shape: tuple) -> torch.Tensor:
    """Backward-compatible wrapper for CONTENT_FIT layout helper."""
    tgt_h, tgt_w = target_canvas_shape
    canvas = render_content_fit(source_tensor, target_w=tgt_w, target_h=tgt_h)
    return (canvas.squeeze(0).permute(1, 2, 0) * 255.0).byte()


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
        self.speaker_path: Optional[np.ndarray] = None

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
            total_frames = int(total_duration * self.job.target_fps) + 30
            
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
                    logger.info(f"[Clip {self.idx}] Computing global trajectory path for {total_frames} frames...")
                    self.crop_path = compute_crop_path(
                        spatial_events=mapped_events,
                        scene_changes=mapped_cuts,
                        fps=float(self.job.target_fps),
                        total_frames=total_frames,
                        crop_ratio=float(self.job.target_resolution[0]) / float(self.job.target_resolution[1]),
                        focal_targets=focal_target_intervals
                    )

            # Pre-compute smoothed speaker POI path for SPLIT_STACK top pane framing
            if self.speaker_path is None:
                self.speaker_path = compute_crop_path(
                    spatial_events=mapped_events,
                    scene_changes=mapped_cuts,
                    fps=float(self.job.target_fps),
                    total_frames=total_frames,
                    crop_ratio=1.0,
                    focal_targets=[(0, total_frames, "SPEAKER_PRIMARY")],
                    lock_cy=False
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
            for seg_idx, seg in enumerate(self.clip.segments):
                seg_start = seg.segment_start if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
                seg_end = seg.segment_end if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
                
                cmd = [
                    "ffmpeg", "-y",
                    "-hwaccel", "nvdec",
                    "-extra_hw_frames", "2",
                    "-ss", f"{seg_start:.3f}",
                    "-to", f"{seg_end:.3f}",
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
                    self.empty_slots.acquire()
                    if self.stop_event.is_set():
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
                    
                proc.terminate()
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
                slot = self.decode_to_inference.get()
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
                if self.crop_path is not None and frame_idx < len(self.crop_path):
                    cx_norm, cy_norm = self.crop_path[frame_idx]
                else:
                    cx_norm, cy_norm = 0.5, 0.5

                # Source frame as float [1, C, H, W] in [0, 1]
                frame = self.decode_slots[slot].permute(2, 0, 1).unsqueeze(0).float() / 255.0

                # Multi-Layout Semantic Framing Dispatch: SPEAKER_SOLO, SPLIT_STACK, CONTENT_FIT
                raw_layout = getattr(current_seg, "layout_mode", "SPEAKER_SOLO") if hasattr(current_seg, "layout_mode") else current_seg.get("layout_mode", "SPEAKER_SOLO")
                norm_layout = str(raw_layout).upper().strip()

                if norm_layout in ("SPLIT_STACK", "SPLIT_SCREEN"):
                    ft = getattr(current_seg, "focal_target", "FOCAL_DISPLAY") if hasattr(current_seg, "focal_target") else current_seg.get("focal_target", "FOCAL_DISPLAY")
                    if self.speaker_path is not None and frame_idx < len(self.speaker_path):
                        spk_cx, spk_cy = self.speaker_path[frame_idx]
                    else:
                        spk_cx, spk_cy = cx_norm, 0.40

                    rendered_tensor = render_split_stack(
                        frame=frame,
                        cx_norm=cx_norm,
                        cy_norm=cy_norm,
                        speaker_cx=spk_cx,
                        speaker_cy=spk_cy,
                        focal_target=str(ft),
                        target_w=target_w,
                        target_h=target_h
                    )
                    # For SPLIT_STACK: Anchor subtitles directly across the divider boundary at Y ≈ 960px
                    sub_baseline_y = 960.0
                elif norm_layout in ("CONTENT_FIT", "BROLL", "GRAPHIC", "FULL_SCREEN", "B_ROLL"):
                    rendered_tensor = render_content_fit(
                        frame=frame,
                        target_w=target_w,
                        target_h=target_h
                    )
                    # For CONTENT_FIT: Anchor subtitles at Y = 65% (Y ≈ 1248px)
                    sub_baseline_y = float(target_h) * 0.65
                else:  # SPEAKER_SOLO (Full 9:16 Portrait)
                    rendered_tensor = render_speaker_solo(
                        frame=frame,
                        cx_norm=cx_norm,
                        target_w=target_w,
                        target_h=target_h
                    )
                    # For SPEAKER_SOLO: Anchor subtitles at Y = 65% (Y ≈ 1248px)
                    sub_baseline_y = float(target_h) * 0.65

                processed = (rendered_tensor.squeeze(0).permute(1, 2, 0) * 255.0).byte()

                # Subtitle Compositor (Premultiplied Alpha Porter-Duff Over)
                if self.subtitle_compositor and self.job.transcript_words:
                    frame_float = processed.unsqueeze(0).float() / 255.0
                    processed_with_subs = self.subtitle_compositor.composite_frame(
                        frame=frame_float,
                        timestamp=original_time,
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
            self.inference_done.set()

    # ─── Thread 3: Encode Thread ─────────────────────────────────────────────

    def _encode_thread(self, wav_path: str, output_path: str, target_w: int, target_h: int, codec_to_use: str):
        try:
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

            cmd += [
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                output_path
            ]
            
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            
            while not self.stop_event.is_set():
                slot = self.inference_to_encode.get()
                if slot is None:
                    break
                
                # Wait for inference processing to finish on GPU
                self.inference_events[slot].synchronize()
                
                # Copy processed vertical frame to host CPU and write to pipe
                cpu_bytes = self.render_slots[slot].cpu().numpy().tobytes()
                proc.stdin.write(cpu_bytes)
                
                # Free the slot back to decode thread
                self.empty_slots.release()
                
            proc.stdin.close()
            stderr_output = proc.stderr.read().decode('utf-8', errors='ignore')
            proc.wait()
            
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg encoding failed with returncode {proc.returncode}: {stderr_output}")
                
        except Exception as e:
            logger.critical(f"HARDWARE TRACE - Encode thread failed: {e}", exc_info=True)
            self.errors.append(f"HARDWARE TRACE - Encode thread error: {e}")
            self.stop_event.set()
            for _ in range(RING_BUFFER_SIZE):
                try:
                    self.empty_slots.release()
                except ValueError:
                    pass
        finally:
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
