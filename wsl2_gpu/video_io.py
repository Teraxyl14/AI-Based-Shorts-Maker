"""
Project Aether V3 — Zero-Copy Hardware Video I/O

Provides GPU-native video decoding (NVDEC) and encoding (NVENC) via PyNvVideoCodec
with DLPack zero-copy tensor interchange. Eliminates all CPU FFmpeg pipes, NumPy arrays,
and host-device PCIe copies from the hot render path.

Automatic fallback to FFmpeg subprocess pipes when PyNvVideoCodec or WSL2 driver
libraries are unavailable.
"""

import os
import logging
import subprocess
from typing import Optional, Tuple, List, Iterator

import numpy as np
import torch

logger = logging.getLogger("AetherVideoIO")

# ─── PyNvVideoCodec availability probe ──────────────────────────────────────

_NVC_AVAILABLE = False
nvc = None

try:
    import PyNvVideoCodec as _nvc_mod
    nvc = _nvc_mod
    _NVC_AVAILABLE = True
    logger.info("PyNvVideoCodec loaded — zero-copy NVDEC/NVENC enabled.")
except ImportError:
    logger.warning(
        "PyNvVideoCodec not available. Falling back to FFmpeg pipe I/O. "
        "Install nvidia-pynvvideocodec and mount /usr/lib/wsl/lib for zero-copy GPU video."
    )


# ═════════════════════════════════════════════════════════════════════════════
# DECODER: NVDEC → DLPack → PyTorch CUDA Tensor (zero-copy)
# ═════════════════════════════════════════════════════════════════════════════

class NvDecoder:
    """Zero-copy GPU video decoder using NVDEC via PyNvVideoCodec.

    Decodes video frames directly into planar RGB CUDA tensors [3, H, W]
    using DLPack interchange — no host RAM copies.
    """

    def __init__(self, input_path: str, gpu_id: int = 0):
        if not _NVC_AVAILABLE:
            raise RuntimeError("PyNvVideoCodec is not installed. Use FFmpegDecoder instead.")

        self.input_path = input_path
        self.gpu_id = gpu_id
        self._decoder = nvc.SimpleDecoder(
            input_path,
            gpu_id=gpu_id,
            output_color_type=nvc.OutputColorType.RGBP
        )
        self._exhausted = False
        logger.info(f"NvDecoder initialized: {input_path} on GPU:{gpu_id}")

    def decode_frames(self) -> Iterator[torch.Tensor]:
        """Yields decoded frames as [3, H, W] float32 CUDA tensors in [0, 1].

        Each frame is zero-copy transferred from NVDEC output surface
        to a PyTorch CUDA tensor via DLPack.
        """
        while not self._exhausted:
            try:
                raw_frame = next(self._decoder)
                # DLPack zero-copy: NVDEC surface → PyTorch CUDA tensor
                tensor = torch.from_dlpack(raw_frame)
                # Normalize uint8 [0, 255] → float32 [0, 1]
                if tensor.dtype == torch.uint8:
                    tensor = tensor.float() / 255.0
                yield tensor
            except StopIteration:
                self._exhausted = True
                return

    def decode_segment(self, start_s: float, end_s: float,
                       fps: float = 30.0) -> List[torch.Tensor]:
        """Decodes a specific time segment, returning list of [3, H, W] tensors.

        Note: PyNvVideoCodec SimpleDecoder decodes sequentially from the start.
        For segment-based decoding, we skip frames before start_s and stop after end_s.
        """
        frames = []
        start_frame = int(start_s * fps)
        end_frame = int(end_s * fps)
        frame_idx = 0

        for tensor in self.decode_frames():
            if frame_idx >= end_frame:
                break
            if frame_idx >= start_frame:
                frames.append(tensor.clone())
            frame_idx += 1

        logger.info(f"NvDecoder segment [{start_s:.2f}s - {end_s:.2f}s]: {len(frames)} frames")
        return frames

    def close(self):
        """Release decoder resources."""
        self._decoder = None
        self._exhausted = True


class FFmpegDecoder:
    """Fallback CPU-based video decoder using FFmpeg subprocess pipe.

    Reads raw RGB24 frames from FFmpeg stdout and uploads to GPU.
    """

    def __init__(self, input_path: str, gpu_id: int = 0):
        self.input_path = input_path
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self._width, self._height = self._probe_dimensions()
        logger.info(f"FFmpegDecoder initialized: {input_path} ({self._width}x{self._height}) → {self.device}")

    def _probe_dimensions(self) -> Tuple[int, int]:
        """Queries video dimensions via ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                self.input_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            dims = result.stdout.strip().split(",")
            return int(dims[0]), int(dims[1])
        except Exception as e:
            logger.error(f"ffprobe failed: {e}")
            return 1920, 1080  # Sensible default

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def decode_segment(self, start_s: float, end_s: float,
                       fps: float = 30.0) -> List[torch.Tensor]:
        """Decodes a segment via FFmpeg pipe and returns [3, H, W] CUDA tensors."""
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "nvdec",
            "-extra_hw_frames", "2",
            "-ss", f"{start_s:.3f}",
            "-to", f"{end_s:.3f}",
            "-i", self.input_path,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_bytes = self._width * self._height * 3
        frames = []

        while True:
            data = proc.stdout.read(frame_bytes)
            if not data or len(data) < frame_bytes:
                break
            arr = np.frombuffer(data, dtype=np.uint8).reshape(self._height, self._width, 3)
            # Convert HWC uint8 → CHW float32 [0, 1] on GPU
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float().to(self.device) / 255.0
            frames.append(tensor)

        proc.terminate()
        proc.wait()
        logger.info(f"FFmpegDecoder segment [{start_s:.2f}s - {end_s:.2f}s]: {len(frames)} frames")
        return frames

    def decode_frames_pipe(self, start_s: float = 0.0, end_s: Optional[float] = None):
        """Generator that yields frames one-by-one from FFmpeg pipe.

        Each frame is [3, H, W] float32 on self.device.
        """
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "nvdec",
            "-extra_hw_frames", "2",
        ]
        if start_s > 0:
            cmd += ["-ss", f"{start_s:.3f}"]
        if end_s is not None:
            cmd += ["-to", f"{end_s:.3f}"]
        cmd += [
            "-i", self.input_path,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_bytes = self._width * self._height * 3

        try:
            while True:
                data = proc.stdout.read(frame_bytes)
                if not data or len(data) < frame_bytes:
                    break
                arr = np.frombuffer(data, dtype=np.uint8).reshape(self._height, self._width, 3)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).float().to(self.device) / 255.0
                yield tensor
        finally:
            proc.terminate()
            proc.wait()

    def close(self):
        """No persistent resources to release for FFmpeg decoder."""
        pass


# ═════════════════════════════════════════════════════════════════════════════
# ENCODER: PyTorch CUDA Tensor → NVENC (zero-copy) or FFmpeg pipe fallback
# ═════════════════════════════════════════════════════════════════════════════

class NvEncoder:
    """Zero-copy GPU video encoder using NVENC via PyNvVideoCodec.

    Accepts planar [3, H, W] CUDA tensors and encodes directly on GPU
    without host memory copies. Tries AV1 first, falls back to H.264.
    """

    def __init__(self, output_path: str, width: int, height: int,
                 fps: int = 30, gpu_id: int = 0, bitrate: str = "8M"):
        if not _NVC_AVAILABLE:
            raise RuntimeError("PyNvVideoCodec is not installed. Use FFmpegEncoder instead.")

        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.gpu_id = gpu_id
        self._encoder = None
        self._codec_used = None
        self._output_file = None

        # Try AV1 first, fallback to H.264
        for codec in ["av1", "h264"]:
            try:
                self._encoder = nvc.CreateEncoder(
                    width=width,
                    height=height,
                    fmt="RGBP",
                    usecpuinputbuffer=False,
                    codec=codec,
                    preset="p7",
                    tuning_info="high_quality",
                    rc="vbr",
                    bitrate=bitrate,
                    fps=fps,
                    gpu_id=gpu_id
                )
                self._codec_used = codec
                logger.info(f"NvEncoder initialized: {width}x{height} @ {fps}fps, codec={codec}")
                break
            except Exception as e:
                logger.warning(f"NVENC codec '{codec}' init failed: {e}")
                continue

        if self._encoder is None:
            raise RuntimeError("All NVENC codec initializations failed.")

        self._output_file = open(output_path, "wb")

    def encode_frame(self, frame: torch.Tensor):
        """Encodes a single [3, H, W] planar CUDA tensor.

        Frame must be uint8 [0, 255] or float32 [0, 1] (auto-converted).
        """
        if frame.dtype == torch.float32:
            frame = (frame * 255.0).clamp(0, 255).byte()

        # Ensure planar CHW layout
        if frame.dim() == 4:
            frame = frame.squeeze(0)
        assert frame.shape == (3, self.height, self.width), \
            f"Expected [{3}, {self.height}, {self.width}], got {frame.shape}"

        encoded_bytes = self._encoder.Encode(frame)
        if encoded_bytes:
            self._output_file.write(bytes(encoded_bytes))

    def finalize(self):
        """Flushes encoder and closes output file."""
        if self._encoder is not None:
            try:
                # Flush remaining frames
                flushed = self._encoder.Flush()
                if flushed and self._output_file:
                    self._output_file.write(bytes(flushed))
            except Exception as e:
                logger.warning(f"NvEncoder flush error: {e}")

        if self._output_file is not None:
            self._output_file.close()
            self._output_file = None

        self._encoder = None
        logger.info(f"NvEncoder finalized: {self.output_path} (codec={self._codec_used})")

    def __del__(self):
        self.finalize()


class FFmpegEncoder:
    """Fallback CPU-pipe video encoder using FFmpeg subprocess.

    Accepts [3, H, W] or [H, W, 3] tensors, converts to RGB24 bytes,
    and pipes to FFmpeg for encoding.
    """

    def __init__(self, output_path: str, width: int, height: int,
                 fps: int = 30, codec: str = "av1_nvenc",
                 audio_path: Optional[str] = None):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self._codec = codec
        self._proc = None

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
        ]

        if audio_path and os.path.exists(audio_path):
            cmd += ["-i", audio_path, "-map", "0:v", "-map", "1:a"]

        cmd += ["-c:v", codec]

        if codec == "av1_nvenc":
            cmd += ["-preset", "p7", "-tune", "hq", "-b:v", "8M",
                    "-maxrate", "10M", "-bufsize", "12M"]
        elif codec == "libx264":
            cmd += ["-preset", "medium", "-crf", "20"]
        else:
            cmd += ["-preset", "medium", "-crf", "20"]

        if audio_path:
            cmd += ["-c:a", "aac", "-b:a", "128k"]

        cmd += ["-movflags", "+faststart", output_path]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"FFmpegEncoder initialized: {width}x{height} @ {fps}fps, codec={codec}")

    def encode_frame(self, frame: torch.Tensor):
        """Encodes a single frame. Accepts [3, H, W] or [H, W, 3] tensor."""
        if frame.dim() == 4:
            frame = frame.squeeze(0)

        # Convert CHW → HWC if needed
        if frame.shape[0] == 3 and frame.dim() == 3:
            frame = frame.permute(1, 2, 0)

        # Convert to uint8
        if frame.dtype == torch.float32:
            frame = (frame * 255.0).clamp(0, 255).byte()

        cpu_bytes = frame.cpu().numpy().tobytes()
        self._proc.stdin.write(cpu_bytes)

    def finalize(self):
        """Closes FFmpeg pipe and waits for encoding to complete."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                stderr = self._proc.stderr.read().decode('utf-8', errors='ignore')
                self._proc.wait()
                if self._proc.returncode != 0:
                    logger.error(f"FFmpeg encoding failed (rc={self._proc.returncode}): {stderr[-500:]}")
                else:
                    logger.info(f"FFmpegEncoder finalized: {self.output_path}")
            except Exception as e:
                logger.error(f"FFmpegEncoder finalize error: {e}")
            finally:
                self._proc = None

    def __del__(self):
        self.finalize()


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY: Auto-select best available decoder/encoder
# ═════════════════════════════════════════════════════════════════════════════

def create_decoder(input_path: str, gpu_id: int = 0):
    """Creates the best available video decoder.

    Returns NvDecoder if PyNvVideoCodec is available, otherwise FFmpegDecoder.
    """
    if _NVC_AVAILABLE:
        try:
            return NvDecoder(input_path, gpu_id=gpu_id)
        except Exception as e:
            logger.warning(f"NvDecoder init failed, falling back to FFmpeg: {e}")

    return FFmpegDecoder(input_path, gpu_id=gpu_id)


def create_encoder(output_path: str, width: int, height: int,
                   fps: int = 30, gpu_id: int = 0,
                   audio_path: Optional[str] = None,
                   bitrate: str = "8M"):
    """Creates the best available video encoder.

    Returns NvEncoder if PyNvVideoCodec is available, otherwise FFmpegEncoder.
    """
    if _NVC_AVAILABLE:
        try:
            return NvEncoder(output_path, width, height, fps=fps,
                             gpu_id=gpu_id, bitrate=bitrate)
        except Exception as e:
            logger.warning(f"NvEncoder init failed, falling back to FFmpeg: {e}")

    # FFmpeg fallback: try av1_nvenc first, then libx264
    codec = "av1_nvenc"
    try:
        test_cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.1",
            "-c:v", "av1_nvenc", "-f", "null", "-"
        ]
        res = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            codec = "libx264"
    except Exception:
        codec = "libx264"

    return FFmpegEncoder(output_path, width, height, fps=fps,
                         codec=codec, audio_path=audio_path)


def probe_video_info(video_path: str) -> Tuple[int, int, float, float]:
    """Probes video for width, height, fps, and duration.

    Returns:
        (width, height, fps, duration_seconds)
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        import json
        info = json.loads(result.stdout)

        stream = info.get("streams", [{}])[0]
        width = int(stream.get("width", 1920))
        height = int(stream.get("height", 1080))

        # Parse fractional fps (e.g., "30000/1001")
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        # Duration from stream or format
        duration = float(stream.get("duration", 0.0))
        if duration <= 0:
            duration = float(info.get("format", {}).get("duration", 0.0))

        return width, height, fps, duration
    except Exception as e:
        logger.error(f"Video probe failed for {video_path}: {e}")
        return 1920, 1080, 30.0, 0.0
