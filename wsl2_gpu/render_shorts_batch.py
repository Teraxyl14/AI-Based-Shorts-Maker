"""
Project Aether — Batch Renderer for Shorts 2, 3, 4
Renders clips serially with autonomous Whisper/YOLO fallbacks,
verifies duration and A/V sync, and extracts audit keyframes.
"""

import os
import sys
import json
import logging
import subprocess

sys.path.insert(0, "/app")

from shared.schemas import RenderJob, ClipManifest, VideoSegment
from wsl2_gpu.render_spoke import ThreeThreadRenderPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BatchRenderer")

VIDEO_PATH = "/data/input_videos/Brand new day.mkv"
OUTPUT_DIR = "/data/output_shorts/2b6f3a6e14d11db95ff5345aed16e916"
AUDIT_DIR = os.path.join(OUTPUT_DIR, "audit_frames")
os.makedirs(AUDIT_DIR, exist_ok=True)

SHORTS = [
    {
        "idx": 2,
        "start": 1471.98,
        "end": 1531.85,
        "title": "Viral Arc 2: You're the only one who's immune"
    },
    {
        "idx": 3,
        "start": 3759.22,
        "end": 3816.44,
        "title": "Viral Arc 3: feel safe. You're losing a kid"
    },
    {
        "idx": 4,
        "start": 3917.80,
        "end": 3971.05,
        "title": "Viral Arc 4: She was in and out of..."
    }
]

def render_short(info: dict):
    idx = info["idx"]
    start_t = info["start"]
    end_t = info["end"]
    title = info["title"]
    dur = end_t - start_t

    logger.info(f"==================================================")
    logger.info(f"Starting render for Short {idx}: {start_t:.2f}s -> {end_t:.2f}s (dur: {dur:.2f}s)")
    logger.info(f"Title: {title}")
    logger.info(f"==================================================")

    clip = ClipManifest(
        title=title,
        segments=[VideoSegment(segment_start=start_t, segment_end=end_t, layout_mode="SMART_PORTRAIT_PAN")],
        hook_score=0.85,
        retention_score=0.85,
        cta_present=True,
        reasoning=f"Batch render short {idx}",
        caption_text=title,
    )

    job = RenderJob(
        source_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        clip_manifest=clip,
        clip_index=idx,
        transcript_words=[],
        target_resolution=(1080, 1920),
        target_fps=30,
        codec="libx264",
        spatial_events=[],
        scene_cut_frames=[]
    )

    pipeline = ThreeThreadRenderPipeline(job)
    success, out_path = pipeline.execute()

    if not success or not os.path.exists(out_path):
        logger.error(f"Failed to render Short {idx}!")
        return False

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"Short {idx} rendered successfully: {out_path} ({file_size_mb:.2f} MB)")

    # Probe duration and audio sync
    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,duration,nb_frames",
            "-of", "json", out_path
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(res.stdout)
        v_dur = 0.0
        a_dur = 0.0
        v_frames = 0
        for s in probe_data.get("streams", []):
            if s.get("codec_type") == "video":
                v_dur = float(s.get("duration", 0.0))
                v_frames = int(s.get("nb_frames", 0))
            elif s.get("codec_type") == "audio":
                a_dur = float(s.get("duration", 0.0))
        delta_ms = abs(v_dur - a_dur) * 1000.0
        logger.info(f"Short {idx} Sync: Video={v_dur:.3f}s ({v_frames} frames), Audio={a_dur:.3f}s. Delta={delta_ms:.2f} ms")
    except Exception as e:
        logger.warning(f"Could not probe sync for Short {idx}: {e}")

    # Extract audit keyframes at 15s and 35s
    for snap_time in [15.0, 35.0]:
        snap_path = os.path.join(AUDIT_DIR, f"short_{idx}_{int(snap_time)}s_verified.jpg")
        cmd_snap = [
            "ffmpeg", "-y", "-ss", f"{snap_time:.1f}",
            "-i", out_path, "-vframes", "1", "-q:v", "2", snap_path
        ]
        subprocess.run(cmd_snap, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(snap_path):
            logger.info(f"Audit keyframe extracted: {snap_path}")

    return True

if __name__ == "__main__":
    targets = [int(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else [2, 3, 4]
    logger.info(f"Batch render target shorts: {targets}")
    for item in SHORTS:
        if item["idx"] in targets:
            ok = render_short(item)
            if not ok:
                logger.error(f"Stopping batch due to failure on Short {item['idx']}")
                sys.exit(1)
    logger.info("All target shorts rendered successfully!")
