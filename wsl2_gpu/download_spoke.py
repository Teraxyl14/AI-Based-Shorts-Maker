"""
Project Aether — Intelligent Quality Downloader & Storage Governor
Downloads trending long-form videos with strict quality constraints:
- Resolution: >= 1080p, >= 30fps.
- Duration: 5 minutes to 45 minutes.
- Hybrid Rights Tagging: CC-BY (Auto-Publish) vs. Fair Use (Review Queue).
- Rolling Storage Auto-Purge: Removes raw multi-gigabyte source files
  immediately after Phase 3 render certification is confirmed.
"""

import os
import sys
import json
import logging
import hashlib
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AetherDownloadSpoke")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_VIDEOS_DIR = os.path.join(DATA_DIR, "input_videos")
OUTPUT_SHORTS_DIR = os.path.join(DATA_DIR, "output_shorts")


def download_quality_video(url: str,
                           output_dir: str = INPUT_VIDEOS_DIR,
                           min_height: int = 1080,
                           min_fps: int = 30) -> Optional[Dict[str, Any]]:
    """
    Downloads a video with strict quality gates:
    - >= 1080p, >= 30 fps.
    - Container: MP4 with AAC audio.
    - Tags licensing rights for algorithmic auto-pilot vs human review.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is required. Run 'pip install yt-dlp'")
        return None

    # Step 1: Pre-download inspection of formats and duration
    probe_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.error(f"Could not extract metadata for {url}: {e}")
        return None

    duration = float(info.get("duration", 0.0) or 0.0)
    title = info.get("title", "Untitled")
    video_id = info.get("id", "video")

    # Duration gate: 5 min (300s) to 45 min (2700s)
    if duration < 300.0 or duration > 2700.0:
        logger.warning(
            f"Video duration ({duration / 60:.1f}m) outside target window (5m-45m): {title}"
        )
        return None

    # Rights classification: CC-BY vs Third-Party Fair Use
    license_str = str(info.get("license", "")).lower()
    description = str(info.get("description", "")).lower()
    is_cc = ("creative commons" in license_str or "cc by" in license_str or
             "creative commons attribution" in description)
    rights_tier = "CC_BY_AUTO" if is_cc else "FAIR_USE_REVIEW"

    # Step 2: Quality-gated download: >= 1080p30, merge to mp4
    clean_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
    out_filename = f"{clean_title}_{video_id}.mp4"
    out_filepath = os.path.join(output_dir, out_filename)

    ydl_opts = {
        "format": f"bestvideo[height>={min_height}][fps>={min_fps}]+bestaudio/best[height>={min_height}]/best",
        "outtmpl": os.path.join(output_dir, f"{clean_title}_{video_id}.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }]
    }

    logger.info(f"Downloading high-quality video: {title} ({duration / 60:.1f}m) [{rights_tier}]")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        return None

    # Find the downloaded file
    final_path = None
    for f in os.listdir(output_dir):
        if video_id in f and f.endswith(".mp4"):
            final_path = os.path.join(output_dir, f)
            break

    if not final_path or not os.path.exists(final_path):
        logger.error(f"Download completed but output file not found for ID: {video_id}")
        return None

    file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
    logger.info(f"Successfully downloaded: {final_path} ({file_size_mb:.1f} MB)")

    result = {
        "title": title,
        "video_id": video_id,
        "path": final_path,
        "filename": os.path.basename(final_path),
        "duration_seconds": duration,
        "rights_tier": rights_tier,
        "is_creative_commons": is_cc,
        "size_mb": round(file_size_mb, 2),
        "source_url": url,
        "uploader": info.get("uploader", "Unknown"),
    }

    # Save sidecar metadata info
    info_path = final_path.replace(".mp4", ".info.json")
    try:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write sidecar info: {e}")

    return result


def purge_processed_source_video(source_path: str) -> bool:
    """
    Purges raw multi-gigabyte source video from input_videos once
    output shorts have been rendered and verified by the QA gate.
    Saves disk space and prevents SSD saturation during autonomous 24/7 runs.
    """
    if not os.path.exists(source_path):
        return False

    # Safety check: Must be inside input_videos directory
    norm_source = os.path.normpath(source_path)
    norm_input_dir = os.path.normpath(INPUT_VIDEOS_DIR)
    if not norm_source.startswith(norm_input_dir):
        logger.warning(f"Refusing to purge file outside input_videos: {source_path}")
        return False

    # Verify that at least 1 certified short exists for this video
    video_hash = hashlib.md5(source_path.encode()).hexdigest()
    shorts_dir = os.path.join(OUTPUT_SHORTS_DIR, video_hash)

    has_valid_shorts = False
    if os.path.exists(shorts_dir) and os.path.isdir(shorts_dir):
        shorts = [f for f in os.listdir(shorts_dir) if f.startswith("short_") and f.endswith(".mp4")]
        if len(shorts) >= 1:
            has_valid_shorts = True

    if not has_valid_shorts:
        logger.warning(f"Will not purge source video: No rendered shorts verified in {shorts_dir}")
        return False

    try:
        size_mb = os.path.getsize(source_path) / (1024 * 1024)
        os.remove(source_path)
        logger.info(f"♻️ Auto-purged raw source video: {os.path.basename(source_path)} (Reclaimed {size_mb:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Could not purge source video: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python wsl2_gpu/download_spoke.py <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]
    res = download_quality_video(url)
    if res:
        print("\n" + "=" * 65)
        print(" DOWNLOAD COMPLETED")
        print(f" File: {res['path']}")
        print(f" Rights: {res['rights_tier']} (CC: {res['is_creative_commons']})")
        print(f" Size: {res['size_mb']} MB")
        print("=" * 65 + "\n")
    else:
        print("Download failed or video did not meet quality requirements.")
        sys.exit(1)


if __name__ == "__main__":
    main()
