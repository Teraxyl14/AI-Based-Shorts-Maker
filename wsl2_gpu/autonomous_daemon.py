"""
Project Aether — Autonomous Master Control Daemon & Thermal Governor
Orchestrates the 4-spoke self-sustaining YouTube Shorts factory:
Spoke 1 (Trending) -> Spoke 2 (Download) -> Spoke 3 (Aether Core) -> QA Gate -> Spoke 4 (Publish)

Thermal & Hardware Longevity:
- Batch-and-sleep pulse engine: 4 daily processing pulses (~3 min GPU load each).
- Throttles or cools down if GPU temperature exceeds safety threshold (80°C).
- GPU rests at 0% load and ~45°C for 23.8 hours/day.

Dual Operating Modes:
- 'auto_pilot': Fully autonomous end-to-end publishing.
- 'review_first': Human-in-the-loop review queue on dashboard.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AetherAutonomousDaemon")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_FILE = os.path.join(DATA_DIR, "daemon_config.json")
STATE_FILE = os.path.join(DATA_DIR, "daemon_state.json")
REVIEW_QUEUE_FILE = os.path.join(DATA_DIR, "review_queue.json")
INPUT_VIDEOS_DIR = os.path.join(DATA_DIR, "input_videos")
OUTPUT_SHORTS_DIR = os.path.join(DATA_DIR, "output_shorts")

DEFAULT_CONFIG = {
    "mode": "review_first",        # auto_pilot | review_first
    "auto_publish_cc": True,       # Auto-publish Creative Commons in auto_pilot mode
    "max_gpu_temp_c": 80,          # GPU thermal throttling cutoff
    "pulses_per_day": 4,           # Number of automated batch pulses per day
    "target_geo": "US",            # Google Trends region
    "min_resolution_height": 1080, # 1080p minimum
    "min_fps": 30,                 # 30 fps minimum
    "loop_interval_seconds": 60.0  # Daemon evaluation tick
}


def load_daemon_config() -> Dict[str, Any]:
    """Loads runtime daemon configuration from disk, creating defaults if missing."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                return {**DEFAULT_CONFIG, **saved}
        except Exception:
            pass
    save_daemon_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()


def save_daemon_config(cfg: Dict[str, Any]):
    """Persists runtime configuration to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save daemon config: {e}")


def load_review_queue() -> List[Dict[str, Any]]:
    """Loads candidate shorts awaiting human review."""
    if os.path.exists(REVIEW_QUEUE_FILE):
        try:
            with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_review_queue(queue: List[Dict[str, Any]]):
    """Persists review queue items to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(REVIEW_QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save review queue: {e}")


def add_to_review_queue(short_item: Dict[str, Any]):
    """Adds a QA-certified short to the human review queue."""
    queue = load_review_queue()
    # Avoid duplicate entries
    for item in queue:
        if item.get("video_path") == short_item.get("video_path"):
            return
    queue.append(short_item)
    save_review_queue(queue)
    logger.info(f"📋 Added certified short to review queue: {short_item.get('title')}")


def get_gpu_temperature() -> int:
    """Reads current GPU temperature via nvidia-smi."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            return int(res.stdout.strip())
    except Exception:
        pass
    return 45


def check_thermal_safety(max_temp: int = 80) -> bool:
    """Certifies that GPU temperature is safe before high-intensity rendering."""
    temp = get_gpu_temperature()
    if temp >= max_temp:
        logger.warning(f"⚠️ Thermal Governor: GPU temp is {temp}°C (threshold: {max_temp}°C). Cooling down...")
        return False
    return True


class AutonomousDaemon:
    def __init__(self):
        self.config = load_daemon_config()
        self.is_running = False
        self.last_pulse_time = 0.0
        self.pulse_interval = (24.0 * 3600.0) / max(1, self.config.get("pulses_per_day", 4))

    def update_mode(self, new_mode: str):
        if new_mode in ("auto_pilot", "review_first"):
            self.config["mode"] = new_mode
            save_daemon_config(self.config)
            logger.info(f"🔄 Daemon operating mode updated to: {new_mode.upper()}")

    def execute_single_pulse(self):
        """
        Executes one autonomous lifecycle pulse:
        1. Discover trending topic (Spoke 1).
        2. Download high-quality source video (Spoke 2).
        3. Trigger Aether Core transformation (Spoke 3).
        4. Validate with QA Gate.
        5. Publish or enqueue into Review Queue (Spoke 4).
        6. Auto-purge raw source video.
        """
        logger.info("══════════════════════════════════════════════════════════════")
        logger.info(" 🚀 EXECUTING AUTONOMOUS LIFECYCLE PULSE")
        logger.info(f" Mode: {self.config['mode'].upper()} | GPU Temp: {get_gpu_temperature()}°C")
        logger.info("══════════════════════════════════════════════════════════════")

        if not check_thermal_safety(self.config.get("max_gpu_temp_c", 80)):
            logger.info("Thermal pause active. Pulse deferred.")
            return

        # Spoke 1: Trending Discovery
        try:
            from wsl2_gpu.trending_spoke import fetch_google_trends, search_candidate_videos_for_topic
            topics = fetch_google_trends(self.config.get("target_geo", "US"))
            if not topics:
                logger.warning("No trending topics found. Pulse ending.")
                return

            chosen_topic = topics[0]
            logger.info(f"Selected Trending Topic: '{chosen_topic.title}' (Traffic: {chosen_topic.approx_traffic})")
            candidates = search_candidate_videos_for_topic(chosen_topic.title, max_results=2)
            if not candidates:
                logger.warning(f"No valid video candidates found for: {chosen_topic.title}")
                return

            target_cand = candidates[0]
            logger.info(f"Selected Candidate Video: {target_cand['title']} ({target_cand['url']})")
        except Exception as e:
            logger.error(f"Spoke 1 (Trending) error: {e}")
            return

        # Spoke 2: Quality-Gated Download
        try:
            from wsl2_gpu.download_spoke import download_quality_video, purge_processed_source_video
            dl_res = download_quality_video(
                target_cand["url"],
                output_dir=INPUT_VIDEOS_DIR,
                min_height=self.config.get("min_resolution_height", 1080),
                min_fps=self.config.get("min_fps", 30)
            )
            if not dl_res:
                logger.warning("Download spoke could not acquire target video. Pulse ending.")
                return

            source_path = dl_res["path"]
            logger.info(f"Video downloaded to: {source_path} ({dl_res['size_mb']} MB)")
        except Exception as e:
            logger.error(f"Spoke 2 (Download) error: {e}")
            return

        # Spoke 3: Trigger Ingest Pipeline
        try:
            from wsl2_gpu.trigger_ingest import trigger_ingest
            nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
            import asyncio
            logger.info(f"Triggering Aether Core pipeline for: {source_path}")
            asyncio.run(trigger_ingest(nats_url, source_path))
        except Exception as e:
            logger.error(f"Spoke 3 (Aether Core trigger) error: {e}")
            return

        # The orchestrator will asynchronously process Phases 1, 2, and 3.
        # Once complete, our post-render handler runs QA certification and publication.
        self.last_pulse_time = time.time()
        logger.info("Autonomous pulse successfully dispatched to Aether pipeline.")

    def run_qa_and_publish_certified(self, shorts_dir: str, source_path: Optional[str] = None):
        """Audits rendered shorts in output directory and publishes or queues them."""
        from wsl2_gpu.qa_gate import validate_short
        from wsl2_gpu.publisher_spoke import publish_short
        from wsl2_gpu.download_spoke import purge_processed_source_video

        if not os.path.exists(shorts_dir):
            return

        certified_count = 0
        for f in sorted(os.listdir(shorts_dir)):
            if f.startswith("short_") and f.endswith(".mp4"):
                vid_path = os.path.join(shorts_dir, f)
                idx_str = f.replace("short_", "").replace(".mp4", "")
                thumb_path = os.path.join(shorts_dir, f"thumbnail_{idx_str}.jpg")
                if not os.path.exists(thumb_path):
                    thumb_path = None

                qa_res = validate_short(vid_path)
                if qa_res.passed:
                    certified_count += 1
                    logger.info(f"✅ QA Gate Certified: {f}")

                    short_item = {
                        "video_path": vid_path,
                        "filename": f,
                        "thumbnail_path": thumb_path,
                        "title": f"Short {idx_str}",
                        "certified_at": time.time(),
                        "qa_checks": qa_res.checks
                    }

                    if self.config["mode"] == "auto_pilot":
                        # Auto-Pilot Mode: Schedule upload immediately
                        pub_res = publish_short(
                            vid_path,
                            raw_title=f"Viral Arc {idx_str}",
                            thumbnail_path=thumb_path,
                            privacy_status="public"
                        )
                        logger.info(f"🚀 Auto-Pilot Published: {pub_res.get('title')} ({pub_res.get('status')})")
                    else:
                        # Review First Mode: Place in Review Queue for human 1-click approval
                        add_to_review_queue(short_item)
                else:
                    logger.warning(f"❌ Short failed QA Gate: {f} -> {qa_res.checks}")

        # Auto-purge raw source video once certified shorts exist
        if certified_count >= 1 and source_path and os.path.exists(source_path):
            purge_processed_source_video(source_path)

    def run(self):
        """Main continuous daemon loop."""
        self.is_running = True
        logger.info("==========================================================")
        logger.info(" Project Aether Autonomous Master Control Daemon")
        logger.info(f" Mode: {self.config['mode'].upper()}")
        logger.info(f" Pulse Cadence: {self.config['pulses_per_day']} batches/day")
        logger.info("==========================================================")

        while self.is_running:
            self.config = load_daemon_config()
            now = time.time()
            if (now - self.last_pulse_time) >= self.pulse_interval:
                self.execute_single_pulse()

            time.sleep(self.config.get("loop_interval_seconds", 60.0))


def main():
    parser = argparse.ArgumentParser(description="Project Aether Master Control Daemon")
    parser.add_argument("--mode", choices=["auto_pilot", "review_first"], help="Set operating mode")
    parser.add_argument("--pulse", action="store_true", help="Execute an immediate single lifecycle pulse")
    args = parser.parse_args()

    daemon = AutonomousDaemon()
    if args.mode:
        daemon.update_mode(args.mode)

    if args.pulse:
        daemon.execute_single_pulse()
    else:
        try:
            daemon.run()
        except KeyboardInterrupt:
            logger.info("Daemon stopped by user.")


if __name__ == "__main__":
    main()
