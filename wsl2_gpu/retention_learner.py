"""
Project Aether — Closed-Loop Retention Analytics Learner (Horizon 4)
Ingests platform watch-time retention metrics (Average Percentage Viewed, 3s retention)
to build an autonomous memory of viral narrative patterns and few-shot exemplars.

Feedback Loop:
YouTube Analytics -> Learning Memory -> Editorial Engine Few-Shot Prompts -> Higher Retention
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AetherRetentionLearner")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
LEARNING_MEMORY_FILE = os.path.join(DATA_DIR, "learning_memory.json")
ANALYTICS_LEDGER_FILE = os.path.join(DATA_DIR, "analytics_ledger.json")

DEFAULT_LEARNING_MEMORY = {
    "version": "1.0",
    "updated_at": 0.0,
    "high_retention_exemplars": [
        {
            "title_hook": "You're the only one who's immune...",
            "topic": "HERO_DILEMMA",
            "duration": 59.87,
            "apv_percent": 94.2,
            "retention_3s": 88.5,
            "opening_line": "You're the only one who's immune to it.",
            "hook_strategy": "Direct high-stakes personal confrontation in first 1.5 seconds"
        },
        {
            "title_hook": "She was in and out of...",
            "topic": "SUSPENSE_REVEAL",
            "duration": 53.27,
            "apv_percent": 91.0,
            "retention_3s": 85.0,
            "opening_line": "She was in and out of consciousness.",
            "hook_strategy": "Immediate life-or-death curiosity gap with zero preamble"
        }
    ],
    "high_converting_hook_words": [
        "IMMUNE", "SECRET", "IMPOSSIBLE", "DISASTER", "CAUGHT",
        "TRAPPED", "REVEALED", "DANGEROUS", "SHOCKED", "HERO"
    ],
    "optimal_pacing": {
        "ideal_duration_s": 55.0,
        "max_intro_silence_s": 0.8,
        "cut_cadence_s": 3.8
    }
}


def load_learning_memory() -> Dict[str, Any]:
    """Loads viral learning memory, seeding with defaults if absent."""
    if os.path.exists(LEARNING_MEMORY_FILE):
        try:
            with open(LEARNING_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    save_learning_memory(DEFAULT_LEARNING_MEMORY)
    return DEFAULT_LEARNING_MEMORY.copy()


def save_learning_memory(memory: Dict[str, Any]):
    """Persists updated learning memory to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    memory["updated_at"] = time.time()
    try:
        with open(LEARNING_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
        logger.info(f"Updated viral learning memory: {len(memory.get('high_retention_exemplars', []))} exemplars")
    except Exception as e:
        logger.warning(f"Could not save learning memory: {e}")


def record_performance_metric(video_id: str,
                              title: str,
                              apv_percent: float,
                              retention_3s: float,
                              views: int = 0,
                              duration_s: float = 55.0,
                              opening_line: str = "") -> Dict[str, Any]:
    """
    Ingests retention performance metrics from YouTube Analytics.
    If APV >= 85%, graduates the short into high-retention exemplars!
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    ledger = []
    if os.path.exists(ANALYTICS_LEDGER_FILE):
        try:
            with open(ANALYTICS_LEDGER_FILE, "r", encoding="utf-8") as f:
                ledger = json.load(f)
        except Exception:
            pass

    entry = {
        "video_id": video_id,
        "title": title,
        "apv_percent": apv_percent,
        "retention_3s": retention_3s,
        "views": views,
        "duration_s": duration_s,
        "opening_line": opening_line,
        "recorded_at": time.time()
    }
    ledger.append(entry)

    try:
        with open(ANALYTICS_LEDGER_FILE, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not record analytics entry: {e}")

    # Learning update: if short achieved viral retention (>85% APV)
    if apv_percent >= 85.0:
        memory = load_learning_memory()
        exemplars = memory.get("high_retention_exemplars", [])
        exemplars.append({
            "title_hook": title,
            "topic": "VIRAL_PERFORMER",
            "duration": duration_s,
            "apv_percent": apv_percent,
            "retention_3s": retention_3s,
            "opening_line": opening_line,
            "hook_strategy": f"Verified platform performer with {apv_percent:.1f}% APV and {views:,} views"
        })
        # Keep top 10 exemplars by APV
        exemplars.sort(key=lambda x: x.get("apv_percent", 0.0), reverse=True)
        memory["high_retention_exemplars"] = exemplars[:10]
        save_learning_memory(memory)
        logger.info(f"🏆 Graduated viral short '{title}' into high-retention learning memory (APV: {apv_percent}%)")

    return entry


def get_few_shot_editorial_guidance() -> str:
    """
    Formats high-retention exemplars as prompt text for Qwen2.5-VL in editorial_engine.py.
    """
    memory = load_learning_memory()
    exemplars = memory.get("high_retention_exemplars", [])
    hook_words = memory.get("high_converting_hook_words", [])

    lines = [
        "VIRAL RETENTION GROUND TRUTH (LEARNED FROM REAL YOUTUBE SHORTS AUDIENCE DATA):",
        "Target shorts that match these proven high-retention narrative dynamics (>85% Average Percentage Viewed):"
    ]

    for idx, ex in enumerate(exemplars[:4], 1):
        lines.append(
            f"Example {idx}: '{ex.get('title_hook')}' (Retention: {ex.get('apv_percent')}% APV | 3s: {ex.get('retention_3s')}%)\n"
            f"   - Opening Line: \"{ex.get('opening_line')}\"\n"
            f"   - Narrative Strategy: {ex.get('hook_strategy')}\n"
            f"   - Ideal Duration: {ex.get('duration')}s"
        )

    lines.append(f"High-Converting Viral Anchor Keywords: {', '.join(hook_words[:8])}")
    lines.append("Prioritize narrative segments that establish a strong curiosity gap within the first 1.5 seconds!")

    return "\n".join(lines)


def main():
    print("\n" + "=" * 65)
    print(" PROJECT AETHER CLOSED-LOOP RETENTION LEARNER")
    print("=" * 65)
    mem = load_learning_memory()
    print(f" Loaded {len(mem.get('high_retention_exemplars', []))} High-Retention Exemplars:")
    for ex in mem.get("high_retention_exemplars", []):
        print(f"  • {ex['title_hook']} ({ex['apv_percent']}% APV, {ex['retention_3s']}% at 3s)")
    print("\nGenerated Few-Shot Prompt Block:")
    print("-" * 50)
    print(get_few_shot_editorial_guidance())
    print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
