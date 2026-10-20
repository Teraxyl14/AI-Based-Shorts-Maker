"""
Project Aether — Persistent Multi-Batch Extraction Memory Engine
Maintains an immutable record of previously extracted short intervals per video.
Enables processing 2-hour+ feature films and long-form podcasts to produce
30+ unique, non-overlapping 50s-60s shorts across sequential batches.
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger("AetherExtractionMemory")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXTRACTION_LEDGER_FILE = os.path.join(DATA_DIR, "extracted_ranges.json")
SAFETY_BUFFER_SECONDS = 3.0  # Buffer around previously extracted clips to avoid boundary cut-bleed


def compute_video_hash(video_path: str) -> str:
    """Computes a consistent MD5 hash identifier for the source video path."""
    norm_path = os.path.normpath(video_path).replace("\\", "/")
    return hashlib.md5(norm_path.encode("utf-8")).hexdigest()


def load_extraction_ledger() -> Dict[str, Any]:
    """Loads the persistent extraction ledger from disk."""
    if os.path.exists(EXTRACTION_LEDGER_FILE):
        try:
            with open(EXTRACTION_LEDGER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read extraction ledger: {e}")
    return {}


def save_extraction_ledger(ledger: Dict[str, Any]):
    """Persists the extraction ledger atomically to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_file = f"{EXTRACTION_LEDGER_FILE}.tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
        if os.path.exists(EXTRACTION_LEDGER_FILE):
            os.replace(tmp_file, EXTRACTION_LEDGER_FILE)
        else:
            os.rename(tmp_file, EXTRACTION_LEDGER_FILE)
    except Exception as e:
        logger.error(f"Failed to persist extraction ledger: {e}")
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass


def get_excluded_intervals(video_path: str,
                           buffer_seconds: float = SAFETY_BUFFER_SECONDS) -> List[Tuple[float, float]]:
    """
    Returns a list of buffered [start_s, end_s] intervals that have already been extracted
    from the given video, preventing duplicate scenes in subsequent batches.
    """
    v_hash = compute_video_hash(video_path)
    ledger = load_extraction_ledger()
    video_record = ledger.get(v_hash, {})
    raw_intervals = video_record.get("extracted_intervals", [])

    buffered = []
    for s, e in raw_intervals:
        buffered.append((max(0.0, float(s) - buffer_seconds), float(e) + buffer_seconds))
    return buffered


def is_interval_excluded(start_s: float, end_s: float,
                         excluded_intervals: List[Tuple[float, float]]) -> bool:
    """Checks whether [start_s, end_s] overlaps with any previously extracted intervals."""
    for ex_s, ex_e in excluded_intervals:
        # Overlap condition: start < ex_e and end > ex_s
        if start_s < ex_e and end_s > ex_s:
            return True
    return False


def record_extracted_batch(video_path: str,
                           video_duration_seconds: float,
                           shorts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Records a completed batch of extracted shorts into the persistent ledger.
    Increments batch count and appends new intervals.
    """
    if not shorts:
        return {}

    v_hash = compute_video_hash(video_path)
    ledger = load_extraction_ledger()

    if v_hash not in ledger:
        ledger[v_hash] = {
            "source_file": os.path.basename(video_path),
            "video_duration_seconds": round(video_duration_seconds, 2),
            "total_shorts_generated": 0,
            "batches": [],
            "extracted_intervals": []
        }

    rec = ledger[v_hash]
    batch_id = len(rec["batches"]) + 1

    batch_shorts = []
    for s_dict in shorts:
        s_start = float(s_dict.get("start_time_seconds", 0.0))
        s_dur = float(s_dict.get("duration_seconds", 0.0))
        s_end = float(s_dict.get("end_time_seconds", s_start + s_dur))
        title = s_dict.get("title", f"Short {rec['total_shorts_generated'] + len(batch_shorts) + 1}")

        batch_shorts.append({
            "index": rec["total_shorts_generated"] + len(batch_shorts) + 1,
            "start_s": round(s_start, 2),
            "end_s": round(s_end, 2),
            "duration_s": round(s_dur, 2),
            "title": title
        })
        rec["extracted_intervals"].append([round(s_start, 2), round(s_end, 2)])

    batch_entry = {
        "batch_id": batch_id,
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shorts_count": len(batch_shorts),
        "shorts": batch_shorts
    }
    rec["batches"].append(batch_entry)
    rec["total_shorts_generated"] += len(batch_shorts)

    save_extraction_ledger(ledger)
    logger.info(
        f"Saved Batch {batch_id} for '{os.path.basename(video_path)}': "
        f"{len(batch_shorts)} shorts (Total lifetime: {rec['total_shorts_generated']})"
    )
    return rec


def get_video_extraction_summary(video_path: str) -> Dict[str, Any]:
    """Returns extraction coverage, total shorts, and batch history for the web dashboard."""
    v_hash = compute_video_hash(video_path)
    ledger = load_extraction_ledger()
    rec = ledger.get(v_hash, None)

    if not rec:
        return {
            "exists": False,
            "hash": v_hash,
            "total_shorts_generated": 0,
            "batches_count": 0,
            "coverage_percent": 0.0,
            "batches": []
        }

    total_extracted_time = sum((e - s) for s, e in rec.get("extracted_intervals", []))
    dur = max(1.0, float(rec.get("video_duration_seconds", 1.0)))
    coverage_pct = min(100.0, round((total_extracted_time / dur) * 100.0, 1))

    return {
        "exists": True,
        "hash": v_hash,
        "source_file": rec.get("source_file", ""),
        "video_duration_seconds": dur,
        "total_shorts_generated": rec.get("total_shorts_generated", 0),
        "batches_count": len(rec.get("batches", [])),
        "coverage_percent": coverage_pct,
        "extracted_intervals": rec.get("extracted_intervals", []),
        "batches": rec.get("batches", [])
    }
