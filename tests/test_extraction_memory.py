"""
Unit and integration tests for Project Aether's Persistent Multi-Batch Extraction Memory Engine.
Verifies video hashing, interval buffering, collision exclusion, batch incrementing,
timeline coverage calculation, and non-overlapping multi-batch generation across long videos.
"""

import os
import json
import pytest
from wsl2_gpu.extraction_memory import (
    compute_video_hash,
    load_extraction_ledger,
    save_extraction_ledger,
    get_excluded_intervals,
    is_interval_excluded,
    record_extracted_batch,
    get_video_extraction_summary,
    SAFETY_BUFFER_SECONDS,
)


class TestExtractionMemory:

    def test_compute_video_hash_consistency(self):
        h1 = compute_video_hash("data/input_videos/movie.mp4")
        h2 = compute_video_hash("data\\input_videos\\movie.mp4")
        assert h1 == h2, "Path normalization must produce identical hashes across platforms"

    def test_empty_ledger_returns_zero_excluded(self, tmp_path, monkeypatch):
        test_file = str(tmp_path / "extracted_ranges.json")
        monkeypatch.setattr("wsl2_gpu.extraction_memory.EXTRACTION_LEDGER_FILE", test_file)
        monkeypatch.setattr("wsl2_gpu.extraction_memory.DATA_DIR", str(tmp_path))

        excluded = get_excluded_intervals("test_video.mp4")
        assert excluded == []

    def test_is_interval_excluded_logic(self):
        excluded = [(100.0, 160.0), (300.0, 360.0)]

        # Completely before first interval -> not excluded
        assert not is_interval_excluded(50.0, 95.0, excluded)

        # Overlapping start of first interval -> excluded
        assert is_interval_excluded(90.0, 110.0, excluded)

        # Completely inside first interval -> excluded
        assert is_interval_excluded(110.0, 150.0, excluded)

        # Overlapping end of first interval -> excluded
        assert is_interval_excluded(155.0, 200.0, excluded)

        # In gap between intervals -> not excluded
        assert not is_interval_excluded(170.0, 290.0, excluded)

        # Spanning across entire second interval -> excluded
        assert is_interval_excluded(280.0, 380.0, excluded)

    def test_record_batch_and_coverage_summary(self, tmp_path, monkeypatch):
        test_file = str(tmp_path / "extracted_ranges.json")
        monkeypatch.setattr("wsl2_gpu.extraction_memory.EXTRACTION_LEDGER_FILE", test_file)
        monkeypatch.setattr("wsl2_gpu.extraction_memory.DATA_DIR", str(tmp_path))

        video = "C:/Movies/Spider-Man.mkv"
        total_duration = 7200.0  # 2 hours

        # Record Batch 1 (4 shorts)
        batch1_shorts = [
            {"start_time_seconds": 120.0, "duration_seconds": 55.0, "end_time_seconds": 175.0, "title": "Arc 1"},
            {"start_time_seconds": 400.0, "duration_seconds": 58.0, "end_time_seconds": 458.0, "title": "Arc 2"},
            {"start_time_seconds": 900.0, "duration_seconds": 52.0, "end_time_seconds": 952.0, "title": "Arc 3"},
            {"start_time_seconds": 1400.0, "duration_seconds": 60.0, "end_time_seconds": 1460.0, "title": "Arc 4"},
        ]
        rec = record_extracted_batch(video, total_duration, batch1_shorts)
        assert rec["total_shorts_generated"] == 4
        assert len(rec["batches"]) == 1
        assert len(rec["extracted_intervals"]) == 4

        # Verify excluded intervals on subsequent run includes buffer
        excluded = get_excluded_intervals(video, buffer_seconds=3.0)
        assert len(excluded) == 4
        assert excluded[0] == (117.0, 178.0)  # 120-3, 175+3

        # Record Batch 2 (4 more shorts)
        batch2_shorts = [
            {"start_time_seconds": 2100.0, "duration_seconds": 55.0, "end_time_seconds": 2155.0, "title": "Arc 5"},
            {"start_time_seconds": 2800.0, "duration_seconds": 56.0, "end_time_seconds": 2856.0, "title": "Arc 6"},
            {"start_time_seconds": 3500.0, "duration_seconds": 59.0, "end_time_seconds": 3559.0, "title": "Arc 7"},
            {"start_time_seconds": 4200.0, "duration_seconds": 54.0, "end_time_seconds": 4254.0, "title": "Arc 8"},
        ]
        rec2 = record_extracted_batch(video, total_duration, batch2_shorts)
        assert rec2["total_shorts_generated"] == 8
        assert len(rec2["batches"]) == 2
        assert len(rec2["extracted_intervals"]) == 8

        # Verify extraction summary
        summary = get_video_extraction_summary(video)
        assert summary["exists"] is True
        assert summary["total_shorts_generated"] == 8
        assert summary["batches_count"] == 2
        assert summary["coverage_percent"] > 0.0
