"""
Unit and integration tests for Project Aether's Closed-Loop Retention Learner (Horizon 4).
Tests analytics recording, APV graduation threshold (>=85%), and few-shot prompt generation.
"""

import os
import json
import pytest
from wsl2_gpu.retention_learner import (
    load_learning_memory,
    save_learning_memory,
    record_performance_metric,
    get_few_shot_editorial_guidance,
    DEFAULT_LEARNING_MEMORY,
)


class TestRetentionLearner:

    def test_default_memory_loads_cleanly(self, tmp_path, monkeypatch):
        test_mem_file = str(tmp_path / "learning_memory.json")
        monkeypatch.setattr("wsl2_gpu.retention_learner.LEARNING_MEMORY_FILE", test_mem_file)
        monkeypatch.setattr("wsl2_gpu.retention_learner.DATA_DIR", str(tmp_path))

        mem = load_learning_memory()
        assert "high_retention_exemplars" in mem
        assert len(mem["high_retention_exemplars"]) >= 2
        assert "high_converting_hook_words" in mem
        assert "optimal_pacing" in mem

    def test_record_performance_metric_under_threshold(self, tmp_path, monkeypatch):
        test_ledger = str(tmp_path / "analytics_ledger.json")
        test_mem = str(tmp_path / "learning_memory.json")
        monkeypatch.setattr("wsl2_gpu.retention_learner.ANALYTICS_LEDGER_FILE", test_ledger)
        monkeypatch.setattr("wsl2_gpu.retention_learner.LEARNING_MEMORY_FILE", test_mem)
        monkeypatch.setattr("wsl2_gpu.retention_learner.DATA_DIR", str(tmp_path))

        # Record short with 65.0% APV (< 85% threshold)
        entry = record_performance_metric(
            video_id="test_vid_1",
            title="Moderate Retention Video",
            apv_percent=65.0,
            retention_3s=60.0,
            views=1200,
            duration_s=55.0,
            opening_line="Hey guys check this out"
        )
        assert entry["video_id"] == "test_vid_1"
        assert entry["apv_percent"] == 65.0

        # Verify not graduated into exemplars
        mem = load_learning_memory()
        exemplar_titles = [x["title_hook"] for x in mem.get("high_retention_exemplars", [])]
        assert "Moderate Retention Video" not in exemplar_titles

    def test_record_performance_metric_graduates_viral_short(self, tmp_path, monkeypatch):
        test_ledger = str(tmp_path / "analytics_ledger.json")
        test_mem = str(tmp_path / "learning_memory.json")
        monkeypatch.setattr("wsl2_gpu.retention_learner.ANALYTICS_LEDGER_FILE", test_ledger)
        monkeypatch.setattr("wsl2_gpu.retention_learner.LEARNING_MEMORY_FILE", test_mem)
        monkeypatch.setattr("wsl2_gpu.retention_learner.DATA_DIR", str(tmp_path))

        # Record short with 92.5% APV (>= 85% threshold)
        entry = record_performance_metric(
            video_id="viral_vid_99",
            title="Secret Spider-Man Revelation",
            apv_percent=92.5,
            retention_3s=89.0,
            views=50000,
            duration_s=58.2,
            opening_line="They never told you the real reason why."
        )
        assert entry["apv_percent"] == 92.5

        # Verify graduated into top exemplars
        mem = load_learning_memory()
        exemplar_titles = [x["title_hook"] for x in mem.get("high_retention_exemplars", [])]
        assert "Secret Spider-Man Revelation" in exemplar_titles

    def test_few_shot_guidance_formatting(self, tmp_path, monkeypatch):
        test_mem = str(tmp_path / "learning_memory.json")
        monkeypatch.setattr("wsl2_gpu.retention_learner.LEARNING_MEMORY_FILE", test_mem)
        monkeypatch.setattr("wsl2_gpu.retention_learner.DATA_DIR", str(tmp_path))

        guidance = get_few_shot_editorial_guidance()
        assert "VIRAL RETENTION GROUND TRUTH" in guidance
        assert "Average Percentage Viewed" in guidance
        assert "High-Converting Viral Anchor Keywords" in guidance
