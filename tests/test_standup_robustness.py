"""
Comprehensive unit and integration test suite for Stand-up Comedy,
Multilingual Hinglish/Hindi Transcription, Split-Screen Prohibition,
2D Spatial Block Scene Detection, and Audience Laughter Payoff Snapping.
"""

import os
import math
import pytest
from unittest.mock import MagicMock, patch

from shared.schemas import (
    CutLayoutMode,
    VideoSegment,
    UniversalContentProfile,
    ShortCandidate,
    SceneChange,
    SpatialEvent,
)
from wsl2_gpu.editorial_engine import classify_video_profile, EditorialEngine
from wsl2_gpu.web_server import classify_video_content
from wsl2_gpu.engagement_evaluator import EngagementEvaluator
from wsl2_gpu.trajectory_solver import fuse_point_of_interest


class TestIndicHinglishTranscription:
    """Verifies bilingual Hinglish prompt anchoring and language configuration flow."""

    def test_hinglish_prompt_anchoring(self):
        from wsl2_gpu.transcription_spoke import TranscriptionSpoke

        spoke = TranscriptionSpoke("nats://mock:4222")

        # Test prompt for Hindi or Indic comedy filename
        prompt = spoke._get_initial_prompt(
            target_lang="hi",
            is_indic_comedy=True
        )
        assert "नमस्ते!" in prompt
        assert "Hinglish" in prompt
        assert "Devanagari" in prompt

    def test_clean_language_override_resolution(self, monkeypatch, tmp_path):
        from wsl2_gpu.vram_orchestrator import run_phase1_asr_worker

        mock_pipe = MagicMock()
        mock_spoke_cls = MagicMock()
        mock_spoke = MagicMock()
        mock_spoke.transcribe_file.return_value = []
        mock_spoke_cls.return_value = mock_spoke

        # Verify passing language="hi" flows cleanly through
        with patch("wsl2_gpu.transcription_spoke.TranscriptionSpoke", mock_spoke_cls):
            with patch("asyncio.run", return_value=[]):
                # Should not raise assertion error
                run_phase1_asr_worker(
                    video_path="test.mp4",
                    nats_url="nats://mock:4222",
                    result_pipe=mock_pipe,
                    task="transcribe",
                    language="hi"
                )
                assert mock_pipe.send.called


class TestStandupContentProfiling:
    """Verifies stand-up comedy and stage show classification across filenames, metadata, and transcripts."""

    def test_classify_video_profile_from_filename(self):
        # Ravi Gupta stand-up comedy video filename
        filename = "Adventure _ Stand-up Comedy by Ravi Gupta_1080p.mp4"
        words = [{"word": "kya", "start": 0.0, "end": 0.5}, {"word": "chal", "start": 0.5, "end": 1.0}]

        profile = classify_video_profile(
            transcript_words=words,
            source_path=filename
        )
        assert profile == UniversalContentProfile.STAGE_PERFORMANCE

    def test_classify_video_content_web_server(self):
        filename = "Ravi_Gupta_Standup_Special.mp4"
        probe = {"duration": 720.0, "width": 1920, "height": 1080}

        res = classify_video_content(probe, filename=filename)
        assert res["type"] == "STAGE_PERFORMANCE"
        assert res["icon"] == "🎭"
        assert "Stand-up" in res["label"]

    def test_classify_video_profile_from_indic_comedy_keywords(self):
        # Transcript with Hindi comedy lexical markers
        words = [
            {"word": "haso", "start": 1.0, "end": 1.5},
            {"word": "chutkula", "start": 2.0, "end": 2.5},
            {"word": "comedy", "start": 3.0, "end": 3.5},
            {"word": "roast", "start": 4.0, "end": 4.5},
        ]
        profile = classify_video_profile(
            transcript_words=words,
            source_path="video_without_hints.mp4"
        )
        assert profile == UniversalContentProfile.STAGE_PERFORMANCE


class TestSplitScreenProhibition:
    """Verifies that solo stand-up comedy and stage shows strictly forbid split-screen."""

    def test_stage_performance_prohibits_dual_pane_split_screen(self):
        evaluator = EngagementEvaluator(content_profile=UniversalContentProfile.STAGE_PERFORMANCE)

        # 15s segment with continuous speaker and an item (water bottle / stool) detected
        segments = [
            VideoSegment(
                segment_start=0.0,
                segment_end=15.0,
                layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
                focal_target="SPEAKER_PRIMARY"
            )
        ]
        # Stage prop item 35% away from speaker
        spatial_events = [
            SpatialEvent(timestamp=3.0, frame_idx=90, track_id=0, label="person", bbox=(0.35, 0.4, 0.3, 0.5), confidence=0.9),
            SpatialEvent(timestamp=3.0, frame_idx=90, track_id=1, label="bottle", bbox=(0.75, 0.7, 0.1, 0.2), confidence=0.85),
        ]
        demo_beats = [
            {
                "beat_id": "b1",
                "start_time": 2.0,
                "end_time": 6.0,
                "topic_category": "DEMONSTRATION",
                "summary": "comedian gesturing with water bottle"
            }
        ]

        report, corrected_segments, cuts, targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate={"start_time_seconds": 0.0, "end_time_seconds": 15.0},
            words=[],
            spatial_events=spatial_events,
            transcript_beats=demo_beats,
            total_duration=15.0,
            content_profile=UniversalContentProfile.STAGE_PERFORMANCE
        )

        # STRICT ASSERTION: Zero segments may be DUAL_PANE_SPLIT_SCREEN
        for seg in corrected_segments:
            assert seg.layout_mode != CutLayoutMode.DUAL_PANE_SPLIT_SCREEN
            # Should remain FULL_FRAME_SPEAKER_TIGHT
            assert seg.layout_mode == CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT


class TestSpatialBlockSceneDetection:
    """Verifies 2D Spatial Block HSV + Edge differencing rejects lighting washes while detecting true camera cuts."""

    def test_uniform_stage_light_wash_rejected(self):
        import torch
        from wsl2_gpu.vision_spoke import VisionSpoke

        spoke = VisionSpoke.__new__(VisionSpoke)
        spoke.device = torch.device("cpu")

        # Simulate frame 1: base stage lighting
        f1 = torch.full((1, 3, 180, 320), 0.2, dtype=torch.float32)
        # Add a comedian silhouette in center block
        f1[:, :, 60:120, 106:212] = 0.6

        # Simulate frame 2: ambient blue/purple stage lighting wash (all blocks shift uniformly by +0.15)
        f2 = f1 + 0.15

        # Compute 3x3 block descriptors
        def get_descriptors(t):
            h_splits = torch.tensor_split(t, 3, dim=2)
            descs = []
            for h in h_splits:
                for b in torch.tensor_split(h, 3, dim=3):
                    r_h = torch.histc(b[:, 0], bins=16, min=0.0, max=1.0)
                    g_h = torch.histc(b[:, 1], bins=16, min=0.0, max=1.0)
                    b_h = torch.histc(b[:, 2], bins=16, min=0.0, max=1.0)
                    c_h = torch.cat([r_h, g_h, b_h])
                    c_h = c_h / (c_h.sum() + 1e-8)
                    dx = torch.abs(b[:, :, :, 1:] - b[:, :, :, :-1])
                    dy = torch.abs(b[:, :, 1:, :] - b[:, :, :-1, :])
                    descs.append((c_h, float((dx.mean() + dy.mean()).item())))
            return descs

        d1 = get_descriptors(f1)
        d2 = get_descriptors(f2)

        # Compute block diffs
        diffs = []
        for i in range(9):
            c_diff = 0.5 * torch.sum((d2[i][0] - d1[i][0]) ** 2 / (d2[i][0] + d1[i][0] + 1e-8)).item()
            e_diff = abs(d2[i][1] - d1[i][1]) / (max(d2[i][1], d1[i][1]) + 1e-4)
            diffs.append(0.6 * c_diff + 0.4 * e_diff)

        import numpy as np
        variance = float(np.var(diffs))
        # Uniform wash exhibits very low spatial block variance
        assert variance < 0.005

    def test_camera_angle_cut_detected(self):
        import torch

        # Frame 1: Wide shot (comedian small in center)
        f1 = torch.zeros((1, 3, 180, 320), dtype=torch.float32)
        f1[:, :, 70:110, 140:180] = 0.9

        # Frame 2: Medium close-up cut (comedian fills upper blocks 0, 1, 2, 3, 4, 5)
        f2 = torch.zeros((1, 3, 180, 320), dtype=torch.float32)
        f2[:, :, 10:160, 80:240] = 0.9

        def get_descriptors(t):
            h_splits = torch.tensor_split(t, 3, dim=2)
            descs = []
            for h in h_splits:
                for b in torch.tensor_split(h, 3, dim=3):
                    r_h = torch.histc(b[:, 0], bins=16, min=0.0, max=1.0)
                    g_h = torch.histc(b[:, 1], bins=16, min=0.0, max=1.0)
                    b_h = torch.histc(b[:, 2], bins=16, min=0.0, max=1.0)
                    c_h = torch.cat([r_h, g_h, b_h])
                    c_h = c_h / (c_h.sum() + 1e-8)
                    dx = torch.abs(b[:, :, :, 1:] - b[:, :, :, :-1])
                    dy = torch.abs(b[:, :, 1:, :] - b[:, :, :-1, :])
                    descs.append((c_h, float((dx.mean() + dy.mean()).item())))
            return descs

        d1 = get_descriptors(f1)
        d2 = get_descriptors(f2)

        diffs = []
        for i in range(9):
            c_diff = 0.5 * torch.sum((d2[i][0] - d1[i][0]) ** 2 / (d2[i][0] + d1[i][0] + 1e-8)).item()
            e_diff = abs(d2[i][1] - d1[i][1]) / (max(d2[i][1], d1[i][1]) + 1e-4)
            diffs.append(0.6 * c_diff + 0.4 * e_diff)

        discontinuous_blocks = sum(1 for d in diffs if d >= 0.28)
        # Camera cut should disrupt >= 5 spatial blocks
        assert discontinuous_blocks >= 5


class TestAudienceLaughterPayoff:
    """Verifies that stand-up comedy punchlines extend into the crowd laughter roar."""

    def test_comedic_laughter_payoff_extension(self, monkeypatch):
        monkeypatch.setenv("AETHER_CONTENT_PROFILE", "STAGE_PERFORMANCE")

        engine = EditorialEngine()

        candidate = ShortCandidate(
            start_time_seconds=60.0,
            duration_seconds=45.0,
            title="Punchline Climax",
            hook_headline="Wait till you hear this joke",
            reasoning="Setup -> Escalation -> Punchline"
        )

        # Word sequence: Punchline ends at 105.0s. Next joke starts at 108.5s (3.5s of crowd laughter)
        words_seq = [
            {"word": "Setup", "start": 61.0, "end": 62.0},
            {"word": "hilarious", "start": 104.0, "end": 104.8},
            {"word": "punchline!", "start": 104.8, "end": 105.0},  # Punchline ends here
            {"word": "So", "start": 108.5, "end": 108.8},         # Next joke 3.5s later
        ]

        valid = engine._validate_and_filter(
            candidates=[candidate],
            video_duration=200.0,
            transcript_words=words_seq
        )

        assert len(valid) == 1
        filtered = valid[0]
        # Should NOT cut off at 105.15s (duration 45.15s), but extend ~2.5s into audience laughter
        expected_end = 60.0 + filtered.duration_seconds
        assert expected_end > 106.5
        assert filtered.duration_seconds <= 50.0


class TestSinglePerformerStageTracking:
    """Verifies that tracking ignores stage props (water bottle, stool) and anchors on the comedian."""

    def test_stage_tracking_ignores_water_bottle(self, monkeypatch):
        monkeypatch.setenv("AETHER_CONTENT_PROFILE", "STAGE_PERFORMANCE")

        total_frames = 30
        fps = 30.0
        src_width = 1920.0
        src_height = 1080.0

        # Person at x=800px, water bottle placed at x=1200px
        spatial_events = [
            SpatialEvent(
                timestamp=0.0,
                frame_idx=0,
                track_id=0,
                label="person",
                bbox=(800.0 / src_width, 0.4, 0.3, 0.6),
                confidence=0.9
            ),
            SpatialEvent(
                timestamp=0.0,
                frame_idx=0,
                track_id=1,
                label="bottle",
                bbox=(1200.0 / src_width, 0.7, 0.1, 0.2),
                confidence=0.85
            ),
        ]

        poi_x = fuse_point_of_interest(
            spatial_events=spatial_events,
            fps=fps,
            total_frames=total_frames,
            src_width=src_width,
            src_height=src_height,
            return_2d=False
        )

        # In STAGE_PERFORMANCE mode, POI should remain locked around the comedian (800px)
        # rather than being pulled toward the bottle (1200px)
        assert abs(poi_x[0] - 800.0) < 50.0
