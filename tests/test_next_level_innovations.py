"""
Unit and integration test suite for Next-Level Short Creation Innovations:
1. The 3-Second Hook Retention Spike (Dynamic Zoom Punch-in)
2. Peak Expression Viral Thumbnail Extractor
3. Conversational Cross-Cutting (Reaction Cam J/L Cuts)
4. Widescreen Ambient Gaussian Blur Reflection Fill
5. Word-by-Word Kinetic Subtitle Typography
"""

import math
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from shared.schemas import (
    VideoSegment,
    CutLayoutMode,
    UniversalContentProfile,
    SpatialEvent,
)
from wsl2_gpu.trajectory_solver import apply_hook_zoom_punch
from wsl2_gpu.render_spoke import (
    render_cinematic_3zone_916,
    ThreeThreadRenderPipeline,
)
from wsl2_gpu.engagement_evaluator import EngagementEvaluator
from wsl2_gpu.subtitle_compositor import (
    SubtitleCompositor,
    _build_micro_chunks as chunk_words_for_display,
    SCALE_POP_FACTOR,
)


class TestHookRetentionSpike:
    """Verifies the 3-second hook retention zoom punch-in curve."""

    def test_hook_zoom_punch_scale_factor(self):
        fps = 30.0
        total_frames = 90  # 3 seconds
        src_w = 607.5
        src_h = 1080.0

        base_w = np.full(total_frames, src_w, dtype=np.float64)
        base_h = np.full(total_frames, src_h, dtype=np.float64)

        zoomed_w, zoomed_h = apply_hook_zoom_punch(
            base_w, base_h, fps=fps, zoom_duration_s=2.5, max_zoom_pct=0.08
        )

        # Frame 0: Maximum punch-in (~108% scale -> crop dimensions ~8% tighter)
        expected_w0 = src_w / 1.08
        expected_h0 = src_h / 1.08
        assert abs(zoomed_w[0] - expected_w0) < 1.0
        assert abs(zoomed_h[0] - expected_h0) < 1.0

        # Frame 75 (2.5s mark): Zoom smoothly settles back to 100% (base dimensions)
        assert abs(zoomed_w[75] - src_w) < 1.0
        assert abs(zoomed_h[75] - src_h) < 1.0

        # Frame 89 (after 2.5s): Remains strictly at 100%
        assert zoomed_w[89] == src_w
        assert zoomed_h[89] == src_h

    def test_hook_zoom_punch_monotonic_decay(self):
        fps = 30.0
        total_frames = 75
        base_w = np.full(total_frames, 607.5, dtype=np.float64)
        base_h = np.full(total_frames, 1080.0, dtype=np.float64)

        zoomed_w, _ = apply_hook_zoom_punch(base_w, base_h, fps=fps, zoom_duration_s=2.5)

        # Crop width must monotonically widen back from zoomed to normal over time
        for f in range(1, len(zoomed_w)):
            assert zoomed_w[f] >= zoomed_w[f - 1] - 1e-4


class TestPeakExpressionThumbnail:
    """Verifies peak expression and gestural energy frame selection in the Hook window."""

    def test_find_peak_expression_timestamp_with_gestural_energy(self):
        pipeline = ThreeThreadRenderPipeline.__new__(ThreeThreadRenderPipeline)
        mock_job = MagicMock()

        # Spatial events:
        # Event 1 (1.0s): Low-energy face, hands resting at bottom
        ev1 = {
            "timestamp": 1.0,
            "confidence": 0.85,
            "label": "person",
            "keypoints": [
                [0.5, 0.2, 0.9],  # nose
                [0.48, 0.18, 0.9], [0.52, 0.18, 0.9],
                [0.45, 0.2, 0.8], [0.55, 0.2, 0.8],
                [0.4, 0.35, 0.9], [0.6, 0.35, 0.9],
                [0.38, 0.55, 0.8], [0.62, 0.55, 0.8],
                [0.38, 0.85, 0.7], [0.62, 0.85, 0.7],  # wrists down at y=0.85
                [0.45, 0.7, 0.8], [0.55, 0.7, 0.8],
            ]
        }
        # Event 2 (2.2s): High gestural apex, wrists raised at chest/head level (y=0.45)
        ev2 = {
            "timestamp": 2.2,
            "confidence": 0.95,
            "label": "person",
            "keypoints": [
                [0.5, 0.2, 0.95],
                [0.48, 0.18, 0.9], [0.52, 0.18, 0.9],
                [0.45, 0.2, 0.8], [0.55, 0.2, 0.8],
                [0.4, 0.35, 0.9], [0.6, 0.35, 0.9],
                [0.38, 0.45, 0.9], [0.62, 0.45, 0.9],
                [0.35, 0.42, 0.9], [0.65, 0.42, 0.9],  # wrists raised high at y=0.42
                [0.45, 0.7, 0.8], [0.55, 0.7, 0.8],
            ]
        }

        mock_job.spatial_events = [ev1, ev2]
        pipeline.job = mock_job

        best_ts = pipeline._find_peak_expression_timestamp()
        assert best_ts == 2.2

    def test_find_peak_expression_timestamp_fallback_when_empty(self):
        pipeline = ThreeThreadRenderPipeline.__new__(ThreeThreadRenderPipeline)
        mock_job = MagicMock()
        mock_job.spatial_events = []
        pipeline.job = mock_job

        # Should fall back cleanly to 1.5s
        best_ts = pipeline._find_peak_expression_timestamp()
        assert best_ts == 1.5


class TestConversationalReactionCrossCutting:
    """Verifies J-Cut anticipatory lead-in offsets for conversational podcasts."""

    def test_conversational_podcast_applies_j_cut_lead_in(self):
        evaluator = EngagementEvaluator()

        segments = [
            VideoSegment(segment_start=0.0, segment_end=8.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            VideoSegment(segment_start=8.0, segment_end=16.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            VideoSegment(segment_start=16.0, segment_end=24.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
        ]

        # Standard cut without podcast mode: cut exactly on segment start
        standard_cuts = evaluator._compute_editorial_cuts(segments, clip_start=0.0, is_podcast=False)
        assert standard_cuts == [8.0, 16.0]

        # Conversational podcast: 0.60s J-Cut anticipatory lead-in
        podcast_cuts = evaluator._compute_editorial_cuts(segments, clip_start=0.0, is_podcast=True)
        assert podcast_cuts == [7.4, 15.4]


class TestWidescreenAmbientReflectionFill:
    """Verifies Gaussian blur background reflection fill for 16:9 widescreen footage."""

    def test_render_cinematic_3zone_layering(self):
        # Synthetic 16:9 widescreen frame with ambient content across canvas
        frame = torch.full((1, 3, 1080, 1920), 0.6, dtype=torch.float32)

        target_w, target_h = 1080, 1920
        composited = render_cinematic_3zone_916(frame, target_w=target_w, target_h=target_h)

        assert composited.shape == (1, 3, target_h, target_w)
        # Background layer should have non-zero blurred reflection fill (not pitch black)
        # Top zone (Y=100)
        assert composited[:, :, 100, :].mean().item() > 0.0
        # Bottom zone (Y=1800)
        assert composited[:, :, 1800, :].mean().item() > 0.0


class TestWordByWordKineticTypography:
    """Verifies word-by-word micro-chunking and kinetic scaling."""

    def test_word_micro_chunking_bounds(self):
        words = [
            {"word": "He", "start": 0.1, "end": 0.3},
            {"word": "changed", "start": 0.3, "end": 0.6},
            {"word": "one", "start": 0.6, "end": 0.8},
            {"word": "number", "start": 0.8, "end": 1.2},
            {"word": "and", "start": 1.2, "end": 1.4},
            {"word": "left.", "start": 1.4, "end": 1.8},
        ]

        chunks = chunk_words_for_display(words)
        # Each chunk should contain at most 3 words
        for chunk in chunks:
            assert len(chunk) <= 3

        # Period at "left." should end chunk
        assert any(chunk[-1]["word"] == "left." for chunk in chunks)

    def test_scale_pop_factor_defined(self):
        # 14% kinetic pop
        assert SCALE_POP_FACTOR == 1.14
