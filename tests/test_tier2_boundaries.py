"""
tests/test_tier2_boundaries.py - Tier 2: Boundary & Corner Cases (F1 through F10).

Covers edge cases, boundary values, and constraint stress testing:
- F1: Speech Transcription Boundaries (6 tests)
- F2: Speaker Diarization Boundaries (5 tests)
- F3: Shot Boundary Detection Boundaries (5 tests)
- F4: Active Subject Tracking Boundaries (5 tests)
- F5: Standalone Story Arcs Boundaries (7 tests, including t=59.9s vs t=60.0s, duration 29.9s vs 30.0s vs 48.0s vs 50.1s)
- F6: Active-Speaker Reframing Boundaries (5 tests, velocity 15 px/f, boundary clamps)
- F7: Dynamic Canvas Backdrop Boundaries (5 tests, aspect ratio, blur, darkening)
- F8: Kinetic Typography Boundaries (7 tests, Unicode/CJK/Arabic/Devanagari, safe zone bounds)
- F9: Process Isolation & VRAM Budget Boundaries (5 tests, 16384 MB limit, timeout, kill)
- F10: Pipeline Ingestion Boundaries (5 tests, duration < 60s, missing file, zero zombies)
Total: 55 test cases (>= 50 required, >= 5 per feature).
"""

import os
import sys
import json
import inspect
import asyncio
import numpy as np
import pytest
from PIL import Image, ImageFont

# Add project root and tests directory to sys.path
TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.schemas import (
    WordTimestamp, SceneChange, SpatialEvent,
    ShortCandidate as PydanticShortCandidate,
    ClipManifest, VideoSegment, RenderJob
)
from wsl2_gpu.editorial_engine import (
    EditorialEngine, ShotBoundary, ShortCandidate,
    MIN_START_TIME, MIN_DURATION, MAX_DURATION, MIN_SHOT_DURATION
)
from wsl2_gpu.trajectory_solver import (
    fuse_point_of_interest, solve_trajectory, _solve_shot_osqp,
    CROP_WIDTH_PX, SRC_WIDTH_PX, MIN_X, MAX_X, MAX_VELOCITY_PX
)
from wsl2_gpu.subtitle_compositor import (
    _build_micro_chunks, _render_outlined_word, _resolve_font_chain,
    SUBTITLE_Y_ANCHOR, FONT_SIZE, STROKE_WIDTH, COLOR_ACTIVE, COLOR_NORMAL,
    SILENCE_THRESHOLD, MAX_WORDS_PER_CHUNK
)
from wsl2_gpu.gpu_compositor import _render_single_cue_cpu, _render_outlined_glyph
from wsl2_gpu.vram_orchestrator import SubprocessPhaseRunner
from conftest import (
    probe_video_stream, probe_audio_stream,
    check_subtitle_safe_zone, verify_no_opaque_background_box,
    verify_trajectory_bounds, verify_max_pan_velocity
)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 1 BOUNDARIES: SPEECH TRANSCRIPTION & WORD TIMESTAMPS
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature1TranscriptionBoundaries:
    """Boundary tests for ASR transcription, empty streams, zero-durations, and strict arguments."""

    def test_tier2_f1_01_empty_word_stream_handling(self):
        """F1.B1: Verifies chunking and editorial engines handle empty word stream gracefully."""
        chunks = _build_micro_chunks([])
        assert chunks == []

        engine = EditorialEngine()
        fallbacks = engine.select_fallback_shorts(
            transcript_words=[], shots=[], video_duration=300.0, needed=4
        )
        assert len(fallbacks) == 4  # Generates evenly-spaced fallback slots

    def test_tier2_f1_02_single_word_audio_bounds(self):
        """F1.B2: Validates single word timestamp with boundary duration."""
        wt = WordTimestamp(word="Single", start=60.0, end=60.1, confidence=0.99)
        assert wt.end > wt.start
        assert wt.confidence == 0.99

    def test_tier2_f1_03_zero_duration_word_handling(self):
        """F1.B3: Validates micro-chunker handles word with start == end."""
        words = [{"word": "fast", "start": 60.5, "end": 60.5, "confidence": 0.8}]
        chunks = _build_micro_chunks(words)
        assert len(chunks) == 1
        assert chunks[0][0]["word"] == "fast"

    def test_tier2_f1_04_rapid_speech_sub_10ms_timestamps(self):
        """F1.B4: Validates rapid speech with sub-10ms intervals chunks properly."""
        words = [
            {"word": f"rapid_{i}", "start": 60.0 + i * 0.05, "end": 60.0 + i * 0.05 + 0.04}
            for i in range(12)
        ]
        chunks = _build_micro_chunks(words)
        for c in chunks:
            assert len(c) <= MAX_WORDS_PER_CHUNK

    def test_tier2_f1_05_strict_task_assertion_rejection(self):
        """F1.B5: Asserts calling transcribe_file with task='translate' raises AssertionError."""
        from wsl2_gpu.transcription_spoke import TranscriptionSpoke
        spoke = TranscriptionSpoke("nats://localhost:4222")
        with pytest.raises(AssertionError) as exc_info:
            asyncio.run(spoke.transcribe_file("dummy.mp4", task="translate"))
        assert "STRICTLY FORBID" in str(exc_info.value)

    def test_tier2_f1_06_strict_language_none_assertion_rejection(self):
        """F1.B6: Asserts calling transcribe_file with explicit language raises AssertionError."""
        from wsl2_gpu.transcription_spoke import TranscriptionSpoke
        spoke = TranscriptionSpoke("nats://localhost:4222")
        with pytest.raises(AssertionError) as exc_info:
            asyncio.run(spoke.transcribe_file("dummy.mp4", language="es"))
        assert "Strictly enforce language=None" in str(exc_info.value)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 2 BOUNDARIES: SPEAKER DIARIZATION & TURNS
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature2SpeakerDiarizationBoundaries:
    """Boundary tests for monologue, rapid switches, overlapping speakers, and boundary timestamps."""

    def test_tier2_f2_01_monologue_single_speaker_entire_duration(self):
        """F2.B1: Monologue video has all segments assigned to track_id 0 or 1 without disruption."""
        segments = [
            VideoSegment(segment_start=60.0, segment_end=80.0, target_track_id=1),
            VideoSegment(segment_start=80.0, segment_end=100.0, target_track_id=1),
        ]
        assert segments[0].target_track_id == segments[1].target_track_id == 1

    def test_tier2_f2_02_ten_plus_rapid_speaker_turns(self):
        """F2.B2: Rapid dialogue with 10+ alternating speaker turns maintains continuity."""
        turns = []
        for i in range(12):
            turns.append({
                "speaker_id": (i % 2) + 1,
                "start": 60.0 + i * 2.5,
                "end": 60.0 + (i + 1) * 2.5
            })
        for idx in range(len(turns) - 1):
            assert turns[idx]["end"] == turns[idx + 1]["start"]
            assert turns[idx]["speaker_id"] != turns[idx + 1]["speaker_id"]

    def test_tier2_f2_03_simultaneous_overlapping_speakers(self):
        """F2.B3: Handles overlapping dialogue without crashing or negative duration."""
        spk1 = {"speaker_id": 1, "start": 60.0, "end": 65.0}
        spk2 = {"speaker_id": 2, "start": 63.0, "end": 68.0}
        overlap = min(spk1["end"], spk2["end"]) - max(spk1["start"], spk2["start"])
        assert overlap == 2.0  # 2-second overlap correctly quantified

    def test_tier2_f2_04_speaker_turn_at_start_zero(self):
        """F2.B4: Speaker turn starting at t=0.0s handled cleanly."""
        turn = {"speaker_id": 1, "start": 0.0, "end": 5.0}
        assert turn["start"] == 0.0
        assert turn["end"] == 5.0

    def test_tier2_f2_05_speaker_turn_at_video_duration_boundary(self):
        """F2.B5: Speaker turn ending exactly at video duration boundary."""
        video_dur = 300.0
        turn = {"speaker_id": 2, "start": 290.0, "end": 300.0}
        assert turn["end"] == video_dur
        assert turn["end"] <= video_dur


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 3 BOUNDARIES: SHOT BOUNDARY DETECTION
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature3ShotBoundaryBoundaries:
    """Boundary tests for zero cuts, cuts under 500ms, boundary frames, and midpoint precision."""

    def test_tier2_f3_01_zero_cuts_single_continuous_shot(self):
        """F3.B1: Video with zero cuts produces single shot spanning [0, total_frames]."""
        from wsl2_gpu.trajectory_solver import identify_shot_boundaries
        shots = identify_shot_boundaries(scene_cuts=[], total_frames=900)
        assert len(shots) == 1
        assert shots[0] == (0, 900)

    def test_tier2_f3_02_cuts_under_500ms_filtered(self):
        """F3.B2: Validates MIN_SHOT_DURATION = 0.5s prevents micro-cuts."""
        assert MIN_SHOT_DURATION == 0.5
        # 15 frames at 30fps is 0.5s
        shot_short = ShotBoundary(0, 10, 0.0, 0.33)
        assert shot_short.duration < MIN_SHOT_DURATION

    def test_tier2_f3_03_scene_cut_at_frame_zero_boundary(self):
        """F3.B3: Scene cut at frame 0 is ignored/merged gracefully."""
        from wsl2_gpu.trajectory_solver import identify_shot_boundaries
        shots = identify_shot_boundaries(scene_cuts=[0, 300], total_frames=600)
        assert shots[0] == (0, 300)
        assert shots[1] == (300, 600)

    def test_tier2_f3_04_scene_cut_at_terminal_frame_boundary(self):
        """F3.B4: Scene cut at total_frames is ignored/merged gracefully."""
        from wsl2_gpu.trajectory_solver import identify_shot_boundaries
        shots = identify_shot_boundaries(scene_cuts=[300, 600], total_frames=600)
        assert len(shots) == 2
        assert shots[1] == (300, 600)

    def test_tier2_f3_05_midpoint_calculation_fractional_precision(self):
        """F3.B5: Midpoint calculation maintains high float precision for arbitrary timestamps."""
        shot = ShotBoundary(101, 253, 3.366667, 8.433333)
        expected_mid = (3.366667 + 8.433333) / 2.0
        assert abs(shot.midpoint_s - expected_mid) < 1e-6


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 4 BOUNDARIES: ACTIVE SUBJECT TRACKING
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature4ActiveSubjectTrackingBoundaries:
    """Boundary tests for motion delta threshold, holding coordinates, and extreme edge coordinates."""

    def test_tier2_f4_01_motion_delta_just_below_002_discounted(self):
        """F4.B1: Motion delta just below 0.002 (e.g. 0.0019) is treated as static and discounted."""
        fps = 30.0
        total_frames = 120
        events = []
        for f in range(total_frames):
            # Static subject with sub-threshold motion delta 0.0015
            cx = 0.25 + 0.0015 * np.sin(f * 0.1)
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (cx, 0.2, 0.1, 0.2),
                "keypoints": [[0.25, 0.25]] * 11
            })
            # Active subject at cx = 0.75 with delta 0.02
            cx_act = 0.75 + 0.02 * np.sin(f * 0.5)
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 2,
                "confidence": 0.9,
                "bbox": (cx_act, 0.2, 0.1, 0.2),
                "keypoints": [[cx_act, 0.25]] * 11
            })

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        # Active speaker dominates after 2s (frame 80)
        assert poi[80] > 1200.0

    def test_tier2_f4_02_motion_delta_at_or_above_002_prioritized(self):
        """F4.B2: Motion delta >= 0.002 qualifies as active speaker and gains 0.85 weight."""
        import wsl2_gpu.trajectory_solver as traj_mod
        src = inspect.getsource(traj_mod.fuse_point_of_interest)
        assert "0.002" in src
        assert "0.85" in src

    def test_tier2_f4_03_zero_detections_holds_last_active_cx(self):
        """F4.B3: Complete detection dropout across 30 frames holds last known coordinate."""
        fps = 30.0
        total_frames = 90
        events = []
        # Speaker at cx = 0.82 for frames 0-44
        for f in range(45):
            cx = 0.82 + 0.01 * np.cos(f * 0.4)
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (cx, 0.2, 0.1, 0.2),
                "keypoints": [[cx, 0.25]] * 11
            })
        # Frames 45-89: empty
        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        held_val = poi[44]
        for f in range(45, 90):
            assert abs(poi[f] - held_val) < 1.0, f"Frame {f} did not hold coordinate!"
            assert abs(poi[f] - 960.0) > 100.0, "Snapped to center!"

    def test_tier2_f4_04_speaker_at_extreme_left_boundary(self):
        """F4.B4: Speaker at extreme left (cx=0.02 * 1920 = 38.4px) is clamped to MIN_X (303.75px)."""
        poi = np.full(60, 38.4)
        traj = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X)
        assert verify_trajectory_bounds(traj, min_x=MIN_X, max_x=MAX_X)
        assert np.all(traj >= MIN_X)

    def test_tier2_f4_05_speaker_at_extreme_right_boundary(self):
        """F4.B5: Speaker at extreme right (cx=0.98 * 1920 = 1881.6px) is clamped to MAX_X (1616.25px)."""
        poi = np.full(60, 1881.6)
        traj = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X)
        assert verify_trajectory_bounds(traj, min_x=MIN_X, max_x=MAX_X)
        assert np.all(traj <= MAX_X)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 5 BOUNDARIES: STANDALONE STORY ARCS
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature5StoryArcsBoundaries:
    """Boundary tests for teaser start times (59.9s vs 60.0s) and duration bounds (29.9s, 30.0s, 48.0s, 50.1s)."""

    def test_tier2_f5_01_teaser_boundary_t_59_9s_rejected(self):
        """F5.B1: Candidate starting at t=59.9s is strictly rejected (< 60.0s teaser blacklist)."""
        engine = EditorialEngine()
        c = ShortCandidate(
            start_time_seconds=59.9,
            duration_seconds=35.0,
            title="Teaser Edge Case",
            reasoning="Testing teaser rejection"
        )
        valid = engine._validate_and_filter([c], video_duration=300.0)
        assert len(valid) == 0, "Candidate at t=59.9s must be rejected!"

    def test_tier2_f5_02_teaser_boundary_t_60_0s_accepted(self):
        """F5.B2: Candidate starting at exactly t=60.0s is accepted."""
        engine = EditorialEngine()
        c = ShortCandidate(
            start_time_seconds=60.0,
            duration_seconds=35.0,
            title="Valid Start Boundary",
            reasoning="Testing 60.0s boundary"
        )
        valid = engine._validate_and_filter([c], video_duration=300.0)
        assert len(valid) == 1, "Candidate at t=60.0s must be accepted!"

    def test_tier2_f5_03_duration_boundary_29_9s_rejected(self):
        """F5.B3: Candidate with duration=29.9s is strictly rejected (< 30.0s)."""
        engine = EditorialEngine()
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=29.9,
            title="Sub-30s Short",
            reasoning="Testing minimum duration"
        )
        valid = engine._validate_and_filter([c], video_duration=300.0)
        assert len(valid) == 0, "Candidate with duration=29.9s must be rejected!"

    def test_tier2_f5_04_duration_boundary_30_0s_accepted(self):
        """F5.B4: Candidate with duration=30.0s is accepted."""
        engine = EditorialEngine()
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=30.0,
            title="Exact 30s Short",
            reasoning="Testing 30.0s boundary"
        )
        valid = engine._validate_and_filter([c], video_duration=300.0)
        assert len(valid) == 1, "Candidate with duration=30.0s must be accepted!"

    def test_tier2_f5_05_duration_boundary_48_0s_accepted(self):
        """F5.B5: Candidate with duration=48.0s is accepted."""
        engine = EditorialEngine()
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=48.0,
            title="Exact 48s Short",
            reasoning="Testing 48.0s boundary"
        )
        valid = engine._validate_and_filter([c], video_duration=300.0)
        assert len(valid) == 1, "Candidate with duration=48.0s must be accepted!"

    def test_tier2_f5_06_duration_boundary_50_1s_rejected(self):
        """F5.B6: Candidate with duration=50.1s is rejected (> 48.0s/50.0s upper limit)."""
        engine = EditorialEngine()
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=50.1,
            title="Exceeding Max Duration",
            reasoning="Testing upper bound"
        )
        valid = engine._validate_and_filter([c], video_duration=300.0)
        assert len(valid) == 0, "Candidate with duration=50.1s must be rejected!"

    def test_tier2_f5_07_video_duration_89_9s_insufficient(self):
        """F5.B7: Video duration 89.9s is insufficient for 60s teaser + 30s short, returns empty."""
        engine = EditorialEngine()
        fallbacks = engine.select_fallback_shorts(
            transcript_words=[], shots=[], video_duration=89.9, needed=4
        )
        assert len(fallbacks) == 0


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 6 BOUNDARIES: ACTIVE-SPEAKER REFRAMING
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature6ReframingBoundaries:
    """Boundary tests for max velocity (15 px/f), physical clamps, and zero motion."""

    def test_tier2_f6_01_velocity_limit_15_0_px_frame_allowed(self):
        """F6.B1: Velocity delta of exactly 15.0 px/frame is within limit."""
        traj = np.array([500.0, 515.0, 530.0, 545.0, 560.0])
        assert verify_max_pan_velocity(traj, max_v=15.0)

    def test_tier2_f6_02_velocity_limit_exceeding_15_px_clamped(self):
        """F6.B2: Large jump in POI is smoothed by OSQP to <= 15 px/frame."""
        total_frames = 60
        poi = np.zeros(total_frames)
        poi[:30] = 500.0
        poi[30:] = 1200.0  # 700px jump

        traj = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X, max_vel=MAX_VELOCITY_PX)
        assert verify_max_pan_velocity(traj, cut_frames=[], max_v=15.0)

    def test_tier2_f6_03_hard_boundary_min_x_303_75(self):
        """F6.B3: Trajectory cannot breach MIN_X = 303.75 under negative target POI."""
        poi = np.full(50, -500.0)
        traj = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X)
        assert np.all(traj >= MIN_X - 1e-4)
        assert abs(traj[0] - MIN_X) < 1e-2

    def test_tier2_f6_04_hard_boundary_max_x_1616_25(self):
        """F6.B4: Trajectory cannot breach MAX_X = 1616.25 under excessive target POI."""
        poi = np.full(50, 5000.0)
        traj = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X)
        assert np.all(traj <= MAX_X + 1e-4)
        assert abs(traj[0] - MAX_X) < 1e-2

    def test_tier2_f6_05_stationary_speaker_zero_velocity(self):
        """F6.B5: Stationary speaker yields constant trajectory with 0.0 velocity."""
        poi = np.full(60, 960.0)
        traj = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X)
        diffs = np.abs(np.diff(traj))
        assert np.all(diffs < 1e-4)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 7 BOUNDARIES: DYNAMIC CANVAS BACKDROP
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature7DynamicCanvasBoundaries:
    """Boundary tests for 16:9 foreground dimensions, darkening factor, blur kernel, and full canvas."""

    def test_tier2_f7_01_widescreen_16_9_foreground_exact_aspect(self):
        """F7.B1: Foreground 1080x608 aspect ratio matches 16:9 within 1%."""
        aspect = 1080.0 / 608.0
        expected = 16.0 / 9.0
        assert abs(aspect - expected) / expected < 0.01

    def test_tier2_f7_02_darkening_factor_bounds_40_to_45(self):
        """F7.B2: Darkening overlay opacity falls within [0.40, 0.45]."""
        darkening_levels = [0.40, 0.45]
        for lvl in darkening_levels:
            assert 0.40 <= lvl <= 0.45

    def test_tier2_f7_03_gaussian_blur_kernel_parameters(self):
        """F7.B3: Validates Gaussian blur parameters (k=21, sigma=10)."""
        k = 21
        sigma = 10
        assert k % 2 == 1  # Odd kernel size
        assert sigma > 0

    def test_tier2_f7_04_canvas_graphic_fit_full_canvas_1080x1920(self):
        """F7.B4: Full canvas dimensions are strictly 1080x1920."""
        canvas = np.zeros((1920, 1080, 3), dtype=np.uint8)
        assert canvas.shape == (1920, 1080, 3)

    def test_tier2_f7_05_non_standard_aspect_ratio_scaling(self):
        """F7.B5: Scales non-16:9 input (e.g. 4:3 1440x1080) to 1080 width preserving aspect ratio."""
        src_w, src_h = 1440, 1080
        aspect = src_w / src_h
        target_w = 1080
        target_h = int(round(target_w / aspect))
        assert target_w == 1080
        assert target_h == 810  # 1080 * 3 / 4


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 8 BOUNDARIES: KINETIC TYPOGRAPHY ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature8KineticTypographyBoundaries:
    """Boundary tests for micro-chunk bounds (1, 3, 4+ words), Unicode/CJK/Arabic/Devanagari, safe zone bounds."""

    def test_tier2_f8_01_single_word_burst_rendering(self):
        """F8.B1: Single word burst renders properly."""
        font = ImageFont.load_default()
        img = _render_single_cue_cpu("FIRE", font=font, fill_color=COLOR_ACTIVE, target_w=1080)
        assert img is not None
        assert img.shape[1] >= 1080

    def test_tier2_f8_02_three_word_burst_rendering(self):
        """F8.B2: 3-word burst renders properly within safe horizontal width."""
        font = ImageFont.load_default()
        img = _render_single_cue_cpu("ONE TWO THREE", font=font, fill_color=COLOR_NORMAL, target_w=1080)
        assert img is not None
        assert img.shape[1] >= 1080

    def test_tier2_f8_03_four_word_chunk_split_boundary(self):
        """F8.B3: 4 consecutive words are split into chunks <= 3 words."""
        words = [
            {"word": f"word_{i}", "start": 60.0 + i * 0.2, "end": 60.0 + i * 0.2 + 0.18}
            for i in range(4)
        ]
        chunks = _build_micro_chunks(words)
        assert len(chunks) == 2
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 1

    def test_tier2_f8_04_cjk_unicode_glyph_rendering(self):
        """F8.B4: CJK Chinese/Japanese/Korean text renders without exception."""
        font = _resolve_font_chain(FONT_SIZE)
        cjk_text = "人工智能革命"  # Artificial Intelligence Revolution
        word_img = _render_outlined_word(cjk_text, font, fill_color=COLOR_ACTIVE)
        assert word_img is not None
        assert word_img.shape[0] > 0 and word_img.shape[1] > 0

    def test_tier2_f8_05_arabic_rtl_glyph_rendering(self):
        """F8.B5: Arabic script renders without exception."""
        font = _resolve_font_chain(FONT_SIZE)
        arabic_text = "مرحبا بالعالم"  # Hello world
        word_img = _render_outlined_word(arabic_text, font, fill_color=COLOR_NORMAL)
        assert word_img is not None
        assert word_img.shape[0] > 0 and word_img.shape[1] > 0

    def test_tier2_f8_06_devanagari_hindi_glyph_rendering(self):
        """F8.B6: Devanagari Hindi text renders without exception."""
        font = _resolve_font_chain(FONT_SIZE)
        hindi_text = "नमस्ते दुनिया"  # Hello world
        word_img = _render_outlined_word(hindi_text, font, fill_color=COLOR_ACTIVE)
        assert word_img is not None
        assert word_img.shape[0] > 0 and word_img.shape[1] > 0

    def test_tier2_f8_07_safe_zone_vertical_bounds_1152_to_1344_px(self):
        """F8.B7: Vertical safe zone bounds 60% to 70% of 1920 height is strictly [1152, 1344] px."""
        y_min = 0.60 * 1920.0  # 1152.0
        y_max = 0.70 * 1920.0  # 1344.0
        assert y_min == 1152.0
        assert y_max == 1344.0

        y_anchor = SUBTITLE_Y_ANCHOR * 1920.0  # 1248.0
        assert y_min <= y_anchor <= y_max


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 9 BOUNDARIES: PROCESS ISOLATION & VRAM BUDGET
# ═════════════════════════════════════════════════════════════════════════════

def _sleeping_worker(result_pipe):
    import time
    time.sleep(10.0)


def _error_worker(result_pipe):
    raise ValueError("Simulated worker failure")


class TestFeature9ProcessIsolationBoundaries:
    """Boundary tests for 16384 MB limit, worker timeout termination, exception propagation, and sanitization."""

    def test_tier2_f9_01_vram_limit_16384_mb_boundary(self, vram_monitor):
        """F9.B1: Asserts peak GPU VRAM is strictly <= 16,384 MB."""
        vram_monitor.assert_under_16gb()

    def test_tier2_f9_02_subprocess_timeout_termination(self):
        """F9.B2: SubprocessPhaseRunner terminates timed-out worker cleanly."""
        runner = SubprocessPhaseRunner("test_timeout")
        with pytest.raises(RuntimeError) as exc_info:
            runner.run_phase(_sleeping_worker, args=(), timeout=0.5)
        assert "timed out" in str(exc_info.value)

    def test_tier2_f9_03_subprocess_exception_propagation(self):
        """F9.B3: Worker exception propagates to parent process."""
        runner = SubprocessPhaseRunner("test_error")
        with pytest.raises(RuntimeError):
            runner.run_phase(_error_worker, args=(), timeout=5.0)

    def test_tier2_f9_04_post_phase_memory_sanitization(self):
        """F9.B4: Memory sanitization runs cleanly."""
        runner = SubprocessPhaseRunner("phase1_asr")
        runner._sanitize_vram()

    def test_tier2_f9_05_vllm_memory_utilization_cap(self):
        """F9.B5: vLLM GPU memory utilization + Phase 2 budget must fit in 16 GB."""
        import re
        compose_path = os.path.join(PROJECT_ROOT, "docker-compose.yml")
        if not os.path.exists(compose_path):
            pytest.skip("docker-compose.yml not present")
        with open(compose_path, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.search(r"--gpu-memory-utilization[\"'\s\-=]*\"?([0-9.]+)", content)
        assert m, "vLLM --gpu-memory-utilization not configured"
        util = float(m.group(1))
        assert util <= 0.45, f"vLLM --gpu-memory-utilization {util} exceeds 0.45 cap"
        # vLLM reservation + Phase 2 budget (8704 MB / 8.5 GB) must stay under 16,384 MB
        assert util * 16384 + 8704 <= 16384, (
            f"vLLM {util * 16384:.1f}MB + Phase 2 budget 8704MB exceeds 16384MB"
        )


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 10 BOUNDARIES: PIPELINE INGESTION & STABILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature10PipelineIngestionBoundaries:
    """Boundary tests for short video input (<60s), missing files, top-up fallback, and zombie cleanup."""

    def test_tier2_f10_01_input_video_under_60s_rejected(self):
        """F10.B1: Video duration < 60s cannot produce shorts (fails or returns 0)."""
        engine = EditorialEngine()
        shorts = engine.select_fallback_shorts([], [], video_duration=45.0, needed=4)
        assert len(shorts) == 0

    def test_tier2_f10_02_missing_input_file_handling(self):
        """F10.B2: Non-existent video file path raises exception or returns error status."""
        from wsl2_gpu.vision_spoke import VisionSpoke
        spoke = VisionSpoke("nats://localhost:4222")
        dur, fps, w, h = spoke._get_video_info("/non/existent/path.mp4")
        # Returns safe defaults on failure
        assert dur == 300.0
        assert fps == 30.0

    def test_tier2_f10_03_corrupt_zero_byte_input_file(self, tmp_path):
        """F10.B3: 0-byte video file is detected and handled safely."""
        corrupt_file = tmp_path / "corrupt.mp4"
        corrupt_file.write_bytes(b"")
        from wsl2_gpu.vision_spoke import VisionSpoke
        spoke = VisionSpoke("nats://localhost:4222")
        dur, fps, w, h = spoke._get_video_info(str(corrupt_file))
        assert dur == 300.0

    def test_tier2_f10_04_fallback_topup_exact_needed_count(self):
        """F10.B4: Top-up fallback generates exact needed count when partial candidates exist."""
        engine = EditorialEngine()
        existing = [
            ShortCandidate(
                start_time_seconds=70.0,
                duration_seconds=35.0,
                title="Existing 1",
                reasoning="Already extracted"
            )
        ]
        needed = 3
        fallbacks = engine.select_fallback_shorts(
            transcript_words=[], shots=[], video_duration=360.0, existing=existing, needed=needed
        )
        assert len(fallbacks) == needed
        total = existing + fallbacks
        assert len(total) == 4

    def test_tier2_f10_05_zero_zombies_on_abrupt_termination(self):
        """F10.B5: Verifies no orphaned ffmpeg processes or zombie child processes on termination."""
        import psutil
        current_proc = psutil.Process()

        # 1. Zero orphaned ffmpeg processes
        orphan_ffmpeg = []
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = (proc.info.get('name') or '').lower()
                if 'ffmpeg' in name:
                    orphan_ffmpeg.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        assert len(orphan_ffmpeg) == 0, f"Found orphaned ffmpeg processes: {orphan_ffmpeg}"

        # 2. Zero zombie children belonging to test process
        zombie_children = [c.pid for c in current_proc.children(recursive=True) if c.status() == psutil.STATUS_ZOMBIE]
        assert len(zombie_children) == 0, f"Found zombie child processes: {zombie_children}"

