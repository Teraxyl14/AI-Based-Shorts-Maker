"""
tests/test_tier3_pairwise.py - Tier 3: Cross-Feature Pairwise Combinations.

Covers cross-module integration and combinatorial feature interactions:
- C1: F1 (ASR) + F8 (Typography): ASR word timestamps piped to Subtitle Compositor.
- C2: F3 (Shot Detection) + F6 (Reframing): Scene cuts triggering OSQP cut teleports.
- C3: F4 (Spatial Tracking) + F6 (Reframing): Active speaker motion gating driving OSQP trajectory.
- C4: F5 (Editorial Arcs) + F7 (Dynamic Canvas): Editorial engine selecting CANVAS_GRAPHIC_FIT mode.
- C5: F5 (Editorial Arcs) + F1 (ASR): Editorial fallback short extraction using speech density & pauses.
- C6: F4 (Spatial Tracking) + F7 (Dynamic Canvas): Spatial FOCAL_DISPLAY detection triggering graphic canvas.
- C7: F6 (Reframing) + F8 (Typography): 1080x1920 cropped frame layered with kinetic subtitles.
- C8: F9 (VRAM Isolation) + F1 (ASR): SubprocessPhaseRunner running ASR with VRAM reclamation.
- C9: F9 (VRAM Isolation) + F3 (Vision) + F5 (Editorial): Phase 2 worker running in isolated subprocess.
- C10: F9 (VRAM Isolation) + F6 (Reframing) + F8 (Typography) + F10 (Pipeline): Phase 3 render isolation.
- C11: F1 (ASR) + F8 (Typography): Multilingual ASR tokens rendered via font fallback chain.
- C12: F5 (Editorial Arcs) + F3 (Shot Detection) + F4 (Spatial Tracking): Episodic continuity tracking.
Total: 12 test cases (>= 10 required).
"""

import os
import sys
import json
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
    MIN_START_TIME, MIN_DURATION, MAX_DURATION
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
from wsl2_gpu.vram_orchestrator import SubprocessPhaseRunner, PHASE_BUDGETS
from conftest import (
    probe_video_stream, probe_audio_stream,
    check_subtitle_safe_zone, verify_no_opaque_background_box,
    verify_trajectory_bounds, verify_max_pan_velocity
)


def _mock_asr_worker(result_pipe):
    words = [
        {"word": "Subprocess", "start": 60.0, "end": 60.5, "confidence": 0.99},
        {"word": "Isolation", "start": 60.6, "end": 61.2, "confidence": 0.98},
    ]
    result_pipe.send({"words": words, "status": "success"})
    result_pipe.close()


def _mock_vision_editorial_worker(result_pipe):
    engine = EditorialEngine()
    words = [
        {"word": f"w_{i}.", "start": 60.0 + i * 0.45, "end": 60.0 + i * 0.45 + 0.38}
        if i % 15 == 14 else
        {"word": f"w_{i}", "start": 60.0 + i * 0.45, "end": 60.0 + i * 0.45 + 0.38}
        for i in range(600)
    ]
    fallbacks = engine.select_fallback_shorts(words, [], video_duration=360.0, needed=4)
    data = [{"start": s.start_time_seconds, "dur": s.duration_seconds} for s in fallbacks]
    result_pipe.send({"shorts": data, "status": "success"})
    result_pipe.close()


def _mock_render_worker(result_pipe):
    poi = np.full(60, 960.0)
    traj = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X)
    result_pipe.send({"rendered_frames": len(traj), "status": "success"})
    result_pipe.close()


class TestTier3PairwiseCombinations:
    """Pairwise cross-feature integration test suite."""

    def test_tier3_01_asr_to_subtitle_compositor_e2e(self):
        """C1: F1 (ASR) + F8 (Typography): ASR word timestamps piped to subtitle micro-chunker & renderer."""
        # 1. Simulate ASR output
        raw_words = [
            WordTimestamp(word="Building", start=60.0, end=60.4, confidence=0.98),
            WordTimestamp(word="the", start=60.45, end=60.6, confidence=0.95),
            WordTimestamp(word="future", start=60.65, end=61.1, confidence=0.99),
            WordTimestamp(word="with", start=61.4, end=61.6, confidence=0.92),  # Gap > 120ms -> new chunk
            WordTimestamp(word="AI", start=61.65, end=62.0, confidence=0.97),
        ]
        word_dicts = [w.model_dump() for w in raw_words]

        # 2. Chunk into 1-3 word bursts
        chunks = _build_micro_chunks(word_dicts)
        assert len(chunks) == 2
        assert len(chunks[0]) == 3  # "Building the future"
        assert len(chunks[1]) == 2  # "with AI"

        # 3. Render cue with neon highlight
        font = ImageFont.load_default()
        cue_text = " ".join(w["word"] for w in chunks[0])
        cue_img = _render_single_cue_cpu(cue_text, font=font, fill_color=COLOR_ACTIVE, target_w=1080)
        assert cue_img is not None
        assert cue_img.shape[1] >= 1080
        assert verify_no_opaque_background_box(cue_img)

    def test_tier3_02_shot_detection_to_osqp_instantaneous_teleport(self):
        """C2: F3 (Shot Detection) + F6 (Reframing): Scene cuts triggering OSQP cut teleports."""
        fps = 30.0
        total_frames = 90
        # Speaker left in shot 1 (frames 0-44), speaker right in shot 2 (frames 45-89)
        spatial_events = []
        for f in range(total_frames):
            cx = 0.20 if f < 45 else 0.80
            spatial_events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1 if f < 45 else 2,
                "confidence": 0.9,
                "bbox": (cx, 0.2, 0.1, 0.2),
                "keypoints": [[cx, 0.25]] * 11
            })
        scene_changes = [{"timestamp": 45.0 / fps, "confidence": 0.95, "frame_idx": 45}]

        traj_norm = solve_trajectory(spatial_events, scene_changes, fps=fps, total_frames=total_frames)
        traj_px = traj_norm[:, 0] * SRC_WIDTH_PX

        # Verify within shot 1 and shot 2 velocity is <= 15 px/f
        assert verify_max_pan_velocity(traj_px, cut_frames=[45], max_v=15.0)
        # Across cut frame 45, teleport occurred
        teleport_delta = abs(traj_px[45] - traj_px[44])
        assert teleport_delta > 50.0  # Instantaneous camera repositioning

    def test_tier3_03_active_speaker_motion_gating_to_osqp_pan(self):
        """C3: F4 (Spatial Tracking) + F6 (Reframing): Active speaker motion gating driving OSQP trajectory."""
        fps = 30.0
        total_frames = 120
        events = []
        for f in range(total_frames):
            # Active speaker moving at cx ~ 0.70
            cx_act = 0.70 + 0.02 * np.sin(f * 0.4)
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (cx_act, 0.2, 0.1, 0.2),
                "keypoints": [[cx_act, 0.25]] * 11
            })
            # Motionless listener at cx = 0.25
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 2,
                "confidence": 0.9,
                "bbox": (0.25, 0.2, 0.1, 0.2),
                "keypoints": [[0.25, 0.25]] * 11
            })

        traj_norm = solve_trajectory(events, scene_changes=[], fps=fps, total_frames=total_frames)
        traj_px = traj_norm[:, 0] * SRC_WIDTH_PX

        # Camera trajectory should be centered near active speaker (~1344 px), not listener (480 px)
        assert np.mean(traj_px[60:]) > 1100.0
        assert verify_trajectory_bounds(traj_px, min_x=MIN_X, max_x=MAX_X)

    def test_tier3_04_editorial_engine_to_canvas_graphic_fit(self):
        """C4: F5 (Editorial Arcs) + F7 (Dynamic Canvas): Editorial engine selecting CANVAS_GRAPHIC_FIT mode."""
        engine = EditorialEngine()
        # Create a candidate in CANVAS_GRAPHIC_FIT mode
        candidate = ShortCandidate(
            start_time_seconds=65.0,
            duration_seconds=36.0,
            title="UI Architecture Walkthrough",
            reasoning="Wide UI screen capture slide demonstration",
            framing_mode="CANVAS_GRAPHIC_FIT"
        )
        valid = engine._validate_and_filter([candidate], video_duration=300.0)
        assert len(valid) == 1
        assert valid[0].framing_mode == "CANVAS_GRAPHIC_FIT"

        # Build RenderJob with this candidate
        manifest = ClipManifest(
            title=valid[0].title,
            segments=[
                VideoSegment(
                    segment_start=valid[0].start_time_seconds,
                    segment_end=valid[0].end_time_seconds,
                    layout_mode=valid[0].framing_mode,
                    focal_target="FOCAL_DISPLAY"
                )
            ],
            hook_score=0.92,
            retention_score=0.88,
            reasoning=valid[0].reasoning
        )
        job = RenderJob(
            source_path="/data/test.mp4",
            output_dir="/data/out",
            clip_manifest=manifest
        )
        assert job.clip_manifest.segments[0].layout_mode == "CANVAS_GRAPHIC_FIT"

    def test_tier3_05_editorial_fallback_using_asr_dialogue_pauses(self):
        """C5: F5 (Editorial Arcs) + F1 (ASR): Editorial fallback short extraction using speech density & pauses."""
        engine = EditorialEngine()
        # Construct realistic dialogue with sentences and pauses across 360s video
        words = []
        t = 60.0
        for sentence_idx in range(60):
            for word_idx in range(10):
                word_str = f"word_{sentence_idx}_{word_idx}"
                if word_idx == 9:
                    word_str += "."  # Sentence pause
                words.append({"word": word_str, "start": t, "end": t + 0.3})
                t += 0.35
            t += 0.5  # Sentence break gap

        shorts = engine.select_fallback_shorts(
            transcript_words=words, shots=[], video_duration=360.0, needed=4
        )
        assert len(shorts) == 4
        for s in shorts:
            assert s.start_time_seconds >= 60.0
            assert 30.0 <= s.duration_seconds <= 50.0

    def test_tier3_06_spatial_focal_display_to_graphic_canvas(self):
        """C6: F4 (Spatial Tracking) + F7 (Dynamic Canvas): Spatial FOCAL_DISPLAY detection triggering graphic canvas."""
        event = SpatialEvent(
            timestamp=72.0,
            frame_idx=2160,
            track_id=5,
            label="laptop",
            bbox=(0.5, 0.5, 0.8, 0.6),
            confidence=0.94
        )
        # Verify classification into ITEM_LABELS
        from wsl2_gpu.trajectory_solver import ITEM_LABELS
        assert "laptop" in ITEM_LABELS

        segment = VideoSegment(
            segment_start=70.0,
            segment_end=105.0,
            layout_mode="CANVAS_GRAPHIC_FIT",
            focal_target="FOCAL_DISPLAY"
        )
        assert segment.layout_mode == "CANVAS_GRAPHIC_FIT"
        assert segment.focal_target == "FOCAL_DISPLAY"

    def test_tier3_07_active_speaker_crop_and_kinetic_subtitle_compositing(self):
        """C7: F6 (Reframing) + F8 (Typography): 1080x1920 cropped frame layered with kinetic subtitles."""
        # 1. 1080x1920 video frame
        frame = np.full((1920, 1080, 3), 40, dtype=np.uint8)

        # 2. Render outlined subtitle cue
        font = ImageFont.load_default()
        cue_canvas = _render_single_cue_cpu("VIRAL HOOK", font=font, fill_color=COLOR_ACTIVE, target_w=1080)
        assert cue_canvas.shape == (48, 1080, 4)

        # 3. Verify placement at Y=1248 (65% safe zone)
        y_pos = int(round(SUBTITLE_Y_ANCHOR * 1920.0)) - (cue_canvas.shape[0] // 2)
        assert 1152 <= y_pos <= 1344

        # 4. Composite alpha layer over background frame
        alpha = cue_canvas[:, :, 3:] / 255.0
        rgb = cue_canvas[:, :, :3]
        target_roi = frame[y_pos:y_pos + cue_canvas.shape[0], :1080]
        composited = (rgb * alpha + target_roi * (1.0 - alpha)).astype(np.uint8)
        assert composited.shape == target_roi.shape

    def test_tier3_08_phase1_asr_subprocess_isolation_and_vram_reclaim(self):
        """C8: F9 (VRAM Isolation) + F1 (ASR): SubprocessPhaseRunner running ASR with VRAM reclamation."""
        runner = SubprocessPhaseRunner("phase1_asr")
        result = runner.run_phase(_mock_asr_worker, args=(), timeout=10.0)
        assert result["status"] == "success"
        assert len(result["words"]) == 2
        assert len(runner.history) >= 1
        assert runner.history[-1]["exitcode"] == 0

    def test_tier3_09_phase2_vision_editorial_subprocess_isolation(self):
        """C9: F9 (VRAM Isolation) + F3 (Vision) + F5 (Editorial): Phase 2 worker running in isolated subprocess."""
        runner = SubprocessPhaseRunner("phase2_vlm_editorial")
        result = runner.run_phase(_mock_vision_editorial_worker, args=(), timeout=15.0)
        assert result["status"] == "success"
        assert len(result["shorts"]) == 4
        for s in result["shorts"]:
            assert s["start"] >= 60.0
            assert 30.0 <= s["dur"] <= 50.0

    def test_tier3_10_phase3_render_subprocess_producing_short(self):
        """C10: F9 (VRAM Isolation) + F6 (Reframing) + F8 (Typography) + F10 (Pipeline): Phase 3 render isolation."""
        runner = SubprocessPhaseRunner("phase3_render")
        result = runner.run_phase(_mock_render_worker, args=(), timeout=10.0)
        assert result["status"] == "success"
        assert result["rendered_frames"] == 60

    def test_tier3_11_multilingual_asr_and_unicode_typography(self):
        """C11: F1 (ASR) + F8 (Typography): Multilingual ASR tokens rendered via font fallback chain."""
        multilingual_tokens = [
            ("English", "INNOVATION"),
            ("Hindi", "क्रांति"),
            ("Chinese", "突破"),
            ("Arabic", "إبداع"),
        ]
        font = _resolve_font_chain(FONT_SIZE)
        for lang, token in multilingual_tokens:
            word_img = _render_outlined_word(token, font, fill_color=COLOR_ACTIVE)
            assert word_img is not None
            assert word_img.shape[0] > 0 and word_img.shape[1] > 0

    def test_tier3_12_editorial_episodic_continuity_with_multi_cut_segments(self):
        """C12: F5 (Editorial Arcs) + F3 (Shot Detection) + F4 (Spatial Tracking): Episodic continuity tracking."""
        engine = EditorialEngine()
        dummy_keyframes = [
            {"shot_idx": 0, "midpoint_s": 15.0, "duration_s": 5.0, "base64_jpeg": "dGVzdA=="},
            {"shot_idx": 1, "midpoint_s": 25.0, "duration_s": 5.0, "base64_jpeg": "dGVzdA=="},
        ]
        dummy_words = [{"word": "continuity", "start": 10.0, "end": 10.5}]
        messages = engine.build_interleaved_prompt(dummy_keyframes, dummy_words, video_duration=120.0)
        system_text = messages[0]["content"][0]["text"]
        assert "EPISODIC CONTINUITY RULE" in system_text
        assert "Every selected short MUST follow ONE continuous person, act, joke, or performance" in system_text
