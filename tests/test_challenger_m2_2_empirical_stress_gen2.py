"""
tests/test_challenger_m2_2_empirical_stress_gen2.py - Milestone 2 Challenger 2 Empirical Stress Test Suite (Gen 2).

Adversarial stress-testing of Features 6 & 7:
1. Feature 6: OSQP Trajectory Solver & Fallback
   - Non-convergence / infeasibility statuses cleanly fall back to velocity-clamped RTS Kalman smoother.
   - Degenerate trajectory lengths (0, 1, 2, 3 frames) handled without unhandled exceptions.
   - Boundary clamping strictly holds within [303.75, 1616.25] px on extreme coordinates ([-1e9, 1e9]).
   - Inter-frame horizontal velocity analysis: tests 15.0 px/frame velocity ceiling under adversarial inputs.
   - Empirical characterization of OSQP unpolished ADMM tolerance residual (absence of post-solve clamp).
2. Feature 7: Dynamic Canvas Graphic Fit
   - Blurred backdrop darkening factor: 45% darkening (* 0.55) on solid white and natural inputs.
   - Foreground centering: 1080x608 uncropped foreground centered at Y=656 (Y in [656, 1264]).
   - Extreme aspect ratios: 16:9, 4:3, 21:9, 1:1, ultrawide 32:9, triple-wide 48:9.
   - Ultra-tall (>9:16) aspect ratio boundary limitation characterization.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
import torch

TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.schemas import SceneChange, SpatialEvent
from wsl2_gpu.trajectory_solver import (
    _solve_shot_osqp, solve_trajectory, fuse_point_of_interest,
    MIN_X, MAX_X, MAX_VELOCITY_PX, CROP_WIDTH_PX, SRC_WIDTH_PX
)
from wsl2_gpu.render_spoke import render_canvas_graphic_fit
from conftest import verify_trajectory_bounds, verify_max_pan_velocity


# ═════════════════════════════════════════════════════════════════════════════
# 1. FEATURE 6 EMPIRICAL STRESS TESTS: OSQP & RTS KALMAN FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature6OSQPAndKalmanFallback:
    """Empirical challenge tests for OSQP solver convergence, fallbacks, and boundaries."""

    @pytest.mark.parametrize("status", [
        "non_converged",
        "primal_infeasible",
        "dual_infeasible",
        "max_iter_reached",
        "time_limit_reached",
        "user_limit_reached",
        "unsolved"
    ])
    def test_osqp_fallback_on_all_non_converged_statuses(self, status):
        """Stress: Verify RTS Kalman fallback activates across every non-converged OSQP status code."""
        mock_res = MagicMock()
        mock_res.info.status = status
        mock_res.x = None

        with patch("osqp.OSQP.solve", return_value=mock_res):
            poi = np.array([300.0, 1600.0, 300.0, 1600.0, 300.0, 1600.0, 300.0])
            out = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X, max_vel=MAX_VELOCITY_PX)

            assert len(out) == len(poi)
            diffs = np.abs(np.diff(out))
            assert np.all(diffs <= MAX_VELOCITY_PX + 1e-4), (
                f"Status {status} fallback breached velocity limit: max {diffs.max()}"
            )
            assert np.all(out >= MIN_X - 1e-4) and np.all(out <= MAX_X + 1e-4), (
                f"Status {status} fallback breached physical bounds: min={out.min()}, max={out.max()}"
            )

    def test_osqp_fallback_when_result_x_is_none(self):
        """Stress: Verify fallback triggers if OSQP reports 'solved' but returns x=None."""
        mock_res = MagicMock()
        mock_res.info.status = "solved"
        mock_res.x = None

        with patch("osqp.OSQP.solve", return_value=mock_res):
            poi = np.array([400.0, 1500.0, 400.0])
            out = _solve_shot_osqp(poi)
            assert len(out) == 3
            assert np.all(np.abs(np.diff(out)) <= MAX_VELOCITY_PX + 1e-4)

    def test_top_level_solve_trajectory_graceful_v2_fallback_on_setup_exception(self):
        """Stress: When OSQP setup throws an unexpected RuntimeError, solve_trajectory falls back to V2 smoother."""
        with patch("osqp.OSQP.setup", side_effect=RuntimeError("Simulated OSQP setup hardware/matrix failure")):
            events = [
                {"timestamp": 0.2, "bbox": (0.75, 0.5, 0.2, 0.4), "confidence": 0.9, "label": "person"}
            ]
            traj = solve_trajectory(events, [], fps=30.0, total_frames=45)
            assert traj.shape == (45, 2)
            assert np.all(traj[:, 0] >= 0.0) and np.all(traj[:, 0] <= 1.0)
            assert np.all(traj[:, 1] >= 0.0) and np.all(traj[:, 1] <= 1.0)

    @pytest.mark.parametrize("length", [0, 1, 2, 3])
    def test_degenerate_trajectory_lengths_in_solve_shot_osqp(self, length):
        """Stress: Degenerate shot lengths (0, 1, 2, 3 frames) execute without zero-size array reductions."""
        poi = np.linspace(350.0, 1550.0, length)
        out = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X, max_vel=MAX_VELOCITY_PX)
        assert len(out) == length
        if length > 0:
            assert np.all(out >= MIN_X - 1e-4) and np.all(out <= MAX_X + 1e-4)
        if length > 1:
            assert np.all(np.abs(np.diff(out)) <= MAX_VELOCITY_PX + 1e-4)

    @pytest.mark.parametrize("total_frames", [0, 1, 2, 5])
    def test_degenerate_trajectory_lengths_in_solve_trajectory(self, total_frames):
        """Stress: solve_trajectory handles total_frames <= 0, 1, 2 cleanly."""
        out = solve_trajectory([], [], fps=30.0, total_frames=total_frames)
        assert out.shape == (total_frames, 2)
        if total_frames > 0:
            assert np.all(out[:, 0] >= 0.0) and np.all(out[:, 0] <= 1.0)

    def test_extreme_numerical_boundary_clamping(self):
        """Stress: Extremely large positive/negative coordinate inputs are strictly clamped to [303.75, 1616.25]."""
        poi = np.array([-1e12, -1e6, -500.0, 0.0, 303.74, 303.75, 1616.25, 1616.26, 2000.0, 1e6, 1e12])
        out = _solve_shot_osqp(poi)
        assert np.all(out >= MIN_X)
        assert np.all(out <= MAX_X)

        events = [
            {"timestamp": 0.1, "bbox": (-10.0, 0.5, 0.2, 0.2), "confidence": 0.9, "label": "person"},
            {"timestamp": 0.5, "bbox": (10.0, 0.5, 0.2, 0.2), "confidence": 0.9, "label": "person"}
        ]
        traj = solve_trajectory(events, [], fps=30.0, total_frames=60)
        cx_px = traj[:, 0] * SRC_WIDTH_PX
        assert np.all(cx_px >= MIN_X - 1e-4)
        assert np.all(cx_px <= MAX_X + 1e-4)

    def test_osqp_step_jump_velocity_smoothing(self):
        """Stress: 1000px instantaneous step jump within a single shot is smoothly damped <= 15.0 px/frame."""
        poi = np.array([400.0] * 40 + [1400.0] * 40)
        out = _solve_shot_osqp(poi)
        diffs = np.abs(np.diff(out))
        assert diffs.max() <= MAX_VELOCITY_PX + 1e-4
        assert out.min() >= MIN_X - 1e-4 and out.max() <= MAX_X + 1e-4

    def test_characterize_osqp_unpolished_velocity_residual(self):
        """Empirical Challenge: Characterize the OSQP solver ADMM residual when polishing fails.
        
        Demonstrates that under adversarial alternating inputs where OSQP polish fails,
        the lack of a post-solve velocity clamp in _solve_shot_osqp allows a small
        residual velocity excess (~0.0159 px/frame) above MAX_VELOCITY_PX.
        """
        np.random.seed(12345)
        # Recreate the exact adversarial trial (Trial 381: alternating extreme step jumps, T=98)
        for trial in range(382):
            T = np.random.randint(10, 200)
            mode = trial % 5
            if mode == 0:
                poi = np.random.uniform(-5000, 5000, size=T)
            elif mode == 1:
                poi = np.random.choice([0.0, 1920.0], size=T)
            elif mode == 2:
                poi = 960.0 + 1000.0 * np.sin(np.arange(T) * 1.5)
            elif mode == 3:
                poi = np.full(T, 500.0)
                poi[np.random.choice(T, size=min(5, T // 2), replace=False)] = 1800.0
            else:
                poi = 960.0 + np.cumsum(np.random.normal(0, 100, size=T))

            if trial == 381:
                out = _solve_shot_osqp(poi)
                diffs = np.abs(np.diff(out))
                max_diff = diffs.max()
                excess = max_diff - MAX_VELOCITY_PX
                # Document empirical finding: excess is small (< 0.02 px) but nonzero
                assert 0.0 < excess < 0.05, f"Expected unpolished residual excess in [0.0, 0.05], got {excess}"


# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURE 7 EMPIRICAL STRESS TESTS: DYNAMIC CANVAS GRAPHIC FIT
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature7CanvasGraphicFitEmpirical:
    """Empirical challenge tests for Canvas Graphic Fit blurred backdrop and aspect ratios."""

    def test_backdrop_45_percent_darkening_and_foreground_placement(self):
        """Verify: Backdrop is darkened by exactly 45% (* 0.55), foreground is 1080x608 at Y=656."""
        # Solid white test tensor [1, 3, 1080, 1920]
        frame = torch.ones(1, 3, 1080, 1920, dtype=torch.float32)
        out = render_canvas_graphic_fit(frame)

        assert out.shape == (1, 3, 1920, 1080)

        # Region 1: Backdrop top (Y in [0, 656))
        bg_top = out[0, :, 0:656, :]
        assert torch.allclose(bg_top, torch.tensor(0.55), atol=1e-3), (
            f"Backdrop top expected 0.55 (45% darkened), got mean {bg_top.mean():.4f}"
        )

        # Region 2: Backdrop bottom (Y in [1264, 1920))
        bg_bottom = out[0, :, 1264:1920, :]
        assert torch.allclose(bg_bottom, torch.tensor(0.55), atol=1e-3), (
            f"Backdrop bottom expected 0.55 (45% darkened), got mean {bg_bottom.mean():.4f}"
        )

        # Region 3: Centered uncropped foreground (Y in [656, 1264))
        fg = out[0, :, 656:1264, :]
        assert torch.allclose(fg, torch.tensor(1.0), atol=1e-3), (
            f"Foreground expected unaltered 1.0, got mean {fg.mean():.4f}"
        )

    @pytest.mark.parametrize("name,w,h", [
        ("16:9 Widescreen", 1920, 1080),
        ("4:3 Standard TV", 1440, 1080),
        ("21:9 Ultra-Wide", 2560, 1080),
        ("1:1 Square", 1080, 1080),
        ("32:9 Super Ultra-Wide", 3840, 1080),
        ("2.39:1 CinemaScope", 2560, 1071),
        ("IMAX 1.43:1", 1920, 1342),
        ("Triple Monitor 48:9", 5760, 1080),
        ("Old PC 5:4", 1280, 1024),
        ("Low-res 4:3", 640, 480)
    ])
    def test_extreme_aspect_ratios_render_successfully(self, name, w, h):
        """Stress: All standard, wide, ultra-wide, and square aspect ratios render without slicing faults."""
        frame = torch.rand(1, 3, h, w, dtype=torch.float32)
        out = render_canvas_graphic_fit(frame)
        assert out.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_characterize_ultra_tall_aspect_ratio_boundary(self):
        """Empirical Challenge: Characterize the ultra-tall (>9:16) aspect ratio boundary condition.
        
        Demonstrates that inputs taller than 9:16 (e.g. 1080x2400) cause a shape mismatch
        because foreground height exceeds 1920, generating a negative start_y offset.
        """
        frame_tall = torch.zeros(1, 3, 2400, 1080)
        with pytest.raises(RuntimeError) as exc_info:
            render_canvas_graphic_fit(frame_tall)
        assert "The expanded size of the tensor" in str(exc_info.value)
