"""
tests/test_challenger_m2_2_stress.py - Milestone 2 Challenger 2 Empirical Stress Test Suite.

Adversarial stress-testing of:
- Feature 6: OSQP solver, RTS Kalman fallback on non-convergence / infeasibility with <= 15.0 px/f clamp,
  boundary clamping [303.75, 1616.25], degenerate shot lengths, and top-level solver fallback.
- Feature 7: Dynamic Canvas backdrop on standard, wide, ultra-wide, and non-standard / ultra-tall aspect ratios.
"""

import sys
import os
import unittest
from unittest.mock import patch
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
    MIN_X, MAX_X, MAX_VELOCITY_PX, CROP_WIDTH_PX, SRC_WIDTH_PX,
    apply_canvas_graphic_fit
)
from wsl2_gpu.render_spoke import render_canvas_graphic_fit, render_smart_portrait_pan
from conftest import verify_trajectory_bounds, verify_max_pan_velocity


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 6 EMPIRICAL STRESS TESTS: OSQP & RTS KALMAN FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature6EmpiricalStress:
    """Adversarial stress harness for Feature 6 (OSQP trajectory solver & RTS Kalman fallback)."""

    def test_osqp_rts_kalman_fallback_on_non_convergence(self):
        """Stress: Verify RTS Kalman fallback triggers when OSQP does not converge, and clamps velocity <= 15.0 px/f."""
        class MockNonConvergedResult:
            class info:
                status = "non_converged"
            x = None

        np.random.seed(1337)
        with patch("osqp.OSQP.solve", return_value=MockNonConvergedResult()):
            for trial in range(30):
                T = np.random.randint(15, 150)
                # Adversarial POI with alternating extreme steps
                poi = np.random.choice([200.0, 500.0, 960.0, 1500.0, 1800.0], size=T)
                smoothed = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X, max_vel=MAX_VELOCITY_PX)

                assert len(smoothed) == T
                assert verify_trajectory_bounds(smoothed, min_x=MIN_X, max_x=MAX_X), (
                    f"Trial {trial}: RTS Kalman fallback breached bounds: min={smoothed.min()}, max={smoothed.max()}"
                )
                assert verify_max_pan_velocity(smoothed, max_v=MAX_VELOCITY_PX, tolerance=1e-4), (
                    f"Trial {trial}: RTS Kalman fallback breached 15.0 px/frame clamp: max_diff={np.max(np.abs(np.diff(smoothed)))}"
                )

    def test_osqp_rts_kalman_fallback_on_primal_infeasible(self):
        """Stress: Verify RTS Kalman fallback triggers on primal infeasible solver status."""
        class MockInfeasibleResult:
            class info:
                status = "primal_infeasible"
            x = None

        with patch("osqp.OSQP.solve", return_value=MockInfeasibleResult()):
            poi = np.array([400.0, 1500.0, 400.0, 1500.0, 400.0, 1500.0])
            smoothed = _solve_shot_osqp(poi, min_x=MIN_X, max_x=MAX_X, max_vel=MAX_VELOCITY_PX)
            diffs = np.abs(np.diff(smoothed))
            assert np.all(diffs <= MAX_VELOCITY_PX + 1e-4)
            assert np.all(smoothed >= MIN_X - 1e-4) and np.all(smoothed <= MAX_X + 1e-4)

    def test_osqp_degenerate_shot_lengths(self):
        """Stress: Verify OSQP solver handles degenerate shot lengths (0, 1, 2, 3 frames) without unhandled exceptions."""
        # T = 0
        empty_out = _solve_shot_osqp(np.zeros(0))
        assert len(empty_out) == 0

        # T = 1
        single_out = _solve_shot_osqp(np.array([200.0]))
        assert len(single_out) == 1
        assert single_out[0] == MIN_X  # 200 clamped to MIN_X

        # T = 2
        pair_out = _solve_shot_osqp(np.array([400.0, 1600.0]))
        assert len(pair_out) == 2
        assert abs(pair_out[1] - pair_out[0]) <= MAX_VELOCITY_PX + 1e-2

        # T = 3
        tri_out = _solve_shot_osqp(np.array([400.0, 1600.0, 400.0]))
        assert len(tri_out) == 3
        assert np.all(np.abs(np.diff(tri_out)) <= MAX_VELOCITY_PX + 1e-2)

    def test_solve_trajectory_zero_and_single_frame_safeguard(self):
        """Stress: Verify top-level solve_trajectory handles total_frames <= 0 and total_frames == 1."""
        out_zero = solve_trajectory([], [], total_frames=0)
        assert out_zero.shape == (0, 2)

        out_one = solve_trajectory([], [], total_frames=1)
        assert out_one.shape == (1, 2)
        assert 0.0 <= out_one[0, 0] <= 1.0

    def test_solve_trajectory_unhandled_exception_falls_back_to_v2(self):
        """Stress: Verify solve_trajectory gracefully degrades to V2 smoother if OSQP solver raises an unexpected exception."""
        with patch("wsl2_gpu.trajectory_solver._solve_shot_osqp", side_effect=RuntimeError("Simulated OSQP solver crash")):
            spatial_events = [
                {"timestamp": 0.5, "bbox": (0.6, 0.4, 0.2, 0.3), "confidence": 0.9, "label": "person"}
            ]
            traj = solve_trajectory(spatial_events, [], fps=30.0, total_frames=60)
            assert traj.shape == (60, 2)
            assert np.all(traj[:, 0] >= 0.0) and np.all(traj[:, 0] <= 1.0)

    def test_osqp_extreme_numerical_sawtooth(self):
        """Stress: Verify OSQP solver handles high-frequency extreme sawtooth inputs without instability."""
        T = 120
        # Sawtooth bouncing between 0 and 1920 every 2 frames
        sawtooth = np.array([0.0 if i % 2 == 0 else 1920.0 for i in range(T)])
        traj = _solve_shot_osqp(sawtooth, min_x=MIN_X, max_x=MAX_X, max_vel=MAX_VELOCITY_PX)
        assert len(traj) == T
        assert verify_trajectory_bounds(traj, min_x=MIN_X, max_x=MAX_X)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 7 EMPIRICAL STRESS TESTS: DYNAMIC CANVAS ASPECT RATIOS
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature7EmpiricalStress:
    """Adversarial stress harness for Feature 7 (Dynamic Canvas backdrop & aspect ratios)."""

    def test_canvas_standard_16_9_widescreen(self):
        """Verify standard 16:9 input (1920x1080) produces 1080x1920 canvas."""
        frame = torch.zeros(1, 3, 1080, 1920)
        out = render_canvas_graphic_fit(frame)
        assert out.shape == (1, 3, 1920, 1080)
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_canvas_non_standard_4_3_and_1_1(self):
        """Verify 4:3 (1440x1080) and 1:1 (1080x1080) produce 1080x1920 canvas without slicing errors."""
        f_4_3 = torch.zeros(1, 3, 1080, 1440)
        out_4_3 = render_canvas_graphic_fit(f_4_3)
        assert out_4_3.shape == (1, 3, 1920, 1080)

        f_1_1 = torch.zeros(1, 3, 1080, 1080)
        out_1_1 = render_canvas_graphic_fit(f_1_1)
        assert out_1_1.shape == (1, 3, 1920, 1080)

    def test_canvas_ultrawide_21_9_and_32_9(self):
        """Verify 21:9 (2560x1080) and 32:9 (3840x1080) ultra-wide ratios produce valid 1080x1920 canvas."""
        f_21_9 = torch.zeros(1, 3, 1080, 2560)
        out_21_9 = render_canvas_graphic_fit(f_21_9)
        assert out_21_9.shape == (1, 3, 1920, 1080)

        f_32_9 = torch.zeros(1, 3, 1080, 3840)
        out_32_9 = render_canvas_graphic_fit(f_32_9)
        assert out_32_9.shape == (1, 3, 1920, 1080)

    def test_canvas_exact_9_16_portrait(self):
        """Verify exact 9:16 portrait input (1080x1920) produces 1080x1920 canvas."""
        f_9_16 = torch.zeros(1, 3, 1920, 1080)
        out_9_16 = render_canvas_graphic_fit(f_9_16)
        assert out_9_16.shape == (1, 3, 1920, 1080)

    def test_canvas_ultra_tall_aspect_ratio_failure_demonstration(self):
        """Empirical Challenge: Demonstrate that inputs taller than 9:16 (e.g. 9:20, 1080x2400)
        cause RuntimeError due to negative start_y in render_canvas_graphic_fit.
        
        This tests the empirical vulnerability in Feature 7 cover scale implementation.
        """
        # Ultra-tall phone screen capture / UI demo: 1080x2400 (aspect 9:20)
        frame_tall = torch.zeros(1, 3, 2400, 1080)
        with pytest.raises(RuntimeError) as exc_info:
            render_canvas_graphic_fit(frame_tall)
        
        # Verify exact error signature: shape mismatch from negative slice
        assert "The expanded size of the tensor" in str(exc_info.value)
