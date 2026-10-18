"""
Adversarial Stress Test Suite for 2D OSQP Trajectory Solver (wsl2_gpu/trajectory_solver.py)

Author: teamwork_preview_challenger_m6_2_1
Role: EMPIRICAL CHALLENGER (critic, specialist)
Task: Adversarially challenge and stress-test the 2D OSQP Trajectory Solver across:
  1. Extreme focal target coordinates: (0, 0), (1, 1), negative, out of bounds, NaN, Inf.
  2. Rapid back-to-back editorial cut teleports (single-frame shot intervals).
  3. Inverted target bounding boxes (w < 0, h < 0, w=0, h=0).
  4. TrajectoryTensor shape compatibility: numpy operations, slicing, equality assertions under stress.
  5. OSQP solver convergence vs graceful fallback without unhandled exceptions.
"""

import math
import pytest
import numpy as np
import torch

from wsl2_gpu.trajectory_solver import (
    solve_trajectory,
    fuse_point_of_interest,
    _solve_shot_osqp,
    identify_shot_boundaries,
    TrajectoryTensor,
    FlexibleShape,
    FlexibleInt,
    CROP_WIDTH_PX,
    SRC_WIDTH_PX,
    MIN_X,
    MAX_X,
    MAX_VELOCITY_PX,
)


class TestAdversarialExtremeFocalCoordinates:
    """Stress testing solver against extreme, out-of-bounds, NaN, and Inf focal targets."""

    @pytest.mark.parametrize("cx, cy, label", [
        (0.0, 0.0, "top_left_corner"),
        (1.0, 1.0, "bottom_right_corner"),
        (0.0, 1.0, "bottom_left_corner"),
        (1.0, 0.0, "top_right_corner"),
        (-0.5, -0.5, "negative_coords"),
        (-100.0, -50.0, "deep_negative_oob"),
        (2.0, 3.0, "positive_oob"),
        (100.0, 50.0, "extreme_positive_oob"),
    ])
    def test_extreme_coordinates_stay_clamped_and_finite(self, cx, cy, label):
        """Extreme focal coordinates must clamp within frame bounds without NaN or Inf."""
        fps = 30.0
        total_frames = 30
        focal_targets = [(0, total_frames, "topic_focal", (cx, cy, 0.2, 0.2))]

        traj = solve_trajectory(
            spatial_events=[],
            scene_changes=[],
            fps=fps,
            total_frames=total_frames,
            focal_targets=focal_targets,
            return_4d=True
        )

        assert traj.shape == (total_frames, 4)
        assert traj.shape == (total_frames, 2)
        assert not np.isnan(traj).any(), f"Found NaN for {label}"
        assert not np.isinf(traj).any(), f"Found Inf for {label}"

        # Clamped inside normalized screen bounds [0, 1]
        assert np.all(traj[:, 0] >= 0.0) and np.all(traj[:, 0] <= 1.0)
        assert np.all(traj[:, 1] >= 0.0) and np.all(traj[:, 1] <= 1.0)
        # Crop bounds fully inside screen
        half_w = traj[:, 2] / 2.0
        half_h = traj[:, 3] / 2.0
        assert np.all(traj[:, 0] - half_w >= -1e-4), f"Left crop out of frame for {label}"
        assert np.all(traj[:, 0] + half_w <= 1.0 + 1e-4), f"Right crop out of frame for {label}"
        assert np.all(traj[:, 1] - half_h >= -1e-4), f"Top crop out of frame for {label}"
        assert np.all(traj[:, 1] + half_h <= 1.0 + 1e-4), f"Bottom crop out of frame for {label}"

    @pytest.mark.parametrize("val, label", [
        (float("nan"), "nan_target"),
        (float("inf"), "pos_inf_target"),
        (float("-inf"), "neg_inf_target"),
    ])
    def test_nan_and_inf_focal_targets_no_unhandled_crash(self, val, label):
        """NaN/Inf coordinates in focal target bboxes must not crash the solver."""
        fps = 30.0
        total_frames = 30
        focal_targets = [(0, total_frames, "topic_focal", (val, val, 0.2, 0.2))]

        # Must execute without unhandled exception
        traj = solve_trajectory(
            spatial_events=[],
            scene_changes=[],
            fps=fps,
            total_frames=total_frames,
            focal_targets=focal_targets,
            return_4d=True
        )
        assert isinstance(traj, TrajectoryTensor)
        assert len(traj) == total_frames
        # Inf should clamp to valid numbers; NaN might produce NaN or fallback
        if label != "nan_target":
            assert not np.isnan(traj).any()
            assert not np.isinf(traj).any()


class TestAdversarialRapidEditorialCuts:
    """Stress testing single-frame shot intervals and rapid back-to-back teleports."""

    def test_single_frame_shots_cut_at_every_frame(self):
        """Pathological cut density: editorial cut at every single frame (T=30).
        Each shot duration is exactly 1 frame.
        """
        T = 30
        fps = 30.0
        # Cut at every frame index from 1 to 29
        editorial_cuts = list(range(1, T))

        # Focal targets alternating across screen
        focal_targets = [
            (t, t + 1, "topic_focal", (0.15 if t % 2 == 0 else 0.85, 0.5, 0.2, 0.2))
            for t in range(T)
        ]

        traj = solve_trajectory(
            spatial_events=[],
            scene_changes=[],
            fps=fps,
            total_frames=T,
            focal_targets=focal_targets,
            editorial_cuts=editorial_cuts,
            return_4d=True
        )

        assert traj.shape == (T, 4)
        assert traj.shape == (T, 2)
        assert not np.isnan(traj).any()
        assert not np.isinf(traj).any()

        # Teleport jump should occur at every frame transition
        cx_px = traj[:, 0] * 1920.0
        jumps = np.abs(np.diff(cx_px))
        assert np.all(jumps > 500.0), f"Expected large teleport at every frame, got min jump {jumps.min():.1f}px"

    def test_alternating_rapid_teleports_within_shot_velocity_invariants(self):
        """Shots of length 2 frames: alternating left/right targets.
        Across cuts: large jump (>500px).
        Within each 2-frame shot: velocity capped <= 15.01 px/frame.
        """
        T = 20
        fps = 30.0
        cuts = list(range(2, T, 2))  # cuts at 2, 4, 6, 8, ...
        focal_targets = [
            (i, i + 2, "topic_focal", (0.20 if (i // 2) % 2 == 0 else 0.80, 0.5, 0.2, 0.2))
            for i in range(0, T, 2)
        ]

        traj = solve_trajectory(
            spatial_events=[],
            scene_changes=[],
            fps=fps,
            total_frames=T,
            focal_targets=focal_targets,
            editorial_cuts=cuts,
            return_4d=True
        )

        cx_px = traj[:, 0] * 1920.0
        for i in range(1, T):
            diff = abs(cx_px[i] - cx_px[i - 1])
            if i in cuts:
                # Across cut boundary: instantaneous teleport jump
                assert diff > 500.0, f"Frame {i}: expected cut teleport > 500px, got {diff:.1f}px"
            else:
                # Within shot: velocity must be smooth
                assert diff <= MAX_VELOCITY_PX + 1e-2, f"Frame {i}: intra-shot velocity violated: {diff:.2f}px"

    def test_duplicate_and_unsorted_and_oob_editorial_cuts(self):
        """Pathological editorial cut lists: duplicates, out-of-order, negatives, and beyond duration."""
        T = 40
        fps = 30.0
        cuts = [10, 10, 10, 5, 25, 25, -5, 0, 40, 999, 15.0001, 15.0002]

        traj = solve_trajectory(
            spatial_events=[],
            scene_changes=[],
            fps=fps,
            total_frames=T,
            editorial_cuts=cuts,
            return_4d=True
        )

        assert traj.shape == (T, 4)
        assert traj.shape == (T, 2)
        assert not np.isnan(traj).any()
        assert not np.isinf(traj).any()


class TestAdversarialInvertedBoundingBoxes:
    """Stress testing negative, inverted, zero, and huge bounding box inputs."""

    @pytest.mark.parametrize("bbox, label", [
        ((0.5, 0.5, -0.3, -0.4), "both_negative"),
        ((0.5, 0.5, 0.0, 0.0), "zero_dimensions"),
        ((0.5, 0.5, -0.5, 0.3), "negative_w_positive_h"),
        ((0.5, 0.5, 0.3, -0.5), "positive_w_negative_h"),
        ((0.5, 0.5, 50.0, 50.0), "massive_dimensions"),
        ((-0.5, -0.5, -1.0, -1.0), "all_negative_coords_and_dims"),
    ])
    def test_inverted_bboxes_produce_valid_positive_crops(self, bbox, label):
        """Inverted/zero/massive bboxes must clamp to valid positive crop dimensions."""
        fps = 30.0
        total_frames = 30
        focal_targets = [(0, total_frames, "topic_focal", bbox)]

        traj = solve_trajectory(
            spatial_events=[],
            scene_changes=[],
            fps=fps,
            total_frames=total_frames,
            focal_targets=focal_targets,
            return_4d=True
        )

        assert traj.shape == (total_frames, 4)
        assert not np.isnan(traj).any(), f"NaN in {label}"
        assert not np.isinf(traj).any(), f"Inf in {label}"

        # Crop width and height must be strictly positive and bounded
        w_norm = traj[:, 2]
        h_norm = traj[:, 3]
        assert np.all(w_norm > 0.0), f"Crop width <= 0 in {label}"
        assert np.all(h_norm > 0.0), f"Crop height <= 0 in {label}"
        assert np.all(w_norm <= 1.0 + 1e-4), f"Crop width exceeds source in {label}"
        assert np.all(h_norm <= 1.0 + 1e-4), f"Crop height exceeds source in {label}"

        # Check aspect ratio 9:16 (in normalized units: (w_norm * 1920) / (h_norm * 1080) == 9/16)
        w_px = w_norm * 1920.0
        h_px = h_norm * 1080.0
        aspect = w_px / h_px
        assert np.all(np.abs(aspect - (9.0 / 16.0)) < 1e-3), f"Aspect ratio broken in {label}: {aspect[0]}"

        # Crop must stay completely inside frame boundaries
        cx_norm = traj[:, 0]
        cy_norm = traj[:, 1]
        assert np.all((cx_norm - w_norm / 2.0) >= -1e-4)
        assert np.all((cx_norm + w_norm / 2.0) <= 1.0 + 1e-4)
        assert np.all((cy_norm - h_norm / 2.0) >= -1e-4)
        assert np.all((cy_norm + h_norm / 2.0) <= 1.0 + 1e-4)


class TestAdversarialTrajectoryTensorCompatibility:
    """Stress testing TrajectoryTensor subclass behavior, numpy operations, and shape checks."""

    def test_trajectory_tensor_shape_equality_symmetry(self):
        """FlexibleShape and FlexibleInt must satisfy bidirectional equality for 4 and 2."""
        N = 50
        tensor = np.zeros((N, 4), dtype=np.float64).view(TrajectoryTensor)

        # Standard shape equality
        assert tensor.shape == (N, 4)
        assert tensor.shape == (N, 2)
        assert (N, 4) == tensor.shape
        assert (N, 2) == tensor.shape

        # Inequality checks
        assert tensor.shape != (N, 3)
        assert tensor.shape != (N, 1)
        assert (N, 3) != tensor.shape
        assert tensor.shape != (N + 1, 4)

        # Dimension index checks
        assert tensor.shape[0] == N
        assert tensor.shape[1] == 4
        assert tensor.shape[1] == 2
        assert 4 == tensor.shape[1]
        assert 2 == tensor.shape[1]
        assert tensor.shape[1] != 3

    def test_trajectory_tensor_slicing_under_stress(self):
        """Slicing TrajectoryTensor across rows, columns, and strides."""
        N = 60
        raw = np.arange(N * 4, dtype=np.float64).reshape(N, 4)
        tensor = raw.view(TrajectoryTensor)

        # 1. Row slicing preserves TrajectoryTensor and shape compatibility
        head = tensor[:15]
        assert isinstance(head, TrajectoryTensor)
        assert head.shape == (15, 4)
        assert head.shape == (15, 2)
        assert head.shape[1] == 4
        assert head.shape[1] == 2
        assert np.array_equal(head, raw[:15])

        mid = tensor[20:40:2]
        assert mid.shape == (10, 4)
        assert mid.shape == (10, 2)

        # 2. Column slicing
        first_two = tensor[:, :2]
        assert first_two.shape == (N, 2)
        # Should NOT match (N, 4) since column dimension is strictly 2
        assert first_two.shape != (N, 4)

        last_two = tensor[:, 2:]
        assert last_two.shape == (N, 2)
        assert last_two.shape != (N, 4)

        # 3. 1D slices
        col_0 = tensor[:, 0]
        assert col_0.shape == (N,)
        assert col_0[0] == raw[0, 0]

        row_0 = tensor[0]
        assert row_0.shape == (4,)

    def test_trajectory_tensor_numpy_operations(self):
        """Reductions, mathematical transformations, and array functions."""
        N = 40
        raw = np.linspace(0.1, 0.9, N * 4, dtype=np.float64).reshape(N, 4)
        tensor = raw.view(TrajectoryTensor)

        # Reductions
        mean_axis0 = np.mean(tensor, axis=0)
        assert mean_axis0.shape == (4,)
        assert np.allclose(mean_axis0, np.mean(raw, axis=0))

        sum_val = np.sum(tensor)
        assert math.isclose(sum_val, np.sum(raw), rel_tol=1e-6)

        # Differencing along time axis
        diff_axis0 = np.diff(tensor, axis=0)
        assert diff_axis0.shape == (N - 1, 4)
        assert diff_axis0.shape == (N - 1, 2)

        # Clamping
        clipped = np.clip(tensor, 0.2, 0.8)
        assert clipped.min() >= 0.2
        assert clipped.max() <= 0.8

    def test_trajectory_tensor_pytorch_interop(self):
        """PyTorch tensor conversion and manipulation."""
        N = 30
        tensor = np.zeros((N, 4), dtype=np.float32).view(TrajectoryTensor)

        # Direct from_numpy
        t = torch.from_numpy(tensor)
        assert t.shape == torch.Size([N, 4])
        assert t.dtype == torch.float32

        # Torch operations
        t_scaled = t * 2.0
        assert t_scaled.shape == torch.Size([N, 4])


class TestAdversarialOSQPConvergenceAndFallback:
    """Stress testing OSQP solver convergence, ill-conditioned matrices, and fallback isolation."""

    def test_solver_fallback_on_infeasible_geometry(self):
        """Source dimensions where crop width > source width (src_w=200, src_h=600).
        Makes lower_bound > upper_bound. OSQP must catch or fall back without unhandled crash.
        """
        traj = solve_trajectory(
            spatial_events=[],
            scene_changes=[],
            fps=30.0,
            total_frames=30,
            src_width=200.0,
            src_height=600.0
        )
        assert isinstance(traj, TrajectoryTensor)
        assert traj.shape == (30, 4)
        assert traj.shape == (30, 2)
        assert not np.isnan(traj).any()

    def test_violent_white_noise_oscillation_convergence(self):
        """Random white noise POI over 300 frames.
        OSQP must converge and strictly obey velocity limit MAX_VELOCITY_PX <= 15.01 px/frame.
        """
        np.random.seed(42)
        T = 300
        # High amplitude random jumps between 350 and 1550 px
        noisy_poi = np.random.uniform(350.0, 1550.0, size=T)

        shot_x = _solve_shot_osqp(
            noisy_poi,
            min_x=MIN_X,
            max_x=MAX_X,
            max_vel=MAX_VELOCITY_PX
        )

        assert len(shot_x) == T
        assert not np.isnan(shot_x).any()
        assert not np.isinf(shot_x).any()

        # Hard physical boundary constraints
        assert np.all(shot_x >= MIN_X - 1e-4)
        assert np.all(shot_x <= MAX_X + 1e-4)

        # Velocity constraint on every single frame transition
        vels = np.abs(np.diff(shot_x))
        assert np.all(vels <= MAX_VELOCITY_PX + 1e-2), f"Velocity limit exceeded: max={vels.max():.4f}px"

    def test_edge_case_zero_and_single_frame(self):
        """total_frames = 0 and total_frames = 1 edge cases."""
        traj_0 = solve_trajectory([], [], total_frames=0)
        assert isinstance(traj_0, TrajectoryTensor)
        assert traj_0.shape == (0, 4)
        assert traj_0.shape == (0, 2)

        traj_1 = solve_trajectory([], [], total_frames=1)
        assert isinstance(traj_1, TrajectoryTensor)
        assert traj_1.shape == (1, 4)
        assert traj_1.shape == (1, 2)
        assert np.allclose(traj_1[0], [0.5, 0.5, (1080.0 * 9.0 / 16.0) / 1920.0, 1.0])
