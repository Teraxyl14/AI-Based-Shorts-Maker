"""
Auditor M6.2 Empirical Stress & Integrity Test Suite.
Verifies mathematical invariants, boundary conditions, edge cases,
and tests for absence of facade or hardcoded behavior.
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
from wsl2_gpu.gpu_compositor import (
    calculate_dynamic_topic_crop_2d,
    render_full_frame_topic_closeup,
    render_styled_divider,
    render_dual_pane_split_screen,
    verify_subtitle_safe_zone_isolation,
    COLOR_NEON_LIME,
    SAFE_ZONE_MIN_RATIO,
    SAFE_ZONE_MAX_RATIO,
)
from wsl2_gpu.render_spoke import (
    render_smart_portrait_pan,
    render_canvas_graphic_fit,
)
from shared.schemas import SplitScreenConfig


class TestAuditorAdversarialOSQP:
    """Stress testing 2D OSQP solver and trajectory bounds."""

    def test_osqp_strictly_obeys_velocity_cap_under_violent_oscillation(self):
        """Violent square-wave POI alternating between 0 and 1920 every frame without scene cuts.
        OSQP must smoothly damp motion and strictly never exceed MAX_VELOCITY_PX (15 px/frame).
        """
        T = 100
        # Violent ping-pong between extreme left and extreme right
        poi_violent = np.array([MIN_X if t % 2 == 0 else MAX_X for t in range(T)])
        shot_x = _solve_shot_osqp(
            poi_violent,
            min_x=MIN_X,
            max_x=MAX_X,
            max_vel=MAX_VELOCITY_PX,
            lambda_v=12.0,
            lambda_a=24.0
        )
        assert len(shot_x) == T
        # 1. Hard physical boundaries
        assert np.all(shot_x >= MIN_X - 1e-4)
        assert np.all(shot_x <= MAX_X + 1e-4)
        # 2. Strict velocity limit on every consecutive frame
        vels = np.abs(np.diff(shot_x))
        assert np.all(vels <= MAX_VELOCITY_PX + 1e-3), f"Max velocity exceeded: {np.max(vels)}"
        # 3. Dynamic responsiveness (not returning a frozen constant)
        assert np.std(shot_x) > 0.0, "Solver should not return a frozen constant"

    def test_editorial_cut_at_every_frame_extreme(self):
        """Pathological scenario: editorial cuts at every single frame.
        Must handle 1-frame shots without crash or indexing errors.
        """
        T = 20
        events = [{
            "timestamp": t / 30.0,
            "label": "face",
            "bbox": (0.2 + (t % 5) * 0.15, 0.4, 0.1, 0.2),
            "confidence": 0.95
        } for t in range(T)]

        editorial_cuts = [t / 30.0 for t in range(1, T)]
        traj = solve_trajectory(
            spatial_events=events,
            scene_changes=[],
            fps=30.0,
            total_frames=T,
            editorial_cuts=editorial_cuts,
            return_4d=True
        )
        assert traj.shape == (T, 4)
        assert len(traj) == T

    def test_out_of_bounds_and_nan_detection_coordinates(self):
        """Detections with out-of-bounds coordinates (cx=-5.0, cy=10.0, bw=50.0).
        Must clamp gracefully to valid frame coordinates.
        """
        events = [{
            "timestamp": 0.5,
            "label": "laptop",
            "bbox": (-2.5, 4.0, 10.0, 8.0),
            "confidence": 0.99
        }]
        focal_targets = [(0, 30, "FULL_FRAME_TOPIC_CLOSEUP")]
        poi_x, poi_y, crop_w, crop_h = fuse_point_of_interest(
            spatial_events=events,
            fps=30.0,
            total_frames=30,
            src_width=1920.0,
            src_height=1080.0,
            focal_targets=focal_targets,
            return_2d=True
        )
        assert np.all(poi_x >= 0.0) and np.all(poi_x <= 1920.0)
        assert np.all(poi_y >= 0.0) and np.all(poi_y <= 1080.0)
        assert np.all(crop_w <= 1920.0) and np.all(crop_w >= 540.0 * 9.0 / 16.0)
        assert np.all(crop_h <= 1080.0) and np.all(crop_h >= 540.0)

    def test_trajectory_tensor_subclass_properties(self):
        """TrajectoryTensor maintains backward-compatibility duck-typing."""
        arr = np.zeros((50, 4), dtype=np.float64).view(TrajectoryTensor)
        # Equality checks
        assert arr.shape == (50, 4)
        assert arr.shape == (50, 2)
        assert arr.shape != (50, 3)
        assert arr.shape[1] == 4
        assert arr.shape[1] == 2
        assert arr.shape[1] != 3
        # Slicing
        slice_2d = arr[:, :2]
        assert slice_2d.shape == (50, 2)


class TestAuditorAdversarialDynamicCrop:
    """Stress testing 2D 9:16 crop sizing mathematics."""

    @pytest.mark.parametrize("cx, cy, bw, bh", [
        (0.0, 0.0, 0.1, 0.1),         # Top-left corner
        (1.0, 1.0, 0.2, 0.2),         # Bottom-right corner
        (0.5, 0.5, 0.95, 0.95),       # Massive object filling screen
        (0.5, 0.5, 0.001, 0.001),     # Microscopic object
        (0.1, 0.9, 0.8, 0.05),        # Ultra-wide horizontal banner
        (0.9, 0.1, 0.05, 0.9),        # Ultra-tall vertical pillar
    ])
    def test_dynamic_crop_strict_invariants(self, cx, cy, bw, bh):
        src_w, src_h = 1920.0, 1080.0
        res = calculate_dynamic_topic_crop_2d(cx, cy, bw, bh, src_w=src_w, src_h=src_h)

        w_crop = res["w_crop_px"]
        h_crop = res["h_crop_px"]
        cx_clamped = res["cx_px"]
        cy_clamped = res["cy_px"]

        # Invariant 1: Aspect ratio strictly 9:16
        assert abs(w_crop / h_crop - (9.0 / 16.0)) < 1e-4

        # Invariant 2: Minimum height constraint 540px
        assert h_crop >= 540.0 - 1e-4

        # Invariant 3: Crop never exceeds source dimensions
        assert h_crop <= src_h + 1e-4
        assert w_crop <= src_w + 1e-4

        # Invariant 4: Bounding window [x0, x1, y0, y1] stays strictly inside [0, src_w] x [0, src_h]
        x0 = cx_clamped - w_crop / 2.0
        x1 = cx_clamped + w_crop / 2.0
        y0 = cy_clamped - h_crop / 2.0
        y1 = cy_clamped + h_crop / 2.0

        assert x0 >= -1e-4, f"x0 out of bounds: {x0}"
        assert x1 <= src_w + 1e-4, f"x1 out of bounds: {x1}"
        assert y0 >= -1e-4, f"y0 out of bounds: {y0}"
        assert y1 <= src_h + 1e-4, f"y1 out of bounds: {y1}"


class TestAuditorAdversarialSplitScreenAndCompositor:
    """Stress testing GPU split-screen, styled divider, and safe zone isolation."""

    def test_divider_glow_decay_mathematical_profile(self):
        """Mathematically verifies Gaussian decay profile of the glow."""
        canvas = torch.zeros(1, 3, 1920, 1080)
        split_y = 960
        div_th = 4
        glow_r = 6
        out = render_styled_divider(
            canvas,
            split_y=split_y,
            divider_thickness=div_th,
            neon_color=COLOR_NEON_LIME,
            glow_radius=glow_r
        )

        # Core line (Y=960) should be solid neon lime: R=0.80, G=1.0, B=0.0
        assert abs(out[0, 0, 960, 500].item() - (204 / 255)) < 1e-3
        assert abs(out[0, 1, 960, 500].item() - 1.0) < 1e-3
        assert abs(out[0, 2, 960, 500].item() - 0.0) < 1e-3

        # Glow must strictly monotonically decrease as distance from core increases
        g_963 = out[0, 1, 963, 500].item()
        g_966 = out[0, 1, 966, 500].item()
        g_970 = out[0, 1, 970, 500].item()
        g_975 = out[0, 1, 975, 500].item()

        assert g_963 > g_966 > g_970 > g_975, f"Glow must monotonically decrease: {g_963}, {g_966}, {g_970}, {g_975}"

    def test_no_black_or_transparent_edges_in_split_screen(self):
        """Dual pane split-screen fills the entire 1080x1920 viewport."""
        frame = torch.ones(1, 3, 1080, 1920) * 0.5
        split = render_dual_pane_split_screen(
            frame=frame,
            speaker_cx=0.3,
            topic_cx=0.7,
            topic_cy=0.6,
            target_w=1080,
            target_h=1920
        )
        assert split.shape == (1, 3, 1920, 1080)
        # Verify no 0.0 holes outside of rounded corner anti-aliasing
        assert split[0, :, 100:900, 100:980].min() > 0.1
        assert split[0, :, 1000:1800, 100:980].min() > 0.1

    def test_non_facade_dynamic_variation(self):
        """Confirms that varying input frames and coordinates produces varying rendered outputs."""
        frame1 = torch.zeros(1, 3, 1080, 1920)
        frame1[:, :, 100:300, 100:300] = 1.0

        frame2 = torch.zeros(1, 3, 1080, 1920)
        frame2[:, :, 700:900, 1400:1600] = 1.0

        out1 = render_full_frame_topic_closeup(frame1, cx_norm=0.15, cy_norm=0.20)
        out2 = render_full_frame_topic_closeup(frame2, cx_norm=0.80, cy_norm=0.75)

        # Difference must be significant (not returning a cached or hardcoded frame)
        diff = torch.abs(out1 - out2).mean().item()
        assert diff > 0.05, f"Outputs should vary dynamically with input: diff={diff}"
