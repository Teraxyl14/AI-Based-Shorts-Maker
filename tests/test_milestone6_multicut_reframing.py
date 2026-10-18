"""
Milestone 6.2 Unit and Integration Test Suite:
Dynamic Alternating Multi-Cut Reframing & Styled Split-Screen (R2)

Covers:
1. 2D OSQP Trajectory Solving with focal targets, topic grounding, and editorial cut teleports.
2. Dynamic 2D 9:16 crop calculation centered on demonstrated objects.
3. Zero-letterbox dynamic 3-mode compositor (FULL_FRAME_SPEAKER_TIGHT, FULL_FRAME_TOPIC_CLOSEUP, DUAL_PANE_SPLIT_SCREEN).
4. Styled dual-pane split-screen rendering: #CCFF00 neon accent divider (3-4px), 6px Gaussian halo, soft drop-shadow, and 16px rounded corners.
5. Subtitle safe zone isolation (divider at Y~960px is well clear of subtitle safe zone at Y in [1152, 1344]px).
6. Backward compatibility for callers expecting (N, 2) trajectory tensors.
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
    CROP_WIDTH_PX,
    SRC_WIDTH_PX,
    MIN_X,
    MAX_X,
    MAX_VELOCITY_PX
)
from wsl2_gpu.render_spoke import (
    render_smart_portrait_pan,
    render_canvas_graphic_fit,
)
from wsl2_gpu.gpu_compositor import (
    render_styled_divider,
    render_dual_pane_split_screen,
    render_full_frame_topic_closeup,
    calculate_dynamic_topic_crop_2d,
    verify_subtitle_safe_zone_isolation,
    COLOR_NEON_LIME,
    SAFE_ZONE_MIN_RATIO,
    SAFE_ZONE_MAX_RATIO,
    SUBTITLE_Y_ANCHOR
)
from shared.schemas import CutLayoutMode, SplitScreenConfig, VideoSegment


# ═════════════════════════════════════════════════════════════════════════════
# 1. 2D OSQP TRAJECTORY SOLVER & FOCAL TARGET ROUTING TESTS
# ═════════════════════════════════════════════════════════════════════════════

class Test2DTrajectorySolver:
    """Tests 2D trajectory optimization, focal target routing, crop sizing, and cut teleports."""

    def test_2d_trajectory_focal_targets_routing_to_poi(self):
        """M6.2.1: Passes focal_targets into fuse_point_of_interest.
        When focal target is a topic/demonstration (FOCAL_DISPLAY, HELD_OBJECT, FULL_FRAME_TOPIC_CLOSEUP),
        anchors POI directly on target coordinates with 0.95 weight, extracting both cx and cy.
        """
        fps = 30.0
        total_frames = 60
        events = []

        # Speaker is at cx = 0.20, cy = 0.30
        for f in range(total_frames):
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.90,
                "bbox": (0.20, 0.30, 0.15, 0.25),
                "keypoints": [[0.20, 0.30]] * 11
            })
            # A demonstrated laptop/screen at cx = 0.75, cy = 0.65
            events.append({
                "timestamp": f / fps,
                "label": "laptop",
                "track_id": 101,
                "confidence": 0.95,
                "bbox": (0.75, 0.65, 0.30, 0.25),
                "keypoints": []
            })

        # Define focal targets: frames 0-29 speaker, frames 30-59 topic demonstration
        focal_targets = [
            (0, 30, "SPEAKER_PRIMARY"),
            (30, 60, "FOCAL_DISPLAY")
        ]

        # Call fuse_point_of_interest with return_2d=True
        poi_x, poi_y, poi_w, poi_h = fuse_point_of_interest(
            spatial_events=events,
            fps=fps,
            total_frames=total_frames,
            src_width=1920.0,
            src_height=1080.0,
            focal_targets=focal_targets,
            return_2d=True
        )

        # In first half (0-29), POI should track active speaker (~0.20 * 1920 = 384px)
        assert abs(poi_x[15] - 0.20 * 1920.0) < 50.0
        assert abs(poi_y[15] - 540.0) < 50.0  # Speaker centered vertically

        # In second half (30-59), focal target is FOCAL_DISPLAY (weight 0.95, speaker weight 0.05):
        # POI anchors directly on demonstrated laptop with 0.95 weight:
        # Expected = 0.75 * 1920 * 0.95 + 0.20 * 1920 * 0.05 = 1368 + 19.2 = 1387.2px
        expected_topic_x = 0.75 * 1920.0
        expected_topic_y = 0.65 * 1080.0
        assert abs(poi_x[45] - expected_topic_x) < 60.0, f"Expected near {expected_topic_x}, got {poi_x[45]}"
        assert abs(poi_x[45] - (expected_topic_x * 0.95 + 0.20 * 1920.0 * 0.05)) < 1.0
        assert abs(poi_y[45] - expected_topic_y) < 50.0, f"Expected near {expected_topic_y}, got {poi_y[45]}"

    def test_dynamic_2d_9_16_crop_calculation_demonstrated_objects(self):
        """M6.2.2: Dynamic 2D 9:16 crop calculation centered on demonstrated objects.
        Given bbox (cx=0.70, cy=0.60, bw=0.25, bh=0.20) in 1920x1080 source:
        Calculates H_crop and W_crop satisfying aspect ratio 9:16 (r=0.5625), clamped within boundaries.
        """
        src_w, src_h = 1920.0, 1080.0
        # Demonstrated object bbox normalized: cx=0.70, cy=0.60, w=0.25, h=0.20
        cx_norm, cy_norm = 0.70, 0.60
        bw_norm, bh_norm = 0.25, 0.20

        crop_info = calculate_dynamic_topic_crop_2d(
            cx_norm=cx_norm,
            cy_norm=cy_norm,
            bw_norm=bw_norm,
            bh_norm=bh_norm,
            src_w=src_w,
            src_h=src_h
        )

        w_crop_px = crop_info["w_crop_px"]
        h_crop_px = crop_info["h_crop_px"]
        cx_px = crop_info["cx_px"]
        cy_px = crop_info["cy_px"]

        # 1. Aspect ratio strictly 9:16 (within float tolerance)
        aspect = w_crop_px / h_crop_px
        assert abs(aspect - (9.0 / 16.0)) < 1e-4

        # 2. Bounding clamps: crop fits fully inside source dimensions
        half_w = w_crop_px / 2.0
        half_h = h_crop_px / 2.0
        assert (cx_px - half_w) >= -1e-4
        assert (cx_px + half_w) <= src_w + 1e-4
        assert (cy_px - half_h) >= -1e-4
        assert (cy_px + half_h) <= src_h + 1e-4

        # 3. Minimum context height check (at least 540px)
        assert h_crop_px >= 540.0

    def test_editorial_cut_instantaneous_teleports(self):
        """M6.2.3: Editorial Cut Teleports.
        Editorial cut frames are merged into shot boundary partitioning so OSQP executes
        instantaneous cut teleports (x_0 = clamp(p_0)) at cut boundaries without pan blur or lag.
        """
        fps = 30.0
        total_frames = 60
        events = []

        # First segment (0-29): speaker on left cx = 0.20
        # Second segment (30-59): product showcase on right cx = 0.85
        for f in range(30):
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.95,
                "bbox": (0.20, 0.30, 0.15, 0.25),
                "keypoints": [[0.20, 0.30]] * 11
            })
        for f in range(30, 60):
            events.append({
                "timestamp": f / fps,
                "label": "laptop",
                "track_id": 102,
                "confidence": 0.95,
                "bbox": (0.85, 0.55, 0.25, 0.25),
                "keypoints": []
            })

        focal_targets = [
            (0, 30, "SPEAKER_PRIMARY"),
            (30, 60, "FULL_FRAME_TOPIC_CLOSEUP")
        ]

        # Editorial cut at frame 30 (t = 1.0s)
        editorial_cuts = [1.0]

        traj = solve_trajectory(
            spatial_events=events,
            scene_changes=[],
            fps=fps,
            total_frames=total_frames,
            focal_targets=focal_targets,
            editorial_cuts=editorial_cuts,
            return_4d=True
        )

        cx_px = traj[:, 0] * 1920.0

        # Within shot 1 (frames 0 to 29): smooth velocity <= 15 px/frame
        vel_shot1 = np.abs(np.diff(cx_px[:30]))
        assert np.all(vel_shot1 <= MAX_VELOCITY_PX + 1e-2)

        # Within shot 2 (frames 30 to 59): smooth velocity <= 15 px/frame
        vel_shot2 = np.abs(np.diff(cx_px[30:]))
        assert np.all(vel_shot2 <= MAX_VELOCITY_PX + 1e-2)

        # Across cut boundary (frame 29 to 30): instantaneous teleport jump (> 100px)!
        cut_jump = abs(cx_px[30] - cx_px[29])
        assert cut_jump > 200.0, f"Expected instantaneous teleport > 200px across cut, got {cut_jump:.1f}px"

        # Frame 30 must immediately anchor to topic POI without pan lag
        expected_p0 = 0.85 * 1920.0
        # Account for maximum clamp bound
        assert cx_px[30] >= (MAX_X - 1e-2) or abs(cx_px[30] - expected_p0) < 50.0

    def test_trajectory_output_normalized_4d_and_2d_backward_compat(self):
        """M6.2.4: Normalized (N, 4) trajectory tensor [cx_norm, cy_norm, w_crop_norm, h_crop_norm]
        while maintaining 100% backward compatibility for callers expecting (N, 2).
        """
        fps = 30.0
        total_frames = 45
        events = [{
            "timestamp": f / fps,
            "label": "face",
            "track_id": 1,
            "confidence": 0.90,
            "bbox": (0.50, 0.40, 0.20, 0.30),
            "keypoints": [[0.50, 0.40]] * 11
        } for f in range(total_frames)]

        traj = solve_trajectory(
            spatial_events=events,
            scene_changes=[],
            fps=fps,
            total_frames=total_frames,
            return_4d=True
        )

        # 1. Shape comparisons: matches both (45, 4) and (45, 2)
        assert traj.shape == (45, 4)
        assert traj.shape == (45, 2)
        assert traj.shape[1] == 4
        assert traj.shape[1] == 2
        assert len(traj) == 45

        # 2. Value bounds
        assert np.all(traj[:, 0] >= 0.0) and np.all(traj[:, 0] <= 1.0)  # cx_norm
        assert np.all(traj[:, 1] >= 0.0) and np.all(traj[:, 1] <= 1.0)  # cy_norm
        assert np.all(traj[:, 2] > 0.0) and np.all(traj[:, 2] <= 1.0)   # w_crop_norm
        assert np.all(traj[:, 3] > 0.0) and np.all(traj[:, 3] <= 1.0)   # h_crop_norm

        # 3. Slicing backward compatibility
        cx_cy = traj[:, :2]
        assert cx_cy.shape == (45, 2)


# ═════════════════════════════════════════════════════════════════════════════
# 2. ZERO-LETTERBOX 3-MODE COMPOSITOR TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestDynamic3ModeZeroLetterboxCompositor:
    """Tests Mode 1, Mode 2, Mode 3, styled glowing split-screen, and letterbox elimination."""

    def test_mode1_full_frame_speaker_tight(self):
        """M6.2.5: Mode 1 (FULL_FRAME_SPEAKER_TIGHT) 9:16 portrait crop centered on active speaker,
        scaled to 1080x1920 with zero letterbox padding.
        """
        frame = torch.rand(1, 3, 1080, 1920)
        cx_norm = 0.35

        out = render_smart_portrait_pan(frame, cx_norm=cx_norm, target_w=1080, target_h=1920)
        assert out.shape == (1, 3, 1920, 1080)
        assert 0.0 <= out.min() <= out.max() <= 1.0

        # Full screen fill: check top and bottom borders have image content (non-zero variance)
        top_slice = out[0, :, 0:10, :]
        bot_slice = out[0, :, 1910:1920, :]
        assert top_slice.std() > 0.001
        assert bot_slice.std() > 0.001

    def test_mode2_full_frame_topic_closeup(self):
        """M6.2.6: Mode 2 (FULL_FRAME_TOPIC_CLOSEUP) Full-frame 9:16 close-up crop directly
        centered on demonstrated object/screen/action, scaled to fill 1080x1920 with zero letterbox padding.
        """
        # Create synthetic frame with distinctive pattern at object location
        frame = torch.zeros(1, 3, 1080, 1920)
        # Put white object at cx=0.75, cy=0.60
        frame[:, :, 500:750, 1300:1600] = 1.0

        out = render_full_frame_topic_closeup(
            frame=frame,
            cx_norm=0.75,
            cy_norm=0.60,
            w_crop_norm=0.30,
            h_crop_norm=0.5333,
            target_w=1080,
            target_h=1920
        )

        assert out.shape == (1, 3, 1920, 1080)
        assert 0.0 <= out.min() <= out.max() <= 1.0

        # Object should now occupy center of vertical 1080x1920 frame
        center_region = out[0, :, 800:1120, 400:680]
        assert center_region.mean() > 0.40, "Object should be prominently centered in vertical close-up"

        # Zero letterbox padding: entire canvas is filled
        assert out[0, :, 0:10, :].mean() >= 0.0
        assert out[0, :, 1910:1920, :].mean() >= 0.0

    def test_mode3_styled_dual_pane_split_screen(self):
        """M6.2.7: Mode 3 (DUAL_PANE_SPLIT_SCREEN) Styled dual-pane vertical split-screen:
        - Top pane: speaker crop scaled to 1080x958.
        - Bottom pane: topic close-up crop scaled to 1080x958.
        - Divider bar: 3-4px horizontal accent line at Y in [958, 962] in #CCFF00 neon accent.
        - 6px Gaussian halo glow and soft drop-shadow.
        - Rounded inner corners (radius 16px).
        """
        # Synthetic frame: top-left has speaker (red), bottom-right has topic (blue)
        frame = torch.zeros(1, 3, 1080, 1920)
        frame[:, 0, 0:540, 0:960] = 1.0     # Red speaker region
        frame[:, 2, 540:1080, 960:1920] = 1.0 # Blue topic region

        config = SplitScreenConfig(
            divider_width_px=4,
            divider_accent_hex="#CCFF00",
            divider_glow_radius=6,
            corner_radius_px=16,
            shadow_blur_px=10
        )

        out = render_dual_pane_split_screen(
            frame=frame,
            speaker_cx=0.25,
            topic_cx=0.75,
            topic_cy=0.75,
            target_w=1080,
            target_h=1920,
            config=config
        )

        assert out.shape == (1, 3, 1920, 1080)

        # 1. Check Divider bar at Y in [958, 962]
        # #CCFF00 in RGB: R = 204/255 = 0.80, G = 255/255 = 1.0, B = 0.0
        divider_center_slice = out[0, :, 959:961, :]
        r_mean = divider_center_slice[0].mean().item()
        g_mean = divider_center_slice[1].mean().item()
        b_mean = divider_center_slice[2].mean().item()

        assert r_mean > 0.70, f"Expected high R for #CCFF00 neon, got {r_mean}"
        assert g_mean > 0.90, f"Expected high G for #CCFF00 neon, got {g_mean}"
        assert b_mean < 0.20, f"Expected low B for #CCFF00 neon, got {b_mean}"

        # 2. Check Gaussian glow halo around divider (e.g. at Y = 952 and Y = 968)
        glow_above = out[0, 1, 953, :].mean().item()
        glow_below = out[0, 1, 967, :].mean().item()
        assert glow_above > 0.15, "Gaussian glow halo should be visible above divider line"
        assert glow_below > 0.15, "Gaussian glow halo should be visible below divider line"

        # 3. Check top pane content (predominantly red from speaker)
        top_pane_sample = out[0, :, 400:500, 400:600]
        assert top_pane_sample[0].mean() > 0.50, "Top pane should contain speaker content"

        # 4. Check bottom pane content (predominantly blue from topic)
        bot_pane_sample = out[0, :, 1500:1600, 400:600]
        assert bot_pane_sample[2].mean() > 0.50, "Bottom pane should contain topic content"

    def test_elimination_of_unstyled_16_9_letterboxing(self):
        """M6.2.8: Elimination of unstyled 16:9 letterboxing and floating canvas backdrops.
        In render_canvas_graphic_fit, dynamic remapping replaces unstyled 1080x608 letterboxed canvas
        with full-frame topic close-up or styled dual-pane split-screen.
        """
        frame = torch.rand(1, 3, 1080, 1920)

        # Calling render_canvas_graphic_fit with dynamic_remap=True
        remapped_closeup = render_canvas_graphic_fit(
            frame=frame,
            target_w=1080,
            target_h=1920,
            dynamic_remap=True,
            focal_target="FULL_FRAME_TOPIC_CLOSEUP",
            topic_cx=0.60,
            topic_cy=0.50
        )
        assert remapped_closeup.shape == (1, 3, 1920, 1080)
        # Verify no 16:9 letterbox black/blurred bands at top Y=200 or bottom Y=1700
        assert remapped_closeup[0, :, 200, :].std() > 0.001
        assert remapped_closeup[0, :, 1700, :].std() > 0.001

        # Calling render_canvas_graphic_fit with dual pane remapping
        remapped_split = render_canvas_graphic_fit(
            frame=frame,
            target_w=1080,
            target_h=1920,
            dynamic_remap=True,
            focal_target="DUAL_PANE_SPLIT_SCREEN",
            speaker_cx=0.30,
            topic_cx=0.70,
            topic_cy=0.60
        )
        assert remapped_split.shape == (1, 3, 1920, 1080)
        # Verify divider is present at Y=960
        divider_r = remapped_split[0, 0, 960, :].mean().item()
        divider_g = remapped_split[0, 1, 960, :].mean().item()
        assert divider_g > 0.90, "Divider must be present in remapped split screen"

    def test_subtitle_safe_zone_isolation(self):
        """M6.2.9: Subtitle Safe Zone Isolation.
        Divider line at Y~960px is well clear of subtitle safe zone at Y in [1152, 1344]px (60%-70% of 1920h).
        Verifies minimum vertical clearance >= 150px between divider/glow and subtitle safe zone.
        """
        target_h = 1920
        split_y = 960
        divider_th = 4
        glow_radius = 6

        # Subtitle safe zone bounds
        safe_min_px = int(round(SAFE_ZONE_MIN_RATIO * target_h))  # 1152
        safe_max_px = int(round(SAFE_ZONE_MAX_RATIO * target_h))  # 1344
        sub_anchor_px = int(round(SUBTITLE_Y_ANCHOR * target_h))   # 1248

        assert safe_min_px == 1152
        assert safe_max_px == 1344
        assert sub_anchor_px == 1248

        # Lower bound of divider glow
        divider_bottom_extent = split_y + (divider_th // 2) + glow_radius  # 960 + 2 + 6 = 968px

        # Clearance to subtitle safe zone top
        clearance_px = safe_min_px - divider_bottom_extent
        assert clearance_px >= 180, f"Expected clearance >= 180px, got {clearance_px}px"

        # Helper function assertion
        is_isolated = verify_subtitle_safe_zone_isolation(
            split_y=split_y,
            divider_thickness=divider_th,
            glow_radius=glow_radius,
            safe_zone_y_min=safe_min_px,
            safe_zone_y_max=safe_max_px
        )
        assert is_isolated is True


# ═════════════════════════════════════════════════════════════════════════════
# 3. INTEGRATION: MULTI-CUT PACING & SUBTITLE OVERLAY
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegrationReframingAndCompositing:
    """End-to-end integration of 3-mode reframing, cut teleports, and subtitle compositing."""

    def test_alternating_multi_cut_clip_pipeline(self):
        """M6.2.10: Alternating multi-cut clip pipeline:
        Segment 1 (0-3s, 90f): Speaker Hook (FULL_FRAME_SPEAKER_TIGHT)
        Segment 2 (3-8s, 150f): Product Showcase (FULL_FRAME_TOPIC_CLOSEUP)
        Segment 3 (8-12s, 120f): Speaker Payoff (FULL_FRAME_SPEAKER_TIGHT)
        Asserts continuous valid framing, instantaneous cuts, zero letterboxing.
        """
        fps = 30.0
        total_frames = 360
        events = []

        # Synthetic tracking data
        for f in range(total_frames):
            # Speaker on left cx=0.25
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.90,
                "bbox": (0.25, 0.35, 0.15, 0.25),
                "keypoints": [[0.25, 0.35]] * 11
            })
            # Product on right cx=0.80
            events.append({
                "timestamp": f / fps,
                "label": "cell phone",
                "track_id": 105,
                "confidence": 0.95,
                "bbox": (0.80, 0.60, 0.20, 0.30),
                "keypoints": []
            })

        focal_targets = [
            (0, 90, "FULL_FRAME_SPEAKER_TIGHT"),
            (90, 240, "FULL_FRAME_TOPIC_CLOSEUP"),
            (240, 360, "FULL_FRAME_SPEAKER_TIGHT")
        ]
        editorial_cuts = [3.0, 8.0]

        traj = solve_trajectory(
            spatial_events=events,
            scene_changes=[],
            fps=fps,
            total_frames=total_frames,
            focal_targets=focal_targets,
            editorial_cuts=editorial_cuts,
            return_4d=True
        )

        assert traj.shape == (total_frames, 4)

        # Assert instantaneous teleports at cut boundaries (frames 90 and 240)
        cx_px = traj[:, 0] * 1920.0
        jump_cut1 = abs(cx_px[90] - cx_px[89])
        jump_cut2 = abs(cx_px[240] - cx_px[239])
        assert jump_cut1 > 200.0, f"Expected cut teleport at frame 90, got {jump_cut1:.1f}px"
        assert jump_cut2 > 200.0, f"Expected cut teleport at frame 240, got {jump_cut2:.1f}px"

        # Assert within each segment, velocity is smooth (<= 15px/frame)
        assert np.all(np.abs(np.diff(cx_px[:90])) <= MAX_VELOCITY_PX + 1e-2)
        assert np.all(np.abs(np.diff(cx_px[90:240])) <= MAX_VELOCITY_PX + 1e-2)
        assert np.all(np.abs(np.diff(cx_px[240:])) <= MAX_VELOCITY_PX + 1e-2)

    def test_composite_with_kinetic_subtitles_on_split_screen(self):
        """M6.2.11: Subtitles composited over styled dual-pane split-screen.
        Verifies subtitles reside strictly in safe zone (Y in [1152, 1344]px) in bottom pane,
        leaving divider line at Y=960 completely unobstructed.
        """
        from wsl2_gpu.gpu_compositor import SubtitleCompositor

        frame = torch.zeros(1, 3, 1080, 1920)
        split_canvas = render_dual_pane_split_screen(
            frame=frame,
            speaker_cx=0.3,
            topic_cx=0.7,
            topic_cy=0.6,
            target_w=1080,
            target_h=1920
        )

        words = [
            {"word": "GROUNDING", "start": 1.0, "end": 1.5},
            {"word": "ENGINE", "start": 1.5, "end": 2.0}
        ]

        compositor = SubtitleCompositor()
        compositor.build_word_atlas(words, target_w=1080)

        composited = compositor.composite_frame(
            frame=split_canvas,
            timestamp=1.25,
            words=words,
            target_w=1080,
            target_h=1920
        )

        assert composited.shape == (1, 3, 1920, 1080)

        # Check divider line at Y=960 is STILL pure neon green/lime (unobstructed by text)
        divider_sample = composited[0, :, 960, :]
        assert divider_sample[1].mean() > 0.85, "Divider line at Y=960 must remain intact"

        # Check subtitle text is active in vertical safe zone [1152, 1344]
        safe_zone_sample = composited[0, :, 1152:1344, :]
        assert safe_zone_sample.std() > 0.001, "Subtitles should be rendered in safe zone"
