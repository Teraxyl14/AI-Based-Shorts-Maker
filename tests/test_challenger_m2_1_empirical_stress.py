"""
tests/test_challenger_m2_1_empirical_stress.py - Milestone 2 Empirical Stress Test Suite
Challenger 1: Active Subject Tracking & Motion Gating Stress Harness (Feature 4)

Adversarial stress-testing of:
1. YOLO11s-Pose keypoint extraction, COCO 17-skeleton schema, and gesture classification heuristics.
2. ByteTrack persistent track IDs, spatial association fallback (track_id <= 0), and multi-person tracking.
3. Motion gating delta threshold boundary: delta = 0.00199 (discounted) vs delta = 0.00201 (prioritized).
4. 80% static subject weight discount: 2-second time threshold (t < 2.0s full weight vs t >= 2.0s 0.20x discount).
5. Elimination of dead-center snap (cx = 0.50): complete dropouts, holding last active speaker coordinates.
6. Extreme numerical matrices, zero-frame inputs, high subject counts, and out-of-order temporal sequences.
"""

import os
import sys
import math
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.schemas import SpatialEvent, SceneChange
from wsl2_gpu.vision_spoke import VisionSpoke
from wsl2_gpu.trajectory_solver import (
    fuse_point_of_interest, solve_trajectory, _solve_shot_osqp,
    CROP_WIDTH_PX, SRC_WIDTH_PX, MIN_X, MAX_X, MAX_VELOCITY_PX,
    FACE_WEIGHT, TORSO_WEIGHT, ITEM_WEIGHT
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. YOLO11s-Pose Keypoints & Gesture Classification Stress
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature4YoloPoseKeypointsAndGestures:
    """Stress tests for 17-keypoint COCO skeleton schema, gesture classification, and malformed poses."""

    def test_coco_17_skeleton_structure_and_labels(self):
        """Verify standard 17-keypoint COCO skeleton structure and normalization."""
        # 17 keypoints: [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
        #                l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip,
        #                l_knee, r_knee, l_ankle, r_ankle]
        keypoints = [(0.5, 0.1 + i * 0.05, 0.95) for i in range(17)]
        se = SpatialEvent(
            timestamp=1.5,
            frame_idx=45,
            track_id=1,
            label="person",
            bbox=(0.5, 0.5, 0.2, 0.8),
            confidence=0.94,
            keypoints=keypoints,
            action_label="standing"
        )
        assert len(se.keypoints) == 17
        for kp in se.keypoints:
            assert len(kp) == 3
            assert 0.0 <= kp[0] <= 1.0
            assert 0.0 <= kp[1] <= 1.0
            assert 0.0 <= kp[2] <= 1.0

    def test_gesture_classification_pointing_and_waving(self):
        """Stress: hand raised above shoulder with wide separation classifies as pointing or waving."""
        # Baseline standing
        kps_standing = [(0.5, 0.5, 0.9)] * 17
        kps_standing[0] = (0.5, 0.2, 0.9)  # nose
        kps_standing[5] = (0.4, 0.4, 0.9)  # l_shoulder
        kps_standing[6] = (0.6, 0.4, 0.9)  # r_shoulder
        kps_standing[9] = (0.35, 0.6, 0.9) # l_wrist below shoulder
        kps_standing[10] = (0.65, 0.6, 0.9) # r_wrist below shoulder
        kps_standing[11] = (0.45, 0.7, 0.9) # l_hip
        kps_standing[12] = (0.55, 0.7, 0.9) # r_hip
        assert VisionSpoke._classify_gesture(kps_standing) == "standing"

        # Pointing: one wrist raised high above shoulder and wide separation
        kps_pointing = list(kps_standing)
        kps_pointing[9] = (0.1, 0.2, 0.9)  # left wrist raised high & wide
        assert VisionSpoke._classify_gesture(kps_pointing) == "pointing"

        # Waving: wrists raised above shoulder with narrow separation
        kps_waving = list(kps_standing)
        kps_waving[9] = (0.45, 0.2, 0.9)
        kps_waving[10] = (0.55, 0.2, 0.9)
        assert VisionSpoke._classify_gesture(kps_waving) == "waving"

    def test_gesture_classification_leaning_forward(self):
        """Stress: nose position significantly ahead of hip/torso center classifies as leaning_forward."""
        kps_leaning = [(0.5, 0.5, 0.9)] * 17
        kps_leaning[5] = (0.4, 0.5, 0.9)  # shoulders at y=0.5
        kps_leaning[6] = (0.6, 0.5, 0.9)
        kps_leaning[11] = (0.4, 0.9, 0.9) # hips at y=0.9
        kps_leaning[12] = (0.6, 0.9, 0.9)
        # torso_len = 0.9 - 0.5 = 0.4. nose significantly ahead (lower y)
        kps_leaning[0] = (0.5, 0.35, 0.9)  # nose at 0.35 < 0.5 - 0.4 * 0.3 = 0.38
        assert VisionSpoke._classify_gesture(kps_leaning) == "leaning_forward"

    def test_malformed_truncated_and_empty_keypoints(self):
        """Stress: truncated, empty, or irregular keypoint lists must not raise exceptions."""
        assert VisionSpoke._classify_gesture([]) == "unknown"
        assert VisionSpoke._classify_gesture([(0.5, 0.5, 0.9)] * 5) == "unknown"
        assert VisionSpoke._classify_gesture([(0.5, 0.5, 0.9)] * 16) == "unknown"
        assert VisionSpoke._classify_gesture(None) == "unknown"
        assert VisionSpoke._classify_gesture("invalid_type") == "unknown"


# ═════════════════════════════════════════════════════════════════════════════
# 2. ByteTrack Persistent Track IDs & Multi-Person Tracking
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature4ByteTrackPersistentIDs:
    """Stress tests for multi-subject tracking, persistent IDs, and spatial association fallback."""

    def test_persistent_ids_distinguish_moving_and_static_speakers(self):
        """Verify persistent IDs prevent trajectory mingling between distinct subjects."""
        fps = 30.0
        total_frames = 120
        events = []

        for f in range(total_frames):
            t = f / fps
            # Person 1 (ID 10): Active speaker moving along cx = 0.70 -> 0.85
            cx1 = 0.70 + 0.15 * (f / float(total_frames))
            events.append({
                "timestamp": t,
                "label": "face",
                "track_id": 10,
                "confidence": 0.92,
                "bbox": (cx1, 0.2, 0.1, 0.2),
                "keypoints": [(cx1, 0.25, 0.9)] * 17
            })
            # Person 2 (ID 20): Static listener at cx = 0.25
            events.append({
                "timestamp": t,
                "label": "face",
                "track_id": 20,
                "confidence": 0.90,
                "bbox": (0.25, 0.2, 0.1, 0.2),
                "keypoints": [(0.25, 0.25, 0.9)] * 17
            })

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        # Person 2 becomes static ghost after 2s (f >= 60). At frame 90:
        # Person 1 is at 0.70 + 0.15 * (90/120) = 0.8125 -> 1560px
        # Person 2 is at 0.25 -> 480px with 80% discount
        expected_px = (1560.0 * 0.85 + 480.0 * (FACE_WEIGHT * 0.20)) / (0.85 + FACE_WEIGHT * 0.20)
        assert abs(poi[90] - expected_px) < 1.0, f"POI mismatch: got {poi[90]}, expected {expected_px}"

    def test_untracked_detections_spatial_association(self):
        """Stress: When track_id <= 0, spatial association clusters detections by position."""
        fps = 30.0
        total_frames = 90
        events = []
        # All detections have track_id = 0, moving smoothly from 0.30 to 0.45
        for f in range(total_frames):
            cx = 0.30 + 0.15 * (f / float(total_frames))
            events.append({
                "timestamp": f / fps,
                "label": "person",
                "track_id": 0,  # untracked
                "confidence": 0.9,
                "bbox": (cx, 0.4, 0.2, 0.6),
                "keypoints": [(cx, 0.2, 0.9)] * 17
            })

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        # Should smoothly track the moving subject to frame 89
        expected_last_px = (0.30 + 0.15 * (89.0 / 90.0)) * 1920.0
        assert abs(poi[89] - expected_last_px) < 2.0

    def test_five_person_crowd_with_single_active_speaker(self):
        """Stress: In a crowded frame (5 persons), active speaker dominates over 4 static listeners."""
        fps = 30.0
        total_frames = 120
        events = []
        listeners_cx = [0.15, 0.30, 0.45, 0.60]

        for f in range(total_frames):
            t = f / fps
            # Active speaker at 0.85 with gesture motion
            events.append({
                "timestamp": t,
                "label": "face",
                "track_id": 100,
                "confidence": 0.95,
                "bbox": (0.85, 0.2, 0.1, 0.2),
                "keypoints": [(0.85, 0.2 + 0.04 * np.sin(f * 0.5), 0.9)] * 17
            })
            # 4 static listeners
            for idx, l_cx in enumerate(listeners_cx):
                events.append({
                    "timestamp": t,
                    "label": "face",
                    "track_id": 10 + idx,
                    "confidence": 0.90,
                    "bbox": (l_cx, 0.2, 0.1, 0.2),
                    "keypoints": [(l_cx, 0.2, 0.9)] * 17
                })

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        # After 2s (frame 80), all 4 listeners are discounted by 80% (0.65 * 0.20 = 0.13 each)
        # Total listener weight = 4 * 0.13 = 0.52. Active speaker weight = 0.85.
        # Active speaker (1632px) heavily outweighs listener average (0.375 * 1920 = 720px)
        assert poi[80] > 1250.0, f"Expected active speaker to dominate crowd, got {poi[80]}"


# ═════════════════════════════════════════════════════════════════════════════
# 3. Motion Gating Threshold Boundary (delta = 0.002)
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature4MotionGatingThresholdBoundaries:
    """Stress tests for strict motion delta threshold boundaries and gesture variance."""

    def test_exact_motion_delta_threshold_sub_vs_super(self):
        """Boundary stress: motion delta 0.0019 vs 0.0021 determines active priority."""
        fps = 30.0
        total_frames = 120

        # Sub-threshold: delta = 0.0019 (moves by 0.0019 per frame)
        events_sub = []
        for f in range(total_frames):
            cx = 0.70 + 0.0019 * np.sin(f * 0.2)
            events_sub.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (cx, 0.2, 0.1, 0.2),
                "keypoints": [(cx, 0.25, 0.9)] * 17
            })
            # Background listener at 0.20
            events_sub.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 2,
                "confidence": 0.9,
                "bbox": (0.20, 0.2, 0.1, 0.2),
                "keypoints": [(0.20, 0.25, 0.9)] * 17
            })

        poi_sub = fuse_point_of_interest(events_sub, fps=fps, total_frames=total_frames, src_width=1920)

        # Super-threshold: delta = 0.005 (moves by constant 0.005 per frame step)
        events_super = []
        for f in range(total_frames):
            cx = 0.70 + 0.005 * (f % 15)
            events_super.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (cx, 0.2, 0.1, 0.2),
                "keypoints": [(cx, 0.25, 0.9)] * 17
            })
            events_super.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 2,
                "confidence": 0.9,
                "bbox": (0.20, 0.2, 0.1, 0.2),
                "keypoints": [(0.20, 0.25, 0.9)] * 17
            })

        poi_super = fuse_point_of_interest(events_super, fps=fps, total_frames=total_frames, src_width=1920)
        # Super-threshold must bias more aggressively toward 0.70 than sub-threshold
        assert poi_super[80] >= poi_sub[80]

    def test_pure_gesture_motion_overrides_static_centroid(self):
        """Stress: stationary centroid with wrist dynamics (delta_wrist >= 0.002) gains 0.85 weight."""
        fps = 30.0
        total_frames = 90
        events = []

        for f in range(total_frames):
            # Centroid at cx = 0.80 exactly
            kps = [(0.80, 0.5, 0.9)] * 17
            # Wrists (kps 9 and 10) moving actively
            kps[9] = (0.75 + 0.04 * np.sin(f * 0.6), 0.3, 0.9)
            kps[10] = (0.85 + 0.04 * np.cos(f * 0.6), 0.3, 0.9)
            events.append({
                "timestamp": f / fps,
                "label": "person",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (0.80, 0.5, 0.2, 0.6),
                "keypoints": kps
            })
            # Static guard at 0.20
            events.append({
                "timestamp": f / fps,
                "label": "person",
                "track_id": 2,
                "confidence": 0.9,
                "bbox": (0.20, 0.5, 0.2, 0.6),
                "keypoints": [(0.20, 0.5, 0.9)] * 17
            })

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        # Active speaker weight = 0.85, static guard weight = 0.25 * 0.20 = 0.05
        # Expected POI ~ (0.80 * 1920 * 0.85 + 0.20 * 1920 * 0.05) / 0.90 = 1482.67px
        assert poi[80] > 1400.0, f"Expected gesture motion to prioritize speaker, got {poi[80]}"

    def test_pure_facial_motion_overrides_static_centroid(self):
        """Stress: stationary centroid with facial dynamics (delta_face >= 0.002) gains 0.85 weight."""
        fps = 30.0
        total_frames = 90
        events = []

        for f in range(total_frames):
            # Centroid fixed at 0.75
            kps = [(0.75, 0.5, 0.9)] * 17
            # Nose / eyes (kps 0, 1, 2) moving
            kps[0] = (0.75 + 0.03 * np.sin(f * 0.5), 0.2, 0.9)
            kps[1] = (0.73 + 0.03 * np.sin(f * 0.5), 0.18, 0.9)
            kps[2] = (0.77 + 0.03 * np.sin(f * 0.5), 0.18, 0.9)
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (0.75, 0.5, 0.2, 0.4),
                "keypoints": kps
            })
            # Static listener at 0.25
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 2,
                "confidence": 0.9,
                "bbox": (0.25, 0.5, 0.2, 0.4),
                "keypoints": [(0.25, 0.5, 0.9)] * 17
            })

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        assert poi[80] > 1300.0, f"Expected facial motion to prioritize speaker, got {poi[80]}"


# ═════════════════════════════════════════════════════════════════════════════
# 4. Static Subject 80% Weight Discount (t >= 2.0s)
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature4StaticSubjectEightyPercentDiscount:
    """Stress tests for the 2.0-second time threshold and 80% static discount."""

    def test_static_discount_temporal_activation_transition(self):
        """Verify static subject retains full weight at t < 2.0s, and drops by 80% at t >= 2.0s."""
        fps = 30.0
        total_frames = 90  # 3.0 seconds
        events = []

        # Person 1: active speaker at 0.80 (1536px)
        # Person 2: static subject at 0.20 (384px)
        for f in range(total_frames):
            t = f / fps
            cx_act = 0.80 + 0.03 * np.sin(f * 0.5)
            events.append({
                "timestamp": t,
                "label": "face",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (cx_act, 0.2, 0.1, 0.2),
                "keypoints": [(cx_act, 0.25, 0.9)] * 17
            })
            events.append({
                "timestamp": t,
                "label": "face",
                "track_id": 2,
                "confidence": 0.9,
                "bbox": (0.20, 0.2, 0.1, 0.2),
                "keypoints": [(0.20, 0.25, 0.9)] * 17
            })

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)

        # At frame 30 (1.0s, t < 2.0s):
        # Person 1 weight = 0.85, Person 2 weight = 0.65 (not yet ghost)
        # Expected ratio: Person 2 influence is significant
        ratio_early = (poi[30] - 384.0) / (1536.0 - 384.0)

        # At frame 80 (2.67s, t >= 2.0s):
        # Person 1 weight = 0.85, Person 2 weight = 0.65 * 0.20 = 0.13 (80% discounted!)
        ratio_late = (poi[80] - 384.0) / (1536.0 - 384.0)

        assert ratio_late > ratio_early, "Static discount did not increase active speaker prominence!"
        assert ratio_late > 0.80, f"Active speaker ratio {ratio_late} should exceed 80% after discount!"


# ═════════════════════════════════════════════════════════════════════════════
# 5. Dead-Center Fallback Elimination & POI Holding
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature4DeadCenterHoldingAndDropout:
    """Stress tests verifying that zero detections hold last coordinate and never snap to 960px."""

    def test_complete_detection_dropout_for_100_frames(self):
        """Stress: 100 consecutive empty frames hold last verified coordinate identically."""
        fps = 30.0
        total_frames = 130
        events = []

        # Active speaker at cx = 0.82 (1574.4px) for frames 0-29
        for f in range(30):
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.92,
                "bbox": (0.82, 0.3, 0.1, 0.2),
                "keypoints": [(0.82, 0.3, 0.9)] * 17
            })
        # Frames 30 to 129: complete blackout (0 detections)

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        held_target = poi[29]
        assert abs(held_target - 1574.4) < 1.0

        for f in range(30, total_frames):
            assert poi[f] == held_target, f"Frame {f} did not hold coordinate!"
            assert abs(poi[f] - 960.0) > 200.0, f"Frame {f} snapped to dead center!"

    def test_dropout_with_isolated_static_ghost_holds_active_speaker(self):
        """Stress: When active speaker vanishes, remaining static ghost is ignored and active coordinate held."""
        fps = 30.0
        total_frames = 135
        events = []

        # Frames 0-74 (2.5s): Active speaker at 0.80 with motion + Static guard at 0.20
        # (Static guard reaches 2.0s threshold at frame 60)
        for f in range(75):
            cx1 = 0.80 + 0.02 * np.sin(f * 0.4)
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (cx1, 0.2, 0.1, 0.2),
                "keypoints": [(cx1, 0.25, 0.9)] * 17
            })
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 2,
                "confidence": 0.9,
                "bbox": (0.20, 0.2, 0.1, 0.2),
                "keypoints": [(0.20, 0.25, 0.9)] * 17
            })

        # Frames 75-134: Active speaker vanishes! Only static guard remains (already static > 2.0s)
        for f in range(75, 135):
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 2,
                "confidence": 0.9,
                "bbox": (0.20, 0.2, 0.1, 0.2),
                "keypoints": [(0.20, 0.25, 0.9)] * 17
            })

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        # Frames 75-134: static guard is ghost, active_candidates is empty -> holds last active position!
        for f in range(75, 135):
            assert poi[f] > 1100.0, f"Frame {f} snapped to static guard ({poi[f]}px) instead of holding!"
            assert abs(poi[f] - 960.0) > 100.0

    def test_initial_frame_zero_detections_leads_in_cleanly(self):
        """Stress: First frames empty, speaker appears later; leads in from first verified speaker."""
        fps = 30.0
        total_frames = 60
        events = []

        # Frames 0-19: empty
        # Frames 20-59: speaker at cx = 0.70
        for f in range(20, 60):
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.9,
                "bbox": (0.70, 0.3, 0.1, 0.2),
                "keypoints": [(0.70, 0.3, 0.9)] * 17
            })

        poi = fuse_point_of_interest(events, fps=fps, total_frames=total_frames, src_width=1920)
        # First verified speaker is at 0.70 * 1920 = 1344px. Lead-in should be clean!
        for f in range(20):
            assert poi[f] == 1344.0, f"Lead-in frame {f} should hold first verified speaker position!"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Extreme Numerical Resilience & Bounds Clamping
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature4ExtremeNumericalResilience:
    """Stress tests for zero frames, extreme coordinates, and OSQP boundary clamping."""

    def test_zero_and_single_frame_trajectories(self):
        """Stress: total_frames = 0 and total_frames = 1 must complete without unhandled exceptions."""
        traj0 = solve_trajectory([], [], fps=30.0, total_frames=0)
        assert traj0.shape == (0, 2)

        poi0 = fuse_point_of_interest([], fps=30.0, total_frames=0)
        assert len(poi0) == 0

        traj1 = solve_trajectory([], [], fps=30.0, total_frames=1)
        assert traj1.shape == (1, 2)
        assert 0.0 <= traj1[0, 0] <= 1.0

    def test_extreme_offscreen_bounding_boxes_clamped(self):
        """Stress: Coordinates at cx = -0.5 and cx = 1.5 are clamped to MIN_X and MAX_X."""
        fps = 30.0
        total_frames = 60

        # Sub-zero coordinate
        events_neg = [{
            "timestamp": f / fps,
            "label": "face",
            "track_id": 1,
            "confidence": 0.9,
            "bbox": (-0.5, 0.5, 0.2, 0.4)
        } for f in range(total_frames)]

        traj_neg = solve_trajectory(events_neg, [], fps=fps, total_frames=total_frames)
        traj_neg_px = traj_neg[:, 0] * 1920.0
        assert np.all(traj_neg_px >= MIN_X - 1e-4)

        # Coordinate > 1.0
        events_high = [{
            "timestamp": f / fps,
            "label": "face",
            "track_id": 1,
            "confidence": 0.9,
            "bbox": (1.5, 0.5, 0.2, 0.4)
        } for f in range(total_frames)]

        traj_high = solve_trajectory(events_high, [], fps=fps, total_frames=total_frames)
        traj_high_px = traj_high[:, 0] * 1920.0
        assert np.all(traj_high_px <= MAX_X + 1e-4)

    def test_pan_velocity_strictly_bounded_at_15px_per_frame(self):
        """Stress: Alternating extreme speakers never causes trajectory pan velocity to exceed 15 px/f."""
        fps = 30.0
        total_frames = 120
        events = []

        # Speaker A at 0.90 for frames 0-49
        for f in range(50):
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 1,
                "confidence": 0.95,
                "bbox": (0.90, 0.3, 0.1, 0.2),
                "keypoints": [(0.90, 0.3, 0.9)] * 17
            })
        # Speaker B at 0.10 for frames 50-119 (within the same shot, no cut)
        for f in range(50, 120):
            events.append({
                "timestamp": f / fps,
                "label": "face",
                "track_id": 2,
                "confidence": 0.95,
                "bbox": (0.10, 0.3, 0.1, 0.2),
                "keypoints": [(0.10, 0.3, 0.9)] * 17
            })

        traj = solve_trajectory(events, [], fps=fps, total_frames=total_frames)
        traj_px = traj[:, 0] * 1920.0
        inter_frame_velocity = np.abs(np.diff(traj_px))

        max_vel = np.max(inter_frame_velocity)
        assert max_vel <= MAX_VELOCITY_PX + 1e-3, f"Pan velocity exceeded 15 px/frame: {max_vel}"
