"""
Adversarial Stress Test Suite for Milestone 6.1:
Vision Tracking, Proximity Calculation, Synthetic ROI, and Topic Grounding.

Stress targets:
1. compute_wrist_object_proximity:
   - truncated keypoints (len 0 to 10)
   - empty / None keypoints
   - NaN / Inf coordinates and confidence values
   - single-element or 2-element tuples
   - extreme coordinate magnitudes (-1e6, 1e6)
2. associate_hand_object_proximity:
   - 0 detections
   - 1,000 simultaneous detections in a single frame
   - worst-case density (500 persons x 500 objects, 250k checks)
   - extreme bounding box sizes (0 area, negative, 1e6, Inf, NaN)
   - mixed frame_idx / timestamp keys
   - extreme proximity thresholds (0.0, Inf, negative)
3. generate_synthetic_demonstration_roi:
   - distorted human keypoints (negative, > 1.0, NaN coordinates)
   - inverted wrists (left wrist on right side, varying heights)
   - inverted torso (person upside down)
   - overlapping objects (50 overlapping, held_product presence, NaN bbox)
   - coordinate clamping assertions (cx in [0.15, 0.85], cy in [0.20, 0.80])
4. extract_transcript_beats & ground_beat_to_visuals:
   - empty transcripts
   - single-word and short sentence transcripts
   - out-of-order words (descending timestamps, backwards jumps)
   - zero visual events
   - temporal window fallbacks (primary window, +-3.0s, fallback to all)
   - all semantic categories with held, screen, speaker, and empty fallbacks
   - large-scale throughput stress (5,000 words + 1,000 spatial events)
"""

import time
import math
import pytest
from typing import List, Tuple

from shared.schemas import (
    WordTimestamp,
    SpeakerTurn,
    SpatialEvent,
    TranscriptBeat,
    SemanticTopicCategory,
    VisualGroundingTarget,
    GroundingEntity,
)
from wsl2_gpu.vision_spoke import (
    compute_wrist_object_proximity,
    associate_hand_object_proximity,
    generate_synthetic_demonstration_roi,
)
from wsl2_gpu.fusion_consumer import (
    extract_transcript_beats,
    ground_beat_to_visuals,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. COMPUTE_WRIST_OBJECT_PROXIMITY ADVERSARIAL STRESS TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestWristObjectProximityStress:
    """Stress tests and boundary conditions for compute_wrist_object_proximity."""

    def test_empty_and_none_keypoints(self):
        """Verifies compute_wrist_object_proximity returns (inf, -1) on empty or None keypoints."""
        bbox = (0.5, 0.5, 0.2, 0.2)
        dist, idx = compute_wrist_object_proximity([], bbox)
        assert dist == float("inf")
        assert idx == -1

        dist, idx = compute_wrist_object_proximity(None, bbox)  # type: ignore
        assert dist == float("inf")
        assert idx == -1

    def test_truncated_keypoints_lengths_0_to_10(self):
        """Verifies all keypoint lists with length < 11 return (inf, -1) without IndexError."""
        bbox = (0.5, 0.5, 0.2, 0.2)
        for length in range(11):
            kps = [(0.5, 0.5, 0.9)] * length
            dist, idx = compute_wrist_object_proximity(kps, bbox)
            assert dist == float("inf"), f"Failed for length {length}"
            assert idx == -1

    def test_exact_length_11_left_and_right_wrists(self):
        """Verifies length 11 keypoints correctly evaluates left wrist (9) and right wrist (10)."""
        bbox = (0.50, 0.50, 0.20, 0.20)
        kps = [(0.0, 0.0, 0.0)] * 11
        kps[9] = (0.52, 0.50, 0.9)   # left wrist dist = 0.02
        kps[10] = (0.60, 0.50, 0.9)  # right wrist dist = 0.10

        dist, idx = compute_wrist_object_proximity(kps, bbox)
        assert pytest.approx(dist, 1e-4) == 0.02
        assert idx == 9

        # Invert: right wrist closer
        kps[9] = (0.70, 0.50, 0.9)   # dist = 0.20
        kps[10] = (0.49, 0.50, 0.9)  # dist = 0.01
        dist, idx = compute_wrist_object_proximity(kps, bbox)
        assert pytest.approx(dist, 1e-4) == 0.01
        assert idx == 10

    def test_short_tuples_and_missing_confidence(self):
        """Verifies keypoints with length 2 tuples default confidence to 1.0, length < 2 are ignored."""
        bbox = (0.50, 0.50, 0.20, 0.20)
        kps = [(0.0, 0.0, 0.0)] * 17
        # 2-element tuple for left wrist
        kps[9] = (0.51, 0.50)  # type: ignore
        kps[10] = (0.0,)        # type: ignore (short tuple, should be ignored)

        dist, idx = compute_wrist_object_proximity(kps, bbox)
        assert pytest.approx(dist, 1e-4) == 0.01
        assert idx == 9

        # Both short tuples
        kps[9] = ()     # type: ignore
        kps[10] = (0.5,) # type: ignore
        dist, idx = compute_wrist_object_proximity(kps, bbox)
        assert dist == float("inf")
        assert idx == -1

    def test_nan_and_inf_coordinates(self):
        """Verifies NaN and Inf coordinates do not raise exceptions and return (inf, -1)."""
        bbox = (0.50, 0.50, 0.20, 0.20)

        # Both wrists NaN
        kps_all_nan = [(float("nan"), float("nan"), 0.9)] * 17
        dist, idx = compute_wrist_object_proximity(kps_all_nan, bbox)
        assert dist == float("inf")
        assert idx == -1

        # One wrist NaN, other wrist valid
        kps_partial_nan = [(0.0, 0.0, 0.0)] * 17
        kps_partial_nan[9] = (float("nan"), 0.5, 0.9)
        kps_partial_nan[10] = (0.53, 0.50, 0.9)
        dist, idx = compute_wrist_object_proximity(kps_partial_nan, bbox)
        assert pytest.approx(dist, 1e-4) == 0.03
        assert idx == 10

        # Wrists at Inf
        kps_inf = [(0.0, 0.0, 0.0)] * 17
        kps_inf[9] = (float("inf"), float("inf"), 0.9)
        kps_inf[10] = (-float("inf"), 0.5, 0.9)
        dist, idx = compute_wrist_object_proximity(kps_inf, bbox)
        assert dist == float("inf")
        assert idx == -1

        # Object bbox with NaN
        kps_valid = [(0.0, 0.0, 0.0)] * 17
        kps_valid[9] = (0.5, 0.5, 0.9)
        nan_bbox = (float("nan"), float("nan"), 0.2, 0.2)
        dist, idx = compute_wrist_object_proximity(kps_valid, nan_bbox)
        assert dist == float("inf")
        assert idx == -1

    def test_extreme_coordinate_magnitudes(self):
        """Verifies extreme numerical magnitudes do not overflow."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[9] = (1e6, -1e6, 1.0)
        kps[10] = (-1e6, 1e6, 1.0)
        bbox = (0.0, 0.0, 0.1, 0.1)

        dist, idx = compute_wrist_object_proximity(kps, bbox)
        expected = math.sqrt((1e6)**2 + (-1e6)**2)
        assert pytest.approx(dist, rel=1e-3) == expected
        assert idx in (9, 10)

    def test_zero_confidence_and_zero_coords_ignored(self):
        """Verifies (0.0, 0.0, 0.0) keypoints are considered undetected and ignored."""
        bbox = (0.0, 0.0, 0.2, 0.2)
        kps = [(0.0, 0.0, 0.0)] * 17
        dist, idx = compute_wrist_object_proximity(kps, bbox)
        assert dist == float("inf")
        assert idx == -1


# ═════════════════════════════════════════════════════════════════════════════
# 2. ASSOCIATE_HAND_OBJECT_PROXIMITY ADVERSARIAL STRESS TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestAssociateHandObjectProximityStress:
    """Stress tests for associate_hand_object_proximity under scale and boundary inputs."""

    def test_zero_detections(self):
        """Verifies associate_hand_object_proximity cleanly handles empty list."""
        assert associate_hand_object_proximity([]) == []

    def test_1000_simultaneous_detections(self):
        """Verifies scalability and throughput with 1,000 simultaneous detections in a single frame."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[9] = (0.50, 0.50, 0.9)
        kps[10] = (0.52, 0.50, 0.9)

        # 100 people and 900 objects
        events = []
        for i in range(100):
            events.append(SpatialEvent(
                timestamp=1.0, frame_idx=30, track_id=i + 1, label="person",
                bbox=(0.5, 0.5, 0.3, 0.6), confidence=0.9, keypoints=kps
            ))

        # 450 close objects, 450 distant objects
        for j in range(450):
            events.append(SpatialEvent(
                timestamp=1.0, frame_idx=30, track_id=200 + j, label="laptop",
                bbox=(0.51, 0.50, 0.15, 0.15), confidence=0.85
            ))
        for j in range(450):
            events.append(SpatialEvent(
                timestamp=1.0, frame_idx=30, track_id=700 + j, label="cup",
                bbox=(0.95, 0.95, 0.05, 0.05), confidence=0.75
            ))

        t0 = time.perf_counter()
        associated = associate_hand_object_proximity(events)
        elapsed = time.perf_counter() - t0

        assert len(associated) == 1000
        assert elapsed < 0.50, f"1000 detections took {elapsed:.4f}s (exceeded 0.5s budget)"

        held_count = sum(1 for e in associated if e.action_label == "held_product")
        assert held_count == 450
        for e in associated:
            if 200 <= e.track_id < 650:
                assert e.action_label == "held_product"
            elif e.track_id >= 700:
                assert e.action_label != "held_product"

    def test_worst_case_density_500x500(self):
        """Stress-tests worst-case density (500 distant persons x 500 distant objects = 250k checks)."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[9] = (0.1, 0.1, 0.9)
        kps[10] = (0.1, 0.1, 0.9)

        events = []
        for i in range(500):
            events.append(SpatialEvent(
                timestamp=2.0, frame_idx=60, track_id=i + 1, label="person",
                bbox=(0.1, 0.1, 0.1, 0.1), confidence=0.9, keypoints=kps
            ))
        for j in range(500):
            events.append(SpatialEvent(
                timestamp=2.0, frame_idx=60, track_id=600 + j, label="bottle",
                bbox=(0.9, 0.9, 0.02, 0.02), confidence=0.8
            ))

        t0 = time.perf_counter()
        associated = associate_hand_object_proximity(events)
        elapsed = time.perf_counter() - t0

        assert len(associated) == 1000
        assert elapsed < 1.0, f"Worst case density took {elapsed:.4f}s"
        assert sum(1 for e in associated if e.action_label == "held_product") == 0

    def test_extreme_bounding_box_sizes(self):
        """Verifies zero area, negative, massive, inf, and NaN bounding box sizes."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[9] = (0.5, 0.5, 0.9)
        kps[10] = (0.5, 0.5, 0.9)
        person = SpatialEvent(
            timestamp=1.0, frame_idx=1, track_id=1, label="person",
            bbox=(0.5, 0.5, 0.3, 0.6), confidence=0.9, keypoints=kps
        )

        test_cases = [
            # (bbox, expected_held)
            ((0.5, 0.5, 0.0, 0.0), True),               # zero area, dist=0 < 0.18 fallback
            ((0.5, 0.5, 1000.0, 1000.0), True),         # massive bbox, dist=0 < 1200
            ((-10.0, -10.0, -1.0, -1.0), False),        # negative coordinates far away
            ((0.5, 0.5, float("inf"), float("inf")), True),  # infinite bbox
            ((0.5, 0.5, float("nan"), float("nan")), False), # NaN bbox dimensions
        ]

        for bbox, expected in test_cases:
            obj = SpatialEvent(
                timestamp=1.0, frame_idx=1, track_id=10, label="laptop",
                bbox=bbox, confidence=0.9
            )
            res = associate_hand_object_proximity([person, obj])
            res_obj = next(e for e in res if e.track_id == 10)
            if expected:
                assert res_obj.action_label == "held_product", f"Failed for {bbox}"
            else:
                assert res_obj.action_label != "held_product", f"Failed for {bbox}"

    def test_mixed_frame_idx_and_timestamps(self):
        """Verifies grouping correctly falls back to timestamp when frame_idx == 0."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[9] = (0.5, 0.5, 0.9)

        # Two events at t=0.00, frame_idx=0
        p1 = SpatialEvent(timestamp=0.0, frame_idx=0, track_id=1, label="person",
                          bbox=(0.5, 0.5, 0.3, 0.6), keypoints=kps)
        o1 = SpatialEvent(timestamp=0.0, frame_idx=0, track_id=101, label="laptop",
                          bbox=(0.51, 0.50, 0.2, 0.2))

        # Two events at t=5.00, frame_idx=0 (different timestamp)
        p2 = SpatialEvent(timestamp=5.0, frame_idx=0, track_id=2, label="person",
                          bbox=(0.1, 0.1, 0.3, 0.6), keypoints=kps)
        o2 = SpatialEvent(timestamp=5.0, frame_idx=0, track_id=102, label="cup",
                          bbox=(0.9, 0.9, 0.1, 0.1))

        res = associate_hand_object_proximity([p1, o1, p2, o2])
        res_o1 = next(e for e in res if e.track_id == 101)
        res_o2 = next(e for e in res if e.track_id == 102)
        assert res_o1.action_label == "held_product"
        assert res_o2.action_label != "held_product"

    def test_custom_proximity_thresholds(self):
        """Verifies explicit proximity_threshold override (0.0, inf, negative)."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[9] = (0.50, 0.50, 0.9)

        p = SpatialEvent(timestamp=1.0, frame_idx=1, track_id=1, label="person",
                         bbox=(0.5, 0.5, 0.3, 0.6), keypoints=kps)
        obj = SpatialEvent(timestamp=1.0, frame_idx=1, track_id=101, label="laptop",
                           bbox=(0.55, 0.50, 0.2, 0.2)) # dist = 0.05

        # threshold = 0.0 -> dist(0.05) < 0.0 is False
        obj.action_label = None
        associate_hand_object_proximity([p, obj], proximity_threshold=0.0)
        assert obj.action_label is None

        # threshold = 0.10 -> dist(0.05) < 0.10 is True
        obj.action_label = None
        associate_hand_object_proximity([p, obj], proximity_threshold=0.10)
        assert obj.action_label == "held_product"


# ═════════════════════════════════════════════════════════════════════════════
# 3. GENERATE_SYNTHETIC_DEMONSTRATION_ROI ADVERSARIAL STRESS TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSyntheticDemonstrationRoiStress:
    """Stress tests for generate_synthetic_demonstration_roi."""

    def test_distorted_and_out_of_bounds_keypoints(self):
        """Verifies synthetic ROI coordinates are strictly clamped to [0.15, 0.85] and [0.20, 0.80]."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[5] = (0.4, 0.3, 0.9)
        kps[6] = (0.6, 0.3, 0.9)
        kps[11] = (0.4, 0.7, 0.9)
        kps[12] = (0.6, 0.7, 0.9)

        # Wrist far to the left (negative x)
        kps[9] = (-10.0, 0.5, 0.9)
        kps[10] = (-5.0, 0.5, 0.9)
        p = SpatialEvent(
            timestamp=1.0, frame_idx=1, track_id=1, label="person",
            bbox=(0.5, 0.5, 0.4, 0.7), confidence=0.9, keypoints=kps, action_label="pointing"
        )
        roi = generate_synthetic_demonstration_roi(p, [])
        assert roi is not None
        cx, cy, w, h = roi.bbox
        assert cx == 0.15  # clamped to min 0.15
        assert cy == 0.50
        assert w == 0.25 and h == 0.35

        # Wrist far to the bottom right (> 1.0)
        kps[9] = (50.0, 50.0, 0.9)
        kps[10] = (20.0, 20.0, 0.9)
        p.keypoints = kps
        roi = generate_synthetic_demonstration_roi(p, [])
        assert roi is not None
        cx, cy, _, _ = roi.bbox
        assert cx == 0.85  # clamped to max 0.85
        assert cy == 0.80  # clamped to max 0.80

    def test_inverted_wrists_and_height_priority(self):
        """Verifies higher wrist (smaller y) is prioritized when both wrists are active."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[5] = (0.4, 0.3, 0.9)
        kps[6] = (0.6, 0.3, 0.9)
        kps[11] = (0.4, 0.7, 0.9)
        kps[12] = (0.6, 0.7, 0.9)

        # Left wrist higher up (y=0.40) than right wrist (y=0.60)
        kps[9] = (0.45, 0.40, 0.9)
        kps[10] = (0.55, 0.60, 0.9)
        p = SpatialEvent(
            timestamp=1.0, frame_idx=1, track_id=1, label="person",
            bbox=(0.5, 0.5, 0.4, 0.7), confidence=0.9, keypoints=kps
        )
        roi = generate_synthetic_demonstration_roi(p, [])
        assert roi is not None
        assert roi.bbox[0] == 0.45
        assert roi.bbox[1] == 0.40

        # Invert: Right wrist higher up (y=0.35) than left wrist (y=0.55)
        kps[9] = (0.45, 0.55, 0.9)
        kps[10] = (0.55, 0.35, 0.9)
        p.keypoints = kps
        roi = generate_synthetic_demonstration_roi(p, [])
        assert roi is not None
        assert roi.bbox[0] == 0.55
        assert roi.bbox[1] == 0.35

    def test_inverted_torso_upside_down(self):
        """Verifies upside down person with pointing action generates bounded ROI, but returns None without gesture."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[5] = (0.4, 0.8, 0.9)   # shoulders at bottom
        kps[6] = (0.6, 0.8, 0.9)
        kps[11] = (0.4, 0.2, 0.9)  # hips at top
        kps[12] = (0.6, 0.2, 0.9)
        kps[9] = (0.45, 0.5, 0.9)
        kps[10] = (0.55, 0.5, 0.9)

        # Case 1: Pointing gesture present
        p_demo = SpatialEvent(
            timestamp=1.0, frame_idx=1, track_id=1, label="person",
            bbox=(0.5, 0.5, 0.4, 0.7), confidence=0.9, keypoints=kps, action_label="pointing"
        )
        roi = generate_synthetic_demonstration_roi(p_demo, [])
        assert roi is not None
        assert roi.action_label == "held_product"

        # Case 2: No gesture -> Active wrist check fails for upside-down person
        p_no_demo = SpatialEvent(
            timestamp=1.0, frame_idx=1, track_id=1, label="person",
            bbox=(0.5, 0.5, 0.4, 0.7), confidence=0.9, keypoints=kps, action_label=None
        )
        assert generate_synthetic_demonstration_roi(p_no_demo, []) is None

    def test_overlapping_objects_and_suppression(self):
        """Verifies synthetic ROI is suppressed if any existing object is within proximity or held."""
        kps = [(0.0, 0.0, 0.0)] * 17
        kps[5] = (0.4, 0.3, 0.9)
        kps[6] = (0.6, 0.3, 0.9)
        kps[9] = (0.45, 0.5, 0.9)
        kps[10] = (0.55, 0.5, 0.9)
        kps[11] = (0.4, 0.7, 0.9)
        kps[12] = (0.6, 0.7, 0.9)

        p = SpatialEvent(
            timestamp=1.0, frame_idx=1, track_id=1, label="person",
            bbox=(0.5, 0.5, 0.4, 0.7), confidence=0.9, keypoints=kps, action_label="holding"
        )

        # 50 overlapping distant objects -> Not near hands, ROI should NOT be suppressed
        distant_objs = [
            SpatialEvent(timestamp=1.0, frame_idx=1, track_id=100 + i, label="cup",
                         bbox=(0.9, 0.9, 0.05, 0.05), confidence=0.8)
            for i in range(50)
        ]
        roi = generate_synthetic_demonstration_roi(p, distant_objs)
        assert roi is not None

        # Add 1 object near hands -> Must suppress
        near_obj = SpatialEvent(
            timestamp=1.0, frame_idx=1, track_id=999, label="laptop",
            bbox=(0.46, 0.51, 0.15, 0.15), confidence=0.9
        )
        assert generate_synthetic_demonstration_roi(p, distant_objs + [near_obj]) is None

        # Object with held_product tag anywhere -> Must suppress
        held_obj = SpatialEvent(
            timestamp=1.0, frame_idx=1, track_id=998, label="box",
            bbox=(0.9, 0.9, 0.1, 0.1), confidence=0.9, action_label="held_product"
        )
        assert generate_synthetic_demonstration_roi(p, [held_obj]) is None


# ═════════════════════════════════════════════════════════════════════════════
# 4. TRANSCRIPT BEATS & VISUAL GROUNDING ADVERSARIAL STRESS TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestTopicGroundingAdversarialStress:
    """Stress tests for extract_transcript_beats and ground_beat_to_visuals."""

    def test_empty_transcripts(self):
        """Verifies empty words list returns empty beat list."""
        assert extract_transcript_beats([], []) == []

    def test_single_word_transcript(self):
        """Verifies single word transcript does not crash and produces exactly one beat."""
        words = [WordTimestamp(word="Incredible!", start=1.0, end=1.5, confidence=0.95)]
        beats = extract_transcript_beats(words, [])
        assert len(beats) == 1
        assert beats[0].beat_id == "beat_01"
        assert beats[0].start_time == 1.0
        assert beats[0].end_time == 1.5
        assert beats[0].duration == 0.5
        assert beats[0].visual_grounding is None

    def test_out_of_order_words_scrambled(self):
        """Verifies scrambled/non-monotonic timestamps degrade gracefully without exceptions."""
        words = [
            WordTimestamp(word="ending", start=50.0, end=50.5, confidence=0.9),
            WordTimestamp(word="middle", start=25.0, end=25.5, confidence=0.9),
            WordTimestamp(word="beginning", start=10.0, end=10.5, confidence=0.9),
            WordTimestamp(word="look", start=60.0, end=60.5, confidence=0.9),
            WordTimestamp(word="at", start=5.0, end=5.5, confidence=0.9),
            WordTimestamp(word="this.", start=0.0, end=0.5, confidence=0.9),
        ]
        beats = extract_transcript_beats(words, [])
        assert len(beats) >= 1
        for b in beats:
            assert b.duration >= 0.0
            assert b.beat_id.startswith("beat_")

    def test_zero_visual_events(self):
        """Verifies beats extracted with zero visual events have visual_grounding=None."""
        words = [
            WordTimestamp(word="The", start=65.0, end=65.3, confidence=0.9),
            WordTimestamp(word="benchmark", start=65.3, end=65.8, confidence=0.9),
            WordTimestamp(word="screen", start=65.8, end=66.2, confidence=0.9),
            WordTimestamp(word="displays.", start=66.2, end=66.8, confidence=0.9),
        ]
        beats = extract_transcript_beats(words, spatial_events=[])
        assert len(beats) == 1
        assert beats[0].visual_grounding is None
        assert beats[0].topic_category == SemanticTopicCategory.SCREEN_DEMONSTRATION

    def test_ground_beat_zero_events_returns_none(self):
        """Verifies ground_beat_to_visuals explicitly returns None when spatial_events is empty."""
        beat = TranscriptBeat(
            beat_id="b1", start_time=10.0, end_time=15.0,
            topic_category=SemanticTopicCategory.PRODUCT_SHOWCASE,
            summary="Demonstrating GPU", word_indices=(0, 5)
        )
        assert ground_beat_to_visuals(beat, []) is None

    def test_temporal_window_hierarchical_fallbacks(self):
        """Verifies hierarchical fallbacks: primary window -> +-3s window -> all events."""
        beat = TranscriptBeat(
            beat_id="b_test", start_time=10.0, end_time=12.0,
            topic_category=SemanticTopicCategory.PRODUCT_SHOWCASE,
            summary="Product", word_indices=(0, 2)
        )

        # 1. Event in primary window [9.0, 13.0]
        ev_primary = SpatialEvent(
            timestamp=11.0, frame_idx=330, track_id=101, label="laptop",
            bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, action_label="held_product"
        )
        target1 = ground_beat_to_visuals(beat, [ev_primary])
        assert target1 is not None and target1.track_id == 101

        # 2. Event in fallback window [8.0, 14.0] (center is 11.0, +-3s is [8.0, 14.0])
        ev_secondary = SpatialEvent(
            timestamp=8.5, frame_idx=255, track_id=102, label="laptop",
            bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, action_label="held_product"
        )
        target2 = ground_beat_to_visuals(beat, [ev_secondary])
        assert target2 is not None and target2.track_id == 102

        # 3. Event way outside window (t=100.0) -> Fallback to all events
        ev_far = SpatialEvent(
            timestamp=100.0, frame_idx=3000, track_id=103, label="laptop",
            bbox=(0.5, 0.5, 0.2, 0.2), confidence=0.9, action_label="held_product"
        )
        target3 = ground_beat_to_visuals(beat, [ev_far])
        assert target3 is not None and target3.track_id == 103

    def test_all_semantic_categories_grounding_targets(self):
        """Verifies all semantic categories map to appropriate target_type and anchor_mode."""
        spatial_events = [
            SpatialEvent(timestamp=10.0, frame_idx=300, track_id=1, label="person",
                         bbox=(0.5, 0.3, 0.3, 0.5), confidence=0.95),
            SpatialEvent(timestamp=10.0, frame_idx=300, track_id=101, label="laptop",
                         bbox=(0.5, 0.6, 0.2, 0.2), confidence=0.90, action_label="held_product"),
            SpatialEvent(timestamp=10.0, frame_idx=300, track_id=102, label="monitor",
                         bbox=(0.6, 0.5, 0.5, 0.5), confidence=0.92),
        ]

        # SPEAKER_HOOK
        b_hook = TranscriptBeat(beat_id="b1", start_time=9.5, end_time=10.5,
                                topic_category=SemanticTopicCategory.SPEAKER_HOOK,
                                summary="Hook", word_indices=(0, 2))
        tgt = ground_beat_to_visuals(b_hook, spatial_events)
        assert tgt is not None and tgt.target_type == "SPEAKER"
        assert tgt.anchor_mode == "TOP_ALIGNED"

        # PRODUCT_SHOWCASE
        b_prod = TranscriptBeat(beat_id="b2", start_time=9.5, end_time=10.5,
                                topic_category=SemanticTopicCategory.PRODUCT_SHOWCASE,
                                summary="Product", word_indices=(2, 4))
        tgt = ground_beat_to_visuals(b_prod, spatial_events)
        assert tgt is not None and tgt.target_type == "PRODUCT"
        assert tgt.anchor_mode == "HAND_REGION"
        assert tgt.secondary_bbox == (0.5, 0.3, 0.3, 0.5)  # Speaker secondary

        # HANDS_ON_ACTION
        b_action = TranscriptBeat(beat_id="b3", start_time=9.5, end_time=10.5,
                                  topic_category=SemanticTopicCategory.HANDS_ON_ACTION,
                                  summary="Action", word_indices=(4, 6))
        tgt = ground_beat_to_visuals(b_action, spatial_events)
        assert tgt is not None and tgt.target_type == "HELD_OBJECT"

        # SCREEN_DEMONSTRATION
        b_screen = TranscriptBeat(beat_id="b4", start_time=9.5, end_time=10.5,
                                  topic_category=SemanticTopicCategory.SCREEN_DEMONSTRATION,
                                  summary="Screen", word_indices=(6, 8))
        tgt = ground_beat_to_visuals(b_screen, spatial_events)
        assert tgt is not None and tgt.target_type == "SCREEN"
        assert tgt.primary_bbox in ((0.5, 0.6, 0.2, 0.2), (0.6, 0.5, 0.5, 0.5))

        # DIALOGUE_CONTEXT
        b_dialogue = TranscriptBeat(beat_id="b5", start_time=9.5, end_time=10.5,
                                    topic_category=SemanticTopicCategory.DIALOGUE_CONTEXT,
                                    summary="Dialogue", word_indices=(8, 10))
        tgt = ground_beat_to_visuals(b_dialogue, spatial_events)
        assert tgt is not None and tgt.target_type == "SPEAKER"

    def test_large_scale_throughput_and_stress(self):
        """Throughput stress: 5,000 words + 1,000 spatial events executes in < 0.25 seconds."""
        words = [
            WordTimestamp(word=f"word_{i}", start=i * 0.2, end=i * 0.2 + 0.18, confidence=0.95)
            for i in range(5000)
        ]
        spatial = [
            SpatialEvent(timestamp=i * 0.5, frame_idx=i * 15, track_id=1, label="person",
                         bbox=(0.5, 0.4, 0.3, 0.6), confidence=0.95)
            for i in range(1000)
        ]

        t0 = time.perf_counter()
        beats = extract_transcript_beats(words, spatial)
        elapsed = time.perf_counter() - t0

        assert len(beats) > 250
        assert elapsed < 0.25, f"Large scale beat extraction took {elapsed:.4f}s"
        assert all(b.visual_grounding is not None for b in beats)
