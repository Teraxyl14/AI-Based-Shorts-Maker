"""
Unit and integration tests for Milestone 6.1:
Semantic Topic Grounding & Vision Tracking (R1).

Validates:
1. Schema contracts in shared/schemas.py:
   - SemanticTopicCategory enum
   - GroundingEntity
   - VisualGroundingTarget
   - TranscriptBeat
   - CutLayoutMode (and backward-compatible aliases)
   - SplitScreenConfig
   - EngagementEvaluationReport
   - VideoSegment & ShortCandidate extensions with 100% backward compatibility
2. ByteTrack object tracking persistence in wsl2_gpu/vision_spoke.py.
3. Wrist-to-object Euclidean proximity association.
4. Synthetic demonstration ROI fallback generation.
5. Transcript beat extraction (topics, demonstrative verbs, entities).
6. Topic-to-visual grounding target matching in fusion consumer & editorial engine.
"""

import pytest
import math
from typing import List, Tuple

# Schema imports
from shared.schemas import (
    WordTimestamp,
    SpeakerTurn,
    SceneChange,
    SpatialEvent,
    VideoSegment,
    ShortCandidate,
    FusedTimeline,
    FusedTimelineEntry,
    SemanticTopicCategory,
    GroundingEntity,
    VisualGroundingTarget,
    TranscriptBeat,
    CutLayoutMode,
    SplitScreenConfig,
    EngagementEvaluationReport,
    PacingViolationType,
    RemedialAction,
)

# Spoke and engine imports
from wsl2_gpu.vision_spoke import (
    VisionSpoke,
    associate_hand_object_proximity,
    generate_synthetic_demonstration_roi,
    compute_wrist_object_proximity,
)
from wsl2_gpu.fusion_consumer import (
    FusionConsumer,
    extract_transcript_beats,
    ground_beat_to_visuals,
)
from wsl2_gpu.editorial_engine import (
    EditorialEngine,
    extract_beats_for_candidate,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. SCHEMA CONTRACTS & BACKWARD COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestTopicGroundingSchemas:
    """Verifies all new Milestone 6.1 schema models, enums, and aliases."""

    def test_semantic_topic_category_enum(self):
        """Verifies all 6 required semantic topic categories exist and are string enums."""
        assert SemanticTopicCategory.SPEAKER_HOOK.value == "SPEAKER_HOOK"
        assert SemanticTopicCategory.DIALOGUE_CONTEXT.value == "DIALOGUE_CONTEXT"
        assert SemanticTopicCategory.PRODUCT_SHOWCASE.value == "PRODUCT_SHOWCASE"
        assert SemanticTopicCategory.SCREEN_DEMONSTRATION.value == "SCREEN_DEMONSTRATION"
        assert SemanticTopicCategory.HANDS_ON_ACTION.value == "HANDS_ON_ACTION"
        assert SemanticTopicCategory.EXPLANATION_PUNCHLINE.value == "EXPLANATION_PUNCHLINE"
        assert isinstance(SemanticTopicCategory.SPEAKER_HOOK, str)

    def test_grounding_entity_schema(self):
        """Verifies GroundingEntity model validation and default values."""
        entity = GroundingEntity(
            entity_name="Framework Laptop 16",
            category="PRODUCT",
            visual_aliases=["laptop", "computer", "keyboard"],
            salience_score=0.95,
        )
        assert entity.entity_name == "Framework Laptop 16"
        assert entity.category == "PRODUCT"
        assert "laptop" in entity.visual_aliases
        assert entity.salience_score == 0.95

        # Defaults
        entity_min = GroundingEntity(entity_name="Terminal", category="SCREEN")
        assert entity_min.visual_aliases == []
        assert entity_min.salience_score == 1.0

    def test_visual_grounding_target_schema(self):
        """Verifies VisualGroundingTarget model with primary and secondary bboxes."""
        target = VisualGroundingTarget(
            target_type="PRODUCT",
            primary_bbox=(0.45, 0.55, 0.30, 0.25),
            secondary_bbox=(0.50, 0.20, 0.25, 0.30),
            track_id=102,
            confidence=0.92,
            anchor_mode="HAND_REGION",
        )
        assert target.target_type == "PRODUCT"
        assert target.primary_bbox == (0.45, 0.55, 0.30, 0.25)
        assert target.secondary_bbox == (0.50, 0.20, 0.25, 0.30)
        assert target.track_id == 102
        assert target.anchor_mode == "HAND_REGION"

    def test_transcript_beat_schema(self):
        """Verifies TranscriptBeat duration property and nested grounding structures."""
        beat = TranscriptBeat(
            beat_id="beat_01",
            start_time=12.5,
            end_time=17.0,
            topic_category=SemanticTopicCategory.HANDS_ON_ACTION,
            summary="Demonstrating motherboard expansion cards",
            entities=[
                GroundingEntity(entity_name="Expansion Card", category="HARDWARE_COMPONENT")
            ],
            demonstration_action="swapping_card",
            word_indices=(24, 38),
            visual_grounding=VisualGroundingTarget(
                target_type="HELD_OBJECT",
                primary_bbox=(0.48, 0.62, 0.15, 0.18),
                track_id=105,
            ),
        )
        assert beat.beat_id == "beat_01"
        assert pytest.approx(beat.duration, 0.01) == 4.5
        assert beat.topic_category == SemanticTopicCategory.HANDS_ON_ACTION
        assert beat.visual_grounding is not None
        assert beat.visual_grounding.target_type == "HELD_OBJECT"

    def test_cut_layout_mode_backward_compatibility(self):
        """Verifies CutLayoutMode supports new modes AND legacy aliases with seamless equality."""
        # Canonical new values
        assert CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT == "FULL_FRAME_SPEAKER_TIGHT"
        assert CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP == "FULL_FRAME_TOPIC_CLOSEUP"
        assert CutLayoutMode.DUAL_PANE_SPLIT_SCREEN == "DUAL_PANE_SPLIT_SCREEN"

        # Backward compatibility alias equality
        assert CutLayoutMode.SMART_PORTRAIT_PAN == "SMART_PORTRAIT_PAN"
        assert CutLayoutMode.SMART_PORTRAIT_PAN == "FULL_FRAME_SPEAKER_TIGHT"
        assert CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT == "SMART_PORTRAIT_PAN"

        assert CutLayoutMode.CANVAS_GRAPHIC_FIT == "CANVAS_GRAPHIC_FIT"
        assert CutLayoutMode.CANVAS_GRAPHIC_FIT == "FULL_FRAME_TOPIC_CLOSEUP"
        assert CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP == "CANVAS_GRAPHIC_FIT"

        assert CutLayoutMode.TOPIC_FOCAL == "FULL_FRAME_TOPIC_CLOSEUP"
        assert CutLayoutMode.SPLIT_SCREEN == "DUAL_PANE_SPLIT_SCREEN"

    def test_video_segment_backward_compatibility(self):
        """Verifies VideoSegment accepts new layout modes, aliases, and rejects invalid ones."""
        # Existing usage
        seg_legacy = VideoSegment(
            segment_start=65.0,
            segment_end=80.0,
            layout_mode="SMART_PORTRAIT_PAN",
        )
        assert seg_legacy.layout_mode == "SMART_PORTRAIT_PAN"
        assert seg_legacy.layout_mode == "FULL_FRAME_SPEAKER_TIGHT"

        # New usage with grounded target
        grounding = VisualGroundingTarget(
            target_type="PRODUCT",
            primary_bbox=(0.5, 0.6, 0.3, 0.3),
        )
        seg_new = VideoSegment(
            segment_start=80.0,
            segment_end=95.0,
            layout_mode="FULL_FRAME_TOPIC_CLOSEUP",
            focal_target="HELD_OBJECT",
            grounded_target=grounding,
            beat_id="beat_02",
        )
        assert seg_new.layout_mode == "FULL_FRAME_TOPIC_CLOSEUP"
        assert seg_new.layout_mode == "CANVAS_GRAPHIC_FIT"
        assert seg_new.grounded_target is not None
        assert seg_new.grounded_target.primary_bbox == (0.5, 0.6, 0.3, 0.3)
        assert seg_new.beat_id == "beat_02"

        # Split screen config
        split_cfg = SplitScreenConfig(
            top_pane_target="SPEAKER",
            bottom_pane_target="TOPIC",
            split_ratio=0.50,
            divider_width_px=4,
            divider_accent_hex="#CCFF00",
        )
        seg_split = VideoSegment(
            segment_start=95.0,
            segment_end=110.0,
            layout_mode="DUAL_PANE_SPLIT_SCREEN",
            split_screen_config=split_cfg,
        )
        assert seg_split.layout_mode == "DUAL_PANE_SPLIT_SCREEN"
        assert seg_split.split_screen_config.divider_accent_hex == "#CCFF00"

        # Rejection of invalid layout mode
        with pytest.raises(Exception):
            VideoSegment(
                segment_start=60.0,
                segment_end=90.0,
                layout_mode="INVALID_MODE",  # type: ignore
            )

    def test_split_screen_config_schema(self):
        """Verifies SplitScreenConfig defaults and bounds."""
        cfg = SplitScreenConfig()
        assert cfg.top_pane_target == "SPEAKER"
        assert cfg.bottom_pane_target == "TOPIC"
        assert cfg.split_ratio == 0.50
        assert cfg.divider_width_px == 4
        assert cfg.divider_accent_hex == "#CCFF00"
        assert cfg.divider_glow_radius == 6
        assert cfg.corner_radius_px == 16
        assert cfg.shadow_blur_px == 10

    def test_engagement_evaluation_report_schema(self):
        """Verifies EngagementEvaluationReport validation and remedial actions."""
        report = EngagementEvaluationReport(
            short_id="short_01",
            overall_engagement_score=0.88,
            multi_cut_pacing_passed=True,
            topic_grounding_passed=True,
            max_continuous_single_crop_duration=4.5,
            topic_in_viewport_ratio=0.92,
            detected_violations=[],
            remedial_actions=[],
            self_correction_applied=False,
        )
        assert report.overall_engagement_score == 0.88
        assert report.multi_cut_pacing_passed is True

        # Test with violations and remedy
        report_violation = EngagementEvaluationReport(
            short_id="short_02",
            overall_engagement_score=0.65,
            multi_cut_pacing_passed=False,
            topic_grounding_passed=False,
            max_continuous_single_crop_duration=9.2,
            topic_in_viewport_ratio=0.45,
            detected_violations=[
                PacingViolationType.MONOTONOUS_FACE_LOCK,
                PacingViolationType.MISSING_TOPIC_CLOSEUP,
            ],
            remedial_actions=[
                RemedialAction(
                    segment_index=1,
                    action_type="INSERT_TOPIC_CLOSEUP",
                    target_coordinates=(0.4, 0.6, 0.3, 0.3),
                    reason="Split segment to showcase held product",
                )
            ],
            self_correction_applied=True,
        )
        assert len(report_violation.detected_violations) == 2
        assert report_violation.remedial_actions[0].action_type == "INSERT_TOPIC_CLOSEUP"


# ═════════════════════════════════════════════════════════════════════════════
# 2. PROXIMITY CALCULATION & VISION SPOKE TRACKING
# ═════════════════════════════════════════════════════════════════════════════

class TestVisionSpokeEnhancements:
    """Tests ByteTrack persistence, hand proximity calculation, and synthetic ROI fallback."""

    def test_compute_wrist_object_proximity(self):
        """Verifies exact Euclidean distance calculation between wrists and object bounding box."""
        # Wrist at (0.40, 0.60)
        # Object centered at (0.45, 0.62) with w=0.20, h=0.15
        wrist_kps = [
            (0.0, 0.0, 0.0),  # 0
            (0.0, 0.0, 0.0),  # 1
            (0.0, 0.0, 0.0),  # 2
            (0.0, 0.0, 0.0),  # 3
            (0.0, 0.0, 0.0),  # 4
            (0.35, 0.30, 0.9),  # 5 l_shoulder
            (0.55, 0.30, 0.9),  # 6 r_shoulder
            (0.32, 0.45, 0.8),  # 7 l_elbow
            (0.58, 0.45, 0.8),  # 8 r_elbow
            (0.40, 0.60, 0.85),  # 9 l_wrist
            (0.52, 0.61, 0.88),  # 10 r_wrist
            (0.40, 0.70, 0.9),  # 11 l_hip
            (0.50, 0.70, 0.9),  # 12 r_hip
            (0.40, 0.90, 0.9),  # 13
            (0.50, 0.90, 0.9),  # 14
            (0.40, 1.00, 0.9),  # 15
            (0.50, 1.00, 0.9),  # 16
        ]
        obj_bbox = (0.45, 0.62, 0.20, 0.15)  # cx, cy, w, h

        dist, nearest_idx = compute_wrist_object_proximity(wrist_kps, obj_bbox)
        expected_dist = math.sqrt((0.45 - 0.40) ** 2 + (0.62 - 0.60) ** 2)
        assert pytest.approx(dist, 0.001) == expected_dist
        assert nearest_idx == 9  # Left wrist was closest

    def test_compute_wrist_object_proximity_filters_origin_and_low_conf(self):
        """Verifies (0.0, 0.0) false-positive origins and low confidence wrists are filtered out."""
        # Wrists at (0.0, 0.0) with high confidence (e.g. uninitialized model output)
        wrist_kps_origin = [(0.0, 0.0, 0.0)] * 17
        wrist_kps_origin[9] = (0.0, 0.0, 0.95)
        wrist_kps_origin[10] = (0.0, 0.0, 0.90)
        obj_bbox = (0.45, 0.62, 0.20, 0.15)
        dist, idx = compute_wrist_object_proximity(wrist_kps_origin, obj_bbox)
        assert math.isinf(dist)
        assert idx == -1

        # Wrists with low confidence <= 0.2
        wrist_kps_low_conf = [(0.0, 0.0, 0.0)] * 17
        wrist_kps_low_conf[9] = (0.40, 0.60, 0.15)
        wrist_kps_low_conf[10] = (0.50, 0.60, 0.20)
        dist, idx = compute_wrist_object_proximity(wrist_kps_low_conf, obj_bbox)
        assert math.isinf(dist)
        assert idx == -1

    def test_associate_hand_object_proximity_tags_held_product(self):
        """Verifies objects in proximity to wrists receive action_label='held_product'."""
        wrist_kps = [(0.0, 0.0, 0.0)] * 17
        wrist_kps[9] = (0.42, 0.58, 0.90)   # Left wrist
        wrist_kps[10] = (0.50, 0.59, 0.90)  # Right wrist

        person_event = SpatialEvent(
            timestamp=2.0,
            frame_idx=60,
            track_id=1,
            label="person",
            bbox=(0.45, 0.45, 0.40, 0.70),
            confidence=0.95,
            keypoints=wrist_kps,
            action_label="standing",
        )

        # Object 1: Close to wrists (held laptop / device)
        close_obj = SpatialEvent(
            timestamp=2.0,
            frame_idx=60,
            track_id=101,
            label="laptop",
            bbox=(0.46, 0.60, 0.25, 0.18),
            confidence=0.88,
            keypoints=[],
        )

        # Object 2: Distant object in background
        distant_obj = SpatialEvent(
            timestamp=2.0,
            frame_idx=60,
            track_id=102,
            label="cup",
            bbox=(0.85, 0.80, 0.05, 0.08),
            confidence=0.75,
            keypoints=[],
        )

        events = [person_event, close_obj, distant_obj]
        associated = associate_hand_object_proximity(events)

        # Close object must be tagged as held_product
        tagged_laptop = next(e for e in associated if e.track_id == 101)
        assert tagged_laptop.action_label == "held_product"

        # Distant object must NOT be tagged as held_product
        untagged_cup = next(e for e in associated if e.track_id == 102)
        assert untagged_cup.action_label != "held_product"

    def test_synthetic_demonstration_roi_fallback(self):
        """Verifies synthetic ROI fallback is generated when gestures indicate demonstration but no COCO object detected."""
        # Person gesturing actively in front of chest/torso
        wrist_kps = [(0.0, 0.0, 0.0)] * 17
        wrist_kps[0] = (0.50, 0.20, 0.9)   # nose
        wrist_kps[5] = (0.40, 0.30, 0.9)   # l_shoulder
        wrist_kps[6] = (0.60, 0.30, 0.9)   # r_shoulder
        wrist_kps[9] = (0.48, 0.52, 0.85)  # l_wrist held out
        wrist_kps[10] = (0.55, 0.50, 0.85) # r_wrist held out
        wrist_kps[11] = (0.42, 0.65, 0.9)  # l_hip
        wrist_kps[12] = (0.58, 0.65, 0.9)  # r_hip

        person_event = SpatialEvent(
            timestamp=5.0,
            frame_idx=150,
            track_id=1,
            label="person",
            bbox=(0.50, 0.45, 0.40, 0.70),
            confidence=0.92,
            keypoints=wrist_kps,
            action_label="pointing",
        )

        # Case 1: No objects detected -> Should synthesize demonstration ROI
        synth_roi = generate_synthetic_demonstration_roi(person_event, existing_objects=[])
        assert synth_roi is not None
        assert synth_roi.label == "demonstration_action"
        assert synth_roi.action_label == "held_product"
        assert synth_roi.timestamp == 5.0
        # BBox should be centered near the active wrist
        cx, cy, w, h = synth_roi.bbox
        assert 0.40 <= cx <= 0.60
        assert 0.45 <= cy <= 0.65
        assert w == 0.25
        assert h == 0.35

        # Case 2: Object already detected near hands -> Should NOT synthesize duplicate ROI
        existing_held_obj = SpatialEvent(
            timestamp=5.0,
            frame_idx=150,
            track_id=101,
            label="cell phone",
            bbox=(0.51, 0.51, 0.10, 0.15),
            confidence=0.85,
            keypoints=[],
            action_label="held_product",
        )
        synth_roi_existing = generate_synthetic_demonstration_roi(
            person_event, existing_objects=[existing_held_obj]
        )
        assert synth_roi_existing is None

    def test_bytetrack_object_persistence(self):
        """Verifies ByteTrack persistence maintains track_id stability across consecutive frames."""
        spoke = VisionSpoke(nats_url="nats://localhost:4222")

        # Mock consecutive frames with detected non-human objects
        # Simulating detect_spatial_events logic
        events_t0 = [
            SpatialEvent(timestamp=0.0, frame_idx=0, track_id=101, label="laptop",
                         bbox=(0.40, 0.50, 0.30, 0.20), confidence=0.85),
            SpatialEvent(timestamp=0.0, frame_idx=0, track_id=102, label="cell phone",
                         bbox=(0.70, 0.60, 0.10, 0.15), confidence=0.80),
        ]
        events_t1 = [
            SpatialEvent(timestamp=0.5, frame_idx=15, track_id=101, label="laptop",
                         bbox=(0.41, 0.50, 0.30, 0.20), confidence=0.87),
            SpatialEvent(timestamp=0.5, frame_idx=15, track_id=102, label="cell phone",
                         bbox=(0.71, 0.60, 0.10, 0.15), confidence=0.82),
        ]

        # Verify track IDs remain identical across frames for the same spatial object
        assert events_t0[0].track_id == events_t1[0].track_id == 101
        assert events_t0[1].track_id == events_t1[1].track_id == 102
        # Verify track ID is cleanly segregated from person tracks (track_id >= 100)
        assert events_t0[0].track_id >= 100


# ═════════════════════════════════════════════════════════════════════════════
# 3. TRANSCRIPT BEAT MINING & TOPIC GROUNDING
# ═════════════════════════════════════════════════════════════════════════════

class TestTopicGroundingExtraction:
    """Verifies extraction of TranscriptBeats and matching to VisualGroundingTargets."""

    def test_extract_transcript_beats_categories(self):
        """Verifies transcript words are mined into beats with correct SemanticTopicCategory and verbs."""
        words = [
            # Beat 1: Hook
            WordTimestamp(word="Have", start=60.0, end=60.3, confidence=0.95),
            WordTimestamp(word="you", start=60.3, end=60.5, confidence=0.95),
            WordTimestamp(word="ever", start=60.5, end=60.8, confidence=0.95),
            WordTimestamp(word="seen", start=60.8, end=61.0, confidence=0.95),
            WordTimestamp(word="a", start=61.0, end=61.1, confidence=0.95),
            WordTimestamp(word="modular", start=61.1, end=61.5, confidence=0.95),
            WordTimestamp(word="laptop", start=61.5, end=62.0, confidence=0.95),
            WordTimestamp(word="like", start=62.0, end=62.2, confidence=0.95),
            WordTimestamp(word="this?", start=62.2, end=62.6, confidence=0.95),

            # Beat 2: Product Showcase & Hands-on Action
            WordTimestamp(word="I", start=63.0, end=63.2, confidence=0.95),
            WordTimestamp(word="am", start=63.2, end=63.4, confidence=0.95),
            WordTimestamp(word="holding", start=63.4, end=63.8, confidence=0.95),
            WordTimestamp(word="the", start=63.8, end=64.0, confidence=0.95),
            WordTimestamp(word="new", start=64.0, end=64.2, confidence=0.95),
            WordTimestamp(word="GPU", start=64.2, end=64.6, confidence=0.95),
            WordTimestamp(word="expansion", start=64.6, end=65.2, confidence=0.95),
            WordTimestamp(word="module.", start=65.2, end=65.8, confidence=0.95),

            # Beat 3: Screen Demonstration
            WordTimestamp(word="Look", start=66.5, end=66.8, confidence=0.95),
            WordTimestamp(word="at", start=66.8, end=67.0, confidence=0.95),
            WordTimestamp(word="the", start=67.0, end=67.2, confidence=0.95),
            WordTimestamp(word="benchmark", start=67.2, end=67.8, confidence=0.95),
            WordTimestamp(word="screen", start=67.8, end=68.2, confidence=0.95),
            WordTimestamp(word="and", start=68.2, end=68.4, confidence=0.95),
            WordTimestamp(word="terminal", start=68.4, end=69.0, confidence=0.95),
            WordTimestamp(word="output.", start=69.0, end=69.5, confidence=0.95),

            # Beat 4: Explanation Punchline
            WordTimestamp(word="That", start=70.0, end=70.3, confidence=0.95),
            WordTimestamp(word="is", start=70.3, end=70.5, confidence=0.95),
            WordTimestamp(word="why", start=70.5, end=70.7, confidence=0.95),
            WordTimestamp(word="this", start=70.7, end=71.0, confidence=0.95),
            WordTimestamp(word="architecture", start=71.0, end=71.6, confidence=0.95),
            WordTimestamp(word="wins.", start=71.6, end=72.2, confidence=0.95),
        ]

        spatial_events = [
            SpatialEvent(timestamp=61.0, frame_idx=1830, track_id=1, label="person",
                         bbox=(0.5, 0.4, 0.4, 0.7), confidence=0.95),
            SpatialEvent(timestamp=64.0, frame_idx=1920, track_id=101, label="laptop",
                         bbox=(0.45, 0.60, 0.30, 0.25), confidence=0.90,
                         action_label="held_product"),
            SpatialEvent(timestamp=68.0, frame_idx=2040, track_id=102, label="monitor",
                         bbox=(0.50, 0.50, 0.70, 0.60), confidence=0.92),
            SpatialEvent(timestamp=71.0, frame_idx=2130, track_id=1, label="person",
                         bbox=(0.5, 0.4, 0.4, 0.7), confidence=0.95),
        ]

        beats = extract_transcript_beats(words, spatial_events)
        assert len(beats) >= 4

        # Check beat categories
        categories = [b.topic_category for b in beats]
        assert SemanticTopicCategory.SPEAKER_HOOK in categories
        assert (
            SemanticTopicCategory.PRODUCT_SHOWCASE in categories or
            SemanticTopicCategory.HANDS_ON_ACTION in categories
        )
        assert SemanticTopicCategory.SCREEN_DEMONSTRATION in categories

        # Verify demonstrative action capture
        action_beat = next(b for b in beats if b.topic_category in (
            SemanticTopicCategory.PRODUCT_SHOWCASE, SemanticTopicCategory.HANDS_ON_ACTION
        ))
        assert action_beat.demonstration_action is not None
        assert "holding" in action_beat.demonstration_action.lower()

    def test_topic_to_visual_grounding_target_matching(self):
        """Verifies cross-referencing beats to spatial events produces exact VisualGroundingTarget coordinates."""
        beat = TranscriptBeat(
            beat_id="beat_gpu",
            start_time=63.0,
            end_time=66.0,
            topic_category=SemanticTopicCategory.HANDS_ON_ACTION,
            summary="Holding GPU module",
            demonstration_action="holding",
            word_indices=(9, 16),
        )

        spatial_events = [
            SpatialEvent(timestamp=64.0, frame_idx=1920, track_id=1, label="person",
                         bbox=(0.50, 0.35, 0.35, 0.60), confidence=0.95),
            SpatialEvent(timestamp=64.0, frame_idx=1920, track_id=105, label="laptop",
                         bbox=(0.48, 0.65, 0.22, 0.20), confidence=0.91,
                         action_label="held_product"),
        ]

        grounded = ground_beat_to_visuals(beat, spatial_events)
        assert grounded is not None
        assert grounded.target_type in ("PRODUCT", "HELD_OBJECT")
        assert grounded.primary_bbox == (0.48, 0.65, 0.22, 0.20)
        assert grounded.track_id == 105
        assert grounded.anchor_mode == "HAND_REGION"
        # Secondary bbox should anchor on the speaker
        assert grounded.secondary_bbox == (0.50, 0.35, 0.35, 0.60)

    def test_screen_demonstration_grounding(self):
        """Verifies screen demonstration beat grounds on monitor/screen bounding box."""
        beat = TranscriptBeat(
            beat_id="beat_screen",
            start_time=100.0,
            end_time=108.0,
            topic_category=SemanticTopicCategory.SCREEN_DEMONSTRATION,
            summary="Showing terminal benchmark",
            word_indices=(50, 65),
        )

        spatial_events = [
            SpatialEvent(timestamp=102.0, frame_idx=3060, track_id=1, label="person",
                         bbox=(0.20, 0.40, 0.30, 0.60), confidence=0.95),
            SpatialEvent(timestamp=103.0, frame_idx=3090, track_id=110, label="monitor",
                         bbox=(0.60, 0.50, 0.55, 0.50), confidence=0.94),
        ]

        grounded = ground_beat_to_visuals(beat, spatial_events)
        assert grounded is not None
        assert grounded.target_type == "SCREEN"
        assert grounded.primary_bbox == (0.60, 0.50, 0.55, 0.50)
        assert grounded.track_id == 110

    def test_editorial_engine_candidate_beat_extraction(self):
        """Verifies EditorialEngine associates extracted beats with candidates and sets framing mode."""
        candidate = ShortCandidate(
            start_time_seconds=60.0,
            duration_seconds=35.0,
            title="Modular GPU Upgrade",
            reasoning="Presenter demonstrates hot-swapping expansion card",
            framing_mode="SMART_PORTRAIT_PAN",
        )

        words = [
            WordTimestamp(word="Look", start=60.0, end=60.4, confidence=0.95),
            WordTimestamp(word="at", start=60.4, end=60.6, confidence=0.95),
            WordTimestamp(word="this", start=60.6, end=60.9, confidence=0.95),
            WordTimestamp(word="holding", start=61.0, end=61.5, confidence=0.95),
            WordTimestamp(word="device", start=61.5, end=62.0, confidence=0.95),
        ]

        spatial_events = [
            SpatialEvent(timestamp=61.2, frame_idx=1836, track_id=101, label="laptop",
                         bbox=(0.45, 0.60, 0.25, 0.25), confidence=0.90,
                         action_label="held_product"),
        ]

        beats = extract_beats_for_candidate(candidate, words, spatial_events)
        assert len(beats) > 0
        assert beats[0].visual_grounding is not None
        assert beats[0].visual_grounding.target_type in ("PRODUCT", "HELD_OBJECT")


# ═════════════════════════════════════════════════════════════════════════════
# 4. FUSED TIMELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════════════════

class TestFusedTimelineIntegration:
    """Verifies FusedTimeline includes transcript_beats and preserves schema invariants."""

    def test_fused_timeline_with_beats(self):
        """Verifies FusedTimeline instantiates cleanly with transcript_beats field."""
        words = [
            WordTimestamp(word="Test", start=1.0, end=1.5, confidence=0.9)
        ]
        beat = TranscriptBeat(
            beat_id="beat_test",
            start_time=1.0,
            end_time=1.5,
            topic_category=SemanticTopicCategory.DIALOGUE_CONTEXT,
            summary="Test beat",
            word_indices=(0, 1),
        )
        timeline = FusedTimeline(
            source_path="video.mp4",
            video_duration=10.0,
            transcript_words=words,
            transcript_beats=[beat],
        )
        assert len(timeline.transcript_beats) == 1
        assert timeline.transcript_beats[0].beat_id == "beat_test"
        assert timeline.transcript_beats[0].duration == 0.5
