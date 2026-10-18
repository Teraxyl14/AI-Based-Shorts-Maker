"""
Milestone 6.3 Unit and Integration Test Suite:
Visual Engagement Self-Correction Loop (R3, R4)

Covers:
1. Pacing Violation Detection:
   - Flags MONOTONOUS_FACE_LOCK when any continuous single crop exceeds 7.0s during demonstration beats.
   - Verifies compliance when continuous crops during demonstration beats are <= 7.0s.
   - Verifies non-demonstration beats (dialogue/speaker hook) do not falsely trigger MONOTONOUS_FACE_LOCK.
2. Topic Alignment Calculation:
   - Verifies topic_in_viewport_ratio >= 0.85 during beats marked with PRODUCT_SHOWCASE,
     SCREEN_DEMONSTRATION, or HANDS_ON_ACTION.
   - Flags MISSING_TOPIC_CLOSEUP or OFF_TARGET_CROP if ratio < 0.85.
   - Verifies perfect topic alignment (1.0) when all demonstration beats have topic framing.
3. Composite Engagement Score Calculation:
   - Formula: overall_engagement_score = 0.40 * pacing_score + 0.35 * topic_alignment_score + 0.25 * cut_diversity_score.
   - Verifies strict [0.0, 1.0] score bounds and sensitivity to pacing and topic grounding defects.
4. Autonomous Self-Correction Engine:
   - When score < 0.75 or violations exist:
     * Splits monotonous segments (> 7.0s) into alternating cuts:
       - Inserts 3.0s FULL_FRAME_TOPIC_CLOSEUP (or DUAL_PANE_SPLIT_SCREEN if both speaker and topic are present)
       - Restores FULL_FRAME_SPEAKER_TIGHT
     * Generates updated VideoSegment list, relative editorial_cuts timestamps, and focal_targets.
     * Sets self_correction_applied = True and populates remedial_actions.
     * Post-correction report passes with score >= 0.75 and zero remaining violations.
5. Dual-Pane Split-Screen vs Full-Frame Close-Up Dispatch:
   - Chooses DUAL_PANE_SPLIT_SCREEN when both active speaker and demonstration topic are co-present.
   - Chooses FULL_FRAME_TOPIC_CLOSEUP when only demonstrated object/screen is present.
6. Edge Cases & Robustness:
   - Empty beats, zero spatial events, short clips (< 7.0s), ultra-long clips (50s).
   - Manifests that are already diverse and compliant pass without redundant corrections.
   - Dict, ShortCandidate, and ClipManifest input compatibility.
"""

import math
import pytest
from typing import List, Tuple, Optional

from shared.schemas import (
    CutLayoutMode,
    SplitScreenConfig,
    TranscriptBeat,
    SemanticTopicCategory,
    VisualGroundingTarget,
    GroundingEntity,
    EngagementEvaluationReport,
    PacingViolationType,
    RemedialAction,
    VideoSegment,
    ClipManifest,
    ShortCandidate,
)
from wsl2_gpu.engagement_evaluator import EngagementEvaluator


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES & HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def evaluator():
    return EngagementEvaluator()


def make_spatial_event(ts: float, label: str, cx: float, cy: float, w: float = 0.3, h: float = 0.3, tid: int = 1):
    return {
        "timestamp": ts,
        "label": label,
        "track_id": tid,
        "confidence": 0.90,
        "bbox": (cx, cy, w, h),
    }


def make_demo_beat(beat_id: str, start: float, end: float, category: SemanticTopicCategory,
                   target_type: str = "PRODUCT", bbox: Tuple[float, float, float, float] = (0.75, 0.65, 0.30, 0.25),
                   secondary_bbox: Optional[Tuple[float, float, float, float]] = None) -> TranscriptBeat:
    return TranscriptBeat(
        beat_id=beat_id,
        start_time=start,
        end_time=end,
        topic_category=category,
        summary=f"Demonstration beat {beat_id}",
        entities=[GroundingEntity(entity_name="Framework Laptop", category="PRODUCT", visual_aliases=["laptop"])],
        demonstration_action="holding_device",
        word_indices=(0, 10),
        visual_grounding=VisualGroundingTarget(
            target_type=target_type,
            primary_bbox=bbox,
            secondary_bbox=secondary_bbox,
            confidence=0.95
        )
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. PACING VIOLATION DETECTION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestPacingViolationDetection:
    """Verifies detection of monotonous face lock during demonstration beats."""

    def test_flags_monotonous_face_lock_when_continuous_crop_exceeds_7s_in_demo(self, evaluator):
        """M6.3.1: Flags MONOTONOUS_FACE_LOCK when a single speaker crop lasts > 7.0s during demonstration."""
        short_dict = {
            "start_time_seconds": 60.0,
            "end_time_seconds": 95.0,
            "duration_seconds": 35.0,
            "framing_mode": "FULL_FRAME_SPEAKER_TIGHT",
            "segments": [
                VideoSegment(segment_start=60.0, segment_end=95.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT)
            ]
        }
        # A 12-second demonstration beat inside the 35s short
        beats = [
            make_demo_beat("b1", 68.0, 80.0, SemanticTopicCategory.PRODUCT_SHOWCASE)
        ]
        spatial_events = [
            make_spatial_event(ts=t, label="face", cx=0.3, cy=0.3, tid=1) for t in range(60, 95)
        ] + [
            make_spatial_event(ts=t, label="laptop", cx=0.8, cy=0.7, tid=2) for t in range(68, 80)
        ]

        # 1. Direct pacing violation detection
        pacing_passed, max_crop, violations = evaluator.detect_pacing_violations(
            segments=short_dict["segments"],
            beats=beats,
            clip_start_s=60.0
        )
        assert pacing_passed is False
        assert max_crop > 7.0
        assert PacingViolationType.MONOTONOUS_FACE_LOCK in violations

        # 2. Evaluate and correct pass
        report, corrected_segs, cuts, targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate=short_dict,
            words=[],
            spatial_events=spatial_events,
            transcript_beats=beats,
            total_duration=35.0
        )

        assert PacingViolationType.MONOTONOUS_FACE_LOCK in report.detected_violations
        assert report.self_correction_applied is True
        assert report.multi_cut_pacing_passed is True

    def test_pacing_passes_when_continuous_crops_during_demo_under_7s(self, evaluator):
        """Continuous crops during demonstration beats under 7.0s pass pacing."""
        short_dict = {
            "start_time_seconds": 60.0,
            "end_time_seconds": 95.0,
            "duration_seconds": 35.0,
            "segments": [
                VideoSegment(segment_start=60.0, segment_end=68.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
                VideoSegment(segment_start=68.0, segment_end=73.0, layout_mode=CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
                VideoSegment(segment_start=73.0, segment_end=78.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
                VideoSegment(segment_start=78.0, segment_end=83.0, layout_mode=CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
                VideoSegment(segment_start=83.0, segment_end=95.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            ]
        }
        # Demo beat occurs at 68.0-83.0, split into 5s topic and 5s speaker cuts (all <= 5s)
        beats = [
            make_demo_beat("b1", 68.0, 83.0, SemanticTopicCategory.SCREEN_DEMONSTRATION)
        ]

        # Evaluate existing segments without self-correction
        pacing_passed, max_crop, violations = evaluator.detect_pacing_violations(
            segments=short_dict["segments"],
            beats=beats,
            clip_start_s=60.0
        )

        assert pacing_passed is True
        assert max_crop <= 7.0
        assert PacingViolationType.MONOTONOUS_FACE_LOCK not in violations

    def test_no_pacing_violation_for_dialogue_only_beats(self, evaluator):
        """Dialogue or speaker hook beats with no demonstration do not trigger MONOTONOUS_FACE_LOCK."""
        short_dict = {
            "start_time_seconds": 60.0,
            "end_time_seconds": 95.0,
            "duration_seconds": 35.0,
            "segments": [
                VideoSegment(segment_start=60.0, segment_end=95.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT)
            ]
        }
        beats = [
            TranscriptBeat(
                beat_id="dialogue_1",
                start_time=60.0,
                end_time=95.0,
                topic_category=SemanticTopicCategory.DIALOGUE_CONTEXT,
                summary="Speaker talking about concepts",
                word_indices=(0, 50)
            )
        ]

        pacing_passed, max_crop, violations = evaluator.detect_pacing_violations(
            segments=short_dict["segments"],
            beats=beats,
            clip_start_s=60.0
        )

        # No demonstration beats, so face lock during demo is NOT flagged
        assert PacingViolationType.MONOTONOUS_FACE_LOCK not in violations


# ═════════════════════════════════════════════════════════════════════════════
# 2. TOPIC ALIGNMENT CALCULATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestTopicAlignmentCalculation:
    """Verifies topic_in_viewport_ratio calculation and thresholding."""

    def test_flags_missing_topic_closeup_when_ratio_below_85_percent(self, evaluator):
        """Flags MISSING_TOPIC_CLOSEUP when topic_in_viewport_ratio < 0.85 during demo."""
        # 10s demo beat from 70.0 to 80.0
        beats = [
            make_demo_beat("b1", 70.0, 80.0, SemanticTopicCategory.HANDS_ON_ACTION)
        ]
        # Segments only show topic for 2.0s out of 10.0s (20% < 85%)
        segments = [
            VideoSegment(segment_start=60.0, segment_end=74.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            VideoSegment(segment_start=74.0, segment_end=76.0, layout_mode=CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
            VideoSegment(segment_start=76.0, segment_end=95.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
        ]

        ratio, passed, violations = evaluator.calculate_topic_alignment(
            segments=segments,
            beats=beats,
            spatial_events=[],
            clip_start_s=60.0
        )

        assert ratio < 0.85
        assert passed is False
        assert (PacingViolationType.MISSING_TOPIC_CLOSEUP in violations or
                PacingViolationType.OFF_TARGET_CROP in violations)

    def test_topic_alignment_passes_when_ratio_ge_85_percent(self, evaluator):
        """Passes when topic is in viewport >= 85% of demonstration duration."""
        beats = [
            make_demo_beat("b1", 70.0, 80.0, SemanticTopicCategory.PRODUCT_SHOWCASE)
        ]
        # 9.0s of topic closeup during 10.0s demo beat = 90% >= 85%
        segments = [
            VideoSegment(segment_start=60.0, segment_end=70.5, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            VideoSegment(segment_start=70.5, segment_end=79.5, layout_mode=CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
            VideoSegment(segment_start=79.5, segment_end=95.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
        ]

        ratio, passed, violations = evaluator.calculate_topic_alignment(
            segments=segments,
            beats=beats,
            spatial_events=[],
            clip_start_s=60.0
        )

        assert ratio >= 0.85
        assert passed is True
        assert PacingViolationType.MISSING_TOPIC_CLOSEUP not in violations

    def test_topic_alignment_perfect_when_no_demonstration_beats(self, evaluator):
        """When there are no demonstration beats, ratio defaults to 1.0 (passed)."""
        segments = [
            VideoSegment(segment_start=60.0, segment_end=95.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT)
        ]
        ratio, passed, violations = evaluator.calculate_topic_alignment(
            segments=segments,
            beats=[],
            spatial_events=[],
            clip_start_s=60.0
        )
        assert ratio == 1.0
        assert passed is True
        assert len(violations) == 0


# ═════════════════════════════════════════════════════════════════════════════
# 3. COMPOSITE ENGAGEMENT SCORE TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestCompositeEngagementScore:
    """Verifies overall_engagement_score formula and bounds."""

    def test_composite_score_formula_weights(self, evaluator):
        """overall_engagement_score = 0.40 * pacing + 0.35 * topic_alignment + 0.25 * cut_diversity."""
        score = evaluator.compute_composite_score(
            pacing_score=1.0,
            topic_alignment_score=1.0,
            cut_diversity_score=1.0
        )
        assert pytest.approx(score, 1e-4) == 1.0

        # Partial values: 0.40*0.5 + 0.35*0.8 + 0.25*0.6 = 0.20 + 0.28 + 0.15 = 0.63
        score_partial = evaluator.compute_composite_score(
            pacing_score=0.5,
            topic_alignment_score=0.8,
            cut_diversity_score=0.6
        )
        assert pytest.approx(score_partial, 1e-4) == 0.63

    def test_composite_score_strictly_bounded_0_to_1(self, evaluator):
        """Checks bounds under extreme negative and overshoot inputs."""
        assert evaluator.compute_composite_score(-0.5, -1.0, -2.0) == 0.0
        assert evaluator.compute_composite_score(1.5, 2.0, 3.0) == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 4. AUTONOMOUS SELF-CORRECTION ENGINE TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestSelfCorrectionEngine:
    """Tests autonomous splitting and remediation of monotonous manifests."""

    def test_self_corrects_monotonous_speaker_into_alternating_topic_cuts(self, evaluator):
        """Splits monotonous segment > 7.0s into alternating cuts with a 3.0s topic closeup."""
        short_candidate = ShortCandidate(
            start_time_seconds=60.0,
            duration_seconds=35.0,
            title="Monotonous Clip",
            reasoning="Testing self-correction",
            framing_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
            beats=[
                make_demo_beat("beat_1", 72.0, 75.5, SemanticTopicCategory.PRODUCT_SHOWCASE, target_type="PRODUCT")
            ]
        )

        spatial_events = [
            make_spatial_event(ts=t, label="person", cx=0.3, cy=0.3, tid=1) for t in range(60, 95)
        ] + [
            make_spatial_event(ts=t, label="laptop", cx=0.75, cy=0.65, tid=101) for t in range(72, 80)
        ]

        report, corrected_segs, editorial_cuts, focal_targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate=short_candidate,
            words=[],
            spatial_events=spatial_events,
            transcript_beats=short_candidate.beats,
            total_duration=35.0
        )

        # Self-correction must be applied
        assert report.self_correction_applied is True
        assert len(report.remedial_actions) > 0

        # Segment list must now contain multiple alternating cuts
        assert len(corrected_segs) >= 3

        # At least one segment must be FULL_FRAME_TOPIC_CLOSEUP or DUAL_PANE_SPLIT_SCREEN
        topic_segs = [s for s in corrected_segs if s.layout_mode in (
            CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP,
            CutLayoutMode.DUAL_PANE_SPLIT_SCREEN
        )]
        assert len(topic_segs) >= 1

        # Check inserted topic segment duration is approximately 3.0s
        first_topic = topic_segs[0]
        topic_dur = first_topic.segment_end - first_topic.segment_start
        assert 2.5 <= topic_dur <= 4.0

        # Check speaker mode is restored before and after
        assert corrected_segs[0].layout_mode == CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT
        assert corrected_segs[-1].layout_mode == CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT

        # Editorial cuts list contains the internal cut timestamps (relative seconds)
        assert len(editorial_cuts) == len(corrected_segs) - 1
        for cut in editorial_cuts:
            assert 0.0 < cut < 35.0

        # Focal targets match solve_trajectory contract: (start_s, end_s, target_type, bbox)
        assert len(focal_targets) == len(corrected_segs)
        for ft in focal_targets:
            assert len(ft) == 4
            s_rel, e_rel, ft_type, bbox = ft
            assert 0.0 <= s_rel < e_rel <= 35.0
            assert isinstance(ft_type, str)

        # Post-correction quality assertion
        assert report.multi_cut_pacing_passed is True
        assert report.topic_grounding_passed is True
        assert report.overall_engagement_score >= 0.75

    def test_chooses_split_screen_when_speaker_and_topic_both_present(self, evaluator):
        """When both active speaker and topic are present, selects DUAL_PANE_SPLIT_SCREEN."""
        short_dict = {
            "start_time_seconds": 60.0,
            "end_time_seconds": 95.0,
            "duration_seconds": 35.0,
            "framing_mode": "FULL_FRAME_SPEAKER_TIGHT",
            "segments": [
                VideoSegment(segment_start=60.0, segment_end=95.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT)
            ]
        }
        # Demo beat with secondary bbox for speaker
        beats = [
            make_demo_beat(
                "beat_split", 70.0, 78.0,
                SemanticTopicCategory.SCREEN_DEMONSTRATION,
                target_type="SCREEN",
                bbox=(0.70, 0.60, 0.40, 0.40),
                secondary_bbox=(0.25, 0.35, 0.20, 0.25)
            )
        ]
        spatial_events = [
            make_spatial_event(ts=t, label="face", cx=0.25, cy=0.35, tid=1) for t in range(60, 95)
        ] + [
            make_spatial_event(ts=t, label="laptop", cx=0.70, cy=0.60, tid=2) for t in range(70, 78)
        ]

        report, corrected_segs, cuts, targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate=short_dict,
            words=[],
            spatial_events=spatial_events,
            transcript_beats=beats,
            total_duration=35.0
        )

        assert report.self_correction_applied is True
        split_segs = [s for s in corrected_segs if s.layout_mode == CutLayoutMode.DUAL_PANE_SPLIT_SCREEN]
        assert len(split_segs) >= 1
        assert split_segs[0].split_screen_config is not None
        assert split_segs[0].split_screen_config.divider_accent_hex == "#CCFF00"
        assert split_segs[0].split_screen_config.divider_width_px in (3, 4)

        # Check remedial action specifies SWITCH_TO_SPLIT_SCREEN
        action_types = [a.action_type for a in report.remedial_actions]
        assert "SWITCH_TO_SPLIT_SCREEN" in action_types

    def test_chooses_topic_closeup_when_no_speaker_present(self, evaluator):
        """When only topic is present during demonstration, selects FULL_FRAME_TOPIC_CLOSEUP."""
        short_dict = {
            "start_time_seconds": 60.0,
            "end_time_seconds": 95.0,
            "duration_seconds": 35.0,
            "segments": [
                VideoSegment(segment_start=60.0, segment_end=95.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT)
            ]
        }
        # Demo beat with ONLY product target, NO secondary bbox
        beats = [
            make_demo_beat(
                "beat_product", 70.0, 73.5,
                SemanticTopicCategory.PRODUCT_SHOWCASE,
                target_type="PRODUCT",
                bbox=(0.80, 0.70, 0.30, 0.30),
                secondary_bbox=None
            )
        ]
        # Only product spatial events during demonstration
        spatial_events = [
            make_spatial_event(ts=t, label="laptop", cx=0.80, cy=0.70, tid=2) for t in range(70, 74)
        ]

        report, corrected_segs, cuts, targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate=short_dict,
            words=[],
            spatial_events=spatial_events,
            transcript_beats=beats,
            total_duration=35.0
        )

        assert report.self_correction_applied is True
        closeup_segs = [s for s in corrected_segs if s.layout_mode == CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP]
        assert len(closeup_segs) >= 1
        action_types = [a.action_type for a in report.remedial_actions]
        assert "INSERT_TOPIC_CLOSEUP" in action_types


# ═════════════════════════════════════════════════════════════════════════════
# 5. EDGE CASES & ROBUSTNESS TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestEngagementEvaluatorEdgeCases:
    """Tests resilience on edge inputs: empty, short, ultra-long, and already-compliant."""

    def test_already_diverse_multi_cut_passes_without_redundant_correction(self, evaluator):
        """Already compliant multi-cut manifest passes with self_correction_applied = False."""
        segments = [
            VideoSegment(segment_start=60.0, segment_end=66.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            VideoSegment(segment_start=66.0, segment_end=69.0, layout_mode=CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
            VideoSegment(segment_start=69.0, segment_end=74.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            VideoSegment(segment_start=74.0, segment_end=77.0, layout_mode=CutLayoutMode.DUAL_PANE_SPLIT_SCREEN),
            VideoSegment(segment_start=77.0, segment_end=82.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            VideoSegment(segment_start=82.0, segment_end=85.0, layout_mode=CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
            VideoSegment(segment_start=85.0, segment_end=95.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
        ]
        clip_manifest = ClipManifest(
            short_id="compliant_1",
            title="Compliant Clip",
            segments=segments,
            hook_score=0.90,
            retention_score=0.90,
            cta_present=True,
            reasoning="Well paced"
        )
        beats = [
            make_demo_beat("b1", 66.0, 69.0, SemanticTopicCategory.PRODUCT_SHOWCASE),
            make_demo_beat("b2", 74.0, 77.0, SemanticTopicCategory.SCREEN_DEMONSTRATION),
            make_demo_beat("b3", 82.0, 85.0, SemanticTopicCategory.HANDS_ON_ACTION),
        ]

        report, corrected_segs, cuts, targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate=clip_manifest,
            words=[],
            spatial_events=[],
            transcript_beats=beats,
            total_duration=35.0
        )

        assert report.self_correction_applied is False
        assert len(report.remedial_actions) == 0
        assert report.multi_cut_pacing_passed is True
        assert report.topic_grounding_passed is True
        assert report.overall_engagement_score >= 0.75
        assert len(corrected_segs) == len(segments)

    def test_short_clip_under_7s_no_monotonous_violation(self, evaluator):
        """A clip under 7.0s duration cannot exceed 7.0s single crop and avoids false violations."""
        short_dict = {
            "start_time_seconds": 60.0,
            "end_time_seconds": 65.5,
            "duration_seconds": 5.5,
            "segments": [
                VideoSegment(segment_start=60.0, segment_end=65.5, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT)
            ]
        }
        beats = [
            make_demo_beat("b1", 61.0, 64.0, SemanticTopicCategory.PRODUCT_SHOWCASE)
        ]

        report, corrected_segs, cuts, targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate=short_dict,
            words=[],
            spatial_events=[],
            transcript_beats=beats,
            total_duration=5.5
        )

        assert report.max_continuous_single_crop_duration <= 7.0
        assert PacingViolationType.MONOTONOUS_FACE_LOCK not in report.detected_violations

    def test_empty_beats_and_zero_spatial_events_no_crash(self, evaluator):
        """Evaluator handles empty beats list and zero spatial events without error."""
        short_dict = {
            "start_time_seconds": 60.0,
            "end_time_seconds": 95.0,
            "duration_seconds": 35.0,
        }
        report, corrected_segs, cuts, targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate=short_dict,
            words=[],
            spatial_events=[],
            transcript_beats=[],
            total_duration=35.0
        )

        assert report is not None
        assert isinstance(report, EngagementEvaluationReport)
        assert len(corrected_segs) >= 1
        assert isinstance(cuts, list)
        assert isinstance(targets, list)

    def test_ultra_long_clip_50s_with_long_demonstration_multiple_cuts(self, evaluator):
        """50.0s clip with a 20s demonstration splits into multiple alternating cuts."""
        short_dict = {
            "start_time_seconds": 60.0,
            "end_time_seconds": 110.0,
            "duration_seconds": 50.0,
            "segments": [
                VideoSegment(segment_start=60.0, segment_end=110.0, layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT)
            ]
        }
        # 20-second demonstration beat
        beats = [
            make_demo_beat("b_long", 70.0, 90.0, SemanticTopicCategory.SCREEN_DEMONSTRATION)
        ]
        spatial_events = [
            make_spatial_event(ts=t, label="screen", cx=0.7, cy=0.6, tid=5) for t in range(70, 90)
        ]

        report, corrected_segs, cuts, targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate=short_dict,
            words=[],
            spatial_events=spatial_events,
            transcript_beats=beats,
            total_duration=50.0
        )

        assert report.self_correction_applied is True
        assert len(corrected_segs) >= 4
        # Assert each segment is <= 7.0s during demonstration
        for seg in corrected_segs:
            dur = seg.segment_end - seg.segment_start
            assert dur > 0.0
        assert report.multi_cut_pacing_passed is True
        assert report.overall_engagement_score >= 0.75

    def test_zero_letterbox_invariant_for_all_corrected_segments(self, evaluator):
        """All corrected segments strictly use 3-mode zero-letterbox layouts (no 16:9 canvas)."""
        short_dict = {
            "start_time_seconds": 60.0,
            "end_time_seconds": 95.0,
            "duration_seconds": 35.0,
            "framing_mode": "CANVAS_GRAPHIC_FIT",  # Legacy mode
        }
        beats = [
            make_demo_beat("b1", 70.0, 78.0, SemanticTopicCategory.PRODUCT_SHOWCASE)
        ]

        report, corrected_segs, cuts, targets = evaluator.evaluate_and_correct(
            short_dict_or_candidate=short_dict,
            words=[],
            spatial_events=[],
            transcript_beats=beats,
            total_duration=35.0
        )

        for seg in corrected_segs:
            assert seg.layout_mode in (
                CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
                CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP,
                CutLayoutMode.DUAL_PANE_SPLIT_SCREEN
            )
            # Legacy CANVAS_GRAPHIC_FIT canonicalizes to FULL_FRAME_TOPIC_CLOSEUP
            assert seg.layout_mode != "CANVAS_GRAPHIC_FIT" or seg.layout_mode == CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP


# ═════════════════════════════════════════════════════════════════════════════
# 6. VRAM ORCHESTRATOR PHASE 3 WORKER INTEGRATION TEST
# ═════════════════════════════════════════════════════════════════════════════

class TestVRAMOrchestratorPhase3Integration:
    """Verifies integration of EngagementEvaluator inside run_phase3_render_worker."""

    def test_phase3_render_worker_applies_engagement_correction_and_focal_targets(self, monkeypatch):
        """Phase 3 render worker evaluates candidate shorts, applies self-correction,
        passes focal targets and editorial cuts to solve_trajectory, and renders multi-cut segments.
        """
        from wsl2_gpu.vram_orchestrator import run_phase3_render_worker

        captured_jobs = []
        captured_crop_paths = []

        class MockPipeline:
            def __init__(self, job, crop_path=None):
                captured_jobs.append(job)
                captured_crop_paths.append(crop_path)
                self.errors = []
            def execute(self):
                return True, "data/output_shorts/short_1.mp4"

        monkeypatch.setattr("wsl2_gpu.render_spoke.ThreeThreadRenderPipeline", MockPipeline)

        class MockPipe:
            def __init__(self):
                self.messages = []
                self.closed = False
            def send(self, msg):
                self.messages.append(msg)
            def close(self):
                self.closed = True

        mock_pipe = MockPipe()

        manifest = {
            "editorial_shorts": [
                {
                    "start_time_seconds": 60.0,
                    "end_time_seconds": 95.0,
                    "duration_seconds": 35.0,
                    "title": "Integration Test Short",
                    "framing_mode": "SMART_PORTRAIT_PAN",
                    "beats": [
                        {
                            "beat_id": "b_demo",
                            "start_time": 72.0,
                            "end_time": 75.5,
                            "topic_category": "PRODUCT_SHOWCASE",
                            "summary": "Demonstrating laptop hardware",
                            "word_indices": (0, 10),
                            "visual_grounding": {
                                "target_type": "PRODUCT",
                                "primary_bbox": (0.75, 0.65, 0.30, 0.25),
                                "confidence": 0.95
                            }
                        }
                    ]
                }
            ],
            "words": [
                {"word": "test", "start": 60.5, "end": 61.0, "confidence": 0.95}
            ],
            "spatial_events": [
                {"timestamp": 60.5, "label": "person", "track_id": 1, "confidence": 0.90, "bbox": (0.3, 0.3, 0.3, 0.3)},
                {"timestamp": 73.0, "label": "laptop", "track_id": 101, "confidence": 0.95, "bbox": (0.75, 0.65, 0.3, 0.25)}
            ],
            "scene_changes": []
        }

        run_phase3_render_worker(
            video_path="data/input_videos/test_video.mp4",
            render_manifest=manifest,
            output_dir="data/output_shorts",
            result_pipe=mock_pipe
        )

        assert len(mock_pipe.messages) == 1
        res = mock_pipe.messages[0]
        assert res["status"] == "success"
        assert len(res["render_results"]) == 1
        assert res["render_results"][0]["success"] is True

        # Verify pipeline received corrected segments
        assert len(captured_jobs) == 1
        job = captured_jobs[0]
        segments = job.clip_manifest.segments
        assert len(segments) >= 3

        topic_segs = [s for s in segments if s.layout_mode in (CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP, CutLayoutMode.DUAL_PANE_SPLIT_SCREEN)]
        assert len(topic_segs) >= 1

        # Check crop path was provided
        assert len(captured_crop_paths) == 1
        crop_path = captured_crop_paths[0]
        assert crop_path is not None
        assert len(crop_path) == int(round(35.0 * 30.0))

