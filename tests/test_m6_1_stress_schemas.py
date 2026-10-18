"""
Adversarial Stress Test Suite for Milestone 6.1 Schemas (shared/schemas.py).

Targets:
1. Extreme timestamps (negative, inverted, huge, sub-millisecond precision, NaN, Inf).
2. Unicode, multilingual, emojis, control characters, empty strings, massive payloads in TranscriptBeat & GroundingEntity.
3. CutLayoutMode equality checks, alias mappings, transitivity failure analysis, and hashability.
4. VideoSegment and ShortCandidate serialization, JSON round-trips, and default value propagation.
5. Clean input rejection without crashes (verifying pydantic.ValidationError).
"""

import math
import pytest
from typing import List, Tuple
from pydantic import ValidationError

from shared.schemas import (
    SemanticTopicCategory,
    GroundingEntity,
    VisualGroundingTarget,
    TranscriptBeat,
    CutLayoutMode,
    SplitScreenConfig,
    PacingViolationType,
    RemedialAction,
    EngagementEvaluationReport,
    VideoSegment,
    ShortCandidate,
    FusedTimeline,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. EXTREME TIMESTAMPS & FLOATING POINT PRECISION
# ═════════════════════════════════════════════════════════════════════════════

class TestExtremeTimestamps:
    """Stress tests boundary and extreme timestamp values."""

    def test_transcript_beat_negative_timestamps(self):
        """Verifies TranscriptBeat handles negative timestamps without crashing and duration calculates correctly."""
        beat = TranscriptBeat(
            beat_id="beat_neg",
            start_time=-15.5,
            end_time=-5.0,
            summary="Pre-roll timeline beat",
            word_indices=(0, 2),
        )
        assert beat.start_time == -15.5
        assert beat.end_time == -5.0
        assert pytest.approx(beat.duration, 0.001) == 10.5

    def test_transcript_beat_inverted_timestamps(self):
        """Verifies when end_time < start_time, .duration property safely clips to 0.0 instead of negative."""
        beat = TranscriptBeat(
            beat_id="beat_inverted",
            start_time=50.0,
            end_time=30.0,
            summary="Inverted timestamp beat",
            word_indices=(5, 10),
        )
        assert beat.duration == 0.0

    def test_transcript_beat_huge_timestamps(self):
        """Verifies handling of extremely large timestamps without overflow or loss of stability."""
        huge_start = 1e12  # ~31,000 years
        huge_end = huge_start + 45.0
        beat = TranscriptBeat(
            beat_id="beat_huge",
            start_time=huge_start,
            end_time=huge_end,
            summary="Astronomical timestamp beat",
            word_indices=(1000, 1050),
        )
        assert beat.start_time == 1e12
        assert pytest.approx(beat.duration, 0.01) == 45.0

    def test_transcript_beat_submillisecond_floating_precision(self):
        """Verifies sub-millisecond precision and 0.1 + 0.2 floating point behavior."""
        start = 0.1 + 0.2  # 0.30000000000000004
        end = 0.30000000000000004 + 0.000001  # 1 microsecond later
        beat = TranscriptBeat(
            beat_id="beat_precision",
            start_time=start,
            end_time=end,
            summary="Microsecond beat",
            word_indices=(0, 1),
        )
        assert pytest.approx(beat.duration, 1e-9) == 1e-6

    def test_transcript_beat_nan_and_inf_handling(self):
        """Verifies behavior when float('inf') or float('nan') are passed into TranscriptBeat."""
        # Inf
        beat_inf = TranscriptBeat(
            beat_id="beat_inf",
            start_time=float("inf"),
            end_time=float("inf"),
            summary="Inf beat",
            word_indices=(0, 1),
        )
        # inf - inf = nan; max(0.0, nan) in Python returns 0.0
        assert beat_inf.duration == 0.0

        # NaN
        beat_nan = TranscriptBeat(
            beat_id="beat_nan",
            start_time=float("nan"),
            end_time=float("nan"),
            summary="NaN beat",
            word_indices=(0, 1),
        )
        assert math.isnan(beat_nan.start_time)
        assert beat_nan.duration == 0.0

    def test_short_candidate_boundary_precision(self):
        """Verifies exact boundaries for ShortCandidate (start >= 60.0, 30.0 <= duration <= 50.0)."""
        # Exactly on the lower boundary (should pass)
        sc_exact_low = ShortCandidate(
            start_time_seconds=60.0,
            duration_seconds=30.0,
        )
        assert sc_exact_low.start_time_seconds == 60.0
        assert sc_exact_low.duration_seconds == 30.0

        # Exactly on the upper boundary (should pass)
        sc_exact_high = ShortCandidate(
            start_time_seconds=120.0,
            duration_seconds=50.0,
        )
        assert sc_exact_high.duration_seconds == 50.0

        # Floating point epsilon below start_time (must fail)
        with pytest.raises(ValidationError):
            ShortCandidate(
                start_time_seconds=59.9999999999999,
                duration_seconds=35.0,
            )

        # Floating point epsilon below duration (must fail)
        with pytest.raises(ValidationError):
            ShortCandidate(
                start_time_seconds=60.0,
                duration_seconds=29.9999999999999,
            )

        # Floating point epsilon above duration (must fail)
        with pytest.raises(ValidationError):
            ShortCandidate(
                start_time_seconds=60.0,
                duration_seconds=50.0000000000001,
            )

        # NaN should be rejected
        with pytest.raises(ValidationError):
            ShortCandidate(
                start_time_seconds=float("nan"),
                duration_seconds=35.0,
            )
        with pytest.raises(ValidationError):
            ShortCandidate(
                start_time_seconds=60.0,
                duration_seconds=float("nan"),
            )

    def test_video_segment_extreme_timestamps(self):
        """Verifies VideoSegment accepts zero, negative, and large timestamps."""
        seg = VideoSegment(
            segment_start=0.0,
            segment_end=1e6,
        )
        assert seg.segment_start == 0.0
        assert seg.segment_end == 1e6


# ═════════════════════════════════════════════════════════════════════════════
# 2. UNICODE, MULTILINGUAL, AND EMPTY ENTITIES/SUMMARIES
# ═════════════════════════════════════════════════════════════════════════════

class TestUnicodeAndMultilingualEntities:
    """Stress tests Unicode, RTL, CJK, Emojis, empty, and massive string inputs."""

    def test_grounding_entity_multilingual_scripts(self):
        """Verifies GroundingEntity handles diverse scripts (Japanese, Arabic, Cyrillic, Hindi, accents)."""
        scripts = [
            ("モジュラーノートPC", ["ラップトップ", "コンピューター"]),  # Japanese
            ("كمبيوتر محمول معياري", ["شاشة", "لوحة مفاتيح"]),        # Arabic RTL
            ("Модульный ноутбук", ["лэптоп", "экран"]),              # Cyrillic
            ("मॉड्यूलर लैपटॉप", ["कंप्यूटर", "स्क्रीन"]),           # Devanagari
            ("Système modulaire révolutionnaire", ["ordinateur"]),     # French
            ("Nahtloses modulares Gehäuse", ["Rechner"]),             # German
        ]
        for name, aliases in scripts:
            entity = GroundingEntity(
                entity_name=name,
                category="PRODUCT",
                visual_aliases=aliases,
                salience_score=0.9,
            )
            assert entity.entity_name == name
            assert entity.visual_aliases == aliases

    def test_grounding_entity_emojis_and_special_symbols(self):
        """Verifies GroundingEntity handles emojis, ZWJ sequences, and math symbols."""
        emojis_and_symbols = [
            "Framework 💻 16 🚀",
            "Hardware Mod 👨‍💻🛠️",
            "ΔV = I_sp * g_0 * ln(m_0 / m_f) ∫ ∮",
            "Special chars: < > & \" ' / \\ | ~ ` ! @ # $ % ^ * ( ) _ +",
        ]
        for text in emojis_and_symbols:
            entity = GroundingEntity(
                entity_name=text,
                category="PRODUCT",
                visual_aliases=[text],
            )
            assert entity.entity_name == text

    def test_grounding_entity_empty_and_whitespace_strings(self):
        """Verifies GroundingEntity allows empty and whitespace-only names and aliases."""
        entity = GroundingEntity(
            entity_name="",
            category="PRODUCT",
            visual_aliases=["", "   ", "\t"],
            salience_score=0.0,
        )
        assert entity.entity_name == ""
        assert len(entity.visual_aliases) == 3
        assert entity.salience_score == 0.0

    def test_grounding_entity_massive_strings(self):
        """Stress tests GroundingEntity with massive 100,000-character payload."""
        massive_name = "Framework_" * 10000  # 100,000 chars
        entity = GroundingEntity(
            entity_name=massive_name,
            category="PRODUCT",
            visual_aliases=[massive_name[:500]],
        )
        assert len(entity.entity_name) == 100000

    def test_grounding_entity_control_characters(self):
        """Verifies GroundingEntity with newlines, tabs, and null-bytes."""
        ctrl_text = "Line 1\nLine 2\r\nLine 3\tTabbed\x00Null"
        entity = GroundingEntity(
            entity_name=ctrl_text,
            category="DOCUMENT",
        )
        assert entity.entity_name == ctrl_text

    def test_transcript_beat_unicode_summary(self):
        """Verifies TranscriptBeat handles multilingual and emoji summaries cleanly."""
        summary_text = "Présentation du GPU modulaire 🚀 avec benchmarks en direct (4K@120fps) 🔥"
        beat = TranscriptBeat(
            beat_id="beat_multi",
            start_time=10.0,
            end_time=15.0,
            summary=summary_text,
            word_indices=(5, 12),
        )
        assert beat.summary == summary_text


# ═════════════════════════════════════════════════════════════════════════════
# 3. CUTLAYOUTMODE EQUALITY, ALIASES, & HASHABILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestCutLayoutModeAdversarial:
    """Stress tests CutLayoutMode equality matrix, cross-aliases, and hashability."""

    def test_canonical_modes_identity(self):
        """Verifies canonical modes equal themselves as enum and as string."""
        canonicals = [
            CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
            CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP,
            CutLayoutMode.DUAL_PANE_SPLIT_SCREEN,
        ]
        for mode in canonicals:
            assert mode == mode
            assert mode == mode.value
            assert mode.value == mode

    def test_alias_to_canonical_bidirectional_equality(self):
        """Verifies bidirectional equality for all (alias, canonical) pairs."""
        pairs = [
            (CutLayoutMode.SMART_PORTRAIT_PAN, CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            (CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT, CutLayoutMode.SMART_PORTRAIT_PAN),
            ("SMART_PORTRAIT_PAN", CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT),
            (CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT, "SMART_PORTRAIT_PAN"),
            (CutLayoutMode.SMART_PORTRAIT_PAN, "FULL_FRAME_SPEAKER_TIGHT"),
            ("FULL_FRAME_SPEAKER_TIGHT", CutLayoutMode.SMART_PORTRAIT_PAN),

            (CutLayoutMode.CANVAS_GRAPHIC_FIT, CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
            (CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP, CutLayoutMode.CANVAS_GRAPHIC_FIT),
            ("CANVAS_GRAPHIC_FIT", CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
            (CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP, "CANVAS_GRAPHIC_FIT"),

            (CutLayoutMode.TOPIC_FOCAL, CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
            (CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP, CutLayoutMode.TOPIC_FOCAL),
            ("TOPIC_FOCAL", CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
            (CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP, "TOPIC_FOCAL"),

            (CutLayoutMode.SPLIT_SCREEN, CutLayoutMode.DUAL_PANE_SPLIT_SCREEN),
            (CutLayoutMode.DUAL_PANE_SPLIT_SCREEN, CutLayoutMode.SPLIT_SCREEN),
            ("SPLIT_SCREEN", CutLayoutMode.DUAL_PANE_SPLIT_SCREEN),
            (CutLayoutMode.DUAL_PANE_SPLIT_SCREEN, "SPLIT_SCREEN"),
        ]
        for a, b in pairs:
            assert a == b, f"Equality failed for {a!r} == {b!r}"

    def test_transitivity_between_co_aliases(self):
        """
        Verifies alias transitivity holds across all co-aliases pointing to the same canonical mode.
        TOPIC_FOCAL and CANVAS_GRAPHIC_FIT both map to FULL_FRAME_TOPIC_CLOSEUP.
        Asserts that CutLayoutMode.TOPIC_FOCAL == CutLayoutMode.CANVAS_GRAPHIC_FIT evaluates to True.
        """
        assert CutLayoutMode.TOPIC_FOCAL == CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP
        assert CutLayoutMode.CANVAS_GRAPHIC_FIT == CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP

        # Alias-to-alias comparison evaluates to True via canonical mapping:
        assert CutLayoutMode.TOPIC_FOCAL == CutLayoutMode.CANVAS_GRAPHIC_FIT
        assert CutLayoutMode.CANVAS_GRAPHIC_FIT == CutLayoutMode.TOPIC_FOCAL

    def test_inequality_between_different_modes(self):
        """Verifies distinct layout modes evaluate to False."""
        distinct_pairs = [
            (CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT, CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP),
            (CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT, CutLayoutMode.DUAL_PANE_SPLIT_SCREEN),
            (CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP, CutLayoutMode.DUAL_PANE_SPLIT_SCREEN),
            (CutLayoutMode.SMART_PORTRAIT_PAN, CutLayoutMode.CANVAS_GRAPHIC_FIT),
            (CutLayoutMode.SMART_PORTRAIT_PAN, CutLayoutMode.SPLIT_SCREEN),
            (CutLayoutMode.SMART_PORTRAIT_PAN, "INVALID_LAYOUT"),
            (CutLayoutMode.SMART_PORTRAIT_PAN, 42),
            (CutLayoutMode.SMART_PORTRAIT_PAN, None),
            (CutLayoutMode.SMART_PORTRAIT_PAN, object()),
        ]
        for a, b in distinct_pairs:
            assert not (a == b), f"Expected False for {a!r} == {b!r}"

    def test_cut_layout_mode_hashability(self):
        """
        Verifies CutLayoutMode implements __hash__ consistent with __eq__.
        Asserts that hash(mode) works, set(CutLayoutMode) succeeds without TypeError,
        and dict lookups using CutLayoutMode keys work correctly.
        """
        # Hashing works without TypeError
        h_speaker = hash(CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT)
        h_pan = hash(CutLayoutMode.SMART_PORTRAIT_PAN)
        assert h_speaker == h_pan

        h_topic = hash(CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP)
        h_canvas = hash(CutLayoutMode.CANVAS_GRAPHIC_FIT)
        h_focal = hash(CutLayoutMode.TOPIC_FOCAL)
        assert h_topic == h_canvas == h_focal

        h_split = hash(CutLayoutMode.DUAL_PANE_SPLIT_SCREEN)
        h_split_alias = hash(CutLayoutMode.SPLIT_SCREEN)
        assert h_split == h_split_alias

        # set(CutLayoutMode) succeeds and collapses aliases to canonical modes
        mode_set = set(CutLayoutMode)
        assert len(mode_set) == 3
        assert CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT in mode_set
        assert CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP in mode_set
        assert CutLayoutMode.DUAL_PANE_SPLIT_SCREEN in mode_set

        # Dict key lookups work across aliases
        mapping = {
            CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT: "speaker_worker",
            CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP: "topic_worker",
            CutLayoutMode.DUAL_PANE_SPLIT_SCREEN: "split_worker",
        }
        assert mapping[CutLayoutMode.SMART_PORTRAIT_PAN] == "speaker_worker"
        assert mapping[CutLayoutMode.CANVAS_GRAPHIC_FIT] == "topic_worker"
        assert mapping[CutLayoutMode.TOPIC_FOCAL] == "topic_worker"
        assert mapping[CutLayoutMode.SPLIT_SCREEN] == "split_worker"


# ═════════════════════════════════════════════════════════════════════════════
# 4. SERIALIZATION, JSON ROUND-TRIPS, & DEFAULTS PROPAGATION
# ═════════════════════════════════════════════════════════════════════════════

class TestSerializationAndRoundTrip:
    """Stress tests JSON serialization, round-trip fidelity, and default value propagation."""

    def test_video_segment_defaults_propagation(self):
        """Verifies VideoSegment propagates all default values on minimal instantiation."""
        seg = VideoSegment(segment_start=60.0, segment_end=80.0)
        assert seg.target_track_id == 0
        assert seg.layout_mode == CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT
        assert seg.layout_mode == "SMART_PORTRAIT_PAN"
        assert seg.focal_target == "SPEAKER_PRIMARY"
        assert seg.grounded_target is None
        assert seg.split_screen_config is None
        assert seg.beat_id is None

    def test_video_segment_roundtrip_with_alias(self):
        """Verifies VideoSegment initialized with alias serializes and deserializes cleanly."""
        seg = VideoSegment(
            segment_start=65.0,
            segment_end=85.0,
            layout_mode="SMART_PORTRAIT_PAN",
        )
        json_data = seg.model_dump_json()
        deserialized = VideoSegment.model_validate_json(json_data)

        assert deserialized.segment_start == 65.0
        assert deserialized.segment_end == 85.0
        assert deserialized.layout_mode == "SMART_PORTRAIT_PAN"
        assert deserialized.layout_mode == CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT

    def test_short_candidate_deep_roundtrip(self):
        """Verifies complex ShortCandidate with nested TranscriptBeats serializes and deserializes accurately."""
        candidate = ShortCandidate(
            start_time_seconds=75.0,
            duration_seconds=40.0,
            title="Adversarial Short 🚀",
            reasoning="Hook on presenter holding motherboard module",
            framing_mode=CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP,
            hook_window=(0.0, 4.5),
            first_word_delay=0.35,
            payoff_snapped=True,
            beats=[
                TranscriptBeat(
                    beat_id="beat_stress_1",
                    start_time=75.0,
                    end_time=82.0,
                    topic_category=SemanticTopicCategory.HANDS_ON_ACTION,
                    summary="Swapping expansion module 🛠️",
                    entities=[
                        GroundingEntity(
                            entity_name="GPU Module",
                            category="HARDWARE_COMPONENT",
                            visual_aliases=["expansion card", "graphics card"],
                            salience_score=0.98,
                        )
                    ],
                    demonstration_action="swapping_module",
                    word_indices=(100, 115),
                    visual_grounding=VisualGroundingTarget(
                        target_type="HELD_OBJECT",
                        primary_bbox=(0.45, 0.65, 0.20, 0.25),
                        secondary_bbox=(0.50, 0.30, 0.30, 0.40),
                        track_id=105,
                        confidence=0.94,
                        anchor_mode="HAND_REGION",
                    ),
                )
            ],
        )

        json_str = candidate.model_dump_json()
        restored = ShortCandidate.model_validate_json(json_str)

        assert restored.start_time_seconds == 75.0
        assert restored.duration_seconds == 40.0
        assert restored.title == "Adversarial Short 🚀"
        assert restored.framing_mode == CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP
        assert restored.framing_mode == "CANVAS_GRAPHIC_FIT"
        assert restored.first_word_delay == 0.35
        assert restored.payoff_snapped is True
        assert len(restored.beats) == 1

        beat = restored.beats[0]
        assert beat.beat_id == "beat_stress_1"
        assert beat.duration == 7.0
        assert beat.topic_category == SemanticTopicCategory.HANDS_ON_ACTION
        assert beat.entities[0].entity_name == "GPU Module"
        assert beat.entities[0].salience_score == 0.98
        assert beat.visual_grounding.target_type == "HELD_OBJECT"
        assert beat.visual_grounding.primary_bbox == (0.45, 0.65, 0.20, 0.25)

    def test_split_screen_config_roundtrip(self):
        """Verifies SplitScreenConfig defaults and JSON round-trip."""
        cfg = SplitScreenConfig()
        json_str = cfg.model_dump_json()
        restored = SplitScreenConfig.model_validate_json(json_str)

        assert restored.top_pane_target == "SPEAKER"
        assert restored.bottom_pane_target == "TOPIC"
        assert restored.split_ratio == 0.50
        assert restored.divider_width_px == 4
        assert restored.divider_accent_hex == "#CCFF00"
        assert restored.divider_glow_radius == 6
        assert restored.corner_radius_px == 16
        assert restored.shadow_blur_px == 10

    def test_engagement_evaluation_report_roundtrip(self):
        """Verifies EngagementEvaluationReport JSON round-trip with enums and remedial actions."""
        report = EngagementEvaluationReport(
            short_id="short_99",
            overall_engagement_score=0.82,
            multi_cut_pacing_passed=False,
            topic_grounding_passed=True,
            max_continuous_single_crop_duration=8.5,
            topic_in_viewport_ratio=0.89,
            detected_violations=[PacingViolationType.MONOTONOUS_FACE_LOCK],
            remedial_actions=[
                RemedialAction(
                    segment_index=2,
                    action_type="INSERT_TOPIC_CLOSEUP",
                    target_coordinates=(0.5, 0.6, 0.3, 0.3),
                    reason="Cut away from face to topic",
                )
            ],
            self_correction_applied=True,
        )

        json_str = report.model_dump_json()
        restored = EngagementEvaluationReport.model_validate_json(json_str)

        assert restored.short_id == "short_99"
        assert restored.overall_engagement_score == 0.82
        assert restored.detected_violations == [PacingViolationType.MONOTONOUS_FACE_LOCK]
        assert len(restored.remedial_actions) == 1
        assert restored.remedial_actions[0].action_type == "INSERT_TOPIC_CLOSEUP"
        assert restored.self_correction_applied is True


# ═════════════════════════════════════════════════════════════════════════════
# 5. CLEAN INPUT REJECTION (NO CRASHES, VALIDATIONERROR RAISED)
# ═════════════════════════════════════════════════════════════════════════════

class TestCleanInputRejection:
    """Stress tests rejection of invalid inputs, verifying clean pydantic.ValidationError without crashing."""

    def test_reject_invalid_category_in_grounding_entity(self):
        """Verifies invalid category string is rejected with ValidationError."""
        with pytest.raises(ValidationError):
            GroundingEntity(entity_name="test", category="INVALID_CAT")

    def test_reject_out_of_bounds_salience_score(self):
        """Verifies salience_score < 0.0 or > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            GroundingEntity(entity_name="test", category="PRODUCT", salience_score=-0.1)
        with pytest.raises(ValidationError):
            GroundingEntity(entity_name="test", category="PRODUCT", salience_score=1.1)

    def test_reject_invalid_target_type_in_visual_grounding(self):
        """Verifies invalid target_type is rejected."""
        with pytest.raises(ValidationError):
            VisualGroundingTarget(target_type="INVALID_TYPE", primary_bbox=(0.5, 0.5, 0.2, 0.2))

    def test_reject_out_of_bounds_confidence_in_visual_grounding(self):
        """Verifies confidence < 0.0 or > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            VisualGroundingTarget(
                target_type="PRODUCT",
                primary_bbox=(0.5, 0.5, 0.2, 0.2),
                confidence=-0.05,
            )
        with pytest.raises(ValidationError):
            VisualGroundingTarget(
                target_type="PRODUCT",
                primary_bbox=(0.5, 0.5, 0.2, 0.2),
                confidence=1.05,
            )

    def test_reject_invalid_anchor_mode(self):
        """Verifies invalid anchor_mode is rejected."""
        with pytest.raises(ValidationError):
            VisualGroundingTarget(
                target_type="PRODUCT",
                primary_bbox=(0.5, 0.5, 0.2, 0.2),
                anchor_mode="BOTTOM_ALIGNED",
            )

    def test_reject_malformed_bbox_tuple(self):
        """Verifies primary_bbox with incorrect length is rejected."""
        with pytest.raises(ValidationError):
            VisualGroundingTarget(
                target_type="PRODUCT",
                primary_bbox=(0.5, 0.5),  # 2 elements instead of 4
            )
        with pytest.raises(ValidationError):
            VisualGroundingTarget(
                target_type="PRODUCT",
                primary_bbox=(0.5, 0.5, 0.2, 0.2, 0.1),  # 5 elements
            )

    def test_reject_malformed_word_indices_tuple(self):
        """Verifies word_indices with incorrect length is rejected."""
        with pytest.raises(ValidationError):
            TranscriptBeat(
                beat_id="b1",
                start_time=1.0,
                end_time=2.0,
                summary="test",
                word_indices=(1,),  # 1 element instead of 2
            )

    def test_reject_invalid_cut_layout_mode_in_video_segment(self):
        """Verifies invalid layout mode string in VideoSegment is rejected."""
        with pytest.raises(ValidationError):
            VideoSegment(
                segment_start=60.0,
                segment_end=80.0,
                layout_mode="UNSUPPORTED_LAYOUT_MODE",
            )

    def test_reject_invalid_focal_target_in_video_segment(self):
        """Verifies invalid focal target string in VideoSegment is rejected."""
        with pytest.raises(ValidationError):
            VideoSegment(
                segment_start=60.0,
                segment_end=80.0,
                focal_target="NON_EXISTENT_TARGET",
            )

    def test_reject_short_candidate_out_of_bounds_values(self):
        """Verifies start < 60, duration < 30, duration > 50 are cleanly rejected."""
        with pytest.raises(ValidationError):
            ShortCandidate(start_time_seconds=45.0, duration_seconds=35.0)
        with pytest.raises(ValidationError):
            ShortCandidate(start_time_seconds=70.0, duration_seconds=20.0)
        with pytest.raises(ValidationError):
            ShortCandidate(start_time_seconds=70.0, duration_seconds=55.0)

    def test_reject_invalid_split_screen_pane_targets(self):
        """Verifies invalid pane targets in SplitScreenConfig are rejected."""
        with pytest.raises(ValidationError):
            SplitScreenConfig(top_pane_target="INVALID_PANE")
        with pytest.raises(ValidationError):
            SplitScreenConfig(bottom_pane_target="INVALID_PANE")

    def test_reject_invalid_remedial_action_type(self):
        """Verifies invalid remedial action type is rejected."""
        with pytest.raises(ValidationError):
            RemedialAction(
                segment_index=0,
                action_type="UNSUPPORTED_ACTION",
                reason="test",
            )

    def test_reject_out_of_bounds_engagement_score(self):
        """Verifies overall_engagement_score < 0.0 or > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            EngagementEvaluationReport(
                short_id="s1",
                overall_engagement_score=-0.1,
                multi_cut_pacing_passed=True,
                topic_grounding_passed=True,
                max_continuous_single_crop_duration=5.0,
                topic_in_viewport_ratio=0.8,
            )
        with pytest.raises(ValidationError):
            EngagementEvaluationReport(
                short_id="s1",
                overall_engagement_score=1.1,
                multi_cut_pacing_passed=True,
                topic_grounding_passed=True,
                max_continuous_single_crop_duration=5.0,
                topic_in_viewport_ratio=0.8,
            )
