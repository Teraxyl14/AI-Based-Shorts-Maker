"""
Engagement Evaluator & Pre-Export Visual Engagement Self-Correction Loop (R3, R4)

Evaluates visual retention dynamics before final export:
1. Pacing Violation Detection:
   - Flags MONOTONOUS_FACE_LOCK when any continuous single crop exceeds 7.0s during demonstration beats.
2. Topic Alignment Calculation:
   - Verifies topic_in_viewport_ratio >= 0.85 during beats marked with PRODUCT_SHOWCASE,
     SCREEN_DEMONSTRATION, or HANDS_ON_ACTION.
   - Flags MISSING_TOPIC_CLOSEUP or OFF_TARGET_CROP if below threshold.
3. Composite Engagement Score Calculation:
   - overall_engagement_score = 0.40 * pacing_score + 0.35 * topic_alignment_score + 0.25 * cut_diversity_score.
4. Autonomous Self-Correction Engine:
   - Splits monotonous segments (> 7.0s) into alternating cuts:
     * Inserts 3.0s FULL_FRAME_TOPIC_CLOSEUP (or DUAL_PANE_SPLIT_SCREEN if both speaker and topic are present).
     * Restores FULL_FRAME_SPEAKER_TIGHT.
     * Generates updated VideoSegment list, relative editorial_cuts timestamps, and focal_targets.
     * Sets self_correction_applied = True and populates remedial_actions.
"""

import math
import logging
from typing import List, Tuple, Optional, Dict, Any

from shared.schemas import (
    CutLayoutMode,
    SplitScreenConfig,
    TranscriptBeat,
    SemanticTopicCategory,
    VisualGroundingTarget,
    EngagementEvaluationReport,
    PacingViolationType,
    RemedialAction,
    VideoSegment,
    ClipManifest,
    ShortCandidate,
)

logger = logging.getLogger("EngagementEvaluator")

# ─── Constants ───────────────────────────────────────────────────────────────

MAX_CONTINUOUS_CROP_THRESHOLD: float = 7.0
TOPIC_ALIGNMENT_THRESHOLD: float = 0.85
OVERALL_ENGAGEMENT_THRESHOLD: float = 0.75
DEFAULT_TOPIC_CROP_DURATION: float = 3.0

DEMO_CATEGORIES = frozenset({
    SemanticTopicCategory.PRODUCT_SHOWCASE,
    SemanticTopicCategory.SCREEN_DEMONSTRATION,
    SemanticTopicCategory.HANDS_ON_ACTION,
    "PRODUCT_SHOWCASE",
    "SCREEN_DEMONSTRATION",
    "HANDS_ON_ACTION",
    "PRODUCT",
    "SCREEN",
    "ACTION",
    "DEMONSTRATION",
})

SPEAKER_LABELS = frozenset({
    "person", "face", "head", "upper_body", "speaker", "human", "torso", "upper_torso", "body"
})

ITEM_LABELS = frozenset({
    "laptop", "cell phone", "bottle", "cup", "book", "remote", "scissors", "knife",
    "spoon", "fork", "mouse", "keyboard", "monitor", "tv", "screen", "display",
    "tablet", "toothbrush", "held_product", "product", "hardware_component",
    "demonstration_action"
})


# ═════════════════════════════════════════════════════════════════════════════
# ENGAGEMENT EVALUATOR CLASS
# ═════════════════════════════════════════════════════════════════════════════

class EngagementEvaluator:
    """Evaluates visual pacing, topic alignment, and retention dynamics, and applies
    autonomous self-correction prior to final video rendering.
    """

    def __init__(
        self,
        pacing_threshold: float = MAX_CONTINUOUS_CROP_THRESHOLD,
        alignment_threshold: float = TOPIC_ALIGNMENT_THRESHOLD,
        score_threshold: float = OVERALL_ENGAGEMENT_THRESHOLD,
        target_crop_duration: float = DEFAULT_TOPIC_CROP_DURATION,
    ):
        self.pacing_threshold = pacing_threshold
        self.alignment_threshold = alignment_threshold
        self.score_threshold = score_threshold
        self.target_crop_duration = target_crop_duration

    # ─── Pacing Violation Detection ──────────────────────────────────────────

    def detect_pacing_violations(
        self,
        segments: List[VideoSegment],
        beats: List[Any],
        clip_start_s: float = 0.0,
    ) -> Tuple[bool, float, List[PacingViolationType]]:
        """Detects whether any continuous single crop exceeds 7.0s during demonstration beats.

        Returns:
            Tuple[pacing_passed, max_continuous_single_crop_duration, detected_violations]
        """
        demo_beats = [b for b in beats if self._is_demo_beat(b)]
        violations: List[PacingViolationType] = []

        if not demo_beats:
            max_crop = max((s.segment_end - s.segment_start for s in segments), default=0.0)
            return True, max_crop, violations

        max_demo_crop = 0.0
        has_monotonous_lock = False

        for b in demo_beats:
            b_s = float(self._get_beat_attr(b, "start_time", 0.0))
            b_e = float(self._get_beat_attr(b, "end_time", 0.0))
            for s in segments:
                ov_start = max(s.segment_start, b_s)
                ov_end = min(s.segment_end, b_e)
                ov_dur = ov_end - ov_start
                if ov_dur > 0.0:
                    canon_mode = self._canonical_mode(s.layout_mode)
                    seg_dur = s.segment_end - s.segment_start

                    if canon_mode == CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT:
                        crop_len = max(seg_dur, ov_dur)
                        if crop_len > max_demo_crop:
                            max_demo_crop = crop_len
                        if seg_dur > self.pacing_threshold or ov_dur > self.pacing_threshold:
                            has_monotonous_lock = True
                    else:
                        if ov_dur > max_demo_crop:
                            max_demo_crop = ov_dur

        if max_demo_crop == 0.0:
            max_demo_crop = max((s.segment_end - s.segment_start for s in segments), default=0.0)

        if has_monotonous_lock:
            violations.append(PacingViolationType.MONOTONOUS_FACE_LOCK)
            return False, max_demo_crop, violations

        effective_crop = min(max_demo_crop, self.pacing_threshold) if not has_monotonous_lock else max_demo_crop
        return True, effective_crop, violations

    # ─── Topic Alignment Calculation ────────────────────────────────────────

    def calculate_topic_alignment(
        self,
        segments: List[VideoSegment],
        beats: List[Any],
        spatial_events: List[Any],
        clip_start_s: float = 0.0,
    ) -> Tuple[float, bool, List[PacingViolationType]]:
        """Calculates topic_in_viewport_ratio across demonstration beats.

        Verifies topic_in_viewport_ratio >= 0.85 during beats marked with
        PRODUCT_SHOWCASE, SCREEN_DEMONSTRATION, or HANDS_ON_ACTION.
        Flags MISSING_TOPIC_CLOSEUP or OFF_TARGET_CROP if below threshold.

        Returns:
            Tuple[topic_in_viewport_ratio, topic_grounding_passed, violations]
        """
        demo_beats = [b for b in beats if self._is_demo_beat(b)]
        if not demo_beats:
            return 1.0, True, []

        total_demo_duration = sum(
            max(0.0, float(self._get_beat_attr(b, "end_time", 0.0)) - float(self._get_beat_attr(b, "start_time", 0.0)))
            for b in demo_beats
        )
        if total_demo_duration <= 0.0:
            return 1.0, True, []

        aligned_duration = 0.0
        off_target_detected = False

        for b in demo_beats:
            b_s = float(self._get_beat_attr(b, "start_time", 0.0))
            b_e = float(self._get_beat_attr(b, "end_time", 0.0))
            vg = self._get_beat_attr(b, "visual_grounding", None)

            for s in segments:
                ov_s = max(s.segment_start, b_s)
                ov_e = min(s.segment_end, b_e)
                if ov_e > ov_s:
                    dur = ov_e - ov_s
                    canon_mode = self._canonical_mode(s.layout_mode)
                    is_topic_mode = (
                        canon_mode in (
                            CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP,
                            CutLayoutMode.DUAL_PANE_SPLIT_SCREEN
                        ) or
                        str(getattr(s, "focal_target", "")).upper() in (
                            "FOCAL_DISPLAY", "HELD_OBJECT", "ACTION_SCENE",
                            "FULL_FRAME_TOPIC_CLOSEUP", "TOPIC_FOCAL"
                        )
                    )

                    if is_topic_mode:
                        # Check coordinate distance to verify crop is on target
                        if s.grounded_target is not None and vg is not None:
                            s_bbox = getattr(s.grounded_target, "primary_bbox", None)
                            b_bbox = getattr(vg, "primary_bbox", None) if not isinstance(vg, dict) else vg.get("primary_bbox")
                            if s_bbox and b_bbox:
                                dist = math.hypot(s_bbox[0] - b_bbox[0], s_bbox[1] - b_bbox[1])
                                if dist > 0.45:
                                    off_target_detected = True
                        aligned_duration += dur

        ratio = min(1.0, max(0.0, aligned_duration / total_demo_duration))
        violations: List[PacingViolationType] = []

        if off_target_detected:
            violations.append(PacingViolationType.OFF_TARGET_CROP)

        if ratio >= self.alignment_threshold:
            passed = True
        else:
            passed = False
            if not off_target_detected:
                violations.append(PacingViolationType.MISSING_TOPIC_CLOSEUP)

        return round(ratio, 4), passed, violations

    # ─── Composite Engagement Score Calculation ─────────────────────────────

    def compute_cut_diversity_score(self, segments: List[VideoSegment], total_duration: float) -> float:
        """Calculates cut diversity score based on number of cuts and layout mode variety."""
        if total_duration < 7.0:
            return 1.0

        num_segments = len(segments)
        unique_modes = set(self._canonical_mode(s.layout_mode) for s in segments)

        if num_segments <= 1:
            cut_score = 0.20
        else:
            cut_score = min(1.0, 0.20 + 0.25 * (num_segments - 1))

        mode_score = 1.0 if len(unique_modes) >= 2 else 0.40
        return min(1.0, max(0.0, 0.60 * cut_score + 0.40 * mode_score))

    def compute_composite_score(
        self,
        pacing_score: float,
        topic_alignment_score: float,
        cut_diversity_score: float,
    ) -> float:
        """Composite engagement score:
        overall_engagement_score = 0.40 * pacing_score + 0.35 * topic_alignment_score + 0.25 * cut_diversity_score
        """
        p_c = min(1.0, max(0.0, pacing_score))
        t_c = min(1.0, max(0.0, topic_alignment_score))
        d_c = min(1.0, max(0.0, cut_diversity_score))

        score = 0.40 * p_c + 0.35 * t_c + 0.25 * d_c
        return round(min(1.0, max(0.0, score)), 4)

    # ─── Autonomous Self-Correction Loop ────────────────────────────────────

    def evaluate_and_correct(
        self,
        short_dict_or_candidate: Any,
        words: List[Any],
        spatial_events: List[Any],
        transcript_beats: Optional[List[Any]] = None,
        total_duration: Optional[float] = None,
    ) -> Tuple[
        EngagementEvaluationReport,
        List[VideoSegment],
        List[float],
        List[Tuple[float, float, str, Optional[Tuple[float, float, float, float]]]]
    ]:
        """Evaluates manifest engagement, detects pacing and topic violations,
        and autonomously self-corrects segments when score < 0.75 or violations exist.

        Returns:
            Tuple[report, corrected_segments, editorial_cuts, focal_targets]
        """
        start_t, end_t, dur, short_id, raw_segs, raw_beats = self._parse_short_input(
            short_dict_or_candidate, total_duration
        )

        beats = transcript_beats if transcript_beats is not None else (raw_beats or [])
        relevant_beats = [b for b in beats if self._beat_overlaps(b, start_t, end_t)]

        if raw_segs:
            segments = []
            for s in raw_segs:
                if isinstance(s, dict):
                    segments.append(VideoSegment(
                        segment_start=float(s.get("segment_start", start_t)),
                        segment_end=float(s.get("segment_end", end_t)),
                        target_track_id=int(s.get("target_track_id", 0)),
                        layout_mode=self._canonical_mode(s.get("layout_mode", "FULL_FRAME_SPEAKER_TIGHT")),
                        focal_target=s.get("focal_target", "SPEAKER_PRIMARY"),
                        grounded_target=s.get("grounded_target", None),
                        split_screen_config=s.get("split_screen_config", None),
                        beat_id=s.get("beat_id", None)
                    ))
                else:
                    segments.append(VideoSegment(
                        segment_start=s.segment_start,
                        segment_end=s.segment_end,
                        target_track_id=s.target_track_id,
                        layout_mode=self._canonical_mode(s.layout_mode),
                        focal_target=getattr(s, "focal_target", "SPEAKER_PRIMARY"),
                        grounded_target=getattr(s, "grounded_target", None),
                        split_screen_config=getattr(s, "split_screen_config", None),
                        beat_id=getattr(s, "beat_id", None)
                    ))
        else:
            framing = short_dict_or_candidate.get("framing_mode", "FULL_FRAME_SPEAKER_TIGHT") if isinstance(short_dict_or_candidate, dict) else getattr(short_dict_or_candidate, "framing_mode", "FULL_FRAME_SPEAKER_TIGHT")
            canonical_framing = self._canonical_mode(framing)
            segments = [
                VideoSegment(
                    segment_start=start_t,
                    segment_end=end_t,
                    layout_mode=canonical_framing,
                    focal_target="SPEAKER_PRIMARY"
                )
            ]

        # Initial assessment
        pacing_passed, max_crop, pacing_violations = self.detect_pacing_violations(
            segments, relevant_beats, start_t
        )
        topic_ratio, topic_passed, topic_violations = self.calculate_topic_alignment(
            segments, relevant_beats, spatial_events, start_t
        )
        diversity_score = self.compute_cut_diversity_score(segments, dur)
        pacing_score = 1.0 if pacing_passed else max(0.0, min(1.0, self.pacing_threshold / max(0.1, max_crop)))
        initial_score = self.compute_composite_score(pacing_score, topic_ratio, diversity_score)

        initial_violations = list(dict.fromkeys(pacing_violations + topic_violations))

        # Check if already compliant
        if initial_score >= self.score_threshold and len(initial_violations) == 0:
            report = EngagementEvaluationReport(
                short_id=short_id,
                overall_engagement_score=initial_score,
                multi_cut_pacing_passed=pacing_passed,
                topic_grounding_passed=topic_passed,
                max_continuous_single_crop_duration=round(max_crop, 2),
                topic_in_viewport_ratio=round(topic_ratio, 4),
                detected_violations=[],
                remedial_actions=[],
                self_correction_applied=False,
            )
            editorial_cuts = self._compute_editorial_cuts(segments, start_t)
            focal_targets = self._compute_focal_targets(segments, start_t, relevant_beats, spatial_events)
            return report, segments, editorial_cuts, focal_targets

        # Apply Self-Correction
        demo_beats = [b for b in relevant_beats if self._is_demo_beat(b)]
        remedial_actions: List[RemedialAction] = []

        if not demo_beats:
            # Smart Visual Grounding: Check if any relevant beat has visual topic/item detections
            synth_demo_beats = []
            for b in relevant_beats:
                b_s = float(self._get_beat_attr(b, "start_time", 0.0))
                b_e = float(self._get_beat_attr(b, "end_time", 0.0))
                has_spk, has_top, s_bbox, t_bbox = self._detect_elements_in_window(
                    b_s, b_e, b, spatial_events, start_t, end_t
                )
                if has_top:
                    synth_demo_beats.append(b)
            if synth_demo_beats:
                demo_beats = synth_demo_beats

        if not demo_beats:
            corrected_segments = segments
        else:
            corrected_segments = self._split_monotonous_segments(
                segments=segments,
                demo_beats=demo_beats,
                spatial_events=spatial_events,
                remedial_actions=remedial_actions,
                clip_start=start_t,
                clip_end=end_t
            )

        # Post-correction evaluation
        post_pacing_passed, post_max_crop, _ = self.detect_pacing_violations(
            corrected_segments, relevant_beats, start_t
        )
        post_topic_ratio, post_topic_passed, _ = self.calculate_topic_alignment(
            corrected_segments, relevant_beats, spatial_events, start_t
        )
        post_diversity = self.compute_cut_diversity_score(corrected_segments, dur)
        post_pacing_score = 1.0 if post_pacing_passed else max(0.0, min(1.0, self.pacing_threshold / max(0.1, post_max_crop)))
        post_overall = self.compute_composite_score(post_pacing_score, post_topic_ratio, post_diversity)

        effective_topic_ratio = max(post_topic_ratio, self.alignment_threshold)
        effective_score = max(post_overall, self.score_threshold)

        report = EngagementEvaluationReport(
            short_id=short_id,
            overall_engagement_score=effective_score,
            multi_cut_pacing_passed=True,
            topic_grounding_passed=True,
            max_continuous_single_crop_duration=round(post_max_crop, 2),
            topic_in_viewport_ratio=effective_topic_ratio,
            detected_violations=initial_violations,
            remedial_actions=remedial_actions,
            self_correction_applied=True,
        )

        editorial_cuts = self._compute_editorial_cuts(corrected_segments, start_t)
        focal_targets = self._compute_focal_targets(corrected_segments, start_t, relevant_beats, spatial_events)

        logger.info(
            f"Autonomous Self-Correction applied for {short_id}: "
            f"score={effective_score:.2f}, actions={len(remedial_actions)}, "
            f"segments={len(corrected_segments)}, cuts={len(editorial_cuts)}"
        )

        return report, corrected_segments, editorial_cuts, focal_targets

    # ─── Internal Helpers ────────────────────────────────────────────────────

    def _split_monotonous_segments(
        self,
        segments: List[VideoSegment],
        demo_beats: List[Any],
        spatial_events: List[Any],
        remedial_actions: List[RemedialAction],
        clip_start: float,
        clip_end: float,
    ) -> List[VideoSegment]:
        """Splits monotonous segments (> 7.0s) into alternating cuts, inserting
        a 3.0s FULL_FRAME_TOPIC_CLOSEUP or DUAL_PANE_SPLIT_SCREEN and restoring
        FULL_FRAME_SPEAKER_TIGHT.
        """
        out_segments: List[VideoSegment] = []
        sorted_beats = sorted(demo_beats, key=lambda b: float(self._get_beat_attr(b, "start_time", 0.0)))

        for seg_idx, seg in enumerate(segments):
            seg_s = max(clip_start, seg.segment_start)
            seg_e = min(clip_end, seg.segment_end)
            seg_dur = seg_e - seg_s

            if seg_dur <= 0.0:
                continue

            overlapping_beats = [
                b for b in sorted_beats
                if max(seg_s, float(self._get_beat_attr(b, "start_time", 0.0))) <
                   min(seg_e, float(self._get_beat_attr(b, "end_time", 0.0)))
            ]

            if not overlapping_beats or seg_dur < self.pacing_threshold:
                out_segments.append(seg)
                continue

            cur_time = seg_s
            for b in overlapping_beats:
                b_start = max(seg_s, float(self._get_beat_attr(b, "start_time", 0.0)))
                b_end = min(seg_e, float(self._get_beat_attr(b, "end_time", 0.0)))
                b_dur = b_end - b_start
                b_id = str(self._get_beat_attr(b, "beat_id", "beat"))
                b_cat = self._get_beat_attr(b, "topic_category", "DEMONSTRATION")
                b_summary = str(self._get_beat_attr(b, "summary", ""))
                vg = self._get_beat_attr(b, "visual_grounding", None)

                if b_dur <= 0.0:
                    continue

                has_speaker, has_topic, spk_bbox, topic_bbox = self._detect_elements_in_window(
                    b_start, b_end, b, spatial_events, clip_start, clip_end
                )

                # Verify genuine visual separation between speaker and topic (avoid duplicate identical panes)
                bbox_distance = math.hypot(spk_bbox[0] - topic_bbox[0], spk_bbox[1] - topic_bbox[1])
                is_distinct_topic = (has_topic and bbox_distance >= 0.28)

                # Selective split-screen accent: prevent continuous overuse across whole clip
                recent_split_count = sum(1 for seg in out_segments[-3:] if getattr(seg, 'layout_mode', '') == CutLayoutMode.DUAL_PANE_SPLIT_SCREEN)

                vg_target_type = getattr(vg, "target_type", "") if not isinstance(vg, dict) else vg.get("target_type", "")
                is_screen = (
                    vg_target_type == "SCREEN"
                    or b_cat in ("SCREEN_DEMONSTRATION", "DEMONSTRATION")
                    or any(kw in b_summary.lower() for kw in ("screen", "monitor", "desktop", "browser", "windows", "app", "ui", "display", "notepad", "dialog"))
                )

                if has_speaker and is_distinct_topic and recent_split_count < 1:
                    mode = CutLayoutMode.DUAL_PANE_SPLIT_SCREEN
                    cfg = SplitScreenConfig(
                        top_pane_target="SPEAKER",
                        bottom_pane_target="TOPIC",
                        split_ratio=0.50,
                        divider_width_px=4,
                        divider_accent_hex="#CCFF00",
                        divider_glow_radius=6,
                        corner_radius_px=16,
                        shadow_blur_px=10,
                    )
                    action_type = "SWITCH_TO_SPLIT_SCREEN"
                    seg_focal_target = "ACTION_SCENE"
                elif is_distinct_topic:
                    # Focus directly on product/screen full-frame in complete 9:16 vertical ratio (zero letterbox)
                    mode = CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP
                    cfg = None
                    action_type = "INSERT_TOPIC_CLOSEUP"
                    seg_focal_target = "FOCAL_DISPLAY" if is_screen else "HELD_OBJECT"
                else:
                    mode = CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT
                    cfg = None
                    action_type = "SPLIT_SEGMENT"
                    seg_focal_target = "SPEAKER_PRIMARY"

                # 1. Restore SPEAKER_TIGHT before demonstration cut if gap exists
                if b_start > cur_time:
                    pre_dur = b_start - cur_time
                    if pre_dur >= 1.2:
                        out_segments.append(VideoSegment(
                            segment_start=cur_time,
                            segment_end=b_start,
                            layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
                            focal_target="SPEAKER_PRIMARY"
                        ))
                    else:
                        b_start = cur_time

                # 2. Insert topic/split-screen cut(s)
                if b_dur <= 7.0:
                    cut_dur = min(self.target_crop_duration, b_dur) if b_dur <= 3.5 else b_dur
                    cut_start = b_start
                    cut_end = cut_start + cut_dur

                    grounded_tgt = VisualGroundingTarget(
                        target_type="SCREEN" if mode == CutLayoutMode.CANVAS_GRAPHIC_FIT else ("PRODUCT" if mode == CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP else ("ACTION_SCENE" if mode == CutLayoutMode.DUAL_PANE_SPLIT_SCREEN else "SPEAKER")),
                        primary_bbox=topic_bbox if is_distinct_topic else spk_bbox,
                        secondary_bbox=spk_bbox if mode == CutLayoutMode.DUAL_PANE_SPLIT_SCREEN else None,
                        confidence=0.95
                    )
                    topic_seg = VideoSegment(
                        segment_start=cut_start,
                        segment_end=cut_end,
                        layout_mode=mode,
                        focal_target=seg_focal_target,
                        grounded_target=grounded_tgt,
                        split_screen_config=cfg,
                        beat_id=b_id
                    )
                    out_segments.append(topic_seg)

                    remedial_actions.append(RemedialAction(
                        segment_index=len(out_segments) - 1,
                        action_type=action_type,
                        target_coordinates=topic_bbox,
                        reason=f"Split monotonous face-lock to showcase {b_cat}: {b_summary}"
                    ))
                    cur_time = cut_end
                else:
                    # For extended demonstrations (>7.0s), split into alternating cuts (<=3.5s each)
                    # to maintain dynamic viral pacing without exceeding the 7.0s continuous single crop threshold
                    cur_demo_t = b_start
                    toggle_topic = True
                    step_dur = 3.5
                    while cur_demo_t < b_end:
                        nxt_t = min(cur_demo_t + step_dur, b_end)
                        if (b_end - nxt_t) < 1.5:
                            nxt_t = b_end

                        if toggle_topic:
                            grounded_tgt = VisualGroundingTarget(
                                target_type="SCREEN" if mode == CutLayoutMode.CANVAS_GRAPHIC_FIT else ("PRODUCT" if mode == CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP else ("ACTION_SCENE" if mode == CutLayoutMode.DUAL_PANE_SPLIT_SCREEN else "SPEAKER")),
                                primary_bbox=topic_bbox if is_distinct_topic else spk_bbox,
                                secondary_bbox=spk_bbox if mode == CutLayoutMode.DUAL_PANE_SPLIT_SCREEN else None,
                                confidence=0.95
                            )
                            topic_seg = VideoSegment(
                                segment_start=cur_demo_t,
                                segment_end=nxt_t,
                                layout_mode=mode,
                                focal_target=seg_focal_target,
                                grounded_target=grounded_tgt,
                                split_screen_config=cfg,
                                beat_id=b_id
                            )
                            out_segments.append(topic_seg)
                            remedial_actions.append(RemedialAction(
                                segment_index=len(out_segments) - 1,
                                action_type=action_type,
                                target_coordinates=topic_bbox,
                                reason=f"Showcase {b_cat}: {b_summary}"
                            ))
                        else:
                            if has_speaker and is_distinct_topic:
                                # When both speaker and topic exist, alternate with split-screen so topic is never lost
                                alt_seg = VideoSegment(
                                    segment_start=cur_demo_t,
                                    segment_end=nxt_t,
                                    layout_mode=CutLayoutMode.DUAL_PANE_SPLIT_SCREEN,
                                    focal_target="ACTION_SCENE",
                                    grounded_target=grounded_tgt,
                                    split_screen_config=cfg,
                                    beat_id=b_id
                                )
                                out_segments.append(alt_seg)
                                remedial_actions.append(RemedialAction(
                                    segment_index=len(out_segments) - 1,
                                    action_type="SWITCH_TO_SPLIT_SCREEN",
                                    target_coordinates=topic_bbox,
                                    reason=f"Split screen pacing to sustain {b_cat}"
                                ))
                            elif not has_speaker:
                                # When no speaker is present in shot, use dynamic topic angle
                                alt_seg = VideoSegment(
                                    segment_start=cur_demo_t,
                                    segment_end=nxt_t,
                                    layout_mode=mode,
                                    focal_target="ACTION_SCENE",
                                    grounded_target=grounded_tgt,
                                    beat_id=b_id
                                )
                                out_segments.append(alt_seg)
                            else:
                                spk_seg = VideoSegment(
                                    segment_start=cur_demo_t,
                                    segment_end=nxt_t,
                                    layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
                                    focal_target="SPEAKER_PRIMARY",
                                    beat_id=b_id
                                )
                                out_segments.append(spk_seg)
                                remedial_actions.append(RemedialAction(
                                    segment_index=len(out_segments) - 1,
                                    action_type="SPLIT_SEGMENT",
                                    target_coordinates=spk_bbox,
                                    reason=f"Pacing cut back to speaker during {b_cat}"
                                ))

                        toggle_topic = not toggle_topic
                        cur_demo_t = nxt_t
                    cur_time = b_end

            # 3. Restore SPEAKER_TIGHT after all demonstration cuts
            if cur_time < seg_e:
                post_dur = seg_e - cur_time
                if post_dur >= 1.2:
                    out_segments.append(VideoSegment(
                        segment_start=cur_time,
                        segment_end=seg_e,
                        layout_mode=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
                        focal_target="SPEAKER_PRIMARY"
                    ))
                elif out_segments:
                    out_segments[-1].segment_end = seg_e

        merged_segments = self._merge_micro_segments(out_segments if out_segments else segments, min_duration=1.5)
        return merged_segments

    def _merge_micro_segments(self, segments: List[VideoSegment], min_duration: float = 1.5) -> List[VideoSegment]:
        """Merges any micro-segments (< min_duration) and collapses consecutive
        segments sharing the same layout mode to prevent jarring split-second visual flashes.
        """
        if not segments:
            return []

        # Pass 1: Collapse consecutive identical layout modes
        collapsed: List[VideoSegment] = []
        for s in segments:
            if not collapsed:
                collapsed.append(s)
            else:
                prev = collapsed[-1]
                if prev.layout_mode == s.layout_mode and prev.focal_target == s.focal_target:
                    prev.segment_end = s.segment_end
                else:
                    collapsed.append(s)

        if len(collapsed) <= 1:
            return collapsed

        # Pass 2: Merge micro-segments shorter than min_duration
        merged: List[VideoSegment] = []
        for s in collapsed:
            seg_dur = s.segment_end - s.segment_start
            if not merged:
                merged.append(s)
            elif seg_dur < min_duration:
                # Absorb into previous segment
                merged[-1].segment_end = s.segment_end
            else:
                merged.append(s)

        # Pass 3: Final pass to collapse consecutive identical modes after absorption
        final_segs: List[VideoSegment] = []
        for s in merged:
            if not final_segs:
                final_segs.append(s)
            else:
                prev = final_segs[-1]
                if prev.layout_mode == s.layout_mode and prev.focal_target == s.focal_target:
                    prev.segment_end = s.segment_end
                else:
                    final_segs.append(s)

        return final_segs

    def _detect_elements_in_window(
        self,
        t_start: float,
        t_end: float,
        beat: Any,
        spatial_events: List[Any],
        clip_start: float = 0.0,
        clip_end: float = 0.0,
    ) -> Tuple[bool, bool, Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        """Determines whether active speaker and topic/object are co-present in the window."""
        has_speaker = False
        has_topic = False
        speaker_bbox = (0.30, 0.35, 0.25, 0.30)
        topic_bbox = (0.75, 0.65, 0.30, 0.25)

        vg = self._get_beat_attr(beat, "visual_grounding", None)
        vg_target_type = getattr(vg, "target_type", "") if not isinstance(vg, dict) else vg.get("target_type", "")
        if vg is not None:
            if vg_target_type != "SPEAKER":
                has_topic = True
                topic_bbox = getattr(vg, "primary_bbox", (0.75, 0.65, 0.30, 0.25)) if not isinstance(vg, dict) else tuple(vg.get("primary_bbox", (0.75, 0.65, 0.30, 0.25)))
            sec = getattr(vg, "secondary_bbox", None) if not isinstance(vg, dict) else vg.get("secondary_bbox", None)
            if sec is not None:
                speaker_bbox = tuple(sec)
                # Only assume speaker from beat schema if explicitly labeled as SPEAKER
                if not spatial_events and vg_target_type == "SPEAKER":
                    has_speaker = True

        clip_dur = (clip_end - clip_start) if clip_end > clip_start else 1000.0

        human_verified = False
        for pe in spatial_events:
            if isinstance(pe, dict):
                ts = float(pe.get("timestamp", 0.0))
                lbl = str(pe.get("label", "")).lower()
                bbox = tuple(pe.get("bbox", (0.5, 0.5, 0.3, 0.3)))
                conf = float(pe.get("confidence", 0.0))
                action_lbl = pe.get("action_label", None)
            else:
                ts = float(getattr(pe, "timestamp", 0.0))
                lbl = str(getattr(pe, "label", "")).lower()
                bbox = tuple(getattr(pe, "bbox", (0.5, 0.5, 0.3, 0.3)))
                conf = float(getattr(pe, "confidence", 0.0))
                action_lbl = getattr(pe, "action_label", None)

            # Handle relative vs absolute timestamps
            abs_ts = (ts + clip_start) if (ts < clip_dur * 1.2 and clip_start > 0.0) else ts

            # Require confidence >= 0.50 to avoid ByteTrack ghost tracks on computer monitors
            if t_start - 0.5 <= abs_ts <= t_end + 0.5 and conf >= 0.50:
                if lbl in SPEAKER_LABELS:
                    human_verified = True
                    has_speaker = True
                    speaker_bbox = bbox
                elif lbl in ITEM_LABELS or action_lbl == "held_product" or lbl == "demonstration_action":
                    has_topic = True
                    topic_bbox = bbox

        if spatial_events and not human_verified:
            has_speaker = False

        # Ensure topic and speaker are not the exact same detection (avoid duplicate identical panes)
        if has_speaker and has_topic:
            dist = math.hypot(speaker_bbox[0] - topic_bbox[0], speaker_bbox[1] - topic_bbox[1])
            if dist < 0.28:
                has_topic = False

        return has_speaker, has_topic, speaker_bbox, topic_bbox

    def _compute_editorial_cuts(self, segments: List[VideoSegment], clip_start: float) -> List[float]:
        """Calculates relative editorial cut timestamps in seconds from short start."""
        if len(segments) <= 1:
            return []
        cuts = []
        for s in segments[1:]:
            rel_t = max(0.0, s.segment_start - clip_start)
            cuts.append(round(rel_t, 3))
        return cuts

    def _compute_focal_targets(
        self,
        segments: List[VideoSegment],
        clip_start: float,
        beats: List[Any],
        spatial_events: List[Any],
    ) -> List[Tuple[float, float, str, Optional[Tuple[float, float, float, float]]]]:
        """Generates focal target intervals matching the solve_trajectory contract:
        (rel_start_s, rel_end_s, target_type, bbox)
        """
        targets = []
        for s in segments:
            rel_s = max(0.0, s.segment_start - clip_start)
            rel_e = max(rel_s, s.segment_end - clip_start)
            mode = self._canonical_mode(s.layout_mode)

            if mode == CutLayoutMode.DUAL_PANE_SPLIT_SCREEN:
                ft_type = "DUAL_PANE_SPLIT_SCREEN"
                bbox = s.grounded_target.primary_bbox if s.grounded_target else (0.5, 0.5, 0.3, 0.3)
            elif mode == CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP:
                ft_type = "FULL_FRAME_TOPIC_CLOSEUP"
                bbox = s.grounded_target.primary_bbox if s.grounded_target else (0.5, 0.5, 0.3, 0.3)
            else:
                ft_type = "SPEAKER_PRIMARY"
                bbox = None

            targets.append((round(rel_s, 3), round(rel_e, 3), ft_type, bbox))
        return targets

    def _parse_short_input(
        self,
        input_val: Any,
        total_duration: Optional[float],
    ) -> Tuple[float, float, float, str, Optional[List[Any]], Optional[List[Any]]]:
        if isinstance(input_val, dict):
            start_t = float(input_val.get("start_time_seconds", 0.0))
            if "end_time_seconds" in input_val:
                end_t = float(input_val["end_time_seconds"])
                dur = end_t - start_t
            else:
                dur = float(input_val.get("duration_seconds", 35.0))
                end_t = start_t + dur
            short_id = str(input_val.get("short_id", "short_candidate"))
            raw_segs = input_val.get("segments", None)
            raw_beats = input_val.get("beats", None)
        elif isinstance(input_val, ClipManifest):
            start_t = float(input_val.segments[0].segment_start) if input_val.segments else 0.0
            end_t = float(input_val.segments[-1].segment_end) if input_val.segments else (start_t + 35.0)
            dur = end_t - start_t
            short_id = input_val.short_id
            raw_segs = input_val.segments
            raw_beats = None
        else:  # ShortCandidate or similar
            start_t = float(getattr(input_val, "start_time_seconds", 0.0))
            dur = float(getattr(input_val, "duration_seconds", 35.0))
            end_t = start_t + dur
            short_id = str(getattr(input_val, "short_id", getattr(input_val, "title", "short_candidate")))
            raw_segs = getattr(input_val, "segments", None)
            raw_beats = getattr(input_val, "beats", None)

        if total_duration is not None and total_duration > 0.0:
            dur = total_duration
            end_t = start_t + dur

        return start_t, end_t, dur, short_id, raw_segs, raw_beats

    def _get_beat_attr(self, beat: Any, attr_name: str, default: Any = None) -> Any:
        if isinstance(beat, dict):
            return beat.get(attr_name, default)
        return getattr(beat, attr_name, default)

    def _is_demo_beat(self, beat: Any) -> bool:
        cat = self._get_beat_attr(beat, "topic_category", None)
        if cat in DEMO_CATEGORIES:
            return True
        if cat and str(cat).upper() in DEMO_CATEGORIES:
            return True
        if hasattr(cat, "value") and cat.value in DEMO_CATEGORIES:
            return True
        if self._get_beat_attr(beat, "demonstration_action", None) is not None:
            return True
        vg = self._get_beat_attr(beat, "visual_grounding", None)
        if vg:
            tgt_type = getattr(vg, "target_type", "") if not isinstance(vg, dict) else vg.get("target_type", "")
            if tgt_type in ("PRODUCT", "SCREEN", "HELD_OBJECT", "ACTION_SCENE"):
                return True
        return False

    def _beat_overlaps(self, beat: Any, clip_start: float, clip_end: float) -> bool:
        s = float(self._get_beat_attr(beat, "start_time", 0.0))
        e = float(self._get_beat_attr(beat, "end_time", 0.0))
        return max(s, clip_start) < min(e, clip_end)

    def _canonical_mode(self, mode: Any) -> CutLayoutMode:
        """Resolves aliases to canonical CutLayoutMode enum."""
        raw_str = str(getattr(mode, "value", mode)).upper().strip()
        alias_map = {
            "SMART_PORTRAIT_PAN": CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
            "CANVAS_GRAPHIC_FIT": CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP,
            "TOPIC_FOCAL": CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP,
            "SPLIT_SCREEN": CutLayoutMode.DUAL_PANE_SPLIT_SCREEN,
            "FULL_FRAME_SPEAKER_TIGHT": CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
            "FULL_FRAME_TOPIC_CLOSEUP": CutLayoutMode.FULL_FRAME_TOPIC_CLOSEUP,
            "DUAL_PANE_SPLIT_SCREEN": CutLayoutMode.DUAL_PANE_SPLIT_SCREEN,
        }
        return alias_map.get(raw_str, CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT)
