import uuid
import math
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Tuple, Annotated, Dict, Any, Union
import operator



# ─── Phase 1: Transcription Spoke Output ────────────────────────────────────

class WordTimestamp(BaseModel):
    word: str
    start: float = Field(..., description="Start time of the word in seconds")
    end: float = Field(..., description="End time of the word in seconds")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    speaker_id: Optional[str] = Field(default=None, description="Speaker identifier e.g. SPEAKER_00, SPEAKER_01")


class SpeakerTurn(BaseModel):
    """Contiguous speech segment attributed to a single speaker."""
    speaker_id: str = Field(..., description="Speaker identifier e.g. SPEAKER_00, SPEAKER_01")
    start: float = Field(..., description="Start time of the speaker turn in seconds")
    end: float = Field(..., description="End time of the speaker turn in seconds")
    confidence: float = Field(default=1.0, description="Diarization confidence score between 0.0 and 1.0")
    words: List[WordTimestamp] = Field(default_factory=list, description="Word timestamps contained within this speaker turn")
    text: Optional[str] = Field(default=None, description="Spoken text within this turn")

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class AudioTranscriptionResult(BaseModel):
    """Complete audio stage result contract bridging ASR/Diarization to Narrative Arc Engine."""
    source_path: str = Field(..., description="Path to source audio or video file")
    detected_language: str = Field(default="auto", description="Auto-detected native language code (e.g., 'en', 'hi')")
    duration: float = Field(..., description="Total audio duration in seconds")
    words: List[WordTimestamp] = Field(default_factory=list, description="Word timestamps with speaker attribution")
    speaker_turns: List[SpeakerTurn] = Field(default_factory=list, description="Segmented speaker turns")

    def __iter__(self):
        """Allows direct iteration over words for backward compatibility with list interfaces."""
        return iter(self.words)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        return self.words[idx]



# ─── Phase 1: Vision Pre-Pass Output ────────────────────────────────────────

class SceneChange(BaseModel):
    """Kornia-detected scene boundary event."""
    timestamp: float = Field(..., description="Time of the scene change in seconds")
    confidence: float = Field(..., description="Scene change confidence 0.0 to 1.0")
    frame_idx: int = Field(..., description="Frame index in the source video")
    change_type: str = Field(default="hard_cut", description="hard_cut, dissolve, or fade")

class SpatialEvent(BaseModel):
    """YOLOE-26 / YOLO11s-Pose / Saliency detection event."""
    timestamp: float = Field(..., description="Time of the detection in seconds")
    frame_idx: int = Field(..., description="Frame index in the source video")
    track_id: int = Field(default=0, description="Tracked object ID within the frame")
    label: str = Field(default="person", description="Object class label (e.g., person, laptop, saliency)")
    bbox: Tuple[float, float, float, float] = Field(..., description="Bounding box (cx, cy, w, h) normalized 0-1")
    confidence: float = Field(default=1.0, description="Confidence score")
    keypoints: List[Tuple[float, float, float]] = Field(
        default_factory=list,
        description="Optional 17-point COCO skeleton if label is person"
    )
    action_label: Optional[str] = Field(default=None, description="Detected gesture or action")


# ─── Legacy VisualHook (kept for backward compat in LangGraph state) ────────

class VisualHook(BaseModel):
    frame_idx: int
    timestamp: float
    hook_type: str = Field(..., description="e.g. face, gesture, transition, visual_climax")
    description: str
    arousal_score: float = Field(..., description="Emotional/engagement arousal score between 0.0 and 1.0")


# ─── Topic-to-Visual Grounding Schemas (Milestone 6.1 / R1) ─────────────────

class UniversalContentProfile(str, Enum):
    """Universal video content profile used for adaptive narrative and visual framing."""
    STAGE_PERFORMANCE = "STAGE_PERFORMANCE"       # Standup comedy, solo theatrical acts, keynotes
    TECH_REVIEW_DEMO = "TECH_REVIEW_DEMO"         # Tech presenter, hardware, monitors, screencasts, gadgets
    GAMEPLAY_STREAM = "GAMEPLAY_STREAM"           # Facecam + gameplay HUD/action
    WILDLIFE_NATURE = "WILDLIFE_NATURE"           # Animal movement, landscapes, non-human subjects
    CONVERSATIONAL_PODCAST = "CONVERSATIONAL_PODCAST" # 2+ seated speakers, turn-taking dialogue
    CINEMATIC_NARRATIVE = "CINEMATIC_NARRATIVE"   # Multi-character films, dramatic action, general movies


class SemanticTopicCategory(str, Enum):
    SPEAKER_HOOK = "SPEAKER_HOOK"
    DIALOGUE_CONTEXT = "DIALOGUE_CONTEXT"
    PRODUCT_SHOWCASE = "PRODUCT_SHOWCASE"
    SCREEN_DEMONSTRATION = "SCREEN_DEMONSTRATION"
    HANDS_ON_ACTION = "HANDS_ON_ACTION"
    EXPLANATION_PUNCHLINE = "EXPLANATION_PUNCHLINE"


class GroundingEntity(BaseModel):
    """Specific entity, product, or UI component identified in a transcript beat."""
    entity_name: str = Field(..., description="Entity name e.g. 'Framework Laptop 16', 'GPU Module', 'Terminal UI'")
    category: Literal["PRODUCT", "SCREEN", "HARDWARE_COMPONENT", "DOCUMENT", "PERSON", "CONCEPT"]
    visual_aliases: List[str] = Field(default_factory=list, description="COCO/YOLO class labels e.g. ['laptop', 'keyboard', 'cell phone']")
    salience_score: float = Field(default=1.0, ge=0.0, le=1.0)


class VisualGroundingTarget(BaseModel):
    """Dynamic spatial coordinates of the discussed topic or focal element."""
    target_type: Literal["SPEAKER", "PRODUCT", "SCREEN", "HELD_OBJECT", "ACTION_SCENE"]
    primary_bbox: Tuple[float, float, float, float] = Field(
        ..., description="(cx, cy, w, h) normalized 0.0 to 1.0"
    )
    secondary_bbox: Optional[Tuple[float, float, float, float]] = Field(
        default=None, description="Optional secondary bbox for dual-element shots (e.g. speaker + screen)"
    )
    track_id: Optional[int] = Field(default=None, description="ByteTrack ID if applicable")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    anchor_mode: Literal["CENTER", "TOP_ALIGNED", "HAND_REGION"] = Field(default="CENTER")


class TranscriptBeat(BaseModel):
    """Narrative beat linking transcript slice to semantic topic and visual grounding target."""
    beat_id: str = Field(..., description="Unique beat ID e.g. 'beat_01'")
    start_time: float = Field(..., description="Beat start timestamp in seconds")
    end_time: float = Field(..., description="Beat end timestamp in seconds")
    topic_category: SemanticTopicCategory = Field(default=SemanticTopicCategory.DIALOGUE_CONTEXT)
    summary: str = Field(..., description="1-sentence description of topic/action")
    entities: List[GroundingEntity] = Field(default_factory=list)
    demonstration_action: Optional[str] = Field(default=None, description="e.g. 'holding_device', 'pointing_screen', 'unscrewing'")
    word_indices: Tuple[int, int] = Field(..., description="(start_word_idx, end_word_idx) in transcript")
    visual_grounding: Optional[VisualGroundingTarget] = Field(default=None)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


# ─── Multi-Cut Framing & Split-Screen Schemas (Milestone 6.1 / R2) ──────────

class CutLayoutMode(str, Enum):
    FULL_FRAME_SPEAKER_TIGHT = "FULL_FRAME_SPEAKER_TIGHT"
    FULL_FRAME_TOPIC_CLOSEUP = "FULL_FRAME_TOPIC_CLOSEUP"
    DUAL_PANE_SPLIT_SCREEN = "DUAL_PANE_SPLIT_SCREEN"

    # Backward compatibility aliases
    SMART_PORTRAIT_PAN = "SMART_PORTRAIT_PAN"
    CANVAS_GRAPHIC_FIT = "CANVAS_GRAPHIC_FIT"
    TOPIC_FOCAL = "TOPIC_FOCAL"
    SPLIT_SCREEN = "SPLIT_SCREEN"

    def _canonical(self, val: str) -> str:
        return self._CANONICAL_MAP.get(val, val)

    def __eq__(self, other):
        if isinstance(other, str):
            val1 = self.value
            val2 = getattr(other, "value", other)
            return self._canonical(val1) == self._canonical(val2)
        return super().__eq__(other)

    def __hash__(self):
        return hash(self._canonical(self.value))


CutLayoutMode._CANONICAL_MAP = {
    "SMART_PORTRAIT_PAN": "FULL_FRAME_SPEAKER_TIGHT",
    "CANVAS_GRAPHIC_FIT": "FULL_FRAME_TOPIC_CLOSEUP",
    "TOPIC_FOCAL": "FULL_FRAME_TOPIC_CLOSEUP",
    "SPLIT_SCREEN": "DUAL_PANE_SPLIT_SCREEN",
}


class SplitScreenConfig(BaseModel):
    """Configuration for styled dual-pane vertical split-screen."""
    top_pane_target: Literal["SPEAKER", "TOPIC", "SCREEN"] = Field(default="SPEAKER")
    bottom_pane_target: Literal["SPEAKER", "TOPIC", "SCREEN"] = Field(default="TOPIC")
    split_ratio: float = Field(default=0.50, description="Height fraction of top pane (0.40 to 0.60)")
    divider_width_px: int = Field(default=4, description="Width of glowing neon accent divider line (3-4px)")
    divider_accent_hex: str = Field(default="#CCFF00", description="Neon accent hex matching kinetic subtitle glow")
    divider_glow_radius: int = Field(default=6, description="Gaussian glow radius in pixels")
    corner_radius_px: int = Field(default=16, description="Rounded inner corner radius for panes")
    shadow_blur_px: int = Field(default=10, description="Drop shadow blur for pane separation")


class PacingViolationType(str, Enum):
    MONOTONOUS_FACE_LOCK = "MONOTONOUS_FACE_LOCK"
    MISSING_TOPIC_CLOSEUP = "MISSING_TOPIC_CLOSEUP"
    RAPID_JITTER_CUTS = "RAPID_JITTER_CUTS"
    OFF_TARGET_CROP = "OFF_TARGET_CROP"


class RemedialAction(BaseModel):
    segment_index: int
    action_type: Literal[
        "INSERT_TOPIC_CLOSEUP", "SWITCH_TO_SPLIT_SCREEN", "SHIFT_CROP_CENTER",
        "SPLIT_SEGMENT", "INSERT_GRAPHIC_FIT", "CANVAS_UNCERTAINTY_FIT"
    ]
    target_coordinates: Optional[Tuple[float, float, float, float]] = None
    reason: str


class EngagementEvaluationReport(BaseModel):
    short_id: str
    overall_engagement_score: float = Field(..., ge=0.0, le=1.0)
    multi_cut_pacing_passed: bool
    topic_grounding_passed: bool
    max_continuous_single_crop_duration: float
    topic_in_viewport_ratio: float
    detected_violations: List[PacingViolationType] = Field(default_factory=list)
    remedial_actions: List[RemedialAction] = Field(default_factory=list)
    self_correction_applied: bool = False


# ─── Phase 2: Fusion Consumer Output ────────────────────────────────────────

class FusedTimelineEntry(BaseModel):
    """A single timestamped entry in the unified text-annotated visual timeline."""
    timestamp: float
    word: Optional[str] = None
    word_end: Optional[float] = None
    word_confidence: Optional[float] = None
    speaker_id: Optional[str] = None
    scene_changes_nearby: List[SceneChange] = Field(default_factory=list)
    spatial_events_nearby: List[SpatialEvent] = Field(default_factory=list)
    hook_analysis: Optional[str] = Field(
        default=None,
        description="LLM-generated hook assessment for this timeline segment"
    )
    arousal_score: float = Field(default=0.5, description="Engagement score 0.0 to 1.0")

class FusedTimeline(BaseModel):
    """Complete merged audio-visual timeline for a single video."""
    source_path: str
    video_duration: float = Field(..., description="Total video duration in seconds")
    entries: List[FusedTimelineEntry] = Field(default_factory=list)
    scene_changes: List[SceneChange] = Field(default_factory=list)
    spatial_events: List[SpatialEvent] = Field(default_factory=list)
    transcript_words: List[WordTimestamp] = Field(default_factory=list)
    speaker_turns: List[SpeakerTurn] = Field(default_factory=list)
    transcript_beats: List[TranscriptBeat] = Field(
        default_factory=list,
        description="Extracted narrative & demonstration beats with visual grounding"
    )



# ─── Debate Graph Models ────────────────────────────────────────────────────

class VideoSegment(BaseModel):
    """A discrete, non-contiguous block of video defined by start and end timestamps.

    Segments are the atomic units of the multi-cut assembly engine.
    Each segment maps to a contiguous slice of the source video timeline,
    but segments within a clip are NOT required to be contiguous — they
    form jump-cuts when stitched.
    """
    segment_start: float
    segment_end: float
    target_track_id: int = Field(default=0, description="Speaker or track identifier")
    layout_mode: CutLayoutMode = Field(
        default=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
        description="Universal Layout Mode: "
                    "'FULL_FRAME_SPEAKER_TIGHT' (SMART_PORTRAIT_PAN) = Full-height 9:16 vertical crop (ch=1.0, cw=607.5px) "
                    "with OSQP-optimized horizontal pan tracking the active speaker. Zero digital zoom. "
                    "'FULL_FRAME_TOPIC_CLOSEUP' (CANVAS_GRAPHIC_FIT) = Tight 9:16 focal crop centered "
                    "on the demonstrated subject or screen. Zero letterboxing. "
                    "Legacy canvas mode: uncropped 16:9 foreground (1080x608) centered vertically "
                    "at Y=656 over a 40%-darkened Gaussian-blurred (σ=10, k=21) background. "
                    "'DUAL_PANE_SPLIT_SCREEN' = Styled vertical split with neon divider."
    )
    focal_target: Literal[
        "SPEAKER_PRIMARY", "SPEAKER_REACTION", "FOCAL_DISPLAY",
        "HELD_OBJECT", "ACTION_SCENE"
    ] = Field(
        default="SPEAKER_PRIMARY",
        description="Script-grounded semantic framing target: determines which entity "
                    "the 9:16 crop anchors on. SPEAKER_PRIMARY = active speaker face/torso; "
                    "SPEAKER_REACTION = non-speaking listener; FOCAL_DISPLAY = on-screen UI/monitor/phone/tablet; "
                    "HELD_OBJECT = hands + held item; "
                    "ACTION_SCENE = full saliency centroid for dynamic scenes."
    )
    grounded_target: Optional[VisualGroundingTarget] = Field(
        default=None, description="Pre-computed exact screen crop coordinates for this segment"
    )
    split_screen_config: Optional[SplitScreenConfig] = Field(
        default=None, description="Populated only if layout_mode == DUAL_PANE_SPLIT_SCREEN"
    )
    beat_id: Optional[str] = Field(default=None, description="Referenced TranscriptBeat ID")

class ClipManifest(BaseModel):
    """Manifest for a compiled short, composed of multiple non-contiguous stitched segments."""
    short_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique short identifier")
    title: str = Field(default="", description="Hook label / title for the short")
    segments: List[VideoSegment]
    hook_score: float = Field(..., description="Calculated hook score 0.0 to 1.0")
    retention_score: float = Field(..., description="Predicted user retention score 0.0 to 1.0")
    cta_present: bool = Field(default=False, description="Whether Call To Action is present/suggested at the end")
    reasoning: str = Field(..., description="Detailed narrative reason for selecting this clip")
    caption_text: Optional[str] = Field(default=None, description="Transcript text slice for this clip segment, used for caption generation")
    editor_version: int = Field(default=1, description="Version of the editor that created this clip")
    director_critique: Optional[str] = Field(default=None, description="Critique from the Director for this clip")

class CuttingManifest(BaseModel):
    clips: List[ClipManifest] = Field(default_factory=list)
    debate_iterations: int = Field(default=0, description="Number of debate rounds completed")
    consensus_reached: bool = Field(default=False)
    content_profile: Optional[UniversalContentProfile] = Field(default=None, description="Global classified video genre/profile")


# ─── VLM Editorial Engine Output ────────────────────────────────────────────

class ShortCandidate(BaseModel):
    """Output schema for the shot-midpoint VLM editorial engine.

    Enforces hard constraints:
        - start_time_seconds >= 60.0 (reject intro teasers)
        - duration_seconds in [30.0, 50.0] for single-clip, up to 120.0s for multi-segment
        - segments: optional multi-segment list for non-linear storytelling
        - framing_mode: CutLayoutMode or legacy aliases
    """
    segments: Optional[List[Tuple[float, float]]] = Field(default=None, description="Disjoint non-linear segments [(s1, e1), (s2, e2), ...] composing this short")
    start_time_seconds: float = Field(..., description="Start time must be >= 60s to reject intro teasers")
    duration_seconds: float = Field(..., description="Short duration must be 30-50s for single cuts, or >= 50s for multi-segment")
    title: str = Field(default="", description="Hook title for the short")
    reasoning: str = Field(default="", description="Narrative arc reasoning (Hook → Escalation → Payoff)")
    framing_mode: CutLayoutMode = Field(
        default=CutLayoutMode.FULL_FRAME_SPEAKER_TIGHT,
        description="Per-short framing mode based on visual content analysis"
    )
    hook_window: Tuple[float, float] = Field(default=(0.0, 5.0))
    first_word_delay: Optional[float] = Field(default=None)
    payoff_snapped: bool = Field(default=False)
    beats: List[TranscriptBeat] = Field(default_factory=list, description="Associated transcript beats")
    content_profile: Optional[UniversalContentProfile] = Field(default=None, description="Classified video profile for this short")

    @field_validator("start_time_seconds")
    @classmethod
    def validate_start_time(cls, v: float) -> float:
        if math.isnan(v) or v < 60.0:
            raise ValueError("Input should be greater than or equal to 60")
        return v

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, v: float, info) -> float:
        if math.isnan(v) or v < 30.0:
            raise ValueError("Input should be greater than or equal to 30")
        has_segments = bool(info.data.get("segments")) if info.data else False
        if not has_segments and v > 50.0:
            raise ValueError("Input should be less than or equal to 50")
        if has_segments and v > 120.0:
            raise ValueError("Input should be less than or equal to 120")
        return v

    @property
    def end_time_seconds(self) -> float:
        """Returns the calculated end timestamp of the short candidate."""
        return round(self.start_time_seconds + self.duration_seconds, 2)



# ─── Render Job ──────────────────────────────────────────────────────────────

class RenderJob(BaseModel):
    source_path: str
    output_dir: str
    clip_manifest: ClipManifest
    clip_index: int = Field(default=1, description="1-based index for output filename (short_1.mp4, short_2.mp4, ...)")
    transcript_words: List[dict] = Field(default_factory=list, description="Word timestamps for the full video, used to generate captions")
    target_resolution: Tuple[int, int] = Field(default=(1080, 1920), description="(width, height)")
    target_fps: int = Field(default=30)
    codec: str = Field(default="av1_nvenc")
    spatial_events: List[dict] = Field(default_factory=list, description="YOLOE-26 spatial events for active tracking")
    crop_path_data: List[List[float]] = Field(default_factory=list, description="Pre-computed [N, 2] crop center path (normalized cx, cy per frame)")
    scene_cut_frames: List[int] = Field(default_factory=list, description="Frame indices of detected scene cuts for trajectory teleports")
    bgm_path: Optional[str] = Field(default=None, description="Optional path to background music track for auto-ducked mixing")


# ─── Pipeline State (LangGraph root schema) ──────────────────────────────────

class PipelineState(BaseModel):
    source_path: str
    video_name_hash: str
    output_dir: str
    content_profile: Optional[UniversalContentProfile] = None

    # Phase 1 raw metadata — Annotated reducers for LangGraph state merging
    transcript_words: Annotated[List[WordTimestamp], operator.add] = Field(default_factory=list)
    speaker_turns: Annotated[List[SpeakerTurn], operator.add] = Field(default_factory=list)
    visual_hooks: Annotated[List[VisualHook], operator.add] = Field(default_factory=list)

    # Phase 2 fused metadata
    fused_timeline: Optional[FusedTimeline] = None

    # Debate loop state
    cutting_manifest: Optional[CuttingManifest] = None

    # System fields
    current_stage: str = Field(default="ingest")
    error_log: Annotated[List[str], operator.add] = Field(default_factory=list)
    pipeline_complete: bool = Field(default=False)
