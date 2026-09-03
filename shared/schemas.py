import uuid

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple, Annotated
import operator


# ─── Phase 1: Transcription Spoke Output ────────────────────────────────────

class WordTimestamp(BaseModel):
    word: str
    start: float = Field(..., description="Start time of the word in seconds")
    end: float = Field(..., description="End time of the word in seconds")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")


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


# ─── Phase 2: Fusion Consumer Output ────────────────────────────────────────

class FusedTimelineEntry(BaseModel):
    """A single timestamped entry in the unified text-annotated visual timeline."""
    timestamp: float
    word: Optional[str] = None
    word_end: Optional[float] = None
    word_confidence: Optional[float] = None
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
    layout_mode: Literal["speaker_full", "split_screen", "FULL_SCREEN", "B_ROLL", "broll", "graphic"] = Field(
        default="speaker_full",
        description="Render layout: 'speaker_full' = face-tracked full-canvas crop; "
                    "'split_screen' = 50/50 vertical split with B-roll in the lower half; 'broll'/'graphic' = full-canvas pillarbox"
    )
    focal_target: Literal[
        "SPEAKER_PRIMARY", "SPEAKER_REACTION", "FOCAL_DISPLAY",
        "HELD_OBJECT", "ACTION_SCENE"
    ] = Field(
        default="SPEAKER_PRIMARY",
        description="Script-grounded semantic framing target: determines which entity "
                    "the 9:16 crop anchors on. SPEAKER_PRIMARY = active speaker face/torso; "
                    "SPEAKER_REACTION = non-speaking listener; FOCAL_DISPLAY = on-screen UI/monitor/phone/tablet; "
                    "HELD_OBJECT = hands + held item with 1.30x macro zoom; "
                    "ACTION_SCENE = full saliency centroid for dynamic scenes."
    )
    camera_target: str = Field(
        default="SPEAKER_FACE",
        description="Legacy text-guided dynamic target for cinematic framing interpolation (e.g. SPEAKER_FACE, GLOBAL_SALIENCY, laptop, keyboard)"
    )
    emphasis_zoom: bool = Field(
        default=False,
        description="When True, the render spoke applies an additional 1.25x zoom "
                    "punch-in effect on this segment for maximum visual emphasis"
    )

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


# ─── Pipeline State (LangGraph root schema) ──────────────────────────────────

class PipelineState(BaseModel):
    source_path: str
    video_name_hash: str
    output_dir: str

    # Phase 1 raw metadata — Annotated reducers for LangGraph state merging
    transcript_words: Annotated[List[WordTimestamp], operator.add] = Field(default_factory=list)
    visual_hooks: Annotated[List[VisualHook], operator.add] = Field(default_factory=list)

    # Phase 2 fused metadata
    fused_timeline: Optional[FusedTimeline] = None

    # Debate loop state
    cutting_manifest: Optional[CuttingManifest] = None

    # System fields
    current_stage: str = Field(default="ingest")
    error_log: Annotated[List[str], operator.add] = Field(default_factory=list)
    pipeline_complete: bool = Field(default=False)
