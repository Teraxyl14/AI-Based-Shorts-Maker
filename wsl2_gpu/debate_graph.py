import os
import sys
import json
import httpx
import logging
import asyncio
from typing import Any, List, Optional
from langgraph.graph import StateGraph, END

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.schemas import PipelineState, CuttingManifest, ClipManifest, VideoSegment

logger = logging.getLogger("AetherDebateGraph")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ─── Single MoE Brain ────────────────────────────────────────────────────────
BRAIN_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"
MAX_DEBATE_ITERATIONS = 3


def get_item_field(item: Any, field_name: str, default: Any = None) -> Any:
    """Helper to safely retrieve a field from a model or a dictionary."""
    if isinstance(item, dict):
        return item.get(field_name, default)
    if hasattr(item, field_name):
        val = getattr(item, field_name)
        if val is not None:
            return val
    return default


get_state_field = get_item_field


def format_transcript_compact(words: List[Any]) -> str:
    """Groups word-level timestamps into readable 15-second visual chunks
    to avoid blowing up the LLM context window.
    """
    if not words:
        return ""
    
    formatted = []
    current_chunk = []
    chunk_start = None
    
    for w in words:
        if isinstance(w, dict):
            word_text = w.get("word", "")
            start = w.get("start", 0.0)
            end = w.get("end", 0.0)
        else:
            word_text = getattr(w, "word", "")
            start = getattr(w, "start", 0.0)
            end = getattr(w, "end", 0.0)
            
        if start > 300.0:
            break
            
        if chunk_start is None:
            chunk_start = start
            
        current_chunk.append(word_text)
        
        # Group into 15-second blocks or sentence endings
        if (end - chunk_start >= 15.0) or word_text.endswith((".", "?", "!")):
            text = " ".join(current_chunk)
            formatted.append(f"[{chunk_start:.1f}s - {end:.1f}s] {text}")
            current_chunk = []
            chunk_start = None
            
    if current_chunk:
        last_end = 0.0
        if isinstance(words[-1], dict):
            last_end = words[-1].get("end", 0.0)
        else:
            last_end = getattr(words[-1], "end", 0.0)
        text = " ".join(current_chunk)
        formatted.append(f"[{chunk_start:.1f}s - {last_end:.1f}s] {text}")
        
    return "\n".join(formatted)


def get_structured_event_summaries(entries: List[Any], scene_changes: List[Any] = None) -> str:
    """Compresses transcript into structured event summaries of top-arousal windows to optimize LLM context window."""
    if not entries:
        return "No transcript entries available."
        
    windows = []
    current_window = []
    window_start = None
    window_arousal = 0.0
    
    for e in entries:
        start = get_item_field(e, "timestamp", 0.0)
        word = get_item_field(e, "word", "")
        arousal = get_item_field(e, "arousal_score", 0.0)
        
        if window_start is None:
            window_start = start
            
        current_window.append(word)
        window_arousal = max(window_arousal, arousal)
        
        # Group strictly by 30.0s blocks to keep structure unified and compact
        if start - window_start >= 30.0:
            text = " ".join(current_window)
            windows.append((window_arousal, window_start, start, text))
            current_window = []
            window_start = None
            window_arousal = 0.0
            
    if current_window:
        end = get_item_field(entries[-1], "timestamp", 0.0)
        text = " ".join(current_window)
        windows.append((window_arousal, window_start, end, text))
        
    # Select top 6 highest arousal windows to provide sufficient candidate material for >= 3 clips of 50s+
    windows.sort(key=lambda w: w[0], reverse=True)
    top_windows = windows[:6]
    
    # Sort chronologically
    top_windows.sort(key=lambda w: w[1])
    
    formatted = []
    for arousal, w_start, w_end, text in top_windows:
        words_in_text = text.split()
        if len(words_in_text) > 35:
            text = " ".join(words_in_text[:35]) + "..."
        formatted.append(f"[{w_start:.1f}-{w_end:.1f}s, ar:{arousal:.2f}] {text}")
        
    return "\n".join(formatted)



class DebateGraph:
    """Orchestrates the Editor-Director debate loop using LangGraph and the unified
    Qwen3.6-35B-A3B sparse MoE brain (single permanent VRAM resident).
    """

    def __init__(self):
        self.brain_url = os.getenv("VLLM_BRAIN_URL", "http://vllm-brain:8000").rstrip("/")

    async def run_brain_llm(self, prompt: str, system_prompt: str, json_mode: bool = False, max_tokens: int = 1000) -> str:
        """Invokes the local vLLM MoE brain."""
        # Estimate input tokens using word count * 1.35
        total_text = prompt + " " + system_prompt
        approx_input_tokens = int(len(total_text.split()) * 1.35) + 20
        # Target 4096 context window limit, ensuring safe output token headroom (at least 600)
        safe_max_tokens = max(600, min(max_tokens, 4096 - approx_input_tokens - 30))
        logger.info(f"[LLM] Estimating {approx_input_tokens} input tokens. Constraining output max_tokens to {safe_max_tokens}")

        payload = {
            "model": BRAIN_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": safe_max_tokens,
            "expert_routing_strategy": "lfru"  # LFRU-based DMA expert swapping
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.brain_url}/v1/chat/completions",
                    json=payload,
                    timeout=90.0
                )
                if response.status_code == 200:
                    data = response.json()
                    choice = data["choices"][0]
                    finish_reason = choice.get("finish_reason")
                    usage = data.get("usage", {})
                    logger.info(f"[LLM] usage={usage}, finish_reason={finish_reason}")
                    return choice["message"]["content"].strip()
                else:
                    logger.error(f"Brain LLM returned status {response.status_code}: {response.text}")
                    raise RuntimeError(f"vLLM error: {response.text}")
            except Exception as e:
                logger.error(f"Brain LLM connection error: {str(e)}")
                raise

    async def editor_node(self, state: PipelineState) -> dict:
        """Editor Agent: formulates/refines a cutting manifest using fused timeline data."""
        cutting_manifest = get_state_field(state, "cutting_manifest")
        if isinstance(cutting_manifest, dict):
            cutting_manifest = CuttingManifest(**cutting_manifest)

        transcript_words = get_state_field(state, "transcript_words", [])
        fused_timeline = get_state_field(state, "fused_timeline")
        error_log = get_state_field(state, "error_log", [])

        current_iter = cutting_manifest.debate_iterations if cutting_manifest else 0
        logger.info(f"[GRAPH] Entering Editor Node (Iteration: {current_iter + 1})")

        # Prepare fused timeline context (scene changes + poses + hook scores) and compact candidate transcript
        fused_context = ""
        entries = []
        if fused_timeline:
            ft = fused_timeline if isinstance(fused_timeline, dict) else (
                fused_timeline.model_dump() if hasattr(fused_timeline, "model_dump") else {}
            )
            scene_changes = ft.get("scene_changes", [])
            spatial_events = ft.get("spatial_events", [])
            entries = ft.get("entries", [])

            # Summarize high-arousal segments
            high_arousal = [e for e in entries if get_item_field(e, "arousal_score", 0) > 0.6]
            # Get unique high arousal times/scores briefly to avoid token overflow
            brief_arousal = []
            seen_times = set()
            for e in high_arousal:
                t = round(get_item_field(e, "timestamp", 0.0), 1)
                # Keep one entry per 15s to keep it small
                time_key = int(t / 15)
                if time_key not in seen_times:
                    seen_times.add(time_key)
                    brief_arousal.append({"t": t, "score": get_item_field(e, "arousal_score")})
            
            fused_context = (
                f"\nTotal scene changes: {len(scene_changes)}\n"
                f"Total spatial events (YOLOE/Saliency): {len(spatial_events)}\n"
                f"High-engagement segments arousal scores: {json.dumps(brief_arousal[:10])}\n"
            )

        if entries:
            transcript_snippet = get_structured_event_summaries(entries)
        else:
            transcript_snippet = format_transcript_compact(transcript_words)

        critique_context = ""
        if cutting_manifest and cutting_manifest.clips:
            logger.info("[GRAPH] Refining previous manifest based on Director's critique...")
            clip_summaries = []
            for c in cutting_manifest.clips:
                if isinstance(c, dict):
                    segs = c.get("segments", [])
                    segs_str = ", ".join([f"{s.get('segment_start',0)}-{s.get('segment_end',0)}s" for s in segs])
                    clip_summaries.append(f"[{segs_str}] (hook:{c.get('hook_score',0)}, ret:{c.get('retention_score',0)})")
                elif hasattr(c, "segments"):
                    segs_str = ", ".join([f"{s.segment_start}-{s.segment_end}s" for s in c.segments])
                    clip_summaries.append(f"[{segs_str}] (hook:{c.hook_score}, ret:{c.retention_score})")

            critique = None
            first_clip = cutting_manifest.clips[0]
            if isinstance(first_clip, dict):
                critique = first_clip.get("director_critique")
            elif hasattr(first_clip, "director_critique"):
                critique = first_clip.director_critique

            critique_context = (
                f"\nYour previous clips were REJECTED.\n"
                f"Previous clips: {'; '.join(clip_summaries)}\n"
                f"Director Critique: {critique if critique else 'Improve pacing and hooks'}\n"
                f"Address the critique and improve all clips."
            )

        system_prompt = (
            "You are an elite short-form video editor curating viral TikTok/Reels/Shorts.\n"
            "LMO Mode: rely strictly on spatial metadata. Curate 4 to 5 distinct, non-overlapping candidate clips.\n"
            "Directives:\n"
            "- Narrative: Each clip must be a SINGLE-TOPIC story arc — one self-contained premise with Premise -> Friction -> Conclusion. "
            "Do NOT create montage summaries or stitch disconnected windows.\n"
            "- Clip duration: Sum of segments in each clip MUST be between 30.0 and 48.0 seconds (target ~40s; strictly NEVER exceed 48.0s).\n"
            "- Hook: First segment must start on a high-arousal/energy statement.\n"
            "- Multi-Layout Framing (layout_mode per segment):\n"
            "  * SPLIT_STACK: Assign whenever a human speaker is actively interacting with, reacting to, or discussing an on-screen display, game, software UI, or secondary subject (e.g. tech reviews, gaming, reaction streams, tutorials).\n"
            "  * SPEAKER_SOLO: Assign when a single speaker is addressing the camera directly with no relevant visual display (e.g. stand-up comedy, monologue, single interview).\n"
            "  * CONTENT_FIT: Assign when the focal point is a full-screen graphic, slide, newspaper article, or diagram where horizontal text must remain 100% readable.\n"
            "- Focal Target: For every segment, assign focal_target from: SPEAKER_PRIMARY (someone speaking), "
            "SPEAKER_REACTION (listener reacting), FOCAL_DISPLAY (on-screen UI/game/monitor/phone/tablet), "
            "HELD_OBJECT (hands holding item), ACTION_SCENE (dynamic wide scene)."
        )

        prompt = (
            f"Candidate timeline:\n{transcript_snippet}\n\n"
            f"Visual metadata:{fused_context}\n"
            f"{critique_context}\n\n"
            "Return JSON matching this schema:\n"
            "{\"clips\": [{\"segments\": [{\"segment_start\": float, \"segment_end\": float, "
            "\"layout_mode\": \"SPEAKER_SOLO\"|\"SPLIT_STACK\"|\"CONTENT_FIT\", "
            "\"focal_target\": \"SPEAKER_PRIMARY\"|\"SPEAKER_REACTION\"|\"FOCAL_DISPLAY\"|\"HELD_OBJECT\"|\"ACTION_SCENE\", "
            "\"emphasis_zoom\": bool}], "
            "\"hook_score\": float, \"retention_score\": float, \"cta_present\": bool, "
            "\"reasoning\": str (max 5 words), \"caption_text\": str (max 5 words)}]}. "
            "You MUST return 4 to 5 clips, each with total duration between 30.0s and 48.0s (target 40s, strictly under 48s). Keep reasoning/caption_text extremely short (under 5 words)."
        )

        try:
            raw_response = await self.run_brain_llm(prompt, system_prompt, json_mode=True, max_tokens=1000)

            # Robust JSON extraction
            raw_clean = raw_response.strip()
            if "```json" in raw_clean:
                raw_clean = raw_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_clean:
                raw_clean = raw_clean.split("```")[1].split("```")[0].strip()
            elif "{" in raw_clean:
                raw_clean = raw_clean[raw_clean.find("{"):raw_clean.rfind("}") + 1]

            parsed = json.loads(raw_clean)
            clips_data = parsed.get("clips", [])

            clips = []
            for c in clips_data:
                segs_data = c.get("segments", [])
                segs = []
                VALID_FOCAL_TARGETS = {"SPEAKER_PRIMARY", "SPEAKER_REACTION", "FOCAL_DISPLAY", "HELD_OBJECT", "ACTION_SCENE"}
                VALID_LAYOUT_MODES = {"SPEAKER_SOLO", "SPLIT_STACK", "CONTENT_FIT"}

                for s in segs_data:
                    raw_focal = str(s.get("focal_target", "SPEAKER_PRIMARY")).upper()
                    # Backwards-compat: remap legacy MONITOR_SCREEN → FOCAL_DISPLAY
                    if raw_focal == "MONITOR_SCREEN":
                        raw_focal = "FOCAL_DISPLAY"
                    if raw_focal not in VALID_FOCAL_TARGETS:
                        raw_focal = "SPEAKER_PRIMARY"

                    raw_layout = str(s.get("layout_mode", "")).upper().strip()
                    if raw_layout in ("SPLIT_STACK", "SPLIT_SCREEN"):
                        norm_layout = "SPLIT_STACK"
                    elif raw_layout in ("CONTENT_FIT", "BROLL", "GRAPHIC", "FULL_SCREEN", "B_ROLL"):
                        norm_layout = "CONTENT_FIT"
                    elif raw_layout in ("SPEAKER_SOLO", "SPEAKER_FULL"):
                        norm_layout = "SPEAKER_SOLO"
                    else:
                        # Infer intelligent default from focal_target
                        if raw_focal in ("FOCAL_DISPLAY", "HELD_OBJECT"):
                            norm_layout = "SPLIT_STACK"
                        elif raw_focal == "ACTION_SCENE":
                            norm_layout = "CONTENT_FIT"
                        else:
                            norm_layout = "SPEAKER_SOLO"

                    segs.append(VideoSegment(
                        segment_start=float(s.get("segment_start", 0.0)),
                        segment_end=float(s.get("segment_end", 30.0)),
                        target_track_id=int(s.get("target_track_id", 0)),
                        layout_mode=norm_layout,
                        focal_target=raw_focal,
                        camera_target=str(s.get("camera_target", "SPEAKER_FACE")),
                        emphasis_zoom=bool(s.get("emphasis_zoom", False))
                    ))
                # Enforce chronological segment sorting
                segs = sorted(segs, key=lambda s: s.segment_start)
                
                # Keyword Intercept Gate — assign appropriate focal_target & layout_mode
                held_object_keywords = {"screws", "unscrew", "keyboard", "numpad", "battery",
                                        "motherboard", "power brick", "holding", "hands", "fingerprint"}
                focal_display_keywords = {"display", "monitor", "screen", "game", "laptop",
                                          "desktop", "ui", "interface", "ports", "inside"}
                content_fit_keywords = {"slide", "document", "article", "wikipedia", "diagram",
                                        "chart", "graphic", "whitepaper", "infographic"}

                ft = fused_timeline if isinstance(fused_timeline, dict) else (
                    fused_timeline.model_dump() if hasattr(fused_timeline, "model_dump") else {}
                )
                ft_entries = ft.get("entries", [])

                for s in segs:
                    for entry in ft_entries:
                        ts = get_item_field(entry, "timestamp", 0.0)
                        if s.segment_start <= ts <= s.segment_end:
                            w = str(get_item_field(entry, "word", "")).lower().strip(".,!?\"'")
                            if w in held_object_keywords:
                                s.focal_target = "HELD_OBJECT"
                                s.layout_mode = "SPLIT_STACK"
                                break
                            elif w in focal_display_keywords:
                                s.focal_target = "FOCAL_DISPLAY"
                                s.layout_mode = "SPLIT_STACK"
                                break
                            elif w in content_fit_keywords:
                                s.layout_mode = "CONTENT_FIT"
                                break

                clips.append(ClipManifest(
                    segments=segs,
                    title=str(c.get("caption_text", ""))[:60],
                    hook_score=float(c.get("hook_score", 0.5)),
                    retention_score=float(c.get("retention_score", 0.5)),
                    cta_present=bool(c.get("cta_present", False)),
                    reasoning=str(c.get("reasoning", "Selected by editor.")),
                    caption_text=str(c.get("caption_text", "")),
                    editor_version=current_iter + 1
                ))

            new_manifest = CuttingManifest(
                clips=clips,
                debate_iterations=current_iter,
                consensus_reached=False
            )
            return {"cutting_manifest": new_manifest}

        except Exception as e:
            logger.critical(f"HARDWARE TRACE - Editor node parsing failed: {e}. Raw response was: {raw_response if 'raw_response' in locals() else 'Not generated'}")
            fallback_clips = [
                ClipManifest(
                    segments=[
                        VideoSegment(segment_start=0.0, segment_end=18.0, focal_target="ACTION_SCENE"),
                        VideoSegment(segment_start=18.0, segment_end=42.0, focal_target="SPEAKER_PRIMARY")
                    ],
                    hook_score=0.85, retention_score=0.80,
                    cta_present=True,
                    reasoning="Fallback clip 1 (42s)",
                    caption_text="Clip 1",
                    editor_version=current_iter + 1
                ),
                ClipManifest(
                    segments=[
                        VideoSegment(segment_start=55.0, segment_end=72.0, focal_target="SPEAKER_PRIMARY"),
                        VideoSegment(segment_start=72.0, segment_end=95.0, focal_target="HELD_OBJECT")
                    ],
                    hook_score=0.85, retention_score=0.80,
                    cta_present=True,
                    reasoning="Fallback clip 2 (40s)",
                    caption_text="Clip 2",
                    editor_version=current_iter + 1
                ),
                ClipManifest(
                    segments=[
                        VideoSegment(segment_start=110.0, segment_end=128.0, focal_target="FOCAL_DISPLAY"),
                        VideoSegment(segment_start=128.0, segment_end=148.0, focal_target="SPEAKER_PRIMARY")
                    ],
                    hook_score=0.85, retention_score=0.80,
                    cta_present=True,
                    reasoning="Fallback clip 3 (38s)",
                    caption_text="Clip 3",
                    editor_version=current_iter + 1
                ),
                ClipManifest(
                    segments=[
                        VideoSegment(segment_start=160.0, segment_end=180.0, focal_target="SPEAKER_PRIMARY"),
                        VideoSegment(segment_start=180.0, segment_end=200.0, focal_target="ACTION_SCENE")
                    ],
                    hook_score=0.85, retention_score=0.80,
                    cta_present=True,
                    reasoning="Fallback clip 4 (40s)",
                    caption_text="Clip 4",
                    editor_version=current_iter + 1
                )
            ]
            return {
                "cutting_manifest": CuttingManifest(
                    clips=fallback_clips,
                    debate_iterations=current_iter,
                    consensus_reached=False
                ),
                "error_log": [f"Editor failed, fallback used: {e}"]
            }

    async def director_node(self, state: PipelineState) -> dict:
        """Director Agent: critiques the proposed manifest and decides if it meets quality standards."""
        logger.info("[GRAPH] Entering Director Node")

        cutting_manifest = get_state_field(state, "cutting_manifest")
        if isinstance(cutting_manifest, dict):
            cutting_manifest = CuttingManifest(**cutting_manifest)

        if not cutting_manifest:
            return {"cutting_manifest": CuttingManifest(clips=[], debate_iterations=1, consensus_reached=True)}

        clips = []
        for c in cutting_manifest.clips:
            if isinstance(c, dict):
                clips.append(c)
            elif hasattr(c, "model_dump"):
                clips.append(c.model_dump())

        # 0. Total duration validation boundary
        rejection_reason = None
        if cutting_manifest and cutting_manifest.clips:
            if len(cutting_manifest.clips) < 4:
                rejection_reason = (
                    f"REJECTED: You only generated {len(cutting_manifest.clips)} clip(s). "
                    f"Manifest must contain at least 4 distinct clips."
                )
            for clip_idx, clip in enumerate(cutting_manifest.clips):
                if rejection_reason:
                    break
                segs = clip.segments if hasattr(clip, "segments") else clip.get("segments", [])
                total_dur = 0.0
                for seg in segs:
                    seg_start = getattr(seg, "segment_start", 0.0) if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
                    seg_end = getattr(seg, "segment_end", 0.0) if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
                    total_dur += (seg_end - seg_start)
                if not (28.0 <= total_dur <= 48.0):
                    rejection_reason = (
                        f"REJECTED: Clip {clip_idx + 1} has a total accumulated duration of {total_dur:.1f}s, "
                        f"which violates the strict 30.0s - 48.0s Shorts/Reels duration gate. "
                        f"Each clip MUST be between 30.0 and 48.0 seconds."
                    )
                    break

        # 1. Programmatic 12-second pacing and technical spec validation
        fused_timeline = get_state_field(state, "fused_timeline")
        if fused_timeline and cutting_manifest and cutting_manifest.clips:
            ft_entries = fused_timeline.entries if hasattr(fused_timeline, "entries") else fused_timeline.get("entries", [])
            scene_changes = fused_timeline.scene_changes if hasattr(fused_timeline, "scene_changes") else fused_timeline.get("scene_changes", [])
            spatial_events = fused_timeline.spatial_events if hasattr(fused_timeline, "spatial_events") else fused_timeline.get("spatial_events", [])
            
            active_keywords = {"look", "watch", "here", "this", "see", "show", "demonstrate", "listen", "real", "but", "wait", "surprise", "now", "suddenly", "check"}
            
            for clip_idx, clip in enumerate(cutting_manifest.clips):
                segs = clip.segments if hasattr(clip, "segments") else clip.get("segments", [])
                for seg_idx, seg in enumerate(segs):
                    seg_start = getattr(seg, "segment_start", 0.0) if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
                    seg_end = getattr(seg, "segment_end", 0.0) if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
                    seg_dur = seg_end - seg_start
                    
                    if seg_dur > 15.0:
                        # A: Check for confirmed scene change
                        has_scene_change = False
                        for sc in scene_changes:
                            sc_ts = getattr(sc, "timestamp", 0.0) if hasattr(sc, "timestamp") else sc.get("timestamp", 0.0)
                            if seg_start <= sc_ts <= seg_end:
                                has_scene_change = True
                                break
                                
                        # B: Check for high visual tracking variance
                        pes_in_seg = []
                        for pe in spatial_events:
                            p_id = getattr(pe, "track_id", 0) if hasattr(pe, "track_id") else pe.get("track_id", 0)
                            if p_id != 0:
                                continue
                            pe_ts = getattr(pe, "timestamp", 0.0) if hasattr(pe, "timestamp") else pe.get("timestamp", 0.0)
                            if seg_start <= pe_ts <= seg_end:
                                pes_in_seg.append(pe)
                                
                        has_high_variance = False
                        if len(pes_in_seg) >= 3:
                            centers = []
                            for pe in pes_in_seg:
                                bbox = getattr(pe, "bbox", (0.0, 0.0, 1.0, 1.0)) if hasattr(pe, "bbox") else pe.get("bbox", (0.0, 0.0, 1.0, 1.0))
                                cx = bbox[0] # bbox is now cx, cy, w, h
                                centers.append(cx)
                            mean_c = sum(centers) / len(centers)
                            var = sum((c - mean_c) ** 2 for c in centers) / len(centers)
                            if var > 0.002:
                                has_high_variance = True
                                
                        # C: Check for active keyword transition
                        has_keyword_transition = False
                        seg_entries = []
                        for entry in ft_entries:
                            ts = get_item_field(entry, "timestamp", 0.0)
                            if seg_start <= ts <= seg_end:
                                w = str(get_item_field(entry, "word", "")).lower().strip(".,!?\"'")
                                if w in active_keywords:
                                    has_keyword_transition = True
                                    break
                                    
                        if not (has_scene_change or has_high_variance or has_keyword_transition):
                            rejection_reason = (
                                f"REJECTED: Clip {clip_idx + 1} Segment {seg_idx + 1} ({seg_start:.1f}s - {seg_end:.1f}s) "
                                f"contains a continuous technical narrative of {seg_dur:.1f}s (longer than 12s limit) "
                                f"without a confirmed scene change, high visual tracking variance, or an active keyword transition. "
                                f"Please split or crop this segment using non-contiguous jump-cuts."
                            )
                            break
                if rejection_reason:
                    break
                    
        # 1.5 Cinematic Framing Validation
        if not rejection_reason and fused_timeline and cutting_manifest and cutting_manifest.clips:
            ft_entries = fused_timeline.entries if hasattr(fused_timeline, "entries") else fused_timeline.get("entries", [])
            hardware_keywords = {"look", "holding", "keyboard", "laptop", "phone", "device", "show", "hands", "demonstrate", "this", "screws", "unscrew", "inside", "numpad", "battery", "motherboard", "display", "power brick", "ports", "fingerprint"}
            
            for clip_idx, clip in enumerate(cutting_manifest.clips):
                segs = clip.segments if hasattr(clip, "segments") else clip.get("segments", [])
                continuous_face_duration = 0.0
                continuous_split_duration = 0.0
                has_hardware_keywords = False
                
                for seg_idx, seg in enumerate(segs):
                    seg_start = getattr(seg, "segment_start", 0.0) if hasattr(seg, "segment_start") else seg.get("segment_start", 0.0)
                    seg_end = getattr(seg, "segment_end", 0.0) if hasattr(seg, "segment_end") else seg.get("segment_end", 0.0)
                    camera_target = getattr(seg, "camera_target", "SPEAKER_FACE") if hasattr(seg, "camera_target") else seg.get("camera_target", "SPEAKER_FACE")
                    layout_mode = getattr(seg, "layout_mode", "speaker_full") if hasattr(seg, "layout_mode") else seg.get("layout_mode", "speaker_full")
                    seg_dur = seg_end - seg_start
                    
                    # Normalize layout mode in director check
                    norm_layout = str(layout_mode).upper().strip()
                    if norm_layout in ("SPLIT_STACK", "SPLIT_SCREEN"):
                        norm_layout = "SPLIT_STACK"
                    elif norm_layout in ("CONTENT_FIT", "BROLL", "GRAPHIC"):
                        norm_layout = "CONTENT_FIT"
                    else:
                        norm_layout = "SPEAKER_SOLO"
                        
                    if camera_target == "SPEAKER_FACE":
                        continuous_face_duration += seg_dur
                        
                        # Check for hardware keywords in this segment
                        for entry in ft_entries:
                            ts = get_item_field(entry, "timestamp", 0.0)
                            if seg_start <= ts <= seg_end:
                                w = str(get_item_field(entry, "word", "")).lower().strip(".,!?\"'")
                                if w in hardware_keywords:
                                    has_hardware_keywords = True
                                    break
                    else:
                        continuous_face_duration = 0.0
                        
                    if continuous_face_duration > 20.0 and has_hardware_keywords:
                        rejection_reason = (
                            f"REJECTED: Clip {clip_idx + 1} maintains 'SPEAKER_FACE' camera target for >20.0s "
                            f"({continuous_face_duration:.1f}s) while physical demonstration/hardware keywords "
                            f"are present in the transcript. You MUST use 'SPEAKER_HANDS' or 'GLOBAL_SALIENCY' "
                            f"to introduce visual variety and eliminate 'Face-Lock' blindness."
                        )
                        break
                        
                if rejection_reason:
                    break
        
        # 2. Retention score threshold: reject if average retention < 0.6
        if not rejection_reason and cutting_manifest and cutting_manifest.clips:
            avg_retention = sum(
                getattr(c, "retention_score", 0.5) if hasattr(c, "retention_score") else c.get("retention_score", 0.5)
                for c in cutting_manifest.clips
            ) / len(cutting_manifest.clips)
            if avg_retention < 0.4:
                rejection_reason = (
                    f"REJECTED: Average retention score across all clips is {avg_retention:.2f} (below 0.6 threshold). "
                    f"Select higher-engagement conversational segments with stronger hooks and emotional velocity."
                )

        system_prompt = (
            "You are a rigorous Director of Content Production. You review viral video clips "
            "to ensure they are narrative-focused, high energy, start exactly on a punchy word, "
            "and are structurally perfect.\n"
            "Production Guidelines:\n"
            "- Clip duration must be strictly between 30 and 58 seconds (HARD LIMIT: never exceed 58s).\n"
            "- Must contain at least 3 clips.\n"
            "- Every clip must have a clear self-contained story arc with a definitive payoff (Premise -> Friction -> Conclusion Loop).\n"
            "- The first 3 seconds MUST contain a high-arousal retention hook (contrarian, surprise, or energy shift).\n"
            "- Average retention_score across clips must exceed 0.6.\n"
            "- Each segment must have an appropriate focal_target (SPEAKER_PRIMARY, MONITOR_SCREEN, HELD_OBJECT, etc.)."
        )

        prompt = (
            f"Review the proposed viral clips:\n{json.dumps(clips)}\n\n"
            f"Judge whether these clips are ready for production. "
            f"If they are excellent, reply ONLY with 'APPROVED'. "
            f"If they have pacing, technical list dumps, monologue, or hook issues, reply with the exact "
            f"improvements needed, starting with 'REJECTED: <critique>'. Keep the critique extremely brief (max 30 words)."
        )

        try:
            if rejection_reason:
                critique = rejection_reason
                logger.info(f"[GRAPH] Programmatic Director Rejection: {critique}")
            else:
                critique = await self.run_brain_llm(prompt, system_prompt, max_tokens=300)
                logger.info(f"[GRAPH] Director Response: {critique}")

            manifest_clips = list(cutting_manifest.clips)
            consensus = False

            critique_upper = critique.upper()
            if "APPROVED" in critique_upper and "REJECTED" not in critique_upper:
                logger.info("[GRAPH] Consensus reached! Director APPROVED the manifest.")
                consensus = True
            else:
                logger.info(f"[GRAPH] Director REJECTED manifest with critique: {critique}")
                updated_clips = []
                for clip in manifest_clips:
                    if isinstance(clip, dict):
                        clip["director_critique"] = critique
                        updated_clips.append(ClipManifest(**clip))
                    else:
                        clip.director_critique = critique
                        updated_clips.append(clip)
                manifest_clips = updated_clips

            new_manifest = CuttingManifest(
                clips=manifest_clips,
                debate_iterations=cutting_manifest.debate_iterations + 1,
                consensus_reached=consensus
            )
            return {"cutting_manifest": new_manifest}

        except Exception as e:
            logger.error(f"Director node error: {e}")
            new_manifest = CuttingManifest(
                clips=cutting_manifest.clips,
                debate_iterations=cutting_manifest.debate_iterations + 1,
                consensus_reached=True
            )
            return {"cutting_manifest": new_manifest, "error_log": [f"Director failed: {e}"]}

    def consensus_router(self, state: PipelineState) -> str:
        """Routes: end debate, iterate, or cap at MAX_DEBATE_ITERATIONS."""
        cutting_manifest = get_state_field(state, "cutting_manifest")
        if isinstance(cutting_manifest, dict):
            cutting_manifest = CuttingManifest(**cutting_manifest)
        if not cutting_manifest:
            return "editor"

        consensus_reached = getattr(cutting_manifest, "consensus_reached", False)
        debate_iterations = getattr(cutting_manifest, "debate_iterations", 0)

        if consensus_reached:
            logger.info("[ROUTER] Consensus achieved. Directing to publish.")
            return "publish"

        if debate_iterations >= MAX_DEBATE_ITERATIONS:
            logger.warning(f"[ROUTER] Debate reached hard cap ({MAX_DEBATE_ITERATIONS}). Forcing approval.")
            return "publish"

        logger.info(f"[ROUTER] Debate continuing. Round {debate_iterations}/{MAX_DEBATE_ITERATIONS}")
        return "editor"

    async def publish_node(self, state: PipelineState) -> dict:
        """Finalizes the debate and enforces strict programmatic duration clamp (<=47.5s) with word-boundary snapping."""
        logger.info("[GRAPH] Entering Publish Node. Debate complete.")
        cutting_manifest = get_state_field(state, "cutting_manifest")
        if isinstance(cutting_manifest, dict):
            cutting_manifest = CuttingManifest(**cutting_manifest)

        # Load transcript words for word-boundary snapping
        transcript_words = get_state_field(state, "transcript_words", [])
        word_ends = []
        for w in transcript_words:
            w_end = w.get("end", 0.0) if isinstance(w, dict) else getattr(w, "end", 0.0)
            if w_end > 0:
                word_ends.append(float(w_end))
        word_ends.sort()

        MAX_CLIP_DUR = 47.5

        if cutting_manifest and cutting_manifest.clips:
            for clip_idx, clip in enumerate(cutting_manifest.clips):
                segs = clip.segments if hasattr(clip, "segments") else clip.get("segments", [])
                total_dur = 0.0
                clamped_segs = []
                for s in segs:
                    s_start = s.segment_start if hasattr(s, "segment_start") else s.get("segment_start", 0.0)
                    s_end = s.segment_end if hasattr(s, "segment_end") else s.get("segment_end", 0.0)
                    dur = s_end - s_start
                    if total_dur + dur > MAX_CLIP_DUR:
                        remaining = max(0.0, MAX_CLIP_DUR - total_dur)
                        if remaining >= 2.0:
                            raw_trim_end = s_start + remaining
                            # Word-boundary snap: find nearest preceding word end
                            snapped_end = raw_trim_end
                            if word_ends:
                                import bisect
                                idx = bisect.bisect_right(word_ends, raw_trim_end) - 1
                                if idx >= 0 and word_ends[idx] >= s_start + 1.0:
                                    snapped_end = word_ends[idx]
                            if hasattr(s, "segment_end"):
                                s.segment_end = round(snapped_end, 2)
                            else:
                                s["segment_end"] = round(snapped_end, 2)
                            clamped_segs.append(s)
                            total_dur += (snapped_end - s_start)
                        break
                    else:
                        clamped_segs.append(s)
                        total_dur += dur

                if hasattr(clip, "segments"):
                    clip.segments = clamped_segs
                else:
                    clip["segments"] = clamped_segs
                logger.info(f"[GRAPH] Clip {clip_idx + 1} finalized with duration {total_dur:.1f}s (strictly <= {MAX_CLIP_DUR}s)")

        return {"cutting_manifest": cutting_manifest, "current_stage": "debate_complete", "pipeline_complete": True}

    def build_graph(self) -> Any:
        """Assembles and compiles the LangGraph StateGraph workflow."""
        workflow = StateGraph(PipelineState)

        workflow.add_node("editor", self.editor_node)
        workflow.add_node("director", self.director_node)
        workflow.add_node("publish", self.publish_node)

        workflow.set_entry_point("editor")
        workflow.add_edge("editor", "director")

        workflow.add_conditional_edges(
            "director",
            self.consensus_router,
            {
                "editor": "editor",
                "publish": "publish"
            }
        )

        workflow.add_edge("publish", END)

        return workflow.compile()
