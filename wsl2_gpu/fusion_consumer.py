import os
import sys
import json
import asyncio
import logging
import time
import httpx
import nats
from nats.js.api import ConsumerConfig
from typing import List, Optional, Tuple, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.schemas import (
    WordTimestamp, SceneChange, SpatialEvent, SpeakerTurn,
    FusedTimelineEntry, FusedTimeline,
    SemanticTopicCategory, GroundingEntity, VisualGroundingTarget, TranscriptBeat
)
from shared.nats_subjects import METADATA_AUDIO, METADATA_VISION, METADATA_FUSED

logger = logging.getLogger("AetherFusionConsumer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Proximity windows for merging events to transcript words
SCENE_CHANGE_WINDOW = 1.0   # seconds: scene changes within ±1s of a word
POSE_WINDOW = 0.5           # seconds: pose events within ±0.5s of a word

BRAIN_MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"


# ─── Semantic Topic & Visual Grounding Engine ────────────────────────────────

SCREEN_KEYWORDS = {
    "screen", "monitor", "display", "terminal", "benchmark", "window",
    "code", "browser", "dashboard", "ui", "tab", "app", "interface",
    "graph", "chart", "ide", "editor", "webpage", "website", "tablet"
}

ACTION_KEYWORDS = {
    "holding", "unscrew", "unscrewing", "plug", "plugging", "press",
    "pressing", "swap", "swapping", "disassemble", "take apart", "inside",
    "ports", "motherboard", "screws", "battery", "hand", "hands",
    "demonstrate", "demonstrating", "showing", "look at this", "watch this",
    "check this out", "check out", "right here", "this one", "notice how",
    "take a look", "as you can see", "here is", "putting", "assembling",
    "connecting", "disconnecting", "opening", "closing", "replacing", "installing"
}

PRODUCT_KEYWORDS = {
    "laptop", "device", "phone", "gpu", "cpu", "hardware", "module",
    "specs", "chassis", "camera", "gadget", "keyboard", "product",
    "expansion", "chip", "board", "circuit", "processor", "cooler",
    "component", "card", "drive", "memory", "ram", "fan", "heatsink"
}

HOOK_TRIGGERS = [
    "have you ever", "why does", "what if", "look at this", "secret",
    "stop doing", "check this", "did you know", "how to", "the real reason"
]

PUNCHLINE_TRIGGERS = [
    "that is why", "that's why", "in conclusion", "turns out", "finally",
    "bottom line", "wins", "results", "verdict", "recommend", "takeaway"
]


def ground_beat_to_visuals(
    beat: TranscriptBeat,
    spatial_events: List[Any]
) -> Optional[VisualGroundingTarget]:
    """Cross-references a TranscriptBeat to spatial detection events to determine
    the exact focal target bounding box coordinates.
    """
    if not spatial_events:
        return None

    def _get_ts(e):
        return float(e.get("timestamp", 0.0) if isinstance(e, dict) else getattr(e, "timestamp", 0.0))

    def _get_lbl(e):
        return str(e.get("label", "") if isinstance(e, dict) else getattr(e, "label", "")).lower().strip()

    def _get_bbox(e):
        return tuple(e.get("bbox", (0.5, 0.5, 0.3, 0.3)) if isinstance(e, dict) else getattr(e, "bbox", (0.5, 0.5, 0.3, 0.3)))

    def _get_act(e):
        return e.get("action_label", None) if isinstance(e, dict) else getattr(e, "action_label", None)

    def _get_tid(e):
        return int(e.get("track_id", 0) if isinstance(e, dict) else getattr(e, "track_id", 0))

    def _get_conf(e):
        return float(e.get("confidence", 0.5) if isinstance(e, dict) else getattr(e, "confidence", 0.5))

    # Temporal window: nearby events within [beat.start_time - 1.0, beat.end_time + 1.0]
    window_events = [
        e for e in spatial_events
        if (_get_ts(e) >= beat.start_time - 1.0 and _get_ts(e) <= beat.end_time + 1.0)
    ]
    if not window_events:
        # Fallback to events within ±3.0s
        mid = (beat.start_time + beat.end_time) / 2.0
        window_events = [
            e for e in spatial_events
            if abs(_get_ts(e) - mid) <= 3.0
        ]
    if not window_events:
        window_events = spatial_events

    person_events = [e for e in window_events if _get_lbl(e) == "person"]
    non_person_events = [e for e in window_events if _get_lbl(e) not in ("person", "saliency")]

    speaker_bbox = _get_bbox(person_events[0]) if person_events else None

    # 1. Product Showcase or Hands-On Action
    if beat.topic_category in (SemanticTopicCategory.PRODUCT_SHOWCASE, SemanticTopicCategory.HANDS_ON_ACTION):
        # Prioritize held_product tagged objects or synthetic demo rois
        held_objs = [e for e in non_person_events if _get_act(e) == "held_product"]
        target_obj = held_objs[0] if held_objs else (non_person_events[0] if non_person_events else None)

        if target_obj:
            t_type = "PRODUCT" if beat.topic_category == SemanticTopicCategory.PRODUCT_SHOWCASE else "HELD_OBJECT"
            anchor = "HAND_REGION" if (_get_act(target_obj) == "held_product" or "holding" in (beat.demonstration_action or "")) else "CENTER"
            return VisualGroundingTarget(
                target_type=t_type,
                primary_bbox=_get_bbox(target_obj),
                secondary_bbox=speaker_bbox,
                track_id=_get_tid(target_obj),
                confidence=_get_conf(target_obj),
                anchor_mode=anchor
            )
        elif person_events:
            return VisualGroundingTarget(
                target_type="SPEAKER",
                primary_bbox=_get_bbox(person_events[0]),
                track_id=_get_tid(person_events[0]),
                confidence=_get_conf(person_events[0]),
                anchor_mode="TOP_ALIGNED"
            )

    # 2. Screen Demonstration
    elif beat.topic_category == SemanticTopicCategory.SCREEN_DEMONSTRATION:
        screen_labels = {"monitor", "laptop", "tv", "cell phone", "screen", "tablet"}
        screen_objs = [e for e in non_person_events if _get_lbl(e) in screen_labels]
        target_obj = screen_objs[0] if screen_objs else (non_person_events[0] if non_person_events else None)

        if target_obj:
            return VisualGroundingTarget(
                target_type="SCREEN",
                primary_bbox=_get_bbox(target_obj),
                secondary_bbox=speaker_bbox,
                track_id=_get_tid(target_obj),
                confidence=_get_conf(target_obj),
                anchor_mode="CENTER"
            )
        elif person_events:
            return VisualGroundingTarget(
                target_type="SPEAKER",
                primary_bbox=_get_bbox(person_events[0]),
                track_id=_get_tid(person_events[0]),
                confidence=_get_conf(person_events[0]),
                anchor_mode="TOP_ALIGNED"
            )

    # 3. Speaker Hook, Punchline, or Dialogue Context
    if person_events:
        return VisualGroundingTarget(
            target_type="SPEAKER",
            primary_bbox=_get_bbox(person_events[0]),
            track_id=_get_tid(person_events[0]),
            confidence=_get_conf(person_events[0]),
            anchor_mode="TOP_ALIGNED"
        )
    elif non_person_events:
        return VisualGroundingTarget(
            target_type="PRODUCT",
            primary_bbox=_get_bbox(non_person_events[0]),
            track_id=_get_tid(non_person_events[0]),
            confidence=_get_conf(non_person_events[0]),
            anchor_mode="CENTER"
        )

    return None


def extract_transcript_beats(
    words: List[Any],
    spatial_events: List[Any],
    speaker_turns: Optional[List[Any]] = None
) -> List[TranscriptBeat]:
    """Extracts narrative and demonstration beats from transcript words and groups them
    by semantic topic category, correlating both spoken content and visual spatial events.
    """
    if not words:
        return []

    def _w_start(w):
        return float(w.get("start", 0.0) if isinstance(w, dict) else getattr(w, "start", 0.0))

    def _w_end(w):
        return float(w.get("end", 0.0) if isinstance(w, dict) else getattr(w, "end", 0.0))

    def _w_text(w):
        return str(w.get("word", "") if isinstance(w, dict) else getattr(w, "word", "")).strip()

    # Partition words into sentence-like or pause-delimited beat chunks
    chunks: List[List[Tuple[int, Any]]] = []
    current_chunk: List[Tuple[int, Any]] = []

    for idx, w in enumerate(words):
        current_chunk.append((idx, w))
        word_text = _w_text(w)

        # Check sentence ending punctuation or large pause
        is_punct_end = word_text.endswith((".", "!", "?"))
        has_pause = (idx < len(words) - 1 and (_w_start(words[idx + 1]) - _w_end(w)) >= 0.7)
        chunk_len = len(current_chunk)

        if (is_punct_end or has_pause or chunk_len >= 16) and chunk_len >= 4:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        if chunks and len(current_chunk) < 3:
            chunks[-1].extend(current_chunk)
        else:
            chunks.append(current_chunk)

    beats: List[TranscriptBeat] = []

    def _get_ts(e):
        return float(e.get("timestamp", 0.0) if isinstance(e, dict) else getattr(e, "timestamp", 0.0))

    def _get_lbl(e):
        return str(e.get("label", "") if isinstance(e, dict) else getattr(e, "label", "")).lower().strip()

    def _get_act(e):
        return e.get("action_label", None) if isinstance(e, dict) else getattr(e, "action_label", None)

    for b_idx, chunk in enumerate(chunks):
        start_idx = chunk[0][0]
        end_idx = chunk[-1][0]
        c_words = [item[1] for item in chunk]
        start_t = _w_start(c_words[0])
        end_t = _w_end(c_words[-1])
        full_text = " ".join(_w_text(cw) for cw in c_words)
        lower_text = full_text.lower()

        # Identify demonstration verbs from speech
        demo_action = None
        for v in ["holding", "unscrew", "unscrewing", "swapping", "plugging", "pointing",
                  "demonstrating", "typing", "pressing", "showing", "replacing", "assembling"]:
            if v in lower_text:
                demo_action = v
                break

        # Check overlapping visual events in [start_t - 0.5, end_t + 0.5]
        window_events = [
            e for e in spatial_events
            if (start_t - 0.5 <= _get_ts(e) <= end_t + 0.5)
        ]
        held_visual_objs = [e for e in window_events if _get_act(e) == "held_product" or _get_lbl(e) == "demonstration_action"]
        screen_visual_objs = [e for e in window_events if _get_lbl(e) in ("monitor", "laptop", "tv", "cell phone", "tablet")]
        other_visual_objs = [e for e in window_events if _get_lbl(e) not in ("person", "saliency")]

        # Categorize
        is_hook = any(h in lower_text for h in HOOK_TRIGGERS) or (start_t < 5.0 and b_idx == 0)
        has_screen = any(sk in lower_text for sk in SCREEN_KEYWORDS) or bool(screen_visual_objs)
        has_action = any(ak in lower_text for ak in ACTION_KEYWORDS) or (demo_action is not None) or bool(held_visual_objs)
        has_product = any(pk in lower_text for pk in PRODUCT_KEYWORDS) or bool(other_visual_objs)
        has_punchline = any(pt in lower_text for pt in PUNCHLINE_TRIGGERS)

        entities: List[GroundingEntity] = []

        if is_hook:
            cat = SemanticTopicCategory.SPEAKER_HOOK
            entities.append(GroundingEntity(entity_name="Speaker", category="PERSON", visual_aliases=["person"]))
        elif has_action and (held_visual_objs or demo_action is not None or any(ak in lower_text for ak in ACTION_KEYWORDS)):
            cat = SemanticTopicCategory.HANDS_ON_ACTION
            matched_prod = [pk for pk in PRODUCT_KEYWORDS if pk in lower_text]
            if held_visual_objs:
                lbl = _get_lbl(held_visual_objs[0])
                e_name = lbl.replace("_", " ").title() if lbl != "demonstration_action" else "Demonstration"
            elif matched_prod:
                e_name = matched_prod[0].title()
            else:
                e_name = "Demonstration"
            entities.append(GroundingEntity(entity_name=e_name, category="HARDWARE_COMPONENT", visual_aliases=["laptop", "keyboard", "device"]))
        elif has_screen:
            cat = SemanticTopicCategory.SCREEN_DEMONSTRATION
            matched_screen = [sk for sk in SCREEN_KEYWORDS if sk in lower_text]
            if screen_visual_objs:
                e_name = _get_lbl(screen_visual_objs[0]).title()
            elif matched_screen:
                e_name = matched_screen[0].title()
            else:
                e_name = "Display"
            entities.append(GroundingEntity(entity_name=e_name, category="SCREEN", visual_aliases=["monitor", "laptop", "tv"]))
        elif has_product:
            cat = SemanticTopicCategory.PRODUCT_SHOWCASE
            matched_prod = [pk for pk in PRODUCT_KEYWORDS if pk in lower_text]
            if other_visual_objs:
                e_name = _get_lbl(other_visual_objs[0]).title()
            elif matched_prod:
                e_name = matched_prod[0].title()
            else:
                e_name = "Product"
            entities.append(GroundingEntity(entity_name=e_name, category="PRODUCT", visual_aliases=["laptop", "cell phone", "keyboard"]))
        elif has_punchline:
            cat = SemanticTopicCategory.EXPLANATION_PUNCHLINE
            entities.append(GroundingEntity(entity_name="Core Takeaway", category="CONCEPT", visual_aliases=["person"]))
        else:
            cat = SemanticTopicCategory.DIALOGUE_CONTEXT
            entities.append(GroundingEntity(entity_name="Speaker Dialogue", category="PERSON", visual_aliases=["person"]))

        beat = TranscriptBeat(
            beat_id=f"beat_{b_idx + 1:02d}",
            start_time=round(start_t, 2),
            end_time=round(end_t, 2),
            topic_category=cat,
            summary=full_text[:80],
            entities=entities,
            demonstration_action=demo_action,
            word_indices=(start_idx, end_idx),
            visual_grounding=None
        )

        # Cross-reference with spatial events
        beat.visual_grounding = ground_beat_to_visuals(beat, spatial_events)
        beats.append(beat)

    return beats


class FusionConsumer:
    """Phase 2 Sequential Fusion Join: merges transcription and vision pre-pass streams,
    then feeds the unified text-annotated timeline to the local MoE brain for hook analysis.

    Waits for both metadata.audio and metadata.vision for the same source_path before
    producing a fused metadata.fused output.
    """

    def __init__(self, nats_url: str):
        self.nats_url = nats_url
        self.nc = None
        self.js = None
        self.brain_url = os.getenv("VLLM_BRAIN_URL", "http://vllm-brain:8000").rstrip("/")

        # Pending partial results keyed by source_path
        self.pending_audio: dict = {}   # source_path -> payload
        self.pending_vision: dict = {}  # source_path -> payload

    async def run(self):
        """Connects to NATS and consumes both metadata streams."""
        logger.info(f"Connecting to NATS at {self.nats_url}...")
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()

        # Durable pull subscriptions with hardened max_ack_pending
        self.audio_sub = await self.js.pull_subscribe(
            subject=METADATA_AUDIO,
            durable="fusion_audio",
            config=ConsumerConfig(ack_wait=600.0, max_ack_pending=12)
        )
        self.vision_sub = await self.js.pull_subscribe(
            subject=METADATA_VISION,
            durable="fusion_vision",
            config=ConsumerConfig(ack_wait=600.0, max_ack_pending=12)
        )

        logger.info("Fusion Consumer initialized. Polling for audio and vision metadata...")

        try:
            while True:
                # Poll both streams
                await self._poll_audio()
                await self._poll_vision()

                # Check for any complete pairs
                ready_paths = set(self.pending_audio.keys()) & set(self.pending_vision.keys())
                for source_path in ready_paths:
                    await self._fuse_and_publish(source_path)

                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("Fusion Consumer cancelled, shutting down...")
        finally:
            if self.nc:
                await self.nc.close()

    async def _poll_audio(self):
        """Fetches pending audio metadata messages."""
        try:
            msgs = await self.audio_sub.fetch(batch=1, timeout=1.0)
            for msg in msgs:
                payload = json.loads(msg.data.decode("utf-8"))
                source_path = payload.get("source_path")
                logger.info(f"Received audio metadata for: {source_path}")
                self.pending_audio[source_path] = payload
                await msg.ack()
        except nats.errors.TimeoutError:
            pass

    async def _poll_vision(self):
        """Fetches pending vision metadata messages."""
        try:
            msgs = await self.vision_sub.fetch(batch=1, timeout=1.0)
            for msg in msgs:
                payload = json.loads(msg.data.decode("utf-8"))
                source_path = payload.get("source_path")
                logger.info(f"Received vision metadata for: {source_path}")
                self.pending_vision[source_path] = payload
                await msg.ack()
        except nats.errors.TimeoutError:
            pass

    async def _fuse_and_publish(self, source_path: str):
        """Merges audio + vision data, runs LLM hook analysis, publishes fused timeline."""
        logger.info(f"🔀 Fusing audio + vision metadata for: {source_path}")
        t0 = time.perf_counter()

        audio_data = self.pending_audio.pop(source_path)
        vision_data = self.pending_vision.pop(source_path)

        try:
            # Parse raw data into models
            words = [WordTimestamp(**w) for w in audio_data.get("words", [])]
            scene_changes = [SceneChange(**sc) for sc in vision_data.get("scene_changes", [])]
            spatial_events = [SpatialEvent(**pe) for pe in vision_data.get("spatial_events", [])]
            video_duration = float(vision_data.get("video_duration", 300.0))

            # ── Step 1: Merge into unified timeline entries ──────────────────
            entries = []
            for word in words:
                word_mid = (word.start + word.end) / 2.0

                # Find scene changes within ±SCENE_CHANGE_WINDOW of this word
                nearby_scenes = [
                    sc for sc in scene_changes
                    if abs(sc.timestamp - word_mid) <= SCENE_CHANGE_WINDOW
                ]

                # Find pose events within ±POSE_WINDOW of this word
                nearby_poses = [
                    pe for pe in spatial_events
                    if abs(pe.timestamp - word_mid) <= POSE_WINDOW
                ]

                entries.append(FusedTimelineEntry(
                    timestamp=word.start,
                    word=word.word,
                    word_end=word.end,
                    word_confidence=word.confidence,
                    scene_changes_nearby=nearby_scenes,
                    spatial_events_nearby=nearby_poses
                ))

            logger.info(f"Merged {len(words)} words with {len(scene_changes)} scene changes "
                       f"and {len(spatial_events)} pose events into {len(entries)} timeline entries.")

            # ── Step 2: Extract transcript beats & LLM hook analysis ─────────
            transcript_beats = extract_transcript_beats(words, spatial_events)
            await self._run_hook_analysis(entries, scene_changes, spatial_events)

            # Clear heavy keypoint coordinate data before NATS serialization to prevent MaxPayloadError
            for pe in spatial_events:
                pe.keypoints = []
            for entry in entries:
                for pe in entry.spatial_events_nearby:
                    pe.keypoints = []

            # ── Step 3: Publish fused timeline to NATS ───────────────────────
            fused = FusedTimeline(
                source_path=source_path,
                video_duration=video_duration,
                entries=entries,
                scene_changes=scene_changes,
                spatial_events=spatial_events,
                transcript_words=words,
                transcript_beats=transcript_beats
            )

            fused_payload = {
                "source_path": source_path,
                "fused_timeline": fused.model_dump(),
                "timestamp": time.time()
            }

            await self.js.publish(METADATA_FUSED, json.dumps(fused_payload).encode("utf-8"))
            elapsed = time.perf_counter() - t0
            logger.info(f"✅ Published fused timeline to '{METADATA_FUSED}' in {elapsed:.2f}s "
                        f"({len(transcript_beats)} transcript beats)")

        except Exception as e:
            logger.error(f"Fusion failed for {source_path}: {e}", exc_info=True)
            # Re-publish a minimal fused timeline so the pipeline doesn't stall
            try:
                words = [WordTimestamp(**w) for w in audio_data.get("words", [])]
                minimal_fused = FusedTimeline(
                    source_path=source_path,
                    video_duration=float(vision_data.get("video_duration", 300.0)),
                    entries=[FusedTimelineEntry(
                        timestamp=w.start, word=w.word, word_end=w.end,
                        word_confidence=w.confidence
                    ) for w in words],
                    transcript_words=words,
                    transcript_beats=extract_transcript_beats(words, spatial_events)
                )
                fallback_payload = {
                    "source_path": source_path,
                    "fused_timeline": minimal_fused.model_dump(),
                    "timestamp": time.time()
                }
                await self.js.publish(METADATA_FUSED, json.dumps(fallback_payload).encode("utf-8"))
                logger.warning("Published minimal fallback fused timeline (without LLM hook analysis).")
            except Exception as e2:
                logger.error(f"Even fallback fusion publish failed: {e2}")

    async def _run_hook_analysis(self, entries: List[FusedTimelineEntry],
                                  scene_changes: List[SceneChange],
                                  spatial_events: List[SpatialEvent]):
        """Sends windowed timeline segments to the unified MoE brain for hook scoring."""
        logger.info("Running LLM hook analysis on fused timeline segments...")

        # Group entries into 30-second windows for analysis
        window_size = 30.0
        if not entries:
            return

        total_duration = entries[-1].timestamp if entries else 0
        windows = []
        current_window_start = 0.0

        while current_window_start < total_duration:
            window_end = current_window_start + window_size
            window_entries = [e for e in entries
                            if e.timestamp >= current_window_start and e.timestamp < window_end]

            if window_entries:
                # Build context string for this window
                text = " ".join(e.word for e in window_entries if e.word)
                scene_count = sum(1 for e in window_entries for _ in e.scene_changes_nearby)
                gesture_summary = []
                for e in window_entries:
                    for p in e.spatial_events_nearby:
                        if p.action_label and p.action_label != "standing":
                            gesture_summary.append(f"{p.action_label}@{p.timestamp:.1f}s")

                windows.append({
                    "start": current_window_start,
                    "end": window_end,
                    "text": text,
                    "scene_changes": scene_count,
                    "gestures": gesture_summary[:5],  # Limit to avoid token overflow
                    "entries": window_entries
                })

            current_window_start += window_size

        # Analyze each window via the unified brain
        async with httpx.AsyncClient() as client:
            for window in windows:
                try:
                    prompt = (
                        f"Analyze this video segment ({window['start']:.0f}s-{window['end']:.0f}s) "
                        f"for viral short-form potential:\n\n"
                        f"Transcript: \"{window['text']}\"\n"
                        f"Scene changes in window: {window['scene_changes']}\n"
                        f"Notable body language: {', '.join(window['gestures']) if window['gestures'] else 'none detected'}\n\n"
                        f"Rate this segment's hook potential from 0.0 to 1.0 and explain why. "
                        f"Respond with JSON: {{\"arousal_score\": <float>, \"hook_analysis\": \"<reason>\"}}"
                    )

                    payload = {
                        "model": BRAIN_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a viral content analyst. Evaluate video segments for short-form engagement potential based on transcript content, scene dynamics, and body language cues."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 200
                    }

                    response = await client.post(
                        f"{self.brain_url}/v1/chat/completions",
                        json=payload,
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"].strip()
                        # Extract JSON
                        if "{" in content:
                            content = content[content.find("{"):content.rfind("}") + 1]
                        parsed = json.loads(content)
                        arousal = float(parsed.get("arousal_score", 0.5))
                        analysis = str(parsed.get("hook_analysis", ""))

                        # Apply scores to the window's entries
                        for entry in window["entries"]:
                            entry.arousal_score = arousal
                            entry.hook_analysis = analysis

                        logger.info(f"Window {window['start']:.0f}-{window['end']:.0f}s: arousal={arousal:.2f}")
                    else:
                        logger.warning(f"Brain returned {response.status_code} for window "
                                      f"{window['start']:.0f}-{window['end']:.0f}s")

                except Exception as e:
                    logger.warning(f"Hook analysis failed for window "
                                  f"{window['start']:.0f}-{window['end']:.0f}s: {e}")


if __name__ == "__main__":
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    consumer = FusionConsumer(nats_url)
    try:
        asyncio.run(consumer.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
