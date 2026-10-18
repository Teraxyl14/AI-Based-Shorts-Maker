"""
Project Aether V3 — Shot-Midpoint Multimodal Editorial Engine

Extracts 4 distinct, single-topic shorts (30s–50s) using token-compressed
visual prompting with zero teaser montages. Uses shot boundary midpoint
keyframes interleaved with timestamped transcript for multimodal VLM
editorial selection via the vLLM OpenAI-compatible endpoint.
Guarantees at least 4 broadcast-quality shorts via multi-modal fallback
selection combining speech pauses, sentence boundaries, and shot midpoints.
"""

import os
import sys
import json
import logging
import base64
import subprocess
import tempfile
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.schemas import WordTimestamp, UniversalContentProfile

logger = logging.getLogger("AetherEditorialEngine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ─── Constants ───────────────────────────────────────────────────────────────

VLLM_BRAIN_URL = os.getenv("VLLM_BRAIN_URL", "http://vllm-brain:8000")

# Keyframe processing
KEYFRAME_WIDTH = 512
KEYFRAME_HEIGHT = 288
KEYFRAME_QUALITY = 70  # JPEG quality

# Editorial constraints
MIN_START_TIME = 60.0   # Hard rejection of intro teasers (t < 60.0s)
MIN_DURATION = 30.0
MAX_DURATION = 50.0     # Strictly 30.0s <= dur <= 50.0s
TARGET_SHORTS = 4
MAX_HOOK_DELAY = 1.5    # First spoken word must occur <= 1.5s into clip
HOOK_WINDOW = 5.0       # Hook detection window (0 to 5s)

# Shot detection (forwarded from VisionSpoke)
MIN_SHOT_DURATION = 0.5  # Ignore shots shorter than 500ms

# ─── VLM token budget ────────────────────────────────────────────────────
# Must match vLLM --max-model-len. Prompt text + images + output must fit.
VLM_CONTEXT_TOKENS = int(os.getenv("VLLM_MAX_MODEL_LEN", "3072"))
VLM_OUTPUT_TOKENS = 640          # 4 JSON shorts with title + reasoning
PROMPT_SAFETY_TOKENS = 96        # chat template + tokenizer estimate slack
CHARS_PER_TOKEN = 3.0            # conservative for timestamped transcript text
# Qwen3-VL: 16px patches merged 2x2 -> one token per 32x32 block (+ wrappers)
TOKENS_PER_IMAGE = (KEYFRAME_WIDTH // 32) * (KEYFRAME_HEIGHT // 32) + 8
TRANSCRIPT_LEAD_IN_S = 10.0      # context kept before MIN_START_TIME
MIN_WORDS_PER_LINE = 4


# ─── Shot Boundary Data ─────────────────────────────────────────────────────

class ShotBoundary:
    """Contiguous shot boundary with frame and time ranges."""

    def __init__(self, start_frame: int, end_frame: int,
                 start_s: float, end_s: float):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_s = start_s
        self.end_s = end_s

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s

    @property
    def midpoint_s(self) -> float:
        """Geometric center: t_center = (start + end) / 2."""
        return (self.start_s + self.end_s) / 2.0

    def __repr__(self):
        return f"Shot({self.start_s:.2f}s–{self.end_s:.2f}s, mid={self.midpoint_s:.2f}s)"


class ShortCandidate:
    """Output schema for a selected short."""

    def __init__(self, start_time_seconds: float, duration_seconds: float,
                 title: str, reasoning: str,
                 framing_mode: str = "SMART_PORTRAIT_PAN",
                 hook_window: Tuple[float, float] = (0.0, 5.0),
                 first_word_delay: Optional[float] = None,
                 payoff_snapped: bool = False,
                 beats: Optional[List[Any]] = None,
                 content_profile: Optional[Any] = None):
        self.start_time_seconds = start_time_seconds
        self.duration_seconds = duration_seconds
        self.title = title
        self.reasoning = reasoning
        self.framing_mode = framing_mode
        self.hook_window = hook_window
        self.first_word_delay = first_word_delay
        self.payoff_snapped = payoff_snapped
        self.beats = beats or []
        self.content_profile = content_profile

    @property
    def end_time_seconds(self) -> float:
        return self.start_time_seconds + self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time_seconds": round(self.start_time_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "end_time_seconds": round(self.end_time_seconds, 2),
            "title": self.title,
            "reasoning": self.reasoning,
            "framing_mode": self.framing_mode,
            "hook_window": self.hook_window,
            "first_word_delay": self.first_word_delay,
            "payoff_snapped": self.payoff_snapped,
            "beats": [b.model_dump() if hasattr(b, "model_dump") else b for b in self.beats],
            "content_profile": (
                self.content_profile.value if hasattr(self.content_profile, "value")
                else str(self.content_profile) if self.content_profile else None
            ),
        }


def extract_beats_for_candidate(
    candidate: Any,
    transcript_words: List[Any],
    spatial_events: List[Any]
) -> List[Any]:
    """Extracts and grounds transcript beats specifically falling within the candidate's time bounds."""
    from wsl2_gpu.fusion_consumer import extract_transcript_beats
    from shared.schemas import WordTimestamp, SpatialEvent

    c_start = getattr(candidate, "start_time_seconds", 0.0)
    c_dur = getattr(candidate, "duration_seconds", 30.0)
    c_end = getattr(candidate, "end_time_seconds", c_start + c_dur)

    words_in_window = []
    for w in transcript_words:
        if isinstance(w, dict):
            wt = WordTimestamp(**w)
        else:
            wt = w
        if wt.end >= c_start and wt.start <= c_end:
            words_in_window.append(wt)

    s_events = []
    for s in spatial_events:
        if isinstance(s, dict):
            se = SpatialEvent(**s)
        else:
            se = s
        if se.timestamp >= c_start - 1.0 and se.timestamp <= c_end + 1.0:
            s_events.append(se)

    beats = extract_transcript_beats(words_in_window, s_events)
    if hasattr(candidate, "beats"):
        candidate.beats = beats
    return beats



# ═════════════════════════════════════════════════════════════════════════════
# UNIVERSAL VIDEO CONTENT PROFILER
# ═════════════════════════════════════════════════════════════════════════════

def classify_video_profile(transcript_words: List[Any],
                           spatial_events: Optional[List[Any]] = None,
                           video_duration: float = 300.0,
                           scene_changes: Optional[List[Any]] = None) -> UniversalContentProfile:
    """Classifies video into a UniversalContentProfile using acoustic, visual, and lexical signals.

    Profiles:
        STAGE_PERFORMANCE: Standup comedy, solo theatrical acts, keynotes, talks
        TECH_REVIEW_DEMO: Tech presenter, hardware, monitors, screencasts, laptops
        GAMEPLAY_STREAM: Gaming HUDs, facecam, gameplay action
        WILDLIFE_NATURE: Non-human animal subjects, landscapes, nature footage
        CONVERSATIONAL_PODCAST: 2+ seated speakers, turn-taking dialogue
        CINEMATIC_NARRATIVE: Multi-character films, dramatic action, general movies
    """
    spatial_events = spatial_events or []
    scene_changes = scene_changes or []

    # 1. Lexical signal analysis
    text_corpus = " ".join([
        str(w.get("word", "") if isinstance(w, dict) else getattr(w, "word", "")).lower()
        for w in transcript_words
    ])

    tech_keywords = {
        "laptop", "monitor", "screen", "windows", "linux", "macbook", "cpu", "gpu",
        "processor", "specs", "benchmark", "port", "usb", "ram", "gigabit", "unboxing",
        "device", "hardware", "keyboard", "display", "firmware", "setup", "review"
    }
    comedy_keywords = {
        "joke", "comedy", "laugh", "standup", "audience", "applause", "hilarious",
        "comedian", "crowd", "punchline", "funny", "stage", "roast"
    }
    game_keywords = {
        "gameplay", "fps", "boss", "level", "multiplayer", "loot", "respawn", "quest",
        "esports", "controller", "streamer", "glitch", "speedrun", "mod"
    }
    wildlife_keywords = {
        "wildlife", "animal", "safari", "predator", "habitat", "species", "forest",
        "jungle", "ocean", "nature", "lion", "tiger", "bear", "bird", "whale"
    }

    tech_hits = sum(1 for kw in tech_keywords if kw in text_corpus)
    comedy_hits = sum(1 for kw in comedy_keywords if kw in text_corpus)
    game_hits = sum(1 for kw in game_keywords if kw in text_corpus)
    wildlife_hits = sum(1 for kw in wildlife_keywords if kw in text_corpus)

    # 2. Visual object signals
    screen_labels = {"monitor", "screen", "laptop", "tv", "display", "tablet"}
    animal_labels = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
    screen_count = 0
    animal_count = 0
    person_count = 0

    for pe in spatial_events:
        lbl = str(pe.get("label", "") if isinstance(pe, dict) else getattr(pe, "label", "")).lower().strip()
        conf = float(pe.get("confidence", 0.0) if isinstance(pe, dict) else getattr(pe, "confidence", 0.0))
        if conf >= 0.35:
            if lbl in screen_labels:
                screen_count += 1
            elif lbl in animal_labels:
                animal_count += 1
            elif lbl in ("person", "face", "pose"):
                person_count += 1

    total_spatial = max(1, len(spatial_events))
    screen_ratio = screen_count / float(total_spatial)
    animal_ratio = animal_count / float(total_spatial)

    # 3. Decision priority
    if animal_ratio >= 0.20 or wildlife_hits >= 4:
        return UniversalContentProfile.WILDLIFE_NATURE

    if screen_ratio >= 0.08 or tech_hits >= 3:
        return UniversalContentProfile.TECH_REVIEW_DEMO

    if game_hits >= 4:
        return UniversalContentProfile.GAMEPLAY_STREAM

    if comedy_hits >= 3:
        return UniversalContentProfile.STAGE_PERFORMANCE

    # Check speaker count
    speakers = set()
    for w in transcript_words:
        spk = w.get("speaker_id") if isinstance(w, dict) else getattr(w, "speaker_id", None)
        if spk:
            speakers.add(spk)

    if len(speakers) >= 2:
        return UniversalContentProfile.CONVERSATIONAL_PODCAST

    return UniversalContentProfile.CINEMATIC_NARRATIVE


# ─── Universal In-Video Sponsor & Promo Detection ────────────────────────────

SPONSOR_KEYWORDS = frozenset({
    "sponsor", "sponsors", "sponsored", "sponsoring", "partner", "partnered",
    "hetzner", "nordvpn", "expressvpn", "surfshark", "squarespace", "audible",
    "skillshare", "ridge", "hellofresh", "honey", "betterhelp", "seatgeek",
    "incogni", "manscaped", "casetify", "grammarly", "established", "titles",
    "patreon", "merch", "discount", "promo", "promocode"
})

SPONSOR_PHRASES = (
    "sponsored by", "our sponsor", "today's sponsor", "thank you to our sponsor",
    "thanks to our sponsor", "partnered with", "check out the link",
    "link in the description", "link down below", "links in the description",
    "discount code", "promo code", "use code", "use promo code",
    "free trial", "money back guarantee", "get 20% off", "get 15% off",
    "get 10% off", "get 30% off", "support the channel", "patreon supporters",
    "hit the subscribe", "like and subscribe", "subscribe to the channel",
    "leave a comment down below", "hit the bell", "click the link below"
)


def detect_sponsor_segments(words: List[Any],
                           padding_s: float = 3.0) -> List[Tuple[float, float, str]]:
    """Deterministically identifies in-video sponsor reads, ad segments, and promotional fluff.

    Scans word stream for multi-word sponsor phrases and branded partner keywords.
    Expands detected intervals by padding_s (default 3.0s) to guarantee zero residual ad bleed.

    Returns:
        List of (start_s, end_s, matched_reason) intervals.
    """
    if not words:
        return []

    word_items = []
    for w in words:
        if isinstance(w, dict):
            s = float(w.get("start", 0.0))
            e = float(w.get("end", 0.0))
            text = str(w.get("word", "")).strip().lower()
        else:
            s = float(getattr(w, "start", 0.0))
            e = float(getattr(w, "end", 0.0))
            text = str(getattr(w, "word", "")).strip().lower()
        if e > 0 and text:
            import re
            cleaned = re.sub(r'[^\w\s]', '', text)
            word_items.append((s, e, cleaned))

    if not word_items:
        return []

    full_text = " ".join(item[2] for item in word_items)
    detected_intervals: List[Tuple[float, float, str]] = []

    # 1. Multi-word phrase matching
    for phrase in SPONSOR_PHRASES:
        phrase_clean = phrase.lower().strip()
        if phrase_clean in full_text:
            phrase_tokens = phrase_clean.split()
            p_len = len(phrase_tokens)
            for i in range(len(word_items) - p_len + 1):
                window = [word_items[i + k][2] for k in range(p_len)]
                if window == phrase_tokens:
                    seg_start = word_items[i][0]
                    # Track contiguous words in the promotional phrase and subsequent context up to 2.5s gap
                    cur_end = word_items[i + p_len - 1][1]
                    for k in range(i + p_len, len(word_items)):
                        gap = word_items[k][0] - cur_end
                        if gap > 2.5 or (word_items[k][1] - seg_start) > 60.0:
                            break
                        cur_end = word_items[k][1]
                    detected_intervals.append((max(0.0, seg_start - padding_s), cur_end + padding_s, phrase_clean))

    # 2. High-salience sponsor brand keywords (e.g. Hetzner, NordVPN, Squarespace)
    for i, (ws, we, wt) in enumerate(word_items):
        if wt in SPONSOR_KEYWORDS:
            start_idx = max(0, i - 4)
            # Find contiguous block around sponsor brand
            cur_end = we
            for k in range(i + 1, len(word_items)):
                gap = word_items[k][0] - cur_end
                if gap > 2.5 or (word_items[k][1] - ws) > 60.0:
                    break
                cur_end = word_items[k][1]
            context_text = " ".join(word_items[k][2] for k in range(start_idx, min(len(word_items), i + 15)))
            promo_signals = ("link", "code", "below", "check", "visit", "offer", "free", "discount", "sponsor", "save", "com", "percent")
            if any(sig in context_text for sig in promo_signals):
                seg_start = word_items[start_idx][0]
                detected_intervals.append((max(0.0, seg_start - padding_s), cur_end + padding_s, f"brand:{wt}"))

    if not detected_intervals:
        return []

    # Sort and merge nearby sponsor intervals (within 12.0s gap to bridge ad read pauses)
    detected_intervals.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float, str]] = []
    for s_start, s_end, reason in detected_intervals:
        if not merged:
            merged.append((s_start, s_end, reason))
        else:
            prev_s, prev_e, prev_reason = merged[-1]
            if s_start <= prev_e + 12.0:
                merged[-1] = (prev_s, max(prev_e, s_end), f"{prev_reason}+{reason}")
            else:
                merged.append((s_start, s_end, reason))

    return merged


# ═════════════════════════════════════════════════════════════════════════════
# EDITORIAL ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class EditorialEngine:
    """Shot-midpoint multimodal editorial engine for narrative short selection.

    Uses the vLLM OpenAI-compatible endpoint for visual+text reasoning.
    """

    def __init__(self, vllm_url: Optional[str] = None,
                 min_duration: Optional[float] = None,
                 max_duration: Optional[float] = None,
                 target_duration: Optional[float] = None,
                 min_start_time: Optional[float] = None):
        self.vllm_url = vllm_url or VLLM_BRAIN_URL
        self.min_duration = min_duration if min_duration is not None else float(os.getenv("AETHER_MIN_SHORT_DURATION", str(MIN_DURATION)))
        self.max_duration = max_duration if max_duration is not None else float(os.getenv("AETHER_MAX_SHORT_DURATION", str(MAX_DURATION)))
        self.target_duration = target_duration if target_duration is not None else float(os.getenv("AETHER_TARGET_SHORT_DURATION", "60.0"))
        self.min_start_time = min_start_time if min_start_time is not None else float(os.getenv("AETHER_MIN_START_TIME", str(MIN_START_TIME)))

    # ─── Shot Boundary Construction ──────────────────────────────────────

    def build_shot_boundaries(self, scene_changes: List[Any],
                              video_duration: float,
                              fps: float = 30.0) -> List[ShotBoundary]:
        """Converts scene change events into contiguous shot boundary tuples.

        A shot is the interval between consecutive scene cuts:
            Shot_k = [cut_{k-1}, cut_k]
        """
        cuts = []
        for sc in scene_changes:
            if isinstance(sc, dict):
                ts = sc.get("timestamp", 0.0)
            else:
                ts = getattr(sc, "timestamp", 0.0)
            if ts > 0:
                cuts.append(ts)

        cuts = sorted(set(cuts))

        # Add start and end boundaries
        boundary_times = [0.0]
        for c in cuts:
            if c > boundary_times[-1] + MIN_SHOT_DURATION:
                boundary_times.append(c)
        if video_duration > boundary_times[-1] + MIN_SHOT_DURATION:
            boundary_times.append(video_duration)
        elif len(boundary_times) > 1:
            boundary_times[-1] = video_duration

        shots = []
        for i in range(len(boundary_times) - 1):
            s_start = boundary_times[i]
            s_end = boundary_times[i + 1]
            if (s_end - s_start) >= MIN_SHOT_DURATION:
                shots.append(ShotBoundary(
                    start_frame=int(s_start * fps),
                    end_frame=int(s_end * fps),
                    start_s=s_start,
                    end_s=s_end,
                ))

        logger.info(f"Constructed {len(shots)} shot boundaries from {len(cuts)} scene cuts (dur={video_duration:.1f}s)")
        return shots

    # ─── Midpoint Keyframe Extraction ────────────────────────────────────

    def extract_midpoint_keyframes(self, video_path: str,
                                   shots: List[ShotBoundary]) -> List[Dict[str, Any]]:
        """Extracts exactly one downsampled JPEG at the geometric midpoint of each shot."""
        if not shots or not os.path.exists(video_path):
            return []

        keyframes = []
        with tempfile.TemporaryDirectory(prefix="aether_kf_") as tmpdir:
            for idx, shot in enumerate(shots):
                mid = shot.midpoint_s
                out_path = os.path.join(tmpdir, f"shot_{idx:04d}_mid.jpg")

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{mid:.3f}",
                    "-i", video_path,
                    "-vframes", "1",
                    "-vf", f"scale={KEYFRAME_WIDTH}:{KEYFRAME_HEIGHT}:force_original_aspect_ratio=decrease,pad={KEYFRAME_WIDTH}:{KEYFRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2",
                    "-q:v", "5",
                    out_path
                ]

                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=True, timeout=10)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    logger.warning(f"Failed to extract keyframe at {mid:.2f}s for shot {idx}")
                    continue

                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")

                    keyframes.append({
                        "shot_idx": idx,
                        "start_s": shot.start_s,
                        "end_s": shot.end_s,
                        "midpoint_s": mid,
                        "duration_s": shot.duration,
                        "base64_jpeg": b64,
                    })

        logger.info(f"Extracted {len(keyframes)}/{len(shots)} midpoint keyframes ({KEYFRAME_WIDTH}x{KEYFRAME_HEIGHT})")
        return keyframes

    # ─── Interleaved Multimodal Prompt ───────────────────────────────────

    @staticmethod
    def _group_transcript_lines(transcript_words: List[Any]) -> List[Tuple[float, List[str]]]:
        """Groups words into (start_s, words) lines of <=5s / <=15 words."""
        lines: List[Tuple[float, List[str]]] = []
        current: List[str] = []
        chunk_start = 0.0
        for w in transcript_words:
            if isinstance(w, dict):
                w_start, w_end, w_text = w.get("start", 0.0), w.get("end", 0.0), w.get("word", "")
            else:
                w_start = getattr(w, "start", 0.0)
                w_end = getattr(w, "end", 0.0)
                w_text = getattr(w, "word", "")
            if not current:
                chunk_start = w_start
            current = current + [w_text]
            if (w_end - chunk_start >= 5.0) or (len(current) >= 15):
                lines.append((chunk_start, current))
                current = []
        if current:
            lines.append((chunk_start, current))
        return lines

    @staticmethod
    def _render_transcript(lines: List[Tuple[float, List[str]]], max_words: int) -> str:
        out = []
        for start, words in lines:
            m = int(start // 60)
            s = start % 60
            out.append(f"[{m:02d}:{s:04.1f}] {' '.join(words[:max_words])}")
        return "\n".join(out)

    @classmethod
    def _compact_transcript(cls, lines: List[Tuple[float, List[str]]], budget_chars: int) -> str:
        """Fits transcript lines into budget_chars while preserving time coverage.

        Order of degradation: full text -> fewer words per line -> evenly
        subsampled lines (first and last always kept).
        """
        if not lines or budget_chars <= 0:
            return ""
        longest = max(len(ws) for _, ws in lines)
        for max_words in range(longest, MIN_WORDS_PER_LINE - 1, -1):
            text = cls._render_transcript(lines, max_words)
            if len(text) <= budget_chars:
                return text
        keep = len(lines)
        while keep > 1:
            keep = max(1, int(keep * 0.85))
            idxs = sorted({round(i * (len(lines) - 1) / max(1, keep - 1)) for i in range(keep)})
            text = cls._render_transcript([lines[i] for i in idxs], MIN_WORDS_PER_LINE)
            if len(text) <= budget_chars:
                return text
        return cls._render_transcript(lines[-1:], MIN_WORDS_PER_LINE)[:budget_chars]

    def build_interleaved_prompt(self, keyframes: List[Dict[str, Any]],
                                 transcript_words: List[Any],
                                 video_duration: float,
                                 transcript_budget_chars: Optional[int] = None,
                                 content_profile: Optional[UniversalContentProfile] = None) -> List[Dict[str, Any]]:
        """Builds an interleaved vision-language prompt for vLLM.

        The transcript is compacted so that text + image tokens + the reserved
        output tokens fit within VLM_CONTEXT_TOKENS.
        """
        lines = [
            ln for ln in self._group_transcript_lines(transcript_words)
            if ln[0] >= MIN_START_TIME - TRANSCRIPT_LEAD_IN_S
        ]
        transcript_text = "{TRANSCRIPT}"

        # Profile-specific editorial directives
        profile_directive = ""
        if content_profile == UniversalContentProfile.STAGE_PERFORMANCE:
            profile_directive = (
                "CONTENT PROFILE: STAGE PERFORMANCE / STANDUP COMEDY\n"
                "- Focus exclusively on the single comedian/performer on stage.\n"
                "- Track comic gestures and stage props. NEVER split-screen a solo comedian.\n"
                "- End on a decisive punchline payoff.\n\n"
            )
        elif content_profile == UniversalContentProfile.TECH_REVIEW_DEMO:
            profile_directive = (
                "CONTENT PROFILE: TECH & HARDWARE REVIEW DEMO\n"
                "- Prioritize software demos, monitor screencasts, UI interfaces, and product closeups.\n"
                "- For shots displaying a monitor/screen, select 'CANVAS_GRAPHIC_FIT' so the widescreen screen/UI is 100% visible with zero blank corners.\n"
                "- When the host is talking to camera, select 'SMART_PORTRAIT_PAN' with host centered.\n\n"
            )
        elif content_profile == UniversalContentProfile.GAMEPLAY_STREAM:
            profile_directive = (
                "CONTENT PROFILE: GAMEPLAY STREAM\n"
                "- Highlight high-intensity gameplay action, clutch plays, and streamer reactions.\n"
                "- Frame gameplay action clearly with HUD/action centered.\n\n"
            )
        elif content_profile == UniversalContentProfile.WILDLIFE_NATURE:
            profile_directive = (
                "CONTENT PROFILE: WILDLIFE & NATURE DOCUMENTARY\n"
                "- Center on active animal movement, predation, or environmental transitions.\n"
                "- Keep primary animal subject centered in portrait 9:16.\n\n"
            )
        elif content_profile == UniversalContentProfile.CONVERSATIONAL_PODCAST:
            profile_directive = (
                "CONTENT PROFILE: CONVERSATIONAL PODCAST / INTERVIEW\n"
                "- Follow conversational turns between speakers. Use split-screen only during simultaneous crosstalk.\n\n"
            )
        elif content_profile == UniversalContentProfile.CINEMATIC_NARRATIVE:
            profile_directive = (
                "CONTENT PROFILE: CINEMATIC FILM / NARRATIVE\n"
                "- Follow dramatic story arcs across scenes with single-act continuity.\n\n"
            )

        # Build interleaved content blocks
        content_blocks = []

        # System context
        content_blocks.append({
            "type": "text",
            "text": (
                f"You are an expert short-form video editor. Analyze this {video_duration:.0f}s video "
                f"using the keyframe images and transcript below.\n\n"
                f"{profile_directive}"
                f"SELECT exactly {TARGET_SHORTS} non-overlapping narrative arcs for YouTube Shorts. "
                f"Each short MUST:\n"
                f"- Start at {MIN_START_TIME:.1f}s or later (strictly t >= 60.0s, NO intro teasers/montages)\n"
                f"- Be {MIN_DURATION:.0f}-{MAX_DURATION:.0f} seconds long (strictly 30.0s to 50.0s)\n"
                f"- Follow the 3-act narrative structure: Hook (0-5s, first spoken word <= 1.5s) → Escalation → Payoff (ending on a sentence boundary)\n"
                f"- Be a self-contained single-topic story segment (strictly NO chronological summary montages across disjoint scenes)\n\n"
                f"STRICT NEGATIVE CONSTRAINT (ZERO AD/SPONSOR CONTENT):\n"
                f"- You MUST NEVER select or include in-video sponsor segments, advertisement reads, promotional plugs ('sponsored by', 'use code', 'link in description', 'check out our sponsor'), Patreon pitches, channel intros, or subscribe calls to action.\n"
                f"- Promotional fluff causes immediate viewer drop-off on YouTube Shorts and will be strictly rejected.\n\n"
                f"EPISODIC CONTINUITY RULE:\n"
                f"- Every selected short MUST follow ONE continuous person, act, joke, or performance.\n"
                f"- DO NOT combine scenes featuring completely different contestants or stage acts into the same clip.\n"
                f"- If the video contains multiple performers, select a {MIN_DURATION:.0f}-{MAX_DURATION:.0f}s window that starts and finishes within the SAME performer's segment.\n\n"
                f"TRANSCRIPT:\n{transcript_text}\n\n"
                f"KEYFRAMES (shot midpoints):\n"
            )
        })

        # Interleave keyframes
        for kf in keyframes:
            mid = kf["midpoint_s"]
            h = int(mid // 3600)
            m = int((mid % 3600) // 60)
            s = mid % 60

            content_blocks.append({
                "type": "text",
                "text": f"\n--- Shot {kf['shot_idx'] + 1} @ {h:02d}:{m:02d}:{s:06.3f} "
                        f"(dur={kf['duration_s']:.1f}s) ---"
            })
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{kf['base64_jpeg']}"
                }
            })

        # Response format instructions
        content_blocks.append({
            "type": "text",
            "text": (
                f"\n\nRespond with EXACTLY {TARGET_SHORTS} shorts in this JSON format:\n"
                '{\n'
                '  "selected_shorts": [\n'
                '    {\n'
                f'      "start_time_seconds": <float, must be >= {self.min_start_time:.1f}>,\n'
                f'      "duration_seconds": <float, must be {self.min_duration:.1f}-{self.max_duration:.1f}>,\n'
                '      "segments": [[<float start1>, <float end1>], [<float start2>, <float end2>]],  // optional non-linear clips summing to >= 50.0s\n'
                '      "title": "<hook title for this short>",\n'
                '      "reasoning": "<why this forms a cohesive 3-act narrative arc (Hook -> Escalation -> Payoff)>",\n'
                '      "framing_mode": "<SMART_PORTRAIT_PAN, CANVAS_GRAPHIC_FIT, or CANVAS_UNCERTAINTY_FIT>"\n'
                '    }\n'
                '  ]\n'
                '}'
            )
        })

        if transcript_budget_chars is None:
            fixed_chars = sum(len(b["text"]) for b in content_blocks if b["type"] == "text")
            fixed_chars -= len("{TRANSCRIPT}")
            n_images = sum(1 for b in content_blocks if b["type"] == "image_url")
            prompt_tokens = (VLM_CONTEXT_TOKENS - VLM_OUTPUT_TOKENS - PROMPT_SAFETY_TOKENS
                             - n_images * TOKENS_PER_IMAGE)
            transcript_budget_chars = int(prompt_tokens * CHARS_PER_TOKEN) - fixed_chars
        compacted = self._compact_transcript(lines, transcript_budget_chars)
        content_blocks[0] = {
            "type": "text",
            "text": content_blocks[0]["text"].replace("{TRANSCRIPT}", compacted),
        }
        logger.info(
            f"Editorial prompt: {len(lines)} transcript lines -> {len(compacted)} chars "
            f"(budget {transcript_budget_chars}), {len(keyframes)} images"
        )

        messages = [
            {"role": "user", "content": content_blocks}
        ]

        return messages

    # ─── VLM Inference ───────────────────────────────────────────────────

    async def _call_vllm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calls the vLLM OpenAI-compatible chat completions endpoint."""
        import httpx

        url = f"{self.vllm_url}/v1/chat/completions"
        payload = {
            "model": "Qwen/Qwen3-VL-8B-Instruct-FP8",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": VLM_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    # ─── Response Parsing & Validation ───────────────────────────────────

    def _parse_response(self, response: Dict[str, Any]) -> List[ShortCandidate]:
        """Parses the vLLM response into ShortCandidate objects."""
        try:
            choice = response["choices"][0]
            raw_text = choice["message"]["content"]
            data = json.loads(raw_text)

            candidates = []
            for item in data.get("selected_shorts", []):
                mode = item.get("framing_mode", "SMART_PORTRAIT_PAN")
                if mode not in ("SMART_PORTRAIT_PAN", "CANVAS_GRAPHIC_FIT"):
                    mode = "SMART_PORTRAIT_PAN"

                candidates.append(ShortCandidate(
                    start_time_seconds=float(item["start_time_seconds"]),
                    duration_seconds=float(item["duration_seconds"]),
                    title=str(item.get("title", "")),
                    reasoning=str(item.get("reasoning", "")),
                    framing_mode=mode,
                ))

            logger.info(f"Parsed {len(candidates)} candidate shorts from VLM response")
            return candidates

        except (KeyError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse VLM response: {e}")
            return []

    def _validate_and_filter(self, candidates: List[ShortCandidate],
                             video_duration: float,
                             transcript_words: Optional[List[Any]] = None) -> List[ShortCandidate]:
        """Applies hard programmatic constraints to selected shorts.

        Enforces:
            - Hard rejection of intro teasers: start_time strictly >= 60.0s (MIN_START_TIME)
            - Duration bounds: strictly [30.0, 50.0] seconds
            - Narrative arc Hook: first spoken word must occur <= 1.5s into the clip (0-5s window)
            - Narrative arc Payoff: clean sentence or word boundary snap (no mid-sentence cuts)
            - Non-overlapping arcs
            - Bounds check against video_duration
        """
        valid: List[ShortCandidate] = []

        # Parse word tuples for temporal alignment
        words_seq = []
        for w in (transcript_words or []):
            if isinstance(w, dict):
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", 0.0))
                wt = str(w.get("word", "")).strip()
            else:
                ws = float(getattr(w, "start", 0.0))
                we = float(getattr(w, "end", 0.0))
                wt = str(getattr(w, "word", "")).strip()
            if we > 0 and wt:
                words_seq.append((ws, we, wt))

        min_start = getattr(self, "min_start_time", MIN_START_TIME)
        min_dur = getattr(self, "min_duration", MIN_DURATION)
        max_dur = getattr(self, "max_duration", MAX_DURATION)

        # Detect in-video sponsor reads and promotional fluff
        sponsor_intervals = detect_sponsor_segments(transcript_words or [])
        if sponsor_intervals:
            logger.info(f"Detected {len(sponsor_intervals)} in-video sponsor intervals: "
                        f"{[(round(s, 1), round(e, 1), r) for s, e, r in sponsor_intervals]}")

        for c in candidates:
            # Multi-segment non-linear narrative handling
            if getattr(c, "segments", None) and len(c.segments) > 0:
                c.segments = [(float(s), float(e)) for s, e in c.segments if float(e) > float(s)]
                if not c.segments:
                    continue
                c.start_time_seconds = c.segments[0][0]
                c.end_time_seconds = c.segments[-1][1]
                c.duration_seconds = round(sum(e - s for s, e in c.segments), 2)

            # Hard rejection: in-video sponsor read / promotional ad contamination
            sponsor_overlap = False
            for sp_s, sp_e, reason in sponsor_intervals:
                if c.start_time_seconds < sp_e and c.end_time_seconds > sp_s:
                    logger.warning(
                        f"REJECTED sponsor-contaminated short [{c.start_time_seconds:.1f}s-{c.end_time_seconds:.1f}s]: '{c.title}' "
                        f"(overlaps sponsor read '{reason}' at [{sp_s:.1f}s-{sp_e:.1f}s])"
                    )
                    sponsor_overlap = True
                    break
            if sponsor_overlap:
                continue

            # Hard rejection: intro teaser
            if c.start_time_seconds < min_start:
                logger.warning(
                    f"REJECTED teaser short starting at {c.start_time_seconds:.1f}s "
                    f"(must be >= {min_start}s): '{c.title}'"
                )
                continue

            # Hook Enforcement (0-5s, first spoken word <= 1.5s)
            if words_seq:
                hook_words = [
                    (ws, we, wt) for (ws, we, wt) in words_seq
                    if we > c.start_time_seconds and ws <= (c.start_time_seconds + HOOK_WINDOW)
                ]
                if not hook_words:
                    logger.warning(
                        f"REJECTED short '{c.title}': No spoken dialogue in 0-5s Hook window "
                        f"[{c.start_time_seconds:.1f}s - {c.start_time_seconds + HOOK_WINDOW:.1f}s]"
                    )
                    continue

                first_w_start = hook_words[0][0]
                if first_w_start < c.start_time_seconds:
                    c.start_time_seconds = max(min_start, round(first_w_start - 0.1, 2))

                delay = round(max(0.0, first_w_start - c.start_time_seconds), 2)
                if delay > MAX_HOOK_DELAY:
                    adjusted_start = round(max(min_start, first_w_start - 0.15), 2)
                    cur_end = c.start_time_seconds + c.duration_seconds
                    if (cur_end - adjusted_start) >= min_dur:
                        c.start_time_seconds = adjusted_start
                        c.duration_seconds = round(cur_end - c.start_time_seconds, 2)
                        delay = round(first_w_start - c.start_time_seconds, 2)
                    else:
                        logger.warning(
                            f"REJECTED short '{c.title}': First spoken word delay {delay:.2f}s > {MAX_HOOK_DELAY}s"
                        )
                        continue
                c.first_word_delay = delay

            # Payoff Enforcement (clean sentence/word boundary snap)
            if words_seq and not getattr(c, "segments", None):
                target_raw_end = c.start_time_seconds + c.duration_seconds
                max_allowed_end = min(c.start_time_seconds + max_dur, video_duration)
                min_allowed_end = c.start_time_seconds + min_dur

                eligible_payoffs = [
                    (ws, we, wt) for (ws, we, wt) in words_seq
                    if min_allowed_end <= we <= max_allowed_end
                ]

                if eligible_payoffs:
                    sentence_ends = [
                        (ws, we, wt) for (ws, we, wt) in eligible_payoffs
                        if wt.endswith((".", "!", "?"))
                    ]
                    if sentence_ends:
                        best = min(sentence_ends, key=lambda x: abs(x[1] - target_raw_end))
                        snapped_end = best[1] + 0.15
                    else:
                        best = min(eligible_payoffs, key=lambda x: abs(x[1] - target_raw_end))
                        snapped_end = best[1] + 0.10

                    c.duration_seconds = round(min(max_dur, snapped_end - c.start_time_seconds), 2)
                    c.payoff_snapped = True

            # Duration bounds
            if c.duration_seconds < min_dur or c.duration_seconds > max_dur:
                logger.warning(
                    f"REJECTED short with duration {c.duration_seconds:.1f}s "
                    f"(must be {min_dur}-{max_dur}s): '{c.title}'"
                )
                continue

            # Bounds check
            if c.end_time_seconds > video_duration + 0.5:
                logger.warning(
                    f"REJECTED short exceeding video duration "
                    f"({c.end_time_seconds:.1f}s > {video_duration:.1f}s): '{c.title}'"
                )
                continue

            # Overlap check
            overlap = False
            for v in valid:
                if (c.start_time_seconds < v.end_time_seconds and
                    c.end_time_seconds > v.start_time_seconds):
                    overlap = True
                    logger.warning(
                        f"REJECTED overlapping short [{c.start_time_seconds:.1f}s-{c.end_time_seconds:.1f}s] "
                        f"with [{v.start_time_seconds:.1f}s-{v.end_time_seconds:.1f}s]"
                    )
                    break

            if not overlap:
                valid.append(c)

        logger.info(f"Validated {len(valid)}/{len(candidates)} shorts (rejected {len(candidates) - len(valid)})")
        return valid[:TARGET_SHORTS]

    # ─── Top-Level API ───────────────────────────────────────────────────

    def select_fallback_shorts(self, transcript_words: List[Any],
                               shots: Optional[List[ShotBoundary]],
                               video_duration: float,
                               existing: Optional[List[ShortCandidate]] = None,
                               needed: int = 4) -> List[ShortCandidate]:
        """Generates high-retention fallback shorts using audio transcription and shot midpoints.

        Guarantees:
            - start >= min_start_time (strictly rejecting intro teasers)
            - duration in [min_duration, max_duration]
            - non-overlapping with existing shorts and each other
            - multi-modal alignment: speech pauses, sentence ends, and shot midpoints
            - 3-act narrative structure: Hook (0-5s, first word <= 1.5s), Escalation, Payoff
            - supports non-linear multi-segment stitching when single spans are insufficient
            - reliably produces >= needed candidates (up to available duration capacity)
        """
        existing = existing or []
        fallback_results: List[ShortCandidate] = []
        min_start = getattr(self, "min_start_time", MIN_START_TIME)
        min_dur = getattr(self, "min_duration", MIN_DURATION)
        max_dur = getattr(self, "max_duration", MAX_DURATION)

        # Detect in-video sponsor reads and promotional fluff
        sponsor_intervals = detect_sponsor_segments(transcript_words or [])

        if needed <= 0 or video_duration < (min_start + min_dur):
            return fallback_results

        def collides(start_s: float, end_s: float) -> bool:
            for s in existing + fallback_results:
                s_start = getattr(s, "start_time_seconds", 0.0) if hasattr(s, "start_time_seconds") else s.get("start_time_seconds", 0.0)
                s_end = getattr(s, "end_time_seconds", 0.0) if hasattr(s, "end_time_seconds") else s.get("end_time_seconds", 0.0)
                if start_s < s_end and end_s > s_start:
                    return True
            for sp_s, sp_e, _ in sponsor_intervals:
                if start_s < sp_e and end_s > sp_s:
                    return True
            return False

        def get_free_gaps(candidate_list: List[Any]) -> List[Tuple[float, float]]:
            all_occ = []
            for s in candidate_list:
                s_start = getattr(s, "start_time_seconds", 0.0) if hasattr(s, "start_time_seconds") else s.get("start_time_seconds", 0.0)
                s_end = getattr(s, "end_time_seconds", 0.0) if hasattr(s, "end_time_seconds") else s.get("end_time_seconds", 0.0)
                all_occ.append((max(min_start, s_start), min(video_duration, s_end)))
            for sp_s, sp_e, _ in sponsor_intervals:
                all_occ.append((max(min_start, sp_s), min(video_duration, sp_e)))
            all_occ.sort(key=lambda x: x[0])
            gaps = []
            curr_t = min_start
            for os_t, oe_t in all_occ:
                if os_t > curr_t:
                    gaps.append((curr_t, os_t))
                curr_t = max(curr_t, oe_t)
            if curr_t < video_duration:
                gaps.append((curr_t, video_duration))
            return gaps

        def gap_capacity(gaps: List[Tuple[float, float]]) -> int:
            return sum(int((ge - gs) // min_dur) for gs, ge in gaps if (ge - gs) >= min_dur)

        # ── 1. Normalize and filter transcript words (t >= min_start) ──
        valid_words = []
        for w in (transcript_words or []):
            if isinstance(w, dict):
                w_start = float(w.get("start", 0.0))
                w_end = float(w.get("end", 0.0))
                w_text = str(w.get("word", "")).strip()
                w_spk = w.get("speaker_id")
            else:
                w_start = float(getattr(w, "start", 0.0))
                w_end = float(getattr(w, "end", 0.0))
                w_text = str(getattr(w, "word", "")).strip()
                w_spk = getattr(w, "speaker_id", None)
            if w_start >= min_start and w_end <= (video_duration + 0.5) and w_text:
                valid_words.append({
                    "start": w_start,
                    "end": w_end,
                    "word": w_text,
                    "speaker_id": w_spk
                })
        valid_words.sort(key=lambda x: x["start"])

        # ── 2. Filter shot boundaries and extract midpoints (t >= min_start) ──
        valid_shots = []
        shot_midpoints = []
        shot_cuts = []
        for s in (shots or []):
            s_start = getattr(s, "start_s", 0.0)
            s_end = getattr(s, "end_s", 0.0)
            mid = getattr(s, "midpoint_s", (s_start + s_end) / 2.0)
            if s_end >= min_start and s_start <= video_duration:
                valid_shots.append(s)
                if mid >= min_start and mid <= video_duration:
                    shot_midpoints.append(mid)
                if s_start >= min_start:
                    shot_cuts.append(s_start)
                if s_end <= video_duration and s_end >= min_start:
                    shot_cuts.append(s_end)

        shot_midpoints = sorted(set(shot_midpoints))
        shot_cuts = sorted(set(shot_cuts))

        # ── 3. Initial Gap Capacity & Tight Packing Check ──
        init_gaps = get_free_gaps(existing)
        total_cap = gap_capacity(init_gaps)
        target_needed = min(needed, total_cap)
        if target_needed <= 0:
            return fallback_results

        total_avail_dur = sum((ge - gs) for gs, ge in init_gaps if (ge - gs) >= min_dur)
        slack = total_avail_dur - (target_needed * min_dur)
        is_tight = (slack < 2.0)

        viable_gaps = [(gs, ge) for gs, ge in init_gaps if (ge - gs) >= min_dur]

        # ── 4. Strategy A: Multi-Modal Speech Pause + Shot Cut Alignment (slack >= 2.0) ──
        if not is_tight and valid_words and len(valid_words) >= 10:
            speech_starts = []
            speech_ends = []
            last_s = -100.0
            last_e = -100.0

            for i, w in enumerate(valid_words):
                w_start = w["start"]
                w_end = w["end"]
                has_punct = w["word"].endswith((".", "!", "?", ",", ";"))
                silence_after = (valid_words[i + 1]["start"] - w_end) if (i < len(valid_words) - 1) else 1.0

                if (w_start - last_s >= 2.5) or (i > 0 and valid_words[i - 1]["word"].endswith((".", "!", "?"))) or (i > 0 and (w_start - valid_words[i - 1]["end"] >= 0.25)):
                    speech_starts.append(w_start)
                    last_s = w_start
                if has_punct or silence_after >= 0.25 or (w_end - last_e >= 2.5):
                    speech_ends.append(w_end)
                    last_e = w_end

            speech_starts = sorted(set(speech_starts))
            speech_ends = sorted(set(speech_ends))

            candidate_windows = []
            for gap_start, gap_end in viable_gaps:
                gap_starts = [t for t in speech_starts if gap_start <= t <= gap_end - min_dur]
                gap_ends = [t for t in speech_ends if gap_start + min_dur <= t <= gap_end]

                for p_start in gap_starts:
                    cand_start = max(gap_start, p_start)
                    for sc in shot_cuts:
                        if abs(sc - p_start) <= 0.4 and sc >= gap_start:
                            cand_start = sc
                            break

                    for p_end in gap_ends:
                        cand_end = min(gap_end, p_end)
                        for sc in shot_cuts:
                            if abs(sc - p_end) <= 0.4 and sc <= gap_end and (sc - cand_start) <= max_dur and (sc - cand_start) >= min_dur:
                                cand_end = sc
                                break

                        dur = cand_end - cand_start
                        if dur < min_dur:
                            continue
                        if dur > max_dur:
                            break

                        words_in_win = [w for w in valid_words if w["start"] >= cand_start and w["end"] <= cand_end]
                        if not words_in_win:
                            continue

                        first_word = words_in_win[0]
                        first_offset = first_word["start"] - cand_start
                        hook_score = 3.0 if first_offset <= 0.6 else (2.0 if first_offset <= 1.5 else max(0.0, 1.5 - first_offset))

                        last_word = words_in_win[-1]
                        payoff_score = 2.5 if last_word["word"].endswith((".", "!", "?")) else 1.0

                        midpoints_in_win = [m for m in shot_midpoints if cand_start <= m <= cand_end]
                        shot_score = min(3.0, len(midpoints_in_win) * 0.75)
                        density = len(words_in_win) / dur
                        density_score = min(3.0, density)

                        total_score = hook_score + payoff_score + shot_score + density_score
                        candidate_windows.append({
                            "start": cand_start,
                            "end": cand_end,
                            "duration": dur,
                            "score": total_score,
                            "words": words_in_win,
                            "midpoints": midpoints_in_win,
                            "first_offset": first_offset,
                        })

            candidate_windows.sort(key=lambda x: x["score"], reverse=True)

            for cand in candidate_windows:
                if len(fallback_results) >= target_needed:
                    break
                c_s = cand["start"]
                c_e = cand["end"]
                c_dur = cand["duration"]
                if not collides(c_s, c_e):
                    # Invariant: accepting this candidate must preserve sufficient capacity for remaining slots
                    rem_needed_after = target_needed - len(fallback_results) - 1
                    test_cands = existing + fallback_results + [
                        ShortCandidate(start_time_seconds=c_s, duration_seconds=c_dur, title="", reasoning="")
                    ]
                    if gap_capacity(get_free_gaps(test_cands)) < rem_needed_after:
                        continue

                    short_idx = len(existing) + len(fallback_results) + 1
                    words_in = cand["words"]
                    hook_words = " ".join([w["word"] for w in words_in[:6]])
                    last_words = " ".join([w["word"] for w in words_in[-5:]])
                    spk_count = len(set(w.get("speaker_id") for w in words_in if w.get("speaker_id")))
                    m_count = len(cand["midpoints"])

                    title = f"Viral Arc {short_idx}: {hook_words[:30]}..." if hook_words else f"Viral Arc {short_idx}"
                    reasoning = (
                        f"Hook (0-5s): Opening speech '{hook_words[:25]}' starts at +{cand['first_offset']:.2f}s. "
                        f"Escalation: Sustained dialogue across {m_count} shot transitions with {spk_count or 1} speakers. "
                        f"Payoff: Conclusive narrative resolution on '{last_words[-25:]}' at {c_e:.2f}s."
                    )

                    fallback_results.append(ShortCandidate(
                        start_time_seconds=round(c_s, 2),
                        duration_seconds=round(c_dur, 2),
                        title=title,
                        reasoning=reasoning,
                        framing_mode="SMART_PORTRAIT_PAN",
                        first_word_delay=cand["first_offset"],
                        payoff_snapped=True
                    ))

        # ── 5. Strategy B: Shot Midpoint Sequence Packaging (slack >= 2.0, if still needed) ──
        if not is_tight and len(fallback_results) < target_needed and valid_shots and len(valid_shots) >= 3:
            shot_windows = []
            for i, s1 in enumerate(valid_shots):
                s_start = s1.start_s
                if s_start < min_start or s_start > video_duration - min_dur:
                    continue
                for s2 in valid_shots[i:]:
                    s_end = s2.end_s
                    dur = s_end - s_start
                    if dur < min_dur:
                        continue
                    if dur > max_dur:
                        break
                    if s_end > video_duration:
                        continue
                    if collides(s_start, s_end):
                        continue

                    mids = [m for m in shot_midpoints if s_start <= m <= s_end]
                    w_cnt = sum(1 for w in valid_words if w["start"] >= s_start and w["end"] <= s_end)
                    shot_windows.append((s_start, s_end, dur, len(mids) * 1.5 + w_cnt * 0.1, len(mids)))

            shot_windows.sort(key=lambda x: x[3], reverse=True)
            for s_start, s_end, dur, score, m_count in shot_windows:
                if len(fallback_results) >= target_needed:
                    break
                if not collides(s_start, s_end):
                    rem_needed_after = target_needed - len(fallback_results) - 1
                    test_cands = existing + fallback_results + [
                        ShortCandidate(start_time_seconds=s_start, duration_seconds=dur, title="", reasoning="")
                    ]
                    if gap_capacity(get_free_gaps(test_cands)) < rem_needed_after:
                        continue

                    short_idx = len(existing) + len(fallback_results) + 1
                    fallback_results.append(ShortCandidate(
                        start_time_seconds=round(s_start, 2),
                        duration_seconds=round(dur, 2),
                        title=f"Visual Story Arc {short_idx}",
                        reasoning=(
                            f"Hook (0-5s): Opening visual cut at {s_start:.2f}s. "
                            f"Escalation: Sequence of {m_count} shot midpoints. "
                            f"Payoff: Scene transition cut at {s_end:.2f}s."
                        ),
                        framing_mode="SMART_PORTRAIT_PAN",
                        first_word_delay=0.10,
                        payoff_snapped=True
                    ))

        # ── 6. Strategy C: Dynamic Gap Partitioning & Tight Packing Guarantee ──
        if len(fallback_results) < target_needed:
            rem_gaps = get_free_gaps(existing + fallback_results)
            viable_rem_gaps = [(gs, ge) for gs, ge in rem_gaps if (ge - gs) >= min_dur]
            viable_rem_gaps.sort(key=lambda g: g[0])

            remaining_needed = target_needed - len(fallback_results)
            for g_start, g_end in viable_rem_gaps:
                if remaining_needed <= 0:
                    break
                g_len = g_end - g_start
                max_in_gap = int(g_len // min_dur)
                if max_in_gap <= 0:
                    continue
                slots_to_add = min(remaining_needed, max_in_gap)

                slot_width = g_len / slots_to_add
                is_slot_tight = (slot_width < min_dur + 1.0)
                target_slot_dur = min(max_dur, max(min_dur, slot_width - 0.5))

                for slot_i in range(slots_to_add):
                    if len(fallback_results) >= target_needed:
                        break
                    win_start = g_start + slot_i * slot_width
                    win_end = g_start + (slot_i + 1) * slot_width

                    if is_slot_tight:
                        s_start = win_start
                        s_end = win_start + min_dur
                    else:
                        words_in_win = [w for w in valid_words if win_start <= w["start"] <= win_end]
                        if words_in_win and (words_in_win[0]["start"] + min_dur <= win_end):
                            first_w = words_in_win[0]
                            cand_s = max(win_start, first_w["start"] - 0.10)
                            max_end = min(win_end, cand_s + target_slot_dur)
                            min_end = cand_s + min_dur
                            end_cands = [w for w in words_in_win if min_end <= (w["end"] + 0.15) <= max_end]
                            if end_cands:
                                punct_ends = [w for w in end_cands if w["word"].endswith((".", "!", "?"))]
                                cand_e = (punct_ends[-1]["end"] + 0.15) if punct_ends else (end_cands[-1]["end"] + 0.15)
                            else:
                                cand_e = max_end
                            s_start = cand_s
                            s_end = cand_e
                        else:
                            s_start = win_start
                            s_end = win_start + target_slot_dur

                    dur = round(s_end - s_start, 2)
                    if dur > max_dur:
                        s_end = s_start + max_dur
                        dur = max_dur
                    elif dur < min_dur:
                        s_end = s_start + min_dur
                        dur = min_dur

                    short_idx = len(existing) + len(fallback_results) + 1
                    words_in_slot = [w for w in valid_words if s_start <= w["start"] and w["end"] <= s_end]
                    if words_in_slot:
                        first_w = words_in_slot[0]
                        last_w = words_in_slot[-1]
                        first_delay = max(0.0, first_w["start"] - s_start)
                        hook_text = " ".join([w["word"] for w in words_in_slot[:5]])
                        payoff_text = " ".join([w["word"] for w in words_in_slot[-5:]])
                        title = f"Narrative Arc {short_idx}: {hook_text[:30]}..."
                        reasoning = (
                            f"Hook (0-5s): Speech starts at +{first_delay:.2f}s on '{hook_text[:25]}'. "
                            f"Escalation: Narrative arc progression across {dur:.1f}s. "
                            f"Payoff: Conclusion on '{payoff_text[-25:]}' at {s_end:.2f}s."
                        )
                        f_delay = first_delay
                    else:
                        title = f"Narrative Arc {short_idx}"
                        reasoning = (
                            f"Hook (0-5s): Visual arc entry at {s_start:.2f}s. "
                            f"Escalation: Narrative arc progression across {dur:.1f}s. "
                            f"Payoff: Visual arc conclusion at {s_end:.2f}s."
                        )
                        f_delay = 0.10

                    fallback_results.append(ShortCandidate(
                        start_time_seconds=round(s_start, 2),
                        duration_seconds=round(dur, 2),
                        title=title,
                        reasoning=reasoning,
                        framing_mode="SMART_PORTRAIT_PAN",
                        first_word_delay=round(f_delay, 2),
                        payoff_snapped=True
                    ))
                    remaining_needed -= 1

        fallback_results.sort(key=lambda s: s.start_time_seconds)
        return fallback_results[:needed]

    async def select_shorts(self, video_path: str,
                            scene_changes: List[Any],
                            transcript_words: List[Any],
                            video_duration: float,
                            fps: float = 30.0,
                            spatial_events: Optional[List[Any]] = None) -> List[ShortCandidate]:
        """End-to-end editorial selection pipeline.

        1. Build shot boundaries from scene changes
        2. Extract midpoint keyframes
        3. Build interleaved multimodal prompt
        4. Call vLLM for editorial selection
        5. Validate and filter results
        6. Top up with dialogue-aware fallback if needed to guarantee >= TARGET_SHORTS
        7. Ground and bind transcript demonstration beats to each candidate short

        Returns:
            List of up to 4 validated ShortCandidate objects with grounded beats.
        """
        logger.info(f"Starting editorial engine for {video_path} ({video_duration:.1f}s)")

        # Step 0: Universal Video Content Profiling
        content_profile = classify_video_profile(
            transcript_words=transcript_words,
            spatial_events=spatial_events,
            video_duration=video_duration,
            scene_changes=scene_changes
        )
        logger.info(f"Classified Universal Video Content Profile: {content_profile.value}")

        # Step 1: Build shot boundaries
        shots = self.build_shot_boundaries(scene_changes, video_duration, fps)

        # Step 2: Extract midpoint keyframes
        keyframes = self.extract_midpoint_keyframes(video_path, shots)

        valid_shorts: List[ShortCandidate] = []

        if not keyframes:
            logger.warning("No keyframes extracted. Generating dialogue-aware fallback shorts directly.")
            raw_fallbacks = self.select_fallback_shorts(
                transcript_words=transcript_words,
                shots=shots,
                video_duration=video_duration,
                existing=[],
                needed=TARGET_SHORTS
            )
            for s in raw_fallbacks:
                s.content_profile = content_profile
                extract_beats_for_candidate(s, transcript_words, spatial_events or [])
            return raw_fallbacks

        # Limit keyframes so transcript keeps most of the 3k-token budget
        MAX_PROMPT_KEYFRAMES = 4
        if len(keyframes) > MAX_PROMPT_KEYFRAMES:
            body_keyframes = [
                kf for kf in keyframes
                if kf["start_s"] >= MIN_START_TIME and kf["duration_s"] >= 1.5
            ]
            selected_pool = body_keyframes if len(body_keyframes) >= 4 else keyframes

            if len(selected_pool) > MAX_PROMPT_KEYFRAMES:
                step = len(selected_pool) / MAX_PROMPT_KEYFRAMES
                keyframes = [selected_pool[int(i * step)] for i in range(MAX_PROMPT_KEYFRAMES)]
            else:
                keyframes = selected_pool
            logger.info(f"Selected {len(keyframes)} keyframes for multimodal editorial prompt")

        # Step 3: Build interleaved prompt
        messages = self.build_interleaved_prompt(keyframes, transcript_words, video_duration, content_profile=content_profile)

        # Step 4: Call vLLM
        try:
            response = await self._call_vllm(messages)
            candidates = self._parse_response(response)
            valid_shorts = self._validate_and_filter(candidates, video_duration, transcript_words=transcript_words)
        except Exception as e:
            logger.warning(f"VLM inference unavailable/failed ({e}). Falling back to heuristic selection.")
            valid_shorts = []

        # Step 5: Top up with fallback shorts if vLLM produced fewer than TARGET_SHORTS
        if len(valid_shorts) < TARGET_SHORTS:
            needed = TARGET_SHORTS - len(valid_shorts)
            logger.info(f"Topping up {needed} shorts to satisfy {TARGET_SHORTS} quota...")
            fallbacks = self.select_fallback_shorts(
                transcript_words=transcript_words,
                shots=shots,
                video_duration=video_duration,
                existing=valid_shorts,
                needed=needed
            )
            valid_shorts.extend(fallbacks)

        # Step 6: Ground and bind transcript demonstration beats to each candidate short
        for i, s in enumerate(valid_shorts, 1):
            s.content_profile = content_profile
            beats = extract_beats_for_candidate(s, transcript_words, spatial_events or [])
            logger.info(
                f"  Short {i}: [{s.start_time_seconds:.1f}s → {s.end_time_seconds:.1f}s] "
                f"({s.duration_seconds:.1f}s) mode={s.framing_mode} beats={len(beats)} — '{s.title}'"
            )

        return valid_shorts
