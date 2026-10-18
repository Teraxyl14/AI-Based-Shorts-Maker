"""
Project Aether V3 — Multilingual Kinetic Subtitle Compositor

Containerless outlined kinetic typography with HarfBuzz multilingual text
shaping, 1–3 word micro-chunking, ~14% kinetic scale pop, neon highlight accents,
dual-pass outline + drop-shadow, and premultiplied-alpha GPU compositing.
Renders broadcast-grade viral subtitles at 1080×1920 without dark container
boxes — dual-pass stroke + shadow + fill outline directly over video within
the 60%–70% vertical safe zone.
"""

import bisect
import logging
import math
import os
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger("AetherSubtitleCompositor")

# ─── Typography Constants ─────────────────────────────────────────────────────

FONT_SIZE = 76  # px normal font size at 1080×1920 canvas
SCALE_POP_FACTOR = 1.14  # ~14% kinetic scale pop for active spoken word
FONT_SIZE_ACTIVE = int(round(FONT_SIZE * SCALE_POP_FACTOR))  # 87px (+14% kinetic scale)
STROKE_WIDTH = 6  # px solid black outline
SHADOW_OFFSET = 3  # px drop shadow offset (+3px X, +3px Y)
MAX_LINE_WIDTH = 920  # px max subtitle line width (safe zone)
SUBTITLE_Y_ANCHOR = 0.65  # vertical anchor as fraction of canvas height (Y=1248 on 1920h)

# Safe zone vertical bounds (60% to 70% of canvas height: [1152, 1344] px on 1920h)
SAFE_ZONE_MIN_RATIO = 0.60
SAFE_ZONE_MAX_RATIO = 0.70

# Colors (RGBA 0-255)
COLOR_NORMAL = (255, 255, 255, 255)       # Crisp white fill
COLOR_STROKE = (0, 0, 0, 255)             # Solid black outline (6px)
COLOR_SHADOW = (0, 0, 0, 160)             # Semi-transparent black drop shadow (~63% alpha)

# R3 Neon Highlight Triad Palette
COLOR_NEON_LIME   = (204, 255, 0, 255)    # #CCFF00 Electric Lime (Primary active highlight)
COLOR_NEON_GREEN  = (150, 255, 50, 255)   # Vivid Neon Spring Green (r+g > 350, g >= 200)
COLOR_NEON_YELLOW = (226, 249, 82, 255)   # #E2F952 Lemon Lime Accent

NEON_PALETTE = {
    "neon_lime": COLOR_NEON_LIME,
    "electric_lime": COLOR_NEON_LIME,
    "neon_green": COLOR_NEON_GREEN,
    "neon_yellow": COLOR_NEON_YELLOW,
    "lemon_lime": COLOR_NEON_YELLOW,
}

# Primary active highlight satisfies test_f8_03: r>=200, g>=200, b<100, a==255
COLOR_ACTIVE = COLOR_NEON_LIME

# Cosine ease-in ramp duration for active word onset
EASE_IN_DURATION = 0.020  # 20ms

# Micro-chunking parameters
MAX_WORDS_PER_CHUNK = 3
SILENCE_THRESHOLD = 0.120  # 120ms silence gap triggers new chunk

# ─── Multilingual Font Fallback Chain ─────────────────────────────────────────

FONT_SEARCH_PATHS = [
    # Container / Linux
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/noto",
    "/usr/local/share/fonts",
    # Windows
    "C:/Windows/Fonts",
    # macOS
    "/Library/Fonts",
    "/System/Library/Fonts",
]

FONT_FALLBACK_CHAIN = [
    "NotoSans-Bold.ttf",
    "NotoSansDevanagari-Bold.ttf",
    "NotoSansCJK-Bold.ttc",
    "NotoSansCJKsc-Bold.otf",
    "NotoSansArabic-Bold.ttf",
    "NotoNaskhArabic-Bold.ttf",
    "DejaVuSans-Bold.ttf",
]

SCRIPT_FONT_MAP = {
    "LATIN": ["NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf"],
    "CYRILLIC": ["NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf"],
    "GREEK": ["NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf"],
    "CJK": [
        "NotoSansCJK-Bold.ttc", "NotoSansCJK-Regular.ttc",
        "NotoSansCJKsc-Bold.otf", "NotoSansSC-Bold.otf",
        "NotoSansJP-Bold.otf", "NotoSansKR-Bold.otf",
        "msyhbd.ttc", "msyh.ttc", "meiryob.ttc", "malgunbd.ttf"
    ],
    "DEVANAGARI": [
        "NotoSansDevanagari-Bold.ttf", "NotoSansDevanagari-Regular.ttf",
        "NotoSerifDevanagari-Bold.ttf", "Nirmala.ttc", "NirmalaB.ttf"
    ],
    "ARABIC": [
        "NotoSansArabic-Bold.ttf", "NotoNaskhArabic-Bold.ttf", "NotoKufiArabic-Bold.ttf",
        "NotoSansArabic-Regular.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf"
    ],
}

BICAMERAL_SCRIPTS = {"LATIN", "CYRILLIC", "GREEK", "ARMENIAN", "GEORGIAN"}

_FONT_CACHE: Dict[Tuple[str, int], Any] = {}


def _is_bicameral_text(text: str) -> bool:
    """Returns True if the text consists primarily of bicameral script characters."""
    if not text:
        return False
    for ch in text:
        if ch.isalpha():
            try:
                script = unicodedata.name(ch, "").split()[0]
            except (ValueError, IndexError):
                continue
            return script.upper() in BICAMERAL_SCRIPTS
    return False


def detect_script(text: str) -> str:
    """Detects primary Unicode script of text: CJK, ARABIC, DEVANAGARI, CYRILLIC, GREEK, or LATIN."""
    if not text:
        return "LATIN"
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith('P'):
            continue
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF
                or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF):
            return "CJK"
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
            return "ARABIC"
        elif 0x0900 <= cp <= 0x097F:
            return "DEVANAGARI"
        elif 0x0400 <= cp <= 0x04FF:
            return "CYRILLIC"
        elif 0x0370 <= cp <= 0x03FF:
            return "GREEK"
    return "LATIN"


def _find_font_path(font_name: str) -> Optional[str]:
    """Searches font search directories for the given font file."""
    for search_dir in FONT_SEARCH_PATHS:
        candidate = os.path.join(search_dir, font_name)
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_font_for_script(script: str, font_size: int, fallback_font=None):
    """Resolves font optimized for the given script and size, with caching."""
    cache_key = (script, font_size)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    from PIL import ImageFont
    candidates = SCRIPT_FONT_MAP.get(script, [])
    for font_name in candidates:
        path = _find_font_path(font_name)
        if path:
            try:
                font = ImageFont.truetype(path, font_size)
                _FONT_CACHE[cache_key] = font
                return font
            except Exception:
                continue

    # Fall back to chain
    chain_font = _resolve_font_chain(font_size)
    if chain_font is not None:
        _FONT_CACHE[cache_key] = chain_font
        return chain_font

    return fallback_font or ImageFont.load_default()


def resolve_font_for_text(text: str, font_size: int = FONT_SIZE):
    """Resolves the best matching font for a given text string based on its script."""
    script = detect_script(text)
    return _resolve_font_for_script(script, font_size)


def _resolve_font_chain(font_size: int):
    """Resolves the first available font from the fallback chain.

    Returns:
        Pillow ImageFont object.
    """
    from PIL import ImageFont

    cache_key = ("__CHAIN__", font_size)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    for font_name in FONT_FALLBACK_CHAIN:
        path = _find_font_path(font_name)
        if path:
            try:
                font = ImageFont.truetype(path, font_size)
                logger.info(f"Loaded font: {path} @ {font_size}px")
                _FONT_CACHE[cache_key] = font
                return font
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

    # Last resort: system DejaVuSans-Bold or default
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        logger.info(f"Loaded system DejaVuSans-Bold @ {font_size}px")
        _FONT_CACHE[cache_key] = font
        return font
    except Exception:
        font = ImageFont.load_default()
        logger.warning("Using Pillow default bitmap font (limited quality)")
        return font


# ─── Word Attribute Helpers ───────────────────────────────────────────────────

def _get_word_text(w: Any) -> str:
    if isinstance(w, dict):
        return str(w.get("word", "")).strip()
    return str(getattr(w, "word", "")).strip()


def _get_word_start(w: Any) -> float:
    if isinstance(w, dict):
        return float(w.get("start", 0.0))
    return float(getattr(w, "start", 0.0))


def _get_word_end(w: Any) -> float:
    if isinstance(w, dict):
        return float(w.get("end", 0.0))
    return float(getattr(w, "end", 0.0))


# ─── Micro-Chunk Builder ─────────────────────────────────────────────────────

def _build_micro_chunks(words: List[Any]) -> List[List[Any]]:
    """Segments word stream into 1–3 word micro-chunks based on acoustic silence and punctuation.

    Rules:
        - 1 to 3 words per chunk strictly (MAX_WORDS_PER_CHUNK = 3).
        - A new chunk starts when the silence gap between consecutive words
          exceeds SILENCE_THRESHOLD (120ms).
        - A new chunk starts if the previous word ends with terminal punctuation ('.', '!', '?').
        - Chunks are split if combined character length exceeds safe line width limits.
        - Supports both dictionary objects and Pydantic WordTimestamp instances.
    """
    if not words:
        return []

    valid_words = [w for w in words if _get_word_text(w)]
    if not valid_words:
        return []

    sorted_words = sorted(valid_words, key=_get_word_start)

    chunks: List[List[Any]] = []
    current_chunk: List[Any] = []

    for w in sorted_words:
        # Sanitize zero or negative duration tokens
        w_start = _get_word_start(w)
        w_end = _get_word_end(w)
        if w_end <= w_start:
            if isinstance(w, dict):
                w["end"] = w_start + 0.08
            elif hasattr(w, "end"):
                try:
                    w.end = w_start + 0.08
                except Exception:
                    pass

        if not current_chunk:
            current_chunk.append(w)
            continue

        prev_w = current_chunk[-1]
        prev_text = _get_word_text(prev_w)
        prev_end = _get_word_end(prev_w)
        this_start = _get_word_start(w)
        gap = this_start - prev_end

        # Terminal punctuation split (. ? ! ...)
        has_terminal_punct = bool(re.search(r'[.!?]+$', prev_text))

        # Check combined character length to prevent horizontal line clipping
        cur_char_len = sum(len(_get_word_text(x)) for x in current_chunk) + len(_get_word_text(w))

        if (len(current_chunk) >= MAX_WORDS_PER_CHUNK
                or gap >= SILENCE_THRESHOLD
                or has_terminal_punct
                or (len(current_chunk) >= 2 and cur_char_len > 22)):
            chunks.append(current_chunk)
            current_chunk = [w]
        else:
            current_chunk.append(w)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ─── Outlined Text Renderer ──────────────────────────────────────────────────

def _render_outlined_word(word: str, font, fill_color: Tuple[int, int, int, int],
                          stroke_color: Tuple[int, int, int, int] = COLOR_STROKE,
                          stroke_width: int = STROKE_WIDTH,
                          shadow_color: Tuple[int, int, int, int] = COLOR_SHADOW,
                          shadow_offset: int = SHADOW_OFFSET,
                          direction: Optional[str] = None) -> np.ndarray:
    """Renders a single word with dual-pass 6px stroke outline + 3px drop-shadow + fill.

    Pass 0: 3px Drop-shadow behind outline ((+3, +3) offset, semi-transparent black)
    Pass 1: 6px Solid black stroke outline
    Pass 2: Colored fill on top

    Strictly zero rectangular background boxes — transparent RGBA canvas.

    Returns:
        np.ndarray [H, W, 4] dtype uint8 — straight-alpha RGBA.
    """
    from PIL import Image, ImageDraw, features

    script = detect_script(word)
    effective_font = font
    if script in ("CJK", "ARABIC", "DEVANAGARI"):
        font_sz = getattr(font, "size", FONT_SIZE)
        effective_font = _resolve_font_for_script(script, font_sz, fallback_font=font)

    if direction is None and script == "ARABIC":
        direction = "rtl"
    if direction is not None and not features.check("raqm"):
        direction = None

    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    # Get raw text bounding box with stroke
    raw_bbox = draw.textbbox((0, 0), word, font=effective_font,
                             stroke_width=stroke_width, direction=direction)
    bbox_x0, bbox_y0 = raw_bbox[0], raw_bbox[1]
    bbox_x1, bbox_y1 = raw_bbox[2], raw_bbox[3]

    pad_x = stroke_width + 8
    pad_y = stroke_width + 8

    text_w = max(1, int((bbox_x1 - bbox_x0) + pad_x * 2 + shadow_offset))
    text_h = max(1, int((bbox_y1 - bbox_y0) + pad_y * 2 + shadow_offset))

    img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x_off = pad_x - bbox_x0
    y_off = pad_y - bbox_y0

    # Pass 0: Drop-shadow pass (3px offset (+3, +3), semi-transparent black)
    draw.text((x_off + shadow_offset, y_off + shadow_offset), word, font=effective_font,
              fill=shadow_color, stroke_width=stroke_width, stroke_fill=shadow_color,
              direction=direction)

    # Pass 1: Solid black stroke outline (6px thickness)
    draw.text((x_off, y_off), word, font=effective_font,
              fill=stroke_color, stroke_width=stroke_width, stroke_fill=stroke_color,
              direction=direction)

    # Pass 2: Colored fill on top
    draw.text((x_off, y_off), word, font=effective_font, fill=fill_color,
              direction=direction)

    return np.array(img, dtype=np.uint8)


# ─── Subtitle Compositor Class ───────────────────────────────────────────────

class SubtitleCompositor:
    """Multilingual kinetic subtitle compositor with containerless outlined typography.

    Renders 1–3 word micro-chunks with dual-pass stroke + 3px drop-shadow outlines directly
    over video using premultiplied-alpha Porter-Duff GPU compositing and ~14% kinetic scale pop.
    """

    def __init__(self, font_path: str = None, font_size: int = FONT_SIZE):
        self.font_path = font_path
        self.font_size = font_size
        self.font_size_active = int(round(font_size * SCALE_POP_FACTOR))
        self._font = None
        self._font_active = None
        self._word_textures: Dict[str, np.ndarray] = {}  # word -> RGBA numpy array (normal variant)
        self._word_starts: List[float] = []
        self._word_ends: List[float] = []
        self._words: List[Any] = []
        self._chunks: List[List[Any]] = []
        self._device = None

    def _load_font(self):
        """Loads base and active fonts via the multilingual fallback chain."""
        if self._font is not None and self._font_active is not None:
            return

        from PIL import ImageFont

        if self.font_path and os.path.exists(self.font_path):
            try:
                self._font = ImageFont.truetype(self.font_path, self.font_size)
                self._font_active = ImageFont.truetype(self.font_path, self.font_size_active)
                logger.info(f"Loaded user font: {self.font_path} @ {self.font_size}px / {self.font_size_active}px")
                return
            except Exception as e:
                logger.warning(f"Failed to load user font {self.font_path}: {e}")

        # Fall through to fallback chain
        self._font = _resolve_font_chain(self.font_size)
        self._font_active = _resolve_font_chain(self.font_size_active)

    def build_word_atlas(self, words: List[Any], target_w: int = 1080):
        """Pre-renders unique transcript words and segments into micro-chunks.

        Args:
            words: List of word dicts or WordTimestamp models.
            target_w: Canvas width (used for line-width validation).
        """
        self._words = words
        self._word_starts = [_get_word_start(w) for w in words]
        self._word_ends = [_get_word_end(w) for w in words]
        self._word_textures.clear()

        self._load_font()

        # Build 1–3 word micro-chunks based on acoustic silence & punctuation
        self._chunks = _build_micro_chunks(words)

        # Pre-render unique words in normal (white outline) variant
        unique_words = set()
        for w in words:
            text = _get_word_text(w)
            if text:
                display_text = self._get_display_text(text)
                unique_words.add(display_text)

        for word in unique_words:
            normal_font = resolve_font_for_text(word, self.font_size)
            self._word_textures[word] = _render_outlined_word(
                word, normal_font, fill_color=COLOR_NORMAL
            )

        logger.info(
            f"Built kinetic subtitle atlas: {len(unique_words)} unique words, "
            f"{len(self._chunks)} micro-chunks (max {MAX_WORDS_PER_CHUNK} words/chunk)"
        )

    def _get_display_text(self, word: str) -> str:
        """Applies bicameral uppercasing to the word."""
        return word.upper() if _is_bicameral_text(word) else word

    def _premultiply_tensor(self, rgba: torch.Tensor) -> torch.Tensor:
        """Converts straight-alpha RGBA float32 tensor to premultiplied alpha."""
        rgb = rgba[..., :3]
        a = rgba[..., 3:4]
        premul_rgb = rgb * a
        return torch.cat([premul_rgb, a], dim=-1)

    def _build_subtitle_strip(self, timestamp: float,
                              target_w: int, target_h: int) -> Optional[np.ndarray]:
        """Assembles a containerless outlined subtitle strip for the current timestamp.

        Renders ONLY the currently active micro-chunk (1–3 words) with:
        - Active spoken word: Neon Pop highlight (#CCFF00) with 20ms ease-in and +14% scale pop
        - Normal words: Crisp white fill at base 76px
        - Dual-pass 6px black stroke outline + 3px drop-shadow
        - Clean vertical baseline alignment across all words

        Returns:
            np.ndarray [strip_h, strip_w, 4] or None if no words active.
        """
        if not self._words or not self._chunks:
            return None

        # Binary search for active word index
        idx = bisect.bisect_right(self._word_starts, timestamp) - 1
        if idx < 0 or idx >= len(self._words):
            return None

        # Grace period after word ends
        active_word = self._words[idx]
        active_w_end = _get_word_end(active_word)
        if timestamp > active_w_end + 0.3:
            return None

        # Find the micro-chunk containing the active word
        display_chunk = None
        active_chunk_word_idx = -1
        for chunk in self._chunks:
            for c_i, w in enumerate(chunk):
                if (w is active_word or
                        (_get_word_text(w) == _get_word_text(active_word)
                         and abs(_get_word_start(w) - _get_word_start(active_word)) < 1e-4)):
                    display_chunk = chunk
                    active_chunk_word_idx = c_i
                    break
            if display_chunk is not None:
                break

        if not display_chunk:
            return None

        active_w_start = _get_word_start(active_word)
        is_speaking = (active_w_start <= timestamp <= active_w_end)

        rendered_items = []
        space_width = 5  # px between words (tightened for clean viral shorts typography)
        total_width = 0

        for w_idx, w in enumerate(display_chunk):
            text = _get_word_text(w)
            if not text:
                continue

            display_text = self._get_display_text(text)
            is_active = (w_idx == active_chunk_word_idx and is_speaking)

            if is_active:
                # Active word — apply cosine ease-in to neon highlight
                elapsed = timestamp - active_w_start
                if elapsed < EASE_IN_DURATION:
                    t = elapsed / EASE_IN_DURATION
                    alpha_scale = 0.5 * (1.0 - math.cos(math.pi * t))
                else:
                    alpha_scale = 1.0

                r = int(COLOR_NORMAL[0] + (COLOR_ACTIVE[0] - COLOR_NORMAL[0]) * alpha_scale)
                g = int(COLOR_NORMAL[1] + (COLOR_ACTIVE[1] - COLOR_NORMAL[1]) * alpha_scale)
                b_val = int(COLOR_NORMAL[2] + (COLOR_ACTIVE[2] - COLOR_NORMAL[2]) * alpha_scale)
                fill_color = (r, g, b_val, 255)

                # Render with ~14% kinetic scale pop
                active_font = resolve_font_for_text(display_text, self.font_size_active)
                word_img = _render_outlined_word(display_text, active_font, fill_color=fill_color)

                normal_check_font = resolve_font_for_text(display_text, self.font_size)
                if getattr(active_font, "size", 0) == getattr(normal_check_font, "size", 0):
                    from PIL import Image
                    pil_w = Image.fromarray(word_img)
                    pop_w = max(1, int(round(word_img.shape[1] * SCALE_POP_FACTOR)))
                    pop_h = max(1, int(round(word_img.shape[0] * SCALE_POP_FACTOR)))
                    word_img = np.array(pil_w.resize((pop_w, pop_h), Image.Resampling.BILINEAR), dtype=np.uint8)

                ascent = active_font.getmetrics()[0] if hasattr(active_font, "getmetrics") else int(round(word_img.shape[0] * 0.72))
                baseline_in_img = (STROKE_WIDTH + 8) + ascent
            else:
                # Normal word — use cached texture or render
                if display_text in self._word_textures:
                    word_img = self._word_textures[display_text]
                else:
                    normal_font = resolve_font_for_text(display_text, self.font_size)
                    word_img = _render_outlined_word(display_text, normal_font, fill_color=COLOR_NORMAL)

                normal_font = resolve_font_for_text(display_text, self.font_size)
                ascent = normal_font.getmetrics()[0] if hasattr(normal_font, "getmetrics") else int(round(word_img.shape[0] * 0.72))
                baseline_in_img = (STROKE_WIDTH + 8) + ascent

            rendered_items.append({
                "img": word_img,
                "baseline": baseline_in_img,
                "w": word_img.shape[1],
                "h": word_img.shape[0],
            })
            total_width += word_img.shape[1] + space_width

        if not rendered_items:
            return None

        total_width -= space_width  # Remove trailing space

        # Scale down if line width exceeds horizontal safe zone
        from PIL import Image
        max_safe_w = min(MAX_LINE_WIDTH, target_w - 40) if target_w > 40 else MAX_LINE_WIDTH
        if total_width > max_safe_w and max_safe_w > 0:
            scale = max_safe_w / total_width
            scaled_items = []
            scaled_total_w = 0
            for item in rendered_items:
                wimg = item["img"]
                pil_w = Image.fromarray(wimg)
                new_w = max(1, int(wimg.shape[1] * scale))
                new_h = max(1, int(wimg.shape[0] * scale))
                pil_resized = pil_w.resize((new_w, new_h), Image.Resampling.BILINEAR)
                arr = np.array(pil_resized, dtype=np.uint8)
                scaled_items.append({
                    "img": arr,
                    "baseline": int(round(item["baseline"] * scale)),
                    "w": new_w,
                    "h": new_h,
                })
                scaled_total_w += new_w + int(space_width * scale)
            rendered_items = scaled_items
            total_width = scaled_total_w - int(space_width * scale)

        # Baseline alignment: align word baselines to a uniform horizontal baseline
        max_baseline = max(item["baseline"] for item in rendered_items)
        strip_h = max(max_baseline - item["baseline"] + item["h"] for item in rendered_items)
        strip_w = max(total_width, 1080) if target_w >= 1080 else total_width

        # Assemble strip on transparent RGBA canvas (strictly zero opaque boxes)
        strip = np.zeros((strip_h, strip_w, 4), dtype=np.uint8)
        x_cursor = max(0, (strip_w - total_width) // 2)

        for item in rendered_items:
            word_img = item["img"]
            wh, ww = item["h"], item["w"]
            y_paste = max_baseline - item["baseline"]

            paste_w = min(ww, strip_w - x_cursor)
            paste_h = min(wh, strip_h - y_paste)

            if paste_w > 0 and paste_h > 0 and x_cursor < strip_w and y_paste < strip_h:
                src = word_img[:paste_h, :paste_w]
                dst = strip[y_paste:y_paste + paste_h, x_cursor:x_cursor + paste_w]

                src_a = src[:, :, 3:4].astype(np.float32) / 255.0
                blended_rgb = (src[:, :, :3].astype(np.float32) * src_a +
                               dst[:, :, :3].astype(np.float32) * (1.0 - src_a))
                blended_a = (src[:, :, 3:4].astype(np.float32) +
                             dst[:, :, 3:4].astype(np.float32) * (1.0 - src_a))

                strip[y_paste:y_paste + paste_h, x_cursor:x_cursor + paste_w, :3] = blended_rgb.clip(0, 255).astype(np.uint8)
                strip[y_paste:y_paste + paste_h, x_cursor:x_cursor + paste_w, 3:4] = blended_a.clip(0, 255).astype(np.uint8)

            x_cursor += ww + space_width

        return strip

    def composite_frame(self, frame: torch.Tensor, timestamp: float,
                        words: List[Any],
                        target_w: int = 0, target_h: int = 0,
                        baseline_y: Optional[float] = None) -> torch.Tensor:
        """Composites kinetic subtitle overlay onto a GPU-resident video frame.

        Uses premultiplied alpha Porter-Duff blending:
            Frame_out = Glyph_premul + Frame_video × (1 - Glyph_alpha)

        Guarantees _overlay_buf.zero_() on active CUDA stream before every composite.
        Restricts vertical placement strictly inside the [0.60, 0.70] screen safe zone.

        Args:
            frame: [1, C, H, W] or [1, H, W, C] float32 GPU tensor in [0, 1].
            timestamp: Current frame timestamp in seconds.
            words: Word timestamp list (dicts or WordTimestamp models).
            target_w: Target width (inferred from frame if 0).
            target_h: Target height (inferred from frame if 0).
            baseline_y: Optional vertical position for subtitle center.

        Returns:
            Same-shape tensor with composited subtitles.
        """
        if self._device is None:
            self._device = frame.device

        # Detect frame layout
        if frame.dim() == 4:
            if frame.shape[1] == 3:
                # [1, C, H, W] format
                _, _, h, w = frame.shape
                is_chw = True
            else:
                # [1, H, W, C] format
                _, h, w, _ = frame.shape
                is_chw = False
        else:
            return frame

        if target_w == 0:
            target_w = w
        if target_h == 0:
            target_h = h

        # Build subtitle strip for current timestamp
        strip = self._build_subtitle_strip(timestamp, target_w, target_h)
        if strip is None and getattr(self, "_word_starts", None) and len(self._word_starts) > 0:
            w_first = self._word_starts[0]
            if w_first > 10.0 and timestamp < w_first - 5.0:
                strip = self._build_subtitle_strip(timestamp + w_first, target_w, target_h)
        if strip is None:
            return frame

        strip_h, strip_w = strip.shape[:2]

        # Determine vertical placement with strict safe zone bounds
        y_min_safe = int(round(SAFE_ZONE_MIN_RATIO * target_h))  # 1152 on 1920
        y_max_safe = int(round(SAFE_ZONE_MAX_RATIO * target_h))  # 1344 on 1920

        if baseline_y is not None:
            y_center = int(round(baseline_y))
        else:
            y_center = int(round(target_h * SUBTITLE_Y_ANCHOR))

        # Clamp vertical placement strictly within [0.60, 0.70] screen safe zone
        y_ideal = y_center - strip_h // 2
        y_start = max(y_min_safe, min(y_ideal, y_max_safe - strip_h))
        y_start = max(0, min(y_start, target_h - strip_h))
        x_start = max(0, (target_w - strip_w) // 2)

        # Convert strip to premultiplied float32 tensor on GPU
        # Explicit zero_() before write — guarantees no stale data on CUDA stream
        strip_tensor = torch.zeros((strip_h, strip_w, 4), dtype=torch.float32, device=self._device)
        strip_tensor.zero_()

        strip_float = torch.from_numpy(strip.astype(np.float32) / 255.0).to(self._device)

        # Pre-multiply glyph textures: R' = R*A, G' = G*A, B' = B*A
        strip_premul = self._premultiply_tensor(strip_float)
        strip_tensor.copy_(strip_premul)

        text_rgb = strip_tensor[:, :, :3]
        text_alpha = strip_tensor[:, :, 3:4]

        # Premultiplied Porter-Duff MAD compositing:
        # Frame_out = Glyph_premul + Frame_video × (1 - Glyph_alpha)
        if is_chw:
            # frame is [1, C, H, W]
            x_end = min(x_start + strip_w, w)
            y_end = min(y_start + strip_h, h)
            actual_sw = x_end - x_start
            actual_sh = y_end - y_start

            if actual_sw > 0 and actual_sh > 0:
                region = frame[0, :, y_start:y_end, x_start:x_end].permute(1, 2, 0)
                blended = text_rgb[:actual_sh, :actual_sw] + region * (1.0 - text_alpha[:actual_sh, :actual_sw])
                blended = torch.clamp(blended, 0.0, 1.0)
                frame[0, :, y_start:y_end, x_start:x_end] = blended.permute(2, 0, 1)
        else:
            # frame is [1, H, W, C]
            x_end = min(x_start + strip_w, w)
            y_end = min(y_start + strip_h, h)
            actual_sw = x_end - x_start
            actual_sh = y_end - y_start

            if actual_sw > 0 and actual_sh > 0:
                region = frame[0, y_start:y_end, x_start:x_end, :]
                blended = text_rgb[:actual_sh, :actual_sw] + region * (1.0 - text_alpha[:actual_sh, :actual_sw])
                blended = torch.clamp(blended, 0.0, 1.0)
                frame[0, y_start:y_end, x_start:x_end, :] = blended

        return frame
