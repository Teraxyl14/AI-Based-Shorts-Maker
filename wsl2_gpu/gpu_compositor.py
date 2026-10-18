"""
Project Aether V3 — Pre-Rendered GPU Glyph Atlas Compositor

Zero-CPU kinetic typography with pre-rendered GPU atlases and stream-safe
MAD (Multiply-Add) alpha compositing. Pre-renders the complete subtitle atlas
(both normal and ~14% scaled neon-highlighted variants) at startup with
dual-pass 6px outline + 3px drop-shadow within the 60%–70% vertical safe zone.

Compositing formula (premultiplied alpha Porter-Duff Over):
    Frame_out = Text_premul + Frame_video · (1.0 - Text_alpha)

All glyph textures are stored in premultiplied alpha format:
    R' = R · A,  G' = G · A,  B' = B · A
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

logger = logging.getLogger("AetherGPUCompositor")

# ─── Typography Constants ─────────────────────────────────────────────────────

FONT_SIZE = 76          # px normal font size at 1080×1920 canvas
SCALE_POP_FACTOR = 1.14  # ~14% kinetic scale pop for active spoken word
FONT_SIZE_ACTIVE = int(round(FONT_SIZE * SCALE_POP_FACTOR))  # 87px (+14% kinetic scale)
STROKE_WIDTH = 6        # px solid black outline
SHADOW_OFFSET = 3       # px drop-shadow offset (+3px X, +3px Y)
MAX_LINE_WIDTH = 920    # px max subtitle line width (safe zone)
SUBTITLE_Y_ANCHOR = 0.65  # vertical anchor as fraction of canvas height (Y=1248 on 1920h)

# Safe zone vertical bounds (60% to 70% of canvas height: [1152, 1344] px on 1920h)
SAFE_ZONE_MIN_RATIO = 0.60
SAFE_ZONE_MAX_RATIO = 0.70

# Colors (RGBA 0-255)
COLOR_NORMAL = (255, 255, 255, 255)       # Crisp white fill
COLOR_STROKE = (0, 0, 0, 255)             # Solid black outline (6px)
COLOR_SHADOW = (0, 0, 0, 160)             # Semi-transparent black drop-shadow (~63% alpha)

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

_GPU_FONT_CACHE: Dict[Tuple[str, int], Any] = {}


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
    if cache_key in _GPU_FONT_CACHE:
        return _GPU_FONT_CACHE[cache_key]

    from PIL import ImageFont
    candidates = SCRIPT_FONT_MAP.get(script, [])
    for font_name in candidates:
        path = _find_font_path(font_name)
        if path:
            try:
                font = ImageFont.truetype(path, font_size)
                _GPU_FONT_CACHE[cache_key] = font
                return font
            except Exception:
                continue

    chain_font = _resolve_font(font_size)
    if chain_font is not None:
        _GPU_FONT_CACHE[cache_key] = chain_font
        return chain_font

    return fallback_font or ImageFont.load_default()


def resolve_font_for_text(text: str, font_size: int = FONT_SIZE):
    """Resolves the best matching font for a given text string based on its script."""
    script = detect_script(text)
    return _resolve_font_for_script(script, font_size)


def _resolve_font(font_size: int):
    """Resolves the first available font from the fallback chain."""
    from PIL import ImageFont

    cache_key = ("__CHAIN__", font_size)
    if cache_key in _GPU_FONT_CACHE:
        return _GPU_FONT_CACHE[cache_key]

    for font_name in FONT_FALLBACK_CHAIN:
        path = _find_font_path(font_name)
        if path:
            try:
                font = ImageFont.truetype(path, font_size)
                logger.info(f"GPU Atlas: Loaded font {path} @ {font_size}px")
                _GPU_FONT_CACHE[cache_key] = font
                return font
            except Exception:
                continue

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        _GPU_FONT_CACHE[cache_key] = font
        return font
    except Exception:
        font = ImageFont.load_default()
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

        has_terminal_punct = bool(re.search(r'[.!?]+$', prev_text))
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


# ─── Pre-Rendered Glyph Rendering (One-Time CPU Cost) ────────────────────────

def _render_single_cue_cpu(text: str, font,
                           fill_color: Tuple[int, int, int, int],
                           stroke_width: int = STROKE_WIDTH,
                           target_w: int = 1080) -> np.ndarray:
    """Renders a single subtitle cue with 3px drop-shadow + 6px outline and right-margin padding.

    Preserves cue_canvas.shape == (48, 1080, 4) contract for default font.

    Returns:
        np.ndarray [H, W, 4] uint8 — PREMULTIPLIED alpha RGBA.
    """
    from PIL import Image, ImageDraw

    script = detect_script(text)
    effective_font = font
    if script in ("CJK", "ARABIC", "DEVANAGARI"):
        font_sz = getattr(font, "size", FONT_SIZE)
        effective_font = _resolve_font_for_script(script, font_sz, fallback_font=font)

    from PIL import features
    direction = "rtl" if (script == "ARABIC" and features.check("raqm")) else None

    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    raw_bbox = draw.textbbox((0, 0), text, font=effective_font,
                             stroke_width=stroke_width, direction=direction)
    bbox_x0, bbox_y0 = raw_bbox[0], raw_bbox[1]
    bbox_x1, bbox_y1 = raw_bbox[2], raw_bbox[3]

    text_w = max(1, int((bbox_x1 - bbox_x0) + stroke_width * 2 + 16))
    text_h = max(1, int((bbox_y1 - bbox_y0) + stroke_width * 2 + 16))

    canvas_w = max(target_w, text_w + 120)
    canvas_h = text_h

    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad_x = stroke_width + 8
    pad_y = stroke_width + 4
    x_off = pad_x - bbox_x0
    y_off = pad_y - bbox_y0

    # Pass 0: 3px Drop-shadow (semi-transparent black)
    draw.text((x_off + SHADOW_OFFSET, y_off + SHADOW_OFFSET), text, font=effective_font,
              fill=COLOR_SHADOW, stroke_width=stroke_width, stroke_fill=COLOR_SHADOW,
              direction=direction)

    # Pass 1: Solid black stroke outline (6px thickness)
    draw.text((x_off, y_off), text, font=effective_font,
              fill=COLOR_STROKE, stroke_width=stroke_width, stroke_fill=COLOR_STROKE,
              direction=direction)

    # Pass 2: Colored fill on top
    draw.text((x_off, y_off), text, font=effective_font, fill=fill_color,
              direction=direction)

    # Convert to premultiplied alpha
    arr = np.array(img, dtype=np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    arr[:, :, 0:3] *= alpha

    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _render_outlined_glyph(text: str, font,
                           fill_color: Tuple[int, int, int, int],
                           stroke_width: int = STROKE_WIDTH) -> np.ndarray:
    """Renders outlined text glyph with 3px drop-shadow + 6px stroke + fill.

    Returns:
        np.ndarray [H, W, 4] uint8 — PREMULTIPLIED alpha RGBA.
    """
    from PIL import Image, ImageDraw

    script = detect_script(text)
    effective_font = font
    if script in ("CJK", "ARABIC", "DEVANAGARI"):
        font_sz = getattr(font, "size", FONT_SIZE)
        effective_font = _resolve_font_for_script(script, font_sz, fallback_font=font)

    from PIL import features
    direction = "rtl" if (script == "ARABIC" and features.check("raqm")) else None

    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    raw_bbox = draw.textbbox((0, 0), text, font=effective_font,
                             stroke_width=stroke_width, direction=direction)
    bbox_x0, bbox_y0 = raw_bbox[0], raw_bbox[1]
    bbox_x1, bbox_y1 = raw_bbox[2], raw_bbox[3]

    pad_x = stroke_width + 8
    pad_y = stroke_width + 8

    text_w = max(1, int((bbox_x1 - bbox_x0) + pad_x * 2 + SHADOW_OFFSET))
    text_h = max(1, int((bbox_y1 - bbox_y0) + pad_y * 2 + SHADOW_OFFSET))

    img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x_off = pad_x - bbox_x0
    y_off = pad_y - bbox_y0

    # Pass 0: 3px Drop-shadow
    draw.text((x_off + SHADOW_OFFSET, y_off + SHADOW_OFFSET), text, font=effective_font,
              fill=COLOR_SHADOW, stroke_width=stroke_width, stroke_fill=COLOR_SHADOW,
              direction=direction)

    # Pass 1: Black stroke outline (6px)
    draw.text((x_off, y_off), text, font=effective_font,
              fill=COLOR_STROKE, stroke_width=stroke_width, stroke_fill=COLOR_STROKE,
              direction=direction)

    # Pass 2: Colored fill on top
    draw.text((x_off, y_off), text, font=effective_font, fill=fill_color,
              direction=direction)

    # Convert to premultiplied alpha
    arr = np.array(img, dtype=np.float32) / 255.0
    alpha = arr[:, :, 3:4]
    arr[:, :, 0:3] *= alpha  # R' = R·A, G' = G·A, B' = B·A

    return (arr * 255.0).clip(0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# GPU GLYPH ATLAS
# ═════════════════════════════════════════════════════════════════════════════

class GPUGlyphAtlas:
    """Pre-rendered GPU-resident glyph atlas for zero-CPU subtitle compositing.

    At initialization:
        1. Chunks transcript into 1–3 word micro-chunks
        2. Pre-renders ALL unique word glyphs (normal 76px and active 87px ~14% scaled variants)
        3. Converts to premultiplied alpha
        4. Uploads entire atlas to GPU (single H2D transfer)

    Per frame:
        1. Index lookup to find active chunk + word highlight
        2. Pure CUDA tensor MAD blend (zero CPU calls)
    """

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._font = None
        self._font_active = None

        # Atlas storage: word_text → GPU tensor [H, W, 4] premultiplied float32
        self._normal_glyphs: Dict[str, torch.Tensor] = {}
        self._active_glyphs: Dict[str, torch.Tensor] = {}
        self._normal_baselines: Dict[str, int] = {}
        self._active_baselines: Dict[str, int] = {}

        # Word timing index
        self._words: List[Any] = []
        self._word_starts: List[float] = []
        self._word_ends: List[float] = []
        self._chunks: List[List[Any]] = []

        # Pre-assembled chunk strips: chunk_idx → (normal_strip, active_word_idx → highlight_strip)
        self._chunk_strips_normal: Dict[int, torch.Tensor] = {}
        self._chunk_strips_active: Dict[int, Dict[int, torch.Tensor]] = {}

        self._space_width = 5  # px between words (tightened for clean viral shorts typography)

    def build_atlas(self, words: List[Any],
                    target_w: int = 1080, target_h: int = 1920):
        """Pre-renders all unique transcript glyphs and uploads to GPU."""
        self._words = words
        self._word_starts = [_get_word_start(w) for w in words]
        self._word_ends = [_get_word_end(w) for w in words]

        self._font = _resolve_font(FONT_SIZE)
        self._font_active = _resolve_font(FONT_SIZE_ACTIVE)
        self._chunks = _build_micro_chunks(words)

        # Step 1: Pre-render all unique word glyphs (normal + active variants)
        unique_display_words = set()
        for w in words:
            text = _get_word_text(w)
            if text:
                display = text.upper() if _is_bicameral_text(text) else text
                unique_display_words.add(display)

        logger.info(f"GPU Atlas: Pre-rendering {len(unique_display_words)} unique word glyphs...")

        from PIL import Image
        for word in unique_display_words:
            word_font = resolve_font_for_text(word, FONT_SIZE)
            word_font_active = resolve_font_for_text(word, FONT_SIZE_ACTIVE)

            # Normal variant (white fill @ 76px)
            normal_np = _render_outlined_glyph(word, word_font, fill_color=COLOR_NORMAL)
            self._normal_glyphs[word] = torch.from_numpy(
                normal_np.astype(np.float32) / 255.0
            ).to(self._device)
            ascent = word_font.getmetrics()[0] if hasattr(word_font, "getmetrics") else int(round(normal_np.shape[0] * 0.72))
            self._normal_baselines[word] = (STROKE_WIDTH + 8) + ascent

            # Active variant (neon lime fill @ 87px / +14% scale pop)
            active_np = _render_outlined_glyph(word, word_font_active, fill_color=COLOR_ACTIVE)

            # Direct image scale fallback if font is non-scalable bitmap font
            if getattr(word_font_active, "size", 0) == getattr(word_font, "size", 0):
                pil_w = Image.fromarray(active_np)
                pop_w = max(1, int(round(active_np.shape[1] * SCALE_POP_FACTOR)))
                pop_h = max(1, int(round(active_np.shape[0] * SCALE_POP_FACTOR)))
                active_np = np.array(pil_w.resize((pop_w, pop_h), Image.Resampling.BILINEAR), dtype=np.uint8)

            self._active_glyphs[word] = torch.from_numpy(
                active_np.astype(np.float32) / 255.0
            ).to(self._device)
            ascent_act = word_font_active.getmetrics()[0] if hasattr(word_font_active, "getmetrics") else int(round(active_np.shape[0] * 0.72))
            self._active_baselines[word] = (STROKE_WIDTH + 8) + ascent_act

        # Step 2: Pre-assemble chunk strips on GPU
        for chunk_idx, chunk in enumerate(self._chunks):
            normal_strip = self._assemble_strip_gpu(chunk, highlight_word_idx=-1, target_w=target_w)
            if normal_strip is not None:
                self._chunk_strips_normal[chunk_idx] = normal_strip

            self._chunk_strips_active[chunk_idx] = {}
            for word_idx in range(len(chunk)):
                active_strip = self._assemble_strip_gpu(chunk, highlight_word_idx=word_idx, target_w=target_w)
                if active_strip is not None:
                    self._chunk_strips_active[chunk_idx][word_idx] = active_strip

        logger.info(
            f"GPU Atlas built: {len(unique_display_words)} glyphs, "
            f"{len(self._chunks)} chunks, "
            f"{sum(len(v) for v in self._chunk_strips_active.values())} highlight variants "
            f"(all on {self._device})"
        )

    def _assemble_strip_gpu(self, chunk: List[Any],
                            highlight_word_idx: int,
                            target_w: int) -> Optional[torch.Tensor]:
        """Assembles a subtitle strip on GPU with baseline alignment from pre-rendered glyphs."""
        items = []
        for i, w in enumerate(chunk):
            text = _get_word_text(w)
            if not text:
                continue
            display = text.upper() if _is_bicameral_text(text) else text

            if i == highlight_word_idx:
                glyph = self._active_glyphs.get(display)
                baseline = self._active_baselines.get(display, 0)
            else:
                glyph = self._normal_glyphs.get(display)
                baseline = self._normal_baselines.get(display, 0)

            if glyph is None:
                glyph = self._normal_glyphs.get(display)
                baseline = self._normal_baselines.get(display, 0)

            if glyph is not None:
                items.append({
                    "glyph": glyph,
                    "baseline": baseline,
                    "h": glyph.shape[0],
                    "w": glyph.shape[1]
                })

        if not items:
            return None

        total_width = sum(it["w"] for it in items) + self._space_width * (len(items) - 1)
        max_baseline = max(it["baseline"] for it in items)
        max_height = max(max_baseline - it["baseline"] + it["h"] for it in items)

        max_safe_w = min(MAX_LINE_WIDTH, target_w - 40) if target_w > 40 else MAX_LINE_WIDTH
        scale = min(1.0, max_safe_w / float(total_width)) if total_width > max_safe_w else 1.0

        strip_w = total_width
        strip_h = max_height

        strip = torch.zeros((strip_h, strip_w, 4), dtype=torch.float32, device=self._device)
        x_cursor = 0

        for it in items:
            glyph = it["glyph"]
            gh, gw = it["h"], it["w"]
            y_paste = max_baseline - it["baseline"]
            paste_w = gw
            paste_h = min(gh, strip_h - y_paste)

            if paste_w > 0 and paste_h > 0 and (x_cursor + paste_w) <= strip_w and y_paste < strip_h:
                src = glyph[:paste_h, :paste_w]
                src_alpha = src[:, :, 3:4]

                dst = strip[y_paste:y_paste + paste_h, x_cursor:x_cursor + paste_w]
                dst_rgb = dst[:, :, :3]
                dst_alpha = dst[:, :, 3:4]

                blended_rgb = src[:, :, :3] + dst_rgb * (1.0 - src_alpha)
                blended_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)

                strip[y_paste:y_paste + paste_h, x_cursor:x_cursor + paste_w, :3] = blended_rgb
                strip[y_paste:y_paste + paste_h, x_cursor:x_cursor + paste_w, 3:4] = blended_alpha

            x_cursor += gw + self._space_width

        if scale < 1.0:
            scaled_w = max(1, int(round(strip_w * scale)))
            scaled_h = max(1, int(round(strip_h * scale)))
            strip_4d = strip.permute(2, 0, 1).unsqueeze(0)  # [1, 4, H, W]
            scaled_4d = torch.nn.functional.interpolate(
                strip_4d, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False
            )
            strip = scaled_4d.squeeze(0).permute(1, 2, 0)

        return strip

    def _find_active_chunk_and_word(self, timestamp: float) -> Tuple[int, int, int]:
        """Finds the active chunk index and highlighted word index for a timestamp.

        Enforces single active word index lookup without overlapping double highlights.

        Returns:
            (chunk_idx, word_idx_in_chunk, global_word_idx) or (-1, -1, -1) if none.
        """
        if not self._words or not self._chunks:
            return -1, -1, -1

        idx = bisect.bisect_right(self._word_starts, timestamp) - 1
        if idx < 0 or idx >= len(self._words):
            return -1, -1, -1

        active_word = self._words[idx]
        w_end = _get_word_end(active_word)
        if timestamp > w_end + 0.3:
            return -1, -1, -1

        w_start = _get_word_start(active_word)
        is_speaking = (w_start <= timestamp <= w_end)

        for chunk_idx, chunk in enumerate(self._chunks):
            for word_idx, w in enumerate(chunk):
                if (w is active_word or
                        (_get_word_text(w) == _get_word_text(active_word)
                         and abs(_get_word_start(w) - w_start) < 1e-4)):
                    if is_speaking:
                        return chunk_idx, word_idx, idx
                    else:
                        return chunk_idx, -1, idx

        return -1, -1, -1

    def composite_frame(self, frame: torch.Tensor, timestamp: float,
                        target_w: int = 1080, target_h: int = 1920,
                        baseline_y: Optional[float] = None) -> torch.Tensor:
        """Composites subtitle overlay onto a GPU-resident video frame.

        ZERO CPU calls — pure CUDA tensor operations using pre-rendered atlas.
        Restricts vertical placement strictly inside the [0.60, 0.70] screen safe zone.

        Compositing formula (premultiplied alpha Porter-Duff Over):
            Frame_out = Text_premul + Frame_video · (1.0 - Text_alpha)
        """
        if not self._words or not self._chunks:
            return frame

        chunk_idx, word_idx, global_idx = self._find_active_chunk_and_word(timestamp)
        if chunk_idx < 0:
            return frame

        # Select the appropriate pre-assembled strip (with or without highlight)
        if word_idx >= 0 and chunk_idx in self._chunk_strips_active:
            strip = self._chunk_strips_active[chunk_idx].get(word_idx)
            if strip is None:
                strip = self._chunk_strips_normal.get(chunk_idx)
        else:
            strip = self._chunk_strips_normal.get(chunk_idx)

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

        # Clamp vertical placement strictly within safe zone bounds
        y_ideal = y_center - strip_h // 2
        y_start = max(y_min_safe, min(y_ideal, y_max_safe - strip_h))
        y_start = max(0, min(y_start, target_h - strip_h))
        x_start = max(0, (target_w - strip_w) // 2)

        x_end = min(x_start + strip_w, target_w)
        y_end = min(y_start + strip_h, target_h)
        actual_sw = x_end - x_start
        actual_sh = y_end - y_start

        if actual_sw <= 0 or actual_sh <= 0:
            return frame

        text_rgb = strip[:actual_sh, :actual_sw, :3]
        text_alpha = strip[:actual_sh, :actual_sw, 3:4]

        # Detect frame layout: [1, C, H, W] (CHW)
        if frame.dim() == 4 and frame.shape[1] == 3:
            region = frame[0, :, y_start:y_end, x_start:x_end]
            region_hwc = region.permute(1, 2, 0)

            blended = text_rgb + region_hwc * (1.0 - text_alpha)
            frame[0, :, y_start:y_end, x_start:x_end] = blended.permute(2, 0, 1)
        elif frame.dim() == 4:
            # [1, H, W, C]
            region = frame[0, y_start:y_end, x_start:x_end, :]
            blended = text_rgb + region * (1.0 - text_alpha)
            frame[0, y_start:y_end, x_start:x_end, :] = blended

        return frame


# ═════════════════════════════════════════════════════════════════════════════
# MILESTONE 6.2: DYNAMIC 3-MODE ZERO-LETTERBOX COMPOSITING & STYLED SPLIT-SCREEN
# ═════════════════════════════════════════════════════════════════════════════

def calculate_dynamic_topic_crop_2d(cx_norm: float,
                                    cy_norm: float,
                                    bw_norm: float,
                                    bh_norm: float,
                                    src_w: float = 1920.0,
                                    src_h: float = 1080.0,
                                    target_aspect: float = 9.0 / 16.0) -> Dict[str, float]:
    """Calculates dynamic 2D 9:16 crop bounding centered on demonstrated object or screen.
    Guarantees aspect ratio strictly 9:16, minimum context padding, and boundary clamping.
    """
    bw_px = bw_norm * src_w
    bh_px = bh_norm * src_h
    h_needed = max(bh_px * 1.4, (bw_px / target_aspect) * 1.4, 540.0)
    h_crop_px = min(h_needed, src_h)
    w_crop_px = h_crop_px * target_aspect

    half_w = w_crop_px / 2.0
    half_h = h_crop_px / 2.0

    cx_px = cx_norm * src_w
    cy_px = cy_norm * src_h

    cx_px_clamped = max(half_w, min(src_w - half_w, cx_px))
    cy_px_clamped = max(half_h, min(src_h - half_h, cy_px))

    return {
        "w_crop_px": w_crop_px,
        "h_crop_px": h_crop_px,
        "cx_px": cx_px_clamped,
        "cy_px": cy_px_clamped,
        "w_crop_norm": w_crop_px / src_w,
        "h_crop_norm": h_crop_px / src_h,
        "cx_norm": cx_px_clamped / src_w,
        "cy_norm": cy_px_clamped / src_h,
    }


def render_full_frame_topic_closeup(frame: torch.Tensor,
                                     cx_norm: float,
                                     cy_norm: float = 0.5,
                                     w_crop_norm: Optional[float] = None,
                                     h_crop_norm: Optional[float] = None,
                                     target_w: int = 1080,
                                     target_h: int = 1920) -> torch.Tensor:
    """Mode 2 (FULL_FRAME_TOPIC_CLOSEUP):
    Full-frame 9:16 close-up crop directly centered on demonstrated object/screen/action,
    scaled to fill 1080x1920 with zero letterbox padding.
    """
    B, C, src_h, src_w = frame.shape
    aspect = float(target_w) / float(target_h)

    if w_crop_norm is None or h_crop_norm is None:
        crop_info = calculate_dynamic_topic_crop_2d(
            cx_norm=cx_norm, cy_norm=cy_norm,
            bw_norm=0.30, bh_norm=0.30,
            src_w=float(src_w), src_h=float(src_h),
            target_aspect=aspect
        )
        crop_w_px = crop_info["w_crop_px"]
        crop_h_px = crop_info["h_crop_px"]
        cx_px = crop_info["cx_px"]
        cy_px = crop_info["cy_px"]
    else:
        crop_h_px = min(float(src_h), max(540.0, h_crop_norm * float(src_h)))
        crop_w_px = crop_h_px * aspect
        half_w = crop_w_px / 2.0
        half_h = crop_h_px / 2.0
        cx_px = max(half_w, min(float(src_w) - half_w, cx_norm * float(src_w)))
        cy_px = max(half_h, min(float(src_h) - half_h, cy_norm * float(src_h)))

    x0 = max(0, min(int(src_w - crop_w_px), int(round(cx_px - crop_w_px / 2.0))))
    x1 = int(round(x0 + crop_w_px))
    y0 = max(0, min(int(src_h - crop_h_px), int(round(cy_px - crop_h_px / 2.0))))
    y1 = int(round(y0 + crop_h_px))

    cropped = frame[:, :, y0:y1, x0:x1]
    resized = torch.nn.functional.interpolate(
        cropped, size=(target_h, target_w), mode='bilinear', align_corners=False
    )
    return resized.clamp(0.0, 1.0)


def render_styled_divider(canvas: torch.Tensor,
                          split_y: int = 960,
                          divider_thickness: int = 4,
                          neon_color: Tuple[int, int, int, int] = COLOR_NEON_LIME,
                          glow_radius: int = 6,
                          shadow_blur: int = 10,
                          corner_radius: int = 16) -> torch.Tensor:
    """Renders a styled divider line between split-screen panes:
    - 3-4px horizontal accent line at Y ~ split_y in #CCFF00 neon accent.
    - Soft drop-shadow behind and around the divider line.
    - 6px Gaussian halo glow falloff.
    - 16px inner corner rounding mask for adjacent pane edges.
    """
    B, C, H, W = canvas.shape

    half_th = divider_thickness // 2
    neon_r = neon_color[0] / 255.0
    neon_g = neon_color[1] / 255.0
    neon_b = neon_color[2] / 255.0

    # 1. Soft drop shadow behind and around divider line
    for dy in range(-16, 17):
        y_pos = split_y + dy
        if 0 <= y_pos < H:
            dist = abs(dy)
            if dist > half_th:
                atten = 1.0 - 0.40 * math.exp(- ((dist - half_th) ** 2) / (2.0 * ((shadow_blur / 2.0) ** 2)))
                canvas[:, :, y_pos, :] *= atten

    # 2. Gaussian Halo Glow (6px Gaussian falloff)
    glow_sigma = max(2.5, float(glow_radius))
    for dy in range(-16, 17):
        y_pos = split_y + dy
        if 0 <= y_pos < H:
            dist = abs(dy)
            if dist > half_th:
                glow = 0.70 * math.exp(- ((dist - half_th) ** 2) / (2.0 * (glow_sigma ** 2)))
                canvas[:, 0, y_pos, :] = torch.clamp(canvas[:, 0, y_pos, :] + neon_r * glow, 0.0, 1.0)
                canvas[:, 1, y_pos, :] = torch.clamp(canvas[:, 1, y_pos, :] + neon_g * glow, 0.0, 1.0)
                canvas[:, 2, y_pos, :] = torch.clamp(canvas[:, 2, y_pos, :] + neon_b * glow, 0.0, 1.0)

    # 3. Solid Neon Accent Core Line
    y_start = max(0, split_y - half_th)
    y_end = min(H, split_y + half_th + (divider_thickness % 2))
    canvas[:, 0, y_start:y_end, :] = neon_r
    canvas[:, 1, y_start:y_end, :] = neon_g
    canvas[:, 2, y_start:y_end, :] = neon_b

    # 4. Rounded inner corners (radius = 16px) for the pane inner edges adjacent to divider
    if corner_radius > 0:
        cr = corner_radius
        for dy_c in range(cr):
            y_t = y_start - 1 - dy_c
            if 0 <= y_t < H:
                for dx_c in range(cr):
                    dist_corner_l = math.sqrt((cr - dx_c)**2 + (cr - dy_c)**2)
                    if dist_corner_l > cr:
                        fade = max(0.0, 1.0 - (dist_corner_l - cr) / 3.0)
                        canvas[:, :, y_t, dx_c] *= fade
                    x_r = W - 1 - dx_c
                    dist_corner_r = math.sqrt((cr - dx_c)**2 + (cr - dy_c)**2)
                    if dist_corner_r > cr:
                        fade = max(0.0, 1.0 - (dist_corner_r - cr) / 3.0)
                        canvas[:, :, y_t, x_r] *= fade

            y_b = y_end + dy_c
            if 0 <= y_b < H:
                for dx_c in range(cr):
                    dist_corner_l = math.sqrt((cr - dx_c)**2 + (cr - dy_c)**2)
                    if dist_corner_l > cr:
                        fade = max(0.0, 1.0 - (dist_corner_l - cr) / 3.0)
                        canvas[:, :, y_b, dx_c] *= fade
                    x_r = W - 1 - dx_c
                    dist_corner_r = math.sqrt((cr - dx_c)**2 + (cr - dy_c)**2)
                    if dist_corner_r > cr:
                        fade = max(0.0, 1.0 - (dist_corner_r - cr) / 3.0)
                        canvas[:, :, y_b, x_r] *= fade

    return canvas.clamp(0.0, 1.0)


def render_dual_pane_split_screen(frame: torch.Tensor,
                                  speaker_cx: float = 0.5,
                                  topic_cx: float = 0.5,
                                  topic_cy: float = 0.5,
                                  target_w: int = 1080,
                                  target_h: int = 1920,
                                  config: Optional[Any] = None) -> torch.Tensor:
    """Mode 3 (DUAL_PANE_SPLIT_SCREEN):
    Styled dual-pane vertical split-screen:
    - Top pane: speaker crop scaled to 1080x958.
    - Bottom pane: topic close-up crop scaled to 1080x958.
    - Divider bar: 3-4px horizontal accent line at Y in [958, 962] in #CCFF00 neon accent.
    - 6px Gaussian halo glow and soft drop-shadow.
    - Rounded inner corners (radius 16px).
    """
    B, C, src_h, src_w = frame.shape
    device = frame.device

    div_th = getattr(config, "divider_width_px", 4) if config else 4
    split_r = getattr(config, "split_ratio", 0.50) if config else 0.50
    glow_r = getattr(config, "divider_glow_radius", 6) if config else 6
    corner_r = getattr(config, "corner_radius_px", 16) if config else 16
    shadow_b = getattr(config, "shadow_blur_px", 10) if config else 10

    split_y = int(round(target_h * split_r))
    half_th = div_th // 2

    top_h = split_y - half_th
    bot_h = target_h - (split_y + half_th + (div_th % 2))

    # Top pane: speaker crop
    top_aspect = float(target_w) / float(top_h)
    crop_h_top = float(src_h)
    crop_w_top = crop_h_top * (float(target_w) / float(target_h)) * 1.5
    crop_w_top = min(float(src_w), max(crop_w_top, crop_h_top * 0.75))
    crop_h_top = crop_w_top / top_aspect
    if crop_h_top > src_h:
        crop_h_top = float(src_h)
        crop_w_top = crop_h_top * top_aspect

    top_x0 = max(0, min(int(src_w - crop_w_top), int(round(speaker_cx * src_w - crop_w_top / 2.0))))
    top_x1 = int(round(top_x0 + crop_w_top))
    top_y0 = 0
    top_y1 = int(round(crop_h_top))
    top_cropped = frame[:, :, top_y0:top_y1, top_x0:top_x1]
    top_pane = torch.nn.functional.interpolate(
        top_cropped, size=(top_h, target_w), mode='bilinear', align_corners=False
    )

    # Bottom pane: topic close-up crop
    bot_aspect = float(target_w) / float(bot_h)
    crop_h_bot = min(float(src_h), max(540.0, float(src_h) * 0.65))
    crop_w_bot = crop_h_bot * bot_aspect
    if crop_w_bot > src_w:
        crop_w_bot = float(src_w)
        crop_h_bot = crop_w_bot / bot_aspect

    bot_x0 = max(0, min(int(src_w - crop_w_bot), int(round(topic_cx * src_w - crop_w_bot / 2.0))))
    bot_x1 = int(round(bot_x0 + crop_w_bot))
    bot_y0 = max(0, min(int(src_h - crop_h_bot), int(round(topic_cy * src_h - crop_h_bot / 2.0))))
    bot_y1 = int(round(bot_y0 + crop_h_bot))
    bot_cropped = frame[:, :, bot_y0:bot_y1, bot_x0:bot_x1]
    bot_pane = torch.nn.functional.interpolate(
        bot_cropped, size=(bot_h, target_w), mode='bilinear', align_corners=False
    )

    # Assemble canvas
    canvas = torch.zeros((B, C, target_h, target_w), dtype=torch.float32, device=device)
    canvas[:, :, :top_h, :] = top_pane
    canvas[:, :, target_h - bot_h:, :] = bot_pane

    # Render styled glowing divider
    canvas = render_styled_divider(
        canvas=canvas,
        split_y=split_y,
        divider_thickness=div_th,
        neon_color=COLOR_NEON_LIME,
        glow_radius=glow_r,
        shadow_blur=shadow_b,
        corner_radius=corner_r
    )

    return canvas


def verify_subtitle_safe_zone_isolation(split_y: int = 960,
                                        divider_thickness: int = 4,
                                        glow_radius: int = 6,
                                        safe_zone_y_min: int = 1152,
                                        safe_zone_y_max: int = 1344) -> bool:
    """Verifies that the split-screen divider line and Gaussian glow halo at Y ~ split_y
    maintain strict separation from the subtitle safe zone [safe_zone_y_min, safe_zone_y_max].
    Ensures minimum clearance >= 150px.
    """
    half_th = divider_thickness // 2
    divider_bottom_extent = split_y + half_th + glow_radius
    clearance = safe_zone_y_min - divider_bottom_extent
    return clearance >= 150


from wsl2_gpu.subtitle_compositor import SubtitleCompositor
GPUCompositor = SubtitleCompositor
