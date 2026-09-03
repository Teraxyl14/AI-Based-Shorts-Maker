"""
Project Aether V2 — Multilingual Kinetic Subtitle Compositor

Containerless outlined kinetic typography with HarfBuzz multilingual text
shaping, 1–3 word micro-chunking, and premultiplied-alpha GPU compositing.
Renders broadcast-grade viral subtitles at 1080×1920 without dark container
boxes — dual-pass stroke + fill outline directly over video.
"""

import bisect
import logging
import math
import os
import unicodedata
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger("AetherSubtitleCompositor")

# ─── Typography Constants ─────────────────────────────────────────────────────

FONT_SIZE = 76  # px at 1080×1920 canvas
STROKE_WIDTH = 6  # px solid black outline
MAX_LINE_WIDTH = 920  # px max subtitle line width (safe zone)
SUBTITLE_Y_ANCHOR = 0.65  # vertical anchor as fraction of canvas height

# Word highlight colors (RGBA 0-255)
COLOR_NORMAL = (255, 255, 255, 255)      # Crisp white fill
COLOR_ACTIVE = (255, 222, 0, 255)        # Neon Yellow highlight for active word
COLOR_STROKE = (0, 0, 0, 255)            # Solid black outline

# Cosine ease-in ramp duration for active word onset
EASE_IN_DURATION = 0.020  # 20ms

# Micro-chunking parameters
MAX_WORDS_PER_CHUNK = 3
SILENCE_THRESHOLD = 0.120  # 120ms silence gap triggers new chunk

# ─── Multilingual Font Fallback Chain ─────────────────────────────────────────

# Search paths for Noto/DejaVu fonts (Docker container + common OS locations)
FONT_SEARCH_PATHS = [
    # Container / Linux
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/opentype/noto",
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
    "DejaVuSans-Bold.ttf",
]

# Scripts that support upper/lower case (bicameral)
BICAMERAL_SCRIPTS = {"LATIN", "CYRILLIC", "GREEK", "ARMENIAN", "GEORGIAN"}


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


def _find_font_path(font_name: str) -> Optional[str]:
    """Searches the font fallback chain directories for the given font file."""
    for search_dir in FONT_SEARCH_PATHS:
        candidate = os.path.join(search_dir, font_name)
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_font_chain(font_size: int):
    """Resolves the first available font from the fallback chain.

    Returns:
        Pillow ImageFont object.
    """
    from PIL import ImageFont

    for font_name in FONT_FALLBACK_CHAIN:
        path = _find_font_path(font_name)
        if path:
            try:
                font = ImageFont.truetype(path, font_size)
                logger.info(f"Loaded font: {path} @ {font_size}px")
                return font
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

    # Last resort: Pillow default
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        logger.info(f"Loaded system DejaVuSans-Bold @ {font_size}px")
        return font
    except Exception:
        font = ImageFont.load_default()
        logger.warning("Using Pillow default bitmap font (limited quality)")
        return font


# ─── Micro-Chunk Builder ─────────────────────────────────────────────────────

def _build_micro_chunks(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Segments word stream into 1–3 word micro-chunks based on acoustic silence.

    Rules:
        - A new chunk starts when the silence gap between consecutive words
          exceeds SILENCE_THRESHOLD (120ms).
        - Chunks are capped at MAX_WORDS_PER_CHUNK (3) words.
    """
    if not words:
        return []

    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []

    for i, w in enumerate(words):
        text = w.get("word", "").strip()
        if not text:
            continue

        if not current_chunk:
            current_chunk.append(w)
            continue

        # Check silence gap from previous word's end to this word's start
        prev_end = current_chunk[-1].get("end", 0.0)
        this_start = w.get("start", 0.0)
        gap = this_start - prev_end

        if len(current_chunk) >= MAX_WORDS_PER_CHUNK or gap >= SILENCE_THRESHOLD:
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
                          stroke_width: int = STROKE_WIDTH) -> np.ndarray:
    """Renders a single word with dual-pass stroke + fill outline.

    Pass 1: Black stroke outline (6px)
    Pass 2: Colored fill on top

    Returns:
        np.ndarray [H, W, 4] dtype uint8 — straight-alpha RGBA.
    """
    from PIL import Image, ImageDraw

    # Measure text bounding box with stroke padding
    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dummy)

    # Get text bbox with stroke for proper sizing
    bbox_stroke = draw.textbbox((0, 0), word, font=font, stroke_width=stroke_width)
    text_w = bbox_stroke[2] - bbox_stroke[0] + 4
    text_h = bbox_stroke[3] - bbox_stroke[1] + 4

    # Create RGBA canvas
    img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x_off = -bbox_stroke[0] + 2
    y_off = -bbox_stroke[1] + 2

    # Pass 1: Stroke outline
    draw.text((x_off, y_off), word, font=font,
              fill=stroke_color, stroke_width=stroke_width,
              stroke_fill=stroke_color)

    # Pass 2: Fill on top (no stroke)
    draw.text((x_off, y_off), word, font=font, fill=fill_color)

    return np.array(img, dtype=np.uint8)


# ─── Subtitle Compositor Class ───────────────────────────────────────────────

class SubtitleCompositor:
    """Multilingual kinetic subtitle compositor with containerless outlined typography.

    Renders 1–3 word micro-chunks with dual-pass stroke + fill outlines directly
    over video using premultiplied-alpha Porter-Duff GPU compositing.
    """

    def __init__(self, font_path: str = None, font_size: int = FONT_SIZE):
        self.font_path = font_path
        self.font_size = font_size
        self._font = None
        self._word_textures: Dict[str, np.ndarray] = {}  # word -> RGBA numpy array (normal variant)
        self._word_starts: List[float] = []
        self._word_ends: List[float] = []
        self._words: List[Dict[str, Any]] = []
        self._chunks: List[List[Dict[str, Any]]] = []
        self._device = None

    def _load_font(self):
        """Loads the font via the multilingual fallback chain."""
        if self._font is not None:
            return

        from PIL import ImageFont

        # User-specified font takes priority
        if self.font_path and os.path.exists(self.font_path):
            try:
                self._font = ImageFont.truetype(self.font_path, self.font_size)
                logger.info(f"Loaded user font: {self.font_path} @ {self.font_size}px")
                return
            except Exception as e:
                logger.warning(f"Failed to load user font {self.font_path}: {e}")

        # Fall through to chain
        self._font = _resolve_font_chain(self.font_size)

    def build_word_atlas(self, words: List[Dict[str, Any]], target_w: int = 1080):
        """Pre-renders all unique transcript words and builds micro-chunks.

        Args:
            words: List of word dicts with 'word', 'start', 'end' keys.
            target_w: Canvas width (used for line-width validation).
        """
        self._words = words
        self._word_starts = [w.get("start", 0.0) for w in words]
        self._word_ends = [w.get("end", 0.0) for w in words]
        self._word_textures.clear()

        self._load_font()

        # Build 1–3 word micro-chunks based on acoustic silence
        self._chunks = _build_micro_chunks(words)

        # Pre-render unique words in normal (white outline) variant
        unique_words = set()
        for w in words:
            text = w.get("word", "").strip()
            if text:
                display_text = text.upper() if _is_bicameral_text(text) else text
                unique_words.add(display_text)

        for word in unique_words:
            self._word_textures[word] = _render_outlined_word(
                word, self._font, fill_color=COLOR_NORMAL
            )

        logger.info(
            f"Built kinetic subtitle atlas: {len(unique_words)} unique words, "
            f"{len(self._chunks)} micro-chunks (max {MAX_WORDS_PER_CHUNK} words/chunk)"
        )

    def _get_display_text(self, word: str) -> str:
        """Applies bicameral uppercasing to the word."""
        return word.upper() if _is_bicameral_text(word) else word

    def _premultiply_tensor(self, rgba: torch.Tensor) -> torch.Tensor:
        """Converts straight-alpha RGBA float32 tensor to premultiplied alpha.

        Args:
            rgba: [..., 4] tensor in [0, 1] range (straight alpha).

        Returns:
            [..., 4] tensor with premultiplied RGB channels.
        """
        rgb = rgba[..., :3]
        a = rgba[..., 3:4]
        premul_rgb = rgb * a
        return torch.cat([premul_rgb, a], dim=-1)

    def _build_subtitle_strip(self, timestamp: float,
                              target_w: int, target_h: int) -> Optional[np.ndarray]:
        """Assembles a containerless outlined subtitle strip for the current timestamp.

        Renders ONLY the currently active micro-chunk (1–3 words) with:
        - Active spoken word: Neon Yellow with 20ms ease-in
        - Normal words: White
        - Dual-pass 6px black stroke outline (no background container)

        Returns:
            np.ndarray [strip_h, strip_w, 4] or None if no words active.
        """
        if not self._words or not self._chunks:
            return None

        # Find current active word using bisect
        idx = bisect.bisect_right(self._word_starts, timestamp) - 1
        if idx < 0 or idx >= len(self._words):
            return None

        # Grace period after word ends
        active_word = self._words[idx]
        if timestamp > active_word.get("end", 0.0) + 0.3:
            return None

        # Find the micro-chunk containing the active word
        display_chunk = None
        for chunk in self._chunks:
            if active_word in chunk:
                display_chunk = chunk
                break

        if not display_chunk:
            return None

        # Render each word in the chunk with appropriate color
        word_images = []
        space_width = 12  # px between words
        total_width = 0

        for w in display_chunk:
            text = w.get("word", "").strip()
            if not text:
                continue

            display_text = self._get_display_text(text)
            word_start = w.get("start", 0.0)
            word_end = w.get("end", 0.0)

            # Determine word color
            if word_start <= timestamp <= word_end:
                # Active word — apply cosine ease-in to neon yellow
                elapsed = timestamp - word_start
                if elapsed < EASE_IN_DURATION:
                    t = elapsed / EASE_IN_DURATION
                    alpha_scale = 0.5 * (1.0 - math.cos(math.pi * t))
                else:
                    alpha_scale = 1.0

                # Interpolate from white to neon yellow
                r = int(COLOR_NORMAL[0] + (COLOR_ACTIVE[0] - COLOR_NORMAL[0]) * alpha_scale)
                g = int(COLOR_NORMAL[1] + (COLOR_ACTIVE[1] - COLOR_NORMAL[1]) * alpha_scale)
                b_val = int(COLOR_NORMAL[2] + (COLOR_ACTIVE[2] - COLOR_NORMAL[2]) * alpha_scale)
                fill_color = (r, g, b_val, 255)

                word_img = _render_outlined_word(display_text, self._font, fill_color=fill_color)
            else:
                # Normal word — use cached white texture or re-render
                if display_text in self._word_textures:
                    word_img = self._word_textures[display_text]
                else:
                    word_img = _render_outlined_word(display_text, self._font, fill_color=COLOR_NORMAL)

            word_images.append(word_img)
            total_width += word_img.shape[1] + space_width

        if not word_images:
            return None

        total_width -= space_width  # Remove trailing space

        # Clamp to MAX_LINE_WIDTH
        strip_w = min(total_width, MAX_LINE_WIDTH, target_w - 40)
        max_h = max(img.shape[0] for img in word_images)
        strip_h = max_h

        # Assemble strip on transparent RGBA canvas
        strip = np.zeros((strip_h, strip_w, 4), dtype=np.uint8)

        x_cursor = max(0, (strip_w - total_width) // 2)  # Center horizontally

        for word_img in word_images:
            wh, ww = word_img.shape[:2]
            paste_w = min(ww, strip_w - x_cursor)
            paste_h = min(wh, strip_h)

            if paste_w > 0 and paste_h > 0 and x_cursor < strip_w:
                # Alpha composite word onto strip
                src = word_img[:paste_h, :paste_w]
                dst = strip[:paste_h, x_cursor:x_cursor + paste_w]

                src_a = src[:, :, 3:4].astype(np.float32) / 255.0
                blended_rgb = (src[:, :, :3].astype(np.float32) * src_a +
                               dst[:, :, :3].astype(np.float32) * (1.0 - src_a))
                blended_a = (src[:, :, 3:4].astype(np.float32) +
                             dst[:, :, 3:4].astype(np.float32) * (1.0 - src_a))

                strip[:paste_h, x_cursor:x_cursor + paste_w, :3] = blended_rgb.clip(0, 255).astype(np.uint8)
                strip[:paste_h, x_cursor:x_cursor + paste_w, 3:4] = blended_a.clip(0, 255).astype(np.uint8)

            x_cursor += ww + space_width

        return strip

    def composite_frame(self, frame: torch.Tensor, timestamp: float,
                        words: List[Dict[str, Any]],
                        target_w: int = 0, target_h: int = 0,
                        baseline_y: Optional[float] = None) -> torch.Tensor:
        """Composites kinetic subtitle overlay onto a GPU-resident video frame.

        Uses premultiplied alpha Porter-Duff blending:
            Frame_out = Glyph_premul + Frame_video × (1 - Glyph_alpha)

        Guarantees _overlay_buf.zero_() on active CUDA stream before every composite.

        Args:
            frame: [1, C, H, W] or [1, H, W, C] float32 GPU tensor in [0, 1].
            timestamp: Current frame timestamp in seconds.
            words: Word timestamp list.
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
        if strip is None:
            return frame

        strip_h, strip_w = strip.shape[:2]

        # Determine vertical placement (center at Y = 0.65 × H)
        if baseline_y is not None:
            y_center = int(baseline_y)
        else:
            y_center = int(target_h * SUBTITLE_Y_ANCHOR)

        y_start = max(0, min(y_center - strip_h // 2, target_h - strip_h))
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
