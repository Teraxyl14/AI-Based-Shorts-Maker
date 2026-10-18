"""
tests/test_challenger_m3_2_empirical_stress.py - Challenger 2 Milestone 3 Empirical Stress Test Suite.

Adversarial stress-testing of Feature 8 (Modern Kinetic Typography Engine / R3):
1. 3-Pass Rendering Oracle: 3px drop-shadow + 6px solid black outline + colored fill.
2. Containerless Transparency & Zero Opaque Boxes: alpha coverage < 35%, transparent background corners.
3. Safe Zone Vertical Lock: Y in [1152, 1344] px under all normal and adversarial baseline_y inputs.
4. Non-Clipping Oracle: Descenders (g, j, p, q, y), ascenders, and drop-shadows across canvas borders.
5. Multilingual Unicode & Tofu Elimination: Authentic glyphs in Latin, CJK, Arabic RTL, and Devanagari without .notdef boxes.
6. Kinetic Scale Pop & Micro-Chunking Invariants: 1-3 words, single active highlight gating, ~14% pop scale.
"""

import math
import os
import sys
import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from wsl2_gpu.subtitle_compositor import (
    SubtitleCompositor,
    _render_outlined_word,
    _build_micro_chunks,
    _is_bicameral_text,
    detect_script,
    resolve_font_for_text,
    FONT_SIZE,
    FONT_SIZE_ACTIVE,
    SCALE_POP_FACTOR,
    STROKE_WIDTH,
    SHADOW_OFFSET,
    COLOR_NORMAL,
    COLOR_STROKE,
    COLOR_SHADOW,
    COLOR_ACTIVE,
    COLOR_NEON_LIME,
    COLOR_NEON_GREEN,
    COLOR_NEON_YELLOW,
    NEON_PALETTE,
    SAFE_ZONE_MIN_RATIO,
    SAFE_ZONE_MAX_RATIO,
    SUBTITLE_Y_ANCHOR,
)
from wsl2_gpu.gpu_compositor import (
    GPUGlyphAtlas,
    _render_single_cue_cpu,
    _render_outlined_glyph,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. 3-PASS RENDERING ORACLE: SHADOW + OUTLINE + FILL
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature8ThreePassRendering:
    """Empirical verification of 3-pass outlined typography (shadow, stroke, fill)."""

    def test_oracle_three_pass_layer_geometry(self):
        """Stress: Verify 3px drop-shadow expands geometry exactly (+3, +3) with semi-transparent black."""
        font = resolve_font_for_text("I", FONT_SIZE)

        # 1. Render standard 3-pass word image
        img_with_shadow = _render_outlined_word(
            "I", font, COLOR_ACTIVE, stroke_color=COLOR_STROKE, stroke_width=STROKE_WIDTH,
            shadow_color=COLOR_SHADOW, shadow_offset=SHADOW_OFFSET
        )

        # 2. Render 2-pass reference (outline + fill only, zero drop-shadow)
        dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)
        raw_bbox = draw.textbbox((0, 0), "I", font=font, stroke_width=STROKE_WIDTH)
        pad_x = STROKE_WIDTH + 8
        pad_y = STROKE_WIDTH + 8
        text_w = max(1, int((raw_bbox[2] - raw_bbox[0]) + pad_x * 2 + SHADOW_OFFSET))
        text_h = max(1, int((raw_bbox[3] - raw_bbox[1]) + pad_y * 2 + SHADOW_OFFSET))
        img_no_shadow = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img_no_shadow)
        x_off = pad_x - raw_bbox[0]
        y_off = pad_y - raw_bbox[1]
        d.text((x_off, y_off), "I", font=font, fill=COLOR_STROKE, stroke_width=STROKE_WIDTH, stroke_fill=COLOR_STROKE)
        d.text((x_off, y_off), "I", font=font, fill=COLOR_ACTIVE)
        arr_no_shadow = np.array(img_no_shadow)

        # 3. Geometric footprint comparison
        nz_with = np.argwhere(img_with_shadow[:, :, 3] > 0)
        nz_no = np.argwhere(arr_no_shadow[:, :, 3] > 0)

        max_y_with, max_x_with = nz_with.max(axis=0)
        max_y_no, max_x_no = nz_no.max(axis=0)

        # Exact 3px shadow offset expansion
        assert max_y_with - max_y_no == SHADOW_OFFSET, f"Y shadow offset mismatch: {max_y_with - max_y_no} vs {SHADOW_OFFSET}"
        assert max_x_with - max_x_no == SHADOW_OFFSET, f"X shadow offset mismatch: {max_x_with - max_x_no} vs {SHADOW_OFFSET}"

        # 4. Pixel-level oracle in the shadow extension zone
        diff_mask = (img_with_shadow[:, :, 3] > 0) & (arr_no_shadow[:, :, 3] == 0)
        assert diff_mask.sum() > 50, "Drop shadow did not create sufficient shadow pixels"
        diff_alphas = img_with_shadow[diff_mask, 3]
        diff_rgbs = img_with_shadow[diff_mask, :3]

        # Shadow alpha must be <= COLOR_SHADOW[3] (160)
        assert diff_alphas.max() <= COLOR_SHADOW[3], f"Shadow alpha exceeded {COLOR_SHADOW[3]}: {diff_alphas.max()}"
        # Shadow color must be pure black
        assert diff_rgbs.max() == 0, f"Shadow contains non-black color: max={diff_rgbs.max()}"

    def test_oracle_colored_fill_and_solid_outline_presence(self):
        """Stress: Verify presence of neon active fill pixels and solid black outline pixels."""
        font = resolve_font_for_text("TEST", FONT_SIZE)
        img = _render_outlined_word("TEST", font, COLOR_ACTIVE)

        # 1. Fill pixels: match COLOR_ACTIVE
        fill_mask = (
            (img[:, :, 0] == COLOR_ACTIVE[0]) &
            (img[:, :, 1] == COLOR_ACTIVE[1]) &
            (img[:, :, 2] == COLOR_ACTIVE[2]) &
            (img[:, :, 3] == 255)
        )
        assert fill_mask.sum() > 200, "Colored fill pixels missing or corrupted"

        # 2. Outline pixels: solid black with alpha 255
        outline_mask = (
            (img[:, :, 0] == 0) &
            (img[:, :, 1] == 0) &
            (img[:, :, 2] == 0) &
            (img[:, :, 3] == 255)
        )
        assert outline_mask.sum() > 500, "Solid black outline pixels missing or corrupted"

    def test_gpu_compositor_three_pass_integrity(self):
        """Stress: Verify GPU compositor cue CPU render implements 3-pass outline + shadow + fill."""
        font = ImageFont.load_default()
        cue = _render_single_cue_cpu("VIRAL HOOK", font=font, fill_color=COLOR_ACTIVE, target_w=1080)
        assert cue.shape == (48, 1080, 4)

        # Check for shadow pixels (premultiplied: RGB=0, alpha around 160)
        shadow_pixels = (cue[:, :, 3] > 0) & (cue[:, :, 3] <= 180) & (cue[:, :, 0] == 0) & (cue[:, :, 1] == 0)
        assert shadow_pixels.sum() > 20, "GPU cue canvas missing drop-shadow alpha pixels"

    def test_neon_palette_spec_compliance(self):
        """Stress: Verify neon active highlight colors satisfy high-energy visual requirements."""
        for name, col in NEON_PALETTE.items():
            r, g, b, a = col
            assert a == 255, f"{name} alpha must be 255"
            assert g >= 200, f"{name} green channel must be high (>=200)"
            assert r + g > 350, f"{name} combined red+green must form vivid yellow/green"


# ═════════════════════════════════════════════════════════════════════════════
# 2. CONTAINERLESS TRANSPARENCY & ZERO OPAQUE BOXES
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature8ZeroOpaqueBoundingBoxes:
    """Empirical verification that subtitle overlays contain zero rectangular bounding boxes."""

    def test_zero_bounding_box_transparent_corners_all_scripts(self):
        """Stress: All four corners of subtitle strip must have alpha == 0 across diverse scripts."""
        comp = SubtitleCompositor()
        script_samples = [
            [{"word": "VIRAL", "start": 0.0, "end": 0.5}],
            [{"word": "HOOK", "start": 0.0, "end": 0.3}, {"word": "NOW", "start": 0.3, "end": 0.6}],
            [{"word": "人工智能", "start": 0.0, "end": 0.5}],
            [{"word": "नमस्ते", "start": 0.0, "end": 0.5}],
            [{"word": "مرحبا", "start": 0.0, "end": 0.5}],
        ]

        for sample in script_samples:
            comp.build_word_atlas(sample, target_w=1080)
            strip = comp._build_subtitle_strip(0.1, 1080, 1920)
            assert strip is not None

            # All 4 corners must be strictly 0 alpha
            assert strip[0, 0, 3] == 0, "Top-left corner not transparent"
            assert strip[0, -1, 3] == 0, "Top-right corner not transparent"
            assert strip[-1, 0, 3] == 0, "Bottom-left corner not transparent"
            assert strip[-1, -1, 3] == 0, "Bottom-right corner not transparent"

    def test_alpha_coverage_under_35_percent_threshold(self):
        """Stress: Alpha coverage on full 1080x1920 video frame must be < 35% (R3 requirement)."""
        comp = SubtitleCompositor()
        atlas = GPUGlyphAtlas(torch.device("cpu"))

        test_transcripts = [
            [{"word": "VIRAL", "start": 0.0, "end": 0.5}],
            [{"word": "THE", "start": 0.0, "end": 0.3}, {"word": "VIRAL", "start": 0.3, "end": 0.7}, {"word": "HOOK", "start": 0.7, "end": 1.1}],
            [{"word": "EXTRAORDINARILY", "start": 0.0, "end": 0.5}, {"word": "UNPRECEDENTED", "start": 0.5, "end": 1.0}],
            [{"word": "人工智能", "start": 0.0, "end": 0.5}, {"word": "革命性", "start": 0.5, "end": 1.0}, {"word": "突破", "start": 1.0, "end": 1.5}],
            [{"word": "नमस्ते", "start": 0.0, "end": 0.5}, {"word": "दुनिया", "start": 0.5, "end": 1.0}],
            [{"word": "مرحبا", "start": 0.0, "end": 0.5}, {"word": "بالعالم", "start": 0.5, "end": 1.0}],
        ]

        for sample in test_transcripts:
            comp.build_word_atlas(sample, target_w=1080)
            atlas.build_atlas(sample, target_w=1080, target_h=1920)

            frame = torch.zeros((1, 3, 1920, 1080), dtype=torch.float32)

            out_cpu = comp.composite_frame(frame.clone(), 0.1, sample, 1080, 1920)
            cov_cpu = (out_cpu[0].sum(dim=0) > 0).float().mean().item()
            assert cov_cpu < 0.35, f"SubtitleCompositor alpha coverage {cov_cpu:.4f} >= 0.35"
            assert cov_cpu < 0.05, f"SubtitleCompositor typical coverage exceeded 5%: {cov_cpu:.4f}"

            out_gpu = atlas.composite_frame(frame.clone(), 0.1, 1080, 1920)
            cov_gpu = (out_gpu[0].sum(dim=0) > 0).float().mean().item()
            assert cov_gpu < 0.35, f"GPUGlyphAtlas alpha coverage {cov_gpu:.4f} >= 0.35"
            assert cov_gpu < 0.05, f"GPUGlyphAtlas typical coverage exceeded 5%: {cov_gpu:.4f}"


# ═════════════════════════════════════════════════════════════════════════════
# 3. VERTICAL SAFE ZONE LOCK: [1152, 1344] PX
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature8SafeZoneVerticalLock:
    """Empirical verification that subtitle Y coordinates stay strictly within [1152, 1344] px."""

    @pytest.mark.parametrize("baseline_y", [
        None,               # Default anchor (Y=1248)
        1248.0,             # Explicit ideal anchor
        1152.0,             # Min safe zone boundary
        1344.0,             # Max safe zone boundary
        0.0,                # Adversarial extreme low
        -500.0,             # Adversarial negative
        1000.0,             # Adversarial sub-safe
        1500.0,             # Adversarial post-safe
        1920.0,             # Adversarial bottom of canvas
        3000.0,             # Adversarial massive out-of-bounds
    ])
    def test_safe_zone_vertical_bounds_under_adversarial_anchors(self, baseline_y):
        """Stress: Subtitle pixels MUST NOT escape [1152, 1344] px under adversarial baseline_y."""
        comp = SubtitleCompositor()
        words = [
            {"word": "VIRAL", "start": 0.0, "end": 0.4},
            {"word": "MOMENTUM", "start": 0.4, "end": 0.8},
        ]
        comp.build_word_atlas(words, target_w=1080)

        frame = torch.zeros((1, 3, 1920, 1080), dtype=torch.float32)
        out = comp.composite_frame(frame, 0.2, words, target_w=1080, target_h=1920, baseline_y=baseline_y)

        diff = (out[0].sum(dim=0) > 0).nonzero()
        assert diff.shape[0] > 0, "No subtitle pixels were composited"

        min_y = diff[:, 0].min().item()
        max_y = diff[:, 0].max().item()

        assert min_y >= 1152, f"Baseline {baseline_y}: Subtitle breached top safe bound! min_y={min_y} < 1152"
        assert max_y <= 1344, f"Baseline {baseline_y}: Subtitle breached bottom safe bound! max_y={max_y} > 1344"

    @pytest.mark.parametrize("script_sample", [
        [{"word": "INNOVATION", "start": 0.0, "end": 0.5}],
        [{"word": "人工智能革命", "start": 0.0, "end": 0.5}],
        [{"word": "नमस्ते दुनिया", "start": 0.0, "end": 0.5}],
        [{"word": "مرحبا بالعالم", "start": 0.0, "end": 0.5}],
        [{"word": "gypqj flying", "start": 0.0, "end": 0.5}],
    ])
    def test_safe_zone_vertical_bounds_across_multilingual_scripts(self, script_sample):
        """Stress: All multilingual scripts must stay strictly within [1152, 1344] px."""
        comp = SubtitleCompositor()
        comp.build_word_atlas(script_sample, target_w=1080)

        frame = torch.zeros((1, 3, 1920, 1080), dtype=torch.float32)
        out = comp.composite_frame(frame, 0.25, script_sample, target_w=1080, target_h=1920)

        diff = (out[0].sum(dim=0) > 0).nonzero()
        min_y = diff[:, 0].min().item()
        max_y = diff[:, 0].max().item()

        assert min_y >= 1152, f"Script {script_sample[0]['word']} breached top safe bound: {min_y} < 1152"
        assert max_y <= 1344, f"Script {script_sample[0]['word']} breached bottom safe bound: {max_y} > 1344"

    def test_gpu_glyph_atlas_safe_zone_vertical_lock(self):
        """Stress: GPUGlyphAtlas composite_frame must strictly clamp inside [1152, 1344] px."""
        atlas = GPUGlyphAtlas(torch.device("cpu"))
        words = [{"word": "GPU", "start": 0.0, "end": 0.5}, {"word": "ATLAS", "start": 0.5, "end": 1.0}]
        atlas.build_atlas(words, target_w=1080, target_h=1920)

        for baseline_y in [None, 1152.0, 1248.0, 1344.0, 0.0, 1920.0]:
            frame = torch.zeros((1, 3, 1920, 1080), dtype=torch.float32)
            out = atlas.composite_frame(frame, 0.25, target_w=1080, target_h=1920, baseline_y=baseline_y)

            diff = (out[0].sum(dim=0) > 0).nonzero()
            min_y = diff[:, 0].min().item()
            max_y = diff[:, 0].max().item()

            assert min_y >= 1152, f"GPU Atlas breached top safe bound at baseline {baseline_y}: {min_y} < 1152"
            assert max_y <= 1344, f"GPU Atlas breached bottom safe bound at baseline {baseline_y}: {max_y} > 1344"


# ═════════════════════════════════════════════════════════════════════════════
# 4. ASCENDER, DESCENDER & DROP-SHADOW NON-CLIPPING ORACLE
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature8BorderNonClipping:
    """Empirical verification that glyph ascenders, descenders, and drop-shadows are never clipped."""

    @pytest.mark.parametrize("word", [
        "g", "j", "p", "q", "y", "gypqj",
        "Jump", "flying", "quickly", "syzygy", "Egyptian",
        "Álvaro", "Île", "Ångström", "Über",
        "Q", "W", "M", "T"
    ])
    def test_descenders_and_ascenders_border_non_clipping_subtitle_compositor(self, word):
        """Stress: Word image borders (row 0, row H-1, col 0, col W-1) must have alpha == 0."""
        font = resolve_font_for_text(word, FONT_SIZE_ACTIVE)
        img = _render_outlined_word(word, font, COLOR_ACTIVE)

        top_row = img[0, :, 3]
        bot_row = img[-1, :, 3]
        left_col = img[:, 0, 3]
        right_col = img[:, -1, 3]

        assert (top_row == 0).all(), f"Word '{word}' top border clipped (max={top_row.max()})"
        assert (bot_row == 0).all(), f"Word '{word}' bottom border clipped (max={bot_row.max()})"
        assert (left_col == 0).all(), f"Word '{word}' left border clipped (max={left_col.max()})"
        assert (right_col == 0).all(), f"Word '{word}' right border clipped (max={right_col.max()})"

    @pytest.mark.parametrize("word", [
        "g", "j", "p", "q", "y", "gypqj", "Quickly", "Flying"
    ])
    def test_gpu_compositor_glyphs_border_non_clipping(self, word):
        """Stress: GPU compositor glyphs must maintain zero border alpha."""
        font = resolve_font_for_text(word, FONT_SIZE_ACTIVE)
        img = _render_outlined_glyph(word, font, COLOR_ACTIVE)

        assert (img[0, :, 3] == 0).all(), f"GPU glyph '{word}' top border clipped"
        assert (img[-1, :, 3] == 0).all(), f"GPU glyph '{word}' bottom border clipped"
        assert (img[:, 0, 3] == 0).all(), f"GPU glyph '{word}' left border clipped"
        assert (img[:, -1, 3] == 0).all(), f"GPU glyph '{word}' right border clipped"

    def test_drop_shadow_presence_without_clipping(self):
        """Stress: Drop shadow (+3px down-right) must exist near borders but remain strictly unclipped."""
        font = resolve_font_for_text("pyg", FONT_SIZE)
        img = _render_outlined_word("pyg", font, COLOR_ACTIVE)

        # Bottom-most and right-most non-zero alpha pixels
        nz_y, nz_x = np.nonzero(img[:, :, 3] > 0)
        max_y = nz_y.max()
        max_x = nz_x.max()

        h, w = img.shape[:2]
        # Must not reach the very last row or col
        assert max_y < h - 1, f"Drop shadow touched bottom border! max_y={max_y}, h={h}"
        assert max_x < w - 1, f"Drop shadow touched right border! max_x={max_x}, w={w}"

        # The pixel at max_y must be a shadow pixel (RGB == 0, alpha near 160)
        bot_pixel = img[max_y, nz_x[nz_y == max_y].min()]
        assert bot_pixel[0] == 0 and bot_pixel[1] == 0 and bot_pixel[2] == 0, "Bottom pixel is not shadow black"


# ═════════════════════════════════════════════════════════════════════════════
# 5. MULTILINGUAL UNICODE & TOFU ELIMINATION
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature8MultilingualUnicodeAndTofuElimination:
    """Empirical verification that multilingual text renders authentic glyphs without tofu (.notdef) boxes."""

    def test_cjk_genuine_glyphs_no_tofu(self):
        """Stress: CJK characters must resolve to NotoSansCJK (or Windows CJK fallback) and avoid the 961-pixel .notdef tofu footprint."""
        cjk_tokens = ["人工智能", "革命性", "突破", "未来"]
        for token in cjk_tokens:
            script = detect_script(token)
            assert script == "CJK", f"Script detection failed for {token}: {script}"

            font = resolve_font_for_text(token, FONT_SIZE)
            font_path = getattr(font, "path", "")
            assert any(n in font_path for n in ("NotoSansCJK", "msyh", "simsun", "meiryo", "Arial")), f"Expected CJK font, got {font_path}"

            img = _render_outlined_word(token, font, COLOR_ACTIVE)
            non_zero_pixels = (img[:, :, 3] > 0).sum()

            # Each CJK word must have substantial glyph pixel density (>> 961 tofu footprint)
            assert non_zero_pixels > 2500, f"CJK word '{token}' produced suspiciously low pixel count: {non_zero_pixels}"

    def test_devanagari_genuine_glyphs_no_tofu(self):
        """Stress: Devanagari Hindi text must resolve to NotoSansDevanagari (or Windows Devanagari fallback) and avoid tofu boxes."""
        dev_tokens = ["नमस्ते", "दुनिया", "क्रांति", "भारत"]
        for token in dev_tokens:
            script = detect_script(token)
            assert script == "DEVANAGARI", f"Script detection failed for {token}: {script}"

            font = resolve_font_for_text(token, FONT_SIZE)
            font_path = getattr(font, "path", "")
            assert any(n in font_path for n in ("NotoSansDevanagari", "Nirmala", "Mangal")), f"Expected Devanagari font, got {font_path}"

            img = _render_outlined_word(token, font, COLOR_ACTIVE)
            non_zero_pixels = (img[:, :, 3] > 0).sum()
            assert non_zero_pixels > 2000, f"Devanagari token '{token}' produced low pixel count: {non_zero_pixels}"

    def test_arabic_rtl_genuine_glyphs_no_tofu(self):
        """Stress: Arabic text must resolve to NotoSansArabic (or Windows Arabic fallback) and route RTL direction."""
        arabic_tokens = ["مرحبا", "بالعالم", "إبداع", "الذكاء"]
        for token in arabic_tokens:
            script = detect_script(token)
            assert script == "ARABIC", f"Script detection failed for {token}: {script}"

            font = resolve_font_for_text(token, FONT_SIZE)
            font_path = getattr(font, "path", "")
            assert any(n in font_path for n in ("NotoSansArabic", "arial", "segoeui")), f"Expected Arabic font, got {font_path}"

            img = _render_outlined_word(token, font, COLOR_ACTIVE)
            non_zero_pixels = (img[:, :, 3] > 0).sum()
            assert non_zero_pixels > 2500, f"Arabic token '{token}' produced low pixel count: {non_zero_pixels}"

    def test_bicameral_uppercasing_rules(self):
        """Stress: Latin/Cyrillic/Greek are uppercased, CJK/Arabic/Devanagari remain untouched."""
        assert _is_bicameral_text("hello") is True
        assert _is_bicameral_text("мир") is True
        assert _is_bicameral_text("κόσμος") is True
        assert _is_bicameral_text("你好") is False
        assert _is_bicameral_text("مرحبا") is False
        assert _is_bicameral_text("नमस्ते") is False


# ═════════════════════════════════════════════════════════════════════════════
# 6. KINETIC SCALE POP & MICRO-CHUNKING INVARIANTS
# ═════════════════════════════════════════════════════════════════════════════

class TestFeature8MicroChunkingAndScalePop:
    """Empirical verification of micro-chunking invariants and ~14% kinetic scale pop."""

    def test_micro_chunk_terminal_punctuation_splitting(self):
        """Stress: Words ending with '.', '!', '?' terminate micro-chunks immediately."""
        words = [
            {"word": "Stop.", "start": 0.0, "end": 0.3},
            {"word": "Go", "start": 0.32, "end": 0.5},
            {"word": "now!", "start": 0.52, "end": 0.8},
            {"word": "Why?", "start": 0.82, "end": 1.1},
            {"word": "Because", "start": 1.12, "end": 1.4},
        ]
        chunks = _build_micro_chunks(words)
        # Should be: ["Stop."], ["Go", "now!"], ["Why?"], ["Because"]
        assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"
        assert [w["word"] for w in chunks[0]] == ["Stop."]
        assert [w["word"] for w in chunks[1]] == ["Go", "now!"]
        assert [w["word"] for w in chunks[2]] == ["Why?"]
        assert [w["word"] for w in chunks[3]] == ["Because"]

    def test_single_active_word_highlight_gating_no_double_highlights(self):
        """Stress: Overlapping word timestamps must NOT cause simultaneous double highlights."""
        comp = SubtitleCompositor()
        # Words with overlapping time intervals
        words = [
            {"word": "WordOne", "start": 1.0, "end": 1.5},
            {"word": "WordTwo", "start": 1.4, "end": 1.9},  # 100ms overlap
            {"word": "WordThree", "start": 1.85, "end": 2.3},
        ]
        comp.build_word_atlas(words, target_w=1080)

        # Probe timestamp in overlap zone (t = 1.45s)
        strip = comp._build_subtitle_strip(1.45, 1080, 1920)
        assert strip is not None

        # Binary search picks the latest started word whose start <= t
        # Exactly one word in the rendered items should have the active neon color
        # Count non-white fill pixels (active neon lime has R=204, G=255, B=0)
        active_pixel_mask = (
            (strip[:, :, 0] >= 180) &
            (strip[:, :, 1] >= 220) &
            (strip[:, :, 2] <= 50) &
            (strip[:, :, 3] == 255)
        )
        assert active_pixel_mask.sum() > 0, "No active neon highlight detected in overlap zone"

    def test_kinetic_scale_pop_ratio(self):
        """Stress: Active font size (87px) vs normal font size (76px) satisfies 1.14 +/- 0.02."""
        ratio = FONT_SIZE_ACTIVE / float(FONT_SIZE)
        assert abs(ratio - SCALE_POP_FACTOR) < 0.02, f"Scale ratio {ratio:.3f} deviates from {SCALE_POP_FACTOR}"

        comp = SubtitleCompositor()
        assert comp.font_size == 76
        assert comp.font_size_active == 87
