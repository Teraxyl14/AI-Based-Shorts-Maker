"""
Milestone 6.2 Empirical Stress Test Suite (Adversarial Challenger)
Compositor Stress Testing: Zero-Letterbox Dynamic 3-Mode Compositor

Targets:
1. Pixel-level verification: Assert ZERO unstyled 16:9 letterbox padding in Mode 2 and Mode 3 outputs.
2. Glowing divider aesthetics: Verify RGB color (204, 255, 0) (#CCFF00), 3-4px line thickness, Gaussian halo blur, and boundary symmetry.
3. Subtitle safe zone isolation: Test collision boundaries with multi-line kinetic subtitle text rendered at Y in [1152, 1344]px.
4. Extreme input dimensions: Test single-pixel frames, ultra-wide (32:9), and ultra-tall (9:32) aspect ratios.
"""

import math
import pytest
import numpy as np
import torch

from wsl2_gpu.gpu_compositor import (
    render_styled_divider,
    render_dual_pane_split_screen,
    render_full_frame_topic_closeup,
    calculate_dynamic_topic_crop_2d,
    verify_subtitle_safe_zone_isolation,
    SubtitleCompositor,
    COLOR_NEON_LIME,
    SAFE_ZONE_MIN_RATIO,
    SAFE_ZONE_MAX_RATIO,
    SUBTITLE_Y_ANCHOR,
)
from wsl2_gpu.render_spoke import (
    render_smart_portrait_pan,
    render_canvas_graphic_fit,
)
from shared.schemas import SplitScreenConfig


# ═════════════════════════════════════════════════════════════════════════════
# 1. PIXEL-LEVEL VERIFICATION: ZERO UNSTYLED 16:9 LETTERBOX PADDING
# ═════════════════════════════════════════════════════════════════════════════

class TestPixelLevelZeroLetterboxElimination:
    """Empirical verification that Mode 2 and Mode 3 eliminate unstyled 16:9 letterboxing."""

    def test_mode2_zero_letterbox_frequency_analysis(self):
        """Mode 2 (FULL_FRAME_TOPIC_CLOSEUP):
        Verifies that top, middle, and bottom vertical regions have active, high-frequency
        content rather than blurred/darkened letterbox padding.
        """
        torch.manual_seed(42)
        frame = torch.rand(1, 3, 1080, 1920)

        # Legacy unstyled letterbox
        legacy = render_canvas_graphic_fit(frame, dynamic_remap=False)

        # Mode 2 full-frame closeup
        m2 = render_full_frame_topic_closeup(frame, cx_norm=0.5, cy_norm=0.5)

        assert m2.shape == (1, 3, 1920, 1080)

        def vertical_gradient_magnitude(img, y_start, y_end):
            region = img[:, :, y_start:y_end, :]
            dy = torch.abs(region[:, :, 1:, :] - region[:, :, :-1, :])
            return dy.mean().item()

        # In legacy letterbox, top (100..500) and bottom (1400..1800) are heavily blurred
        legacy_top_grad = vertical_gradient_magnitude(legacy, 100, 500)
        legacy_mid_grad = vertical_gradient_magnitude(legacy, 700, 1100)
        legacy_bot_grad = vertical_gradient_magnitude(legacy, 1400, 1800)

        assert legacy_top_grad < 0.005, f"Legacy letterbox should be blurred at top, got {legacy_top_grad}"
        assert legacy_bot_grad < 0.005, f"Legacy letterbox should be blurred at bottom, got {legacy_bot_grad}"
        assert legacy_mid_grad > 0.15, "Legacy letterbox middle should have sharp content"

        # In Mode 2, all regions must have sharp, unblurred gradient (> 0.08)
        m2_top_grad = vertical_gradient_magnitude(m2, 100, 500)
        m2_mid_grad = vertical_gradient_magnitude(m2, 700, 1100)
        m2_bot_grad = vertical_gradient_magnitude(m2, 1400, 1800)

        assert m2_top_grad > 0.08, f"Mode 2 top must have sharp content, got {m2_top_grad}"
        assert m2_mid_grad > 0.08, f"Mode 2 middle must have sharp content, got {m2_mid_grad}"
        assert m2_bot_grad > 0.08, f"Mode 2 bottom must have sharp content, got {m2_bot_grad}"

        # Standard deviation across every 100px band must be strictly non-zero
        for y_band in range(0, 1920, 100):
            band = m2[0, :, y_band:min(1920, y_band + 100), :]
            assert band.std().item() > 0.05, f"Band at Y={y_band} appears unstyled/flat"

    def test_mode3_dual_pane_zero_letterbox_fill(self):
        """Mode 3 (DUAL_PANE_SPLIT_SCREEN):
        Verifies top pane (0..958) and bottom pane (962..1920) are fully populated with active
        sharp content and zero black bars or blurred letterbox bands.
        """
        torch.manual_seed(99)
        frame = torch.rand(1, 3, 1080, 1920)

        m3 = render_dual_pane_split_screen(frame, speaker_cx=0.3, topic_cx=0.7, topic_cy=0.6)
        assert m3.shape == (1, 3, 1920, 1080)

        # Top pane check
        top_slice = m3[0, :, 50:900, :]
        assert top_slice.std().item() > 0.10, "Top pane must contain active dynamic content"

        # Bottom pane check
        bot_slice = m3[0, :, 1000:1850, :]
        assert bot_slice.std().item() > 0.10, "Bottom pane must contain active dynamic content"

        # Boundary checks: zero black border rows at top or bottom
        assert m3[0, :, 0, :].mean().item() > 0.10
        assert m3[0, :, 1919, :].mean().item() > 0.10
        assert m3[0, :, :, 0].mean().item() > 0.10
        assert m3[0, :, :, 1079].mean().item() > 0.10

    @pytest.mark.parametrize("aspect_name, src_shape", [
        ("16:9", (1080, 1920)),
        ("4:3", (1080, 1440)),
        ("1:1", (1080, 1080)),
        ("21:9", (1080, 2560)),
        ("32:9", (1080, 3840)),
        ("9:16", (1920, 1080)),
        ("9:32", (3840, 1080)),
    ])
    def test_zero_letterbox_across_aspect_ratios(self, aspect_name, src_shape):
        """Asserts zero unstyled letterbox padding across diverse aspect ratios."""
        h, w = src_shape
        torch.manual_seed(123)
        frame = torch.rand(1, 3, h, w)

        # Mode 2
        m2 = render_full_frame_topic_closeup(frame, cx_norm=0.5, cy_norm=0.5)
        assert m2.shape == (1, 3, 1920, 1080)
        assert m2[0, :, 0:200, :].std().item() > 0.05
        assert m2[0, :, 1720:1920, :].std().item() > 0.05

        # Mode 3
        m3 = render_dual_pane_split_screen(frame, speaker_cx=0.5, topic_cx=0.5, topic_cy=0.5)
        assert m3.shape == (1, 3, 1920, 1080)
        assert m3[0, :, 0:200, :].std().item() > 0.05
        assert m3[0, :, 1720:1920, :].std().item() > 0.05


# ═════════════════════════════════════════════════════════════════════════════
# 2. GLOWING DIVIDER AESTHETICS: COLOR, THICKNESS, AND GAUSSIAN HALO
# ═════════════════════════════════════════════════════════════════════════════

class TestGlowingDividerAesthetics:
    """Stress tests aesthetic requirements: RGB (204, 255, 0), thickness, and halo blur."""

    def test_divider_rgb_color_and_thickness_4px(self):
        """Verifies core line RGB is #CCFF00 (204, 255, 0) and thickness is exactly 4px."""
        canvas = torch.full((1, 3, 1920, 1080), 0.2, dtype=torch.float32)
        out = render_styled_divider(canvas, split_y=960, divider_thickness=4, glow_radius=6, shadow_blur=10)

        expected_r = 204.0 / 255.0  # 0.8000
        expected_g = 255.0 / 255.0  # 1.0000
        expected_b = 0.0            # 0.0000

        # For thickness 4 and split_y 960: y_start = 958, y_end = 962
        core_rows = [958, 959, 960, 961]
        for y in core_rows:
            r = out[0, 0, y, 500].item()
            g = out[0, 1, y, 500].item()
            b = out[0, 2, y, 500].item()
            assert abs(r - expected_r) < 1e-4, f"Row {y} R expected {expected_r}, got {r}"
            assert abs(g - expected_g) < 1e-4, f"Row {y} G expected {expected_g}, got {g}"
            assert abs(b - expected_b) < 1e-4, f"Row {y} B expected {expected_b}, got {b}"

        # Exact thickness count: exactly 4 consecutive rows with full core RGB
        is_core = (torch.abs(out[0, 0, :, 500] - expected_r) < 1e-4) & \
                  (torch.abs(out[0, 1, :, 500] - expected_g) < 1e-4) & \
                  (torch.abs(out[0, 2, :, 500] - expected_b) < 1e-4)
        core_count = is_core.sum().item()
        assert core_count == 4, f"Expected exactly 4px core thickness, got {core_count}px"

    def test_divider_rgb_color_and_thickness_3px(self):
        """Verifies core line RGB is #CCFF00 and thickness is exactly 3px for thickness=3."""
        canvas = torch.full((1, 3, 1920, 1080), 0.2, dtype=torch.float32)
        out = render_styled_divider(canvas, split_y=960, divider_thickness=3, glow_radius=6, shadow_blur=10)

        expected_r = 204.0 / 255.0
        expected_g = 255.0 / 255.0
        expected_b = 0.0

        core_rows = [959, 960, 961]
        for y in core_rows:
            r = out[0, 0, y, 500].item()
            g = out[0, 1, y, 500].item()
            b = out[0, 2, y, 500].item()
            assert abs(r - expected_r) < 1e-4, f"Row {y} R expected {expected_r}, got {r}"
            assert abs(g - expected_g) < 1e-4, f"Row {y} G expected {expected_g}, got {g}"
            assert abs(b - expected_b) < 1e-4, f"Row {y} B expected {expected_b}, got {b}"

        is_core = (torch.abs(out[0, 0, :, 500] - expected_r) < 1e-4) & \
                  (torch.abs(out[0, 1, :, 500] - expected_g) < 1e-4) & \
                  (torch.abs(out[0, 2, :, 500] - expected_b) < 1e-4)
        core_count = is_core.sum().item()
        assert core_count == 3, f"Expected exactly 3px core thickness, got {core_count}px"

    def test_gaussian_halo_blur_monotonic_falloff(self):
        """Verifies Gaussian halo blur falloff decreases monotonically with distance from core."""
        # Black canvas to measure pure additive glow
        canvas = torch.zeros((1, 3, 1920, 1080), dtype=torch.float32)
        glow_radius = 6
        out = render_styled_divider(canvas, split_y=960, divider_thickness=4, glow_radius=glow_radius, shadow_blur=10)

        # Above core line (y_start = 958): y = 957, 956, 955, ..., 945
        glow_values = [out[0, 1, y, 500].item() for y in range(957, 944, -1)]

        # Check strict monotonic decay
        for i in range(len(glow_values) - 1):
            assert glow_values[i] > glow_values[i + 1], (
                f"Gaussian halo at d={i+1} ({glow_values[i]:.4f}) must be > d={i+2} ({glow_values[i+1]:.4f})"
            )

        # Check Gaussian mathematical conformance: glow ~ exp(-d^2 / (2 * sigma^2))
        sigma = float(glow_radius)
        expected_ratio = math.exp(- (1.0 ** 2) / (2.0 * sigma ** 2)) / math.exp(- (6.0 ** 2) / (2.0 * sigma ** 2))
        actual_ratio = glow_values[0] / glow_values[5]
        assert abs(actual_ratio - expected_ratio) < 0.05, (
            f"Glow ratio at 1px vs 6px expected ~{expected_ratio:.3f}, got {actual_ratio:.3f}"
        )

    def test_divider_glow_boundary_continuity_defect_identification(self):
        """CRITICAL CHALLENGE: Identifies the 1-pixel glow gap/discontinuity bug at Y=962 for thickness=4.
        
        At divider_thickness=4:
        y_start = 958, y_end = 962. Core covers [958, 961].
        At dy = +2, y_pos = 962, dist = 2 <= half_th (2).
        Because dist <= half_th, no glow is applied at Y=962!
        And because y_end = 962 (exclusive), no core is applied at Y=962!
        Therefore, Y=962 is an untreated 1px gap where glow is 0.000, creating an asymmetric halo!
        """
        canvas = torch.zeros((1, 3, 1920, 1080), dtype=torch.float32)
        out = render_styled_divider(canvas, split_y=960, divider_thickness=4, glow_radius=6, shadow_blur=10)

        glow_1px_above = out[0, 1, 957, 500].item()  # 1px above core line (y=958)
        glow_1px_below = out[0, 1, 962, 500].item()  # 1px below core line (y=961)

        # In a symmetric, well-formed divider, glow 1px above and 1px below should match:
        # Here we document the empirical defect:
        is_asymmetric_gap = (glow_1px_below == 0.0 and glow_1px_above > 0.60)
        assert is_asymmetric_gap, (
            f"Expected defect observation: glow at Y=962 is {glow_1px_below}, while Y=957 is {glow_1px_above}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. SUBTITLE SAFE ZONE ISOLATION: MULTI-LINE KINETIC TYPOGRAPHY OVERLAY
# ═════════════════════════════════════════════════════════════════════════════

class TestSubtitleSafeZoneIsolation:
    """Stress tests boundary isolation between divider at Y=960 and subtitle safe zone [1152, 1344]."""

    def test_multi_line_bursts_zero_collision_with_divider(self):
        """Tests multi-line kinetic subtitle text rendered at Y in [1152, 1344].
        Asserts zero collision / zero alteration of divider line at Y in [944, 976].
        """
        frame = torch.zeros(1, 3, 1080, 1920)
        split_canvas = render_dual_pane_split_screen(frame, target_w=1080, target_h=1920)

        # Multi-line kinetic words
        multiline_words = [
            {"word": "FIRST_LINE_TEXT", "start": 1.0, "end": 1.5},
            {"word": "SECOND_LINE_TEXT", "start": 1.5, "end": 2.0},
            {"word": "THIRD_LINE_TEXT", "start": 2.0, "end": 2.5},
        ]

        comp = SubtitleCompositor()
        comp.build_word_atlas(multiline_words, target_w=1080)

        for t in [1.25, 1.75, 2.25]:
            composited = comp.composite_frame(
                split_canvas.clone(),
                timestamp=t,
                words=multiline_words,
                target_w=1080,
                target_h=1920
            )

            # 1. Divider region Y in [944, 976] must be 100% identical to un-subtitled canvas
            orig_divider = split_canvas[0, :, 944:976, :]
            comp_divider = composited[0, :, 944:976, :]
            divider_diff = torch.abs(orig_divider - comp_divider).max().item()
            assert divider_diff == 0.0, f"Subtitle collided with divider at t={t}: diff={divider_diff}"

            # 2. Above safe zone (Y < 1152) must have ZERO subtitle bleeding
            orig_above = split_canvas[0, :, :1152, :]
            comp_above = composited[0, :, :1152, :]
            above_diff = torch.abs(orig_above - comp_above).max().item()
            assert above_diff == 0.0, f"Subtitle bled above safe zone (Y < 1152) at t={t}: diff={above_diff}"

            # 3. Subtitle text must be active in vertical safe zone [1152, 1344]
            sz_diff = torch.abs(composited[0, :, 1152:1344, :] - split_canvas[0, :, 1152:1344, :]).max().item()
            assert sz_diff > 0.5, f"Subtitle text should be rendered in safe zone at t={t}"

    def test_adversarial_baseline_y_clamping_to_safe_zone(self):
        """Adversarially requests baseline_y directly on divider (baseline_y=960).
        Verifies that composite_frame strictly clamps placement so text cannot move above Y=1152.
        """
        frame = torch.zeros(1, 3, 1080, 1920)
        split_canvas = render_dual_pane_split_screen(frame, target_w=1080, target_h=1920)

        words = [{"word": "ADVERSARIAL", "start": 1.0, "end": 2.0}]
        comp = SubtitleCompositor()
        comp.build_word_atlas(words, target_w=1080)

        # Force baseline_y = 960 (attempting collision with divider)
        out = comp.composite_frame(
            split_canvas.clone(),
            timestamp=1.5,
            words=words,
            target_w=1080,
            target_h=1920,
            baseline_y=960.0
        )

        # Assert no change above 1152
        diff_above = torch.abs(out[0, :, :1152, :] - split_canvas[0, :, :1152, :]).max().item()
        assert diff_above == 0.0, f"Subtitle leaked above safe zone despite clamp: diff={diff_above}"

    def test_split_ratio_safe_zone_collision_detection(self):
        """Verifies that verify_subtitle_safe_zone_isolation detects collisions when split_ratio
        places the divider within 150px of or inside the subtitle safe zone.
        """
        # Default split_ratio 0.50 (split_y = 960) -> 184px clearance >= 150px -> PASS
        assert verify_subtitle_safe_zone_isolation(split_y=960) is True

        # split_ratio 0.55 (split_y = 1056) -> clearance 88px < 150px -> FAIL
        assert verify_subtitle_safe_zone_isolation(split_y=1056) is False

        # split_ratio 0.65 (split_y = 1248, inside safe zone) -> clearance -104px -> FAIL
        assert verify_subtitle_safe_zone_isolation(split_y=1248) is False


# ═════════════════════════════════════════════════════════════════════════════
# 4. EXTREME INPUT DIMENSIONS: 1x1, ULTRA-WIDE 32:9, AND ULTRA-TALL 9:32
# ═════════════════════════════════════════════════════════════════════════════

class TestExtremeInputDimensions:
    """Stress tests extreme input dimensions across Mode 1, Mode 2, and Mode 3."""

    def test_single_pixel_frame_resilience(self):
        """Single-pixel input frame: frame = torch.rand(1, 3, 1, 1)."""
        f_1x1 = torch.rand(1, 3, 1, 1)

        # Mode 1
        m1 = render_smart_portrait_pan(f_1x1, cx_norm=0.5)
        assert m1.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(m1).any()
        assert not torch.isinf(m1).any()

        # Mode 2
        m2 = render_full_frame_topic_closeup(f_1x1, cx_norm=0.5, cy_norm=0.5)
        assert m2.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(m2).any()
        assert not torch.isinf(m2).any()

        # Mode 3
        m3 = render_dual_pane_split_screen(f_1x1, speaker_cx=0.5, topic_cx=0.5, topic_cy=0.5)
        assert m3.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(m3).any()
        assert not torch.isinf(m3).any()

    def test_ultrawide_32_9_aspect_ratio(self):
        """Ultra-wide 32:9 input frame: (1080, 3840)."""
        f_32_9 = torch.rand(1, 3, 1080, 3840)

        # Mode 1
        m1 = render_smart_portrait_pan(f_32_9, cx_norm=0.5)
        assert m1.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(m1).any()

        # Mode 2
        m2 = render_full_frame_topic_closeup(f_32_9, cx_norm=0.85, cy_norm=0.5)
        assert m2.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(m2).any()

        # Mode 3
        m3 = render_dual_pane_split_screen(f_32_9, speaker_cx=0.2, topic_cx=0.8, topic_cy=0.5)
        assert m3.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(m3).any()

    def test_ultratall_9_32_aspect_ratio(self):
        """Ultra-tall 9:32 input frame: (3840, 1080)."""
        f_9_32 = torch.rand(1, 3, 3840, 1080)

        # Mode 1
        m1 = render_smart_portrait_pan(f_9_32, cx_norm=0.5)
        assert m1.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(m1).any()

        # Mode 2
        m2 = render_full_frame_topic_closeup(f_9_32, cx_norm=0.5, cy_norm=0.2)
        assert m2.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(m2).any()

        # Mode 3
        m3 = render_dual_pane_split_screen(f_9_32, speaker_cx=0.5, topic_cx=0.5, topic_cy=0.7)
        assert m3.shape == (1, 3, 1920, 1080)
        assert not torch.isnan(m3).any()

    def test_extreme_aspect_1x1000_and_1000x1(self):
        """Extreme 1000:1 and 1:1000 aspect ratio frames."""
        f_tall = torch.rand(1, 3, 1000, 1)
        f_wide = torch.rand(1, 3, 1, 1000)

        for f in [f_tall, f_wide]:
            m2 = render_full_frame_topic_closeup(f, cx_norm=0.5, cy_norm=0.5)
            assert m2.shape == (1, 3, 1920, 1080)
            assert not torch.isnan(m2).any()

            m3 = render_dual_pane_split_screen(f, speaker_cx=0.5, topic_cx=0.5, topic_cy=0.5)
            assert m3.shape == (1, 3, 1920, 1080)
            assert not torch.isnan(m3).any()
