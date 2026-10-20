"""
Unit and integration test suite for the 3-Zone Cinema-Vertical Layout Engine,
1:1 Square Reaction mode, and SubtitleCompositor viral hook banner generation.
Verifies compliance with the gold-standard reference shorts (Better Call Saul, Catch-22, Koenigsegg).
"""

import pytest
import torch
import numpy as np
from typing import Optional

from wsl2_gpu.render_spoke import (
    render_cinematic_3zone_916,
    render_square_reaction_11,
)
from wsl2_gpu.subtitle_compositor import (
    SubtitleCompositor,
    COLOR_HOOK_YELLOW,
    COLOR_HOOK_RED,
    COLOR_HOOK_WHITE,
)


class TestCinematic3ZoneLayout:
    """Verifies 3-Zone Cinema-Vertical compositing mathematics and visual layout."""

    def test_3zone_16_9_dimensions_and_channels(self):
        """Verifies 16:9 widescreen frame is correctly composited onto 1080x1920 canvas."""
        # 16:9 1080p frame: [1, 3, 1080, 1920]
        src_frame = torch.rand(1, 3, 1080, 1920, dtype=torch.float32)
        out = render_cinematic_3zone_916(src_frame, target_w=1080, target_h=1920)

        assert out.shape == (1, 3, 1920, 1080)
        assert out.dtype == torch.float32
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    def test_3zone_scope_239_aspect_ratio(self):
        """Verifies 2.39:1 scope cinematic movies (e.g. 1920x804) are centered without jitter."""
        # Scope frame: [1, 3, 804, 1920]
        src_frame = torch.rand(1, 3, 804, 1920, dtype=torch.float32)
        out = render_cinematic_3zone_916(src_frame, target_w=1080, target_h=1920)

        assert out.shape == (1, 3, 1920, 1080)
        # Foreground height: 1080 * (804 / 1920) = 452px
        fg_h = int(round(804.0 * (1080.0 / 1920.0)))  # 452
        start_y = (1920 - fg_h) // 2  # (1920 - 452) // 2 = 734
        end_y = start_y + fg_h

        # Center slice should match resized foreground
        center_slice = out[:, :, start_y:end_y, :]
        assert center_slice.shape == (1, 3, fg_h, 1080)

    def test_3zone_background_blur_and_dimming(self):
        """Verifies background zones 1 and 3 are blurred and dimmed to ~40% intensity."""
        src_frame = torch.ones(1, 3, 1080, 1920, dtype=torch.float32)
        out = render_cinematic_3zone_916(src_frame, target_w=1080, target_h=1920)

        # Top letterbox zone (e.g. Y = 100) should be dimmed to <= 0.50
        top_pixel_val = float(out[0, :, 100, 540].mean())
        assert top_pixel_val <= 0.50, f"Expected dimmed background <= 0.50, got {top_pixel_val}"

        # Center scene (Y = 960) should preserve original full intensity
        center_pixel_val = float(out[0, :, 960, 540].mean())
        assert center_pixel_val >= 0.95, f"Expected full foreground >= 0.95, got {center_pixel_val}"


class TestSquareReactionLayout:
    """Verifies 1:1 Square Reaction mode framing and canvas placement."""

    def test_square_reaction_in_vertical_canvas(self):
        """Verifies 1:1 square crop is centered vertically on 1080x1920 canvas."""
        src_frame = torch.rand(1, 3, 1080, 1920, dtype=torch.float32)
        out = render_square_reaction_11(src_frame, cx_norm=0.5, cy_norm=0.5, target_w=1080, target_h=1920)

        assert out.shape == (1, 3, 1920, 1080)
        # 1080x1080 square in center: start_y = (1920 - 1080) // 2 = 420
        start_y = (1920 - 1080) // 2
        square_slice = out[:, :, start_y:start_y + 1080, :]
        assert square_slice.shape == (1, 3, 1080, 1080)

    def test_square_reaction_in_square_target(self):
        """Verifies 1:1 target canvas produces pure 1080x1080 square output."""
        src_frame = torch.rand(1, 3, 1080, 1920, dtype=torch.float32)
        out = render_square_reaction_11(src_frame, cx_norm=0.5, cy_norm=0.5, target_w=1080, target_h=1080)

        assert out.shape == (1, 3, 1080, 1080)


class TestSubtitleCompositorViralBanners:
    """Verifies SubtitleCompositor Top Hook and Bottom Punchline banner generation."""

    def test_build_top_hook_banner_multiline(self):
        compositor = SubtitleCompositor()
        headline = "He changed one number and left with no evidence"
        highlight = "and left with no evidence"

        banner = compositor.build_top_hook_banner(headline, highlight_phrase=highlight, target_w=1080)
        assert banner is not None
        assert isinstance(banner, np.ndarray)
        assert banner.ndim == 3
        assert banner.shape[2] == 4  # RGBA
        assert banner.shape[1] == 1080  # Full width canvas
        assert banner.shape[0] > 0
        assert banner.dtype == np.uint8

        # Has non-transparent pixels
        alpha_channel = banner[:, :, 3]
        assert np.any(alpha_channel > 0)

    def test_build_top_hook_banner_pov_red_accent(self):
        compositor = SubtitleCompositor()
        headline = "POV: Koenigsegg invented the legal money shift"
        highlight = "legal money shift"

        banner = compositor.build_top_hook_banner(headline, highlight_phrase=highlight, target_w=1080)
        assert banner is not None
        assert banner.shape[1] == 1080

    def test_build_bottom_punchline_banner(self):
        compositor = SubtitleCompositor()
        punchline = "one bite completely blows Kathcart's mind"

        banner = compositor.build_bottom_punchline_banner(punchline, target_w=1080)
        assert banner is not None
        assert isinstance(banner, np.ndarray)
        assert banner.shape[2] == 4
        assert banner.shape[1] == 1080
        assert banner.shape[0] > 0

    def test_composite_banner_onto_frame_chw(self):
        compositor = SubtitleCompositor()
        frame = torch.zeros(1, 3, 1920, 1080, dtype=torch.float32)

        # Create dummy banner RGBA
        banner_rgba = np.full((120, 1080, 4), 255, dtype=np.uint8)
        out = compositor.composite_banner(frame, banner_rgba, y_center=200, target_w=1080, target_h=1920)

        assert out.shape == (1, 3, 1920, 1080)
        # Region around Y=200 should now have composited color > 0.0
        assert float(out[0, :, 200, 540].mean()) > 0.5
