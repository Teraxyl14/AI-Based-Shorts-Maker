"""
Token-budget tests for the VLM editorial prompt.

Regression: a 332s transcript (~1,200 words) + 6 keyframes + max_tokens=1024
exceeded vLLM's 3,072-token context (HTTP 400), silently forcing heuristic
fallback selection. The prompt must always fit the configured context window.
"""
import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from wsl2_gpu import editorial_engine as ee  # noqa: E402


def _words(duration_s: float, wps: float = 3.6):
    n = int(duration_s * wps)
    step = duration_s / n
    return [
        {"word": f"word{i % 97}", "start": i * step, "end": i * step + step * 0.8}
        for i in range(n)
    ]


def _keyframes(n: int, duration_s: float):
    return [
        {"shot_idx": i, "start_s": 0.0, "end_s": 1.0,
         "midpoint_s": (i + 0.5) * duration_s / n, "duration_s": 3.0,
         "base64_jpeg": "AAAA"}
        for i in range(n)
    ]


def _estimated_prompt_tokens(messages):
    text_chars = 0
    images = 0
    for block in messages[0]["content"]:
        if block["type"] == "text":
            text_chars += len(block["text"])
        else:
            images += 1
    return text_chars / ee.CHARS_PER_TOKEN + images * ee.TOKENS_PER_IMAGE


@pytest.mark.parametrize("duration_s", [120.0, 332.4, 1800.0, 3600.0])
def test_prompt_fits_context_window(duration_s):
    engine = ee.EditorialEngine()
    msgs = engine.build_interleaved_prompt(_keyframes(6, duration_s), _words(duration_s), duration_s)
    budget = ee.VLM_CONTEXT_TOKENS - ee.VLM_OUTPUT_TOKENS - ee.PROMPT_SAFETY_TOKENS
    assert _estimated_prompt_tokens(msgs) <= budget


def test_output_reservation_leaves_room_for_four_shorts():
    # ~120 tokens per JSON short incl. reasoning; 4 shorts must fit.
    assert ee.VLM_OUTPUT_TOKENS >= ee.TARGET_SHORTS * 120


def test_transcript_preserves_selectable_window_coverage():
    """Compaction must keep timestamps spanning the selectable range (>= 60s)."""
    duration_s = 1800.0
    engine = ee.EditorialEngine()
    msgs = engine.build_interleaved_prompt(_keyframes(6, duration_s), _words(duration_s), duration_s)
    text = msgs[0]["content"][0]["text"]
    assert "[01:" in text or "[02:" in text          # early selectable region present
    assert "[29:" in text or "[28:" in text          # tail of video present
    assert "[00:1" not in text                       # pre-teaser lines (<~50s) dropped


def test_short_video_transcript_kept_verbatim():
    duration_s = 120.0
    words = _words(duration_s)
    engine = ee.EditorialEngine()
    msgs = engine.build_interleaved_prompt(_keyframes(4, duration_s), words, duration_s)
    text = msgs[0]["content"][0]["text"]
    late_words = [w["word"] for w in words if w["start"] >= 60.0]
    # All words in the selectable window survive when budget allows.
    assert all(w in text for w in set(late_words))
