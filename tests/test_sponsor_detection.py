"""
Unit tests for deterministic in-video sponsor read detection and exclusion in AetherEditorialEngine.
"""

import pytest
from wsl2_gpu.editorial_engine import EditorialEngine, detect_sponsor_segments
from shared.schemas import ShortCandidate, WordTimestamp


def test_detects_sponsor_phrases():
    """Detects multi-word sponsor phrases like 'sponsored by' and 'link down below'."""
    transcript = [
        {"word": "today's", "start": 10.0, "end": 10.4},
        {"word": "video", "start": 10.5, "end": 10.8},
        {"word": "is", "start": 10.9, "end": 11.0},
        {"word": "sponsored", "start": 11.1, "end": 11.5},
        {"word": "by", "start": 11.6, "end": 11.8},
        {"word": "our", "start": 11.9, "end": 12.0},
        {"word": "friends", "start": 12.1, "end": 12.4},
        {"word": "at", "start": 12.5, "end": 12.6},
        {"word": "hetzner", "start": 12.7, "end": 13.0},
        {"word": "check", "start": 13.1, "end": 13.3},
        {"word": "out", "start": 13.4, "end": 13.5},
        {"word": "the", "start": 13.6, "end": 13.7},
        {"word": "link", "start": 13.8, "end": 14.0},
        {"word": "down", "start": 14.1, "end": 14.3},
        {"word": "below", "start": 14.4, "end": 14.7},
    ]

    segments = detect_sponsor_segments(transcript, padding_s=1.0)
    assert len(segments) >= 1
    # Check that sponsor window spans around 10s to 14.7s
    s_start, s_end, reason = segments[0]
    assert s_start <= 11.1
    assert s_end >= 14.7


def test_hetzner_sponsor_read_detected_from_testvid():
    """Detects the exact Hetzner sponsor read from TestVid.mp4 (53.2s to 68.2s)."""
    transcript = [
        {"word": "we're", "start": 53.22, "end": 53.40},
        {"word": "still", "start": 53.45, "end": 53.60},
        {"word": "going", "start": 53.65, "end": 53.80},
        {"word": "to", "start": 53.85, "end": 53.95},
        {"word": "get", "start": 54.00, "end": 54.10},
        {"word": "to", "start": 54.15, "end": 54.25},
        {"word": "our", "start": 54.30, "end": 54.40},
        {"word": "sponsor", "start": 54.45, "end": 54.85},
        {"word": "though", "start": 54.90, "end": 55.10},
        {"word": "hetzner", "start": 55.36, "end": 55.80},
        {"word": "with", "start": 55.85, "end": 56.00},
        {"word": "their", "start": 56.05, "end": 56.20},
        {"word": "new", "start": 56.25, "end": 56.40},
        {"word": "singapore", "start": 56.45, "end": 56.90},
        {"word": "location", "start": 56.95, "end": 57.30},
        {"word": "check", "start": 64.40, "end": 64.60},
        {"word": "out", "start": 64.65, "end": 64.80},
        {"word": "their", "start": 64.85, "end": 65.00},
        {"word": "metered", "start": 65.05, "end": 65.40},
        {"word": "options", "start": 65.45, "end": 65.80},
        {"word": "using", "start": 65.85, "end": 66.10},
        {"word": "our", "start": 66.15, "end": 66.30},
        {"word": "link", "start": 66.35, "end": 66.50},
        {"word": "down", "start": 66.55, "end": 66.70},
        {"word": "below", "start": 66.75, "end": 67.10},
    ]

    segments = detect_sponsor_segments(transcript, padding_s=2.0)
    assert len(segments) >= 1
    # Check that sponsor start covers 53s-54s and ends after 67s
    sp_start, sp_end, reason = segments[0]
    assert sp_start <= 53.22
    assert sp_end >= 67.10


def test_zero_false_positive_on_clean_content():
    """Asserts that normal dialogue about programming or tech does not trigger sponsor detection."""
    clean_transcript = [
        {"word": "windows", "start": 1.0, "end": 1.4},
        {"word": "xp", "start": 1.5, "end": 1.8},
        {"word": "was", "start": 1.9, "end": 2.1},
        {"word": "the", "start": 2.2, "end": 2.3},
        {"word": "first", "start": 2.4, "end": 2.7},
        {"word": "operating", "start": 2.8, "end": 3.2},
        {"word": "system", "start": 3.3, "end": 3.6},
        {"word": "to", "start": 3.7, "end": 3.8},
        {"word": "unify", "start": 3.9, "end": 4.2},
        {"word": "the", "start": 4.3, "end": 4.4},
        {"word": "consumer", "start": 4.5, "end": 4.9},
        {"word": "and", "start": 5.0, "end": 5.1},
        {"word": "business", "start": 5.2, "end": 5.6},
        {"word": "lines", "start": 5.7, "end": 6.0},
    ]

    segments = detect_sponsor_segments(clean_transcript)
    assert len(segments) == 0


def test_editorial_engine_rejects_sponsor_candidate():
    """EditorialEngine._validate_and_filter rejects a candidate short intersecting a sponsor read."""
    engine = EditorialEngine(min_start_time=10.0, min_duration=30.0, max_duration=60.0)

    transcript = [
        {"word": "start", "start": 65.0, "end": 65.5},
        {"word": "sponsored", "start": 80.0, "end": 80.5},
        {"word": "by", "start": 80.6, "end": 80.8},
        {"word": "our", "start": 80.9, "end": 81.0},
        {"word": "partner", "start": 81.1, "end": 81.5},
        {"word": "link", "start": 82.0, "end": 82.2},
        {"word": "down", "start": 82.3, "end": 82.5},
        {"word": "below", "start": 82.6, "end": 82.9},
        {"word": "end", "start": 95.0, "end": 95.5},
    ]

    # Candidate 1: overlaps sponsor segment (65s to 105s) -> must be rejected
    cand_dirty = ShortCandidate(
        start_time_seconds=65.0,
        duration_seconds=40.0,
        title="Dirty Sponsor Short",
        reasoning="Test",
        framing_mode="SMART_PORTRAIT_PAN"
    )

    # Candidate 2: clean short (120s to 160s) -> must be accepted
    cand_clean = ShortCandidate(
        start_time_seconds=120.0,
        duration_seconds=35.0,
        title="Clean Tech Short",
        reasoning="Test",
        framing_mode="SMART_PORTRAIT_PAN"
    )

    # Words for clean candidate
    clean_words = transcript + [
        {"word": "here", "start": 120.5, "end": 120.8},
        {"word": "is", "start": 120.9, "end": 121.0},
        {"word": "great", "start": 121.1, "end": 121.4},
        {"word": "code", "start": 121.5, "end": 121.8},
        {"word": "finish.", "start": 154.5, "end": 154.9},
    ]

    valid = engine._validate_and_filter(
        candidates=[cand_dirty, cand_clean],
        video_duration=200.0,
        transcript_words=clean_words
    )

    titles = [c.title for c in valid]
    assert "Dirty Sponsor Short" not in titles
    assert "Clean Tech Short" in titles


def test_select_fallback_shorts_excludes_sponsor_intervals():
    """select_fallback_shorts never schedules a fallback candidate inside a sponsor read interval."""
    engine = EditorialEngine(min_start_time=10.0, min_duration=30.0, max_duration=40.0)

    # Transcript with sponsor read from 40s to 60s
    words = [
        {"word": "opening", "start": 10.5, "end": 11.0},
        {"word": "speech", "start": 11.1, "end": 11.5},
        {"word": "topic.", "start": 38.5, "end": 39.0},
        # Sponsor block:
        {"word": "sponsored", "start": 42.0, "end": 42.5},
        {"word": "by", "start": 42.6, "end": 42.8},
        {"word": "link", "start": 43.0, "end": 43.2},
        {"word": "down", "start": 43.3, "end": 43.5},
        {"word": "below.", "start": 43.6, "end": 44.0},
        # Resume genuine content:
        {"word": "next", "start": 65.0, "end": 65.3},
        {"word": "content", "start": 65.4, "end": 65.8},
        {"word": "concludes.", "start": 100.0, "end": 100.5},
    ]

    fallbacks = engine.select_fallback_shorts(
        transcript_words=words,
        shots=None,
        video_duration=200.0,
        existing=[],
        needed=2
    )

    for fb in fallbacks:
        # None of the fallbacks should overlap the sponsor window (40.0s to 55.0s)
        assert not (fb.start_time_seconds < 50.0 and fb.end_time_seconds > 40.0), \
            f"Fallback short [{fb.start_time_seconds}-{fb.end_time_seconds}] overlapped sponsor read!"
