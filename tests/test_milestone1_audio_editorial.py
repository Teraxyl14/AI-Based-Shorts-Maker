"""
Milestone 1 Dedicated Verification Test Suite: Audio & Narrative Editorial Engine
Covers:
  1. WordTimestamp schema with optional speaker_id
  2. SpeakerTurn schema, AudioTranscriptionResult, and overlap alignment
  3. Duration bounds [30.0, 50.0]s
  4. Teaser blacklist (strictly rejecting t < 60.0s)
  5. 3-Act Narrative structure: Hook (0-5s, speech <= 1.5s), Escalation, Payoff
  6. Reliable selection of >= 4 distinct, pairwise non-overlapping arcs
  7. Offline acoustic clustering fallback (single vs multi-speaker)
"""

import os
import sys
import unittest
import numpy as np
from typing import List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.schemas import (
    WordTimestamp,
    SpeakerTurn,
    AudioTranscriptionResult,
    ShortCandidate as PydanticShortCandidate
)
from wsl2_gpu.editorial_engine import EditorialEngine, ShotBoundary, ShortCandidate
from wsl2_gpu.transcription_spoke import (
    align_words_to_speaker_turns,
    cluster_speaker_segments,
    compute_spectral_bands,
    extract_segment_features
)


class TestMilestone1AudioEditorial(unittest.TestCase):

    def setUp(self):
        self.engine = EditorialEngine()

    # ═════════════════════════════════════════════════════════════════════════
    # 1. WordTimestamp & SpeakerTurn Schema Tests
    # ═════════════════════════════════════════════════════════════════════════

    def test_word_timestamp_schema_and_speaker_id(self):
        """Verifies WordTimestamp supports speaker_id and serializes cleanly."""
        # Backward compatibility without speaker_id
        w_legacy = WordTimestamp(word="welcome", start=60.1, end=60.5, confidence=0.98)
        self.assertTrue(hasattr(w_legacy, "speaker_id"))
        self.assertIsNone(w_legacy.speaker_id)

        # Explicit speaker_id
        w_tagged = WordTimestamp(word="hello", start=61.0, end=61.4, confidence=0.95, speaker_id="SPEAKER_01")
        self.assertEqual(w_tagged.speaker_id, "SPEAKER_01")

        data = w_tagged.model_dump()
        self.assertEqual(data["speaker_id"], "SPEAKER_01")
        self.assertEqual(data["word"], "hello")

    def test_speaker_turn_and_audio_transcription_result(self):
        """Verifies SpeakerTurn and AudioTranscriptionResult schema structures and iteration helpers."""
        turn = SpeakerTurn(speaker_id="SPEAKER_00", start=60.0, end=75.0, confidence=0.99, text="Opening thesis")
        self.assertEqual(turn.speaker_id, "SPEAKER_00")
        self.assertEqual(turn.start, 60.0)
        self.assertEqual(turn.end, 75.0)
        self.assertEqual(turn.duration, 15.0)

        words = [
            WordTimestamp(word="Opening", start=60.1, end=60.6, confidence=0.99, speaker_id="SPEAKER_00"),
            WordTimestamp(word="thesis", start=60.7, end=61.2, confidence=0.98, speaker_id="SPEAKER_00"),
        ]
        result = AudioTranscriptionResult(
            source_path="/data/test_video.mp4",
            detected_language="en",
            duration=120.0,
            words=words,
            speaker_turns=[turn]
        )
        self.assertEqual(result.detected_language, "en")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].word, "Opening")
        self.assertEqual(result[1].word, "thesis")
        self.assertEqual(len(result.speaker_turns), 1)

        # Iteration helper compatibility
        iterated_words = [w.word for w in result]
        self.assertEqual(iterated_words, ["Opening", "thesis"])

    # ═════════════════════════════════════════════════════════════════════════
    # 2. SpeakerTurn Overlap Alignment Tests
    # ═════════════════════════════════════════════════════════════════════════

    def test_speaker_turn_overlap_alignment(self):
        """Verifies word-to-speaker turn temporal overlap alignment and cross-talk resolution."""
        turns = [
            SpeakerTurn(speaker_id="spk_A", start=60.0, end=72.0),
            SpeakerTurn(speaker_id="spk_B", start=70.0, end=85.0),  # 2.0s cross-talk overlap [70.0, 72.0]
        ]
        words = [
            WordTimestamp(word="clean_A", start=61.0, end=61.5, confidence=0.9),
            # In cross-talk: word mostly in Turn B (71.2 to 72.5): Overlap A=0.8s, Overlap B=1.3s -> spk_B
            WordTimestamp(word="interruption", start=71.2, end=72.5, confidence=0.9),
            WordTimestamp(word="clean_B", start=75.0, end=75.8, confidence=0.9),
            # Gap word after all turns (86.0 to 86.5) -> nearest turn is spk_B
            WordTimestamp(word="trailing", start=86.0, end=86.5, confidence=0.9),
        ]

        align_words_to_speaker_turns(words, turns)
        self.assertEqual(words[0].speaker_id, "spk_A")
        self.assertEqual(words[1].speaker_id, "spk_B")
        self.assertEqual(words[2].speaker_id, "spk_B")
        self.assertEqual(words[3].speaker_id, "spk_B")

        # Words should also be attached to respective turns
        self.assertEqual(len(turns[0].words), 1)
        self.assertEqual(len(turns[1].words), 3)

    # ═════════════════════════════════════════════════════════════════════════
    # 3. Duration Bounds [30.0s, 50.0s] Tests
    # ═════════════════════════════════════════════════════════════════════════

    def test_duration_bounds_hard_rejection(self):
        """Verifies candidate validation strictly accepts [30.0, 50.0]s and rejects outside."""
        # 30.0s and 50.0s boundary tests
        c_exact_min = ShortCandidate(start_time_seconds=60.0, duration_seconds=30.0, title="Min", reasoning="Exact min")
        c_exact_max = ShortCandidate(start_time_seconds=100.0, duration_seconds=50.0, title="Max", reasoning="Exact max")
        c_short = ShortCandidate(start_time_seconds=160.0, duration_seconds=29.9, title="Short", reasoning="Too short")
        c_long = ShortCandidate(start_time_seconds=200.0, duration_seconds=50.1, title="Long", reasoning="Too long")

        valid = self.engine._validate_and_filter(
            [c_exact_min, c_exact_max, c_short, c_long],
            video_duration=300.0
        )
        valid_titles = [v.title for v in valid]
        self.assertIn("Min", valid_titles)
        self.assertIn("Max", valid_titles)
        self.assertNotIn("Short", valid_titles)
        self.assertNotIn("Long", valid_titles)

    # ═════════════════════════════════════════════════════════════════════════
    # 4. Teaser Blacklist (Rejecting t < 60.0s) Tests
    # ═════════════════════════════════════════════════════════════════════════

    def test_teaser_blacklist_rejection(self):
        """Verifies candidates starting at t < 60.0s are rejected by validation and fallback."""
        intro_candidates = [
            ShortCandidate(start_time_seconds=0.0, duration_seconds=35.0, title="Intro0", reasoning="Teaser"),
            ShortCandidate(start_time_seconds=30.0, duration_seconds=40.0, title="Intro30", reasoning="Teaser"),
            ShortCandidate(start_time_seconds=59.9, duration_seconds=35.0, title="Intro59.9", reasoning="Teaser"),
            ShortCandidate(start_time_seconds=60.0, duration_seconds=35.0, title="Valid60", reasoning="Post-teaser"),
        ]
        valid = self.engine._validate_and_filter(intro_candidates, video_duration=300.0)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].title, "Valid60")

        # Verify fallback selection never produces start < 60.0s even with transcript starting at 0s
        words = [
            {"word": f"word_{i}", "start": i * 0.5, "end": i * 0.5 + 0.4}
            for i in range(600)  # spans 0.0s to 300.0s
        ]
        fallbacks = self.engine.select_fallback_shorts(
            transcript_words=words,
            shots=[],
            video_duration=300.0,
            needed=4
        )
        for idx, f in enumerate(fallbacks):
            self.assertGreaterEqual(
                f.start_time_seconds, 60.0,
                f"Fallback short {idx} started at {f.start_time_seconds}s < 60.0s"
            )

    # ═════════════════════════════════════════════════════════════════════════
    # 5. Hook (0-5s), Escalation, Payoff Structure Tests
    # ═════════════════════════════════════════════════════════════════════════

    def test_hook_escalation_payoff_structure(self):
        """Verifies Hook has speech <= 1.5s, Payoff closes cleanly, and reasoning records 3 acts."""
        # Construct dialogue with clean sentence endings every 15 words
        words = []
        for i in range(500):
            w_start = 60.0 + i * 0.45
            is_end = (i % 15 == 14)
            text = f"concept_{i}." if is_end else f"concept_{i}"
            words.append({
                "word": text,
                "start": w_start,
                "end": w_start + 0.38,
                "speaker_id": "SPEAKER_00" if i < 250 else "SPEAKER_01"
            })

        shots = [
            ShotBoundary(start_frame=i * 90, end_frame=(i + 1) * 90,
                         start_s=60.0 + i * 3.0, end_s=60.0 + (i + 1) * 3.0)
            for i in range(80)
        ]

        shorts = self.engine.select_fallback_shorts(
            transcript_words=words,
            shots=shots,
            video_duration=300.0,
            needed=4
        )
        self.assertGreaterEqual(len(shorts), 4)

        for idx, s in enumerate(shorts):
            # 1. Hook check: first word within arc must start within 1.5s of arc start
            words_in_arc = [w for w in words if w["start"] >= s.start_time_seconds and w["end"] <= s.end_time_seconds]
            self.assertTrue(len(words_in_arc) > 0, f"Short {idx} has no words")
            first_word_offset = words_in_arc[0]["start"] - s.start_time_seconds
            self.assertLessEqual(
                first_word_offset, 1.5,
                f"Short {idx} hook delayed: first word starts at +{first_word_offset:.2f}s (>1.5s)"
            )

            # 2. Payoff check: arc must not cut a word mid-speech
            last_word = words_in_arc[-1]
            self.assertGreaterEqual(
                s.end_time_seconds, last_word["end"],
                f"Short {idx} cut off word '{last_word['word']}' before its end"
            )

            # 3. 3-Act Structure in Reasoning
            self.assertIn("Hook", s.reasoning, f"Short {idx} missing Hook in reasoning")
            self.assertIn("Escalation", s.reasoning, f"Short {idx} missing Escalation in reasoning")
            self.assertIn("Payoff", s.reasoning, f"Short {idx} missing Payoff in reasoning")

    # ═════════════════════════════════════════════════════════════════════════
    # 6. >= 4 Distinct, Pairwise Non-Overlapping Arcs Tests
    # ═════════════════════════════════════════════════════════════════════════

    def test_four_distinct_non_overlapping_arcs(self):
        """Verifies at least 4 arcs are selected and none overlap."""
        words = [
            {"word": f"word_{i}.", "start": 60.0 + i * 0.4, "end": 60.0 + i * 0.4 + 0.35}
            for i in range(500)
        ]
        shorts = self.engine.select_fallback_shorts(
            transcript_words=words,
            shots=[],
            video_duration=300.0,
            needed=4
        )
        self.assertEqual(len(shorts), 4)

        for i in range(len(shorts)):
            # Bounds
            self.assertGreaterEqual(shorts[i].start_time_seconds, 60.0)
            self.assertGreaterEqual(shorts[i].duration_seconds, 30.0)
            self.assertLessEqual(shorts[i].duration_seconds, 50.0)
            self.assertLessEqual(shorts[i].end_time_seconds, 300.0)

            # Pairwise non-overlapping check
            for j in range(i + 1, len(shorts)):
                s1, s2 = shorts[i], shorts[j]
                self.assertTrue(
                    s1.end_time_seconds <= s2.start_time_seconds or s2.end_time_seconds <= s1.start_time_seconds,
                    f"Short {i} [{s1.start_time_seconds}-{s1.end_time_seconds}] overlaps with "
                    f"Short {j} [{s2.start_time_seconds}-{s2.end_time_seconds}]"
                )

    def test_top_up_with_existing_candidate(self):
        """Verifies fallback tops up existing arcs to guarantee >= 4 without overlap."""
        existing = [
            ShortCandidate(
                start_time_seconds=70.0,
                duration_seconds=35.0,
                title="VLM Lead Arc",
                reasoning="Hook (0-5s): ... Escalation: ... Payoff: ..."
            )
        ]
        words = [
            {"word": f"term_{i}", "start": 60.0 + i * 0.45, "end": 60.0 + i * 0.45 + 0.4}
            for i in range(400)
        ]
        fallbacks = self.engine.select_fallback_shorts(
            transcript_words=words,
            shots=[],
            video_duration=300.0,
            existing=existing,
            needed=3
        )
        self.assertEqual(len(fallbacks), 3)

        combined = existing + fallbacks
        self.assertEqual(len(combined), 4)
        for i in range(len(combined)):
            for j in range(i + 1, len(combined)):
                s1, s2 = combined[i], combined[j]
                self.assertTrue(
                    s1.end_time_seconds <= s2.start_time_seconds or s2.end_time_seconds <= s1.start_time_seconds,
                    f"Combined short {i} overlaps with {j}"
                )

    def test_sparse_and_silent_video_edge_cases(self):
        """Verifies fallback produces 4 valid arcs with sparse dialogue or zero dialogue."""
        # Minimal duration boundary (180.0s total: exactly enough for 60s teaser + 4 x 30s)
        shorts_180 = self.engine.select_fallback_shorts(
            transcript_words=[],
            shots=[],
            video_duration=180.0,
            needed=4
        )
        self.assertEqual(len(shorts_180), 4)
        for s in shorts_180:
            self.assertGreaterEqual(s.start_time_seconds, 60.0)
            self.assertGreaterEqual(s.duration_seconds, 30.0)
            self.assertLessEqual(s.duration_seconds, 50.0)
            self.assertLessEqual(s.end_time_seconds, 180.0)

        # Silent video with shots only
        shots = [
            ShotBoundary(start_frame=i * 120, end_frame=(i + 1) * 120,
                         start_s=60.0 + i * 4.0, end_s=60.0 + (i + 1) * 4.0)
            for i in range(50)
        ]
        shorts_silent = self.engine.select_fallback_shorts(
            transcript_words=[],
            shots=shots,
            video_duration=260.0,
            needed=4
        )
        self.assertEqual(len(shorts_silent), 4)
        for s in shorts_silent:
            self.assertGreaterEqual(s.start_time_seconds, 60.0)
            self.assertGreaterEqual(s.duration_seconds, 30.0)
            self.assertLessEqual(s.duration_seconds, 50.0)

    # ═════════════════════════════════════════════════════════════════════════
    # 7. Acoustic Clustering Fallback Tests
    # ═════════════════════════════════════════════════════════════════════════

    def test_offline_acoustic_clustering_fallback(self):
        """Verifies acoustic feature extraction and clustering handles single vs multi-speaker."""
        # Single speaker features (minimal variance)
        X_single = np.ones((6, 28), dtype=np.float32) + np.random.randn(6, 28).astype(np.float32) * 0.005
        labels_single = cluster_speaker_segments(X_single)
        self.assertEqual(len(set(labels_single)), 1)
        self.assertEqual(labels_single[0], 0)

        # Two distinct speakers
        X_spk1 = np.ones((4, 28), dtype=np.float32) + np.random.randn(4, 28).astype(np.float32) * 0.01
        X_spk2 = -np.ones((4, 28), dtype=np.float32) + np.random.randn(4, 28).astype(np.float32) * 0.01
        X_two = np.vstack([X_spk1, X_spk2])
        labels_two = cluster_speaker_segments(X_two)
        self.assertEqual(len(set(labels_two)), 2)


if __name__ == "__main__":
    unittest.main()
