"""
Adversarial Stress Test Suite for Milestone 1 (Audio Diarization & Arc Partitioning)
Executed by: teamwork_preview_challenger_m1_2 (Milestone 1 Challenger 2)

Covers:
1. Speaker diarization alignment:
   - Overlapping turns (unequal overlap, exact ties)
   - Edge-boundary words (start exactly at turn end, end exactly at turn start)
   - Zero-overlap words (words in silence gaps, before first turn, after last turn)
   - Single-speaker audio vs multi-speaker audio in acoustic clustering
   - Zero-norm acoustic feature vector stability (silence edge cases)
2. Fallback arc selection:
   - Sparse transcripts across diverse video durations
   - Short videos (< 120s)
   - Long videos (> 1800s)
   - Pairwise non-overlapping guarantee
   - Quota assertion: >= 4 distinct arcs when duration >= 180.0s
   - 3-Act narrative structure & Pydantic schema validation
"""

import os
import sys
import unittest
import time
import numpy as np

# Ensure project root is on sys.path
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
    extract_segment_features,
    TranscriptionSpoke
)


class TestMilestone1AdversarialChallenger(unittest.TestCase):

    def setUp(self):
        self.engine = EditorialEngine()

    # ═════════════════════════════════════════════════════════════════════════
    # 1. SPEAKER DIARIZATION ALIGNMENT CHALLENGES
    # ═════════════════════════════════════════════════════════════════════════

    def test_overlapping_turns_and_crosstalk_assignment(self):
        """Adversarial Challenge: Overlapping speaker turns with varied word alignments."""
        t1 = SpeakerTurn(speaker_id="SPEAKER_A", start=10.0, end=20.0)
        t2 = SpeakerTurn(speaker_id="SPEAKER_B", start=16.0, end=26.0)

        words = [
            # Word mostly in T1 (15.0 to 18.0: overlap T1=3.0, T2=2.0) -> SPEAKER_A
            WordTimestamp(word="predominant_A", start=15.0, end=18.0, confidence=0.9),
            # Word mostly in T2 (18.0 to 22.0: overlap T1=2.0, T2=4.0) -> SPEAKER_B
            WordTimestamp(word="predominant_B", start=18.0, end=22.0, confidence=0.9),
            # Exact 50-50 tie in overlap (17.0 to 19.0: overlap T1=2.0, T2=2.0)
            WordTimestamp(word="exact_tie", start=17.0, end=19.0, confidence=0.9),
        ]

        align_words_to_speaker_turns(words, [t1, t2])
        self.assertEqual(words[0].speaker_id, "SPEAKER_A")
        self.assertEqual(words[1].speaker_id, "SPEAKER_B")
        self.assertIn(words[2].speaker_id, ["SPEAKER_A", "SPEAKER_B"])
        # Both turns should contain their respective assigned words
        self.assertTrue(len(t1.words) >= 1)
        self.assertTrue(len(t2.words) >= 1)

    def test_edge_boundary_words_alignment(self):
        """Adversarial Challenge: Words touching turn boundaries with zero mathematical overlap."""
        t1 = SpeakerTurn(speaker_id="SPEAKER_00", start=10.0, end=20.0)
        t2 = SpeakerTurn(speaker_id="SPEAKER_01", start=30.0, end=40.0)

        words = [
            # Word starts exactly when T1 ends: [20.0, 20.4] -> nearest is T1 (dist=0.20)
            WordTimestamp(word="at_end_boundary", start=20.0, end=20.4, confidence=0.9),
            # Word ends exactly when T1 starts: [9.5, 10.0] -> nearest is T1 (dist=0.25)
            WordTimestamp(word="at_start_boundary", start=9.5, end=10.0, confidence=0.9),
            # Word ends exactly when T2 starts: [29.6, 30.0] -> nearest is T2 (dist=0.20)
            WordTimestamp(word="at_t2_start", start=29.6, end=30.0, confidence=0.9),
        ]

        align_words_to_speaker_turns(words, [t1, t2])
        self.assertEqual(words[0].speaker_id, "SPEAKER_00")
        self.assertEqual(words[1].speaker_id, "SPEAKER_00")
        self.assertEqual(words[2].speaker_id, "SPEAKER_01")

    def test_zero_overlap_words_in_silence_gaps(self):
        """Adversarial Challenge: Words falling completely into silence gaps between distant turns."""
        t1 = SpeakerTurn(speaker_id="SPEAKER_00", start=10.0, end=20.0)
        t2 = SpeakerTurn(speaker_id="SPEAKER_01", start=50.0, end=60.0)

        words = [
            # Word before all turns: [2.0, 2.5] (mid 2.25) -> nearest is T1
            WordTimestamp(word="pre_turns", start=2.0, end=2.5, confidence=0.85),
            # Word in gap closer to T1: [23.0, 23.6] (mid 23.3) -> dist T1=3.3, dist T2=26.7 -> SPEAKER_00
            WordTimestamp(word="gap_near_t1", start=23.0, end=23.6, confidence=0.85),
            # Word in gap closer to T2: [44.0, 44.6] (mid 44.3) -> dist T1=24.3, dist T2=5.7 -> SPEAKER_01
            WordTimestamp(word="gap_near_t2", start=44.0, end=44.6, confidence=0.85),
            # Word after all turns: [70.0, 70.5] (mid 70.25) -> nearest is T2
            WordTimestamp(word="post_turns", start=70.0, end=70.5, confidence=0.85),
        ]

        align_words_to_speaker_turns(words, [t1, t2])
        self.assertEqual(words[0].speaker_id, "SPEAKER_00")
        self.assertEqual(words[1].speaker_id, "SPEAKER_00")
        self.assertEqual(words[2].speaker_id, "SPEAKER_01")
        self.assertEqual(words[3].speaker_id, "SPEAKER_01")

    def test_empty_inputs_speaker_alignment(self):
        """Adversarial Challenge: Empty words list and empty speaker turns list."""
        # 1. Empty words
        t1 = SpeakerTurn(speaker_id="SPEAKER_00", start=10.0, end=20.0)
        empty_words = []
        align_words_to_speaker_turns(empty_words, [t1])
        self.assertEqual(len(empty_words), 0)

        # 2. Empty turns -> defaults words to SPEAKER_00
        words = [
            WordTimestamp(word="orphan1", start=10.0, end=10.5, confidence=0.9),
            WordTimestamp(word="orphan2", start=12.0, end=12.5, confidence=0.9, speaker_id="EXISTING_SPK"),
        ]
        align_words_to_speaker_turns(words, [])
        self.assertEqual(words[0].speaker_id, "SPEAKER_00")
        self.assertEqual(words[1].speaker_id, "EXISTING_SPK")

    def test_single_vs_multi_speaker_acoustic_clustering(self):
        """Adversarial Challenge: Multi-speaker separation vs single-speaker clustering."""
        # Single speaker (intra-speaker variance < 0.25)
        np.random.seed(42)
        X_single = np.ones((8, 28), dtype=np.float32) + np.random.randn(8, 28).astype(np.float32) * 0.005
        labels_single = cluster_speaker_segments(X_single)
        self.assertEqual(len(set(labels_single)), 1)
        self.assertEqual(labels_single[0], 0)

        # 2 distinct speakers (each with 3 samples)
        X_spk1 = np.ones((3, 28), dtype=np.float32) + np.random.randn(3, 28).astype(np.float32) * 0.01
        X_spk2 = -np.ones((3, 28), dtype=np.float32) + np.random.randn(3, 28).astype(np.float32) * 0.01
        labels_two = cluster_speaker_segments(np.vstack([X_spk1, X_spk2]))
        self.assertEqual(len(set(labels_two)), 2)

        # 3 distinct orthogonal speakers (each with 3 samples)
        e1 = np.zeros((3, 28), dtype=np.float32); e1[:, 0] = 5.0
        e2 = np.zeros((3, 28), dtype=np.float32); e2[:, 1] = 5.0
        e3 = np.zeros((3, 28), dtype=np.float32); e3[:, 2] = 5.0
        labels_three = cluster_speaker_segments(np.vstack([e1, e2, e3]))
        self.assertEqual(len(set(labels_three)), 3)

    def test_zero_feature_vector_acoustic_clustering_stability(self):
        """Adversarial Challenge: Zero-energy feature vectors from silent/short segments must not raise ValueError."""
        # When audio segments are silent or < 512 samples, feature extraction yields all zeros.
        # This must return single-speaker zeros rather than crashing with scipy ValueError.
        X_silent = np.zeros((4, 28), dtype=np.float32)
        try:
            labels_silent = cluster_speaker_segments(X_silent)
            self.assertEqual(len(labels_silent), 4)
            self.assertEqual(len(set(labels_silent)), 1)
        except ValueError as e:
            self.fail(f"cluster_speaker_segments crashed on silent zero vectors with: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # 2. FALLBACK ARC SELECTION BOUNDARIES & DURATION CHALLENGES
    # ═════════════════════════════════════════════════════════════════════════

    def test_short_videos_under_120s_graceful_bounds(self):
        """Adversarial Challenge: Videos < 120s must safely produce up to mathematically possible arcs without crashing."""
        # 1. Video duration < 60s (intro teaser zone) -> 0 arcs
        shorts_45 = self.engine.select_fallback_shorts([], [], video_duration=45.0, needed=4)
        self.assertEqual(len(shorts_45), 0)

        # 2. Video duration 60.0s to 89.9s -> 0 arcs (since min clip is 30.0s, need >= 90.0s)
        shorts_80 = self.engine.select_fallback_shorts([], [], video_duration=80.0, needed=4)
        self.assertEqual(len(shorts_80), 0)

        # 3. Video duration 100.0s (< 120s) -> exactly 1 arc possible (60s to 100s)
        shorts_100 = self.engine.select_fallback_shorts([], [], video_duration=100.0, needed=4)
        self.assertEqual(len(shorts_100), 1)
        self.assertGreaterEqual(shorts_100[0].start_time_seconds, 60.0)
        self.assertTrue(30.0 <= shorts_100[0].duration_seconds <= 50.0)
        self.assertLessEqual(shorts_100[0].end_time_seconds, 100.0)

        # 4. Video duration 119.9s (< 120s) -> 1 arc possible
        shorts_119 = self.engine.select_fallback_shorts([], [], video_duration=119.9, needed=4)
        self.assertEqual(len(shorts_119), 1)
        self.assertGreaterEqual(shorts_119[0].start_time_seconds, 60.0)
        self.assertTrue(30.0 <= shorts_119[0].duration_seconds <= 50.0)

    def test_long_videos_over_1800s_performance_and_validity(self):
        """Adversarial Challenge: Long videos (> 1800s / 30m up to 1h) must process rapidly and produce >= 4 valid arcs."""
        for dur in [1800.0, 3600.0]:
            t0 = time.perf_counter()
            # Dense transcript up to 500 words
            words = [
                {"word": f"word_{i}.", "start": 60.0 + i * 0.8, "end": 60.0 + i * 0.8 + 0.5}
                for i in range(500)
            ]
            shorts = self.engine.select_fallback_shorts(
                transcript_words=words,
                shots=[],
                video_duration=dur,
                needed=4
            )
            elapsed = time.perf_counter() - t0

            # Performance requirement: fallback arc selection must complete in under 1.0s
            self.assertLess(elapsed, 1.0, f"select_fallback_shorts took {elapsed:.2f}s on {dur}s video (>1.0s limit)")
            self.assertEqual(len(shorts), 4)

            # Validate each short
            for idx, s in enumerate(shorts):
                self.assertGreaterEqual(s.start_time_seconds, 60.0)
                self.assertTrue(30.0 <= s.duration_seconds <= 50.0)
                self.assertLessEqual(s.end_time_seconds, dur)

    def test_pairwise_non_overlapping_arcs_strict_guarantee(self):
        """Adversarial Challenge: Arcs must NEVER have temporal overlap across any test conditions."""
        test_durations = [180.0, 240.0, 300.0, 600.0, 1800.0]
        for dur in test_durations:
            words = [
                {"word": f"w_{i}.", "start": 60.0 + i * 0.5, "end": 60.0 + i * 0.5 + 0.4}
                for i in range(int((dur - 60) * 1.5))
            ]
            shorts = self.engine.select_fallback_shorts(
                transcript_words=words,
                shots=[],
                video_duration=dur,
                needed=4
            )
            for i in range(len(shorts)):
                for j in range(i + 1, len(shorts)):
                    s1 = shorts[i]
                    s2 = shorts[j]
                    self.assertTrue(
                        s1.end_time_seconds <= s2.start_time_seconds or s2.end_time_seconds <= s1.start_time_seconds,
                        f"Overlap detected on dur={dur}s: Arc {i} [{s1.start_time_seconds}, {s1.end_time_seconds}] "
                        f"overlaps with Arc {j} [{s2.start_time_seconds}, {s2.end_time_seconds}]"
                    )

    def test_guarantee_four_arcs_on_dense_and_silent_videos(self):
        """Adversarial Challenge: Video with duration >= 180s must produce >= 4 distinct arcs under dense speech or silence."""
        # 1. Pure silence at boundary 180.0s (60s teaser + 4 x 30s)
        shorts_silent_180 = self.engine.select_fallback_shorts([], [], video_duration=180.0, needed=4)
        self.assertEqual(len(shorts_silent_180), 4, f"Expected 4 arcs on silent 180s video, got {len(shorts_silent_180)}")

        # 2. Dense transcript at 180.0s
        words_dense = [
            {"word": f"token_{i}.", "start": 60.0 + i * 0.4, "end": 60.0 + i * 0.4 + 0.35}
            for i in range(300)
        ]
        shorts_dense_180 = self.engine.select_fallback_shorts(words_dense, [], video_duration=180.0, needed=4)
        self.assertEqual(len(shorts_dense_180), 4, f"Expected 4 arcs on dense 180s video, got {len(shorts_dense_180)}")

    def test_guarantee_four_arcs_on_sparse_transcript_boundary(self):
        """Adversarial Challenge: Video with duration >= 180s MUST produce >= 4 arcs even with sparse transcripts.

        Hypothesis: When transcript words are sparse (e.g. 4 isolated words across 180s or 1 word every 100s across 1800s),
        word-boundary snapping in Strategy C causes static slots to collide or fall below MIN_DURATION,
        resulting in quota starvation (< 4 arcs).
        """
        # Test Case A: 180s video with 4 sparse words
        words_sparse_180 = [
            {"word": "intro_point", "start": 60.5, "end": 60.9},
            {"word": "mid_point", "start": 105.0, "end": 105.4},
            {"word": "pivot_point", "start": 130.0, "end": 130.4},
            {"word": "climax_point", "start": 160.0, "end": 160.4},
        ]
        shorts_sparse_180 = self.engine.select_fallback_shorts(
            transcript_words=words_sparse_180,
            shots=[],
            video_duration=180.0,
            needed=4
        )
        self.assertEqual(
            len(shorts_sparse_180), 4,
            f"CRITICAL DEFECT: Fallback arc selector produced only {len(shorts_sparse_180)}/4 arcs "
            f"on 180.0s video with sparse transcript!"
        )

    def test_guarantee_four_arcs_on_sparse_long_video(self):
        """Adversarial Challenge: 1800s video with sparse words (1 word every 100s) must produce >= 4 arcs."""
        words_sparse_1800 = [
            {"word": f"speech_{i}", "start": 65.0 + i * 100.0, "end": 65.0 + i * 100.0 + 0.5}
            for i in range(17)
        ]
        shorts_sparse_1800 = self.engine.select_fallback_shorts(
            transcript_words=words_sparse_1800,
            shots=[],
            video_duration=1800.0,
            needed=4
        )
        self.assertEqual(
            len(shorts_sparse_1800), 4,
            f"CRITICAL DEFECT: Fallback arc selector produced only {len(shorts_sparse_1800)}/4 arcs "
            f"on 1800.0s video with sparse transcript!"
        )

    def test_pydantic_schema_strict_compliance(self):
        """Adversarial Challenge: All fallback arcs must conform to Pydantic ShortCandidate contract."""
        shorts = self.engine.select_fallback_shorts([], [], video_duration=300.0, needed=4)
        for s in shorts:
            # Must validate cleanly against Pydantic schema
            pydantic_cand = PydanticShortCandidate(
                start_time_seconds=s.start_time_seconds,
                duration_seconds=s.duration_seconds,
                title=s.title,
                reasoning=s.reasoning,
                framing_mode=s.framing_mode
            )
            self.assertGreaterEqual(pydantic_cand.start_time_seconds, 60.0)
            self.assertTrue(30.0 <= pydantic_cand.duration_seconds <= 50.0)


if __name__ == "__main__":
    unittest.main()
