"""
tests/test_challenger_m1_2_empirical_stress.py - Milestone 1 Iteration 2 Empirical Stress Suite.

Adversarial stress-testing of:
1. Acoustic clustering: zero-norm, subnormal, NaN, Inf, extreme magnitudes, single-speaker, and multi-speaker feature matrices.
2. Narrative arc contracts: Hook (0-5s, speech entry <= 1.5s), Escalation, Payoff (no word clipping, punctuation closure), 
   teaser blacklist (t >= 60.0s), duration bounds (30.0-50.0s), and pairwise non-overlapping >= 4 quota guarantee.
"""

import time
import math
import asyncio
import numpy as np
import pytest

from shared.schemas import (
    WordTimestamp, SpeakerTurn, ShortCandidate as PydanticShortCandidate,
    ClipManifest, VideoSegment, CuttingManifest
)
from wsl2_gpu.transcription_spoke import (
    cluster_speaker_segments, align_words_to_speaker_turns
)
from wsl2_gpu.editorial_engine import (
    EditorialEngine, ShortCandidate, ShotBoundary,
    MIN_START_TIME, MIN_DURATION, MAX_DURATION, MAX_HOOK_DELAY
)
from wsl2_gpu.debate_graph import DebateGraph


class TestAcousticClusteringExtremeMatrices:
    """Stress tests for acoustic embedding clustering under extreme numeric matrices."""

    def test_zero_norm_and_empty_matrices(self):
        """Verify handling of empty, single, and all-zero feature matrices."""
        # Empty input
        res_empty = cluster_speaker_segments(np.zeros((0, 28), dtype=np.float32))
        assert len(res_empty) == 0

        # Single zero sample
        res_single_zero = cluster_speaker_segments(np.zeros((1, 28), dtype=np.float32))
        assert np.array_equal(res_single_zero, np.zeros(1, dtype=int))

        # Single non-zero sample
        res_single_val = cluster_speaker_segments(np.full((1, 28), 5.0, dtype=np.float32))
        assert np.array_equal(res_single_val, np.zeros(1, dtype=int))

        # Multiple zero samples (4, 10, 100)
        for n in [4, 10, 100]:
            res = cluster_speaker_segments(np.zeros((n, 28), dtype=np.float32))
            assert len(res) == n
            assert np.all(res == 0)

    def test_subnormal_and_near_zero_vectors(self):
        """Verify handling of subnormal floats and vectors with norms below 1e-4."""
        # Extremely small values below numerical resolution
        X_subnormal = np.full((8, 28), 1e-15, dtype=np.float32)
        res = cluster_speaker_segments(X_subnormal)
        assert len(res) == 8
        assert np.all(res == 0)

        # Norms just below 1e-4 threshold
        X_under_thresh = np.full((6, 28), 1e-5, dtype=np.float32)  # norm = sqrt(28)*1e-5 ~ 5.29e-5 < 1e-4
        res_under = cluster_speaker_segments(X_under_thresh)
        assert len(res_under) == 6
        assert np.all(res_under == 0)

    def test_non_finite_nan_and_inf_matrices(self):
        """Verify complete resilience against NaN, +Inf, -Inf without raising exceptions."""
        # All NaN
        X_nan = np.full((6, 28), np.nan, dtype=np.float32)
        res_nan = cluster_speaker_segments(X_nan)
        assert len(res_nan) == 6
        assert np.all(res_nan == 0)

        # All +Inf and -Inf
        X_inf = np.full((6, 28), np.inf, dtype=np.float32)
        res_inf = cluster_speaker_segments(X_inf)
        assert len(res_inf) == 6
        assert np.all(res_inf == 0)

        X_neginf = np.full((6, 28), -np.inf, dtype=np.float32)
        res_neginf = cluster_speaker_segments(X_neginf)
        assert len(res_neginf) == 6
        assert np.all(res_neginf == 0)

        # Mixed NaN, Inf, and zero vectors
        X_mixed_nonfinite = np.zeros((6, 28), dtype=np.float32)
        X_mixed_nonfinite[0, :] = np.nan
        X_mixed_nonfinite[1, :] = np.inf
        X_mixed_nonfinite[2, :] = -np.inf
        res_mixed_nf = cluster_speaker_segments(X_mixed_nonfinite)
        assert len(res_mixed_nf) == 6
        assert np.all(res_mixed_nf == 0)

    def test_mixed_non_finite_and_valid_multi_speaker(self):
        """Verify that valid speech segments cluster properly when interleaved with NaN/Inf silent chunks."""
        # Rows 0, 1: NaN / Inf (silent or corrupted chunks)
        # Rows 2, 3: Speaker A (dim 0 high)
        # Rows 4, 5: Speaker B (dim 1 high)
        X = np.zeros((6, 28), dtype=np.float32)
        X[0, :] = np.nan
        X[1, :] = np.inf
        X[2:4, 0] = 8.0
        X[4:6, 1] = 8.0

        res = cluster_speaker_segments(X)
        assert len(res) == 6
        # Corrupted/silent rows 0 and 1 must default to label 0
        assert res[0] == 0
        assert res[1] == 0
        # Speaker A segments must share a cluster
        assert res[2] == res[3]
        # Speaker B segments must share a cluster
        assert res[4] == res[5]
        # Speaker A and Speaker B must be distinct
        assert res[2] != res[4]

    def test_single_speaker_variations(self):
        """Verify that single speaker audio with small acoustic perturbations maps to 0."""
        # Identical non-zero vectors
        X_identical = np.zeros((5, 28), dtype=np.float32)
        X_identical[:, 0] = 3.0
        X_identical[:, 5] = 2.0
        res_id = cluster_speaker_segments(X_identical)
        assert np.all(res_id == 0)

        # Small perturbations within single speaker threshold (< 0.25 cosine distance)
        rng = np.random.RandomState(42)
        base = np.zeros(28, dtype=np.float32)
        base[0:10] = 2.0
        X_perturbed = np.array([base + rng.normal(0, 0.05, size=28) for _ in range(8)], dtype=np.float32)
        res_pert = cluster_speaker_segments(X_perturbed)
        assert np.all(res_pert == 0)

    def test_multi_speaker_orthogonal_profiles(self):
        """Verify accurate separation of 2, 3, and 4 distinct acoustic speaker profiles."""
        # 2 Speakers
        X_2 = np.zeros((6, 28), dtype=np.float32)
        X_2[0:3, 0] = 5.0
        X_2[3:6, 1] = 5.0
        res_2 = cluster_speaker_segments(X_2)
        assert len(set(res_2)) == 2
        assert res_2[0] == res_2[1] == res_2[2]
        assert res_2[3] == res_2[4] == res_2[5]
        assert res_2[0] != res_2[3]

        # 3 Speakers
        X_3 = np.zeros((9, 28), dtype=np.float32)
        X_3[0:3, 0] = 5.0
        X_3[3:6, 1] = 5.0
        X_3[6:9, 2] = 5.0
        res_3 = cluster_speaker_segments(X_3)
        assert len(set(res_3)) == 3
        assert res_3[0] == res_3[1] == res_3[2]
        assert res_3[3] == res_3[4] == res_3[5]
        assert res_3[6] == res_3[7] == res_3[8]

        # 4 Speakers
        X_4 = np.zeros((12, 28), dtype=np.float32)
        X_4[0:3, 0] = 5.0
        X_4[3:6, 1] = 5.0
        X_4[6:9, 2] = 5.0
        X_4[9:12, 3] = 5.0
        res_4 = cluster_speaker_segments(X_4, max_speakers=4)
        assert len(set(res_4)) == 4

    def test_large_numeric_magnitudes(self):
        """Verify numerical stability under huge feature magnitudes."""
        # 1. Magnitudes within float32 square range (1e15)
        X_large = np.zeros((4, 28), dtype=np.float32)
        X_large[0:2, 0] = 1e15
        X_large[2:4, 1] = 1e15
        res = cluster_speaker_segments(X_large)
        assert len(res) == 4
        assert len(set(res)) == 2

        # 2. Magnitudes overflowing float32 (1e30) must not crash and default safely
        X_overflow = np.zeros((4, 28), dtype=np.float32)
        X_overflow[0:2, 0] = 1e30
        X_overflow[2:4, 1] = 1e30
        res_overflow = cluster_speaker_segments(X_overflow)
        assert len(res_overflow) == 4
        assert np.all(res_overflow == 0)

    def test_high_volume_speaker_segments(self):
        """Stress-test performance and memory with 600 segments across 3 speakers."""
        X_vol = np.zeros((600, 28), dtype=np.float32)
        for i in range(600):
            spk = i % 3
            X_vol[i, spk] = 4.0
        t0 = time.perf_counter()
        res = cluster_speaker_segments(X_vol, max_speakers=4)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0  # Must finish within 5 seconds
        assert len(res) == 600
        assert len(set(res)) == 3


class TestWordToSpeakerTurnAdversarialAlignment:
    """Stress tests for word timestamp to speaker turn alignment."""

    def test_alignment_with_crosstalk_and_silence_gaps(self):
        """Verify alignment across overlapping turns and long silence intervals."""
        turns = [
            SpeakerTurn(speaker_id="SPEAKER_00", start=60.0, end=70.0, confidence=0.9),
            SpeakerTurn(speaker_id="SPEAKER_01", start=68.0, end=80.0, confidence=0.9),  # 2s overlap
            SpeakerTurn(speaker_id="SPEAKER_02", start=100.0, end=110.0, confidence=0.9), # 20s gap
        ]
        words = [
            # In turn 0 non-overlapping
            WordTimestamp(word="hello", start=61.0, end=61.5, confidence=0.95),
            # In overlap window [68.0, 70.0], closer to turn 0 (68.0 - 69.5 -> overlap with turn 0 is 1.5s, turn 1 is 1.5s tie)
            WordTimestamp(word="overlap_turn0", start=68.1, end=69.9, confidence=0.95),
            # In overlap window [68.0, 70.0], purely inside turn 1 after 70.0s
            WordTimestamp(word="turn1_pure", start=71.0, end=72.0, confidence=0.95),
            # In the 20s silence gap at 90.0s (midpoint 90.25 -> dist to turn 1 end is 10.25s, dist to turn 2 start is 9.75s)
            WordTimestamp(word="silence_word", start=90.0, end=90.5, confidence=0.95),
        ]

        align_words_to_speaker_turns(words, turns)
        assert words[0].speaker_id == "SPEAKER_00"
        assert words[2].speaker_id == "SPEAKER_01"
        # Silence word is closer to Turn 2 (start 100.0) than Turn 1 (end 80.0)
        assert words[3].speaker_id == "SPEAKER_02"


class TestNarrativeArcContractsAndEditorialFallback:
    """Stress tests for Hook, Escalation, Payoff, and Quota contracts."""

    def setup_method(self):
        self.engine = EditorialEngine()

    def test_hook_speech_delay_hard_constraint(self):
        """Verify that every generated arc enforces Hook speech delay <= 1.5s."""
        words = []
        for i in range(100):
            t = 70.0 + i * 0.8
            words.append({"word": f"speech_{i}.", "start": t, "end": t + 0.4})

        shorts = self.engine.select_fallback_shorts(
            transcript_words=words, shots=[], video_duration=300.0, needed=4
        )
        assert len(shorts) >= 4
        for idx, s in enumerate(shorts):
            words_in_s = [w for w in words if s.start_time_seconds <= w["start"] <= s.end_time_seconds]
            if words_in_s:
                first_w = words_in_s[0]
                delay = first_w["start"] - s.start_time_seconds
                assert delay <= MAX_HOOK_DELAY, (
                    f"Arc {idx} failed hook contract: first word at {first_w['start']}s, "
                    f"arc start {s.start_time_seconds}s (delay {delay:.2f}s > {MAX_HOOK_DELAY}s)"
                )

    def test_payoff_clean_closure_and_no_word_truncation(self):
        """Verify Payoff contract: arc end time does not truncate any spoken word mid-speech."""
        words = []
        for i in range(200):
            t = 60.0 + i * 0.5
            is_period = (i % 10 == 9)
            text = f"word_{i}." if is_period else f"word_{i}"
            words.append({"word": text, "start": t, "end": t + 0.4})

        shorts = self.engine.select_fallback_shorts(
            transcript_words=words, shots=[], video_duration=300.0, needed=4
        )
        assert len(shorts) >= 4
        for idx, s in enumerate(shorts):
            # A word is truncated if speech starts before arc ends, but finishes after arc ends
            truncated = [w for w in words if w["start"] < s.end_time_seconds and w["end"] > s.end_time_seconds]
            assert len(truncated) == 0, (
                f"Arc {idx} truncated {len(truncated)} word(s) mid-speech: {truncated} (arc end {s.end_time_seconds}s)"
            )

    def test_three_act_structure_in_reasoning(self):
        """Verify that all generated arcs explicitly document Hook (0-5s), Escalation, and Payoff."""
        words = [{"word": f"word_{i}.", "start": 60.0 + i * 0.5, "end": 60.0 + i * 0.5 + 0.4} for i in range(120)]
        shorts = self.engine.select_fallback_shorts(
            transcript_words=words, shots=[], video_duration=240.0, needed=4
        )
        assert len(shorts) == 4
        for idx, s in enumerate(shorts):
            assert "Hook" in s.reasoning, f"Arc {idx} missing Hook in reasoning"
            assert "Escalation" in s.reasoning, f"Arc {idx} missing Escalation in reasoning"
            assert "Payoff" in s.reasoning, f"Arc {idx} missing Payoff in reasoning"

    def test_strict_teaser_blacklist_and_duration_bounds(self):
        """Verify start_time >= 60.0s and 30.0s <= duration <= 50.0s across all generated arcs."""
        test_durations = [180.0, 240.0, 360.0, 1800.0]
        for dur in test_durations:
            words = [{"word": f"w_{i}.", "start": 60.0 + i * 0.6, "end": 60.0 + i * 0.6 + 0.4} for i in range(int(dur))]
            shorts = self.engine.select_fallback_shorts(
                transcript_words=words, shots=[], video_duration=dur, needed=4
            )
            assert len(shorts) == 4, f"Failed to produce 4 arcs on duration {dur}s"
            for idx, s in enumerate(shorts):
                assert s.start_time_seconds >= 60.0, f"Arc {idx} on dur {dur} started at {s.start_time_seconds} < 60.0s"
                assert 30.0 <= s.duration_seconds <= 50.0, f"Arc {idx} on dur {dur} duration {s.duration_seconds} out of [30, 50]"
                assert s.end_time_seconds <= dur, f"Arc {idx} on dur {dur} exceeded video duration"

    def test_pairwise_non_overlapping_arcs(self):
        """Verify that no two arcs overlap on the timeline."""
        for dur in [180.0, 181.0, 240.0, 600.0]:
            words = [{"word": f"w_{i}", "start": 60.0 + i * 0.4, "end": 60.0 + i * 0.4 + 0.3} for i in range(300)]
            shorts = self.engine.select_fallback_shorts(
                transcript_words=words, shots=[], video_duration=dur, needed=4
            )
            assert len(shorts) == 4
            sorted_arcs = sorted(shorts, key=lambda s: s.start_time_seconds)
            for i in range(len(sorted_arcs) - 1):
                cur_end = sorted_arcs[i].end_time_seconds
                nxt_start = sorted_arcs[i + 1].start_time_seconds
                assert cur_end <= nxt_start + 1e-4, (
                    f"Overlap detected on dur {dur}s: Arc {i} ends at {cur_end:.2f}s, "
                    f"Arc {i+1} starts at {nxt_start:.2f}s"
                )

    def test_extreme_silence_video_four_arcs_guarantee(self):
        """Verify that a 100% silent video (0 words, 0 shots) on 180.0s produces exactly 4 valid non-overlapping arcs."""
        shorts = self.engine.select_fallback_shorts(
            transcript_words=[], shots=[], video_duration=180.0, needed=4
        )
        assert len(shorts) == 4, f"Expected 4 arcs on 100% silent 180s video, got {len(shorts)}"
        for idx, s in enumerate(shorts):
            assert s.start_time_seconds >= 60.0
            assert 30.0 <= s.duration_seconds <= 50.0
            assert s.end_time_seconds <= 180.0
            assert "Hook" in s.reasoning
            assert "Escalation" in s.reasoning
            assert "Payoff" in s.reasoning

    def test_debate_graph_director_unconditional_teaser_rejection(self):
        """Verify that DebateGraph director unconditionally rejects clips starting before 60.0s."""
        clips = [
            ClipManifest(segments=[VideoSegment(segment_start=59.5, segment_end=95.0)], hook_score=0.9, retention_score=0.9, reasoning="Teaser clip 1"),
            ClipManifest(segments=[VideoSegment(segment_start=100.0, segment_end=135.0)], hook_score=0.9, retention_score=0.9, reasoning="Clip 2"),
            ClipManifest(segments=[VideoSegment(segment_start=140.0, segment_end=175.0)], hook_score=0.9, retention_score=0.9, reasoning="Clip 3"),
            ClipManifest(segments=[VideoSegment(segment_start=180.0, segment_end=215.0)], hook_score=0.9, retention_score=0.9, reasoning="Clip 4"),
        ]
        manifest = CuttingManifest(clips=clips)
        state = {
            "cutting_manifest": manifest,
            "fused_timeline": None,  # Crucial test: fused_timeline is None!
            "critique_history": []
        }
        graph = DebateGraph()
        res_state = asyncio.run(graph.director_node(state))
        critique = res_state["cutting_manifest"].clips[0].director_critique
        assert critique is not None
        assert "REJECTED" in critique
        assert "intro teaser zone" in critique
        assert not res_state["cutting_manifest"].consensus_reached

    def test_tight_packing_top_up_with_existing_candidate(self):
        """Verify dynamic gap allocation tops up exactly into available tight 180s slots."""
        # Existing candidate at [90.0, 120.0] on a 180.0s video.
        # Free gaps: [60.0, 90.0] (30s -> 1 slot) and [120.0, 180.0] (60s -> 2 slots). Total capacity = 3.
        existing = [
            ShortCandidate(start_time_seconds=90.0, duration_seconds=30.0, title="Pre-existing 1", reasoning="Act 1")
        ]
        shorts = self.engine.select_fallback_shorts(
            transcript_words=[], shots=[], video_duration=180.0, needed=3, existing=existing
        )
        assert len(shorts) == 3
        all_arcs = sorted(existing + shorts, key=lambda s: s.start_time_seconds)
        assert len(all_arcs) == 4
        # Verify no overlap across all 4 arcs
        for i in range(len(all_arcs) - 1):
            assert all_arcs[i].end_time_seconds <= all_arcs[i + 1].start_time_seconds + 1e-4

    def test_sparse_long_video_600s_five_words_stress(self):
        """Verify fallback on 600s video with only 5 words across 10 minutes produces >= 4 arcs."""
        sparse_words = [
            {"word": "anomaly.", "start": 70.0, "end": 70.5},
            {"word": "discovery.", "start": 160.0, "end": 160.5},
            {"word": "escalation.", "start": 290.0, "end": 290.5},
            {"word": "resolution.", "start": 440.0, "end": 440.5},
            {"word": "conclusion.", "start": 570.0, "end": 570.5},
        ]
        shorts = self.engine.select_fallback_shorts(
            transcript_words=sparse_words, shots=[], video_duration=600.0, needed=4
        )
        assert len(shorts) == 4
        for idx, s in enumerate(shorts):
            assert s.start_time_seconds >= 60.0
            assert 30.0 <= s.duration_seconds <= 50.0
            assert s.end_time_seconds <= 600.0
            # Hook contract: first word in arc starts <= 1.5s or visual entry recorded
            words_in_s = [w for w in sparse_words if s.start_time_seconds <= w["start"] <= s.end_time_seconds]
            if words_in_s:
                assert words_in_s[0]["start"] - s.start_time_seconds <= MAX_HOOK_DELAY

    def test_interleaved_silence_and_speaker_turns_clustering(self):
        """Verify alternating conversation with silent chunks accurately clusters speakers."""
        # 6 segments: Spk A (0), Silent (1), Spk B (2), Silent (3), Spk A (4), Spk B (5)
        X = np.zeros((6, 28), dtype=np.float32)
        X[0, 0:5] = 4.0  # Spk A
        # X[1] is silent
        X[2, 10:15] = 4.0 # Spk B
        # X[3] is silent
        X[4, 0:5] = 4.2  # Spk A
        X[5, 10:15] = 3.9 # Spk B

        res = cluster_speaker_segments(X, max_speakers=4)
        assert len(res) == 6
        # Silent segments default to 0
        assert res[1] == 0
        assert res[3] == 0
        # Spk A segments share cluster
        assert res[0] == res[4]
        # Spk B segments share cluster
        assert res[2] == res[5]
        # Spk A and Spk B are distinct
        assert res[0] != res[2]
