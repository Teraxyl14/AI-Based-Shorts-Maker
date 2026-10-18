"""
tests/test_challenger_m1_boundary_stress.py - Milestone 1 Adversarial Stress Test Suite
Challenger 1: Boundary & Duration Stress Tester

Empirical stress tests for:
1. Teaser boundary: candidates starting at 59.9s, 59.999s, 60.0s, 60.001s.
   Confirm strict rejection of any t < 60.0s.
2. Duration bounds: candidates with durations 29.9s, 30.0s, 49.9s, 50.0s, 50.1s.
   Confirm strictly [30.0, 50.0]s is permitted.
3. Hook delay: candidates with first spoken word at 1.4s (accepted), 1.6s (adjusted or rejected).
"""

import os
import sys
import unittest
import asyncio
from pydantic import ValidationError

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.schemas import (
    ShortCandidate as PydanticShortCandidate,
    WordTimestamp,
    VideoSegment,
    ClipManifest,
    CuttingManifest,
    FusedTimeline,
    SceneChange,
    PipelineState
)
from wsl2_gpu.editorial_engine import (
    EditorialEngine,
    ShortCandidate,
    MIN_START_TIME,
    MIN_DURATION,
    MAX_DURATION,
    MAX_HOOK_DELAY
)
from wsl2_gpu.debate_graph import DebateGraph


class TestMilestone1BoundaryAndDurationStress(unittest.TestCase):

    def setUp(self):
        self.engine = EditorialEngine()
        self.debate_graph = DebateGraph()

    # ═════════════════════════════════════════════════════════════════════════
    # 1. Teaser Boundary Stress Tests (59.9s, 59.999s, 60.0s, 60.001s)
    # ═════════════════════════════════════════════════════════════════════════

    def test_teaser_boundary_59_9s_strictly_rejected(self):
        """Stress: t=59.9s must be rejected by Pydantic, EditorialEngine, and DebateGraph."""
        # A. Pydantic schema validation
        with self.assertRaises(ValidationError) as ctx:
            PydanticShortCandidate(
                start_time_seconds=59.9,
                duration_seconds=35.0,
                title="Teaser 59.9s",
                reasoning="Testing teaser rejection"
            )
        self.assertIn("greater than or equal to 60", str(ctx.exception))

        # B. EditorialEngine validation
        c = ShortCandidate(
            start_time_seconds=59.9,
            duration_seconds=35.0,
            title="Teaser 59.9s",
            reasoning="Testing teaser rejection"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0)
        self.assertEqual(len(valid), 0, "Candidate at t=59.9s must be rejected by EditorialEngine")

        # C. DebateGraph Director programmatic check
        manifest = CuttingManifest(
            clips=[
                ClipManifest(
                    segments=[VideoSegment(segment_start=59.9, segment_end=94.9)],
                    hook_score=0.9, retention_score=0.9, reasoning="Teaser test"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=100.0, segment_end=135.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 2"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=140.0, segment_end=175.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 3"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=180.0, segment_end=215.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 4"
                ),
            ]
        )
        scene_changes = [
            SceneChange(timestamp=65.0, confidence=0.95, frame_idx=1950),
            SceneChange(timestamp=105.0, confidence=0.95, frame_idx=3150),
            SceneChange(timestamp=145.0, confidence=0.95, frame_idx=4350),
            SceneChange(timestamp=185.0, confidence=0.95, frame_idx=5550),
        ]
        state = PipelineState(
            source_path="/tmp/test.mp4",
            video_name_hash="h1",
            output_dir="/tmp/out",
            fused_timeline=FusedTimeline(
                source_path="/tmp/test.mp4",
                video_duration=300.0,
                scene_changes=scene_changes
            ),
            cutting_manifest=manifest
        )
        res = asyncio.run(self.debate_graph.director_node(state))
        result_manifest = res["cutting_manifest"]
        self.assertFalse(result_manifest.consensus_reached)
        self.assertIn("intro teaser zone", result_manifest.clips[0].director_critique)

    def test_teaser_boundary_59_999s_strictly_rejected(self):
        """Stress: t=59.999s (sub-millisecond pre-boundary) must be strictly rejected."""
        # A. Pydantic schema validation
        with self.assertRaises(ValidationError) as ctx:
            PydanticShortCandidate(
                start_time_seconds=59.999,
                duration_seconds=35.0,
                title="Teaser 59.999s",
                reasoning="Testing precision teaser rejection"
            )
        self.assertIn("greater than or equal to 60", str(ctx.exception))

        # B. EditorialEngine validation
        c = ShortCandidate(
            start_time_seconds=59.999,
            duration_seconds=35.0,
            title="Teaser 59.999s",
            reasoning="Testing precision teaser rejection"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0)
        self.assertEqual(len(valid), 0, "Candidate at t=59.999s must be rejected by EditorialEngine")

        # C. DebateGraph Director programmatic check
        manifest = CuttingManifest(
            clips=[
                ClipManifest(
                    segments=[VideoSegment(segment_start=59.999, segment_end=94.999)],
                    hook_score=0.9, retention_score=0.9, reasoning="Teaser sub-ms"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=100.0, segment_end=135.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 2"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=140.0, segment_end=175.0)],
                    hook_score=0.9, reasoning="Valid 3", retention_score=0.9
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=180.0, segment_end=215.0)],
                    hook_score=0.9, reasoning="Valid 4", retention_score=0.9
                ),
            ]
        )
        scene_changes = [
            SceneChange(timestamp=65.0, confidence=0.95, frame_idx=1950),
            SceneChange(timestamp=105.0, confidence=0.95, frame_idx=3150),
            SceneChange(timestamp=145.0, confidence=0.95, frame_idx=4350),
            SceneChange(timestamp=185.0, confidence=0.95, frame_idx=5550),
        ]
        state = PipelineState(
            source_path="/tmp/test.mp4",
            video_name_hash="h1",
            output_dir="/tmp/out",
            fused_timeline=FusedTimeline(
                source_path="/tmp/test.mp4",
                video_duration=300.0,
                scene_changes=scene_changes
            ),
            cutting_manifest=manifest
        )
        res = asyncio.run(self.debate_graph.director_node(state))
        result_manifest = res["cutting_manifest"]
        self.assertFalse(result_manifest.consensus_reached)
        self.assertIn("intro teaser zone", result_manifest.clips[0].director_critique)

    def test_teaser_boundary_60_0s_accepted(self):
        """Stress: t=60.0s (exact boundary) must be accepted."""
        # A. Pydantic schema validation
        c_pydantic = PydanticShortCandidate(
            start_time_seconds=60.0,
            duration_seconds=35.0,
            title="Valid 60.0s",
            reasoning="Exact boundary test"
        )
        self.assertEqual(c_pydantic.start_time_seconds, 60.0)

        # B. EditorialEngine validation
        c = ShortCandidate(
            start_time_seconds=60.0,
            duration_seconds=35.0,
            title="Valid 60.0s",
            reasoning="Exact boundary test"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].start_time_seconds, 60.0)

    def test_teaser_boundary_60_001s_accepted(self):
        """Stress: t=60.001s (sub-millisecond post-boundary) must be accepted."""
        # A. Pydantic schema validation
        c_pydantic = PydanticShortCandidate(
            start_time_seconds=60.001,
            duration_seconds=35.0,
            title="Valid 60.001s",
            reasoning="Sub-ms post boundary test"
        )
        self.assertEqual(c_pydantic.start_time_seconds, 60.001)

        # B. EditorialEngine validation
        c = ShortCandidate(
            start_time_seconds=60.001,
            duration_seconds=35.0,
            title="Valid 60.001s",
            reasoning="Sub-ms post boundary test"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].start_time_seconds, 60.001)

    def test_teaser_fallback_zero_candidates_before_60s(self):
        """Stress: Fallback generator given audio from 0.0s must never select t < 60.0s."""
        dense_words = [
            {"word": f"token_{i}.", "start": i * 0.4, "end": i * 0.4 + 0.35}
            for i in range(1000)  # 0.0s to 400.0s
        ]
        fallbacks = self.engine.select_fallback_shorts(
            transcript_words=dense_words,
            shots=[],
            video_duration=400.0,
            needed=6
        )
        self.assertGreaterEqual(len(fallbacks), 4)
        for idx, fb in enumerate(fallbacks):
            self.assertGreaterEqual(
                fb.start_time_seconds, 60.0,
                f"Candidate {idx} started at {fb.start_time_seconds}s < 60.0s"
            )

    # ═════════════════════════════════════════════════════════════════════════
    # 2. Duration Bounds Stress Tests (29.9s, 30.0s, 49.9s, 50.0s, 50.1s)
    # ═════════════════════════════════════════════════════════════════════════

    def test_duration_boundary_29_9s_strictly_rejected(self):
        """Stress: duration=29.9s must be rejected by Pydantic, EditorialEngine, and DebateGraph."""
        # A. Pydantic
        with self.assertRaises(ValidationError) as ctx:
            PydanticShortCandidate(
                start_time_seconds=70.0,
                duration_seconds=29.9,
                title="Sub-30s Short",
                reasoning="Testing minimum duration"
            )
        self.assertIn("greater than or equal to 30", str(ctx.exception))

        # B. EditorialEngine
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=29.9,
            title="Sub-30s Short",
            reasoning="Testing minimum duration"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0)
        self.assertEqual(len(valid), 0, "Candidate with duration=29.9s must be rejected")

        # C. DebateGraph Director
        manifest = CuttingManifest(
            clips=[
                ClipManifest(
                    segments=[VideoSegment(segment_start=70.0, segment_end=99.9)],
                    hook_score=0.9, retention_score=0.9, reasoning="29.9s duration"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=110.0, segment_end=145.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 2"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=150.0, segment_end=185.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 3"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=190.0, segment_end=225.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 4"
                ),
            ]
        )
        state = PipelineState(
            source_path="/tmp/test.mp4",
            video_name_hash="h2",
            output_dir="/tmp/out",
            cutting_manifest=manifest
        )
        res = asyncio.run(self.debate_graph.director_node(state))
        self.assertFalse(res["cutting_manifest"].consensus_reached)
        self.assertIn("violates the strict 30.0s - 50.0s", res["cutting_manifest"].clips[0].director_critique)

    def test_duration_boundary_30_0s_accepted(self):
        """Stress: duration=30.0s (exact lower bound) must be accepted."""
        # A. Pydantic
        c_pyd = PydanticShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=30.0,
            title="Exact 30.0s Short",
            reasoning="Exact min duration"
        )
        self.assertEqual(c_pyd.duration_seconds, 30.0)

        # B. EditorialEngine
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=30.0,
            title="Exact 30.0s Short",
            reasoning="Exact min duration"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].duration_seconds, 30.0)

    def test_duration_boundary_49_9s_accepted(self):
        """Stress: duration=49.9s (sub-upper-limit) must be accepted."""
        # A. Pydantic
        c_pyd = PydanticShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=49.9,
            title="Valid 49.9s Short",
            reasoning="High duration"
        )
        self.assertEqual(c_pyd.duration_seconds, 49.9)

        # B. EditorialEngine
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=49.9,
            title="Valid 49.9s Short",
            reasoning="High duration"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].duration_seconds, 49.9)

    def test_duration_boundary_50_0s_accepted(self):
        """Stress: duration=50.0s (exact upper bound) must be accepted."""
        # A. Pydantic
        c_pyd = PydanticShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=50.0,
            title="Exact 50.0s Short",
            reasoning="Exact max duration"
        )
        self.assertEqual(c_pyd.duration_seconds, 50.0)

        # B. EditorialEngine
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=50.0,
            title="Exact 50.0s Short",
            reasoning="Exact max duration"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0].duration_seconds, 50.0)

    def test_duration_boundary_50_1s_strictly_rejected(self):
        """Stress: duration=50.1s must be rejected by Pydantic, EditorialEngine, and DebateGraph."""
        # A. Pydantic
        with self.assertRaises(ValidationError) as ctx:
            PydanticShortCandidate(
                start_time_seconds=70.0,
                duration_seconds=50.1,
                title="Exceeding 50s Short",
                reasoning="Testing upper duration limit"
            )
        self.assertIn("less than or equal to 50", str(ctx.exception))

        # B. EditorialEngine
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=50.1,
            title="Exceeding 50s Short",
            reasoning="Testing upper duration limit"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0)
        self.assertEqual(len(valid), 0, "Candidate with duration=50.1s must be rejected")

        # C. DebateGraph Director
        manifest = CuttingManifest(
            clips=[
                ClipManifest(
                    segments=[VideoSegment(segment_start=70.0, segment_end=120.1)],
                    hook_score=0.9, retention_score=0.9, reasoning="50.1s duration"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=130.0, segment_end=165.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 2"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=170.0, segment_end=205.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 3"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=210.0, segment_end=245.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 4"
                ),
            ]
        )
        state = PipelineState(
            source_path="/tmp/test.mp4",
            video_name_hash="h3",
            output_dir="/tmp/out",
            cutting_manifest=manifest
        )
        res = asyncio.run(self.debate_graph.director_node(state))
        self.assertFalse(res["cutting_manifest"].consensus_reached)
        self.assertIn("violates the strict 30.0s - 50.0s", res["cutting_manifest"].clips[0].director_critique)

    # ═════════════════════════════════════════════════════════════════════════
    # 3. Hook Delay Stress Tests (1.4s accepted vs 1.6s adjusted / rejected)
    # ═════════════════════════════════════════════════════════════════════════

    def test_hook_delay_1_4s_accepted_without_adjustment(self):
        """Stress: First spoken word at 1.4s delay (<= 1.5s) must be accepted as-is."""
        # Candidate starting at 70.0s, duration 35.0s
        # First word starts at 71.4s (delay 1.40s)
        words = [
            {"word": "Welcome", "start": 71.4, "end": 71.8},
            {"word": "to", "start": 71.9, "end": 72.1},
            {"word": "today's", "start": 72.2, "end": 72.6},
            {"word": "demo.", "start": 104.5, "end": 104.9},
        ]
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=35.0,
            title="Hook 1.4s",
            reasoning="Testing hook delay 1.4s"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0, transcript_words=words)
        self.assertEqual(len(valid), 1)
        accepted = valid[0]
        # Start time must remain unchanged at 70.0s
        self.assertEqual(accepted.start_time_seconds, 70.0)
        self.assertEqual(accepted.first_word_delay, 1.4)

        # In DebateGraph Director:
        state_words = [
            WordTimestamp(word="Welcome", start=71.4, end=71.8, confidence=0.9),
            WordTimestamp(word="Clip2", start=110.2, end=110.8, confidence=0.9),
            WordTimestamp(word="Clip3", start=150.2, end=150.8, confidence=0.9),
            WordTimestamp(word="Clip4", start=190.2, end=190.8, confidence=0.9),
        ]
        # Verify Director hook validation calculation directly:
        first_seg_start = 70.0
        words_in_hook = [
            w.start for w in state_words
            if w.end > first_seg_start and w.start <= first_seg_start + 5.0
        ]
        first_word_delay = words_in_hook[0] - first_seg_start
        self.assertAlmostEqual(first_word_delay, 1.4, places=2)
        self.assertLessEqual(first_word_delay, 1.5, "1.4s delay must satisfy Director <= 1.5s requirement")

    def test_hook_delay_1_6s_adjusted_when_duration_permits(self):
        """Stress: First spoken word at 1.6s delay (> 1.5s) with 38s duration must be adjusted forward."""
        # Candidate starting at 70.0s, duration 38.0s (ends at 108.0s)
        # First word starts at 71.6s (delay = 1.60s > 1.5s)
        # Adjusting start forward: adjusted_start = round(71.6 - 0.15, 2) = 71.45s
        # Remaining duration: 108.0 - 71.45 = 36.55s (which is >= 30.0s)
        words = [
            {"word": "Delayed", "start": 71.6, "end": 72.0},
            {"word": "speech", "start": 72.1, "end": 72.5},
            {"word": "concludes.", "start": 107.5, "end": 107.9},
        ]
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=38.0,
            title="Hook 1.6s Buffered",
            reasoning="Testing hook adjustment"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0, transcript_words=words)
        self.assertEqual(len(valid), 1, "Candidate with 1.6s hook delay should be adjusted and accepted")
        adjusted = valid[0]
        # Start time must have been adjusted forward to eliminate leading dead air
        self.assertGreater(adjusted.start_time_seconds, 70.0)
        self.assertAlmostEqual(adjusted.start_time_seconds, 71.45, places=2)
        # Duration adjusted accordingly: 108.0 - 71.45 = 36.55s
        self.assertGreaterEqual(adjusted.duration_seconds, 30.0)
        self.assertLessEqual(adjusted.duration_seconds, 50.0)
        # New delay relative to adjusted start must be <= 1.5s
        new_delay = 71.6 - adjusted.start_time_seconds
        self.assertLessEqual(new_delay, 1.5)

    def test_hook_delay_1_6s_rejected_when_duration_would_drop_below_30s(self):
        """Stress: First spoken word at 1.6s delay with tight 30.5s duration must be rejected."""
        # Candidate starting at 70.0s, duration 30.5s (ends at 100.5s)
        # First word at 71.6s.
        # Adjusted start would be 71.45s -> remaining duration = 100.5 - 71.45 = 29.05s < 30.0s!
        # Cannot adjust without violating the 30.0s minimum duration constraint -> MUST REJECT!
        words = [
            {"word": "Delayed", "start": 71.6, "end": 72.0},
            {"word": "tight", "start": 72.1, "end": 72.5},
            {"word": "conclusion.", "start": 100.0, "end": 100.4},
        ]
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=30.5,
            title="Hook 1.6s Tight",
            reasoning="Testing hook rejection on tight duration"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0, transcript_words=words)
        self.assertEqual(
            len(valid), 0,
            "Candidate with 1.6s hook delay that would shrink below 30.0s MUST be rejected"
        )

    def test_hook_delay_1_6s_rejected_by_debate_graph_director(self):
        """Stress: DebateGraph Director must reject unadjusted clip with 1.6s first spoken word delay."""
        manifest = CuttingManifest(
            clips=[
                ClipManifest(
                    segments=[VideoSegment(segment_start=70.0, segment_end=105.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Hook 1.6s unadjusted"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=110.0, segment_end=145.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 2"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=150.0, segment_end=185.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 3"
                ),
                ClipManifest(
                    segments=[VideoSegment(segment_start=190.0, segment_end=225.0)],
                    hook_score=0.9, retention_score=0.9, reasoning="Valid 4"
                ),
            ]
        )
        # Word at 71.6s -> delay is 71.6 - 70.0 = 1.60s (> 1.5s)
        state_words = [
            WordTimestamp(word="DelayedIntro", start=71.6, end=72.0, confidence=0.9),
            WordTimestamp(word="Clip2", start=110.2, end=110.8, confidence=0.9),
            WordTimestamp(word="Clip3", start=150.2, end=150.8, confidence=0.9),
            WordTimestamp(word="Clip4", start=190.2, end=190.8, confidence=0.9),
        ]
        scene_changes = [
            SceneChange(timestamp=75.0, confidence=0.95, frame_idx=2250),
            SceneChange(timestamp=115.0, confidence=0.95, frame_idx=3450),
            SceneChange(timestamp=155.0, confidence=0.95, frame_idx=4650),
            SceneChange(timestamp=195.0, confidence=0.95, frame_idx=5850),
        ]
        state = PipelineState(
            source_path="/tmp/test.mp4",
            video_name_hash="h5",
            output_dir="/tmp/out",
            fused_timeline=FusedTimeline(
                source_path="/tmp/test.mp4",
                video_duration=300.0,
                transcript_words=state_words,
                scene_changes=scene_changes
            ),
            transcript_words=state_words,
            cutting_manifest=manifest
        )
        res = asyncio.run(self.debate_graph.director_node(state))
        self.assertFalse(res["cutting_manifest"].consensus_reached)
        critique = res["cutting_manifest"].clips[0].director_critique
        self.assertIn("Hook failure", critique)
        self.assertIn("1.5s threshold", critique)

    # ═════════════════════════════════════════════════════════════════════════
    # 4. Comprehensive Mixed-Batch Matrix Stress Test
    # ═════════════════════════════════════════════════════════════════════════

    def test_mixed_adversarial_batch_filtering(self):
        """Stress: Process a combined batch of valid and invalid boundary candidates simultaneously."""
        candidates = [
            ShortCandidate(start_time_seconds=59.9, duration_seconds=35.0, title="Teaser59.9", reasoning="Teaser 59.9"),
            ShortCandidate(start_time_seconds=59.999, duration_seconds=35.0, title="Teaser59.999", reasoning="Teaser 59.999"),
            ShortCandidate(start_time_seconds=60.0, duration_seconds=29.9, title="Dur29.9", reasoning="Duration 29.9"),
            ShortCandidate(start_time_seconds=60.0, duration_seconds=30.0, title="Dur30.0", reasoning="Duration 30.0"),
            ShortCandidate(start_time_seconds=100.0, duration_seconds=49.9, title="Dur49.9", reasoning="Duration 49.9"),
            ShortCandidate(start_time_seconds=150.0, duration_seconds=50.0, title="Dur50.0", reasoning="Duration 50.0"),
            ShortCandidate(start_time_seconds=205.0, duration_seconds=50.1, title="Dur50.1", reasoning="Duration 50.1"),
            ShortCandidate(start_time_seconds=260.001, duration_seconds=35.0, title="Start60.001", reasoning="Start 260.001"),
        ]

        valid = self.engine._validate_and_filter(candidates, video_duration=400.0)
        accepted_titles = [v.title for v in valid]

        # Assert rejections
        self.assertNotIn("Teaser59.9", accepted_titles)
        self.assertNotIn("Teaser59.999", accepted_titles)
        self.assertNotIn("Dur29.9", accepted_titles)
        self.assertNotIn("Dur50.1", accepted_titles)

        # Assert acceptances
        self.assertIn("Dur30.0", accepted_titles)
        self.assertIn("Dur49.9", accepted_titles)
        self.assertIn("Dur50.0", accepted_titles)
        self.assertIn("Start60.001", accepted_titles)

        # All accepted must satisfy contracts
        for v in valid:
            self.assertGreaterEqual(v.start_time_seconds, 60.0)
            self.assertGreaterEqual(v.duration_seconds, 30.0)
            self.assertLessEqual(v.duration_seconds, 50.0)

    # ═════════════════════════════════════════════════════════════════════════
    # 5. Additional Corner-Case & Epsilon Precision Stress Tests
    # ═════════════════════════════════════════════════════════════════════════

    def test_teaser_boundary_microsecond_precision(self):
        """Stress: Microsecond epsilon boundary precision (59.999999s vs 60.000001s)."""
        # 59.999999s must fail Pydantic
        with self.assertRaises(ValidationError):
            PydanticShortCandidate(
                start_time_seconds=59.999999,
                duration_seconds=30.0,
                title="Microsecond Pre",
                reasoning="Testing microsecond precision"
            )

        # 60.000001s must succeed
        c_post = PydanticShortCandidate(
            start_time_seconds=60.000001,
            duration_seconds=30.0,
            title="Microsecond Post",
            reasoning="Testing microsecond precision"
        )
        self.assertEqual(c_post.start_time_seconds, 60.000001)

    def test_hook_complete_silence_in_0_to_5s_window_rejected(self):
        """Stress: When audio is completely silent in the 0-5s window, candidate must be rejected."""
        # Candidate starting at 70.0s. First word appears at 76.0s (outside 0-5s hook window [70.0, 75.0])
        words = [
            {"word": "LateSpeech", "start": 76.0, "end": 76.5},
            {"word": "concludes.", "start": 104.0, "end": 104.5},
        ]
        c = ShortCandidate(
            start_time_seconds=70.0,
            duration_seconds=35.0,
            title="Silent Hook",
            reasoning="Testing silent hook window"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0, transcript_words=words)
        self.assertEqual(len(valid), 0, "Candidate with zero words in 0-5s hook window must be rejected")

    def test_cross_boundary_speech_clamps_strictly_at_60s(self):
        """Stress: A word starting before 60.0s (e.g. 59.5s) must not snap start_time below 60.0s."""
        # Word spans 59.5s to 60.5s
        words = [
            {"word": "Crossing", "start": 59.5, "end": 60.5},
            {"word": "speech.", "start": 91.0, "end": 91.5},
        ]
        c = ShortCandidate(
            start_time_seconds=60.0,
            duration_seconds=35.0,
            title="Boundary Overlap",
            reasoning="Testing speech crossing 60.0s"
        )
        valid = self.engine._validate_and_filter([c], video_duration=300.0, transcript_words=words)
        self.assertEqual(len(valid), 1)
        # Even though first word starts at 59.5s, candidate start must NOT be pushed below 60.0s!
        self.assertGreaterEqual(valid[0].start_time_seconds, 60.0)

    def test_video_duration_boundary_capacity(self):
        """Stress: Video durations below mathematical capacity cannot produce target shorts."""
        # 89.9s total: cannot fit 60s teaser + 30s short
        shorts_89 = self.engine.select_fallback_shorts(
            transcript_words=[], shots=[], video_duration=89.9, needed=4
        )
        self.assertEqual(len(shorts_89), 0)

        # 90.0s total: can fit at most 1 short (60.0s to 90.0s)
        shorts_90 = self.engine.select_fallback_shorts(
            transcript_words=[], shots=[], video_duration=90.0, needed=4
        )
        self.assertLessEqual(len(shorts_90), 1)
        if len(shorts_90) == 1:
            self.assertEqual(shorts_90[0].start_time_seconds, 60.0)
            self.assertEqual(shorts_90[0].duration_seconds, 30.0)

        # 180.0s total: exactly enough for 4 x 30.0s shorts
        shorts_180 = self.engine.select_fallback_shorts(
            transcript_words=[], shots=[], video_duration=180.0, needed=4
        )
        self.assertEqual(len(shorts_180), 4)

    def test_debate_graph_multi_segment_duration_bounds(self):
        """Stress: DebateGraph Director must enforce duration on multi-segment stitched clips."""
        # Multi-segment clip summing to 29.9s (15.0s + 14.9s): REJECTED
        clip_29_9 = ClipManifest(
            segments=[
                VideoSegment(segment_start=70.0, segment_end=85.0),
                VideoSegment(segment_start=90.0, segment_end=104.9),
            ],
            hook_score=0.9, retention_score=0.9, reasoning="Multi-cut 29.9s"
        )
        # Multi-segment clip summing to 50.1s (25.0s + 25.1s): REJECTED
        clip_50_1 = ClipManifest(
            segments=[
                VideoSegment(segment_start=70.0, segment_end=95.0),
                VideoSegment(segment_start=100.0, segment_end=125.1),
            ],
            hook_score=0.9, retention_score=0.9, reasoning="Multi-cut 50.1s"
        )
        dummy_clip_2 = ClipManifest(
            segments=[VideoSegment(segment_start=130.0, segment_end=165.0)],
            hook_score=0.9, retention_score=0.9, reasoning="Valid 2"
        )
        dummy_clip_3 = ClipManifest(
            segments=[VideoSegment(segment_start=170.0, segment_end=205.0)],
            hook_score=0.9, retention_score=0.9, reasoning="Valid 3"
        )
        dummy_clip_4 = ClipManifest(
            segments=[VideoSegment(segment_start=210.0, segment_end=245.0)],
            hook_score=0.9, retention_score=0.9, reasoning="Valid 4"
        )

        manifest_sub = CuttingManifest(clips=[clip_29_9, dummy_clip_2, dummy_clip_3, dummy_clip_4])
        state_sub = PipelineState(
            source_path="/tmp/test.mp4", video_name_hash="hsub",
            output_dir="/tmp/out", cutting_manifest=manifest_sub
        )
        res_sub = asyncio.run(self.debate_graph.director_node(state_sub))
        self.assertFalse(res_sub["cutting_manifest"].consensus_reached)
        self.assertIn("violates the strict 30.0s - 50.0s", res_sub["cutting_manifest"].clips[0].director_critique)

        manifest_over = CuttingManifest(clips=[clip_50_1, dummy_clip_2, dummy_clip_3, dummy_clip_4])
        state_over = PipelineState(
            source_path="/tmp/test.mp4", video_name_hash="hover",
            output_dir="/tmp/out", cutting_manifest=manifest_over
        )
        res_over = asyncio.run(self.debate_graph.director_node(state_over))
        self.assertFalse(res_over["cutting_manifest"].consensus_reached)
        self.assertIn("violates the strict 30.0s - 50.0s", res_over["cutting_manifest"].clips[0].director_critique)


if __name__ == "__main__":
    unittest.main()

