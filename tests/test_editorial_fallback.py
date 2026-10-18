import unittest
from wsl2_gpu.editorial_engine import EditorialEngine, ShortCandidate
from wsl2_gpu.orchestrator import Orchestrator


class TestEditorialFallback(unittest.TestCase):

    def setUp(self):
        self.engine = EditorialEngine()
        self.orchestrator = Orchestrator(nats_url="nats://localhost:4222")

    def test_fallback_generates_four_distinct_shorts_with_transcript(self):
        # 300-second video with dialogue
        words = [
            {"word": f"word_{i}", "start": 60.0 + i * 0.4, "end": 60.0 + i * 0.4 + 0.35}
            for i in range(500)
        ]
        shorts = self.engine.select_fallback_shorts(
            transcript_words=words,
            shots=[],
            video_duration=300.0,
            existing=[],
            needed=4
        )
        self.assertEqual(len(shorts), 4)

        for idx, s in enumerate(shorts):
            self.assertGreaterEqual(s.start_time_seconds, 60.0, f"Short {idx} starts before 60s")
            self.assertGreaterEqual(s.duration_seconds, 30.0, f"Short {idx} too short")
            self.assertLessEqual(s.duration_seconds, 50.0, f"Short {idx} too long")
            self.assertLessEqual(s.end_time_seconds, 300.0, f"Short {idx} exceeds video duration")

        # Verify non-overlapping
        for i in range(len(shorts)):
            for j in range(i + 1, len(shorts)):
                s1, s2 = shorts[i], shorts[j]
                overlap = (s1.start_time_seconds < s2.end_time_seconds and
                           s1.end_time_seconds > s2.start_time_seconds)
                self.assertFalse(overlap, f"Short {i} overlaps with Short {j}")

    def test_top_up_when_partial_shorts_exist(self):
        # Existing 1 valid short
        existing = [
            ShortCandidate(
                start_time_seconds=70.0,
                duration_seconds=35.0,
                title="Existing Short 1",
                reasoning="Already selected by VLM"
            )
        ]
        words = [
            {"word": f"word_{i}", "start": 60.0 + i * 0.4, "end": 60.0 + i * 0.4 + 0.35}
            for i in range(500)
        ]
        fallbacks = self.engine.select_fallback_shorts(
            transcript_words=words,
            shots=[],
            video_duration=300.0,
            existing=existing,
            needed=3
        )
        self.assertEqual(len(fallbacks), 3)

        total_shorts = existing + fallbacks
        self.assertEqual(len(total_shorts), 4)
        for i in range(len(total_shorts)):
            for j in range(i + 1, len(total_shorts)):
                s1, s2 = total_shorts[i], total_shorts[j]
                overlap = (s1.start_time_seconds < s2.end_time_seconds and
                           s1.end_time_seconds > s2.start_time_seconds)
                self.assertFalse(overlap, f"Total short {i} overlaps with {j}")

    def test_orchestrator_generate_fallback_shorts(self):
        shorts = self.orchestrator._generate_fallback_shorts(video_duration=300.0, needed=4)
        self.assertEqual(len(shorts), 4)
        for s in shorts:
            self.assertGreaterEqual(s["start_time_seconds"], 60.0)
            self.assertGreaterEqual(s["duration_seconds"], 30.0)
            self.assertLessEqual(s["duration_seconds"], 50.0)


if __name__ == "__main__":
    unittest.main()
