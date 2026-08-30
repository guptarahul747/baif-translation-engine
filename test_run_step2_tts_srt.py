import unittest

import numpy as np

from run_step2_tts_srt import build_tts_timeline


class TtsTimelineTests(unittest.TestCase):
    def test_continuous_mode_has_no_inserted_pause(self):
        clips = [
            {"samples": np.ones(10), "start": 0.0, "end": 1.0},
            {"samples": np.ones(10), "start": 20.0, "end": 21.0},
        ]

        timeline, silence = build_tts_timeline(clips, 10, timing_mode="continuous")

        self.assertEqual(timeline.size, 20)
        self.assertEqual(silence, 0.0)

    def test_compact_mode_removes_artificial_intra_segment_wait(self):
        clips = [
            {"samples": np.ones(10), "start": 0.0, "end": 10.0},
            {"samples": np.ones(10), "start": 10.0, "end": 20.0},
        ]

        timeline, silence = build_tts_timeline(clips, 10, timing_mode="compact")

        self.assertEqual(timeline.size, 22)
        self.assertAlmostEqual(silence, 0.2)

    def test_compact_mode_caps_long_source_silence(self):
        clips = [
            {"samples": np.ones(10), "start": 0.0, "end": 1.0},
            {"samples": np.ones(10), "start": 48.0, "end": 49.0},
        ]

        timeline, silence = build_tts_timeline(clips, 100, timing_mode="compact")

        self.assertEqual(timeline.size, 145)
        self.assertAlmostEqual(silence, 1.25)

    def test_source_mode_retains_absolute_timestamps(self):
        clips = [
            {"samples": np.ones(10), "start": 0.0, "end": 10.0},
            {"samples": np.ones(10), "start": 10.0, "end": 20.0},
        ]

        timeline, silence = build_tts_timeline(clips, 10, timing_mode="source")

        self.assertEqual(timeline.size, 110)
        self.assertAlmostEqual(silence, 9.0)


if __name__ == "__main__":
    unittest.main()
