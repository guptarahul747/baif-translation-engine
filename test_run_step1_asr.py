import unittest
from dataclasses import dataclass

import numpy as np

from run_step1_asr import (
    build_decode_windows,
    clean_repeated_words,
    finalize_segments,
    transcribe_in_chunks,
    validate_chunk_settings,
)


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str


@dataclass
class FakeInfo:
    language: str = "mr"
    language_probability: float = 1.0


class FakeWhisperModel:
    def __init__(self):
        self.window_lengths = []

    def transcribe(self, audio, **kwargs):
        self.window_lengths.append(len(audio))
        # At 10 Hz this segment belongs to the core of each successive window.
        return iter([FakeSegment(2.5, 4.0, "शब्द शब्द शब्द शब्द")]), FakeInfo()


class FakeGapRecoveryModel:
    def __init__(self):
        self.beams = []

    def transcribe(self, audio, **kwargs):
        self.beams.append(kwargs["beam_size"])
        if kwargs["beam_size"] >= 5:
            return iter([FakeSegment(2.0, 6.0, "recovered speech")]), FakeInfo()
        return iter(
            [
                FakeSegment(0.0, 2.0, "first phrase"),
                FakeSegment(12.0, 15.0, "last phrase"),
            ]
        ), FakeInfo()


class AsrChunkingTests(unittest.TestCase):
    def test_repetition_cleanup_limits_pathological_word_runs(self):
        self.assertEqual(
            clean_repeated_words("आहे, आहे आहे आहे पुढे"),
            "आहे, आहे पुढे",
        )
        self.assertEqual(clean_repeated_words("हो हो पुढे"), "हो हो पुढे")

    def test_decode_windows_cover_timeline_with_bounded_overlap(self):
        windows = build_decode_windows(65.0, chunk_seconds=25.0, context_seconds=2.5)
        self.assertEqual(
            windows,
            [
                (0.0, 25.0, 0.0, 27.5),
                (25.0, 50.0, 22.5, 52.5),
                (50.0, 65.0, 47.5, 65.0),
            ],
        )
        self.assertTrue(all(end - start <= 30.0 for _, _, start, end in windows))

    def test_invalid_window_larger_than_whisper_context_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_chunk_settings(26.0, 2.5)

    def test_chunk_transcription_offsets_segments_and_processes_every_core(self):
        model = FakeWhisperModel()
        audio = np.zeros(650, dtype=np.float32)

        segments, languages, cleaned = transcribe_in_chunks(
            model=model,
            audio=audio,
            sample_rate=10,
            requested_lang="mr",
            beam_size=5,
            chunk_seconds=25.0,
            context_seconds=2.5,
        )

        self.assertEqual(model.window_lengths, [275, 300, 175])
        self.assertEqual([item["start"] for item in segments], [2.5, 25.0, 50.0])
        self.assertEqual([item["id"] for item in segments], [1, 2, 3])
        self.assertTrue(all(item["text"] == "शब्द शब्द" for item in segments))
        self.assertGreater(languages["mr"], 0)
        self.assertEqual(cleaned, 3)

    def test_finalize_segments_removes_overlap_duplicates(self):
        segments = finalize_segments(
            [
                {"start": 1.0, "end": 3.0, "text": "hello world"},
                {"start": 2.8, "end": 4.0, "text": "hello world"},
                {"start": 4.1, "end": 6.0, "text": "world again today"},
            ]
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["end"], 4.0)
        self.assertEqual(segments[1]["id"], 2)

    def test_audio_present_gap_is_retried_with_quality_beam(self):
        model = FakeGapRecoveryModel()
        audio = np.ones(200, dtype=np.float32)

        segments, _, _ = transcribe_in_chunks(
            model=model,
            audio=audio,
            sample_rate=10,
            requested_lang="mr",
            beam_size=3,
            chunk_seconds=20.0,
            context_seconds=0.0,
        )

        self.assertEqual(model.beams, [3, 5])
        self.assertIn("recovered speech", [item["text"] for item in segments])


if __name__ == "__main__":
    unittest.main()
