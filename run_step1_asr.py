import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from faster_whisper import WhisperModel
from app.modules.ingestion import extract_and_normalize_audio

VAULT_DIR = BASE_DIR / "local_model_vault"
OUTPUT_DIR = BASE_DIR / "storage_vault" / "outputs"

SUPPORTED_LOCALES = {
    "auto": None,
    "mr": "mr",
    "hi": "hi",
    "en": "en",
}

DEFAULT_CHUNK_SECONDS = 28.0
DEFAULT_CHUNK_CONTEXT_SECONDS = 1.0
WHISPER_AUDIO_WINDOW_SECONDS = 30.0
GAP_REPORT_SECONDS = 5.0
SILENCE_DBFS = -40.0
RECOVERY_GAP_SECONDS = 6.0
MAX_RECOVERY_WINDOWS = 8


def find_file(directory: Path, pattern: str):
    if not directory.exists():
        return None
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None


def normalize_word(word: str) -> str:
    return re.sub(
        r"[^\w\u0900-\u097F]",
        "",
        word,
        flags=re.UNICODE,
    ).casefold()


def clean_repeated_words(text: str, max_consecutive: int = 2) -> str:
    """
    Reduce accidental consecutive Whisper repetition.

    Examples:
        "आहे आहे आहे आहे" -> "आहे आहे"
        "yes yes yes yes" -> "yes yes"
    """
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return ""

    words = text.split()
    cleaned = []
    previous_normalized = None
    consecutive_count = 0

    for word in words:
        normalized = normalize_word(word)

        if not normalized:
            cleaned.append(word)
            continue

        if normalized == previous_normalized:
            consecutive_count += 1
        else:
            previous_normalized = normalized
            consecutive_count = 1

        if consecutive_count <= max_consecutive:
            cleaned.append(word)

    return " ".join(cleaned).strip()


def validate_chunk_settings(chunk_seconds: float, context_seconds: float) -> None:
    """Keep every decode window inside Whisper's 30-second audio context."""
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than zero")
    if context_seconds < 0:
        raise ValueError("chunk_context_seconds cannot be negative")
    if chunk_seconds + (2 * context_seconds) > WHISPER_AUDIO_WINDOW_SECONDS:
        raise ValueError(
            "chunk_seconds + 2 * chunk_context_seconds must be no more than "
            f"{WHISPER_AUDIO_WINDOW_SECONDS:g} seconds"
        )


def build_decode_windows(
    duration: float,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    context_seconds: float = DEFAULT_CHUNK_CONTEXT_SECONDS,
) -> list[tuple[float, float, float, float]]:
    """
    Return complete, ordered decode windows.

    Each tuple contains ``(core_start, core_end, window_start, window_end)``.
    The core regions cover the audio exactly once. Context on either side avoids
    losing words cut by a chunk boundary, while midpoint ownership prevents
    duplicate output from the overlap.
    """
    validate_chunk_settings(chunk_seconds, context_seconds)
    if duration <= 0:
        return []

    windows = []
    core_start = 0.0
    while core_start < duration:
        core_end = min(duration, core_start + chunk_seconds)
        window_start = max(0.0, core_start - context_seconds)
        window_end = min(duration, core_end + context_seconds)
        windows.append((core_start, core_end, window_start, window_end))
        core_start = core_end
    return windows


def _normalized_words(text: str) -> list[str]:
    return [word for word in (normalize_word(part) for part in text.split()) if word]


def _trim_repeated_prefix(previous_text: str, current_text: str) -> str:
    """Remove a duplicated multi-word phrase introduced at a chunk boundary."""
    previous_words = _normalized_words(previous_text)
    current_parts = current_text.split()
    current_words = _normalized_words(current_text)
    if len(previous_words) < 2 or len(current_words) < 2:
        return current_text

    maximum = min(10, len(previous_words), len(current_words))
    for overlap in range(maximum, 1, -1):
        if previous_words[-overlap:] == current_words[:overlap]:
            return " ".join(current_parts[overlap:]).strip()
    return current_text


def finalize_segments(segments: list[dict]) -> list[dict]:
    """Sort chunk output and remove overlap-only duplicate text."""
    ordered = sorted(segments, key=lambda item: (item["start"], item["end"]))
    finalized = []

    for item in ordered:
        current = dict(item)
        if finalized:
            previous = finalized[-1]
            previous_normalized = " ".join(_normalized_words(previous["text"]))
            current_normalized = " ".join(_normalized_words(current["text"]))

            # The same segment can occasionally be returned by both overlapping
            # windows. Keep one copy and retain its widest timestamp range.
            if (
                current_normalized
                and current_normalized == previous_normalized
                and current["start"] <= previous["end"] + 0.5
            ):
                previous["end"] = max(previous["end"], current["end"])
                continue

            if current["start"] <= previous["end"] + 1.0:
                current["text"] = _trim_repeated_prefix(
                    previous["text"], current["text"]
                )
                if not current["text"]:
                    continue

        finalized.append(current)

    for segment_id, item in enumerate(finalized, start=1):
        item["id"] = segment_id
    return finalized


def transcribe_in_chunks(
    model,
    audio: np.ndarray,
    sample_rate: int,
    requested_lang: str | None,
    beam_size: int,
    use_vad: bool = False,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    context_seconds: float = DEFAULT_CHUNK_CONTEXT_SECONDS,
) -> tuple[list[dict], dict[str, float], int]:
    """
    Decode every part of an audio timeline in bounded overlapping windows.

    Faster-Whisper normally advances through long files using predicted
    timestamp tokens. A bad timestamp or repetition loop can move that cursor
    forward by tens of seconds. Independent windows put a hard upper bound on
    that failure and guarantee that the next window is still decoded.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if audio.ndim != 1:
        raise ValueError("normalized ASR audio must be mono")

    duration = len(audio) / float(sample_rate)
    windows = build_decode_windows(duration, chunk_seconds, context_seconds)
    collected = []
    language_scores: dict[str, float] = {}
    repetitions_cleaned = 0

    transcribe_kwargs = dict(
        language=requested_lang,
        task="transcribe",
        beam_size=max(1, int(beam_size)),
        # Bounded chunks and n-gram blocking contain repetition without the
        # expensive multi-temperature retry loop.
        temperature=0.0,
        initial_prompt=None,
        condition_on_previous_text=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.75,
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
        vad_filter=use_vad,
    )
    if use_vad:
        transcribe_kwargs["vad_parameters"] = {
            "threshold": 0.35,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 1000,
            "speech_pad_ms": 500,
        }

    for window_number, window in enumerate(windows, start=1):
        core_start, core_end, window_start, window_end = window
        sample_start = max(0, int(round(window_start * sample_rate)))
        sample_end = min(len(audio), int(round(window_end * sample_rate)))
        audio_window = np.ascontiguousarray(audio[sample_start:sample_end])

        print(
            f"🔎 Chunk {window_number}/{len(windows)}: "
            f"{core_start:.2f}s → {core_end:.2f}s",
            flush=True,
        )
        segment_iterator, info = model.transcribe(audio_window, **transcribe_kwargs)

        info_language = (getattr(info, "language", None) or requested_lang or "unknown")
        info_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        language_scores[info_language] = language_scores.get(info_language, 0.0) + max(
            info_probability, 0.01
        ) * (core_end - core_start)

        is_last_window = window_number == len(windows)
        for raw_segment in segment_iterator:
            raw_text = re.sub(r"\s+", " ", (raw_segment.text or "")).strip()
            if not raw_text:
                continue

            text = clean_repeated_words(raw_text, max_consecutive=2)
            if not text:
                continue
            if text != raw_text:
                repetitions_cleaned += 1

            start = max(0.0, min(duration, window_start + float(raw_segment.start)))
            end = max(start, min(duration, window_start + float(raw_segment.end)))
            midpoint = start + ((end - start) / 2.0)

            owns_midpoint = core_start <= midpoint < core_end
            if is_last_window and math.isclose(midpoint, core_end, abs_tol=0.001):
                owns_midpoint = True
            if not owns_midpoint:
                continue

            collected.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                }
            )

    finalized = finalize_segments(collected)
    recovery_windows = []
    candidate_gaps = []
    if finalized:
        candidate_gaps.append((0.0, float(finalized[0]["start"])))
        candidate_gaps.extend(
            (float(previous["end"]), float(current["start"]))
            for previous, current in zip(finalized, finalized[1:])
        )
        candidate_gaps.append((float(finalized[-1]["end"]), duration))

    for gap_start, gap_end in candidate_gaps:
        gap_duration = gap_end - gap_start
        if gap_duration < RECOVERY_GAP_SECONDS:
            continue

        gap_dbfs = _gap_dbfs(audio, sample_rate, gap_start, gap_end)
        if gap_dbfs <= SILENCE_DBFS:
            continue

        relative_windows = build_decode_windows(
            gap_duration,
            chunk_seconds=chunk_seconds,
            context_seconds=context_seconds,
        )
        for relative_core_start, relative_core_end, _, _ in relative_windows:
            core_start = gap_start + relative_core_start
            core_end = gap_start + relative_core_end
            core_duration = core_end - core_start
            available_context = max(
                0.0,
                (WHISPER_AUDIO_WINDOW_SECONDS - core_duration) / 2.0,
            )
            recovery_context = min(5.0, available_context)
            window_start = max(0.0, core_start - recovery_context)
            window_end = min(duration, core_end + recovery_context)
            recovery_windows.append(
                (core_start, core_end, window_start, window_end, gap_dbfs)
            )

    if recovery_windows:
        recovery_windows = recovery_windows[:MAX_RECOVERY_WINDOWS]
        print(
            f"🩹 Recovering {len(recovery_windows)} internal gap(s) "
            "that still contain audio...",
            flush=True,
        )
        recovery_kwargs = dict(transcribe_kwargs)
        recovery_kwargs["beam_size"] = max(5, int(beam_size))
        recovery_kwargs["temperature"] = 0.0

        recovered = []
        for recovery_number, recovery_window in enumerate(
            recovery_windows, start=1
        ):
            core_start, core_end, window_start, window_end, gap_dbfs = (
                recovery_window
            )
            print(
                f"🩹 Gap {recovery_number}/{len(recovery_windows)}: "
                f"{core_start:.2f}s → {core_end:.2f}s ({gap_dbfs:.1f} dBFS)",
                flush=True,
            )
            sample_start = max(0, int(round(window_start * sample_rate)))
            sample_end = min(len(audio), int(round(window_end * sample_rate)))
            audio_window = np.ascontiguousarray(audio[sample_start:sample_end])
            segment_iterator, _ = model.transcribe(
                audio_window,
                **recovery_kwargs,
            )

            for raw_segment in segment_iterator:
                raw_text = re.sub(r"\s+", " ", (raw_segment.text or "")).strip()
                if not raw_text:
                    continue
                text = clean_repeated_words(raw_text, max_consecutive=2)
                if not text:
                    continue
                if text != raw_text:
                    repetitions_cleaned += 1

                start = max(
                    0.0,
                    min(duration, window_start + float(raw_segment.start)),
                )
                end = max(
                    start,
                    min(duration, window_start + float(raw_segment.end)),
                )
                midpoint = start + ((end - start) / 2.0)
                if not core_start <= midpoint < core_end:
                    continue
                recovered.append(
                    {
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                    }
                )

        if recovered:
            print(
                f"✅ Recovered {len(recovered)} additional segment(s)",
                flush=True,
            )
            finalized = finalize_segments(finalized + recovered)

    return finalized, language_scores, repetitions_cleaned


def _gap_dbfs(audio: np.ndarray, sample_rate: int, start: float, end: float) -> float:
    sample_start = max(0, int(start * sample_rate))
    sample_end = min(len(audio), int(end * sample_rate))
    samples = audio[sample_start:sample_end]
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-12))


def report_timeline_gaps(
    segments: list[dict],
    audio: np.ndarray,
    sample_rate: int,
    duration: float,
    minimum_gap: float = GAP_REPORT_SECONDS,
) -> None:
    """Explain large transcript gaps without inventing subtitle content."""
    previous_end = 0.0
    gaps = []
    for segment in segments:
        start = float(segment["start"])
        if start - previous_end >= minimum_gap:
            gaps.append((previous_end, start))
        previous_end = max(previous_end, float(segment["end"]))
    if duration - previous_end >= minimum_gap:
        gaps.append((previous_end, duration))

    for start, end in gaps:
        dbfs = _gap_dbfs(audio, sample_rate, start, end)
        classification = (
            "silence/very-low audio" if dbfs <= SILENCE_DBFS else "audio present"
        )
        print(
            f"ℹ️ Transcript gap {start:.2f}s → {end:.2f}s "
            f"({end - start:.2f}s, {classification}, {dbfs:.1f} dBFS)",
            flush=True,
        )


def run_whisper_extraction(
    media_path: str,
    src_lang: str = "mr",
    beam_size: int = 5,
    cpu_threads: int | None = None,
    use_vad: bool = False,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_context_seconds: float = DEFAULT_CHUNK_CONTEXT_SECONDS,
):
    print("=" * 72, flush=True)
    print("🎙️ STAGE 1: OFFLINE SPEECH RECOGNITION", flush=True)
    print("=" * 72, flush=True)

    input_path = Path(media_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    print(f"📥 Input: {input_path.name}", flush=True)
    print("🎧 Extracting/normalizing audio to 16 kHz mono WAV...", flush=True)

    t0 = time.perf_counter()
    wav_path = Path(extract_and_normalize_audio(str(input_path)))
    print(
        f"✅ Audio ready in {time.perf_counter() - t0:.2f}s: {wav_path}",
        flush=True,
    )

    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio after normalization: {wav_path}")
    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio after normalization, got {sample_rate} Hz"
        )
    if audio.size == 0:
        raise ValueError(f"No audio samples were extracted from: {input_path}")
    duration = len(audio) / float(sample_rate)
    validate_chunk_settings(chunk_seconds, chunk_context_seconds)

    whisper_dir = VAULT_DIR / "whisper"
    model_bin = find_file(whisper_dir, "model.bin") or find_file(whisper_dir, "*.bin")
    if not model_bin:
        raise FileNotFoundError(
            "Whisper model.bin not found under local_model_vault/whisper"
        )

    model_dir = model_bin.parent
    threads = cpu_threads or min(8, max(1, os.cpu_count() or 4))

    print(f"🧠 Loading Faster-Whisper from: {model_dir}", flush=True)
    print(
        f"⚙️ CPU threads={threads}, compute_type=int8, beam_size={beam_size}",
        flush=True,
    )

    t0 = time.perf_counter()
    model = WhisperModel(
        str(model_dir),
        device="cpu",
        compute_type="int8",
        cpu_threads=threads,
    )
    print(
        f"✅ Whisper loaded in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    requested_lang = SUPPORTED_LOCALES.get(src_lang.lower())

    print(
        f"🗣️ Transcribing language={src_lang.upper()} "
        f"(beam={beam_size}, VAD={'ON' if use_vad else 'OFF'}, "
        f"chunks={chunk_seconds:g}s + {chunk_context_seconds:g}s context)...",
        flush=True,
    )
    print(
        f"⏱️ Audio duration: {duration:.2f}s ({duration / 60:.2f} min)",
        flush=True,
    )

    t_asr = time.perf_counter()

    extracted, language_scores, repeated_segments_cleaned = transcribe_in_chunks(
        model=model,
        audio=audio,
        sample_rate=sample_rate,
        requested_lang=requested_lang,
        beam_size=beam_size,
        use_vad=use_vad,
        chunk_seconds=chunk_seconds,
        context_seconds=chunk_context_seconds,
    )

    detected_code = (
        max(language_scores, key=language_scores.get)
        if language_scores
        else (requested_lang or src_lang or "unknown")
    ).lower()
    print(f"🌐 ASR language: {detected_code.upper()}", flush=True)

    for item in extracted:
        pct = (item["end"] / duration * 100.0) if duration else 0.0
        print(
            f"[{item['id']:03d}] {item['start']:7.2f}s → {item['end']:7.2f}s "
            f"({pct:5.1f}%) | {item['text']}",
            flush=True,
        )

    report_timeline_gaps(extracted, audio, sample_rate, duration)

    elapsed = time.perf_counter() - t_asr
    print(
        f"✅ ASR completed: {len(extracted)} segments in {elapsed:.2f}s",
        flush=True,
    )

    if repeated_segments_cleaned:
        print(
            f"🧹 Repetition cleanup applied to "
            f"{repeated_segments_cleaned} segment(s)",
            flush=True,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_json = OUTPUT_DIR / "step1_whisper_output.json"
    detected_json = OUTPUT_DIR / f"step1_whisper_output_{detected_code}.json"

    for path in (output_json, detected_json):
        path.write_text(
            json.dumps(extracted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"📄 Transcript: {output_json}", flush=True)
    print("=" * 72, flush=True)

    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline multilingual Faster-Whisper ASR"
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Input audio/video path",
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_flag",
        default=None,
        help="Input audio/video path (alternative form)",
    )
    parser.add_argument(
        "--lang",
        "-l",
        default="mr",
        choices=["auto", "mr", "hi", "en"],
        help="Source language",
    )
    parser.add_argument(
        "--beam_size",
        "-b",
        type=int,
        default=5,
        help="Whisper decoding beam size (quality default: 5)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU thread count (default: up to 8)",
    )
    parser.add_argument(
        "--use-vad",
        action="store_true",
        help=(
            "Enable permissive Silero VAD. "
            "Default is OFF to reduce the risk of spoken audio being cut."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help="Full-coverage ASR core chunk size (default: 28 seconds)",
    )
    parser.add_argument(
        "--chunk-context-seconds",
        type=float,
        default=DEFAULT_CHUNK_CONTEXT_SECONDS,
        help="Context added around each chunk (default: 1 second)",
    )

    args = parser.parse_args()
    target = args.input_flag or args.input_file

    if not target:
        parser.print_help()
        sys.exit(1)

    run_whisper_extraction(
        target,
        src_lang=args.lang,
        beam_size=max(1, args.beam_size),
        cpu_threads=args.threads,
        use_vad=args.use_vad,
        chunk_seconds=args.chunk_seconds,
        chunk_context_seconds=args.chunk_context_seconds,
    )
