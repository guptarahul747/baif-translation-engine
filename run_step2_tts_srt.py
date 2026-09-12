import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf
import srt

BASE_DIR = Path(__file__).resolve().parent
VAULT_DIR = BASE_DIR / "local_model_vault"
DEFAULT_INPUT = BASE_DIR / "storage_vault" / "outputs" / "step2_offline_translated.json"
OUTPUT_DIR = BASE_DIR / "storage_vault" / "outputs"

TTS_FOLDERS = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
}

TTS_TIMING_MODES = {"continuous", "compact", "source"}
COMPACT_LEADING_PAUSE_SECONDS = 0.5
COMPACT_BASE_PAUSE_SECONDS = 0.18
COMPACT_MAX_PAUSE_SECONDS = 1.25


def normalize_tts_text(text: str, target_lang: str) -> str:
    if not text:
        return ""

    cleaned = text.replace("▁", " ").replace("\u2581", " ")

    if target_lang == "hi":
        number_map = {
            "10,000": "दस हजार",
            "10000": "दस हजार",
            "12,000": "बारह हजार",
            "12000": "बारह हजार",
            "1700": "सत्रह सौ",
            "19वीं": "उन्नीसवीं",
            "20वीं": "बीसवीं",
            "19th": "उन्नीसवीं",
            "20th": "बीसवीं",
        }
        for source, replacement in number_map.items():
            cleaned = cleaned.replace(source, replacement)

    cleaned = re.sub(r"[\"'_\[\]{}]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def find_file(directory: Path, pattern: str):
    if not directory.exists():
        return None
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None


def get_tts_engine(target_lang: str) -> sherpa_onnx.OfflineTts:
    if target_lang not in TTS_FOLDERS:
        raise ValueError("TTS target must be one of: en, hi, mr")

    tts_dir = VAULT_DIR / "tts" / TTS_FOLDERS[target_lang]
    onnx_file = find_file(tts_dir, "*.onnx")
    tokens_file = find_file(tts_dir, "tokens.txt")
    espeak_dirs = list(tts_dir.rglob("espeak-ng-data"))
    espeak_dir = espeak_dirs[0] if espeak_dirs else tts_dir / "espeak-ng-data"

    if not onnx_file:
        raise FileNotFoundError(f"ONNX TTS model not found in {tts_dir}")
    if not tokens_file:
        raise FileNotFoundError(f"tokens.txt not found in {tts_dir}")
    if not espeak_dir.exists():
        raise FileNotFoundError(f"espeak-ng-data not found in {tts_dir}")

    print(f"⏳ Loading {target_lang.upper()} Sherpa-ONNX TTS...", flush=True)
    t0 = time.perf_counter()

    vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(onnx_file),
        tokens=str(tokens_file),
        lexicon="",
        data_dir=str(espeak_dir),
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=1.0,
    )
    model_config = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits_config,
        num_threads=min(8, max(1, os.cpu_count() or 4)),
        provider="cpu",
    )
    engine = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(model=model_config)
    )

    print(f"✅ TTS loaded in {time.perf_counter() - t0:.2f}s", flush=True)
    return engine


def build_tts_timeline(
    clips: list[dict],
    sample_rate: int,
    timing_mode: str = "compact",
) -> tuple[np.ndarray, float]:
    """
    Assemble generated clips without repeatedly copying the whole waveform.

    ``continuous`` is the downloadable audio: generated clips are joined with
    no artificial silence. ``compact`` keeps a short natural pause plus a
    capped source-scene gap.
    ``source`` retains the old absolute timestamp alignment for users who need
    the dubbed track to follow the original video timing exactly.
    """
    if timing_mode not in TTS_TIMING_MODES:
        raise ValueError("TTS timing mode must be one of: continuous, compact, source")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    parts = []
    timeline_samples = 0
    inserted_silence_samples = 0
    previous_source_end = None

    for clip in clips:
        samples = np.asarray(clip["samples"], dtype=np.float32)
        if samples.size == 0:
            continue

        source_start = max(0.0, float(clip.get("start", 0.0)))
        source_end = max(source_start, float(clip.get("end", source_start)))

        if timing_mode == "source":
            desired_start = int(round(source_start * sample_rate))
            pause_samples = max(0, desired_start - timeline_samples)
        elif timing_mode == "continuous":
            pause_samples = 0
        elif previous_source_end is None:
            pause_seconds = min(source_start, COMPACT_LEADING_PAUSE_SECONDS)
            pause_samples = int(round(pause_seconds * sample_rate))
        else:
            source_gap = max(0.0, source_start - previous_source_end)
            pause_seconds = min(
                COMPACT_MAX_PAUSE_SECONDS,
                COMPACT_BASE_PAUSE_SECONDS + source_gap,
            )
            pause_samples = int(round(pause_seconds * sample_rate))

        if pause_samples:
            parts.append(np.zeros(pause_samples, dtype=np.float32))
            timeline_samples += pause_samples
            inserted_silence_samples += pause_samples

        parts.append(samples)
        timeline_samples += samples.size
        previous_source_end = source_end

    if not parts:
        return np.array([], dtype=np.float32), 0.0

    return (
        np.concatenate(parts),
        inserted_silence_samples / float(sample_rate),
    )


def generate_srt_and_audio(
    input_json_path: str,
    target_lang: str = "hi",
    timing_mode: str = "continuous",
):
    target_lang = target_lang.lower()
    if target_lang not in TTS_FOLDERS:
        raise ValueError("Supported target languages: en, hi, mr")
    if timing_mode not in TTS_TIMING_MODES:
        raise ValueError("TTS timing mode must be one of: continuous, compact, source")

    input_path = Path(input_json_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Translated JSON not found: {input_path}")

    segments = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(segments, list):
        raise ValueError("Translated JSON must contain a list")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lang_key = f"translation_{target_lang}"

    print("=" * 64, flush=True)
    print(f"🎤 STAGE 3: TTS + SRT ({target_lang.upper()})", flush=True)
    print("=" * 64, flush=True)
    print(f"Segments: {len(segments)}", flush=True)
    print(f"Audio pacing: {timing_mode}", flush=True)

    subtitles = []
    for index, segment in enumerate(segments, start=1):
        raw_text = (
            segment.get(lang_key)
            or segment.get("translated_text")
            or segment.get("text")
            or ""
        )
        subtitle_text = re.sub(r"\s+", " ", raw_text).strip()
        subtitles.append(
            srt.Subtitle(
                index=index,
                start=datetime.timedelta(seconds=float(segment.get("start", 0.0))),
                end=datetime.timedelta(seconds=float(segment.get("end", segment.get("start", 0.0) + 1.0))),
                content=subtitle_text,
            )
        )

    srt_path = OUTPUT_DIR / f"video_subtitles_{target_lang}.srt"
    srt_path.write_text(srt.compose(subtitles), encoding="utf-8")
    print(f"📄 SRT generated: {srt_path}", flush=True)

    engine = get_tts_engine(target_lang)
    sample_rate = None
    generated_clips = []

    t0 = time.perf_counter()
    for index, segment in enumerate(segments, start=1):
        raw_text = (
            segment.get(lang_key)
            or segment.get("translated_text")
            or segment.get("text")
            or ""
        )
        clean_text = normalize_tts_text(raw_text, target_lang)
        if not clean_text:
            continue

        print(
            f"🎙️ [{index}/{len(segments)}] {clean_text[:80]}",
            flush=True,
        )
        audio = engine.generate(clean_text, sid=0, speed=1.0)
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            print(f"⚠️ Segment {index} produced no audio", flush=True)
            continue

        if sample_rate is None:
            sample_rate = int(audio.sample_rate)

        generated_clips.append(
            {
                "samples": samples,
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", segment.get("start", 0.0))),
            }
        )

    if sample_rate is None:
        raise RuntimeError("TTS generated no audio samples")

    # The downloadable track contains no inserted pauses. Build the video
    # track from the same generated clips, aligned to original timestamps, so
    # it follows the source video without re-running speech synthesis.
    timeline, inserted_silence = build_tts_timeline(
        generated_clips,
        sample_rate,
        timing_mode=timing_mode,
    )
    video_timeline, video_inserted_silence = build_tts_timeline(
        generated_clips,
        sample_rate,
        timing_mode="source",
    )

    if timeline.size == 0:
        raise RuntimeError("TTS generated no audio samples")

    wav_path = OUTPUT_DIR / f"video_dubbed_{target_lang}.wav"
    video_wav_path = OUTPUT_DIR / f"video_dubbed_{target_lang}_video_timed.wav"
    sf.write(str(wav_path), timeline, sample_rate)
    sf.write(str(video_wav_path), video_timeline, sample_rate)

    print(f"✅ TTS completed in {time.perf_counter() - t0:.2f}s", flush=True)
    print(f"⏸️ Inserted pause time: {inserted_silence:.2f}s", flush=True)
    print(f"🔊 WAV generated: {wav_path}", flush=True)
    print(f"🎬 Video-timed WAV generated: {video_wav_path}", flush=True)
    print(f"⏸️ Video pause time: {video_inserted_silence:.2f}s", flush=True)
    print("=" * 64, flush=True)
    return srt_path, wav_path


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_INPUT)
    target_language = sys.argv[2] if len(sys.argv) > 2 else "hi"
    audio_timing = sys.argv[3] if len(sys.argv) > 3 else "continuous"
    generate_srt_and_audio(
        input_file,
        target_lang=target_language,
        timing_mode=audio_timing,
    )
