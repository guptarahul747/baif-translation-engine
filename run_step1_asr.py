import argparse
import json
import os
import sys
import time
from pathlib import Path

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

INITIAL_PROMPTS = {
    "mr": "शेळीपालन, शेळीपालक, पशुसखी, व्यवस्थापन, शेती, चारा, आजार, लसीकरण, पैदास, आहार.",
    "hi": "बकरी पालन, पशुपालन, किसान, चारा, स्वास्थ्य, प्रबंधन, टीकाकरण, आहार.",
    "en": "Goat rearing, livestock management, animal husbandry, fodder, health care, agriculture.",
}


def find_file(directory: Path, pattern: str):
    if not directory.exists():
        return None
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None


def run_whisper_extraction(
    media_path: str,
    src_lang: str = "mr",
    beam_size: int = 5,
    cpu_threads: int | None = None,
):
    print("=" * 64, flush=True)
    print("🎙️ STAGE 1: OFFLINE SPEECH RECOGNITION", flush=True)
    print("=" * 64, flush=True)

    input_path = Path(media_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    print(f"📥 Input: {input_path.name}", flush=True)
    print("🎧 Extracting/normalizing audio to 16 kHz mono WAV...", flush=True)
    t0 = time.perf_counter()
    wav_path = Path(extract_and_normalize_audio(str(input_path)))
    print(f"✅ Audio ready in {time.perf_counter() - t0:.2f}s: {wav_path}", flush=True)

    whisper_dir = VAULT_DIR / "whisper"
    model_bin = find_file(whisper_dir, "model.bin") or find_file(whisper_dir, "*.bin")
    if not model_bin:
        raise FileNotFoundError("Whisper model.bin not found under local_model_vault/whisper")

    model_dir = model_bin.parent
    threads = cpu_threads or min(8, max(1, os.cpu_count() or 4))

    print(f"🧠 Loading Faster-Whisper from: {model_dir}", flush=True)
    print(f"⚙️ CPU threads={threads}, compute_type=int8, beam_size={beam_size}", flush=True)
    t0 = time.perf_counter()
    model = WhisperModel(
        str(model_dir),
        device="cpu",
        compute_type="int8",
        cpu_threads=threads,
    )
    print(f"✅ Whisper loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    requested_lang = SUPPORTED_LOCALES.get(src_lang.lower())
    prompt = INITIAL_PROMPTS.get(src_lang.lower(), "") if requested_lang else ""

    print(
        f"🗣️ Transcribing language={src_lang.upper()} with VAD enabled...",
        flush=True,
    )
    t_asr = time.perf_counter()

    segments_iter, info = model.transcribe(
        str(wav_path),
        language=requested_lang,
        task="transcribe",
        beam_size=beam_size,
        temperature=0.0,
        initial_prompt=prompt or None,
        condition_on_previous_text=False,
        repetition_penalty=1.2,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
    )

    duration = float(info.duration or 0.0)
    print(
        f"🌐 Detected: {info.language.upper()} "
        f"({info.language_probability * 100:.1f}% confidence)",
        flush=True,
    )
    if duration:
        print(f"⏱️ Media speech duration: {duration:.2f}s ({duration / 60:.2f} min)", flush=True)

    extracted = []
    accepted_id = 0

    for raw_index, seg in enumerate(segments_iter, start=1):
        text = (seg.text or "").strip()
        if not text:
            continue

        compact = text.replace(" ", "")
        if len(compact) > 10 and len(set(compact)) <= 2:
            print(f"⚠️ Skipping suspicious repeated segment #{raw_index}", flush=True)
            continue

        accepted_id += 1
        item = {
            "id": accepted_id,
            "start": round(float(seg.start), 2),
            "end": round(float(seg.end), 2),
            "text": text,
        }
        extracted.append(item)

        pct = (float(seg.end) / duration * 100.0) if duration else 0.0
        print(
            f"[{accepted_id:03d}] {seg.start:7.2f}s → {seg.end:7.2f}s "
            f"({pct:5.1f}%) | {text}",
            flush=True,
        )

    elapsed = time.perf_counter() - t_asr
    print(f"✅ ASR completed: {len(extracted)} segments in {elapsed:.2f}s", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_json = OUTPUT_DIR / "step1_whisper_output.json"
    detected_json = OUTPUT_DIR / f"step1_whisper_output_{info.language}.json"

    for path in (output_json, detected_json):
        path.write_text(
            json.dumps(extracted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"📄 Transcript: {output_json}", flush=True)
    print("=" * 64, flush=True)
    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline multilingual Faster-Whisper ASR")
    parser.add_argument("input_file", nargs="?", default=None)
    parser.add_argument("--input", "-i", dest="input_flag", default=None)
    parser.add_argument(
        "--lang",
        "-l",
        default="mr",
        choices=["auto", "mr", "hi", "en"],
    )
    parser.add_argument("--beam_size", "-b", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None)
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
    )
