import sys
import json
import argparse
import time
from pathlib import Path

# Force project root into path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from faster_whisper import WhisperModel
from app.modules.ingestion import extract_and_normalize_audio

VAULT_DIR = BASE_DIR / "local_model_vault"

SUPPORTED_LOCALES = {
    "auto": None,     # Auto-detect
    "mr": "mr",       # Marathi
    "hi": "hi",       # Hindi
    "en": "en"        # English
}

# Domain-specific Devanagari prompts to lock vocabulary and script
INITIAL_PROMPTS = {
    "mr": "शेळीपालन, शेळीपालक, पशुसखी, व्यवस्थापन, शेती, चारा, आजार, लसीकरण, पैदास, आहार.",
    "hi": "बकरी पालन, पशुपालन, शेतकरी, चारा, स्वास्थ्य, प्रबंधन, टीकाकरण, आहार.",
    "en": "Goat rearing, livestock management, animal husbandry, fodder, health care."
}

def find_file(directory: Path, pattern: str):
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None

def run_whisper_extraction(video_or_audio_path: str, src_lang: str = "mr", beam_size: int = 5):
    print("==================================================")
    print("🎙️ STEP 1: ANTI-HALLUCINATION ASR TRANSCRIPTION")
    print("==================================================")
    
    # 1. Normalize input media ONCE
    print(f"\n1️⃣ Extracting 16kHz Mono PCM WAV from: {Path(video_or_audio_path).name}")
    wav_path = extract_and_normalize_audio(video_or_audio_path)
    print(f"   └─ Normalized WAV path: {wav_path}")

    # 2. Locate Whisper model weights
    whisper_bin = find_file(VAULT_DIR / "whisper", "*.bin")
    if not whisper_bin:
        raise FileNotFoundError("Could not locate Whisper .bin model weights in local_model_vault/whisper/")
    
    model_dir = whisper_bin.parent
    print(f"\n2️⃣ Loading Faster-Whisper model from: {model_dir.relative_to(BASE_DIR)}")
    t0 = time.time()
    model = WhisperModel(
        str(model_dir),
        device="cpu",
        compute_type="int8",
        cpu_threads=4
    )
    print(f"   └─ ✅ Model loaded in {time.time() - t0:.2f}s")

    # 3. Configure Locale & Anti-Hallucination Parameters
    target_locale = SUPPORTED_LOCALES.get(src_lang.lower(), "mr")
    prompt = INITIAL_PROMPTS.get(target_locale, "")

    print(f"\n3️⃣ Transcribing Speech (Locale: [{target_locale.upper()}], Beam: {beam_size})...")
    print(f"   └─ Devanagari Prompt Anchor: {prompt}")

    t_start = time.time()
    segments, info = model.transcribe(
        str(wav_path),
        language=target_locale,
        beam_size=beam_size,
        temperature=0.0,
        initial_prompt=prompt,
        condition_on_previous_text=False,  # CRITICAL: Stops hallucination propagation
        repetition_penalty=1.2,            # CRITICAL: Prevents token repetition loops
        compression_ratio_threshold=2.4,   # Discards compressed looping gibberish
        no_speech_threshold=0.6,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200
        )
    )
    
    total_duration = info.duration or 1.0
    detected_lang = info.language
    print(f"   └─ 🌐 Language: {detected_lang.upper()} (Confidence: {info.language_probability * 100:.1f}%)")
    print(f"   └─ ⏱️ Audio Duration: {total_duration:.2f}s (~{total_duration / 60:.1f} mins)\n")

    extracted_data = []
    print("----------------------------------------------------------------------")
    print(f" {'ID':<4} | {'TIMESTAMP':<15} | {'PROGRESS':<8} | {'TRANSCRIPTION TEXT'}")
    print("----------------------------------------------------------------------")
    
    # 4. Stream segments in real-time
    for idx, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        # Filter out empty or broken tokens
        if not text or len(set(text.replace(" ", ""))) <= 2 and len(text) > 10:
            continue

        pct = min(100.0, (seg.end / total_duration) * 100)
        time_str = f"{seg.start:05.1f}s -> {seg.end:05.1f}s"
        
        segment_obj = {
            "id": idx,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text
        }
        extracted_data.append(segment_obj)
        
        # Real-time console streaming
        print(f" #{idx:<3} | {time_str:<15} | {pct:>5.1f}%   | {text}")

    total_time = time.time() - t_start
    print("----------------------------------------------------------------------")
    print(f"✅ Successfully extracted {len(extracted_data)} clean segments in {total_time:.2f}s!")
    
    # 5. Save outputs
    output_dir = BASE_DIR / "storage_vault" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_json = output_dir / "step1_whisper_output.json"
    tagged_json = output_dir / f"step1_whisper_output_{detected_lang}.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
    with open(tagged_json, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
    print(f"📄 Saved transcript JSON to: {output_json.relative_to(BASE_DIR)}")
    print("==================================================")
    
    return output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Robust Multilingual ASR")
    parser.add_argument("input_file", nargs="?", default=None, help="Input video/audio file (positional)")
    parser.add_argument("--input", "-i", dest="input_flag", type=str, default=None, help="Input video/audio file (--input)")
    parser.add_argument("--lang", "-l", type=str, default="mr", choices=["auto", "mr", "hi", "en"], help="Source locale (default: mr)")
    parser.add_argument("--beam_size", "-b", type=int, default=5, help="Beam size (default: 5)")

    args = parser.parse_args()
    target_file = args.input_flag or args.input_file
    
    if not target_file:
        parser.print_help()
        sys.exit(1)
        
    run_whisper_extraction(target_file, src_lang=args.lang, beam_size=args.beam_size)