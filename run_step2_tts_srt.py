import json
import re
import sys
import datetime
import srt
import soundfile as sf
import numpy as np
from pathlib import Path
import sherpa_onnx

# Setup base paths
BASE_DIR = Path(__file__).resolve().parent
VAULT_DIR = BASE_DIR / "local_model_vault"
DEFAULT_INPUT = BASE_DIR / "storage_vault" / "outputs" / "step2_offline_translated.json"
OUTPUT_DIR = BASE_DIR / "storage_vault" / "outputs"


def normalize_hindi_text(text: str) -> str:
    """Normalizes Hindi text, expands numbers to words, and strips problematic symbols."""
    if not text:
        return ""

    # 1. Strip SentencePiece meta-characters
    cleaned = text.replace("▁", " ").replace("\u2581", " ")

    # 2. Convert digits/numbers into Hindi words
    number_map = {
        "10,000": "दस हजार", "10000": "दस हजार",
        "12,000": "बारह हजार", "12000": "बारह हजार",
        "1700": "सत्रह सौ", "19वीं": "उन्नीसवीं",
        "20वीं": "बीसवीं", "19th": "उन्नीसवीं",
        "20th": "बीसवीं", "19": "उन्नीस", "20": "बीस",
    }
    for num_str, hindi_word in number_map.items():
        cleaned = cleaned.replace(num_str, hindi_word)

    # 3. Remove punctuation that breaks G2P phonemizer
    cleaned = re.sub(r"[।॥\"'\-_!?:;,()\[\]{}]", " ", cleaned)

    # Normalize whitespace
    return re.sub(r"\s+", " ", cleaned).strip()


def find_file(directory: Path, pattern: str) -> Path:
    """Recursively finds the first matching file in a directory."""
    if not directory.exists():
        return None
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None


def get_tts_engine(target_lang: str = "hi") -> sherpa_onnx.OfflineTts:
    """Loads Sherpa-ONNX VITS engine with dynamic espeak-ng G2P phonemization."""
    folder_name = "hindi" if target_lang == "hi" else "marathi"
    tts_dir = VAULT_DIR / "tts" / folder_name

    if not tts_dir.exists():
        raise FileNotFoundError(f"❌ TTS model directory does not exist: {tts_dir}")

    # 1. Discover ONNX model file
    onnx_file = find_file(tts_dir, "*.onnx")
    if not onnx_file:
        raise FileNotFoundError(f"❌ ONNX model file (*.onnx) not found in {tts_dir}")

    # 2. Discover tokens.txt
    tokens_file = find_file(tts_dir, "tokens.txt")
    if not tokens_file:
        raise FileNotFoundError(f"❌ tokens.txt not found in {tts_dir}")

    # 3. Discover espeak-ng-data directory
    espeak_matches = list(tts_dir.rglob("espeak-ng-data"))
    espeak_dir = espeak_matches[0] if espeak_matches else (tts_dir / "espeak-ng-data")

    if not espeak_dir.exists():
        raise FileNotFoundError(
            f"❌ Missing 'espeak-ng-data' directory in {tts_dir}.\n"
            "Please ensure espeak-ng-data is downloaded into the model folder."
        )

    print(f"⚡ Initializing Sherpa-ONNX TTS Engine ({target_lang.upper()})...")
    print(f"   ├─ Model File : {onnx_file.name}")
    print(f"   ├─ Tokens     : {tokens_file.name}")
    print(f"   ├─ Phonemizer : Dynamic espeak-ng G2P (Lexicon set to empty string)")
    print(f"   └─ Espeak Data: {espeak_dir.name}\n")

    # CRITICAL FIX: Set lexicon to "" so Sherpa-ONNX routes ALL words through espeak-ng
    vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(onnx_file),
        tokens=str(tokens_file),
        lexicon="",  # MUST BE EMPTY STRING FOR PIPER/ESPEAK MODELS
        data_dir=str(espeak_dir),
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=1.0
    )

    model_config = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits_config,
        num_threads=4,
        provider="cpu"
    )

    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model_config))


def generate_srt_and_audio(input_json_path: str, target_lang: str = "hi"):
    print("==================================================")
    print(f"🔊 STEP 2: SUBTITLE & AUDIO SYNTHESIS ({target_lang.upper()})")
    print("==================================================")

    input_path = Path(input_json_path).resolve()

    if not input_path.exists():
        print(f"❌ Input JSON file not found at: {input_path}")
        print("Usage: python3 run_step2_tts_srt.py <path_to_translated_json> [hi|mr]")
        sys.exit(1)

    print(f"📄 Loading translated JSON from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"📖 Loaded {len(segments)} translated segments.\n")

    lang_key = f"translation_{target_lang}"

    # 1. Create .srt Subtitles
    srt_subtitles = []
    for idx, seg in enumerate(segments, start=1):
        raw_text = (
            seg.get(lang_key) or 
            seg.get("translation_hi") or 
            seg.get("translation_mr") or 
            seg.get("translated_text") or 
            seg.get("text")
        )
        clean_text = normalize_hindi_text(raw_text)
        
        sub = srt.Subtitle(
            index=idx,
            start=datetime.timedelta(seconds=seg["start"]),
            end=datetime.timedelta(seconds=seg["end"]),
            content=clean_text
        )
        srt_subtitles.append(sub)

    srt_path = OUTPUT_DIR / f"video_subtitles_{target_lang}.srt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(srt_subtitles))
    print(f"📄 Generated .srt Subtitles: {srt_path.relative_to(BASE_DIR)}\n")

    # 2. Synthesize Spoken Audio Clips
    tts_engine = get_tts_engine(target_lang)
    audio_clips = []
    sample_rate = 22050

    print(f"🎙️ Synthesizing voice audio for {len(segments)} segments...")
    for idx, seg in enumerate(segments, start=1):
        raw_text = (
            seg.get(lang_key) or 
            seg.get("translation_hi") or 
            seg.get("translation_mr") or 
            seg.get("translated_text") or 
            seg.get("text")
        )
        clean_text = normalize_hindi_text(raw_text)

        if not clean_text:
            print(f" ⚠️ [Segment {idx:02d}/{len(segments):02d}] Empty text, skipping.")
            continue

        try:
            audio = tts_engine.generate(clean_text, sid=0, speed=1.0)
            samples = np.array(audio.samples, dtype=np.float32)

            if samples.size > 0:
                audio_clips.append(samples)
                sample_rate = audio.sample_rate
                print(f" ├─ [Segment {idx:02d}/{len(segments):02d}] \"{clean_text[:35]}...\"")
            else:
                print(f" ⚠️ [Segment {idx:02d}/{len(segments):02d}] Generated 0 audio samples.")
        except Exception as err:
            print(f" ❌ [Segment {idx:02d}/{len(segments):02d}] TTS Failed: {err}")

    # 3. Concatenate and Save Audio File
    if audio_clips:
        combined_audio = np.concatenate(audio_clips)
        if combined_audio.size > 0:
            wav_path = OUTPUT_DIR / f"video_dubbed_{target_lang}.wav"
            sf.write(str(wav_path), combined_audio, sample_rate)
            print(f"\n🔊 Generated Dubbed WAV Audio Track: {wav_path.relative_to(BASE_DIR)}")
        else:
            print("\n⚠️ No audio samples generated across all segments.")
    else:
        print("\n⚠️ No valid audio clips were synthesized.")

    print("==================================================")
    print("✅ Step 2 Complete! Subtitles and Audio Track ready.")
    print("==================================================")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_INPUT)
    target_language = sys.argv[2] if len(sys.argv) > 2 else "hi"

    generate_srt_and_audio(input_file, target_lang=target_language)