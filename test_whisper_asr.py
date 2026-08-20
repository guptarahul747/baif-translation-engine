import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from faster_whisper import WhisperModel
from app.modules.ingestion import extract_and_normalize_audio

VAULT_DIR = BASE_DIR / "local_model_vault"

LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "mr": "mr",
    "auto": None
}

def find_file(directory: Path, pattern: str):
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None

def run_whisper_extraction(video_or_audio_path: str, src_lang: str = "en"):
    print("==================================================")
    print(f"🎙️ STEP 1: ASR TRANSCRIPTION ({src_lang.upper()})")
    print("==================================================")
    
    wav_path = extract_and_normalize_audio(video_or_audio_path)
    whisper_bin = find_file(VAULT_DIR / "whisper", "*.bin")
    if not whisper_bin:
        raise FileNotFoundError("Could not locate Whisper model in local_model_vault/whisper/")
    
    model_dir = whisper_bin.parent
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=4)

    target_lang = LANG_MAP.get(src_lang.lower(), None)
    
    print(f"Running ASR with target language: {target_lang}")
    segments, info = model.transcribe(
        str(wav_path), 
        beam_size=5, 
        language=target_lang,
        condition_on_previous_text=False
    )
    
    extracted_data = []
    for idx, seg in enumerate(segments, start=1):
        segment_obj = {
            "id": idx,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        }
        extracted_data.append(segment_obj)
        print(f"[{segment_obj['start']:>6.2f}s -> {segment_obj['end']:>6.2f}s] {segment_obj['text']}")

    output_json = BASE_DIR / "storage_vault" / "outputs" / "step1_whisper_output.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
    return output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--lang", "-l", type=str, default="en")
    args = parser.parse_args()

    run_whisper_extraction(args.input_file, src_lang=args.lang)