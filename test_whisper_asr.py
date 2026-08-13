import sys
import json
from pathlib import Path

# Force project root into path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from faster_whisper import WhisperModel
from app.modules.ingestion import extract_and_normalize_audio

VAULT_DIR = BASE_DIR / "local_model_vault"

def find_file(directory: Path, pattern: str):
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None

def run_whisper_extraction(video_or_audio_path: str):
    print("==================================================")
    print("🎙️ STEP 1: WHISPER AUDIO-TO-TEXT EXTRACTION TEST")
    print("==================================================")
    
    # 1. Normalize input media to 16kHz Mono WAV using FFmpeg
    print(f"\n1️⃣ Extracting 16kHz Mono PCM WAV from: {video_or_audio_path}")
    wav_path = extract_and_normalize_audio(video_or_audio_path)
    print(f"   └─ Extracted WAV path: {wav_path}")

    # 2. Locate Whisper model weights
    whisper_bin = find_file(VAULT_DIR / "whisper", "*.bin")
    if not whisper_bin:
        raise FileNotFoundError("Could not locate Whisper .bin model weights in local_model_vault/whisper/")
    
    model_dir = whisper_bin.parent
    print(f"\n2️⃣ Loading Faster-Whisper model from: {model_dir.relative_to(BASE_DIR)}")
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=4)

    # 3. Transcribe speech to timestamped segments
    print("\n3️⃣ Running Automatic Speech Recognition (ASR)...")
    segments, info = model.transcribe(wav_path, beam_size=5, language="en")
    
    print(f"   └─ Detected Language: {info.language} (Probability: {info.language_probability:.2f})")
    print(f"   └─ Audio Duration: {round(info.duration, 2)} seconds\n")

    extracted_data = []
    print("--------------------------------------------------")
    print("RESULTS: Extracted Text Segments")
    print("--------------------------------------------------")
    
    for idx, seg in enumerate(segments, start=1):
        segment_obj = {
            "id": idx,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        }
        extracted_data.append(segment_obj)
        print(f"[{segment_obj['start']:>6.2f}s -> {segment_obj['end']:>6.2f}s] {segment_obj['text']}")

    print("--------------------------------------------------")
    print(f"✅ Successfully extracted {len(extracted_data)} segments!")
    
    # Save output to JSON for inspection
    output_json = BASE_DIR / "storage_vault" / "outputs" / "step1_whisper_output.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2)
        
    print(f"📄 Saved transcript JSON to: {output_json.relative_to(BASE_DIR)}")
    print("==================================================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_whisper_asr.py <path_to_video_or_audio_file>")
        print("Example: python test_whisper_asr.py my_video.mp4")
        sys.exit(1)

    input_file = sys.argv[1]
    run_whisper_extraction(input_file)