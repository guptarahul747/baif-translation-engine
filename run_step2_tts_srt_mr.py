import json
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Dict, List

# ==========================================
# Language Vault Mapping
# ==========================================
MODEL_MAP = {
    "mr": Path("local_model_vault/tts/marathi/mr_IN-google-medium.onnx")
}


def get_piper_executable() -> str:
    """Detects system 'piper' command or local binary in ./piper/piper."""
    if shutil.which("piper"):
        return "piper"
    
    local_bin = Path("./piper/piper").resolve()
    if local_bin.exists():
        return str(local_bin)
        
    return "piper"


def format_srt_time(seconds: float) -> str:
    """Converts seconds float (e.g. 14.68) to SRT format 'HH:MM:SS,mmm'."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
        secs += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_srt_file(data: List[Dict], output_srt_path: Path):
    """Converts parsed JSON segments into a standard .srt subtitle file."""
    srt_blocks = []
    for idx, item in enumerate(data, start=1):
        start_str = format_srt_time(item.get("start", 0.0))
        end_str = format_srt_time(item.get("end", 0.0))
        text = item["text"]

        srt_block = f"{idx}\n{start_str} --> {end_str}\n{text}\n"
        srt_blocks.append(srt_block)

    output_srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    print(f"📄 Generated SRT file: {output_srt_path}")


def load_translated_json(json_path: Path, lang_code: str) -> List[Dict]:
    """Loads translated items from the JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("segments") or data.get("subtitles") or data.get("data") or []

    items = []
    for idx, entry in enumerate(data, start=1):
        if isinstance(entry, dict):
            text = (
                entry.get(f"translation_{lang_code}")
                or entry.get("translation")
                or entry.get("translated_text")
                or entry.get("target_text")
                or entry.get(f"{lang_code}_text")
                or entry.get("text")
                or ""
            )
            item_id = entry.get("id") or entry.get("index") or idx
            start = entry.get("start", 0.0)
            end = entry.get("end", 0.0)
        else:
            text = str(entry)
            item_id = idx
            start = 0.0
            end = 0.0

        if text.strip():
            items.append({
                "id": str(item_id),
                "text": text.strip(),
                "start": start,
                "end": end
            })

    return items


def generate_segment_tts(piper_bin: str, model_path: Path, text: str, output_wav: Path) -> bool:
    """Invokes Piper TTS for a single text segment."""
    command = [
        piper_bin,
        "--model", str(model_path),
        "--output_file", str(output_wav)
    ]
    
    try:
        subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error synthesizing segment: {e.stderr}")
        return False


def merge_wav_files(wav_list: List[Path], output_path: Path):
    """Combines individual audio segment files into a single WAV file."""
    if not wav_list:
        print("⚠️ No audio segments found to merge.")
        return

    data = []
    params = None

    for wav_file in wav_list:
        if not wav_file.exists():
            continue
        with wave.open(str(wav_file), "rb") as w:
            if params is None:
                params = w.getparams()
            data.append(w.readframes(w.getnframes()))

    if data and params:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as output_w:
            output_w.setparams(params)
            for frames in data:
                output_w.writeframes(frames)
        print(f"🎬 Successfully merged {len(wav_list)} segments into: {output_path}")


def main():
    json_path_arg = sys.argv[1] if len(sys.argv) > 1 else "storage_vault/outputs/step2_offline_translated.json"
    lang_code = sys.argv[2] if len(sys.argv) > 2 else "mr"

    input_json = Path(json_path_arg).resolve()

    print("==========================================")
    print("🚀 Step 2: Offline TTS Audio & SRT Generation")
    print(f"📄 Input JSON : {input_json}")
    print(f"🌐 Language   : {lang_code}")
    print("==========================================")

    model_path = MODEL_MAP.get(lang_code)
    if not model_path or not model_path.exists():
        print(f"❌ Error: Model for language '{lang_code}' not found at '{model_path}'.")
        sys.exit(1)

    piper_bin = get_piper_executable()
    output_dir = input_json.parent / f"tts_segments_{lang_code}"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_wav_path = input_json.parent / f"video_dubbed_{lang_code}.wav"
    final_srt_path = input_json.parent / f"video_subtitles_{lang_code}.srt"

    try:
        segments = load_translated_json(input_json, lang_code)
        print(f" Found {len(segments)} translated entries to process.\n")
    except Exception as e:
        print(f"❌ Failed to parse JSON file: {e}")
        sys.exit(1)

    # 1. Export .srt Subtitle File
    export_srt_file(segments, final_srt_path)

    # 2. Synthesize TTS Segments
    generated_wavs = []
    for item in segments:
        seg_id = item["id"]
        text = item["text"]
        segment_wav = output_dir / f"segment_{int(seg_id):04d}.wav"

        print(f"🎙️ [{seg_id}/{len(segments)}] Synthesizing Marathi: \"{text}\"")
        success = generate_segment_tts(piper_bin, model_path, text, segment_wav)
        if success:
            generated_wavs.append(segment_wav)

    # 3. Merge Audio Segments
    merge_wav_files(generated_wavs, final_wav_path)

    # 4. Clean up temp folder
    print("🧹 Cleaning up temporary audio segments...")
    shutil.rmtree(output_dir)

    print("\n✅ Step 2 Completed!")


if __name__ == "__main__":
    main()
