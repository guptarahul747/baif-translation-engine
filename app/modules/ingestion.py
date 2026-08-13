import sys
from pathlib import Path

# Force project root directory into Python search path for clean imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import os
import platform
import subprocess
import soundfile as sf
from typing import Dict, Any

INPUTS_DIR = BASE_DIR / "storage_vault" / "inputs"


def get_ffmpeg_binary() -> str:
    """Returns system-appropriate FFmpeg command trigger."""
    return "ffmpeg"


def extract_and_normalize_audio(input_media_path: str, output_wav_filename: str = None) -> str:
    """
    Extracts audio from video/audio input and converts it to 16kHz Mono PCM WAV format.

    Args:
        input_media_path: Path to raw uploaded video or audio file.
        output_wav_filename: Optional custom filename for extracted WAV.

    Returns:
        str: Absolute path to normalized 16kHz mono WAV file on disk.
    """
    input_path = Path(input_media_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input media file not found at: {input_media_path}")

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_wav_filename:
        output_path = INPUTS_DIR / output_wav_filename
    else:
        output_path = INPUTS_DIR / f"{input_path.stem}_normalized.wav"

    ffmpeg_cmd = get_ffmpeg_binary()

    # FFmpeg parameters for Faster-Whisper compatibility:
    # -y                  : Overwrite output file if exists
    # -i input_path       : Input media path
    # -vn                 : Disable video (extract audio only)
    # -ac 1               : Force mono channel
    # -ar 16000           : Force 16000 Hz sample rate
    # -c:a pcm_s16le      : Encode as 16-bit PCM WAV
    cmd = [
        ffmpeg_cmd,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]

    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{stderr_output}") from e

    return str(output_path)


def inspect_audio_metadata(wav_path: str) -> Dict[str, Any]:
    """
    Reads sample rate, channel count, and duration from a normalized WAV file.
    
    Returns:
        Dict containing duration (seconds), sample_rate, channels, and total frames.
    """
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found at: {wav_path}")

    info = sf.info(str(path))

    return {
        "file_name": path.name,
        "duration_seconds": round(info.duration, 2),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "format": info.format_str
    }


if __name__ == "__main__":
    print("==================================================")
    print("🎧 TESTING INGESTION MODULE...")
    print("==================================================")

    try:
        ffmpeg_bin = get_ffmpeg_binary()
        res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True)
        print("✅ FFmpeg Binary Detected:\n", res.stdout.split("\n")[0])
        print("✅ Ingestion module ready for media stream processing.")
    except Exception as err:
        print("❌ FFmpeg execution check failed:", err)
    print("==================================================")