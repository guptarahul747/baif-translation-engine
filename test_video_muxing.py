import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "storage_vault" / "outputs"


def check_ffmpeg():
    """Verifies FFmpeg is available in system PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def merge_video_and_audio(
    original_video_path: str,
    target_lang: str = "hi",
    burn_subtitles: bool = False
):
    print("==================================================")
    print("🎬 STEP 3: VIDEO & TRANSLATED AUDIO MUXING")
    print("==================================================")

    if not check_ffmpeg():
        print("❌ FFmpeg is not installed or not available in system PATH.")
        sys.exit(1)

    video_input = Path(original_video_path).resolve()
    audio_input = OUTPUT_DIR / f"video_dubbed_{target_lang}.wav"
    srt_input = OUTPUT_DIR / f"video_subtitles_{target_lang}.srt"
    
    output_video = OUTPUT_DIR / f"final_translated_video_{target_lang}.mp4"

    # Input validation
    if not video_input.exists():
        print(f"❌ Original input video not found: {video_input}")
        sys.exit(1)

    if not audio_input.exists():
        print(f"❌ Dubbed audio track not found: {audio_input.relative_to(BASE_DIR)}")
        print("Please run Step 2 (run_step2_tts_srt.py) first.")
        sys.exit(1)

    print(f"📹 Original Video : {video_input.name}")
    print(f"🔊 Translated Audio: {audio_input.relative_to(BASE_DIR)}")
    if srt_input.exists():
        print(f"📄 Subtitles Track : {srt_input.relative_to(BASE_DIR)}")

    # Construct FFmpeg command
    if burn_subtitles and srt_input.exists():
        print("\n🔥 Burning subtitles directly onto the video stream...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_input),
            "-i", str(audio_input),
            "-vf", f"subtitles={srt_input}",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_video)
        ]
    else:
        print("\n⚡ Remuxing video stream with new audio track (Ultra-fast copy)...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_input),
            "-i", str(audio_input),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",          # Direct stream copy (no re-encoding)
            "-c:a", "aac",           # Encode audio to AAC standard
            "-b:a", "192k",
            "-shortest",
            str(output_video)
        ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"\n✅ Fully Dubbed Video Created Successfully!")
        print(f"🎬 Saved to: {output_video.relative_to(BASE_DIR)}")
        print("==================================================")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FFmpeg error occurred:")
        print(e.stderr.decode("utf-8"))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_video_muxing.py <path_to_original_video> [lang] [--burn-subs]")
        print("Example: python3 test_video_muxing.py my_video.mp4 hi --burn-subs")
        sys.exit(1)

    input_video_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "hi"
    burn_subs = "--burn-subs" in sys.argv

    merge_video_and_audio(input_video_file, target_lang=lang, burn_subtitles=burn_subs)