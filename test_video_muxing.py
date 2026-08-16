import sys
import subprocess
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "storage_vault" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FFMPEG
# ============================================================

def check_ffmpeg():
    """Verify FFmpeg is available in PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print("✅ FFmpeg detected:")
        print(result.stdout.splitlines()[0])
        return True

    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ FFmpeg is not installed or not available in PATH.")
        return False


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_input_video(original_video_path: str) -> Path:
    """
    Resolve the original video path.

    Supports:
      python test_video_muxing.py English.mp4
      python test_video_muxing.py storage_vault/inputs/English.mp4
      python test_video_muxing.py /absolute/path/English.mp4
    """

    path = Path(original_video_path)

    if path.is_absolute() and path.exists():
        return path.resolve()

    # Current directory
    candidate = BASE_DIR / path
    if candidate.exists():
        return candidate.resolve()

    # storage_vault/inputs
    candidate = BASE_DIR / "storage_vault" / "inputs" / path.name
    if candidate.exists():
        return candidate.resolve()

    # app/storage_vault/inputs, if applicable
    candidate = BASE_DIR / "app" / "storage_vault" / "inputs" / path.name
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Original video not found.\n"
        f"Requested: {original_video_path}"
    )


# ============================================================
# MEDIA INFORMATION
# ============================================================

def get_video_duration(video_path: Path) -> float:
    """Get video duration using ffprobe."""

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        return float(result.stdout.strip())

    except Exception as e:
        print(f"⚠️ Could not determine video duration: {e}")
        return 0.0


# ============================================================
# MUXING
# ============================================================

def merge_video_and_audio(
    original_video_path: str,
    target_lang: str = "hi",
    burn_subtitles: bool = False
):
    """
    Create final translated MP4.

    Input:
        Original video
        Translated WAV
        Optional SRT

    Output:
        storage_vault/outputs/final_translated_video_<lang>.mp4
    """

    print("==================================================")
    print("🎬 STEP 3: VIDEO & TRANSLATED AUDIO MUXING")
    print("==================================================")

    # --------------------------------------------------------
    # Check FFmpeg
    # --------------------------------------------------------

    if not check_ffmpeg():
        sys.exit(1)

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    try:
        video_input = resolve_input_video(original_video_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    audio_input = OUTPUT_DIR / f"video_dubbed_{target_lang}.wav"
    srt_input = OUTPUT_DIR / f"video_subtitles_{target_lang}.srt"

    # Final browser-friendly MP4
    output_video = OUTPUT_DIR / f"final_translated_video_{target_lang}.mp4"

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not video_input.exists():
        print(f"❌ Original video not found:")
        print(f"   {video_input}")
        sys.exit(1)

    if not audio_input.exists():
        print(f"❌ Dubbed audio not found:")
        print(f"   {audio_input}")
        print()
        print("Run Step 2 first.")
        sys.exit(1)

    print()
    print(f"📹 Original Video:")
    print(f"   {video_input}")

    print()
    print(f"🔊 Translated Audio:")
    print(f"   {audio_input}")

    if srt_input.exists():
        print()
        print(f"📄 Subtitle File:")
        print(f"   {srt_input}")
    else:
        print()
        print("ℹ️ No SRT file found.")

    # --------------------------------------------------------
    # Get video duration
    # --------------------------------------------------------

    video_duration = get_video_duration(video_input)

    if video_duration > 0:
        print()
        print(f"⏱️ Original video duration: {video_duration:.2f} seconds")

    # --------------------------------------------------------
    # Build FFmpeg command
    # --------------------------------------------------------

    if burn_subtitles and srt_input.exists():

        print()
        print("🔥 Burning subtitles into video...")
        print("🎞️ Re-encoding video as H.264")
        print("🔊 Encoding audio as AAC")

        # Escape subtitle path for FFmpeg filter.
        # macOS absolute paths need special handling.
        subtitle_path = str(srt_input.resolve())

        subtitle_path = (
            subtitle_path
            .replace("\\", "/")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )

        cmd = [
            "ffmpeg",
            "-y",

            # Video input
            "-i", str(video_input),

            # Dubbed audio input
            "-i", str(audio_input),

            # Video
            "-map", "0:v:0",

            # New translated audio
            "-map", "1:a:0",

            # Subtitle burn-in
            "-vf", f"subtitles='{subtitle_path}'",

            # Browser-compatible video
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",

            # Maximum browser compatibility
            "-pix_fmt", "yuv420p",

            # Browser streaming
            "-movflags", "+faststart",

            # Audio
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",

            # Keep video duration
            "-t", str(video_duration) if video_duration > 0 else "999999",

            # Output
            str(output_video)
        ]

    else:

        print()
        print("⚡ Creating browser-compatible MP4...")
        print("🎞️ Video: H.264")
        print("🔊 Audio: AAC")
        print("🌐 MP4 optimized for browser preview")

        cmd = [
            "ffmpeg",
            "-y",

            # Original video
            "-i", str(video_input),

            # Translated audio
            "-i", str(audio_input),

            # Original video stream
            "-map", "0:v:0",

            # Translated audio stream
            "-map", "1:a:0",

            # Browser-compatible H.264
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",

            # Important for Safari/Chrome/Streamlit preview
            "-pix_fmt", "yuv420p",

            # AAC audio
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",

            # Make MP4 seekable/streamable
            "-movflags", "+faststart",

            # Use original video duration
            "-t", str(video_duration) if video_duration > 0 else "999999",

            # Output
            str(output_video)
        ]

    # --------------------------------------------------------
    # Execute FFmpeg
    # --------------------------------------------------------

    print()
    print("🚀 Running FFmpeg...")
    print()

    try:

        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:

            print()
            print("❌ FFmpeg failed.")
            print()
            print(process.stderr)

            sys.exit(1)

    except Exception as e:

        print()
        print(f"❌ Failed to execute FFmpeg: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if not output_video.exists():
        print()
        print("❌ FFmpeg finished but output file was not created.")
        sys.exit(1)

    output_size = output_video.stat().st_size

    if output_size == 0:
        print()
        print("❌ Output video is empty.")
        sys.exit(1)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("==================================================")
    print("✅ VIDEO CREATION SUCCESSFUL")
    print("==================================================")

    print()
    print("🎬 Final Video:")
    print(f"   {output_video}")

    print()
    print("📦 Format:")
    print("   MP4")

    print()
    print("🎞️ Video Codec:")
    print("   H.264 / AVC")

    print()
    print("🔊 Audio Codec:")
    print("   AAC")

    print()
    print("🌐 Browser Compatibility:")
    print("   Chrome / Safari / Edge / Firefox")

    print()
    print(f"📏 File Size:")
    print(f"   {output_size / (1024 * 1024):.2f} MB")

    print()
    print("==================================================")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print()
        print(
            "python3 test_video_muxing.py "
            "<video_path> [lang] [--burn-subs]"
        )

        print()
        print("Examples:")
        print(
            "python3 test_video_muxing.py "
            "storage_vault/inputs/English.mp4 mr"
        )

        print(
            "python3 test_video_muxing.py "
            "storage_vault/inputs/English.mp4 mr --burn-subs"
        )

        sys.exit(1)

    input_video_file = sys.argv[1]

    lang = (
        sys.argv[2]
        if len(sys.argv) > 2 and not sys.argv[2].startswith("--")
        else "hi"
    )

    burn_subs = "--burn-subs" in sys.argv

    merge_video_and_audio(
        input_video_file,
        target_lang=lang,
        burn_subtitles=burn_subs
    )