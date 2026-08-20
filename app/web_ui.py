import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
import streamlit as st
import platform

# ─────────────────────────────────────────────────────────
# Configuration & Paths
# ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_DIR = BASE_DIR / "storage_vault" / "inputs"
OUTPUTS_DIR = BASE_DIR / "storage_vault" / "outputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

# ─────────────────────────────────────────────────────────
# Streamlit Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BAIF Video Dubbing Engine",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎙️ BAIF Offline Video Translation & Dubbing Engine")
st.caption("Complete 4-Stage Pipeline: ASR → Translation → TTS → Video Muxing")

def summarize_transcript(texts: list[str]) -> str:
    """Create a dynamic number of concise bullets from the complete narration."""
    full_text = " ".join(texts)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?।॥])\s+", full_text)
        if sentence.strip()
    ]
    unique_sentences = []
    seen_sentences = set()
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence).casefold()
        if normalized not in seen_sentences:
            seen_sentences.add(normalized)
            unique_sentences.append(sentence)
    sentences = unique_sentences

    if len(sentences) <= 3:
        selected_sentences = sentences
    else:
        # Larger transcripts receive more points, without a fixed maximum.
        summary_count = max(3, round(len(sentences) * 0.4))
        words = re.findall(r"[\w\u0900-\u097F]+", full_text.lower(), flags=re.UNICODE)
        frequencies = {}
        for word in words:
            if len(word) > 2:
                frequencies[word] = frequencies.get(word, 0) + 1

        scored_sentences = []
        for index, sentence in enumerate(sentences):
            sentence_words = re.findall(r"[\w\u0900-\u097F]+", sentence.lower(), flags=re.UNICODE)
            score = sum(frequencies.get(word, 0) for word in sentence_words)
            scored_sentences.append((score / max(len(sentence_words), 1), index, sentence))

        selected_sentences = [
            sentence
            for _, _, sentence in sorted(
                sorted(scored_sentences, reverse=True)[:summary_count],
                key=lambda item: item[1]
            )
        ]

    return "\n".join(f"- {sentence}" for sentence in selected_sentences)


def generate_video_summary(translated_json_path: Path, target_lang: str) -> Path:
    """Create bullet-point summary from all translated dubbed-video text."""
    summary_path = translated_json_path.with_name(f"final_video_summary_{target_lang}.txt")

    if not translated_json_path.exists():
        summary_text = (
            f"Final dubbed video bullet-point summary ({target_lang.upper()}):\n\n"
            f"No translated segments were found for this video."
        )
        summary_path.write_text(summary_text, encoding="utf-8")
        return summary_path

    try:
        with open(translated_json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    except Exception:
        summary_text = (
            f"Final dubbed video bullet-point summary ({target_lang.upper()}):\n\n"
            f"No translated segments were found for this video."
        )
        summary_path.write_text(summary_text, encoding="utf-8")
        return summary_path

    texts = []
    for seg in segments:
        for key in [f"translation_{target_lang}", "translated_text", "text"]:
            text = seg.get(key)
            if isinstance(text, str) and text.strip():
                texts.append(re.sub(r"\s+", " ", text).strip())
                break

    if not texts:
        summary_text = (
            f"Final dubbed video bullet-point summary ({target_lang.upper()}):\n\n"
            f"No translated segments were found for this video."
        )
    else:
        summary_text = (
            f"Final dubbed video bullet-point summary ({target_lang.upper()}):\n\n"
            + summarize_transcript(texts)
        )

    summary_path.write_text(summary_text, encoding="utf-8")
    return summary_path

# ─────────────────────────────────────────────────────────
# Sidebar: File Upload & Language Selection
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Configuration")
    
    # Step 1: Upload Video
    st.subheader("1️⃣ Upload Video File")
    uploaded_file = st.file_uploader(
        "Select a video file (MP4, MKV, MOV)",
        type=["mp4", "mkv", "mov"],
        help="Maximum file size: 500MB"
    )
    
    if uploaded_file:
        st.success(f"✅ File selected: {uploaded_file.name}")
        
        # Save uploaded file
        video_input_path = INPUTS_DIR / uploaded_file.name
        with open(video_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        video_input_path = None
        st.info("No file selected yet")
    
    st.divider()
    
    # Step 2: Language Selection
    st.subheader("2️⃣ Language Selection")
    
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox(
            "Source Language",
            ["English", "Hindi", "Marathi"],
            index=0,
            help="Language of the original video audio"
        )
    
    with col2:
        tgt_lang = st.selectbox(
            "Target Language",
            ["Marathi", "Hindi", "English"],
            index=0,
            help="Language for dubbing output"
        )
    
    if src_lang == tgt_lang:
        st.warning("⚠️ Source and target languages must be different!")
        can_process = False
    else:
        can_process = True
    
    st.divider()
    
    # Step 3: Process Button
    st.subheader("3️⃣ Start Pipeline")
    generate_summary = st.checkbox(
        "📝 Generate summary from all dubbed text",
        value=True,
        help="Create a dynamic number of summary points from the complete translated video text."
    )
    if st.button("🚀 Start Translation & Dubbing", type="primary", disabled=not (video_input_path and can_process), use_container_width=True):
        st.session_state.start_processing = True
        st.session_state.generate_summary = generate_summary

# ─────────────────────────────────────────────────────────
# Main Content Area: Processing & Results
# ─────────────────────────────────────────────────────────
if not video_input_path:
    st.info("👈 Please upload a video file using the sidebar to begin")
    st.stop()

if LANG_MAP[src_lang] == LANG_MAP[tgt_lang]:
    st.error("❌ Source and target languages must be different!")
    st.stop()

# Initialize session state
if "start_processing" not in st.session_state:
    st.session_state.start_processing = False
if "generate_summary" not in st.session_state:
    st.session_state.generate_summary = True

if st.session_state.start_processing:
    src_code = LANG_MAP[src_lang]
    tgt_code = LANG_MAP[tgt_lang]
    
    st.divider()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create logs storage
    logs_dict = {
        "stage1": "",
        "stage2": "",
        "stage3": "",
        "stage4": "",
        "errors": ""
    }
    
    stages = [
        ("🎙️ Speech Recognition (ASR)", 0.25, "stage1"),
        ("🌐 Machine Translation", 0.50, "stage2"),
        ("🎤 Text-to-Speech Synthesis", 0.75, "stage3"),
        ("🎬 Video Muxing", 1.0, "stage4")
    ]
    
    try:
        # ────────────────────────────────────────────
        # STAGE 1: Speech Recognition (ASR)
        # ────────────────────────────────────────────
        status_text.info(f"⏳ {stages[0][0]}... Please wait")
        progress_bar.progress(0.05)
        
        result = subprocess.run(
        ["python3", str(BASE_DIR / "test_whisper_asr.py"), str(video_input_path), "--lang", src_code],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
        )
        
        logs_dict["stage1"] = result.stdout
        
        if result.returncode != 0:
            logs_dict["errors"] += f"\n❌ Stage 1 Error:\n{result.stderr}\n"
            status_text.error(f"❌ {stages[0][0]} failed!")
            st.error("Please try again or check the input video file.")
            st.stop()
        
        progress_bar.progress(stages[0][1])
        status_text.success(f"✅ {stages[0][0]} completed!")
        
        # ────────────────────────────────────────────
        # STAGE 2: Translation
        # ────────────────────────────────────────────
        status_text.info(f"⏳ {stages[1][0]}... Please wait")
        progress_bar.progress(0.30)
        
        result = subprocess.run(
        ["python3", str(BASE_DIR / "run_step1_translation.py"), 
        str(OUTPUTS_DIR / "step1_whisper_output.json"), tgt_code, src_code],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR)
        )
        
        logs_dict["stage2"] = result.stdout
        
        if result.returncode != 0:
            logs_dict["errors"] += f"\n❌ Stage 2 Error:\n{result.stderr}\n"
            status_text.error(f"❌ {stages[1][0]} failed!")
            st.error("Translation process encountered an error.")
            st.stop()
        
        progress_bar.progress(stages[1][1])
        status_text.success(f"✅ {stages[1][0]} completed!")
        
        # ────────────────────────────────────────────
        # STAGE 3: Text-to-Speech
        # ────────────────────────────────────────────
        status_text.info(f"⏳ {stages[2][0]}... Please wait")
        progress_bar.progress(0.55)
        
        # Use the Sherpa-ONNX path for all languages, including Marathi.
        # The legacy Marathi script requires an external `piper` executable.
        tts_script = "run_step2_tts_srt.py"
        
        result = subprocess.run(
            ["python3", str(BASE_DIR / tts_script),
             str(OUTPUTS_DIR / "step2_offline_translated.json"), tgt_code],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR)
        )
        
        logs_dict["stage3"] = result.stdout
        
        if result.returncode != 0:
            logs_dict["errors"] += f"\n❌ Stage 3 Error:\n{result.stderr}\n"
            status_text.error(f"❌ {stages[2][0]} failed!")
            st.error("Audio synthesis encountered an error.")
            if result.stdout:
                st.code(result.stdout, language="text")
            if result.stderr:
                st.code(result.stderr, language="text")
            st.stop()
        
        progress_bar.progress(stages[2][1])
        status_text.success(f"✅ {stages[2][0]} completed!")
        
        # ────────────────────────────────────────────
        # STAGE 4: Video Muxing
        # ────────────────────────────────────────────
        status_text.info(f"⏳ {stages[3][0]}... Please wait")
        progress_bar.progress(0.85)
        
        result = subprocess.run(
            ["python3", str(BASE_DIR / "test_video_muxing.py"),
             str(video_input_path), tgt_code],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR)
        )
        
        logs_dict["stage4"] = result.stdout
        
        if result.returncode != 0:
            logs_dict["errors"] += f"\n❌ Stage 4 Error:\n{result.stderr}\n"
            status_text.error(f"❌ {stages[3][0]} failed!")
            st.error("Video muxing encountered an error. Check if FFmpeg is installed correctly.")
            st.stop()
        
        # Resolve the exact output file created by Stage 4.
        # test_video_muxing.py prints the final saved path, so use that path
        # instead of assuming a fixed filename.
        final_video_path = None

        stage4_output = (result.stdout or "") + "\n" + (result.stderr or "")

        saved_match = re.search(
            r"Final video saved as\\s*[:=]?\\s*(.+?\\.mp4)\\s*$",
            stage4_output,
            re.IGNORECASE | re.MULTILINE
        )

        if saved_match:
            saved_path = Path(saved_match.group(1).strip().strip('"').strip("'"))

            if not saved_path.is_absolute():
                saved_path = (BASE_DIR / saved_path).resolve()
            else:
                saved_path = saved_path.resolve()

            if saved_path.exists():
                final_video_path = saved_path

        # Current naming-convention fallback.
        if final_video_path is None:
            candidate = OUTPUTS_DIR / f"final_translated_video_{tgt_code}.mp4"
            if candidate.exists():
                final_video_path = candidate.resolve()

        # Future-proof fallback: find the newest MP4 recursively.
        if final_video_path is None:
            video_files = list(OUTPUTS_DIR.rglob("*.mp4"))
            if video_files:
                final_video_path = max(
                    video_files,
                    key=lambda p: p.stat().st_mtime
                ).resolve()

        if final_video_path is None:
            logs_dict["errors"] += (
                "\n❌ Stage 4 completed, but Streamlit could not locate the generated MP4.\n"
                f"Expected output directory: {OUTPUTS_DIR}\n"
                f"Stage 4 output:\n{stage4_output}\n"
            )
            status_text.error("❌ Video was generated, but the output file could not be located.")
            st.error("⚠️ Final video not found. Please check the output path below.")
            st.code(
                f"Expected output directory:\n{OUTPUTS_DIR}\n\n"
                f"Stage 4 output:\n{stage4_output}"
            )
            st.stop()

        progress_bar.progress(stages[3][1])
        status_text.success(f"✅ {stages[3][0]} completed!")

        summary_path = None
        if st.session_state.generate_summary:
            status_text.info("⏳ Generating summary from all dubbed text...")
            translated_json = OUTPUTS_DIR / "step2_offline_translated.json"
            summary_path = generate_video_summary(translated_json, tgt_code)
            status_text.success("✅ Complete video summary created!")
        
        # ────────────────────────────────────────────
        # SUCCESS: Display Results
        # ────────────────────────────────────────────
        progress_bar.progress(1.0)
        status_text.empty()
        
        st.success("🎉 Pipeline Completed Successfully!")
        
        st.divider()
        st.subheader("📺 Preview & Download")
        
        # Use the exact output path resolved after Stage 4.
        if final_video_path and final_video_path.exists():
            
            # Display video preview
            st.video(str(final_video_path))
            
            # Download button and folder button
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with open(final_video_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Dubbed Video",
                        data=f,
                        file_name=f"dubbed_{tgt_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📁 Open Output Folder", use_container_width=True):
                    if platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", str(OUTPUTS_DIR)])
                    elif platform.system() == "Windows":
                        subprocess.run(["explorer", str(OUTPUTS_DIR)])
                    else:  # Linux
                        subprocess.run(["xdg-open", str(OUTPUTS_DIR)])
                    st.info(f"📂 Opened output folder")
            
            with col3:
                if st.button("🔄 Process Another Video", use_container_width=True):
                    st.session_state.start_processing = False
                    st.rerun()
            
            st.divider()

            if summary_path and summary_path.exists():
                st.subheader("📝 Final Dubbed Video Summary")
                summary_text = summary_path.read_text(encoding="utf-8")
                st.code(summary_text, language="text")
                with open(summary_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Summary",
                        data=f,
                        file_name=f"final_summary_{tgt_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

            st.divider()
            
            # Show output files summary
            st.subheader("📂 Generated Files")
            output_files = {
                "🎬 Dubbed Video": final_video_path,
                "📝 Transcript": OUTPUTS_DIR / "step1_whisper_output.json",
                "🌐 Translation": OUTPUTS_DIR / "step2_offline_translated.json",
                "📄 Subtitles": OUTPUTS_DIR / f"video_subtitles_{tgt_code}.srt",
                "📝 Video Summary": summary_path
            }
            
            file_count = 0
            for name, path in output_files.items():
                if path is not None and path.exists():
                    file_count += 1
            
            st.success(f"✅ {file_count} files generated in `storage_vault/outputs/`")
            
            # Show logs in expandable section
            st.divider()
            with st.expander("📋 View Processing Logs", expanded=False):
                if logs_dict["errors"]:
                    st.warning("⚠️ Errors Encountered:")
                    st.code(logs_dict["errors"], language="text")
                
                st.subheader("Stage 1: Speech Recognition")
                if logs_dict["stage1"]:
                    st.code(logs_dict["stage1"], language="text")
                else:
                    st.info("No logs available")
                
                st.subheader("Stage 2: Translation")
                if logs_dict["stage2"]:
                    st.code(logs_dict["stage2"], language="text")
                else:
                    st.info("No logs available")
                
                st.subheader("Stage 3: Text-to-Speech")
                if logs_dict["stage3"]:
                    st.code(logs_dict["stage3"], language="text")
                else:
                    st.info("No logs available")
                
                st.subheader("Stage 4: Video Muxing")
                if logs_dict["stage4"]:
                    st.code(logs_dict["stage4"], language="text")
                else:
                    st.info("No logs available")
        else:
            st.error("⚠️ Final video not found. Please check the output folder.")
            st.code(
                f"Expected output directory:\n{OUTPUTS_DIR}\n\n"
                f"Resolved final video path:\n{final_video_path}"
            )
    
    except Exception as e:
        progress_bar.progress(0)
        status_text.error(f"❌ Error: {str(e)}")
        st.stop()
