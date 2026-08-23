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
# Helper Function for Real-Time Terminal Streaming
# ─────────────────────────────────────────────────────────
def run_command_with_live_logs(cmd_list, cwd):
    """
    Executes a command, streams output line-by-line to the terminal instantly,
    and captures the output for Streamlit UI logging.
    """
    # Environment override to force unbuffered output from child Python scripts
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Ensures standard tqdm progress output

    process = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr (progress bars) into stdout
        text=True,
        bufsize=1,  # Line-buffered
        cwd=cwd,
        env=env
    )

    full_output = []
    
    # Read output character/line in real-time
    if process.stdout:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line, end='', flush=True)  # Print immediately to terminal
                full_output.append(line)

    returncode = process.wait()
    return returncode, "".join(full_output)

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
            index=2,  # Default to Marathi
            help="Language of the original video audio"
        )
    
    with col2:
        tgt_lang = st.selectbox(
            "Target Language",
            ["Hindi", "Marathi", "English"],
            index=0,  # Default to Hindi
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
    if st.button("🚀 Start Translation & Dubbing", type="primary", disabled=not (video_input_path and can_process), use_container_width=True):
        st.session_state.start_processing = True

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
        
        returncode, stdout_str = run_command_with_live_logs(
            [
                "python3",
                str(BASE_DIR / "run_step1_asr.py"),
                str(video_input_path),
                "--lang",
                src_code
            ],
            cwd=str(BASE_DIR)
        )
        
        logs_dict["stage1"] = stdout_str
        
        if returncode != 0:
            logs_dict["errors"] += f"\n❌ Stage 1 Error:\n{stdout_str}\n"
            status_text.error(f"❌ {stages[0][0]} failed!")
            st.error("Please try again or check the input video file.")
            st.stop()
        
        progress_bar.progress(stages[0][1])
        status_text.success(f"✅ {stages[0][0]} completed!")
        
        # ────────────────────────────────────────────
        # STAGE 2: Translation (e.g. Marathi -> Hindi)
        # ────────────────────────────────────────────
        status_text.info(f"⏳ {stages[1][0]}... Please wait")
        progress_bar.progress(0.30)
        
        returncode, stdout_str = run_command_with_live_logs(
            [
                "python3",
                str(BASE_DIR / "run_step1_translation.py"), 
                str(OUTPUTS_DIR / "step1_whisper_output.json"),
                tgt_code,
                src_code
            ],
            cwd=str(BASE_DIR)
        )
        
        logs_dict["stage2"] = stdout_str
        
        if returncode != 0:
            logs_dict["errors"] += f"\n❌ Stage 2 Error:\n{stdout_str}\n"
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
        
        # Resolve correct TTS script based on target language
        if tgt_code == "mr" and (BASE_DIR / "run_step2_tts_srt_mr.py").exists():
            tts_script = "run_step2_tts_srt_mr.py"
        elif tgt_code == "hi" and (BASE_DIR / "run_step2_tts_srt_hi.py").exists():
            tts_script = "run_step2_tts_srt_hi.py"
        else:
            tts_script = "run_step2_tts_srt.py"
        
        returncode, stdout_str = run_command_with_live_logs(
            [
                "python3",
                str(BASE_DIR / tts_script),
                str(OUTPUTS_DIR / "step2_offline_translated.json"),
                tgt_code
            ],
            cwd=str(BASE_DIR)
        )
        
        logs_dict["stage3"] = stdout_str
        
        if returncode != 0:
            logs_dict["errors"] += f"\n❌ Stage 3 Error:\n{stdout_str}\n"
            status_text.error(f"❌ {stages[2][0]} failed!")
            st.error("Audio synthesis encountered an error.")
            st.stop()
        
        progress_bar.progress(stages[2][1])
        status_text.success(f"✅ {stages[2][0]} completed!")
        
        # ────────────────────────────────────────────
        # STAGE 4: Video Muxing
        # ────────────────────────────────────────────
        status_text.info(f"⏳ {stages[3][0]}... Please wait")
        progress_bar.progress(0.85)
        
        returncode, stdout_str = run_command_with_live_logs(
            [
                "python3",
                str(BASE_DIR / "test_video_muxing.py"),
                str(video_input_path),
                tgt_code
            ],
            cwd=str(BASE_DIR)
        )
        
        logs_dict["stage4"] = stdout_str
        
        if returncode != 0:
            logs_dict["errors"] += f"\n❌ Stage 4 Error:\n{stdout_str}\n"
            status_text.error(f"❌ {stages[3][0]} failed!")
            st.error("Video muxing encountered an error. Check if FFmpeg is installed correctly.")
            st.stop()
        
        # Resolve output MP4 path dynamically
        final_video_path = None
        saved_match = re.search(
            r"Final video saved as\s*[:=]?\s*(.+\.mp4)\s*$",
            stdout_str,
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

        # Language-specific output fallback
        if final_video_path is None:
            candidate = OUTPUTS_DIR / f"final_translated_video_{tgt_code}.mp4"
            if candidate.exists():
                final_video_path = candidate.resolve()

        # Recency fallback
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
                f"Stage 4 output:\n{stdout_str}\n"
            )
            status_text.error("❌ Video was generated, but the output file could not be located.")
            st.error("⚠️ Final video not found. Please check the output path below.")
            st.code(
                f"Expected output directory:\n{OUTPUTS_DIR}\n\n"
                f"Stage 4 output:\n{stdout_str}"
            )
            st.stop()

        progress_bar.progress(stages[3][1])
        status_text.success(f"✅ {stages[3][0]} completed!")
        
        # ────────────────────────────────────────────
        # SUCCESS: Display Results
        # ────────────────────────────────────────────
        progress_bar.progress(1.0)
        status_text.empty()
        
        st.success("🎉 Pipeline Completed Successfully!")
        
        st.divider()
        st.subheader("📺 Preview & Download")
        
        if final_video_path and final_video_path.exists():
            st.video(str(final_video_path))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with open(final_video_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Dubbed Video",
                        data=f,
                        file_name=f"dubbed_{src_code}_to_{tgt_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📁 Open Output Folder", use_container_width=True):
                    if platform.system() == "Darwin":
                        subprocess.run(["open", str(OUTPUTS_DIR)])
                    elif platform.system() == "Windows":
                        subprocess.run(["explorer", str(OUTPUTS_DIR)])
                    else:
                        subprocess.run(["xdg-open", str(OUTPUTS_DIR)])
                    st.info("📂 Opened output folder")
            
            with col3:
                if st.button("🔄 Process Another Video", use_container_width=True):
                    st.session_state.start_processing = False
                    st.rerun()
            
            st.divider()
            
            # Summary of generated files
            st.subheader("📂 Generated Files")
            output_files = {
                "🎬 Dubbed Video": final_video_path,
                "📝 Transcript": OUTPUTS_DIR / "step1_whisper_output.json",
                "🌐 Translation": OUTPUTS_DIR / "step2_offline_translated.json",
                "📄 Subtitles": OUTPUTS_DIR / f"video_subtitles_{tgt_code}.srt"
            }
            
            file_count = sum(1 for p in output_files.values() if p.exists())
            st.success(f"✅ {file_count} files generated in `storage_vault/outputs/`")
            
            st.divider()
            with st.expander("📋 View Processing Logs", expanded=False):
                if logs_dict["errors"]:
                    st.warning("⚠️ Errors Encountered:")
                    st.code(logs_dict["errors"], language="text")
                
                st.subheader("Stage 1: Speech Recognition")
                st.code(logs_dict["stage1"] or "No logs available", language="text")
                
                st.subheader("Stage 2: Translation")
                st.code(logs_dict["stage2"] or "No logs available", language="text")
                
                st.subheader("Stage 3: Text-to-Speech")
                st.code(logs_dict["stage3"] or "No logs available", language="text")
                
                st.subheader("Stage 4: Video Muxing")
                st.code(logs_dict["stage4"] or "No logs available", language="text")
        else:
            st.error("⚠️ Final video not found. Please check the output folder.")
    
    except Exception as e:
        progress_bar.progress(0)
        status_text.error(f"❌ Error: {str(e)}")
        st.stop()