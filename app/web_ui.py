import streamlit as st
import requests
import time
import pandas as pd
import srt
import datetime

# --- API CONFIG ---
API_URL = "http://127.0.0.1:8000"

# --- PAGE SETUP ---
st.set_page_config(
    page_title="BAIF Offline Dubbing Engine",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 BAIF Offline Translation & Dubbing Workspace")
st.caption("Air-Gapped On-Premises Video/Audio Translation Pipeline (Whisper → IndicTrans2 → Sherpa-ONNX)")
st.divider()

# --- UI STATE MANAGEMENT ---
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "status" not in st.session_state:
    st.session_state.status = None
if "segments" not in st.session_state:
    st.session_state.segments = []

# --- TOP ROW: INGESTION & TRACKING ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Media")
    uploaded_file = st.file_uploader("Upload Video or Audio (MP4, WAV, MP3)", type=["mp4", "wav", "mp3", "mkv", "mov"])
    
    target_lang = st.selectbox(
        "Select Target Dubbing Language", 
        options=[("Hindi", "hi"), ("Marathi", "mr")], 
        format_func=lambda x: x[0]
    )
    
    if st.button("🚀 Process & Translate", use_container_width=True) and uploaded_file:
        with st.spinner("Uploading to Offline Engine..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"target_language": target_lang[1]}
            
            try:
                res = requests.post(f"{API_URL}/process", files=files, data=data)
                res.raise_for_status()
                payload = res.json()
                
                st.session_state.job_id = payload["job_id"]
                st.session_state.status = payload["status"]
                
                # Check for SHA-256 instant DB hit
                if payload.get("cached"):
                    st.session_state.segments = payload.get("segments", [])
                    st.success("⚡ Instant Cache Hit! Media was previously processed.")
                else:
                    st.success("Background job queued successfully!")
                
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to connect to backend API: {e}")

with col2:
    st.subheader("2. Pipeline Status")
    
    if st.session_state.job_id:
        if st.session_state.status in ["QUEUED", "PROCESSING"]:
            try:
                status_res = requests.get(f"{API_URL}/status/{st.session_state.job_id}")
                if status_res.status_code == 200:
                    status_data = status_res.json()
                    st.session_state.status = status_data.get("status")
                    
                    if st.session_state.status in ["QUEUED", "PROCESSING"]:
                        st.info(f"⏳ **{st.session_state.status}**: {status_data.get('progress')}")
                        time.sleep(2) # Auto-poll interval
                        st.rerun()
                    elif st.session_state.status == "COMPLETED":
                        st.session_state.segments = status_data.get("result", {}).get("segments", [])
                        st.success("🎉 Pipeline Processing Complete!")
                        st.rerun()
                    elif st.session_state.status == "FAILED":
                        st.error(f"❌ Failed: {status_data.get('reason')}")
            except Exception as e:
                st.warning("Waiting for backend response...")
                time.sleep(2)
                st.rerun()
                
        elif st.session_state.status == "COMPLETED":
            st.success(f"🎉 Pipeline Processing Complete! (Job ID: `{st.session_state.job_id}`)")
        elif st.session_state.status == "FAILED":
            st.error("❌ Job Processing Failed.")
    else:
        st.info("Awaiting media upload...")

st.divider()

# --- BOTTOM ROW: HUMAN-IN-THE-LOOP & MEDIA PREVIEW ---
if st.session_state.status == "COMPLETED" and st.session_state.segments:
    st.subheader("3. Human-in-the-Loop (HITL) Subtitle Editor")
    
    df = pd.DataFrame(st.session_state.segments)
    if "translated_text" in df.columns:
        df = df[["start", "end", "text", "translated_text"]]
    
    st.caption("Review and refine the translated segments below. Edits will dynamically update the `.srt` download file.")
    
    # Interactive Data Editor
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        column_config={
            "start": st.column_config.NumberColumn("Start (s)", disabled=True),
            "end": st.column_config.NumberColumn("End (s)", disabled=True),
            "text": st.column_config.TextColumn("Original English", disabled=True),
            "translated_text": st.column_config.TextColumn("Translated Text (Editable)")
        },
        hide_index=True
    )
    
    # Dynamic SRT Generation
    def generate_srt_content(dataframe):
        srt_subtitles = []
        for idx, row in dataframe.iterrows():
            sub = srt.Subtitle(
                index=idx + 1,
                start=datetime.timedelta(seconds=float(row["start"])),
                end=datetime.timedelta(seconds=float(row["end"])),
                content=str(row["translated_text"])
            )
            srt_subtitles.append(sub)
        return srt.compose(srt_subtitles)

    final_srt = generate_srt_content(edited_df)
    
    st.subheader("4. Final Media Outputs")
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    
    with dl_col1:
        st.download_button(
            label="📄 Download Refined Subtitles (.srt)",
            data=final_srt,
            file_name=f"{st.session_state.job_id}_subtitles.srt",
            mime="text/plain",
            use_container_width=True
        )
        
    with dl_col2:
        try:
            # We fetch audio server-side to prevent CORS/IP resolution issues on intranets
            audio_res = requests.get(f"{API_URL}/download/{st.session_state.job_id}/audio")
            if audio_res.status_code == 200:
                st.download_button(
                    label="🔊 Download Dubbed Audio (.wav)",
                    data=audio_res.content,
                    file_name=f"{st.session_state.job_id}_dubbed.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
        except Exception:
            st.error("Audio download unavailable.")

    st.divider()
    
    st.subheader("5. Media Preview")
    prev_col1, prev_col2 = st.columns(2)
    
    with prev_col1:
        st.markdown("**Original Uploaded Media**")
        if uploaded_file:
            if "video" in uploaded_file.type:
                st.video(uploaded_file)
            else:
                st.audio(uploaded_file)
                
    with prev_col2:
        st.markdown("**Synthesized Dubbed Audio**")
        if 'audio_res' in locals() and audio_res.status_code == 200:
            st.audio(audio_res.content, format="audio/wav")