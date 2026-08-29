import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

# Make project root importable when Streamlit launches this file directly.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.modules.database import (  # noqa: E402
    build_ui_cache_key,
    compute_file_hash,
    compute_text_hash,
    delete_ui_cached_result,
    get_ui_cached_result,
    init_db,
    save_ui_cached_result,
)

INPUTS_DIR = BASE_DIR / "storage_vault" / "inputs"
OUTPUTS_DIR = BASE_DIR / "storage_vault" / "outputs"
CACHE_DIR = BASE_DIR / "storage_vault" / "cache"

INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
init_db()

PYTHON_BIN = sys.executable

PIPELINE_VERSION = "quality-v2-beam5-context"

LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}

VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "wmv", "mkv", "flv", "webm"}
AUDIO_EXTENSIONS = {"mp3", "wav", "aac", "m4a", "flac", "wma", "ogg"}
MEDIA_EXTENSIONS = sorted(VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)

st.set_page_config(
    page_title="Team Anuwad | BAIF Translation Studio",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ BAIF Anuwad Studio (अनुवाद)")
st.caption(
    "Developed by **Team Anuwad** "
)

# Initialize Session State
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "has_started" not in st.session_state:
    st.session_state.has_started = False


def summarize_transcript(texts: list[str]) -> str:
    full_text = " ".join(texts).strip()
    if not full_text:
        return ""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?।॥])\s+", full_text)
        if sentence.strip()
    ]

    unique_sentences = []
    seen = set()
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence).casefold()
        if normalized not in seen:
            seen.add(normalized)
            unique_sentences.append(sentence)

    sentences = unique_sentences
    if len(sentences) <= 3:
        selected = sentences
    else:
        summary_count = max(3, round(len(sentences) * 0.4))
        words = re.findall(r"[\w\u0900-\u097F]+", full_text.lower(), flags=re.UNICODE)
        frequencies = {}
        for word in words:
            if len(word) > 2:
                frequencies[word] = frequencies.get(word, 0) + 1

        scored = []
        for index, sentence in enumerate(sentences):
            sentence_words = re.findall(
                r"[\w\u0900-\u097F]+", sentence.lower(), flags=re.UNICODE
            )
            score = sum(frequencies.get(word, 0) for word in sentence_words)
            scored.append((score / max(len(sentence_words), 1), index, sentence))

        selected = [
            sentence
            for _, _, sentence in sorted(
                sorted(scored, reverse=True)[:summary_count],
                key=lambda item: item[1],
            )
        ]

    return "\n".join(f"- {sentence}" for sentence in selected)


def generate_summary(translated_json: Path, target_lang: str) -> Path:
    segments = json.loads(translated_json.read_text(encoding="utf-8"))
    texts = []
    for segment in segments:
        text = (
            segment.get(f"translation_{target_lang}")
            or segment.get("translated_text")
            or segment.get("text")
        )
        if isinstance(text, str) and text.strip():
            texts.append(re.sub(r"\s+", " ", text).strip())

    summary = summarize_transcript(texts)
    summary_path = OUTPUTS_DIR / f"final_summary_{target_lang}.txt"
    summary_path.write_text(
        f"Summary ({target_lang.upper()}):\n\n{summary or 'No summary available.'}\n",
        encoding="utf-8",
    )
    return summary_path


def run_command(command: list[str], title: str) -> None:
    """Run process silently without UI logs."""
    process = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"{title} failed with exit code {process.returncode}.\n\n{process.stderr}")


def save_text_as_segments(text: str) -> Path:
    output = OUTPUTS_DIR / "step1_whisper_output.json"
    output.write_text(
        json.dumps(
            [{"id": 1, "start": 0.0, "end": 1.0, "text": text.strip()}],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def copy_if_exists(source: Optional[Path], destination: Path) -> Optional[Path]:
    if source is None or not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def cache_path_to_db(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def db_path_to_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def cache_entry_is_usable(
    cached: dict,
    *,
    require_tts: bool,
    require_summary: bool,
    require_video: bool,
) -> bool:
    translated = db_path_to_path(cached.get("translated_json_path"))
    if translated is None or not translated.exists():
        return False

    if require_tts:
        audio = db_path_to_path(cached.get("audio_path"))
        srt = db_path_to_path(cached.get("srt_path"))
        if audio is None or not audio.exists() or srt is None or not srt.exists():
            return False

    if require_summary:
        summary = db_path_to_path(cached.get("summary_path"))
        if summary is None or not summary.exists():
            return False

    if require_video:
        video = db_path_to_path(cached.get("video_path"))
        if video is None or not video.exists():
            return False

    return True


def render_results(
    *,
    translated_json: Path,
    target_lang: str,
    generate_tts: bool,
    audio_path: Optional[Path],
    srt_path: Optional[Path],
    summary_path: Optional[Path],
    final_video: Optional[Path],
    cache_hit: bool,
) -> None:
    if cache_hit:
        st.success("🎯 Cache hit — reused the previously completed result.")

    segments = json.loads(translated_json.read_text(encoding="utf-8"))
    translated_texts = [
        segment.get(f"translation_{target_lang}")
        or segment.get("translated_text")
        or ""
        for segment in segments
    ]
    if summary_path and summary_path.exists():
        st.subheader("📝 Summary")
        summary_text = summary_path.read_text(encoding="utf-8")
        st.text_area("Summary output", value=summary_text, height=180)
        st.download_button(
            "⬇️ Summary",
            data=summary_path.read_bytes(),
            file_name=summary_path.name,
            mime="text/plain",
        )

    if generate_tts and audio_path and audio_path.exists():
        st.subheader("🔊 Translated Audio")
        st.audio(str(audio_path))

    if final_video and final_video.exists():
        st.subheader("🎬 Final Translated Video")
        st.video(str(final_video))
        st.download_button(
            "⬇️ Final Video",
            data=final_video.read_bytes(),
            file_name=final_video.name,
            mime="video/mp4",
        )


def reset_translation():
    st.session_state.is_processing = False
    st.session_state.has_started = False


def render_workflow(steps_state: dict):
    """Renders the vertical step workflow in UI."""
    st.subheader("📋 Workflow Steps")
    for step_num, info in steps_state.items():
        status_icon = "⏳"
        if info["status"] == "running":
            status_icon = "🔄"
        elif info["status"] == "done":
            status_icon = "✅"
        elif info["status"] == "error":
            status_icon = "❌"

        st.markdown(f"**Step {step_num}: {info['name']}** — {status_icon} *{info['status'].title()}*")


with st.sidebar:
    st.header("📋 Configuration")

    input_mode = st.radio(
        "Input Type",
        ["Audio / Video", "Text"],
        disabled=st.session_state.is_processing,
    )

    uploaded_path = None
    uploaded_file_name = None
    text_input = ""

    if input_mode == "Audio / Video":
        uploaded_file = st.file_uploader(
            "Upload audio or video",
            type=MEDIA_EXTENSIONS,
            accept_multiple_files=False,  # Single item upload to avoid "+" icon
            disabled=st.session_state.is_processing,
            help=(
                "Video: MP4, MOV, AVI, WMV, MKV, FLV, WebM | "
                "Audio: MP3, WAV, AAC, M4A, FLAC, WMA, OGG"
            ),
        )
        if uploaded_file:
            uploaded_file_name = uploaded_file.name
            uploaded_path = INPUTS_DIR / uploaded_file.name
            uploaded_path.write_bytes(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name}")
    else:
        text_input = st.text_area(
            "Text to translate",
            height=180,
            placeholder="Enter English, Hindi or Marathi text...",
            disabled=st.session_state.is_processing,
        )

    st.divider()
    source_name = st.selectbox(
        "Source Language",
        list(LANG_MAP.keys()),
        index=0,
        disabled=st.session_state.is_processing,
    )
    target_name = st.selectbox(
        "Target Language",
        list(LANG_MAP.keys()),
        index=1,
        disabled=st.session_state.is_processing,
    )
    source_lang = LANG_MAP[source_name]
    target_lang = LANG_MAP[target_name]

    if source_lang == target_lang:
        st.warning("Source and target languages must be different.")

    generate_tts = st.checkbox("Generate translated voice", value=True, disabled=st.session_state.is_processing)
    generate_summary_option = st.checkbox("Generate summary", value=True, disabled=st.session_state.is_processing)
    burn_subtitles = st.checkbox("Burn subtitles into video", value=False, disabled=st.session_state.is_processing)

    valid_input = bool(uploaded_path) if input_mode == "Audio / Video" else bool(text_input.strip())
    can_start = valid_input and source_lang != target_lang and not st.session_state.is_processing

    start = st.button(
        "🚀 Start Translation",
        type="primary",
        use_container_width=True,
        disabled=not can_start,
    )

    # Show Reset button below Start Translation once translation process has started/completed
    if st.session_state.has_started:
        st.button(
            "🔄 Reset / New Translation",
            use_container_width=True,
            on_click=reset_translation,
        )

if not can_start and not start and not st.session_state.has_started:
    st.info("Choose different source/target languages and provide an input to begin.")

if start:
    st.session_state.is_processing = True
    st.session_state.has_started = True
    st.rerun()

if st.session_state.has_started and st.session_state.is_processing:
    progress = st.progress(0)
    workflow_container = st.empty()

    # Define dynamic workflow steps based on options
    steps_state = {
        1: {"name": "Speech Recognition / Text Setup", "status": "pending"},
        2: {"name": f"Translation ({source_name} → {target_name})", "status": "pending"},
        3: {"name": "Speech Synthesis (TTS & SRT)", "status": "pending" if generate_tts else "skipped"},
        4: {"name": "Final Muxing & Summary", "status": "pending"},
    }

    with workflow_container.container():
        render_workflow(steps_state)

    try:
        # Cache Check
        if input_mode == "Audio / Video":
            assert uploaded_path is not None
            input_hash = compute_file_hash(str(uploaded_path))
            input_type = "media"
            extension = uploaded_path.suffix.lower().lstrip(".")
        else:
            input_hash = compute_text_hash(text_input)
            input_type = "text"
            extension = ""

        cache_key = build_ui_cache_key(
            input_hash=input_hash,
            input_type=input_type,
            source_language=source_lang,
            target_language=target_lang,
            generate_tts=generate_tts,
            generate_summary=generate_summary_option,
            burn_subtitles=burn_subtitles,
            pipeline_version=PIPELINE_VERSION,
        )

        require_video = (
            input_mode == "Audio / Video"
            and extension in VIDEO_EXTENSIONS
            and generate_tts
        )

        cached = get_ui_cached_result(cache_key)
        if cached and cache_entry_is_usable(
            cached,
            require_tts=generate_tts,
            require_summary=generate_summary_option,
            require_video=require_video,
        ):
            for step_key in steps_state:
                if steps_state[step_key]["status"] != "skipped":
                    steps_state[step_key]["status"] = "done"

            with workflow_container.container():
                render_workflow(steps_state)

            progress.progress(100)
            render_results(
                translated_json=db_path_to_path(cached["translated_json_path"]),
                target_lang=target_lang,
                generate_tts=generate_tts,
                audio_path=db_path_to_path(cached.get("audio_path")),
                srt_path=db_path_to_path(cached.get("srt_path")),
                summary_path=db_path_to_path(cached.get("summary_path")),
                final_video=db_path_to_path(cached.get("video_path")),
                cache_hit=True,
            )
            st.session_state.is_processing = False
            st.stop()

        elif cached:
            delete_ui_cached_result(cache_key)

        # Stage 1: Speech Recognition or text preparation
        steps_state[1]["status"] = "running"
        with workflow_container.container():
            render_workflow(steps_state)

        if input_mode == "Audio / Video":
            progress.progress(10)
            run_command(
                [
                    PYTHON_BIN,
                    str(BASE_DIR / "run_step1_asr.py"),
                    str(uploaded_path),
                    "--lang",
                    source_lang,
                    "--beam_size",
                    "5",
                    "--threads",
                    "8",
                ],
                "ASR",
            )
        else:
            save_text_as_segments(text_input)

        steps_state[1]["status"] = "done"
        progress.progress(25)

        # Stage 2: Translation
        steps_state[2]["status"] = "running"
        with workflow_container.container():
            render_workflow(steps_state)

        translated_json = OUTPUTS_DIR / "step2_offline_translated.json"
        run_command(
            [
                PYTHON_BIN,
                str(BASE_DIR / "run_step1_translation.py"),
                str(OUTPUTS_DIR / "step1_whisper_output.json"),
                target_lang,
                source_lang,
            ],
            "Translation",
        )

        steps_state[2]["status"] = "done"
        progress.progress(50)

        # Stage 3: TTS + SRT
        srt_path = OUTPUTS_DIR / f"video_subtitles_{target_lang}.srt"
        audio_path = OUTPUTS_DIR / f"video_dubbed_{target_lang}.wav"

        if generate_tts:
            steps_state[3]["status"] = "running"
            with workflow_container.container():
                render_workflow(steps_state)

            run_command(
                [
                    PYTHON_BIN,
                    str(BASE_DIR / "run_step2_tts_srt.py"),
                    str(translated_json),
                    target_lang,
                ],
                "TTS/SRT",
            )
            steps_state[3]["status"] = "done"

        progress.progress(75)

        # Stage 4: Video Muxing & Summary
        steps_state[4]["status"] = "running"
        with workflow_container.container():
            render_workflow(steps_state)

        final_video = None
        if require_video:
            mux_cmd = [
                PYTHON_BIN,
                str(BASE_DIR / "test_video_muxing.py"),
                str(uploaded_path),
                target_lang,
            ]
            if burn_subtitles:
                mux_cmd.append("--burn-subs")
            run_command(mux_cmd, "Video muxing")
            candidate = OUTPUTS_DIR / f"final_translated_video_{target_lang}.mp4"
            if candidate.exists():
                final_video = candidate

        summary_path = None
        if generate_summary_option:
            summary_path = generate_summary(translated_json, target_lang)

        steps_state[4]["status"] = "done"
        with workflow_container.container():
            render_workflow(steps_state)

        progress.progress(90)

        # Cache artifacts
        cache_entry_dir = CACHE_DIR / cache_key
        cache_entry_dir.mkdir(parents=True, exist_ok=True)

        cached_transcript = copy_if_exists(
            OUTPUTS_DIR / "step1_whisper_output.json",
            cache_entry_dir / "transcript.json",
        )
        cached_translation = copy_if_exists(
            translated_json,
            cache_entry_dir / "translation.json",
        )
        if cached_translation is None:
            raise RuntimeError("Translated JSON was not produced, so the result cannot be cached.")

        cached_audio = copy_if_exists(
            audio_path if generate_tts else None,
            cache_entry_dir / "translated_audio.wav",
        )
        cached_srt = copy_if_exists(
            srt_path if generate_tts else None,
            cache_entry_dir / "subtitles.srt",
        )
        cached_summary = copy_if_exists(
            summary_path,
            cache_entry_dir / "summary.txt",
        )
        cached_video = copy_if_exists(
            final_video,
            cache_entry_dir / "final_video.mp4",
        )

        save_ui_cached_result(
            cache_key=cache_key,
            input_hash=input_hash,
            input_type=input_type,
            file_name=uploaded_file_name,
            source_language=source_lang,
            target_language=target_lang,
            generate_tts=generate_tts,
            generate_summary=generate_summary_option,
            burn_subtitles=burn_subtitles,
            pipeline_version=PIPELINE_VERSION,
            transcript_json_path=cache_path_to_db(cached_transcript),
            translated_json_path=cache_path_to_db(cached_translation),
            audio_path=cache_path_to_db(cached_audio),
            srt_path=cache_path_to_db(cached_srt),
            summary_path=cache_path_to_db(cached_summary),
            video_path=cache_path_to_db(cached_video),
            metadata={
                "source_name": source_name,
                "target_name": target_name,
            },
        )

        progress.progress(100)
        st.session_state.is_processing = False

        render_results(
            translated_json=cached_translation,
            target_lang=target_lang,
            generate_tts=generate_tts,
            audio_path=cached_audio,
            srt_path=cached_srt,
            summary_path=cached_summary,
            final_video=cached_video,
            cache_hit=False,
        )

    except Exception as exc:
        progress.progress(0)
        st.session_state.is_processing = False
        st.error("❌ Processing failed")
        st.exception(exc)