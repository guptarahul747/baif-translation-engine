import json
import re
import shutil
import subprocess
import sys
import threading
import traceback
from html import escape
from pathlib import Path
from typing import Callable, Optional

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

# Resilient dual-fallback import for Domain Lexicon (zero visual footprint)
try:
    from app.modules.domain_lexicon import normalize_text
except ModuleNotFoundError:
    try:
        from domain_lexicon import normalize_text
    except ModuleNotFoundError:
        def normalize_text(text: str, lang: str = "mr") -> str:
            return text.strip() if text else ""

INPUTS_DIR = BASE_DIR / "storage_vault" / "inputs"
OUTPUTS_DIR = BASE_DIR / "storage_vault" / "outputs"
CACHE_DIR = BASE_DIR / "storage_vault" / "cache"

INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
init_db()

PYTHON_BIN = sys.executable

# IMPORTANT: bump this whenever ASR/translation logic or models change enough
# that old results should not be reused.
PIPELINE_VERSION = "quality-v11-consolidated-domain-lexicon"

LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}

VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "wmv", "mkv", "flv", "webm"}
AUDIO_EXTENSIONS = {"mp3", "wav", "aac", "m4a", "flac", "wma", "ogg"}
MEDIA_EXTENSIONS = sorted(VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)

# Automatic offline configuration: preserves full audio coverage (no VAD gaps),
# uses overlap context to prevent missed boundary words, and uses a beam size
# that gives a strong accuracy/speed balance on CPU.
AUTOMATIC_ASR_PROFILE = {
    "key": "automatic-balanced",
    "beam_size": 4,
    "chunk_seconds": 28,
    "context_seconds": 1,
    "use_vad": False,
}

UI_LANGUAGES = {"English": "en", "हिन्दी": "hi", "मराठी": "mr"}
LANGUAGE_LABELS = {
    "en": {"en": "English", "hi": "Hindi", "mr": "Marathi"},
    "hi": {"en": "अंग्रेज़ी", "hi": "हिंदी", "mr": "मराठी"},
    "mr": {"en": "इंग्रजी", "hi": "हिंदी", "mr": "मराठी"},
}
UI_COPY = {
    "en": {
        "setup": "⚙️ Translation Setup", "select_ui": "Select language",
        "badge": "100% offline · Open source · On premises", "hero": "BAIF Anuwad Studio",
        "tagline": "Transcribe, translate, dub and caption English, Hindi and Marathi media—entirely on BAIF infrastructure.",
        "feature_asr": "🎙️ Speech recognition", "feature_asr_detail": "Full-coverage speech-to-text",
        "feature_translation": "🌐 Regional translation", "feature_translation_detail": "English · हिन्दी · मराठी",
        "feature_outputs": "🎬 Ready-to-use outputs", "feature_outputs_detail": "Text, dubbed voice, SRT and translated video",
        "input_type": "Input Type", "media": "Audio / Video", "text": "Text",
        "upload": "Upload audio or video", "selected_file": "Selected file",
        "text_to_translate": "Text to translate", "text_placeholder": "Enter English, Hindi or Marathi text...",
        "source": "Source Language", "target": "Target Language",
        "different_languages": "Source and target languages must be different.",
        "voice": "Generate translated voice", "summary": "Generate summary",
        "burn_subtitles": "Burn subtitles into video", "start": "🚀 Start Translation",
        "translation_setup": "Choose different source/target languages and provide an input to begin.",
        "locked": "🔒 Processing in progress. Upload and configuration controls are locked until this job finishes.",
        "completed": "✅ Processing completed — controls are unlocked for the next video",
        "translated_text": "🌐 Translated Text", "translation": "Translation",
        "translation_json": "⬇️ Translation JSON", "dubbed_audio": "⬇️ Dubbed Audio",
        "subtitles": "⬇️ SRT Subtitles", "summary_heading": "📝 Summary",
        "summary_output": "Summary output", "download_summary": "⬇️ Summary",
        "translated_audio": "🔊 Translated Audio", "final_video": "🎬 Final Translated Video",
        "download_video": "⬇️ Final Video", "cache_hit": "🎯 Previous result loaded — no processing was needed.",
        "last_error": "❌ The last processing job failed", "error_details": "Show error details",
        "stop": "⏹ Stop",
    },
    "hi": {
        "setup": "⚙️ अनुवाद सेटअप", "select_ui": "भाषा चुनें",
        "badge": "100% ऑफ़लाइन · ओपन सोर्स · ऑन-प्रिमाइसेस", "hero": "BAIF अनुवाद स्टूडियो",
        "tagline": "अंग्रेज़ी, हिंदी और मराठी मीडिया का ट्रांसक्रिप्शन, अनुवाद, डबिंग और कैप्शन—पूरी तरह BAIF के ढाँचे पर।",
        "feature_asr": "🎙️ वाक् पहचान", "feature_asr_detail": "पूरे ऑडियो का टेक्स्ट रूपांतरण",
        "feature_translation": "🌐 क्षेत्रीय अनुवाद", "feature_translation_detail": "अंग्रेज़ी · हिंदी · मराठी",
        "feature_outputs": "🎬 तैयार आउटपुट", "feature_outputs_detail": "टेक्स्ट, डब आवाज़, SRT और अनुवादित वीडियो",
        "input_type": "इनपुट प्रकार", "media": "ऑडियो / वीडियो", "text": "टेक्स्ट",
        "upload": "ऑडियो या वीडियो अपलोड करें", "selected_file": "चुनी गई फ़ाइल",
        "text_to_translate": "अनुवाद के लिए टेक्स्ट", "text_placeholder": "अंग्रेज़ी, हिंदी या मराठी में टेक्स्ट लिखें...",
        "source": "स्रोत भाषा", "target": "लक्ष्य भाषा",
        "different_languages": "स्रोत और लक्ष्य भाषा अलग होनी चाहिए।",
        "voice": "अनुवादित आवाज़ बनाएँ", "summary": "सारांश बनाएँ",
        "burn_subtitles": "वीडियो में उपशीर्षक जोड़ें", "start": "🚀 अनुवाद शुरू करें",
        "translation_setup": "अलग स्रोत/लक्ष्य भाषा चुनें और शुरू करने के लिए इनपुट दें।",
        "locked": "🔒 प्रक्रिया चल रही है। काम पूरा होने तक अपलोड और सेटिंग उपलब्ध नहीं हैं।",
        "completed": "✅ प्रक्रिया पूरी हुई — अगली फ़ाइल के लिए नियंत्रण उपलब्ध हैं",
        "translated_text": "🌐 अनुवादित टेक्स्ट", "translation": "अनुवाद",
        "translation_json": "⬇️ अनुवाद JSON", "dubbed_audio": "⬇️ डब की गई ऑडियो",
        "subtitles": "⬇️ SRT उपशीर्षक", "summary_heading": "📝 सारांश",
        "summary_output": "सारांश", "download_summary": "⬇️ सारांश डाउनलोड करें",
        "translated_audio": "🔊 अनुवादित ऑडियो", "final_video": "🎬 अंतिम अनुवादित वीडियो",
        "download_video": "⬇️ वीडियो डाउनलोड करें", "cache_hit": "🎯 पिछला परिणाम तैयार है — कोई प्रक्रिया नहीं चली।",
        "last_error": "❌ पिछली प्रक्रिया पूरी नहीं हो सकी", "error_details": "त्रुटि विवरण दिखाएँ",
        "stop": "⏹ रोकें",
    },
    "mr": {
        "setup": "⚙️ भाषांतर मांडणी", "select_ui": "भाषा निवडा",
        "badge": "100% ऑफलाइन · ओपन सोर्स · ऑन-प्रिमाइसेस", "hero": "BAIF अनुवाद स्टुडिओ",
        "tagline": "इंग्रजी, हिंदी आणि मराठी माध्यमांचे लिप्यंतरण, भाषांतर, डबिंग आणि कॅप्शन—पूर्णपणे BAIF पायाभूत सुविधांवर।",
        "feature_asr": "🎙️ वाणी ओळख", "feature_asr_detail": "संपूर्ण ऑडिओचे मजकुरात रूपांतर",
        "feature_translation": "🌐 प्रादेशिक भाषांतर", "feature_translation_detail": "इंग्रजी · हिंदी · मराठी",
        "feature_outputs": "🎬 वापरासाठी तयार आउटपुट", "feature_outputs_detail": "मजकूर, डब आवाज, SRT आणि भाषांतरित व्हिडिओ",
        "input_type": "इनपुट प्रकार", "media": "ऑडिओ / व्हिडिओ", "text": "मजकूर",
        "upload": "ऑडिओ किंवा व्हिडिओ अपलोड करा", "selected_file": "निवडलेली फाईल",
        "text_to_translate": "भाषांतरासाठी मजकूर", "text_placeholder": "इंग्रजी, हिंदी किंवा मराठीत मजकूर लिहा...",
        "source": "स्त्रोत भाषा", "target": "लक्ष्य भाषा",
        "different_languages": "स्त्रोत आणि लक्ष्य भाषा वेगळी असणे आवश्यक आहे।",
        "voice": "भाषांतरित आवाज तयार करा", "summary": "सारांश तयार करा",
        "burn_subtitles": "व्हिडिओमध्ये उपशीर्षके जोडा", "start": "🚀 भाषांतर सुरू करा",
        "translation_setup": "वेगळी स्त्रोत/लक्ष्य भाषा निवडा आणि सुरू करण्यासाठी इनपुट द्या।",
        "locked": "🔒 प्रक्रिया सुरू आहे. काम पूर्ण होईपर्यंत अपलोड आणि सेटिंग्ज बंद आहेत।",
        "completed": "✅ प्रक्रिया पूर्ण झाली — पुढील फाईलसाठी नियंत्रणे उपलब्ध आहेत",
        "translated_text": "🌐 भाषांतरित मजकूर", "translation": "भाषांतर",
        "translation_json": "⬇️ भाषांतर JSON", "dubbed_audio": "⬇️ डब केलेला ऑडिओ",
        "subtitles": "⬇️ SRT उपशीर्षके", "summary_heading": "📝 सारांश",
        "summary_output": "सारांश", "download_summary": "⬇️ सारांश डाउनलोड करा",
        "translated_audio": "🔊 भाषांतरित ऑडिओ", "final_video": "🎬 अंतिम भाषांतरित व्हिडिओ",
        "download_video": "⬇️ व्हिडिओ डाउनलोड करा", "cache_hit": "🎯 मागील निकाल तयार आहे — प्रक्रिया चालवली नाही।",
        "last_error": "❌ मागील प्रक्रिया पूर्ण होऊ शकली नाही", "error_details": "त्रुटी तपशील दाखवा",
        "stop": "⏹ थांबवा",
    },
}


def t(key: str) -> str:
    return UI_COPY[st.session_state.get("ui_language", "en")].get(key, key)


st.set_page_config(
    page_title="BAIF Offline Translation Engine",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource
def get_processing_lock() -> threading.Lock:
    """Allow only one media pipeline to use the local models at a time."""
    return threading.Lock()


PROCESSING_LOCK = get_processing_lock()

for state_key, default_value in {
    "processing": False,
    "job_requested": False,
    "stop_requested": False,
    "keep_awake_process": None,
    "last_result": None,
    "last_error": None,
}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value

st.session_state.ui_language = UI_LANGUAGES.get(
    st.session_state.get("ui_language_picker", "English"), "en"
)


def request_stop() -> None:
    """Ask the running local command to stop at its next safe update."""
    st.session_state.stop_requested = True
    st.session_state.processing = False
    st.session_state.job_requested = False
    stop_keep_awake()
    st.session_state.last_result = None
    st.session_state.last_error = None
    if PROCESSING_LOCK.locked():
        PROCESSING_LOCK.release()


def start_keep_awake() -> None:
    """Prevent idle system sleep while an on-premises translation is running."""
    if sys.platform != "darwin" or st.session_state.get("keep_awake_process"):
        return
    try:
        st.session_state.keep_awake_process = subprocess.Popen(
            ["caffeinate", "-i"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError):
        st.session_state.keep_awake_process = None


def stop_keep_awake() -> None:
    process = st.session_state.get("keep_awake_process")
    if process and process.poll() is None:
        process.terminate()
    st.session_state.keep_awake_process = None


header_spacer, language_picker, stop_picker = st.columns([6.6, 1.8, 1.8])
with language_picker:
    selected_ui_language = st.selectbox(
        "Select language",
        list(UI_LANGUAGES),
        key="ui_language_picker",
        label_visibility="collapsed",
    )
st.session_state.ui_language = UI_LANGUAGES[selected_ui_language]
with stop_picker:
    st.button(
        t("stop"),
        key="stop_translation",
        use_container_width=True,
        disabled=not st.session_state.processing,
        on_click=request_stop,
        help="Stops the current translation at the next safe step.",
    )


def queue_processing() -> None:
    st.session_state.processing = True
    st.session_state.job_requested = True
    st.session_state.stop_requested = False
    start_keep_awake()
    st.session_state.last_result = None
    st.session_state.last_error = None


st.markdown(
    """
    <style>
        :root {
            --baif-red: #168a63;
            --baif-red-dark: #0a624c;
            --baif-red-deep: #063d31;
            --baif-red-soft: #ecf8f1;
            --baif-orange: #7ccf97;
            --baif-ink: #123d31;
            --baif-muted: #4d6b60;
        }
        .stApp {
            background:
                radial-gradient(circle at 90% 0%, rgba(124,207,151,.22), transparent 25rem),
                radial-gradient(circle at 68% 8%, rgba(22,138,99,.14), transparent 20rem),
                linear-gradient(180deg, #edf9f2 0%, #ffffff 46%);
        }
        [data-testid="stHeader"] {
            background: linear-gradient(90deg, #063d31, #0f694f) !important;
            border-bottom: 1px solid rgba(194,235,211,.64);
        }
        [data-testid="stStatusWidget"] {
            position: fixed !important;
            top: .3rem !important;
            right: 23rem !important;
            z-index: 100000 !important;
        }
        [data-testid="stDeployButton"] { display: none !important; }
        [data-testid="stToolbar"] {
            display: none !important;
        }
        [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        [data-testid="stDeployButton"] button,
        [data-testid="stToolbar"] button:first-child {
            min-height: 2.15rem !important;
            padding: .35rem .9rem !important;
            border: 1px solid rgba(194,235,211,.72) !important;
            border-radius: 10px !important;
            background: linear-gradient(90deg, #0a624c, #168a63) !important;
            box-shadow: 0 4px 10px rgba(6,61,49,.24);
        }
        [data-testid="stDeployButton"] button:hover,
        [data-testid="stToolbar"] button:first-child:hover {
            background: linear-gradient(90deg, #084d3c, #0e7255) !important;
        }
        /* Visible action bar above the BAIF heading. */
        section.main [data-testid="stHorizontalBlock"]:has([data-baseweb="select"]) {
            align-items: center !important;
            min-height: 3.4rem !important;
            margin: 0 0 .45rem !important;
            padding: .35rem .65rem !important;
            border: 1px solid rgba(194,235,211,.64) !important;
            border-radius: 18px !important;
            background: linear-gradient(90deg, #063d31, #0f694f) !important;
            box-shadow: 0 6px 14px rgba(6,61,49,.15) !important;
        }
        [data-testid="stMarkdownContainer"] [data-testid="stHeaderActionElements"],
        [data-testid="stMarkdownContainer"] a[href^="#"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(165deg, #063d31 0%, #0f7859 48%, #22a072 100%);
            border-right: 2px solid rgba(255,255,255,.34);
            box-shadow: 8px 0 24px rgba(6,61,49,.15);
        }
        [data-testid="stSidebarContent"] {
            padding-top: .65rem !important;
        }
        [data-testid="stSidebarContent"] > div:nth-child(2) {
            padding-top: 1.5rem !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-testid="stAlert"] p {
            color: inherit !important;
        }
        [data-baseweb="select"] > div {
            border: 1px solid rgba(255,255,255,.38) !important;
            box-shadow: 0 0 0 2px rgba(255,255,255,.08);
            background: #ffffff !important;
            color: var(--baif-ink) !important;
        }
        [data-baseweb="select"] [role="combobox"],
        [data-baseweb="select"] input {
            background: #ffffff !important;
            color: var(--baif-ink) !important;
        }
        [data-baseweb="select"] svg {
            fill: var(--baif-red-dark) !important;
        }
        [data-baseweb="select"] > div:focus-within {
            border-color: #9cdbb5 !important;
            box-shadow: 0 0 0 3px rgba(156,219,181,.22);
        }
        [role="listbox"],
        [data-baseweb="menu"],
        [data-baseweb="popover"] {
            background: #ffffff !important;
            border: 1px solid #9bcfb1 !important;
            box-shadow: 0 10px 24px rgba(6,61,49,.18) !important;
        }
        [role="option"] {
            background: #ffffff !important;
            color: var(--baif-ink) !important;
        }
        [role="option"]:hover,
        [role="option"][aria-selected="true"] {
            background: #ecf8f1 !important;
            color: var(--baif-red-dark) !important;
        }
        .baif-hero {
            padding: 2.1rem 2.3rem;
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,.56);
            outline: 4px solid rgba(22,138,99,.13);
            outline-offset: 2px;
            color: #ffffff;
            background: linear-gradient(120deg, #0a624c 0%, #168a63 52%, #49ad7d 100%);
            box-shadow: 0 18px 46px rgba(10,98,76,.27);
            margin: .25rem 0 1.2rem;
            position: relative;
            overflow: hidden;
        }
        .baif-hero:after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -70px;
            top: -115px;
            border-radius: 50%;
            background: linear-gradient(145deg, rgba(189,237,205,.54), rgba(255,255,255,.12));
        }
        .baif-hero:before {
            content: "";
            position: absolute;
            width: 150px;
            height: 150px;
            right: 120px;
            bottom: -115px;
            border-radius: 50%;
            background: rgba(255,255,255,.09);
        }
        .baif-badge {
            display: inline-block;
            padding: .32rem .72rem;
            border-radius: 999px;
            background: rgba(255,255,255,.18);
            border: 1px solid rgba(255,255,255,.42);
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
        }
        .baif-hero h1 {
            color: #ffffff;
            margin: .75rem 0 .4rem;
            font-size: clamp(2rem, 4vw, 3.15rem);
            line-height: 1.06;
        }
        .baif-hero p {
            color: rgba(255,255,255,.88);
            margin: 0;
            max-width: 760px;
            font-size: 1.02rem;
        }
        .baif-feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .85rem;
            margin-bottom: 1.25rem;
        }
        .baif-feature {
            position: relative;
            background: rgba(255,255,255,.92);
            border: 1px solid #cfe9da;
            border-top: 3px solid var(--baif-red);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 22px rgba(10,98,76,.10), 0 0 0 1px rgba(22,138,99,.05);
            transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
        }
        .baif-feature:after {
            content: "";
            position: absolute;
            top: 0;
            right: 1.05rem;
            width: 34px;
            height: 3px;
            border-radius: 0 0 6px 6px;
            background: var(--baif-orange);
        }
        .baif-feature:hover {
            transform: translateY(-3px);
            border-color: #90cbaa;
            box-shadow: 0 12px 28px rgba(10,98,76,.17), 0 0 0 2px rgba(124,207,151,.18);
        }
        .baif-feature strong { color: var(--baif-red-dark); display: block; }
        .baif-feature span { color: var(--baif-muted); font-size: .88rem; }
        div.stButton > button[kind="primary"] {
            border: 1px solid rgba(255,255,255,.72);
            color: #ffffff;
            background: linear-gradient(90deg, #168a63, #49ad7d);
            box-shadow: 0 8px 20px rgba(22,138,99,.29), 0 0 0 2px rgba(22,138,99,.14);
            font-weight: 800;
        }
        div.stButton > button[kind="primary"]:hover {
            color: #ffffff;
            background: linear-gradient(90deg, #0e7255, #31986d);
            transform: translateY(-1px);
        }
        div.stButton > button[kind="primary"]:disabled {
            color: #edf7f1;
            background: #8aa99a;
            box-shadow: none;
            cursor: not-allowed;
            transform: none;
        }
        div.stButton > button[kind="secondary"] {
            border: 1px solid #89c4a5;
            color: #ffffff;
            background: linear-gradient(90deg, #0a624c, #168a63);
            box-shadow: 0 5px 12px rgba(6,61,49,.18);
            font-weight: 700;
        }
        div.stButton > button[kind="secondary"]:hover {
            color: #ffffff;
            background: linear-gradient(90deg, #084d3c, #0e7255);
        }
        [data-testid="stDownloadButton"] > button {
            border: 1px solid #75b99a !important;
            color: #ffffff !important;
            background: linear-gradient(90deg, #063d31, #0a624c) !important;
            box-shadow: 0 5px 12px rgba(6,61,49,.20) !important;
            font-weight: 700 !important;
        }
        [data-testid="stDownloadButton"] > button:hover {
            color: #ffffff !important;
            background: linear-gradient(90deg, #042f26, #084d3c) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed rgba(54,166,113,.48);
            background: #f7fcf9;
            border-radius: 14px;
            box-shadow: inset 0 0 0 4px rgba(255,255,255,.92), 0 6px 16px rgba(22,138,99,.08);
        }
        [data-testid="stFileUploaderDropzone"] button {
            border: 1px solid rgba(194,235,211,.72) !important;
            background: linear-gradient(90deg, #0e7255, #299b6d) !important;
            color: #ffffff !important;
            box-shadow: 0 5px 12px rgba(6,61,49,.22);
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: linear-gradient(90deg, #084d3c, #137b59) !important;
            color: #ffffff !important;
        }
        [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] {
            display: none !important;
        }
        /* Hide Streamlit's built-in drop hint after selection. The selected
           filename below is the only label shown inside the white area. */
        [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzone"] > div > span,
        [data-testid="stFileUploaderDropzone"] small {
            display: none !important;
        }
        [data-testid="stFileUploaderDropzone"] button span {
            display: inline !important;
        }
        [data-testid="stFileUploaderFile"] { display: none !important; }
        .baif-selected-file {
            position: relative;
            z-index: 2;
            margin: .7rem .85rem 1rem;
            color: #ffffff;
            font-size: .9rem;
            font-weight: 750;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            pointer-events: none;
        }
        section.main h2,
        section.main h3,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3 {
            color: var(--baif-red-dark) !important;
            font-weight: 800;
        }
        [data-testid="stTextArea"] textarea {
            background: #f8fdf9 !important;
            color: var(--baif-ink) !important;
            border: 1px solid #a8d7bd !important;
            border-radius: 12px !important;
        }
        [data-testid="stTextArea"] label { color: var(--baif-red-dark) !important; }
        .baif-inline-notice {
            color: #ffffff;
            font-weight: 700;
            margin: .25rem 0 .75rem;
        }
        [data-testid="stProgress"] p {
            color: var(--baif-ink) !important;
            font-weight: 750;
        }
        [data-baseweb="progress-bar"] > div {
            background: #dcefe4 !important;
        }
        [data-baseweb="progress-bar"] > div > div > div {
            background: linear-gradient(90deg, #0e7255, #33a973 62%, #8edca7);
        }
        .baif-progress-status {
            margin: .55rem 0 1.1rem;
            color: var(--baif-ink);
            font-size: .96rem;
        }
        .baif-progress-status strong { color: var(--baif-red-dark); }
        [data-testid="stAlert"],
        [data-testid="stAlert"] > div {
            background: var(--baif-red-soft) !important;
            border: 1px solid #cce8d8 !important;
            box-shadow: 0 4px 12px rgba(22,138,99,.06);
        }
        [data-testid="stAlert"] p {
            color: var(--baif-ink) !important;
            font-weight: 600;
        }
        .baif-lock {
            padding: .8rem .9rem;
            border-radius: 12px;
            background: rgba(174,226,193,.26);
            border: 1px solid rgba(196,235,211,.62);
            color: #ffffff;
            font-size: .88rem;
            margin-bottom: .8rem;
        }
        @media (max-width: 760px) {
            .baif-feature-grid { grid-template-columns: 1fr; }
            .baif-hero { padding: 1.5rem; }
        }
    </style>
    <section class="baif-hero">
        <span class="baif-badge">__BADGE__</span>
        <h1>__HERO__</h1>
        <p>__TAGLINE__</p>
    </section>
    <section class="baif-feature-grid">
        <div class="baif-feature"><strong>__FEATURE_ASR__</strong><span>__FEATURE_ASR_DETAIL__</span></div>
        <div class="baif-feature"><strong>__FEATURE_TRANSLATION__</strong><span>__FEATURE_TRANSLATION_DETAIL__</span></div>
        <div class="baif-feature"><strong>__FEATURE_OUTPUTS__</strong><span>__FEATURE_OUTPUTS_DETAIL__</span></div>
    </section>
    """.replace("__BADGE__", t("badge")).replace("__HERO__", t("hero")).replace(
        "__TAGLINE__", t("tagline")
    ).replace("__FEATURE_ASR__", t("feature_asr")).replace(
        "__FEATURE_ASR_DETAIL__", t("feature_asr_detail")
    ).replace("__FEATURE_TRANSLATION__", t("feature_translation")).replace(
        "__FEATURE_TRANSLATION_DETAIL__", t("feature_translation_detail")
    ).replace("__FEATURE_OUTPUTS__", t("feature_outputs")).replace(
        "__FEATURE_OUTPUTS_DETAIL__", t("feature_outputs_detail")
    ),
    unsafe_allow_html=True,
)


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


def set_pipeline_progress(
    progress_bar,
    status_placeholder,
    percent: float,
    label: str,
) -> None:
    value = max(0, min(100, int(round(percent))))
    progress_bar.progress(value)
    status_placeholder.markdown(
        f'<div class="baif-progress-status"><strong>{value}% complete</strong>'
        f' · {escape(label)}</div>',
        unsafe_allow_html=True,
    )


def make_fraction_progress_callback(
    progress_bar,
    status_placeholder,
    pattern: str,
    start_percent: float,
    end_percent: float,
    label: str,
) -> Callable[[str], None]:
    compiled = re.compile(pattern)

    def update(line: str) -> None:
        match = compiled.search(line)
        if not match:
            return
        current = int(match.group(1))
        total = max(1, int(match.group(2)))
        fraction = min(1.0, current / total)
        percent = start_percent + ((end_percent - start_percent) * fraction)
        set_pipeline_progress(progress_bar, status_placeholder, percent, label)

    return update


def stream_command(
    command: list[str],
    title: str,
    on_output: Optional[Callable[[str], None]] = None,
) -> str:
    """Run a child process while retaining output only for error reporting."""
    lines = []
    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        if st.session_state.get("stop_requested", False):
            process.terminate()
            process.wait(timeout=10)
            raise InterruptedError("Translation stopped by the user.")
        line = line.rstrip("\n")
        lines.append(line)
        if on_output is not None:
            on_output(line)

    return_code = process.wait()
    output = "\n".join(lines)
    if return_code != 0:
        raise RuntimeError(f"{title} failed with exit code {return_code}.\n\n{output}")
    return output


def save_text_as_segments(text: str, lang: str = "mr") -> Path:
    cleaned = normalize_text(text, lang=lang) if text else ""
    output = OUTPUTS_DIR / "step1_whisper_output.json"
    output.write_text(
        json.dumps(
            [{"id": 1, "start": 0.0, "end": 1.0, "text": cleaned}],
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
    """Store paths relative to project root so the Desktop repo is portable."""
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
        st.success(t("cache_hit"))

    segments = json.loads(translated_json.read_text(encoding="utf-8"))
    translated_texts = [
        segment.get(f"translation_{target_lang}")
        or segment.get("translated_text")
        or ""
        for segment in segments
    ]

    st.subheader(t("translated_text"))
    st.text_area(
        t("translation"),
        value="\n".join(text for text in translated_texts if text),
        height=220,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            t("translation_json"),
            data=translated_json.read_bytes(),
            file_name=translated_json.name,
            mime="application/json",
        )
    with col2:
        if generate_tts and audio_path and audio_path.exists():
            st.download_button(
                t("dubbed_audio"),
                data=audio_path.read_bytes(),
                file_name=audio_path.name,
                mime="audio/wav",
            )
    with col3:
        if generate_tts and srt_path and srt_path.exists():
            st.download_button(
                t("subtitles"),
                data=srt_path.read_bytes(),
                file_name=srt_path.name,
                mime="text/plain",
            )

    if summary_path and summary_path.exists():
        st.subheader(t("summary_heading"))
        summary_text = summary_path.read_text(encoding="utf-8")
        st.text_area(t("summary_output"), value=summary_text, height=180)
        st.download_button(
            t("download_summary"),
            data=summary_path.read_bytes(),
            file_name=summary_path.name,
            mime="text/plain",
        )

    if generate_tts and audio_path and audio_path.exists():
        st.subheader(t("translated_audio"))
        st.audio(str(audio_path))

    if final_video and final_video.exists():
        st.subheader(t("final_video"))
        st.video(str(final_video))
        st.download_button(
            t("download_video"),
            data=final_video.read_bytes(),
            file_name=final_video.name,
            mime="video/mp4",
        )


def make_stored_result(
    *,
    translated_json: Path,
    target_lang: str,
    generate_tts: bool,
    audio_path: Optional[Path],
    srt_path: Optional[Path],
    summary_path: Optional[Path],
    final_video: Optional[Path],
    cache_hit: bool,
) -> dict:
    return {
        "translated_json": str(translated_json),
        "target_lang": target_lang,
        "generate_tts": generate_tts,
        "audio_path": str(audio_path) if audio_path else None,
        "srt_path": str(srt_path) if srt_path else None,
        "summary_path": str(summary_path) if summary_path else None,
        "final_video": str(final_video) if final_video else None,
        "cache_hit": cache_hit,
    }


def finish_processing(*, result: Optional[dict] = None, error: Optional[str] = None) -> None:
    st.session_state.last_result = result
    st.session_state.last_error = error
    st.session_state.processing = False
    st.session_state.job_requested = False
    st.session_state.stop_requested = False
    stop_keep_awake()
    if PROCESSING_LOCK.locked():
        PROCESSING_LOCK.release()
    st.rerun()


def render_stored_result(result: dict) -> None:
    render_results(
        translated_json=Path(result["translated_json"]),
        target_lang=result["target_lang"],
        generate_tts=result["generate_tts"],
        audio_path=Path(result["audio_path"]) if result.get("audio_path") else None,
        srt_path=Path(result["srt_path"]) if result.get("srt_path") else None,
        summary_path=(
            Path(result["summary_path"]) if result.get("summary_path") else None
        ),
        final_video=Path(result["final_video"]) if result.get("final_video") else None,
        cache_hit=result.get("cache_hit", False),
    )


controls_locked = bool(st.session_state.processing or PROCESSING_LOCK.locked())

with st.sidebar:
    st.header(t("setup"))

    if controls_locked:
        st.markdown(
            f'<div class="baif-lock">{t("locked")}</div>',
            unsafe_allow_html=True,
        )

    input_mode = st.radio(
        t("input_type"),
        ["media", "text"],
        format_func=t,
        disabled=controls_locked,
    )

    uploaded_path = None
    uploaded_file_name = None
    text_input = ""

    if input_mode == "media":
        uploaded_file = st.file_uploader(
            t("upload"),
            type=MEDIA_EXTENSIONS,
            help=(
                "Video: MP4, MOV, AVI, WMV, MKV, FLV, WebM | "
                "Audio: MP3, WAV, AAC, M4A, FLAC, WMA, OGG"
            ),
            disabled=controls_locked,
        )
        if uploaded_file:
            uploaded_file_name = uploaded_file.name
            uploaded_path = INPUTS_DIR / uploaded_file.name
            uploaded_path.write_bytes(uploaded_file.getbuffer())
            st.markdown(
                f'<div class="baif-selected-file">✅ {escape(uploaded_file.name)}</div>',
                unsafe_allow_html=True,
            )
    else:
        text_input = st.text_area(
            t("text_to_translate"),
            height=180,
            placeholder=t("text_placeholder"),
            disabled=controls_locked,
        )

    st.divider()
    source_lang = st.selectbox(
        t("source"),
        list(LANGUAGE_LABELS[st.session_state.ui_language]),
        index=0,
        format_func=lambda code: LANGUAGE_LABELS[st.session_state.ui_language][code],
        disabled=controls_locked,
    )
    target_lang = st.selectbox(
        t("target"),
        list(LANGUAGE_LABELS[st.session_state.ui_language]),
        index=1,
        format_func=lambda code: LANGUAGE_LABELS[st.session_state.ui_language][code],
        disabled=controls_locked,
    )
    source_name = LANGUAGE_LABELS[st.session_state.ui_language][source_lang]
    target_name = LANGUAGE_LABELS[st.session_state.ui_language][target_lang]

    if source_lang == target_lang:
        st.markdown(
            f'<div class="baif-inline-notice">⚠️ {t("different_languages")}</div>',
            unsafe_allow_html=True,
        )

    # The offline engine picks the quality/speed balance automatically.
    asr_profile = AUTOMATIC_ASR_PROFILE

    generate_tts = st.checkbox(
        t("voice"),
        value=True,
        disabled=controls_locked,
    )
    generate_summary_option = st.checkbox(
        t("summary"),
        value=True,
        disabled=controls_locked,
    )
    burn_subtitles = st.checkbox(
        t("burn_subtitles"),
        value=False,
        disabled=controls_locked,
    )

    effective_pipeline_version = f"{PIPELINE_VERSION}-{asr_profile['key']}"

    valid_input = bool(uploaded_path) if input_mode == "media" else bool(text_input.strip())
    can_start = valid_input and source_lang != target_lang

    st.button(
        t("start"),
        type="primary",
        use_container_width=True,
        disabled=controls_locked or not can_start,
        on_click=queue_processing,
    )

start = bool(st.session_state.job_requested)

if not can_start and not start and not st.session_state.last_result:
    st.info(t("translation_setup"))

if start:
    lock_acquired = PROCESSING_LOCK.acquire(blocking=False)
    if not lock_acquired:
        st.session_state.processing = False
        st.session_state.job_requested = False
        st.session_state.last_error = (
            "Another video started processing first. Please try again when it finishes."
        )
        st.rerun()

    st.session_state.job_requested = False
    progress = st.progress(0)
    status_line = st.empty()
    set_pipeline_progress(progress, status_line, 0, "Preparing your translation")

    try:
        # ------------------------------------------------------------------
        # Build exact cache key BEFORE any expensive model work.
        # ------------------------------------------------------------------
        if input_mode == "media":
            assert uploaded_path is not None
            set_pipeline_progress(
                progress,
                status_line,
                2,
                "Checking whether this file was processed before",
            )
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
            pipeline_version=effective_pipeline_version,
        )

        require_video = (
            input_mode == "media"
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
            set_pipeline_progress(
                progress,
                status_line,
                100,
                "Your previous translation is ready",
            )
            finish_processing(
                result=make_stored_result(
                translated_json=db_path_to_path(cached["translated_json_path"]),
                target_lang=target_lang,
                generate_tts=generate_tts,
                audio_path=db_path_to_path(cached.get("audio_path")),
                srt_path=db_path_to_path(cached.get("srt_path")),
                summary_path=db_path_to_path(cached.get("summary_path")),
                final_video=db_path_to_path(cached.get("video_path")),
                cache_hit=True,
                )
            )
        elif cached:
            # Row exists but one or more cached artifact files were removed.
            delete_ui_cached_result(cache_key)

        # ------------------------------------------------------------------
        # CACHE MISS: run the real pipeline.
        # ------------------------------------------------------------------
        # Stage 1: ASR or text preparation
        if input_mode == "media":
            set_pipeline_progress(
                progress, status_line, 5, "Speech recognition in progress"
            )
            stream_command(
                [
                    PYTHON_BIN,
                    str(BASE_DIR / "run_step1_asr.py"),
                    str(uploaded_path),
                    "--lang",
                    source_lang,
                    "--beam_size",
                    str(asr_profile["beam_size"]),
                    "--threads",
                    "8",
                    "--chunk-seconds",
                    str(asr_profile["chunk_seconds"]),
                    "--chunk-context-seconds",
                    str(asr_profile["context_seconds"]),
                    *(["--use-vad"] if asr_profile["use_vad"] else []),
                ],
                "ASR",
                on_output=make_fraction_progress_callback(
                    progress,
                    status_line,
                    r"Chunk\s+(\d+)/(\d+)",
                    5,
                    32,
                    "Speech recognition in progress",
                ),
            )
        else:
            set_pipeline_progress(progress, status_line, 5, "Preparing your text")
            save_text_as_segments(text_input, lang=source_lang)

        set_pipeline_progress(progress, status_line, 35, "Speech recognition complete")

        # Stage 2: Translation - ALL SIX COMBINATIONS
        set_pipeline_progress(
            progress,
            status_line,
            38,
            f"Translating from {source_name} to {target_name}",
        )
        translated_json = OUTPUTS_DIR / "step2_offline_translated.json"
        stream_command(
            [
                PYTHON_BIN,
                str(BASE_DIR / "run_step1_translation.py"),
                str(OUTPUTS_DIR / "step1_whisper_output.json"),
                target_lang,
                source_lang,
            ],
            "Translation",
        )
        set_pipeline_progress(progress, status_line, 55, "Translation complete")

        # Stage 3: TTS + SRT
        srt_path = OUTPUTS_DIR / f"video_subtitles_{target_lang}.srt"
        audio_path = OUTPUTS_DIR / f"video_dubbed_{target_lang}.wav"

        if generate_tts:
            set_pipeline_progress(
                progress, status_line, 58, "Creating the translated voice"
            )
            stream_command(
                [
                    PYTHON_BIN,
                    str(BASE_DIR / "run_step2_tts_srt.py"),
                    str(translated_json),
                    target_lang,
                ],
                "TTS/SRT",
                on_output=make_fraction_progress_callback(
                    progress,
                    status_line,
                    r"\[(\d+)/(\d+)\]",
                    58,
                    80,
                    "Creating the translated voice",
                ),
            )
        else:
            set_pipeline_progress(
                progress, status_line, 58, "Preparing subtitles and output files"
            )

        set_pipeline_progress(progress, status_line, 82, "Voice and subtitles ready")

        # Stage 4: video muxing / summary
        final_video = None
        if require_video:
            set_pipeline_progress(
                progress, status_line, 85, "Creating your translated video"
            )
            mux_cmd = [
                PYTHON_BIN,
                str(BASE_DIR / "test_video_muxing.py"),
                str(uploaded_path),
                target_lang,
            ]
            if burn_subtitles:
                mux_cmd.append("--burn-subs")
            stream_command(mux_cmd, "Video muxing")
            candidate = OUTPUTS_DIR / f"final_translated_video_{target_lang}.mp4"
            if candidate.exists():
                final_video = candidate
        else:
            set_pipeline_progress(progress, status_line, 85, "Finalizing your files")

        summary_path = None
        if generate_summary_option:
            set_pipeline_progress(progress, status_line, 94, "Creating a short summary")
            summary_path = generate_summary(translated_json, target_lang)

        set_pipeline_progress(progress, status_line, 96, "Saving your translated files")

        # ------------------------------------------------------------------
        # Save immutable copies of the artifacts into this cache entry.
        # Fixed pipeline output names can be overwritten by the next job,
        # so cached results MUST use their own directory.
        # ------------------------------------------------------------------
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
            pipeline_version=effective_pipeline_version,
            transcript_json_path=cache_path_to_db(cached_transcript),
            translated_json_path=cache_path_to_db(cached_translation),
            audio_path=cache_path_to_db(cached_audio),
            srt_path=cache_path_to_db(cached_srt),
            summary_path=cache_path_to_db(cached_summary),
            video_path=cache_path_to_db(cached_video),
            metadata={
                "source_name": source_name,
                "target_name": target_name,
                "asr_profile": asr_profile["key"],
                "audio_timing_mode": "continuous-download/source-video",
            },
        )

        set_pipeline_progress(progress, status_line, 100, "Finished — your files are ready")

        finish_processing(
            result=make_stored_result(
                translated_json=cached_translation,
                target_lang=target_lang,
                generate_tts=generate_tts,
                audio_path=cached_audio,
                srt_path=cached_srt,
                summary_path=cached_summary,
                final_video=cached_video,
                cache_hit=False,
            )
        )

    except InterruptedError:
        set_pipeline_progress(progress, status_line, 0, "Translation stopped")
        finish_processing()
    except Exception as exc:
        set_pipeline_progress(progress, status_line, 0, "Processing could not be completed")
        finish_processing(error=f"{exc}\n\n{traceback.format_exc()}")
    finally:
        # Streamlit's Stop control interrupts execution outside the normal
        # exception path. Always release UI state so the next render is clean.
        if st.session_state.processing:
            st.session_state.processing = False
            st.session_state.job_requested = False
            st.session_state.stop_requested = False
            stop_keep_awake()
            st.session_state.last_result = None
            st.session_state.last_error = None
            if PROCESSING_LOCK.locked():
                PROCESSING_LOCK.release()
            st.rerun()

if not start and st.session_state.last_error:
    st.error(t("last_error"))
    with st.expander(t("error_details"), expanded=True):
        st.code(st.session_state.last_error, language="text")
elif not start and st.session_state.last_result:
    st.success(t("completed"))
    render_stored_result(st.session_state.last_result)