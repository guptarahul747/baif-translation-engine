import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import timedelta
import streamlit as st
import numpy as np
import soundfile as sf
import srt
import sentencepiece as spm
import ctranslate2
from faster_whisper import WhisperModel
import sherpa_onnx

# -------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VAULT_DIR = BASE_DIR / "local_model_vault"
TEMP_DIR = BASE_DIR / "storage_vault" / "web_cache"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

LANG_MAP = {
    "English": {"code": "en", "tag": "eng_Latn", "voice": "english/en_US-lessac-medium.onnx", "tokens": "english/tokens.txt"},
    "Hindi": {"code": "hi", "tag": "hin_Deva", "voice": "hindi/hi_IN-pratham-medium.onnx", "tokens": "hindi/tokens.txt"},
    "Marathi": {"code": "mr", "tag": "mar_Deva", "voice": "marathi/mr_IN-google-medium.onnx", "tokens": "marathi/tokens.txt"}
}

# -------------------------------------------------------------
# Native IndicTrans2 Engine (Direct + Pivot)
# -------------------------------------------------------------
class NativeIndicTranslator:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.translator = ctranslate2.Translator(str(model_dir), device="cpu")

        src_spm = model_dir / "vocab" / "model.SRC"
        tgt_spm = model_dir / "vocab" / "model.TGT"
        if not src_spm.exists():
            src_spm = next(model_dir.rglob("model.SRC"))
            tgt_spm = next(model_dir.rglob("model.TGT"))

        self.sp_src = spm.SentencePieceProcessor()
        self.sp_src.load(str(src_spm))
        self.sp_tgt = spm.SentencePieceProcessor()
        self.sp_tgt.load(str(tgt_spm))

        src_vocab_path = model_dir / "source_vocabulary.json"
        if not src_vocab_path.exists():
            src_vocab_path = next(model_dir.rglob("source_vocabulary.json"))
        with open(src_vocab_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            self.src_vocab = set(v_data if isinstance(v_data, list) else v_data.keys())

    def get_tag(self, lang_key: str) -> str:
        code = LANG_MAP[lang_key]["tag"]
        for c in [code, f"_{code}_", f"__{code}__"]:
            if c in self.src_vocab:
                return c
        return code

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_tag = self.get_tag(src_lang)
        tgt_tag = self.get_tag(tgt_lang)

        tokens = self.sp_src.encode(text, out_type=str)
        input_tokens = [src_tag, tgt_tag] + tokens

        results = self.translator.translate_batch([input_tokens])
        out_tokens = results[0].hypotheses[0]

        special_tags = {src_tag, tgt_tag, "<s>", "</s>", "<unk>", "<pad>", "eng_Latn", "hin_Deva", "mar_Deva", "_eng_Latn_", "_hin_Deva_", "_mar_Deva_"}
        clean_tokens = [t for t in out_tokens if t not in special_tags and not (t.startswith("__") and t.endswith("__"))]

        try:
            return self.sp_tgt.decode_pieces(clean_tokens)
        except Exception:
            return "".join(clean_tokens).replace(" ", " ").strip()

# -------------------------------------------------------------
# Cached Engines
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Whisper ASR Engine...")
def load_whisper():
    return WhisperModel(str(VAULT_DIR / "whisper"), device="cpu", compute_type="int8")

@st.cache_resource(show_spinner="Loading Translation Engines...")
def load_translators():
    return {
        "en-indic": NativeIndicTranslator(VAULT_DIR / "indictrans2" / "en-indic"),
        "indic-en": NativeIndicTranslator(VAULT_DIR / "indictrans2" / "indic-en")
    }

def get_tts_engine(tgt_lang: str):
    info = LANG_MAP[tgt_lang]
    model_path = VAULT_DIR / "tts" / info["voice"]
    tokens_path = VAULT_DIR / "tts" / info["tokens"]
    data_dir = VAULT_DIR / "tts" / info["voice"].split("/")[0] / "espeak-ng-data"

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_path),
                tokens=str(tokens_path),
                data_dir=str(data_dir) if data_dir.exists() else "",
            ),
            provider="cpu",
            num_threads=4,
        )
    )
    return sherpa_onnx.OfflineTts(tts_config)

def translate_pipeline(text: str, src_lang: str, tgt_lang: str, engines: dict) -> str:
    if src_lang == tgt_lang or not text.strip():
        return text
    src_code, tgt_code = LANG_MAP[src_lang]["code"], LANG_MAP[tgt_lang]["code"]

    if src_code == "en" and tgt_code in ["hi", "mr"]:
        return engines["en-indic"].translate(text, src_lang, tgt_lang)
    elif src_code in ["hi", "mr"] and tgt_code == "en":
        return engines["indic-en"].translate(text, src_lang, tgt_lang)
    elif src_code in ["hi", "mr"] and tgt_code in ["hi", "mr"]:
        intermediate_en = engines["indic-en"].translate(text, src_lang, "English")
        return engines["en-indic"].translate(intermediate_en, "English", tgt_lang)
    return text

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.set_page_config(page_title="BAIF Multilingual Video Dubber", page_icon="🎙️", layout="wide")

st.title("🎙️ BAIF Multilingual Video Translation & Dubbing Engine")
st.caption("100% Offline AI Pipeline supporting bidirectional English, Hindi, and Marathi")

col1, col2 = st.columns(2)
with col1:
    src_lang = st.selectbox("Source Language (Video Audio)", ["English", "Hindi", "Marathi"], index=0)
with col2:
    tgt_lang = st.selectbox("Target Language (Dubbed Audio)", ["Marathi", "Hindi", "English"], index=0)

uploaded_file = st.file_uploader("Upload Video (MP4, MKV, MOV)", type=["mp4", "mkv", "mov"])

if uploaded_file and st.button("🚀 Process & Dub Video", type="primary"):
    if src_lang == tgt_lang:
        st.warning("Please choose different Source and Target languages.")
        st.stop()

    video_input_path = TEMP_DIR / uploaded_file.name
    with open(video_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    status = st.status("Executing Pipeline...", expanded=True)

    # 1. Extract Audio
    status.write("🎵 Extracting audio from video...")
    audio_wav_path = TEMP_DIR / "extracted.wav"
    subprocess.run(["ffmpeg", "-y", "-i", str(video_input_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_wav_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Whisper ASR
    status.write("🗣️ Transcribing speech with Whisper...")
    whisper_model = load_whisper()
    segments, _ = whisper_model.transcribe(str(audio_wav_path), language=LANG_MAP[src_lang]["code"])
    raw_segments = list(segments)

    # 3. Translation
    status.write(f"🌐 Translating segments ({src_lang} ➔ {tgt_lang})...")
    translators = load_translators()
    processed = []
    for seg in raw_segments:
        tr_text = translate_pipeline(seg.text, src_lang, tgt_lang, translators)
        processed.append({"start": seg.start, "end": seg.end, "src": seg.text, "tgt": tr_text})

    # 4. Generate Subtitles
    subs = [srt.Subtitle(index=i, start=timedelta(seconds=d["start"]), end=timedelta(seconds=d["end"]), content=d["tgt"]) for i, d in enumerate(processed, 1)]
    srt_out = TEMP_DIR / f"subtitles_{LANG_MAP[tgt_lang]['code']}.srt"
    with open(srt_out, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))

    # 5. TTS Audio Synthesis
    status.write(f"🎙️ Synthesizing {tgt_lang} audio tracks...")
    tts_engine = get_tts_engine(tgt_lang)
    sample_rate = 22050
    audio_segments = []
    for d in processed:
        audio = tts_engine.generate(d["tgt"], sid=0, speed=1.0)
        if len(audio.samples) > 0:
            sample_rate = audio.sample_rate
            audio_segments.append((d["start"], audio.samples))

    # Timestamp-synchronized stitching
    max_end = max(d["end"] for d in processed)
    total_len = int(max_end * sample_rate) + (sample_rate * 2)
    master_audio = np.zeros(total_len, dtype=np.float32)

    for start_sec, samples in audio_segments:
        s_idx = int(start_sec * sample_rate)
        e_idx = s_idx + len(samples)
        if e_idx > len(master_audio):
            master_audio = np.pad(master_audio, (0, e_idx - len(master_audio)))
        master_audio[s_idx:e_idx] = samples

    dubbed_audio_path = TEMP_DIR / "dubbed_audio.wav"
    sf.write(str(dubbed_audio_path), master_audio, sample_rate)

    # 6. Muxing
    status.write("🎬 Muxing dubbed video...")
    final_video = TEMP_DIR / f"dubbed_{LANG_MAP[tgt_lang]['code']}.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_input_path),
        "-i", str(dubbed_audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", str(final_video)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    status.update(label="✅ Dubbing Complete!", state="complete", expanded=False)

    # Output Results
    st.divider()
    st.subheader("📺 Dubbed Video Output")
    st.video(str(final_video))

    c1, c2 = st.columns(2)
    with c1:
        with open(final_video, "rb") as f:
            st.download_button("⬇️ Download Dubbed Video (.mp4)", f, file_name=f"dubbed_{LANG_MAP[tgt_lang]['code']}.mp4")
    with c2:
        with open(srt_out, "rb") as f:
            st.download_button("⬇️ Download Subtitles (.srt)", f, file_name=f"subtitles_{LANG_MAP[tgt_lang]['code']}.srt")

    st.subheader("📝 Transcript & Translations")
    st.dataframe(processed, use_container_width=True)