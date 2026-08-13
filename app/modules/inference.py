import json
import re
import sys
import datetime
import srt
import soundfile as sf
import numpy as np
from pathlib import Path
import ctranslate2
import sherpa_onnx

try:
    import sentencepiece as spm
except ImportError:
    spm = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VAULT_DIR = BASE_DIR / "local_model_vault"


def find_file(directory: Path, pattern: str) -> Path:
    if not directory.exists():
        return None
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None


def find_model_dir(search_dir: Path) -> Path:
    matches = list(search_dir.rglob("model.bin"))
    if matches:
        return matches[0].parent
    matches = list(search_dir.rglob("config.json"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(f"Could not locate model inside {search_dir}")


def normalize_hindi_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("▁", " ").replace("\u2581", " ")
    number_map = {
        "10,000": "दस हजार", "10000": "दस हजार",
        "12,000": "बारह हजार", "12000": "बारह हजार",
        "1700": "सत्रह सौ", "19वीं": "उन्नीसवीं",
        "20वीं": "बीसवीं", "19th": "उन्नीसवीं",
        "20th": "बीसवीं", "19": "उन्नीस", "20": "बीस",
    }
    for num_str, hindi_word in number_map.items():
        cleaned = cleaned.replace(num_str, hindi_word)
    cleaned = re.sub(r"[।॥\"'\-_!?:;,()\[\]{}]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


class PipelineInferenceEngine:
    def __init__(self):
        self.nmt_translator = None
        self.sp_processor = None
        self.tts_engines = {}

    def load_nmt(self):
        """Loads IndicTrans2 and SentencePiece processor."""
        if self.nmt_translator is None:
            it2_vault = VAULT_DIR / "indictrans2"
            it2_dir = find_model_dir(it2_vault)
            self.nmt_translator = ctranslate2.Translator(str(it2_dir), device="cpu", inter_threads=4)

            spm_file = find_file(it2_vault, "model.SRC") or find_file(it2_vault, "*.model") or find_file(it2_vault, "*.spm")
            if not spm_file:
                raise FileNotFoundError(f"SentencePiece model not found in {it2_vault}")
            
            self.sp_processor = spm.SentencePieceProcessor()
            self.sp_processor.load(str(spm_file))

    def translate_segments(self, segments: list, target_lang: str = "hi") -> list:
        """Batch translates text segments into target Indic language."""
        self.load_nmt()

        src_tag = "eng_Latn"
        tgt_tag = "hin_Deva" if target_lang == "hi" else "mar_Deva"

        tokenized_inputs = []
        for seg in segments:
            raw_text = seg["text"].strip()
            subwords = self.sp_processor.encode(raw_text, out_type=str)
            tokenized_inputs.append([src_tag] + subwords + [tgt_tag])

        results = self.nmt_translator.translate_batch(tokenized_inputs)

        translated_segments = []
        for seg, res in zip(segments, results):
            hypothesis = res.hypotheses[0]
            clean_tokens = [tok for tok in hypothesis if tok not in [src_tag, tgt_tag, "</s>", "<s>"]]
            raw_translation = self.sp_processor.decode(clean_tokens)
            translated_text = normalize_hindi_text(raw_translation)

            seg_entry = dict(seg)
            seg_entry[f"translation_{target_lang}"] = translated_text
            translated_segments.append(seg_entry)

        return translated_segments

    def get_tts_engine(self, target_lang: str = "hi") -> sherpa_onnx.OfflineTts:
        """Loads Sherpa-ONNX VITS engine with dynamic espeak-ng phonemization."""
        if target_lang in self.tts_engines:
            return self.tts_engines[target_lang]

        folder_name = "hindi" if target_lang == "hi" else "marathi"
        tts_dir = VAULT_DIR / "tts" / folder_name

        onnx_file = find_file(tts_dir, "*.onnx")
        tokens_file = find_file(tts_dir, "tokens.txt")
        espeak_matches = list(tts_dir.rglob("espeak-ng-data"))
        espeak_dir = espeak_matches[0] if espeak_matches else (tts_dir / "espeak-ng-data")

        vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(onnx_file),
            tokens=str(tokens_file),
            lexicon="",  # Must be empty string to use dynamic espeak G2P
            data_dir=str(espeak_dir) if espeak_dir.exists() else "",
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0
        )

        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits_config,
            num_threads=4,
            provider="cpu"
        )

        engine = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=model_config))
        self.tts_engines[target_lang] = engine
        return engine

    def synthesize_speech(self, segments: list, target_lang: str = "hi") -> np.ndarray:
        """Synthesizes dubbed audio array from translated segments."""
        tts_engine = self.get_tts_engine(target_lang)
        lang_key = f"translation_{target_lang}"
        audio_clips = []

        for seg in segments:
            raw_text = seg.get(lang_key) or seg.get("text")
            clean_text = normalize_hindi_text(raw_text)
            if clean_text:
                audio = tts_engine.generate(clean_text, sid=0, speed=1.0)
                samples = np.array(audio.samples, dtype=np.float32)
                if samples.size > 0:
                    audio_clips.append(samples)

        return np.concatenate(audio_clips) if audio_clips else np.array([], dtype=np.float32)


# Global Singleton Instance
inference_engine = PipelineInferenceEngine()