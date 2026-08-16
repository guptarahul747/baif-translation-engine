import os
import sys
import json
import time
from pathlib import Path
import sentencepiece as spm
import ctranslate2

# -------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VAULT_DIR = BASE_DIR / "local_model_vault"
OUTPUT_DIR = BASE_DIR / "storage_vault" / "test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard FLORES-200 Language Codes used by IndicTrans2
LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva"
}

# Test Sentences
SAMPLE_INPUTS = {
    "en": "Agriculture is the foundation of rural development and food security.",
    "hi": "कृषि ग्रामीण विकास और खाद्य सुरक्षा की नींव है।",
    "mr": "शेती हा ग्रामीण विकासाचा आणि अन्न सुरक्षेचा पाया आहे."
}

# All 6 Bidirectional Translation Combinations
TRANSLATION_PAIRS = [
    ("en", "hi", "Direct (en-indic)"),
    ("en", "mr", "Direct (en-indic)"),
    ("hi", "en", "Direct (indic-en)"),
    ("mr", "en", "Direct (indic-en)"),
    ("hi", "mr", "Pivot via English (indic-en -> en-indic)"),
    ("mr", "hi", "Pivot via English (indic-en -> en-indic)")
]

def print_header(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

# -------------------------------------------------------------
# Native IndicTrans2 Engine (Using model.SRC & model.TGT)
# -------------------------------------------------------------
class NativeIndicTranslator:
    """
    Offline IndicTrans2 engine that loads SentencePiece model.SRC for encoding
    and model.TGT for decoding directly without huggingface transformers.
    """
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.translator = ctranslate2.Translator(str(model_dir), device="cpu")

        # 1. Locate Source & Target SentencePiece models
        src_spm = model_dir / "vocab" / "model.SRC"
        tgt_spm = model_dir / "vocab" / "model.TGT"

        if not src_spm.exists():
            # Fallback search if nested
            src_spm = next(model_dir.rglob("model.SRC"))
            tgt_spm = next(model_dir.rglob("model.TGT"))

        self.sp_src = spm.SentencePieceProcessor()
        self.sp_src.load(str(src_spm))

        self.sp_tgt = spm.SentencePieceProcessor()
        self.sp_tgt.load(str(tgt_spm))

        # 2. Load Source Vocabulary to match language tag formatting
        self.src_vocab = set()
        src_vocab_path = model_dir / "source_vocabulary.json"
        if not src_vocab_path.exists():
            src_vocab_path = next(model_dir.rglob("source_vocabulary.json"))
        
        with open(src_vocab_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            if isinstance(v_data, list):
                self.src_vocab = set(v_data)
            elif isinstance(v_data, dict):
                self.src_vocab = set(v_data.keys())

    def get_lang_tag(self, lang_key: str) -> str:
        code = LANG_CODES.get(lang_key, lang_key)
        candidates = [code, f"_{code}_", f"__{code}__", f"__{code}"]
        for c in candidates:
            if c in self.src_vocab:
                return c
        return code

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_tag = self.get_lang_tag(src_lang)
        tgt_tag = self.get_lang_tag(tgt_lang)

        # 1. Source SentencePiece tokenization
        tokens = self.sp_src.encode(text, out_type=str)

        # 2. Format input tokens with language tags: [src_lang, tgt_lang] + tokens
        input_tokens = [src_tag, tgt_tag] + tokens

        # 3. CTranslate2 batch translation
        results = self.translator.translate_batch([input_tokens])
        out_tokens = results[0].hypotheses[0]

        # 4. Filter special tokens and tags
        special_tags = {
            src_tag, tgt_tag, "<s>", "</s>", "<unk>", "<pad>",
            "eng_Latn", "hin_Deva", "mar_Deva",
            "_eng_Latn_", "_hin_Deva_", "_mar_Deva_"
        }
        clean_tokens = [t for t in out_tokens if t not in special_tags and not (t.startswith("__") and t.endswith("__"))]

        # 5. Target SentencePiece detokenization
        try:
            return self.sp_tgt.decode_pieces(clean_tokens)
        except Exception:
            return "".join(clean_tokens).replace(" ", " ").strip()

# -------------------------------------------------------------
# 1. Verify Local Model Files on Disk
# -------------------------------------------------------------
def verify_vault_files():
    print_header("1. Checking Local Model Vault Files")
    
    en_indic_dir = VAULT_DIR / "indictrans2" / "en-indic"
    indic_en_dir = VAULT_DIR / "indictrans2" / "indic-en"
    
    status = {
        "Whisper ASR": (VAULT_DIR / "whisper" / "model.bin").exists(),
        "IndicTrans2 (en-indic)": (en_indic_dir / "model.bin").exists() and (en_indic_dir / "vocab" / "model.SRC").exists(),
        "IndicTrans2 (indic-en)": (indic_en_dir / "model.bin").exists() and (indic_en_dir / "vocab" / "model.SRC").exists(),
        "TTS Hindi (Pratham)": (VAULT_DIR / "tts" / "hindi" / "tokens.txt").exists(),
        "TTS English (Lessac)": (VAULT_DIR / "tts" / "english" / "tokens.txt").exists(),
        "TTS Marathi (Google)": (VAULT_DIR / "tts" / "marathi" / "mr_IN-google-medium.onnx").exists(),
        "TTS Marathi eSpeak Data": (VAULT_DIR / "tts" / "marathi" / "espeak-ng-data").exists()
    }

    all_ok = True
    for name, ok in status.items():
        if ok:
            print(f"  ✅ {name:<25} : Ready")
        else:
            print(f"  ❌ {name:<25} : MISSING")
            all_ok = False
            
    return all_ok, en_indic_dir, indic_en_dir

# -------------------------------------------------------------
# 2. Verify Faster-Whisper ASR
# -------------------------------------------------------------
def verify_whisper() -> bool:
    print_header("2. Verifying Faster-Whisper ASR Engine")
    try:
        from faster_whisper import WhisperModel
        whisper_dir = str(VAULT_DIR / "whisper")
        t0 = time.time()
        _ = WhisperModel(whisper_dir, device="cpu", compute_type="int8")
        print(f"  ✅ Whisper ASR loaded successfully in {time.time() - t0:.2f}s")
        return True
    except Exception as e:
        print(f"  ❌ Whisper loading failed: {e}")
        return False

# -------------------------------------------------------------
# 3. Test All 6 Bidirectional Translation Combinations
# -------------------------------------------------------------
def verify_translations(en_indic_dir: Path, indic_en_dir: Path) -> bool:
    print_header("3. Verifying All 6 Bidirectional Translation Combinations")
    try:
        print(f"  ⏳ Initializing en-indic engine...")
        engine_en_indic = NativeIndicTranslator(en_indic_dir)
        print("  ↳ ✅ en-indic initialized (vocab/model.SRC & model.TGT loaded)")

        print(f"  ⏳ Initializing indic-en engine...")
        engine_indic_en = NativeIndicTranslator(indic_en_dir)
        print("  ↳ ✅ indic-en initialized (vocab/model.SRC & model.TGT loaded)")

        def translate_pair(text: str, src: str, tgt: str) -> str:
            # Direct English -> Indic
            if src == "en" and tgt in ["hi", "mr"]:
                return engine_en_indic.translate(text, src, tgt)
            # Direct Indic -> English
            elif src in ["hi", "mr"] and tgt == "en":
                return engine_indic_en.translate(text, src, tgt)
            # Pivot Indic -> Indic (via English)
            elif src in ["hi", "mr"] and tgt in ["hi", "mr"]:
                en_intermediate = engine_indic_en.translate(text, src, "en")
                return engine_en_indic.translate(en_intermediate, "en", tgt)
            return text

        all_passed = True
        for src, tgt, mode in TRANSLATION_PAIRS:
            input_text = SAMPLE_INPUTS[src]
            t0 = time.time()
            try:
                out = translate_pair(input_text, src, tgt)
                print(f"\n  🌐 [{src.upper()} ➔ {tgt.upper()}] ({mode}) - {time.time() - t0:.2f}s")
                print(f"     Input  : {input_text}")
                print(f"     Output : {out}")
            except Exception as e:
                print(f"  ❌ [{src.upper()} ➔ {tgt.upper()}] Failed: {e}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"  ❌ Translation engine failed: {e}")
        return False

# -------------------------------------------------------------
# 4. Test TTS Audio Synthesis (English, Hindi, Marathi)
# -------------------------------------------------------------
def verify_tts() -> bool:
    print_header("4. Verifying Sherpa-ONNX TTS Synthesis (en, hi, mr)")
    try:
        import sherpa_onnx
        import soundfile as sf

        tts_configs = {
            "hi": {
                "model": VAULT_DIR / "tts" / "hindi" / "hi_IN-pratham-medium.onnx",
                "tokens": VAULT_DIR / "tts" / "hindi" / "tokens.txt",
                "data_dir": VAULT_DIR / "tts" / "hindi" / "espeak-ng-data",
                "text": "कृषि क्षेत्र में आपका स्वागत है।"
            },
            "en": {
                "model": VAULT_DIR / "tts" / "english" / "en_US-lessac-medium.onnx",
                "tokens": VAULT_DIR / "tts" / "english" / "tokens.txt",
                "data_dir": VAULT_DIR / "tts" / "english" / "espeak-ng-data",
                "text": "Welcome to the agricultural translation system."
            },
            "mr": {
                "model": VAULT_DIR / "tts" / "marathi" / "mr_IN-google-medium.onnx",
                "tokens": VAULT_DIR / "tts" / "marathi" / "tokens.txt",
                "data_dir": VAULT_DIR / "tts" / "marathi" / "espeak-ng-data",
                "text": "शेती विषयक कार्यक्रमात आपले स्वागत आहे."
            }
        }

        all_passed = True
        for lang, cfg in tts_configs.items():
            try:
                t_config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=str(cfg["model"]),
                            tokens=str(cfg["tokens"]),
                            data_dir=str(cfg["data_dir"]) if cfg["data_dir"].exists() else "",
                        ),
                        provider="cpu",
                        num_threads=4,
                    )
                )

                if not t_config.validate():
                    print(f"  ❌ TTS Config Invalid for '{lang}'")
                    all_passed = False
                    continue

                engine = sherpa_onnx.OfflineTts(t_config)
                audio = engine.generate(cfg["text"], sid=0, speed=1.0)

                out_wav = OUTPUT_DIR / f"verify_tts_{lang}.wav"
                sf.write(str(out_wav), audio.samples, audio.sample_rate)
                duration = len(audio.samples) / audio.sample_rate
                print(f"  🎙️ [{lang.upper()} Voice] Synthesized '{cfg['text']}' ➔ {out_wav.name} ({duration:.2f}s)")
            except Exception as e:
                print(f"  ❌ TTS Generation failed for '{lang}': {e}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"  ❌ TTS setup failed: {e}")
        return False

# -------------------------------------------------------------
# Main Verification Runner
# -------------------------------------------------------------
def main():
    print("=" * 65)
    print("🧪 BAIF TRANSLATION ENGINE - MULTI-MODEL SYSTEM VERIFICATION")
    print("=" * 65)

    files_ok, en_indic_dir, indic_en_dir = verify_vault_files()
    whisper_ok = verify_whisper()
    nmt_ok = verify_translations(en_indic_dir, indic_en_dir)
    tts_ok = verify_tts()

    print_header("FINAL VERIFICATION SUMMARY")
    print(f"  • Model Vault Files      : {'✅ PASSED' if files_ok else '⚠️ INCOMPLETE'}")
    print(f"  • Whisper ASR Engine     : {'✅ PASSED' if whisper_ok else '❌ FAILED'}")
    print(f"  • 6/6 NMT Combinations   : {'✅ PASSED' if nmt_ok else '❌ FAILED'}")
    print(f"  • 3/3 TTS Voice Engines  : {'✅ PASSED' if tts_ok else '❌ FAILED'}")
    print("=" * 65)

    if files_ok and whisper_ok and nmt_ok and tts_ok:
        print("🎉 ALL SYSTEMS GO: Full bidirectional translation and dubbing ready!\n")
    else:
        print("⚠️ Check the errors indicated above.\n")

if __name__ == "__main__":
    main()