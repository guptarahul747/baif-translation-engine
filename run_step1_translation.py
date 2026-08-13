import json
import re
import sys
from pathlib import Path
import ctranslate2

try:
    import sentencepiece as spm
except ImportError:
    print("❌ 'sentencepiece' module is missing. Please run: pip install sentencepiece")
    sys.exit(1)

# Setup base paths
BASE_DIR = Path(__file__).resolve().parent
VAULT_DIR = BASE_DIR / "local_model_vault"
DEFAULT_INPUT = BASE_DIR / "storage_vault" / "outputs" / "step1_whisper_output.json"
OUTPUT_JSON = BASE_DIR / "storage_vault" / "outputs" / "step2_offline_translated.json"


def find_model_dir(search_dir: Path) -> Path:
    """Finds directory containing CTranslate2 model files."""
    matches = list(search_dir.rglob("model.bin"))
    if matches:
        return matches[0].parent
    matches = list(search_dir.rglob("config.json"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(f"Could not locate model files inside {search_dir}")


def find_spm_file(search_dir: Path) -> Path:
    """Finds SentencePiece source model file (model.SRC)."""
    src_matches = list(search_dir.rglob("model.SRC"))
    if src_matches:
        return src_matches[0]
    
    for match in search_dir.rglob("*"):
        if match.is_file() and match.name.endswith((".SRC", ".model", ".spm")) and not match.name.endswith((".bin", ".json")):
            return match
    return None


def clean_sentencepiece_text(text: str) -> str:
    """Replaces SentencePiece whitespace markers (\u2581 / ▁) with standard spaces."""
    if not text:
        return ""
    # Replace SentencePiece meta-character \u2581 and literal ▁ with standard space
    cleaned = text.replace("\u2581", " ").replace("▁", " ")
    # Collapse multiple spaces into one and trim
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def translate_whisper_json(input_json_path: str, target_lang: str = "hi"):
    print("==================================================")
    print(f"🌐 STEP 1: TRANSLATING WHISPER JSON ({target_lang.upper()})")
    print("==================================================")

    input_path = Path(input_json_path).resolve()

    if not input_path.exists():
        print(f"❌ Input file not found at: {input_path}")
        print("Usage: python3 run_step1_translation.py <path_to_whisper_json> [hi|mr]")
        sys.exit(1)

    print(f"📄 Loading Whisper JSON from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"📖 Loaded {len(segments)} segments from Whisper transcript.\n")

    # 1. Locate CTranslate2 model directory
    it2_vault = VAULT_DIR / "indictrans2"
    it2_dir = find_model_dir(it2_vault)
    print(f"⚡ Loading IndicTrans2 CTranslate2 model from: {it2_dir.relative_to(BASE_DIR)}")
    translator = ctranslate2.Translator(str(it2_dir), device="cpu", inter_threads=4)

    # 2. Locate SentencePiece model file
    spm_file = find_spm_file(it2_vault)
    if not spm_file:
        print(f"\n❌ Could not find SentencePiece file inside: {it2_vault}")
        sys.exit(1)

    print(f"🔤 Loading SentencePiece Processor from: {spm_file.relative_to(BASE_DIR)}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_file))

    # Language tags for IndicTrans2
    src_tag = "eng_Latn"
    tgt_tag = "hin_Deva" if target_lang == "hi" else "mar_Deva"

    # 3. Tokenize inputs with SentencePiece & Language Tags
    print(f"\n🔄 Tokenizing and translating to '{target_lang}'...")
    tokenized_inputs = []
    
    for seg in segments:
        raw_text = seg["text"].strip()
        subwords = sp.encode(raw_text, out_type=str)
        formatted_tokens = [src_tag] + subwords + [tgt_tag]
        tokenized_inputs.append(formatted_tokens)

    # 4. Batch Translate via CTranslate2
    results = translator.translate_batch(tokenized_inputs)

    # 5. Decode outputs & sanitize SentencePiece spaces
    translated_segments = []
    print("\n--------------------------------------------------")
    print(f"TRANSLATION PREVIEW ({target_lang.upper()})")
    print("--------------------------------------------------")

    for seg, res in zip(segments, results):
        hypothesis = res.hypotheses[0]
        
        # Filter out special language tags before decoding
        clean_tokens = [tok for tok in hypothesis if tok not in [src_tag, tgt_tag, "</s>", "<s>"]]
        raw_translation = sp.decode(clean_tokens)
        
        # Post-process SentencePiece meta-character replacement
        translated_text = clean_sentencepiece_text(raw_translation)
        
        seg_entry = dict(seg)
        seg_entry[f"translation_{target_lang}"] = translated_text
        translated_segments.append(seg_entry)

        print(f"[{seg['start']:>6.2f}s -> {seg['end']:>6.2f}s]")
        print(f" ├─ EN: {seg['text']}")
        print(f" └─ {target_lang.upper()}: {translated_text}\n")

    print("--------------------------------------------------")

    # 6. Save Output JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)

    print(f"✅ Translation Complete! Saved clean translations to: {OUTPUT_JSON.relative_to(BASE_DIR)}")
    print("==================================================")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_INPUT)
    target_language = sys.argv[2] if len(sys.argv) > 2 else "hi"
    
    translate_whisper_json(input_file, target_lang=target_language)