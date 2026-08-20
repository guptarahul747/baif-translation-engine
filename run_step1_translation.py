import json
import re
import sys
from pathlib import Path
import ctranslate2

try:
    import sentencepiece as spm
except ImportError:
    print("❌ 'sentencepiece' missing. Run: pip install sentencepiece")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
VAULT_DIR = BASE_DIR / "local_model_vault" / "indictrans2"
DEFAULT_INPUT = BASE_DIR / "storage_vault" / "outputs" / "step1_whisper_output.json"
OUTPUT_JSON = BASE_DIR / "storage_vault" / "outputs" / "step2_offline_translated.json"

FLORES_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva"
}

def resolve_paths(src_lang: str, tgt_lang: str):
    """Dynamically routes between en-indic and indic-en/indic-indic model folders."""
    direction = "en-indic" if src_lang == "en" else ("indic-en" if tgt_lang == "en" else "indic-indic")
    
    target_dir = VAULT_DIR / f"{direction}-1b-ct2"
    if not target_dir.exists():
        target_dir = VAULT_DIR / direction
    if not target_dir.exists():
        target_dir = VAULT_DIR

    model_matches = list(target_dir.rglob("model.bin"))
    if not model_matches:
        raise FileNotFoundError(f"Could not find model.bin inside {target_dir}")
    model_dir = model_matches[0].parent

    spm_matches = list(target_dir.rglob("*.model")) + list(target_dir.rglob("*.SRC")) + list(target_dir.rglob("*.spm"))
    spm_matches = [f for f in spm_matches if not f.name.endswith((".bin", ".json"))]

    src_spm = next((f for f in spm_matches if "SRC" in f.name or "src" in f.name), spm_matches[0])
    tgt_spm = next((f for f in spm_matches if "TGT" in f.name or "tgt" in f.name), spm_matches[-1])

    return model_dir, src_spm, tgt_spm

def clean_sentencepiece_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\u2581", " ").replace("▁", " ")
    return re.sub(r"\s+", " ", cleaned).strip()

def translate_whisper_json(input_json_path: str, src_lang: str = "en", target_lang: str = "hi"):
    src_tag = FLORES_TAGS.get(src_lang.lower(), "eng_Latn")
    tgt_tag = FLORES_TAGS.get(target_lang.lower(), "hin_Deva")

    input_path = Path(input_json_path).resolve()
    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    model_dir, src_spm_path, tgt_spm_path = resolve_paths(src_lang, target_lang)
    translator = ctranslate2.Translator(str(model_dir), device="cpu", inter_threads=4)

    sp_src = spm.SentencePieceProcessor()
    sp_src.load(str(src_spm_path))
    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(str(tgt_spm_path))

    tokenized_inputs, target_prefixes = [], []
    for seg in segments:
        subwords = sp_src.encode_as_pieces(seg["text"].strip())
        formatted_tokens = [src_tag, tgt_tag] + subwords + ["</s>"]
        tokenized_inputs.append(formatted_tokens)
        target_prefixes.append([tgt_tag])

    results = translator.translate_batch(
        tokenized_inputs,
        target_prefix=target_prefixes,
        beam_size=5,
        max_decoding_length=256
    )

    translated_segments = []
    for seg, res in zip(segments, results):
        clean_tokens = [tok for tok in res.hypotheses[0] if tok not in [src_tag, tgt_tag, "</s>", "<s>", "<unk>"]]
        translated_text = clean_sentencepiece_text(sp_tgt.decode_pieces(clean_tokens))
        
        seg_entry = dict(seg)
        seg_entry[f"translation_{target_lang}"] = translated_text
        translated_segments.append(seg_entry)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_INPUT)
    target_language = sys.argv[2] if len(sys.argv) > 2 else "hi"
    source_language = sys.argv[3] if len(sys.argv) > 3 else "en"
    
    translate_whisper_json(input_file, src_lang=source_language, target_lang=target_language)   