import json
import re
import sys
import argparse
from pathlib import Path

import ctranslate2

try:
    import sentencepiece as spm
except ImportError:
    print("❌ 'sentencepiece' missing. Run:")
    print("   pip install sentencepiece")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent

VAULT_DIR = BASE_DIR / "local_model_vault" / "indictrans2"

DEFAULT_INPUT = (
    BASE_DIR
    / "storage_vault"
    / "outputs"
    / "step1_whisper_output.json"
)

OUTPUT_JSON = (
    BASE_DIR
    / "storage_vault"
    / "outputs"
    / "step2_offline_translated.json"
)


# ---------------------------------------------------------
# FLORES-200 LANGUAGE TAGS
# ---------------------------------------------------------

FLORES_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
}


# ---------------------------------------------------------
# MODEL PATH RESOLUTION
# ---------------------------------------------------------

def resolve_paths(src_lang: str, tgt_lang: str):
    """
    Resolve the correct IndicTrans2 CTranslate2 model.

    Supported translation directions:

        English -> Indian language
        Indian language -> English

    Marathi -> Hindi is handled as:

        Marathi -> English -> Hindi

    The model vault may contain either of these layouts:

        local_model_vault/indictrans2/
        ├── en-indic-1b-ct2/
        │   └── ctranslate2_model/
        │       ├── model.bin
        │       └── vocab/
        │           ├── model.SRC
        │           └── model.TGT

    OR:

        local_model_vault/indictrans2/
        └── indic-en/
            └── indic-en-1b-ct2/
                └── ctranslate2_model/
                    ├── model.bin
                    └── vocab/
                        ├── model.SRC
                        └── model.TGT
    """

    src_lang = src_lang.lower()
    tgt_lang = tgt_lang.lower()

    # -----------------------------------------------------
    # Determine model direction
    # -----------------------------------------------------

    if src_lang == "en":

        direction = "en-indic"

    elif tgt_lang == "en":

        direction = "indic-en"

    else:

        raise ValueError(
            f"Direct {src_lang} -> {tgt_lang} translation "
            f"is not supported.\n"
            f"Use English as the intermediate language."
        )

    print(f"🔎 Translation model direction: {direction}")

    # -----------------------------------------------------
    # Possible model root directories
    # -----------------------------------------------------

    search_roots = [
        VAULT_DIR / direction,
        VAULT_DIR / f"{direction}-1b-ct2",
    ]

    print("🔍 Searching model directories:")

    for root in search_roots:
        print(f"   {root}")

    # -----------------------------------------------------
    # Search recursively for model.bin
    # -----------------------------------------------------

    model_candidates = []

    for root in search_roots:

        if not root.exists():
            continue

        for model_file in root.rglob("model.bin"):
            model_candidates.append(model_file)

    # Remove duplicates while preserving order
    model_candidates = list(dict.fromkeys(model_candidates))

    if not model_candidates:

        raise FileNotFoundError(
            f"\n❌ Could not find model.bin for {direction}.\n\n"
            f"Searched under:\n"
            + "\n".join(
                f"   {root}"
                for root in search_roots
            )
        )

    print("🔎 Found model files:")

    for model_file in model_candidates:
        print(f"   {model_file}")

    # -----------------------------------------------------
    # Find model.bin with matching vocab files
    # -----------------------------------------------------

    for model_file in model_candidates:

        model_dir = model_file.parent

        vocab_dir = model_dir / "vocab"

        src_spm = vocab_dir / "model.SRC"
        tgt_spm = vocab_dir / "model.TGT"

        if src_spm.exists() and tgt_spm.exists():

            print()
            print("✅ IndicTrans2 model found")
            print(f"   Model : {model_dir}")
            print(f"   SRC   : {src_spm}")
            print(f"   TGT   : {tgt_spm}")

            return (
                model_dir,
                src_spm,
                tgt_spm,
            )

    # -----------------------------------------------------
    # model.bin exists but vocabulary is missing
    # -----------------------------------------------------

    raise FileNotFoundError(
        f"\n❌ Found model.bin for {direction}, "
        f"but matching vocabulary files were not found.\n\n"
        f"Expected:\n"
        f"   model.SRC\n"
        f"   model.TGT\n\n"
        f"Found model files:\n"
        + "\n".join(
            f"   {model_file}"
            for model_file in model_candidates
        )
    )


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_sentencepiece_text(text: str) -> str:

    if not text:
        return ""

    # SentencePiece underscore character
    text = text.replace("\u2581", " ")

    # Literal ▁ character
    text = text.replace("▁", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------
# SINGLE TRANSLATION
# ---------------------------------------------------------

def translate_segments(
    segments,
    src_lang: str,
    tgt_lang: str,
):
    """
    Translate one direction.

    Supported:

        en -> hi
        en -> mr
        mr -> en
        hi -> en
    """

    src_lang = src_lang.lower()
    tgt_lang = tgt_lang.lower()

    if src_lang not in FLORES_TAGS:
        raise ValueError(
            f"Unsupported source language: {src_lang}"
        )

    if tgt_lang not in FLORES_TAGS:
        raise ValueError(
            f"Unsupported target language: {tgt_lang}"
        )

    src_tag = FLORES_TAGS[src_lang]
    tgt_tag = FLORES_TAGS[tgt_lang]

    print()
    print("--------------------------------------------------")
    print(
        f"🌐 TRANSLATION: "
        f"{src_lang.upper()} → {tgt_lang.upper()}"
    )
    print("--------------------------------------------------")

    # -----------------------------------------------------
    # Resolve model
    # -----------------------------------------------------

    (
        model_dir,
        src_spm_path,
        tgt_spm_path,
    ) = resolve_paths(
        src_lang,
        tgt_lang,
    )

    # -----------------------------------------------------
    # Load CTranslate2 model
    # -----------------------------------------------------

    print()
    print("🧠 Loading CTranslate2 model...")
    print(f"   Device: CPU")
    print(f"   Model : {model_dir}")

    translator = ctranslate2.Translator(
        str(model_dir),
        device="cpu",
        inter_threads=4,
    )

    print("✅ CTranslate2 model loaded")

    # -----------------------------------------------------
    # Load SentencePiece models
    # -----------------------------------------------------

    print("🔤 Loading SentencePiece models...")

    sp_src = spm.SentencePieceProcessor()
    sp_src.load(str(src_spm_path))

    sp_tgt = spm.SentencePieceProcessor()
    sp_tgt.load(str(tgt_spm_path))

    print("✅ SentencePiece models loaded")

    # -----------------------------------------------------
    # Prepare input
    # -----------------------------------------------------

    tokenized_inputs = []
    target_prefixes = []
    valid_segments = []

    for seg in segments:

        text = seg.get("text", "").strip()

        if not text:
            continue

        subwords = sp_src.encode_as_pieces(text)

        formatted_tokens = (
            [src_tag, tgt_tag]
            + subwords
            + ["</s>"]
        )

        tokenized_inputs.append(
            formatted_tokens
        )

        target_prefixes.append(
            [tgt_tag]
        )

        valid_segments.append(seg)

    if not tokenized_inputs:

        print("⚠️ No valid segments found for translation.")

        return []

    print()
    print(
        f"📦 Segments to translate: "
        f"{len(tokenized_inputs)}"
    )

    # -----------------------------------------------------
    # Translation
    # -----------------------------------------------------

    print("🚀 Starting translation...")
    print()

    results = translator.translate_batch(
        tokenized_inputs,
        target_prefix=target_prefixes,
        beam_size=5,
        max_decoding_length=256,
    )

    # -----------------------------------------------------
    # Process results
    # -----------------------------------------------------

    translated_segments = []

    for index, (seg, res) in enumerate(
        zip(valid_segments, results),
        start=1,
    ):

        clean_tokens = [
            token
            for token in res.hypotheses[0]
            if token not in [
                src_tag,
                tgt_tag,
                "</s>",
                "<s>",
                "<unk>",
            ]
        ]

        translated_text = clean_sentencepiece_text(
            sp_tgt.decode_pieces(
                clean_tokens
            )
        )

        entry = dict(seg)

        entry[
            f"translation_{tgt_lang}"
        ] = translated_text

        translated_segments.append(entry)

        print(
            f"[{seg['start']:>7.2f}s -> "
            f"{seg['end']:>7.2f}s]"
        )

        print(
            f"  [{index}/{len(valid_segments)}]"
        )

        print(
            f"  SRC: {seg['text']}"
        )

        print(
            f"  OUT: {translated_text}"
        )

        print()

    return translated_segments


# ---------------------------------------------------------
# MARATHI → ENGLISH → HINDI
# ---------------------------------------------------------

def translate_marathi_to_hindi(segments):
    """
    Marathi -> English -> Hindi.

    This is used because the current IndicTrans2
    models in the local vault support:

        Marathi -> English
        English -> Hindi

    but do not directly perform:

        Marathi -> Hindi
    """

    print()
    print("==================================================")
    print("🇮🇳 MARATHI → ENGLISH → HINDI")
    print("==================================================")

    # -----------------------------------------------------
    # STEP A
    # Marathi -> English
    # -----------------------------------------------------

    print()
    print("📌 STEP A: Marathi → English")

    mr_to_en = translate_segments(
        segments,
        src_lang="mr",
        tgt_lang="en",
    )

    if not mr_to_en:

        raise RuntimeError(
            "❌ Marathi → English produced no translations."
        )

    print()
    print(
        f"✅ Marathi → English completed "
        f"({len(mr_to_en)} segments)"
    )

    # -----------------------------------------------------
    # STEP B
    # English -> Hindi
    # -----------------------------------------------------

    print()
    print("📌 STEP B: English → Hindi")

    english_segments = []

    for seg in mr_to_en:

        english_segments.append(
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["translation_en"],
            }
        )

    en_to_hi = translate_segments(
        english_segments,
        src_lang="en",
        tgt_lang="hi",
    )

    if not en_to_hi:

        raise RuntimeError(
            "❌ English → Hindi produced no translations."
        )

    print()
    print(
        f"✅ English → Hindi completed "
        f"({len(en_to_hi)} segments)"
    )

    # -----------------------------------------------------
    # Map Hindi translations back to original segments
    # -----------------------------------------------------

    hindi_by_id = {
        seg["id"]: seg["translation_hi"]
        for seg in en_to_hi
    }

    final_segments = []

    for seg in mr_to_en:

        entry = dict(seg)

        entry["translation_hi"] = (
            hindi_by_id.get(
                seg["id"],
                "",
            )
        )

        final_segments.append(entry)

    return final_segments


# ---------------------------------------------------------
# MAIN TRANSLATION FUNCTION
# ---------------------------------------------------------

def translate_whisper_json(
    input_json_path: str,
    src_lang: str = "en",
    target_lang: str = "hi",
):
    """
    Translate Whisper JSON.

    Examples:

        English -> Hindi:

            source = en
            target = hi

        Marathi -> English:

            source = mr
            target = en

        Marathi -> Hindi:

            source = mr
            target = hi

            Automatically:

                Marathi -> English -> Hindi
    """

    input_path = Path(
        input_json_path
    ).resolve()

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not input_path.exists():

        raise FileNotFoundError(
            f"❌ Input JSON does not exist:\n"
            f"{input_path}"
        )

    print()
    print(
        f"📄 Input JSON: {input_path}"
    )

    # -----------------------------------------------------
    # Load Whisper JSON
    # -----------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:

        segments = json.load(f)

    if not isinstance(
        segments,
        list,
    ):

        raise ValueError(
            "❌ Whisper JSON must contain "
            "a list of segments."
        )

    print(
        f"📊 Whisper segments loaded: "
        f"{len(segments)}"
    )

    # -----------------------------------------------------
    # Normalize languages
    # -----------------------------------------------------

    src_lang = src_lang.lower()
    target_lang = target_lang.lower()

    # -----------------------------------------------------
    # Validate languages
    # -----------------------------------------------------

    if src_lang not in FLORES_TAGS:

        raise ValueError(
            f"Unsupported source language: "
            f"{src_lang}"
        )

    if target_lang not in FLORES_TAGS:

        raise ValueError(
            f"Unsupported target language: "
            f"{target_lang}"
        )

    # -----------------------------------------------------
    # Special pipeline:
    #
    # Marathi -> English -> Hindi
    # -----------------------------------------------------

    if (
        src_lang == "mr"
        and target_lang == "hi"
    ):

        translated_segments = (
            translate_marathi_to_hindi(
                segments
            )
        )

    else:

        translated_segments = (
            translate_segments(
                segments,
                src_lang=src_lang,
                tgt_lang=target_lang,
            )
        )

    # -----------------------------------------------------
    # Validate translation result
    # -----------------------------------------------------

    if not translated_segments:

        raise RuntimeError(
            "❌ Translation returned no segments."
        )

    # -----------------------------------------------------
    # Ensure output directory exists
    # -----------------------------------------------------

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Save translation JSON
    # -----------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            translated_segments,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print()
    print("==================================================")
    print("✅ TRANSLATION COMPLETED")
    print("==================================================")
    print(
        f"📄 Output: {OUTPUT_JSON}"
    )
    print(
        f"📊 Translated segments: "
        f"{len(translated_segments)}"
    )

    return OUTPUT_JSON


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Offline IndicTrans2 translation pipeline. "
            "Marathi → Hindi automatically uses "
            "Marathi → English → Hindi."
        )
    )

    # -----------------------------------------------------
    # Input JSON
    # -----------------------------------------------------

    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Whisper JSON input file",
    )

    # -----------------------------------------------------
    # Target language
    # -----------------------------------------------------

    parser.add_argument(
        "target_language",
        nargs="?",
        default="hi",
        choices=[
            "en",
            "hi",
            "mr",
        ],
        help="Target language",
    )

    # -----------------------------------------------------
    # Source language
    # -----------------------------------------------------

    parser.add_argument(
        "source_language",
        nargs="?",
        default="en",
        choices=[
            "en",
            "hi",
            "mr",
        ],
        help="Source language",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Run translation
    # -----------------------------------------------------

    translate_whisper_json(
        args.input_file,
        src_lang=args.source_language,
        target_lang=args.target_language,
    )