import json
import re
import sys
import argparse
import time
import os
from pathlib import Path

import ctranslate2


# =========================================================
# OPTIONAL: IndicTrans2 Indic -> Indic dependencies
# =========================================================

try:
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )
    from IndicTransToolkit.processor import IndicProcessor

except ImportError:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    IndicProcessor = None


# =========================================================
# SentencePiece
# =========================================================

try:
    import sentencepiece as spm

except ImportError:
    print("❌ 'sentencepiece' is missing.")
    print()
    print("Run:")
    print("   pip install sentencepiece")
    sys.exit(1)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VAULT_DIR = (
    BASE_DIR
    / "local_model_vault"
    / "indictrans2"
)


# =========================================================
# LOCAL INDIC -> INDIC MODEL
# =========================================================

INDIC_INDIC_MODEL_DIR = (
    VAULT_DIR
    / "indic-indic-1B"
)


# =========================================================
# MODEL CACHE
# =========================================================

_INDIC_INDIC_TOKENIZER = None
_INDIC_INDIC_MODEL = None
_INDIC_INDIC_PROCESSOR = None
_INDIC_INDIC_DEVICE = None


# =========================================================
# CONFIGURATION
# =========================================================

# IMPORTANT:
# Keep this small for Apple Silicon / MPS stability.
INDIC_BATCH_SIZE = 4

# Maximum generated sequence length.
INDIC_MAX_LENGTH = 256

# Beam size.
INDIC_NUM_BEAMS = 5


# =========================================================
# HF CACHE
# =========================================================

HF_CACHE_DIR = (
    BASE_DIR
    / "storage_vault"
    / "hf_model_cache"
)


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = (
    BASE_DIR
    / "storage_vault"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# DEFAULT FILES
# =========================================================

DEFAULT_INPUT = (
    OUTPUT_DIR
    / "step1_whisper_output.json"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "step2_offline_translated.json"
)


# =========================================================
# FLORES-200 LANGUAGE TAGS
# =========================================================

FLORES_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
}


# =========================================================
# FILE VALIDATION HELPER
# =========================================================

def file_ok(path: Path) -> bool:
    """
    Return True when the supplied path exists,
    is a file and is not empty.
    """

    try:
        return (
            path.is_file()
            and path.stat().st_size > 0
        )

    except OSError:
        return False


# =========================================================
# TIMER HELPERS
# =========================================================

def elapsed_seconds(start_time: float) -> float:
    return time.time() - start_time


def format_elapsed(seconds: float) -> str:
    """
    Format seconds as:
        12.4s
        1m 12.4s
        1h 02m 12.4s
    """

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return (
            f"{minutes}m "
            f"{remaining_seconds:04.1f}s"
        )

    hours = int(minutes // 60)
    minutes = minutes % 60

    return (
        f"{hours}h "
        f"{minutes:02d}m "
        f"{remaining_seconds:04.1f}s"
    )


# =========================================================
# MPS SYNCHRONIZATION
# =========================================================

def synchronize_mps():
    """
    MPS operations are asynchronous.
    Synchronization makes timing and error reporting
    more reliable.
    """

    if torch is None:
        return

    try:

        if (
            hasattr(
                torch,
                "mps"
            )
            and torch.mps.is_available()
        ):

            torch.mps.synchronize()

    except Exception:
        pass


# =========================================================
# MODEL DIRECTION
# =========================================================

def get_model_direction(
    src_lang: str,
    tgt_lang: str,
) -> str:
    """
    Determine which IndicTrans2 model is required.

    Direct routes:

        EN -> HI
        EN -> MR

        HI -> EN
        MR -> EN

    Indic -> Indic:

        MR -> HI
        HI -> MR

    use the local IndicTrans2 Indic -> Indic 1B model.
    """

    src_lang = src_lang.lower()
    tgt_lang = tgt_lang.lower()

    if src_lang == "en":
        return "en-indic"

    if tgt_lang == "en":
        return "indic-en"

    return "indic-indic-1B"


# =========================================================
# FIND MODEL.BIN
# =========================================================

def find_model_bin(
    direction: str,
) -> Path:

    direction_root = (
        VAULT_DIR
        / direction
    )

    if not direction_root.exists():

        raise FileNotFoundError(
            f"\n❌ Model directory does not exist:\n"
            f"   {direction_root}"
        )

    candidates = sorted(
        direction_root.rglob(
            "model.bin"
        )
    )

    if not candidates:

        raise FileNotFoundError(
            f"\n❌ model.bin was not found "
            f"for {direction}.\n\n"
            f"Searched:\n"
            f"   {direction_root}"
        )

    direct_model = (
        direction_root
        / "model.bin"
    )

    if direct_model.exists():
        return direct_model

    return candidates[0]


# =========================================================
# FIND TOKENIZER FILES
# =========================================================

def find_tokenizer_files(
    direction: str,
):

    model_root = (
        VAULT_DIR
        / direction
    )

    cache_root = (
        HF_CACHE_DIR
        / direction
    )

    print()
    print(
        "🔤 Searching tokenizer files..."
    )

    print(
        f"   Local model root : "
        f"{model_root}"
    )

    print(
        f"   HF cache root    : "
        f"{cache_root}"
    )

    vocab_candidates = []

    # -----------------------------------------------------
    # Local vocab
    # -----------------------------------------------------

    vocab_candidates.append(
        model_root
        / "vocab"
    )

    # -----------------------------------------------------
    # HF cache vocab
    # -----------------------------------------------------

    vocab_candidates.append(
        cache_root
        / "vocab"
    )

    # -----------------------------------------------------
    # Recursive local search
    # -----------------------------------------------------

    if model_root.exists():

        for path in model_root.rglob(
            "vocab"
        ):

            if path.is_dir():
                vocab_candidates.append(
                    path
                )

    # -----------------------------------------------------
    # Recursive cache search
    # -----------------------------------------------------

    if cache_root.exists():

        for path in cache_root.rglob(
            "vocab"
        ):

            if path.is_dir():
                vocab_candidates.append(
                    path
                )

    # -----------------------------------------------------
    # Remove duplicates
    # -----------------------------------------------------

    unique_candidates = []

    for path in vocab_candidates:

        try:
            path = path.resolve()

        except OSError:
            continue

        if path not in unique_candidates:
            unique_candidates.append(path)

    # -----------------------------------------------------
    # Find matching pair
    # -----------------------------------------------------

    for vocab_dir in unique_candidates:

        src_spm = (
            vocab_dir
            / "model.SRC"
        )

        tgt_spm = (
            vocab_dir
            / "model.TGT"
        )

        if (
            src_spm.exists()
            and tgt_spm.exists()
        ):

            print(
                "   ✅ Tokenizer pair found"
            )

            print(
                f"      Directory : "
                f"{vocab_dir}"
            )

            print(
                f"      model.SRC : "
                f"{src_spm}"
            )

            print(
                f"      model.TGT : "
                f"{tgt_spm}"
            )

            return (
                src_spm,
                tgt_spm,
            )

    # -----------------------------------------------------
    # Detailed diagnostics
    # -----------------------------------------------------

    found_src = []
    found_tgt = []

    search_locations = [
        model_root,
        cache_root,
        HF_CACHE_DIR,
    ]

    for root in search_locations:

        if not root.exists():
            continue

        found_src.extend(
            root.rglob(
                "model.SRC"
            )
        )

        found_tgt.extend(
            root.rglob(
                "model.TGT"
            )
        )

    found_src = list(
        dict.fromkeys(
            str(
                p.resolve()
            )
            for p in found_src
        )
    )

    found_tgt = list(
        dict.fromkeys(
            str(
                p.resolve()
            )
            for p in found_tgt
        )
    )

    error = (
        f"\n❌ Tokenizer files could not "
        f"be resolved for {direction}.\n\n"
        f"Expected both:\n"
        f"   model.SRC\n"
        f"   model.TGT\n\n"
        f"Search locations:\n"
        +
        "\n".join(
            f"   {root}"
            for root in search_locations
        )
    )

    if found_src:

        error += (
            "\n\nFound model.SRC files:\n"
            +
            "\n".join(
                f"   {p}"
                for p in found_src
            )
        )

    if found_tgt:

        error += (
            "\n\nFound model.TGT files:\n"
            +
            "\n".join(
                f"   {p}"
                for p in found_tgt
            )
        )

    raise FileNotFoundError(
        error
    )


# =========================================================
# RESOLVE COMPLETE CT2 MODEL
# =========================================================

def resolve_paths(
    src_lang: str,
    tgt_lang: str,
):

    direction = get_model_direction(
        src_lang,
        tgt_lang,
    )

    print()
    print(
        f"🔎 Translation model direction: "
        f"{direction}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model_file = find_model_bin(
        direction
    )

    model_dir = (
        model_file.parent
    )

    print()
    print(
        "🔍 Found model:"
    )

    print(
        f"   model.bin : "
        f"{model_file}"
    )

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    (
        src_spm,
        tgt_spm,
    ) = find_tokenizer_files(
        direction
    )

    print()
    print(
        "✅ Complete IndicTrans2 "
        "model configuration found"
    )

    print(
        f"   Model directory : "
        f"{model_dir}"
    )

    print(
        f"   Model file      : "
        f"{model_file}"
    )

    print(
        f"   SRC tokenizer   : "
        f"{src_spm}"
    )

    print(
        f"   TGT tokenizer   : "
        f"{tgt_spm}"
    )

    return (
        model_dir,
        src_spm,
        tgt_spm,
    )


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_sentencepiece_text(
    text: str,
) -> str:

    if not text:
        return ""

    text = text.replace(
        "\u2581",
        " "
    )

    text = text.replace(
        "▁",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CTRANSLATE2 TRANSLATION
# =========================================================

def translate_segments(
    segments,
    src_lang: str,
    tgt_lang: str,
):

    src_lang = src_lang.lower()
    tgt_lang = tgt_lang.lower()

    if src_lang not in FLORES_TAGS:
        raise ValueError(
            f"Unsupported source language: "
            f"{src_lang}"
        )

    if tgt_lang not in FLORES_TAGS:
        raise ValueError(
            f"Unsupported target language: "
            f"{tgt_lang}"
        )

    if src_lang == tgt_lang:
        raise ValueError(
            "Source and target languages "
            "must be different."
        )

    src_tag = FLORES_TAGS[
        src_lang
    ]

    tgt_tag = FLORES_TAGS[
        tgt_lang
    ]

    print()
    print("-" * 50)

    print(
        f"🌐 TRANSLATION: "
        f"{src_lang.upper()} → "
        f"{tgt_lang.upper()}"
    )

    print("-" * 50)

    (
        model_dir,
        src_spm_path,
        tgt_spm_path,
    ) = resolve_paths(
        src_lang,
        tgt_lang,
    )

    # -----------------------------------------------------
    # Load CTranslate2
    # -----------------------------------------------------

    print()
    print(
        "🧠 Loading CTranslate2 model..."
    )

    print(
        "   Device: CPU"
    )

    print(
        f"   Model : {model_dir}"
    )

    start_time = time.time()

    translator = (
        ctranslate2.Translator(
            str(model_dir),
            device="cpu",
            inter_threads=4,
        )
    )

    print(
        f"✅ CTranslate2 model loaded "
        f"in {elapsed_seconds(start_time):.2f}s"
    )

    # -----------------------------------------------------
    # SentencePiece
    # -----------------------------------------------------

    print()
    print(
        "🔤 Loading SentencePiece models..."
    )

    sp_src = (
        spm.SentencePieceProcessor()
    )

    sp_src.load(
        str(src_spm_path)
    )

    sp_tgt = (
        spm.SentencePieceProcessor()
    )

    sp_tgt.load(
        str(tgt_spm_path)
    )

    print(
        "✅ model.SRC loaded"
    )

    print(
        "✅ model.TGT loaded"
    )

    # -----------------------------------------------------
    # Prepare batch
    # -----------------------------------------------------

    tokenized_inputs = []
    target_prefixes = []
    valid_segments = []

    for seg in segments:

        text = str(
            seg.get(
                "text",
                ""
            )
        ).strip()

        if not text:
            continue

        subwords = (
            sp_src.encode_as_pieces(
                text
            )
        )

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

        valid_segments.append(
            seg
        )

    if not tokenized_inputs:

        print(
            "⚠️ No valid segments found."
        )

        return []

    print()
    print(
        f"📦 Segments to translate: "
        f"{len(tokenized_inputs)}"
    )

    print(
        "🚀 Starting CTranslate2 inference..."
    )

    start_time = time.time()

    results = (
        translator.translate_batch(
            tokenized_inputs,
            target_prefix=target_prefixes,
            beam_size=5,
            max_decoding_length=256,
        )
    )

    print(
        f"✅ Translation inference "
        f"completed in "
        f"{elapsed_seconds(start_time):.2f}s"
    )

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

    translated_segments = []

    total = len(valid_segments)

    for index, (
        seg,
        result,
    ) in enumerate(
        zip(
            valid_segments,
            results,
        ),
        start=1,
    ):

        if not result.hypotheses:

            translated_text = ""

        else:

            output_tokens = (
                result.hypotheses[0]
            )

            clean_tokens = []

            for token in output_tokens:

                if token in {
                    src_tag,
                    tgt_tag,
                    "</s>",
                    "<s>",
                    "<unk>",
                    "<pad>",
                }:
                    continue

                clean_tokens.append(
                    token
                )

            try:

                translated_text = (
                    sp_tgt.decode_pieces(
                        clean_tokens
                    )
                )

            except Exception:

                translated_text = (
                    "".join(
                        clean_tokens
                    )
                )

            translated_text = (
                clean_sentencepiece_text(
                    translated_text
                )
            )

        entry = dict(seg)

        entry[
            f"translation_{tgt_lang}"
        ] = translated_text

        translated_segments.append(
            entry
        )

    return translated_segments


# =========================================================
# LOAD LOCAL INDIC -> INDIC 1B
# =========================================================

def load_indic_indic_model(
    force_cpu=False,
):

    global _INDIC_INDIC_TOKENIZER
    global _INDIC_INDIC_MODEL
    global _INDIC_INDIC_PROCESSOR
    global _INDIC_INDIC_DEVICE

    # -----------------------------------------------------
    # Already loaded
    # -----------------------------------------------------

    if (
        not force_cpu
        and _INDIC_INDIC_MODEL is not None
        and _INDIC_INDIC_TOKENIZER is not None
        and _INDIC_INDIC_PROCESSOR is not None
    ):

        return (
            _INDIC_INDIC_TOKENIZER,
            _INDIC_INDIC_MODEL,
            _INDIC_INDIC_PROCESSOR,
            _INDIC_INDIC_DEVICE,
        )

    # -----------------------------------------------------
    # CPU reload requested
    # -----------------------------------------------------

    if force_cpu:

        print()
        print(
            "🔄 Reinitializing IndicTrans2 "
            "on CPU..."
        )

        _INDIC_INDIC_TOKENIZER = None
        _INDIC_INDIC_MODEL = None
        _INDIC_INDIC_PROCESSOR = None
        _INDIC_INDIC_DEVICE = None

        if torch is not None:

            try:

                if hasattr(
                    torch,
                    "mps",
                ):

                    torch.mps.empty_cache()

            except Exception:
                pass

    # -----------------------------------------------------
    # Dependency validation
    # -----------------------------------------------------

    if (
        torch is None
        or AutoTokenizer is None
        or AutoModelForSeq2SeqLM is None
        or IndicProcessor is None
    ):

        raise RuntimeError(
            "IndicTrans2 Indic → Indic dependencies "
            "are missing.\n\n"
            "Install them in the active venv:\n\n"
            "  pip install torch transformers "
            "IndicTransToolkit"
        )

    # -----------------------------------------------------
    # Model directory
    # -----------------------------------------------------

    if not INDIC_INDIC_MODEL_DIR.exists():

        raise FileNotFoundError(
            "IndicTrans2 Indic → Indic model "
            "directory was not found:\n\n"
            f"  {INDIC_INDIC_MODEL_DIR}\n\n"
            "Run the Indic-Indic downloader first."
        )

    print()
    print(
        "🔍 Checking IndicTrans2 "
        "Indic → Indic model files..."
    )

    print(
        f"   Directory: "
        f"{INDIC_INDIC_MODEL_DIR}"
    )

    # -----------------------------------------------------
    # Required configuration files
    # -----------------------------------------------------

    required = [
        "config.json",
        "configuration_indictrans.py",
        "modeling_indictrans.py",
        "tokenization_indictrans.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    missing = []

    for filename in required:

        path = (
            INDIC_INDIC_MODEL_DIR
            / filename
        )

        if not file_ok(path):

            missing.append(
                filename
            )

    # -----------------------------------------------------
    # Weight files
    # -----------------------------------------------------

    weight_files = []

    weight_files.extend(
        INDIC_INDIC_MODEL_DIR.glob(
            "*.safetensors"
        )
    )

    weight_files.extend(
        INDIC_INDIC_MODEL_DIR.glob(
            "*.bin"
        )
    )

    valid_weight_files = [
        p
        for p in weight_files
        if file_ok(p)
    ]

    # -----------------------------------------------------
    # Validate model
    # -----------------------------------------------------

    if (
        missing
        or not valid_weight_files
    ):

        details = []

        if missing:

            details.append(
                "Missing files:\n  "
                +
                "\n  ".join(
                    missing
                )
            )

        if not valid_weight_files:

            details.append(
                "No valid model weight "
                "file found (*.bin / *.safetensors)."
            )

        raise RuntimeError(
            "❌ IndicTrans2 Indic → Indic "
            "model is incomplete.\n\n"
            +
            "\n\n".join(
                details
            )
        )

    print(
        "   ✅ Required model files found"
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    if (
        not force_cpu
        and torch is not None
        and hasattr(
            torch.backends,
            "mps"
        )
        and torch.backends.mps.is_available()
    ):

        device = "mps"
        dtype = torch.float16

    else:

        device = "cpu"
        dtype = torch.float32

    print()
    print(
        "🧠 Loading IndicTrans2 "
        "Indic → Indic 1B..."
    )

    print(
        f"   Model : "
        f"{INDIC_INDIC_MODEL_DIR}"
    )

    print(
        f"   Device: "
        f"{device.upper()}"
    )

    print(
        f"   Dtype : "
        f"{dtype}"
    )

    print(
        "   Mode  : Local files only"
    )

    start_time = time.time()

    try:

        # -------------------------------------------------
        # Tokenizer
        # -------------------------------------------------

        print()
        print(
            "🔤 Loading tokenizer..."
        )

        tokenizer = (
            AutoTokenizer.from_pretrained(
                str(
                    INDIC_INDIC_MODEL_DIR
                ),
                trust_remote_code=True,
                local_files_only=True,
                use_fast=False,
            )
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        print(
            "🧠 Loading 1B model weights..."
        )

        model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                str(
                    INDIC_INDIC_MODEL_DIR
                ),
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=dtype,
            )
        )

        # -------------------------------------------------
        # Move model
        # -------------------------------------------------

        print(
            f"⚙️ Moving model to "
            f"{device.upper()}..."
        )

        model = model.to(
            device
        )

        model.eval()

        # -------------------------------------------------
        # Processor
        # -------------------------------------------------

        processor = (
            IndicProcessor(
                inference=True
            )
        )

    except Exception as exc:

        # -------------------------------------------------
        # If MPS loading fails, retry CPU automatically.
        # -------------------------------------------------

        if device == "mps" and not force_cpu:

            print()
            print(
                "⚠️ MPS model initialization failed."
            )

            print(
                f"   Reason: "
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "   ↪ Falling back to CPU..."
            )

            return load_indic_indic_model(
                force_cpu=True
            )

        raise RuntimeError(
            "❌ Failed to load the local "
            "IndicTrans2 Indic → Indic 1B model.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            "Verify that torch, transformers "
            "and IndicTransToolkit are installed "
            "in the active venv."
        ) from exc

    # -----------------------------------------------------
    # Cache
    # -----------------------------------------------------

    _INDIC_INDIC_TOKENIZER = (
        tokenizer
    )

    _INDIC_INDIC_MODEL = (
        model
    )

    _INDIC_INDIC_PROCESSOR = (
        processor
    )

    _INDIC_INDIC_DEVICE = (
        device
    )

    print()
    print(
        f"✅ IndicTrans2 Indic → Indic "
        f"1B loaded in "
        f"{format_elapsed(elapsed_seconds(start_time))}"
    )

    return (
        tokenizer,
        model,
        processor,
        device,
    )


# =========================================================
# CPU FALLBACK
# =========================================================

def switch_indic_model_to_cpu():
    """
    Move the currently loaded IndicTrans2 model to CPU.

    Used when an MPS inference operation fails.
    """

    global _INDIC_INDIC_MODEL
    global _INDIC_INDIC_DEVICE

    if _INDIC_INDIC_MODEL is None:
        return

    print()
    print(
        "🔄 Switching IndicTrans2 "
        "from MPS → CPU..."
    )

    try:

        synchronize_mps()

    except Exception:
        pass

    try:

        _INDIC_INDIC_MODEL = (
            _INDIC_INDIC_MODEL.to(
                "cpu"
            )
        )

        _INDIC_INDIC_MODEL.eval()

        _INDIC_INDIC_DEVICE = "cpu"

        try:

            if hasattr(
                torch,
                "mps",
            ):

                torch.mps.empty_cache()

        except Exception:
            pass

        print(
            "   ✅ CPU fallback ready"
        )

    except Exception as exc:

        raise RuntimeError(
            "❌ Could not move IndicTrans2 "
            "model to CPU.\n\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc


# =========================================================
# INDIC TRANS2 SINGLE BATCH
# =========================================================

def run_indic_batch(
    input_sentences,
    tokenizer,
    model,
    processor,
    device,
    src_tag,
    tgt_tag,
):
    """
    Execute one small IndicTrans2 batch.

    The batch size is controlled by the caller.
    """

    # -----------------------------------------------------
    # Processor
    # -----------------------------------------------------

    batch = (
        processor.preprocess_batch(
            input_sentences,
            src_lang=src_tag,
            tgt_lang=tgt_tag,
        )
    )

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    with torch.no_grad():

        generated_tokens = (
            model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=INDIC_MAX_LENGTH,
                num_beams=INDIC_NUM_BEAMS,
                num_return_sequences=1,
            )
        )

    # -----------------------------------------------------
    # MPS synchronization
    # -----------------------------------------------------

    if device == "mps":

        synchronize_mps()

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

    generated_tokens = (
        generated_tokens
        .detach()
        .cpu()
    )

    decoded = (
        tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )

    translations = (
        processor.postprocess_batch(
            decoded,
            lang=tgt_tag,
        )
    )

    return translations


# =========================================================
# DIRECT INDIC -> INDIC
# =========================================================

def translate_indic_indic_segments(
    segments,
    src_lang: str,
    tgt_lang: str,
):

    src_lang = src_lang.lower()
    tgt_lang = tgt_lang.lower()

    if (
        src_lang not in FLORES_TAGS
        or tgt_lang not in FLORES_TAGS
    ):

        raise ValueError(
            "Unsupported Indic language."
        )

    if (
        src_lang == "en"
        or tgt_lang == "en"
    ):

        raise ValueError(
            "translate_indic_indic_segments "
            "is only for Indic → Indic."
        )

    src_tag = FLORES_TAGS[
        src_lang
    ]

    tgt_tag = FLORES_TAGS[
        tgt_lang
    ]

    print()
    print("=" * 60)

    print(
        f"🌐 DIRECT INDIC TRANSLATION: "
        f"{src_lang.upper()} → "
        f"{tgt_lang.upper()}"
    )

    print("=" * 60)

    print(
        "🧠 Engine: "
        "IndicTrans2 Indic → Indic 1B"
    )

    print(
        f"   Source: {src_tag}"
    )

    print(
        f"   Target: {tgt_tag}"
    )

    print(
        f"   Batch : {INDIC_BATCH_SIZE} segments"
    )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    (
        tokenizer,
        model,
        processor,
        device,
    ) = load_indic_indic_model()

    # -----------------------------------------------------
    # Prepare segments
    # -----------------------------------------------------

    valid_segments = []
    input_sentences = []

    for seg in segments:

        text = str(
            seg.get(
                "text",
                ""
            )
        ).strip()

        if not text:
            continue

        valid_segments.append(
            seg
        )

        input_sentences.append(
            text
        )

    if not input_sentences:

        print(
            "⚠️ No valid segments found."
        )

        return []

    total = len(
        input_sentences
    )

    total_batches = (
        (
            total
            + INDIC_BATCH_SIZE
            - 1
        )
        // INDIC_BATCH_SIZE
    )

    print()
    print(
        f"📦 Segments to translate: "
        f"{total}"
    )

    print(
        f"📦 Batches: "
        f"{total_batches} "
        f"(batch size = "
        f"{INDIC_BATCH_SIZE})"
    )

    print(
        f"🧠 Device: "
        f"{device.upper()}"
    )

    print()
    print(
        "🚀 Starting IndicTrans2 inference..."
    )

    overall_start = time.time()

    translated_texts = []

    # -----------------------------------------------------
    # Process 4 segments at a time
    # -----------------------------------------------------

    for batch_number, batch_start in enumerate(
        range(
            0,
            total,
            INDIC_BATCH_SIZE,
        ),
        start=1,
    ):

        batch_end = min(
            batch_start
            + INDIC_BATCH_SIZE,
            total,
        )

        batch_sentences = (
            input_sentences[
                batch_start:batch_end
            ]
        )

        batch_size = len(
            batch_sentences
        )

        batch_start_time = time.time()

        # -------------------------------------------------
        # Run batch
        # -------------------------------------------------

        try:

            translations = run_indic_batch(
                batch_sentences,
                tokenizer,
                model,
                processor,
                device,
                src_tag,
                tgt_tag,
            )

        except Exception as exc:

            # ---------------------------------------------
            # Automatic MPS -> CPU fallback
            # ---------------------------------------------

            if device == "mps":

                print()
                print(
                    f"⚠️ MPS failed on batch "
                    f"{batch_number}/{total_batches}"
                )

                print(
                    f"   Reason: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    "   ↪ Retrying this batch "
                    "on CPU..."
                )

                switch_indic_model_to_cpu()

                device = "cpu"

                # -----------------------------------------
                # Retry same batch
                # -----------------------------------------

                retry_start = time.time()

                translations = run_indic_batch(
                    batch_sentences,
                    tokenizer,
                    _INDIC_INDIC_MODEL,
                    processor,
                    device,
                    src_tag,
                    tgt_tag,
                )

                retry_elapsed = (
                    elapsed_seconds(
                        retry_start
                    )
                )

                print(
                    f"   ✅ CPU retry completed "
                    f"in "
                    f"{format_elapsed(retry_elapsed)}"
                )

            else:

                raise RuntimeError(
                    f"IndicTrans2 inference failed "
                    f"on batch "
                    f"{batch_number}/{total_batches}.\n\n"
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        # -------------------------------------------------
        # Validate batch result
        # -------------------------------------------------

        if len(translations) != batch_size:

            raise RuntimeError(
                f"IndicTrans2 returned "
                f"{len(translations)} translations "
                f"for {batch_size} inputs "
                f"in batch "
                f"{batch_number}/{total_batches}."
            )

        # -------------------------------------------------
        # Clean translations
        # -------------------------------------------------

        for translated_text in translations:

            translated_text = (
                clean_sentencepiece_text(
                    str(
                        translated_text
                    )
                )
            )

            translated_texts.append(
                translated_text
            )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        processed = batch_end

        percent = (
            processed / total
        ) * 100

        batch_elapsed = (
            elapsed_seconds(
                batch_start_time
            )
        )

        overall_elapsed = (
            elapsed_seconds(
                overall_start
            )
        )

        # Estimated remaining time.
        if processed > 0:

            avg_per_segment = (
                overall_elapsed
                / processed
            )

            remaining = (
                total - processed
            )

            eta = (
                avg_per_segment
                * remaining
            )

        else:

            eta = 0

        print(
            f"[{processed}/{total}] "
            f"{percent:5.1f}% | "
            f"batch {batch_number}/{total_batches} | "
            f"{batch_size} seg | "
            f"batch {format_elapsed(batch_elapsed)} | "
            f"elapsed {format_elapsed(overall_elapsed)} | "
            f"ETA {format_elapsed(eta)} | "
            f"{device.upper()}"
        )

    # -----------------------------------------------------
    # Build output
    # -----------------------------------------------------

    translated_segments = []

    for seg, translated_text in zip(
        valid_segments,
        translated_texts,
    ):

        entry = dict(seg)

        entry[
            f"translation_{tgt_lang}"
        ] = translated_text

        translated_segments.append(
            entry
        )

    total_elapsed = (
        elapsed_seconds(
            overall_start
        )
    )

    print()
    print(
        "✅ IndicTrans2 translation finished"
    )

    print(
        f"   Segments : {total}"
    )

    print(
        f"   Batches  : {total_batches}"
    )

    print(
        f"   Device   : {device.upper()}"
    )

    print(
        f"   Elapsed  : "
        f"{format_elapsed(total_elapsed)}"
    )

    return translated_segments


# =========================================================
# MARATHI -> HINDI
# =========================================================

def translate_marathi_to_hindi(
    segments,
):

    print()
    print(
        "🇮🇳 Marathi → Hindi"
    )

    print(
        "   Using DIRECT "
        "IndicTrans2 Indic → Indic 1B"
    )

    return (
        translate_indic_indic_segments(
            segments,
            src_lang="mr",
            tgt_lang="hi",
        )
    )


# =========================================================
# HINDI -> MARATHI
# =========================================================

def translate_hindi_to_marathi(
    segments,
):

    print()
    print(
        "🇮🇳 Hindi → Marathi"
    )

    print(
        "   Using DIRECT "
        "IndicTrans2 Indic → Indic 1B"
    )

    return (
        translate_indic_indic_segments(
            segments,
            src_lang="hi",
            tgt_lang="mr",
        )
    )


# =========================================================
# ATOMIC JSON OUTPUT
# =========================================================

def write_output_json(
    translated_segments,
    output_path: Path,
):
    """
    Write JSON to a temporary file first.

    The real output file is replaced only after
    json.dump() completes successfully.

    This prevents a failed translation from leaving
    behind a partially written output JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    try:

        # -------------------------------------------------
        # Remove stale temp file
        # -------------------------------------------------

        if temp_path.exists():

            try:
                temp_path.unlink()

            except OSError:
                pass

        # -------------------------------------------------
        # Write temporary output
        # -------------------------------------------------

        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                translated_segments,
                f,
                ensure_ascii=False,
                indent=2,
            )

            # Make sure Python has flushed everything.
            f.flush()

            try:
                os.fsync(
                    f.fileno()
                )

            except OSError:
                pass

        # -------------------------------------------------
        # Validate temporary JSON
        # -------------------------------------------------

        with open(
            temp_path,
            "r",
            encoding="utf-8",
        ) as f:

            json.load(f)

        # -------------------------------------------------
        # Atomic replacement
        # -------------------------------------------------

        os.replace(
            temp_path,
            output_path,
        )

    except Exception:

        # -------------------------------------------------
        # Never leave partial temporary files.
        # -------------------------------------------------

        try:

            if temp_path.exists():
                temp_path.unlink()

        except OSError:
            pass

        raise


# =========================================================
# MAIN TRANSLATION FUNCTION
# =========================================================

def translate_whisper_json(
    input_json_path: str,
    src_lang: str = "en",
    target_lang: str = "hi",
):

    input_path = (
        Path(
            input_json_path
        ).resolve()
    )

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not input_path.exists():

        raise FileNotFoundError(
            f"\n❌ Input JSON does not exist:\n"
            f"{input_path}"
        )

    if not input_path.is_file():

        raise FileNotFoundError(
            f"\n❌ Input path is not a file:\n"
            f"{input_path}"
        )

    if input_path.stat().st_size == 0:

        raise ValueError(
            f"\n❌ Input JSON is empty:\n"
            f"{input_path}"
        )

    print()
    print(
        f"📄 Input JSON: "
        f"{input_path}"
    )

    # -----------------------------------------------------
    # Load JSON
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

    print()
    print(
        f"📊 Whisper segments loaded: "
        f"{len(segments)}"
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    src_lang = (
        src_lang.lower()
    )

    target_lang = (
        target_lang.lower()
    )

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

    if src_lang == target_lang:

        raise ValueError(
            "Source and target languages "
            "must be different."
        )

    # -----------------------------------------------------
    # Route
    # -----------------------------------------------------

    translation_start = time.time()

    if (
        src_lang == "mr"
        and target_lang == "hi"
    ):

        translated_segments = (
            translate_marathi_to_hindi(
                segments
            )
        )

    elif (
        src_lang == "hi"
        and target_lang == "mr"
    ):

        translated_segments = (
            translate_hindi_to_marathi(
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

    translation_elapsed = (
        elapsed_seconds(
            translation_start
        )
    )

    # -----------------------------------------------------
    # Validate output
    # -----------------------------------------------------

    if not translated_segments:

        raise RuntimeError(
            "❌ Translation returned "
            "no segments."
        )

    # -----------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT write OUTPUT_JSON until every translation
    # batch has successfully completed.
    # -----------------------------------------------------

    print()
    print(
        "💾 Preparing final output..."
    )

    write_output_json(
        translated_segments,
        OUTPUT_JSON,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 60)

    print(
        "✅ TRANSLATION COMPLETED"
    )

    print("=" * 60)

    print(
        f"📄 Output: "
        f"{OUTPUT_JSON}"
    )

    print(
        f"📊 Translated segments: "
        f"{len(translated_segments)}"
    )

    print(
        f"🌐 Route: "
        f"{src_lang.upper()} → "
        f"{target_lang.upper()}"
    )

    print(
        f"⏱️ Translation time: "
        f"{format_elapsed(translation_elapsed)}"
    )

    if (
        src_lang in {"mr", "hi"}
        and target_lang in {"mr", "hi"}
    ):

        print(
            "🧠 Model: "
            "IndicTrans2 Indic → Indic 1B"
        )

        print(
            f"📁 Model: "
            f"{INDIC_INDIC_MODEL_DIR}"
        )

        print(
            f"📦 Batch size: "
            f"{INDIC_BATCH_SIZE}"
        )

    return OUTPUT_JSON


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Offline IndicTrans2 translation pipeline. "
            "Supports English ↔ Indic using CTranslate2 "
            "and direct Indic ↔ Indic using the local "
            "IndicTrans2 1B model."
        )
    )

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------

    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(
            DEFAULT_INPUT
        ),
        help=(
            "Whisper JSON input file"
        ),
    )

    # -----------------------------------------------------
    # Target
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
        help=(
            "Target language"
        ),
    )

    # -----------------------------------------------------
    # Source
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
        help=(
            "Source language"
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Execute
    # -----------------------------------------------------

    try:

        translate_whisper_json(
            args.input_file,
            src_lang=args.source_language,
            target_lang=args.target_language,
        )

    except KeyboardInterrupt:

        print()
        print(
            "⚠️ Translation interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 60)

        print(
            "❌ TRANSLATION FAILED"
        )

        print("=" * 60)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "⚠️ Output JSON was NOT updated."
        )

        sys.exit(1)