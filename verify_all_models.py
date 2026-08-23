import os
import sys
import json
import time
from pathlib import Path

import ctranslate2


# =============================================================
# PROJECT PATHS
# =============================================================

BASE_DIR = Path(__file__).resolve().parent

VAULT_DIR = BASE_DIR / "local_model_vault"

OUTPUT_DIR = BASE_DIR / "storage_vault" / "test_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
# LANGUAGE CODES
# =============================================================

LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
}


# =============================================================
# TEST SENTENCES
# =============================================================

SAMPLE_INPUTS = {
    "en": (
        "Agriculture is the foundation of rural development "
        "and food security."
    ),
    "hi": (
        "कृषि ग्रामीण विकास और खाद्य सुरक्षा की नींव है।"
    ),
    "mr": (
        "शेती हा ग्रामीण विकासाचा आणि अन्न सुरक्षेचा पाया आहे."
    ),
}


# =============================================================
# TRANSLATION COMBINATIONS
# =============================================================

TRANSLATION_PAIRS = [
    ("en", "hi", "Direct English → Hindi"),
    ("en", "mr", "Direct English → Marathi"),
    ("hi", "en", "Direct Hindi → English"),
    ("mr", "en", "Direct Marathi → English"),
    (
        "hi",
        "mr",
        "Pivot Hindi → English → Marathi",
    ),
    (
        "mr",
        "hi",
        "Pivot Marathi → English → Hindi",
    ),
]


# =============================================================
# PRINT HEADER
# =============================================================

def print_header(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================
# INDIC TRANS 2 MODEL VALIDATION
# =============================================================

def validate_indictrans2_directory(
    model_dir: Path,
    direction: str,
) -> bool:
    """
    Validate the actual IndicTrans2 CT2 model layout used by
    this BAIF project.

    Expected:

        model.bin
        config.json
        source_vocabulary.json
        target_vocabulary.json

    No vocab/model.SRC or vocab/model.TGT is required.
    """

    print()
    print(f"  🔎 Checking {direction} model")
    print(f"     Directory: {model_dir}")

    if not model_dir.exists():
        print("     ❌ Directory does not exist")
        return False

    required_files = [
        "model.bin",
        "config.json",
        "source_vocabulary.json",
        "target_vocabulary.json",
    ]

    all_present = True

    for filename in required_files:

        path = model_dir / filename

        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)

            print(
                f"     ✅ {filename:<25} "
                f"({size_mb:,.1f} MB)"
            )
        else:
            print(
                f"     ❌ {filename:<25} MISSING"
            )
            all_present = False

    if not all_present:
        return False

    # ---------------------------------------------------------
    # Validate config.json
    # ---------------------------------------------------------

    try:

        with open(
            model_dir / "config.json",
            "r",
            encoding="utf-8",
        ) as f:

            config = json.load(f)

        print(
            f"     ✅ config.json readable"
        )

        if isinstance(config, dict):

            print(
                f"        decoder_start_token: "
                f"{config.get('decoder_start_token')}"
            )

            print(
                f"        eos_token: "
                f"{config.get('eos_token')}"
            )

    except Exception as e:

        print(
            f"     ❌ config.json invalid: {e}"
        )

        return False

    # ---------------------------------------------------------
    # Validate source vocabulary
    # ---------------------------------------------------------

    try:

        with open(
            model_dir / "source_vocabulary.json",
            "r",
            encoding="utf-8",
        ) as f:

            source_vocab = json.load(f)

        if isinstance(source_vocab, dict):

            source_count = len(source_vocab)

        elif isinstance(source_vocab, list):

            source_count = len(source_vocab)

        else:

            source_count = 0

        print(
            f"     ✅ source_vocabulary.json "
            f"({source_count:,} entries)"
        )

    except Exception as e:

        print(
            f"     ❌ Source vocabulary invalid: {e}"
        )

        return False

    # ---------------------------------------------------------
    # Validate target vocabulary
    # ---------------------------------------------------------

    try:

        with open(
            model_dir / "target_vocabulary.json",
            "r",
            encoding="utf-8",
        ) as f:

            target_vocab = json.load(f)

        if isinstance(target_vocab, dict):

            target_count = len(target_vocab)

        elif isinstance(target_vocab, list):

            target_count = len(target_vocab)

        else:

            target_count = 0

        print(
            f"     ✅ target_vocabulary.json "
            f"({target_count:,} entries)"
        )

    except Exception as e:

        print(
            f"     ❌ Target vocabulary invalid: {e}"
        )

        return False

    return True


# =============================================================
# LOAD INDIC TRANS 2 CT2 ENGINE
# =============================================================

def load_translation_engine(
    model_dir: Path,
    name: str,
):
    """
    Load a CTranslate2 IndicTrans2 model.

    This intentionally does NOT look for:

        vocab/model.SRC
        vocab/model.TGT

    because the current BAIF model format does not contain them.
    """

    print()
    print(
        f"  ⏳ Loading CTranslate2 engine: {name}"
    )

    print(
        f"     Model directory:"
    )
    print(
        f"     {model_dir}"
    )

    start = time.time()

    try:

        translator = ctranslate2.Translator(
            str(model_dir),
            device="cpu",
            inter_threads=4,
        )

        elapsed = time.time() - start

        print(
            f"  ✅ {name} loaded successfully "
            f"in {elapsed:.2f}s"
        )

        return translator

    except Exception as e:

        print(
            f"  ❌ {name} failed to load"
        )

        print(
            f"     Error: {type(e).__name__}: {e}"
        )

        return None


# =============================================================
# 1. VERIFY LOCAL MODEL VAULT
# =============================================================

def verify_vault_files():

    print_header(
        "1. Checking Local Model Vault Files"
    )

    whisper_dir = (
        VAULT_DIR / "whisper"
    )

    en_indic_dir = (
        VAULT_DIR
        / "indictrans2"
        / "en-indic"
    )

    indic_en_dir = (
        VAULT_DIR
        / "indictrans2"
        / "indic-en"
    )

    status = {}

    # ---------------------------------------------------------
    # Whisper
    # ---------------------------------------------------------

    status["Whisper ASR"] = (
        whisper_dir / "model.bin"
    ).is_file()

    # ---------------------------------------------------------
    # IndicTrans2 EN → Indic
    # ---------------------------------------------------------

    status["IndicTrans2 (en-indic)"] = (
        validate_indictrans2_directory(
            en_indic_dir,
            "en-indic",
        )
    )

    # ---------------------------------------------------------
    # IndicTrans2 Indic → EN
    # ---------------------------------------------------------

    status["IndicTrans2 (indic-en)"] = (
        validate_indictrans2_directory(
            indic_en_dir,
            "indic-en",
        )
    )

    # ---------------------------------------------------------
    # TTS
    # ---------------------------------------------------------

    status["TTS Hindi (Pratham)"] = (
        VAULT_DIR
        / "tts"
        / "hindi"
        / "tokens.txt"
    ).is_file()

    status["TTS English (Lessac)"] = (
        VAULT_DIR
        / "tts"
        / "english"
        / "tokens.txt"
    ).is_file()

    status["TTS Marathi (Google)"] = (
        VAULT_DIR
        / "tts"
        / "marathi"
        / "mr_IN-google-medium.onnx"
    ).is_file()

    status["TTS Marathi eSpeak Data"] = (
        VAULT_DIR
        / "tts"
        / "marathi"
        / "espeak-ng-data"
    ).is_dir()

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    all_ok = True

    print()

    for name, ok in status.items():

        if ok:

            print(
                f"  ✅ {name:<30} : Ready"
            )

        else:

            print(
                f"  ❌ {name:<30} : MISSING"
            )

            all_ok = False

    return (
        all_ok,
        en_indic_dir,
        indic_en_dir,
    )


# =============================================================
# 2. VERIFY FASTER WHISPER
# =============================================================

def verify_whisper() -> bool:

    print_header(
        "2. Verifying Faster-Whisper ASR Engine"
    )

    try:

        from faster_whisper import WhisperModel

        whisper_dir = str(
            VAULT_DIR / "whisper"
        )

        start = time.time()

        _ = WhisperModel(
            whisper_dir,
            device="cpu",
            compute_type="int8",
        )

        elapsed = time.time() - start

        print(
            f"  ✅ Whisper ASR loaded successfully "
            f"in {elapsed:.2f}s"
        )

        return True

    except Exception as e:

        print(
            f"  ❌ Whisper loading failed:"
        )

        print(
            f"     {type(e).__name__}: {e}"
        )

        return False


# =============================================================
# 3. VERIFY INDIC TRANS 2 ENGINES
# =============================================================

def verify_translation_engines(
    en_indic_dir: Path,
    indic_en_dir: Path,
):

    print_header(
        "3. Verifying IndicTrans2 CTranslate2 Engines"
    )

    en_indic_engine = load_translation_engine(
        en_indic_dir,
        "en-indic",
    )

    indic_en_engine = load_translation_engine(
        indic_en_dir,
        "indic-en",
    )

    en_indic_ok = (
        en_indic_engine is not None
    )

    indic_en_ok = (
        indic_en_engine is not None
    )

    print()

    print(
        f"  • en-indic engine : "
        f"{'✅ READY' if en_indic_ok else '❌ FAILED'}"
    )

    print(
        f"  • indic-en engine : "
        f"{'✅ READY' if indic_en_ok else '❌ FAILED'}"
    )

    return (
        en_indic_ok,
        indic_en_ok,
        en_indic_engine,
        indic_en_engine,
    )


# =============================================================
# 4. VERIFY SIX TRANSLATION ROUTES
# =============================================================

def verify_translation_combinations(
    en_indic_engine,
    indic_en_engine,
):

    print_header(
        "4. Verifying All 6 Bidirectional Translation Routes"
    )

    routes = [
        (
            "EN → HI",
            "Direct English → Hindi",
            en_indic_engine,
        ),
        (
            "EN → MR",
            "Direct English → Marathi",
            en_indic_engine,
        ),
        (
            "HI → EN",
            "Direct Hindi → English",
            indic_en_engine,
        ),
        (
            "MR → EN",
            "Direct Marathi → English",
            indic_en_engine,
        ),
        (
            "HI → MR",
            "Hindi → English → Marathi",
            "PIVOT",
        ),
        (
            "MR → HI",
            "Marathi → English → Hindi",
            "PIVOT",
        ),
    ]

    results = []

    for route, description, engine in routes:

        print()
        print(
            f"  🌐 {route}"
        )

        print(
            f"     Mode: {description}"
        )

        # -----------------------------------------------------
        # Direct route
        # -----------------------------------------------------

        if engine != "PIVOT":

            if engine is None:

                print(
                    "     ❌ Engine unavailable"
                )

                results.append(False)
                continue

            try:

                # We only perform an engine-level smoke test
                # here because these model directories do not
                # contain model.SRC/model.TGT SentencePiece files.

                engine.translate_batch(
                    [
                        [
                            "<s>",
                            "</s>",
                        ]
                    ],
                    beam_size=1,
                    max_decoding_length=8,
                )

                print(
                    "     ✅ CTranslate2 inference engine "
                    "responded"
                )

                results.append(True)

            except Exception as e:

                print(
                    "     ❌ CTranslate2 inference failed:"
                )

                print(
                    f"        {type(e).__name__}: {e}"
                )

                results.append(False)

        # -----------------------------------------------------
        # Pivot route
        # -----------------------------------------------------

        else:

            if (
                en_indic_engine is None
                or indic_en_engine is None
            ):

                print(
                    "     ❌ Required pivot engines unavailable"
                )

                results.append(False)
                continue

            try:

                # Verify both directions involved in pivot.
                #
                # Actual language tokenization is intentionally
                # not attempted here because the downloaded
                # model directories do not contain SentencePiece
                # model.SRC/model.TGT files.

                indic_en_engine.translate_batch(
                    [
                        [
                            "<s>",
                            "</s>",
                        ]
                    ],
                    beam_size=1,
                    max_decoding_length=8,
                )

                en_indic_engine.translate_batch(
                    [
                        [
                            "<s>",
                            "</s>",
                        ]
                    ],
                    beam_size=1,
                    max_decoding_length=8,
                )

                print(
                    "     ✅ Both pivot CTranslate2 engines "
                    "responded"
                )

                results.append(True)

            except Exception as e:

                print(
                    "     ❌ Pivot engine verification failed:"
                )

                print(
                    f"        {type(e).__name__}: {e}"
                )

                results.append(False)

    passed = sum(
        1 for result in results
        if result
    )

    print()
    print(
        f"  📊 Translation route engine checks: "
        f"{passed}/{len(results)}"
    )

    return (
        passed == len(results)
    )


# =============================================================
# 5. VERIFY SHERPA-ONNX TTS
# =============================================================

def verify_tts() -> bool:

    print_header(
        "5. Verifying Sherpa-ONNX TTS Synthesis (en, hi, mr)"
    )

    try:

        import sherpa_onnx
        import soundfile as sf

        tts_configs = {

            "hi": {
                "model": (
                    VAULT_DIR
                    / "tts"
                    / "hindi"
                    / "hi_IN-pratham-medium.onnx"
                ),
                "tokens": (
                    VAULT_DIR
                    / "tts"
                    / "hindi"
                    / "tokens.txt"
                ),
                "data_dir": (
                    VAULT_DIR
                    / "tts"
                    / "hindi"
                    / "espeak-ng-data"
                ),
                "text": (
                    "कृषि क्षेत्र में आपका स्वागत है।"
                ),
            },

            "en": {
                "model": (
                    VAULT_DIR
                    / "tts"
                    / "english"
                    / "en_US-lessac-medium.onnx"
                ),
                "tokens": (
                    VAULT_DIR
                    / "tts"
                    / "english"
                    / "tokens.txt"
                ),
                "data_dir": (
                    VAULT_DIR
                    / "tts"
                    / "english"
                    / "espeak-ng-data"
                ),
                "text": (
                    "Welcome to the agricultural "
                    "translation system."
                ),
            },

            "mr": {
                "model": (
                    VAULT_DIR
                    / "tts"
                    / "marathi"
                    / "mr_IN-google-medium.onnx"
                ),
                "tokens": (
                    VAULT_DIR
                    / "tts"
                    / "marathi"
                    / "tokens.txt"
                ),
                "data_dir": (
                    VAULT_DIR
                    / "tts"
                    / "marathi"
                    / "espeak-ng-data"
                ),
                "text": (
                    "शेती विषयक कार्यक्रमात आपले "
                    "स्वागत आहे."
                ),
            },
        }

        all_passed = True

        for lang, cfg in tts_configs.items():

            try:

                t_config = (
                    sherpa_onnx.OfflineTtsConfig(
                        model=(
                            sherpa_onnx
                            .OfflineTtsModelConfig(
                                vits=(
                                    sherpa_onnx
                                    .OfflineTtsVitsModelConfig(
                                        model=str(
                                            cfg["model"]
                                        ),
                                        tokens=str(
                                            cfg["tokens"]
                                        ),
                                        data_dir=(
                                            str(
                                                cfg["data_dir"]
                                            )
                                            if cfg[
                                                "data_dir"
                                            ].exists()
                                            else ""
                                        ),
                                    )
                                ),
                                provider="cpu",
                                num_threads=4,
                            )
                        )
                    )
                )

                if not t_config.validate():

                    print(
                        f"  ❌ TTS Config Invalid "
                        f"for '{lang}'"
                    )

                    all_passed = False
                    continue

                start = time.time()

                engine = sherpa_onnx.OfflineTts(
                    t_config
                )

                audio = engine.generate(
                    cfg["text"],
                    sid=0,
                    speed=1.0,
                )

                out_wav = (
                    OUTPUT_DIR
                    / f"verify_tts_{lang}.wav"
                )

                sf.write(
                    str(out_wav),
                    audio.samples,
                    audio.sample_rate,
                )

                duration = (
                    len(audio.samples)
                    / audio.sample_rate
                )

                elapsed = time.time() - start

                print(
                    f"  🎙️ [{lang.upper()} Voice] "
                    f"Synthesized successfully"
                )

                print(
                    f"     Output  : {out_wav}"
                )

                print(
                    f"     Duration: {duration:.2f}s"
                )

                print(
                    f"     Time    : {elapsed:.2f}s"
                )

            except Exception as e:

                print(
                    f"  ❌ TTS Generation failed "
                    f"for '{lang}':"
                )

                print(
                    f"     {type(e).__name__}: {e}"
                )

                all_passed = False

        return all_passed

    except Exception as e:

        print(
            f"  ❌ TTS setup failed:"
        )

        print(
            f"     {type(e).__name__}: {e}"
        )

        return False


# =============================================================
# MAIN
# =============================================================

def main():

    print()
    print("=" * 65)
    print(
        "🧪 BAIF TRANSLATION ENGINE - "
        "MULTI-MODEL SYSTEM VERIFICATION"
    )
    print("=" * 65)

    # ---------------------------------------------------------
    # 1. Model Vault
    # ---------------------------------------------------------

    (
        files_ok,
        en_indic_dir,
        indic_en_dir,
    ) = verify_vault_files()

    # ---------------------------------------------------------
    # 2. Whisper
    # ---------------------------------------------------------

    whisper_ok = verify_whisper()

    # ---------------------------------------------------------
    # 3. IndicTrans2 CT2 engines
    # ---------------------------------------------------------

    (
        en_indic_ok,
        indic_en_ok,
        en_indic_engine,
        indic_en_engine,
    ) = verify_translation_engines(
        en_indic_dir,
        indic_en_dir,
    )

    # ---------------------------------------------------------
    # 4. Translation route checks
    # ---------------------------------------------------------

    nmt_ok = verify_translation_combinations(
        en_indic_engine,
        indic_en_engine,
    )

    # ---------------------------------------------------------
    # 5. TTS
    # ---------------------------------------------------------

    tts_ok = verify_tts()

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print_header(
        "FINAL VERIFICATION SUMMARY"
    )

    print(
        "  • Model Vault Files      : "
        f"{'✅ PASSED' if files_ok else '❌ INCOMPLETE'}"
    )

    print(
        "  • Whisper ASR Engine     : "
        f"{'✅ PASSED' if whisper_ok else '❌ FAILED'}"
    )

    print(
        "  • IndicTrans2 en-indic   : "
        f"{'✅ PASSED' if en_indic_ok else '❌ FAILED'}"
    )

    print(
        "  • IndicTrans2 indic-en   : "
        f"{'✅ PASSED' if indic_en_ok else '❌ FAILED'}"
    )

    print(
        "  • 6/6 NMT Engine Routes  : "
        f"{'✅ PASSED' if nmt_ok else '❌ FAILED'}"
    )

    print(
        "  • 3/3 TTS Voice Engines  : "
        f"{'✅ PASSED' if tts_ok else '❌ FAILED'}"
    )

    print("=" * 65)

    # ---------------------------------------------------------
    # Overall result
    # ---------------------------------------------------------

    if (
        files_ok
        and whisper_ok
        and en_indic_ok
        and indic_en_ok
        and nmt_ok
        and tts_ok
    ):

        print()
        print(
            "🎉 ALL MODEL ENGINES ARE READY!"
        )

        print(
            "✅ Whisper ASR"
        )

        print(
            "✅ IndicTrans2 English → Indic"
        )

        print(
            "✅ IndicTrans2 Indic → English"
        )

        print(
            "✅ Hindi / Marathi / English TTS"
        )

        print()
        print(
            "⚠️ NOTE:"
        )

        print(
            "The NMT checks above validate the "
            "CTranslate2 engines."
        )

        print(
            "Actual sentence translation still "
            "requires the tokenizer/inference "
            "layer used by this IndicTrans2 model."
        )

        print()

    else:

        print()
        print(
            "⚠️ SYSTEM VERIFICATION FAILED."
        )

        print(
            "Check the errors above."
        )

        print()


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()