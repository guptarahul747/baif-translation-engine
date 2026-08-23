#!/usr/bin/env python3
"""
BAIF IndicTrans2 Indic -> Indic system verification.

Usage:
    python check_indic_indic.py

Optional:
    python check_indic_indic.py /path/to/indic-indic-1B

This script only verifies the local model.
It does NOT download, modify, or delete any files.
"""

import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_DIR = (
    BASE_DIR
    / "local_model_vault"
    / "indictrans2"
    / "indic-indic-1B"
)

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-1B"


# Required files for the IndicTrans2 Indic -> Indic 1B model.
# tokenizer.json is intentionally NOT required because this
# model uses IndicTrans2's custom tokenizer files.
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "configuration_indictrans.py",
    "modeling_indictrans.py",
    "tokenization_indictrans.py",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.SRC",
    "model.TGT",
    "dict.SRC.json",
    "dict.TGT.json",
]

CUSTOM_CODE_FILES = [
    "configuration_indictrans.py",
    "modeling_indictrans.py",
    "tokenization_indictrans.py",
]

JSON_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "dict.SRC.json",
    "dict.TGT.json",
]

WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
)


def file_ok(path):
    """Check that a file exists and is not empty."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def format_size(size_bytes):
    """Convert bytes into a readable size."""
    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def check_required_files(model_dir):
    """Check all required files."""
    missing = []

    for filename in REQUIRED_FILES:
        if not file_ok(model_dir / filename):
            missing.append(filename)

    return len(missing) == 0, missing


def check_weights(model_dir):
    """Check model weight files."""
    weight_files = []

    for pattern in WEIGHT_PATTERNS:
        weight_files.extend(
            path
            for path in model_dir.glob(pattern)
            if file_ok(path)
        )

    # Remove duplicates.
    weight_files = sorted(
        set(weight_files),
        key=lambda path: path.name.lower(),
    )

    total_size = sum(
        path.stat().st_size
        for path in weight_files
    )

    return (
        len(weight_files) > 0,
        weight_files,
        total_size,
    )


def check_custom_code(model_dir):
    """Check IndicTrans2 custom Python files."""
    missing = [
        filename
        for filename in CUSTOM_CODE_FILES
        if not file_ok(model_dir / filename)
    ]

    return len(missing) == 0, missing


def check_json(model_dir):
    """Validate important JSON files."""
    invalid = []

    for filename in JSON_FILES:

        path = model_dir / filename

        if not file_ok(path):
            continue

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                json.load(handle)

        except Exception:
            invalid.append(filename)

    return len(invalid) == 0, invalid


def check_incomplete_files(model_dir):
    """Check for incomplete/temporary download files."""
    incomplete = []

    for path in model_dir.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.name.endswith(".incomplete")
            or path.name.endswith(".tmp")
        ):
            incomplete.append(path)

    return len(incomplete) == 0, incomplete


def check_weight_indexes(model_dir):
    """Check Hugging Face weight index files if present."""
    errors = []

    for index_path in model_dir.glob("*.index.json"):

        try:

            data = json.loads(
                index_path.read_text(
                    encoding="utf-8"
                )
            )

            weight_map = data.get(
                "weight_map",
                {}
            )

            if not isinstance(
                weight_map,
                dict
            ) or not weight_map:

                errors.append(
                    f"{index_path.name}: "
                    "empty weight_map"
                )

                continue

            shard_names = sorted(
                set(weight_map.values())
            )

            for shard_name in shard_names:

                if not file_ok(
                    model_dir / shard_name
                ):

                    errors.append(
                        f"{index_path.name}: "
                        f"missing {shard_name}"
                    )

        except Exception as exc:

            errors.append(
                f"{index_path.name}: {exc}"
            )

    return len(errors) == 0, errors


def verify(model_dir):
    """Run all verification checks."""

    result = {}

    result["directory"] = (
        model_dir.exists()
        and model_dir.is_dir()
    )

    if not result["directory"]:
        return result

    (
        result["required_files"],
        result["missing_files"],
    ) = check_required_files(model_dir)

    (
        result["weights"],
        result["weight_files"],
        result["weight_size"],
    ) = check_weights(model_dir)

    (
        result["custom_code"],
        result["custom_missing"],
    ) = check_custom_code(model_dir)

    (
        result["json"],
        result["invalid_json"],
    ) = check_json(model_dir)

    (
        result["integrity"],
        result["incomplete_files"],
    ) = check_incomplete_files(model_dir)

    (
        result["weight_indexes"],
        result["weight_index_errors"],
    ) = check_weight_indexes(model_dir)

    result["success"] = all(
        [
            result["directory"],
            result["required_files"],
            result["weights"],
            result["custom_code"],
            result["json"],
            result["integrity"],
            result["weight_indexes"],
        ]
    )

    return result


def print_failure_details(result):

    missing = result.get(
        "missing_files",
        []
    )

    if missing:

        print()
        print("Missing / empty files:")

        for filename in missing:
            print(
                f"  ❌ {filename}"
            )

    custom_missing = result.get(
        "custom_missing",
        []
    )

    if custom_missing:

        print()
        print("Missing custom code:")

        for filename in custom_missing:
            print(
                f"  ❌ {filename}"
            )

    invalid_json = result.get(
        "invalid_json",
        []
    )

    if invalid_json:

        print()
        print("Invalid JSON:")

        for filename in invalid_json:
            print(
                f"  ❌ {filename}"
            )

    incomplete = result.get(
        "incomplete_files",
        []
    )

    if incomplete:

        print()
        print("Incomplete download files:")

        for path in incomplete:
            print(
                f"  ❌ {path}"
            )

    index_errors = result.get(
        "weight_index_errors",
        []
    )

    if index_errors:

        print()
        print("Weight index errors:")

        for error in index_errors:
            print(
                f"  ❌ {error}"
            )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Verify the BAIF IndicTrans2 "
            "Indic -> Indic local model."
        )
    )

    parser.add_argument(
        "model_dir",
        nargs="?",
        default=str(
            DEFAULT_MODEL_DIR
        ),
        help=(
            "Path to the local "
            "IndicTrans2 Indic -> Indic "
            "model directory."
        ),
    )

    args = parser.parse_args()

    model_dir = Path(
        args.model_dir
    ).expanduser()

    print()

    print(
        "============================================================"
    )

    print(
        "          BAIF INDIC-INDIC SYSTEM VERIFICATION"
    )

    print(
        "============================================================"
    )

    print()

    # ---------------------------------------------------------
    # Model directory check
    # ---------------------------------------------------------

    if not model_dir.exists():

        print(
            "❌ IndicTrans2 Indic → Indic"
        )

        print()

        print(
            "⚠️ SYSTEM VERIFICATION FAILED."
        )

        print()

        print(
            "Model directory was not found:"
        )

        print(
            f"  {model_dir}"
        )

        print()

        print(
            "Check the model path above."
        )

        print()

        return 1

    # ---------------------------------------------------------
    # Run verification
    # ---------------------------------------------------------

    result = verify(
        model_dir
    )

    # ---------------------------------------------------------
    # IndicTrans2 Indic -> Indic
    # ---------------------------------------------------------

    if (
        result["required_files"]
        and result["weights"]
        and result["custom_code"]
    ):

        print(
            "✅ IndicTrans2 Indic → Indic"
        )

    else:

        print(
            "❌ IndicTrans2 Indic → Indic"
        )

    # ---------------------------------------------------------
    # Model Configuration
    # ---------------------------------------------------------

    if (
        result["required_files"]
        and result["json"]
    ):

        print(
            "✅ Model Configuration"
        )

    else:

        print(
            "❌ Model Configuration"
        )

    # ---------------------------------------------------------
    # Custom Code
    # ---------------------------------------------------------

    if result["custom_code"]:

        print(
            "✅ IndicTrans2 Custom Code"
        )

    else:

        print(
            "❌ IndicTrans2 Custom Code"
        )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer_ok = (
        file_ok(
            model_dir
            / "tokenizer_config.json"
        )
        and
        file_ok(
            model_dir
            / "special_tokens_map.json"
        )
        and
        result["json"]
    )

    if tokenizer_ok:

        print(
            "✅ Tokenizer Configuration"
        )

    else:

        print(
            "❌ Tokenizer Configuration"
        )

    # ---------------------------------------------------------
    # Translation Dictionaries
    # ---------------------------------------------------------

    dictionaries_ok = all(
        file_ok(
            model_dir / filename
        )
        for filename in [
            "model.SRC",
            "model.TGT",
            "dict.SRC.json",
            "dict.TGT.json",
        ]
    )

    if dictionaries_ok:

        print(
            "✅ Translation Dictionaries"
        )

    else:

        print(
            "❌ Translation Dictionaries"
        )

    # ---------------------------------------------------------
    # Model Weights
    # ---------------------------------------------------------

    if result["weights"]:

        weight_count = len(
            result["weight_files"]
        )

        weight_size = format_size(
            result["weight_size"]
        )

        print(
            f"✅ Model Weights "
            f"({weight_count} file(s), "
            f"{weight_size})"
        )

    else:

        print(
            "❌ Model Weights"
        )

    # ---------------------------------------------------------
    # Download Integrity
    # ---------------------------------------------------------

    if (
        result["integrity"]
        and result["weight_indexes"]
    ):

        print(
            "✅ Download Integrity"
        )

    else:

        print(
            "❌ Download Integrity"
        )

    print()

    print(
        "------------------------------------------------------------"
    )

    print()

    # =========================================================
    # SUCCESS
    # =========================================================

    if result["success"]:

        print(
            "✅ SYSTEM VERIFICATION SUCCESSFUL"
        )

        print()

        print(
            "The IndicTrans2 Indic → Indic model "
            "is completely downloaded."
        )

        print(
            "All required model components are "
            "present and valid."
        )

        print()

        print(
            "Model:"
        )

        print(
            f"  {MODEL_NAME}"
        )

        print()

        print(
            "Location:"
        )

        print(
            f"  {model_dir.resolve()}"
        )

        print()

        print(
            "⚠️ NOTE:"
        )

        print(
            "The checks above validate the "
            "downloaded IndicTrans2 Indic → "
            "Indic model components."
        )

        print(
            "Actual sentence translation still "
            "requires successful model loading "
            "and the tokenizer/inference layer "
            "used by this IndicTrans2 model."
        )

        print()

    # =========================================================
    # FAILURE
    # =========================================================

    else:

        print(
            "⚠️ SYSTEM VERIFICATION FAILED."
        )

        print(
            "One or more IndicTrans2 Indic → "
            "Indic components are missing or invalid."
        )

        print_failure_details(
            result
        )

        print()

        print(
            "Check the errors above."
        )

        print()

    print(
        "============================================================"
    )

    print()

    return (
        0
        if result["success"]
        else 1
    )


if __name__ == "__main__":

    sys.exit(
        main()
    )