#!/usr/bin/env python3
"""Fast pre-flight verification for BAIF deployment."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

PACKAGES = [
    ("streamlit", "Streamlit"),
    ("faster_whisper", "faster-whisper"),
    ("ctranslate2", "CTranslate2"),
    ("sentencepiece", "SentencePiece"),
    ("sherpa_onnx", "sherpa-onnx"),
    ("soundfile", "soundfile"),
    ("srt", "srt"),
    ("numpy", "NumPy"),
]

REQUIRED_FILES = [
    "local_model_vault/whisper/model.bin",
    "local_model_vault/indictrans2/en-indic/model.bin",
    "local_model_vault/indictrans2/en-indic/vocab/model.SRC",
    "local_model_vault/indictrans2/en-indic/vocab/model.TGT",
    "local_model_vault/indictrans2/indic-en/model.bin",
    "local_model_vault/indictrans2/indic-en/vocab/model.SRC",
    "local_model_vault/indictrans2/indic-en/vocab/model.TGT",
    "local_model_vault/tts/english/en_US-lessac-medium.onnx",
    "local_model_vault/tts/hindi/hi_IN-pratham-medium.onnx",
    "local_model_vault/tts/marathi/mr_IN-google-medium.onnx",
]


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def main() -> int:
    failures = 0
    print("BAIF installation pre-flight\n")

    py_ok = sys.version_info[:2] == (3, 12)
    print(f"{mark(py_ok)} Python {sys.version.split()[0]} (recommended: 3.12.x)")
    failures += 0 if py_ok else 1

    for exe in ("ffmpeg", "ffprobe"):
        ok = shutil.which(exe) is not None
        print(f"{mark(ok)} {exe} on PATH")
        failures += 0 if ok else 1

    for module, label in PACKAGES:
        try:
            importlib.import_module(module)
            ok = True
        except Exception as exc:
            ok = False
            print(f"❌ {label}: {exc}")
        if ok:
            print(f"✅ {label}")
        failures += 0 if ok else 1

    for rel in REQUIRED_FILES:
        p = Path(rel)
        ok = p.is_file() and p.stat().st_size > 0
        print(f"{mark(ok)} {rel}")
        failures += 0 if ok else 1

    print()
    if failures:
        print(f"❌ Pre-flight failed with {failures} problem(s).")
        return 1

    print("✅ Pre-flight passed.")
    print("Next: python verify_all_models.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
