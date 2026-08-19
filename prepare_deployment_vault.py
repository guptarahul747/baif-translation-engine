#!/usr/bin/env python3
"""Create the minimal stable BAIF production model vault.

Copies only the model directories used by the last-known-good pipeline:
- faster-whisper model
- IndicTrans2 en-indic CTranslate2 model
- IndicTrans2 indic-en CTranslate2 model
- English/Hindi/Marathi TTS folders

Experimental model folders, caches, storage, and virtual environments are not copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_SOURCE = Path("local_model_vault")
DEFAULT_DEST = Path("deployment_model_vault")

COPY_MAP = {
    "whisper": "whisper",
    "indictrans2/en-indic": "indictrans2/en-indic",
    "indictrans2/indic-en": "indictrans2/indic-en",
    "tts/english": "tts/english",
    "tts/hindi": "tts/hindi",
    "tts/marathi": "tts/marathi",
}

REQUIRED_FILES = [
    "whisper/model.bin",
    "whisper/config.json",
    "indictrans2/en-indic/model.bin",
    "indictrans2/en-indic/config.json",
    "indictrans2/en-indic/vocab/model.SRC",
    "indictrans2/en-indic/vocab/model.TGT",
    "indictrans2/indic-en/model.bin",
    "indictrans2/indic-en/config.json",
    "indictrans2/indic-en/vocab/model.SRC",
    "indictrans2/indic-en/vocab/model.TGT",
    "tts/english/en_US-lessac-medium.onnx",
    "tts/english/en_US-lessac-medium.onnx.json",
    "tts/hindi/hi_IN-pratham-medium.onnx",
    "tts/hindi/hi_IN-pratham-medium.onnx.json",
    "tts/marathi/mr_IN-google-medium.onnx",
    "tts/marathi/mr_IN-google-medium.onnx.json",
]

REQUIRED_DIRS = [
    "tts/english/espeak-ng-data",
    "tts/hindi/espeak-ng-data",
    "tts/marathi/espeak-ng-data",
]


def validate(vault: Path) -> None:
    missing = []
    for rel in REQUIRED_FILES:
        p = vault / rel
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(rel)
    for rel in REQUIRED_DIRS:
        if not (vault / rel).is_dir():
            missing.append(rel + "/")
    if missing:
        raise RuntimeError("Missing required production artifacts:\n  - " + "\n  - ".join(missing))


def build_manifest(vault: Path) -> dict:
    files = []
    total = 0
    for p in sorted(vault.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total += size
            files.append({"path": p.relative_to(vault).as_posix(), "size_bytes": size})
    return {
        "pipeline": "BAIF stable last-known-good",
        "description": "Minimal production model vault; no experimental Indic-Indic 320M model.",
        "file_count": len(files),
        "total_size_bytes": total,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()

    if not source.is_dir():
        raise SystemExit(f"Source model vault not found: {source}")

    if dest.exists():
        if not args.force:
            raise SystemExit(f"Destination already exists: {dest}\nUse --force to rebuild it.")
        shutil.rmtree(dest)

    print(f"Preparing production vault from: {source}")
    for src_rel, dst_rel in COPY_MAP.items():
        src = source / src_rel
        dst = dest / dst_rel
        if not src.is_dir():
            raise SystemExit(f"Required source directory not found: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Copying {src_rel} -> {dst_rel}")
        shutil.copytree(src, dst)

    validate(dest)
    manifest = build_manifest(dest)
    (dest / "DEPLOYMENT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    gib = manifest["total_size_bytes"] / (1024 ** 3)
    print("\n✅ Production model vault created successfully")
    print(f"   Location : {dest}")
    print(f"   Files    : {manifest['file_count']}")
    print(f"   Size     : {gib:.2f} GiB")
    print("   Includes : Whisper, en-indic, indic-en, English/Hindi/Marathi TTS")
    print("   Excludes : experimental Indic-Indic 320M and other development models")


if __name__ == "__main__":
    main()
