#!/usr/bin/env python3
"""Download the exact BAIF production model vault with persistent resume."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Enforce long timeouts before importing huggingface_hub
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "3600"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download

DEFAULT_DEST = Path("local_model_vault")

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
    "DEPLOYMENT_MANIFEST.json",
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
        raise RuntimeError("Downloaded vault is incomplete:\n - " + "\n - ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.environ.get("BAIF_MODEL_REPO_ID"))
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    if not args.repo_id:
        raise SystemExit("Provide --repo-id YOUR_ORG/YOUR_PRIVATE_MODEL_REPO or set BAIF_MODEL_REPO_ID.")

    dest = args.dest.resolve()
    token = os.environ.get("HF_TOKEN") or None

    print(f"📦 Syncing production model vault from: {args.repo_id}")
    print(f"📁 Destination: {dest}")

    # Download directly to target directory so interrupted files resume rather than restart
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="model",
                local_dir=str(dest),
                token=token,
                max_workers=1,  # Critical: Single thread gives full bandwidth to 4.5GB model.bin
                resume_download=True,
            )
            break
        except Exception as e:
            print(f"\n⚠️ Attempt {attempt}/{max_retries} encountered connection drop: {e}")
            if attempt == max_retries:
                print("❌ Max retries reached.", file=sys.stderr)
                sys.exit(1)
            print("⏳ Retrying and resuming incomplete files in 5 seconds...")
            time.sleep(5)

    # Clean local cache metadata if present
    shutil.rmtree(dest / ".cache", ignore_errors=True)

    validate(dest)
    print("\n✅ All production models downloaded and validated successfully!")


if __name__ == "__main__":
    main()