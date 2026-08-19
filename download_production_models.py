#!/usr/bin/env python3
"""Download the exact BAIF production model vault during first-time setup.

The model package should be the private Hugging Face repository created by
publish_production_model_vault.py. After download, runtime inference uses only
local_model_vault and does not need internet or a Hugging Face token.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

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
        raise RuntimeError("Downloaded vault is incomplete:\n  - " + "\n  - ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.environ.get("BAIF_MODEL_REPO_ID"))
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.repo_id:
        raise SystemExit("Provide --repo-id YOUR_ORG/YOUR_PRIVATE_MODEL_REPO or set BAIF_MODEL_REPO_ID.")

    dest = args.dest.resolve()
    tmp = dest.parent / (dest.name + ".download_tmp")

    if dest.exists():
        if not args.force:
            raise SystemExit(f"Destination already exists: {dest}\nUse --force only if you intentionally want to replace it.")
        shutil.rmtree(dest)
    if tmp.exists():
        shutil.rmtree(tmp)

    token = os.environ.get("HF_TOKEN") or None

    print(f"Downloading production model vault from private repo: {args.repo_id}")
    print("This is a one-time internet-connected setup step.")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=str(tmp),
        token=token,
    )

    # snapshot_download may create local metadata under .cache; it is not needed at runtime.
    shutil.rmtree(tmp / ".cache", ignore_errors=True)

    validate(tmp)
    tmp.rename(dest)

    print("\n✅ Production models downloaded and validated")
    print(f"   Local runtime vault: {dest}")
    print("You can now remove Hugging Face credentials and disconnect internet after verification.")


if __name__ == "__main__":
    main()
