#!/usr/bin/env python3
"""Publish the exact verified production model vault to a private Hugging Face model repo.

Run once from an internet-connected development machine after:
    python prepare_deployment_vault.py

Authentication:
- `hf auth login`, OR
- set HF_TOKEN in the environment temporarily.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.environ.get("BAIF_MODEL_REPO_ID"))
    parser.add_argument("--folder", type=Path, default=Path("deployment_model_vault"))
    args = parser.parse_args()

    if not args.repo_id:
        raise SystemExit("Provide --repo-id YOUR_ORG/YOUR_PRIVATE_MODEL_REPO or set BAIF_MODEL_REPO_ID.")

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Model vault not found: {folder}\nRun prepare_deployment_vault.py first.")

    required = folder / "DEPLOYMENT_MANIFEST.json"
    if not required.is_file():
        raise SystemExit("DEPLOYMENT_MANIFEST.json is missing. Re-run prepare_deployment_vault.py.")

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)

    print(f"Creating/using private model repository: {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )

    print(f"Uploading exact production vault from: {folder}")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=".",
        commit_message="Publish verified BAIF production model vault",
    )

    print("\n✅ Upload complete")
    print(f"Private repo: {args.repo_id}")
    print("BAIF machines can now download this exact vault during first-time internet setup.")


if __name__ == "__main__":
    main()
