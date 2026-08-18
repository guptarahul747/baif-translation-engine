import os
import sys
import json
import shutil
import ssl
import tarfile
import urllib.request
from pathlib import Path
import onnx
from huggingface_hub import snapshot_download, hf_hub_download


def build_ssl_context():
    """Build an SSL context that works even when certifi is not installed."""
    try:
        import certifi
        cafile = certifi.where()
        if cafile:
            return ssl.create_default_context(cafile=cafile)
    except Exception:
        pass
    return ssl.create_default_context()

BASE_VAULT = Path(__file__).resolve().parent / "local_model_vault"

MODELS_CONFIG = {
    "Whisper ASR Medium (INT8)": {
        "repo_id": "Systran/faster-whisper-medium",
        "local_dir": BASE_VAULT / "whisper",
        "allow_patterns": ["*.bin", "*.json", "*.txt"]
    },
    "IndicTrans2 (English -> Indic)": {
        "repo_id": "adalat-ai/ct2-rotary-indictrans2-en-indic-1B",
        "local_dir": BASE_VAULT / "indictrans2" / "en-indic",
        "allow_patterns": ["*"]
    },
    "IndicTrans2 (Indic -> English)": {
        "repo_id": "adalat-ai/ct2-rotary-indictrans2-indic-en-1B",
        "local_dir": BASE_VAULT / "indictrans2" / "indic-en",
        "allow_patterns": ["*"]
    },
    "Sherpa-ONNX TTS (Hindi - Pratham)": {
        "repo_id": "csukuangfj/vits-piper-hi_IN-pratham-medium",
        "local_dir": BASE_VAULT / "tts" / "hindi",
        "allow_patterns": ["*"]
    },
    "Sherpa-ONNX TTS (English - Lessac)": {
        "repo_id": "csukuangfj/vits-piper-en_US-lessac-medium",
        "local_dir": BASE_VAULT / "tts" / "english",
        "allow_patterns": ["*"]
    }
}

def clean_symlinks(directory: Path):
    """Removes existing symlinks in directory so HF download doesn't trigger SameFileError."""
    if not directory.exists():
        return
    for root, _, files in os.walk(directory):
        for f in files:
            p = Path(root) / f
            if p.is_symlink():
                p.unlink()

def flatten_ct2_directory(target_dir: Path):
    """Ensures model.bin and configs live in the target root, not nested subfolders."""
    if not (target_dir / "model.bin").exists():
        for sub_bin in target_dir.rglob("model.bin"):
            src_folder = sub_bin.parent
            for item in src_folder.iterdir():
                dest = target_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            # remove leftover empty nested folder
            for nested in target_dir.iterdir():
                if nested.is_dir() and nested.name != target_dir.name:
                    shutil.rmtree(str(nested), ignore_errors=True)
            break

def download_universal_espeak(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    espeak_final = target_dir / "espeak-ng-data"

    if espeak_final.exists() and (espeak_final / "mr_dict").exists():
        return

    print("  ↳ 📦 Fetching universal espeak-ng-data archive...")
    archive_path = target_dir / "espeak-ng-data.tar.bz2"
    try:
        ssl_context = build_ssl_context()
        request = urllib.request.Request(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(request, context=ssl_context, timeout=60) as response, open(archive_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        with tarfile.open(str(archive_path), "r:bz2") as tar:
            tar.extractall(path=str(target_dir))
        if archive_path.exists():
            archive_path.unlink()
        print("  ↳ ✅ Universal espeak-ng-data configured.")
    except Exception as e:
        print(f"  ❌ Failed to download espeak-ng-data: {e}", file=sys.stderr)

def setup_marathi_model():
    print("📦 Setting up Sherpa-ONNX TTS (Marathi - Google Medium)...")
    marathi_dir = BASE_VAULT / "tts" / "marathi"
    marathi_dir.mkdir(parents=True, exist_ok=True)

    files = ["mr_IN-google-medium.onnx", "mr_IN-google-medium.onnx.json"]
    for filename in files:
        target_file = marathi_dir / filename
        if not target_file.exists() or target_file.is_symlink():
            if target_file.is_symlink():
                target_file.unlink()
            downloaded_path = hf_hub_download(
                repo_id="rhasspy/piper-voices",
                filename=f"mr/mr_IN/google/medium/{filename}",
                local_dir=str(marathi_dir)
            )
            real_source = Path(os.path.realpath(downloaded_path))
            if real_source != target_file:
                shutil.copyfile(str(real_source), str(target_file))

    nested_root = marathi_dir / "mr"
    if nested_root.exists():
        shutil.rmtree(str(nested_root))

    json_path = marathi_dir / "mr_IN-google-medium.onnx.json"
    tokens_path = marathi_dir / "tokens.txt"
    onnx_path = marathi_dir / "mr_IN-google-medium.onnx"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        phoneme_map = cfg.get("phoneme_id_map", {})
        with open(tokens_path, "w", encoding="utf-8") as f:
            for token, ids in phoneme_map.items():
                if len(token) == 1:
                    for token_id in ids:
                        f.write(f"{token} {token_id}\n")

        model = onnx.load(str(onnx_path))
        existing_meta = {p.key: p for p in model.metadata_props}
        espeak_voice = cfg.get("espeak", {}).get("voice", "mr")

        required_meta = {
            "sample_rate": str(cfg.get("audio", {}).get("sample_rate", 22050)),
            "n_speakers": str(cfg.get("num_speakers", 1)),
            "version": "1",
            "model_type": "vits",
            "comment": "piper",
            "language": "mr",
            "voice": espeak_voice,
            "add_blank": "1"
        }

        updated = False
        for k, v in required_meta.items():
            if k in existing_meta:
                if existing_meta[k].value != v:
                    existing_meta[k].value = v
                    updated = True
            else:
                meta = model.metadata_props.add()
                meta.key = k
                meta.value = str(v)
                updated = True

        if updated:
            onnx.save(model, str(onnx_path))

    download_universal_espeak(marathi_dir)
    print(f"✅ Marathi TTS ready at: {marathi_dir}\n")

def download_all():
    print("=" * 65)
    print("🚀 Provisioning Full Bidirectional Multi-Language Model Vault")
    print("=" * 65 + "\n")

    for name, config in MODELS_CONFIG.items():
        local_dir = Path(config["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"📦 Checking / Downloading {name}...")

        # Clean existing symlinks to prevent SameFileError
        clean_symlinks(local_dir)

        try:
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=str(local_dir),
                allow_patterns=config.get("allow_patterns"),
                resume_download=True,
                local_dir_use_symlinks=False
            )

            if "indictrans2" in str(local_dir):
                flatten_ct2_directory(local_dir)

            print(f"✅ Ready at: {local_dir}\n")
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}\n", file=sys.stderr)

    try:
        setup_marathi_model()
    except Exception as e:
        print(f"❌ Failed to setup Marathi TTS: {e}\n", file=sys.stderr)

    print("=" * 65)
    print("🎉 All models verified and provisioned in local_model_vault!")
    print("==========================================================")

if __name__ == "__main__":
    download_all()