import json
import os
import shutil
import ssl
import urllib.request
from pathlib import Path

# --- Configuration ---
VAULT_DIR = Path("./local_model_vault/tts/marathi")
VOICES_INDEX_URL = "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json"
BASE_HF_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def get_ssl_context():
    """Bypasses macOS certificate verification issues."""
    return ssl._create_unverified_context()


def download_file(url: str, dest_path: Path):
    """Downloads a file with progress output."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    )
    with urllib.request.urlopen(req, context=get_ssl_context()) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8192

        while chunk := response.read(chunk_size):
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  └─ {dest_path.name}: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        print()


def setup_marathi_piper():
    print("🧹 Cleaning old Marathi TTS vault...")
    if VAULT_DIR.exists():
        shutil.rmtree(VAULT_DIR)
    VAULT_DIR.mkdir(parents=True, exist_ok=True)

    print("🔍 Fetching official Piper voices index...")
    try:
        req = urllib.request.Request(
            VOICES_INDEX_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, context=get_ssl_context()) as resp:
            voices_data = json.loads(resp.read().decode())

        # Find any Marathi voice entry in the official catalog
        marathi_voices = {
            key: info for key, info in voices_data.items()
            if key.startswith("mr_") or info.get("language", {}).get("code") == "mr_IN"
        }

        if not marathi_voices:
            print("❌ No Marathi voices found in the official Piper catalog.")
            return

        # Pick the primary Marathi voice key
        voice_key = list(marathi_voices.keys())[0]
        voice_info = marathi_voices[voice_key]
        print(f"🎯 Selected model: '{voice_key}'")

        files_map = voice_info.get("files", {})
        print(f"📥 Downloading model bundle...")

        for rel_path in files_map.keys():
            file_url = f"{BASE_HF_URL}/{rel_path}"
            filename = Path(rel_path).name
            dest_file = VAULT_DIR / filename
            print(f"  Downloading {filename}...")
            download_file(file_url, dest_file)

        print("\n✅ Marathi Piper TTS assets successfully installed!")
        print(f"📁 Vault location: {VAULT_DIR.resolve()}\n")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")


if __name__ == "__main__":
    setup_marathi_piper()