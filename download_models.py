import os
import sys
from huggingface_hub import snapshot_download

BASE_VAULT = os.path.join(os.path.dirname(__file__), "local_model_vault")

MODELS = {
    # STAGE 1: Speech-to-Text
    "Whisper ASR (INT8)": {
        "repo_id": "Systran/faster-whisper-medium",
        "local_dir": os.path.join(BASE_VAULT, "whisper"),
        "allow_patterns": ["*.bin", "*.json", "*.txt"]
    },
    
    # STAGE 2: AI4BHARAT Translation Engine (STILL HERE & UNCHANGED!)
    "IndicTrans2 NMT (AI4Bharat)": {
        "repo_id": "adalat-ai/ct2-rotary-indictrans2-en-indic-1B",
        "local_dir": os.path.join(BASE_VAULT, "indictrans2"),
        "allow_patterns": ["*.bin", "*.json", "*.txt", "*model*"]
    },
    
    # STAGE 3a: Voice Generation for Hindi
    "Sherpa-ONNX TTS (Hindi)": {
        "repo_id": "csukuangfj/vits-piper-hi_IN-pratham-medium",
        "local_dir": os.path.join(BASE_VAULT, "tts", "hindi"),
        "allow_patterns": ["*.onnx", "*.txt", "*.json"]
    },
    
    # STAGE 3b: Voice Generation for Marathi
    "Sherpa-ONNX TTS (Marathi)": {
        "repo_id": "shreyask/bol-tts-marathi-onnx",
        "local_dir": os.path.join(BASE_VAULT, "tts", "marathi"),
        "allow_patterns": ["*.onnx", "*.txt", "*.json"]
    }
}

def download_all_models():
    print("🚀 Provisioning Local Model Vault...")
    for name, config in MODELS.items():
        print(f"📦 Downloading {name}...")
        try:
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=config["local_dir"],
                allow_patterns=config["allow_patterns"],
                resume_download=True
            )
            print(f"✅ Saved to: {config['local_dir']}\n")
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}\n", file=sys.stderr)

if __name__ == "__main__":
    download_all_models()
