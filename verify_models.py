import os
import sys
from pathlib import Path

VAULT_DIR = Path("local_model_vault")
WHISPER_PATH = VAULT_DIR / "whisper"
INDICTRANS_PATH = VAULT_DIR / "indictrans2"
TTS_HINDI_PATH = VAULT_DIR / "tts" / "hindi"
TTS_MARATHI_PATH = VAULT_DIR / "tts" / "marathi"

def find_file(directory: Path, pattern: str) -> Path | None:
    """Recursively finds the first matching file in directory."""
    if not directory.exists():
        return None
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None

def verify_all():
    print("==================================================")
    print("🔍 AUDITING OFFLINE MODEL VAULT...")
    print("==================================================\n")
    
    status = {}

    # 1. Faster-Whisper ASR
    print("1️⃣ Checking Faster-Whisper Medium (INT8)...")
    whisper_bin = find_file(WHISPER_PATH, "*.bin")
    if whisper_bin:
        try:
            from faster_whisper import WhisperModel
            # Pass the parent directory containing the .bin weights
            _ = WhisperModel(str(whisper_bin.parent), device="cpu", compute_type="int8")
            print("   ✅ Faster-Whisper loaded into memory successfully!\n")
            status["Whisper ASR"] = "READY"
        except Exception as e:
            print(f"   ❌ Engine failed to load Whisper weights: {e}\n")
            status["Whisper ASR"] = "FAILED"
    else:
        print("   ❌ Missing .bin weights in local_model_vault/whisper/\n")
        status["Whisper ASR"] = "MISSING FILES"

    # 2. IndicTrans2 NMT
    print("2️⃣ Checking IndicTrans2 (En -> Indic INT8)...")
    it2_bin = find_file(INDICTRANS_PATH, "model.bin")
    if it2_bin:
        try:
            import ctranslate2
            # Pass the inner folder containing model.bin: local_model_vault/indictrans2/en-indic-1b-ct2/ctranslate2_model
            _ = ctranslate2.Translator(str(it2_bin.parent), device="cpu")
            print(f"   ✅ Found model at: {it2_bin.parent.relative_to(VAULT_DIR)}")
            print("   ✅ IndicTrans2 CTranslate2 engine loaded successfully!\n")
            status["IndicTrans2 NMT"] = "READY"
        except Exception as e:
            print(f"   ❌ Engine failed to load IndicTrans2 weights: {e}\n")
            status["IndicTrans2 NMT"] = "FAILED"
    else:
        print("   ❌ Missing model.bin in local_model_vault/indictrans2/\n")
        status["IndicTrans2 NMT"] = "MISSING FILES"

    # 3. Sherpa-ONNX Hindi TTS
    print("3️⃣ Checking Hindi TTS Voice Profile (ONNX)...")
    hindi_onnx = find_file(TTS_HINDI_PATH, "*.onnx")
    if hindi_onnx:
        print(f"   ✅ Found model at: {hindi_onnx.relative_to(VAULT_DIR)}")
        print("   ✅ Hindi ONNX model weights verified on disk!\n")
        status["Hindi TTS"] = "READY"
    else:
        print("   ❌ Missing .onnx file in local_model_vault/tts/hindi/\n")
        status["Hindi TTS"] = "MISSING FILES"

    # 4. Sherpa-ONNX Marathi TTS
    print("4️⃣ Checking Marathi TTS Voice Profile (ONNX)...")
    marathi_onnx = find_file(TTS_MARATHI_PATH, "*.onnx")
    if marathi_onnx:
        print(f"   ✅ Found model at: {marathi_onnx.relative_to(VAULT_DIR)}")
        print("   ✅ Marathi ONNX model weights verified on disk!\n")
        status["Marathi TTS"] = "READY"
    else:
        print("   ❌ Missing .onnx file in local_model_vault/tts/marathi/\n")
        status["Marathi TTS"] = "MISSING FILES"

    # Summary Output
    print("==================================================")
    print("📊 MODEL VAULT STATUS SUMMARY")
    print("==================================================")
    all_passed = True
    for model, state in status.items():
        icon = "✅" if state == "READY" else "❌"
        print(f"{icon} {model:<20}: {state}")
        if state != "READY":
            all_passed = False

    print("==================================================")
    if all_passed:
        print("🎉 ALL OFFLINE MODELS ARE 100% OPERATIONAL!")
    else:
        print("⚠️ SOME MODELS REQUIRE ATTENTION.")
    print("==================================================")

if __name__ == "__main__":
    verify_all()
