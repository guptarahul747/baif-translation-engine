# 🌾 BAIF Offline Multilingual Translation Engine
## Complete Setup, Deployment & Operations Guide

> **Battle-tested on Apple M4 (Dev) + Windows 11 Intel i5 (Production)**  
> Every command copy-paste ready. Every known error documented with its fix.

---

## 📋 Pre-Flight Checklist

| Item | Requirement | Notes |
|------|-------------|-------|
| RAM | ≥ 16 GB | Models need ~2.5 GB headroom |
| Storage | ≥ 10 GB free | Models + vault |
| CPU | Apple M4 / Intel i5 11th Gen+ | ARM64 / AVX-2 acceleration |
| OS | macOS (M4 dev) / Windows 11 (production) | Both fully supported |
| Python | 3.12.x only | ≥ 3.10 required by ctranslate2 |
| Internet | Setup only | 100% offline after model download |
| Cost | ₹0 recurring | All open-source models |

> ⚠️ **Safe to install Python 3.12 alongside existing Python 3.9** — they live in separate paths and never conflict. Project runs entirely inside a virtual environment.

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│      STAGE 1        │     │      STAGE 2        │     │      STAGE 3        │     │      STAGE 4        │
│  Speech Recognition │ ──► │ Machine Translation │ ──► │  Voice Synthesis    │ ──► │   Video Muxing      │
│   Faster-Whisper    │     │   IndicTrans2 NMT   │     │  Sherpa-ONNX VITS   │     │      FFmpeg         │
│      INT8 CPU       │     │   SOV Grammar       │     │  Hindi / Marathi    │     │   Audio Merge       │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

| Stage | Task | Model | Format |
|-------|------|-------|--------|
| Stage 1 | ASR | `Systran/faster-whisper-medium` | CTranslate2 INT8 |
| Stage 2 | NMT | `adalat-ai/ct2-rotary-indictrans2-en-indic-1B` | CTranslate2 |
| Stage 3a | Hindi TTS | `csukuangfj/vits-piper-hi_IN-pratham-medium` | ONNX |
| Stage 3b | Marathi TTS | `csukuangfj/vits-piper-mr_IN-google-medium` | ONNX |

---

## 🗂️ Platform Architecture Matrix

| Parameter | Mac (Dev) | Windows (BAIF Production) |
|-----------|-----------|--------------------------|
| OS | macOS Apple Silicon M4 | Windows 11 Enterprise 64-bit |
| CPU | 10-Core ARM64 | Intel Core i5 6-Core |
| Acceleration | Apple Accelerate C++ | Intel AVX-2 Vector Instructions |
| Python | 3.12 via `.pkg` installer | 3.12 AMD64 installer |
| Package Manager | Homebrew (manual install) | Direct installers |
| FFmpeg | 6.x via Homebrew | 6.x static build |

---

## 📁 Repository Structure

```
baif-translation-engine/
├── app/
│   ├── main.py                      ← FastAPI backend orchestrator
│   ├── web_ui.py                    ← Streamlit operator dashboard
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── database.py              ← SQLite SHA-256 cache engine
│   │   ├── ingestion.py             ← FFmpeg + Silero VAD wrappers
│   │   └── inference.py             ← CTranslate2 / ONNX interfaces
│   ├── run_app.sh                   ← macOS boot script
│   └── run_server.bat               ← Windows boot script
├── local_model_vault/               ← All AI models (offline)
│   ├── whisper/                     ← Faster-Whisper Medium INT8
│   ├── indictrans2/                 ← IndicTrans2 En-Indic CT2
│   └── tts/
│       ├── hindi/                   ← hi_IN-pratham-medium.onnx
│       └── marathi/                 ← mr_IN-google-medium.onnx
├── storage_vault/
│   ├── inputs/                      ← Raw uploaded media files
│   └── outputs/                     ← Translated media + subtitles
├── logs/                            ← Runtime logs
├── download_models.py               ← Model download script
├── verify_models.py                 ← Model integrity checker
├── test_whisper_asr.py              ← Stage 1: ASR test script
├── run_step1_translation.py         ← Stage 2: NMT translation
├── run_step2_tts_srt.py             ← Stage 3: Hindi TTS
├── run_step2_tts_srt_mr.py          ← Stage 3: Marathi TTS
├── test_video_muxing.py             ← Stage 4: Video mux test
└── requirements.txt
```

---

## 🗂️ Step 0 — Create Project Skeleton

Run once on either platform:

```bash
# macOS
mkdir -p baif-translation-engine && cd baif-translation-engine
mkdir -p app/modules \
         local_model_vault/whisper \
         local_model_vault/indictrans2 \
         local_model_vault/tts/hindi \
         local_model_vault/tts/marathi \
         storage_vault/inputs \
         storage_vault/outputs \
         logs
```

```powershell
# Windows PowerShell
mkdir baif-translation-engine
cd baif-translation-engine
mkdir app\modules, `
      local_model_vault\whisper, `
      local_model_vault\indictrans2, `
      local_model_vault\tts\hindi, `
      local_model_vault\tts\marathi, `
      storage_vault\inputs, `
      storage_vault\outputs, `
      logs
```

---

## 🍏 macOS Setup (Apple M4 + Xcode 26)

> ⚠️ **Xcode 26 known issue:** Standard Homebrew installer triggers a Command Line Tools popup that fails with *"not currently available from the Software Update server"* — Apple hasn't released CLT for Xcode 26 yet. Follow exactly as written below.

---

### Mac Step 1 — Fix xcode-select

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

Verify:
```bash
xcode-select -p
# Expected: /Applications/Xcode.app/Contents/Developer

clang --version
# Expected: Apple clang version 17.x.x
```

---

### Mac Step 2 — Install Homebrew (manual method)

> Standard Homebrew installer fails on Xcode 26 with CLT popup. This manual method bypasses it entirely.

```bash
cd /opt
sudo mkdir homebrew
sudo chown $(whoami) homebrew
curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip-components 1 -C homebrew
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zprofile
source ~/.zprofile
brew update
```

Verify:
```bash
brew --version
# Expected: Homebrew 6.x.x
```

---

### Mac Step 3 — Install pkg-config + FFmpeg 6.x

> ⚠️ **FFmpeg 6.x is required.** The `av` package (dependency of faster-whisper) is incompatible with FFmpeg 7.x/8.x and throws `AV_OPT_TYPE_CHANNEL_LAYOUT` build errors.

```bash
brew install pkg-config
brew install ffmpeg@6
brew link ffmpeg@6 --force
```

Verify:
```bash
pkg-config --version      # 0.29.x
ffmpeg -version | head -1 # ffmpeg version 6.x.x
```

---

### Mac Step 4 — Install Python 3.12

> Install via direct `.pkg` — avoids any Homebrew/CLT dependency.

```bash
curl -O https://www.python.org/ftp/python/3.12.3/python-3.12.3-macos11.pkg
open python-3.12.3-macos11.pkg
```

Follow the GUI installer. Verify:
```bash
python3.12 --version
# Expected: Python 3.12.3
```

> ✅ Do NOT add Homebrew Python path to `.zprofile` if you installed via `.pkg` — it adds itself automatically.

---

### Mac Step 5 — Create Virtual Environment

```bash
cd baif-translation-engine
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

You should see `(venv)` prefix:
```
(venv) yourname@MacBook baif-translation-engine %
```

> 💡 Every new terminal session: `source venv/bin/activate`

---

### Mac Step 6 — Install sherpa-onnx First

> ⚠️ Must be installed separately before `requirements.txt` to resolve ARM64 wheel correctly.

```bash
pip install sherpa-onnx==1.13.4
```

---

### Mac Step 7 — Install All Dependencies

```bash
cat > requirements.txt << 'EOF'
streamlit==1.32.0
fastapi==0.110.0
uvicorn==0.28.0
python-multipart==0.0.9
faster-whisper==1.0.1
ctranslate2==4.1.0
huggingface_hub==0.21.4
soundfile==0.12.1
srt==3.5.3
openpyxl==3.1.2
numpy==1.26.4
EOF

pip install -r requirements.txt
```

> ⚠️ `librosa` excluded — pulls in `av` which conflicts with FFmpeg 8.x  
> ⚠️ `sherpa-onnx` excluded — already installed in Step 6  
> ⏱️ Takes 8–12 minutes — don't interrupt

---

### Mac Step 8 — Verify Installation

```bash
python3 -c "
import importlib.metadata
packages = [
    'faster-whisper', 'ctranslate2', 'sherpa-onnx',
    'streamlit', 'fastapi', 'soundfile', 'srt',
    'numpy', 'huggingface-hub', 'python-multipart',
    'uvicorn', 'openpyxl',
]
print('━' * 45)
print('  BAIF Dependency Verification')
print('━' * 45)
all_ok = True
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f'  ✅  {pkg:<25} {version}')
    except importlib.metadata.PackageNotFoundError:
        print(f'  ❌  {pkg:<25} NOT FOUND')
        all_ok = False
print('━' * 45)
print('  ✅  All dependencies verified.' if all_ok else '  ⚠️  Some packages missing.')
print('━' * 45)
"
```

Expected output:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BAIF Dependency Verification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅  faster-whisper            1.0.1
  ✅  ctranslate2               4.1.0
  ✅  sherpa-onnx               1.13.4
  ✅  streamlit                 1.32.0
  ✅  fastapi                   0.110.0
  ✅  soundfile                 0.12.1
  ✅  srt                       3.5.3
  ✅  numpy                     1.26.4
  ✅  huggingface-hub           0.21.4
  ✅  python-multipart          0.0.9
  ✅  uvicorn                   0.28.0
  ✅  openpyxl                  3.1.2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅  All dependencies verified.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### Mac Step 9 — Download AI Models

Save as `download_models.py` in project root:

```python
import os
import sys
from huggingface_hub import snapshot_download

BASE_VAULT = os.path.join(os.path.dirname(__file__), "local_model_vault")

MODELS = {
    "Whisper ASR (INT8)": {
        "repo_id": "Systran/faster-whisper-medium",
        "local_dir": os.path.join(BASE_VAULT, "whisper"),
        "allow_patterns": ["*.bin", "*.json", "*.txt"]
    },
    "IndicTrans2 NMT (English → Indic)": {
        "repo_id": "adalat-ai/ct2-rotary-indictrans2-en-indic-1B",
        "local_dir": os.path.join(BASE_VAULT, "indictrans2"),
        "allow_patterns": ["*.bin", "*.json", "*.txt", "*model*"]
    },
    "Sherpa-ONNX TTS (Hindi)": {
        "repo_id": "csukuangfj/vits-piper-hi_IN-pratham-medium",
        "local_dir": os.path.join(BASE_VAULT, "tts", "hindi"),
        "allow_patterns": ["*.onnx", "*.txt", "*.json"]
    },
    "Sherpa-ONNX TTS (Marathi)": {
        "repo_id": "csukuangfj/vits-piper-mr_IN-google-medium",
        "local_dir": os.path.join(BASE_VAULT, "tts", "marathi"),
        "allow_patterns": ["*.onnx", "*.txt", "*.json"]
    }
}

def download_all_models():
    print("=" * 50)
    print("🚀 Starting Offline Model Vault Provisioning...")
    print("=" * 50 + "\n")
    for name, config in MODELS.items():
        print(f"📦 Downloading {name}...")
        print(f"   Repo : {config['repo_id']}")
        print(f"   Path : {config['local_dir']}")
        try:
            snapshot_download(
                repo_id=config["repo_id"],
                local_dir=config["local_dir"],
                allow_patterns=config["allow_patterns"],
            )
            print(f"✅ Done: {name}\n")
        except Exception as e:
            print(f"❌ Failed: {name}\n   Error: {e}\n", file=sys.stderr)
    print("=" * 50)
    print("🎉 All models synchronized!")
    print("=" * 50)

if __name__ == "__main__":
    download_all_models()
```

Run:
```bash
python3 download_models.py
```

> ℹ️ Uses `adalat-ai` repo — **no HuggingFace account or token required**  
> ⏱️ Total ~2.1 GB · Auto-resumes if interrupted · Safe to re-run

**Model sizes:**

| Model | Folder | Size |
|-------|--------|------|
| Faster-Whisper Medium INT8 | `local_model_vault/whisper/` | ~770 MB |
| IndicTrans2 1B INT8 | `local_model_vault/indictrans2/` | ~1.2 GB |
| VITS Hindi ONNX | `local_model_vault/tts/hindi/` | ~60 MB |
| VITS Marathi ONNX | `local_model_vault/tts/marathi/` | ~60 MB |

---

### Mac Step 10 — Initialize Database

```bash
python3 app/modules/database.py
# Expected: ✅ Database schema initialized
```

---

### Mac Step 11 — Create Boot Script

```bash
cat > app/run_app.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
source venv/bin/activate
echo "🚀 Starting FastAPI Backend on port 8000..."
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
sleep 2
echo "🌐 Starting Streamlit UI on port 8501..."
streamlit run app/web_ui.py --server.port 8501
EOF

chmod +x app/run_app.sh
```

---

## 🪟 Windows 11 Setup (BAIF Production Server)

> This is the **primary deployment target**. Run all commands in **PowerShell as Administrator**.

---

### Win Step 1 — Install Python 3.12

1. Download: **[Python 3.12.3 AMD64 Installer](https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe)**
2. Run installer
3. ✅ **Check "Add Python to PATH"** on first screen — critical
4. Click **Install Now**

Verify in new PowerShell:
```powershell
python --version
# Expected: Python 3.12.3
```

---

### Win Step 2 — Install FFmpeg 6.x

> ⚠️ Use FFmpeg **6.x only** — NOT 7.x or 8.x. The `av` package is incompatible with FFmpeg 7+.

1. Download: **[FFmpeg 6.1.2 Essentials Build](https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-6.1.2-essentials_build.zip)**
2. Extract ZIP → rename folder to `ffmpeg` → move to `C:\ffmpeg\`

Add to System PATH:
```powershell
$ffmpegPath = "C:\ffmpeg\bin"
$currentPath = [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($currentPath -notlike "*$ffmpegPath*") {
    [System.Environment]::SetEnvironmentVariable("PATH", "$currentPath;$ffmpegPath", "Machine")
    Write-Host "✅ FFmpeg added to PATH"
} else {
    Write-Host "✅ FFmpeg already in PATH"
}
```

Close and reopen PowerShell, verify:
```powershell
ffmpeg -version | Select-Object -First 1
# Expected: ffmpeg version 6.1.2
```

---

### Win Step 3 — Install Visual C++ Redistributables

Required for CTranslate2 and Sherpa-ONNX native runtimes.

1. Download: **[VC++ Redistributables VS 2015–2022](https://aka.ms/vs/17/release/vc_redist.x64.exe)**
2. Run → Install → restart if prompted

---

### Win Step 4 — Create Virtual Environment

```powershell
cd C:\path\to\baif-translation-engine
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
```

You should see `(venv)` prefix:
```
(venv) PS C:\baif-translation-engine>
```

> Execution policy error? Fix with:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

> 💡 Every new PowerShell session: `.\venv\Scripts\Activate.ps1`

---

### Win Step 5 — Install sherpa-onnx First

```powershell
pip install sherpa-onnx==1.13.4
```

---

### Win Step 6 — Install All Dependencies

```powershell
@"
streamlit==1.32.0
fastapi==0.110.0
uvicorn==0.28.0
python-multipart==0.0.9
faster-whisper==1.0.1
ctranslate2==4.1.0
huggingface_hub==0.21.4
soundfile==0.12.1
srt==3.5.3
openpyxl==3.1.2
numpy==1.26.4
"@ | Out-File -FilePath requirements.txt -Encoding UTF8

pip install -r requirements.txt
```

> ⏱️ Takes 8–12 minutes — don't interrupt

---

### Win Step 7 — Verify Installation

```powershell
python -c "
import importlib.metadata
packages = [
    'faster-whisper', 'ctranslate2', 'sherpa-onnx',
    'streamlit', 'fastapi', 'soundfile', 'srt',
    'numpy', 'huggingface-hub', 'python-multipart',
    'uvicorn', 'openpyxl',
]
print('=' * 45)
print('  BAIF Dependency Verification')
print('=' * 45)
all_ok = True
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f'  OK  {pkg:<25} {version}')
    except importlib.metadata.PackageNotFoundError:
        print(f'  MISSING  {pkg}')
        all_ok = False
print('=' * 45)
print('  All verified.' if all_ok else '  Some packages missing.')
print('=' * 45)
"
```

---

### Win Step 8 — Download AI Models

```powershell
python download_models.py
```

> ℹ️ Total ~2.1 GB · No HuggingFace account needed · Auto-resumes if interrupted

---

### Win Step 9 — Verify Model Integrity

```powershell
python verify_models.py
```

---

### Win Step 10 — Initialize Database

```powershell
python app\modules\database.py
```

---

### Win Step 11 — Create Boot Script

```powershell
@"
@echo off
cd /d "%~dp0.."
call venv\Scripts\activate.bat
echo.
echo Starting BAIF FastAPI Backend on port 8000...
start /B uvicorn app.main:app --host 0.0.0.0 --port 8000
timeout /t 3 /nobreak > NUL
echo Starting BAIF Streamlit UI on port 8501...
streamlit run app\web_ui.py --server.port 8501 --server.address 0.0.0.0
"@ | Out-File -FilePath app\run_server.bat -Encoding ASCII
```

> ℹ️ `0.0.0.0` binding allows all BAIF intranet users to access via server IP

---

## 🚀 Running the App

### Start

| Platform | Command |
|----------|---------|
| Mac (dev) | `./app/run_app.sh` |
| Windows (production) | `.\app\run_server.bat` |

| Access Point | URL |
|---|---|
| Local browser | `http://localhost:8501` |
| BAIF intranet | `http://192.168.1.X:8501` |
| API docs | `http://localhost:8000/docs` |

---

### Stop

| Situation | Mac | Windows |
|-----------|-----|---------|
| Terminal open | `Ctrl + C` | `Ctrl + C` |
| Background | `pkill -f "streamlit run" && pkill -f "uvicorn"` | `taskkill /F /IM "streamlit.exe"` |
| By port | `lsof -ti:8501 \| xargs kill -9` | `netstat -ano \| findstr :8501` → `taskkill /PID <PID> /F` |

---

### Run in Background (Mac)

```bash
mkdir -p logs
nohup ./app/run_app.sh > logs/app.log 2>&1 &

# Check status
ps aux | grep -E "streamlit|uvicorn"

# View logs
tail -f logs/app.log
```

---

## 🚀 Pipeline Execution Guide

### Pre-Pipeline Setup — Download Marathi TTS Model

Before running the complete pipeline, download the Marathi TTS model:

```bash
python3 setup_marathi_piper.py
```

> ℹ️ This downloads the Marathi Piper TTS model to `local_model_vault/tts/marathi/`  
> ⏱️ ~64 MB download · One-time setup

---

### Generate Final Video — Complete Pipeline Execution

Once all models are downloaded, run these four commands sequentially to generate the final dubbed video with Marathi audio and subtitles:

> 📁 **Sample Video File Location:**  
> `storage_vault/inputs/SampleVideo.mp4`

#### Command 1: Speech Recognition (ASR)

```bash
python3 test_whisper_asr.py storage_vault/inputs/SampleVideo.mp4
```

**Input:** `storage_vault/inputs/SampleVideo.mp4`  
**Output:** `storage_vault/outputs/step1_whisper_output.json`

---

#### Command 2: Machine Translation (NMT)

```bash
python3 run_step1_translation.py storage_vault/outputs/step1_whisper_output.json mr
```

**Input:** `storage_vault/outputs/step1_whisper_output.json`  
**Output:** `storage_vault/outputs/step2_offline_translated.json`  
**Language:** `mr` (Marathi)

---

#### Command 3: Text-to-Speech (TTS) — Marathi

```bash
python3 run_step2_tts_srt_mr.py storage_vault/outputs/step2_offline_translated.json mr
```

**Input:** `storage_vault/outputs/step2_offline_translated.json`  
**Output:** 
- `storage_vault/outputs/audio_segments/` — Individual `.wav` files
- `storage_vault/outputs/video_subtitles_mr.srt` — Marathi subtitle file

---

#### Command 4: Video Muxing — Final Dubbed Video

```bash
python3 test_video_muxing.py storage_vault/inputs/SampleVideo.mp4 mr
```

**Input:** 
- `storage_vault/inputs/SampleVideo.mp4` — Original video
- Audio segments and subtitles from Stage 3

**Output:** `storage_vault/outputs/translated_video_mr.mp4` — Final dubbed video with Marathi audio and burned-in subtitles

---

### Complete Pipeline Summary

| Step | Command | Input | Output |
|------|---------|-------|--------|
| 1 | `python3 test_whisper_asr.py storage_vault/inputs/SampleVideo.mp4` | Video file | Transcription JSON |
| 2 | `python3 run_step1_translation.py storage_vault/outputs/step1_whisper_output.json mr` | Transcription | Translated JSON |
| 3 | `python3 run_step2_tts_srt_mr.py storage_vault/outputs/step2_offline_translated.json mr` | Translation | Audio + SRT file |
| 4 | `python3 test_video_muxing.py storage_vault/inputs/SampleVideo.mp4 mr` | Video + Audio | Final dubbed video |

---

## 🎙️ TTS Module Reference

### Hindi TTS (`run_step2_tts_srt.py`)

```python
import os
from pathlib import Path
import sherpa_onnx

HINDI_TTS_DIR = Path("local_model_vault/tts/hindi")
MODEL_PATH    = HINDI_TTS_DIR / "hi_IN-pratham-medium.onnx"
TOKENS_PATH   = HINDI_TTS_DIR / "tokens.txt"
DATA_DIR      = HINDI_TTS_DIR / "espeak-ng-data"

def initialize_hindi_tts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Hindi model not found at: {MODEL_PATH}")

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(MODEL_PATH),
                tokens=str(TOKENS_PATH) if TOKENS_PATH.exists() else "",
                data_dir=str(DATA_DIR) if DATA_DIR.exists() else "",
            ),
            provider="cpu",
            num_threads=4,
        )
    )

    if not tts_config.validate():
        raise ValueError("Invalid Hindi TTS config. Check model files.")

    return sherpa_onnx.OfflineTts(tts_config)

def generate_hindi_audio(text: str, output_wav_path: str):
    tts   = initialize_hindi_tts()
    audio = tts.generate(text, sid=0, speed=1.0)

    if len(audio.samples) == 0:
        print(f"⚠️ Warning: Empty audio for: '{text}'")
        return

    sherpa_onnx.write_wave(output_wav_path, audio.samples, audio.sample_rate)
    print(f"✅ Hindi audio saved: {output_wav_path}")

if __name__ == "__main__":
    sample_text = "नमस्ते, यह हिंदी आवाज निर्माण का एक उदाहरण है।"
    generate_hindi_audio(sample_text, "output_hindi.wav")
```

---

## 🔧 Troubleshooting

### macOS Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Homebrew CLT popup fails | Xcode 26 CLT not on Apple servers | Use manual Homebrew install via GitHub tarball (Mac Step 2) |
| `xcode-select: invalid developer directory` | CLT path wrong | `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` |
| `AV_OPT_TYPE_CHANNEL_LAYOUT` build error | FFmpeg 8.x incompatible with `av` | `brew install ffmpeg@6 && brew link ffmpeg@6 --force` |
| `pkg-config is required for building PyAV` | pkg-config missing | `brew install pkg-config` |
| `.pkg` gives `com.apple.installer error -1` | Corrupted download | Delete and re-download `python-3.12.3-macos11.pkg` |
| `sherpa-onnx` not found after pip install | ARM64 wheel conflict | `pip install sherpa-onnx==1.13.4` separately before requirements |
| `module 'srt' has no __version__` | `srt` package quirk | Not an error — use `importlib.metadata.version('srt')` |
| `No module named 'pkg_resources'` | Not in Python 3.12 | Use `importlib.metadata` instead |
| `zsh: command not found: #` | Pasted comment line | Harmless — ignore |

### Windows Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `python` not found | PATH not set | Re-run installer, check "Add to PATH" |
| `ffmpeg` not found | PATH not set | Verify `C:\ffmpeg\bin` in System PATH, restart PowerShell |
| `activate.ps1` execution error | Script policy | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `vcruntime140.dll` missing | VC++ not installed | Install VC++ Redistributables (Win Step 3) |
| `ctranslate2` import error | VC++ missing | Install redistributables + restart PowerShell |
| Port 8501 in use | Another process | `netstat -ano \| findstr :8501` → `taskkill /PID <PID> /F` |

### Both Platforms

| Error | Fix |
|-------|-----|
| `401 gated repo` on IndicTrans2 | Use `adalat-ai/ct2-rotary-indictrans2-en-indic-1B` — no login needed |
| `FileNotFoundError` on TTS model | Run `python download_models.py` |
| Model download interrupted | Re-run — `snapshot_download` auto-resumes |
| `resume_download` deprecation warning | Remove from script — resuming is now default |
| Empty WAV generated | Check input is valid Hindi/Marathi unicode |
| `(venv)` not showing | `source venv/bin/activate` (Mac) / `.\venv\Scripts\Activate.ps1` (Windows) |

---

## 🚀 Quick Reference

```
Web UI (local)       →  http://localhost:8501
API Docs             →  http://localhost:8000/docs
BAIF Intranet        →  http://192.168.1.X:8501
Database             →  storage_vault/translation_cache.db
Outputs              →  storage_vault/outputs/
Logs                 →  logs/app.log
Models               →  local_model_vault/
```

```
Recurring cost       →  ₹0
Internet after setup →  None
GPU required         →  None
RAM usage            →  < 2.5 GB
CPU speedup          →  4× vs FP32 (AVX-2 / Apple Accelerate)
```
