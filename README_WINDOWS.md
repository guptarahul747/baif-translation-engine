# BAIF Offline Translation Engine — Windows Installation & Operation Guide

This guide provides both a **One-Click Automated Solution** and a **Step-by-Step Manual Process** for setting up and running the BAIF offline translation pipeline on Windows x64.

---

## 🌟 Quick Start (One-Click Scripts)

For the easiest setup experience, use the provided Windows batch scripts (or PowerShell scripts).

### 1. Setup (One-Click)
Double-click `setup_windows.bat` or run in PowerShell:
```powershell
.\setup_windows.ps1
```
**What this script does automatically:**
- Detects Python 3.12 x64 on your system.
- Detects FFmpeg / FFprobe and automatically adds `C:\ffmpeg\bin` (or standard paths) to your Windows PATH.
- Creates the local virtual environment (`.venv`).
- Installs and upgrades all runtime and setup dependencies (`pip`, `wheel`, `packaging`, `streamlit`, `faster-whisper`, `ctranslate2`, `sherpa-onnx`, `soundfile`, `srt`, `sentencepiece`, `numpy`, `setuptools`).
- Downloads and provisions all required models into `local_model_vault/`.
- Runs complete pre-flight and model inference verification.

*Optional parameters for private repository downloads:*
```powershell
.\setup_windows.ps1 -ModelRepoId "your-org/baif-production-model-vault" -HfToken "hf_your_token"
```

### 2. Validation (One-Click)
Double-click `validate_windows.bat` or run in PowerShell:
```powershell
.\validate_windows.ps1
```
**What this script does:**
- Validates Python 3.12, FFmpeg, and FFprobe availability.
- Validates all package imports and dependency health.
- Validates all local model files in `local_model_vault/`.
- Runs end-to-end inference verification for:
  - **Whisper ASR Engine**
  - **6/6 Bidirectional NMT Translation Combinations** (EN↔HI, EN↔MR, HI↔MR)
  - **3/3 TTS Voice Engines** (English, Hindi, Marathi)
- If any check fails, it highlights the exact manual remedy needed.

### 3. Run the Application (One-Click)
Double-click `run_windows.bat` or run in PowerShell:
```powershell
.\run_windows.bat
```
- Automatically configures environment variables (`PYTHONUTF8=1`, `PATH`).
- Starts the Streamlit application.
- Opens your default web browser to: **http://localhost:8501**

---

## 📖 Step-by-Step Manual Installation Process

If you prefer to perform each step manually or need custom deployment control, follow the instructions below.

### Step 1: Install System Prerequisites
Install the following 64-bit software on Windows:
1. **Git for Windows**: [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. **Python 3.12 x64**: [https://www.python.org/downloads/](https://www.python.org/downloads/) *(Enable "Add python.exe to PATH" during setup)*
3. **FFmpeg x64**: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/) *(Extract to `C:\ffmpeg\bin` and add to PATH)*
4. **Microsoft Visual C++ 2015–2022 Redistributable x64**: [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

Verify in PowerShell:
```powershell
py -3.12 --version
git --version
ffmpeg -version
ffprobe -version
```

### Step 2: Clone the Repository & Checkout Branch
```powershell
mkdir C:\BAIF
cd C:\BAIF
git clone <YOUR_REPOSITORY_URL>
cd baif-translation-engine
git checkout baif-swagatika
git branch --show-current
```

### Step 3: Create Python Virtual Environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version
```

### Step 4: Install Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip "setuptools<70" "wheel==0.43.0" "packaging<24,>=16.8"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-setup.txt
.\.venv\Scripts\python.exe -m pip check
```
*Expected: `No broken requirements found.`*

### Step 5: Download & Provision Model Vault
#### Option A: Direct Production Download (Recommended)
```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe download_models.py
```

#### Option B: From Private Hugging Face Repository
```powershell
$env:BAIF_MODEL_REPO_ID = "swagatika15/baif-production-model-vault"
$env:HF_TOKEN = "hf_your_read_token"
.\.venv\Scripts\python.exe download_production_models.py
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
```

#### Verified Local Model Vault Structure:
```text
local_model_vault/
├── whisper/
│   ├── model.bin
│   └── config.json
├── indictrans2/
│   ├── en-indic/
│   │   ├── model.bin
│   │   ├── config.json
│   │   └── vocab/ (model.SRC, model.TGT)
│   └── indic-en/
│       ├── model.bin
│       ├── config.json
│       └── vocab/ (model.SRC, model.TGT)
└── tts/
    ├── english/ (en_US-lessac-medium.onnx, tokens.txt, espeak-ng-data)
    ├── hindi/   (hi_IN-pratham-medium.onnx, tokens.txt, espeak-ng-data)
    └── marathi/ (mr_IN-google-medium.onnx, tokens.txt, espeak-ng-data)
```

### Step 6: Verify Installation & Models
```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe verify_installation.py
.\.venv\Scripts\python.exe verify_all_models.py
```
*Expected summary:*
```text
• Model Vault Files      : ✅ PASSED
• Whisper ASR Engine     : ✅ PASSED
• 6/6 NMT Combinations   : ✅ PASSED
• 3/3 TTS Voice Engines  : ✅ PASSED
```

### Step 7: Launch Application
```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py
```
Open browser to `http://localhost:8501`.

---

## 🔒 Offline Operation & Disconnecting Internet

1. After verification succeeds, remove any temporary tokens:
   ```powershell
   Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
   Remove-Item Env:BAIF_MODEL_REPO_ID -ErrorAction SilentlyContinue
   ```
2. Disconnect the machine from the internet.
3. Launch using `.\run_windows.bat`.
4. Test with a new, uncached file. All translation, transcription, and TTS synthesis operations will execute completely locally and offline.

---

## 🛠️ Troubleshooting

### 1. `ffmpeg` or `ffprobe` is not recognized
- Make sure FFmpeg is extracted (e.g. to `C:\ffmpeg\bin`).
- Run `setup_windows.bat` or `validate_windows.bat` — it automatically detects `C:\ffmpeg\bin` and adds it to your user PATH.
- Or manually add `C:\ffmpeg\bin` to Windows System/User Environment Variables.

### 2. Python 3.12 not recognized
- Install Python 3.12 x64 from python.org.
- Ensure `py -3.12 --version` or `python --version` returns `3.12.x`.

### 3. Port 8501 already in use
- Launch on a custom port:
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run app\web_ui.py --server.port 8502
  ```
  or with the run script:
  ```powershell
  .\run_windows.ps1 -Port 8502
  ```

### 4. Character Encoding Warnings in PowerShell
- Set UTF-8 encoding in your PowerShell session:
  ```powershell
  $env:PYTHONUTF8 = "1"
  $env:PYTHONIOENCODING = "utf-8"
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  ```
