# BAIF Anuwad Studio — Windows Installation and Handover

This is the production guide for the final BAIF Windows laptop. After installation, the application runs fully offline: no HSBC systems, cloud APIs, account login, or internet connection is used during normal translation.

## Final laptop requirements

- Windows 10/11 x64, 16 GB RAM or more, and at least 15 GB free storage
- Python 3.12 x64
- FFmpeg x64 (`ffmpeg` and `ffprobe` available on `PATH`)
- Microsoft Visual C++ 2015–2022 Redistributable x64
- BAIF application source and the approved `local_model_vault`

## Normal daily startup

```powershell
Set-Location C:\BAIF\baif-translation-engine
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py
```

Open `http://localhost:8501`. Internet is not required.

## Method 1 — Prepared USB drive (no internet on final laptop)

Use this method when the final BAIF laptop must never be connected to the internet.

### Prepare the USB drive on a staging Windows PC

The staging PC must be Windows x64 with Python 3.12 x64. It may use internet once.

1. Create this layout on the USB drive:

```text
BAIF_INSTALL/
  baif-translation-engine/  application source and approved local_model_vault
  wheelhouse/               offline Python package wheels
  installers/               Python, FFmpeg, VC++ and optional Git installers
```

2. Copy a clean application source folder to `BAIF_INSTALL/baif-translation-engine`. Do not copy `.venv`, `storage_vault`, `__pycache__`, caches, or user media.

3. Copy the approved and verified production vault to `BAIF_INSTALL/baif-translation-engine/local_model_vault`. It must contain Whisper, `indictrans2/en-indic`, `indictrans2/indic-en`, and English/Hindi/Marathi TTS folders.

4. Download all runtime package wheels to the USB, using the same Python version and architecture as the final laptop:

```powershell
py -3.12 -m pip download --only-binary=:all: --dest E:\BAIF_INSTALL\wheelhouse -r requirements.txt
```

Replace `E:` with the USB drive letter. Include Python 3.12 x64, FFmpeg x64, and the VC++ Redistributable x64 installers under `installers`.

### Install on the final BAIF laptop

1. Copy the application folder from USB to `C:\BAIF\baif-translation-engine`.
2. Install Python 3.12 x64, FFmpeg x64, and VC++ from the USB. Add the FFmpeg `bin` directory to `PATH`.
3. Open a new PowerShell window and verify:

```powershell
py -3.12 --version
ffmpeg -version
ffprobe -version
```

4. Create the environment and install only from the USB wheelhouse:

```powershell
Set-Location C:\BAIF\baif-translation-engine
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-index --find-links E:\BAIF_INSTALL\wheelhouse -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

5. Verify and start:

```powershell
.\.venv\Scripts\python.exe verify_installation.py
.\.venv\Scripts\python.exe verify_all_models.py
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py
```

Process one new short audio/video file as the final acceptance test. No Hugging Face token, Git login, or internet is needed for this method.

## Method 2 — One-time internet setup

Use this method only when the final laptop is allowed temporary internet during setup.

### 1. Install local prerequisites

Install BAIF-approved versions of Python 3.12 x64 (enable **Add python.exe to PATH**), Git for Windows, FFmpeg x64, and Microsoft Visual C++ 2015–2022 Redistributable x64. Add the FFmpeg `bin` folder to `PATH`.

Verify in a new PowerShell window:

```powershell
py -3.12 --version
git --version
ffmpeg -version
ffprobe -version
```

### 2. Download the application and packages

```powershell
New-Item -ItemType Directory -Force C:\BAIF | Out-Null
Set-Location C:\BAIF
git clone <REPO_URL> baif-translation-engine
Set-Location .\baif-translation-engine
git checkout <BRANCH>

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-setup.txt
.\.venv\Scripts\python.exe -m pip check
```

### 3. Download the approved model vault once

Use a temporary read-only Hugging Face token only in the current PowerShell session:

```powershell
$env:BAIF_MODEL_REPO_ID = "<MODEL_REPO_ID>"
$env:HF_TOKEN = "hf_your_read_token"
.\.venv\Scripts\python.exe download_production_models.py
```

### 4. Verify, remove credentials, and prove offline use

```powershell
.\.venv\Scripts\python.exe verify_installation.py
.\.venv\Scripts\python.exe verify_all_models.py
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:BAIF_MODEL_REPO_ID -ErrorAction SilentlyContinue
```

Disconnect internet. Start the app and process a **new uncached** file:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py
```

If that succeeds, the final laptop is ready for offline BAIF use.

## Operating notes

- Configure Windows **Sleep** to **Never when plugged in** for production use. Locking the screen should not stop processing, but sleep, hibernation, and closing the lid can pause local applications.
- Never copy a `.venv` between machines; create it locally.
- Do not copy a previous `storage_vault`; it can contain user media and cached results.
- Remove Hugging Face tokens and any internet credentials after handover.
