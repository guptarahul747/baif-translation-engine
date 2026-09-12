# BAIF Anuwad Studio — macOS Installation

This guide is for a BAIF Mac development or backup machine. The same application, models, and outputs run fully offline after setup.

## Prerequisites

- macOS with 16 GB RAM or more and 15 GB free storage
- Python 3.12
- FFmpeg (`ffmpeg` and `ffprobe` on `PATH`)
- The application source and approved `local_model_vault`

Install and verify prerequisites when internet is available:

```bash
brew install git python@3.12 ffmpeg
git --version
python3.12 --version
ffmpeg -version
ffprobe -version
```

## Method 1 — Prepared USB drive

1. On a staging Mac with the same CPU architecture, copy a clean repository and the approved `local_model_vault` to the USB. Exclude `.venv`, `storage_vault`, Python caches, and user media.
2. Download offline runtime wheels to the USB:

```bash
python3.12 -m pip download --only-binary=:all: --dest /Volumes/BAIF_INSTALL/wheelhouse -r requirements.txt
```

3. Include approved macOS installers for Python 3.12 and FFmpeg if the final Mac cannot use Homebrew.
4. Copy the application to an approved local folder, for example `~/BAIF/baif-translation-engine`.
5. Create the environment and install from the USB only:

```bash
cd ~/BAIF/baif-translation-engine
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links /Volumes/BAIF_INSTALL/wheelhouse -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python verify_installation.py
.venv/bin/python verify_all_models.py
```

## Method 2 — One-time internet setup

```bash
git clone <REPO_URL> baif-translation-engine
cd baif-translation-engine
git checkout <BRANCH>
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-setup.txt
export BAIF_MODEL_REPO_ID="<MODEL_REPO_ID>"
export HF_TOKEN="hf_your_read_token"
.venv/bin/python download_production_models.py
.venv/bin/python verify_installation.py
.venv/bin/python verify_all_models.py
unset HF_TOKEN
unset BAIF_MODEL_REPO_ID
```

Disconnect internet and process a new uncached file before handover.

## Start the application

```bash
cd ~/BAIF/baif-translation-engine
.venv/bin/python -m streamlit run app/web_ui.py
```

Open `http://localhost:8501`. macOS jobs use `caffeinate` while a translation is active to prevent idle system sleep; closing the laptop lid can still suspend processing.
