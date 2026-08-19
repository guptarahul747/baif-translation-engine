# BAIF Offline Translation Engine — macOS Fresh Installation

This guide installs the **stable last-known-good BAIF pipeline** from scratch on a Mac.

## Final runtime capabilities

- Fully offline after first-time setup
- English, Hindi and Marathi
- English → Hindi
- English → Marathi
- Hindi → English
- Marathi → English
- Hindi → Marathi through English pivot
- Marathi → Hindi through English pivot
- Text, audio and video input through the Streamlit UI
- Offline Whisper ASR
- Offline IndicTrans2 translation
- Offline English/Hindi/Marathi TTS
- Subtitles, dubbed audio/video and optional summary
- Persistent completed-result cache

---

## 0. Values needed before setup

Have these ready:

```text
REPO_URL=<your Git repository URL>
BRANCH=<your deployment branch>
MODEL_REPO_ID=<private Hugging Face model repo prepared with MODEL_DISTRIBUTION_GUIDE.md>
```

The BAIF Mac must have internet access only for the installation/download steps.

---

## 1. Install system prerequisites

Recommended: Homebrew.

```bash
brew install git
brew install python@3.12
brew install ffmpeg
```

Verify:

```bash
git --version
python3.12 --version
ffmpeg -version
ffprobe -version
```

Python should be `3.12.x`.

If Homebrew is not permitted, install Git, Python 3.12 and FFmpeg using BAIF-approved installers and make sure they are available on `PATH`.

---

## 2. Clone the repository

```bash
git clone <REPO_URL>
cd baif-translation-engine
git checkout <BRANCH>
```

Check:

```bash
git branch --show-current
```

---

## 3. Create the virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
python --version
which python
```

The Python path should point inside this repo's `.venv`.

---

## 4. Install Python dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-setup.txt
python -m pip check
```

`requirements-setup.txt` is only needed for the one-time model download utility.

---

## 5. Download the exact production models

The production model package is stored in a private Hugging Face repository prepared from the verified development machine.

### Option A — temporary environment token

```bash
export BAIF_MODEL_REPO_ID="<MODEL_REPO_ID>"
export HF_TOKEN="hf_your_read_token"

python download_production_models.py
```

When download succeeds:

```bash
unset HF_TOKEN
```

### Option B — interactive Hugging Face login

```bash
hf auth login
export BAIF_MODEL_REPO_ID="<MODEL_REPO_ID>"
python download_production_models.py
```

After successful download and verification, you can run:

```bash
hf auth logout
```

The resulting directory must be:

```text
local_model_vault/
├── whisper/
├── indictrans2/
│   ├── en-indic/
│   └── indic-en/
└── tts/
    ├── english/
    ├── hindi/
    └── marathi/
```

No experimental Indic-Indic 320M model is required by this stable build.

---

## 6. Run the fast installation pre-flight

```bash
python verify_installation.py
```

Expected ending:

```text
✅ Pre-flight passed.
Next: python verify_all_models.py
```

---

## 7. Run full model verification

```bash
python verify_all_models.py
```

The important result is:

```text
Whisper ASR Engine       ✅
6/6 NMT combinations     ✅
3/3 TTS voice engines    ✅
```

Do not proceed to handover if any required model fails.

---

## 8. Start the application

```bash
streamlit run app/web_ui.py
```

Open:

```text
http://localhost:8501
```

The stable Streamlit UI executes the pipeline scripts directly. A separate FastAPI/Uvicorn process is **not required** for this deployment.

---

## 9. End-to-end online setup test

Before disconnecting internet, test at least:

1. English → Hindi
2. Marathi → Hindi
3. Hindi → Marathi
4. one TTS output
5. one video with subtitles/dubbed output
6. repeat the exact same request and confirm a cache hit

---

## 10. Prove offline operation

After all tests pass:

1. remove/unset `HF_TOKEN`
2. log out from Hugging Face if interactive login was used
3. disconnect internet/Wi-Fi
4. restart Streamlit
5. repeat a **new uncached** translation

```bash
streamlit run app/web_ui.py
```

The translation must still complete using `local_model_vault` only.

---

## 11. Normal daily startup

```bash
cd /path/to/baif-translation-engine
source .venv/bin/activate
streamlit run app/web_ui.py
```

---

## Troubleshooting

### `python3.12: command not found`

Verify Python 3.12 installation:

```bash
brew list python@3.12
brew --prefix python@3.12
```

### `ffmpeg: command not found`

```bash
brew install ffmpeg
```

Then open a new Terminal.

### Model download says unauthorized

The model repository is private. Confirm:

- the Hugging Face account/token has read access
- `BAIF_MODEL_REPO_ID` is correct
- the token was not expired/revoked

### Model vault already exists

Do not overwrite a working production vault casually. If intentionally rebuilding:

```bash
python download_production_models.py --force
```

### Streamlit port is already in use

```bash
streamlit run app/web_ui.py --server.port 8502
```

---

## 12. Third-party model licensing

Before production handover, review `MODEL_LICENSE_NOTES.md`. Individual TTS voices can have terms that differ from the repository-level license.
