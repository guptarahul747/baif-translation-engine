# BAIF Offline Translation Engine — Windows Fresh Installation

This is the recommended guide for the **final BAIF Windows machine**.

It installs the stable last-known-good pipeline and downloads the exact verified model vault while internet is temporarily available. After verification, the application runs offline.

## Final runtime capabilities

- Fully offline after first-time setup
- English, Hindi and Marathi
- All 6 translation directions
- Hindi ↔ Marathi through the proven English pivot
- Offline Whisper ASR
- Offline IndicTrans2 CTranslate2 translation
- Offline English/Hindi/Marathi TTS
- Text/audio/video UI
- subtitles, dubbed output and optional summary
- persistent completed-result cache

---

## 0. Values needed before setup

Have these ready:

```text
REPO_URL=<your Git repository URL>
BRANCH=<your deployment branch>
MODEL_REPO_ID=<private Hugging Face model repo prepared with MODEL_DISTRIBUTION_GUIDE.md>
```

The Windows PC needs internet only for initial software/package/model installation.

---

## 1. Install system prerequisites

Install BAIF-approved x64 versions of:

1. **Git for Windows**
2. **Python 3.12 x64**
3. **FFmpeg x64**
4. **Microsoft Visual C++ 2015–2022 Redistributable x64**

During Python installation, enable the option to add Python to PATH if your BAIF policy allows it.

For FFmpeg, add its `bin` directory to Windows PATH, for example:

```text
C:\ffmpeg\bin
```

Open a **new PowerShell** window after installation and verify:

```powershell
git --version
py -3.12 --version
ffmpeg -version
ffprobe -version
```

Python should be `3.12.x`.

---

## 2. Clone the repository

Choose an approved local directory, for example:

```powershell
New-Item -ItemType Directory -Force C:\BAIF | Out-Null
Set-Location C:\BAIF

git clone <REPO_URL>
Set-Location .\baif-translation-engine
git checkout <BRANCH>
```

Verify:

```powershell
git branch --show-current
```

---

## 3. Create the virtual environment

```powershell
py -3.12 -m venv .venv
```

You do **not** have to activate the environment. Using its Python executable directly avoids PowerShell execution-policy issues.

Verify:

```powershell
.\.venv\Scripts\python.exe --version
```

---

## 4. Install Python dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-setup.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements-setup.txt` is used only for the first-time model download utility.

---

## 5. Download the exact production models

The stable production vault should already have been published to a private Hugging Face model repository using `MODEL_DISTRIBUTION_GUIDE.md`.

### Recommended — temporary token in the current PowerShell session

```powershell
$env:BAIF_MODEL_REPO_ID = "<MODEL_REPO_ID>"
$env:HF_TOKEN = "hf_your_read_token"

.\.venv\Scripts\python.exe download_production_models.py
```

After successful download:

```powershell
Remove-Item Env:HF_TOKEN
```

You may also remove the repo ID environment variable after setup:

```powershell
Remove-Item Env:BAIF_MODEL_REPO_ID
```

The script creates:

```text
C:\BAIF\baif-translation-engine\local_model_vault\
├── whisper\
├── indictrans2\
│   ├── en-indic\
│   └── indic-en\
└── tts\
    ├── english\
    ├── hindi\
    └── marathi\
```

The stable production build does **not** require the experimental Indic-Indic 320M model.

---

## 6. Run the installation pre-flight

```powershell
.\.venv\Scripts\python.exe verify_installation.py
```

Expected ending:

```text
✅ Pre-flight passed.
Next: python verify_all_models.py
```

---

## 7. Run full model verification

```powershell
.\.venv\Scripts\python.exe verify_all_models.py
```

Confirm:

```text
Whisper ASR Engine       ✅
6/6 NMT combinations     ✅
3/3 TTS voice engines    ✅
```

If any required model fails, fix it before internet access is removed.

---

## 8. Start the Streamlit application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py
```

Open:

```text
http://localhost:8501
```

The stable UI runs the pipeline scripts directly. Do **not** start FastAPI/Uvicorn separately for this deployment.

---

## 9. End-to-end setup test while internet is still available

Test at minimum:

1. English → Hindi
2. English → Marathi
3. Hindi → English
4. Marathi → English
5. Hindi → Marathi
6. Marathi → Hindi
7. one English TTS output
8. one Hindi TTS output
9. one Marathi TTS output
10. a short video with dubbed/subtitle output
11. run the exact same job twice and confirm the second run uses the cache

---

## 10. Prove the final machine is offline-capable

After all verification passes:

1. remove `HF_TOKEN`
2. remove any saved Hugging Face credentials if you used interactive login
3. disconnect network/internet
4. restart Streamlit
5. process a **new uncached** file

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py
```

The new translation must work with no network connection.

---

## 11. Normal daily Windows startup

```powershell
Set-Location C:\BAIF\baif-translation-engine
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py
```

No internet, Hugging Face token or Python-environment activation is required for normal runtime.

---

## Optional PowerShell activation

If BAIF policy permits PowerShell scripts:

```powershell
.\.venv\Scripts\Activate.ps1
```

If execution policy blocks it, do not change corporate policy just for this project; use `.\.venv\Scripts\python.exe` commands shown above.

---

## Troubleshooting

### `py -3.12` does not work

Confirm Python 3.12 x64 is installed. On some managed machines use:

```powershell
python --version
```

If `python` is Python 3.12, create the environment with:

```powershell
python -m venv .venv
```

### `ffmpeg` is not recognized

Confirm the FFmpeg `bin` folder is on PATH, close PowerShell and open a new window.

```powershell
where.exe ffmpeg
where.exe ffprobe
```

### Import/DLL error when loading native packages

Confirm the Microsoft Visual C++ 2015–2022 Redistributable **x64** is installed, then reopen PowerShell/restart Windows if required.

### Private model repo returns 401/403

Check:

- token has read permission
- token/account has access to the private model repository
- `MODEL_REPO_ID` is correct

### Re-download intentionally

Only if the local vault is known to be incomplete/corrupt:

```powershell
.\.venv\Scripts\python.exe download_production_models.py --force
```

### Port 8501 already in use

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py --server.port 8502
```

---

## 12. Third-party model licensing

Before production handover, review `MODEL_LICENSE_NOTES.md`. Individual TTS voices can have terms that differ from the repository-level license.
