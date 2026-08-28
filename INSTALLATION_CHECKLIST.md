# BAIF Final Installation / Handover Checklist

Use this checklist on every BAIF machine before internet access is removed.

## Repository

- [ ] Correct repository cloned
- [ ] Correct deployment branch checked out
- [ ] `requirements.txt` present
- [ ] `requirements-setup.txt` present
- [ ] `download_production_models.py` present
- [ ] `verify_installation.py` present

## System software

- [ ] Python 3.12 x64/arm64 as appropriate
- [ ] Git available
- [ ] FFmpeg available
- [ ] FFprobe available
- [ ] Windows only: Microsoft Visual C++ 2015–2022 Redistributable x64 installed

## Python environment

- [ ] `.venv` created locally on the BAIF machine
- [ ] `requirements.txt` installed successfully
- [ ] `requirements-setup.txt` installed successfully
- [ ] `python -m pip check` reports no broken requirements

## Model vault

- [ ] private production model repo ID confirmed
- [ ] one-time read token used only during download
- [ ] `local_model_vault/whisper` present
- [ ] `local_model_vault/indictrans2/en-indic` present
- [ ] `local_model_vault/indictrans2/indic-en` present
- [ ] `local_model_vault/tts/english` present
- [ ] `local_model_vault/tts/hindi` present
- [ ] `local_model_vault/tts/marathi` present
- [ ] no experimental Indic-Indic 320M model required

## Verification

- [ ] `verify_installation.py` passes
- [ ] `verify_all_models.py` passes
- [ ] English → Hindi works
- [ ] English → Marathi works
- [ ] Hindi → English works
- [ ] Marathi → English works
- [ ] Hindi → Marathi works
- [ ] Marathi → Hindi works
- [ ] English TTS works
- [ ] Hindi TTS works
- [ ] Marathi TTS works
- [ ] audio input works
- [ ] video input works
- [ ] subtitles/output generation works
- [ ] same exact request returns a cache hit

## Security/offline handover

- [ ] `HF_TOKEN` removed/unset
- [ ] Hugging Face login removed if used
- [ ] no token committed to Git/configuration files
- [ ] internet disconnected
- [ ] Streamlit restarts successfully offline
- [ ] a **new uncached** translation works with internet disconnected

## Final startup command

macOS:

```bash
source .venv/bin/activate
streamlit run app/web_ui.py
```

Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py
```
