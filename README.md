# BAIF Anuwad Studio

BAIF Anuwad Studio is an on-premises, fully offline application for translating English, Hindi, and Marathi text, audio, and video. It is designed for BAIF infrastructure and has no dependency on HSBC systems or services.

## Capabilities

- Text input and MP3, WAV, AAC, M4A, FLAC, WMA, OGG audio
- MP4, MOV, AVI, WMV, MKV, FLV, WebM video
- Speech recognition for Marathi, Hindi, and English
- Translation between all six language pairs
- Translated text, continuous translated WAV audio, SRT subtitles, and optional dubbed video
- Local result cache and a clear, end-user progress indicator
- Fully offline runtime after the local model vault is installed

## Offline architecture

```text
Media/Text → Faster-Whisper ASR → IndicTrans2 CTranslate2 → Sherpa-ONNX TTS → SRT / FFmpeg video
```

All models are read from `local_model_vault/`. Normal runtime does not download models, use a token, or make web requests.

## Installation guides

- Final BAIF Windows laptop: [README_WINDOWS.md](README_WINDOWS.md)
- macOS development/backup machine: [README_MAC.md](README_MAC.md)
- Windows handover text file: [BAIF_WINDOWS_INSTALLATION_STEPS.txt](BAIF_WINDOWS_INSTALLATION_STEPS.txt)
- macOS handover text file: [BAIF_MAC_INSTALLATION_STEPS.txt](BAIF_MAC_INSTALLATION_STEPS.txt)
- Handover checklist: [INSTALLATION_CHECKLIST.md](INSTALLATION_CHECKLIST.md)
- Production model-vault packaging: [MODEL_DISTRIBUTION_GUIDE.md](MODEL_DISTRIBUTION_GUIDE.md)

The Windows guide includes both supported methods:

1. installation from a prepared USB drive with no internet on the final laptop;
2. installation with one-time internet access, followed by an offline acceptance test.

## Normal launch

Windows PowerShell:

```powershell
Set-Location C:\BAIF\baif-translation-engine
.\.venv\Scripts\python.exe -m streamlit run app\web_ui.py
```

macOS:

```bash
cd ~/BAIF/baif-translation-engine
.venv/bin/python -m streamlit run app/web_ui.py
```

Open `http://localhost:8501`.

## Dependencies

- `requirements.txt`: required at runtime on both Windows and macOS.
- `requirements-setup.txt`: one-time internet/model-download utility only; not needed after `local_model_vault` has been installed.

## Verification before handover

Run these from the project folder after installing the runtime dependencies and model vault:

```text
verify_installation.py
verify_all_models.py
```

Then disconnect internet and process a new uncached media file. Do not hand over the machine until that offline test passes.
