# Third-Party Model License Notes

This file is a deployment due-diligence note, not legal advice.

The BAIF runtime uses third-party model artifacts. Repository-level licenses and individual voice/dataset terms can differ, so BAIF should retain the upstream attribution/license information and confirm the intended deployment use is compatible.

## IndicTrans2

- Source: AI4Bharat IndicTrans2
- The distilled IndicTrans2 model cards used for tokenizer/provenance identify an MIT license.
- Production runtime uses the already-converted CTranslate2 artifacts from the verified development vault.

## Faster-Whisper

- Runtime library: faster-whisper (MIT)
- Whisper model is the CTranslate2-format Whisper model already present in the verified development vault.

## Piper / Sherpa-ONNX TTS voices

The `rhasspy/piper-voices` repository is marked MIT, but each voice can reference its own source dataset/license.

### Hindi — `hi_IN-pratham-medium`

The upstream model card states the source dataset/license as **CC BY-NC-SA 4.0**. This is non-commercial and share-alike. BAIF should confirm its intended use fits these terms before production deployment.

Source model card:
https://huggingface.co/rhasspy/piper-voices/tree/main/hi/hi_IN/pratham/medium

### Marathi — `mr_IN-google-medium`

The upstream model card states the source dataset/license as **CC BY-SA 4.0**.

Source history/model information:
https://huggingface.co/rhasspy/piper-voices/tree/main/mr/mr_IN/google/medium

### English — `en_US-lessac-medium`

The upstream voice model card references the Lessac Blizzard 2013 dataset and its specific dataset license. BAIF should retain/review those terms.

Source model card:
https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium

## Recommendation

Before final BAIF handover:

1. retain upstream model cards/license notices with the deployment documentation;
2. confirm BAIF's use of the Hindi voice is non-commercial if retaining `hi_IN-pratham-medium`;
3. if BAIF requires an OSI-style unrestricted production voice license, replace the affected voice model and re-run TTS verification before deployment.
