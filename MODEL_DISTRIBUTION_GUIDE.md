# BAIF Model Distribution Guide

This guide is for the developer/maintainer preparing the production model package **before** BAIF machine setup.

The final BAIF machine has internet only during first-time setup and cannot receive files manually. To guarantee the BAIF machine uses the exact same models as the verified development build, publish the already-working **minimal production model vault** to a **private Hugging Face model repository** once.

## What is published

Only these stable model components are published:

```text
whisper/
indictrans2/
  en-indic/
  indic-en/
tts/
  english/
  hindi/
  marathi/
DEPLOYMENT_MANIFEST.json
```

The following are intentionally excluded:

- experimental `indic-indic-dist-320M`
- incomplete/broken `en-indic-1b-ct2`
- `.venv`
- `storage_vault`
- Hugging Face cache folders
- application output/cache files

## One-time developer steps

Run from the **known-good development repo**.

### 1. Activate the working environment

macOS:

```bash
cd ~/Desktop/baif-translation-engine
source .venv/bin/activate
```

### 2. Install the publishing utility

```bash
python -m pip install -r requirements-setup.txt
```

### 3. Build a clean model package

```bash
python prepare_deployment_vault.py --force
```

Expected folder:

```text
deployment_model_vault/
```

The script validates the required models and creates `DEPLOYMENT_MANIFEST.json`.

### 4. Create/use a private Hugging Face account or organization repo

Authenticate using either:

```bash
hf auth login
```

or a temporary environment variable:

```bash
export HF_TOKEN="hf_..."
```

Use a token with permission to create/write the private model repository.

### 5. Publish the exact vault

Choose a private repo ID, for example:

```text
your-org/baif-production-model-vault
```

Then run:

```bash
python publish_production_model_vault.py \
  --repo-id your-org/baif-production-model-vault
```

The script creates the repository as **private** if needed and uploads the exact contents of `deployment_model_vault`.

### 6. Record the model repo ID securely

The BAIF setup instructions need only the repo ID, for example:

```text
your-org/baif-production-model-vault
```

Do **not** commit a Hugging Face token to Git.

## Token use on BAIF machines

A read token is needed only during the one-time download from the private repository.

After `download_production_models.py` succeeds and model verification passes:

- remove/unset the token
- optionally run `hf auth logout` if interactive login was used
- disconnect internet
- normal BAIF translation continues from `local_model_vault`

## Why this approach is used

The stable BAIF pipeline uses pre-converted CTranslate2 IndicTrans2 folders. Re-downloading upstream Transformer checkpoints and converting them again on every BAIF machine could create model/version differences. Publishing the already-verified vault guarantees deployment uses the same artifacts that passed your development tests.
