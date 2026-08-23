#!/usr/bin/env python3
"""Download and verify ai4bharat/indictrans2-indic-indic-1B."""
import argparse, os, shutil, sys
from pathlib import Path
MODEL_ID='ai4bharat/indictrans2-indic-indic-1B'
BASE_DIR=Path(__file__).resolve().parent
TARGET_DIR=BASE_DIR/'local_model_vault'/'indictrans2'/'indic-indic-1B'
REQUIRED=['config.json','generation_config.json','configuration_indictrans.py','modeling_indictrans.py','tokenizer_config.json','special_tokens_map.json','model.SRC','model.TGT','dict.SRC.json','dict.TGT.json','pytorch_model.bin']
def log(s): print(f'[Indic-Indic] {s}',flush=True)
def deps():
    try: import huggingface_hub
    except ImportError: print("ERROR: install with: pip install 'huggingface_hub==0.21.4'");sys.exit(1)
    if int(huggingface_hub.__version__.split('.')[0])>=1: print(f"ERROR: huggingface_hub {huggingface_hub.__version__} is incompatible with your Transformers 4.x stack. Run: pip install 'huggingface_hub==0.21.4'");sys.exit(1)
    log(f'huggingface_hub: {huggingface_hub.__version__}')
def auth():
    from huggingface_hub import HfApi
    try:
        x=HfApi().whoami(); log('Hugging Face login OK: '+str(x.get('name') or x.get('fullname') or 'authenticated user'))
    except Exception as e:
        print('ERROR: Hugging Face authentication failed.');print(e);print('\nRun: huggingface-cli login');print(f'Then accept access conditions for {MODEL_ID}');sys.exit(1)
def clean():
    if TARGET_DIR.exists(): shutil.rmtree(TARGET_DIR); log(f'Removed {TARGET_DIR}')
    hf_home=Path(os.environ.get('HF_HOME',Path.home()/'.cache'/'huggingface'))
    cache=hf_home/'hub'/'models--ai4bharat--indictrans2-indic-indic-1B'
    if cache.exists(): shutil.rmtree(cache); log(f'Removed HF cache {cache}')
def download():
    from huggingface_hub import snapshot_download
    TARGET_DIR.parent.mkdir(parents=True,exist_ok=True)
    log(f'Model: {MODEL_ID}');log(f'Target: {TARGET_DIR}');log('Starting/resuming download...')
    try:
        snapshot_download(repo_id=MODEL_ID,local_dir=str(TARGET_DIR),local_dir_use_symlinks=False,resume_download=True)
    except TypeError:
        snapshot_download(repo_id=MODEL_ID,local_dir=str(TARGET_DIR),local_dir_use_symlinks=False)
    except Exception as e:
        print('\nDOWNLOAD FAILED');print(type(e).__name__+':',e);print('\nRun this same script again to resume.');sys.exit(2)
def verify():
    if not TARGET_DIR.exists(): print('ERROR: target directory missing');sys.exit(3)
    missing=[]
    for f in REQUIRED:
        p=TARGET_DIR/f
        if not p.is_file(): missing.append(f)
        else: log(f'OK {f:30s} {p.stat().st_size/(1024**2):,.1f} MB')
    incomplete=list(TARGET_DIR.rglob('*.incomplete'))
    if incomplete: print('\nERROR: incomplete files remain:');[print(' ',x) for x in incomplete];sys.exit(4)
    if missing: print('\nERROR: missing files:');[print(' ',x) for x in missing];sys.exit(5)
    size=(TARGET_DIR/'pytorch_model.bin').stat().st_size
    if size<4*1024**3: print(f'ERROR: pytorch_model.bin only {size/(1024**3):.2f} GB');sys.exit(6)
    total=sum(p.stat().st_size for p in TARGET_DIR.rglob('*') if p.is_file())
    print('\n============================================================');print('DOWNLOAD + VERIFICATION SUCCESSFUL');print('============================================================');print(f'Model directory:\n  {TARGET_DIR}');print(f'Total size: {total/(1024**3):.2f} GB');print('\nReady for local Transformers/PyTorch loading.');print('NOTE: this is NOT a CTranslate2 model.');print('============================================================')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--clean',action='store_true',help='Delete ONLY this Indic-Indic model cache and target before downloading.');a=ap.parse_args();print('\n=== BAIF IndicTrans2 Indic -> Indic downloader ===\n');deps();
    if a.clean: clean()
    auth();download();verify()
if __name__=='__main__': main()
