import sys
from pathlib import Path

# Force project root into Python search path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import os
import uuid
import shutil
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

# Dual import fallback to guarantee module resolution
try:
    from app.modules.database import init_db, compute_file_hash, get_cached_job, save_job
    from app.modules.ingestion import extract_and_normalize_audio
    from app.modules.inference import TranslationPipeline
except ModuleNotFoundError:
    from modules.database import init_db, compute_file_hash, get_cached_job, save_job
    from modules.ingestion import extract_and_normalize_audio
    from modules.inference import TranslationPipeline

app = FastAPI(
    title="BAIF Offline Translation API",
    description="Air-gapped backend API for multilingual translation & TTS synthesis",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_DIR = BASE_DIR / "storage_vault" / "inputs"
OUTPUTS_DIR = BASE_DIR / "storage_vault" / "outputs"

pipeline: TranslationPipeline = None
job_status_tracker: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
def startup_event():
    global pipeline
    init_db()
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("🚀 Initializing Translation Pipeline in background...", flush=True)
    pipeline = TranslationPipeline()


def run_translation_job(job_id: str, raw_file_path: str, file_hash: str, file_name: str, target_lang: str):
    """Background worker executing pipeline stages with real-time logging."""
    try:
        print(f"\n==========================================", flush=True)
        print(f"🎬 STARTING JOB: {job_id} ({file_name})", flush=True)
        print(f"==========================================", flush=True)
        
        job_status_tracker[job_id] = {"status": "PROCESSING", "progress": "Stage 1/3: Extracting 16kHz audio..."}
        print("🎧 [Stage 1/3] Extracting audio via FFmpeg...", flush=True)
        wav_path = extract_and_normalize_audio(raw_file_path, output_wav_filename=f"{job_id}.wav")
        print(f"✅ Audio extracted to: {wav_path}", flush=True)
        
        job_status_tracker[job_id] = {"status": "PROCESSING", "progress": "Stage 2/3: Transcribing & Translating..."}
        print("🌐 [Stage 2/3] Running Whisper ASR & IndicTrans2 NMT...", flush=True)
        result = pipeline.process_pipeline(wav_path, target_lang=target_lang, job_id=job_id)
        
        if result.get("status") == "FAILED":
            job_status_tracker[job_id] = {"status": "FAILED", "reason": result.get("reason", "Unknown error")}
            print(f"❌ JOB FAILED: {result.get('reason')}", flush=True)
            return

        job_status_tracker[job_id] = {"status": "PROCESSING", "progress": "Stage 3/3: Saving to Cache Database..."}
        print("💾 [Stage 3/3] Saving results to SQLite cache...", flush=True)
        save_job(
            job_id=job_id,
            file_name=file_name,
            file_hash=file_hash,
            storage_path=raw_file_path,
            target_language=target_lang,
            segments=result["segments"]
        )
        
        job_status_tracker[job_id] = {
            "status": "COMPLETED",
            "result": result
        }
        print(f"🎉 JOB {job_id} COMPLETED SUCCESSFULLY!\n", flush=True)
    except Exception as e:
        print(f"💥 PIPELINE ERROR on job {job_id}: {str(e)}", flush=True)
        job_status_tracker[job_id] = {"status": "FAILED", "reason": str(e)}


@app.get("/")
def read_root():
    return {"status": "online", "engine": "BAIF Offline Engine"}


@app.post("/process")
async def process_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form("hi")
):
    if target_language not in ["hi", "mr"]:
        raise HTTPException(status_code=400, detail="Target language must be 'hi' or 'mr'.")

    print(f"\n📥 Receiving incoming file stream: {file.filename}...", flush=True)
    temp_file_path = INPUTS_DIR / f"upload_{file.filename}"
    
    # Stream in 1MB chunks to prevent memory locking
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"⚡ File saved to disk. Computing SHA-256 fingerprint...", flush=True)
    file_hash = compute_file_hash(str(temp_file_path))

    cached = get_cached_job(file_hash, target_language)
    if cached:
        print(f"🎯 SHA-256 Cache Hit! Returning instant results for {file.filename}", flush=True)
        return {
            "job_id": cached["job_id"],
            "cached": True,
            "status": "COMPLETED",
            "message": "Instant sub-second cache hit!",
            "segments": cached["segments"]
        }

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_status_tracker[job_id] = {"status": "QUEUED", "progress": "Job queued for background processing"}

    print(f"🚀 Dispatching background job {job_id}...", flush=True)
    background_tasks.add_task(
        run_translation_job,
        job_id=job_id,
        raw_file_path=str(temp_file_path),
        file_hash=file_hash,
        file_name=file.filename,
        target_lang=target_language
    )

    return {
        "job_id": job_id,
        "cached": False,
        "status": "QUEUED",
        "message": "Processing started in background."
    }


@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_status_tracker:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return job_status_tracker[job_id]


@app.get("/download/{job_id}/{file_type}")
def download_output(job_id: str, file_type: str):
    if file_type == "srt":
        target_path = list(OUTPUTS_DIR.glob(f"{job_id}_*.srt"))
    elif file_type == "audio":
        target_path = list(OUTPUTS_DIR.glob(f"{job_id}_*_dubbed.wav"))
    else:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    if not target_path or not target_path[0].exists():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(path=target_path[0], filename=target_path[0].name)