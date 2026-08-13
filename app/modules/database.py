import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "storage_vault" / "translation_cache.db"

def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS translation_jobs (
        job_id TEXT PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_hash TEXT UNIQUE NOT NULL,
        storage_path TEXT NOT NULL,
        target_language TEXT NOT NULL,
        status TEXT DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcript_segments (
        segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        start_time REAL NOT NULL,
        end_time REAL NOT NULL,
        source_text TEXT NOT NULL,
        translated_text TEXT NOT NULL,
        FOREIGN KEY(job_id) REFERENCES translation_jobs(job_id) ON DELETE CASCADE
    );
    """)
    
    conn.commit()
    conn.close()

def compute_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_cached_job(file_hash: str, target_language: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM translation_jobs WHERE file_hash = ? AND target_language = ? AND status = 'COMPLETED'",
        (file_hash, target_language)
    )
    job = cursor.fetchone()
    if not job:
        conn.close()
        return None
        
    job_dict = dict(job)
    cursor.execute(
        "SELECT start_time, end_time, source_text, translated_text FROM transcript_segments WHERE job_id = ?",
        (job_dict["job_id"],)
    )
    job_dict["segments"] = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return job_dict

def save_job(job_id: str, file_name: str, file_hash: str, storage_path: str, target_language: str, segments: List[Dict[str, Any]]) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO translation_jobs (job_id, file_name, file_hash, storage_path, target_language, status)
        VALUES (?, ?, ?, ?, ?, 'COMPLETED')
    """, (job_id, file_name, file_hash, storage_path, target_language))
    
    for seg in segments:
        cursor.execute("""
            INSERT INTO transcript_segments (job_id, start_time, end_time, source_text, translated_text)
            VALUES (?, ?, ?, ?, ?)
        """, (job_id, seg["start"], seg["end"], seg["text"], seg["translated_text"]))
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("✅ Database schema initialized at storage_vault/translation_cache.db")