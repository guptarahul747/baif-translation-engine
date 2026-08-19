import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "storage_vault" / "translation_cache.db"


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize both:
      1) the original FastAPI cache tables (kept for compatibility), and
      2) the Streamlit v2 cache used by the current all-6-language UI.

    The new UI cache deliberately uses a separate table so an existing
    translation_cache.db can be upgraded without destructive migrations.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Existing backend schema - preserved so app/main.py is not broken.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS translation_jobs (
            job_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            storage_path TEXT NOT NULL,
            target_language TEXT NOT NULL,
            status TEXT DEFAULT 'PENDING',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_segments (
            segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            source_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES translation_jobs(job_id) ON DELETE CASCADE
        );
        """
    )

    # Streamlit cache. This cache is safe for all 6 directions and UI options.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ui_translation_cache (
            cache_key TEXT PRIMARY KEY,
            input_hash TEXT NOT NULL,
            input_type TEXT NOT NULL,
            file_name TEXT,
            source_language TEXT NOT NULL,
            target_language TEXT NOT NULL,
            generate_tts INTEGER NOT NULL,
            generate_summary INTEGER NOT NULL,
            burn_subtitles INTEGER NOT NULL,
            pipeline_version TEXT NOT NULL,
            transcript_json_path TEXT,
            translated_json_path TEXT NOT NULL,
            audio_path TEXT,
            srt_path TEXT,
            summary_path TEXT,
            video_path TEXT,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ui_translation_cache_lookup
        ON ui_translation_cache (
            input_hash,
            source_language,
            target_language,
            pipeline_version
        );
        """
    )

    conn.commit()
    conn.close()


def compute_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(1024 * 1024), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_text_hash(text: str) -> str:
    normalized = text.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def build_ui_cache_key(
    *,
    input_hash: str,
    input_type: str,
    source_language: str,
    target_language: str,
    generate_tts: bool,
    generate_summary: bool,
    burn_subtitles: bool,
    pipeline_version: str,
) -> str:
    """Create a deterministic key for one exact UI processing configuration."""
    payload = {
        "input_hash": input_hash,
        "input_type": input_type,
        "source_language": source_language,
        "target_language": target_language,
        "generate_tts": bool(generate_tts),
        "generate_summary": bool(generate_summary),
        "burn_subtitles": bool(burn_subtitles),
        "pipeline_version": pipeline_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_ui_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM ui_translation_cache WHERE cache_key = ?",
        (cache_key,),
    )
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return None

    cursor.execute(
        "UPDATE ui_translation_cache SET last_accessed = CURRENT_TIMESTAMP WHERE cache_key = ?",
        (cache_key,),
    )
    conn.commit()
    result = dict(row)
    conn.close()

    if result.get("metadata_json"):
        try:
            result["metadata"] = json.loads(result["metadata_json"])
        except json.JSONDecodeError:
            result["metadata"] = {}
    else:
        result["metadata"] = {}

    return result


def save_ui_cached_result(
    *,
    cache_key: str,
    input_hash: str,
    input_type: str,
    file_name: Optional[str],
    source_language: str,
    target_language: str,
    generate_tts: bool,
    generate_summary: bool,
    burn_subtitles: bool,
    pipeline_version: str,
    translated_json_path: str,
    transcript_json_path: Optional[str] = None,
    audio_path: Optional[str] = None,
    srt_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    video_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO ui_translation_cache (
            cache_key,
            input_hash,
            input_type,
            file_name,
            source_language,
            target_language,
            generate_tts,
            generate_summary,
            burn_subtitles,
            pipeline_version,
            transcript_json_path,
            translated_json_path,
            audio_path,
            srt_path,
            summary_path,
            video_path,
            metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            file_name = excluded.file_name,
            transcript_json_path = excluded.transcript_json_path,
            translated_json_path = excluded.translated_json_path,
            audio_path = excluded.audio_path,
            srt_path = excluded.srt_path,
            summary_path = excluded.summary_path,
            video_path = excluded.video_path,
            metadata_json = excluded.metadata_json,
            last_accessed = CURRENT_TIMESTAMP
        """,
        (
            cache_key,
            input_hash,
            input_type,
            file_name,
            source_language,
            target_language,
            int(generate_tts),
            int(generate_summary),
            int(burn_subtitles),
            pipeline_version,
            transcript_json_path,
            translated_json_path,
            audio_path,
            srt_path,
            summary_path,
            video_path,
            json.dumps(metadata or {}, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def delete_ui_cached_result(cache_key: str) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ui_translation_cache WHERE cache_key = ?", (cache_key,))
    conn.commit()
    conn.close()


def clear_ui_cache() -> int:
    """Delete only UI cache database rows. Cached files are managed by the UI."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS count FROM ui_translation_cache")
    row = cursor.fetchone()
    count = int(row["count"]) if row else 0
    cursor.execute("DELETE FROM ui_translation_cache")
    conn.commit()
    conn.close()
    return count


# -----------------------------------------------------------------------------
# Original FastAPI cache functions - kept unchanged for compatibility.
# -----------------------------------------------------------------------------

def get_cached_job(file_hash: str, target_language: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM translation_jobs WHERE file_hash = ? AND target_language = ? AND status = 'COMPLETED'",
        (file_hash, target_language),
    )
    job = cursor.fetchone()
    if not job:
        conn.close()
        return None

    job_dict = dict(job)
    cursor.execute(
        "SELECT start_time, end_time, source_text, translated_text FROM transcript_segments WHERE job_id = ?",
        (job_dict["job_id"],),
    )
    job_dict["segments"] = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return job_dict


def save_job(
    job_id: str,
    file_name: str,
    file_hash: str,
    storage_path: str,
    target_language: str,
    segments: List[Dict[str, Any]],
) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO translation_jobs (
            job_id, file_name, file_hash, storage_path, target_language, status
        ) VALUES (?, ?, ?, ?, ?, 'COMPLETED')
        """,
        (job_id, file_name, file_hash, storage_path, target_language),
    )

    for seg in segments:
        cursor.execute(
            """
            INSERT INTO transcript_segments (
                job_id, start_time, end_time, source_text, translated_text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                job_id,
                seg["start"],
                seg["end"],
                seg["text"],
                seg.get("translated_text")
                or next(
                    (
                        value
                        for key, value in seg.items()
                        if key.startswith("translation_") and isinstance(value, str)
                    ),
                    "",
                ),
            ),
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"✅ Database initialized at {DB_PATH}")
