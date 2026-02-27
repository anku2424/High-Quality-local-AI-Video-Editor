import os
import shutil
import subprocess
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="Local OpenAI Key Demo")
logger = logging.getLogger("uvicorn.error")

# For local development only; replace in production.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "local-dev-secret-change-me"),
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
UPLOAD_DIR = Path("uploads")
VIDEO_DIR = UPLOAD_DIR / "videos"
AUDIO_DIR = UPLOAD_DIR / "audio"
CHUNK_DIR = UPLOAD_DIR / "chunks"
TRANSCRIPT_DIR = UPLOAD_DIR / "transcripts"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRANSCRIBE_FILE_BYTES = 25 * 1024 * 1024
# 16kHz mono PCM WAV is ~32KB/s. Keep chunk duration under the API limit with margin.
CHUNK_SECONDS = 600
TRANSCRIPTION_MODEL = "whisper-1"

transcription_jobs: dict[str, dict[str, Any]] = {}
jobs_lock = Lock()


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"


def extract_audio_from_video(video_path: Path, audio_path: Path) -> None:
    # Extract mono 16k WAV for downstream processing.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def split_audio_for_transcription(audio_path: Path, job_id: str) -> list[Path]:
    if audio_path.stat().st_size <= MAX_TRANSCRIBE_FILE_BYTES:
        return [audio_path]

    job_chunk_dir = CHUNK_DIR / job_id
    job_chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_pattern = job_chunk_dir / "chunk_%03d.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-f",
            "segment",
            "-segment_time",
            str(CHUNK_SECONDS),
            "-reset_timestamps",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(chunk_pattern),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    chunks = sorted(job_chunk_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("Audio chunking failed.")
    return chunks


def shift_timestamps(items: list[dict[str, Any]], offset_seconds: float) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        for key in ("start", "end"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                row[key] = value + offset_seconds
        shifted.append(row)
    return shifted


def transcribe_audio_file(
    audio_path: Path,
    api_key: str,
    job_id: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    client = OpenAI(api_key=api_key)
    transcription_parts: list[str] = []
    all_segments: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []
    chunks = split_audio_for_transcription(audio_path, job_id)

    for index, chunk_path in enumerate(chunks):
        with chunk_path.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        payload = result.model_dump() if hasattr(result, "model_dump") else {}
        text = str(payload.get("text", "")).strip()
        if text:
            transcription_parts.append(text)
        offset_seconds = float(index * CHUNK_SECONDS) if len(chunks) > 1 else 0.0
        segments = payload.get("segments") or []
        words = payload.get("words") or []
        if isinstance(segments, list):
            all_segments.extend(shift_timestamps(segments, offset_seconds))
        if isinstance(words, list):
            all_words.extend(shift_timestamps(words, offset_seconds))

    return "\n".join(transcription_parts), all_segments, all_words


def set_transcription_job(
    job_id: str,
    status: str,
    text: str = "",
    error: str = "",
    transcript_json_path: str = "",
    transcript_text_path: str = "",
) -> None:
    with jobs_lock:
        transcription_jobs[job_id] = {
            "status": status,
            "text": text,
            "error": error,
            "transcript_json_path": transcript_json_path,
            "transcript_text_path": transcript_text_path,
        }


def get_transcription_job(job_id: Optional[str]) -> dict[str, Any]:
    if not job_id:
        return {
            "status": "idle",
            "text": "",
            "error": "",
            "transcript_json_path": "",
            "transcript_text_path": "",
        }
    with jobs_lock:
        return transcription_jobs.get(
            job_id,
            {
                "status": "idle",
                "text": "",
                "error": "",
                "transcript_json_path": "",
                "transcript_text_path": "",
            },
        )


def run_transcription_job(job_id: str, audio_path: str, api_key: str) -> None:
    set_transcription_job(job_id, "processing")
    try:
        text, segments, words = transcribe_audio_file(Path(audio_path), api_key, job_id)
        transcript_text_path = TRANSCRIPT_DIR / f"{job_id}.txt"
        transcript_json_path = TRANSCRIPT_DIR / f"{job_id}.json"

        transcript_text_path.write_text(text, encoding="utf-8")
        transcript_json_path.write_text(
            json.dumps(
                {
                    "text": text,
                    "segments": segments,
                    "words": words,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        set_transcription_job(
            job_id,
            "completed",
            text=text,
            transcript_json_path=str(transcript_json_path),
            transcript_text_path=str(transcript_text_path),
        )
    except Exception as exc:
        logger.exception("Transcription job failed for job_id=%s audio_path=%s", job_id, audio_path)
        set_transcription_job(
            job_id,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
        )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    api_key = request.session.get("openai_api_key")
    if api_key:
        return RedirectResponse(url="/studio", status_code=303)

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/studio", response_class=HTMLResponse)
async def studio(request: Request):
    api_key = request.session.get("openai_api_key")
    if not api_key:
        return RedirectResponse(url="/", status_code=303)

    video_uploaded = request.session.pop("video_uploaded", False)
    job_id = request.session.get("transcription_job_id")
    transcription_job = get_transcription_job(job_id)
    return templates.TemplateResponse(
        "studio.html",
        {
            "request": request,
            "masked_key": mask_api_key(api_key),
            "video_uploaded": video_uploaded,
            "transcription_status": transcription_job["status"],
            "transcription_text": transcription_job["text"],
            "transcription_error": transcription_job["error"],
        },
    )


@app.post("/save-key")
async def save_key(request: Request, api_key: str = Form(...)):
    # Remove accidental spaces/newlines from pasted keys.
    normalized_key = "".join(api_key.split())
    request.session["openai_api_key"] = normalized_key
    return RedirectResponse(url="/studio", status_code=303)


@app.post("/upload-video")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    api_key = request.session.get("openai_api_key")
    if not api_key:
        return RedirectResponse(url="/", status_code=303)

    suffix = Path(video.filename or "").suffix
    file_id = uuid4().hex
    target_file = VIDEO_DIR / f"{file_id}{suffix}"
    target_audio = AUDIO_DIR / f"{file_id}.wav"

    with target_file.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    await video.close()
    extract_audio_from_video(target_file, target_audio)
    job_id = uuid4().hex
    set_transcription_job(job_id, "processing")
    background_tasks.add_task(run_transcription_job, job_id, str(target_audio), api_key)

    request.session["uploaded_video_path"] = str(target_file)
    request.session["uploaded_audio_path"] = str(target_audio)
    request.session["transcription_job_id"] = job_id
    request.session["transcription_text_path"] = str(TRANSCRIPT_DIR / f"{job_id}.txt")
    request.session["transcription_json_path"] = str(TRANSCRIPT_DIR / f"{job_id}.json")
    request.session["video_uploaded"] = True
    return RedirectResponse(url="/studio", status_code=303)


@app.get("/transcription-status")
async def transcription_status(request: Request):
    job_id = request.session.get("transcription_job_id")
    job = get_transcription_job(job_id)
    return JSONResponse(job)


@app.post("/clear-key")
async def clear_key(request: Request):
    job_id = request.session.get("transcription_job_id")
    if job_id:
        with jobs_lock:
            transcription_jobs.pop(job_id, None)
    request.session.pop("openai_api_key", None)
    request.session.pop("uploaded_video_path", None)
    request.session.pop("uploaded_audio_path", None)
    request.session.pop("transcription_job_id", None)
    request.session.pop("transcription_text_path", None)
    request.session.pop("transcription_json_path", None)
    request.session.pop("video_uploaded", None)
    return RedirectResponse(url="/", status_code=303)
