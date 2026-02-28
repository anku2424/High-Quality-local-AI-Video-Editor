import os
import shutil
import subprocess
import json
import logging
import re
from pathlib import Path, PurePosixPath
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
app.mount("/uploads", StaticFiles(directory="uploads", check_dir=False), name="uploads")
UPLOAD_DIR = Path("uploads")
VIDEO_DIR = UPLOAD_DIR / "videos"
AUDIO_DIR = UPLOAD_DIR / "audio"
CHUNK_DIR = UPLOAD_DIR / "chunks"
TRANSCRIPT_DIR = UPLOAD_DIR / "transcripts"
SUBTITLE_DIR = UPLOAD_DIR / "subtitles"
RENDER_DIR = UPLOAD_DIR / "rendered"
BROLL_DIR = UPLOAD_DIR / "broll"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
RENDER_DIR.mkdir(parents=True, exist_ok=True)
BROLL_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRANSCRIBE_FILE_BYTES = 25 * 1024 * 1024
# 16kHz mono PCM WAV is ~32KB/s. Keep chunk duration under the API limit with margin.
CHUNK_SECONDS = 600
TRANSCRIPTION_MODEL = "whisper-1"
MAX_KARAOKE_WORDS = 7
MAX_KARAOKE_DURATION_SECONDS = 3.8
MAX_KARAOKE_GAP_SECONDS = 0.65
MIN_WORDS_BEFORE_PUNCT_BREAK = 3
DEFAULT_FONT_SIZE = 58
MIN_FONT_SIZE = 24
MAX_FONT_SIZE = 200
DEFAULT_BORDER_SIZE = 3
MIN_BORDER_SIZE = 0
MAX_BORDER_SIZE = 20
DEFAULT_HIGHLIGHT_COLOR = "#FFD700"
DEFAULT_NON_HIGHLIGHT_COLOR = "#F2F2F2"
DEFAULT_BORDER_COLOR = "#000000"
TRANSPARENT_ASS_COLOR = "&HFF000000"
BROLL_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
BROLL_PLACEMENT_PROMPT_ID = "pmpt_69a1c74939e48197b22c4c4135b557890eb70db7c075888b"
BROLL_PLACEMENT_PROMPT_VERSION = "3"
BROLL_PLACEMENT_SCHEMA = {
    "name": "image_placement_response",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "images": {
                "type": "array",
                "description": "List of image placements with non-overlapping time ranges.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "image_name": {
                            "type": "string",
                            "description": "The name of the image file.",
                        },
                        "placements": {
                            "type": "array",
                            "description": "List of time ranges where the image should appear.",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "start_time": {
                                        "type": "number",
                                        "description": "Start time in seconds (inclusive).",
                                    },
                                    "end_time": {
                                        "type": "number",
                                        "description": "End time in seconds (exclusive).",
                                    },
                                },
                                "required": ["start_time", "end_time"],
                            },
                        },
                    },
                    "required": ["image_name", "placements"],
                },
            }
        },
        "required": ["images"],
    },
    "strict": True,
}
COLOR_SWATCHES = [
    "#FFFFFF",
    "#F2F2F2",
    "#D9D9D9",
    "#C9C9C9",
    "#000000",
    "#FFD700",
    "#FFC107",
    "#FF9F1A",
    "#FF6B6B",
    "#FF4D88",
    "#A55EEA",
    "#7E57C2",
    "#4D96FF",
    "#00B8FF",
    "#00C2A8",
    "#57CC99",
]

transcription_jobs: dict[str, dict[str, Any]] = {}
jobs_lock = Lock()


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"


def get_default_job_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "text": "",
        "error": "",
        "transcript_json_path": "",
        "transcript_text_path": "",
        "subtitle_status": "idle",
        "subtitle_error": "",
        "karaoke_ass_path": "",
        "karaoke_video_path": "",
        "font_size": DEFAULT_FONT_SIZE,
        "border_size": DEFAULT_BORDER_SIZE,
        "highlight_color": DEFAULT_HIGHLIGHT_COLOR,
        "non_highlight_color": DEFAULT_NON_HIGHLIGHT_COLOR,
        "border_color": DEFAULT_BORDER_COLOR,
    }


def clamp_font_size(raw_size: Any) -> int:
    try:
        font_size = int(raw_size)
    except (TypeError, ValueError):
        return DEFAULT_FONT_SIZE
    return max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, font_size))


def clamp_border_size(raw_size: Any) -> int:
    try:
        border_size = int(raw_size)
    except (TypeError, ValueError):
        return DEFAULT_BORDER_SIZE
    return max(MIN_BORDER_SIZE, min(MAX_BORDER_SIZE, border_size))


def normalize_hex_color(raw_color: Any, default_color: str) -> str:
    color = str(raw_color or "").strip()
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", color) is None:
        return default_color
    if not color.startswith("#"):
        color = f"#{color}"
    return color.upper()


def hex_to_ass_bgr(hex_color: str, alpha_hex: str = "00") -> str:
    safe_hex = normalize_hex_color(hex_color, DEFAULT_NON_HIGHLIGHT_COLOR)
    rr = safe_hex[1:3]
    gg = safe_hex[3:5]
    bb = safe_hex[5:7]
    return f"&H{alpha_hex}{bb}{gg}{rr}"


def is_supported_broll_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in BROLL_IMAGE_EXTENSIONS


def sanitize_uploaded_relative_path(raw_name: str) -> Path:
    candidate = (raw_name or "").replace("\\", "/").strip()
    parts: list[str] = []
    for part in PurePosixPath(candidate).parts:
        if part in {"", ".", ".."}:
            continue
        safe_part = re.sub(r"[^A-Za-z0-9._ -]", "_", part).strip()
        if safe_part:
            parts.append(safe_part)

    if not parts:
        raise ValueError("Invalid uploaded file path.")
    return Path(*parts)


def load_word_timestamps(transcript_json_path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(transcript_json_path).read_text(encoding="utf-8"))
    words = payload.get("words") or []
    if not isinstance(words, list):
        return []
    normalized_words: list[dict[str, Any]] = []
    for item in words:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item or "word" not in item:
            continue
        normalized_words.append(
            {
                "start": item["start"],
                "end": item["end"],
                "word": item["word"],
            }
        )
    return normalized_words


def list_broll_image_names(broll_dir_path: str) -> list[str]:
    root = Path(broll_dir_path)
    if not root.exists():
        return []
    names: list[str] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in BROLL_IMAGE_EXTENSIONS:
            continue
        names.append(file_path.name)
    return sorted(dict.fromkeys(names))


def generate_broll_placement_result(
    api_key: str,
    transcript_json_path: str,
    broll_dir_path: str,
) -> dict[str, Any]:
    transcript_path = Path(transcript_json_path)
    if not transcript_path.exists():
        raise ValueError("Transcript JSON file not found.")

    broll_path = Path(broll_dir_path)
    if not broll_path.exists():
        raise ValueError("B-roll folder path not found.")

    word_timestamps = load_word_timestamps(str(transcript_path))
    broll_image_names = list_broll_image_names(str(broll_path))
    if not word_timestamps:
        raise ValueError("No word timestamps available.")
    if not broll_image_names:
        raise ValueError("No b-roll images found.")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        prompt={
            "id": BROLL_PLACEMENT_PROMPT_ID,
            # "version": BROLL_PLACEMENT_PROMPT_VERSION,
            "variables": {
                "word_timestamps": json.dumps(word_timestamps, ensure_ascii=False),
                "broll_image_names": json.dumps(broll_image_names, ensure_ascii=False),
            },
        },
        text={
            "format": {
                "type": "json_schema",
                "name": BROLL_PLACEMENT_SCHEMA["name"],
                "schema": BROLL_PLACEMENT_SCHEMA["schema"],
                "strict": BROLL_PLACEMENT_SCHEMA["strict"],
            }
        },
    )

    output_text = (response.output_text or "").strip()
    if not output_text:
        logger.error("B-roll placement response had empty output_text. response_id=%s", response.id)
        raise RuntimeError("B-roll placement response was empty.")

    try:
        placement_result = json.loads(output_text)
    except json.JSONDecodeError as exc:
        logger.exception("B-roll placement response is not valid JSON. response_id=%s", response.id)
        raise RuntimeError("B-roll placement response is not valid JSON.") from exc

    pretty = json.dumps(placement_result, indent=2, ensure_ascii=False)
    print("\nB-roll placement result:")
    print(pretty)
    logger.info("B-roll placement result:\n%s", pretty)
    return placement_result


def index_broll_images_by_name(broll_dir_path: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicate_names: set[str] = set()

    for file_path in sorted(broll_dir_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in BROLL_IMAGE_EXTENSIONS:
            continue
        if file_path.name in index:
            duplicate_names.add(file_path.name)
            continue
        index[file_path.name] = file_path

    if duplicate_names:
        logger.warning(
            "Duplicate b-roll image names detected; first file is used for: %s",
            ", ".join(sorted(duplicate_names)),
        )
    return index


def build_broll_overlay_plan(
    placement_result: dict[str, Any],
    broll_dir_path: Path,
) -> list[dict[str, Any]]:
    image_entries = placement_result.get("images")
    if not isinstance(image_entries, list):
        raise RuntimeError("B-roll placement response is missing 'images'.")

    image_index = index_broll_images_by_name(broll_dir_path)
    overlay_plan: list[dict[str, Any]] = []

    for image_entry in image_entries:
        if not isinstance(image_entry, dict):
            continue

        image_name = str(image_entry.get("image_name") or "").strip()
        if not image_name:
            continue

        image_path = image_index.get(image_name)
        if image_path is None:
            logger.warning("Skipping missing b-roll image from placement result: %s", image_name)
            continue

        raw_placements = image_entry.get("placements")
        if not isinstance(raw_placements, list):
            continue

        placements: list[tuple[float, float]] = []
        for placement in raw_placements:
            if not isinstance(placement, dict):
                continue
            start_time = max(0.0, to_float(placement.get("start_time"), -1.0))
            end_time = max(0.0, to_float(placement.get("end_time"), -1.0))
            if end_time <= start_time:
                continue
            placements.append((start_time, end_time))

        if not placements:
            continue

        placements.sort(key=lambda item: item[0])
        overlay_plan.append(
            {
                "image_name": image_name,
                "image_path": image_path,
                "placements": placements,
            }
        )

    return overlay_plan


def build_broll_placement_clips(
    overlay_plan: list[dict[str, Any]],
    video_duration: float,
) -> list[dict[str, Any]]:
    clips: list[dict[str, Any]] = []
    max_duration = max(0.0, video_duration)
    for item in overlay_plan:
        image_path = item["image_path"]
        for start_time, end_time in item["placements"]:
            start = min(max(0.0, float(start_time)), max_duration)
            end = min(max(0.0, float(end_time)), max_duration)
            if end <= start:
                continue
            clips.append(
                {
                    "image_path": image_path,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                }
            )
    clips.sort(key=lambda clip: (clip["start"], clip["end"]))
    return clips


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if streams:
            width = int(streams[0].get("width") or 0)
            height = int(streams[0].get("height") or 0)
            if width > 0 and height > 0:
                return width, height
    except Exception:
        logger.warning("Failed to probe video dimensions for %s; using fallback.", video_path)

    return 1920, 1080


def get_video_duration_seconds(video_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return max(0.0, to_float((result.stdout or "").strip(), 0.0))
    except Exception:
        logger.warning("Failed to probe video duration for %s; using fallback.", video_path)
        return 0.0


def run_ffmpeg_checked(command: list[str], error_label: str) -> None:
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        stderr_text = (result.stderr or "").strip()
        if stderr_text:
            logger.error(
                "%s failed (code=%s):\n%s",
                error_label,
                result.returncode,
                stderr_text[-4000:],
            )
        raise RuntimeError(f"{error_label} failed with exit code {result.returncode}.")


def render_broll_video(
    video_path: Path,
    broll_dir_path: Path,
    placement_result: dict[str, Any],
    output_path: Path,
) -> None:
    if not video_path.exists():
        raise ValueError("Source video for b-roll render was not found.")
    if not broll_dir_path.exists():
        raise ValueError("B-roll directory was not found.")

    overlay_plan = build_broll_overlay_plan(placement_result, broll_dir_path)
    if not overlay_plan:
        raise RuntimeError("No valid b-roll placements were available to render.")

    video_width, video_height = get_video_dimensions(video_path)
    video_duration = get_video_duration_seconds(video_path)
    clips = build_broll_placement_clips(overlay_plan, video_duration)
    if not clips:
        raise RuntimeError("No in-range b-roll clips were available to render.")

    render_id = uuid4().hex[:8]
    temp_dir = RENDER_DIR / f".broll_tmp_{render_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scale_filter = f"scale=w={video_width}:h={video_height}:force_original_aspect_ratio=increase,crop={video_width}:{video_height}"

    # Normalize all images to same resolution and format once, then reuse with fast looped inputs.
    placements_by_source: dict[Path, list[tuple[float, float]]] = {}
    for clip in clips:
        source_path = Path(str(clip["image_path"]))
        placements_by_source.setdefault(source_path, []).append(
            (float(clip["start"]), float(clip["end"]))
        )

    prepared_inputs: list[dict[str, Any]] = []
    try:
        for idx, source_path in enumerate(sorted(placements_by_source.keys()), start=1):
            prepared_path = temp_dir / f"prepared_{idx:03d}.jpg"
            prep_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(source_path),
                "-frames:v",
                "1",
                "-vf",
                scale_filter,
                str(prepared_path),
            ]
            run_ffmpeg_checked(prep_cmd, "ffmpeg b-roll image normalization")
            prepared_inputs.append(
                {
                    "prepared_path": prepared_path,
                    "placements": sorted(placements_by_source[source_path], key=lambda item: item[0]),
                }
            )

        if not prepared_inputs:
            raise RuntimeError("No b-roll images were prepared for rendering.")

        render_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
        ]
        for prepared in prepared_inputs:
            render_cmd.extend(
                [
                    "-loop",
                    "1",
                    "-i",
                    str(prepared["prepared_path"]),
                ]
            )

        filter_parts = ["[0:v]setpts=PTS-STARTPTS[v0]"]
        current_label = "v0"
        for idx, prepared in enumerate(prepared_inputs, start=1):
            next_label = f"v{idx}"
            enable_expression = "+".join(
                f"between(t,{start:.3f},{end:.3f})" for start, end in prepared["placements"]
            )
            filter_parts.append(
                f"[{current_label}][{idx}:v]overlay=x=0:y=0:enable='{enable_expression}':eof_action=pass[{next_label}]"
            )
            current_label = next_label

        render_cmd.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                f"[{current_label}]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(output_path),
            ]
        )
        run_ffmpeg_checked(render_cmd, "ffmpeg b-roll overlay render")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


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


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_ass_timestamp(seconds: float) -> str:
    safe = max(0.0, seconds)
    total_cs = int(round(safe * 100))
    hours = total_cs // 360000
    minutes = (total_cs % 360000) // 6000
    secs = (total_cs % 6000) // 100
    centiseconds = total_cs % 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", " ")
    )


def should_split_batch(
    batch: list[dict[str, Any]],
    next_word: Optional[dict[str, Any]],
) -> bool:
    if not batch:
        return False

    first_start = to_float(batch[0].get("start"))
    current_end = to_float(batch[-1].get("end"), first_start)
    duration = max(0.0, current_end - first_start)
    current_text = str(batch[-1].get("word") or "").strip()

    if len(batch) >= MAX_KARAOKE_WORDS:
        return True
    if duration >= MAX_KARAOKE_DURATION_SECONDS:
        return True
    if (
        len(batch) >= MIN_WORDS_BEFORE_PUNCT_BREAK
        and re.search(r"[.!?]$|[,;:]$", current_text) is not None
    ):
        return True
    if next_word is None:
        return True

    next_start = to_float(next_word.get("start"), current_end)
    if max(0.0, next_start - current_end) > MAX_KARAOKE_GAP_SECONDS:
        return True
    return False


def words_to_karaoke_events(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []

    for idx, word in enumerate(words):
        text = str(word.get("word") or "").strip()
        if not text:
            continue

        cleaned = dict(word)
        cleaned["word"] = text
        batch.append(cleaned)

        next_word = words[idx + 1] if idx + 1 < len(words) else None
        if not should_split_batch(batch, next_word):
            continue

        start = to_float(batch[0].get("start"))
        end = to_float(batch[-1].get("end"), start + 0.01)
        tokens: list[str] = []
        for word_idx, item in enumerate(batch):
            word_start = to_float(item.get("start"), start)
            if word_idx + 1 < len(batch):
                next_start = to_float(batch[word_idx + 1].get("start"), word_start)
            else:
                next_start = to_float(item.get("end"), word_start + 0.01)

            duration_cs = max(1, int(round(max(0.01, next_start - word_start) * 100)))
            tokens.append(r"{\k" + str(duration_cs) + "}" + escape_ass_text(item["word"]))

        text_line = " ".join(tokens).strip()
        if text_line:
            events.append(
                {
                    "start": start,
                    "end": max(end, start + 0.05),
                    "text": text_line,
                }
            )
        batch = []

    return events


def create_karaoke_ass(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    ass_path: Path,
    font_size: int = DEFAULT_FONT_SIZE,
    border_size: int = DEFAULT_BORDER_SIZE,
    highlight_color: str = DEFAULT_HIGHLIGHT_COLOR,
    non_highlight_color: str = DEFAULT_NON_HIGHLIGHT_COLOR,
    border_color: str = DEFAULT_BORDER_COLOR,
) -> None:
    sorted_words = sorted(words, key=lambda item: to_float(item.get("start")))
    sorted_segments = sorted(segments, key=lambda item: to_float(item.get("start")))

    if not sorted_words:
        raise RuntimeError("No word-level timestamps available for karaoke rendering.")

    events: list[dict[str, Any]] = []
    used_indexes: set[int] = set()
    cursor = 0
    tolerance = 0.08

    for segment in sorted_segments:
        segment_start = to_float(segment.get("start"))
        segment_end = to_float(segment.get("end"), segment_start)

        segment_words: list[dict[str, Any]] = []
        while cursor < len(sorted_words):
            word = sorted_words[cursor]
            word_start = to_float(word.get("start"))
            word_end = to_float(word.get("end"), word_start)

            if word_end < segment_start - tolerance:
                cursor += 1
                continue
            if word_start > segment_end + tolerance:
                break

            segment_words.append(word)
            used_indexes.add(cursor)
            cursor += 1

        if segment_words:
            events.extend(words_to_karaoke_events(segment_words))

    leftovers = [word for idx, word in enumerate(sorted_words) if idx not in used_indexes]
    if leftovers:
        events.extend(words_to_karaoke_events(leftovers))

    if not events:
        raise RuntimeError("No karaoke events were generated from transcript timestamps.")

    safe_font_size = clamp_font_size(font_size)
    safe_border_size = clamp_border_size(border_size)
    safe_highlight_color = normalize_hex_color(highlight_color, DEFAULT_HIGHLIGHT_COLOR)
    safe_non_highlight_color = normalize_hex_color(
        non_highlight_color,
        DEFAULT_NON_HIGHLIGHT_COLOR,
    )
    safe_border_color = normalize_hex_color(
        border_color,
        DEFAULT_BORDER_COLOR,
    )
    header = """[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,{font_size},{primary},{secondary},{outline},{back},0,0,0,0,100,100,0,0,1,{outline_size},0.6,2,90,90,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
        font_size=safe_font_size,
        primary=hex_to_ass_bgr(safe_highlight_color),
        secondary=hex_to_ass_bgr(safe_non_highlight_color),
        outline=hex_to_ass_bgr(safe_border_color),
        back=TRANSPARENT_ASS_COLOR,
        outline_size=safe_border_size,
    )
    lines = [header.rstrip("\n")]
    for event in events:
        start = format_ass_timestamp(to_float(event["start"]))
        end = format_ass_timestamp(to_float(event["end"]))
        lines.append(f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{event['text']}")

    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def burn_karaoke_subtitles(video_path: Path, ass_path: Path, output_path: Path) -> None:
    ass_filter_path = ass_path.resolve().as_posix().replace("'", r"\'")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"ass='{ass_filter_path}'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


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
    status: Optional[str] = None,
    text: Optional[str] = None,
    error: Optional[str] = None,
    transcript_json_path: Optional[str] = None,
    transcript_text_path: Optional[str] = None,
    subtitle_status: Optional[str] = None,
    subtitle_error: Optional[str] = None,
    karaoke_ass_path: Optional[str] = None,
    karaoke_video_path: Optional[str] = None,
    font_size: Optional[int] = None,
    border_size: Optional[int] = None,
    highlight_color: Optional[str] = None,
    non_highlight_color: Optional[str] = None,
    border_color: Optional[str] = None,
) -> None:
    with jobs_lock:
        state = get_default_job_state()
        state.update(transcription_jobs.get(job_id, {}))

        if status is not None:
            state["status"] = status
        if text is not None:
            state["text"] = text
        if error is not None:
            state["error"] = error
        if transcript_json_path is not None:
            state["transcript_json_path"] = transcript_json_path
        if transcript_text_path is not None:
            state["transcript_text_path"] = transcript_text_path
        if subtitle_status is not None:
            state["subtitle_status"] = subtitle_status
        if subtitle_error is not None:
            state["subtitle_error"] = subtitle_error
        if karaoke_ass_path is not None:
            state["karaoke_ass_path"] = karaoke_ass_path
        if karaoke_video_path is not None:
            state["karaoke_video_path"] = karaoke_video_path
        if font_size is not None:
            state["font_size"] = clamp_font_size(font_size)
        if border_size is not None:
            state["border_size"] = clamp_border_size(border_size)
        if highlight_color is not None:
            state["highlight_color"] = normalize_hex_color(
                highlight_color,
                DEFAULT_HIGHLIGHT_COLOR,
            )
        if non_highlight_color is not None:
            state["non_highlight_color"] = normalize_hex_color(
                non_highlight_color,
                DEFAULT_NON_HIGHLIGHT_COLOR,
            )
        if border_color is not None:
            state["border_color"] = normalize_hex_color(
                border_color,
                DEFAULT_BORDER_COLOR,
            )

        transcription_jobs[job_id] = state


def get_transcription_job(job_id: Optional[str]) -> dict[str, Any]:
    if not job_id:
        return get_default_job_state()
    with jobs_lock:
        state = get_default_job_state()
        state.update(transcription_jobs.get(job_id, {}))
        return state


def run_transcription_job(job_id: str, audio_path: str, api_key: str) -> None:
    set_transcription_job(
        job_id,
        status="processing",
        error="",
        subtitle_status="idle",
        subtitle_error="",
        karaoke_ass_path="",
        karaoke_video_path="",
    )
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
            status="completed",
            text=text,
            transcript_json_path=str(transcript_json_path),
            transcript_text_path=str(transcript_text_path),
            subtitle_status="idle",
            subtitle_error="",
        )
    except Exception as exc:
        logger.exception("Transcription job failed for job_id=%s audio_path=%s", job_id, audio_path)
        set_transcription_job(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            subtitle_status="idle",
        )


def run_subtitle_burn_job(
    job_id: str,
    video_path: str,
    transcript_json_path: str,
    font_size: int,
    border_size: int,
    highlight_color: str,
    non_highlight_color: str,
    border_color: str,
) -> None:
    safe_font_size = clamp_font_size(font_size)
    safe_border_size = clamp_border_size(border_size)
    safe_highlight_color = normalize_hex_color(highlight_color, DEFAULT_HIGHLIGHT_COLOR)
    safe_non_highlight_color = normalize_hex_color(
        non_highlight_color,
        DEFAULT_NON_HIGHLIGHT_COLOR,
    )
    safe_border_color = normalize_hex_color(
        border_color,
        DEFAULT_BORDER_COLOR,
    )
    set_transcription_job(
        job_id,
        subtitle_status="processing",
        subtitle_error="",
        karaoke_ass_path="",
        karaoke_video_path="",
        font_size=safe_font_size,
        border_size=safe_border_size,
        highlight_color=safe_highlight_color,
        non_highlight_color=safe_non_highlight_color,
        border_color=safe_border_color,
    )

    try:
        transcript_path = Path(transcript_json_path)
        if not transcript_path.exists():
            raise RuntimeError("Transcript JSON not found. Please transcribe again.")

        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        segments = payload.get("segments") or []
        words = payload.get("words") or []

        render_id = uuid4().hex[:8]
        karaoke_ass_path = SUBTITLE_DIR / f"{job_id}_{render_id}.ass"
        karaoke_video_path = RENDER_DIR / f"{job_id}_{render_id}_karaoke.mp4"

        create_karaoke_ass(
            segments=segments,
            words=words,
            ass_path=karaoke_ass_path,
            font_size=safe_font_size,
            border_size=safe_border_size,
            highlight_color=safe_highlight_color,
            non_highlight_color=safe_non_highlight_color,
            border_color=safe_border_color,
        )
        burn_karaoke_subtitles(Path(video_path), karaoke_ass_path, karaoke_video_path)

        set_transcription_job(
            job_id,
            subtitle_status="completed",
            subtitle_error="",
            karaoke_ass_path=str(karaoke_ass_path),
            karaoke_video_path=str(karaoke_video_path),
        )
    except Exception as exc:
        logger.exception("Subtitle burn failed for job_id=%s video_path=%s", job_id, video_path)
        set_transcription_job(
            job_id,
            subtitle_status="failed",
            subtitle_error=f"{type(exc).__name__}: {exc}",
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
    broll_uploaded = request.session.pop("broll_uploaded", False)
    broll_image_count = int(request.session.get("broll_image_count", 0))
    broll_rendered_video_path = request.session.get("broll_rendered_video_path", "")
    broll_render_error = request.session.pop("broll_render_error", "")
    return templates.TemplateResponse(
        "studio.html",
        {
            "request": request,
            "masked_key": mask_api_key(api_key),
            "video_uploaded": video_uploaded,
            "broll_uploaded": broll_uploaded,
            "broll_image_count": broll_image_count,
            "transcription_status": transcription_job["status"],
            "transcription_text": transcription_job["text"],
            "transcription_error": transcription_job["error"],
            "subtitle_status": transcription_job["subtitle_status"],
            "subtitle_error": transcription_job["subtitle_error"],
            "karaoke_video_path": transcription_job["karaoke_video_path"],
            "selected_font_size": transcription_job["font_size"],
            "selected_border_size": transcription_job["border_size"],
            "selected_highlight_color": transcription_job["highlight_color"],
            "selected_non_highlight_color": transcription_job["non_highlight_color"],
            "selected_border_color": transcription_job["border_color"],
            "color_swatches": COLOR_SWATCHES,
            "broll_rendered_video_path": broll_rendered_video_path,
            "broll_render_error": broll_render_error,
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
    set_transcription_job(
        job_id,
        status="processing",
        error="",
        subtitle_status="idle",
        subtitle_error="",
        karaoke_ass_path="",
        karaoke_video_path="",
        font_size=DEFAULT_FONT_SIZE,
        border_size=DEFAULT_BORDER_SIZE,
        highlight_color=DEFAULT_HIGHLIGHT_COLOR,
        non_highlight_color=DEFAULT_NON_HIGHLIGHT_COLOR,
        border_color=DEFAULT_BORDER_COLOR,
    )
    background_tasks.add_task(run_transcription_job, job_id, str(target_audio), api_key)

    request.session["uploaded_video_path"] = str(target_file)
    request.session["uploaded_audio_path"] = str(target_audio)
    request.session["transcription_job_id"] = job_id
    request.session["transcription_text_path"] = str(TRANSCRIPT_DIR / f"{job_id}.txt")
    request.session["transcription_json_path"] = str(TRANSCRIPT_DIR / f"{job_id}.json")
    request.session["karaoke_ass_path"] = ""
    request.session["karaoke_video_path"] = ""
    request.session.pop("broll_set_id", None)
    request.session.pop("broll_dir_path", None)
    request.session.pop("broll_image_count", None)
    request.session.pop("broll_uploaded", None)
    request.session.pop("broll_rendered_video_path", None)
    request.session.pop("broll_render_error", None)
    request.session["video_uploaded"] = True
    return RedirectResponse(url="/studio", status_code=303)


@app.post("/upload-broll-folder")
async def upload_broll_folder(
    request: Request,
    broll_images: list[UploadFile] = File(...),
):
    api_key = request.session.get("openai_api_key")
    if not api_key:
        return RedirectResponse(url="/", status_code=303)

    broll_set_id = uuid4().hex
    target_root = BROLL_DIR / broll_set_id
    saved_count = 0

    for image in broll_images:
        filename = image.filename or ""
        if not is_supported_broll_image(filename):
            await image.close()
            continue

        try:
            relative_path = sanitize_uploaded_relative_path(filename)
        except ValueError:
            await image.close()
            continue

        target_file = target_root / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with target_file.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        await image.close()
        saved_count += 1

    request.session["broll_set_id"] = broll_set_id
    request.session["broll_dir_path"] = str(target_root)
    request.session["broll_image_count"] = saved_count
    request.session["broll_uploaded"] = True
    request.session["broll_rendered_video_path"] = ""
    request.session["broll_render_error"] = ""

    transcript_json_path = request.session.get("transcription_json_path")
    if saved_count > 0 and transcript_json_path:
        try:
            placement_result = generate_broll_placement_result(
                api_key=api_key,
                transcript_json_path=str(transcript_json_path),
                broll_dir_path=str(target_root),
            )
            placement_path = target_root / "placement_result.json"
            placement_path.write_text(
                json.dumps(placement_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            job_id = request.session.get("transcription_job_id")
            job = get_transcription_job(job_id)
            source_video_path = (
                job.get("karaoke_video_path") or request.session.get("uploaded_video_path")
            )
            if not source_video_path:
                raise RuntimeError("No source video path found for b-roll rendering.")

            broll_video_output_path = RENDER_DIR / f"{broll_set_id}_broll.mp4"
            render_broll_video(
                video_path=Path(str(source_video_path)),
                broll_dir_path=target_root,
                placement_result=placement_result,
                output_path=broll_video_output_path,
            )
            request.session["broll_rendered_video_path"] = str(broll_video_output_path)
            logger.info("B-roll video rendered at %s", broll_video_output_path)
        except Exception:
            request.session["broll_render_error"] = "B-roll render failed."
            logger.exception(
                "Automatic b-roll placement/render failed for broll_set_id=%s",
                broll_set_id,
            )

    return RedirectResponse(url="/studio", status_code=303)


@app.post("/burn-subtitles")
async def burn_subtitles(
    request: Request,
    background_tasks: BackgroundTasks,
    font_size: int = Form(DEFAULT_FONT_SIZE),
    border_size: int = Form(DEFAULT_BORDER_SIZE),
    highlight_color: str = Form(DEFAULT_HIGHLIGHT_COLOR),
    non_highlight_color: str = Form(DEFAULT_NON_HIGHLIGHT_COLOR),
    border_color: str = Form(DEFAULT_BORDER_COLOR),
):
    job_id = request.session.get("transcription_job_id")
    if not job_id:
        return RedirectResponse(url="/studio", status_code=303)

    job = get_transcription_job(job_id)
    if job.get("status") != "completed":
        return RedirectResponse(url="/studio", status_code=303)
    if job.get("subtitle_status") == "processing":
        return RedirectResponse(url="/studio", status_code=303)

    video_path = request.session.get("uploaded_video_path")
    transcript_json_path = request.session.get("transcription_json_path")
    if not video_path or not transcript_json_path:
        set_transcription_job(
            job_id,
            subtitle_status="failed",
            subtitle_error="Missing video or transcript path.",
        )
        return RedirectResponse(url="/studio", status_code=303)

    safe_font_size = clamp_font_size(font_size)
    safe_border_size = clamp_border_size(border_size)
    safe_highlight_color = normalize_hex_color(highlight_color, DEFAULT_HIGHLIGHT_COLOR)
    safe_non_highlight_color = normalize_hex_color(
        non_highlight_color,
        DEFAULT_NON_HIGHLIGHT_COLOR,
    )
    safe_border_color = normalize_hex_color(
        border_color,
        DEFAULT_BORDER_COLOR,
    )
    set_transcription_job(
        job_id,
        subtitle_status="processing",
        subtitle_error="",
        karaoke_ass_path="",
        karaoke_video_path="",
        font_size=safe_font_size,
        border_size=safe_border_size,
        highlight_color=safe_highlight_color,
        non_highlight_color=safe_non_highlight_color,
        border_color=safe_border_color,
    )
    request.session["broll_rendered_video_path"] = ""
    request.session["broll_render_error"] = ""

    background_tasks.add_task(
        run_subtitle_burn_job,
        job_id,
        str(video_path),
        str(transcript_json_path),
        safe_font_size,
        safe_border_size,
        safe_highlight_color,
        safe_non_highlight_color,
        safe_border_color,
    )
    return RedirectResponse(url="/studio", status_code=303)


@app.get("/transcription-status")
async def transcription_status(request: Request):
    job_id = request.session.get("transcription_job_id")
    job = get_transcription_job(job_id)
    return JSONResponse(job)


@app.post("/generate-broll-placement")
async def generate_broll_placement(request: Request):
    api_key = request.session.get("openai_api_key")
    if not api_key:
        return JSONResponse({"error": "Missing API key in session."}, status_code=400)

    transcript_json_path = request.session.get("transcription_json_path")
    if not transcript_json_path:
        return JSONResponse({"error": "Missing transcript JSON path."}, status_code=400)

    broll_dir_path = request.session.get("broll_dir_path")
    if not broll_dir_path:
        return JSONResponse({"error": "Missing b-roll folder upload."}, status_code=400)

    try:
        placement_result = generate_broll_placement_result(
            api_key=api_key,
            transcript_json_path=str(transcript_json_path),
            broll_dir_path=str(broll_dir_path),
        )
        return JSONResponse({"result": placement_result})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        logger.exception("B-roll placement generation failed.")
        return JSONResponse({"error": "B-roll placement generation failed."}, status_code=500)


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
    request.session.pop("karaoke_ass_path", None)
    request.session.pop("karaoke_video_path", None)
    request.session.pop("broll_set_id", None)
    request.session.pop("broll_dir_path", None)
    request.session.pop("broll_image_count", None)
    request.session.pop("broll_uploaded", None)
    request.session.pop("broll_rendered_video_path", None)
    request.session.pop("broll_render_error", None)
    request.session.pop("video_uploaded", None)
    return RedirectResponse(url="/", status_code=303)
