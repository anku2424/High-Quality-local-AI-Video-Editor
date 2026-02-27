import os
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI(title="Local OpenAI Key Demo")

# For local development only; replace in production.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "local-dev-secret-change-me"),
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}"


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
    return templates.TemplateResponse(
        "studio.html",
        {
            "request": request,
            "masked_key": mask_api_key(api_key),
            "video_uploaded": video_uploaded,
        },
    )


@app.post("/save-key")
async def save_key(request: Request, api_key: str = Form(...)):
    # Remove accidental spaces/newlines from pasted keys.
    normalized_key = "".join(api_key.split())
    request.session["openai_api_key"] = normalized_key
    return RedirectResponse(url="/studio", status_code=303)


@app.post("/upload-video")
async def upload_video(request: Request, video: UploadFile = File(...)):
    api_key = request.session.get("openai_api_key")
    if not api_key:
        return RedirectResponse(url="/", status_code=303)

    suffix = Path(video.filename or "").suffix
    target_file = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    with target_file.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    await video.close()

    request.session["uploaded_video_path"] = str(target_file)
    request.session["video_uploaded"] = True
    return RedirectResponse(url="/studio", status_code=303)


@app.post("/clear-key")
async def clear_key(request: Request):
    request.session.pop("openai_api_key", None)
    request.session.pop("uploaded_video_path", None)
    request.session.pop("video_uploaded", None)
    return RedirectResponse(url="/", status_code=303)
