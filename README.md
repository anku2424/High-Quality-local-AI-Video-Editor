# FastAPI Local App (OpenAI Key UI)

A simple local FastAPI web app with a UI to collect an OpenAI API key.

## Project structure

```
.
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── static
│   │   └── styles.css
│   └── templates
│       ├── index.html
│       └── studio.html
├── uploads
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## Run locally

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the server:

```bash
uvicorn app.main:app --reload
```

4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Notes

- After saving your API key, the app redirects to `/studio`.
- The studio page lets you clear the key and upload a video.
- After upload, a success prompt shows: `video uploaded`.
- Uploaded files are stored in the local `uploads/` folder.
- The API key is stored in a signed session cookie for local demo use.
- Change `SESSION_SECRET` before any non-local usage.
