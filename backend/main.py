# backend/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from routers import enroll, auth
import os, re

app = FastAPI(
    title="Multimodal Biometric Authentication System",
    description="Keystroke Dynamics + Speech Biometrics",
    version="1.0.0"
)

# CORS — allow localhost and any ngrok tunnel URL
# ngrok URLs look like: https://abc123.ngrok-free.app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ngrok URL changes every session — allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(enroll.router, prefix="/api/enroll", tags=["Enrollment"])
app.include_router(auth.router,   prefix="/api/auth",   tags=["Authentication"])

# ── Serve frontend as static files ────────────────────────────────────────────
frontend_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
)

# ── Dynamic API_BASE injection ────────────────────────────────────────────────
# When accessed via ngrok, the browser sends requests to http://127.0.0.1:8000
# which is your local machine — not accessible to remote users.
# This middleware rewrites api.js on the fly so API_BASE always points to
# whatever host the user is actually accessing (ngrok URL or localhost).
@app.middleware("http")
async def inject_api_base(request: Request, call_next):
    response = await call_next(request)

    # Only rewrite api.js
    if request.url.path == "/static/js/api.js":
        # Determine the actual host the request came from
        forwarded_host = request.headers.get("x-forwarded-host")
        forwarded_proto = request.headers.get("x-forwarded-proto", "https")
        host = request.headers.get("host", "127.0.0.1:8000")

        if forwarded_host:
            api_base = f"{forwarded_proto}://{forwarded_host}/api"
        else:
            proto = "https" if request.url.scheme == "https" else "http"
            api_base = f"{proto}://{host}/api"

        # Read the original file and replace the API_BASE line
        api_js_path = os.path.join(frontend_dir, "js", "api.js")
        with open(api_js_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = re.sub(
            r'const API_BASE\s*=\s*["\'].*?["\'];',
            f'const API_BASE = "{api_base}";',
            content
        )

        return HTMLResponse(
            content=content,
            media_type="application/javascript",
            status_code=200,
        )

    return response

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/pages/login.html")

@app.get("/enroll")
def go_enroll():
    return RedirectResponse(url="/static/pages/enroll.html")

@app.get("/login")
def go_login():
    return RedirectResponse(url="/static/pages/login.html")

@app.get("/dashboard")
def go_dashboard():
    return RedirectResponse(url="/static/pages/dashboard.html")

@app.get("/health")
def health_check():
    return {"status": "ok"}