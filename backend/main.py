# backend/main.py

# ── Windows symlink fix — must be before ALL other imports ────────────────────
# Prevents WinError 1314 when SpeechBrain downloads the ECAPA-TDNN model.
# HuggingFace uses symlinks by default; on Windows this requires Developer Mode
# or admin rights. These vars force file copies instead.
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "warning"

# Forgives duplicate OpenMP runtimes. CTranslate2 (faster-whisper) and torch
# (SpeechBrain) each ship their own libiomp5md.dll; co-loading aborts the
# Python process on Windows without a traceback. This env var makes the second
# loader reuse the first's runtime instead of crashing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
from routers import enroll, auth
from utils.debug_logger import init_logging, get_logger, request_logging_middleware, log_error
import os

init_logging()
_startup_logger = get_logger("startup")


# Pre-load faster-whisper at startup, BEFORE any request can trigger torch/
# SpeechBrain imports. Load order matters on Windows: whichever of CTranslate2
# vs torch loads its native MKL/OpenMP runtime first wins; loading torch first
# has been observed to silently kill the process when faster-whisper initialises
# later. Warming Whisper here flips the order so ECAPA loads second.
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        enroll._get_whisper_model()
        _startup_logger.info("Whisper preloaded", extra={"stage": "startup", "user": "-"})
    except Exception as e:
        print(f"[startup] Whisper preload skipped: {type(e).__name__}: {e}")
        log_error("startup", "-", e, extra={"phase": "whisper_preload"})
    yield


app = FastAPI(
    title="Multimodal Biometric Authentication System",
    description="Keystroke Dynamics + Speech Biometrics",
    version="1.0.0",
    lifespan=lifespan,
)

# HTTP request logger — writes one line per request to logs/debug.log
app.middleware("http")(request_logging_middleware)

# Global exception catch — any unhandled error gets a full stack trace logged
from fastapi.requests import Request as _Req
from fastapi.responses import JSONResponse as _JSON

@app.exception_handler(Exception)
async def _log_unhandled(request: _Req, exc: Exception):
    log_error("unhandled", "-", exc, extra={"path": request.url.path, "method": request.method})
    return _JSON(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

# CORS — allow localhost and any ngrok tunnel URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# ── Serve all JS and HTML files with no-cache headers ─────────────────────────
# Prevents remote browsers from serving stale cached versions of JS files.
# api.js now uses window.location.host to auto-detect the correct API URL,
# so no server-side rewriting is needed anymore.

def _serve_js(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, media_type="application/javascript",
                        headers={"Cache-Control": "no-store"})

def _serve_html(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, media_type="text/html",
                        headers={"Cache-Control": "no-store"})

@app.get("/static/js/api.js")
async def serve_api_js():
    return _serve_js(os.path.join(frontend_dir, "js", "api.js"))

@app.get("/static/js/speech.js")
async def serve_speech_js():
    return _serve_js(os.path.join(frontend_dir, "js", "speech.js"))

@app.get("/static/js/enroll.js")
async def serve_enroll_js():
    return _serve_js(os.path.join(frontend_dir, "js", "enroll.js"))

@app.get("/static/js/auth_flow.js")
async def serve_auth_flow_js():
    return _serve_js(os.path.join(frontend_dir, "js", "auth_flow.js"))

@app.get("/static/js/keystroke.js")
async def serve_keystroke_js():
    return _serve_js(os.path.join(frontend_dir, "js", "keystroke.js"))

@app.get("/static/js/reset.js")
async def serve_reset_js():
    return _serve_js(os.path.join(frontend_dir, "js", "reset.js"))

@app.get("/static/pages/enroll.html")
async def serve_enroll_html():
    return _serve_html(os.path.join(frontend_dir, "pages", "enroll.html"))

@app.get("/static/pages/login.html")
async def serve_login_html():
    return _serve_html(os.path.join(frontend_dir, "pages", "login.html"))

@app.get("/static/pages/forgot-password.html")
async def serve_forgot_password_html():
    return _serve_html(os.path.join(frontend_dir, "pages", "forgot-password.html"))

@app.get("/static/pages/reset.html")
async def serve_reset_html():
    return _serve_html(os.path.join(frontend_dir, "pages", "reset.html"))


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