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
# Serves api.js with API_BASE rewritten to match the actual host (ngrok or localhost).
# Must be a route, not a middleware — middleware fires after StaticFiles already
# streams the file, making the response body inaccessible for rewriting.
@app.get("/static/js/api.js")
async def serve_api_js(request: Request):
    # Determine the actual host from request headers
    forwarded_host  = request.headers.get("x-forwarded-host")
    forwarded_proto = request.headers.get("x-forwarded-proto", "https")
    host            = request.headers.get("host", "127.0.0.1:8000")

    if forwarded_host:
        api_base = f"{forwarded_proto}://{forwarded_host}/api"
    else:
        proto    = "https" if request.url.scheme == "https" else "http"
        api_base = f"{proto}://{host}/api"

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
        headers={"Cache-Control": "no-store"},  # prevent browser caching old API_BASE
    )

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