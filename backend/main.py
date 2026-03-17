# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from routers import enroll, auth
import os

app = FastAPI(
    title="Multimodal Biometric Authentication System",
    description="Keystroke Dynamics + Speech Biometrics",
    version="1.0.0"
)

# CORS — restrict to localhost; change to your domain if deploying
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(enroll.router, prefix="/api/enroll", tags=["Enrollment"])
app.include_router(auth.router,   prefix="/api/auth",   tags=["Authentication"])

# ── Serve frontend as static files ────────────────────────────────────────────
# Resolves to: biometric_auth/frontend/
frontend_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
)

app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/pages/enroll.html")

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