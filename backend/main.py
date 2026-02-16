# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import enroll, auth

app = FastAPI(
    title="Multimodal Biometric Authentication System",
    description="Keystroke Dynamics + Speech Biometrics",
    version="1.0.0"
)

# Fix CORS — allow Live Server origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(enroll.router, prefix="/api/enroll", tags=["Enrollment"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])

@app.get("/")
def root():
    return {
        "system": "Multimodal Biometric Authentication",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}