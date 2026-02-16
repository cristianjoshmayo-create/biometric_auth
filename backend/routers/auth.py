# backend/routers/auth.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import hashlib

from database.db import get_db
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion, AuthLog

router = APIRouter()

# ── Schemas ───────────────────────────────────────────────

class KeystrokeAuth(BaseModel):
    username: str
    dwell_times: List[float]
    flight_times: List[float]
    typing_speed: float
    dwell_mean: float = 0
    dwell_std: float = 0
    flight_mean: float = 0
    flight_std: float = 0
    p2p_mean: float = 0
    p2p_std: float = 0
    rhythm_cv: float = 0
    typing_speed_cpm: float = 0

class VoiceAuth(BaseModel):
    username: str
    mfcc_features: List[float]

class SecurityAuth(BaseModel):
    username: str
    answer: str

# ── Auth Threshold ────────────────────────────────────────
KEYSTROKE_THRESHOLD = 0.40   # Phase 4 basic threshold
VOICE_THRESHOLD     = 0.50

# ── Helper: cosine similarity ─────────────────────────────
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0

# ── Helper: feature similarity ────────────────────────────
def feature_similarity(enrolled, live, tolerance=0.30):
    """Compare two feature arrays with tolerance"""
    enrolled = np.array(enrolled)
    live = np.array(live)
    min_len = min(len(enrolled), len(live))
    enrolled = enrolled[:min_len]
    live = live[:min_len]

    # Normalize difference
    diffs = np.abs(enrolled - live) / (np.abs(enrolled) + 1e-6)
    score = float(np.mean(diffs < tolerance))
    return score

# ── Log auth attempt ──────────────────────────────────────
def log_attempt(db, user_id, method, confidence, result):
    log = AuthLog(
        user_id=user_id,
        auth_method=method,
        confidence_score=confidence,
        result=result
    )
    db.add(log)
    db.commit()

# ── Endpoints ─────────────────────────────────────────────

@router.post("/keystroke")
def verify_keystroke(payload: KeystrokeAuth, db: Session = Depends(get_db)):
    # Get user
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if flagged
    if user.is_flagged:
        raise HTTPException(status_code=403, detail="Account flagged")

    # Get enrolled template
    template = db.query(KeystrokeTemplate).filter(
        KeystrokeTemplate.user_id == user.id
    ).first()
    if not template:
        raise HTTPException(status_code=404, detail="No keystroke template found")

    # Compare dwell times
    dwell_score = feature_similarity(
        template.dwell_times,
        payload.dwell_times,
        tolerance=0.30
    )

    # Compare flight times
    flight_score = feature_similarity(
        template.flight_times,
        payload.flight_times,
        tolerance=0.35
    )

    # Compare typing speed
    speed_diff = abs(template.typing_speed - payload.typing_speed)
    speed_score = float(speed_diff < (template.typing_speed * 0.30))

    # Final confidence score
    confidence = (0.50 * dwell_score) + (0.30 * flight_score) + (0.20 * speed_score)

    authenticated = confidence >= KEYSTROKE_THRESHOLD

    print(f"Keystroke auth — user: {payload.username}")
    print(f"  dwell: {dwell_score:.3f}, flight: {flight_score:.3f}, speed: {speed_score:.3f}")
    print(f"  confidence: {confidence:.3f}, result: {'PASS' if authenticated else 'FAIL'}")

    # Log attempt
    log_attempt(
        db, user.id, "keystroke",
        confidence,
        "granted" if authenticated else "denied"
    )

    # Flag user if too many failed attempts
    if not authenticated:
        recent_fails = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.result == "denied"
        ).count()

        if recent_fails >= 5:
            user.is_flagged = True
            db.commit()

    return {
        "authenticated": authenticated,
        "confidence": confidence,
        "breakdown": {
            "dwell_score": dwell_score,
            "flight_score": flight_score,
            "speed_score": speed_score
        }
    }


@router.post("/voice")
def verify_voice(payload: VoiceAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    template = db.query(VoiceTemplate).filter(
        VoiceTemplate.user_id == user.id
    ).first()
    if not template:
        raise HTTPException(status_code=404, detail="No voice template found")

    # Compare MFCC features using cosine similarity
    confidence = cosine_similarity(
        template.mfcc_features,
        payload.mfcc_features
    )

    # Normalize to 0-1 range
    confidence = (confidence + 1) / 2

    authenticated = confidence >= VOICE_THRESHOLD

    print(f"Voice auth — user: {payload.username}")
    print(f"  confidence: {confidence:.3f}, result: {'PASS' if authenticated else 'FAIL'}")

    log_attempt(
        db, user.id, "voice",
        confidence,
        "granted" if authenticated else "denied"
    )

    return {
        "authenticated": authenticated,
        "confidence": confidence
    }


@router.get("/security-question/{username}")
def get_security_question(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sq = db.query(SecurityQuestion).filter(
        SecurityQuestion.user_id == user.id
    ).first()
    if not sq:
        raise HTTPException(status_code=404, detail="No security question found")

    return {"question": sq.question}


@router.post("/security")
def verify_security(payload: SecurityAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sq = db.query(SecurityQuestion).filter(
        SecurityQuestion.user_id == user.id
    ).first()
    if not sq:
        raise HTTPException(status_code=404, detail="No security question found")

    # Hash the provided answer
    answer_hash = hashlib.sha256(
        payload.answer.strip().lower().encode()
    ).hexdigest()

    authenticated = answer_hash == sq.answer_hash

    print(f"Security question auth — user: {payload.username}")
    print(f"  result: {'PASS' if authenticated else 'FAIL'}")

    log_attempt(
        db, user.id, "security_question",
        1.0 if authenticated else 0.0,
        "granted" if authenticated else "denied"
    )

    # Flag user if security question also fails
    if not authenticated:
        user.is_flagged = True
        db.commit()
        print(f"  ⚠️ User {payload.username} flagged!")

    return {
        "authenticated": authenticated,
        "confidence": 1.0 if authenticated else 0.0
    }