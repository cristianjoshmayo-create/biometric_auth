# backend/routers/enroll.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import hashlib
import base64
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os

from database.db import get_db
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion

router = APIRouter()

# ── Schemas ───────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str

class KeystrokeEnroll(BaseModel):
    username: str
    dwell_times: List[float]
    flight_times: List[float]
    typing_speed: float

class VoiceEnroll(BaseModel):
    username: str
    mfcc_features: List[float]

class SecurityEnroll(BaseModel):
    username: str
    question: str
    answer: str

class AudioData(BaseModel):
    audio_data: str
    audio_format: str = "webm"
    username: str | None = None
# ── Endpoints ─────────────────────────────────────────────

@router.post("/user")
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        return {"success": True, "message": "User already exists", "user_id": existing.id}

    new_user = User(username=payload.username)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"success": True, "message": "User created", "user_id": new_user.id}


@router.post("/keystroke")
def enroll_keystroke(payload: KeystrokeEnroll, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete existing template if re-enrolling
    existing = db.query(KeystrokeTemplate).filter(
        KeystrokeTemplate.user_id == user.id
    ).first()
    
    if existing:
        db.delete(existing)
        db.flush()  # ← this forces the delete BEFORE the insert

    template = KeystrokeTemplate(
        user_id=user.id,
        dwell_times=payload.dwell_times,
        flight_times=payload.flight_times,
        typing_speed=payload.typing_speed
    )
    db.add(template)
    db.commit()

    return {"success": True, "message": "Keystroke template saved"}


@router.post("/voice")
def enroll_voice(payload: VoiceEnroll, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    existing = db.query(VoiceTemplate).filter(
        VoiceTemplate.user_id == user.id
    ).first()
    if existing:
        db.delete(existing)
        db.flush()  # ← force delete before insert

    template = VoiceTemplate(
        user_id=user.id,
        mfcc_features=payload.mfcc_features
    )
    db.add(template)
    db.commit()

    return {"success": True, "message": "Voice template saved"}


@router.post("/security")
def enroll_security(payload: SecurityEnroll, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    answer_hash = hashlib.sha256(
        payload.answer.strip().lower().encode()
    ).hexdigest()

    existing = db.query(SecurityQuestion).filter(
        SecurityQuestion.user_id == user.id
    ).first()
    if existing:
        db.delete(existing)
        db.flush()  # ← force delete before insert

    sq = SecurityQuestion(
        user_id=user.id,
        question=payload.question,
        answer_hash=answer_hash
    )
    db.add(sq)
    db.commit()

    return {"success": True, "message": "Security question saved"}

@router.post("/extract-mfcc")
async def extract_mfcc(payload: AudioData, db: Session = Depends(get_db)):
    input_path = None
    wav_path = None
    try:
        audio_bytes = base64.b64decode(payload.audio_data)
        audio_format = payload.audio_format or "webm"

        print(f"Audio bytes: {len(audio_bytes)}, format: {audio_format}")

        # Save with correct extension
        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}", delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            input_path = tmp.name

        print(f"Saved to: {input_path}")

        # Convert to wav using pydub
        wav_path = input_path.replace(f".{audio_format}", ".wav")

        from pydub import AudioSegment
        audio_segment = AudioSegment.from_file(input_path, format=audio_format)
        audio_segment = (
            audio_segment
            .set_frame_rate(16000)
            .set_channels(1)
        )
        audio_segment.export(wav_path, format="wav")
        print(f"Converted to wav successfully")

        # Load with librosa
        audio, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
        print(f"Audio duration: {len(audio)/sample_rate:.2f}s")

        if len(audio) < 1600:
            return {
                "success": False,
                "detail": "Audio too short, speak for at least 2 seconds"
            }

        # Extract MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1).tolist()
        print(f"MFCC extracted: {len(mfcc_mean)} features")

        return {"success": True, "mfcc_features": mfcc_mean}

    except Exception as e:
        print(f"MFCC ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "detail": str(e)}

    finally:
        for path in [input_path, wav_path]:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except:
                pass