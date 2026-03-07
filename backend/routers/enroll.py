# backend/routers/enroll.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import hashlib
import base64
import numpy as np
import librosa
import tempfile
import os

from database.db import get_db
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str

class KeystrokeEnroll(BaseModel):
    username: str

    # Raw arrays
    dwell_times:  List[float]
    flight_times: List[float]
    typing_speed: float = 0

    # Core timing
    dwell_mean:   float = 0
    dwell_std:    float = 0
    dwell_median: float = 0
    dwell_min:    float = 0
    dwell_max:    float = 0

    flight_mean:   float = 0
    flight_std:    float = 0
    flight_median: float = 0

    p2p_mean: float = 0
    p2p_std:  float = 0
    r2r_mean: float = 0
    r2r_std:  float = 0

    # Digraphs — updated to match "Biometric Voice Keystroke Authentication"
    digraph_th: float = 0
    digraph_he: float = 0
    digraph_bi: float = 0
    digraph_io: float = 0
    digraph_om: float = 0
    digraph_me: float = 0
    digraph_et: float = 0
    digraph_tr: float = 0
    digraph_ri: float = 0
    digraph_ic: float = 0
    digraph_vo: float = 0
    digraph_oi: float = 0
    digraph_ce: float = 0
    digraph_ke: float = 0
    digraph_ey: float = 0
    digraph_ys: float = 0
    digraph_st: float = 0
    digraph_ro: float = 0
    digraph_ok: float = 0
    digraph_au: float = 0
    digraph_ut: float = 0
    digraph_en: float = 0
    digraph_nt: float = 0
    digraph_ti: float = 0
    digraph_ca: float = 0
    digraph_at: float = 0
    digraph_on: float = 0

    # Behavioral
    typing_speed_cpm:        float = 0
    typing_duration:         float = 0
    rhythm_mean:             float = 0
    rhythm_std:              float = 0
    rhythm_cv:               float = 0
    pause_count:             float = 0
    pause_mean:              float = 0
    backspace_ratio:         float = 0
    backspace_count:         float = 0
    hand_alternation_ratio:  float = 0
    same_hand_sequence_mean: float = 0
    finger_transition_ratio: float = 0
    seek_time_mean:          float = 0
    seek_time_count:         float = 0

    # Shift-lag (new)
    shift_lag_mean:  float = 0
    shift_lag_std:   float = 0
    shift_lag_count: float = 0

    # Normalized ratios (new)
    dwell_mean_norm:  float = 0
    dwell_std_norm:   float = 0
    flight_mean_norm: float = 0
    flight_std_norm:  float = 0
    p2p_std_norm:     float = 0
    r2r_mean_norm:    float = 0
    shift_lag_norm:   float = 0


class VoiceEnroll(BaseModel):
    username: str
    mfcc_features: List[float]

class SecurityEnroll(BaseModel):
    username: str
    question: str
    answer: str

class AudioData(BaseModel):
    audio_data:   str
    audio_format: str = "webm"
    username:     Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/user")
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
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

    # Count existing attempts for this user
    existing_count = db.query(KeystrokeTemplate).filter(
        KeystrokeTemplate.user_id == user.id
    ).count()

    # Determine enrollment session
    # Each fresh enrollment run starts a new session
    # For now we increment attempt_number per save
    attempt_number = existing_count + 1

    template = KeystrokeTemplate(
        user_id        = user.id,
        attempt_number = attempt_number,

        # Raw arrays
        dwell_times  = payload.dwell_times,
        flight_times = payload.flight_times,
        typing_speed = payload.typing_speed,

        # Core timing
        dwell_mean   = payload.dwell_mean,
        dwell_std    = payload.dwell_std,
        dwell_median = payload.dwell_median,
        dwell_min    = payload.dwell_min,
        dwell_max    = payload.dwell_max,

        flight_mean   = payload.flight_mean,
        flight_std    = payload.flight_std,
        flight_median = payload.flight_median,

        p2p_mean = payload.p2p_mean,
        p2p_std  = payload.p2p_std,
        r2r_mean = payload.r2r_mean,
        r2r_std  = payload.r2r_std,

        # Digraphs
        digraph_th = payload.digraph_th,
        digraph_he = payload.digraph_he,
        digraph_bi = payload.digraph_bi,
        digraph_io = payload.digraph_io,
        digraph_om = payload.digraph_om,
        digraph_me = payload.digraph_me,
        digraph_et = payload.digraph_et,
        digraph_tr = payload.digraph_tr,
        digraph_ri = payload.digraph_ri,
        digraph_ic = payload.digraph_ic,
        digraph_vo = payload.digraph_vo,
        digraph_oi = payload.digraph_oi,
        digraph_ce = payload.digraph_ce,
        digraph_ke = payload.digraph_ke,
        digraph_ey = payload.digraph_ey,
        digraph_ys = payload.digraph_ys,
        digraph_st = payload.digraph_st,
        digraph_ro = payload.digraph_ro,
        digraph_ok = payload.digraph_ok,
        digraph_au = payload.digraph_au,
        digraph_ut = payload.digraph_ut,
        digraph_en = payload.digraph_en,
        digraph_nt = payload.digraph_nt,
        digraph_ti = payload.digraph_ti,
        digraph_ca = payload.digraph_ca,
        digraph_at = payload.digraph_at,
        digraph_on = payload.digraph_on,

        # Behavioral
        typing_speed_cpm        = payload.typing_speed_cpm,
        typing_duration         = payload.typing_duration,
        rhythm_mean             = payload.rhythm_mean,
        rhythm_std              = payload.rhythm_std,
        rhythm_cv               = payload.rhythm_cv,
        pause_count             = payload.pause_count,
        pause_mean              = payload.pause_mean,
        backspace_ratio         = payload.backspace_ratio,
        backspace_count         = payload.backspace_count,
        hand_alternation_ratio  = payload.hand_alternation_ratio,
        same_hand_sequence_mean = payload.same_hand_sequence_mean,
        finger_transition_ratio = payload.finger_transition_ratio,
        seek_time_mean          = payload.seek_time_mean,
        seek_time_count         = payload.seek_time_count,

        # Shift-lag
        shift_lag_mean  = payload.shift_lag_mean,
        shift_lag_std   = payload.shift_lag_std,
        shift_lag_count = payload.shift_lag_count,

        # Normalized ratios
        dwell_mean_norm  = payload.dwell_mean_norm,
        dwell_std_norm   = payload.dwell_std_norm,
        flight_mean_norm = payload.flight_mean_norm,
        flight_std_norm  = payload.flight_std_norm,
        p2p_std_norm     = payload.p2p_std_norm,
        r2r_mean_norm    = payload.r2r_mean_norm,
        shift_lag_norm   = payload.shift_lag_norm,
    )

    db.add(template)
    db.commit()

    print(f"✅ Keystroke attempt #{attempt_number} saved for user '{payload.username}'")
    print(f"   dwell_mean={payload.dwell_mean:.2f}ms  flight_mean={payload.flight_mean:.2f}ms  cpm={payload.typing_speed_cpm:.1f}")

    return {
        "success":        True,
        "message":        f"Keystroke attempt #{attempt_number} saved",
        "attempt_number": attempt_number,
    }


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
        db.flush()

    template = VoiceTemplate(
        user_id       = user.id,
        mfcc_features = payload.mfcc_features,
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
        db.flush()

    sq = SecurityQuestion(
        user_id     = user.id,
        question    = payload.question,
        answer_hash = answer_hash,
    )
    db.add(sq)
    db.commit()
    return {"success": True, "message": "Security question saved"}


@router.post("/extract-mfcc")
async def extract_mfcc(payload: AudioData, db: Session = Depends(get_db)):
    input_path = None
    wav_path   = None
    try:
        audio_bytes  = base64.b64decode(payload.audio_data)
        audio_format = payload.audio_format or "webm"

        print(f"\n{'='*60}")
        print(f"VOICE VALIDATION PIPELINE")
        print(f"{'='*60}")
        print(f"Audio bytes: {len(audio_bytes)}, format: {audio_format}")

        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
            tmp.write(audio_bytes)
            input_path = tmp.name

        wav_path = input_path.replace(f".{audio_format}", ".wav")

        from pydub import AudioSegment
        audio_segment = (
            AudioSegment.from_file(input_path, format=audio_format)
            .set_frame_rate(16000)
            .set_channels(1)
            .set_sample_width(2)
        )
        audio_segment.export(wav_path, format="wav")

        audio, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
        total_duration = len(audio) / sample_rate
        print(f"Total duration: {total_duration:.2f}s")

        # ── WebRTC VAD ────────────────────────────────────────────────────
        import webrtcvad
        vad = webrtcvad.Vad(3)

        with open(wav_path, 'rb') as f:
            wav_data = f.read()

        audio_data   = wav_data[44:]
        frame_duration = 30
        frame_size   = int(sample_rate * frame_duration / 1000) * 2

        voiced_frames = 0
        total_frames  = 0
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i:i + frame_size]
            if len(frame) == frame_size:
                total_frames += 1
                if vad.is_speech(frame, sample_rate):
                    voiced_frames += 1

        voice_ratio    = voiced_frames / total_frames if total_frames > 0 else 0
        speech_duration = (voiced_frames * frame_duration) / 1000

        print(f"WebRTC VAD: voice_ratio={voice_ratio:.2%}  speech={speech_duration:.2f}s")

        if voice_ratio < 0.40:
            return {"success": False, "detail": f"Insufficient voice detected ({voice_ratio:.0%}). Please speak clearly."}
        if speech_duration < 1.5:
            return {"success": False, "detail": f"Speech too short ({speech_duration:.1f}s). Please speak the full phrase."}

        # ── Energy & Spectral ─────────────────────────────────────────────
        rms_energy = np.sqrt(np.mean(audio**2))
        if rms_energy < 0.02:
            return {"success": False, "detail": f"Audio too quiet (RMS: {rms_energy:.4f}). Please speak louder."}

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        mean_centroid     = np.mean(spectral_centroid)
        if mean_centroid < 600 or mean_centroid > 5000:
            return {"success": False, "detail": f"Audio doesn't sound like speech (centroid: {mean_centroid:.0f}Hz)."}

        zcr      = librosa.feature.zero_crossing_rate(audio)
        mean_zcr = np.mean(zcr)
        if mean_zcr < 0.02 or mean_zcr > 0.5:
            return {"success": False, "detail": f"Audio has unusual characteristics (ZCR: {mean_zcr:.4f})."}

        # ── MFCC ─────────────────────────────────────────────────────────
        mfccs    = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std  = np.std(mfccs, axis=1)

        if np.mean(mfcc_std) < 5.0:
            return {"success": False, "detail": "Audio lacks variation typical of speech."}
        if mfcc_mean[0] < -500 or mfcc_mean[0] > 500:
            return {"success": False, "detail": f"Audio spectrum unusual (MFCC0: {mfcc_mean[0]:.0f})."}

        print(f"✅ ALL VALIDATIONS PASSED")
        return {
            "success":      True,
            "mfcc_features": mfcc_mean.tolist(),
            "validation": {
                "voice_ratio":        float(voice_ratio),
                "speech_duration":    float(speech_duration),
                "rms_energy":         float(rms_energy),
                "spectral_centroid":  float(mean_centroid),
            }
        }

    except Exception as e:
        print(f"❌ MFCC ERROR: {e}")
        import traceback; traceback.print_exc()
        return {"success": False, "detail": str(e)}

    finally:
        for path in [input_path, wav_path]:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except:
                pass