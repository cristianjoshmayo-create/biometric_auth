# backend/routers/enroll.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import func
import hashlib
import base64
import numpy as np
import librosa
import tempfile
import os
import bcrypt  # ← ADDED

from database.db import get_db
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion
import threading
import sys

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-TRAIN HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _run_training(username: str):
    """Runs in a background thread — trains keystroke model after enrollment."""
    try:
        # Add project root to path so ml/ is importable
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        )
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from ml.train_keystroke_rf import train_random_forest
        print(f"\n🔄 Auto-training keystroke model for '{username}' ...")
        model_path = train_random_forest(username)
        if model_path:
            print(f"✅ Auto-training complete → {model_path}")
        else:
            print(f"⚠  Auto-training returned no model for '{username}'")
    except Exception as e:
        import traceback
        print(f"❌ Auto-training failed for '{username}': {e}")
        traceback.print_exc()


def trigger_training(username: str):
    """Fire-and-forget: starts training in background, API responds immediately."""
    t = threading.Thread(target=_run_training, args=(username,), daemon=True)
    t.start()


def _run_voice_training(username: str):
    """Runs in a background thread — trains voice model after enrollment."""
    try:
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        )
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from ml.train_voice_cnn import train_voice_model
        print(f"\n🔄 Auto-training voice model for '{username}' ...")
        model_path = train_voice_model(username)
        if model_path:
            print(f"✅ Voice auto-training complete → {model_path}")
        else:
            print(f"⚠  Voice auto-training returned no model for '{username}'")
    except Exception as e:
        import traceback
        print(f"❌ Voice auto-training failed for '{username}': {e}")
        traceback.print_exc()


def trigger_voice_training(username: str):
    """Fire-and-forget: starts voice training in background."""
    t = threading.Thread(target=_run_voice_training, args=(username,), daemon=True)
    t.start()

# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    password: str  # ← ADDED

class KeystrokeEnroll(BaseModel):
    username: str
    dwell_times:  List[float]
    flight_times: List[float]
    typing_speed: float = 0
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
    shift_lag_mean:  float = 0
    shift_lag_std:   float = 0
    shift_lag_count: float = 0
    dwell_mean_norm:  float = 0
    dwell_std_norm:   float = 0
    flight_mean_norm: float = 0
    flight_std_norm:  float = 0
    p2p_std_norm:     float = 0
    r2r_mean_norm:    float = 0
    shift_lag_norm:   float = 0


class VoiceEnroll(BaseModel):
    username:     str
    mfcc_features: List[float]
    mfcc_std:      List[float] = []
    pitch_mean:    float = 0
    pitch_std:     float = 0
    speaking_rate: float = 0
    energy_mean:   float = 0
    energy_std:    float = 0
    zcr_mean:               float = 0
    spectral_centroid_mean: float = 0
    spectral_rolloff_mean:  float = 0


class SecurityEnroll(BaseModel):
    username: str
    question: str
    answer:   str


class AudioData(BaseModel):
    audio_data:   str
    audio_format: str = "webm"
    username:     Optional[str] = None


class ClearEnrollPayload(BaseModel):
    username: str
    confirm:  str


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/user")
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        return {"success": True, "message": "User already exists", "user_id": existing.id}

    # ← ADDED: validate and hash password
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    hashed = bcrypt.hashpw(
        payload.password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')

    new_user = User(username=payload.username, password_hash=hashed)  # ← UPDATED
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"success": True, "message": "User created", "user_id": new_user.id}


@router.post("/keystroke")
def enroll_keystroke(payload: KeystrokeEnroll, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    existing_count = db.query(KeystrokeTemplate).filter(
        KeystrokeTemplate.user_id == user.id
    ).count()

    max_order = db.query(func.max(KeystrokeTemplate.sample_order)).filter(
        KeystrokeTemplate.user_id == user.id
    ).scalar() or 0

    template = KeystrokeTemplate(
        user_id        = user.id,
        attempt_number = existing_count + 1,
        source         = "enrollment",
        sample_order   = max_order + 1,
        dwell_times    = payload.dwell_times,
        flight_times   = payload.flight_times,
        typing_speed   = payload.typing_speed,
        dwell_mean     = payload.dwell_mean,
        dwell_std      = payload.dwell_std,
        dwell_median   = payload.dwell_median,
        dwell_min      = payload.dwell_min,
        dwell_max      = payload.dwell_max,
        flight_mean    = payload.flight_mean,
        flight_std     = payload.flight_std,
        flight_median  = payload.flight_median,
        p2p_mean       = payload.p2p_mean,
        p2p_std        = payload.p2p_std,
        r2r_mean       = payload.r2r_mean,
        r2r_std        = payload.r2r_std,
        digraph_th     = payload.digraph_th,
        digraph_he     = payload.digraph_he,
        digraph_bi     = payload.digraph_bi,
        digraph_io     = payload.digraph_io,
        digraph_om     = payload.digraph_om,
        digraph_me     = payload.digraph_me,
        digraph_et     = payload.digraph_et,
        digraph_tr     = payload.digraph_tr,
        digraph_ri     = payload.digraph_ri,
        digraph_ic     = payload.digraph_ic,
        digraph_vo     = payload.digraph_vo,
        digraph_oi     = payload.digraph_oi,
        digraph_ce     = payload.digraph_ce,
        digraph_ke     = payload.digraph_ke,
        digraph_ey     = payload.digraph_ey,
        digraph_ys     = payload.digraph_ys,
        digraph_st     = payload.digraph_st,
        digraph_ro     = payload.digraph_ro,
        digraph_ok     = payload.digraph_ok,
        digraph_au     = payload.digraph_au,
        digraph_ut     = payload.digraph_ut,
        digraph_en     = payload.digraph_en,
        digraph_nt     = payload.digraph_nt,
        digraph_ti     = payload.digraph_ti,
        digraph_ca     = payload.digraph_ca,
        digraph_at     = payload.digraph_at,
        digraph_on     = payload.digraph_on,
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
        shift_lag_mean          = payload.shift_lag_mean,
        shift_lag_std           = payload.shift_lag_std,
        shift_lag_count         = payload.shift_lag_count,
        dwell_mean_norm         = payload.dwell_mean_norm,
        dwell_std_norm          = payload.dwell_std_norm,
        flight_mean_norm        = payload.flight_mean_norm,
        flight_std_norm         = payload.flight_std_norm,
        p2p_std_norm            = payload.p2p_std_norm,
        r2r_mean_norm           = payload.r2r_mean_norm,
        shift_lag_norm          = payload.shift_lag_norm,
    )
    db.add(template)
    db.commit()

    attempt_num = existing_count + 1
    print(f"✅ Keystroke attempt #{attempt_num} for '{payload.username}' | "
          f"dwell={payload.dwell_mean:.1f}ms flight={payload.flight_mean:.1f}ms "
          f"cpm={payload.typing_speed_cpm:.0f}")

    # Auto-train after every sample (background thread — won't block the response)
    training_started = False
    if attempt_num >= 1:
        trigger_training(payload.username)
        training_started = True

    return {
        "success":          True,
        "message":          f"Keystroke attempt #{attempt_num} saved",
        "attempt_number":   attempt_num,
        "training_started": training_started,
        "training_note":    "Model is being trained in the background." if training_started else "",
    }


@router.post("/voice")
def enroll_voice(payload: VoiceEnroll, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Warn if frontend is still sending only 13 features (old behaviour)
    missing_features = (
        not payload.mfcc_std or all(v == 0 for v in payload.mfcc_std)
    ) and payload.pitch_mean == 0 and payload.energy_mean == 0

    if missing_features:
        print(f"⚠  PARTIAL ENROLLMENT for '{payload.username}': "
              "mfcc_std / pitch / energy are all zero — "
              "frontend is still sending only mfcc_features (13 values). "
              "Deploy the fixed speech.js, enroll.js, and api.js.")

    existing_count = db.query(VoiceTemplate).filter(
        VoiceTemplate.user_id == user.id
    ).count()

    template = VoiceTemplate(
        user_id        = user.id,
        attempt_number = existing_count + 1,
        mfcc_features  = payload.mfcc_features,
        mfcc_std       = payload.mfcc_std if payload.mfcc_std else [],
        pitch_mean     = payload.pitch_mean,
        pitch_std      = payload.pitch_std,
        speaking_rate  = payload.speaking_rate,
        energy_mean    = payload.energy_mean,
        energy_std     = payload.energy_std,
        zcr_mean               = payload.zcr_mean,
        spectral_centroid_mean = payload.spectral_centroid_mean,
        spectral_rolloff_mean  = payload.spectral_rolloff_mean,
    )
    db.add(template)
    db.commit()

    attempt_num = existing_count + 1
    print(f"✅ Voice attempt #{attempt_num} for '{payload.username}' | "
          f"pitch={payload.pitch_mean:.1f}Hz  energy={payload.energy_mean:.4f}  "
          f"rate={payload.speaking_rate:.2f}  "
          f"mfcc_std={'✓' if payload.mfcc_std else '✗ MISSING'}")

    # Auto-train after every sample (background thread — won't block the response)
    trigger_voice_training(payload.username)

    return {
        "success":           True,
        "message":           f"Voice attempt #{attempt_num} saved",
        "attempt_number":    attempt_num,
        "has_full_features": not missing_features,
        "training_started":  True,
        "training_note":     "Voice model is being trained in the background.",
    }


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


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO PROCESSING + FULL FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/extract-mfcc")
async def extract_mfcc(payload: AudioData, db: Session = Depends(get_db)):
    """
    Validates audio quality then extracts 34 speaker features:
      - 13 MFCC means + 13 MFCC stds  (spectral shape)
      - pitch mean + std               (fundamental frequency)
      - speaking rate                  (behavioral)
      - energy mean + std              (loudness pattern)
      - ZCR mean                       (voicing)
      - spectral centroid mean         (brightness)
      - spectral rolloff mean          (frequency distribution)
    """
    input_path = None
    wav_path   = None

    try:
        audio_bytes  = base64.b64decode(payload.audio_data)
        audio_format = payload.audio_format or "webm"

        print(f"\n{'='*60}")
        print(f"VOICE FEATURE EXTRACTION PIPELINE")
        print(f"Audio: {len(audio_bytes)} bytes  format: {audio_format}")
        print(f"{'='*60}")

        # Save and convert to WAV
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

        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        total_duration = len(audio) / sr
        print(f"Duration: {total_duration:.2f}s")

        # ── VAD Check ─────────────────────────────────────────────────────
        import webrtcvad
        vad = webrtcvad.Vad(3)

        with open(wav_path, 'rb') as f:
            wav_data = f.read()

        audio_pcm    = wav_data[44:]
        frame_size   = int(sr * 30 / 1000) * 2
        voiced = total = 0

        for i in range(0, len(audio_pcm) - frame_size, frame_size):
            frame = audio_pcm[i:i + frame_size]
            if len(frame) == frame_size:
                total += 1
                if vad.is_speech(frame, sr):
                    voiced += 1

        voice_ratio     = voiced / total if total > 0 else 0
        speech_duration = (voiced * 30) / 1000

        print(f"VAD: {voice_ratio:.0%} voice  {speech_duration:.1f}s speech")

        if voice_ratio < 0.40:
            return {"success": False,
                    "detail": f"Insufficient voice ({voice_ratio:.0%}). Speak clearly."}
        if speech_duration < 1.5:
            return {"success": False,
                    "detail": f"Too short ({speech_duration:.1f}s). Speak the full phrase."}

        # ── Energy check ───────────────────────────────────────────────────
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 0.02:
            return {"success": False,
                    "detail": f"Audio too quiet (RMS={rms:.4f}). Speak louder."}

        # ── Spectral sanity ────────────────────────────────────────────────
        centroid      = librosa.feature.spectral_centroid(y=audio, sr=sr)
        mean_centroid = float(np.mean(centroid))
        if mean_centroid < 600 or mean_centroid > 5000:
            return {"success": False,
                    "detail": f"Audio doesn't sound like speech (centroid={mean_centroid:.0f}Hz)."}

        zcr      = librosa.feature.zero_crossing_rate(audio)
        mean_zcr = float(np.mean(zcr))
        if mean_zcr < 0.02 or mean_zcr > 0.5:
            return {"success": False,
                    "detail": f"Unusual audio characteristics (ZCR={mean_zcr:.4f})."}

        # ── MFCC extraction (13 coefficients) ─────────────────────────────
        mfccs     = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std  = np.std(mfccs, axis=1)

        if np.mean(mfcc_std) < 5.0:
            return {"success": False,
                    "detail": "Audio lacks variation typical of speech."}

        # ── Pitch extraction ───────────────────────────────────────────────
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            voiced_f0  = f0[voiced_flag & ~np.isnan(f0)]
            pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            pitch_std  = float(np.std(voiced_f0))  if len(voiced_f0) > 0 else 0.0
        except Exception:
            pitch_mean = 0.0
            pitch_std  = 0.0

        # ── Speaking rate ──────────────────────────────────────────────────
        hop_length    = 512
        rms_frames    = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        rms_threshold = np.mean(rms_frames) * 0.5
        peaks         = np.where(
            (rms_frames[1:-1] > rms_frames[:-2]) &
            (rms_frames[1:-1] > rms_frames[2:]) &
            (rms_frames[1:-1] > rms_threshold)
        )[0]
        speaking_rate = float(len(peaks) / total_duration) if total_duration > 0 else 0.0

        # ── Energy features ────────────────────────────────────────────────
        energy_frames = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        energy_mean   = float(np.mean(energy_frames))
        energy_std    = float(np.std(energy_frames))

        # ── Spectral rolloff ───────────────────────────────────────────────
        rolloff      = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        rolloff_mean = float(np.mean(rolloff))

        # ── Summary ────────────────────────────────────────────────────────
        print(f"✅ ALL CHECKS PASSED")
        print(f"  MFCC[0] mean={mfcc_mean[0]:.1f}  std_mean={np.mean(mfcc_std):.2f}")
        print(f"  Pitch  mean={pitch_mean:.1f}Hz  std={pitch_std:.1f}Hz")
        print(f"  Rate   {speaking_rate:.2f} peaks/s")
        print(f"  Energy mean={energy_mean:.4f}  std={energy_std:.4f}")
        print(f"  Centroid={mean_centroid:.0f}Hz  Rolloff={rolloff_mean:.0f}Hz")
        print(f"  ZCR={mean_zcr:.4f}")

        return {
            "success":       True,
            "mfcc_features": mfcc_mean.tolist(),
            "mfcc_std":      mfcc_std.tolist(),
            "pitch_mean":    pitch_mean,
            "pitch_std":     pitch_std,
            "speaking_rate": speaking_rate,
            "energy_mean":   energy_mean,
            "energy_std":    energy_std,
            "zcr_mean":      mean_zcr,
            "spectral_centroid_mean": mean_centroid,
            "spectral_rolloff_mean":  rolloff_mean,
            "validation": {
                "voice_ratio":       float(voice_ratio),
                "speech_duration":   float(speech_duration),
                "rms_energy":        rms,
                "spectral_centroid": mean_centroid,
            }
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
        return {"success": False, "detail": str(e)}

    finally:
        for path in [input_path, wav_path]:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except:
                pass


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: GET /debug/voice-count
#  Verify how many DB rows exist and whether features are complete.
#  Usage: GET /api/enroll/debug/voice-count?username=hope
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/debug/voice-count")
def debug_voice_count(username: str = Query(...), db: Session = Depends(get_db)):
    """
    Returns row count + per-row feature summary for a user's voice enrollment.

    Good response (3 complete rows):
    {
        "total_attempts": 3,
        "rows": [
            { "attempt": 1, "mfcc0": -399.5, "pitch": 165.2, "energy": 0.042, "has_std": true },
            { "attempt": 2, "mfcc0": -412.1, "pitch": 171.8, "energy": 0.038, "has_std": true },
            { "attempt": 3, "mfcc0": -388.7, "pitch": 159.4, "energy": 0.051, "has_std": true }
        ],
        "verdict": "✅ Good — all attempts have full features"
    }

    If total_attempts=1 or pitch=0, the frontend fix hasn't been deployed yet.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")

    templates = (
        db.query(VoiceTemplate)
        .filter(VoiceTemplate.user_id == user.id)
        .order_by(VoiceTemplate.attempt_number.asc())
        .all()
    )

    rows = []
    for t in templates:
        mfcc = list(t.mfcc_features or [])
        std  = list(t.mfcc_std or [])
        rows.append({
            "attempt":    t.attempt_number,
            "mfcc0":      round(mfcc[0], 2) if mfcc else None,
            "pitch":      round(float(t.pitch_mean or 0), 2),
            "energy":     round(float(t.energy_mean or 0), 5),
            "zcr":        round(float(t.zcr_mean or 0), 5),
            "has_std":    bool(std and any(v != 0 for v in std)),
            "enrolled_at": str(t.enrolled_at) if hasattr(t, "enrolled_at") else "n/a",
        })

    incomplete = [r for r in rows if not r["has_std"] or r["pitch"] == 0]

    return {
        "username":        username,
        "total_attempts":  len(templates),
        "complete_rows":   len(rows) - len(incomplete),
        "incomplete_rows": len(incomplete),
        "rows":            rows,
        "verdict": (
            "✅ Good — all attempts have full features"
            if not incomplete else
            f"⚠  {len(incomplete)} row(s) missing pitch/std — "
            "re-enroll after deploying the fixed speech.js, enroll.js, api.js"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: POST /re-enroll/clear
#  Wipe all voice rows for a user so they can re-enroll cleanly.
#  Call this ONCE per user after deploying the frontend fixes.
#  Usage: POST /api/enroll/re-enroll/clear
#         Body: { "username": "hope", "confirm": "yes-delete" }
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/re-enroll/clear")
def clear_voice_enrollment(payload: ClearEnrollPayload, db: Session = Depends(get_db)):
    if payload.confirm != "yes-delete":
        raise HTTPException(
            status_code=400,
            detail='Set confirm="yes-delete" to proceed.'
        )

    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    deleted = (
        db.query(VoiceTemplate)
        .filter(VoiceTemplate.user_id == user.id)
        .delete()
    )
    db.commit()

    # Also delete the stale trained model file if present
    from pathlib import Path
    model_path = (
        Path(__file__).parent.parent.parent
        / "ml" / "models"
        / f"{payload.username}_voice_cnn.pkl"
    )
    model_deleted = False
    if model_path.exists():
        model_path.unlink()
        model_deleted = True

    print(f"🗑  Cleared {deleted} voice row(s) for '{payload.username}'")

    return {
        "success":       True,
        "rows_deleted":  deleted,
        "model_deleted": model_deleted,
        "message":       (
            f"Cleared {deleted} enrollment row(s) for '{payload.username}'. "
            "Re-enroll now using the updated frontend."
        ),
    }