# backend/routers/auth.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import hashlib
import pickle
import os

from database.db import get_db
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion, AuthLog

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class KeystrokeAuth(BaseModel):
    username: str
    dwell_times:  List[float]
    flight_times: List[float]
    typing_speed: float = 0

    # Core timing
    dwell_mean:    float = 0
    dwell_std:     float = 0
    dwell_median:  float = 0
    dwell_min:     float = 0
    dwell_max:     float = 0
    flight_mean:   float = 0
    flight_std:    float = 0
    flight_median: float = 0
    p2p_mean:      float = 0
    p2p_std:       float = 0
    r2r_mean:      float = 0
    r2r_std:       float = 0

    # FIX: correct digraphs from "biometric voice keystroke authentication"
    # OLD had: digraph_in, digraph_er, digraph_an, digraph_ed, digraph_to, digraph_it
    # None of these appear in the phrase so they were always 0 at auth time,
    # causing a feature vector mismatch with the trained model every single login.
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

    # FIX: entirely missing from old schema — RF model trained on these features,
    # sending 0 for all of them degraded every prediction.
    shift_lag_mean:   float = 0
    shift_lag_std:    float = 0
    shift_lag_count:  float = 0
    dwell_mean_norm:  float = 0
    dwell_std_norm:   float = 0
    flight_mean_norm: float = 0
    flight_std_norm:  float = 0
    p2p_std_norm:     float = 0
    r2r_mean_norm:    float = 0
    shift_lag_norm:   float = 0


# FIX: old VoiceAuth only had mfcc_features (13 values).
# The trained model uses 34 features — the missing 21 were always 0
# at auth time, which is why every voice was accepted.
class VoiceAuth(BaseModel):
    username:      str
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


class SecurityAuth(BaseModel):
    username: str
    answer:   str


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _model_dir() -> str:
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'ml', 'models'
    ))


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def feature_similarity(enrolled, live, tolerance=0.30):
    if not enrolled or not live or len(enrolled) == 0 or len(live) == 0:
        return 0.0
    enrolled = np.array(enrolled)
    live     = np.array(live)
    min_len  = min(len(enrolled), len(live))
    enrolled = enrolled[:min_len]
    live     = live[:min_len]
    enrolled_safe = np.where(np.abs(enrolled) < 1e-6, 1e-6, enrolled)
    diffs = np.abs(enrolled - live) / (np.abs(enrolled_safe) + 1e-6)
    diffs = np.nan_to_num(diffs, nan=1.0, posinf=1.0, neginf=1.0)
    score = float(np.mean(diffs < tolerance))
    if np.isnan(score) or np.isinf(score):
        return 0.0
    return max(0.0, min(1.0, score))


def log_attempt(db, user_id, method, confidence, result):
    log = AuthLog(
        user_id=user_id,
        auth_method=method,
        confidence_score=float(confidence),
        result=result
    )
    db.add(log)
    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  KEYSTROKE AUTH
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/keystroke")
def verify_keystroke(payload: KeystrokeAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.is_flagged:
        raise HTTPException(status_code=403, detail="Account flagged")

    model_path = os.path.join(_model_dir(), f"{payload.username}_keystroke_rf.pkl")

    if os.path.exists(model_path):
        print(f"[keystroke] Using RF model for '{payload.username}'")
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # FIX: old code used model_data['model'], model_data['scaler'],
            # and model_data['genuine_features'] — none of these keys exist.
            # train_keystroke_rf.py saves: 'pipeline', 'feature_names', 'profile_mean'.
            # The KeyError silently fell through to confidence=0.5 → always denied.
            pipeline     = model_data['pipeline']      # was: model_data['model']
            feat_names   = model_data['feature_names']
            profile_mean = model_data['profile_mean']  # was: model_data['genuine_features']
            threshold    = model_data['threshold']     # use saved EER threshold

            # Build feature vector in the exact order the model was trained on
            vec = np.array([
                float(getattr(payload, name, 0.0) or 0.0)
                for name in feat_names
            ]).reshape(1, -1)

            # Pipeline includes the scaler — no separate scaler.transform() needed
            rf_score = float(pipeline.predict_proba(vec)[0][1])

            # Mahalanobis proximity to enrolled profile
            # (replaces cosine similarity which was always ~0.95 → inflated every score)
            profile_std = model_data.get('profile_std', None)
            if profile_std is not None and len(profile_std) == len(profile_mean):
                safe_std  = np.where(profile_std < 1e-6, 1e-6, profile_std)
                z         = np.abs((vec[0] - profile_mean) / safe_std)
                mah_score = float(1.0 / (1.0 + np.exp(np.mean(z) - 1.5)))
            else:
                diff      = np.linalg.norm(vec[0] - profile_mean)
                scale     = np.linalg.norm(profile_mean) + 1e-9
                mah_score = float(max(0, 1 - diff / scale))

            # Fused: 65% RF + 35% proximity
            confidence = 0.65 * rf_score + 0.35 * mah_score

            # Adaptive threshold based on login history
            login_count = db.query(AuthLog).filter(
                AuthLog.user_id == user.id,
                AuthLog.result == "granted",
                AuthLog.auth_method == "keystroke"
            ).count()

            if login_count < 5:
                effective_threshold = max(threshold * 0.85, 0.30)
                phase = "early"
            elif login_count < 15:
                effective_threshold = threshold
                phase = "growth"
            else:
                effective_threshold = min(threshold * 1.1, 0.80)
                phase = "mature"

            authenticated = confidence >= effective_threshold

            print(f"  RF={rf_score:.3f}  Mah={mah_score:.3f}  "
                  f"Fused={confidence:.3f}  Threshold={effective_threshold:.3f} "
                  f"[{phase}]  → {'PASS' if authenticated else 'FAIL'}")

        except Exception as e:
            print(f"[keystroke] RF model error: {e}")
            import traceback; traceback.print_exc()
            # Hard fail — don't silently grant access on model errors
            confidence    = 0.0
            authenticated = False

    else:
        # No model trained yet — simple dwell-time similarity fallback
        print(f"[keystroke] No RF model for '{payload.username}', using fallback")
        template = db.query(KeystrokeTemplate).filter(
            KeystrokeTemplate.user_id == user.id
        ).first()
        if not template:
            raise HTTPException(status_code=404, detail="No keystroke template found")

        confidence    = feature_similarity(template.dwell_times, payload.dwell_times, 0.35)
        authenticated = confidence >= 0.40

    print(f"[keystroke] user={payload.username}  "
          f"confidence={confidence:.3f}  result={'PASS' if authenticated else 'FAIL'}")

    log_attempt(db, user.id, "keystroke", confidence,
                "granted" if authenticated else "denied")

    if not authenticated:
        total_denials = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.result == "denied"
        ).count()
        if total_denials >= 5:
            user.is_flagged = True
            db.commit()

    return {"authenticated": authenticated, "confidence": float(confidence)}


# ─────────────────────────────────────────────────────────────────────────────
#  VOICE AUTH
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/voice")
def verify_voice(payload: VoiceAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    model_path = os.path.join(_model_dir(), f"{payload.username}_voice_cnn.pkl")

    if os.path.exists(model_path):
        print(f"[voice] Using trained .pkl model for '{payload.username}'")
        try:
            from ml.train_voice_cnn import predict_voice

            # FIX: old code passed only payload.mfcc_features (13 values).
            # predict_voice() needs the full 34-feature dict — the missing 21
            # were always 0, making every voice score near-identical → accept all.
            feature_dict = {
                "mfcc_features":          payload.mfcc_features,
                "mfcc_std":               payload.mfcc_std,
                "pitch_mean":             payload.pitch_mean,
                "pitch_std":              payload.pitch_std,
                "speaking_rate":          payload.speaking_rate,
                "energy_mean":            payload.energy_mean,
                "energy_std":             payload.energy_std,
                "zcr_mean":               payload.zcr_mean,
                "spectral_centroid_mean": payload.spectral_centroid_mean,
                "spectral_rolloff_mean":  payload.spectral_rolloff_mean,
            }

            result = predict_voice(payload.username, feature_dict)

            if "error" in result:
                raise ValueError(result["error"])

            confidence    = result["fused_score"] / 100.0
            authenticated = result["match"]

            print(f"  MLP={result['confidence']:.1f}%  "
                  f"Mah={result['mahalanobis']:.1f}%  "
                  f"Fused={result['fused_score']:.1f}%  "
                  f"Threshold={result['threshold']:.1f}%  "
                  f"→ {'PASS' if authenticated else 'FAIL'}")

        except Exception as e:
            print(f"[voice] Model error: {e}")
            import traceback; traceback.print_exc()
            confidence    = 0.0
            authenticated = False

    else:
        # No trained model — cosine similarity fallback on MFCC means only
        print(f"[voice] No trained model for '{payload.username}', using cosine fallback")
        template = db.query(VoiceTemplate).filter(
            VoiceTemplate.user_id == user.id
        ).first()
        if not template:
            raise HTTPException(status_code=404, detail="No voice template found")

        a = np.array(template.mfcc_features or [])
        b = np.array(payload.mfcc_features  or [])
        if len(a) > 0 and len(b) > 0:
            min_len = min(len(a), len(b))
            a, b    = a[:min_len], b[:min_len]
            denom   = np.linalg.norm(a) * np.linalg.norm(b)
            cos_sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0
            confidence = (cos_sim + 1) / 2
        else:
            confidence = 0.0

        authenticated = confidence >= 0.75  # higher bar for raw cosine fallback

    print(f"[voice] user={payload.username}  "
          f"confidence={confidence:.3f}  result={'PASS' if authenticated else 'FAIL'}")

    log_attempt(db, user.id, "voice", confidence,
                "granted" if authenticated else "denied")

    return {"authenticated": authenticated, "confidence": float(confidence)}


# ─────────────────────────────────────────────────────────────────────────────
#  SECURITY QUESTION AUTH
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/security-question/{username}")
def get_security_question(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    sq = db.query(SecurityQuestion).filter(SecurityQuestion.user_id == user.id).first()
    if not sq:
        raise HTTPException(status_code=404, detail="No security question found")
    return {"question": sq.question}


@router.post("/security")
def verify_security(payload: SecurityAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sq = db.query(SecurityQuestion).filter(SecurityQuestion.user_id == user.id).first()
    if not sq:
        raise HTTPException(status_code=404, detail="No security question found")

    answer_hash   = hashlib.sha256(payload.answer.strip().lower().encode()).hexdigest()
    authenticated = answer_hash == sq.answer_hash

    print(f"[security] '{payload.username}' → {'PASS' if authenticated else 'FAIL'}")

    log_attempt(db, user.id, "security_question",
                1.0 if authenticated else 0.0,
                "granted" if authenticated else "denied")

    if not authenticated:
        user.is_flagged = True
        db.commit()
        print(f"  ⚠️ User '{payload.username}' flagged!")

    return {
        "authenticated": authenticated,
        "confidence":    1.0 if authenticated else 0.0
    }