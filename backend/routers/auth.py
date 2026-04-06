# backend/routers/auth.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import hashlib
import pickle
import os
import sys
import subprocess
import bcrypt  # ← ADDED

from utils.fusion import fuse_keystroke_scores, fuse_voice_scores, fuse_multimodal

from database.db import get_db
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion, AuthLog
from schemas import VoiceFeatures

router = APIRouter()

MAX_SAMPLES = 50

def _safe_filename(username: str) -> str:
    """Sanitize email for use as filename. user@gmail.com → user_at_gmail_com"""
    return username.replace("@", "_at_").replace(".", "_").replace(" ", "_")

# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

# ← ADDED
class PasswordAuth(BaseModel):
    username: str
    password: str


class KeystrokeAuth(BaseModel):
    username: str
    dwell_times:  List[float]
    flight_times: List[float]
    typing_speed: float = 0
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
    # FIX: phrase-specific bigram timings — sent as a dict from keystroke.js.
    # The training script strips inactive hardcoded digraphs and appends these
    # as extra_XX features. Without this field they were silently zeroed at
    # auth time, making the login vector look nothing like the enrolled vector.
    extra_digraphs:   Optional[dict] = {}


class VoiceAuth(VoiceFeatures):
    username: str


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
#  PASSWORD AUTH  ← ADDED
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/password")
def verify_password(payload: PasswordAuth, db: Session = Depends(get_db)):
    # Normalise email — lowercase so login works regardless of capitalisation
    email = payload.username.strip().lower()
    user = db.query(User).filter(User.username == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.is_flagged:
        raise HTTPException(status_code=403, detail="Account flagged")
    if not user.password_hash:
        raise HTTPException(status_code=400, detail="No password set for this user")

    authenticated = bcrypt.checkpw(
        payload.password.encode('utf-8'),
        user.password_hash.encode('utf-8')
    )

    print(f"[password] '{email}' → {'PASS' if authenticated else 'FAIL'}")

    log_attempt(db, user.id, "password",
                1.0 if authenticated else 0.0,
                "granted" if authenticated else "denied")

    if not authenticated:
        pw_denials = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.auth_method == "password",
            AuthLog.result == "denied"
        ).count()
        if pw_denials >= 5:
            user.is_flagged = True
            db.commit()

    return {
        "authenticated": bool(authenticated),
        "confidence":    1.0 if authenticated else 0.0
    }


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

    model_path = os.path.join(_model_dir(), f"{_safe_filename(payload.username)}_keystroke_rf.pkl")

    if os.path.exists(model_path):
        print(f"[keystroke] Using RF model for '{payload.username}'")
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            pipeline     = model_data['pipeline']
            feat_names   = model_data['feature_names']
            profile_mean = model_data['profile_mean']
            threshold    = model_data['threshold']

            # FIX: features whose names start with "extra_" were produced by
            # the phrase-aware digraph filter in train_keystroke_rf.py.
            # They must be looked up from payload.extra_digraphs (a dict keyed
            # by the 2-letter bigram), NOT via getattr which always returns 0.0.
            # Zeroing every extra_* feature was the primary reason keystroke
            # auth always failed for users with unique passphrases.
            extra_map = payload.extra_digraphs or {}
            vec = np.array([
                float(extra_map.get(name[6:], 0.0) or 0.0)
                if name.startswith("extra_")
                else float(getattr(payload, name, 0.0) or 0.0)
                for name in feat_names
            ]).reshape(1, -1)

            rf_score = float(pipeline.predict_proba(vec)[0][1])

            profile_std = model_data.get('profile_std', None)
            if profile_std is not None and len(profile_std) == len(profile_mean):
                var       = profile_std ** 2
                safe_var  = np.where(var < 1e-10, 1e-10, var)
                diff      = vec[0] - profile_mean
                d_sq      = float(np.sum(diff ** 2 / safe_var))
                d_sq_norm = d_sq / max(len(vec[0]), 1)
                # Clamp exponent to prevent overflow (np.exp overflows at ~710)
                exponent  = float(np.clip(2.5 * (d_sq_norm - 1.0), -500, 500))
                mah_score = float(1.0 / (1.0 + np.exp(exponent)))
            else:
                diff      = np.linalg.norm(vec[0] - profile_mean)
                scale     = np.linalg.norm(profile_mean) + 1e-9
                mah_score = float(max(0, 1 - diff / scale))

            # ── Hard sanity gates ─────────────────────────────────────────────
            # These catch cases where the RF is overconfident but raw timing
            # numbers are obviously wrong for this user.  Gates fire BEFORE the
            # fused score so RF=1.0 cannot override clearly anomalous typing.
            hard_reject   = False
            reject_reason = ""

            fn_list = list(feat_names)
            def _fv(name):
                try:
                    return float(vec[0][fn_list.index(name)])
                except (ValueError, IndexError):
                    return None

            # Gate 1: dwell_mean z-score (how long keys are held)
            if not hard_reject and profile_std is not None and 'dwell_mean' in fn_list:
                e_dwell     = float(profile_mean[fn_list.index('dwell_mean')])
                e_dwell_std = float(profile_std[fn_list.index('dwell_mean')]) + 1e-9
                l_dwell     = _fv('dwell_mean')
                if l_dwell is not None:
                    dwell_z = abs(l_dwell - e_dwell) / e_dwell_std
                    # FIX: tightened from 4.5 → 3.0 z-score.
                    # 4.5 std is extreme — only fires if the impostor types
                    # nearly 5x slower/faster than the genuine user. At 3.0,
                    # groupmates who differ by 3 standard deviations (a clearly
                    # different typing rhythm) are hard-rejected immediately.
                    if dwell_z > 3.0:
                        hard_reject   = True
                        reject_reason = (f"dwell_mean z={dwell_z:.1f} "
                                         f"(live={l_dwell:.0f}ms enrolled={e_dwell:.0f}ms)")

            # Gate 2: typing_speed_cpm z-score
            if not hard_reject and profile_std is not None and 'typing_speed_cpm' in fn_list:
                e_cpm     = float(profile_mean[fn_list.index('typing_speed_cpm')])
                e_cpm_std = float(profile_std[fn_list.index('typing_speed_cpm')]) + 1e-9
                l_cpm     = _fv('typing_speed_cpm')
                if l_cpm is not None:
                    cpm_z = abs(l_cpm - e_cpm) / e_cpm_std
                    if cpm_z > 3.0:
                        hard_reject   = True
                        reject_reason = (f"typing_speed_cpm z={cpm_z:.1f} "
                                         f"(live={l_cpm:.0f} enrolled={e_cpm:.0f})")

            # Gate 3: Mahalanobis hard floor.
            # FIX: raised from 0.05 → 0.15. mah_score < 0.05 only fired for
            # extreme outliers (d_sq_norm >> 3). Groupmates who type similarly
            # land at mah_score ~0.10–0.30; raising the floor catches them
            # before the RF score can override.
            if not hard_reject and mah_score < 0.15:
                hard_reject   = True
                reject_reason = (f"Mahalanobis floor breach "
                                 f"(mah={mah_score:.4f}, d_sq_norm={d_sq_norm:.2f})")

            if hard_reject:
                confidence    = 0.0
                authenticated = False
                print(f"  \u26d4 Hard reject: {reject_reason}")
            else:
                # Intra-modal fusion: RF (75%) + Mahalanobis (25%).
                # Weights are defined in utils/fusion.py — single source of truth.
                confidence    = fuse_keystroke_scores(rf_score, mah_score)
                authenticated = confidence >= threshold

            print(f"  RF={rf_score:.3f}  Mah={mah_score:.3f}  "
                  f"Fused={confidence:.3f}  Threshold={threshold:.3f}"
                  f"  \u2192 {'PASS' if authenticated else 'FAIL'}")

        except Exception as e:
            print(f"[keystroke] RF model error: {e}")
            import traceback; traceback.print_exc()
            confidence    = 0.0
            authenticated = False

    else:
        # No trained model — hard reject.
        # The old dwell-time similarity fallback (threshold 0.40) was too
        # loose and granted access to anyone with a vaguely similar typing
        # speed.  Until the RF model is trained, deny all keystroke attempts
        # so the user falls through to voice → security question.
        print(f"[keystroke] No RF model for '{payload.username}' — hard reject until model is trained")
        confidence    = 0.0
        authenticated = False

    print(f"[keystroke] user={payload.username}  "
          f"confidence={confidence:.3f}  result={'PASS' if authenticated else 'FAIL'}")

    log_attempt(db, user.id, "keystroke", confidence,
                "granted" if authenticated else "denied")

    # ═════════════════════════════════════════════════════════════════
    # ADAPTIVE LEARNING: Save login sample + Auto-retrain
    # Only save high-confidence matches — borderline passes risk saving
    # an impostor's sample as genuine data, corrupting the model.
    # ═════════════════════════════════════════════════════════════════
    SAVE_CONFIDENCE_MINIMUM = 0.70

    if authenticated and confidence >= SAVE_CONFIDENCE_MINIMUM:
        # Get current sample count
        total_samples = db.query(KeystrokeTemplate).filter(
            KeystrokeTemplate.user_id == user.id
        ).count()

        # Delete oldest if at limit (rolling window)
        if total_samples >= MAX_SAMPLES:
            oldest = db.query(KeystrokeTemplate).filter(
                KeystrokeTemplate.user_id == user.id
            ).order_by(KeystrokeTemplate.sample_order.asc()).first()
            if oldest:
                print(f"  🗑️ Deleting oldest sample (order={oldest.sample_order})")
                db.delete(oldest)
                db.flush()

        # Get next order number
        max_order = db.query(func.max(KeystrokeTemplate.sample_order)).filter(
            KeystrokeTemplate.user_id == user.id
        ).scalar() or 0

        # Save new login sample with all features
        new_sample = KeystrokeTemplate(
            user_id=user.id,
            attempt_number=min(total_samples + 1, MAX_SAMPLES),
            source="login",
            sample_order=max_order + 1,
            dwell_times=payload.dwell_times,
            flight_times=payload.flight_times,
            typing_speed=payload.typing_speed,
            dwell_mean=payload.dwell_mean,
            dwell_std=payload.dwell_std,
            dwell_median=payload.dwell_median,
            dwell_min=payload.dwell_min,
            dwell_max=payload.dwell_max,
            flight_mean=payload.flight_mean,
            flight_std=payload.flight_std,
            flight_median=payload.flight_median,
            p2p_mean=payload.p2p_mean,
            p2p_std=payload.p2p_std,
            r2r_mean=payload.r2r_mean,
            r2r_std=payload.r2r_std,
            digraph_th=payload.digraph_th,
            digraph_he=payload.digraph_he,
            digraph_bi=payload.digraph_bi,
            digraph_io=payload.digraph_io,
            digraph_om=payload.digraph_om,
            digraph_me=payload.digraph_me,
            digraph_et=payload.digraph_et,
            digraph_tr=payload.digraph_tr,
            digraph_ri=payload.digraph_ri,
            digraph_ic=payload.digraph_ic,
            digraph_vo=payload.digraph_vo,
            digraph_oi=payload.digraph_oi,
            digraph_ce=payload.digraph_ce,
            digraph_ke=payload.digraph_ke,
            digraph_ey=payload.digraph_ey,
            digraph_ys=payload.digraph_ys,
            digraph_st=payload.digraph_st,
            digraph_ro=payload.digraph_ro,
            digraph_ok=payload.digraph_ok,
            digraph_au=payload.digraph_au,
            digraph_ut=payload.digraph_ut,
            digraph_en=payload.digraph_en,
            digraph_nt=payload.digraph_nt,
            digraph_ti=payload.digraph_ti,
            digraph_ca=payload.digraph_ca,
            digraph_at=payload.digraph_at,
            digraph_on=payload.digraph_on,
            typing_speed_cpm=payload.typing_speed_cpm,
            typing_duration=payload.typing_duration,
            rhythm_mean=payload.rhythm_mean,
            rhythm_std=payload.rhythm_std,
            rhythm_cv=payload.rhythm_cv,
            pause_count=payload.pause_count,
            pause_mean=payload.pause_mean,
            backspace_ratio=payload.backspace_ratio,
            backspace_count=payload.backspace_count,
            hand_alternation_ratio=payload.hand_alternation_ratio,
            same_hand_sequence_mean=payload.same_hand_sequence_mean,
            finger_transition_ratio=payload.finger_transition_ratio,
            seek_time_mean=payload.seek_time_mean,
            seek_time_count=payload.seek_time_count,
            shift_lag_mean=payload.shift_lag_mean,
            shift_lag_std=payload.shift_lag_std,
            shift_lag_count=payload.shift_lag_count,
            dwell_mean_norm=payload.dwell_mean_norm,
            dwell_std_norm=payload.dwell_std_norm,
            flight_mean_norm=payload.flight_mean_norm,
            flight_std_norm=payload.flight_std_norm,
            p2p_std_norm=payload.p2p_std_norm,
            r2r_mean_norm=payload.r2r_mean_norm,
            shift_lag_norm=payload.shift_lag_norm,
            # FIX: persist phrase-specific bigram timings so adaptive
            # retraining can load them via extract_feature_vector(extra_keys=...)
            extra_digraphs=payload.extra_digraphs or {},
        )
        db.add(new_sample)
        db.commit()

        updated_count = db.query(KeystrokeTemplate).filter(
            KeystrokeTemplate.user_id == user.id
        ).count()

        print(f"  💾 Saved login sample (total: {updated_count}/{MAX_SAMPLES})")

        # Determine adaptive retrain interval
        login_count_total = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.auth_method == "keystroke",
            AuthLog.result == "granted"
        ).count()

        if updated_count <= 10:
            retrain_interval = 2   # Every 2 logins (fast bootstrap)
        elif updated_count <= 30:
            retrain_interval = 5   # Every 5 logins (learning phase)
        else:
            retrain_interval = 10  # Every 10 logins (mature phase)

        should_retrain = False
        retrain_reason = ""

        # Check interval-based retraining
        if login_count_total % retrain_interval == 0:
            should_retrain = True
            retrain_reason = f"interval ({retrain_interval} logins)"

        # Also retrain at key milestones
        milestones = [10, 20, 30, 40, 50]
        if updated_count in milestones:
            should_retrain = True
            retrain_reason = f"milestone ({updated_count} samples)"

        if should_retrain:
            print(f"  🔄 Triggering retrain: {retrain_reason}")
            try:
                script_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'ml', 'train_keystroke_rf.py'
                )
                subprocess.Popen(
                    [sys.executable, script_path, payload.username],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                print(f"  ✅ Retraining started in background")
            except Exception as e:
                print(f"  ⚠️ Retrain failed: {e}")

    # Handle failed attempts — count keystroke method only
    if not authenticated:
        ks_denials = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.auth_method == "keystroke",
            AuthLog.result == "denied"
        ).count()
        if ks_denials >= 10:
            user.is_flagged = True
            db.commit()

    return {"authenticated": bool(authenticated), "confidence": float(confidence)}


# ─────────────────────────────────────────────────────────────────────────────
#  VOICE AUTH
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/voice")
def verify_voice(payload: VoiceAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    model_path = os.path.join(_model_dir(), f"{_safe_filename(payload.username)}_voice_cnn.pkl")

    # ── PRIMARY AUTH: ECAPA-TDNN pretrained speaker verification ────────────
    # Works after the first enrollment recording. No pkl model file needed —
    # just the profile built by save_enrollment() during voice enrollment.
    try:
        project_root = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..'
        ))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from ml.voice_ecapa import predict_voice as ecapa_predict

        ecapa_result  = ecapa_predict(
            payload.username,
            getattr(payload, 'ecapa_embedding', [])
        )
        confidence    = ecapa_result["similarity"]
        authenticated = ecapa_result["match"]

        print(f"  [ECAPA] similarity={ecapa_result['similarity']:.4f}  "
              f"threshold={ecapa_result['threshold']:.2f}  "
              f"n_enrolled={ecapa_result.get('n_enrollment', '?')}  "
              f"→ {'PASS' if authenticated else 'FAIL'}")

        if "error" in ecapa_result:
            print(f"  [ECAPA] note: {ecapa_result['error']}")

    except Exception as e:
        print(f"[voice] ECAPA error: {e}")
        import traceback; traceback.print_exc()
        confidence    = 0.0
        authenticated = False

    # ── SECONDARY (logged only — not used for decision) ───────────────────
    # CNN and Resemblyzer still run so you can compare scores in the console.
    # Their results do NOT affect whether the user is granted access.
    try:
        from ml.train_voice_cnn import predict_voice as cnn_predict
        if os.path.exists(model_path):
            feature_dict = {
                "mfcc_features":          payload.mfcc_features,
                "mfcc_std":               payload.mfcc_std,
                "delta_mfcc_mean":        payload.delta_mfcc_mean,
                "delta2_mfcc_mean":       payload.delta2_mfcc_mean,
                "pitch_mean":             payload.pitch_mean,
                "pitch_std":              payload.pitch_std,
                "speaking_rate":          payload.speaking_rate,
                "energy_mean":            payload.energy_mean,
                "energy_std":             payload.energy_std,
                "zcr_mean":               payload.zcr_mean,
                "spectral_centroid_mean": payload.spectral_centroid_mean,
                "spectral_rolloff_mean":  payload.spectral_rolloff_mean,
                "spectral_flux_mean":     payload.spectral_flux_mean,
                "voiced_fraction":        payload.voiced_fraction,
            }
            cnn_r = cnn_predict(payload.username, feature_dict)
            print(f"  [CNN log-only] fused={cnn_r.get('fused_score', 0):.1f}%")
        else:
            print(f"  [CNN log-only] no model yet for '{payload.username}'")
    except Exception as _e:
        print(f"  [CNN log-only] skipped — {_e}")

    try:
        from ml.voice_resemblyzer import predict_voice as resem_predict
        resem_r = resem_predict(payload.username, getattr(payload, 'resemblyzer_embedding', []))
        print(f"  [Resemblyzer log-only] similarity={resem_r.get('similarity', 0):.4f}")
    except Exception as _e:
        print(f"  [Resemblyzer log-only] skipped — {_e}")

    print(f"[voice] user={payload.username}  "
          f"confidence={confidence:.3f}  result={'PASS' if authenticated else 'FAIL'}")

    log_attempt(db, user.id, "voice", confidence,
                "granted" if authenticated else "denied")

    # Handle failed voice attempts — count voice method only to avoid cross-method false lockout
    if not authenticated:
        voice_denials = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.auth_method == "voice",
            AuthLog.result == "denied"
        ).count()
        if voice_denials >= 10:
            user.is_flagged = True
            db.commit()
            print(f"  ⚠️ User '{payload.username}' flagged after {voice_denials} voice failures")

    # ── Adaptive learning: save login sample and retrain periodically ─────
    if authenticated:
        MAX_VOICE_SAMPLES = 50

        total_voice = db.query(VoiceTemplate).filter(
            VoiceTemplate.user_id == user.id
        ).count()

        # Rolling window — delete oldest if at limit
        if total_voice >= MAX_VOICE_SAMPLES:
            oldest = db.query(VoiceTemplate).filter(
                VoiceTemplate.user_id == user.id
            ).order_by(VoiceTemplate.attempt_number.asc()).first()
            if oldest:
                db.delete(oldest)
                db.flush()

        # Save this login attempt as a new voice sample (with ALL v2 fields)
        new_voice = VoiceTemplate(
            user_id        = user.id,
            attempt_number = min(total_voice + 1, MAX_VOICE_SAMPLES),
            mfcc_features  = payload.mfcc_features,
            mfcc_std       = payload.mfcc_std,
            pitch_mean     = payload.pitch_mean,
            pitch_std      = payload.pitch_std,
            speaking_rate  = payload.speaking_rate,
            energy_mean    = payload.energy_mean,
            energy_std     = payload.energy_std,
            zcr_mean               = payload.zcr_mean,
            spectral_centroid_mean = payload.spectral_centroid_mean,
            spectral_rolloff_mean  = payload.spectral_rolloff_mean,
            delta_mfcc_mean        = payload.delta_mfcc_mean,    # v2
            delta2_mfcc_mean       = payload.delta2_mfcc_mean,   # v2
            spectral_flux_mean     = payload.spectral_flux_mean, # v2
            voiced_fraction        = payload.voiced_fraction,    # v2
            snr_db                 = payload.snr_db,             # v2
        )
        db.add(new_voice)
        db.commit()

        updated_voice_count = db.query(VoiceTemplate).filter(
            VoiceTemplate.user_id == user.id
        ).count()

        print(f"  💾 Saved voice login sample (total: {updated_voice_count}/{MAX_VOICE_SAMPLES})")

        # Adaptive retrain intervals
        successful_voice_logins = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.auth_method == "voice",
            AuthLog.result == "granted"
        ).count()

        if updated_voice_count <= 10:
            retrain_interval = 2
        elif updated_voice_count <= 30:
            retrain_interval = 5
        else:
            retrain_interval = 10

        voice_milestones = [10, 20, 30, 40, 50]
        should_retrain = (
            successful_voice_logins % retrain_interval == 0 or
            updated_voice_count in voice_milestones
        )

        if should_retrain:
            retrain_reason = (
                f"milestone ({updated_voice_count} samples)"
                if updated_voice_count in voice_milestones
                else f"interval ({retrain_interval} logins)"
            )
            print(f"  🔄 Triggering voice retrain: {retrain_reason}")
            try:
                script_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'ml', 'train_voice_cnn.py'
                )
                lock_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'ml', 'models', f"{_safe_filename(payload.username)}_voice_cnn.pkl.retraining"
                )
                # Skip if another retrain is already running for this user
                if not os.path.exists(lock_path):
                    open(lock_path, 'w').close()
                    subprocess.Popen(
                        [sys.executable, script_path, payload.username,
                         f"--lock={lock_path}"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    )
                    print(f"  ✅ Voice retraining started in background")
                else:
                    print(f"  ⏭  Voice retrain skipped — already running")
            except Exception as e:
                print(f"  ⚠️ Voice retrain failed to start: {e}")

    return {"authenticated": bool(authenticated), "confidence": float(confidence)}


# ─────────────────────────────────────────────────────────────────────────────
#  SECURITY QUESTION AUTH
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/phrase/{email}")
def get_phrase(email: str, db: Session = Depends(get_db)):
    """Return the unique passphrase assigned to this user.
    Called by login page so it can display the correct phrase to type/speak."""
    normalised = email.strip().lower()
    user = db.query(User).filter(User.username == normalised).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.phrase:
        raise HTTPException(status_code=404, detail="No phrase assigned to this user")
    return {"phrase": user.phrase}

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
        # Require 3 failed security answers before flagging — one typo shouldn't lock an account
        sq_denials = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.auth_method == "security_question",
            AuthLog.result == "denied"
        ).count()
        if sq_denials >= 3:
            user.is_flagged = True
            db.commit()
            print(f"  ⚠️ User '{payload.username}' flagged after {sq_denials} security Q failures")

    return {
        "authenticated": authenticated,
        "confidence":    1.0 if authenticated else 0.0
    }