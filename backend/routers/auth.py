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

from utils.fusion import fuse_keystroke_scores
from utils.crypto import decrypt
from utils.debug_logger import log_auth_stage, log_error

# ── Pending keystroke sample cache ────────────────────────────────────────────
# Keystroke samples are only saved after the full multimodal login is granted.
# This prevents polluting training data with samples from sessions where voice
# later failed (the user would not have been authenticated).
# Key: username (str)  Value: KeystrokeAuth payload
_pending_ks_samples: dict = {}

from database.db import get_db
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion, AuthLog
from schemas import VoiceFeatures

router = APIRouter()

MAX_SAMPLES = 50
# Adaptive pool cap (TypingDNA-style). Enrollment rows are pinned and do NOT
# count against this — only source in ('login_ks','login_fusion') are adaptive.
# When the cap is hit, the lowest-scoring adaptive row is evicted so the pool
# self-cleans toward the user's strongest patterns instead of drifting with age.
MAX_ADAPTIVE = 30

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
    key_dwell_map:    Optional[dict] = {}

    # 4-variant digraph timings + trigraphs (frontend keystroke.js v3.1+).
    # Currently received but not used by the matcher — wire in once the
    # trainer's extract_feature_vector accepts the new dict-based features.
    digraph_dd_map:     Optional[dict] = {}
    digraph_du_map:     Optional[dict] = {}
    digraph_ud_map:     Optional[dict] = {}
    digraph_uu_map:     Optional[dict] = {}
    flight_per_digraph: Optional[dict] = {}
    trigraph_map:       Optional[dict] = {}


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


def log_attempt(db, user_id, method, confidence, result,
                template_maturity=None, effective_threshold=None):
    log = AuthLog(
        user_id=user_id,
        auth_method=method,
        confidence_score=float(confidence),
        result=result,
        template_maturity=template_maturity,
        effective_threshold=effective_threshold,
    )
    db.add(log)
    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
#  Progressive-enrollment policy
#
#  A freshly-enrolled template built from 5 upfront samples is noisy. Rather
#  than reject the genuine user at a hardened threshold from day 1, we start
#  keystroke at a relaxed threshold and harden it as the template stabilizes
#  through real-world samples. Security is preserved because the overall login
#  still requires password AND voice — joint FAR stays far below 10⁻⁴.
#
#  Maturity = count of successful multi-factor-verified logins since enroll.
#    maturity < 3  →  soft    threshold 0.45
#    3 ≤ m < 7    →  ramp    threshold 0.55
#    maturity ≥ 7  →  hardened (stored threshold, typically 0.65)
# ─────────────────────────────────────────────────────────────────────────────

_SOFT_THRESHOLD = 0.45
_RAMP_THRESHOLD = 0.55
_SOFT_MATURITY_END = 3   # below → soft
_RAMP_MATURITY_END = 7   # below → ramp; at/above → hardened


def _keystroke_maturity(db, user_id: int) -> int:
    """
    Count completed logins that verified this user's identity end-to-end —
    either keystroke alone granted access, or fusion (voice recovery) granted.
    Both imply the full multi-factor pipeline accepted the user.
    """
    return db.query(AuthLog).filter(
        AuthLog.user_id == user_id,
        AuthLog.auth_method.in_(("keystroke", "fusion")),
        AuthLog.result == "granted",
    ).count()


def _effective_keystroke_threshold(stored_threshold: float, maturity: int) -> float:
    if maturity < _SOFT_MATURITY_END:
        return _SOFT_THRESHOLD
    if maturity < _RAMP_MATURITY_END:
        return _RAMP_THRESHOLD
    # Gradual hardening from RAMP → stored threshold over maturity 7..15.
    # Avoids the sudden FRR cliff that rejected genuine users whose typing
    # sits in the 0.55–0.70 band. Template keeps absorbing adaptive samples
    # while the threshold tightens slowly toward the calibrated value.
    HARDEN_SPAN = 8
    progress = min(1.0, (maturity - _RAMP_MATURITY_END) / HARDEN_SPAN)
    return _RAMP_THRESHOLD + progress * (float(stored_threshold) - _RAMP_THRESHOLD)


def _personalize_threshold(base: float, s_mean: float, s_std: float,
                           is_inconsistent: bool) -> float:
    # TypingDNA-style per-user calibration: for users whose LOO self-check shows
    # high variance or a low mean, relax the bar toward `s_mean - 2σ` so their
    # own genuine lower tail fits above threshold. Never drops below SOFT and
    # never raises above the base — we only loosen for inconsistent typists.
    if not is_inconsistent or s_std <= 0:
        return base
    personalized = max(_SOFT_THRESHOLD, s_mean - 2.0 * s_std)
    return min(base, personalized)


def _consecutive_denials(db: Session, user_id: int, method: str) -> int:
    # Count denials since the user's last granted attempt for this method.
    # Why: the old all-time count permanently flagged accounts once lifetime
    # typos hit the limit. Consecutive-since-last-success resets on every win.
    last_grant = (
        db.query(AuthLog.attempted_at)
          .filter(AuthLog.user_id == user_id,
                  AuthLog.auth_method == method,
                  AuthLog.result == "granted")
          .order_by(AuthLog.attempted_at.desc())
          .first()
    )
    q = db.query(AuthLog).filter(
        AuthLog.user_id == user_id,
        AuthLog.auth_method == method,
        AuthLog.result == "denied",
    )
    if last_grant is not None:
        q = q.filter(AuthLog.attempted_at > last_grant[0])
    return q.count()


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

    log_auth_stage(
        "password", email,
        "granted" if authenticated else "denied",
        score=1.0 if authenticated else 0.0,
        threshold=1.0,
    )

    return {
        "authenticated": bool(authenticated),
        "confidence":    1.0 if authenticated else 0.0
    }


# ─────────────────────────────────────────────────────────────────────────────
#  KEYSTROKE SAMPLE SAVE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _save_keystroke_sample(
    db: Session,
    user,
    ks_payload,
    source: str = "login",
    ks_score: float | None = None,
    effective_threshold: float | None = None,
):
    """
    Persist a keystroke sample to the database and trigger adaptive retraining
    if a milestone or interval is reached.  Called from two places:
      - verify_keystroke  : when keystroke alone grants access
      - fuse_scores       : when voice recovery + fusion grants access
    """
    # TypingDNA-style save gate: the auth threshold IS the quality bar.
    # A second stricter gate locks out inconsistent typists whose genuine
    # scores cluster near threshold — the adaptive loop never learns their
    # variance. If a sample passed auth, it's genuine enough to absorb.
    # For fusion saves (voice recovered a keystroke-fail), require the sample
    # to have come close to the auth threshold on its own — otherwise voice
    # would drag the keystroke pool off-profile.
    if ks_score is not None:
        thr = float(effective_threshold) if effective_threshold is not None else 0.50
        if source == "login_ks":
            quality_bar = thr
        elif source == "login_fusion":
            quality_bar = thr
        else:
            quality_bar = 0.0
        if ks_score < quality_bar:
            print(f"  🚫 Keystroke sample not saved [{source}] — score {ks_score:.3f} < quality bar {quality_bar:.3f}")
            return

    try:
        total_samples = db.query(KeystrokeTemplate).filter(
            KeystrokeTemplate.user_id == user.id
        ).count()

        # TypingDNA-style FIFO eviction over the ADAPTIVE pool only. Enrollment
        # rows are pinned. When the cap is hit, drop the OLDEST adaptive row —
        # not the weakest — so the pool preserves diversity and drifts with how
        # the user currently types. A strict-monotonic "must beat weakest" gate
        # locks out inconsistent genuine typists once a few high-scoring samples
        # seed the pool; FIFO avoids that failure mode.
        adaptive_q = db.query(KeystrokeTemplate).filter(
            KeystrokeTemplate.user_id == user.id,
            KeystrokeTemplate.source.in_(["login_ks", "login_fusion"]),
        )
        adaptive_count = adaptive_q.count()
        if adaptive_count >= MAX_ADAPTIVE:
            oldest = adaptive_q.order_by(
                KeystrokeTemplate.sample_order.asc()
            ).first()
            if oldest:
                db.delete(oldest)
                db.flush()

        max_order = db.query(func.max(KeystrokeTemplate.sample_order)).filter(
            KeystrokeTemplate.user_id == user.id
        ).scalar() or 0

        new_sample = KeystrokeTemplate(
            user_id=user.id,
            attempt_number=min(total_samples + 1, MAX_SAMPLES),
            source=source,
            sample_order=max_order + 1,
            dwell_times=ks_payload.dwell_times,
            flight_times=ks_payload.flight_times,
            typing_speed=ks_payload.typing_speed,
            dwell_mean=ks_payload.dwell_mean,
            dwell_std=ks_payload.dwell_std,
            dwell_median=ks_payload.dwell_median,
            dwell_min=ks_payload.dwell_min,
            dwell_max=ks_payload.dwell_max,
            flight_mean=ks_payload.flight_mean,
            flight_std=ks_payload.flight_std,
            flight_median=ks_payload.flight_median,
            p2p_mean=ks_payload.p2p_mean,
            p2p_std=ks_payload.p2p_std,
            r2r_mean=ks_payload.r2r_mean,
            r2r_std=ks_payload.r2r_std,
            digraph_th=ks_payload.digraph_th,
            digraph_he=ks_payload.digraph_he,
            digraph_bi=ks_payload.digraph_bi,
            digraph_io=ks_payload.digraph_io,
            digraph_om=ks_payload.digraph_om,
            digraph_me=ks_payload.digraph_me,
            digraph_et=ks_payload.digraph_et,
            digraph_tr=ks_payload.digraph_tr,
            digraph_ri=ks_payload.digraph_ri,
            digraph_ic=ks_payload.digraph_ic,
            digraph_vo=ks_payload.digraph_vo,
            digraph_oi=ks_payload.digraph_oi,
            digraph_ce=ks_payload.digraph_ce,
            digraph_ke=ks_payload.digraph_ke,
            digraph_ey=ks_payload.digraph_ey,
            digraph_ys=ks_payload.digraph_ys,
            digraph_st=ks_payload.digraph_st,
            digraph_ro=ks_payload.digraph_ro,
            digraph_ok=ks_payload.digraph_ok,
            digraph_au=ks_payload.digraph_au,
            digraph_ut=ks_payload.digraph_ut,
            digraph_en=ks_payload.digraph_en,
            digraph_nt=ks_payload.digraph_nt,
            digraph_ti=ks_payload.digraph_ti,
            digraph_ca=ks_payload.digraph_ca,
            digraph_at=ks_payload.digraph_at,
            digraph_on=ks_payload.digraph_on,
            typing_speed_cpm=ks_payload.typing_speed_cpm,
            typing_duration=ks_payload.typing_duration,
            rhythm_mean=ks_payload.rhythm_mean,
            rhythm_std=ks_payload.rhythm_std,
            rhythm_cv=ks_payload.rhythm_cv,
            pause_count=ks_payload.pause_count,
            pause_mean=ks_payload.pause_mean,
            backspace_ratio=ks_payload.backspace_ratio,
            backspace_count=ks_payload.backspace_count,
            hand_alternation_ratio=ks_payload.hand_alternation_ratio,
            same_hand_sequence_mean=ks_payload.same_hand_sequence_mean,
            finger_transition_ratio=ks_payload.finger_transition_ratio,
            seek_time_mean=ks_payload.seek_time_mean,
            seek_time_count=ks_payload.seek_time_count,
            shift_lag_mean=ks_payload.shift_lag_mean,
            shift_lag_std=ks_payload.shift_lag_std,
            shift_lag_count=ks_payload.shift_lag_count,
            dwell_mean_norm=ks_payload.dwell_mean_norm,
            dwell_std_norm=ks_payload.dwell_std_norm,
            flight_mean_norm=ks_payload.flight_mean_norm,
            flight_std_norm=ks_payload.flight_std_norm,
            p2p_std_norm=ks_payload.p2p_std_norm,
            r2r_mean_norm=ks_payload.r2r_mean_norm,
            shift_lag_norm=ks_payload.shift_lag_norm,
            extra_digraphs=ks_payload.extra_digraphs or {},
            key_dwell_map=ks_payload.key_dwell_map or {},
            # 4-variant digraph timings + per-pair flight + trigraph maps.
            # Required for RF retraining to reproduce the enrollment feature
            # vector once the user crosses the profile→RF sample threshold.
            digraph_dd_map=ks_payload.digraph_dd_map or {},
            digraph_du_map=ks_payload.digraph_du_map or {},
            digraph_ud_map=ks_payload.digraph_ud_map or {},
            digraph_uu_map=ks_payload.digraph_uu_map or {},
            flight_per_digraph=ks_payload.flight_per_digraph or {},
            trigraph_map=ks_payload.trigraph_map or {},
            saved_score=float(ks_score) if ks_score is not None else None,
        )
        db.add(new_sample)
        db.commit()

        updated_count = db.query(KeystrokeTemplate).filter(
            KeystrokeTemplate.user_id == user.id
        ).count()
        print(f"  💾 Keystroke sample saved [{source}] (total: {updated_count}/{MAX_SAMPLES})")

        # Adaptive retrain
        login_count_total = db.query(AuthLog).filter(
            AuthLog.user_id == user.id,
            AuthLog.auth_method == "keystroke",
            AuthLog.result == "granted"
        ).count()

        if updated_count <= 10:
            retrain_interval = 3
        elif updated_count <= 30:
            retrain_interval = 5
        else:
            retrain_interval = 10

        should_retrain = (login_count_total % retrain_interval == 0) or (updated_count in [10, 20, 30, 40, 50])
        if should_retrain:
            reason = f"milestone ({updated_count})" if updated_count in [10, 20, 30, 40, 50] else f"interval ({retrain_interval})"
            print(f"  🔄 Triggering retrain: {reason}")
            try:
                script_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    'ml', 'train_keystroke_rf.py'
                )
                subprocess.Popen(
                    [sys.executable, script_path, ks_payload.username],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                print(f"  ✅ Retraining started in background")
            except Exception as e:
                print(f"  ⚠️ Retrain failed: {e}")

    except Exception as _save_err:
        print(f"  ⚠️ Keystroke sample save failed: {_save_err}")


# ─────────────────────────────────────────────────────────────────────────────
#  KEYSTROKE AUTH
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/keystroke")
def verify_keystroke(payload: KeystrokeAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    model_path = os.path.join(_model_dir(), f"{_safe_filename(payload.username)}_keystroke_rf.pkl")

    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            model_type   = model_data.get('model_type', 'rf')
            model_stage  = model_data.get('model_stage', 'late' if model_type != 'profile' else 'early')
            feat_names   = model_data['feature_names']
            profile_mean = model_data['profile_mean']
            stored_threshold = model_data['threshold']

            # Per-user consistency signals (present on profile models only; RF
            # models trained before this field was added fall back to defaults).
            s_mean = float(model_data.get('self_score_mean', 1.0))
            s_std  = float(model_data.get('self_score_std',  0.0))
            is_inconsistent = bool(model_data.get('is_inconsistent', False))

            # Progressive threshold: relaxed during template stabilization.
            maturity  = _keystroke_maturity(db, user.id)
            threshold = _effective_keystroke_threshold(stored_threshold, maturity)
            mode = ("SOFT" if maturity < _SOFT_MATURITY_END else
                    "RAMP" if maturity < _RAMP_MATURITY_END else "HARD")
            print(f"[keystroke] stage={model_stage}  maturity={maturity}  mode={mode}  "
                  f"effective_threshold={threshold:.2f}  (stored={stored_threshold:.2f})")
            if is_inconsistent:
                print(f"[keystroke] inconsistent profile: mean={s_mean:.2f} σ={s_std:.2f} "
                      f"→ will relax toward {max(_SOFT_THRESHOLD, s_mean - 2*s_std):.2f}")

            # Build live feature vector. Must route prefixed feature names to the
            # correct source dict. Previously only extra_* and key_* were handled,
            # silently zeroing every flight_*-per-digraph and trigraph_* feature at
            # auth time — so the live vector looked nothing like what enrollment
            # trained on. Disambiguation: flight_mean / flight_std / flight_median
            # (+ _norm variants) are hardcoded scalar payload attributes that also
            # start with "flight_", so they must NOT be routed to the per-digraph dict.
            _SCALAR_FLIGHT = {
                'flight_mean', 'flight_std', 'flight_median',
                'flight_mean_norm', 'flight_std_norm',
            }
            extra_map    = payload.extra_digraphs    or {}
            key_map      = payload.key_dwell_map     or {}
            flight_map   = payload.flight_per_digraph or {}
            trigraph_map = payload.trigraph_map      or {}
            vec = np.array([
                float(extra_map.get(name[6:], 0.0) or 0.0)
                    if name.startswith("extra_") else
                float(key_map.get(name[4:], 0.0) or 0.0)
                    if name.startswith("key_") else
                float(trigraph_map.get(name[9:], 0.0) or 0.0)
                    if name.startswith("trigraph_") else
                float(flight_map.get(name[7:], 0.0) or 0.0)
                    if name.startswith("flight_") and name not in _SCALAR_FLIGHT else
                float(getattr(payload, name, 0.0) or 0.0)
                for name in feat_names
            ])

            # ── Profile Matcher path (≤20 enrollment samples) ─────────────────
            if model_type == 'profile':
                print(f"[keystroke] Using Profile Matcher (GP A+R) for '{payload.username}'")
                # TypingDNA-literal: fixed stored threshold, no maturity ramp.
                # The maturity ramp was designed for the old Z-score matcher whose
                # enrollment σ was noisy at low n. GP A+R is scale-aware via the
                # ratio test and rank-invariant, so it needs no warmup relaxation.
                if threshold != stored_threshold:
                    print(f"  Using stored threshold {stored_threshold:.2f} "
                          f"(ignoring maturity ramp → {threshold:.2f})")
                    threshold = stored_threshold
                personalized = _personalize_threshold(threshold, s_mean, s_std, is_inconsistent)
                if personalized < threshold:
                    print(f"  ⚑ Inconsistent typist — relaxing threshold "
                          f"{threshold:.2f} → {personalized:.2f} (mean={s_mean:.2f}, σ={s_std:.2f})")
                    threshold = personalized
                _project_root = os.path.normpath(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), '..', '..'
                ))
                if _project_root not in sys.path:
                    sys.path.insert(0, _project_root)
                from ml.keystroke_profile_matcher import compute_profile_score, compute_set_match_score
                profile_std = model_data['profile_std']
                # TypingDNA Same-Text style: if raw enrollment vectors were
                # stored at training time, score against each individually and
                # take top-3 mean — preserves per-pattern variance instead of
                # collapsing to one blurred centroid.
                genuine_vectors = model_data.get('genuine_vectors')
                if genuine_vectors:
                    result = compute_set_match_score(
                        vec, feat_names, genuine_vectors, profile_std,
                        model_stage=model_stage,
                    )
                    _k = result.get('top_k_used', '?')
                    _conf = result.get('confidence', 1.0)
                    print(f"  🎯 Set-match: top-{_k} weighted={result['score']:.3f} conf={_conf:.2f} | per-sample={[f'{s:.2f}' for s in result['per_sample_scores']]}")
                else:
                    result = compute_profile_score(
                        vec, feat_names, profile_mean, profile_std, model_stage=model_stage
                    )
                profile_score = result['score']
                speed_ratio   = result['speed_ratio']
                group_scores  = result['group_scores']

                fn_list = list(feat_names)
                confidence    = profile_score
                authenticated = confidence >= threshold
                hard_reject   = False

                breach = group_scores.get('floor_breach')
                if breach:
                    print(f"  ⛔ {breach}")
                n_valid = group_scores.get('n_valid', '?')
                print(f"  GP score={profile_score:.3f}  speed_ratio={speed_ratio:.2f}  "
                      f"Threshold={threshold:.2f}  "
                      f"A={group_scores['digraph_dist']:.2f} "
                      f"R={group_scores['digraph_rank']:.2f} "
                      f"n_valid={n_valid}  "
                      f"→ {'PASS' if authenticated else 'FAIL'}")

                # Diagnostic: on fail, dump top-10 highest-ratio digraph
                # positions so we can see which keys are dragging the score.
                if not authenticated:
                    details = result.get('details', [])
                    dg_rows = [
                        (nm, z, grp) for (nm, z, grp) in details if grp == 'digraph'
                    ]
                    dg_rows.sort(key=lambda r: r[1], reverse=True)
                    print(f"  [diag] top digraph-group ratio deviations (of {len(dg_rows)}):")
                    for nm, d, _ in dg_rows[:10]:
                        idx_f = fn_list.index(nm) if nm in fn_list else -1
                        if idx_f >= 0:
                            enr = float(profile_mean[idx_f])
                            live_raw = float(vec[idx_f])
                            print(f"    {nm:24s} live={live_raw:7.1f}  "
                                  f"enr={enr:7.1f}  ratio-1={d:+.2f}")

            # ── RF / GBM path (>10 enrollment samples) ────────────────────────
            else:
                print(f"[keystroke] Using RF model for '{payload.username}'")
                pipeline  = model_data['pipeline']
                vec_2d    = vec.reshape(1, -1)
                rf_score  = float(pipeline.predict_proba(vec_2d)[0][1])

                profile_std = model_data.get('profile_std', None)
                if profile_std is not None and len(profile_std) == len(profile_mean):
                    var       = profile_std ** 2
                    safe_var  = np.where(var < 1e-10, 1e-10, var)
                    diff      = vec - profile_mean
                    d_sq      = float(np.sum(diff ** 2 / safe_var))
                    d_sq_norm = d_sq / max(len(vec), 1)
                    exponent  = float(np.clip(2.5 * (d_sq_norm - 1.0), -500, 500))
                    mah_score = float(1.0 / (1.0 + np.exp(exponent)))
                else:
                    diff      = np.linalg.norm(vec - profile_mean)
                    scale     = np.linalg.norm(profile_mean) + 1e-9
                    mah_score = float(max(0, 1 - diff / scale))

                # ── Hard sanity gates ─────────────────────────────────────────
                hard_reject   = False
                reject_reason = ""

                fn_list = list(feat_names)
                def _fv(name):
                    try:
                        return float(vec[fn_list.index(name)])
                    except (ValueError, IndexError):
                        return None

                # Gate 1: dwell_mean z-score
                if not hard_reject and profile_std is not None and 'dwell_mean' in fn_list:
                    e_dwell     = float(profile_mean[fn_list.index('dwell_mean')])
                    e_dwell_std = float(profile_std[fn_list.index('dwell_mean')]) + 1e-9
                    e_dwell_std = max(e_dwell_std, e_dwell * 0.10, 20.0)
                    l_dwell     = _fv('dwell_mean')
                    if l_dwell is not None:
                        dwell_z = abs(l_dwell - e_dwell) / e_dwell_std
                        if dwell_z > 3.0:
                            hard_reject   = True
                            reject_reason = (f"dwell_mean z={dwell_z:.1f} "
                                             f"(live={l_dwell:.0f}ms enrolled={e_dwell:.0f}ms)")

                # Gate 2: typing_speed_cpm z-score
                if not hard_reject and profile_std is not None and 'typing_speed_cpm' in fn_list:
                    e_cpm     = float(profile_mean[fn_list.index('typing_speed_cpm')])
                    e_cpm_std = float(profile_std[fn_list.index('typing_speed_cpm')]) + 1e-9
                    e_cpm_std = max(e_cpm_std, e_cpm * 0.20)
                    l_cpm     = _fv('typing_speed_cpm')
                    if l_cpm is not None:
                        cpm_z = abs(l_cpm - e_cpm) / e_cpm_std
                        if cpm_z > 5.0:
                            hard_reject   = True
                            reject_reason = (f"typing_speed_cpm z={cpm_z:.1f} "
                                             f"(live={l_cpm:.0f} enrolled={e_cpm:.0f})")

                # Gate 3: Mahalanobis hard floor
                if not hard_reject and mah_score < 0.15:
                    hard_reject   = True
                    reject_reason = (f"Mahalanobis floor breach "
                                     f"(mah={mah_score:.4f}, d_sq_norm={d_sq_norm:.2f})")

                if hard_reject:
                    confidence    = 0.0
                    authenticated = False
                    print(f"  ⛔ Hard reject: {reject_reason}")
                else:
                    confidence    = fuse_keystroke_scores(rf_score, mah_score)
                    authenticated = confidence >= threshold

                print(f"  RF={rf_score:.3f}  Mah={mah_score:.3f}  "
                      f"Fused={confidence:.3f}  Threshold={threshold:.3f}"
                      f"  \u2192 {'PASS' if authenticated else 'FAIL'}")

        except Exception as e:
            print(f"[keystroke] model error: {e}")
            import traceback; traceback.print_exc()
            log_error("keystroke", payload.username, e,
                      extra={"phase": "model_eval", "model_path": model_path})
            confidence    = 0.0
            authenticated = False

    else:
        # No trained model — hard reject until enrollment is complete.
        print(f"[keystroke] No model for '{payload.username}' — hard reject until model is trained")
        log_auth_stage("keystroke", payload.username, "denied",
                       score=0.0, extra={"reason": "no_model", "model_path": model_path})
        confidence    = 0.0
        authenticated = False

    print(f"[keystroke] user={payload.username}  "
          f"confidence={confidence:.3f}  result={'PASS' if authenticated else 'FAIL'}")

    # `threshold` / `maturity` only exist when a model was loaded; for the
    # no-model hard-reject path below, fall back to nulls.
    _log_mat = locals().get('maturity')
    _log_thr = locals().get('threshold')
    log_attempt(db, user.id, "keystroke", confidence,
                "granted" if authenticated else "denied",
                template_maturity=_log_mat,
                effective_threshold=_log_thr)

    _extra = {
        "maturity":      _log_mat,
        "mode":          locals().get('mode'),
        "model_type":    locals().get('model_type'),
        "model_stage":   locals().get('model_stage'),
        "rf_score":      locals().get('rf_score'),
        "mah_score":     locals().get('mah_score'),
        "profile_score": locals().get('profile_score'),
        "speed_ratio":   locals().get('speed_ratio'),
        "hard_reject":   locals().get('hard_reject'),
        "reject_reason": locals().get('reject_reason') or None,
    }
    log_auth_stage(
        "keystroke", payload.username,
        "granted" if authenticated else "denied",
        score=float(confidence),
        threshold=float(_log_thr) if _log_thr is not None else None,
        model=locals().get('model_type'),
        extra={k: v for k, v in _extra.items() if v is not None},
    )

    # ═════════════════════════════════════════════════════════════════
    # ADAPTIVE LEARNING
    #
    # Case A — Keystroke alone granted access:
    #   Save the sample immediately.  No need to wait for voice/fusion
    #   because the user is already authenticated by keystroke alone.
    #
    # Case B — Keystroke failed, voice recovery will be tried:
    #   Cache the sample.  /auth/fuse will save it only if the full
    #   multimodal fusion grants access.  This prevents polluting the
    #   training data with samples from sessions the user ultimately
    #   failed to authenticate.
    #
    # Hard reject (confidence == 0): don't cache at all — the sample
    #   carries no useful signal.
    # ═════════════════════════════════════════════════════════════════
    # Continuous enrollment policy.
    #
    # Case A (keystroke alone passed): the effective threshold already reflects
    # the template's maturity stage — a pass at the soft threshold is safe to
    # feed back because the soft regime assumes multi-factor backup, and a
    # user reaching Case A also cleared password. Save the sample.
    #
    # Case B (keystroke failed, voice recovery will run): defer to /fuse —
    # it only saves if the *full* multi-factor pipeline grants access.
    if authenticated:
        _save_keystroke_sample(
            db, user, payload, source="login_ks",
            ks_score=float(confidence),
            effective_threshold=float(_log_thr) if _log_thr is not None else None,
        )
    elif confidence > 0:
        # Case B: low confidence, voice recovery path — hold for fusion
        _pending_ks_samples[payload.username] = payload
        print(f"  ⏳ Keystroke sample cached — will save only if fusion grants access")

    return {"authenticated": bool(authenticated), "confidence": float(confidence)}


# ─────────────────────────────────────────────────────────────────────────────
#  VOICE AUTH
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/voice")
def verify_voice(payload: VoiceAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")


    # ── PRIMARY AUTH: ECAPA-TDNN Speaker Verification ─────────────────────
    # Uses the pretrained SpeechBrain ECAPA-TDNN model (192-dim embeddings).
    # The ecapa_embedding was extracted from raw audio during /extract-mfcc
    # and is sent here as part of the VoiceAuth payload.
    # No per-user training needed — just cosine similarity vs enrolled profile.
    confidence    = 0.0
    authenticated = False
    try:
        project_root = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..'
        ))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from ml.voice_ecapa import predict_voice as ecapa_predict

        ecapa_embedding = getattr(payload, 'ecapa_embedding', []) or []
        if ecapa_embedding:
            ecapa_r       = ecapa_predict(payload.username, ecapa_embedding)
            confidence    = ecapa_r.get("confidence", 0.0) / 100.0   # normalise to [0,1]
            authenticated = ecapa_r.get("match", False)
            print(f"  [ECAPA] similarity={ecapa_r.get('similarity', 0):.4f}  "
                  f"threshold={ecapa_r.get('threshold', 0):.2f}  "
                  f"→ {'PASS' if authenticated else 'FAIL'}")
            if "error" in ecapa_r:
                print(f"  [ECAPA] note: {ecapa_r['error']}")
        else:
            print("  ⚠  ECAPA: no embedding in payload — re-enroll voice to generate ECAPA profile")
            confidence    = 0.0
            authenticated = False

    except Exception as e:
        print(f"[voice] ECAPA error: {e}")
        import traceback; traceback.print_exc()
        log_error("voice", payload.username, e, extra={"phase": "ecapa_predict"})
        confidence    = 0.0
        authenticated = False


    # ── PHRASE VERIFICATION: confirm the correct passphrase was spoken ────
    # Three-signal matching to handle accents and ASR errors:
    #
    #   Signal 1 — Character ratio (difflib):
    #     Compares the full strings character by character.
    #     Good for: clean pronunciation, exact matches.
    #     Weak for: strong accents that shift vowel sounds.
    #
    #   Signal 2 — Word overlap ratio:
    #     Counts how many expected words appear anywhere in the transcript.
    #     Good for: accents, word reordering, one dropped/mispronounced word.
    #     For a 4-word phrase, 3/4 correct words = 0.75 overlap ratio.
    #
    #   Signal 3 — Phonetic ratio (Metaphone-lite):
    #     Each expected word is reduced to a phonetic code (th→0, vowels
    #     dropped, duplicate consonants collapsed) and fuzzy-matched against
    #     the phonetic codes of transcript words. Catches cases where Whisper
    #     shifts vowels or drops a letter — e.g. "think"↔"thank" (both 0NK),
    #     "new"↔"nu", "stiled"↔"style". Standard string comparison misses
    #     these because the character edits look large, but the sounds are
    #     nearly identical.
    #
    #   Final score = max(char_ratio, word_overlap, phonetic_ratio)
    #   so any single strong signal can pass. Threshold = 0.70 — lenient
    #   enough for accents, strict enough to block completely different
    #   phrases.
    #
    # Deliberately NOT biasing Whisper with initial_prompt=expected_phrase:
    # that would make Whisper "hear" the expected phrase even when garbage
    # was spoken — a security hole on a biometric system.
    #
    # If Whisper was unavailable (empty transcript), check is skipped so the
    # system degrades gracefully rather than locking everyone out.
    import difflib, re

    def _phonetic(word: str) -> str:
        """
        Tiny Metaphone-style encoder: reduces a word to the consonant sounds
        an English ASR would most likely share across mishearings.
        Not a full Metaphone — just enough for the common confusions we see
        (think/thank/link/drink, new/renew, stiled/style, etc.).
        """
        w = word.lower()
        # Order matters: digraphs before single-char replacements.
        for old, new in (
            ("tch", "x"), ("sch", "x"), ("sh", "x"), ("ch", "x"),
            ("th", "0"),  ("ph", "f"),  ("ck", "k"), ("kn", "n"),
            ("gn", "n"),  ("wr", "r"),  ("qu", "kw"), ("wh", "w"),
            ("gh", ""),   ("x",  "ks"), ("z",  "s"),  ("v",  "f"),
        ):
            w = w.replace(old, new)
        if not w:
            return ""
        # Keep leading char (even if vowel); drop vowels elsewhere.
        first, rest = w[0], re.sub(r"[aeiouy]", "", w[1:])
        out = [first]
        for ch in rest:
            if ch != out[-1]:        # collapse consecutive duplicates
                out.append(ch)
        return "".join(out)

    transcript      = (getattr(payload, 'transcript', '') or '').strip()
    expected_phrase = decrypt(user.phrase).strip() if user.phrase else ""
    phrase_error    = ""

    if transcript and expected_phrase:
        def _norm(s):
            s = s.lower()
            s = re.sub(r"[^a-z0-9\s]", "", s)
            return re.sub(r"\s+", " ", s).strip()

        norm_expected   = _norm(expected_phrase)
        norm_transcript = _norm(transcript)

        # Signal 1: character-level similarity
        char_ratio = difflib.SequenceMatcher(None, norm_expected, norm_transcript).ratio()

        # Signal 2: word-level overlap
        expected_words    = set(norm_expected.split())
        transcript_words  = set(norm_transcript.split())
        matched_words     = expected_words & transcript_words
        word_overlap      = len(matched_words) / len(expected_words) if expected_words else 0.0

        # Signal 3: phonetic fuzzy match, word-by-word.
        # For each expected word, check whether any transcript word has a
        # phonetic code with ≥0.75 SequenceMatcher ratio. Threshold chosen
        # so that "think"↔"thank" (identical codes) and "style"↔"stiled"
        # (one-char diff on 3-char code) pass, while "think"↔"link" does not.
        expected_phon   = [_phonetic(w) for w in norm_expected.split()   if w]
        transcript_phon = [_phonetic(w) for w in norm_transcript.split() if w]
        phon_hits = 0
        for pe in expected_phon:
            if not pe:
                continue
            for pt in transcript_phon:
                if not pt:
                    continue
                if difflib.SequenceMatcher(None, pe, pt).ratio() >= 0.75:
                    phon_hits += 1
                    break
        phonetic_ratio = phon_hits / len(expected_phon) if expected_phon else 0.0

        final_score  = max(char_ratio, word_overlap, phonetic_ratio)
        phrase_match = final_score >= 0.70

        print(f"  [PHRASE] transcript='{transcript}'  expected='{expected_phrase}'")
        print(f"  [PHRASE] char_ratio={char_ratio:.2f}  word_overlap={word_overlap:.2f}  "
              f"phonetic={phonetic_ratio:.2f}  final={final_score:.2f}  "
              f"→ {'PASS' if phrase_match else 'FAIL'}")

        if authenticated and not phrase_match:
            authenticated = False
            confidence    = 0.0
            phrase_error  = (
                f"Wrong phrase spoken — heard: \"{transcript}\". "
                f"Please say your assigned phrase and try again."
            )
            print("  [VOICE] ECAPA passed but wrong phrase spoken — access denied")
    elif not transcript:
        print("  ⚠  [PHRASE] No transcript — Whisper unavailable, skipping phrase check")
    elif not expected_phrase:
        print("  ⚠  [PHRASE] No phrase assigned to user — skipping phrase check")

    print(f"[voice] user={payload.username}  "
          f"confidence={confidence:.3f}  result={'PASS' if authenticated else 'FAIL'}")

    log_attempt(db, user.id, "voice", confidence,
                "granted" if authenticated else "denied")

    log_auth_stage(
        "voice", payload.username,
        "granted" if authenticated else "denied",
        score=float(confidence),
        model="ecapa-tdnn",
        extra={
            "ecapa_similarity": locals().get('ecapa_r', {}).get('similarity') if 'ecapa_r' in locals() else None,
            "ecapa_threshold":  locals().get('ecapa_r', {}).get('threshold') if 'ecapa_r' in locals() else None,
            "ecapa_note":       locals().get('ecapa_r', {}).get('error') if 'ecapa_r' in locals() else None,
            "transcript":       transcript or None,
            "expected_phrase":  expected_phrase or None,
            "char_ratio":       locals().get('char_ratio'),
            "word_overlap":     locals().get('word_overlap'),
            "phonetic_ratio":   locals().get('phonetic_ratio'),
            "phrase_final":     locals().get('final_score'),
            "phrase_match":     locals().get('phrase_match'),
            "phrase_error":     phrase_error or None,
            "has_embedding":    bool(getattr(payload, 'ecapa_embedding', []) or []),
        },
    )

    # ── Adaptive learning: update ECAPA profile with new login embedding ────
    # ECAPA needs no retraining — just append the new embedding to the profile
    # so the mean vector gradually adapts to the user's voice over time.
    if authenticated:
        try:
            from ml.voice_ecapa import save_enrollment as ecapa_enroll
            ecapa_embedding = getattr(payload, 'ecapa_embedding', []) or []
            if ecapa_embedding:
                ecapa_enroll(payload.username, ecapa_embedding)
                print(f"  💾 ECAPA profile updated with login embedding")
        except Exception as _e:
            print(f"  ⚠️ ECAPA profile update skipped: {_e}")

    return {
        "authenticated": bool(authenticated),
        "confidence":    float(confidence),
        "phrase_error":  phrase_error,
    }


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
    return {"phrase": decrypt(user.phrase)}

@router.post("/security-question")
def get_security_question(payload: PasswordAuth, db: Session = Depends(get_db)):
    # Gated behind password re-verification so a raw email can't leak the
    # (decrypted) security prompt — which would otherwise expose a
    # recovery-path question for any registered address. Using POST + JSON
    # keeps the password out of URLs, logs, and browser history.
    email = payload.username.strip().lower()
    user = db.query(User).filter(User.username == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.password_hash or not bcrypt.checkpw(
        payload.password.encode('utf-8'),
        user.password_hash.encode('utf-8')
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    sq = db.query(SecurityQuestion).filter(SecurityQuestion.user_id == user.id).first()
    if not sq:
        raise HTTPException(status_code=404, detail="No security question found")
    return {"question": decrypt(sq.question)}


@router.post("/security")
def verify_security(payload: SecurityAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sq = db.query(SecurityQuestion).filter(SecurityQuestion.user_id == user.id).first()
    if not sq:
        raise HTTPException(status_code=404, detail="No security question found")

    authenticated = bcrypt.checkpw(
        payload.answer.strip().lower().encode(),
        sq.answer_hash.encode('utf-8')
    )

    print(f"[security] '{payload.username}' → {'PASS' if authenticated else 'FAIL'}")

    log_attempt(db, user.id, "security_question",
                1.0 if authenticated else 0.0,
                "granted" if authenticated else "denied")

    log_auth_stage(
        "security_question", payload.username,
        "granted" if authenticated else "denied",
        score=1.0 if authenticated else 0.0,
    )

    if not authenticated:
        # Recovery path: password already passed, so a wrong answer here is
        # high-signal impostor behavior. 3-strike buffer absorbs typos.
        sq_denials = _consecutive_denials(db, user.id, "security_question")
        if sq_denials >= 3:
            user.is_flagged = True
            db.commit()
            print(f"  ⚠️ User '{payload.username}' flagged after {sq_denials} security Q failures")

    return {
        "authenticated": authenticated,
        "confidence":    1.0 if authenticated else 0.0,
        "flagged":       bool(user.is_flagged),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MULTIMODAL FUSION  (server-side, authoritative)
# ─────────────────────────────────────────────────────────────────────────────

class FusionRequest(BaseModel):
    username:          str
    keystroke_score:   float   # intra-modal keystroke confidence [0, 1]
    voice_score:       float   # intra-modal voice confidence [0, 1]
    keystroke_passed:  bool    # whether keystroke individually exceeded its threshold
    voice_passed:      bool    # whether voice individually exceeded its threshold


@router.post("/fuse")
def fuse_scores(payload: FusionRequest, db: Session = Depends(get_db)):
    """
    Authoritative multimodal fusion endpoint — adaptive threshold.

    Only reached when keystroke score is in the uncertain range (0.55–0.79).
    High-confidence keystroke (≥ 0.80) grants immediately without calling this.

    Two cases handled:

    Case A — keystroke_passed=True (scored ≥ per-user threshold but < 0.80):
        The user is plausibly genuine but keystroke isn't certain enough alone.
        Voice confirms. Fused threshold: 0.58 (lenient — we already passed KS).
        Weights: 0.45 keystroke + 0.55 voice (voice carries more weight as
        the confirming modality).

    Case B — keystroke_passed=False (scored < per-user threshold):
        Keystroke failed. Voice is the recovery path.
        Fused threshold: 0.65 (stricter — need strong voice to compensate).
        Weights: 0.35 keystroke + 0.65 voice (voice dominates since KS failed).

    Both cases require voice >= 0.40 floor — ensures the system is genuinely
    bimodal and a completely failed voice can't be rescued by keystroke alone.
    """
    ks    = float(np.clip(payload.keystroke_score, 0.0, 1.0))
    voice = float(np.clip(payload.voice_score,     0.0, 1.0))

    VOICE_FLOOR = 0.40

    # ── Adaptive keystroke reliability ────────────────────────────────────────
    # Load per-user reliability from the trained model. Inconsistent typists
    # (high enrollment FRR) get a lower reliability → keystroke weight is
    # scaled down proportionally and voice carries more of the decision.
    # Missing / unreadable model → treat as fully reliable (1.0).
    ks_reliability = 1.0
    try:
        _mpath = os.path.join(_model_dir(),
                              f"{_safe_filename(payload.username)}_keystroke_rf.pkl")
        if os.path.exists(_mpath):
            with open(_mpath, 'rb') as _f:
                _md = pickle.load(_f)
            ks_reliability = float(_md.get('ks_reliability', 1.0))
    except Exception:
        ks_reliability = 1.0

    if payload.keystroke_passed:
        # Case A: KS passed but uncertain — voice confirms
        base_ks_w, base_voice_w = 0.45, 0.55
        FUSED_THRESHOLD         = 0.58
        case_label              = "A (ks-passed, voice-confirms)"
    else:
        # Case B: KS failed — voice is recovery, stricter threshold.
        # Threshold at 0.62 (not 0.65) so a strong voice ≥0.95 can still
        # recover when ks=0.0: 0.65*0.95 = 0.6175 ≈ 0.62.
        base_ks_w, base_voice_w = 0.35, 0.65
        FUSED_THRESHOLD         = 0.62
        case_label              = "B (ks-failed, voice-recovery)"

    # Scale keystroke weight by measured reliability; voice absorbs the rest.
    ks_w    = base_ks_w * ks_reliability
    voice_w = 1.0 - ks_w

    fused          = ks_w * ks + voice_w * voice
    voice_floor_ok = voice >= VOICE_FLOOR
    granted        = (fused >= FUSED_THRESHOLD) and voice_floor_ok

    reason = ""
    if not granted:
        if not voice_floor_ok:
            reason = f"voice {voice:.2f} below floor {VOICE_FLOOR}"
        else:
            reason = f"fused {fused:.2f} below threshold {FUSED_THRESHOLD}"

    print(
        f"[fuse] '{payload.username}'  case={case_label}  "
        f"ks_rel={ks_reliability:.2f}  weights=({ks_w:.2f}/{voice_w:.2f})  "
        f"ks={ks:.3f}  voice={voice:.3f}  "
        f"fused={fused:.3f}  floor={'ok' if voice_floor_ok else 'FAIL'}  "
        f"→ {'GRANT' if granted else f'DENY ({reason})'}"
    )

    user = db.query(User).filter(User.username == payload.username).first()
    if user:
        # Record maturity at the time of this attempt so the audit trail shows
        # which progressive-enrollment regime this fusion grant belongs to.
        fuse_maturity = _keystroke_maturity(db, user.id)
        log_attempt(db, user.id, "fusion", fused,
                    "granted" if granted else "denied",
                    template_maturity=fuse_maturity)

    log_auth_stage(
        "fusion", payload.username,
        "granted" if granted else "denied",
        score=float(fused),
        threshold=float(FUSED_THRESHOLD),
        extra={
            "case":            case_label,
            "ks_score":        round(ks, 4),
            "voice_score":     round(voice, 4),
            "ks_passed":       bool(payload.keystroke_passed),
            "voice_passed":    bool(payload.voice_passed),
            "ks_weight":       round(ks_w, 3),
            "voice_weight":    round(voice_w, 3),
            "ks_reliability":  round(ks_reliability, 3),
            "voice_floor_ok":  bool(voice_floor_ok),
            "voice_floor":     VOICE_FLOOR,
            "maturity":        locals().get('fuse_maturity'),
            "reason":          reason or None,
        },
    )

    # ── Save pending keystroke sample (Case B: voice recovery path) ───────────
    # Case A samples (keystroke granted directly) were already saved in
    # verify_keystroke.  Here we only handle Case B: keystroke failed but
    # voice recovery succeeded and fusion granted access.
    if granted:
        ks_payload = _pending_ks_samples.pop(payload.username, None)
        if ks_payload and user:
            _save_keystroke_sample(
                db, user, ks_payload, source="login_fusion",
                ks_score=float(ks),
            )
    else:
        # Login denied — discard the cached sample, do not save
        discarded = _pending_ks_samples.pop(payload.username, None)
        if discarded:
            print(f"  🗑️ Cached keystroke sample discarded (login denied)")

    return {
        "granted":         granted,
        "fused_score":     float(fused),
        "keystroke_score": ks,
        "voice_score":     voice,
        "voice_floor_ok":  voice_floor_ok,
        "threshold":       FUSED_THRESHOLD,
        "case":            case_label,
        "reason":          reason,
    }
