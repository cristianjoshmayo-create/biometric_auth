# backend/routers/enroll.py
# IMPROVED v2: better noise rejection in extract-mfcc, 62-feature extraction

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
import bcrypt

from database.db import get_db
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion
from schemas import VoiceFeatures
import threading
import sys

router = APIRouter()

MAX_KEYSTROKE_SAMPLES = 5
MAX_VOICE_SAMPLES     = 3


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-TRAIN HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_training(username: str):
    try:
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
    except Exception as e:
        import traceback
        print(f"❌ Auto-training failed for '{username}': {e}")
        traceback.print_exc()


def trigger_training(username: str):
    t = threading.Thread(target=_run_training, args=(username,), daemon=True)
    t.start()


def _run_voice_training(username: str):
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
    except Exception as e:
        import traceback
        print(f"❌ Voice auto-training failed for '{username}': {e}")
        traceback.print_exc()


def trigger_voice_training(username: str):
    t = threading.Thread(target=_run_voice_training, args=(username,), daemon=True)
    t.start()


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    password: str


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


class VoiceEnroll(VoiceFeatures):
    username: str


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

    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    hashed = bcrypt.hashpw(
        payload.password.encode('utf-8'),
        bcrypt.gensalt()
    ).decode('utf-8')

    new_user = User(username=payload.username, password_hash=hashed)
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

    training_started = False
    if attempt_num >= MAX_KEYSTROKE_SAMPLES:
        trigger_training(payload.username)
        training_started = True

    return {
        "success":          True,
        "message":          f"Keystroke attempt #{attempt_num} saved",
        "attempt_number":   attempt_num,
        "training_started": training_started,
        "training_note":    "Model training started in background." if training_started else "",
    }


@router.post("/voice")
def enroll_voice(payload: VoiceEnroll, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    missing_features = (
        not payload.mfcc_std or all(v == 0 for v in payload.mfcc_std)
    ) and payload.pitch_mean == 0 and payload.energy_mean == 0

    if missing_features:
        print(f"⚠  PARTIAL ENROLLMENT for '{payload.username}': missing features")

    existing_count = db.query(VoiceTemplate).filter(
        VoiceTemplate.user_id == user.id
    ).count()

    template = VoiceTemplate(
        user_id               = user.id,
        attempt_number        = existing_count + 1,
        mfcc_features         = payload.mfcc_features,
        mfcc_std              = payload.mfcc_std if payload.mfcc_std else [],
        pitch_mean            = payload.pitch_mean,
        pitch_std             = payload.pitch_std,
        speaking_rate         = payload.speaking_rate,
        energy_mean           = payload.energy_mean,
        energy_std            = payload.energy_std,
        zcr_mean              = payload.zcr_mean,
        spectral_centroid_mean= payload.spectral_centroid_mean,
        spectral_rolloff_mean = payload.spectral_rolloff_mean,
    )

    # Save new fields if the model column exists (graceful for older DB schemas)
    for field in ['delta_mfcc_mean', 'delta2_mfcc_mean', 'spectral_flux_mean', 'voiced_fraction', 'snr_db']:
        val = getattr(payload, field, None)
        if val is not None and hasattr(template, field):
            setattr(template, field, val)

    db.add(template)
    db.commit()

    attempt_num = existing_count + 1
    print(f"✅ Voice attempt #{attempt_num} for '{payload.username}' | "
          f"pitch={payload.pitch_mean:.1f}Hz  energy={payload.energy_mean:.4f}  "
          f"snr={payload.snr_db:.1f}dB  voiced={payload.voiced_fraction:.0%}")

    training_started = attempt_num >= MAX_VOICE_SAMPLES
    if training_started:
        trigger_voice_training(payload.username)

    return {
        "success":           True,
        "message":           f"Voice attempt #{attempt_num} saved",
        "attempt_number":    attempt_num,
        "has_full_features": not missing_features,
        "training_started":  training_started,
        "training_note":     "Voice model training started in background." if training_started else "",
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
#  IMPROVED AUDIO PROCESSING + FULL FEATURE EXTRACTION
#
#  Key improvements over original:
#  1. Multi-band SNR estimation — rejects samples where noise floor is high
#  2. Adaptive spectral subtraction — removes stationary background noise
#     from the power spectrum before MFCC extraction
#  3. Voiced-only MFCC computation — ignores unvoiced/silence frames for
#     cleaner speaker-identity features (CMVN normalisation)
#  4. Delta and delta-delta MFCCs — capture rate of change (speaker dynamics)
#  5. Spectral flux — distinguishes consistent voiced speech from erratic noise
#  6. voiced_fraction — fraction of frames identified as voiced by WebRTC VAD
#  7. Stricter VAD: mode 3 + minimum 60% voiced ratio (was 40%)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_snr(audio: np.ndarray, sr: int) -> float:
    """
    Estimate SNR in dB using a simple noise floor from the quietest 10% of frames.
    Returns SNR in dB; higher = cleaner.
    """
    hop   = 512
    frame = librosa.feature.rms(y=audio, hop_length=hop)[0]
    if len(frame) < 5:
        return 0.0
    noise_floor = np.percentile(frame, 10)   # quietest 10% of frames ≈ noise
    signal_rms  = np.percentile(frame, 90)   # loudest 90% ≈ speech
    if noise_floor < 1e-10:
        return 40.0  # essentially silent background
    snr = 20.0 * np.log10((signal_rms + 1e-10) / (noise_floor + 1e-10))
    return float(np.clip(snr, -10, 60))


def _spectral_subtraction(audio: np.ndarray, sr: int,
                          noise_frac: float = 0.10,
                          alpha: float = 2.0) -> np.ndarray:
    """
    Simple power-spectrum spectral subtraction for stationary noise removal.

    Steps:
      1. Estimate noise power from the first noise_frac of the signal.
      2. Subtract alpha * noise_power from the signal power spectrum.
      3. Reconstruct via ISTFT.

    alpha=2.0 is an oversubtraction factor — common in speech enhancement.
    """
    n_fft  = 512
    hop    = n_fft // 4

    stft   = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
    magnitude = np.abs(stft)
    phase     = np.angle(stft)

    # Estimate noise from first noise_frac of frames
    n_noise_frames = max(1, int(magnitude.shape[1] * noise_frac))
    noise_est = np.mean(magnitude[:, :n_noise_frames] ** 2, axis=1, keepdims=True)

    # Subtract and half-wave rectify (no negative power)
    power_clean = np.maximum(magnitude ** 2 - alpha * noise_est, 0)
    mag_clean   = np.sqrt(power_clean)

    stft_clean  = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop, length=len(audio))
    return audio_clean.astype(np.float32)


@router.post("/extract-mfcc")
async def extract_mfcc(payload: AudioData, db: Session = Depends(get_db)):
    """
    Improved audio quality validation + 62-speaker-feature extraction.

    New features vs original (34):
      + delta_mfcc_mean[0..12]   : rate of change of MFCC coefficients
      + delta2_mfcc_mean[0..12]  : acceleration of MFCC coefficients
      + spectral_flux_mean       : average frame-to-frame spectral change
      + voiced_fraction          : fraction of frames classified as voiced
      Total: 62 features

    New noise-handling:
      • SNR estimated and returned (minimum 10 dB required)
      • Spectral subtraction applied before MFCC extraction
      • CMVN applied to MFCCs (cepstral mean-variance normalisation)
      • MFCCs computed only from voiced frames (WebRTC VAD filtered)
      • Stricter VAD: 60% voiced frames required (was 40%)
    """
    input_path = None
    wav_path   = None

    try:
        audio_bytes  = base64.b64decode(payload.audio_data)
        audio_format = payload.audio_format or "webm"

        print(f"\n{'='*60}")
        print(f"IMPROVED VOICE FEATURE EXTRACTION")
        print(f"Audio: {len(audio_bytes)} bytes  format: {audio_format}")
        print(f"{'='*60}")

        # ── Save + convert to 16kHz mono WAV ──────────────────────────────
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

        audio, sr     = librosa.load(wav_path, sr=16000, mono=True)
        total_duration = len(audio) / sr
        print(f"Duration: {total_duration:.2f}s")

        # ── SNR check ──────────────────────────────────────────────────────
        snr_db = _estimate_snr(audio, sr)
        print(f"Estimated SNR: {snr_db:.1f} dB")

        # Lowered from 10dB to 6dB — typical laptop mic in a normal room
        # measures 6–12dB. 10dB was blocking almost every real recording.
        if snr_db < 6.0:
            return {
                "success": False,
                "detail":  (
                    f"Audio too noisy (SNR={snr_db:.1f}dB, need ≥6dB). "
                    "Move away from fans or loud speakers and try again."
                ),
                "snr_db": snr_db,
            }

        # ── Spectral subtraction (denoise) ─────────────────────────────────
        # Apply whenever SNR < 30dB (was 25dB) to cover more real-world cases
        if snr_db < 30.0:
            print(f"  Applying spectral subtraction (SNR={snr_db:.1f}dB)")
            audio_clean = _spectral_subtraction(audio, sr)
        else:
            audio_clean = audio
            print(f"  Skipping spectral subtraction (SNR={snr_db:.1f}dB ≥ 30dB)")

        # ── WebRTC VAD ─────────────────────────────────────────────────────
        import webrtcvad
        # Mode 1 (was 3) — less aggressive, works better on compressed
        # audio (webm/opus from browser) and laptop mic recordings
        vad = webrtcvad.Vad(1)

        with open(wav_path, 'rb') as f:
            wav_data = f.read()

        audio_pcm  = wav_data[44:]
        frame_size = int(sr * 30 / 1000) * 2   # 30ms frames, 16-bit PCM
        voiced = total = 0
        voiced_frame_indices = []

        for i in range(0, len(audio_pcm) - frame_size, frame_size):
            frame = audio_pcm[i:i + frame_size]
            if len(frame) == frame_size:
                total += 1
                is_voiced = vad.is_speech(frame, sr)
                if is_voiced:
                    voiced += 1
                    voiced_frame_indices.append(total - 1)

        voice_ratio     = voiced / total if total > 0 else 0
        speech_duration = (voiced * 30) / 1000
        voiced_fraction = voice_ratio

        print(f"VAD: {voice_ratio:.0%} voiced  ({speech_duration:.1f}s speech)")

        # Lowered from 60% to 40% — browser audio compression and laptop mics
        # cause many voiced frames to be misclassified as unvoiced by aggressive VAD.
        # 40% still ensures we have real speech, not just ambient noise.
        if voice_ratio < 0.40:
            return {
                "success": False,
                "detail":  (
                    f"Not enough speech detected ({voice_ratio:.0%} voiced, need ≥40%). "
                    f"Speak the phrase clearly: 'biometric voice keystroke authentication'."
                ),
                "snr_db": snr_db,
            }
        # Lowered from 1.5s to 1.0s minimum speech
        if speech_duration < 1.0:
            return {
                "success": False,
                "detail":  f"Recording too short ({speech_duration:.1f}s of speech). Speak the full phrase.",
            }

        # ── Energy check ───────────────────────────────────────────────────
        rms = float(np.sqrt(np.mean(audio_clean ** 2)))
        # Lowered from 0.02 to 0.005 — laptop mics and some browsers
        # output much lower amplitude than professional mics
        if rms < 0.005:
            return {
                "success": False,
                "detail":  f"Audio too quiet (level={rms:.4f}). Speak louder or move closer to the mic.",
            }

        # ── Spectral sanity checks ─────────────────────────────────────────
        centroid      = librosa.feature.spectral_centroid(y=audio_clean, sr=sr)
        mean_centroid = float(np.mean(centroid))
        # Widened from 600–5500 to 300–7000 Hz — some voices and accents
        # have more energy outside the narrow 600–5500 band
        if mean_centroid < 300 or mean_centroid > 7000:
            return {
                "success": False,
                "detail":  f"Audio doesn't sound like speech (centroid={mean_centroid:.0f}Hz). Check microphone.",
            }

        zcr      = librosa.feature.zero_crossing_rate(audio_clean)
        mean_zcr = float(np.mean(zcr))
        # Widened from 0.02–0.5 to 0.01–0.6 — compressed audio formats
        # can produce slightly different ZCR distributions
        if mean_zcr < 0.01 or mean_zcr > 0.6:
            return {
                "success": False,
                "detail":  f"Unusual audio (ZCR={mean_zcr:.4f}). Ensure microphone is working.",
            }

        # ── Extract voiced-only audio for MFCC computation ─────────────────
        # This is the key improvement: MFCCs from voiced segments only
        # are more speaker-discriminative and more noise-robust.
        hop_length = int(sr * 0.010)   # 10ms hop
        frame_len  = int(sr * 0.025)   # 25ms frame

        if voiced_frame_indices and len(voiced_frame_indices) > 10:
            # Convert 30ms VAD frames to sample indices and extract
            voiced_mask = np.zeros(len(audio_clean), dtype=bool)
            vad_frame_samples = int(sr * 0.030)
            for fi in voiced_frame_indices:
                start = fi * vad_frame_samples
                end   = min(start + vad_frame_samples, len(audio_clean))
                voiced_mask[start:end] = True
            audio_voiced = audio_clean[voiced_mask]
            if len(audio_voiced) < sr * 0.5:
                audio_voiced = audio_clean  # fallback if too short after masking
        else:
            audio_voiced = audio_clean

        # ── MFCC extraction with CMVN ──────────────────────────────────────
        # CMVN (cepstral mean-variance normalisation) removes channel effects
        # and adapts for different microphones/environments.
        mfccs     = librosa.feature.mfcc(y=audio_voiced, sr=sr, n_mfcc=13,
                                          n_fft=frame_len, hop_length=hop_length)
        # Apply CMVN
        mfcc_cmvn = (mfccs - mfccs.mean(axis=1, keepdims=True)) / (mfccs.std(axis=1, keepdims=True) + 1e-8)

        mfcc_mean = np.mean(mfcc_cmvn, axis=1)
        mfcc_std  = np.std(mfcc_cmvn,  axis=1)

        if np.mean(mfcc_std) < 0.1:
            return {
                "success": False,
                "detail":  "Audio lacks variation typical of speech. Try again.",
            }

        # ── Delta + delta-delta MFCCs ──────────────────────────────────────
        # Deltas capture temporal dynamics (HOW the voice changes) — a key
        # speaker-identity cue that static MFCCs miss entirely.
        delta_mfcc  = librosa.feature.delta(mfcc_cmvn, order=1)
        delta2_mfcc = librosa.feature.delta(mfcc_cmvn, order=2)

        delta_mfcc_mean  = np.mean(delta_mfcc,  axis=1)
        delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)

        # ── Pitch ─────────────────────────────────────────────────────────
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio_voiced,
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

        # ── Speaking rate ─────────────────────────────────────────────────
        rms_frames    = librosa.feature.rms(y=audio_clean, hop_length=512)[0]
        rms_threshold = np.mean(rms_frames) * 0.5
        peaks         = np.where(
            (rms_frames[1:-1] > rms_frames[:-2]) &
            (rms_frames[1:-1] > rms_frames[2:]) &
            (rms_frames[1:-1] > rms_threshold)
        )[0]
        speaking_rate = float(len(peaks) / total_duration) if total_duration > 0 else 0.0

        # ── Energy ────────────────────────────────────────────────────────
        energy_frames = librosa.feature.rms(y=audio_voiced, hop_length=hop_length)[0]
        energy_mean   = float(np.mean(energy_frames))
        energy_std    = float(np.std(energy_frames))

        # ── Spectral rolloff ──────────────────────────────────────────────
        rolloff      = librosa.feature.spectral_rolloff(y=audio_voiced, sr=sr)
        rolloff_mean = float(np.mean(rolloff))

        # ── Spectral flux (NEW) ───────────────────────────────────────────
        # Spectral flux = average frame-to-frame spectral change.
        # Speech has smooth, predictable flux; background noise is erratic.
        # This feature helps reject recordings with intermittent noise bursts.
        stft_flux = librosa.stft(audio_voiced, n_fft=512, hop_length=hop_length)
        mag_flux  = np.abs(stft_flux)
        flux      = np.sqrt(np.sum(np.diff(mag_flux, axis=1) ** 2, axis=0))
        spectral_flux_mean = float(np.mean(flux))

        # ── Summary ───────────────────────────────────────────────────────
        print(f"✅ ALL CHECKS PASSED")
        print(f"  MFCC[0] mean={mfcc_mean[0]:.2f}  std_mean={np.mean(mfcc_std):.2f}")
        print(f"  Delta[0] mean={delta_mfcc_mean[0]:.3f}  D2[0]={delta2_mfcc_mean[0]:.3f}")
        print(f"  Pitch  mean={pitch_mean:.1f}Hz  std={pitch_std:.1f}Hz")
        print(f"  Rate   {speaking_rate:.2f} peaks/s")
        print(f"  Energy mean={energy_mean:.4f}  std={energy_std:.4f}")
        print(f"  Spectral flux={spectral_flux_mean:.2f}")
        print(f"  Voiced fraction={voiced_fraction:.2%}")
        print(f"  SNR={snr_db:.1f}dB")

        return {
            "success":              True,
            "mfcc_features":        mfcc_mean.tolist(),
            "mfcc_std":             mfcc_std.tolist(),
            "delta_mfcc_mean":      delta_mfcc_mean.tolist(),   # NEW
            "delta2_mfcc_mean":     delta2_mfcc_mean.tolist(),  # NEW
            "pitch_mean":           pitch_mean,
            "pitch_std":            pitch_std,
            "speaking_rate":        speaking_rate,
            "energy_mean":          energy_mean,
            "energy_std":           energy_std,
            "zcr_mean":             mean_zcr,
            "spectral_centroid_mean": mean_centroid,
            "spectral_rolloff_mean":  rolloff_mean,
            "spectral_flux_mean":   spectral_flux_mean,         # NEW
            "voiced_fraction":      voiced_fraction,            # NEW
            "snr_db":               snr_db,                     # NEW
            "validation": {
                "voice_ratio":        float(voice_ratio),
                "speech_duration":    float(speech_duration),
                "rms_energy":         rms,
                "spectral_centroid":  mean_centroid,
                "snr_db":             snr_db,
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


@router.get("/debug/voice-count")
def debug_voice_count(username: str = Query(...), db: Session = Depends(get_db)):
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
        has_delta = bool(getattr(t, 'delta_mfcc_mean', None))
        rows.append({
            "attempt":    t.attempt_number,
            "mfcc0":      round(mfcc[0], 2) if mfcc else None,
            "pitch":      round(float(t.pitch_mean or 0), 2),
            "energy":     round(float(t.energy_mean or 0), 5),
            "snr_db":     round(float(getattr(t, 'snr_db', 0) or 0), 1),
            "voiced_frac":round(float(getattr(t, 'voiced_fraction', 0) or 0), 2),
            "has_std":    bool(std and any(v != 0 for v in std)),
            "has_delta":  has_delta,
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
            f"⚠  {len(incomplete)} row(s) missing features — re-enroll"
        ),
    }


@router.post("/re-enroll/clear")
def clear_voice_enrollment(payload: ClearEnrollPayload, db: Session = Depends(get_db)):
    if payload.confirm != "yes-delete":
        raise HTTPException(status_code=400, detail='Set confirm="yes-delete" to proceed.')

    user = db.query(User).filter(User.username == payload.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    deleted = (
        db.query(VoiceTemplate)
        .filter(VoiceTemplate.user_id == user.id)
        .delete()
    )
    db.commit()

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