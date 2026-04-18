# backend/database/models.py

from sqlalchemy import (
    Column, Integer, String, Float,
    ARRAY, ForeignKey, DateTime, Text, Boolean, JSON
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database.db import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(255), unique=True, nullable=False)  # stores email address
    password_hash = Column(String(255), nullable=True)
    phrase        = Column(String(255), nullable=True)  # unique 4-word passphrase per user
    is_flagged    = Column(Boolean, default=False)
    created_at    = Column(DateTime, default=func.now())

    keystroke_template = relationship("KeystrokeTemplate", back_populates="user")
    voice_template     = relationship("VoiceTemplate",     back_populates="user")
    security_question  = relationship("SecurityQuestion",  back_populates="user")
    auth_logs          = relationship("AuthLog",           back_populates="user")


class KeystrokeTemplate(Base):
    __tablename__ = "keystroke_templates"

    id                 = Column(Integer, primary_key=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=False)
    attempt_number     = Column(Integer, default=1)
    enrollment_session = Column(Integer, default=1)
    sample_order       = Column(Integer, default=0)
    source             = Column(String(50), default="enrollment")
    # Score this sample earned at save time. Used for quality-weighted eviction:
    # when the adaptive pool overflows, the lowest-scoring adaptive row is evicted
    # instead of the oldest, bounding centroid drift. Null for enrollment rows.
    #   ALTER TABLE keystroke_templates ADD COLUMN IF NOT EXISTS saved_score FLOAT;
    saved_score        = Column(Float, nullable=True)

    dwell_times  = Column(ARRAY(Float), nullable=False)
    flight_times = Column(ARRAY(Float), nullable=False)
    typing_speed = Column(Float, default=0)

    dwell_mean   = Column(Float, default=0)
    dwell_std    = Column(Float, default=0)
    dwell_median = Column(Float, default=0)
    dwell_min    = Column(Float, default=0)
    dwell_max    = Column(Float, default=0)

    flight_mean   = Column(Float, default=0)
    flight_std    = Column(Float, default=0)
    flight_median = Column(Float, default=0)

    p2p_mean = Column(Float, default=0)
    p2p_std  = Column(Float, default=0)
    r2r_mean = Column(Float, default=0)
    r2r_std  = Column(Float, default=0)

    digraph_th = Column(Float, default=0)
    digraph_he = Column(Float, default=0)
    digraph_bi = Column(Float, default=0)
    digraph_io = Column(Float, default=0)
    digraph_om = Column(Float, default=0)
    digraph_me = Column(Float, default=0)
    digraph_et = Column(Float, default=0)
    digraph_tr = Column(Float, default=0)
    digraph_ri = Column(Float, default=0)
    digraph_ic = Column(Float, default=0)
    digraph_vo = Column(Float, default=0)
    digraph_oi = Column(Float, default=0)
    digraph_ce = Column(Float, default=0)
    digraph_ke = Column(Float, default=0)
    digraph_ey = Column(Float, default=0)
    digraph_ys = Column(Float, default=0)
    digraph_st = Column(Float, default=0)
    digraph_ro = Column(Float, default=0)
    digraph_ok = Column(Float, default=0)
    digraph_au = Column(Float, default=0)
    digraph_ut = Column(Float, default=0)
    digraph_en = Column(Float, default=0)
    digraph_nt = Column(Float, default=0)
    digraph_ti = Column(Float, default=0)
    digraph_ca = Column(Float, default=0)
    digraph_at = Column(Float, default=0)
    digraph_on = Column(Float, default=0)

    extra_digraphs = Column(JSON, default=dict)
    key_dwell_map  = Column(JSON, default=dict)   # { 's': avg_ms, 'p': avg_ms, ... }

    # ── 4-variant digraph timings (Killourhy–Maxion convention) ──────────────
    # extra_digraphs above is DD (press→press) for backward compat.
    # The four maps below let the trainer use any combination.
    #
    # MIGRATION (run once, e.g. via psql):
    #   ALTER TABLE keystroke_templates
    #     ADD COLUMN IF NOT EXISTS digraph_dd_map     JSON DEFAULT '{}',
    #     ADD COLUMN IF NOT EXISTS digraph_du_map     JSON DEFAULT '{}',
    #     ADD COLUMN IF NOT EXISTS digraph_ud_map     JSON DEFAULT '{}',
    #     ADD COLUMN IF NOT EXISTS digraph_uu_map     JSON DEFAULT '{}',
    #     ADD COLUMN IF NOT EXISTS flight_per_digraph JSON DEFAULT '{}',
    #     ADD COLUMN IF NOT EXISTS trigraph_map       JSON DEFAULT '{}';
    digraph_dd_map     = Column(JSON, default=dict)   # press[i] → press[i+1]
    digraph_du_map     = Column(JSON, default=dict)   # press[i] → release[i]
    digraph_ud_map     = Column(JSON, default=dict)   # release[i] → press[i+1]  (per-pair flight)
    digraph_uu_map     = Column(JSON, default=dict)   # release[i] → release[i+1]
    flight_per_digraph = Column(JSON, default=dict)   # alias of UD; convenient for ML
    trigraph_map       = Column(JSON, default=dict)   # press[i] → press[i+2] per trigraph

    typing_speed_cpm        = Column(Float, default=0)
    typing_duration         = Column(Float, default=0)
    rhythm_mean             = Column(Float, default=0)
    rhythm_std              = Column(Float, default=0)
    rhythm_cv               = Column(Float, default=0)
    pause_count             = Column(Float, default=0)
    pause_mean              = Column(Float, default=0)
    backspace_ratio         = Column(Float, default=0)
    backspace_count         = Column(Float, default=0)
    hand_alternation_ratio  = Column(Float, default=0)
    same_hand_sequence_mean = Column(Float, default=0)
    finger_transition_ratio = Column(Float, default=0)
    seek_time_mean          = Column(Float, default=0)
    seek_time_count         = Column(Float, default=0)

    shift_lag_mean  = Column(Float, default=0)
    shift_lag_std   = Column(Float, default=0)
    shift_lag_count = Column(Float, default=0)

    dwell_mean_norm  = Column(Float, default=0)
    dwell_std_norm   = Column(Float, default=0)
    flight_mean_norm = Column(Float, default=0)
    flight_std_norm  = Column(Float, default=0)
    p2p_std_norm     = Column(Float, default=0)
    r2r_mean_norm    = Column(Float, default=0)
    shift_lag_norm   = Column(Float, default=0)

    enrolled_at = Column(DateTime, default=func.now())
    user = relationship("User", back_populates="keystroke_template")


class VoiceTemplate(Base):
    __tablename__ = "voice_templates"

    id             = Column(Integer, primary_key=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    attempt_number = Column(Integer, default=1)

    mfcc_features = Column(ARRAY(Float), nullable=False)
    mfcc_std      = Column(ARRAY(Float), default=list)

    pitch_mean = Column(Float, default=0)
    pitch_std  = Column(Float, default=0)

    speaking_rate = Column(Float, default=0)

    energy_mean = Column(Float, default=0)
    energy_std  = Column(Float, default=0)

    zcr_mean               = Column(Float, default=0)
    spectral_centroid_mean = Column(Float, default=0)
    spectral_rolloff_mean  = Column(Float, default=0)

    delta_mfcc_mean  = Column(ARRAY(Float), default=list)
    delta2_mfcc_mean = Column(ARRAY(Float), default=list)

    spectral_flux_mean = Column(Float, default=0)
    voiced_fraction    = Column(Float, default=0)
    snr_db             = Column(Float, default=0)

    # v4 CNN — raw per-frame MFCC matrix stored as JSON.
    # Shape when loaded: list of T lists of 13 floats  →  (T, 13).
    # The CNN training script converts this to a (39, T_MAX) tensor by
    # computing delta + delta² on the fly and padding to T_MAX=300 frames.
    # Run the migration SQL before deploying this change:
    #   ALTER TABLE voice_templates
    #   ADD COLUMN IF NOT EXISTS mfcc_frames JSON DEFAULT '[]';
    mfcc_frames = Column(JSON, default=list)

    enrolled_at = Column(DateTime, default=func.now())
    user = relationship("User", back_populates="voice_template")


class SecurityQuestion(Base):
    __tablename__ = "security_questions"

    id          = Column(Integer, primary_key=True)
    user_id     = Column(Integer, ForeignKey("users.id"), unique=True)
    question    = Column(Text, nullable=False)
    answer_hash = Column(Text, nullable=False)

    user = relationship("User", back_populates="security_question")


class AuthLog(Base):
    __tablename__ = "auth_logs"

    id               = Column(Integer, primary_key=True)
    user_id          = Column(Integer, ForeignKey("users.id"))
    auth_method      = Column(String(50))
    confidence_score = Column(Float)
    result           = Column(String(20))
    failed_attempts  = Column(Integer, default=0)
    attempted_at     = Column(DateTime, default=func.now())
    # Progressive-enrollment audit trail: which threshold this attempt was
    # judged against, and how many successful logins had shaped the template
    # at the time. Populated for keystroke/fusion rows; null elsewhere.
    template_maturity    = Column(Integer, nullable=True)
    effective_threshold  = Column(Float,   nullable=True)

    user = relationship("User", back_populates="auth_logs")