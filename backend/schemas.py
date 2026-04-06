# backend/schemas.py
# Shared Pydantic base models.
#
# VoiceFeatures is the single source of truth for all voice fields — both
# enrollment (VoiceEnroll) and authentication (VoiceAuth) inherit from it
# so the two can never drift out of sync again.
#
# v4 adds:
#   mfcc_frames : List[List[float]]
#     Raw per-frame MFCC matrix returned by /enroll/extract-mfcc.
#     Shape: (T, 13) stored as a list of lists.
#     Required by the CNN model for sequence-level training and inference.
#     Defaults to [] for backwards compatibility with old enrolled users
#     (they will fall back to CNN score = 0.0 until they re-enroll).

from pydantic import BaseModel
from typing import List


class VoiceFeatures(BaseModel):
    """
    All voice features extracted by /enroll/extract-mfcc.
    Shared by enrollment and authentication — single source of truth.
    """
    mfcc_features:          List[float]
    mfcc_std:               List[float] = []

    # v2 — temporal dynamics (delta MFCCs)
    delta_mfcc_mean:        List[float] = []
    delta2_mfcc_mean:       List[float] = []

    pitch_mean:             float = 0
    pitch_std:              float = 0
    speaking_rate:          float = 0
    energy_mean:            float = 0
    energy_std:             float = 0
    zcr_mean:               float = 0
    spectral_centroid_mean: float = 0
    spectral_rolloff_mean:  float = 0

    # v2 — noise-robustness features
    spectral_flux_mean:     float = 0
    voiced_fraction:        float = 0
    snr_db:                 float = 0   # quality logging only

    # ECAPA-TDNN — 192-dim pretrained speaker embedding
    # Computed by /enroll/extract-mfcc and passed through enroll + auth.
    # Defaults to [] for users enrolled before this update (must re-enroll).
    ecapa_embedding:        List[float] = []

    # Resemblyzer — pretrained 256-dim speaker embedding
    # Computed by extract_mfcc endpoint and passed through enroll/auth flow.
    # Empty list = old enrolled user (before Resemblyzer was added).
    resemblyzer_embedding:  List[float] = []

    # v4 CNN — raw per-frame MFCC matrix (T × 13)
    # Returned by /enroll/extract-mfcc and stored in voice_templates.mfcc_frames.
    # If empty (old enrolled users), CNN inference falls back to score=0.0.
    mfcc_frames:            List[List[float]] = []