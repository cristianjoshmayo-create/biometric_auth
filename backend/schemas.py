# backend/schemas.py
# Shared Pydantic base models.
#
# Previously VoiceEnroll (enroll.py) and VoiceAuth (auth.py) were maintained
# separately and had already drifted — auth.py was missing the 4 v2 fields.
# Both now inherit from VoiceFeatures so they can never drift again.

from pydantic import BaseModel
from typing import List


class VoiceFeatures(BaseModel):
    """
    All 62 voice features extracted by extract-mfcc.
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