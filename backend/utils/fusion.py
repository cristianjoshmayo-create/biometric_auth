# backend/utils/fusion.py
# Multimodal Biometric Score Fusion Module
#
# This module centralises ALL score-level fusion logic for the multimodal
# biometric authentication system. Keeping weights here as a single source
# of truth ensures training, authentication, and evaluation always agree.
#
# Two levels of fusion are implemented:
#
#   Intra-modal  — combines classifier + Mahalanobis scores within one modality
#     fuse_keystroke_scores(rf_score, mah_score)   → float [0, 1]
#     fuse_voice_scores(gbm_score, mah_score)       → float [0, 1]
#
#   Inter-modal  — combines keystroke + voice into one final decision
#     fuse_multimodal(keystroke_score, voice_score, ...) → dict

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  INTRA-MODAL FUSION WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

# Keystroke: RF is the primary classifier; Mahalanobis adds a geometric
# sanity check that prevents an overconfident RF score from granting access
# to a sample that lies far from the genuine cluster. 3:1 weighting keeps
# the decision primarily classifier-driven while catching obvious outliers.
KEYSTROKE_RF_WEIGHT  = 0.85
KEYSTROKE_MAH_WEIGHT = 0.15

# Voice: GBM is the primary classifier; Mahalanobis uses raw (non-CMVN)
# features so it retains per-speaker prosodic variation even after CMVN
# flattens the MFCC means. Slightly higher Mah weight than keystroke (0.30
# vs 0.25) because voice MFCCs are more session-variable and the geometric
# check adds more value as a stability anchor.
VOICE_GBM_WEIGHT = 0.70
VOICE_MAH_WEIGHT = 0.30


# Note: inter-modal (keystroke+voice) weights are NOT defined here — the
# `/fuse` endpoint in routers/auth.py uses case-dependent weights (Case A
# vs Case B) that depend on whether keystroke passed its own threshold.
# See that endpoint for the authoritative inter-modal weighting.


# ─────────────────────────────────────────────────────────────────────────────
#  INTRA-MODAL FUSION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def fuse_keystroke_scores(rf_score: float, mah_score: float) -> float:
    """
    Intra-modal fusion for keystroke dynamics.

    Combines the Random Forest classifier probability with a Mahalanobis
    distance-based proximity score into a single keystroke confidence value.

    Parameters
    ----------
    rf_score  : float [0, 1]
        RF posterior probability of the genuine class.
    mah_score : float [0, 1]
        Mahalanobis proximity score (1.0 = sample is at the profile centre;
        approaches 0.0 as the sample moves further from the genuine cluster).

    Returns
    -------
    float [0, 1]
        Fused keystroke confidence. Compare against the per-user threshold
        stored in the model pkl to make the accept/reject decision.
    """
    rf_score  = float(np.clip(rf_score,  0.0, 1.0))
    mah_score = float(np.clip(mah_score, 0.0, 1.0))
    return KEYSTROKE_RF_WEIGHT * rf_score + KEYSTROKE_MAH_WEIGHT * mah_score


def fuse_voice_scores(gbm_score: float, mah_score: float) -> float:
    """
    Intra-modal fusion for voice biometrics.

    Combines the Gradient Boosting Machine classifier probability with a
    Mahalanobis distance score (computed on raw, non-CMVN features) into a
    single voice confidence value.

    Parameters
    ----------
    gbm_score : float [0, 1]
        GBM posterior probability of the genuine class.
    mah_score : float [0, 1]
        Mahalanobis proximity score computed on raw (non-CMVN) features so
        that prosodic features retain their per-speaker discriminative value.

    Returns
    -------
    float [0, 1]
        Fused voice confidence. Compare against the per-user threshold
        stored in the model pkl to make the accept/reject decision.
    """
    gbm_score = float(np.clip(gbm_score, 0.0, 1.0))
    mah_score = float(np.clip(mah_score, 0.0, 1.0))
    return VOICE_GBM_WEIGHT * gbm_score + VOICE_MAH_WEIGHT * mah_score

