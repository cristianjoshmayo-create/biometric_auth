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
from typing import Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
#  INTRA-MODAL FUSION WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

# Keystroke: RF is the primary classifier; Mahalanobis adds a geometric
# sanity check that prevents an overconfident RF score from granting access
# to a sample that lies far from the genuine cluster. 3:1 weighting keeps
# the decision primarily classifier-driven while catching obvious outliers.
KEYSTROKE_RF_WEIGHT  = 0.75
KEYSTROKE_MAH_WEIGHT = 0.25

# Voice: GBM is the primary classifier; Mahalanobis uses raw (non-CMVN)
# features so it retains per-speaker prosodic variation even after CMVN
# flattens the MFCC means. Slightly higher Mah weight than keystroke (0.30
# vs 0.25) because voice MFCCs are more session-variable and the geometric
# check adds more value as a stability anchor.
VOICE_GBM_WEIGHT = 0.70
VOICE_MAH_WEIGHT = 0.30


# ─────────────────────────────────────────────────────────────────────────────
#  INTER-MODAL FUSION WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

# Equal weighting — keystroke captures fine motor habit; voice captures
# vocal-tract geometry. Neither modality dominates; 50/50 gives each equal
# influence on the final authentication decision.
MULTIMODAL_KEYSTROKE_WEIGHT = 0.50
MULTIMODAL_VOICE_WEIGHT     = 0.50


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


# ─────────────────────────────────────────────────────────────────────────────
#  INTER-MODAL FUSION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def fuse_multimodal(
    keystroke_score: Optional[float] = None,
    voice_score: Optional[float] = None,
    keystroke_threshold: float = 0.55,
    voice_threshold: float = 0.50,
    strategy: str = "weighted_sum",
) -> Dict:
    """
    Inter-modal (multimodal) fusion of keystroke and voice confidence scores.

    Handles partial availability gracefully: if one modality's model is not
    yet trained (score = None), fusion falls back to the available modality
    only rather than rejecting the attempt outright.

    Parameters
    ----------
    keystroke_score     : float [0, 1] or None
        Fused keystroke confidence from fuse_keystroke_scores(), or None if
        the keystroke model has not been trained yet.
    voice_score         : float [0, 1] or None
        Fused voice confidence from fuse_voice_scores(), or None if the
        voice model has not been trained yet.
    keystroke_threshold : float
        Per-user decision threshold for the keystroke modality (stored in pkl).
    voice_threshold     : float
        Per-user decision threshold for the voice modality (stored in pkl).
    strategy            : str
        Fusion rule — one of: 'weighted_sum' | 'and' | 'or' | 'min'.

    Strategies
    ----------
    weighted_sum (default)
        Normalised weighted average of available scores. Threshold is the
        proportionally combined per-modality threshold. Best for balanced
        systems where both modalities are reliable and enrolled.

    and
        BOTH modalities must independently exceed their thresholds. Most
        secure; rejects if either modality fails. Use when both modalities
        are fully enrolled and minimising FAR is the priority.

    or
        AT LEAST ONE modality must exceed its threshold. Most permissive;
        useful during partial enrolment or as a fallback mode. Increases FAR
        — use only when minimising FRR is the priority.

    min
        Fused score = min(available scores). The weakest modality dominates.
        Conservative baseline suitable for high-security scenarios.

    Returns
    -------
    dict with keys:
        fused_score            : float [0, 1]   final combined score
        decision               : bool            True = access granted
        strategy               : str             strategy used
        contributing_modalities: list[str]       modalities included
        individual_scores      : dict[str,float] per-modality scores
    """
    available: Dict[str, float] = {}
    if keystroke_score is not None:
        available["keystroke"] = float(np.clip(keystroke_score, 0.0, 1.0))
    if voice_score is not None:
        available["voice"] = float(np.clip(voice_score, 0.0, 1.0))

    if not available:
        return {
            "fused_score":             0.0,
            "decision":                False,
            "strategy":                strategy,
            "contributing_modalities": [],
            "individual_scores":       {},
            "reason":                  "no biometric scores available",
        }

    thresholds = {
        "keystroke": keystroke_threshold,
        "voice":     voice_threshold,
    }
    weights = {
        "keystroke": MULTIMODAL_KEYSTROKE_WEIGHT,
        "voice":     MULTIMODAL_VOICE_WEIGHT,
    }

    if strategy == "weighted_sum":
        total_w  = sum(weights[m] for m in available)
        fused    = sum(available[m] * weights[m] for m in available) / total_w
        thresh   = sum(thresholds[m] * weights[m] for m in available) / total_w
        decision = fused >= thresh

    elif strategy == "and":
        decisions = {m: available[m] >= thresholds[m] for m in available}
        decision  = all(decisions.values())
        fused     = sum(available.values()) / len(available)

    elif strategy == "or":
        decisions = {m: available[m] >= thresholds[m] for m in available}
        decision  = any(decisions.values())
        fused     = sum(available.values()) / len(available)

    elif strategy == "min":
        fused    = min(available.values())
        decision = fused >= min(thresholds[m] for m in available)

    else:
        raise ValueError(
            f"Unknown fusion strategy: '{strategy}'. "
            f"Valid options: weighted_sum, and, or, min."
        )

    return {
        "fused_score":             float(np.clip(fused, 0.0, 1.0)),
        "decision":                bool(decision),
        "strategy":                strategy,
        "contributing_modalities": list(available.keys()),
        "individual_scores":       dict(available),
    }