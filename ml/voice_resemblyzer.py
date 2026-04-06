# ml/voice_resemblyzer.py
# Resemblyzer-based speaker verification for biometric authentication.
#
# How it works:
#   Resemblyzer ships a pretrained speaker encoder (LSTM trained with GE2E loss
#   on thousands of speakers) that converts any speech segment into a 256-dim
#   embedding vector. Speakers cluster tightly in this space — cosine similarity
#   between two recordings of the same voice is typically 0.80–0.95, while
#   two different speakers score 0.40–0.70.
#
# Enrollment:
#   Each recording produces one 256-dim embedding. After all enrollment
#   recordings are done, their mean is stored as the user's voice profile.
#   Mean averaging is the standard approach and is robust to minor session
#   variation (different mic position, ambient noise level).
#
# Authentication:
#   The login embedding is compared to the stored mean profile using cosine
#   similarity. A similarity ≥ threshold → granted.
#
# Why this is better than the GBM approach for your use case:
#   • Pretrained on thousands of speakers — no per-user training needed
#   • Works well with as few as 1–3 enrollment recordings
#   • Session-robust — the embedding captures vocal-tract geometry, not
#     acoustic conditions, so it generalises across microphones and rooms
#   • Fast inference — one forward pass (~10ms on CPU)
#
# Install:
#   pip install resemblyzer
#
# The pretrained model (~17 MB) downloads automatically on first use.

import os
import sys
import pickle
import numpy as np
from typing import List, Optional

# ── Resemblyzer import with friendly error ────────────────────────────────────
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

# Cosine similarity threshold.
# Values at or above this → genuine user accepted.
# Resemblyzer literature uses 0.75 as a reliable operating point.
# Lowered slightly to 0.72 to accommodate minor cross-session variation
# (e.g. slightly different mic distance or background noise level).
DEFAULT_THRESHOLD = 0.72

EMBEDDING_DIM = 256   # Resemblyzer always outputs 256-dim vectors


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _safe_filename(username: str) -> str:
    return username.replace("@", "_at_").replace(".", "_").replace(" ", "_")


def _model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _profile_path(username: str) -> str:
    return os.path.join(_model_dir(), f"{_safe_filename(username)}_voice_resemblyzer.pkl")


def _load_profile(username: str) -> Optional[dict]:
    path = _profile_path(username)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_profile(data: dict):
    os.makedirs(_model_dir(), exist_ok=True)
    with open(_profile_path(data["username"]), "wb") as f:
        pickle.dump(data, f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return float(np.dot(a, b) / norm)


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_embedding(audio: np.ndarray, sr: int = 16000) -> Optional[List[float]]:
    """
    Convert a raw audio waveform to a 256-dim Resemblyzer speaker embedding.

    Parameters
    ----------
    audio : np.ndarray
        Raw waveform samples (float32, mono).
    sr    : int
        Sample rate. Must be 16000 for Resemblyzer.

    Returns
    -------
    list of 256 floats, or None if Resemblyzer is not installed or audio is bad.
    """
    if not RESEMBLYZER_AVAILABLE:
        print("⚠  resemblyzer not installed — run: pip install resemblyzer")
        return None

    try:
        # Resemblyzer expects 16 kHz float32 mono — preprocess_wav handles
        # resampling and trimming silence automatically.
        wav = preprocess_wav(audio, source_sr=sr)

        if len(wav) < sr * 0.5:
            print("  ⚠  Audio too short for Resemblyzer (< 0.5s after preprocessing)")
            return None

        encoder = VoiceEncoder(device="cpu")
        embedding = encoder.embed_utterance(wav)   # shape: (256,)
        return embedding.tolist()

    except Exception as e:
        print(f"  ⚠  Resemblyzer embedding failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  ENROLLMENT
# ─────────────────────────────────────────────────────────────────────────────

def save_enrollment(username: str, embedding: List[float]) -> dict:
    """
    Append a new enrollment embedding to the user's profile and recompute
    the mean embedding used for authentication.

    Called automatically by enroll.py after each voice enrollment recording.
    No separate "training" step needed — the profile updates immediately.

    Parameters
    ----------
    username  : str   user's email / username
    embedding : list  256-dim embedding from extract_embedding()

    Returns
    -------
    dict with n_enrollment and updated profile info.
    """
    emb = np.array(embedding, dtype=np.float32)

    if len(emb) != EMBEDDING_DIM:
        return {"error": f"Expected {EMBEDDING_DIM}-dim embedding, got {len(emb)}"}

    # Load existing profile or start fresh
    profile = _load_profile(username) or {
        "username":      username,
        "embeddings":    [],
        "mean_embedding": None,
        "threshold":     DEFAULT_THRESHOLD,
        "n_enrollment":  0,
    }

    profile["embeddings"].append(emb.tolist())
    profile["n_enrollment"] = len(profile["embeddings"])

    # Recompute mean profile embedding
    emb_matrix = np.array(profile["embeddings"], dtype=np.float32)  # (N, 256)
    mean_emb   = emb_matrix.mean(axis=0)

    # L2-normalise the mean so cosine similarity is numerically stable
    norm = np.linalg.norm(mean_emb)
    mean_emb = mean_emb / (norm + 1e-10)

    profile["mean_embedding"] = mean_emb.tolist()
    _save_profile(profile)

    print(f"  ✅ Resemblyzer: saved enrollment #{profile['n_enrollment']} for '{username}'")
    return {
        "success":      True,
        "n_enrollment": profile["n_enrollment"],
        "threshold":    profile["threshold"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

def predict_voice(username: str, embedding: List[float]) -> dict:
    """
    Authenticate a voice sample against the stored Resemblyzer profile.

    Parameters
    ----------
    username  : str   user's email / username
    embedding : list  256-dim embedding of the login attempt's audio

    Returns
    -------
    dict with:
        match       : bool   — True = genuine user accepted
        similarity  : float  — cosine similarity [0, 1]
        threshold   : float  — decision threshold
        confidence  : float  — similarity as a percentage (for logging)
        fused_score : float  — same as similarity × 100 (matches old API shape)
    """
    if not embedding or len(embedding) == 0:
        return {
            "match":       False,
            "similarity":  0.0,
            "threshold":   DEFAULT_THRESHOLD,
            "confidence":  0.0,
            "fused_score": 0.0,
            "error":       "No Resemblyzer embedding in payload. Re-enroll.",
        }

    profile = _load_profile(username)
    if profile is None:
        return {
            "match":       False,
            "similarity":  0.0,
            "threshold":   DEFAULT_THRESHOLD,
            "confidence":  0.0,
            "fused_score": 0.0,
            "error":       f"No Resemblyzer profile for '{username}'. Enroll first.",
        }

    mean_emb  = np.array(profile["mean_embedding"], dtype=np.float32)
    login_emb = np.array(embedding,                 dtype=np.float32)

    # L2-normalise login embedding before comparison
    norm = np.linalg.norm(login_emb)
    login_emb = login_emb / (norm + 1e-10)

    similarity = cosine_similarity(login_emb, mean_emb)
    threshold  = profile.get("threshold", DEFAULT_THRESHOLD)
    match      = similarity >= threshold

    print(
        f"\n  Resemblyzer '{username}': "
        f"similarity={similarity:.4f}  threshold={threshold:.2f}  "
        f"n_enrolled={profile['n_enrollment']}  "
        f"→ {'✅ MATCH' if match else '❌ REJECT'}"
    )

    return {
        "match":       bool(match),
        "similarity":  round(similarity, 4),
        "threshold":   round(threshold, 2),
        "confidence":  round(similarity * 100, 2),
        "fused_score": round(similarity * 100, 2),   # keeps shape identical to CNN result
        "n_enrollment": profile["n_enrollment"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI — check a user's profile
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect a user's Resemblyzer profile")
    parser.add_argument("username", nargs="?", default=None)
    args   = parser.parse_args()

    username = args.username or input("Username: ").strip()
    profile  = _load_profile(username)

    if profile is None:
        print(f"No Resemblyzer profile found for '{username}'.")
    else:
        print(f"\nResemblyer profile — {username}")
        print(f"  Enrolled recordings : {profile['n_enrollment']}")
        print(f"  Threshold           : {profile['threshold']}")
        mean = np.array(profile["mean_embedding"])
        print(f"  Mean embedding norm : {np.linalg.norm(mean):.4f} (should be ~1.0)")
        print(f"  Embedding dim       : {len(mean)}")