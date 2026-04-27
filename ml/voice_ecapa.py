# ml/voice_ecapa.py
# ECAPA-TDNN speaker verification via SpeechBrain pretrained model.
#
# ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation
# Time-Delay Neural Network) is a state-of-the-art speaker verification CNN
# pretrained on VoxCeleb (7,000+ speakers, 1M+ utterances). It consistently
# outperforms both traditional MFCC+GBM approaches and simpler LSTMs like
# Resemblyzer on standard speaker verification benchmarks (VoxCeleb1 EER ~0.8%).
#
# How it works in this system:
#   Enrollment : each recording → 192-dim embedding → stored in pkl profile
#                Profile = L2-normalised mean of all enrollment embeddings
#   Auth       : login recording → 192-dim embedding → cosine similarity vs profile
#                similarity ≥ threshold → granted
#
# No per-user training. The pretrained weights already encode speaker identity
# for any voice — new users just add their profile, no retraining needed.
#
# The model (~80 MB) downloads from HuggingFace automatically on first use.
# After that it's cached locally in ml/pretrained_models/ecapa-tdnn/.
#
# Install:
#   pip install speechbrain
#   pip install torch   (if not already installed)

import os
import sys
import pickle
import numpy as np
from typing import List, Optional

# ── Windows symlink fix — applied at MODULE LOAD TIME, before any SpeechBrain ──
# import touches the filesystem.
#
# Root cause of WinError 1314:
#   HuggingFace hub + SpeechBrain's Pretrainer both default to LocalStrategy.SYMLINK.
#   Creating symlinks on Windows requires Developer Mode or admin rights.
#
# Fix — three layers to be 100% sure:
#   1. Env var: tells HF hub to copy instead of symlink.
#   2. Monkey-patch fetching.fetch() default arg → LocalStrategy.COPY.
#   3. Remap LocalStrategy.SYMLINK → COPY so any caller that passes the enum
#      value explicitly (including Pretrainer.collect) still ends up copying.
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "warning"

def _apply_speechbrain_copy_patch():
    """
    Force SpeechBrain to use file-copy instead of symlinks on Windows.
    Must be called before EncoderClassifier (or any SpeechBrain inference class)
    is imported, because Pretrainer reads the strategy at instantiation time.
    """
    try:
        import speechbrain.utils.fetching as _sb_fetch

        _COPY = _sb_fetch.LocalStrategy.COPY

        # Layer 2 — patch the fetch() function so its default is COPY
        _orig_fetch = _sb_fetch.fetch

        def _patched_fetch(filename, source, savedir=None, save_filename=None,
                           local_strategy=_COPY,
                           fetch_config=_sb_fetch.FetchConfig()):
            return _orig_fetch(filename, source, savedir=savedir,
                               save_filename=save_filename,
                               local_strategy=_COPY,   # always COPY, ignore caller's value
                               fetch_config=fetch_config)

        _sb_fetch.fetch = _patched_fetch

        # Layer 3 — remap the SYMLINK enum member to COPY so Pretrainer.collect()
        # which passes local_strategy=LocalStrategy.SYMLINK explicitly also copies.
        try:
            _sb_fetch.LocalStrategy.SYMLINK = _COPY
        except (AttributeError, TypeError):
            pass  # read-only enum on some versions — layer 2 is sufficient

        # Propagate patch to parameter_transfer which has its own 'fetch' reference
        try:
            import speechbrain.utils.parameter_transfer as _sb_pt
            if hasattr(_sb_pt, "fetch"):
                _sb_pt.fetch = _patched_fetch
            # Also patch Pretrainer.collect default kwarg if accessible
            import inspect, functools
            if hasattr(_sb_pt, "Pretrainer"):
                _orig_collect = _sb_pt.Pretrainer.collect

                @functools.wraps(_orig_collect)
                async def _patched_collect(self, *args, **kwargs):
                    kwargs.setdefault("local_strategy", _COPY)
                    kwargs["local_strategy"] = _COPY
                    return await _orig_collect(self, *args, **kwargs)

                _sb_pt.Pretrainer.collect = _patched_collect
        except Exception:
            pass  # non-critical; layers 1+2 already cover most cases

        print("  SpeechBrain patched → LocalStrategy.COPY (no symlinks)")
        return True
    except Exception as e:
        print(f"  SpeechBrain patch skipped: {e}")
        return False

# Apply the patch NOW, before EncoderClassifier is imported below.
_apply_speechbrain_copy_patch()

# ── SpeechBrain import with friendly error ────────────────────────────────────
try:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    ECAPA_AVAILABLE = True
except ImportError:
    ECAPA_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Cosine similarity threshold.
# ECAPA-TDNN embeddings are tightly clustered per speaker so 0.68 is a
# conservative but reliable operating point for short passphrases.
# Bumped from 0.65 → 0.68 in Phase 2 because max-cosine over a multi-slot
# profile inflates scores slightly compared with cosine-vs-mean.
DEFAULT_THRESHOLD = 0.68

EMBEDDING_DIM = 192   # ECAPA-TDNN output dimension

# Multi-embedding profile caps (Phase 2: adaptive cross-device coverage).
# Profile = up to MAX_ENROLL enrollment slots + up to MAX_ADAPTIVE adaptive
# slots. Enrollment slots are protected; adaptive slots use FIFO eviction
# so the most recent device/condition embeddings dominate.
MAX_ENROLL   = 5
MAX_ADAPTIVE = 5

# Model is cached here after the first download so it survives server restarts.
_PRETRAINED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pretrained_models", "ecapa-tdnn"
)


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _safe_filename(username: str) -> str:
    return username.replace("@", "_at_").replace(".", "_").replace(" ", "_")

def _model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def _profile_path(username: str) -> str:
    return os.path.join(_model_dir(), f"{_safe_filename(username)}_voice_ecapa.pkl")

def _load_profile(username: str) -> Optional[dict]:
    path = _profile_path(username)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        profile = pickle.load(f)
    # Phase 2 backward-compat migration: old profiles store `embeddings` as
    # a flat list of float-lists. Convert each to a tagged slot dict so we
    # can distinguish enrollment from adaptive entries and apply FIFO
    # eviction on the adaptive ones only.
    embs = profile.get("embeddings", [])
    if embs and isinstance(embs[0], list):
        from datetime import datetime as _dt, timezone as _tz
        ts = _dt.now(_tz.utc).isoformat()
        profile["embeddings"] = [
            {"vec": e, "source": "enrollment", "ts": ts} for e in embs
        ]
        _save_profile(profile)
    return profile

def _save_profile(data: dict):
    os.makedirs(_model_dir(), exist_ok=True)
    with open(_profile_path(data["username"]), "wb") as f:
        pickle.dump(data, f)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return float(np.dot(a, b) / norm)


# ─────────────────────────────────────────────────────────────────────────────
#  ENCODER (singleton — loaded once, reused for every call)
# ─────────────────────────────────────────────────────────────────────────────

_encoder_instance = None

def _get_encoder() -> Optional["EncoderClassifier"]:
    """
    Load the ECAPA-TDNN encoder once and cache it in memory.
    Downloads the pretrained model from HuggingFace on first call (~80 MB).
    Subsequent calls return the cached instance instantly.
    """
    global _encoder_instance
    if _encoder_instance is not None:
        return _encoder_instance

    if not ECAPA_AVAILABLE:
        print("⚠  speechbrain not installed.")
        print("   Run: pip install speechbrain")
        return None

    try:
        print("  Loading ECAPA-TDNN pretrained model ...")
        print("  (Downloads ~80 MB from HuggingFace on first run — cached after that)")

        # Download / locate the model snapshot in the HuggingFace cache.
        try:
            from huggingface_hub import snapshot_download
            savedir = snapshot_download(repo_id="speechbrain/spkrec-ecapa-voxceleb")
            print(f"  Model cache: {savedir}")
        except Exception as hf_err:
            print(f"  snapshot_download failed ({hf_err}), using local dir")
            savedir = _PRETRAINED_DIR
            os.makedirs(savedir, exist_ok=True)

        _encoder_instance = EncoderClassifier.from_hparams(
            source=savedir,
            savedir=savedir,
            run_opts={"device": "cpu"},
        )
        print("  ✅ ECAPA-TDNN model loaded")
        return _encoder_instance
    except Exception as e:
        print(f"  ❌ Failed to load ECAPA-TDNN model: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_embedding(audio: np.ndarray, sr: int = 16000) -> Optional[List[float]]:
    """
    Convert a raw audio waveform to a 192-dim ECAPA-TDNN speaker embedding.

    Parameters
    ----------
    audio : np.ndarray
        Raw waveform samples, float32, mono. Must be at 16 kHz.
    sr    : int
        Sample rate. ECAPA-TDNN expects 16000 Hz.

    Returns
    -------
    List of 192 floats, or None if the model is unavailable or audio is bad.
    """
    encoder = _get_encoder()
    if encoder is None:
        return None

    try:
        audio_f32 = np.array(audio, dtype=np.float32)

        if len(audio_f32) < sr * 0.5:
            print("  ⚠  Audio too short for ECAPA-TDNN (< 0.5s)")
            return None

        # Normalise amplitude so very quiet recordings still produce good embeddings
        max_amp = np.abs(audio_f32).max()
        if max_amp > 1e-6:
            audio_f32 = audio_f32 / max_amp

        # SpeechBrain expects (batch, time) float32 tensor + relative lengths
        wav_tensor = torch.FloatTensor(audio_f32).unsqueeze(0)  # (1, T)
        wav_lens   = torch.ones(1)                               # relative length = 1.0

        with torch.no_grad():
            embeddings = encoder.encode_batch(wav_tensor, wav_lens)
            # shape: (1, 1, 192)  →  squeeze to (192,)
            embedding = embeddings.squeeze().cpu().numpy()

        if embedding.shape[0] != EMBEDDING_DIM:
            print(f"  ⚠  Unexpected embedding dim {embedding.shape[0]} (expected {EMBEDDING_DIM})")
            return None

        return embedding.tolist()

    except Exception as e:
        print(f"  ⚠  ECAPA-TDNN embedding failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  ENROLLMENT
# ─────────────────────────────────────────────────────────────────────────────

def _new_profile(username: str) -> dict:
    return {
        "username":       username,
        "embeddings":     [],            # list of {vec, source, ts}
        "mean_embedding": None,          # kept for diagnostics; auth uses max-cosine
        "threshold":      DEFAULT_THRESHOLD,
        "n_enrollment":   0,
    }

def _recompute_mean(profile: dict) -> None:
    """Refresh the diagnostic mean_embedding + n_enrollment counters."""
    slots = profile.get("embeddings", [])
    if not slots:
        profile["mean_embedding"] = None
        profile["n_enrollment"]   = 0
        return
    vecs = np.array([s["vec"] for s in slots], dtype=np.float32)
    mean = vecs.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-10)
    profile["mean_embedding"] = mean.tolist()
    profile["n_enrollment"]   = sum(1 for s in slots if s.get("source") == "enrollment")

def save_enrollment(username: str, embedding: List[float]) -> dict:
    """
    Append a new ENROLLMENT slot to the multi-embedding voice profile.

    Called by enroll.py after each voice recording. No separate training step
    needed — the profile updates immediately so auth works after 1 recording.
    Enrollment slots are protected (never evicted by the adaptive FIFO).
    """
    emb = np.array(embedding, dtype=np.float32)
    if len(emb) != EMBEDDING_DIM:
        return {"error": f"Expected {EMBEDDING_DIM}-dim embedding, got {len(emb)}"}

    profile = _load_profile(username) or _new_profile(username)

    from datetime import datetime as _dt, timezone as _tz
    ts = _dt.now(_tz.utc).isoformat()

    enroll_slots = [s for s in profile["embeddings"] if s.get("source") == "enrollment"]
    other_slots  = [s for s in profile["embeddings"] if s.get("source") != "enrollment"]
    enroll_slots.append({"vec": emb.tolist(), "source": "enrollment", "ts": ts})
    if len(enroll_slots) > MAX_ENROLL:
        # Cap enrollment slots too — keep most recent.
        enroll_slots = enroll_slots[-MAX_ENROLL:]
    profile["embeddings"] = enroll_slots + other_slots

    _recompute_mean(profile)
    _save_profile(profile)

    n_total = len(profile["embeddings"])
    print(f"  ✅ ECAPA: enrollment slot saved for '{username}' "
          f"(enroll={profile['n_enrollment']}, total={n_total})")
    return {
        "success":      True,
        "n_enrollment": profile["n_enrollment"],
        "threshold":    profile["threshold"],
    }


def append_adaptive(username: str, embedding: List[float]) -> dict:
    """
    Append an ADAPTIVE slot from a confidently-passed login session.

    Used to silently register new devices/conditions: when fusion (or
    high-margin voice) confirms identity, the login embedding is added so
    the next login on the same device hits it via max-cosine and passes
    voice alone.

    FIFO eviction at MAX_ADAPTIVE; enrollment slots are never touched.
    Caller is responsible for the margin gate (don't reinforce noise).
    """
    emb = np.array(embedding, dtype=np.float32)
    if len(emb) != EMBEDDING_DIM:
        return {"error": f"Expected {EMBEDDING_DIM}-dim embedding, got {len(emb)}"}

    profile = _load_profile(username)
    if profile is None:
        return {"error": f"No ECAPA profile for '{username}' — cannot append adaptive."}

    from datetime import datetime as _dt, timezone as _tz
    ts = _dt.now(_tz.utc).isoformat()

    enroll_slots   = [s for s in profile["embeddings"] if s.get("source") == "enrollment"]
    adaptive_slots = [s for s in profile["embeddings"] if s.get("source") != "enrollment"]
    adaptive_slots.append({"vec": emb.tolist(), "source": "adaptive", "ts": ts})
    if len(adaptive_slots) > MAX_ADAPTIVE:
        # FIFO: drop the oldest adaptive slot.
        adaptive_slots = adaptive_slots[-MAX_ADAPTIVE:]
    profile["embeddings"] = enroll_slots + adaptive_slots

    _recompute_mean(profile)
    _save_profile(profile)

    print(f"  💾 ECAPA: adaptive slot appended for '{username}' "
          f"(enroll={len(enroll_slots)}, adaptive={len(adaptive_slots)})")
    return {
        "success":   True,
        "n_enroll":  len(enroll_slots),
        "n_adaptive": len(adaptive_slots),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

def predict_voice(username: str, embedding: List[float]) -> dict:
    """
    Authenticate a voice sample against the stored ECAPA-TDNN profile.

    Parameters
    ----------
    username  : str
    embedding : list of 192 floats from extract_embedding() at login time

    Returns
    -------
    dict with:
        match       : bool   — True = genuine speaker accepted
        similarity  : float  — cosine similarity [0, 1]
        threshold   : float  — current decision threshold
        confidence  : float  — similarity as percentage (for display)
        fused_score : float  — same as confidence (keeps API shape consistent)
    """
    if not embedding or len(embedding) == 0:
        return {
            "match":       False,
            "similarity":  0.0,
            "threshold":   DEFAULT_THRESHOLD,
            "confidence":  0.0,
            "fused_score": 0.0,
            "error":       "No ECAPA embedding in payload. Re-enroll your voice.",
        }

    profile = _load_profile(username)
    if profile is None:
        return {
            "match":       False,
            "similarity":  0.0,
            "threshold":   DEFAULT_THRESHOLD,
            "confidence":  0.0,
            "fused_score": 0.0,
            "error":       f"No ECAPA profile for '{username}'. Enroll voice first.",
        }

    slots = profile.get("embeddings", [])
    if not slots:
        return {
            "match":       False,
            "similarity":  0.0,
            "threshold":   DEFAULT_THRESHOLD,
            "confidence":  0.0,
            "fused_score": 0.0,
            "error":       f"No embeddings stored for '{username}'.",
        }

    login_emb = np.array(embedding, dtype=np.float32)
    login_emb = login_emb / (np.linalg.norm(login_emb) + 1e-10)

    # Phase 2: max cosine over every stored slot. The slot pool covers
    # different devices / acoustic conditions; the device-matched slot
    # wins, so cross-device drift no longer drags the score down through
    # averaging.
    best_sim    = -1.0
    best_source = None
    for s in slots:
        v = np.array(s["vec"], dtype=np.float32)
        sim = cosine_similarity(login_emb, v)
        if sim > best_sim:
            best_sim    = sim
            best_source = s.get("source", "?")

    n_enroll   = sum(1 for s in slots if s.get("source") == "enrollment")
    n_adaptive = len(slots) - n_enroll

    threshold = profile.get("threshold", DEFAULT_THRESHOLD)
    # Soften threshold for cold-start users with a thin enrollment set;
    # max-of-N already absorbs the +0.02 bump that the old mean-based
    # logic applied at n>=5, so we no longer add it.
    if n_enroll <= 2:
        threshold -= 0.05

    match = best_sim >= threshold

    print(
        f"\n  ECAPA-TDNN '{username}': "
        f"max_sim={best_sim:.4f} (best_slot={best_source})  "
        f"threshold={threshold:.2f}  "
        f"slots=enroll:{n_enroll}+adapt:{n_adaptive}  "
        f"→ {'MATCH' if match else 'REJECT'}"
    )

    return {
        "match":        bool(match),
        "similarity":   round(best_sim, 4),
        "threshold":    round(threshold, 2),
        "confidence":   round(best_sim * 100, 2),
        "fused_score":  round(best_sim * 100, 2),
        "n_enrollment": n_enroll,
        "n_adaptive":   n_adaptive,
        "best_slot":    best_source,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI — inspect or delete a user profile
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect a user's ECAPA-TDNN profile")
    parser.add_argument("username", nargs="?", default=None)
    parser.add_argument("--delete", action="store_true", help="Delete the profile")
    args = parser.parse_args()

    username = args.username or input("Username: ").strip()
    profile  = _load_profile(username)

    if profile is None:
        print(f"No ECAPA profile found for '{username}'.")
    elif args.delete:
        path = _profile_path(username)
        os.remove(path)
        print(f"Deleted ECAPA profile for '{username}'.")
    else:
        print(f"\nECAPA-TDNN profile — {username}")
        print(f"  Enrolled recordings : {profile['n_enrollment']}")
        print(f"  Threshold           : {profile['threshold']}")
        mean = np.array(profile["mean_embedding"])
        print(f"  Mean embedding norm : {np.linalg.norm(mean):.4f}  (should be ~1.0)")
        print(f"  Embedding dim       : {len(mean)}  (should be {EMBEDDING_DIM})")