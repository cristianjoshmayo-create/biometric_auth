# ml/voice_azure.py
# Azure Cognitive Services Speaker Recognition — drop-in replacement for ECAPA-TDNN.
#
# Azure Speaker Recognition API:
#   - Enrollment : POST audio to Azure → they build a voice profile in the cloud
#   - Auth       : POST audio to Azure → they return a confidence score
#   - No local ML model needed — no PyTorch, no SpeechBrain, no DLL issues
#   - Free tier  : 10,000 transactions/month
#
# Setup (one time):
#   1. Go to https://portal.azure.com
#   2. Create a resource → search "Speaker Recognition"
#   3. Select Free tier (F0) → Create
#   4. Copy the API Key and Region from "Keys and Endpoint"
#   5. Set environment variables (or paste directly below):
#        AZURE_SPEECH_KEY=your_key_here
#        AZURE_SPEECH_REGION=your_region_here   (e.g. eastus, southeastasia)
#
# How it works:
#   Enrollment : each WAV recording → POST to Azure → profile stored in Azure cloud
#                profile_id (GUID) saved locally in ml/models/<user>_voice_azure.pkl
#   Auth       : login WAV → POST to Azure → returns "Accept"/"Reject" + confidence
#
# Install:
#   pip install requests

import os
import sys
import pickle
import requests
import numpy as np
import struct
import wave
import io
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
# Set these environment variables, or hardcode them here for development.
AZURE_SPEECH_KEY    = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "")

# Minimum enrollment samples before auth is allowed
MIN_ENROLLMENT_SAMPLES = 3

# ── API endpoints ──────────────────────────────────────────────────────────────
def _base_url():
    return f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/speaker/verification/v2.0"

def _headers():
    return {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/octet-stream",
    }

def _json_headers():
    return {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/json",
    }

# ── Profile storage ────────────────────────────────────────────────────────────
def _safe_filename(username: str) -> str:
    return username.replace("@", "_at_").replace(".", "_").replace(" ", "_")

def _model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def _profile_path(username: str) -> str:
    return os.path.join(_model_dir(), f"{_safe_filename(username)}_voice_azure.pkl")

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

# ── Audio conversion ───────────────────────────────────────────────────────────
def _audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Convert float32 numpy audio array to 16-bit PCM WAV bytes for Azure."""
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sr)      # 16000 Hz
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()

# ── Availability check ─────────────────────────────────────────────────────────
def is_available() -> bool:
    """Return True if Azure credentials are configured."""
    return bool(AZURE_SPEECH_KEY and AZURE_SPEECH_REGION)

# ── Profile management ─────────────────────────────────────────────────────────
def create_profile(username: str) -> Optional[str]:
    """
    Create a new Azure Speaker Verification profile for this user.
    Returns the profile_id (GUID string) or None on failure.
    """
    if not is_available():
        print("  ⚠  Azure credentials not set (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION)")
        return None

    url = f"{_base_url()}/text-independent/profiles"
    body = {"locale": "en-us"}
    try:
        r = requests.post(url, headers=_json_headers(), json=body, timeout=15)
        r.raise_for_status()
        profile_id = r.json()["profileId"]
        print(f"  ✅ Azure: created profile {profile_id} for '{username}'")
        return profile_id
    except Exception as e:
        print(f"  ❌ Azure: failed to create profile: {e}")
        return None

def delete_profile(username: str) -> bool:
    """Delete the Azure cloud profile for this user."""
    profile = _load_profile(username)
    if not profile:
        return False
    profile_id = profile.get("profile_id")
    if not profile_id:
        return False
    try:
        url = f"{_base_url()}/text-independent/profiles/{profile_id}"
        r = requests.delete(url, headers=_json_headers(), timeout=15)
        r.raise_for_status()
        os.remove(_profile_path(username))
        print(f"  ✅ Azure: deleted profile for '{username}'")
        return True
    except Exception as e:
        print(f"  ⚠  Azure: delete failed: {e}")
        return False

# ── Enrollment ─────────────────────────────────────────────────────────────────
def save_enrollment(username: str, audio: np.ndarray, sr: int = 16000) -> dict:
    """
    Enroll a voice recording with Azure Speaker Recognition.

    Called by enroll.py after each voice recording. Audio is sent as WAV
    to Azure who builds and updates the voice profile in the cloud.
    After MIN_ENROLLMENT_SAMPLES recordings Azure marks the profile as enrolled.

    Parameters
    ----------
    username : str
    audio    : np.ndarray  float32 mono waveform at `sr` Hz
    sr       : int         sample rate (default 16000)

    Returns
    -------
    dict with success flag, enrollment count, and enrollment status.
    """
    if not is_available():
        return {"error": "Azure credentials not configured. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION."}

    # Load or create profile
    profile = _load_profile(username)
    if profile is None:
        profile_id = create_profile(username)
        if not profile_id:
            return {"error": "Failed to create Azure voice profile."}
        profile = {
            "username":       username,
            "profile_id":     profile_id,
            "n_enrollment":   0,
            "enrolled":       False,
        }

    profile_id = profile["profile_id"]

    # Convert audio and POST to Azure
    wav_bytes = _audio_to_wav_bytes(audio, sr)
    url = f"{_base_url()}/text-independent/profiles/{profile_id}/enrollments"
    try:
        r = requests.post(url, headers=_headers(), data=wav_bytes, timeout=30)
        r.raise_for_status()
        result = r.json()

        remaining = result.get("remainingEnrollmentsSpeechLength", 0)
        status    = result.get("enrollmentStatus", "unknown")
        speech_s  = result.get("enrollmentsSpeechLength", 0)

        profile["n_enrollment"] += 1
        profile["enrolled"]      = (status == "Enrolled")
        _save_profile(profile)

        print(f"  ✅ Azure enrollment #{profile['n_enrollment']} for '{username}' "
              f"| status={status}  speech={speech_s:.1f}s  remaining={remaining:.1f}s")

        return {
            "success":          True,
            "n_enrollment":     profile["n_enrollment"],
            "enrolled":         profile["enrolled"],
            "status":           status,
            "speech_seconds":   speech_s,
            "remaining_seconds": remaining,
        }

    except requests.HTTPError as e:
        # 403 = not enough speech yet — not a hard error
        msg = f"Azure enrollment HTTP error: {e.response.status_code} {e.response.text}"
        print(f"  ⚠  {msg}")
        profile["n_enrollment"] += 1
        _save_profile(profile)
        return {"error": msg, "n_enrollment": profile["n_enrollment"]}
    except Exception as e:
        print(f"  ❌ Azure enrollment failed: {e}")
        return {"error": str(e)}

# ── Authentication ─────────────────────────────────────────────────────────────
def predict_voice(username: str, audio: np.ndarray, sr: int = 16000) -> dict:
    """
    Authenticate a voice sample against the Azure Speaker Recognition profile.

    Parameters
    ----------
    username : str
    audio    : np.ndarray  float32 mono waveform at `sr` Hz
    sr       : int         sample rate (default 16000)

    Returns
    -------
    dict with:
        match       : bool   — True = genuine speaker accepted
        confidence  : float  — Azure confidence score 0.0–1.0
        fused_score : float  — confidence as percentage (for UI display)
        result      : str    — "Accept" or "Reject" from Azure
    """
    if not is_available():
        return {
            "match": False, "confidence": 0.0, "fused_score": 0.0,
            "error": "Azure credentials not configured.",
        }

    profile = _load_profile(username)
    if profile is None:
        return {
            "match": False, "confidence": 0.0, "fused_score": 0.0,
            "error": f"No Azure profile for '{username}'. Please enroll first.",
        }

    if not profile.get("enrolled"):
        return {
            "match": False, "confidence": 0.0, "fused_score": 0.0,
            "error": f"Azure profile for '{username}' not yet fully enrolled. "
                     f"Need more speech audio ({profile.get('n_enrollment', 0)} recordings so far).",
        }

    profile_id = profile["profile_id"]
    wav_bytes   = _audio_to_wav_bytes(audio, sr)
    url = f"{_base_url()}/text-independent/profiles/{profile_id}/verify"

    try:
        r = requests.post(url, headers=_headers(), data=wav_bytes, timeout=30)
        r.raise_for_status()
        result = r.json()

        az_result  = result.get("result", "Reject")       # "Accept" or "Reject"
        confidence = float(result.get("score", 0.0))      # 0.0 – 1.0
        match      = (az_result == "Accept")

        print(f"\n  Azure Speaker Verify '{username}': "
              f"result={az_result}  score={confidence:.4f}  "
              f"→ {'✅ MATCH' if match else '❌ REJECT'}")

        return {
            "match":        match,
            "confidence":   round(confidence, 4),
            "fused_score":  round(confidence * 100, 2),
            "result":       az_result,
            "n_enrollment": profile.get("n_enrollment", 0),
        }

    except requests.HTTPError as e:
        msg = f"Azure verify HTTP {e.response.status_code}: {e.response.text}"
        print(f"  ❌ {msg}")
        return {"match": False, "confidence": 0.0, "fused_score": 0.0, "error": msg}
    except Exception as e:
        print(f"  ❌ Azure verify failed: {e}")
        return {"match": False, "confidence": 0.0, "fused_score": 0.0, "error": str(e)}


# ── CLI — inspect or delete a user profile ────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect an Azure voice profile")
    parser.add_argument("username", nargs="?", default=None)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    username = args.username or input("Username: ").strip()
    profile  = _load_profile(username)

    if profile is None:
        print(f"No Azure profile found for '{username}'.")
    elif args.delete:
        delete_profile(username)
    else:
        print(f"\nAzure Speaker Recognition profile — {username}")
        print(f"  Profile ID     : {profile['profile_id']}")
        print(f"  Enrollments    : {profile['n_enrollment']}")
        print(f"  Enrolled       : {profile['enrolled']}")