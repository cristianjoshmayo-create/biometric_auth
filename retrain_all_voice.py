"""
retrain_all_voice.py
─────────────────────────────────────────────────────────────────────────────
Retrains voice models for ALL enrolled users in the database.
Run this after upgrading scikit-learn to rebuild stale .pkl files.

Usage
─────
  python retrain_all_voice.py

Place this file in the root of biometric_auth/ (same level as retrain_all_keystroke.py)
"""

import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_PATH = os.path.join(ROOT_DIR, "backend")
ML_PATH      = os.path.join(ROOT_DIR, "ml")
sys.path.insert(0, BACKEND_PATH)
sys.path.insert(0, ML_PATH)

import time
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from database.db import SessionLocal
from database.models import User, VoiceTemplate
from sqlalchemy import func
from train_voice_cnn import train_voice_model


def main():
    print(f"\n{'═'*60}")
    print(f"  RETRAIN ALL VOICE MODELS")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}\n")

    db = SessionLocal()
    try:
        # Only retrain users who have at least 1 voice sample
        voice_counts = dict(
            db.query(VoiceTemplate.user_id, func.count(VoiceTemplate.id))
              .group_by(VoiceTemplate.user_id)
              .all()
        )
        users = db.query(User).order_by(User.id).all()
        eligible = [u for u in users if voice_counts.get(u.id, 0) >= 1]
    finally:
        db.close()

    print(f"  Found {len(eligible)} user(s) with voice enrollment data:\n")
    for i, u in enumerate(eligible, 1):
        vc = voice_counts.get(u.id, 0)
        print(f"  {i:>3}. {u.username}  ({vc} sample(s))")

    print()
    succeeded, failed = [], []

    for i, user in enumerate(eligible, 1):
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(eligible)}] Retraining: {user.username}")
        print(f"{'─'*60}")
        try:
            result = train_voice_model(user.username)
            if result:
                print(f"  ✅ Done: {user.username}")
                succeeded.append(user.username)
            else:
                print(f"  ⚠  Skipped (no valid samples): {user.username}")
                failed.append(user.username)
        except Exception as e:
            print(f"  ❌ Failed: {user.username} — {e}")
            failed.append(user.username)

    print(f"\n{'═'*60}")
    print(f"  RETRAIN COMPLETE — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Succeeded : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print(f"  Failed users:")
        for u in failed:
            print(f"    - {u}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()