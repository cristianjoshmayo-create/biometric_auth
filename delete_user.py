"""
Delete a user completely from the database and the local filesystem.

Edit EMAIL below, then run from the project root:
    python delete_user.py

Removes:
  - users row (cascades manually to keystroke_templates, voice_templates,
    security_questions, auth_logs)
  - ml/models/<sanitized_email>_*.pkl files (keystroke + voice models)
"""

import os
import sys
from pathlib import Path

# ── EDIT THIS ────────────────────────────────────────────────────────────────
EMAIL = "binhs.morillojerime@gmail.com"
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from database.db import SessionLocal
from database.models import (
    User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion, AuthLog,
)


def sanitize(email: str) -> str:
    return email.replace("@", "_at_").replace(".", "_")


def delete_user(email: str) -> None:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == email).first()
        if not user:
            print(f"[db] no user found for {email}")
        else:
            uid = user.id
            ks = db.query(KeystrokeTemplate).filter_by(user_id=uid).delete()
            vc = db.query(VoiceTemplate).filter_by(user_id=uid).delete()
            sq = db.query(SecurityQuestion).filter_by(user_id=uid).delete()
            al = db.query(AuthLog).filter_by(user_id=uid).delete()
            db.delete(user)
            db.commit()
            print(f"[db] deleted user id={uid} ({email})")
            print(f"     keystroke_templates: {ks}")
            print(f"     voice_templates:     {vc}")
            print(f"     security_questions:  {sq}")
            print(f"     auth_logs:           {al}")
    except Exception as e:
        db.rollback()
        print(f"[db] ERROR: {e}")
        raise
    finally:
        db.close()

    models_dir = ROOT / "ml" / "models"
    prefix = sanitize(email)
    removed = 0
    if models_dir.exists():
        for f in models_dir.glob(f"{prefix}_*"):
            try:
                f.unlink()
                print(f"[fs] removed {f.name}")
                removed += 1
            except OSError as e:
                print(f"[fs] failed to remove {f.name}: {e}")
    print(f"[fs] removed {removed} model file(s)")


if __name__ == "__main__":
    confirm = input(f"Delete user '{EMAIL}' from DB and filesystem? [y/N]: ").strip().lower()
    if confirm != "y":
        print("aborted.")
        sys.exit(0)
    delete_user(EMAIL)
    print("done.")
