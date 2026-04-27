"""One-off: delete joshmayo1805@gmail.com and all enrolled data."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from database.db import SessionLocal
from database.models import User, KeystrokeTemplate, VoiceTemplate, SecurityQuestion, AuthLog

TARGET_EMAIL = "joshmayo1805@gmail.com"
MODELS_DIR = ROOT / "ml" / "models"
MODEL_FILES = [
    "joshmayo1805_at_gmail_com_keystroke_rf.pkl",
    "joshmayo1805_at_gmail_com_voice_ecapa.pkl",
]

db = SessionLocal()
try:
    user = db.query(User).filter(User.username == TARGET_EMAIL).first()
    if not user:
        print(f"No user found with email {TARGET_EMAIL}")
        sys.exit(0)

    uid = user.id
    print(f"Found user id={uid} username={user.username}")

    counts = {
        "keystroke_templates": db.query(KeystrokeTemplate).filter(KeystrokeTemplate.user_id == uid).count(),
        "voice_templates":     db.query(VoiceTemplate).filter(VoiceTemplate.user_id == uid).count(),
        "security_questions":  db.query(SecurityQuestion).filter(SecurityQuestion.user_id == uid).count(),
        "auth_logs":           db.query(AuthLog).filter(AuthLog.user_id == uid).count(),
    }
    print("Rows to delete:", counts)

    db.query(KeystrokeTemplate).filter(KeystrokeTemplate.user_id == uid).delete(synchronize_session=False)
    db.query(VoiceTemplate).filter(VoiceTemplate.user_id == uid).delete(synchronize_session=False)
    db.query(SecurityQuestion).filter(SecurityQuestion.user_id == uid).delete(synchronize_session=False)
    db.query(AuthLog).filter(AuthLog.user_id == uid).delete(synchronize_session=False)
    db.delete(user)
    db.commit()
    print("DB rows deleted and committed.")
except Exception as e:
    db.rollback()
    print(f"ERROR — rolled back: {e}")
    raise
finally:
    db.close()

for fname in MODEL_FILES:
    fpath = MODELS_DIR / fname
    if fpath.exists():
        fpath.unlink()
        print(f"Deleted model file: {fname}")
    else:
        print(f"Model file not present: {fname}")

print("Done.")
