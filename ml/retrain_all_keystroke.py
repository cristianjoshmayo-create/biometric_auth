# ml/retrain_all_keystroke.py
# Retrain keystroke models for all enrolled users.
# Run from the project root: python ml/retrain_all_keystroke.py

import sys
import os

backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend')
sys.path.insert(0, backend_path)

from database.db import SessionLocal
from database.models import User, KeystrokeTemplate
from train_keystroke_rf import train_random_forest

db = SessionLocal()
try:
    users = db.query(User).all()
    eligible = [
        u for u in users
        if db.query(KeystrokeTemplate).filter(KeystrokeTemplate.user_id == u.id).count() >= 3
    ]
finally:
    db.close()

print(f"Found {len(eligible)} user(s) with keystroke enrollment data.\n")
for u in eligible:
    print(f"{'='*60}")
    print(f"Retraining: {u.username}")
    train_random_forest(u.username)

print("\nDone.")
