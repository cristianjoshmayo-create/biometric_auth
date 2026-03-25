# retrain_all_keystroke.py
# Retrains keystroke RF models for ALL users in the database.
# Run from the project root:  python retrain_all_keystroke.py

import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(project_root, 'backend')
ml_path      = os.path.join(project_root, 'ml')

for p in [project_root, backend_path, ml_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Imports ───────────────────────────────────────────────────────────────────
from database.db import SessionLocal
from database.models import User, KeystrokeTemplate
from ml.train_keystroke_rf import train_random_forest

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            print("No users found in database.")
            return

        # Only train users that actually have keystroke samples
        eligible = []
        for u in users:
            count = db.query(KeystrokeTemplate).filter(
                KeystrokeTemplate.user_id == u.id
            ).count()
            if count > 0:
                eligible.append((u.username, count))
            else:
                print(f"  ⚠  Skipping '{u.username}' — no keystroke samples")

    finally:
        db.close()

    print(f"\n{'='*60}")
    print(f"  Retraining {len(eligible)} user(s)")
    print(f"{'='*60}\n")

    results = {"ok": [], "failed": []}

    for i, (username, sample_count) in enumerate(eligible, 1):
        print(f"\n[{i}/{len(eligible)}] Training '{username}' ({sample_count} samples)...")
        print("-" * 60)
        try:
            model_path = train_random_forest(username)
            if model_path:
                print(f"  ✅ Done → {os.path.basename(model_path)}")
                results["ok"].append(username)
            else:
                print(f"  ❌ Training returned None for '{username}'")
                results["failed"].append(username)
        except Exception as e:
            import traceback
            print(f"  ❌ Error training '{username}': {e}")
            traceback.print_exc()
            results["failed"].append(username)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RETRAIN COMPLETE")
    print(f"{'='*60}")
    print(f"  ✅ Success : {len(results['ok'])}")
    for u in results["ok"]:
        print(f"       {u}")
    if results["failed"]:
        print(f"  ❌ Failed  : {len(results['failed'])}")
        for u in results["failed"]:
            print(f"       {u}")
    print()


if __name__ == "__main__":
    main()