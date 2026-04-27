"""
One-off cleanup of asther's keystroke samples polluted by impostor attempts
67 and 69 on 2026-04-27 (08:00 and 08:07 Asia/Manila).

Both attempts went through the fusion path (keystroke uncertain -> voice ->
/fuse granted), so the samples were persisted with source="login_fusion".
The keystroke confidences at save time were ~0.8234 and ~0.8488.

Usage:
    python cleanup_asther_polluted.py             # dry-run, prints candidates
    python cleanup_asther_polluted.py --apply     # actually deletes
"""

import sys
sys.path.insert(0, "backend")

from database.db import SessionLocal
from database.models import KeystrokeTemplate, User

TARGET_USER  = "astherlilies@gmail.com"
TARGET_SCORES = [0.8234, 0.8488]   # ks confidences for attempts 67, 69
SCORE_TOL    = 0.005


def main(apply: bool):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == TARGET_USER).first()
        if not user:
            print(f"User '{TARGET_USER}' not found.")
            return

        rows = db.query(KeystrokeTemplate).filter(
            KeystrokeTemplate.user_id == user.id,
            KeystrokeTemplate.source == "login_fusion",
        ).order_by(KeystrokeTemplate.sample_order.desc()).all()

        print(f"All login_fusion rows for {TARGET_USER} ({len(rows)} total):")
        for r in rows:
            print(f"  id={r.id}  sample_order={r.sample_order}  "
                  f"saved_score={r.saved_score}  attempt_number={r.attempt_number}")

        targets = [
            r for r in rows
            if r.saved_score is not None
            and any(abs(r.saved_score - s) < SCORE_TOL for s in TARGET_SCORES)
        ]
        print(f"\nMatched candidates ({len(targets)}):")
        for r in targets:
            print(f"  id={r.id}  saved_score={r.saved_score}  "
                  f"sample_order={r.sample_order}")

        if not targets:
            print("Nothing matched. Re-check TARGET_SCORES or pass IDs explicitly.")
            return

        if not apply:
            print("\n[dry-run] no changes made. re-run with --apply to delete.")
            return

        for r in targets:
            db.delete(r)
        db.commit()
        print(f"\n[ok] deleted {len(targets)} rows.")

    finally:
        db.close()


if __name__ == "__main__":
    main(apply=("--apply" in sys.argv))
