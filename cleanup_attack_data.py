"""
One-off cleanup of the sibling-attack audit rows on 2026-04-24 14:23
(Asia/Manila). Only the AuthLog rows from that single session are removed;
no KeystrokeTemplate rows were polluted (the save gate at line 520 rejected
the sibling's sub-threshold sample).

Usage:
    python cleanup_attack_data.py             # dry-run
    python cleanup_attack_data.py --apply     # actually deletes
"""

import sys

sys.path.insert(0, "backend")

from database.db import SessionLocal
from database.models import AuthLog

# AuthLog ids of the sibling-attack 14:23 session:
#   2096 keystroke denied 0.3647
#   2097 voice     granted 0.7687
#   2098 fusion    granted 0.6273  (Case B — the breach)
ATTACK_LOG_IDS = [2096, 2097, 2098]


def main(apply: bool):
    db = SessionLocal()
    try:
        rows = db.query(AuthLog).filter(AuthLog.id.in_(ATTACK_LOG_IDS)).all()
        print(f"AuthLog rows to delete: {len(rows)}")
        for r in rows:
            print(f"  id={r.id}  method={r.auth_method}  result={r.result}  "
                  f"score={r.confidence_score}  at={r.attempted_at}")

        missing = set(ATTACK_LOG_IDS) - {r.id for r in rows}
        if missing:
            print(f"  [!] not found: {sorted(missing)}")

        if not apply:
            print("\n[dry-run] no changes made. re-run with --apply to delete.")
            return

        for r in rows:
            db.delete(r)
        db.commit()
        print(f"\n[ok] deleted {len(rows)} AuthLog rows.")

    finally:
        db.close()


if __name__ == "__main__":
    main(apply=("--apply" in sys.argv))
