"""One-off: remove polluted samples from the 2026-05-02 14:21 AI-voice attempt
on joshmayo1805@gmail.com. Lists candidates first; pass --apply to delete."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
from database.db import SessionLocal
from database.models import User, KeystrokeTemplate

from ml.voice_ecapa import _load_profile, _save_profile, _recompute_mean

USERNAME = "joshmayo1805@gmail.com"
APPLY = "--apply" in sys.argv

db = SessionLocal()
user = db.query(User).filter(User.username == USERNAME).first()
if not user:
    print(f"User {USERNAME} not found"); sys.exit(1)

# 1) Newest adaptive keystroke row (the 14:21 fusion-grant save)
ks_row = (
    db.query(KeystrokeTemplate)
    .filter(KeystrokeTemplate.user_id == user.id,
            KeystrokeTemplate.source.in_(["login_ks", "login_fusion"]))
    .order_by(KeystrokeTemplate.sample_order.desc())
    .first()
)
print("Latest adaptive keystroke row:")
if ks_row:
    print(f"  id={ks_row.id} source={ks_row.source} order={ks_row.sample_order}")
else:
    print("  (none found)")

# 2) Newest ECAPA adaptive slot
profile = _load_profile(USERNAME)
adaptive_slots = [s for s in (profile or {}).get("embeddings", [])
                  if s.get("source") != "enrollment"]
print(f"ECAPA adaptive slots: {len(adaptive_slots)}")
if adaptive_slots:
    print(f"  latest ts={adaptive_slots[-1].get('ts')}")

if not APPLY:
    print("\nDry run. Re-run with --apply to delete.")
    sys.exit(0)

# Apply
if ks_row:
    db.delete(ks_row); db.commit()
    print(f"Deleted keystroke_templates id={ks_row.id}")

if profile and adaptive_slots:
    enroll_slots = [s for s in profile["embeddings"] if s.get("source") == "enrollment"]
    profile["embeddings"] = enroll_slots + adaptive_slots[:-1]
    _recompute_mean(profile)
    _save_profile(profile)
    print(f"Removed latest ECAPA adaptive slot "
          f"(remaining adaptive={len(adaptive_slots)-1})")

db.close()
