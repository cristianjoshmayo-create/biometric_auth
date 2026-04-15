import sys
import os
import glob

sys.path.insert(0, 'backend')

from database.db import SessionLocal
from database.models import AuthLog, SecurityQuestion, VoiceTemplate, KeystrokeTemplate, User

db = SessionLocal()
db.query(AuthLog).delete()
db.query(SecurityQuestion).delete()
db.query(VoiceTemplate).delete()
db.query(KeystrokeTemplate).delete()
db.query(User).delete()
db.commit()
db.close()
print("Database cleared.")

deleted = 0
for f in glob.glob('ml/models/*_keystroke_rf.pkl') + glob.glob('ml/models/*_voice_*.pkl'):
    os.remove(f)
    print(f"Deleted {f}")
    deleted += 1

if deleted == 0:
    print("No model files to delete.")

print("Done.")
