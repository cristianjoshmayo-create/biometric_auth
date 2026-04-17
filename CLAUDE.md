# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Multimodal biometric authentication PoC combining **keystroke dynamics** and **speech biometrics** as a 2FA alternative. Desktop web only — fixed lowercase 4-word passphrases, no mobile/shift/IME support.

## Running the App

```bash
# Start the server (from project root)
start.bat          # Interactive: choose (1) stable/demo or (2) dev with auto-reload

# Or manually:
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000           # stable
uvicorn main:app --host 127.0.0.1 --port 8000 --reload  # dev
```

Two venvs exist: `venv` (primary) and `venv310` (Python 3.10 for torch/SpeechBrain compatibility).

## Train Models Manually

```bash
# Keystroke (Random Forest / GBM / Profile Matcher)
python ml/train_keystroke_rf.py <email>

# Retrain all keystroke models
python retrain_all_keystroke.py

# Voice CNN
python ml/train_voice_cnn.py <email>
```

Models are saved as `.pkl` files in `ml/models/` with sanitized filenames (`@` → `_at_`, `.` → `_`).

## Architecture

### Authentication Flow (4 steps, sequential)

1. **Password** → `POST /api/auth/password` — bcrypt check
2. **Keystroke** → `POST /api/auth/keystroke` — RF/GBM or Profile Matcher scores fused with Mahalanobis distance
3. **Voice** → `POST /api/auth/voice` — ECAPA-TDNN cosine similarity + Whisper phrase verification
4. **Fusion** → `POST /api/auth/fuse` — weighted combination of keystroke + voice scores (only called when keystroke is uncertain, 0.55–0.79)

If keystroke confidence ≥ 0.80, voice is skipped and access is granted immediately.

### Backend (FastAPI, `backend/`)

- `main.py` — app setup, CORS, static file serving with no-cache headers, route mounts
- `routers/enroll.py` — user creation (bcrypt password, Fernet-encrypted phrase), keystroke/voice/security-question enrollment, audio feature extraction (`/extract-mfcc`)
- `routers/auth.py` — password/keystroke/voice/security verification, multimodal fusion (`/fuse`), progressive enrollment (soft → ramp → hardened thresholds), adaptive keystroke sample saving
- `utils/fusion.py` — single source of truth for intra-modal (RF+Mahalanobis) and inter-modal (keystroke+voice) score fusion weights
- `utils/crypto.py` — Fernet encryption/decryption for `users.phrase` and `security_questions.question`
- `database/db.py` — SQLAlchemy engine with Supabase-tuned pool settings (pool_pre_ping, keepalives)
- `database/models.py` — User, KeystrokeTemplate (60+ feature columns + JSON maps for digraph variants), VoiceTemplate, SecurityQuestion, AuthLog
- `schemas.py` — `VoiceFeatures` base class shared by enrollment and auth (single source of truth for voice fields)

### ML Layer (`ml/`)

- `train_keystroke_rf.py` — phrase-aware training: strips inactive digraphs, generates 3-tier synthetic impostors (near/shifted/random), CMU impostor patching, FAR-weighted threshold search. Uses Profile Matcher for ≤10 samples, upgrades to RF/GBM at 11+.
- `keystroke_profile_matcher.py` — TypingDNA-style direct comparison (no classifier): speed normalization, tiered Z-tolerance per feature group, weighted group scoring
- `voice_ecapa.py` — ECAPA-TDNN speaker verification (SpeechBrain pretrained, 192-dim embeddings, cosine similarity)
- `train_voice_cnn.py` — CNN voice model training
- `load_cmu_impostors.py` — builds `cmu_impostor_profiles.pkl` from CMU keystroke dataset

### Frontend (`frontend/`)

- `js/keystroke.js` — captures dwell times, flight times, digraph/trigraph maps (4 variants: DD/DU/UD/UU), per-key dwell map, quality scoring
- `js/speech.js` — mic recording with RNNoise denoising, sends base64 audio to `/extract-mfcc`
- `js/api.js` — auto-detects API base URL from `window.location`, shared voice payload builder
- `js/enroll.js` / `js/auth_flow.js` — enrollment and login UI orchestration
- Pages: `login.html`, `enroll.html`, `dashboard.html`

### Database (PostgreSQL on Supabase)

- `users.phrase` column is **Fernet-encrypted ciphertext** — always call `decrypt()` before using the plaintext
- `security_questions.question` is also encrypted
- DB credentials via `.env` (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, ENCRYPTION_KEY)

## Key Design Patterns

- **Progressive enrollment**: keystroke threshold starts soft (0.45) for new users, ramps to 0.55 after 3 logins, hardens to stored threshold after 7. Maturity = count of granted keystroke/fusion auth logs.
- **Adaptive learning**: successful login keystroke samples are saved back to the DB and trigger periodic retraining (every 2/5/10 logins depending on sample count, capped at 50 samples with FIFO eviction).
- **Pending sample cache**: keystroke samples from failed-keystroke sessions are held in memory (`_pending_ks_samples`) and only persisted if voice fusion grants access — prevents polluting training data.
- **Auto-training**: enrollment triggers background thread training after collecting enough samples (5 keystroke, 3 voice).
- **Phrase-aware features**: only digraphs present in the user's assigned phrase are kept as features; inactive hardcoded digraphs are stripped before training.
- **Windows symlink fix**: `HF_HUB_DISABLE_SYMLINKS=1` set before imports in `main.py` to avoid WinError 1314 with HuggingFace model downloads.

## DB Migrations

Schema changes are applied manually via SQL (see comments in `database/models.py` for ALTER TABLE statements). No migration framework.
