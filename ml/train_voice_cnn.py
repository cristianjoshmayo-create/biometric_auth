# ml/train_voice_cnn.py
# Voice CNN v4 — 1D Convolutional Neural Network for speaker verification
#
# WHY CNN instead of GBM:
#   The old GBM worked on 62 aggregated statistics (mean, std of MFCCs).
#   Aggregation collapses the temporal dimension — two very different
#   speakers can have similar mean MFCCs if their raw sequences differ only
#   in timing patterns. A 1D CNN operates on the raw MFCC frame sequence
#   (shape: 39 × T, with MFCCs + deltas) and learns temporal speaking patterns
#   that are far more speaker-discriminative, especially for short passphrases.
#
# Architecture:
#   Input  : (batch, 39, T)  — 13 MFCCs + 13 Δ-MFCCs + 13 Δ²-MFCCs per frame
#   Block 1: Conv1d(39→64, k=7, pad=3)  → BN → ReLU → MaxPool(2)
#   Block 2: Conv1d(64→128, k=5, pad=2) → BN → ReLU → MaxPool(2)
#   Block 3: Conv1d(128→256, k=3, pad=1)→ BN → ReLU → AdaptiveAvgPool(1)
#   Head   : Linear(256→128) → ReLU → Dropout(0.35) → Linear(128→1) → Sigmoid
#
# Training:
#   • Binary classification: genuine (1) vs impostor (0)
#   • SpecAugment-style augmentation (time mask + freq mask + noise)
#   • CMU-profile synthetic sequences + other enrolled users as impostors
#   • Focal loss — handles class imbalance robustly
#   • AdamW optimiser + cosine annealing LR schedule
#
# Inference:
#   CNN posterior fused with Mahalanobis on raw prosodic features (via fusion.py)
#
# Requirements:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

import sys
import os

backend_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'
)
sys.path.insert(0, backend_path)

import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from database.db import SessionLocal
from database.models import User, VoiceTemplate

# ── PyTorch import with friendly error ────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠  PyTorch not found. Install with:")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("   Falling back to GBM classifier.\n")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

T_MAX      = 300   # fixed sequence length (frames). 300 = ~3 s at 10ms hop.
N_MFCC     = 13    # number of MFCC coefficients
N_CHANNELS = 39    # 13 MFCC + 13 Δ + 13 Δ²   (CNN input channels)
N_FEATURES = 62    # aggregated feature vector for Mahalanobis


# ─────────────────────────────────────────────────────────────────────────────
#  CNN ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class VoiceCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for speaker verification.

    Takes an MFCC sequence of shape (batch, 39, T) and outputs a probability
    in [0, 1] where 1 = genuine speaker and 0 = impostor.

    Three conv blocks with increasing filter depth capture temporal patterns
    at different time scales:
      Block 1 (k=7): broad patterns — phrase-level rhythm
      Block 2 (k=5): mid-range patterns — syllable-level dynamics
      Block 3 (k=3): fine patterns — phoneme transitions

    Global average pooling makes the model length-agnostic so it handles
    any utterance length without padding-induced bias.
    """

    def __init__(self, n_channels: int = N_CHANNELS, dropout: float = 0.35):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.10),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.10),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 39, T) → (batch, 1) logit"""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
#  FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Addresses class imbalance by down-weighting easy negatives.

    FL(p) = -α · (1 − p)^γ · log(p)
    α = 0.75 weights positive class more (fewer genuine samples)
    γ = 2.0  focuses learning on hard impostor examples
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets == 1, probs, 1 - probs)
        alpha = torch.where(targets == 1,
                            torch.full_like(targets, self.alpha),
                            torch.full_like(targets, 1 - self.alpha))
        focal = alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  SEQUENCE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def frames_to_tensor(frames: list) -> np.ndarray:
    """
    Convert stored mfcc_frames (list of T lists of 13 floats) to
    a (39, T_MAX) numpy array by:
      1. Stacking into (T, 13)
      2. Computing delta and delta² along the time axis
      3. Concatenating to (T, 39)
      4. Transposing to (39, T)
      5. Padding or truncating to T_MAX
    """
    arr = np.array(frames, dtype=np.float32)   # (T, 13)
    if arr.ndim != 2 or arr.shape[1] != N_MFCC:
        raise ValueError(f"Expected (T, {N_MFCC}) frames, got {arr.shape}")

    T = arr.shape[0]

    # Compute delta and delta² along time axis (axis=0)
    # Using simple finite differences — same as librosa.feature.delta
    delta  = _delta(arr)   # (T, 13)
    delta2 = _delta(delta) # (T, 13)

    seq = np.concatenate([arr, delta, delta2], axis=1)   # (T, 39)
    seq = seq.T                                           # (39, T)

    # Pad or truncate to T_MAX
    if T >= T_MAX:
        seq = seq[:, :T_MAX]
    else:
        pad = np.zeros((N_CHANNELS, T_MAX - T), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=1)

    return seq.astype(np.float32)


def _delta(x: np.ndarray, width: int = 9) -> np.ndarray:
    """
    Compute delta features using regression over ±(width//2) frames.
    Equivalent to librosa.feature.delta but works on (T, F) arrays.
    """
    T, F = x.shape
    hw   = width // 2
    denom = float(2 * sum(i ** 2 for i in range(1, hw + 1)))
    padded = np.pad(x, ((hw, hw), (0, 0)), mode='edge')
    out = np.zeros_like(x)
    for t in range(T):
        for i in range(1, hw + 1):
            out[t] += i * (padded[t + hw + i] - padded[t + hw - i])
    return out / (denom + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    SpecAugment-inspired augmentation for a (39, T_MAX) MFCC sequence.

    Applied only to genuine samples during training to expand the small
    enrollment set without changing the speaker identity:

    1. Time masking   — zero a random T//8 segment (simulates hesitation)
    2. Freq masking   — zero a random channel block (simulates mic dropout)
    3. Gaussian noise — add small noise (simulates mic variability)
    4. Time shift     — circular shift (simulates phrase-start timing)
    """
    seq = seq.copy()
    C, T = seq.shape

    # 1. Time masking
    if rng.random() < 0.80:
        t_width = rng.integers(1, max(2, T // 8))
        t_start = rng.integers(0, max(1, T - t_width))
        seq[:, t_start:t_start + t_width] = 0.0

    # 2. Frequency (channel) masking
    if rng.random() < 0.80:
        f_width = rng.integers(1, max(2, C // 6))
        f_start = rng.integers(0, max(1, C - f_width))
        seq[f_start:f_start + f_width, :] = 0.0

    # 3. Gaussian noise (σ = 5% of signal std)
    if rng.random() < 0.70:
        noise_std = float(np.std(seq)) * 0.05 + 1e-6
        seq += rng.normal(0, noise_std, seq.shape).astype(np.float32)

    # 4. Circular time shift (±10% of T)
    if rng.random() < 0.50:
        shift = int(rng.integers(-T // 10, T // 10))
        seq = np.roll(seq, shift, axis=1)

    return seq


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_genuine_sequences(db, user_id: int) -> list:
    """Load stored MFCC frame sequences for the target user."""
    templates = (
        db.query(VoiceTemplate)
        .filter(VoiceTemplate.user_id == user_id)
        .order_by(VoiceTemplate.enrolled_at.asc())
        .all()
    )
    seqs = []
    for t in templates:
        frames = getattr(t, 'mfcc_frames', None)
        if not frames or len(frames) < 10:
            print(f"  ⚠  Template id={t.id} has no mfcc_frames — skipping")
            continue
        try:
            tensor = frames_to_tensor(frames)
            seqs.append(tensor)
        except Exception as e:
            print(f"  ⚠  Template id={t.id} frame error: {e}")
    return seqs


def load_impostor_sequences(db, exclude_user_id: int) -> list:
    """Load MFCC sequences from all OTHER enrolled users as real impostors."""
    other_users = db.query(User).filter(User.id != exclude_user_id).all()
    seqs = []
    for u in other_users:
        for t in (
            db.query(VoiceTemplate)
            .filter(VoiceTemplate.user_id == u.id)
            .all()
        ):
            frames = getattr(t, 'mfcc_frames', None)
            if not frames or len(frames) < 10:
                continue
            try:
                seqs.append(frames_to_tensor(frames))
            except Exception:
                continue
    if seqs:
        print(f"  Real impostor sequences: {len(seqs)} from {len(other_users)} other user(s)")
    return seqs


def generate_synthetic_impostor_sequences(profile_mean: np.ndarray,
                                          profile_std: np.ndarray,
                                          n: int,
                                          rng: np.random.Generator) -> list:
    """
    Generate synthetic impostor sequences sampled from CMU impostor profiles.

    Each synthetic impostor is a sequence of T_MAX frames where each frame
    is drawn from a Gaussian parameterised by the user's profile mean/std
    but pushed ≥1 std away to ensure it is distinguishable from genuine.

    profile_mean/std are the 62-feature aggregated profiles already computed
    in training. The first 13 entries correspond to mfcc_mean; we use them
    as the per-channel mean to generate (T_MAX, 13) frame sequences.
    """
    seqs = []
    # Use mfcc mean/std from the 62-feature vector (first 13 = mfcc means,
    # next 13 = mfcc stds — this matches extract_feature_vector order)
    ch_mean = profile_mean[:N_MFCC]
    ch_std  = profile_std[N_MFCC:2 * N_MFCC] + 0.5  # mfcc_std channels

    for _ in range(n):
        # Draw T_MAX frames from a Gaussian, shifted away from genuine
        direction = rng.choice([-1, 1], size=N_MFCC)
        shift     = direction * (rng.uniform(1.0, 2.5, N_MFCC) * ch_std)
        frames    = rng.normal(
            (ch_mean + shift)[None, :],        # (1, 13) broadcast
            ch_std[None, :] * 1.5,
            size=(T_MAX, N_MFCC)
        ).astype(np.float32)
        try:
            seqs.append(frames_to_tensor(frames.tolist()))
        except Exception:
            continue
    return seqs


def generate_genuine_augmented(seqs: list, n: int,
                                rng: np.random.Generator) -> list:
    """Augment genuine sequences to produce N augmented samples."""
    augmented = []
    for _ in range(n):
        base = seqs[rng.integers(0, len(seqs))]
        augmented.append(augment_sequence(base, rng))
    return augmented


# ─────────────────────────────────────────────────────────────────────────────
#  MAHALANOBIS (kept for fusion — operates on 62-feature aggregated vector)
# ─────────────────────────────────────────────────────────────────────────────

def extract_agg_vector(template) -> np.ndarray:
    """62-element CMVN feature vector from a VoiceTemplate row."""
    mfcc_mean   = list(template.mfcc_features or [])
    mfcc_std    = list(template.mfcc_std      or [])
    delta_mean  = list(getattr(template, 'delta_mfcc_mean',  None) or [])
    delta2_mean = list(getattr(template, 'delta2_mfcc_mean', None) or [])
    while len(mfcc_mean)   < 13: mfcc_mean.append(0.0)
    while len(mfcc_std)    < 13: mfcc_std.append(0.0)
    while len(delta_mean)  < 13: delta_mean.append(0.0)
    while len(delta2_mean) < 13: delta2_mean.append(0.0)
    return np.array(
        mfcc_mean[:13] + mfcc_std[:13] + delta_mean[:13] + delta2_mean[:13] + [
            float(getattr(template, 'pitch_mean',             None) or 0),
            float(getattr(template, 'pitch_std',              None) or 0),
            float(getattr(template, 'speaking_rate',          None) or 0),
            float(getattr(template, 'energy_mean',            None) or 0),
            float(getattr(template, 'energy_std',             None) or 0),
            float(getattr(template, 'zcr_mean',               None) or 0),
            float(getattr(template, 'spectral_centroid_mean', None) or 0),
            float(getattr(template, 'spectral_rolloff_mean',  None) or 0),
            float(getattr(template, 'spectral_flux_mean',     None) or 0),
            float(getattr(template, 'voiced_fraction',        None) or 0),
        ],
        dtype=np.float64
    )


def extract_raw_profile_vector(template) -> np.ndarray:
    """36-element non-CMVN vector for Mahalanobis (avoids CMVN mean collapse)."""
    mfcc_std   = list(template.mfcc_std or [])
    delta_mean = list(getattr(template, 'delta_mfcc_mean', None) or [])
    while len(mfcc_std)   < 13: mfcc_std.append(0.0)
    while len(delta_mean) < 13: delta_mean.append(0.0)
    return np.array(
        mfcc_std[:13] + delta_mean[:13] + [
            float(getattr(template, 'pitch_mean',             None) or 0),
            float(getattr(template, 'pitch_std',              None) or 0),
            float(getattr(template, 'speaking_rate',          None) or 0),
            float(getattr(template, 'energy_mean',            None) or 0),
            float(getattr(template, 'energy_std',             None) or 0),
            float(getattr(template, 'zcr_mean',               None) or 0),
            float(getattr(template, 'spectral_centroid_mean', None) or 0),
            float(getattr(template, 'spectral_rolloff_mean',  None) or 0),
            float(getattr(template, 'spectral_flux_mean',     None) or 0),
            float(getattr(template, 'voiced_fraction',        None) or 0),
        ],
        dtype=np.float64
    )


def mahalanobis_score(vec, profile_mean, profile_std) -> float:
    var       = profile_std ** 2
    safe_var  = np.where(var < 1e-10, 1e-10, var)
    diff      = vec - profile_mean
    d_sq      = float(np.sum(diff ** 2 / safe_var))
    d_sq_norm = d_sq / max(len(vec), 1)
    score     = 1.0 / (1.0 + np.exp(2.5 * (d_sq_norm - 1.0)))
    return float(np.clip(score, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _safe_filename(username: str) -> str:
    return username.replace("@", "_at_").replace(".", "_").replace(" ", "_")


def train_voice_model(username: str):
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not installed. Cannot train CNN.")
        print("   Run:  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return None

    print(f"\n{'='*70}")
    print(f"  VOICE CNN v4 — 1D Convolutional Neural Network")
    print(f"  User: {username}")
    print(f"{'='*70}")

    # ── Load data from DB ─────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"❌ User '{username}' not found!")
            return None

        genuine_seqs    = load_genuine_sequences(db, user.id)
        impostor_seqs   = load_impostor_sequences(db, user.id)
        agg_vecs        = []
        raw_vecs        = []

        templates = (
            db.query(VoiceTemplate)
            .filter(VoiceTemplate.user_id == user.id)
            .all()
        )
        for t in templates:
            agg_vecs.append(extract_agg_vector(t))
            raw_vecs.append(extract_raw_profile_vector(t))

        user_id = user.id

    finally:
        try: db.close()
        except Exception: pass

    if not genuine_seqs:
        print("❌ No stored MFCC frame sequences found.")
        print("   Users enrolled before the CNN update must re-enroll to get mfcc_frames stored.")
        print("   Tip: delete their voice templates and enroll again.")
        return None

    n_genuine_real = len(genuine_seqs)
    print(f"\n  Genuine sequences loaded : {n_genuine_real}")

    # ── Aggregated profile (for Mahalanobis at inference) ─────────────────────
    agg_arr          = np.array(agg_vecs)
    profile_mean     = agg_arr.mean(axis=0)
    profile_std      = agg_arr.std(axis=0) if len(agg_vecs) > 1 else np.abs(agg_arr[0]) * 0.10 + 1e-6

    raw_arr          = np.array(raw_vecs)
    raw_profile_mean = raw_arr.mean(axis=0)
    raw_profile_std  = raw_arr.std(axis=0) if len(raw_vecs) > 1 else np.abs(raw_arr[0]) * 0.10 + 1e-6
    raw_profile_std  = np.where(raw_profile_std < 1e-6, 1e-6, raw_profile_std)

    # ── Augment genuine sequences ──────────────────────────────────────────────
    rng     = np.random.default_rng(42)
    n_aug   = max(300, n_genuine_real * 100)
    gen_aug = generate_genuine_augmented(genuine_seqs, n_aug, rng)
    print(f"  Genuine augmented        : {len(gen_aug)}")

    # ── Build impostor pool ───────────────────────────────────────────────────
    n_syn_needed = max(0, len(gen_aug) * 2 - len(impostor_seqs))
    syn_impostors = generate_synthetic_impostor_sequences(
        profile_mean, profile_std, n_syn_needed, rng
    )
    all_impostors = impostor_seqs + syn_impostors
    print(f"  Real impostor sequences  : {len(impostor_seqs)}")
    print(f"  Synthetic impostors      : {len(syn_impostors)}")
    print(f"  Total impostors          : {len(all_impostors)}")

    # ── Build tensors ─────────────────────────────────────────────────────────
    X_genuine   = np.stack(gen_aug, axis=0).astype(np.float32)       # (N_g, 39, T)
    X_impostor  = np.stack(all_impostors, axis=0).astype(np.float32) # (N_i, 39, T)

    X = np.concatenate([X_genuine, X_impostor], axis=0)
    y = np.array([1.0] * len(X_genuine) + [0.0] * len(X_impostor), dtype=np.float32)

    print(f"\n  Training set: {len(X_genuine)} genuine / {len(X_impostor)} impostor")
    print(f"  Input shape : (batch, {N_CHANNELS}, {T_MAX})")

    # ── 5-fold cross-validation for threshold calibration ─────────────────────
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix

    device   = torch.device('cpu')
    skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_int    = y.astype(int)
    probs_cv = np.zeros(len(y), dtype=np.float32)

    print(f"\n  Running 5-fold cross-validation ...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_int)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val      = X[val_idx]

        ds_tr  = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(y_tr).unsqueeze(1)
        )
        loader = DataLoader(ds_tr, batch_size=64, shuffle=True)

        model = VoiceCNN1D(n_channels=N_CHANNELS).to(device)
        opt   = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

        model.train()
        for epoch in range(30):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss   = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(torch.from_numpy(X_val).to(device))
            probs_val  = torch.sigmoid(logits_val).squeeze(1).cpu().numpy()
        probs_cv[val_idx] = probs_val

    # ── Compute Mahalanobis scores for each CV sample ─────────────────────────
    # Use aggregated features (62-vec) as proxy: CNN score fused with Mah
    # The Mah scores for the TRAINING MATRIX can't be computed per-sample
    # (we don't have agg vectors for augmented/synthetic samples).
    # Instead we use CNN CV probability alone for threshold search.
    # At inference, fusion with Mah happens in predict_voice().
    from utils.fusion import fuse_voice_scores

    print(f"\n{'='*70}")
    print(f"  THRESHOLD SEARCH on CNN probability (CV fold estimates)")
    print(f"{'='*70}")
    print(f"  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'EER':>8}")

    best_thresh, best_eer = 0.50, 1.0
    for t in np.arange(0.20, 0.90, 0.02):
        y_pred = (probs_cv >= t).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        cm = confusion_matrix(y_int, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        far  = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr  = fn / (fn + tp) if (fn + tp) > 0 else 0
        eer  = (far + frr) / 2
        mark = " ◄ best" if eer < best_eer else ""
        print(f"  {t:>10.2f}  {far:>8.2%}  {frr:>8.2%}  {eer:>8.2%}{mark}")
        if eer < best_eer:
            best_eer, best_thresh = eer, float(t)

    # Apply a minimum floor — CNN probabilities tend to be more spread out
    # than GBM, so 0.40 is a reasonable minimum for the CNN alone.
    # The Mahalanobis term in fuse_voice_scores() adds extra security.
    final_thresh = max(best_thresh, 0.40)
    print(f"\n  Final threshold : {final_thresh:.2f}  (EER: {best_eer:.2%})")

    # ── Train final model on full dataset ─────────────────────────────────────
    print(f"\n  Training final CNN on full dataset ...")
    ds_full  = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).unsqueeze(1)
    )
    loader_full = DataLoader(ds_full, batch_size=64, shuffle=True)

    final_model = VoiceCNN1D(n_channels=N_CHANNELS).to(device)
    opt_final   = optim.AdamW(final_model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched_final = optim.lr_scheduler.CosineAnnealingLR(opt_final, T_max=50)
    loss_fn     = FocalLoss(alpha=0.75, gamma=2.0)

    final_model.train()
    for epoch in range(50):
        ep_loss = 0.0
        for xb, yb in loader_full:
            xb, yb = xb.to(device), yb.to(device)
            logits = final_model(xb)
            loss   = loss_fn(logits, yb)
            opt_final.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            opt_final.step()
            ep_loss += loss.item()
        sched_final.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/50  loss={ep_loss/len(loader_full):.4f}")

    final_model.eval()

    # ── Save ──────────────────────────────────────────────────────────────────
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_data = {
        'model_state':       final_model.state_dict(),
        'model_arch':        'VoiceCNN1D_v4',
        'n_channels':        N_CHANNELS,
        't_max':             T_MAX,
        'n_mfcc':            N_MFCC,
        'username':          username,
        'user_id':           user_id,
        'n_enrollment':      n_genuine_real,
        # Mahalanobis profile (non-CMVN features, for fusion)
        'raw_profile_mean':  raw_profile_mean,
        'raw_profile_std':   raw_profile_std,
        # Aggregated profile (kept for fallback)
        'profile_mean':      profile_mean,
        'profile_std':       profile_std,
        'threshold':         final_thresh,
        'eer':               float(best_eer),
    }

    model_path = os.path.join(model_dir, f"{_safe_filename(username)}_voice_cnn.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    size_kb = os.path.getsize(model_path) / 1024
    print(f"\n  ✅ CNN model saved → {model_path}  ({size_kb:.1f} KB)")
    print(f"  Threshold: {final_thresh:.2f}   EER: {best_eer:.2%}")
    print(f"{'='*70}\n")
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_voice(username: str, feature_dict: dict) -> dict:
    """
    Authenticate a voice sample using the stored CNN model.

    feature_dict must include:
      mfcc_frames       : list of T lists of 13 floats  (raw MFCC sequence)
      mfcc_std          : list of 13 floats
      delta_mfcc_mean   : list of 13 floats
      pitch_mean, pitch_std, speaking_rate, energy_mean, energy_std,
      zcr_mean, spectral_centroid_mean, spectral_rolloff_mean,
      spectral_flux_mean, voiced_fraction
    """
    model_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_path = os.path.join(model_dir, f"{_safe_filename(username)}_voice_cnn.pkl")

    if not os.path.exists(model_path):
        return {'error': f'No voice model for "{username}". Enroll first.'}

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # ── CNN forward pass ──────────────────────────────────────────────────────
    mfcc_frames = feature_dict.get('mfcc_frames')
    cnn_score   = 0.0

    if TORCH_AVAILABLE and mfcc_frames and len(mfcc_frames) >= 10:
        try:
            seq    = frames_to_tensor(mfcc_frames)                           # (39, T_MAX)
            seq_t  = torch.from_numpy(seq).unsqueeze(0)                      # (1, 39, T_MAX)

            arch       = model_data.get('model_arch', '')
            n_channels = model_data.get('n_channels', N_CHANNELS)

            # Reconstruct model and load weights
            model = VoiceCNN1D(n_channels=n_channels)
            model.load_state_dict(model_data['model_state'])
            model.eval()

            with torch.no_grad():
                logit    = model(seq_t)
                cnn_score = float(torch.sigmoid(logit).item())

        except Exception as e:
            print(f"  ⚠  CNN forward pass failed: {e} — using 0.0")
            cnn_score = 0.0
    else:
        if not mfcc_frames:
            print("  ⚠  No mfcc_frames in payload — CNN score = 0.0")
        cnn_score = 0.0

    # ── Mahalanobis on raw (non-CMVN) profile ────────────────────────────────
    raw_profile_mean = model_data.get('raw_profile_mean')
    raw_profile_std  = model_data.get('raw_profile_std')

    if raw_profile_mean is not None:
        mfcc_std    = list(feature_dict.get('mfcc_std',        [0]*13))
        delta_mean  = list(feature_dict.get('delta_mfcc_mean', [0]*13))
        while len(mfcc_std)   < 13: mfcc_std.append(0.0)
        while len(delta_mean) < 13: delta_mean.append(0.0)

        raw_inf_vec = np.array(
            mfcc_std[:13] + delta_mean[:13] + [
                float(feature_dict.get('pitch_mean',             0)),
                float(feature_dict.get('pitch_std',              0)),
                float(feature_dict.get('speaking_rate',          0)),
                float(feature_dict.get('energy_mean',            0)),
                float(feature_dict.get('energy_std',             0)),
                float(feature_dict.get('zcr_mean',               0)),
                float(feature_dict.get('spectral_centroid_mean', 0)),
                float(feature_dict.get('spectral_rolloff_mean',  0)),
                float(feature_dict.get('spectral_flux_mean',     0)),
                float(feature_dict.get('voiced_fraction',        0)),
            ],
            dtype=np.float64
        )
        mah_score = mahalanobis_score(raw_inf_vec, raw_profile_mean, raw_profile_std)
    else:
        mah_score = 0.5  # neutral fallback

    # ── Fuse CNN + Mahalanobis ────────────────────────────────────────────────
    from utils.fusion import fuse_voice_scores
    fused     = fuse_voice_scores(cnn_score, mah_score)
    threshold = model_data['threshold']
    match     = fused >= threshold

    print(f"\n  Voice CNN '{username}': "
          f"cnn={cnn_score:.4f}  mah={mah_score:.4f}  "
          f"fused={fused:.4f}  thresh={threshold:.2f}  "
          f"→ {'✅ MATCH' if match else '❌ REJECT'}")

    return {
        'match':       bool(match),
        'confidence':  round(cnn_score * 100, 2),
        'mahalanobis': round(mah_score * 100, 2),
        'fused_score': round(fused * 100, 2),
        'threshold':   round(threshold * 100, 2),
        'far':         0.0,
        'frr':         0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("username", nargs="?", default=None)
    parser.add_argument("--lock", default=None)
    args = parser.parse_args()

    username = args.username or input("Username: ").strip()
    try:
        train_voice_model(username)
    finally:
        if args.lock and os.path.exists(args.lock):
            try:
                os.remove(args.lock)
            except Exception:
                pass