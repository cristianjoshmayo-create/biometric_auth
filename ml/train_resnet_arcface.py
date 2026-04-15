# ml/train_resnet_arcface.py
# ResNet-34 + ArcFace Speaker Verification
# Trained on LibriSpeech train-clean-100 + train-clean-360 (1,172 speakers)
#
# Architecture:
#   Input  : Mel-Spectrogram (1, 80, T)
#   Backbone: Thin ResNet-34 (channels: 32, 64, 128, 256)
#   Pooling : Attentive Statistics Pooling (mean + std weighted by attention)
#   Embedding: 256-dim L2-normalized speaker embedding
#   Loss   : ArcFace (margin=0.2, scale=30) during training only
#   Verify : Cosine similarity at inference
#
# Run:
#   python train_resnet_arcface.py
#
# Requirements:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#   pip install librosa scikit-learn matplotlib tqdm numpy

import os
import math
import bisect
import time
import json
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import Counter, defaultdict

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = "./data/librispeech"
MODEL_DIR   = "./models"
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Feature config
SAMPLE_RATE = 16000
N_MELS      = 80       # mel bins
N_FFT       = 512
HOP_LENGTH  = 160      # 10ms at 16kHz
WIN_LENGTH  = 400      # 25ms at 16kHz
T_MAX       = 300      # fixed time frames (~3 seconds)

# Model config
EMBED_DIM   = 256      # speaker embedding dimension
DROPOUT     = 0.2

# ArcFace config  (matched to clovaai/voxceleb_trainer reference)
ARC_MARGIN    = 0.3   # clovaai default; 0.5 over-penalises random init
ARC_SCALE     = 15    # clovaai default; 30 doubles gradient magnitude → NaN
WARMUP_EPOCHS = 3     # ramp margin 0 → ARC_MARGIN over first 3 epochs

# Training config
BATCH_SIZE  = 64
NUM_EPOCHS  = 30
LR          = 1e-3   # clovaai reference value; safe with scale=15
WEIGHT_DECAY= 1e-4
NUM_WORKERS = 4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TF32: uses Tensor Cores for fp32 matmuls on Ampere/Ada Lovelace GPUs.
# Near fp16 speed, full fp32 exponent range (no overflow → no NaN).
# Enabled by default on Ampere+ but setting explicitly ensures it's on.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32      = True

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    f_min=20,
    f_max=8000,
    power=2.0,
)
amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)

resampler_cache = {}

def extract_melspec(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Convert raw audio to log mel-spectrogram.
    waveform : (1, N) tensor
    returns  : (1, N_MELS, T_MAX) tensor  — ready for 2D CNN
    """
    # Resample to 16kHz if needed
    if sample_rate != SAMPLE_RATE:
        if sample_rate not in resampler_cache:
            resampler_cache[sample_rate] = T.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler_cache[sample_rate](waveform)

    # Stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize amplitude
    max_amp = waveform.abs().max()
    if max_amp > 1e-6:
        waveform = waveform / max_amp

    # Mel spectrogram: (1, N_MELS, T)
    mel = mel_transform(waveform)
    mel = amplitude_to_db(mel)

    # Normalize to zero mean, unit variance
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    # Pad or random crop to T_MAX
    T = mel.shape[2]
    if T < T_MAX:
        mel = F.pad(mel, (0, T_MAX - T))
    else:
        start = random.randint(0, T - T_MAX)
        mel   = mel[:, :, start:start + T_MAX]

    return mel  # (1, 80, 300)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class LibriSpeechMelSpec(Dataset):
    """
    Wraps LibriSpeech dataset.
    Returns (mel_spectrogram, speaker_label)
      mel_spectrogram : (1, 80, 300) float32
      speaker_label   : int
    """
    def __init__(self, librispeech_ds, speaker_to_label: dict, augment: bool = True):
        self.dataset          = librispeech_ds
        self.speaker_to_label = speaker_to_label
        self.augment          = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sr, _, speaker_id, _, _ = self.dataset[idx]
        label = self.speaker_to_label[speaker_id]
        mel   = extract_melspec(waveform, sr)

        # Guard: corrupt FLAC files can produce NaN/inf mel.
        # Replace with zeros so the training-loop finite check skips the batch.
        if not torch.isfinite(mel).all():
            mel = torch.zeros(1, N_MELS, T_MAX)

        if self.augment:
            mel = self._augment(mel)
        return mel, label

    def _augment(self, mel: torch.Tensor) -> torch.Tensor:
        """
        SpecAugment: time masking + frequency masking + noise
        Makes model robust to mic differences and background noise.
        """
        mel = mel.clone()
        # Time masking — zero up to 40 frames
        if random.random() < 0.5:
            t = random.randint(1, 40)
            s = random.randint(0, T_MAX - t)
            mel[:, :, s:s + t] = 0.0
        # Frequency masking — zero up to 15 mel bins
        if random.random() < 0.5:
            f = random.randint(1, 15)
            s = random.randint(0, N_MELS - f)
            mel[:, s:s + f, :] = 0.0
        # Additive Gaussian noise
        if random.random() < 0.3:
            mel = mel + torch.randn_like(mel) * 0.05
        return mel


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL — THIN RESNET-34 + ATTENTIVE STATS POOLING + ARCFACE
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Basic ResNet residual block with skip connection.
    Input and output have the same number of channels.
    If downsample=True, spatial dims are halved and channels doubled.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Skip connection: match dimensions if stride > 1 or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # residual / skip connection
        return F.relu(out)


class AttentiveStatsPooling(nn.Module):
    """
    Attentive Statistics Pooling.
    Learns which time frames are more speaker-discriminative
    and weights them accordingly before computing mean + std.

    Input : (batch, channels, time)
    Output: (batch, 2 * channels)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, in_channels, kernel_size=1),
            nn.Softmax(dim=2),  # softmax over time
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C, T)
        weights = self.attention(x)          # (batch, C, T)
        mean    = (weights * x).sum(dim=2)   # weighted mean
        # clamp before sqrt: fp16 rounding can produce tiny negatives → NaN
        var     = (weights * (x - mean.unsqueeze(2)) ** 2).sum(dim=2).clamp(min=0)
        std     = var.sqrt() + 1e-4  # 1e-8 underflows to 0 in fp16; 1e-4 is safe
        return torch.cat([mean, std], dim=1) # (batch, 2C)


class ThinResNet34(nn.Module):
    """
    Thin ResNet-34 for speaker verification.

    "Thin" means we use half the standard ResNet channels:
      Standard ResNet-34 : [64, 128, 256, 512]
      Thin ResNet-34     : [32,  64, 128, 256]

    This fits in 6 GB VRAM and trains faster while still
    achieving strong speaker verification performance.

    Input  : (batch, 1, N_MELS, T_MAX) — 2D mel-spectrogram
    Output : (batch, EMBED_DIM)         — L2-normalized speaker embedding
    """

    def __init__(self, embed_dim: int = 256, n_speakers: int = 1172,
                 dropout: float = 0.2):
        super().__init__()

        # ── Initial conv ─────────────────────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # ── ResNet-34 blocks (thin channels: 32, 64, 128, 256) ───────────────
        # ResNet-34 block counts: [3, 4, 6, 3]
        self.layer1 = self._make_layer(32,  32,  n_blocks=3, stride=1)
        self.layer2 = self._make_layer(32,  64,  n_blocks=4, stride=2)
        self.layer3 = self._make_layer(64,  128, n_blocks=6, stride=2)
        self.layer4 = self._make_layer(128, 256, n_blocks=3, stride=2)

        # ── Collapse frequency dim → time sequence ────────────────────────────
        # After 4 layers with strides, freq dim = 80 / (2^3) = 10
        # We reshape (batch, 256, freq, time) → (batch, 256*freq, time)
        self.freq_collapse = nn.AdaptiveAvgPool2d((1, None))  # (batch, 256, 1, T)

        # ── Attentive statistics pooling (over time) ──────────────────────────
        self.stats_pool = AttentiveStatsPooling(256)  # → (batch, 512)

        # ── Embedding layer ───────────────────────────────────────────────────
        self.embed_fc = nn.Linear(512, embed_dim)
        self.embed_bn = nn.BatchNorm1d(embed_dim)
        self.dropout  = nn.Dropout(dropout)

        # ── ArcFace classification head (training only) ───────────────────────
        # Weight matrix W: (n_speakers, embed_dim) — each row is a class center
        self.arc_weight = nn.Parameter(torch.FloatTensor(n_speakers, embed_dim))
        nn.init.xavier_uniform_(self.arc_weight)

        self._init_weights()

    def _make_layer(self, in_ch: int, out_ch: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized 256-dim speaker embedding.
        Use this at inference time.
        x : (batch, 1, 80, 300)
        returns: (batch, 256)
        """
        x = self.conv1(x)           # (B, 32, 80, 300)
        x = self.layer1(x)          # (B, 32, 80, 300)
        x = self.layer2(x)          # (B, 64, 40, 150)
        x = self.layer3(x)          # (B, 128, 20, 75)
        x = self.layer4(x)          # (B, 256, 10, 38)
        x = self.freq_collapse(x)   # (B, 256, 1, 38)
        x = x.squeeze(2)            # (B, 256, 38)
        x = self.stats_pool(x)      # (B, 512)
        x = self.embed_fc(x)        # (B, 256)
        x = self.embed_bn(x)
        x = self.dropout(x)         # dropout before normalization, not after
        x = F.normalize(x.float(), p=2, dim=1)
        return x

    def arcface_logits(self, embeddings: torch.Tensor,
                       labels: torch.Tensor,
                       margin: float, scale: float) -> torch.Tensor:
        """
        ArcFace loss logits — stable formulation without torch.acos.

        cos(θ + m) = cos(θ)·cos(m) − sin(θ)·sin(m)
        sin(θ)     = sqrt(1 − cos²(θ))

        Avoids acos whose backward (-1/sqrt(1−x²)) blows up near ±1.
        Easy-margin guard: when θ + m > π, clamp to cosine − sin(m)·m
        so the target logit never goes more negative than a linear fallback.
        """
        embeddings = embeddings.float()
        W = F.normalize(self.arc_weight.float(), p=2, dim=1)  # (n_speakers, 256)

        # Cosine similarities: (batch, n_speakers)
        cosine = F.linear(embeddings, W).clamp(-1 + 1e-7, 1 - 1e-7)

        # Precompute margin terms (scalars — computed once per forward)
        cos_m = math.cos(margin)
        sin_m = math.sin(margin)
        # Easy-margin threshold: cos(π − m).  When cosine < threshold,
        # θ > π − m, meaning θ + m > π — the margin would over-penalise.
        threshold = math.cos(math.pi - margin)   # = −cos(m)
        easy_mm   = sin_m * margin               # linear fallback shift

        # cos(θ + m) via trig identity — no acos needed
        sin_theta  = (1.0 - cosine ** 2).clamp(min=0).sqrt()
        phi        = cosine * cos_m - sin_theta * sin_m   # target class logit

        # Easy margin: replace phi with cosine − easy_mm when cosine < threshold
        phi = torch.where(cosine > threshold, phi, cosine - easy_mm)

        # Build logit matrix: target class gets phi, others keep cosine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits  = scale * (one_hot * phi + (1.0 - one_hot) * cosine)

        return logits

    def forward(self, x: torch.Tensor,
                labels: torch.Tensor = None,
                margin: float = ARC_MARGIN) -> torch.Tensor:
        """
        Training forward pass.
        x      : (batch, 1, 80, 300)
        labels : (batch,) speaker labels
        margin : ArcFace angular margin (0 during warmup epochs)
        returns: (batch, n_speakers) ArcFace logits
        """
        emb    = self.get_embedding(x)
        logits = self.arcface_logits(emb, labels, margin, ARC_SCALE)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, margin):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch_idx, (mel, labels) in enumerate(loader):
        mel    = mel.to(device)   # blocking transfer — guarantees data is on GPU
        labels = labels.to(device)

        # Skip batches with corrupt mel (NaN/inf in input → NaN loss)
        if not torch.isfinite(mel).all():
            print(f"\n  WARNING: non-finite mel at batch {batch_idx+1}, skipping")
            continue

        optimizer.zero_grad()

        logits = model(mel, labels, margin=margin)
        loss   = criterion(logits, labels)

        # Skip NaN/inf loss batches rather than aborting — weights stay intact,
        # training continues.  Mirrors GradScaler's behaviour with inf gradients.
        if not torch.isfinite(loss):
            print(f"\n  WARNING: non-finite loss={loss.item():.4f} "
                  f"at batch {batch_idx+1}, skipping")
            continue

        loss.backward()

        # Check for NaN/inf gradients BEFORE clipping.
        # clip_grad_norm_ on inf grads computes inf * 0 = NaN → corrupts weights.
        grad_ok = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None
        )
        if not grad_ok:
            print(f"\n  WARNING: NaN/inf gradients at batch {batch_idx+1}, skipping")
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * mel.size(0)
        total_samples += mel.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  "
                  f"acc={total_correct/total_samples*100:.1f}%", end='\r')

    return total_loss / total_samples, total_correct / total_samples * 100


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for mel, labels in loader:
        mel    = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(mel, labels)
        loss   = criterion(logits, labels)

        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * mel.size(0)
        total_samples += mel.size(0)

    return total_loss / total_samples, total_correct / total_samples * 100


def _get_val_labels(val_ds) -> list:
    """
    Get speaker labels for all samples in val_ds without loading audio.
    val_ds is a Subset of a ConcatDataset of LibriSpeechMelSpec datasets.
    Uses _walker (list of (speaker_id, chapter_id, utterance_id)) directly.
    """
    concat_ds = val_ds.dataset          # ConcatDataset
    cum_sizes = concat_ds.cumulative_sizes
    result = []
    for idx in val_ds.indices:
        ds_idx     = bisect.bisect_right(cum_sizes, idx)
        sample_idx = idx - (cum_sizes[ds_idx - 1] if ds_idx > 0 else 0)
        sub_ds     = concat_ds.datasets[ds_idx]   # LibriSpeechMelSpec
        fileid     = sub_ds.dataset._walker[sample_idx]  # "speaker-chapter-utterance"
        speaker_id = int(fileid.split('-')[0])
        result.append(sub_ds.speaker_to_label[speaker_id])
    return result


@torch.no_grad()
def compute_eer(model, dataset, val_labels: list,
                n_trials: int = 2000, device: str = 'cuda'):
    """
    Compute Equal Error Rate on verification trials.
    Genuine pairs  : same speaker, different utterances
    Impostor pairs : different speakers

    val_labels: precomputed list of integer speaker labels (no audio load).
    """
    from sklearn.metrics import roc_curve

    model.eval()

    # Build speaker → sample index map from precomputed labels (no mel extraction)
    speaker_indices = defaultdict(list)
    for idx, label in enumerate(val_labels):
        speaker_indices[label].append(idx)

    valid_speakers = [s for s, idxs in speaker_indices.items() if len(idxs) >= 2]
    print(f"  Speakers with >=2 utterances: {len(valid_speakers)}")

    n_half = n_trials // 2

    # Pre-sample all trial pairs to collect unique indices for batched embedding
    genuine_pairs, impostor_pairs = [], []
    for _ in range(n_half):
        spk  = random.choice(valid_speakers)
        idxs = speaker_indices[spk]
        i, j = random.sample(range(len(idxs)), 2)
        genuine_pairs.append((idxs[i], idxs[j]))

    for _ in range(n_half):
        s1, s2 = random.sample(valid_speakers, 2)
        impostor_pairs.append((random.choice(speaker_indices[s1]),
                               random.choice(speaker_indices[s2])))

    all_pairs   = genuine_pairs + impostor_pairs
    pair_labels = [1] * n_half + [0] * n_half

    # Compute embeddings in batches for all unique indices (avoids 4K individual fwd passes)
    unique_idx = list({idx for pair in all_pairs for idx in pair})
    emb_cache  = {}
    batch_size  = 32
    for start in range(0, len(unique_idx), batch_size):
        batch_idx = unique_idx[start:start + batch_size]
        mels = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        embs = model.get_embedding(mels)
        for i, idx in enumerate(batch_idx):
            emb_cache[idx] = embs[i].cpu()

    # Score all pairs
    scores = []
    for (idx1, idx2) in all_pairs:
        e1 = emb_cache[idx1].unsqueeze(0)
        e2 = emb_cache[idx2].unsqueeze(0)
        scores.append(F.cosine_similarity(e1, e2).item())

    scores = np.array(scores)
    labels = np.array(pair_labels)

    # Drop NaN scores — caused by zero-norm embeddings (F.normalize(0) = NaN)
    valid     = ~np.isnan(scores)
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f"  WARNING: {n_dropped} NaN scores dropped (zero-norm embeddings)")
    scores = scores[valid]
    labels = labels[valid]

    if len(scores) < 10:
        print("  ERROR: Too few valid scores to compute EER")
        return float('nan'), 0.5

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr    = 1 - tpr
    idx    = np.nanargmin(np.abs(fpr - fnr))
    eer    = (fpr[idx] + fnr[idx]) / 2 * 100
    thresh = float(thresholds[idx])

    return float(eer), thresh


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  ResNet-34 + ArcFace Speaker Verification")
    print("  Dataset : LibriSpeech train-clean-100 + train-clean-360")
    print("=" * 65 + "\n")

    print(f"Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Download datasets ─────────────────────────────────────────────────────
    print("Step 1/6 — Downloading LibriSpeech (saved to ./data/librispeech)")
    print("  train-clean-100 (~6 GB) ...")
    ds_100 = torchaudio.datasets.LIBRISPEECH(
        root=DATA_DIR, url='train-clean-100', download=True)
    print(f"  train-clean-100 loaded  : {len(ds_100):,} utterances")

    print("  train-clean-360 (~23 GB) ...")
    ds_360 = torchaudio.datasets.LIBRISPEECH(
        root=DATA_DIR, url='train-clean-360', download=True)
    print(f"  train-clean-360 loaded  : {len(ds_360):,} utterances")

    # ── Validate audio files ──────────────────────────────────────────────────
    print("\nStep 1b — Scanning for corrupt audio files ...")
    bad_files = []
    for ds_name, ds in [('train-clean-100', ds_100), ('train-clean-360', ds_360)]:
        for i, fileid in enumerate(ds._walker):
            try:
                waveform, sr, *_ = ds[i]
                if not torch.isfinite(waveform).all():
                    bad_files.append((ds_name, fileid))
                    print(f"  CORRUPT (NaN/inf): {ds_name}/{fileid}")
                elif waveform.abs().max() < 1e-9:
                    print(f"  SILENT           : {ds_name}/{fileid}")
            except Exception as e:
                bad_files.append((ds_name, fileid))
                print(f"  UNREADABLE       : {ds_name}/{fileid}  ({e})")
            if (i + 1) % 10000 == 0:
                print(f"  Scanned {i+1:,}/{len(ds._walker):,} files in {ds_name} ...")
    if bad_files:
        print(f"\n  Found {len(bad_files)} corrupt file(s) — they will be skipped during training.")
    else:
        print("  All files OK.")

    # ── Build speaker map ─────────────────────────────────────────────────────
    print("\nStep 2/6 — Building speaker map ...")
    speaker_counter = Counter()
    for ds in [ds_100, ds_360]:
        # _walker holds "speaker-chapter-utterance" strings — no audio load needed
        for fileid in ds._walker:
            speaker_id = int(fileid.split('-')[0])
            speaker_counter[speaker_id] += 1

    all_speakers     = sorted(speaker_counter.keys())
    speaker_to_label = {spk: idx for idx, spk in enumerate(all_speakers)}
    N_SPEAKERS       = len(all_speakers)

    print(f"  Total speakers   : {N_SPEAKERS}")
    print(f"  Total utterances : {sum(speaker_counter.values()):,}")

    speaker_map_path = os.path.join(MODEL_DIR, 'speaker_to_label.pkl')
    with open(speaker_map_path, 'wb') as f:
        pickle.dump({'speaker_to_label': speaker_to_label,
                     'label_to_speaker': {v: k for k, v in speaker_to_label.items()},
                     'n_speakers': N_SPEAKERS}, f)
    print(f"  Speaker map saved: {speaker_map_path}")

    # ── Build datasets ────────────────────────────────────────────────────────
    print("\nStep 3/6 — Building PyTorch datasets ...")
    full_100_train = LibriSpeechMelSpec(ds_100, speaker_to_label, augment=True)
    full_360_train = LibriSpeechMelSpec(ds_360, speaker_to_label, augment=True)
    full_100_val   = LibriSpeechMelSpec(ds_100, speaker_to_label, augment=False)
    full_360_val   = LibriSpeechMelSpec(ds_360, speaker_to_label, augment=False)

    full_ds_train = ConcatDataset([full_100_train, full_360_train])
    full_ds_val   = ConcatDataset([full_100_val,   full_360_val])

    n_total  = len(full_ds_train)
    n_val    = int(n_total * 0.1)
    n_train  = n_total - n_val

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    train_ds = torch.utils.data.Subset(full_ds_train, indices[:n_train])
    val_ds   = torch.utils.data.Subset(full_ds_val,   indices[n_train:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train : {len(train_ds):,} utterances")
    print(f"  Val   : {len(val_ds):,} utterances")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\nStep 4/6 — Building ResNet-34 + ArcFace model ...")
    model  = ThinResNet34(embed_dim=EMBED_DIM, n_speakers=N_SPEAKERS,
                          dropout=DROPOUT).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {params:,}  (~{params/1e6:.1f}M)")
    print(f"  Model size : ~{params*4/1e6:.1f} MB")

    # Test forward pass
    dummy  = torch.randn(2, 1, N_MELS, T_MAX).to(DEVICE)
    dummy_labels = torch.zeros(2, dtype=torch.long).to(DEVICE)
    out    = model(dummy, dummy_labels)
    emb    = model.get_embedding(dummy)
    print(f"  Output shape    : {out.shape}")
    print(f"  Embedding shape : {emb.shape}")
    print(f"  Embedding norm  : {emb.norm(dim=1).mean().item():.4f}  (should be ~1.0)")

    # ── Training setup ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_acc    = 0.0
    best_model_path = os.path.join(MODEL_DIR, 'resnet34_arcface_best.pt')
    history         = {'train_loss': [], 'train_acc': [],
                       'val_loss':   [], 'val_acc':   []}

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStep 5/6 — Training for {NUM_EPOCHS} epochs ...")
    est_min = len(train_loader) * NUM_EPOCHS * 0.10 / 60
    print(f"  Estimated time : ~{est_min:.0f} minutes on RTX 4050")
    print("=" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        # Linear margin warmup: 0 → ARC_MARGIN over WARMUP_EPOCHS
        # Prevents NaN from huge gradients when embeddings are still random
        warmup_margin = ARC_MARGIN * min(1.0, epoch / WARMUP_EPOCHS)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            margin=warmup_margin)

        val_loss, val_acc = validate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        star = 'NEW BEST ⭐' if val_acc > best_val_acc else ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_acc':     val_acc,
                'n_speakers':  N_SPEAKERS,
                'embed_dim':   EMBED_DIM,
                'n_mels':      N_MELS,
                't_max':       T_MAX,
            }, best_model_path)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
              f"margin={warmup_margin:.2f}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s  {star}")

    # Save training history
    with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f)

    print("\n" + "=" * 65)
    print(f"Training complete")
    print(f"Best val accuracy : {best_val_acc:.2f}%")
    print(f"Model saved       : {best_model_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\nStep 6/6 — Computing EER ...")
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])

    val_labels = _get_val_labels(val_ds)
    eer, threshold = compute_eer(model, val_ds, val_labels, n_trials=2000, device=DEVICE)
    print(f"\n===== FINAL RESULTS =====")
    print(f"EER               : {eer:.2f}%")
    print(f"Optimal threshold : {threshold:.4f}")
    print(f"Best val accuracy : {best_val_acc:.2f}%")

    # ── Export ────────────────────────────────────────────────────────────────
    with open(speaker_map_path, 'rb') as f:
        speaker_map = pickle.load(f)

    export_path = os.path.join(MODEL_DIR, 'resnet34_arcface_export.pt')
    torch.save({
        'model_state': model.state_dict(),
        'architecture': {
            'embed_dim':   EMBED_DIM,
            'n_speakers':  N_SPEAKERS,
            'dropout':     DROPOUT,
        },
        'feature_config': {
            'sample_rate': SAMPLE_RATE,
            'n_mels':      N_MELS,
            'n_fft':       N_FFT,
            'hop_length':  HOP_LENGTH,
            'win_length':  WIN_LENGTH,
            't_max':       T_MAX,
        },
        'eval': {
            'eer_percent':       eer,
            'optimal_threshold': threshold,
            'best_val_accuracy': best_val_acc,
            'training_epochs':   NUM_EPOCHS,
            'dataset':           'LibriSpeech train-clean-100 + train-clean-360',
            'n_speakers':        N_SPEAKERS,
        },
        'speaker_to_label': speaker_map['speaker_to_label'],
    }, export_path)

    size_mb = os.path.getsize(export_path) / 1e6
    print(f"\nModel exported")
    print(f"  Path      : {export_path}")
    print(f"  File size : {size_mb:.1f} MB")
    print(f"\nCopy to biometric_auth/ml/models/resnet34_arcface_export.pt")
    print("Then tell me and I will wire it into the system.")


if __name__ == '__main__':
    main()
