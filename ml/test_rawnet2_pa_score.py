"""
RawNet2-PA anti-spoofing scorer (replay-trained).

Loads a RawNet2 checkpoint trained on ASVspoof 2021 PA (Physical Access /
replay attacks) and scores audio files. Output: bonafide probability —
higher = live, lower = replay.

Setup options:
  Option A — official baseline (recommended):
    git clone https://github.com/asvspoof-challenge/2021.git ml/asvspoof2021
    # The PA RawNet2 baseline is at: ml/asvspoof2021/PA/Baseline-RawNet2/
    # If pretrained weights are not bundled, train briefly on ASVspoof 2019 PA:
    #   download dataset: https://datashare.ed.ac.uk/handle/10283/3336
    #   then run their main_PA.py for ~10 epochs (a few hours on CPU/GPU)
    # OR find a community-hosted pretrained checkpoint:
    #   https://github.com/eurecom-asp/rawnet2-antispoofing  (LA, not PA — wrong)
    #   search "rawnet2 PA pretrained pth" on Hugging Face / GitHub releases

  Option B — fallback: train a tiny model on YOUR captures
    (script not included here — would take genuine/ + ai_replay/ as classes
     and train a 2-layer CNN on log-mel spectrograms. Ask Claude to draft it
     if Option A blocks you.)

Usage:
  python ml/test_rawnet2_pa_score.py --ckpt path/to/rawnet2_pa.pth \
      ml/antispoof_samples/genuine ml/antispoof_samples/ai_replay
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

REPO_DIR = Path(__file__).resolve().parent / "asvspoof2021" / "PA" / "Baseline-RawNet2"
TARGET_SR = 16000
N_SAMPLES = 64600  # ~4s, RawNet2 standard input length


def load_model(ckpt_path: Path):
    if not REPO_DIR.exists():
        raise SystemExit(
            f"Missing baseline repo at {REPO_DIR}.\n"
            "Run: git clone https://github.com/asvspoof-challenge/2021.git ml/asvspoof2021"
        )
    sys.path.insert(0, str(REPO_DIR))
    # The baseline defines the model in model.py
    from model import RawNet  # type: ignore
    import yaml
    cfg_path = REPO_DIR / "model_config_RawNet.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNet(cfg["model"], device).to(device)
    state = torch.load(str(ckpt_path), map_location=device)
    # Accept both raw state_dict and {"model_state_dict": ...} formats
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if len(missing) > 5:
            print("       large mismatch — checkpoint may be wrong architecture")
    model.eval()
    return model, device


def load_audio(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    y = y.astype(np.float32)
    if len(y) < N_SAMPLES:
        # tile-pad like the baseline does
        reps = int(np.ceil(N_SAMPLES / len(y)))
        y = np.tile(y, reps)[:N_SAMPLES]
    else:
        y = y[:N_SAMPLES]
    return y


def score(model, device, path: Path) -> float:
    y = load_audio(path)
    x = torch.from_numpy(y).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        # RawNet2 returns 2-class logits; index 1 = bonafide in the baseline
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
    return prob


def collect(args):
    files = []
    for a in args:
        p = Path(a)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}:
                    files.append(f)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to RawNet2-PA .pth checkpoint")
    ap.add_argument("paths", nargs="+", help="wav files or directories")
    args = ap.parse_args()

    print(f"Loading RawNet2-PA from {args.ckpt}...")
    model, device = load_model(Path(args.ckpt))

    files = collect(args.paths)
    if not files:
        print("No audio found.")
        sys.exit(1)

    print(f"\n{'file':<55} {'bonafide':>10}  verdict")
    print("-" * 80)
    by_dir = {}
    for f in files:
        s = score(model, device, f)
        verdict = "LIVE  " if s >= 0.5 else "SPOOF "
        print(f"{f.name:<55} {s:>10.4f}  {verdict}")
        by_dir.setdefault(f.parent.name, []).append(s)

    if len(by_dir) > 1:
        print("\n" + "=" * 80)
        for d, ss in by_dir.items():
            print(f"  {d:<20}  n={len(ss):<3}  mean={np.mean(ss):.4f}  min={min(ss):.4f}  max={max(ss):.4f}")


if __name__ == "__main__":
    main()
