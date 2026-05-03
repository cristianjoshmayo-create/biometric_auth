"""
AASIST anti-spoofing scorer (single-file test).

Loads a pretrained AASIST checkpoint and scores arbitrary audio files.
Output: a "bonafide score" — higher = more likely live, lower = more
likely spoofed (replay / TTS / clone).

First-time setup:
  pip install torch torchaudio soundfile librosa
  # AASIST repo + weights:
  git clone https://github.com/clovaai/aasist.git ml/aasist
  # (the repo ships pretrained weights under aasist/models/weights/)

Usage:
  python ml/test_aasist_score.py <wav> [<wav> ...]
  python ml/test_aasist_score.py ml/antispoof_samples/genuine ml/antispoof_samples/ai_replay
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

AASIST_DIR = Path(__file__).resolve().parent / "aasist"
CKPT = AASIST_DIR / "models" / "weights" / "AASIST.pth"
TARGET_SR = 16000
N_SAMPLES = 64600  # AASIST trained on ~4s @ 16k


def load_model():
    sys.path.insert(0, str(AASIST_DIR))
    from models.AASIST import Model  # type: ignore
    import json
    cfg = json.loads((AASIST_DIR / "config" / "AASIST.conf").read_text())
    model = Model(cfg["model_config"])
    state = torch.load(str(CKPT), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_audio(path: Path) -> torch.Tensor:
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    y = y.astype(np.float32)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]
    return torch.from_numpy(y).unsqueeze(0)


def score(model, path: Path) -> float:
    x = load_audio(path)
    with torch.no_grad():
        _, out = model(x)
        # AASIST returns [spoof_logit, bonafide_logit]; use bonafide prob
        prob = torch.softmax(out, dim=-1)[0, 1].item()
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
    files = collect(sys.argv[1:])
    if not files:
        print("Usage: python ml/test_aasist_score.py <wav_or_dir> ...")
        sys.exit(1)

    print("Loading AASIST...")
    model = load_model()

    print(f"\n{'file':<50} {'bonafide_score':>16}  verdict")
    print("-" * 80)
    for f in files:
        s = score(model, f)
        verdict = "LIVE  " if s >= 0.5 else "SPOOF "
        print(f"{f.name:<50} {s:>16.4f}  {verdict}")


if __name__ == "__main__":
    main()
