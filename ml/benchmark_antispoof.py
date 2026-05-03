"""
Cheap anti-spoof benchmark.

Computes orthogonal liveness signals on every audio file in a folder
(any format librosa can read: wav, mp3, m4a, flac, ogg).

Signals:
  hf_ratio    energy above 4 kHz / total energy.
              Phone-speaker -> mic recapture band-limits to ~3.4-7 kHz,
              so this collapses on replay. Genuine 16 kHz mic capture
              has meaningful HF content.
  hf7_ratio   same but cutoff at 7 kHz. Tighter replay signal.
  spec_flat   spectral flatness, mean over frames. TTS vocoders often
              produce flatter (more noise-like) high-band spectra than
              real speech.
  rolloff95   95th-percentile spectral rolloff in Hz. Drops sharply on
              band-limited replay.

Usage:
  python ml/benchmark_antispoof.py <folder_or_file> [<folder_or_file> ...]

Convention for organizing samples:
  ml/antispoof_samples/genuine/*.wav     real you, direct mic
  ml/antispoof_samples/ai_clone/*.mp3    TTS / voice clone
  ml/antispoof_samples/replay/*.wav      played through speaker, recaptured

Pass each subfolder as an argument and compare distributions.
"""
import sys
from pathlib import Path
import numpy as np
import librosa

TARGET_SR = 16000
HF_CUTOFF_LOW = 4000
HF_CUTOFF_HIGH = 7000


def analyze(path: Path) -> dict:
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    if y.size == 0:
        return {"error": "empty"}

    # Trim leading/trailing silence so band stats reflect speech, not silence.
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    if y_trim.size < TARGET_SR // 4:
        y_trim = y

    n_fft = 2048
    S = np.abs(librosa.stft(y_trim, n_fft=n_fft, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=n_fft)

    total = S.sum()
    hf4 = S[freqs >= HF_CUTOFF_LOW].sum() / max(total, 1e-12)
    hf7 = S[freqs >= HF_CUTOFF_HIGH].sum() / max(total, 1e-12)

    flat = float(librosa.feature.spectral_flatness(S=np.sqrt(S)).mean())
    rolloff = float(np.median(
        librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=TARGET_SR, roll_percent=0.95)
    ))

    return {
        "duration_s": round(len(y_trim) / TARGET_SR, 2),
        "hf4_ratio":  round(float(hf4), 4),
        "hf7_ratio":  round(float(hf7), 4),
        "spec_flat":  round(flat, 4),
        "rolloff95":  round(rolloff, 1),
    }


def collect(targets):
    files = []
    for t in targets:
        p = Path(t)
        if p.is_file():
            files.append((p.parent.name or "root", p))
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
                    files.append((p.name, f))
    return files


def main():
    targets = sys.argv[1:] or ["josh.mp3"]
    files = collect(targets)
    if not files:
        print("No audio files found.")
        sys.exit(1)

    print(f"{'group':<12} {'file':<40} {'dur':>5} {'hf4':>7} {'hf7':>7} {'flat':>7} {'rolloff':>9}")
    print("-" * 90)

    by_group = {}
    for group, path in files:
        try:
            r = analyze(path)
        except Exception as e:
            print(f"{group:<12} {path.name:<40} ERROR: {e}")
            continue
        if "error" in r:
            print(f"{group:<12} {path.name:<40} {r['error']}")
            continue
        print(f"{group:<12} {path.name:<40} "
              f"{r['duration_s']:>5} {r['hf4_ratio']:>7.4f} {r['hf7_ratio']:>7.4f} "
              f"{r['spec_flat']:>7.4f} {r['rolloff95']:>9.1f}")
        by_group.setdefault(group, []).append(r)

    if len(by_group) > 1:
        print("\n" + "=" * 90)
        print(f"{'group':<12} {'n':>3} {'hf4_mean':>10} {'hf7_mean':>10} {'flat_mean':>10} {'rolloff_mean':>14}")
        print("-" * 90)
        for g, rs in by_group.items():
            n = len(rs)
            print(f"{g:<12} {n:>3} "
                  f"{np.mean([r['hf4_ratio']  for r in rs]):>10.4f} "
                  f"{np.mean([r['hf7_ratio']  for r in rs]):>10.4f} "
                  f"{np.mean([r['spec_flat']  for r in rs]):>10.4f} "
                  f"{np.mean([r['rolloff95']  for r in rs]):>14.1f}")


if __name__ == "__main__":
    main()
