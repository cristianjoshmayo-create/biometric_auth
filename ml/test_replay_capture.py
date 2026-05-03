"""
Replay-attack capture harness.

Procedure:
  1. Script assigns you a random 4-word phrase.
  2. You say it LIVE into your laptop mic       -> saved to genuine/
  3. Script then asks you to play that recording
     through your PHONE SPEAKER while the laptop
     mic re-captures it                         -> saved to ai_replay/
  4. Run the AASIST scorer (test_aasist_score.py)
     on the pair and compare scores.

Usage:
  python ml/test_replay_capture.py
  python ml/test_replay_capture.py --duration 5
"""
import argparse
import random
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf

SR = 16000
ROOT = Path(__file__).resolve().parent / "antispoof_samples"
GENUINE_DIR = ROOT / "genuine"
REPLAY_DIR = ROOT / "ai_replay"

WORDS = [
    "river", "candle", "window", "purple", "garden", "silver", "kitchen",
    "thunder", "morning", "lemon", "pencil", "yellow", "marble", "shadow",
    "forest", "copper", "ocean", "ginger", "velvet", "winter",
]


def pick_phrase(n=4):
    return " ".join(random.sample(WORDS, n))


def record(seconds: int, label: str) -> "np.ndarray":
    print(f"\n[{label}] recording {seconds}s in 3...", end="", flush=True)
    for i in (2, 1):
        time.sleep(1); print(f" {i}...", end="", flush=True)
    time.sleep(1); print(" GO")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    print(f"[{label}] done.")
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--tag", default=None, help="filename tag (default: timestamp)")
    args = ap.parse_args()

    GENUINE_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    phrase = pick_phrase()
    tag = args.tag or f"test_{int(time.time())}"

    print("=" * 60)
    print(f"Assigned phrase:  {phrase!r}")
    print(f"Duration:         {args.duration}s per take")
    print(f"Tag:              {tag}")
    print("=" * 60)

    # --- Take 1: live ---
    input("\nSTEP 1: Press ENTER, then say the phrase LIVE into your laptop mic.")
    live = record(args.duration, "LIVE")
    live_path = GENUINE_DIR / f"{tag}_live.wav"
    sf.write(str(live_path), live, SR)
    print(f"saved -> {live_path}")

    # --- Take 2: replay ---
    print("\nSTEP 2: Now copy/play that file through your PHONE SPEAKER")
    print(f"        File to play:  {live_path}")
    print( "        Hold the phone ~15-30cm from the laptop mic.")
    input("Press ENTER when ready to capture the replay.")
    replay = record(args.duration, "REPLAY")
    replay_path = REPLAY_DIR / f"{tag}_replay.wav"
    sf.write(str(replay_path), replay, SR)
    print(f"saved -> {replay_path}")

    print("\n" + "=" * 60)
    print("Both samples captured. Next:")
    print(f"  python ml/test_aasist_score.py {live_path} {replay_path}")
    print(f"  python ml/benchmark_antispoof.py {GENUINE_DIR} {REPLAY_DIR}")


if __name__ == "__main__":
    main()
