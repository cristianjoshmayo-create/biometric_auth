"""
run_all_benchmarks.py
─────────────────────────────────────────────────────────────────────────────
Thesis Chapter 4 — Combined Benchmark Runner

Runs both keystroke dynamics and voice biometrics benchmarks for a given
user and produces a unified results summary suitable for the thesis.

Usage
─────
  python run_all_benchmarks.py <username>
  python run_all_benchmarks.py <username> --keystroke-only
  python run_all_benchmarks.py <username> --voice-only

Output
──────
  <username>_chapter4_results.txt  — full console output captured to file
  <username>_keystroke_benchmark.csv
  <username>_voice_benchmark.csv
"""

import sys
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure this folder is on sys.path so siblings (keystroke_benchmark,
# voice_benchmark, eval_utils) can always be imported with plain names
# regardless of how/where Python is invoked.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Chapter 4 Benchmark Runner")
    parser.add_argument("username", nargs="?", default=None)
    parser.add_argument("--keystroke-only", action="store_true")
    parser.add_argument("--voice-only",     action="store_true")
    args = parser.parse_args()

    username = args.username or input("Enter username: ").strip()

    print(f"\n{'█'*70}")
    print(f"  THESIS CHAPTER 4 — BIOMETRIC AUTHENTICATION BENCHMARK")
    print(f"  User        : {username}")
    print(f"  Started at  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*70}")

    all_results = []

    # ── Keystroke benchmark ───────────────────────────────────────────────────
    if not args.voice_only:
        print(f"\n\n{'▓'*70}")
        print(f"  PART 1 — KEYSTROKE DYNAMICS")
        print(f"{'▓'*70}")
        try:
            from keystroke_benchmark import main as ks_main, build_dataset as ks_build
            ks_main(username)
        except Exception as e:
            print(f"  ❌ Keystroke benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Voice benchmark ───────────────────────────────────────────────────────
    if not args.keystroke_only:
        print(f"\n\n{'▓'*70}")
        print(f"  PART 2 — VOICE BIOMETRICS")
        print(f"{'▓'*70}")
        try:
            from voice_benchmark import main as v_main
            v_main(username)
        except Exception as e:
            print(f"  ❌ Voice benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Multimodal fusion note ────────────────────────────────────────────────
    if not args.keystroke_only and not args.voice_only:
        print(f"\n{'═'*70}")
        print("  MULTIMODAL FUSION NOTE")
        print(f"{'═'*70}")
        print("""
  The production system combines keystroke and voice authentication via
  a serial (AND-gate) fusion strategy:
    1. Keystroke dynamics check (first factor)
    2. Voice biometrics check   (second factor)

  Both must pass independently.  This yields a combined FAR ≈ FAR_ks × FAR_v
  (approximate, assuming independence).

  For example:
    FAR_keystroke = 3.1%  →  FAR_voice = 2.0%
    Combined FAR  ≈ 0.031 × 0.020 ≈ 0.062%   (~16× better than either alone)

  The combined FRR is FAR_ks + FAR_v (either check can reject a genuine user).
  This trade-off between FAR and FRR in a multimodal system is discussed in
  Section 4.5 of the thesis.
""")

    print(f"\n{'█'*70}")
    print(f"  BENCHMARK COMPLETE  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results CSV files saved to: {SCRIPT_DIR}")
    print(f"{'█'*70}\n")


if __name__ == "__main__":
    main()