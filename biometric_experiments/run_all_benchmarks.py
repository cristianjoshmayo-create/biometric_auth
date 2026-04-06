"""
run_all_benchmarks.py
─────────────────────────────────────────────────────────────────────────────
Thesis Chapter 4 — Combined Benchmark Runner

Supports:
  Single user  : python run_all_benchmarks.py <username>
  All users    : python run_all_benchmarks.py --all-users
  Filtered     : python run_all_benchmarks.py --all-users --min-samples 3

Options
───────
  --all-users          Run benchmarks for every enrolled user in the DB
  --keystroke-only     Skip voice benchmarks
  --voice-only         Skip keystroke benchmarks
  --min-samples N      Skip users with fewer than N enrollment samples (default: 3)
  --summary-only       After --all-users run, print the aggregate summary table

Output
──────
  <username>_keystroke_benchmark.csv   — per-user keystroke results
  <username>_voice_benchmark.csv       — per-user voice results
  all_users_summary.csv                — aggregate table across all users
"""

import sys
import os
import argparse
import time
import csv
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
BACKEND_PATH = os.path.join(ROOT_DIR, "backend")
ML_PATH      = os.path.join(ROOT_DIR, "ml")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, BACKEND_PATH)
sys.path.insert(0, ML_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_all_users(min_ks_samples: int = 3, min_voice_samples: int = 1):
    """
    Query the database and return a list of usernames that have enough
    enrollment data to be worth benchmarking.

    Parameters
    ----------
    min_ks_samples    : minimum keystroke enrollment samples required
    min_voice_samples : minimum voice enrollment samples required
                        (set to 0 to include users with no voice data)
    """
    try:
        from database.db import SessionLocal
        from database.models import User, KeystrokeTemplate, VoiceTemplate
        from sqlalchemy import func

        db = SessionLocal()
        try:
            # Count keystroke samples per user
            ks_counts = dict(
                db.query(KeystrokeTemplate.user_id,
                         func.count(KeystrokeTemplate.id))
                  .group_by(KeystrokeTemplate.user_id)
                  .all()
            )
            # Count voice samples per user
            voice_counts = dict(
                db.query(VoiceTemplate.user_id,
                         func.count(VoiceTemplate.id))
                  .group_by(VoiceTemplate.user_id)
                  .all()
            )

            users = db.query(User).order_by(User.id).all()
            result = []
            for u in users:
                ks_n    = ks_counts.get(u.id, 0)
                voice_n = voice_counts.get(u.id, 0)
                if ks_n >= min_ks_samples and voice_n >= min_voice_samples:
                    result.append({
                        "username":      u.username,
                        "ks_samples":    ks_n,
                        "voice_samples": voice_n,
                    })
            return result

        finally:
            db.close()

    except Exception as e:
        print(f"  ❌ Could not query database: {e}")
        return []


def print_user_table(users: list):
    """Print a table of users and their enrollment counts."""
    print(f"\n  {'#':<4} {'Username':<36} {'KS Samples':>11} {'Voice Samples':>13}")
    print(f"  {'─'*4} {'─'*36} {'─'*11} {'─'*13}")
    for i, u in enumerate(users, 1):
        print(f"  {i:<4} {u['username']:<36} {u['ks_samples']:>11} {u['voice_samples']:>13}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE USER BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def run_for_user(username: str, keystroke_only: bool, voice_only: bool) -> dict:
    """
    Run both benchmarks for one user.
    Returns a dict of aggregate metrics for the summary table.
    """
    summary = {"username": username, "status": "OK"}

    # ── Keystroke ─────────────────────────────────────────────────────────────
    if not voice_only:
        print(f"\n{'▓'*70}")
        print(f"  KEYSTROKE DYNAMICS  —  {username}")
        print(f"{'▓'*70}")
        try:
            from keystroke_benchmark import main as ks_main
            ks_main(username)

            # Pull last-written CSV for summary
            safe_user = username.replace("@", "_at_").replace(".", "_")
            csv_path  = os.path.join(SCRIPT_DIR, "results",
                                     f"{safe_user}_keystroke_benchmark.csv")
            if os.path.exists(csv_path):
                with open(csv_path, newline="") as f:
                    rows = list(csv.DictReader(f))
                for row in rows:
                    model = row.get("model", "").replace(" ", "_").lower()
                    summary[f"ks_{model}_acc"] = row.get("accuracy", "")
                    summary[f"ks_{model}_eer"] = row.get("eer", "")
        except Exception as e:
            print(f"  ❌ Keystroke failed for {username}: {e}")
            summary["ks_error"] = str(e)

    # ── Voice ─────────────────────────────────────────────────────────────────
    if not keystroke_only:
        print(f"\n{'▓'*70}")
        print(f"  VOICE BIOMETRICS  —  {username}")
        print(f"{'▓'*70}")
        try:
            from voice_benchmark import main as v_main
            v_main(username)

            safe_user = username.replace("@", "_at_").replace(".", "_")
            csv_path  = os.path.join(SCRIPT_DIR, "results",
                                     f"{safe_user}_voice_benchmark.csv")
            if os.path.exists(csv_path):
                with open(csv_path, newline="") as f:
                    rows = list(csv.DictReader(f))
                for row in rows:
                    model = row.get("model", "").replace(" ", "_").lower()
                    summary[f"voice_{model}_acc"] = row.get("accuracy", "")
                    summary[f"voice_{model}_eer"] = row.get("eer", "")
        except Exception as e:
            print(f"  ❌ Voice failed for {username}: {e}")
            summary["voice_error"] = str(e)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  AGGREGATE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_aggregate_summary(all_summaries: list):
    """Print a cross-user summary table after all benchmarks complete."""
    if not all_summaries:
        return

    print(f"\n\n{'█'*70}")
    print(f"  AGGREGATE SUMMARY — ALL USERS")
    print(f"{'█'*70}")
    print(f"\n  {'Username':<32} {'Status':<8} {'KS EER (RF)':>12} {'Voice EER (LSTM)':>16}")
    print(f"  {'─'*32} {'─'*8} {'─'*12} {'─'*16}")

    for s in all_summaries:
        ks_eer    = s.get("ks_random_forest_acc", s.get("ks_error", "N/A"))
        voice_eer = s.get("voice_lstm_(voice)_acc", s.get("voice_error", "N/A"))
        status    = "❌ ERR" if ("ks_error" in s or "voice_error" in s) else "✅ OK"
        print(f"  {s['username']:<32} {status:<8} {ks_eer:>12} {voice_eer:>16}")

    print(f"{'█'*70}\n")


def save_aggregate_csv(all_summaries: list):
    """Save the full aggregate summary to a CSV."""
    if not all_summaries:
        return
    all_keys = set()
    for s in all_summaries:
        all_keys.update(s.keys())
    fieldnames = ["username", "status"] + sorted(
        k for k in all_keys if k not in ("username", "status")
    )
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "all_users_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in all_summaries:
            writer.writerow({k: s.get(k, "") for k in fieldnames})
    print(f"  📄 Aggregate summary saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  MULTIMODAL FUSION NOTE  (printed once at the end)
# ─────────────────────────────────────────────────────────────────────────────

def print_fusion_note():
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


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chapter 4 Benchmark Runner")
    parser.add_argument("username",          nargs="?", default=None,
                        help="Single username to benchmark")
    parser.add_argument("--all-users",       action="store_true",
                        help="Run benchmarks for ALL enrolled users in the DB")
    parser.add_argument("--keystroke-only",  action="store_true")
    parser.add_argument("--voice-only",      action="store_true")
    parser.add_argument("--min-samples",     type=int, default=3,
                        help="Min keystroke enrollment samples to include user (default: 3)")
    parser.add_argument("--min-voice",       type=int, default=1,
                        help="Min voice enrollment samples to include user (default: 1)")
    args = parser.parse_args()

    # ── Determine user list ────────────────────────────────────────────────────
    if args.all_users:
        print(f"\n{'█'*70}")
        print(f"  THESIS CHAPTER 4 — BENCHMARK ALL USERS")
        print(f"  Min KS samples   : {args.min_samples}")
        print(f"  Min voice samples: {args.min_voice}")
        print(f"  Started at       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'█'*70}")

        print(f"\n  Querying database for eligible users ...")
        users = get_all_users(
            min_ks_samples=args.min_samples,
            min_voice_samples=args.min_voice,
        )

        if not users:
            print("  ❌ No users found with sufficient enrollment data.")
            print(f"     Try lowering --min-samples (current: {args.min_samples})")
            sys.exit(1)

        print(f"\n  Found {len(users)} eligible user(s):")
        print_user_table(users)

        all_summaries = []
        failed        = []

        for i, u in enumerate(users, 1):
            uname = u["username"]
            print(f"\n{'█'*70}")
            print(f"  USER {i}/{len(users)}: {uname}")
            print(f"  KS samples: {u['ks_samples']}  |  "
                  f"Voice samples: {u['voice_samples']}")
            print(f"{'█'*70}")

            try:
                summary = run_for_user(uname, args.keystroke_only, args.voice_only)
                all_summaries.append(summary)
                if "ks_error" in summary or "voice_error" in summary:
                    failed.append(uname)
            except Exception as e:
                print(f"  ❌ Unexpected error for {uname}: {e}")
                failed.append(uname)
                all_summaries.append({"username": uname, "status": "FAILED",
                                       "error": str(e)})

        # ── Final summary ──────────────────────────────────────────────────────
        if not args.keystroke_only and not args.voice_only:
            print_fusion_note()

        print_aggregate_summary(all_summaries)
        save_aggregate_csv(all_summaries)

        print(f"\n{'█'*70}")
        print(f"  ALL-USERS BENCHMARK COMPLETE")
        print(f"  Finished at : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Processed   : {len(users)} user(s)")
        if failed:
            print(f"  Failed      : {len(failed)} — {', '.join(failed)}")
        print(f"  Results in  : {SCRIPT_DIR}")
        print(f"{'█'*70}\n")

    else:
        # ── Single user mode ───────────────────────────────────────────────────
        username = args.username or input("Enter username: ").strip()

        print(f"\n{'█'*70}")
        print(f"  THESIS CHAPTER 4 — BIOMETRIC AUTHENTICATION BENCHMARK")
        print(f"  User        : {username}")
        print(f"  Started at  : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'█'*70}")

        run_for_user(username, args.keystroke_only, args.voice_only)

        if not args.keystroke_only and not args.voice_only:
            print_fusion_note()

        print(f"\n{'█'*70}")
        print(f"  BENCHMARK COMPLETE  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Results CSV files saved to: {SCRIPT_DIR}/results")
        print(f"{'█'*70}\n")


if __name__ == "__main__":
    main()