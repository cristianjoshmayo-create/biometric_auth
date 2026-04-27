"""
Trim impostor voice embedding(s) from asther's ECAPA profile.

Attempt 67 (2026-04-27 08:00) appended an impostor embedding to
astherlilies@gmail.com_voice_ecapa.pkl because voice's own match decision
returned True (sim 0.76 was above threshold). Attempt 69's voice was below
threshold so it was not saved.

Strategy:
  1. Print each stored embedding's cosine similarity to the mean of the
     OTHER embeddings. Genuine samples cluster (high sim ~0.7+); impostor
     embeddings stand out as outliers (much lower sim).
  2. By default, trim only the most-recently-appended embedding (the
     impostor from attempt 67). Use --trim N to remove the last N.
  3. Recompute mean_embedding and n_enrollment, save back.

Usage:
    python cleanup_asther_voice_polluted.py             # dry-run, shows similarities
    python cleanup_asther_voice_polluted.py --apply     # trims last 1
    python cleanup_asther_voice_polluted.py --trim 2 --apply
"""

import sys, os, pickle, argparse
import numpy as np

PROFILE_PATH = "ml/models/astherlilies_at_gmail_com_voice_ecapa.pkl"


def _l2(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-10)


def main(apply: bool, trim: int):
    if not os.path.exists(PROFILE_PATH):
        print(f"Profile not found: {PROFILE_PATH}")
        return

    with open(PROFILE_PATH, "rb") as f:
        prof = pickle.load(f)

    embs = [np.asarray(e, dtype=np.float32) for e in prof.get("embeddings", [])]
    n = len(embs)
    print(f"Profile: {prof['username']}  n_enrollment={prof.get('n_enrollment')}  stored={n}")

    if n < 2:
        print("Not enough embeddings to compute leave-one-out similarity.")
        return

    print("\nLeave-one-out similarity to mean of the other embeddings")
    print("(low value = outlier; impostors stand out at the bottom):\n")
    sims = []
    for i, e in enumerate(embs):
        others = np.array([embs[j] for j in range(n) if j != i])
        mean_others = _l2(others.mean(axis=0))
        s = float(np.dot(_l2(e), mean_others))
        sims.append((i, s))

    for i, s in sims:
        marker = "  <-- LAST (newest)" if i == n - 1 else ""
        print(f"  idx {i:2d}  sim_to_others_mean = {s:+.4f}{marker}")

    print(f"\nWill trim the last {trim} embedding(s): "
          f"indices {list(range(n - trim, n))}")

    if not apply:
        print("\n[dry-run] no changes made. Re-run with --apply to write back.")
        return

    kept = embs[: n - trim]
    if not kept:
        print("Refusing to delete every embedding.")
        return

    matrix = np.array([e.tolist() for e in kept], dtype=np.float32)
    new_mean = matrix.mean(axis=0)
    new_mean = new_mean / (np.linalg.norm(new_mean) + 1e-10)

    prof["embeddings"]     = [e.tolist() for e in kept]
    prof["mean_embedding"] = new_mean.tolist()
    prof["n_enrollment"]   = len(kept)

    with open(PROFILE_PATH, "wb") as f:
        pickle.dump(prof, f)

    print(f"\n[ok] wrote profile back: n_enrollment={len(kept)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--trim",  type=int, default=1,
                    help="number of most-recent embeddings to drop (default 1)")
    args = ap.parse_args()
    main(apply=args.apply, trim=args.trim)
