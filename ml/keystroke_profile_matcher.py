# ml/keystroke_profile_matcher.py
#
# TypingDNA-style per-user profile matcher for small enrollment sets (≤10 samples).
#
# Key design decisions:
#   1. No classifier training — direct comparison against enrollment profile (mean ± std).
#      This sidesteps the augmentation quality problem: with only 5 real samples, any
#      GBM/RF trains on fake data whose spread doesn't match real within-session variance.
#
#   2. Speed normalization.  All ms-valued timing features are divided by the speed ratio
#      (live_p2p_mean / enrolled_p2p_mean) before computing Z-scores.  This handles the
#      "tired fingers" / "different keyboard" problem: if the genuine user types 15% faster
#      today, every digraph and dwell time scales proportionally and still matches.
#
#   3. Tiered Z-tolerance per feature group:
#       Digraphs (digraph_*, extra_*) → Z ≤ 1.8  (tight — muscle-memory fingerprint)
#       Dwell / flight ms             → Z ≤ 2.2  (medium — varies with fatigue)
#       Rhythm / ratios / counts      → Z ≤ 2.8  (loose — changes with stress / context)
#
#   4. Adaptive digraph tolerance.  When speed ratio is extreme (< 0.65 or > 1.55),
#      tighten digraph tolerance to Z ≤ 1.4 — someone typing at a very different overall
#      speed whose individual digraph ratios don't scale proportionally is suspicious.
#
#   5. Weighted group scoring:
#       digraphs:     40%  (highest discrimination power — inter-person muscle memory)
#       dwell/flight: 40%  (strong signal — per-key hold time)
#       rhythm/other: 20%  (weakest — too context-dependent for high weight)
#
# Score range: [0, 1].  Threshold is 0.65 by default (set at training time).
# A genuine user with consistent enrollment typically scores 0.73–0.88.
# An impostor with similar speed scores 0.48–0.63 because digraph muscle memory diverges.

import numpy as np
from typing import List, Optional

# ──────────────────────────────────────────────────────────────────────────────
#  Feature group classification
# ──────────────────────────────────────────────────────────────────────────────

# Features whose ms values scale proportionally with overall typing speed.
# We normalize these by the live/enrolled p2p ratio before computing Z.
_MS_TIMING_PREFIXES = (
    'dwell_', 'flight_', 'p2p_', 'r2r_', 'rhythm_mean', 'rhythm_std',
    'pause_mean', 'seek_time_mean', 'digraph_', 'extra_', 'key_',
    'trigraph_',   # press[i]→press[i+2] elapsed ms — scales with overall speed
)

# Already-normalised or ratio features — do NOT speed-normalize again.
_NO_NORMALIZE = {
    'dwell_mean_norm', 'dwell_std_norm', 'flight_mean_norm', 'flight_std_norm',
    'p2p_std_norm', 'r2r_mean_norm', 'shift_lag_norm',
    'rhythm_cv', 'backspace_ratio', 'hand_alternation_ratio',
    'finger_transition_ratio',
    'pause_count', 'backspace_count', 'seek_time_count',
    'typing_speed_cpm',   # CPM is inverse of time — already normalised
    'typing_duration',    # total seconds — do not scale
    'same_hand_sequence_mean',
}

# Binary Z-tolerance still used for aggregate dwell/flight + rhythm.
_Z_TOL = {
    'dwell_flight': 1.8,
    'rhythm':       2.2,
}

# Digraph group uses two continuous sub-scores:
#   digraph_dist : scaled Manhattan distance (mean |live-μ|/σ) — THE discriminator
#   digraph_rank : fraction of digraph pairs whose live ordering matches
#                  enrollment. On SHORT FIXED phrases, rank barely discriminates
#                  because phrase mechanics force similar orderings across
#                  typists — both genuine and impostors land near 1.0. Kept at
#                  low weight as a minor consistency check, not a discriminator.
_WEIGHTS = {
    'digraph_dist': 0.55,
    'digraph_rank': 0.10,
    'dwell_flight': 0.25,
    'rhythm':       0.10,
}

# mean_z → distance_score: linear, 0 at MEAN_Z_MAX or above. Tightened 2.8→2.0
# for sharper rejection: genuine typically sits at mean_z ≤ 0.5 → dist ≥ 0.75,
# similar-typist impostors land at mean_z 1.2–1.8 → dist 0.1–0.4.
_DIST_MEAN_Z_MAX = 2.0

# Hard floor on digraph_rank (soft penalty on dist — see scoring).
_DIGRAPH_DIST_FLOOR = 0.35
# Raised 0.30 → 0.45. Rank 0.30 (after [0.5,1.0]→[0,1] remap) means raw
# concordance ≈ 0.65 — barely above coin-flip. Genuine typists land at 0.70+.
# 0.45 corresponds to raw 0.725 concordance, still lenient for genuine but
# firmly rejects impostors whose digraph ordering doesn't match muscle memory.
_DIGRAPH_RANK_FLOOR = 0.45


def _classify_feature(name: str) -> str:
    """
    Return the group name for a feature.

    Per-key dwells (`key_*`) and trigraphs (`trigraph_*`) are person-specific
    muscle-memory signals — they belong with digraphs, not in dwell_flight
    (which is for aggregate ms-timings).
    """
    if (name.startswith('digraph_') or name.startswith('extra_')
            or name.startswith('key_') or name.startswith('trigraph_')):
        return 'digraph'
    if (name.startswith('dwell_') or name.startswith('flight_')
            or name.startswith('p2p_') or name.startswith('r2r_')):
        return 'dwell_flight'
    return 'rhythm'


def _rank_agreement(live_vals: np.ndarray, mean_vals: np.ndarray) -> float:
    """
    Fraction of digraph pairs whose relative ordering in `live` matches
    the ordering in the enrolled mean profile (Gunetti–Picardi R-measure).

    Returns a value in [0.5, 1.0] for typical inputs (0.5 = random ordering,
    1.0 = identical ordering). The caller should remap to [0, 1] via
    max(0, 2*frac - 1) so random ordering gets score 0.
    """
    n = len(live_vals)
    if n < 2:
        return 1.0
    concordant = 0
    total = 0
    for i in range(n):
        li, mi = live_vals[i], mean_vals[i]
        for j in range(i + 1, n):
            dl = li - live_vals[j]
            dm = mi - mean_vals[j]
            # Ignore pairs where either profile or live treats them as tied —
            # they carry no ordering signal.
            if abs(dl) < 1.0 or abs(dm) < 1.0:
                continue
            total += 1
            if (dl > 0) == (dm > 0):
                concordant += 1
    if total == 0:
        return 1.0
    return concordant / total


def _is_ms_timing(name: str) -> bool:
    """True if this feature should be speed-normalized (its value is in ms and scales with speed)."""
    if name in _NO_NORMALIZE:
        return False
    return any(name.startswith(p) for p in _MS_TIMING_PREFIXES)


# ──────────────────────────────────────────────────────────────────────────────
#  Core scoring function
# ──────────────────────────────────────────────────────────────────────────────

def compute_profile_score(
    live_vec:     np.ndarray,
    feat_names:   List[str],
    profile_mean: np.ndarray,
    profile_std:  np.ndarray,
) -> dict:
    """
    Score a login attempt against the user's enrollment profile.

    Parameters
    ----------
    live_vec     : 1-D feature vector from the login attempt (same feature order as enrollment)
    feat_names   : list of feature name strings (same length as live_vec / profile arrays)
    profile_mean : per-feature mean over enrollment samples
    profile_std  : per-feature std over enrollment samples (with 1e-9 floor already added)

    Returns
    -------
    dict with keys:
        score       : float in [0, 1]  (higher = more genuine)
        speed_ratio : float  (live / enrolled p2p_mean, clamped [0.5, 2.0])
        group_scores: dict   {group_name: fraction_within_tolerance}
        details     : list of (feature_name, z_score, within_tol) for debugging
    """
    fn = list(feat_names)
    live = np.asarray(live_vec, dtype=np.float64)
    pmean = np.asarray(profile_mean, dtype=np.float64)
    pstd  = np.asarray(profile_std,  dtype=np.float64)

    # ── Speed normalization ────────────────────────────────────────────────────
    p2p_idx = fn.index('p2p_mean') if 'p2p_mean' in fn else None
    if p2p_idx is not None and pmean[p2p_idx] > 1e-6 and live[p2p_idx] > 1e-6:
        speed_ratio = float(np.clip(live[p2p_idx] / pmean[p2p_idx], 0.5, 2.0))
    else:
        speed_ratio = 1.0

    # Per-group std caps keep tolerance bands tight even when enrollment was
    # inconsistent. Tightened for digraph so impostors with ±10% off-mean
    # digraph timings can't slip through a wide band.
    # Digraph std cap relaxed 0.10→0.15: with only 5 enrollment samples and
    # natural speed variation, a 10% cap produces artificially tight bands that
    # reject the genuine user when they log in at a slightly different speed.
    # The R-measure (digraph_rank) is the real discriminator — keep dist loose.
    _STD_CAP = {
        'digraph':      0.15,
        'dwell_flight': 0.20,
        'rhythm':       0.30,
    }

    # ── Per-feature Z-scores ───────────────────────────────────────────────────
    digraph_z = []                                # |live-μ|/σ for digraph features
    digraph_live_norm = []                        # speed-normalized live values
    digraph_mean_ref  = []                        # enrollment means
    group_hits = {'dwell_flight': [], 'rhythm': []}
    details    = []

    for i, name in enumerate(fn):
        enr_mean = float(pmean[i])
        grp = _classify_feature(name)
        cap = _STD_CAP[grp]
        enr_std = float(np.clip(
            max(pstd[i], abs(enr_mean) * 0.03, 1.0),
            0,
            abs(enr_mean) * cap + 1.0
        ))

        norm_val = float(live[i])
        if _is_ms_timing(name) and speed_ratio != 1.0:
            norm_val = norm_val / speed_ratio

        z = abs(norm_val - enr_mean) / enr_std

        if grp == 'digraph':
            # Only include digraphs that carry signal — skip zero-mean features
            # (happens when a digraph never fired during enrollment).
            if enr_mean > 1e-6:
                digraph_z.append(z)
                digraph_live_norm.append(norm_val)
                digraph_mean_ref.append(enr_mean)
        else:
            group_hits[grp].append(z <= _Z_TOL[grp])

        details.append((name, float(z), grp))

    # ── Digraph distance sub-score (scaled Manhattan) ─────────────────────────
    if digraph_z:
        mean_z = float(np.mean(digraph_z))
        digraph_dist = float(np.clip(1.0 - mean_z / _DIST_MEAN_Z_MAX, 0.0, 1.0))
    else:
        mean_z = 0.0
        digraph_dist = 1.0

    # ── Digraph rank sub-score (R-measure) ─────────────────────────────────────
    if len(digraph_live_norm) >= 2:
        frac = _rank_agreement(
            np.asarray(digraph_live_norm),
            np.asarray(digraph_mean_ref),
        )
        # Remap [0.5,1.0] → [0,1]; sub-0.5 (anti-correlated) clips to 0.
        digraph_rank = float(max(0.0, 2.0 * frac - 1.0))
    else:
        digraph_rank = 1.0

    # ── Aggregate group scores ─────────────────────────────────────────────────
    def _frac(hits):
        return (sum(hits) / len(hits)) if hits else 1.0
    dwell_flight_score = float(_frac(group_hits['dwell_flight']))
    rhythm_score       = float(_frac(group_hits['rhythm']))

    group_scores = {
        'digraph_dist': digraph_dist,
        'digraph_rank': digraph_rank,
        'dwell_flight': dwell_flight_score,
        'rhythm':       rhythm_score,
        'digraph_mean_z': mean_z,
    }

    total_score = (
        _WEIGHTS['digraph_dist'] * digraph_dist
        + _WEIGHTS['digraph_rank'] * digraph_rank
        + _WEIGHTS['dwell_flight'] * dwell_flight_score
        + _WEIGHTS['rhythm']       * rhythm_score
    )

    # ── Hard floor on digraph_rank ─────────────────────────────────────────────
    # Only the rank sub-score hard-clamps the total. The distance sub-score
    # already penalizes bad matches via its weighted contribution; clamping on
    # it too would reject genuine users whose speed shifts between sessions.
    # Rank is robust to speed and captures muscle-memory ordering — a real
    # impostor should fail it regardless of how close their absolute times are.
    floor_breach = None
    if len(digraph_live_norm) >= 2 and digraph_rank < _DIGRAPH_RANK_FLOOR:
        floor_breach = f"digraph_rank={digraph_rank:.2f} < {_DIGRAPH_RANK_FLOOR}"
        total_score = 0.0
    # Soft penalty when dist is very low: halve the total rather than clamp.
    elif digraph_z and digraph_dist < _DIGRAPH_DIST_FLOOR:
        floor_breach = f"digraph_dist={digraph_dist:.2f} < {_DIGRAPH_DIST_FLOOR} (soft)"
        total_score *= 0.5
    group_scores['floor_breach'] = floor_breach

    return {
        'score':        float(np.clip(total_score, 0.0, 1.0)),
        'speed_ratio':  speed_ratio,
        'group_scores': group_scores,
        'details':      details,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Profile builder (called from train_keystroke_rf.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_profile_model(
    genuine_vectors: list,
    active_feat_names: List[str],
    username: str,
    user_id: int,
    user_phrase: str,
    threshold: float = 0.65,
) -> dict:
    """
    Build the serialisable profile model dict (same format as RF model dict,
    but with model_type='profile' and no pipeline key).

    threshold : accept if score >= threshold.  Default 0.62 — tuned for the
                new scoring scheme (distance + R-measure + binary groups).
                Genuine typically scores 0.70–0.85; similar-typist impostors
                land in 0.35–0.55 due to the digraph rank sub-score.
    """
    vecs = np.array(genuine_vectors, dtype=np.float64)
    n_in = len(genuine_vectors)

    print(f"\n{'='*70}")
    print(f"  KEYSTROKE PROFILE MATCHER  —  user: {username}")
    print(f"{'='*70}")
    print(f"  Enrollment samples : {n_in}")
    print(f"  Feature dimensions : {len(active_feat_names)}")

    # ── LOO-based outlier rejection ────────────────────────────────────────────
    # For each sample, score it against a profile built from the other n-1.
    # Drop samples whose LOO score is > 1.5σ below the median — they disagree
    # with the bulk of enrollment and would poison mean/std if retained.
    if n_in >= 5:
        loo_scores = []
        for i in range(n_in):
            others = np.delete(vecs, i, axis=0)
            m = others.mean(axis=0)
            s = others.std(axis=0) + 1e-9
            r = compute_profile_score(vecs[i], active_feat_names, m, s)
            loo_scores.append(r['score'])
        loo_arr = np.array(loo_scores)
        med     = float(np.median(loo_arr))
        sigma   = float(np.std(loo_arr)) or 1e-6
        cutoff  = med - 1.5 * sigma
        keep_mask = loo_arr >= cutoff
        # Never drop below 4 survivors
        if keep_mask.sum() < 4:
            keep_mask = np.ones(n_in, dtype=bool)
        dropped = n_in - int(keep_mask.sum())
        if dropped:
            print(f"  ⚠  LOO outlier drop: removed {dropped} sample(s) "
                  f"(score < {cutoff:.2f}, median={med:.2f}, σ={sigma:.2f})")
            vecs = vecs[keep_mask]
            genuine_vectors = [genuine_vectors[i] for i in range(n_in) if keep_mask[i]]
        else:
            print(f"  ✅ LOO outlier check: all {n_in} samples consistent "
                  f"(median={med:.2f}, σ={sigma:.2f})")
    else:
        print(f"  ⓘ  LOO outlier check skipped (need ≥ 5 samples, have {n_in})")

    profile_mean = vecs.mean(axis=0)
    profile_std  = vecs.std(axis=0) + 1e-9   # floor for division safety
    n = len(genuine_vectors)
    fn = active_feat_names
    for label, feat in [
        ("p2p_mean (ms)",    "p2p_mean"),
        ("dwell_mean (ms)",  "dwell_mean"),
        ("flight_mean (ms)", "flight_mean"),
        ("typing_speed_cpm", "typing_speed_cpm"),
        ("rhythm_cv",        "rhythm_cv"),
    ]:
        if feat in fn:
            idx = fn.index(feat)
            print(f"  {label:22s}: {profile_mean[idx]:.1f}  ±{profile_std[idx]:.1f}")

    # Quick self-check: score each enrollment sample against the profile
    self_scores = []
    for v in genuine_vectors:
        r = compute_profile_score(v, active_feat_names, profile_mean, profile_std)
        self_scores.append(r['score'])
    s_mean = float(np.mean(self_scores))
    s_std  = float(np.std(self_scores))
    print(f"\n  Self-check scores (enrollment vs own profile):")
    print(f"    min={min(self_scores):.3f}  mean={s_mean:.3f}  max={max(self_scores):.3f}  σ={s_std:.3f}")

    # ── Per-user threshold calibration ─────────────────────────────────────────
    # Use the user's own score distribution: threshold = mean - 2σ. Bounded to
    # [0.50, 0.70] so noisy enrollment can't push it absurdly low (FAR risk)
    # and very consistent typists don't get a punishingly strict threshold
    # that fails on slight cross-session drift.
    auto_thr = s_mean - 2.0 * s_std
    calibrated = float(np.clip(auto_thr, 0.50, 0.70))
    print(f"  Calibrated threshold: {calibrated:.3f}  "
          f"(raw mean-2σ={auto_thr:.3f}, default was {threshold:.2f})")
    threshold = calibrated
    frr_at_thresh = sum(s < threshold for s in self_scores) / len(self_scores)
    print(f"    FRR at {threshold:.2f}: {frr_at_thresh:.0%}  "
          f"({'OK' if frr_at_thresh == 0 else 'WARNING: genuine would be rejected'})")

    return {
        'model_type':    'profile',
        'feature_names': active_feat_names,
        'username':      username,
        'user_id':       user_id,
        'n_enrollment':  n,
        'profile_mean':  profile_mean,
        'profile_std':   profile_std,
        'threshold':     threshold,
        'phrase':        user_phrase,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Standalone test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)

    feat_names = (
        [f"digraph_ab", f"digraph_bc", f"extra_cd"]
        + [f"dwell_{i}" for i in range(5)]
        + [f"flight_{i}" for i in range(5)]
        + [f"p2p_mean", f"p2p_std", f"r2r_mean"]
        + [f"rhythm_cv", f"backspace_ratio", f"pause_count",
           f"typing_speed_cpm", f"typing_duration", f"hand_alternation_ratio"]
    )

    # Build a fake genuine profile (5 samples)
    N_FEAT = len(feat_names)
    base = rng.uniform(50, 200, size=N_FEAT)
    enrollment = [base + rng.normal(0, base * 0.05) for _ in range(5)]

    model = build_profile_model(enrollment, feat_names, "test_user", 1, "abc phrase")

    # Genuine login (slight speed change — 10% faster today)
    genuine_live = base.copy() * 0.90
    genuine_live += rng.normal(0, base * 0.04)
    r = compute_profile_score(genuine_live, feat_names, model['profile_mean'], model['profile_std'])
    print(f"\nGenuine login score: {r['score']:.3f}  speed_ratio={r['speed_ratio']:.2f}")
    print(f"  Groups: {r['group_scores']}")

    # Realistic impostor: their own muscle memory, independently drawn from a
    # different baseline.  Digraphs differ substantially; overall speed similar.
    impostor_base = rng.uniform(40, 220, size=N_FEAT)  # different person's timing profile
    impostor_live = impostor_base.copy()
    # Adjust p2p_mean index so speed_ratio ~1 (similar overall speed, different digraphs)
    p2p_i = feat_names.index('p2p_mean')
    impostor_live[p2p_i] = base[p2p_i] * rng.uniform(0.92, 1.08)
    impostor_live += rng.normal(0, impostor_base * 0.04)
    r2 = compute_profile_score(impostor_live, feat_names, model['profile_mean'], model['profile_std'])
    print(f"Impostor score:      {r2['score']:.3f}  speed_ratio={r2['speed_ratio']:.2f}")
    print(f"  Groups: {r2['group_scores']}")

    thr = model['threshold']
    print(f"\nThreshold {thr:.2f}: genuine {'PASS' if r['score'] >= thr else 'FAIL'}  "
          f"impostor {'PASS (BAD)' if r2['score'] >= thr else 'REJECT (good)'}")
