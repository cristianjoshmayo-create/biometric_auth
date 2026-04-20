# ml/keystroke_profile_matcher.py
#
# Keystroke matcher implementing TypingDNA's public approach literally
# (Gunetti & Picardi, 2005). Scoring is ratio-based A-measure + rank-
# displacement R-measure over per-digraph DD timings and per-key dwell
# times only. Aggregate rhythm/hand/speed features may be present in the
# feature vector but are ignored by the score.
#
#   A = fraction of (live_i, ref_i) positions where max/min <= t (t=1.25)
#   R = 1 - sum(|rank_live - rank_ref|) / floor(n^2 / 2)
#   score = (A + R) / 2
#
# Same-Text verification compares the live sample against each stored
# enrollment vector individually, then combines the top matches.
#
# Positions with either live or ref below ε (5ms) are excluded — they
# represent digraphs that never fired and carry no signal.

import numpy as np
from typing import List, Optional


# Fixed GP tolerance — TypingDNA's published ratio bound.
_T_RATIO = 1.25
# Positions below this many ms are treated as "missing" (digraph didn't fire).
_EPS_MS = 5.0
# Minimum number of valid positions for A+R to be meaningful.
_MIN_VALID = 4


def _is_scoring_feature(name: str) -> bool:
    """TypingDNA scoring uses only digraph DD timings and per-key dwells."""
    return (name.startswith('digraph_')
            or name.startswith('extra_')
            or name.startswith('key_')
            or name.startswith('trigraph_'))


def _classify_feature(name: str) -> str:
    """Group label used only for diagnostic output."""
    if (name.startswith('digraph_') or name.startswith('extra_')
            or name.startswith('key_') or name.startswith('trigraph_')):
        return 'digraph'
    if (name.startswith('dwell_') or name.startswith('flight_')
            or name.startswith('p2p_') or name.startswith('r2r_')):
        return 'dwell_flight'
    return 'rhythm'


def _stable_ranks(arr: np.ndarray) -> np.ndarray:
    """1-indexed ranks; ties broken stably by index order."""
    order = np.argsort(arr, kind='stable')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def _gp_score(live: np.ndarray, ref: np.ndarray) -> tuple:
    """
    Return (A, R, n_valid) on aligned (live, ref) timing vectors.
    Filters positions where either side is below ε.
    """
    valid = (live >= _EPS_MS) & (ref >= _EPS_MS)
    n = int(valid.sum())
    if n < _MIN_VALID:
        return 0.0, 0.0, n

    vl = live[valid]
    vr = ref[valid]

    ratios = np.maximum(vl, vr) / np.minimum(vl, vr)
    A = float(np.mean(ratios <= _T_RATIO))

    rl = _stable_ranks(vl)
    rr = _stable_ranks(vr)
    max_disp = (n * n) // 2
    if max_disp <= 0:
        R = 1.0
    else:
        disp = float(np.sum(np.abs(rl - rr)))
        R = float(1.0 - disp / max_disp)
    return A, R, n


# ──────────────────────────────────────────────────────────────────────────────
#  Core scoring
# ──────────────────────────────────────────────────────────────────────────────

def compute_profile_score(
    live_vec:     np.ndarray,
    feat_names:   List[str],
    profile_mean: np.ndarray,
    profile_std:  np.ndarray,          # accepted for signature compatibility; unused
    model_stage:  Optional[str] = None,
) -> dict:
    """Gunetti-Picardi A+R score against a single reference vector."""
    fn = list(feat_names)
    live_all = np.asarray(live_vec,     dtype=np.float64)
    ref_all  = np.asarray(profile_mean, dtype=np.float64)

    score_idx = [i for i, n in enumerate(fn) if _is_scoring_feature(n)]

    # Purely informational — for logs. Not used in scoring.
    if 'p2p_mean' in fn:
        p = fn.index('p2p_mean')
        if ref_all[p] > 1e-6 and live_all[p] > 1e-6:
            speed_ratio = float(np.clip(live_all[p] / ref_all[p], 0.25, 4.0))
        else:
            speed_ratio = 1.0
    else:
        speed_ratio = 1.0

    if not score_idx:
        return {
            'score':        0.0,
            'speed_ratio':  speed_ratio,
            'group_scores': {
                'digraph_dist': 0.0, 'digraph_rank': 0.0,
                'dwell_flight': 1.0, 'rhythm': 1.0,
                'digraph_mean_z': 0.0,
                'floor_breach': 'no scoring features in feat_names',
                'stage': (model_stage or 'mid').lower(),
                'n_valid': 0,
            },
            'details': [(n, 0.0, _classify_feature(n)) for n in fn],
        }

    live = live_all[score_idx]
    ref  = ref_all[score_idx]
    A, R, n_valid = _gp_score(live, ref)

    score = (A + R) / 2.0

    # Per-feature diagnostic: ratio-1 as a stand-in for "distance" so the
    # auth.py top-N diag dump still produces something meaningful.
    details = []
    for i, name in enumerate(fn):
        grp = _classify_feature(name)
        if grp == 'digraph':
            l = float(live_all[i]); r = float(ref_all[i])
            if l >= _EPS_MS and r >= _EPS_MS:
                d = max(l, r) / min(l, r) - 1.0
            else:
                d = 0.0
            details.append((name, float(d), grp))
        else:
            details.append((name, 0.0, grp))

    breach = None
    if n_valid < _MIN_VALID:
        breach = f'only {n_valid} valid positions (< {_MIN_VALID})'

    return {
        'score':        float(np.clip(score, 0.0, 1.0)),
        'speed_ratio':  speed_ratio,
        'group_scores': {
            'digraph_dist':   float(A),
            'digraph_rank':   float(R),
            'dwell_flight':   1.0,
            'rhythm':         1.0,
            'digraph_mean_z': 0.0,
            'floor_breach':   breach,
            'stage':          (model_stage or 'mid').lower(),
            'n_valid':        n_valid,
        },
        'details':      details,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Same-Text set matching
# ──────────────────────────────────────────────────────────────────────────────

def compute_set_match_score(
    live_vec:        np.ndarray,
    feat_names:      List[str],
    genuine_vectors: list,
    profile_std:     np.ndarray,        # unused; kept for signature compat
    model_stage:     Optional[str] = None,
    top_k:           Optional[int] = None,
) -> dict:
    """Best-of-N over stored enrollment vectors, weighted toward best match."""
    per_sample = []
    for sample_vec in genuine_vectors:
        r = compute_profile_score(
            live_vec, feat_names,
            np.asarray(sample_vec, dtype=np.float64),
            profile_std,
            model_stage=model_stage,
        )
        per_sample.append(r)

    scores = np.array([r['score'] for r in per_sample])
    n = len(scores)
    if top_k is None:
        k = min(3, max(2, n // 2))
    else:
        k = min(int(top_k), n)
    best_idx = np.argsort(scores)[-k:][::-1]
    top_scores = scores[best_idx]

    if k == 1:
        weights = np.array([1.0])
    elif k == 2:
        weights = np.array([0.65, 0.35])
    elif k == 3:
        weights = np.array([0.55, 0.30, 0.15])
    else:
        weights = 0.55 * (0.55 ** np.arange(k))
        weights = weights / weights.sum()
    final = float(np.sum(top_scores * weights))

    if k >= 2:
        spread = float(top_scores.std())
        confidence = float(np.clip(1.0 - spread * 2.0, 0.0, 1.0))
    else:
        confidence = 1.0

    best = per_sample[int(best_idx[0])]
    best['score']             = final
    best['confidence']        = confidence
    best['per_sample_scores'] = scores.tolist()
    best['top_k_used']        = k
    return best


# ──────────────────────────────────────────────────────────────────────────────
#  Profile model builder (called from train_keystroke_rf.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_profile_model(
    genuine_vectors:   list,
    active_feat_names: List[str],
    username:          str,
    user_id:           int,
    user_phrase:       str,
    model_stage:       Optional[str] = None,
    threshold:         float = 0.65,
) -> dict:
    """
    Build the serialisable profile model dict.
    Fixed threshold (default 0.65) — no per-user calibration. Inconsistent
    typists get downweighted in fusion via `ks_reliability`, not by relaxing
    the keystroke threshold itself.
    """
    vecs = np.array(genuine_vectors, dtype=np.float64)
    n_in = len(genuine_vectors)

    print(f"\n{'='*70}")
    print(f"  KEYSTROKE PROFILE MATCHER (GP A+R)  —  user: {username}")
    print(f"{'='*70}")
    print(f"  Enrollment samples : {n_in}")
    print(f"  Stage              : {(model_stage or 'mid').upper()}")
    print(f"  Feature dimensions : {len(active_feat_names)}")
    print(f"  Scoring features   : "
          f"{sum(1 for n in active_feat_names if _is_scoring_feature(n))} "
          f"(digraph/key/trigraph)")

    # LOO outlier rejection: drop samples whose GP self-match is > 1.5σ below
    # the median. Prevents one noisy sample from polluting the reference set.
    if n_in >= 5:
        loo_scores = []
        for i in range(n_in):
            others = [genuine_vectors[j] for j in range(n_in) if j != i]
            r = compute_set_match_score(
                vecs[i], active_feat_names, others,
                np.zeros(len(active_feat_names)), model_stage=model_stage,
            )
            loo_scores.append(r['score'])
        loo_arr = np.array(loo_scores)
        med   = float(np.median(loo_arr))
        sigma = float(np.std(loo_arr)) or 1e-6
        cutoff = med - 1.5 * sigma
        keep_mask = loo_arr >= cutoff
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
    profile_std  = vecs.std(axis=0) + 1e-9   # retained for RF-path / diagnostic use
    n = len(genuine_vectors)
    fn = active_feat_names
    for label, feat in [
        ("p2p_mean (ms)",    "p2p_mean"),
        ("dwell_mean (ms)",  "dwell_mean"),
        ("flight_mean (ms)", "flight_mean"),
        ("typing_speed_cpm", "typing_speed_cpm"),
    ]:
        if feat in fn:
            idx = fn.index(feat)
            print(f"  {label:22s}: {profile_mean[idx]:.1f}  ±{profile_std[idx]:.1f}")

    # Self-check via leave-one-out set-matching: realistic preview of
    # enrolled-user scores against the final reference set at auth time.
    self_scores = []
    for i, v in enumerate(genuine_vectors):
        others = [genuine_vectors[j] for j in range(len(genuine_vectors)) if j != i]
        if len(others) == 0:
            r = compute_profile_score(v, active_feat_names, profile_mean, profile_std,
                                      model_stage=model_stage)
        else:
            r = compute_set_match_score(v, active_feat_names, others, profile_std,
                                        model_stage=model_stage)
        self_scores.append(r['score'])
    s_mean = float(np.mean(self_scores))
    s_std  = float(np.std(self_scores))
    print(f"\n  Self-check scores (LOO set-match):")
    print(f"    min={min(self_scores):.3f}  mean={s_mean:.3f}  "
          f"max={max(self_scores):.3f}  σ={s_std:.3f}")

    # Fixed threshold, no calibration.
    print(f"  Fixed threshold    : {threshold:.3f}  (no per-user calibration)")
    frr_at_thresh = sum(s < threshold for s in self_scores) / len(self_scores)
    print(f"    FRR at {threshold:.2f}: {frr_at_thresh:.0%}  "
          f"({'OK' if frr_at_thresh == 0 else 'some enrollments would miss — expect fusion fallback'})")

    # Reliability feeds adaptive fusion weighting in auth.py. Floor at 0.15 so
    # keystroke retains a minimal contribution even for very inconsistent typists.
    ks_reliability = float(np.clip(1.0 - frr_at_thresh, 0.15, 1.0))
    print(f"  Keystroke reliability: {ks_reliability:.2f}  (for fusion weighting)")

    # Inconsistency flag: high LOO σ OR low LOO mean means this user's typing
    # varies enough that the global threshold will over-reject them. Auth-side
    # threshold calibration uses these to relax the floor for these users only.
    is_inconsistent = bool(s_std > 0.08 or s_mean < 0.70)
    print(f"  Consistency        : {'INCONSISTENT' if is_inconsistent else 'stable'} "
          f"(mean={s_mean:.2f}, σ={s_std:.2f})")

    return {
        'model_type':       'profile',
        'model_stage':      (model_stage or 'mid'),
        'feature_names':    active_feat_names,
        'username':         username,
        'user_id':          user_id,
        'n_enrollment':     n,
        'profile_mean':     profile_mean,
        'profile_std':      profile_std,
        'genuine_vectors':  [np.asarray(v, dtype=np.float64).tolist() for v in genuine_vectors],
        'threshold':        threshold,
        'phrase':           user_phrase,
        'ks_reliability':   ks_reliability,
        'self_score_mean':  s_mean,
        'self_score_std':   s_std,
        'is_inconsistent':  is_inconsistent,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Standalone self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Build a mixed feat_names list: scoring features (digraph/key) + aggregates.
    digraphs = [f"digraph_{c1}{c2}" for c1, c2 in
                [('t','h'),('h','e'),('e',' '),(' ','q'),('q','u'),('u','i'),
                 ('i','c'),('c','k'),('k',' '),(' ','b'),('b','r'),('r','o')]]
    keys     = [f"key_{c}" for c in "thequickbrownfox"]
    aggregates = ['dwell_mean', 'flight_mean', 'p2p_mean', 'rhythm_cv',
                  'typing_speed_cpm', 'hand_alternation_ratio']
    feat_names = digraphs + keys + aggregates
    N = len(feat_names)

    # Genuine user: each digraph+key timing has a per-person baseline (uniform)
    # plus ~5% within-session noise.
    base = np.concatenate([
        rng.uniform(80, 220, size=len(digraphs)),     # DD ms
        rng.uniform(60, 140, size=len(keys)),         # dwell ms
        np.array([95.0, 85.0, 180.0, 0.35, 240.0, 0.55]),  # aggregates
    ])
    enrollment = [base + rng.normal(0, np.abs(base) * 0.05) for _ in range(5)]

    model = build_profile_model(enrollment, feat_names, "test_user", 1, "the quick brown")

    # Genuine login: same person, different day (10% slower overall, 4% noise).
    genuine = base.copy() * 1.10
    genuine += rng.normal(0, np.abs(base) * 0.04)
    g = compute_set_match_score(genuine, feat_names, enrollment,
                                model['profile_std'], model_stage='mid')
    print(f"\nGenuine login:   score={g['score']:.3f}  "
          f"A={g['group_scores']['digraph_dist']:.2f} "
          f"R={g['group_scores']['digraph_rank']:.2f} "
          f"n_valid={g['group_scores']['n_valid']} "
          f"conf={g.get('confidence', 1.0):.2f}")

    # Impostor: different person, similar overall speed (so naive speed
    # gates can't tell them apart) but independent muscle memory.
    impostor_base = np.concatenate([
        rng.uniform(70, 230, size=len(digraphs)),
        rng.uniform(55, 150, size=len(keys)),
        np.array([90.0, 90.0, 180.0, 0.40, 240.0, 0.58]),  # matched aggregates
    ])
    impostor = impostor_base + rng.normal(0, np.abs(impostor_base) * 0.04)
    i = compute_set_match_score(impostor, feat_names, enrollment,
                                model['profile_std'], model_stage='mid')
    print(f"Impostor login:  score={i['score']:.3f}  "
          f"A={i['group_scores']['digraph_dist']:.2f} "
          f"R={i['group_scores']['digraph_rank']:.2f} "
          f"n_valid={i['group_scores']['n_valid']} "
          f"conf={i.get('confidence', 1.0):.2f}")

    thr = model['threshold']
    print(f"\nThreshold {thr:.2f}: "
          f"genuine  {'PASS ✓' if g['score'] >= thr else 'FAIL ✗'}   "
          f"impostor {'PASS (BAD) ✗' if i['score'] >= thr else 'REJECT ✓'}")
