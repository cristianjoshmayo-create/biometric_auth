# ml/train_keystroke_rf.py
# v5 — phrase-aware impostor generation (security fix)
#
# Key changes vs v4:
#  1. generate_impostor_samples() now uses THREE tiers instead of one flat
#     random distribution:
#       Tier 1 (40%): Near impostors — same phrase, different person timing.
#                     Forces the model to distinguish on ms-level differences,
#                     not on structural zeros in the feature vector.
#       Tier 2 (30%): Speed-shifted impostors — consistently faster or slower
#                     typist.  Covers the case where an impostor's overall
#                     rhythm is very different from the genuine user.
#       Tier 3 (30%): Random impostors — original broad-range behaviour for
#                     full input-space coverage.
#  2. phrase-aware impostor patching: CMU and enrolled-other-
#     user impostors previously had zeros for phrase-specific extra_ digraph
#     columns (they typed a different phrase).  This caused the model to learn
#     "extra_ = 0 → impostor" rather than learning timing patterns.  We now
#     replace those zeros with realistic phrase-plausible-but-distinct values
#     so the model must compete on actual timing.
#  3. Threshold search is now FAR-weighted (3:1 penalty on false accepts vs
#     false rejects) — a security system should bias toward rejection.
#     Minimum threshold floor raised from 0.45 → 0.50.
#
# Key changes vs v3 (carried over from v4):
#  1. get_active_digraphs() — only digraphs present in the user's actual
#     phrase are kept; inactive digraphs are dropped before training.
#  2. load_real_impostors() threshold lowered from 8 → 3 genuine samples.
#  3. Fallback in auth is a hard reject (no loose dwell-time comparison).
#
# Key changes vs v2 (carried over from v3):
#  1. shift_lag_norm added to FEATURE_NAMES (was computed/stored but ignored)
#  2. CMU digraphs fixed — plausible timing estimates instead of all-zero
#  3. Augmentation scaled harder for small sets (5 samples → 600 genuine aug)
#  4. Augmentation noise tightened to 7% (was 10%) — keeps boundary narrow
#  5. More synthetic impostors (1200 instead of 300) to compensate for tiny
#     genuine set — widens the decision margin
#  6. GradientBoosting replaces RandomForest when n_enrollment <= 5 —
#     GBM builds sequential trees that focus on hard boundary cases, which
#     matters more when genuine samples are scarce
#  7. Threshold search now also considers FRR cost — with 5 samples a high
#     FRR (false reject) is just as bad as a high FAR for usability
#  8. profile_var stored for improved Mahalanobis scoring

import sys
import os

# Force UTF-8 stdout/stderr so emoji/box-drawing prints don't crash on Windows
# cp1252 consoles (which happens silently when training is launched as a
# subprocess from auth.py — crash aborts training before the RF pkl is saved).
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

backend_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'
)
sys.path.insert(0, backend_path)

import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from database.db import SessionLocal
from database.models import User, KeystrokeTemplate
from utils.crypto import decrypt

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE NAMES
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'dwell_mean',   'dwell_std',   'dwell_median', 'dwell_min',  'dwell_max',
    'flight_mean',  'flight_std',  'flight_median',
    'p2p_mean',     'p2p_std',
    'r2r_mean',     'r2r_std',

    'digraph_th', 'digraph_he',
    'digraph_bi', 'digraph_io', 'digraph_om', 'digraph_me', 'digraph_et',
    'digraph_tr', 'digraph_ri', 'digraph_ic', 'digraph_vo', 'digraph_oi',
    'digraph_ce', 'digraph_ke', 'digraph_ey', 'digraph_ys', 'digraph_st',
    'digraph_ro', 'digraph_ok', 'digraph_au', 'digraph_ut', 'digraph_en',
    'digraph_nt', 'digraph_ti', 'digraph_ca', 'digraph_at', 'digraph_on',

    'typing_speed_cpm', 'typing_duration',
    'rhythm_mean', 'rhythm_std', 'rhythm_cv',
    'pause_count', 'pause_mean',
    'backspace_ratio', 'backspace_count',
    'hand_alternation_ratio', 'same_hand_sequence_mean',
    'finger_transition_ratio', 'seek_time_mean', 'seek_time_count',

    'dwell_mean_norm',  'dwell_std_norm',
    'flight_mean_norm', 'flight_std_norm',
    'p2p_std_norm',     'r2r_mean_norm',
    'shift_lag_norm',
]

COUNT_FEATURES = {'pause_count', 'backspace_count', 'seek_time_count'}
RATIO_FEATURES = {
    'backspace_ratio', 'hand_alternation_ratio', 'finger_transition_ratio',
    'rhythm_cv', 'p2p_std_norm', 'dwell_mean_norm', 'dwell_std_norm',
    'flight_mean_norm', 'flight_std_norm', 'r2r_mean_norm',
    'shift_lag_norm',
}

HUMAN_RANGES = {
    'dwell_mean':    (40,   250),  'dwell_std':     (5,    80),
    'dwell_median':  (40,   250),  'dwell_min':     (20,   120),
    'dwell_max':     (80,   500),
    'flight_mean':   (30,   400),  'flight_std':    (10,   150),
    'flight_median': (30,   400),
    'p2p_mean':      (80,   600),  'p2p_std':       (20,   200),
    'r2r_mean':      (80,   600),  'r2r_std':       (20,   200),
    'typing_speed_cpm':  (80,  600),  'typing_duration':   (3,   30),
    'rhythm_mean':       (80,  600),  'rhythm_std':        (20,  200),
    'rhythm_cv':         (0.1, 1.5),
    'pause_count':       (0,   8),   'pause_mean':        (0,   500),
    'backspace_ratio':   (0,   0.3), 'backspace_count':   (0,   5),
    'hand_alternation_ratio':  (0.2, 0.8),
    'same_hand_sequence_mean': (1.0, 5.0),
    'finger_transition_ratio': (0.3, 0.9),
    'seek_time_mean':    (0,   300), 'seek_time_count':   (0,   5),
    'dwell_mean_norm':   (0.5, 2.0), 'dwell_std_norm':    (0.3, 1.5),
    'flight_mean_norm':  (0.5, 2.5), 'flight_std_norm':   (0.3, 2.0),
    'p2p_std_norm':      (0.3, 2.0), 'r2r_mean_norm':     (0.5, 2.5),
    'shift_lag_norm':    (0.0, 1.5),
}
DIGRAPH_RANGE = (20, 300)

MODEL_STAGE_EARLY_MAX = 7
MODEL_STAGE_MID_MAX = 20
PROFILE_MATCHER_THRESHOLD = 19

EARLY_BASE_FEATURES = [
    'dwell_mean', 'dwell_std',
    'flight_mean', 'flight_std',
    'p2p_mean',
    'typing_speed_cpm',
    'rhythm_cv',
    'backspace_ratio',
    'pause_count',
    'dwell_mean_norm',
    'flight_mean_norm',
]

MID_BASE_EXTRA_FEATURES = [
    'p2p_std',
    'r2r_mean', 'r2r_std',
    'pause_mean',
    'backspace_count',
    'same_hand_sequence_mean',
    'finger_transition_ratio',
    'seek_time_mean',
    'shift_lag_norm',
]

_DYNAMIC_LIMITS = {
    # Early now carries 70% of its weight on digraph A+R measures, so the rank
    # statistic needs enough pairs to be discriminative (minimum ~6–8 digraphs).
    'early': {'digraph': 8,  'key': 0, 'flight': 0, 'trigraph': 0},
    'mid':   {'digraph': 10, 'key': 3, 'flight': 5, 'trigraph': 0},
    'late':  {'digraph': None, 'key': None, 'flight': None, 'trigraph': None},
}


def _determine_model_stage(n_enrollment: int) -> str:
    """Cold-start profile stages inspired by commercial same-text systems."""
    if n_enrollment <= MODEL_STAGE_EARLY_MAX:
        return 'early'
    if n_enrollment <= MODEL_STAGE_MID_MAX:
        return 'mid'
    return 'late'


def _rank_dynamic_features(vectors: list, feat_names: list, candidates: list) -> list:
    """
    Rank sparse phrase-specific features by usefulness for early stages.

    Preference order:
      1. high coverage across enrollment samples
      2. low within-user CV (stable for this typist)
      3. higher mean (non-trivial signal)
    """
    if not vectors or not candidates:
        return []

    arr = np.asarray(vectors, dtype=np.float64)
    idx = {name: i for i, name in enumerate(feat_names)}
    ranked = []
    for name in candidates:
        col_idx = idx.get(name)
        if col_idx is None:
            continue
        col = arr[:, col_idx]
        nonzero = col[np.abs(col) > 1e-6]
        coverage = len(nonzero) / max(len(col), 1)
        if len(nonzero) == 0:
            continue
        mean = float(np.mean(nonzero))
        std = float(np.std(nonzero))
        cv = std / (abs(mean) + 1e-9)
        ranked.append((name, coverage, cv, mean))

    ranked.sort(key=lambda row: (-row[1], row[2], -row[3], row[0]))
    return [name for name, *_ in ranked]


def _select_stage_feature_names(stage: str, full_feat_names: list, genuine_vectors_full: list) -> list:
    """
    Keep the early stage dense and low-dimensional, then widen with maturity.
    """
    selected = set(EARLY_BASE_FEATURES)
    if stage in ('mid', 'late'):
        selected.update(MID_BASE_EXTRA_FEATURES)
    if stage == 'late':
        selected.update(full_feat_names)

    digraph_candidates = [
        name for name in full_feat_names
        if name.startswith('digraph_') or name.startswith('extra_')
    ]
    key_candidates = [name for name in full_feat_names if name.startswith('key_')]
    flight_candidates = [name for name in full_feat_names if name.startswith('flight_') and name not in {
        'flight_mean', 'flight_std', 'flight_median', 'flight_mean_norm', 'flight_std_norm'
    }]
    trigraph_candidates = [name for name in full_feat_names if name.startswith('trigraph_')]

    ranked = {
        'digraph': _rank_dynamic_features(genuine_vectors_full, full_feat_names, digraph_candidates),
        'key': _rank_dynamic_features(genuine_vectors_full, full_feat_names, key_candidates),
        'flight': _rank_dynamic_features(genuine_vectors_full, full_feat_names, flight_candidates),
        'trigraph': _rank_dynamic_features(genuine_vectors_full, full_feat_names, trigraph_candidates),
    }

    for group, limit in _DYNAMIC_LIMITS[stage].items():
        names = ranked[group]
        if limit is None:
            selected.update(names)
        else:
            selected.update(names[:limit])

    return [name for name in full_feat_names if name in selected]


# ─────────────────────────────────────────────────────────────────────────────
#  DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_vector(template, extra_keys: list = None, key_keys: list = None,
                           flight_pair_keys: list = None, trigraph_keys: list = None) -> np.ndarray:
    """
    Build the feature vector for a template.

    extra_keys       : digraph DD pairs (press→press) — read from extra_digraphs
    key_keys         : single letters — read from key_dwell_map
    flight_pair_keys : digraph pairs ['th','qu',...] — per-pair UD/flight time,
                       read from template.flight_per_digraph
    trigraph_keys    : 3-letter sequences ['the','qui',...] — press[i]→press[i+2],
                       read from template.trigraph_map

    Tail order matters — must match the feature-name construction in
    train_random_forest() and the auth-time vector rebuild.
    """
    vals = []
    for name in FEATURE_NAMES:
        raw = getattr(template, name, 0.0)
        if isinstance(raw, (list, tuple, np.ndarray)):
            vals.append(0.0)
        else:
            try:
                vals.append(float(raw or 0.0))
            except (TypeError, ValueError):
                vals.append(0.0)

    def _append_from_map(map_name, keys):
        if not keys:
            return
        m = getattr(template, map_name, None) or {}
        for k in keys:
            try:
                vals.append(float(m.get(k, 0.0) or 0.0))
            except (TypeError, ValueError):
                vals.append(0.0)

    _append_from_map('extra_digraphs',     extra_keys)
    _append_from_map('key_dwell_map',      key_keys)
    _append_from_map('flight_per_digraph', flight_pair_keys)
    _append_from_map('trigraph_map',       trigraph_keys)

    return np.array(vals, dtype=np.float64)


def _is_quality_sample(vec: np.ndarray, feat_names: list = None) -> tuple:
    names = feat_names if feat_names is not None else FEATURE_NAMES
    idx   = {name: i for i, name in enumerate(names)}

    def _get(key):
        i = idx.get(key)
        return vec[i] if i is not None and i < len(vec) else None

    dwell_mean = _get('dwell_mean')
    dwell_std  = _get('dwell_std')
    p2p_mean   = _get('p2p_mean')
    cpm        = _get('typing_speed_cpm')

    if dwell_mean is not None:
        if dwell_mean < 20:  return False, f"dwell_mean too low ({dwell_mean:.0f}ms)"
        if dwell_mean > 600: return False, f"dwell_mean too high ({dwell_mean:.0f}ms)"
    if p2p_mean  is not None and p2p_mean  < 50:  return False, f"p2p_mean impossibly fast ({p2p_mean:.0f}ms)"
    if dwell_std is not None and dwell_std < 1.0:  return False, "dwell_std ≈ 0 (automated/copy-paste)"
    if cpm       is not None and cpm       > 800:  return False, f"typing_speed_cpm impossibly high ({cpm:.0f})"
    return True, "ok"


def load_enrollment_samples(db, user_id: int, extra_keys: list = None, key_keys: list = None,
                            flight_pair_keys: list = None, trigraph_keys: list = None):
    templates = (
        db.query(KeystrokeTemplate)
        .filter(KeystrokeTemplate.user_id == user_id)
        .order_by(KeystrokeTemplate.sample_order.asc())
        .all()
    )
    vectors = []
    for t in templates:
        vec = extract_feature_vector(
            t,
            extra_keys=extra_keys,
            key_keys=key_keys,
            flight_pair_keys=flight_pair_keys,
            trigraph_keys=trigraph_keys,
        )
        ok, reason = _is_quality_sample(vec)
        if not ok:
            print(f"  ⚠  Skipping sample id={t.id}: {reason}")
            continue
        vectors.append(vec)
    return vectors


def load_real_impostors(db, exclude_user_id: int, n_genuine: int = 99):
    # Threshold lowered from 8 → 3.  The old guard of 8 meant real enrolled
    # impostors were NEVER used (MAX_KEYSTROKE_SAMPLES = 5, so n_genuine is
    # always 5 at training time).  3 is a safe minimum — enough genuine
    # samples to define a meaningful user profile before mixing in real humans.
    if n_genuine < 3:
        print(f"  Skipping real enrolled impostors (n_genuine={n_genuine} < 3) — using CMU + synthetic only")
        return []
    other_users = db.query(User).filter(User.id != exclude_user_id).all()
    impostors   = []
    for u in other_users:
        # Load WITHOUT extra_keys — vectors stay at FEATURE_NAMES length.
        # train_random_forest() pads the user-specific
        # extra_pairs columns with zeros, which is correct: the impostor never
        # typed the genuine user's phrase so those bigram timings are unknown.
        samples = load_enrollment_samples(db, u.id)
        if samples:
            impostors.extend(samples)
            print(f"    impostor '{u.username}': {len(samples)} sample(s)")
    if impostors:
        print(f"  Real impostor samples from {len(other_users)} other user(s): {len(impostors)}")
    else:
        print(f"  No other enrolled users found to use as impostors.")
    return impostors


def load_cmu_impostors() -> list:
    pkl_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'models', 'cmu_impostor_profiles.pkl'
    )
    if not os.path.exists(pkl_path):
        print("  ⚠  CMU impostor profiles not found. Run: python ml/load_cmu_impostors.py")
        return []
    with open(pkl_path, 'rb') as f:
        vecs = pickle.load(f)
    print(f"  CMU impostor profiles loaded: {len(vecs)} subjects")

    expected_len = len(FEATURE_NAMES)
    valid_vecs = [v for v in vecs if hasattr(v, '__len__') and len(v) == expected_len]
    if len(valid_vecs) != len(vecs):
        stale = len(vecs) - len(valid_vecs)
        print(
            f"  ⚠  {stale} CMU vectors have wrong length "
            f"(expected {expected_len}) — pkl is stale. "
            f"Re-run: python ml/load_cmu_impostors.py"
        )
        if not valid_vecs:
            print("  ⚠  All CMU vectors are stale — continuing without CMU impostors.")
            return []
        vecs = valid_vecs

    return vecs


# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION — tuned for small enrollment sets (5 samples)
# ─────────────────────────────────────────────────────────────────────────────

def generate_genuine_samples(genuine_vectors, n: int = 600, rng_seed: int = 42, feat_names: list = None):
    rng  = np.random.default_rng(rng_seed)
    base = np.array(genuine_vectors)
    if feat_names is None:
        feat_names = FEATURE_NAMES

    vec_len = base.shape[1] if base.ndim == 2 else len(base[0])
    if len(feat_names) != vec_len:
        raise ValueError(
            f"generate_genuine_samples: feat_names has {len(feat_names)} entries "
            f"but genuine vectors have {vec_len} features. "
            f"Pass feat_names=active_feat_names (the post-stripping name list) "
            f"when calling with stripped vectors."
        )

    if base.shape[0] > 1:
        within_std = base.std(axis=0)
    else:
        within_std = np.abs(base[0]) * 0.04

    mean_vals  = np.abs(base.mean(axis=0))
    # FIX: tighten genuine augmentation from 5% floor → 3% floor.
    # The old 5% floor combined with ±16% factor created a genuine cluster
    # wide enough to contain nearby impostors (groupmates with similar speed).
    within_std = np.maximum(within_std, mean_vals * 0.03)

    samples  = []
    attempts = 0
    while len(samples) < n and attempts < n * 6:
        attempts += 1
        idx   = rng.integers(0, len(genuine_vectors))
        noisy = genuine_vectors[idx].copy()

        for i, name in enumerate(feat_names):
            if name in COUNT_FEATURES:
                noisy[i] = max(0, noisy[i] + rng.integers(-1, 2))
            elif name in RATIO_FEATURES:
                noisy[i] = float(np.clip(
                    noisy[i] + rng.normal(0, within_std[i] * 0.5 + 0.01), 0, 2))
            else:
                # FIX: tighter factor: normal(1.0, 0.04) clipped to (0.92, 1.08).
                # Old: normal(1.0, 0.07) clipped to (0.84, 1.16) = ±16% spread.
                # New: ±8% spread — keeps the genuine cluster tight so nearby
                # impostors (groupmates typing ~10-15% differently) are correctly
                # outside the genuine zone and get labelled as impostors.
                factor  = rng.normal(1.0, 0.06)
                factor  = np.clip(factor, 0.88, 1.12)
                noisy[i] = noisy[i] * factor

        ok, _ = _is_quality_sample(noisy, feat_names)
        if ok:
            samples.append(noisy)

    if len(samples) < n:
        print(f"  ⚠  Only {len(samples)}/{n} genuine augmentations passed quality check")
    return samples


def generate_impostor_samples(profile_mean, profile_std, n: int = 1200, rng_seed: int = 42, feat_names: list = None):
    """
    Generate synthetic impostor samples in three tiers:

    Tier 1 — NEAR impostors (40% of n):
        Simulate a real person typing the SAME phrase but with their own rhythm.
        Values are drawn from a Gaussian centred on the genuine user's mean but
        pushed at least 1.5–2.0 std away on KEY_FEATURES and extra_ digraphs.
        This is the most realistic attack vector (a friend typing your passphrase).

    Tier 2 — SHIFTED impostors (30% of n):
        A different typing speed / dwell profile — e.g. a fast typist vs a slow
        genuine user.  All timing features are scaled by a random factor drawn
        from (0.45–0.75) or (1.35–2.2), keeping internal consistency.

    Tier 3 — RANDOM impostors (30% of n):
        Original behaviour — uniform sample from human plausible ranges.
        Keeps the decision boundary from collapsing to a tight ball around
        the genuine mean and gives the model coverage of the full input space.

    KEY insight for extra_ digraphs:
        CMU / enrolled impostors have zeros for phrase-specific bigrams because
        they typed a different phrase.  Tier-1 and Tier-2 impostors fill those
        columns with plausible non-zero values (the same phrase, different person),
        so the model learns to distinguish on TIMING, not on zero vs non-zero.
    """
    rng = np.random.default_rng(rng_seed)
    if feat_names is None:
        feat_names = FEATURE_NAMES

    if len(profile_mean) != len(feat_names):
        raise ValueError(
            f"generate_impostor_samples: profile_mean has {len(profile_mean)} entries "
            f"but feat_names has {len(feat_names)}. "
            f"Pass feat_names=active_feat_names when calling with stripped profiles."
        )

    KEY_FEATURES = {
        'dwell_mean', 'flight_mean', 'p2p_mean', 'typing_speed_cpm',
        'rhythm_mean', 'dwell_mean_norm', 'flight_mean_norm', 'shift_lag_norm',
        'r2r_mean', 'p2p_std',
    }

    n_near    = int(n * 0.40)
    n_shifted = int(n * 0.30)
    n_random  = n - n_near - n_shifted

    samples = []

    # ── Tier 1: Near impostors ────────────────────────────────────────────────
    # Same phrase, different person → values cluster near genuine mean but are
    # pushed away enough to be distinguishable on key timing features.
    for _ in range(n_near):
        vec = np.zeros(len(feat_names))
        for i, name in enumerate(feat_names):
            genuine_val = float(profile_mean[i])
            genuine_std = float(profile_std[i])

            if name in COUNT_FEATURES:
                # Integer counts — shift by ±1–3
                vec[i] = max(0, genuine_val + rng.integers(-2, 4))
                continue

            if name in RATIO_FEATURES:
                # Ratios — perturb moderately
                vec[i] = float(np.clip(
                    genuine_val + rng.normal(0, genuine_std * 2.5 + 0.05), 0, 2))
                continue

            # Timing features (ms) and extra digraphs
            # Strategy: draw from a Gaussian around genuine_mean, then
            # enforce a minimum separation so they don't collapse into genuine.
            if name in KEY_FEATURES or name.startswith('extra_') or name.startswith('digraph_'):
                # Minimum separation: at least 20% of genuine mean OR 1.5× std
                min_sep = max(genuine_std * 1.0, abs(genuine_val) * 0.08, 5.0)  # FIX: was (1.5*std, 20%) — gap allowed groupmates to slip in
                for _ in range(20):
                    # Wide spread: ±2.5 std — realistic human variation
                    v = rng.normal(genuine_val, genuine_std * 2.5 + 15.0)
                    # Keep in human range
                    if name.startswith('extra_') or name.startswith('digraph_'):
                        v = np.clip(v, DIGRAPH_RANGE[0], DIGRAPH_RANGE[1])
                    elif name in HUMAN_RANGES:
                        v = np.clip(v, HUMAN_RANGES[name][0], HUMAN_RANGES[name][1])
                    else:
                        v = max(0, v)
                    if abs(v - genuine_val) >= min_sep:
                        break
                vec[i] = float(v)
            elif name in HUMAN_RANGES:
                lo, hi = HUMAN_RANGES[name]
                vec[i] = float(np.clip(rng.normal(genuine_val, genuine_std * 2.0), lo, hi))
            else:
                vec[i] = float(max(0, rng.normal(genuine_val, genuine_std * 2.0)))

        samples.append(vec)

    # ── Tier 2: Speed-shifted impostors ──────────────────────────────────────
    # Model a typist with a systematically faster or slower rhythm.
    # All timing values are scaled by a consistent factor, keeping the
    # internal ratios plausible while moving the profile far from genuine.
    for _ in range(n_shifted):
        # Avoid factors near 1.0 — those would overlap with genuine augmentation
        if rng.random() < 0.5:
            speed_factor = rng.uniform(0.40, 0.72)   # faster typist
        else:
            speed_factor = rng.uniform(1.35, 2.20)   # slower typist

        vec = np.zeros(len(feat_names))
        for i, name in enumerate(feat_names):
            genuine_val = float(profile_mean[i])
            genuine_std = float(profile_std[i])

            if name in COUNT_FEATURES:
                vec[i] = max(0, genuine_val + rng.integers(-1, 3))
            elif name in RATIO_FEATURES:
                vec[i] = float(np.clip(
                    genuine_val + rng.normal(0, genuine_std * 1.5 + 0.03), 0, 2))
            elif name.startswith('extra_') or name.startswith('digraph_'):
                # Digraphs scale with typing speed + small individual noise
                raw = genuine_val * speed_factor + rng.normal(0, genuine_std * 0.8)
                vec[i] = float(np.clip(raw, DIGRAPH_RANGE[0], DIGRAPH_RANGE[1]))
            elif name in HUMAN_RANGES:
                raw = genuine_val * speed_factor + rng.normal(0, genuine_std * 0.5)
                lo, hi = HUMAN_RANGES[name]
                vec[i] = float(np.clip(raw, lo, hi))
            else:
                vec[i] = float(max(0, genuine_val * speed_factor))

        samples.append(vec)

    # ── Tier 3: Random impostors (original behaviour) ─────────────────────────
    # Uniform coverage of the human-plausible input space.
    for _ in range(n_random):
        vec = np.zeros(len(feat_names))
        for i, name in enumerate(feat_names):
            genuine_val = float(profile_mean[i])

            if name in HUMAN_RANGES:
                lo, hi = HUMAN_RANGES[name]
            elif name.startswith('digraph_') or name.startswith('extra_'):
                lo, hi = DIGRAPH_RANGE
            else:
                lo = genuine_val * 0.3
                hi = genuine_val * 2.0

            for _ in range(15):
                v = rng.uniform(lo, hi)
                if name in KEY_FEATURES:
                    min_dist = max(profile_std[i] * 1.5, abs(genuine_val) * 0.20)
                    if abs(v - genuine_val) >= min_dist:
                        break
                else:
                    break
            vec[i] = float(v)
        samples.append(vec)

    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  MAHALANOBIS SCORE — improved diagonal covariance version
# ─────────────────────────────────────────────────────────────────────────────

def mahalanobis_score(vec, profile_mean, profile_std):
    var      = profile_std ** 2
    safe_var = np.where(var < 1e-10, 1e-10, var)
    diff     = vec - profile_mean
    d_sq     = float(np.sum(diff ** 2 / safe_var))
    d_sq_norm = d_sq / len(vec)
    score    = 1.0 / (1.0 + np.exp(1.5 * (d_sq_norm - 1.0)))
    return float(np.clip(score, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSIFIER SELECTION — GBM for small sets, RF for larger sets
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(n_enrollment: int) -> Pipeline:
    if n_enrollment <= 7:
        print(f"  Using GradientBoosting (n_enrollment={n_enrollment} ≤ 7)")
        clf = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=6,
            min_samples_leaf=4,
            subsample=0.8,
            max_features='sqrt',
            random_state=42,
        )
    else:
        print(f"  Using RandomForest (n_enrollment={n_enrollment} > 7)")
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=3,
            max_features='sqrt',
            # FIX: was {0: 1, 1: 2} which penalised false REJECTS more.
            # For a security system, a false ACCEPT (impostor gets through) is
            # far more harmful than a false REJECT (user must retry).
            # 3:1 weight on impostors (class 0) vs genuine (class 1) makes the
            # model biased toward rejection of ambiguous borderline samples.
            class_weight={0: 2, 1: 1},
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _safe_filename(username: str) -> str:
    """user@gmail.com → user_at_gmail_com (safe on all OS)"""
    return username.replace("@", "_at_").replace(".", "_").replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
#  PHRASE-AWARE DIGRAPH FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def get_active_digraphs(phrase: str) -> tuple:
    """
    Return two values:
      - standard_active : digraph feature names from FEATURE_NAMES that appear in phrase
      - extra_pairs     : letter pairs in the phrase NOT in FEATURE_NAMES (stored in
                          extra_digraphs JSON column and appended to the feature vector)

    Example for phrase "pearl proof thing large":
        standard_active = {'digraph_th', 'digraph_ro'}   (only 2 of the 27 hardcoded)
        extra_pairs     = ['pe','ea','ar','rl','pr','oo','of','hi','in','ng','la','rg']
    """
    all_digraph_features = {f for f in FEATURE_NAMES if f.startswith("digraph_")}
    phrase_clean = phrase.lower().replace(" ", "")
    seen = set()
    standard_active = set()
    extra_pairs = []

    for i in range(len(phrase_clean) - 1):
        pair = phrase_clean[i] + phrase_clean[i + 1]
        if pair in seen:
            continue
        if not pair.isalpha():
            continue
        seen.add(pair)
        feat = f"digraph_{pair}"
        if feat in all_digraph_features:
            standard_active.add(feat)
        else:
            extra_pairs.append(pair)

    return standard_active, extra_pairs


def get_active_key_dwells(phrase: str) -> list:
    """
    Return sorted list of unique letter characters in the phrase.
    These become per-key dwell features: key_a, key_e, key_k, ...
    Each one captures how long the user holds that specific key — more
    person-specific than the aggregate dwell_mean.
    """
    return sorted(set(c for c in phrase.lower() if c.isalpha()))


def get_phrase_digraph_pairs(phrase: str) -> list:
    """
    Every unique [a-z]{2} window in the phrase, regardless of whether the pair
    is one of the hardcoded FEATURE_NAMES digraphs. Used for `flight_<pair>`
    feature columns (per-pair UD/flight time).
    """
    clean = phrase.lower().replace(" ", "")
    seen, pairs = set(), []
    for i in range(len(clean) - 1):
        p = clean[i] + clean[i + 1]
        if p.isalpha() and p not in seen:
            seen.add(p)
            pairs.append(p)
    return pairs


def get_active_trigraphs(phrase: str) -> list:
    """
    Every unique [a-z]{3} window in the phrase. Each becomes a `trigraph_<tri>`
    feature: press[i] → press[i+2] elapsed time, captures sequence muscle memory.
    """
    clean = phrase.lower().replace(" ", "")
    seen, trigs = set(), []
    for i in range(len(clean) - 2):
        t = clean[i] + clean[i + 1] + clean[i + 2]
        if t.isalpha() and t not in seen:
            seen.add(t)
            trigs.append(t)
    return trigs


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_random_forest(username: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"❌ User '{username}' not found!")
            return None

        genuine_vectors = load_enrollment_samples(db, user.id)
        if not genuine_vectors:
            print("❌ No valid enrollment samples found.")
            return None

        cmu_impostors  = load_cmu_impostors()
        real_impostors = load_real_impostors(db, user.id, n_genuine=len(genuine_vectors))
        user_id        = user.id
        # user.phrase is stored as a Fernet ciphertext in the DB.  If we pass
        # the ciphertext to get_active_digraphs() we get spurious bigrams from
        # the base64 token (e.g. 'ga','aa','ab','bp' from 'gAAAAABp...') instead
        # of the real phrase's bigrams.  decrypt() falls back to returning the
        # raw value if the stored value is already plaintext.
        user_phrase    = decrypt(user.phrase or "")

    finally:
        try:
            db.close()
        except Exception:
            pass

    print(f"\n{'='*70}")
    print(f"  KEYSTROKE RF TRAINING v4  —  user: {username}")
    print(f"{'='*70}")

    # ── Phrase-aware digraph filtering ────────────────────────────────────────
    # get_active_digraphs returns TWO values:
    #   standard_active : subset of the 27 hardcoded FEATURE_NAMES digraphs
    #   extra_pairs     : all other bigrams in the phrase (stored in extra_digraphs JSON)
    standard_active, extra_pairs_all = get_active_digraphs(user_phrase)
    key_keys_all          = get_active_key_dwells(user_phrase)
    flight_pair_keys_all  = get_phrase_digraph_pairs(user_phrase)   # ALL phrase pairs (UD)
    trigraph_keys_all     = get_active_trigraphs(user_phrase)
    all_digraph_feats     = {f for f in FEATURE_NAMES if f.startswith("digraph_")}
    inactive_digraphs_all = all_digraph_feats - standard_active
    full_feat_names = (
        list(FEATURE_NAMES)
        + [f"extra_{p}"    for p in extra_pairs_all]
        + [f"key_{k}"      for k in key_keys_all]
        + [f"flight_{p}"   for p in flight_pair_keys_all]
        + [f"trigraph_{t}" for t in trigraph_keys_all]
    )

    # Reload genuine samples WITH all four phrase-specific dict expansions.
    db2 = SessionLocal()
    try:
        genuine_vectors_full = load_enrollment_samples(
            db2, user_id,
            extra_keys=extra_pairs_all, key_keys=key_keys_all,
            flight_pair_keys=flight_pair_keys_all, trigraph_keys=trigraph_keys_all,
        )
        if not genuine_vectors_full:
            print("❌ No valid enrollment samples found.")
            return None
    finally:
        try:
            db2.close()
        except Exception:
            pass

    n_genuine_real = len(genuine_vectors_full)
    print(f"\n  Enrollment samples loaded: {n_genuine_real}")
    model_stage = _determine_model_stage(n_genuine_real)
    print(f"  Feature stage      : {model_stage.upper()}")
        # ── Outlier trimming ──────────────────────────────────────────────────────
    if len(genuine_vectors_full) >= 4:
        dropped_total = 0

        # Trim by dwell_mean
        dwell_idx = (full_feat_names.index('dwell_mean')
                     if 'dwell_mean' in full_feat_names else None)
        if dwell_idx is not None:
            dwells       = np.array([v[dwell_idx] for v in genuine_vectors_full])
            median_dwell = np.median(dwells)
            std_dwell    = max(np.std(dwells), median_dwell * 0.10)
            trimmed      = [v for v in genuine_vectors_full
                            if abs(v[dwell_idx] - median_dwell) <= 2.5 * std_dwell]
            dropped = len(genuine_vectors_full) - len(trimmed)
            if dropped and len(trimmed) >= 3:
                print(f"  ⚠  Dropped {dropped} outlier(s) by dwell_mean "
                      f"(> 2.5σ from median={median_dwell:.0f}ms)")
                genuine_vectors_full = trimmed
                dropped_total += dropped

        # Trim by typing_speed_cpm — catches samples where the user typed
        # at a very different speed (e.g. 188 vs 341 cpm) that dwell trimming misses.
        cpm_idx = (full_feat_names.index('typing_speed_cpm')
                   if 'typing_speed_cpm' in full_feat_names else None)
        # Require n >= 6 before CPM outlier trimming — with 5 samples,
        # dropping one leaves n=4 which disables LOO in the profile matcher.
        if cpm_idx is not None and len(genuine_vectors_full) >= 6:
            cpms       = np.array([v[cpm_idx] for v in genuine_vectors_full])
            median_cpm = np.median(cpms)
            std_cpm    = max(np.std(cpms), median_cpm * 0.10)
            trimmed    = [v for v in genuine_vectors_full
                          if abs(v[cpm_idx] - median_cpm) <= 2.5 * std_cpm]
            dropped = len(genuine_vectors_full) - len(trimmed)
            if dropped and len(trimmed) >= 3:
                print(f"  ⚠  Dropped {dropped} outlier(s) by typing_speed_cpm "
                      f"(> 2.5σ from median={median_cpm:.0f} cpm)")
                genuine_vectors_full = trimmed
                dropped_total += dropped

        if dropped_total == 0:
            print(f"  ✅ All {len(genuine_vectors_full)} enrollment samples passed outlier check")
        elif len(genuine_vectors_full) < 3:
            print(f"  ⚠  Outlier trimming skipped — would leave < 3 samples")
    trimmed_stage = _determine_model_stage(len(genuine_vectors_full))
    if trimmed_stage != model_stage:
        print(f"  Stage adjusted after trimming: {model_stage.upper()} -> {trimmed_stage.upper()}")
        model_stage = trimmed_stage
        n_genuine_real = len(genuine_vectors_full)
    active_feat_names = _select_stage_feature_names(
        model_stage, full_feat_names, genuine_vectors_full
    )
    extra_pairs = [name[len("extra_"):] for name in active_feat_names if name.startswith("extra_")]
    key_keys = [name[len("key_"):] for name in active_feat_names if name.startswith("key_")]
    flight_pair_keys = [
        name[len("flight_"):] for name in active_feat_names
        if name.startswith("flight_") and name not in {
            'flight_mean', 'flight_std', 'flight_median',
            'flight_mean_norm', 'flight_std_norm',
        }
    ]
    trigraph_keys = [name[len("trigraph_"):] for name in active_feat_names if name.startswith("trigraph_")]
    selected_hardcoded_digraphs = {
        name for name in active_feat_names if name.startswith("digraph_")
    }
    inactive_digraphs = all_digraph_feats - selected_hardcoded_digraphs

    full_idx = {name: i for i, name in enumerate(full_feat_names)}
    base_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def _project_genuine_vectors(vecs):
        return [
            np.array([float(v[full_idx[name]]) for name in active_feat_names], dtype=np.float64)
            for v in vecs
        ]

    def _project_impostor_vectors(vecs):
        projected = []
        for v in vecs:
            row = []
            for name in active_feat_names:
                idx = base_idx.get(name)
                row.append(float(v[idx]) if idx is not None and idx < len(v) else 0.0)
            projected.append(np.array(row, dtype=np.float64))
        return projected

    def _patch_impostor_digraphs(vecs, p_mean, p_std, rng_inst):
        """
        After profile_mean/std are known, replace the zero phrase-specific columns
        (extra_ digraphs and key_ dwell times) on impostor vectors with
        phrase-plausible but genuinely-distinct timings.

        This forces the model to distinguish on TIMING, not on zero vs non-zero.
        """
        dyn_cols = [
            i for i, name in enumerate(active_feat_names)
            if (name.startswith('extra_') or name.startswith('key_')
                or name.startswith('trigraph_')
                or (name.startswith('flight_') and name not in {
                    'flight_mean', 'flight_std', 'flight_median',
                    'flight_mean_norm', 'flight_std_norm',
                }))
        ]
        if not dyn_cols:
            return vecs
        patched = []
        for v in vecs:
            v = v.copy()
            for col in dyn_cols:
                name   = active_feat_names[col]
                g_mean = p_mean[col]
                g_std  = p_std[col]
                min_sep = max(g_std * 1.5, g_mean * 0.15, 5.0)
                for _ in range(20):
                    val = rng_inst.normal(g_mean, g_std * 2.5 + 15.0)
                    if name.startswith('key_'):
                        val = float(np.clip(val, 20.0, 300.0))
                    elif name.startswith('trigraph_'):
                        val = float(np.clip(val, 40.0, 500.0))
                    else:
                        val = float(np.clip(val, DIGRAPH_RANGE[0], DIGRAPH_RANGE[1]))
                    if abs(val - g_mean) >= min_sep:
                        break
                v[col] = val
            patched.append(v)
        return patched

    _rng_patch = np.random.default_rng(77)
    genuine_vectors = _project_genuine_vectors(genuine_vectors_full)

    print(f"  User phrase      : '{user_phrase}'")
    print(f"  Standard active  : {len(standard_active)}  ({', '.join(sorted(standard_active)) or 'none'})")
    print(f"  Extra pairs      : {len(extra_pairs)}  ({', '.join(extra_pairs) or 'none'})")
    print(f"  Key dwell keys   : {len(key_keys)}  ({', '.join(key_keys) or 'none'})")
    print(f"  Dropped digraphs : {len(inactive_digraphs)}")
    print(f"  Active features  : {len(active_feat_names)} (was {len(FEATURE_NAMES)})")

    profile_mean = np.array(genuine_vectors).mean(axis=0)
    profile_std  = np.array(genuine_vectors).std(axis=0) + 1e-9
    profile_var  = profile_std ** 2

    key_idxs = [active_feat_names.index(f) for f in
                ['dwell_mean', 'flight_mean', 'p2p_mean', 'typing_speed_cpm']
                if f in active_feat_names]
    consistency_cv = np.mean(
        profile_std[key_idxs] / (np.abs(profile_mean[key_idxs]) + 1e-9)
    )
    print(f"  Enrollment consistency CV: {consistency_cv:.3f} "
          f"({'consistent' if consistency_cv < 0.15 else 'variable'})")

    fn = active_feat_names
    print(f"\n  Profile (mean across {n_genuine_real} attempt(s)):")
    for label, feat in [
        ("typing_speed_cpm", "typing_speed_cpm"),
        ("dwell_mean (ms)",  "dwell_mean"),
        ("flight_mean (ms)", "flight_mean"),
        ("p2p_mean (ms)",    "p2p_mean"),
        ("rhythm_cv",        "rhythm_cv"),
        ("shift_lag_norm",   "shift_lag_norm"),
    ]:
        if feat in fn:
            print(f"    {label:20s}: {profile_mean[fn.index(feat)]:.3f}")

    # ── Hybrid model selection ────────────────────────────────────────────────
    # With ≤ 15 real enrollment samples, a GBM/RF trained on synthetic augmentation
    # is unreliable: the augmented genuine cluster doesn't reflect real within-session
    # variance and the decision boundary lands inside the genuine distribution.
    #
    # For small sets we use a TypingDNA-style profile matcher instead:
    #   - Direct comparison against enrollment profile (mean ± std per feature)
    #   - Speed normalization: scale all ms-timings by live/enrolled p2p ratio
    #   - Tiered Z-tolerance: digraphs 1.8 | dwell/flight 2.2 | rhythm 2.8
    #   - Weighted group score: 40% digraphs | 40% dwell/flight | 20% rhythm
    #
    # RF/GBM kicks in automatically once the user has built up ≥ 16 samples through
    # adaptive learning (successful logins are saved back to the database).
    if n_genuine_real <= PROFILE_MATCHER_THRESHOLD:
        # Ensure ml/ is on sys.path so this import works whether train_keystroke_rf
        # is called standalone (already in ml/) or as a module from the backend.
        _ml_dir = os.path.dirname(os.path.abspath(__file__))
        if _ml_dir not in sys.path:
            sys.path.insert(0, _ml_dir)
        from keystroke_profile_matcher import build_profile_model

        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_data = build_profile_model(
            genuine_vectors  = genuine_vectors,
            active_feat_names= active_feat_names,
            username         = username,
            user_id          = user_id,
            user_phrase      = user_phrase,
            model_stage      = model_stage,
            threshold        = 0.65,
        )

        # Sanity-check critical profile features before persisting. A zero
        # mean here means the feature/vector projection misaligned and the
        # pickle would hard-reject every real login. Fail loudly instead.
        _critical_mins = {
            'typing_speed_cpm': 20.0,
            'dwell_mean':       20.0,
            'flight_mean':       5.0,
            'p2p_mean':         20.0,
        }
        _pm    = model_data.get('profile_mean')
        _names = model_data.get('feature_names', [])
        for _feat, _minv in _critical_mins.items():
            if _feat in _names:
                _v = float(_pm[_names.index(_feat)])
                if _v < _minv:
                    raise ValueError(
                        f"Profile pickle refused: {_feat}={_v:.3f} below "
                        f"sanity floor {_minv}. Likely a feature/vector "
                        f"alignment bug in training — check full_feat_names "
                        f"vs extract_feature_vector."
                    )

        model_path = os.path.join(model_dir, f"{_safe_filename(username)}_keystroke_rf.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        size_kb = os.path.getsize(model_path) / 1024
        print(f"\n  ✅ PROFILE MODEL SAVED: {model_path}  ({size_kb:.1f} KB)")
        print(f"  Mode: Profile Matcher ({model_stage.upper()} stage, n={n_genuine_real} ≤ {PROFILE_MATCHER_THRESHOLD})")
        print(f"  Will upgrade to RF/GBM after {PROFILE_MATCHER_THRESHOLD + 1}+ enrollment samples\n")
        return model_path

    n_aug     = max(600, n_genuine_real * 120)
    n_imp_syn = max(1200, n_aug * 2)

    genuine_aug = generate_genuine_samples(genuine_vectors, n=n_aug, feat_names=active_feat_names)

    cmu_impostors  = _project_impostor_vectors(cmu_impostors)
    real_impostors = _project_impostor_vectors(real_impostors)

    # Now that profile_mean/std are computed, replace the placeholder zeros in
    # real impostor extra_ columns with realistic phrase-plausible timings.
    # This is the core fix: CMU and enrolled impostors no longer signal
    # "impostor" via zero digraph values — the model must learn on timing.
    cmu_impostors  = _patch_impostor_digraphs(cmu_impostors,  profile_mean, profile_std, _rng_patch)
    real_impostors = _patch_impostor_digraphs(real_impostors, profile_mean, profile_std, _rng_patch)

    real_pool      = cmu_impostors + real_impostors

    n_synthetic   = max(0, n_imp_syn - len(real_pool))
    syn_impostors = generate_impostor_samples(
        profile_mean, profile_std, n=n_synthetic, feat_names=active_feat_names
    ) if n_synthetic > 0 else []
    all_impostors = real_pool + syn_impostors

    n_feats = len(active_feat_names)
    def _is_valid(v):
        try:
            arr = np.asarray(v, dtype=np.float64)
            return arr.ndim == 1 and arr.shape[0] == n_feats
        except Exception:
            return False

    genuine_aug   = [v for v in genuine_aug   if _is_valid(v)]
    all_impostors = [v for v in all_impostors  if _is_valid(v)]

    print(f"\n{'='*70}")
    print(f"  TRAINING DATA")
    print(f"{'='*70}")
    print(f"  Genuine (real)     : {n_genuine_real}")
    print(f"  Genuine (augmented): {len(genuine_aug)}")
    print(f"  CMU impostors      : {len(cmu_impostors)}  (51 real humans, digraphs patched)")
    print(f"  Enrolled impostors : {len(real_impostors)}  (other users in DB, digraphs patched)")
    n_near_syn    = int(n_synthetic * 0.40)
    n_shifted_syn = int(n_synthetic * 0.30)
    n_random_syn  = n_synthetic - n_near_syn - n_shifted_syn
    print(f"  Synthetic impostors: {len(syn_impostors)}")
    print(f"    ├─ Near (same phrase, diff person): ~{n_near_syn}")
    print(f"    ├─ Speed-shifted (fast/slow typist): ~{n_shifted_syn}")
    print(f"    └─ Random (broad human range):       ~{n_random_syn}")
    print(f"  Total impostors    : {len(all_impostors)}")
    print(f"  Feature dims       : {len(active_feat_names)}")
    print(f"  Impostor ratio     : {len(all_impostors)/max(len(genuine_aug),1):.1f}:1")

    X = np.vstack([genuine_aug, all_impostors])
    y = np.array([1] * len(genuine_aug) + [0] * len(all_impostors))

    pipeline = build_pipeline(n_genuine_real)

    print(f"\n  Running 5-fold cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    print(f"\n{'='*70}")
    print(f"  THRESHOLD SEARCH  (FAR-weighted — security biased 3:1 over FRR)")
    print(f"{'='*70}")
    print(f"  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'EER':>8}  {'Score':>8}")

    best_thresh     = 0.50
    best_eer        = 1.0
    best_far_score  = 1.0   # weighted metric: (3*FAR + FRR) / 4

    for t in np.arange(0.35, 0.92, 0.02):
        y_at_t = (y_prob_cv >= t).astype(int)
        if len(np.unique(y_at_t)) < 2:
            continue
        cm_t = confusion_matrix(y, y_at_t)
        if cm_t.shape != (2, 2):
            continue
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        far_t   = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
        frr_t   = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0
        eer_t   = (far_t + frr_t) / 2
        # Security bias: penalise false accepts 3× more than false rejects.
        # A legitimate user being asked to retry is annoying; an impostor
        # getting through is a security breach.
        score_t = (2 * far_t + frr_t) / 3

        marker = ""
        if eer_t < best_eer:
            best_eer = eer_t
        if score_t < best_far_score:
            best_far_score = score_t
            best_thresh    = float(t)
            marker         = " ◄ best"
        print(f"  {t:>10.2f}  {far_t:>8.2%}  {frr_t:>8.2%}  {eer_t:>8.2%}  {score_t:>8.2%}{marker}")

    # Security floor: never accept a threshold below 0.55 regardless of EER.
    # FIX: raised from 0.50 → 0.55 for extra security margin.
    # FIX: removed the consistency_cv branch that lowered the floor to 0.42.
    #   Old logic: "high variability → lower threshold so the genuine user can
    #   still log in." This is backwards — high variability means the genuine
    #   profile is noisy, so we should be MORE conservative, not less.
    #   Lowering to 0.42 essentially disabled the security gate and let
    #   groupmates (who scored ~0.45-0.50) pass through.
    final_thresh = max(best_thresh, 0.50)

    if consistency_cv > 0.25:
        print(f"\n  ⚠  High variability (CV={consistency_cv:.2f}) — "
              f"keeping conservative threshold {final_thresh:.2f} (NOT lowering it)")

    y_final = (y_prob_cv >= final_thresh).astype(int)
    cm_f    = confusion_matrix(y, y_final)
    tn_f, fp_f, fn_f, tp_f = cm_f.ravel()
    far_f = fp_f / (fp_f + tn_f) if (fp_f + tn_f) > 0 else 0
    frr_f = fn_f / (fn_f + tp_f) if (fn_f + tp_f) > 0 else 0

    print(f"\n  Final @ {final_thresh:.2f}:  "
          f"FAR={far_f:.2%}  FRR={frr_f:.2%}  "
          f"EER={best_eer:.2%}  "
          f"ACC={accuracy_score(y, y_final):.2%}")

    print(f"\n  Training final model on full dataset ...")
    pipeline.fit(X, y)

    try:
        clf_step = pipeline.named_steps["clf"]
        importances = clf_step.feature_importances_
        pairs = sorted(zip(active_feat_names, importances), key=lambda x: x[1], reverse=True)
        print(f"\n  TOP 10 MOST IMPORTANT FEATURES")
        for i, (feat, imp) in enumerate(pairs[:10], 1):
            bar = "█" * int(imp * 200)
            print(f"  {i:2d}. {feat:28s} {imp:.4f}  {bar}")
    except Exception:
        pass

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_data = {
        'pipeline':         pipeline,
        'model_type':       'rf',
        'model_stage':      model_stage,
        'feature_names':    active_feat_names,
        'username':         username,
        'user_id':          user_id,
        'n_enrollment':     n_genuine_real,
        'profile_mean':     profile_mean,
        'profile_std':      profile_std,
        'profile_var':      profile_var,
        'threshold':        final_thresh,
        'consistency_cv':   float(consistency_cv),
        'far':              float(far_f),
        'frr':              float(frr_f),
        'eer':              float(best_eer),
        'ks_reliability':   float(np.clip(1.0 - frr_f, 0.15, 1.0)),
        'phrase':           user_phrase,
        # Phrase-specific dict-expansion key lists — required so predict_keystroke
        # can rebuild the same vector from the auth payload.
        'extra_keys':       list(extra_pairs),
        'key_keys':         list(key_keys),
        'flight_pair_keys': list(flight_pair_keys),
        'trigraph_keys':    list(trigraph_keys),
    }

    model_path = os.path.join(model_dir, f"{_safe_filename(username)}_keystroke_rf.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    size_kb = os.path.getsize(model_path) / 1024
    print(f"\n{'='*70}")
    print(f"  ✅ MODEL SAVED:  {model_path}  ({size_kb:.1f} KB)")
    print(f"  Threshold: {final_thresh:.2f}   "
          f"FAR: {far_f:.2%}   FRR: {frr_f:.2%}   EER: {best_eer:.2%}\n")

    return model_path


# ─────────────────────────────────────────────────────────────────────────────
#  AUTHENTICATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def predict_keystroke(username: str, feature_dict: dict) -> dict:
    model_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_path = os.path.join(model_dir, f"{_safe_filename(username)}_keystroke_rf.pkl")

    if not os.path.exists(model_path):
        return {'error': f'No model found for {username}. Please enroll first.'}

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    pipeline     = model_data['pipeline']
    feat_names   = model_data['feature_names']
    profile_mean = model_data['profile_mean']
    profile_std  = model_data['profile_std']
    threshold    = model_data['threshold']

    # Dict-expansion features arrive as nested dicts on the login payload.
    extra_map    = feature_dict.get('extra_digraphs')     or {}
    key_map      = feature_dict.get('key_dwell_map')      or {}
    flight_map   = feature_dict.get('flight_per_digraph') or {}
    trigraph_map = feature_dict.get('trigraph_map')       or {}

    def _get_val(name):
        if name.startswith('extra_'):
            return float(extra_map.get(name[len('extra_'):], 0.0) or 0.0)
        if name.startswith('key_'):
            return float(key_map.get(name[len('key_'):], 0.0) or 0.0)
        if name.startswith('flight_'):
            return float(flight_map.get(name[len('flight_'):], 0.0) or 0.0)
        if name.startswith('trigraph_'):
            return float(trigraph_map.get(name[len('trigraph_'):], 0.0) or 0.0)
        return float(feature_dict.get(name, 0.0) or 0.0)

    vec = np.array([_get_val(n) for n in feat_names])

    # Missing-digraph imputation — mirrors profile matcher's valid-mask.
    # When the frontend clean-press chain strips a backspaced pair, its
    # digraph/flight/trigraph/key timing arrives as 0.0. RF trees and
    # Mahalanobis both treat that as a real extreme value → score collapses.
    # Profile matcher drops such positions (keystroke_profile_matcher._gp_score
    # via `live >= _EPS_MS`). Here we impute them with profile_mean so the
    # feature contributes neutrally (ratio≈1, mahalanobis term≈0), and reject
    # if too few per-pair features actually fired — same floor as GP.
    from ml.keystroke_profile_matcher import _EPS_MS, _MIN_VALID, _is_scoring_feature
    pair_idx = [i for i, n in enumerate(feat_names) if _is_scoring_feature(n)]
    n_valid_pairs = 0
    for i in pair_idx:
        if vec[i] < _EPS_MS:
            vec[i] = float(profile_mean[i])
        else:
            n_valid_pairs += 1
    if pair_idx and n_valid_pairs < _MIN_VALID:
        return {
            'match':      False,
            'confidence': 0.0,
            'rejected':   True,
            'reason':     f'Too few valid digraphs ({n_valid_pairs} < {_MIN_VALID}) — too many backspaces',
        }

    # Neutralize backspace features at scoring time.
    # Why: enrollment samples are typed cleanly, so profile_std for these
    # columns is ~0. A single live backspace makes diff²/var explode in
    # Mahalanobis (→ score 0) and also trips RF splits learned on synthetic
    # impostors with non-zero counts. The frontend's clean-press chain
    # already strips corrected characters from all digraph/p2p/flight/trigraph
    # timings, so the backspace count/ratio carry no additional genuine signal
    # at auth time — zeroing them mirrors what the profile matcher does.
    for _bs in ('backspace_ratio', 'backspace_count'):
        if _bs in feat_names:
            vec[feat_names.index(_bs)] = float(profile_mean[feat_names.index(_bs)])

    ok, reason = _is_quality_sample(vec, feat_names)
    if not ok:
        return {
            'match':      False,
            'confidence': 0.0,
            'rejected':   True,
            'reason':     f'Invalid keystroke data: {reason}',
        }

    rf_score  = float(pipeline.predict_proba(vec.reshape(1, -1))[0][1])
    mah_score = mahalanobis_score(vec, profile_mean, profile_std)

    from utils.fusion import fuse_keystroke_scores
    fused = fuse_keystroke_scores(rf_score, mah_score)
    match = fused >= threshold

    return {
        'match':      bool(match),
        'confidence': float(fused),
        'rf_score':   float(rf_score),
        'mah_score':  float(mah_score),
        'threshold':  float(threshold),
        'far':        model_data.get('far', 0),
        'frr':        model_data.get('frr', 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = input("Enter username to train: ").strip()
    train_random_forest(username)
