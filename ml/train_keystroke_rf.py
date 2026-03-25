# ml/train_keystroke_rf.py
# v4 — unique-phrase-per-user support
#
# Key changes vs v3:
#  1. get_active_digraphs() — only digraphs present in the user's actual
#     phrase are kept; inactive digraphs are dropped before training so
#     the model never learns "zero digraph = genuine".
#  2. load_real_impostors() now accepts n_genuine and skips real enrolled
#     impostors when the genuine set is too small (< 8 samples), preventing
#     the model boundary from being contaminated when only a few users exist.
#  3. Fallback in auth is now a hard reject (no loose dwell-time comparison).
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


# ─────────────────────────────────────────────────────────────────────────────
#  DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_vector(template, extra_keys: list = None) -> np.ndarray:
    """
    Build the feature vector for a template.
    If extra_keys is provided (a list of digraph pair strings like ['pe','ea',...]),
    those timings are appended from template.extra_digraphs after the standard
    FEATURE_NAMES features.  This is the dynamic-digraph path for users enrolled
    with unique passphrases.
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

    if extra_keys:
        extra_map = getattr(template, 'extra_digraphs', None) or {}
        for pair in extra_keys:
            try:
                vals.append(float(extra_map.get(pair, 0.0) or 0.0))
            except (TypeError, ValueError):
                vals.append(0.0)

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


def load_enrollment_samples(db, user_id: int, extra_keys: list = None):
    templates = (
        db.query(KeystrokeTemplate)
        .filter(KeystrokeTemplate.user_id == user_id)
        .order_by(KeystrokeTemplate.sample_order.asc())
        .all()
    )
    vectors = []
    for t in templates:
        vec = extract_feature_vector(t, extra_keys=extra_keys)
        ok, reason = _is_quality_sample(vec)
        if not ok:
            print(f"  ⚠  Skipping sample id={t.id}: {reason}")
            continue
        vectors.append(vec)
    return vectors


def load_real_impostors(db, exclude_user_id: int, n_genuine: int = 99):
    if n_genuine < 8:
        print(f"  Skipping real enrolled impostors (n_genuine={n_genuine} < 8) — using CMU + synthetic only")
        return []
    other_users = db.query(User).filter(User.id != exclude_user_id).all()
    impostors   = []
    for u in other_users:
        impostors.extend(load_enrollment_samples(db, u.id))
    if impostors:
        print(f"  Real impostor samples from {len(other_users)} other user(s): {len(impostors)}")
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
        within_std = np.abs(base[0]) * 0.07

    mean_vals  = np.abs(base.mean(axis=0))
    within_std = np.maximum(within_std, mean_vals * 0.05)

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
                    noisy[i] + rng.normal(0, within_std[i] * 0.7 + 0.01), 0, 2))
            else:
                factor  = rng.normal(1.0, 0.07)
                factor  = np.clip(factor, 0.84, 1.16)
                noisy[i] = noisy[i] * factor

        ok, _ = _is_quality_sample(noisy, feat_names)
        if ok:
            samples.append(noisy)

    if len(samples) < n:
        print(f"  ⚠  Only {len(samples)}/{n} genuine augmentations passed quality check")
    return samples


def generate_impostor_samples(profile_mean, profile_std, n: int = 1200, rng_seed: int = 42, feat_names: list = None):
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
        'rhythm_mean', 'dwell_mean_norm', 'flight_mean_norm', 'shift_lag_norm'
    }

    samples = []
    for _ in range(n):
        vec = np.zeros(len(feat_names))
        for i, name in enumerate(feat_names):
            if name in HUMAN_RANGES:
                lo, hi = HUMAN_RANGES[name]
            elif name.startswith('digraph_') or name.startswith('extra_'):
                lo, hi = DIGRAPH_RANGE
            else:
                lo = profile_mean[i] * 0.3
                hi = profile_mean[i] * 2.0

            for _ in range(15):
                v = rng.uniform(lo, hi)
                if name in KEY_FEATURES:
                    min_dist = max(profile_std[i] * 1.5, profile_mean[i] * 0.20)
                    if abs(v - profile_mean[i]) >= min_dist:
                        break
                else:
                    break
            vec[i] = v
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
    score    = 1.0 / (1.0 + np.exp(2.5 * (d_sq_norm - 1.0)))
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
            class_weight={0: 1, 1: 2},
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
        user_phrase    = user.phrase or ""

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
    standard_active, extra_pairs = get_active_digraphs(user_phrase)
    all_digraph_feats = {f for f in FEATURE_NAMES if f.startswith("digraph_")}
    inactive_digraphs = all_digraph_feats - standard_active
    drop_indices      = [i for i, n in enumerate(FEATURE_NAMES) if n in inactive_digraphs]
    active_feat_names = (
        [n for n in FEATURE_NAMES if n not in inactive_digraphs]
        + [f"extra_{p}" for p in extra_pairs]
    )

    # Reload genuine samples WITH extra_keys so extra_digraphs are appended
    db2 = SessionLocal()
    try:
        genuine_vectors = load_enrollment_samples(db2, user_id, extra_keys=extra_pairs)
        if not genuine_vectors:
            print("❌ No valid enrollment samples found.")
            return None
    finally:
        try:
            db2.close()
        except Exception:
            pass

    n_genuine_real = len(genuine_vectors)
    print(f"\n  Enrollment samples loaded: {n_genuine_real}")

    def _strip_inactive(vecs):
        """Strip inactive hardcoded digraphs; pad zeros for extra_pairs on impostor vecs."""
        stripped = []
        for v in vecs:
            base = np.delete(v[:len(FEATURE_NAMES)], drop_indices)
            if len(v) > len(FEATURE_NAMES):
                extra = v[len(FEATURE_NAMES):]
            else:
                extra = np.zeros(len(extra_pairs))
            stripped.append(np.concatenate([base, extra]))
        return stripped

    genuine_vectors = _strip_inactive(genuine_vectors)

    print(f"  User phrase      : '{user_phrase}'")
    print(f"  Standard active  : {len(standard_active)}  ({', '.join(sorted(standard_active)) or 'none'})")
    print(f"  Extra pairs      : {len(extra_pairs)}  ({', '.join(extra_pairs) or 'none'})")
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

    n_aug     = max(600, n_genuine_real * 120)
    n_imp_syn = max(1200, n_aug * 2)

    genuine_aug = generate_genuine_samples(genuine_vectors, n=n_aug, feat_names=active_feat_names)

    cmu_impostors  = _strip_inactive(cmu_impostors)
    real_impostors = _strip_inactive(real_impostors)
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
    print(f"  CMU impostors      : {len(cmu_impostors)}  (51 real humans)")
    print(f"  Enrolled impostors : {len(real_impostors)}  (other users in DB)")
    print(f"  Synthetic impostors: {len(syn_impostors)}")
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
    print(f"  THRESHOLD SEARCH")
    print(f"{'='*70}")
    print(f"  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'EER':>8}")

    best_thresh = 0.50
    best_eer    = 1.0

    for t in np.arange(0.35, 0.92, 0.02):
        y_at_t = (y_prob_cv >= t).astype(int)
        if len(np.unique(y_at_t)) < 2:
            continue
        cm_t = confusion_matrix(y, y_at_t)
        if cm_t.shape != (2, 2):
            continue
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        far_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
        frr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0
        eer_t = (far_t + frr_t) / 2

        marker = ""
        if eer_t < best_eer:
            best_eer    = eer_t
            best_thresh = float(t)
            marker      = " ◄ best EER"
        print(f"  {t:>10.2f}  {far_t:>8.2%}  {frr_t:>8.2%}  {eer_t:>8.2%}{marker}")

    final_thresh = max(best_thresh, 0.45)

    if consistency_cv > 0.25:
        final_thresh = max(final_thresh - 0.04, 0.42)
        print(f"\n  ⚠  High variability (CV={consistency_cv:.2f}) → "
              f"threshold adjusted to {final_thresh:.2f}")

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
        'pipeline':       pipeline,
        'feature_names':  active_feat_names,
        'username':       username,
        'user_id':        user_id,
        'n_enrollment':   n_genuine_real,
        'profile_mean':   profile_mean,
        'profile_std':    profile_std,
        'profile_var':    profile_var,
        'threshold':      final_thresh,
        'consistency_cv': float(consistency_cv),
        'far':            float(far_f),
        'frr':            float(frr_f),
        'eer':            float(best_eer),
        'phrase':         user_phrase,
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

    # extra_digraphs from the login payload (dict of pair → mean_ms)
    extra_map = feature_dict.get('extra_digraphs') or {}

    def _get_val(name):
        if name.startswith('extra_'):
            pair = name[len('extra_'):]
            return float(extra_map.get(pair, 0.0) or 0.0)
        return float(feature_dict.get(name, 0.0) or 0.0)

    vec = np.array([_get_val(n) for n in feat_names])

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
    fused     = 0.75 * rf_score + 0.25 * mah_score
    match     = fused >= threshold

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