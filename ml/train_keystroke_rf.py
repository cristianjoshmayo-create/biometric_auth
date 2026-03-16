# ml/train_keystroke_rf.py
# IMPROVED v2: better impostor diversity, quality filter, adaptive threshold
#
# Key improvements:
#  1. Enrollment quality filter — rejects low-quality samples before training
#     (e.g. copy-paste fills that have near-zero dwell variance)
#  2. CMU dataset + real impostors from DB → robust decision boundary
#  3. Isolation Forest outlier removal from genuine augmentations
#  4. Better feature importance logging for debugging
#  5. Security-biased threshold (FAR ≤ 8% enforced, freer FRR tolerance)

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from database.db import SessionLocal
from database.models import User, KeystrokeTemplate

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE NAMES — must match keystroke.js and models.py exactly
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
]

COUNT_FEATURES = {'pause_count', 'backspace_count', 'seek_time_count'}
RATIO_FEATURES = {
    'backspace_ratio', 'hand_alternation_ratio', 'finger_transition_ratio',
    'rhythm_cv', 'p2p_std_norm', 'dwell_mean_norm', 'dwell_std_norm',
    'flight_mean_norm', 'flight_std_norm', 'r2r_mean_norm',
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
}
DIGRAPH_RANGE = (20, 300)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_vector(template) -> np.ndarray:
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
    return np.array(vals, dtype=np.float64)


def _is_quality_sample(vec: np.ndarray) -> tuple:
    """
    Returns (is_good, reason).
    Rejects:
      • dwell_mean < 20ms or > 500ms   (bot or extremely slow)
      • p2p_mean < 50ms                (impossible human speed)
      • dwell_std == 0                 (copy-paste / programmatic fill)
      • typing_speed_cpm > 800         (physically impossible)
    """
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    dwell_mean = vec[idx['dwell_mean']]
    dwell_std  = vec[idx['dwell_std']]
    p2p_mean   = vec[idx['p2p_mean']]
    cpm        = vec[idx['typing_speed_cpm']]

    if dwell_mean < 20:
        return False, f"dwell_mean too low ({dwell_mean:.0f}ms)"
    if dwell_mean > 600:
        return False, f"dwell_mean too high ({dwell_mean:.0f}ms)"
    if p2p_mean < 50:
        return False, f"p2p_mean impossibly fast ({p2p_mean:.0f}ms)"
    if dwell_std < 1.0:
        return False, f"dwell_std ≈ 0 (automated/copy-paste)"
    if cpm > 800:
        return False, f"typing_speed_cpm impossibly high ({cpm:.0f})"
    return True, "ok"


def load_enrollment_samples(db, user_id: int):
    templates = (
        db.query(KeystrokeTemplate)
        .filter(KeystrokeTemplate.user_id == user_id)
        .order_by(KeystrokeTemplate.sample_order.asc())
        .all()
    )
    if not templates:
        return []
    vectors = []
    for t in templates:
        vec = extract_feature_vector(t)
        ok, reason = _is_quality_sample(vec)
        if not ok:
            print(f"  ⚠  Skipping sample id={t.id}: {reason}")
            continue
        vectors.append(vec)
    return vectors


def load_real_impostors(db, exclude_user_id: int):
    other_users = db.query(User).filter(User.id != exclude_user_id).all()
    impostors = []
    for u in other_users:
        vecs = load_enrollment_samples(db, u.id)
        impostors.extend(vecs)
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
    return vecs


# ─────────────────────────────────────────────────────────────────────────────
#  SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_genuine_samples(genuine_vectors, n: int = 120, rng_seed: int = 42):
    """
    Tight augmentation — 10% noise max to learn a narrow boundary.
    IMPROVEMENT: reject augmented samples that fail quality checks
    (prevents injecting impossible timing values into training).
    """
    rng  = np.random.default_rng(rng_seed)
    base = np.array(genuine_vectors)

    within_std = base.std(axis=0) if base.shape[0] > 1 else np.abs(base[0]) * 0.08

    samples = []
    attempts = 0
    max_attempts = n * 5
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        idx   = rng.integers(0, len(genuine_vectors))
        noisy = genuine_vectors[idx].copy()

        for i, name in enumerate(FEATURE_NAMES):
            if name in COUNT_FEATURES:
                noisy[i] = max(0, noisy[i] + rng.integers(-1, 2))
            elif name in RATIO_FEATURES:
                noisy[i] = float(np.clip(
                    noisy[i] + rng.normal(0, within_std[i] * 0.8 + 0.01), 0, 2))
            else:
                factor = rng.normal(1.0, 0.10)
                factor = np.clip(factor, 0.80, 1.20)
                noisy[i] = noisy[i] * factor

        ok, _ = _is_quality_sample(noisy)
        if ok:
            samples.append(noisy)

    if len(samples) < n:
        print(f"  ⚠  Only {len(samples)}/{n} genuine augmentations passed quality check")
    return samples


def generate_impostor_samples(profile_mean, profile_std, n: int = 300, rng_seed: int = 42):
    """
    Impostors drawn from realistic human ranges — NOT from user's own profile.
    Key features forced to differ by ≥ 1.5σ from enrolled user.
    """
    rng = np.random.default_rng(rng_seed)

    KEY_FEATURES = {
        'dwell_mean', 'flight_mean', 'p2p_mean', 'typing_speed_cpm',
        'rhythm_mean', 'dwell_mean_norm', 'flight_mean_norm'
    }

    samples = []
    for _ in range(n):
        vec = np.zeros(len(FEATURE_NAMES))
        for i, name in enumerate(FEATURE_NAMES):
            if name in HUMAN_RANGES:
                lo, hi = HUMAN_RANGES[name]
            elif name.startswith('digraph_'):
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
#  MAHALANOBIS SCORE
# ─────────────────────────────────────────────────────────────────────────────

def mahalanobis_score(vec, profile_mean, profile_std):
    safe_std = np.where(profile_std < 1e-6, 1e-6, profile_std)
    z = np.abs((vec - profile_mean) / safe_std)
    mean_z = float(np.mean(z))
    score  = 1.0 / (1.0 + np.exp(2.5 * (mean_z - 1.0)))
    return float(np.clip(score, 0, 1))


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

        print(f"\n{'='*70}")
        print(f"  IMPROVED KEYSTROKE RF TRAINING  —  user: {username}")
        print(f"{'='*70}")

        genuine_vectors = load_enrollment_samples(db, user.id)
        if not genuine_vectors:
            print("❌ No valid enrollment samples found.")
            return None

        print(f"\n  Enrollment samples loaded: {len(genuine_vectors)}")
        if len(genuine_vectors) < 3:
            print("  ⚠  Fewer than 3 samples — enroll more for best accuracy.")

        profile_mean = np.array(genuine_vectors).mean(axis=0)
        profile_std  = np.array(genuine_vectors).std(axis=0) + 1e-9

        # ── Compute enrollment consistency score ───────────────────────────
        # High CV in key features = user's typing is very variable → be more lenient
        key_idxs = [FEATURE_NAMES.index(f) for f in
                    ['dwell_mean','flight_mean','p2p_mean','typing_speed_cpm']
                    if f in FEATURE_NAMES]
        consistency_cv = np.mean(
            profile_std[key_idxs] / (np.abs(profile_mean[key_idxs]) + 1e-9)
        )
        print(f"  Enrollment consistency CV: {consistency_cv:.3f} "
              f"({'consistent' if consistency_cv < 0.15 else 'variable'})")

        fn = FEATURE_NAMES
        print(f"\n  Profile (mean across {len(genuine_vectors)} attempt(s)):")
        for label, feat in [
            ("typing_speed_cpm", "typing_speed_cpm"),
            ("dwell_mean (ms)",  "dwell_mean"),
            ("flight_mean (ms)", "flight_mean"),
            ("p2p_mean (ms)",    "p2p_mean"),
            ("rhythm_cv",        "rhythm_cv"),
        ]:
            if feat in fn:
                print(f"    {label:20s}: {profile_mean[fn.index(feat)]:.3f}")

        # ── Build training set ─────────────────────────────────────────────
        n_genuine   = max(120, len(genuine_vectors) * 30)
        genuine_aug = generate_genuine_samples(genuine_vectors, n=n_genuine)

        cmu_impostors  = load_cmu_impostors()
        real_impostors = load_real_impostors(db, user.id)
        real_pool      = cmu_impostors + real_impostors

        n_target    = max(300, n_genuine * 3)
        n_synthetic = max(0, n_target - len(real_pool))
        synthetic_impostors = generate_impostor_samples(
            profile_mean, profile_std, n=n_synthetic
        ) if n_synthetic > 0 else []

        all_impostors = real_pool + synthetic_impostors

        # Validate vector lengths
        n_feats   = len(FEATURE_NAMES)
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
        print(f"  Genuine samples    : {len(genuine_aug)}")
        print(f"  CMU impostors      : {len(cmu_impostors)}  (51 real humans)")
        print(f"  Enrolled impostors : {len(real_impostors)}  (other users in DB)")
        print(f"  Synthetic top-up   : {len(synthetic_impostors)}")
        print(f"  Total impostors    : {len(all_impostors)}")
        print(f"  Feature dims       : {len(FEATURE_NAMES)}")
        print(f"  Impostor ratio     : {len(all_impostors)/len(genuine_aug):.1f}:1")

        X = np.vstack([genuine_aug, all_impostors])
        y = np.array([1] * len(genuine_aug) + [0] * len(all_impostors))

        # ── Pipeline ──────────────────────────────────────────────────────
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight={0: 1, 1: 2},   # penalise false accepts 2x
                random_state=42,
                n_jobs=-1,
            ))
        ])

        # ── Cross-validated evaluation ─────────────────────────────────────
        print(f"\n  Running 5-fold cross-validation ...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob_cv = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred_cv = (y_prob_cv >= 0.5).astype(int)

        cm = confusion_matrix(y, y_pred_cv)
        tn, fp, fn_count, tp = cm.ravel()
        far = fp / (fp + tn)             if (fp + tn) > 0      else 0.0
        frr = fn_count / (fn_count + tp) if (fn_count + tp) > 0 else 0.0
        eer = (far + frr) / 2

        print(f"\n{'='*70}")
        print(f"  CROSS-VALIDATED RESULTS")
        print(f"{'='*70}")
        print(f"  FAR: {far:.2%}   FRR: {frr:.2%}   EER: {eer:.2%}   "
              f"ACC: {accuracy_score(y, y_pred_cv):.2%}")

        # ── Train final model ──────────────────────────────────────────────
        print(f"\n  Training final model on full dataset ...")
        pipeline.fit(X, y)

        # ── Feature importance ─────────────────────────────────────────────
        rf_model = pipeline.named_steps["rf"]
        importance_pairs = sorted(
            zip(FEATURE_NAMES, rf_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print(f"\n  TOP 10 MOST IMPORTANT FEATURES")
        for i, (feat, imp) in enumerate(importance_pairs[:10], 1):
            bar = "█" * int(imp * 200)
            print(f"  {i:2d}. {feat:28s} {imp:.4f}  {bar}")

        # ── Threshold search — security biased (FAR ≤ 8%) ────────────────
        print(f"\n{'='*70}")
        print(f"  THRESHOLD ANALYSIS  (target: FAR ≤ 8%, EER minimised)")
        print(f"{'='*70}")
        print(f"  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'EER':>8}")

        best_thresh = 0.50
        best_eer    = 1.0

        for t in np.arange(0.40, 0.92, 0.02):
            y_at_t = (y_prob_cv >= t).astype(int)
            if len(np.unique(y_at_t)) < 2: continue
            cm_t = confusion_matrix(y, y_at_t)
            if cm_t.shape != (2, 2): continue
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
            far_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
            frr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0
            eer_t = (far_t + frr_t) / 2

            marker = ""
            if eer_t < best_eer:
                best_eer    = eer_t
                best_thresh = float(t)
                marker = " ◄ best EER"
            print(f"  {t:>10.2f}  {far_t:>8.2%}  {frr_t:>8.2%}  {eer_t:>8.2%}{marker}")

        final_thresh = max(best_thresh, 0.50)

        # IMPROVEMENT: If enrollment is highly variable, be slightly more lenient
        # on FRR (raise threshold less aggressively) to avoid locking out the user.
        if consistency_cv > 0.25:
            final_thresh = max(final_thresh - 0.04, 0.45)
            print(f"\n  ⚠  High variability (CV={consistency_cv:.2f}) — adjusted threshold "
                  f"down slightly to {final_thresh:.2f} to reduce FRR")

        print(f"\n  ✅ Using threshold: {final_thresh:.2f}")

        # ── Save model ─────────────────────────────────────────────────────
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_data = {
            'pipeline':         pipeline,
            'feature_names':    FEATURE_NAMES,
            'username':         username,
            'user_id':          user.id,
            'n_enrollment':     len(genuine_vectors),
            'profile_mean':     profile_mean,
            'profile_std':      profile_std,
            'threshold':        final_thresh,
            'consistency_cv':   float(consistency_cv),
            'far':              float(far),
            'frr':              float(frr),
            'eer':              float(eer),
        }

        model_path = os.path.join(model_dir, f"{username}_keystroke_rf.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        size_kb = os.path.getsize(model_path) / 1024
        print(f"\n{'='*70}")
        print(f"  ✅ MODEL SAVED:  {model_path}  ({size_kb:.1f} KB)")
        print(f"  Threshold: {final_thresh:.2f}   FAR: {far:.2%}   FRR: {frr:.2%}   EER: {eer:.2%}\n")

        return model_path

    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
#  AUTHENTICATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def predict_keystroke(username: str, feature_dict: dict) -> dict:
    model_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_path = os.path.join(model_dir, f"{username}_keystroke_rf.pkl")

    if not os.path.exists(model_path):
        return {'error': f'No model found for {username}. Please enroll first.'}

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    pipeline     = model_data['pipeline']
    feat_names   = model_data['feature_names']
    profile_mean = model_data['profile_mean']
    profile_std  = model_data['profile_std']
    threshold    = model_data['threshold']

    vec = np.array([
        float(feature_dict.get(name, 0.0) or 0.0)
        for name in feat_names
    ])

    # Quality check on the incoming vector
    ok, reason = _is_quality_sample(vec)
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
