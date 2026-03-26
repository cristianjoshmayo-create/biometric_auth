# ml/train_voice_cnn.py
# FIXED v3: resolves all authentication failures
#
# Bug fixes applied:
#
#  FIX 1 — CMVN profile collapse (primary voice auth failure cause)
#    CMVN subtracts the per-utterance mean, so mfcc_mean[0] stored in DB
#    is always ~0.00 for every user.  When profile_mean ≈ 0 the Mahalanobis
#    distance explodes and fused_score = 0.000.
#    Fix: store separate raw_profile_mean / raw_profile_std built from
#    features NOT affected by CMVN (mfcc_std, delta, prosodic).  These are
#    used exclusively by mahalanobis_score at inference time.
#
#  FIX 2 — Augmentation noise too aggressive (80% → 40% of within_std)
#    80% noise erases the user's identity in augmented samples — GBM trains
#    on noise rather than voice.  40% keeps samples speaker-representative.
#
#  FIX 3 — Threshold calibrated on wrong distribution
#    Threshold searched on raw GBM prob but decision uses 0.70*GBM + 0.30*Mah.
#    Fix: calibrate on fused_cv using the corrected Mah (requires FIX 1).
#
#  FIX 4 — Too few genuine augmentations (150 min → 300 min)
#    max(300, n_genuine * 100) ensures the genuine class is dense enough
#    for GBM to learn a meaningful boundary even at 3 enrollment samples.

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

from database.db import SessionLocal
from database.models import User, VoiceTemplate

N_FEATURES = 62


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_vector(template) -> np.ndarray:
    """62-element CMVN feature vector — used for GBM training."""
    mfcc_mean   = list(template.mfcc_features or [])
    mfcc_std    = list(template.mfcc_std      or [])
    delta_mean  = list(getattr(template, 'delta_mfcc_mean',  None) or [])
    delta2_mean = list(getattr(template, 'delta2_mfcc_mean', None) or [])

    while len(mfcc_mean)   < 13: mfcc_mean.append(0.0)
    while len(mfcc_std)    < 13: mfcc_std.append(0.0)
    while len(delta_mean)  < 13: delta_mean.append(0.0)
    while len(delta2_mean) < 13: delta2_mean.append(0.0)

    return np.array(
        mfcc_mean[:13] + mfcc_std[:13] +
        delta_mean[:13] + delta2_mean[:13] + [
            float(getattr(template, 'pitch_mean',             None) or 0),
            float(getattr(template, 'pitch_std',              None) or 0),
            float(getattr(template, 'speaking_rate',          None) or 0),
            float(getattr(template, 'energy_mean',            None) or 0),
            float(getattr(template, 'energy_std',             None) or 0),
            float(getattr(template, 'zcr_mean',               None) or 0),
            float(getattr(template, 'spectral_centroid_mean', None) or 0),
            float(getattr(template, 'spectral_rolloff_mean',  None) or 0),
            float(getattr(template, 'spectral_flux_mean',     None) or 0),
            float(getattr(template, 'voiced_fraction',        None) or 0),
        ],
        dtype=np.float64
    )


def extract_raw_profile_vector(template) -> np.ndarray:
    """
    FIX 1: 36-element raw (non-CMVN) vector used only for Mahalanobis profile.
    Slices: mfcc_std[0..12] + delta_mean[0..12] + 10 prosodic features.
    These features are NOT zeroed by CMVN so they retain genuine per-speaker
    variation that makes Mahalanobis distance meaningful.
    """
    mfcc_std    = list(template.mfcc_std      or [])
    delta_mean  = list(getattr(template, 'delta_mfcc_mean', None) or [])
    while len(mfcc_std)   < 13: mfcc_std.append(0.0)
    while len(delta_mean) < 13: delta_mean.append(0.0)

    return np.array(
        mfcc_std[:13] + delta_mean[:13] + [
            float(getattr(template, 'pitch_mean',             None) or 0),
            float(getattr(template, 'pitch_std',              None) or 0),
            float(getattr(template, 'speaking_rate',          None) or 0),
            float(getattr(template, 'energy_mean',            None) or 0),
            float(getattr(template, 'energy_std',             None) or 0),
            float(getattr(template, 'zcr_mean',               None) or 0),
            float(getattr(template, 'spectral_centroid_mean', None) or 0),
            float(getattr(template, 'spectral_rolloff_mean',  None) or 0),
            float(getattr(template, 'spectral_flux_mean',     None) or 0),
            float(getattr(template, 'voiced_fraction',        None) or 0),
        ],
        dtype=np.float64
    )


def _raw_vec_from_cmvn_array(cmvn_vec: np.ndarray) -> np.ndarray:
    """
    FIX 1 helper for CV loop: extract the raw-profile slice from a CMVN
    training matrix row.  Layout: mfcc_mean[0:13] | mfcc_std[13:26] |
    delta[26:39] | delta2[39:52] | prosodic[52:62].
    Raw profile uses mfcc_std[13:26] + delta[26:39] + prosodic[52:62].
    """
    return np.concatenate([cmvn_vec[13:26], cmvn_vec[26:39], cmvn_vec[52:62]])


def load_enrollment_samples(db, user_id: int):
    """Returns (cmvn_vectors, raw_profile_vectors)."""
    templates = (
        db.query(VoiceTemplate)
        .filter(VoiceTemplate.user_id == user_id)
        .order_by(VoiceTemplate.enrolled_at.asc())
        .all()
    )
    cmvn_vecs = []
    raw_vecs  = []
    for t in templates:
        vec = extract_feature_vector(t)
        if vec[0] == 0.0 and vec[13] == 0.0:
            print(f"  ⚠  Skipping template id={t.id} (zero MFCC)")
            continue
        cmvn_vecs.append(vec)
        raw_vecs.append(extract_raw_profile_vector(t))
    return cmvn_vecs, raw_vecs


def load_real_impostors(db, exclude_user_id: int) -> list:
    other_users = db.query(User).filter(User.id != exclude_user_id).all()
    impostors = []
    for u in other_users:
        vecs, _ = load_enrollment_samples(db, u.id)
        impostors.extend(vecs)
    return impostors


def generate_synthetic_impostors(profile_mean, profile_std, n=300, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    HUMAN_RANGES = {
        **{i: [(-600,-100),(-80,80),(-60,60),(-50,50),(-40,40),(-35,35),
               (-30,30),(-30,30),(-25,25),(-25,25),(-20,20),(-20,20),(-20,20)][i]
           for i in range(13)},
        **{i: (2, 30)  for i in range(13, 26)},
        **{i: (-15, 15) for i in range(26, 39)},
        **{i: (-10, 10) for i in range(39, 52)},
        52: (80, 320), 53: (5, 60),
        54: (1.0, 8.0),
        55: (0.005, 0.15), 56: (0.002, 0.08),
        57: (0.03, 0.35),
        58: (800, 4500), 59: (1500, 7000), 60: (5.0, 80.0),
        61: (0.3, 0.95),
    }
    samples = []
    for _ in range(n):
        vec = np.zeros(N_FEATURES)
        for i in range(N_FEATURES):
            lo, hi = HUMAN_RANGES.get(i, (profile_mean[i] * 0.3, profile_mean[i] * 1.7 + 1e-6))
            for _ in range(10):
                v = rng.uniform(lo, hi)
                if abs(v - profile_mean[i]) >= max(profile_std[i] * 0.5, 1e-6):
                    break
            vec[i] = v
        samples.append(vec)
    return samples


def generate_genuine_augmentations(genuine_vectors, n=300, rng_seed=42):
    """
    FIX 2: noise capped at 40% of within_std (was 80%).
    Keeps augmented samples recognisably similar to the genuine ones so the
    GBM learns actual speaker features, not noise.
    """
    rng  = np.random.default_rng(rng_seed)
    base = np.array(genuine_vectors)
    if base.shape[0] > 1:
        within_std = base.std(axis=0)
    else:
        within_std = np.abs(base[0]) * 0.08
    within_std = np.where(within_std < 0.3, 0.3, within_std)

    samples = []
    for _ in range(n):
        idx   = rng.integers(0, len(genuine_vectors))
        noisy = genuine_vectors[idx].copy()
        noise = rng.normal(0, within_std * 0.40)   # FIX 2: was 0.80
        noisy = noisy + noise
        # Clamp physically non-negative features
        noisy[55] = max(0.0, noisy[55])
        noisy[56] = max(0.0, noisy[56])
        noisy[57] = max(0.0, noisy[57])
        noisy[60] = max(0.0, noisy[60])
        noisy[61] = float(np.clip(noisy[61], 0.0, 1.0))
        samples.append(noisy)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
#  MAHALANOBIS  (operates on raw-profile vectors — FIX 1)
# ─────────────────────────────────────────────────────────────────────────────

def mahalanobis_score(vec, profile_mean, profile_std):
    var      = profile_std ** 2
    safe_var = np.where(var < 1e-10, 1e-10, var)
    diff     = vec - profile_mean
    d_sq     = float(np.sum(diff ** 2 / safe_var))
    d_sq_norm = d_sq / max(len(vec), 1)
    score    = 1.0 / (1.0 + np.exp(2.5 * (d_sq_norm - 1.0)))
    return float(np.clip(score, 0, 1))


def estimate_enrollment_noise(genuine_vectors):
    if len(genuine_vectors) < 2:
        return 0.3
    base = np.array(genuine_vectors)
    cv = np.abs(base.std(axis=0)) / (np.abs(base.mean(axis=0)) + 1e-9)
    return float(np.clip(np.mean(cv), 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        min_samples_split=6, min_samples_leaf=4,
        subsample=0.8, max_features='sqrt', random_state=42,
    )
    return Pipeline([("scaler", StandardScaler()), ("gbm", clf)])


def _safe_filename(username: str) -> str:
    return username.replace("@", "_at_").replace(".", "_").replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_voice_model(username: str):
    # Phase 1: load all DB data then close connection
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"❌ User '{username}' not found!")
            return None

        genuine_cmvn, genuine_raw = load_enrollment_samples(db, user.id)
        if not genuine_cmvn:
            print("❌ No valid voice enrollment samples.")
            return None

        real_impostors = load_real_impostors(db, user.id)

    finally:
        try:
            db.close()
        except Exception:
            pass

    # Phase 2: all training in memory
    print(f"\n{'='*70}")
    print(f"  VOICE MODEL FIXED v3  —  user: {username}  ({N_FEATURES} features)")
    print(f"{'='*70}")
    print(f"  Enrollment samples: {len(genuine_cmvn)}")
    if len(genuine_cmvn) < 3:
        print("  ⚠  Fewer than 3 samples — accuracy will be lower.")

    # CMVN profile (for pipeline scaler reference only)
    cmvn_arr     = np.array(genuine_cmvn)
    profile_mean = cmvn_arr.mean(axis=0)
    profile_std  = (cmvn_arr.std(axis=0) if len(genuine_cmvn) > 1
                    else np.abs(profile_mean) * 0.10)

    # FIX 1: raw profile for Mahalanobis — non-CMVN features only
    raw_arr          = np.array(genuine_raw)
    raw_profile_mean = raw_arr.mean(axis=0)
    raw_profile_std  = (raw_arr.std(axis=0) if len(genuine_raw) > 1
                        else np.abs(raw_profile_mean) * 0.10 + 1e-6)
    raw_profile_std  = np.where(raw_profile_std < 1e-6, 1e-6, raw_profile_std)

    noise_level = estimate_enrollment_noise(genuine_cmvn)
    print(f"  Enrollment noise level: {noise_level:.2f} "
          f"({'quiet' if noise_level < 0.2 else 'moderate' if noise_level < 0.4 else 'noisy'})")

    # FIX 4: minimum 300 genuine augmentations (was 150)
    n_aug = max(300, len(genuine_cmvn) * 100)
    genuine_aug = generate_genuine_augmentations(genuine_cmvn, n=n_aug)

    if len(real_impostors) >= 20:
        print(f"  Using {len(real_impostors)} real impostor samples.")
        impostor_samples = list(real_impostors)
        while len(impostor_samples) < len(genuine_aug) * 2:
            impostor_samples += generate_synthetic_impostors(profile_mean, profile_std, n=100)
    else:
        impostor_samples = real_impostors + generate_synthetic_impostors(
            profile_mean, profile_std,
            n=max(300, len(genuine_aug) * 2)
        )

    X = np.vstack([genuine_aug, impostor_samples])
    y = np.array([1] * len(genuine_aug) + [0] * len(impostor_samples))

    print(f"  Training data: {len(genuine_aug)} genuine / {len(impostor_samples)} impostor")

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import confusion_matrix, accuracy_score

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("  Running 5-fold cross-validation …")
    y_prob_cv = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    # FIX 3: calibrate threshold on FUSED score using corrected raw Mah
    fused_cv = np.array([
        0.70 * p + 0.30 * mahalanobis_score(
            _raw_vec_from_cmvn_array(X[i]),
            raw_profile_mean, raw_profile_std
        )
        for i, p in enumerate(y_prob_cv)
    ])

    best_thresh, best_eer = 0.50, 1.0
    print(f"\n  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'EER':>8}  (on fused score)")
    for t in np.arange(0.30, 0.90, 0.02):
        y_t = (fused_cv >= t).astype(int)
        if len(np.unique(y_t)) < 2: continue
        cm = confusion_matrix(y, y_t)
        if cm.shape != (2, 2): continue
        tn, fp, fn, tp = cm.ravel()
        far_t = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr_t = fn / (fn + tp) if (fn + tp) > 0 else 0
        eer_t = (far_t + frr_t) / 2
        marker = " ◄" if eer_t < best_eer else ""
        print(f"  {t:>10.2f}  {far_t:>8.2%}  {frr_t:>8.2%}  {eer_t:>8.2%}{marker}")
        if eer_t < best_eer:
            best_eer, best_thresh = eer_t, float(t)

    noise_adj    = noise_level * 0.06
    final_thresh = min(best_thresh + noise_adj, 0.72)
    if noise_adj > 0.01:
        print(f"\n  ⚠  Noise-adaptive: {best_thresh:.2f} → {final_thresh:.2f} (+{noise_adj:.2f})")

    y_final = (fused_cv >= final_thresh).astype(int)
    cm = confusion_matrix(y, y_final)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  Final @ {final_thresh:.2f}:  FAR={far:.2%}  FRR={frr:.2%}  "
          f"EER={best_eer:.2%}  ACC={accuracy_score(y, y_final):.2%}")

    print("\n  Training final model …")
    pipeline.fit(X, y)

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_data = {
        'pipeline':         pipeline,
        'n_features':       N_FEATURES,
        'username':         username,
        'user_id':          user.id,
        'n_enrollment':     len(genuine_cmvn),
        'profile_mean':     profile_mean,       # CMVN (pipeline scaler ref)
        'profile_std':      profile_std,
        'profile_var':      profile_std ** 2,
        # FIX 1: raw (non-CMVN) profile for Mahalanobis at inference
        'raw_profile_mean': raw_profile_mean,
        'raw_profile_std':  raw_profile_std,
        'threshold':        final_thresh,
        'noise_level':      noise_level,
        'far':              float(far),
        'frr':              float(frr),
        'eer':              float(best_eer),
        'model_type':       'gbm_fixed_v3',
    }

    model_path = os.path.join(model_dir, f"{_safe_filename(username)}_voice_cnn.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n  ✅ Model saved → {model_path}  ({os.path.getsize(model_path)/1024:.1f} KB)")
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_voice(username: str, feature_dict: dict) -> dict:
    model_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_path = os.path.join(model_dir, f"{_safe_filename(username)}_voice_cnn.pkl")

    if not os.path.exists(model_path):
        return {'error': f'No voice model for "{username}". Enroll first.'}

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    pipeline     = model_data['pipeline']
    threshold    = model_data['threshold']
    model_n_feat = model_data.get('n_features', 34)

    mfcc_mean   = list(feature_dict.get('mfcc_features',    [0]*13))
    mfcc_std    = list(feature_dict.get('mfcc_std',         [0]*13))
    delta_mean  = list(feature_dict.get('delta_mfcc_mean',  [0]*13))
    delta2_mean = list(feature_dict.get('delta2_mfcc_mean', [0]*13))
    while len(mfcc_mean)   < 13: mfcc_mean.append(0.0)
    while len(mfcc_std)    < 13: mfcc_std.append(0.0)
    while len(delta_mean)  < 13: delta_mean.append(0.0)
    while len(delta2_mean) < 13: delta2_mean.append(0.0)

    pitch_mean   = float(feature_dict.get('pitch_mean',             0))
    pitch_std    = float(feature_dict.get('pitch_std',              0))
    speak_rate   = float(feature_dict.get('speaking_rate',          0))
    energy_mean  = float(feature_dict.get('energy_mean',            0))
    energy_std   = float(feature_dict.get('energy_std',             0))
    zcr          = float(feature_dict.get('zcr_mean',               0))
    centroid     = float(feature_dict.get('spectral_centroid_mean', 0))
    rolloff      = float(feature_dict.get('spectral_rolloff_mean',  0))
    flux         = float(feature_dict.get('spectral_flux_mean',     0))
    voiced_frac  = float(feature_dict.get('voiced_fraction',        0))

    if model_n_feat == 62:
        cmvn_list = (
            mfcc_mean[:13] + mfcc_std[:13] +
            delta_mean[:13] + delta2_mean[:13] + [
                pitch_mean, pitch_std, speak_rate,
                energy_mean, energy_std, zcr,
                centroid, rolloff, flux, voiced_frac,
            ]
        )
    else:
        # Legacy 34-feature model
        cmvn_list = (
            mfcc_mean[:13] + mfcc_std[:13] + [
                pitch_mean, pitch_std, speak_rate,
                energy_mean, energy_std, zcr,
                centroid, rolloff,
            ]
        )

    cmvn_vec = np.array(cmvn_list, dtype=np.float64).reshape(1, -1)
    n = model_data['profile_mean'].shape[0]
    if cmvn_vec.shape[1] < n:
        cmvn_vec = np.hstack([cmvn_vec, np.zeros((1, n - cmvn_vec.shape[1]))])
    elif cmvn_vec.shape[1] > n:
        cmvn_vec = cmvn_vec[:, :n]

    model_prob = float(pipeline.predict_proba(cmvn_vec)[0][1])

    # FIX 1: use raw profile for Mahalanobis if model supports it
    if 'raw_profile_mean' in model_data:
        raw_inf_vec = np.array(
            mfcc_std[:13] + delta_mean[:13] + [
                pitch_mean, pitch_std, speak_rate,
                energy_mean, energy_std, zcr,
                centroid, rolloff, flux, voiced_frac,
            ],
            dtype=np.float64
        )
        mah_score = mahalanobis_score(
            raw_inf_vec,
            model_data['raw_profile_mean'],
            model_data['raw_profile_std']
        )
    else:
        # Legacy fallback — old model without raw profile
        profile_var = model_data.get('profile_var', model_data['profile_std'] ** 2)
        mah_score   = mahalanobis_score(
            cmvn_vec[0],
            model_data['profile_mean'],
            np.sqrt(profile_var)
        )

    fused = 0.70 * model_prob + 0.30 * mah_score
    match = fused >= threshold

    print(f"\n  Voice auth '{username}': model={model_prob:.4f} mah={mah_score:.4f} "
          f"fused={fused:.4f} thresh={threshold:.2f} → {'✅ MATCH' if match else '❌ REJECT'}")

    return {
        'match':       bool(match),
        'confidence':  round(model_prob * 100, 2),
        'mahalanobis': round(mah_score * 100, 2),
        'fused_score': round(fused * 100, 2),
        'threshold':   round(threshold * 100, 2),
        'noise_level': round(model_data.get('noise_level', 0) * 100, 1),
        'far':         round(model_data.get('far', 0) * 100, 2),
        'frr':         round(model_data.get('frr', 0) * 100, 2),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("username", nargs="?", default=None)
    parser.add_argument("--lock", default=None, help="Lock file to delete on completion")
    args = parser.parse_args()

    username = args.username or input("Username: ").strip()
    try:
        train_voice_model(username)
    finally:
        if args.lock and os.path.exists(args.lock):
            try:
                os.remove(args.lock)
            except Exception:
                pass