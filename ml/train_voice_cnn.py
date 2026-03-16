# ml/train_voice_cnn.py
# IMPROVED v2: noise-robust voice model
# Changes:
#  1. 62-feature vector: CMVN MFCCs + delta + delta2 + spectral flux + voiced_fraction
#  2. GradientBoosting replaces MLP (better on small enrollment sets)
#  3. Noise-adaptive threshold tightening
#  4. Backward-compatible with old 34-feature models

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

N_FEATURES = 62  # increased from 34


def extract_feature_vector(template) -> np.ndarray:
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


def load_enrollment_samples(db, user_id: int) -> list:
    templates = (
        db.query(VoiceTemplate)
        .filter(VoiceTemplate.user_id == user_id)
        .order_by(VoiceTemplate.enrolled_at.asc())
        .all()
    )
    vectors = []
    for t in templates:
        vec = extract_feature_vector(t)
        if vec[0] == 0.0 and vec[13] == 0.0:
            print(f"  ⚠  Skipping template id={t.id} (zero MFCC)")
            continue
        vectors.append(vec)
    return vectors


def load_real_impostors(db, exclude_user_id: int) -> list:
    other_users = db.query(User).filter(User.id != exclude_user_id).all()
    impostors = []
    for u in other_users:
        vecs = load_enrollment_samples(db, u.id)
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


def generate_genuine_augmentations(genuine_vectors, n=150, rng_seed=42):
    rng  = np.random.default_rng(rng_seed)
    base = np.array(genuine_vectors)
    within_std = base.std(axis=0) if base.shape[0] > 1 else np.abs(base[0]) * 0.12
    within_std = np.where(within_std < 0.5, 0.5, within_std)
    samples = []
    for _ in range(n):
        idx   = rng.integers(0, len(genuine_vectors))
        noisy = genuine_vectors[idx].copy()
        noise = rng.normal(0, within_std * 0.8)
        noisy = noisy + noise
        noisy[55] = max(0.0, noisy[55])
        noisy[56] = max(0.0, noisy[56])
        noisy[57] = max(0.0, noisy[57])
        noisy[60] = max(0.0, noisy[60])
        noisy[61] = float(np.clip(noisy[61], 0.0, 1.0))
        samples.append(noisy)
    return samples


def mahalanobis_score(vec, profile_mean, profile_std):
    safe_std = np.where(profile_std < 1e-6, 1e-6, profile_std)
    z = np.abs((vec - profile_mean) / safe_std)
    mean_z = float(np.mean(z))
    score = 1.0 / (1.0 + np.exp(2.5 * (mean_z - 1.0)))
    return float(np.clip(score, 0, 1))


def estimate_enrollment_noise(genuine_vectors):
    if len(genuine_vectors) < 2:
        return 0.3
    base = np.array(genuine_vectors)
    cv = np.abs(base.std(axis=0)) / (np.abs(base.mean(axis=0)) + 1e-9)
    return float(np.clip(np.mean(cv), 0, 1))


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


def train_voice_model(username: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            print(f"❌ User '{username}' not found!")
            return None

        print(f"\n{'='*70}")
        print(f"  IMPROVED VOICE MODEL — user: {username}  ({N_FEATURES} features)")
        print(f"{'='*70}")

        genuine_vectors = load_enrollment_samples(db, user.id)
        if not genuine_vectors:
            print("❌ No valid voice enrollment samples.")
            return None

        print(f"  Enrollment samples: {len(genuine_vectors)}")
        if len(genuine_vectors) < 3:
            print("  ⚠  Fewer than 3 samples — accuracy will be lower.")

        profile_mean = np.array(genuine_vectors).mean(axis=0)
        profile_std  = (np.array(genuine_vectors).std(axis=0)
                        if len(genuine_vectors) > 1
                        else np.abs(profile_mean) * 0.10)

        noise_level = estimate_enrollment_noise(genuine_vectors)
        print(f"  Enrollment noise level: {noise_level:.2f} "
              f"({'quiet' if noise_level < 0.2 else 'moderate' if noise_level < 0.4 else 'noisy'})")

        n_aug = max(150, len(genuine_vectors) * 40)
        genuine_aug = generate_genuine_augmentations(genuine_vectors, n=n_aug)

        real_impostors = load_real_impostors(db, user.id)
        if len(real_impostors) >= 20:
            print(f"  Using {len(real_impostors)} real impostor samples.")
            impostor_samples = real_impostors
            while len(impostor_samples) < len(genuine_aug) * 2:
                impostor_samples += generate_synthetic_impostors(profile_mean, profile_std, n=100)
        else:
            impostor_samples = real_impostors + generate_synthetic_impostors(
                profile_mean, profile_std, n=max(300, len(genuine_aug) * 2))

        X = np.vstack([genuine_aug, impostor_samples])
        y = np.array([1] * len(genuine_aug) + [0] * len(impostor_samples))

        print(f"  Training data: {len(genuine_aug)} genuine / {len(impostor_samples)} impostor")

        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import confusion_matrix, accuracy_score

        pipeline = build_pipeline()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("  Running 5-fold cross-validation …")
        y_prob_cv = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

        best_thresh, best_eer = 0.50, 1.0
        print(f"\n  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'EER':>8}")
        for t in np.arange(0.30, 0.90, 0.02):
            y_t = (y_prob_cv >= t).astype(int)
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

        # Noise-adaptive threshold tightening
        noise_adj    = noise_level * 0.08
        final_thresh = min(best_thresh + noise_adj, 0.85)
        if noise_adj > 0.01:
            print(f"\n  ⚠  Noise-adaptive: {best_thresh:.2f} → {final_thresh:.2f} (+{noise_adj:.2f})")

        y_final = (y_prob_cv >= final_thresh).astype(int)
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
            'pipeline':     pipeline,
            'n_features':   N_FEATURES,
            'username':     username,
            'user_id':      user.id,
            'n_enrollment': len(genuine_vectors),
            'profile_mean': profile_mean,
            'profile_std':  profile_std,
            'threshold':    final_thresh,
            'noise_level':  noise_level,
            'far':          float(far),
            'frr':          float(frr),
            'eer':          float(best_eer),
            'model_type':   'gbm_improved_v2',
        }
        model_path = os.path.join(model_dir, f"{username}_voice_cnn.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n  ✅ Model saved → {model_path}  ({os.path.getsize(model_path)/1024:.1f} KB)")
        return model_path
    finally:
        db.close()


def predict_voice(username: str, feature_dict: dict) -> dict:
    model_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_path = os.path.join(model_dir, f"{username}_voice_cnn.pkl")

    if not os.path.exists(model_path):
        return {'error': f'No voice model for "{username}". Enroll first.'}

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    pipeline     = model_data['pipeline']
    profile_mean = model_data['profile_mean']
    profile_std  = model_data['profile_std']
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

    if model_n_feat == 62:
        raw_vec = (
            mfcc_mean[:13] + mfcc_std[:13] +
            delta_mean[:13] + delta2_mean[:13] + [
                float(feature_dict.get('pitch_mean',             0)),
                float(feature_dict.get('pitch_std',              0)),
                float(feature_dict.get('speaking_rate',          0)),
                float(feature_dict.get('energy_mean',            0)),
                float(feature_dict.get('energy_std',             0)),
                float(feature_dict.get('zcr_mean',               0)),
                float(feature_dict.get('spectral_centroid_mean', 0)),
                float(feature_dict.get('spectral_rolloff_mean',  0)),
                float(feature_dict.get('spectral_flux_mean',     0)),
                float(feature_dict.get('voiced_fraction',        0)),
            ]
        )
    else:
        # Backward-compat: old 34-feat model
        raw_vec = (
            mfcc_mean[:13] + mfcc_std[:13] + [
                float(feature_dict.get('pitch_mean',             0)),
                float(feature_dict.get('pitch_std',              0)),
                float(feature_dict.get('speaking_rate',          0)),
                float(feature_dict.get('energy_mean',            0)),
                float(feature_dict.get('energy_std',             0)),
                float(feature_dict.get('zcr_mean',               0)),
                float(feature_dict.get('spectral_centroid_mean', 0)),
                float(feature_dict.get('spectral_rolloff_mean',  0)),
            ]
        )

    vec = np.array(raw_vec, dtype=np.float64).reshape(1, -1)
    n = len(profile_mean)
    if vec.shape[1] < n:
        vec = np.hstack([vec, np.zeros((1, n - vec.shape[1]))])
    elif vec.shape[1] > n:
        vec = vec[:, :n]

    model_prob = float(pipeline.predict_proba(vec)[0][1])
    mah_score  = mahalanobis_score(vec[0], profile_mean, profile_std)
    fused      = 0.70 * model_prob + 0.30 * mah_score
    match      = fused >= threshold

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

    username = sys.argv[1] if len(sys.argv) > 1 else input("Username: ").strip()
    train_voice_model(username)
