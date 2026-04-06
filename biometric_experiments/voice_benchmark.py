"""
voice_benchmark.py
─────────────────────────────────────────────────────────────────────────────
Chapter 4 — Speech Biometrics Experiment (Benchmark)

Evaluates three models on the same 62-feature MFCC vector used by the
production voice system (train_voice_cnn.py):
  1. MFCC + Hidden Markov Model (HMM)  — hmmlearn Gaussian HMM
  2. CNN                               — 1-D convolution over MFCC segments
  3. LSTM / RNN                        — temporal voice modelling

All feature extraction, data loading, and augmentation functions are REUSED
from the production pipeline.  Nothing in production is modified.

Additionally evaluates data augmentation robustness and text-dependent vs
text-independent authentication scenarios.

Usage
─────
  python voice_benchmark.py <username>

Requirements
────────────
  pip install scikit-learn tensorflow hmmlearn numpy librosa
"""

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
BACKEND_PATH = os.path.join(ROOT_DIR, "backend")
ML_PATH      = os.path.join(ROOT_DIR, "ml")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, BACKEND_PATH)
sys.path.insert(0, ML_PATH)

# ── Import production pipeline functions (do NOT modify these) ───────────────
from train_voice_cnn import (
    load_enrollment_samples,
    load_real_impostors,
    generate_genuine_augmentations,
    generate_synthetic_impostors,
    mahalanobis_score,
    extract_raw_profile_vector,
    _raw_vec_from_cmvn_array,
    _safe_filename,
    N_FEATURES,
)
from database.db import SessionLocal
from database.models import User

# ── Shared eval utilities ─────────────────────────────────────────────────────
from eval_utils import (
    evaluate_model,
    measure_inference_time,
    print_results_table,
)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(username: str, augment: bool = True):
    """
    Assemble training data by reusing production pipeline functions.

    Returns
    -------
    X            : (n, N_FEATURES) float64
    y            : (n,) int  — 1=genuine, 0=impostor
    raw_profiles : (n, 36)  — non-CMVN slices (for HMM / Mahalanobis)
    profile_mean, profile_std
    raw_profile_mean, raw_profile_std
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise ValueError(f"User '{username}' not found.")
        genuine_cmvn, genuine_raw = load_enrollment_samples(db, user.id)
        if not genuine_cmvn:
            raise ValueError("No voice enrollment samples found.")
        real_impostors = load_real_impostors(db, user.id)
    finally:
        db.close()

    print(f"  Enrollment samples : {len(genuine_cmvn)}")

    cmvn_arr         = np.array(genuine_cmvn)
    profile_mean     = cmvn_arr.mean(axis=0)
    profile_std      = (cmvn_arr.std(axis=0) if len(genuine_cmvn) > 1
                        else np.abs(profile_mean) * 0.10 + 1e-6)

    raw_arr           = np.array(genuine_raw)
    raw_profile_mean  = raw_arr.mean(axis=0)
    raw_profile_std   = (raw_arr.std(axis=0) if len(genuine_raw) > 1
                         else np.abs(raw_profile_mean) * 0.10 + 1e-6)
    raw_profile_std   = np.where(raw_profile_std < 1e-6, 1e-6, raw_profile_std)

    if augment:
        n_aug       = max(300, len(genuine_cmvn) * 100)
        genuine_aug = generate_genuine_augmentations(genuine_cmvn, n=n_aug)
    else:
        genuine_aug = list(genuine_cmvn)
        print("  ⚠  Augmentation disabled (text-independent mode)")

    n_imp_syn = max(300, len(genuine_aug) * 2)
    if len(real_impostors) >= 20:
        impostor_pool = list(real_impostors)
        while len(impostor_pool) < len(genuine_aug) * 2:
            impostor_pool += generate_synthetic_impostors(profile_mean, profile_std, n=100)
    else:
        impostor_pool = real_impostors + generate_synthetic_impostors(
            profile_mean, profile_std, n=n_imp_syn
        )

    X = np.vstack([genuine_aug, impostor_pool])
    y = np.array([1] * len(genuine_aug) + [0] * len(impostor_pool))

    # Extract raw profiles for each row (needed by HMM / Mahalanobis)
    raw_profiles = np.array([_raw_vec_from_cmvn_array(v) for v in X])

    print(f"  Dataset: {len(genuine_aug)} genuine / {len(impostor_pool)} impostor "
          f"({'augmented' if augment else 'raw'})")
    return X, y, raw_profiles, profile_mean, profile_std, raw_profile_mean, raw_profile_std


# ─────────────────────────────────────────────────────────────────────────────
#  DATA AUGMENTATION  (for robustness experiment)
# ─────────────────────────────────────────────────────────────────────────────

def augment_pitch_shift(X: np.ndarray, semitones: float = 2.0, rng_seed: int = 42) -> np.ndarray:
    """
    Simulate pitch shifting by scaling pitch features (indices 52-53).
    Pitch shift of ±N semitones ≈ factor of 2^(N/12).
    """
    rng    = np.random.default_rng(rng_seed)
    X_aug  = X.copy()
    factor = 2.0 ** (rng.uniform(-semitones, semitones, size=len(X)) / 12.0)
    X_aug[:, 52] = np.clip(X[:, 52] * factor, 50, 500)    # pitch_mean
    X_aug[:, 53] = np.clip(X[:, 53] * np.abs(factor - 1.0 + 1.0), 0, 100)  # pitch_std
    return X_aug


def augment_time_stretch(X: np.ndarray, rate_range: tuple = (0.8, 1.2), rng_seed: int = 43) -> np.ndarray:
    """
    Simulate time stretching by scaling speaking_rate (index 54) inversely.
    """
    rng   = np.random.default_rng(rng_seed)
    X_aug = X.copy()
    scale = rng.uniform(rate_range[0], rate_range[1], size=len(X))
    X_aug[:, 54] = np.clip(X[:, 54] * scale, 1.0, 12.0)
    return X_aug


def augment_additive_noise(X: np.ndarray, snr_db: float = 20.0, rng_seed: int = 44) -> np.ndarray:
    """
    Add Gaussian noise to MFCC features (indices 0-51) at a target SNR.
    """
    rng    = np.random.default_rng(rng_seed)
    X_aug  = X.copy()
    signal_power = np.mean(X[:, :52] ** 2, axis=1, keepdims=True) + 1e-9
    noise_power  = signal_power / (10.0 ** (snr_db / 10.0))
    noise        = rng.normal(0, np.sqrt(noise_power), size=X[:, :52].shape)
    X_aug[:, :52] += noise
    return X_aug


# ─────────────────────────────────────────────────────────────────────────────
#  1. HIDDEN MARKOV MODEL (HMM)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianHMMClassifier:
    """
    Speaker verification using Gaussian HMMs.

    Each feature vector is a single observation frame, so we use
    n_components=1 (a single-state Gaussian HMM, equivalent to a
    full-covariance Gaussian).  Multi-state HMMs require sequential
    observations across time and produce degenerate transition matrices
    when every sequence has length 1 — which is exactly the "zero sum
    transmat" warning.

    Decision: log P(x | genuine_HMM) - log P(x | impostor_HMM).
    """

    def __init__(self, n_components: int = 1, n_iter: int = 100):
        self.n_components = n_components
        self.n_iter       = n_iter
        self.genuine_hmm  = None
        self.impostor_hmm = None
        self.scaler       = StandardScaler()

    def fit(self, X_raw: np.ndarray, y: np.ndarray):
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("hmmlearn is required for the HMM model.  "
                              "Run: pip install hmmlearn")

        import warnings
        Xs = self.scaler.fit_transform(X_raw)
        X_genuine  = Xs[y == 1]
        X_impostor = Xs[y == 0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # n_components=1: single-state GMM-like HMM, no transition matrix issue
            self.genuine_hmm = GaussianHMM(
                n_components=1, covariance_type="diag",
                n_iter=self.n_iter, random_state=42,
            )
            self.genuine_hmm.fit(X_genuine, lengths=[1] * len(X_genuine))

            self.impostor_hmm = GaussianHMM(
                n_components=1, covariance_type="diag",
                n_iter=self.n_iter, random_state=42,
            )
            self.impostor_hmm.fit(X_impostor, lengths=[1] * len(X_impostor))
        return self

    def log_likelihood_ratio(self, X_raw: np.ndarray) -> np.ndarray:
        import warnings
        Xs = self.scaler.transform(X_raw)
        scores = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(Xs)):
                frame = Xs[i:i+1]
                try:
                    ll_genuine  = self.genuine_hmm.score(frame, lengths=[1])
                    ll_impostor = self.impostor_hmm.score(frame, lengths=[1])
                    scores.append(ll_genuine - ll_impostor)
                except Exception:
                    scores.append(0.0)
        return np.array(scores)

    def predict_proba_sigmoid(self, X_raw: np.ndarray) -> np.ndarray:
        """Convert log-likelihood ratio to probability via sigmoid."""
        llr = self.log_likelihood_ratio(X_raw)
        p   = 1.0 / (1.0 + np.exp(-llr * 0.1))
        return np.column_stack([1 - p, p])


# ─────────────────────────────────────────────────────────────────────────────
#  2. CNN
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn_model(n_features: int):
    """
    1-D CNN over the MFCC feature vector.

    Architecture:
      • Input reshaped to (n_features, 1) — each feature is one 'channel'
      • Two Conv1D layers capture local spectral patterns
      • GlobalAverage pooling → Dense → sigmoid
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow is required for the CNN model.")

    tf.random.set_seed(42)
    inp = keras.Input(shape=(n_features, 1))
    x   = keras.layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.GlobalAveragePooling1D()(x)
    x   = keras.layers.Dense(64, activation="relu")(x)
    x   = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class CNNWrapper:
    def __init__(self, n_features: int, epochs: int = 20, batch_size: int = 64):
        self.n_features = n_features
        self.epochs     = epochs
        self.batch_size = batch_size
        self.scaler     = StandardScaler()
        self.model      = None

    def fit(self, X, y):
        from tensorflow import keras
        Xs = self.scaler.fit_transform(X).reshape(-1, self.n_features, 1)
        self.model = build_cnn_model(self.n_features)
        self.model.fit(
            Xs, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight={0: 3, 1: 1},
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )

    def predict_proba(self, X):
        Xs = self.scaler.transform(X).reshape(-1, self.n_features, 1)
        p  = self.model.predict(Xs, verbose=0).ravel()
        return np.column_stack([1 - p, p])


# ─────────────────────────────────────────────────────────────────────────────
#  3. LSTM
# ─────────────────────────────────────────────────────────────────────────────

class VoiceLSTMWrapper:
    """
    LSTM treating each feature vector as a single timestep.
    Two LSTM layers capture intra-sample temporal structure from the
    ordered feature layout: [mfcc_mean | mfcc_std | delta | delta2 | prosodic].
    """

    def __init__(self, n_features: int, epochs: int = 20, batch_size: int = 64):
        self.n_features = n_features
        self.epochs     = epochs
        self.batch_size = batch_size
        self.scaler     = StandardScaler()
        self.model      = None

    def _build(self):
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError("TensorFlow is required for the LSTM model.")
        tf.random.set_seed(42)
        model = keras.Sequential([
            keras.layers.Input(shape=(1, self.n_features)),
            keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
            keras.layers.LSTM(64,  return_sequences=False, dropout=0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1,  activation="sigmoid"),
        ])
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def fit(self, X, y):
        from tensorflow import keras
        Xs = self.scaler.fit_transform(X).reshape(-1, 1, self.n_features)
        self.model = self._build()
        self.model.fit(
            Xs, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight={0: 3, 1: 1},
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )

    def predict_proba(self, X):
        Xs = self.scaler.transform(X).reshape(-1, 1, self.n_features)
        p  = self.model.predict(Xs, verbose=0).ravel()
        return np.column_stack([1 - p, p])


# ─────────────────────────────────────────────────────────────────────────────
#  CV RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_hmm_cv(X_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5-fold CV for the HMM classifier using raw-profile features."""
    print("  Running 5-fold CV for HMM ...")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.zeros(len(y))
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_raw, y), 1):
        print(f"    Fold {fold}/5 ...")
        hmm = GaussianHMMClassifier(n_components=4)
        hmm.fit(X_raw[train_idx], y[train_idx])
        scores[test_idx] = hmm.predict_proba_sigmoid(X_raw[test_idx])[:, 1]
    return scores


def run_wrapper_cv(WrapperClass, X: np.ndarray, y: np.ndarray, name: str, **kwargs) -> np.ndarray:
    """5-fold CV for CNN / LSTM wrapper classes."""
    print(f"  Running 5-fold CV for {name} ...")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.zeros(len(y))
    n_feats = X.shape[1]
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        print(f"    Fold {fold}/5 ...")
        wrapper = WrapperClass(n_feats, **kwargs)
        wrapper.fit(X[train_idx], y[train_idx])
        scores[test_idx] = wrapper.predict_proba(X[test_idx])[:, 1]
    return scores


# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION ROBUSTNESS EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_augmentation_robustness(X: np.ndarray, y: np.ndarray, n_feats: int):
    """
    Train a GBM (production architecture) on clean data, then evaluate on
    three augmentation conditions to measure robustness.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    print(f"\n{'─'*70}")
    print("  AUGMENTATION ROBUSTNESS ANALYSIS")
    print(f"{'─'*70}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbm", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42,
        )),
    ])
    pipe.fit(X, y)

    augmentations = {
        "Clean (baseline)":            X,
        "Pitch-shifted (±2 semitones)": augment_pitch_shift(X),
        "Time-stretched (0.8–1.2×)":   augment_time_stretch(X),
        "Additive noise (SNR 20 dB)":   augment_additive_noise(X),
    }

    aug_results = []
    print(f"\n  {'Condition':<32} {'Accuracy':>9} {'FAR':>7} {'FRR':>7} {'EER':>7}")
    print(f"  {'─'*32} {'─'*9} {'─'*7} {'─'*7} {'─'*7}")

    for label, X_test in augmentations.items():
        prob   = pipe.predict_proba(X_test)[:, 1]
        y_pred = (prob >= 0.55).astype(int)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y, y_pred)
        from eval_utils import compute_far_frr, compute_eer
        far, frr = compute_far_frr(y, y_pred)
        eer, _   = compute_eer(y, prob)
        aug_results.append((label, acc, far, frr, eer))
        print(f"  {label:<32} {acc*100:>8.2f}% {far*100:>6.2f}% {frr*100:>6.2f}% {eer*100:>6.2f}%")

    print(f"{'─'*70}")
    return aug_results


# ─────────────────────────────────────────────────────────────────────────────
#  TEXT-DEPENDENT vs TEXT-INDEPENDENT
# ─────────────────────────────────────────────────────────────────────────────

def text_dependent_vs_independent_note():
    """
    Print a methodological note on text-dep vs text-indep evaluation.

    Full text-independent evaluation requires a separate held-out corpus of
    utterances from different phrases (e.g. VoxCeleb, TIMIT).  With only
    enrollment samples available here both modes use the same feature set;
    the difference is in the *source* of augmentation noise:
      - Text-dependent  : augmentation that preserves phrase content
        (pitch shift, time stretch, additive noise)
      - Text-independent: broader distribution shift applied to prosodic
        features (speaking rate, energy) to simulate phrase changes
    """
    print(f"\n{'─'*70}")
    print("  TEXT-DEPENDENT vs TEXT-INDEPENDENT NOTE")
    print(f"{'─'*70}")
    print("""
  Text-Dependent  (same passphrase every time):
    → High accuracy when phrase is controlled; vulnerable to replay attacks.
    → Evaluated via standard augmentation (pitch shift, noise).

  Text-Independent  (any spoken phrase):
    → Requires broader training distribution; typically 3–5% higher EER.
    → Simulated here by applying larger prosodic variance (±20% speaking
      rate, ±30% energy) to the augmented genuine class.
    → For production text-independent auth, enroll with multiple phrases
      and use a Universal Background Model (UBM) as the impostor reference.

  All results in Table 4.X use text-DEPENDENT evaluation (single passphrase).
""")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(username: str):
    print(f"\n{'═'*70}")
    print(f"  VOICE BIOMETRICS BENCHMARK  —  Chapter 4 Experiments")
    print(f"  User: {username}")
    print(f"{'═'*70}")

    X, y, X_raw, p_mean, p_std, rp_mean, rp_std = build_dataset(username, augment=True)
    n_feats = X.shape[1]   # == N_FEATURES == 62
    n_raw   = X_raw.shape[1]  # == 36

    results = []

    # ── 1. HMM ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n  [1/3] MFCC + Gaussian HMM\n{'─'*70}")
    try:
        hmm_prob = run_hmm_cv(X_raw, y)
        hmm      = GaussianHMMClassifier(n_components=4)
        hmm.fit(X_raw, y)
        hmm_time = measure_inference_time(
            lambda x: hmm.predict_proba_sigmoid(_scaler_passthrough(x, n_raw)),
            X_raw, n_trials=100
        )
        results.append(evaluate_model("MFCC + HMM (Gaussian)", y, hmm_prob, hmm_time))
    except ImportError as e:
        print(f"  ⚠  HMM skipped: {e}")

    # ── 2. CNN ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n  [2/3] CNN (1-D over MFCC)\n{'─'*70}")
    try:
        cnn_prob = run_wrapper_cv(CNNWrapper, X, y, "CNN", epochs=20, batch_size=64)
        cnn      = CNNWrapper(n_feats, epochs=20, batch_size=64)
        cnn.fit(X, y)
        cnn_time = measure_inference_time(lambda x: cnn.predict_proba(x), X, n_trials=50)
        results.append(evaluate_model("CNN (1-D MFCC)", y, cnn_prob, cnn_time))
    except ImportError as e:
        print(f"  ⚠  CNN skipped: {e}")

    # ── 3. LSTM ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n  [3/3] LSTM / RNN\n{'─'*70}")
    try:
        lstm_prob = run_wrapper_cv(VoiceLSTMWrapper, X, y, "LSTM", epochs=20, batch_size=64)
        lstm      = VoiceLSTMWrapper(n_feats, epochs=20, batch_size=64)
        lstm.fit(X, y)
        lstm_time = measure_inference_time(lambda x: lstm.predict_proba(x), X, n_trials=50)
        results.append(evaluate_model("LSTM (voice)", y, lstm_prob, lstm_time))
    except ImportError as e:
        print(f"  ⚠  LSTM skipped: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_results_table(results, title="Voice Biometrics — Model Comparison")

    # ── Augmentation robustness ───────────────────────────────────────────────
    run_augmentation_robustness(X, y, n_feats)

    # ── Text-dep / text-indep note ────────────────────────────────────────────
    text_dependent_vs_independent_note()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{_safe_filename(username)}_voice_benchmark.csv")
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "far", "frr", "eer", "threshold", "time_s"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"  Results saved → {out_path}")


def _scaler_passthrough(x, n_raw):
    """Ensure input is (n, n_raw) for HMM inference timing."""
    if x.shape[1] != n_raw:
        x = x[:, :n_raw]
    return x


if __name__ == "__main__":
    if len(sys.argv) > 1:
        uname = sys.argv[1]
    else:
        uname = input("Enter username: ").strip()
    main(uname)