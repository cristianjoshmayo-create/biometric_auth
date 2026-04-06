"""
keystroke_benchmark.py
─────────────────────────────────────────────────────────────────────────────
Chapter 4 — Keystroke Dynamics Experiment (Benchmark)

Evaluates four models on the same feature vector used by the production system:
  1. Support Vector Machine (SVM, RBF kernel)
  2. Random Forest  (ensemble baseline + feature importance)
  3. K-Nearest Neighbors (KNN)
  4. LSTM / RNN     (TensorFlow / Keras)

All feature extraction, data loading, augmentation, and impostor generation
functions are REUSED from the production pipeline (train_keystroke_rf.py).
Nothing in the production code is modified.

Usage
─────
  python keystroke_benchmark.py <username>

Requirements
────────────
  pip install scikit-learn tensorflow numpy
"""

import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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
from train_keystroke_rf import (
    load_enrollment_samples,
    load_real_impostors,
    load_cmu_impostors,
    generate_genuine_samples,
    generate_impostor_samples,
    get_active_digraphs,
    FEATURE_NAMES,
    _safe_filename,
)
from database.db import SessionLocal
from database.models import User

# ── Import shared evaluation utilities ───────────────────────────────────────
from eval_utils import (
    evaluate_model,
    measure_inference_time,
    print_results_table,
    print_literature_comparison,
)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET BUILDER
#  Mirrors the data-assembly logic in train_keystroke_rf.train_random_forest()
#  without changing any of the production functions.
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(username: str):
    """
    Load and augment the training dataset for a given user.

    Returns
    -------
    X : np.ndarray  (n_samples, n_features)
    y : np.ndarray  (n_samples,)  — 1=genuine, 0=impostor
    active_feat_names : list[str]
    profile_mean, profile_std : np.ndarray  (for LSTM normalisation)
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise ValueError(f"User '{username}' not found in the database.")
        user_id     = user.id
        user_phrase = user.phrase or ""
        genuine_vecs_raw = load_enrollment_samples(db, user_id)
        cmu_impostors    = load_cmu_impostors()
        real_impostors   = load_real_impostors(db, user_id, n_genuine=len(genuine_vecs_raw))
    finally:
        db.close()

    # ── Phrase-aware digraph filtering (same as production) ──────────────────
    standard_active, extra_pairs = get_active_digraphs(user_phrase)
    all_digraph_feats  = {f for f in FEATURE_NAMES if f.startswith("digraph_")}
    inactive_digraphs  = all_digraph_feats - standard_active
    drop_indices       = [i for i, n in enumerate(FEATURE_NAMES) if n in inactive_digraphs]
    active_feat_names  = (
        [n for n in FEATURE_NAMES if n not in inactive_digraphs]
        + [f"extra_{p}" for p in extra_pairs]
    )

    # Reload genuine samples WITH extra_keys
    db2 = SessionLocal()
    try:
        genuine_vecs = load_enrollment_samples(db2, user_id, extra_keys=extra_pairs)
    finally:
        db2.close()

    n_genuine_real = len(genuine_vecs)
    print(f"  Enrollment samples : {n_genuine_real}")
    print(f"  Active features    : {len(active_feat_names)}")

    def _strip(vecs, is_impostor=False):
        rng = np.random.default_rng(42)
        out = []
        for v in vecs:
            base  = np.delete(v[:len(FEATURE_NAMES)], drop_indices)
            extra = (np.zeros(len(extra_pairs))
                     if is_impostor else v[len(FEATURE_NAMES):])
            out.append(np.concatenate([base, extra]))
        return out

    genuine_vecs   = _strip(genuine_vecs)
    cmu_impostors  = _strip(cmu_impostors,  is_impostor=True)
    real_impostors = _strip(real_impostors, is_impostor=True)

    profile_mean = np.array(genuine_vecs).mean(axis=0)
    profile_std  = np.array(genuine_vecs).std(axis=0) + 1e-9

    n_aug     = max(600, n_genuine_real * 120)
    n_imp_syn = max(1200, n_aug * 2)

    genuine_aug   = generate_genuine_samples(genuine_vecs, n=n_aug,      feat_names=active_feat_names)
    syn_impostors = generate_impostor_samples(profile_mean, profile_std,
                                              n=n_imp_syn, feat_names=active_feat_names)
    all_impostors = cmu_impostors + real_impostors + syn_impostors

    n_feats = len(active_feat_names)
    genuine_aug   = [v for v in genuine_aug   if np.asarray(v).shape[0] == n_feats]
    all_impostors = [v for v in all_impostors  if np.asarray(v).shape[0] == n_feats]

    X = np.vstack([genuine_aug, all_impostors])
    y = np.array([1] * len(genuine_aug) + [0] * len(all_impostors))

    print(f"  Dataset: {len(genuine_aug)} genuine aug / {len(all_impostors)} impostor")
    return X, y, active_feat_names, profile_mean, profile_std


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_svm() -> Pipeline:
    """SVM with RBF kernel and probability calibration."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,   # Required for FAR/FRR threshold sweep
            class_weight={0: 3, 1: 1},  # Security bias (same as production)
            random_state=42,
        )),
    ])


def build_rf() -> Pipeline:
    """Random Forest baseline — matches production hyperparameters."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight={0: 3, 1: 1},
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_knn() -> Pipeline:
    """KNN — distance-based behavioural authentication."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  LSTM MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm(n_features: int):
    """
    1-D LSTM for keystroke sequence modelling.

    The flat feature vector is treated as a single timestep sequence of
    shape (1, n_features) — appropriate for per-attempt feature vectors.
    For richer temporal modelling the caller can reshape across multiple
    enrollment attempts before passing to this model.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow is required for the LSTM model.  "
                          "Run: pip install tensorflow")

    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.Input(shape=(1, n_features)),
        keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
        keras.layers.LSTM(64,  return_sequences=False, dropout=0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1,  activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class LSTMWrapper:
    """
    Sklearn-compatible wrapper around the Keras LSTM so it can be passed
    to the same cross_val_predict / measure_inference_time helpers.
    """

    def __init__(self, n_features: int, epochs: int = 20, batch_size: int = 64):
        self.n_features = n_features
        self.epochs     = epochs
        self.batch_size = batch_size
        self.scaler     = StandardScaler()
        self.model      = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import tensorflow as tf
        from tensorflow import keras
        tf.random.set_seed(42)
        Xs = self.scaler.fit_transform(X).reshape(-1, 1, self.n_features)
        self.model = build_lstm(self.n_features)
        self.model.fit(
            Xs, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight={0: 3, 1: 1},
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="loss", patience=5, restore_best_weights=True
                )
            ],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).reshape(-1, 1, self.n_features)
        p  = self.model.predict(Xs, verbose=0).ravel()
        return np.column_stack([1 - p, p])


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_importance(pipeline: Pipeline, feat_names: list, top_n: int = 15):
    """Print ranked feature importances from a fitted RandomForest pipeline."""
    try:
        clf = pipeline.named_steps["clf"]
        importances = clf.feature_importances_
    except AttributeError:
        print("  Feature importance not available for this model type.")
        return

    pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n{'─'*55}")
    print(f"  TOP {top_n} MOST IMPORTANT KEYSTROKE FEATURES (Random Forest)")
    print(f"{'─'*55}")

    highlight = {
        "dwell_mean", "dwell_std", "flight_mean", "flight_std",
        "typing_speed_cpm", "rhythm_cv", "shift_lag_norm",
    }
    digraph_feats = [n for n in feat_names if "digraph" in n or n.startswith("extra_")]

    for rank, (feat, imp) in enumerate(pairs[:top_n], 1):
        bar   = "█" * int(imp * 300)
        tag   = ""
        if feat in highlight:
            tag = "  ← dwell/flight"
        elif feat in digraph_feats:
            tag = "  ← digraph"
        elif "speed" in feat or "rhythm" in feat:
            tag = "  ← speed/rhythm"
        print(f"  {rank:2d}. {feat:<30} {imp:.4f}  {bar}{tag}")

    # Group-level summary
    groups = {
        "Dwell time":      [i for i, n in enumerate(feat_names) if "dwell" in n],
        "Flight time":     [i for i, n in enumerate(feat_names) if "flight" in n],
        "Digraph latency": [i for i, n in enumerate(feat_names) if "digraph" in n or n.startswith("extra_")],
        "Typing speed":    [i for i, n in enumerate(feat_names) if "speed" in n or "rhythm" in n],
        "Pause / misc":    [i for i, n in enumerate(feat_names) if "pause" in n or "backspace" in n],
    }
    print(f"\n  FEATURE GROUP IMPORTANCE")
    print(f"  {'Group':<22} {'Total Importance':>18}")
    print(f"  {'─'*22} {'─'*18}")
    for gname, idxs in groups.items():
        total = sum(importances[i] for i in idxs)
        bar   = "▓" * int(total * 100)
        print(f"  {gname:<22} {total:>16.4f}  {bar}")
    print(f"{'─'*55}")


# ─────────────────────────────────────────────────────────────────────────────
#  CROSS-VALIDATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_sklearn_cv(name: str, pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Run 5-fold stratified CV and return probability scores for genuine class."""
    print(f"\n  Running 5-fold CV for {name} ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    return y_prob


def run_lstm_cv(wrapper: LSTMWrapper, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Manual 5-fold CV for the Keras LSTM wrapper."""
    print(f"\n  Running 5-fold CV for LSTM ...")
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob  = np.zeros(len(y))
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        print(f"    Fold {fold}/5 ...")
        w = LSTMWrapper(wrapper.n_features, wrapper.epochs, wrapper.batch_size)
        w.fit(X[train_idx], y[train_idx])
        y_prob[test_idx] = w.predict_proba(X[test_idx])[:, 1]
    return y_prob


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(username: str):
    print(f"\n{'═'*70}")
    print(f"  KEYSTROKE DYNAMICS BENCHMARK  —  Chapter 4 Experiments")
    print(f"  User: {username}")
    print(f"{'═'*70}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    X, y, feat_names, profile_mean, profile_std = build_dataset(username)
    n_feats = X.shape[1]

    results = []

    # ── 1. SVM ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n  [1/4] Support Vector Machine (SVM, RBF)\n{'─'*70}")
    svm_pipe  = build_svm()
    svm_prob  = run_sklearn_cv("SVM", svm_pipe, X, y)
    svm_pipe.fit(X, y)
    svm_time  = measure_inference_time(lambda x: svm_pipe.predict_proba(x), X)
    results.append(evaluate_model("SVM (RBF)", y, svm_prob, svm_time))

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n  [2/4] Random Forest\n{'─'*70}")
    rf_pipe  = build_rf()
    rf_prob  = run_sklearn_cv("Random Forest", rf_pipe, X, y)
    rf_pipe.fit(X, y)
    rf_time  = measure_inference_time(lambda x: rf_pipe.predict_proba(x), X)
    results.append(evaluate_model("Random Forest", y, rf_prob, rf_time))
    print_feature_importance(rf_pipe, feat_names)

    # ── 3. KNN ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n  [3/4] K-Nearest Neighbors (KNN)\n{'─'*70}")
    knn_pipe  = build_knn()
    knn_prob  = run_sklearn_cv("KNN", knn_pipe, X, y)
    knn_pipe.fit(X, y)
    knn_time  = measure_inference_time(lambda x: knn_pipe.predict_proba(x), X)
    results.append(evaluate_model("KNN (k=7, weighted)", y, knn_prob, knn_time))

    # ── 4. LSTM ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}\n  [4/4] LSTM / RNN\n{'─'*70}")
    lstm_wrapper = LSTMWrapper(n_feats, epochs=20, batch_size=64)
    try:
        lstm_prob = run_lstm_cv(lstm_wrapper, X, y)
        lstm_wrapper.fit(X, y)
        lstm_time = measure_inference_time(lambda x: lstm_wrapper.predict_proba(x), X, n_trials=50)
        results.append(evaluate_model("LSTM (1-step)", y, lstm_prob, lstm_time))
    except ImportError as e:
        print(f"  ⚠  LSTM skipped: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_results_table(results, title="Keystroke Dynamics — Model Comparison")
    print_literature_comparison(results)

    # ── Save CSV for thesis ───────────────────────────────────────────────────
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{_safe_filename(username)}_keystroke_benchmark.csv")
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "far", "frr", "eer", "threshold", "time_s"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        uname = sys.argv[1]
    else:
        uname = input("Enter username: ").strip()
    main(uname)