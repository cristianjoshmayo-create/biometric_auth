# ml/benchmark_keystroke.py
#
# Compares keystroke classification algorithms on the same enrolled-user data,
# so the thesis can defend the claim "we identified the most suitable algorithm".
#
# For every enrolled user we:
#   1. Load their genuine keystroke vectors from the DB.
#   2. Build an impostor pool: CMU subjects + other enrolled users' samples.
#   3. Train each candidate algorithm with stratified 5-fold cross-validation.
#   4. Record EER, FAR@EER, FRR@EER, ROC-AUC, train time (s), infer time (ms).
#
# Algorithms compared:
#   - RandomForest (production)
#   - GradientBoosting (production small-set fallback)
#   - SVM (RBF kernel, probability=True)
#   - k-NN with Manhattan distance (Killourhy & Maxion 2009 baseline)
#   - Logistic Regression (linear baseline)
#   - One-Class SVM (anomaly-style baseline; trained on genuine only)
#   - MLP (small dense net)
#   - ProfileMatcher_GP (Gunetti-Picardi A+R, production cold-start algorithm)
#
# Output: results/keystroke_benchmark.csv  +  results/keystroke_benchmark_summary.csv
#
# Run:  python ml/benchmark_keystroke.py

import os
import sys
import time
import csv
import warnings

warnings.filterwarnings("ignore")

backend_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'
)
sys.path.insert(0, backend_path)

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from database.db import SessionLocal
from database.models import User
from utils.crypto import decrypt

from train_keystroke_rf import (
    FEATURE_NAMES,
    load_enrollment_samples,
    load_cmu_impostors,
    generate_impostor_samples,
    get_active_digraphs,
    get_active_key_dwells,
    get_phrase_digraph_pairs,
    get_active_trigraphs,
)
from keystroke_profile_matcher import compute_set_match_score, _is_scoring_feature


def compute_eer(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(fpr[idx]), float(fnr[idx]), float(thr[idx])


def make_models():
    return {
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=3,
                class_weight={0: 2, 1: 1}, random_state=42, n_jobs=-1,
            )),
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42,
            )),
        ]),
        'SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                        probability=True, class_weight={0: 2, 1: 1}, random_state=42)),
        ]),
        'kNN_Manhattan': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, metric='manhattan')),
        ]),
        'LogReg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight={0: 2, 1: 1},
                                       max_iter=1000, random_state=42)),
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                  random_state=42)),
        ]),
    }


def evaluate_one_class_svm(X, y, n_splits=5):
    """Trained on genuine only; scores both classes by signed distance."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all, y_score_all = [], []
    train_times, infer_times = [], []
    for tr, te in skf.split(X, y):
        X_tr_gen = X[tr][y[tr] == 1]
        if len(X_tr_gen) < 3:
            continue
        scaler = StandardScaler().fit(X_tr_gen)
        X_tr_s = scaler.transform(X_tr_gen)
        X_te_s = scaler.transform(X[te])
        clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
        t0 = time.perf_counter()
        clf.fit(X_tr_s)
        train_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        scores = clf.decision_function(X_te_s)
        infer_times.append((time.perf_counter() - t0) / max(len(X_te_s), 1) * 1000)
        y_true_all.extend(y[te].tolist())
        y_score_all.extend(scores.tolist())
    return np.array(y_true_all), np.array(y_score_all), train_times, infer_times


def patch_impostor_zeros(X_imp, feat_names, p_mean, p_std, seed=77):
    """
    Mirror train_keystroke_rf._patch_impostor_digraphs: replace zero-padded
    phrase-specific columns (extra_/key_/trigraph_/per-pair flight_) on
    real impostor vectors with phrase-plausible but distinct timings.

    Without this, GP A+R masks those zero positions out and the supervised
    classifiers learn a "zero == impostor" shortcut. Patching produces a
    fair comparison across all algorithms.
    """
    if len(X_imp) == 0:
        return X_imp
    rng = np.random.default_rng(seed)
    aggr_flights = {'flight_mean', 'flight_std', 'flight_median',
                    'flight_mean_norm', 'flight_std_norm'}
    dyn_cols = [i for i, n in enumerate(feat_names)
                if (n.startswith('extra_') or n.startswith('key_')
                    or n.startswith('trigraph_')
                    or (n.startswith('flight_') and n not in aggr_flights))]
    if not dyn_cols:
        return X_imp
    out = X_imp.copy()
    for r in range(len(out)):
        for col in dyn_cols:
            if abs(out[r, col]) > 1e-6:
                continue  # already filled (e.g. synthetic) — leave alone
            name = feat_names[col]
            g_mean = p_mean[col]
            g_std = p_std[col]
            min_sep = max(g_std * 1.5, g_mean * 0.15, 5.0)
            val = g_mean
            for _ in range(20):
                v = rng.normal(g_mean, g_std * 2.5 + 15.0)
                if name.startswith('key_'):
                    v = float(np.clip(v, 20.0, 300.0))
                elif name.startswith('trigraph_'):
                    v = float(np.clip(v, 40.0, 500.0))
                else:
                    v = float(np.clip(v, 20.0, 300.0))
                if abs(v - g_mean) >= min_sep:
                    val = v
                    break
            out[r, col] = val
    return out


def evaluate_profile_matcher(X, y, feat_names):
    """
    LOOCV evaluation of the Gunetti-Picardi A+R profile matcher.

    For each genuine sample i, score it via set-match against the other
    n-1 genuine samples. For each impostor, score against the full
    genuine reference set. No "training" beyond storing the references.
    """
    genuine = X[y == 1]
    impostors = X[y == 0]
    if len(genuine) < 4:
        return np.array([]), np.array([]), [], []

    profile_std = np.zeros(X.shape[1])  # accepted but unused by GP scorer
    y_true_all, y_score_all = [], []
    train_times, infer_times = [], []

    for i in range(len(genuine)):
        ref = [genuine[j] for j in range(len(genuine)) if j != i]
        train_times.append(0.0)  # no training; references just stored
        t0 = time.perf_counter()
        r = compute_set_match_score(genuine[i], feat_names, ref, profile_std)
        infer_times.append((time.perf_counter() - t0) * 1000)
        y_true_all.append(1)
        y_score_all.append(r['score'])

    ref_full = [v for v in genuine]
    for v in impostors:
        t0 = time.perf_counter()
        r = compute_set_match_score(v, feat_names, ref_full, profile_std)
        infer_times.append((time.perf_counter() - t0) * 1000)
        y_true_all.append(0)
        y_score_all.append(r['score'])

    return np.array(y_true_all), np.array(y_score_all), train_times, infer_times


def evaluate_classifier(name, model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all, y_score_all = [], []
    train_times, infer_times = [], []
    for tr, te in skf.split(X, y):
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        train_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X[te])[:, 1]
        else:
            scores = model.decision_function(X[te])
        infer_times.append((time.perf_counter() - t0) / max(len(X[te]), 1) * 1000)
        y_true_all.extend(y[te].tolist())
        y_score_all.extend(scores.tolist())
    return np.array(y_true_all), np.array(y_score_all), train_times, infer_times


def build_user_dataset(db, user):
    phrase = decrypt(user.phrase or "")
    _, extra_pairs = get_active_digraphs(phrase)
    key_keys = get_active_key_dwells(phrase)
    flight_pair_keys = get_phrase_digraph_pairs(phrase)
    trigraph_keys = get_active_trigraphs(phrase)

    full_feat_names = (
        list(FEATURE_NAMES)
        + [f"extra_{p}"    for p in extra_pairs]
        + [f"key_{k}"      for k in key_keys]
        + [f"flight_{p}"   for p in flight_pair_keys]
        + [f"trigraph_{t}" for t in trigraph_keys]
    )

    genuine = load_enrollment_samples(
        db, user.id,
        extra_keys=extra_pairs, key_keys=key_keys,
        flight_pair_keys=flight_pair_keys, trigraph_keys=trigraph_keys,
    )
    if len(genuine) < 3:
        return None

    cmu = load_cmu_impostors()
    others = [
        u for u in db.query(User).filter(User.id != user.id).all()
    ]
    other_imposters = []
    for u in others:
        s = load_enrollment_samples(db, u.id)
        if s:
            other_imposters.extend(s)

    base_len = len(FEATURE_NAMES)
    full_len = len(full_feat_names)

    def pad(v):
        v = np.asarray(v, dtype=np.float64)
        if v.shape[0] >= full_len:
            return v[:full_len]
        out = np.zeros(full_len)
        out[:min(v.shape[0], base_len)] = v[:min(v.shape[0], base_len)]
        return out

    g = np.array([pad(v) for v in genuine])
    p_mean = g.mean(axis=0)
    p_std = g.std(axis=0) + 1e-9

    real_imp_raw = np.array([pad(v) for v in cmu] + [pad(v) for v in other_imposters]) \
        if (cmu or other_imposters) else np.empty((0, full_len))

    # Phrase-aware patching of zero-padded scoring columns. Mirrors what
    # train_keystroke_rf.py does in production: prevents GP masking and the
    # "zero column == impostor" shortcut for supervised models.
    real_imp_arr = patch_impostor_zeros(real_imp_raw, full_feat_names, p_mean, p_std)
    real_imp = [v for v in real_imp_arr]

    n_synth = max(0, max(400, len(g) * 80) - len(real_imp))
    syn = generate_impostor_samples(p_mean, p_std, n=n_synth, feat_names=full_feat_names) \
        if n_synth > 0 else []

    X_imp = np.array(real_imp + syn) if (real_imp or syn) else np.empty((0, full_len))

    X = np.vstack([g, X_imp])
    y = np.array([1] * len(g) + [0] * len(X_imp))
    return X, y, len(g), len(real_imp), len(syn), full_feat_names


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, 'keystroke_benchmark.csv')
    summary_path = os.path.join(out_dir, 'keystroke_benchmark_summary.csv')

    db = SessionLocal()
    try:
        users = db.query(User).all()
        print(f"Found {len(users)} users to benchmark.\n")

        all_rows = []
        for user in users:
            print(f"[user] {user.username}")
            ds = build_user_dataset(db, user)
            if ds is None:
                print("  skipped (insufficient genuine samples)\n")
                continue
            X, y, n_gen, n_real_imp, n_syn, feat_names = ds
            print(f"  genuine={n_gen}  real_imp={n_real_imp}  syn_imp={n_syn}  feats={X.shape[1]}")

            for name, model in make_models().items():
                try:
                    y_t, y_s, tt, it = evaluate_classifier(name, model, X, y)
                    eer, far, frr, _ = compute_eer(y_t, y_s)
                    auc = roc_auc_score(y_t, y_s)
                    row = {
                        'user': user.username, 'algorithm': name,
                        'n_genuine': n_gen, 'n_impostor': len(y) - n_gen,
                        'features': X.shape[1],
                        'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr,
                        'auc': auc,
                        'train_time_s': float(np.mean(tt)),
                        'infer_time_ms': float(np.mean(it)),
                    }
                    all_rows.append(row)
                    print(f"    {name:18s}  EER={eer:.3f}  AUC={auc:.3f}  "
                          f"train={np.mean(tt):.2f}s  infer={np.mean(it):.2f}ms")
                except Exception as e:
                    print(f"    {name:18s}  FAILED: {e}")

            # One-Class SVM (genuine-only training)
            try:
                y_t, y_s, tt, it = evaluate_one_class_svm(X, y)
                if len(y_t) > 0:
                    eer, far, frr, _ = compute_eer(y_t, y_s)
                    auc = roc_auc_score(y_t, y_s)
                    row = {
                        'user': user.username, 'algorithm': 'OneClassSVM',
                        'n_genuine': n_gen, 'n_impostor': len(y) - n_gen,
                        'features': X.shape[1],
                        'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr,
                        'auc': auc,
                        'train_time_s': float(np.mean(tt)),
                        'infer_time_ms': float(np.mean(it)),
                    }
                    all_rows.append(row)
                    print(f"    {'OneClassSVM':18s}  EER={eer:.3f}  AUC={auc:.3f}")
            except Exception as e:
                print(f"    OneClassSVM       FAILED: {e}")

            # Profile Matcher (Gunetti-Picardi A+R) — production cold-start
            try:
                y_t, y_s, tt, it = evaluate_profile_matcher(X, y, feat_names)
                if len(y_t) > 0 and len(np.unique(y_t)) >= 2:
                    eer, far, frr, _ = compute_eer(y_t, y_s)
                    auc = roc_auc_score(y_t, y_s)
                    row = {
                        'user': user.username, 'algorithm': 'ProfileMatcher_GP',
                        'n_genuine': n_gen, 'n_impostor': len(y) - n_gen,
                        'features': X.shape[1],
                        'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr,
                        'auc': auc,
                        'train_time_s': float(np.mean(tt)) if tt else 0.0,
                        'infer_time_ms': float(np.mean(it)) if it else 0.0,
                    }
                    all_rows.append(row)
                    print(f"    {'ProfileMatcher_GP':18s}  EER={eer:.3f}  AUC={auc:.3f}")
            except Exception as e:
                print(f"    ProfileMatcher_GP FAILED: {e}")
            print()
    finally:
        db.close()

    if not all_rows:
        print("No results produced.")
        return

    fields = list(all_rows[0].keys())
    with open(rows_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Per-user results -> {rows_path}")

    by_algo = {}
    for r in all_rows:
        by_algo.setdefault(r['algorithm'], []).append(r)
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'n_users', 'mean_eer', 'std_eer',
                    'mean_auc', 'mean_train_s', 'mean_infer_ms'])
        ranked = []
        for algo, rs in by_algo.items():
            eers = [r['eer'] for r in rs]
            aucs = [r['auc'] for r in rs]
            tts = [r['train_time_s'] for r in rs]
            its = [r['infer_time_ms'] for r in rs]
            ranked.append((algo, len(rs), np.mean(eers), np.std(eers),
                           np.mean(aucs), np.mean(tts), np.mean(its)))
        ranked.sort(key=lambda x: x[2])  # ascending EER
        for row in ranked:
            w.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}",
                        f"{row[4]:.4f}", f"{row[5]:.3f}", f"{row[6]:.3f}"])

    print(f"Summary (ranked by EER) -> {summary_path}")
    print("\n=== RANKED SUMMARY ===")
    print(f"{'Algorithm':<18} {'Users':>6} {'meanEER':>9} {'meanAUC':>9} {'train(s)':>10} {'infer(ms)':>10}")
    for r in ranked:
        print(f"{r[0]:<18} {r[1]:>6d} {r[2]:>9.4f} {r[4]:>9.4f} {r[5]:>10.3f} {r[6]:>10.3f}")


if __name__ == "__main__":
    main()
