# ml/benchmark_keystroke_cmu.py
#
# External keystroke-algorithm benchmark on the public CMU dataset
# (Killourhy & Maxion 2009, DSL-StrongPasswordData.csv — 51 subjects, 400 reps
# of password ".tie5Roanl" each).
#
# Why this exists alongside benchmark_keystroke.py:
#   The internal benchmark (10 enrolled users, full 120-feature space) shows
#   how the algorithms rank on our production pipeline. This external benchmark
#   shows the same ranking holds on a public, peer-reviewed dataset at 5x the
#   subject count, with NO synthetic impostors and NO phrase-feature padding —
#   addressing the "EER=0 looks too clean" caveat in the internal benchmark.
#
# Protocol (mirrors Killourhy & Maxion 2009):
#   For each of the 51 CMU subjects acting as "the user":
#     - genuine     : that subject's first 100 reps   (k-fold CV target)
#     - impostors   : first 5 reps × 50 other subjects = 250 real human samples
#                     typing the same password, no synthesis, no padding
#   Algorithms compared: same 8 as benchmark_keystroke.py.
#
# Feature mapping (so the GP profile matcher's _is_scoring_feature picks them up):
#   H.<key>        ->  key_<key>          (per-key dwell)         11 features
#   DD.<a>.<b>     ->  digraph_<a><b>     (key-down to key-down)  10 features
#   UD.<a>.<b>     ->  extra_<a><b>       (key-up to next-down)   10 features
#   Total: 31 scoring features per sample (no aggregates — CMU doesn't capture them).
#
# Output: results/keystroke_benchmark_cmu.csv + results/keystroke_benchmark_cmu_summary.csv
#
# Run:  venv310/Scripts/python.exe ml/benchmark_keystroke_cmu.py

import os
import sys
import csv
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keystroke_profile_matcher import compute_set_match_score


CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', 'DSL-StrongPasswordData.csv')

H_COLS = [
    'H.period', 'H.t', 'H.i', 'H.e', 'H.five',
    'H.Shift.r', 'H.o', 'H.a', 'H.n', 'H.l', 'H.Return'
]
DD_COLS = [
    'DD.period.t', 'DD.t.i', 'DD.i.e', 'DD.e.five',
    'DD.five.Shift.r', 'DD.Shift.r.o', 'DD.o.a', 'DD.a.n', 'DD.n.l', 'DD.l.Return'
]
UD_COLS = [
    'UD.period.t', 'UD.t.i', 'UD.i.e', 'UD.e.five',
    'UD.five.Shift.r', 'UD.Shift.r.o', 'UD.o.a', 'UD.a.n', 'UD.n.l', 'UD.l.Return'
]


def _short(name: str) -> str:
    """`H.Shift.r` -> `Shiftr`; `DD.l.Return` -> `lReturn`."""
    parts = name.split('.')[1:]
    return ''.join(parts)


FEAT_NAMES = (
    [f"key_{_short(c)}"     for c in H_COLS]   # 11 dwell  -> scored by GP
    + [f"digraph_{_short(c)}" for c in DD_COLS]  # 10 DD     -> scored by GP
    + [f"extra_{_short(c)}"   for c in UD_COLS]  # 10 UD     -> scored by GP
)


def load_cmu_dataset(csv_path=CSV_PATH):
    """
    Returns dict: subject -> ndarray of shape (n_reps, 31), values in milliseconds.
    """
    by_subj = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                vec = (
                    [float(row[c]) * 1000 for c in H_COLS]
                    + [float(row[c]) * 1000 for c in DD_COLS]
                    + [float(row[c]) * 1000 for c in UD_COLS]
                )
                by_subj[row['subject']].append(vec)
            except (ValueError, KeyError):
                continue
    return {s: np.array(v, dtype=np.float64) for s, v in by_subj.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics + algorithms (self-contained — no DB / no train_keystroke_rf import)
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(fpr[idx]), float(fnr[idx]), float(thr[idx])


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


def evaluate_classifier(model, X, y, n_splits=5):
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


def evaluate_one_class_svm(X, y, n_splits=5):
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


def evaluate_profile_matcher(X, y, feat_names, gen_subsample=None, seed=42):
    """
    LOOCV on the genuine class against a Gunetti-Picardi A+R reference set.
    For 100+ genuine reps, full LOOCV is ~100×99 set-matches per user — fine.
    `gen_subsample` truncates genuine to N reps if speed becomes a concern.
    """
    rng = np.random.default_rng(seed)
    genuine = X[y == 1]
    impostors = X[y == 0]
    if gen_subsample is not None and len(genuine) > gen_subsample:
        idx = rng.choice(len(genuine), size=gen_subsample, replace=False)
        genuine = genuine[idx]
    if len(genuine) < 4:
        return np.array([]), np.array([]), [], []

    profile_std = np.zeros(X.shape[1])
    y_true_all, y_score_all = [], []
    train_times, infer_times = [], []

    for i in range(len(genuine)):
        ref = [genuine[j] for j in range(len(genuine)) if j != i]
        train_times.append(0.0)
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


# ─────────────────────────────────────────────────────────────────────────────
#  Per-subject benchmark
# ─────────────────────────────────────────────────────────────────────────────

N_GENUINE   = 100   # first 100 reps of the target subject
N_IMP_PER   = 5     # first 5 reps from each of the other 50 subjects -> 250 imp


def build_subject_dataset(target, all_subjects):
    gen = all_subjects[target][:N_GENUINE]
    imp_rows = []
    for s, mat in all_subjects.items():
        if s == target:
            continue
        imp_rows.append(mat[:N_IMP_PER])
    imp = np.vstack(imp_rows) if imp_rows else np.empty((0, gen.shape[1]))
    X = np.vstack([gen, imp])
    y = np.array([1] * len(gen) + [0] * len(imp))
    return X, y, len(gen), len(imp)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Missing CMU CSV at {CSV_PATH}")
        print("   Run: python ml/load_cmu_impostors.py  (it will download the file)")
        return

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(out_dir, exist_ok=True)
    rows_path    = os.path.join(out_dir, 'keystroke_benchmark_cmu.csv')
    summary_path = os.path.join(out_dir, 'keystroke_benchmark_cmu_summary.csv')

    print("Loading CMU dataset ...")
    subjects = load_cmu_dataset()
    print(f"  {len(subjects)} subjects, "
          f"{sum(len(v) for v in subjects.values())} total reps, "
          f"{len(FEAT_NAMES)} features\n")

    all_rows = []
    for i, target in enumerate(sorted(subjects.keys()), start=1):
        X, y, n_gen, n_imp = build_subject_dataset(target, subjects)
        print(f"[{i:>2}/{len(subjects)}] {target}  genuine={n_gen}  impostor={n_imp}")

        for name, model in make_models().items():
            try:
                y_t, y_s, tt, it = evaluate_classifier(model, X, y)
                eer, far, frr, _ = compute_eer(y_t, y_s)
                auc = roc_auc_score(y_t, y_s)
                all_rows.append({
                    'subject': target, 'algorithm': name,
                    'n_genuine': n_gen, 'n_impostor': n_imp,
                    'features': X.shape[1],
                    'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr, 'auc': auc,
                    'train_time_s':  float(np.mean(tt)),
                    'infer_time_ms': float(np.mean(it)),
                })
                print(f"    {name:18s}  EER={eer:.3f}  AUC={auc:.3f}")
            except Exception as e:
                print(f"    {name:18s}  FAILED: {e}")

        try:
            y_t, y_s, tt, it = evaluate_one_class_svm(X, y)
            if len(y_t) > 0 and len(np.unique(y_t)) >= 2:
                eer, far, frr, _ = compute_eer(y_t, y_s)
                auc = roc_auc_score(y_t, y_s)
                all_rows.append({
                    'subject': target, 'algorithm': 'OneClassSVM',
                    'n_genuine': n_gen, 'n_impostor': n_imp,
                    'features': X.shape[1],
                    'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr, 'auc': auc,
                    'train_time_s':  float(np.mean(tt)) if tt else 0.0,
                    'infer_time_ms': float(np.mean(it)) if it else 0.0,
                })
                print(f"    {'OneClassSVM':18s}  EER={eer:.3f}  AUC={auc:.3f}")
        except Exception as e:
            print(f"    OneClassSVM       FAILED: {e}")

        try:
            y_t, y_s, tt, it = evaluate_profile_matcher(X, y, FEAT_NAMES)
            if len(y_t) > 0 and len(np.unique(y_t)) >= 2:
                eer, far, frr, _ = compute_eer(y_t, y_s)
                auc = roc_auc_score(y_t, y_s)
                all_rows.append({
                    'subject': target, 'algorithm': 'ProfileMatcher_GP',
                    'n_genuine': n_gen, 'n_impostor': n_imp,
                    'features': X.shape[1],
                    'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr, 'auc': auc,
                    'train_time_s':  float(np.mean(tt)) if tt else 0.0,
                    'infer_time_ms': float(np.mean(it)) if it else 0.0,
                })
                print(f"    {'ProfileMatcher_GP':18s}  EER={eer:.3f}  AUC={auc:.3f}")
        except Exception as e:
            print(f"    ProfileMatcher_GP FAILED: {e}")
        print()

    if not all_rows:
        print("No results produced.")
        return

    fields = list(all_rows[0].keys())
    with open(rows_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Per-subject results -> {rows_path}")

    by_algo = {}
    for r in all_rows:
        by_algo.setdefault(r['algorithm'], []).append(r)
    ranked = []
    for algo, rs in by_algo.items():
        eers = [r['eer'] for r in rs]
        aucs = [r['auc'] for r in rs]
        tts  = [r['train_time_s']  for r in rs]
        its  = [r['infer_time_ms'] for r in rs]
        ranked.append((algo, len(rs), float(np.mean(eers)), float(np.std(eers)),
                       float(np.mean(aucs)), float(np.mean(tts)), float(np.mean(its))))
    ranked.sort(key=lambda x: x[2])

    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'n_subjects', 'mean_eer', 'std_eer',
                    'mean_auc', 'mean_train_s', 'mean_infer_ms'])
        for r in ranked:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}",
                        f"{r[4]:.4f}", f"{r[5]:.3f}", f"{r[6]:.3f}"])

    print(f"Summary (ranked by EER) -> {summary_path}\n")
    print(f"{'Algorithm':<18} {'Subj':>5} {'meanEER':>9} {'stdEER':>9} "
          f"{'meanAUC':>9} {'train(s)':>10} {'infer(ms)':>10}")
    for r in ranked:
        print(f"{r[0]:<18} {r[1]:>5d} {r[2]:>9.4f} {r[3]:>9.4f} "
              f"{r[4]:>9.4f} {r[5]:>10.3f} {r[6]:>10.3f}")


if __name__ == "__main__":
    main()
