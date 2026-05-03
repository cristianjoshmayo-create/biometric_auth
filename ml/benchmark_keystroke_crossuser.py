# ml/benchmark_keystroke_crossuser.py
#
# Cross-user-only keystroke benchmark — the honest "how often does another
# enrolled user get into your account?" number.
#
# Why this exists
# ---------------
# benchmark_keystroke.py reports near-zero EER on the internal dataset, but
# inflates that result with (a) synthetic impostors drawn from the genuine
# user's own (mu, sigma) and (b) phrase-specific feature columns that other
# users literally don't have. This benchmark removes both:
#
#   - impostor pool        : OTHER enrolled users' samples ONLY
#                            (no CMU, no synthetic)
#   - feature set          : content-independent global aggregates ONLY
#                            (dwell/flight/p2p/rhythm/etc. — drop every
#                             digraph_* / extra_* / trigraph_* / per-pair
#                             flight_* column so phrase content can't leak)
#   - genuine evaluation   : leave-one-out (per-user)
#
# Output:
#   results/keystroke_benchmark_crossuser.csv
#   results/keystroke_benchmark_crossuser_summary.csv
#
# Run:
#   venv310/Scripts/python.exe ml/benchmark_keystroke_crossuser.py

import os
import sys
import csv
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'backend'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve

from database.db import SessionLocal
from database.models import User
from train_keystroke_rf import FEATURE_NAMES, load_enrollment_samples
from keystroke_profile_matcher import compute_set_match_score


# Content-independent feature subset: drop every column that depends on the
# specific phrase being typed. These are global aggregates only.
CONTENT_INDEP_FEATURES = [
    'dwell_mean', 'dwell_std', 'dwell_median', 'dwell_min', 'dwell_max',
    'flight_mean', 'flight_std', 'flight_median',
    'p2p_mean', 'p2p_std',
    'r2r_mean', 'r2r_std',
    'typing_speed_cpm', 'typing_duration',
    'rhythm_mean', 'rhythm_std', 'rhythm_cv',
    'pause_count', 'pause_mean',
    'backspace_ratio', 'backspace_count',
    'hand_alternation_ratio', 'same_hand_sequence_mean',
    'finger_transition_ratio', 'seek_time_mean', 'seek_time_count',
    'dwell_mean_norm', 'dwell_std_norm',
    'flight_mean_norm', 'flight_std_norm',
    'p2p_std_norm', 'r2r_mean_norm',
    'shift_lag_norm',
]

# Indices in the base FEATURE_NAMES vector that we keep.
KEEP_IDX = [FEATURE_NAMES.index(n) for n in CONTENT_INDEP_FEATURES]


def project(vec):
    """Project a base-length feature vector onto the content-independent subset."""
    v = np.asarray(vec, dtype=np.float64)
    base_len = len(FEATURE_NAMES)
    if v.shape[0] < base_len:
        # pad short vectors (older rows) with zeros, then project
        out = np.zeros(base_len)
        out[:v.shape[0]] = v
        v = out
    return v[KEEP_IDX]


def compute_eer(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(fpr[idx]), float(fnr[idx])


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
        'kNN_Manhattan': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5, metric='manhattan')),
        ]),
        'LogReg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight={0: 2, 1: 1},
                                       max_iter=1000, random_state=42)),
        ]),
    }


def evaluate_classifier(model, g, imp):
    """Per-genuine LOOCV: train on (g_minus_one + imp), test the held-out
    genuine sample plus all impostors. Score = predict_proba[:, 1]."""
    y_true_all, y_score_all = [], []
    loo = LeaveOneOut()
    for tr_idx, te_idx in loo.split(g):
        X_tr = np.vstack([g[tr_idx], imp])
        y_tr = np.array([1] * len(tr_idx) + [0] * len(imp))
        model.fit(X_tr, y_tr)
        # held-out genuine
        s_g = model.predict_proba(g[te_idx])[:, 1]
        y_true_all.extend([1] * len(te_idx))
        y_score_all.extend(s_g.tolist())
    # Score impostors once on the full-genuine model (cheap, stable estimate).
    X_full = np.vstack([g, imp])
    y_full = np.array([1] * len(g) + [0] * len(imp))
    model.fit(X_full, y_full)
    s_i = model.predict_proba(imp)[:, 1]
    y_true_all.extend([0] * len(imp))
    y_score_all.extend(s_i.tolist())
    return np.array(y_true_all), np.array(y_score_all)


def evaluate_profile_matcher(g, imp, feat_names):
    """LOOCV genuine + full-set impostor scoring with Gunetti-Picardi A+R."""
    profile_std = np.zeros(len(feat_names))
    y_true_all, y_score_all = [], []
    for i in range(len(g)):
        ref = [g[j] for j in range(len(g)) if j != i]
        r = compute_set_match_score(g[i], feat_names, ref, profile_std)
        y_true_all.append(1)
        y_score_all.append(r['score'])
    ref_full = [v for v in g]
    for v in imp:
        r = compute_set_match_score(v, feat_names, ref_full, profile_std)
        y_true_all.append(0)
        y_score_all.append(r['score'])
    return np.array(y_true_all), np.array(y_score_all)


def main():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        print(f"Found {len(users)} users.\n")

        # Load each user's projected enrollment vectors once.
        per_user = {}
        for u in users:
            samples = load_enrollment_samples(db, u.id)
            if len(samples) >= 3:
                per_user[u.id] = {
                    'username': u.username,
                    'vectors': np.array([project(v) for v in samples]),
                }
        print(f"{len(per_user)} users have >=3 enrollment samples.\n")
    finally:
        db.close()

    if len(per_user) < 2:
        print("Need at least 2 users with samples for cross-user evaluation.")
        return

    feat_names = list(CONTENT_INDEP_FEATURES)
    rows = []
    score_pool = {}  # algo -> (y_true_list, y_score_list) pooled across users

    for uid, info in per_user.items():
        g = info['vectors']
        # Impostor pool = ALL OTHER users' samples
        imp = np.vstack([d['vectors'] for k, d in per_user.items() if k != uid])
        if len(imp) < 3:
            continue
        print(f"[{info['username']}]  genuine={len(g)}  impostor={len(imp)}")

        for name, model in make_models().items():
            try:
                y_t, y_s = evaluate_classifier(model, g, imp)
                eer, far, frr = compute_eer(y_t, y_s)
                auc = roc_auc_score(y_t, y_s)
                rows.append({
                    'user': info['username'], 'algorithm': name,
                    'n_genuine': len(g), 'n_impostor': len(imp),
                    'features': len(feat_names),
                    'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr,
                    'auc': auc,
                })
                pool = score_pool.setdefault(name, ([], []))
                pool[0].extend(y_t.tolist()); pool[1].extend(y_s.tolist())
                print(f"    {name:18s}  EER={eer:.4f}  AUC={auc:.4f}")
            except Exception as e:
                print(f"    {name:18s}  FAILED: {e}")

        try:
            y_t, y_s = evaluate_profile_matcher(g, imp, feat_names)
            if len(np.unique(y_t)) >= 2:
                eer, far, frr = compute_eer(y_t, y_s)
                auc = roc_auc_score(y_t, y_s)
                rows.append({
                    'user': info['username'], 'algorithm': 'ProfileMatcher_GP',
                    'n_genuine': len(g), 'n_impostor': len(imp),
                    'features': len(feat_names),
                    'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr,
                    'auc': auc,
                })
                pool = score_pool.setdefault('ProfileMatcher_GP', ([], []))
                pool[0].extend(y_t.tolist()); pool[1].extend(y_s.tolist())
                print(f"    {'ProfileMatcher_GP':18s}  EER={eer:.4f}  AUC={auc:.4f}")
        except Exception as e:
            print(f"    ProfileMatcher_GP  FAILED: {e}")
        print()

    if not rows:
        print("No results produced.")
        return

    rows_path = os.path.join(RESULTS, 'keystroke_benchmark_crossuser.csv')
    summary_path = os.path.join(RESULTS, 'keystroke_benchmark_crossuser_summary.csv')
    threshold_path = os.path.join(RESULTS, 'keystroke_benchmark_crossuser_thresholds.csv')

    fields = list(rows[0].keys())
    with open(rows_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Per-user results -> {rows_path}")

    by_algo = {}
    for r in rows:
        by_algo.setdefault(r['algorithm'], []).append(r)
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'n_users', 'mean_eer', 'std_eer',
                    'min_eer', 'max_eer',
                    'mean_far', 'mean_frr', 'mean_auc'])
        ranked = []
        for algo, rs in by_algo.items():
            eers = [r['eer'] for r in rs]
            fars = [r['far_at_eer'] for r in rs]
            frrs = [r['frr_at_eer'] for r in rs]
            aucs = [r['auc'] for r in rs]
            ranked.append((algo, len(rs),
                           np.mean(eers), np.std(eers),
                           np.min(eers), np.max(eers),
                           np.mean(fars), np.mean(frrs), np.mean(aucs)))
        ranked.sort(key=lambda x: x[2])
        for row in ranked:
            w.writerow([row[0], row[1],
                        f"{row[2]:.4f}", f"{row[3]:.4f}",
                        f"{row[4]:.4f}", f"{row[5]:.4f}",
                        f"{row[6]:.4f}", f"{row[7]:.4f}", f"{row[8]:.4f}"])

    print(f"Summary (ranked by EER) -> {summary_path}")
    print("\n=== CROSS-USER-ONLY RANKED SUMMARY (mean / std / min / max EER across users) ===")
    print(f"{'Algorithm':<18} {'Users':>6} {'meanEER':>9} {'stdEER':>8} "
          f"{'minEER':>8} {'maxEER':>8} {'meanAUC':>9}")
    for r in ranked:
        print(f"{r[0]:<18} {r[1]:>6d} {r[2]:>9.4f} {r[3]:>8.4f} "
              f"{r[4]:>8.4f} {r[5]:>8.4f} {r[8]:>9.4f}")

    # ── Threshold analysis at deployed production cutoffs ────────────────
    # The production decision rule (backend/routers/auth.py) uses two
    # thresholds on the keystroke score:
    #   < 0.55  -> deny
    #   0.55-0.79 -> route to voice fusion
    #   >= 0.80 -> grant access immediately
    # FAR/FRR at these cutoffs tell us what genuine pass-rate and impostor
    # accept-rate the deployed system is actually operating at.
    THRESHOLDS = [0.55, 0.80]
    thr_rows = []
    print("\n=== FAR / FRR @ DEPLOYED THRESHOLDS (pooled across users) ===")
    print(f"{'Algorithm':<18} {'thr':>5}   "
          f"{'FAR (impostor accept)':>22}  {'FRR (genuine reject)':>22}")
    for algo, (yt, ys) in score_pool.items():
        yt = np.asarray(yt); ys = np.asarray(ys)
        if len(np.unique(yt)) < 2:
            continue
        gen = ys[yt == 1]; imp = ys[yt == 0]
        for t in THRESHOLDS:
            far = float((imp >= t).mean())  # impostor accepted = score >= thr
            frr = float((gen <  t).mean())  # genuine rejected = score <  thr
            thr_rows.append({
                'algorithm': algo, 'threshold': t,
                'far_at_thr': far, 'frr_at_thr': frr,
                'n_genuine': int(len(gen)), 'n_impostor': int(len(imp)),
            })
            print(f"{algo:<18} {t:>5.2f}   {far:>22.4f}  {frr:>22.4f}")
    if thr_rows:
        with open(threshold_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(thr_rows[0].keys()))
            w.writeheader()
            for r in thr_rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"\nThreshold table -> {threshold_path}")


if __name__ == "__main__":
    main()
