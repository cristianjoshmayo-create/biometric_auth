# ml/benchmark_voice.py
#
# Compares speaker-verification feature/classifier combinations on the same
# enrolled-user data, so the thesis can defend the claim "we identified the
# most suitable algorithm for speech feature extraction and classification".
#
# Features compared:
#   - MFCC mean vector (from voice_templates.mfcc_features in DB)
#   - ECAPA-TDNN embedding (192-dim, from per-user voice_ecapa.pkl)
#
# Classifiers / scoring strategies:
#   - MFCC + Cosine to user mean
#   - MFCC + GMM (genuine-only, log-likelihood score)
#   - MFCC + SVM (RBF, genuine vs other-user impostors)
#   - MFCC + kNN (Manhattan, genuine vs impostors)
#   - ECAPA + Cosine (production)
#
# For each enrolled user we treat that user's enrollment voice samples as
# genuine and every other enrolled user's samples as impostors. We compute
# EER, FAR@EER, FRR@EER, ROC-AUC across pooled trials.
#
# Output: results/voice_benchmark.csv  +  results/voice_benchmark_summary.csv
#
# Run:  python ml/benchmark_voice.py

import os
import sys
import csv
import time
import pickle
import warnings

warnings.filterwarnings("ignore")

backend_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'
)
sys.path.insert(0, backend_path)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, roc_curve

from database.db import SessionLocal
from database.models import User, VoiceTemplate


def compute_eer(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(fpr[idx]), float(fnr[idx])


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_user_mfccs(db, user_id):
    rows = db.query(VoiceTemplate).filter(VoiceTemplate.user_id == user_id).all()
    out = []
    for r in rows:
        if r.mfcc_features and len(r.mfcc_features) > 0:
            out.append(np.asarray(r.mfcc_features, dtype=np.float64))
    if not out:
        return np.empty((0, 0))
    L = min(len(v) for v in out)
    return np.array([v[:L] for v in out])


def load_user_ecapa(username):
    safe = username.replace("@", "_at_").replace(".", "_").replace(" ", "_")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'models', f"{safe}_voice_ecapa.pkl")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        d = pickle.load(f)
    embs = d.get('embeddings')
    if embs is None or len(embs) == 0:
        return None
    return np.array([np.asarray(e, dtype=np.float64).flatten() for e in embs])


def loocv_cosine_to_mean(genuine, impostors):
    """For each genuine sample, score = cosine to mean of OTHER genuine samples."""
    y_t, y_s = [], []
    for i in range(len(genuine)):
        ref = np.delete(genuine, i, axis=0)
        if len(ref) == 0:
            continue
        ref_mean = ref.mean(axis=0)
        y_t.append(1); y_s.append(cosine(genuine[i], ref_mean))
    if len(genuine) >= 2:
        ref_mean_full = genuine.mean(axis=0)
        for v in impostors:
            y_t.append(0); y_s.append(cosine(v, ref_mean_full))
    return np.array(y_t), np.array(y_s)


def loocv_gmm(genuine, impostors, n_components=1):
    """GMM trained on genuine; score = log-likelihood."""
    if len(genuine) < 3:
        return np.array([]), np.array([])
    y_t, y_s = [], []
    for i in range(len(genuine)):
        train = np.delete(genuine, i, axis=0)
        scaler = StandardScaler().fit(train)
        train_s = scaler.transform(train)
        nc = min(n_components, max(1, len(train) - 1))
        try:
            gmm = GaussianMixture(n_components=nc, covariance_type='diag',
                                  reg_covar=1e-3, random_state=42).fit(train_s)
        except Exception:
            continue
        y_t.append(1)
        y_s.append(float(gmm.score(scaler.transform(genuine[i:i+1]))))
    if len(genuine) >= 3 and len(impostors) > 0:
        scaler = StandardScaler().fit(genuine)
        nc = min(n_components, len(genuine) - 1)
        gmm = GaussianMixture(n_components=nc, covariance_type='diag',
                              reg_covar=1e-3, random_state=42).fit(scaler.transform(genuine))
        for v in impostors:
            y_t.append(0)
            y_s.append(float(gmm.score(scaler.transform(v.reshape(1, -1)))))
    return np.array(y_t), np.array(y_s)


def loocv_supervised(genuine, impostors, clf_factory):
    """LOOCV on genuine + all impostors as class-0 training pool."""
    if len(genuine) < 3 or len(impostors) < 2:
        return np.array([]), np.array([])
    y_t, y_s = [], []
    imp = np.array(impostors)
    for i in range(len(genuine)):
        gen_train = np.delete(genuine, i, axis=0)
        X_tr = np.vstack([gen_train, imp])
        y_tr = np.array([1] * len(gen_train) + [0] * len(imp))
        scaler = StandardScaler().fit(X_tr)
        clf = clf_factory()
        clf.fit(scaler.transform(X_tr), y_tr)
        if hasattr(clf, 'predict_proba'):
            s = float(clf.predict_proba(scaler.transform(genuine[i:i+1]))[0, 1])
        else:
            s = float(clf.decision_function(scaler.transform(genuine[i:i+1]))[0])
        y_t.append(1); y_s.append(s)
    # Score impostors with model trained on full genuine + leave-one-impostor-out is overkill;
    # use a single model trained on all genuine for impostor scoring.
    X_tr = np.vstack([genuine, imp])
    y_tr = np.array([1] * len(genuine) + [0] * len(imp))
    scaler = StandardScaler().fit(X_tr)
    clf = clf_factory()
    clf.fit(scaler.transform(X_tr), y_tr)
    for v in impostors:
        if hasattr(clf, 'predict_proba'):
            s = float(clf.predict_proba(scaler.transform(v.reshape(1, -1)))[0, 1])
        else:
            s = float(clf.decision_function(scaler.transform(v.reshape(1, -1)))[0])
        y_t.append(0); y_s.append(s)
    return np.array(y_t), np.array(y_s)


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, 'voice_benchmark.csv')
    summary_path = os.path.join(out_dir, 'voice_benchmark_summary.csv')

    db = SessionLocal()
    try:
        users = db.query(User).all()
        print(f"Found {len(users)} users.\n")

        # Pre-load all user MFCC + ECAPA datasets
        user_mfcc = {}
        user_ecapa = {}
        for u in users:
            mf = load_user_mfccs(db, u.id)
            if mf.shape[0] >= 3:
                user_mfcc[u.username] = mf
            ec = load_user_ecapa(u.username)
            if ec is not None and len(ec) >= 3:
                user_ecapa[u.username] = ec

        print(f"Users with ≥3 MFCC samples : {len(user_mfcc)}")
        print(f"Users with ≥3 ECAPA samples: {len(user_ecapa)}\n")

        all_rows = []

        def add(user, algo, y_t, y_s, t_train=0.0, t_infer=0.0):
            if len(y_t) == 0 or len(np.unique(y_t)) < 2:
                return
            eer, far, frr = compute_eer(y_t, y_s)
            auc = roc_auc_score(y_t, y_s)
            all_rows.append({
                'user': user, 'algorithm': algo,
                'n_genuine': int((y_t == 1).sum()),
                'n_impostor': int((y_t == 0).sum()),
                'eer': eer, 'far_at_eer': far, 'frr_at_eer': frr,
                'auc': auc,
                'train_time_s': t_train, 'infer_time_ms': t_infer,
            })
            print(f"    {algo:22s}  EER={eer:.3f}  AUC={auc:.3f}")

        # ---- MFCC-based methods ----
        # Align MFCC dimensionality across users (truncate to min)
        if user_mfcc:
            min_mfcc_len = min(v.shape[1] for v in user_mfcc.values())
            user_mfcc = {k: v[:, :min_mfcc_len] for k, v in user_mfcc.items()}
            print(f"MFCC dim aligned to {min_mfcc_len}\n")

            for username, gen in user_mfcc.items():
                print(f"[MFCC] {username}  n_genuine={len(gen)}")
                imp = np.vstack([v for k, v in user_mfcc.items() if k != username]) \
                    if len(user_mfcc) > 1 else np.empty((0, gen.shape[1]))

                yt, ys = loocv_cosine_to_mean(gen, imp)
                add(username, 'MFCC + Cosine', yt, ys)

                yt, ys = loocv_gmm(gen, imp, n_components=1)
                add(username, 'MFCC + GMM', yt, ys)

                yt, ys = loocv_supervised(gen, imp,
                    lambda: SVC(kernel='rbf', C=1.0, gamma='scale',
                                probability=True, class_weight={0: 2, 1: 1},
                                random_state=42))
                add(username, 'MFCC + SVM_RBF', yt, ys)

                yt, ys = loocv_supervised(gen, imp,
                    lambda: KNeighborsClassifier(n_neighbors=3, metric='manhattan'))
                add(username, 'MFCC + kNN_Manhattan', yt, ys)
                print()

        # ---- ECAPA cosine ----
        if user_ecapa:
            for username, gen in user_ecapa.items():
                print(f"[ECAPA] {username}  n_genuine={len(gen)}")
                imp = np.vstack([v for k, v in user_ecapa.items() if k != username]) \
                    if len(user_ecapa) > 1 else np.empty((0, gen.shape[1]))
                yt, ys = loocv_cosine_to_mean(gen, imp)
                add(username, 'ECAPA + Cosine', yt, ys)
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

    by_algo = {}
    for r in all_rows:
        by_algo.setdefault(r['algorithm'], []).append(r)
    ranked = []
    for algo, rs in by_algo.items():
        eers = [r['eer'] for r in rs]
        aucs = [r['auc'] for r in rs]
        ranked.append((algo, len(rs), float(np.mean(eers)), float(np.std(eers)),
                       float(np.mean(aucs))))
    ranked.sort(key=lambda x: x[2])

    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'n_users', 'mean_eer', 'std_eer', 'mean_auc'])
        for r in ranked:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])

    print(f"\nPer-user results -> {rows_path}")
    print(f"Summary (ranked by EER) -> {summary_path}\n")
    print(f"{'Algorithm':<24} {'Users':>6} {'meanEER':>9} {'stdEER':>9} {'meanAUC':>9}")
    for r in ranked:
        print(f"{r[0]:<24} {r[1]:>6d} {r[2]:>9.4f} {r[3]:>9.4f} {r[4]:>9.4f}")


if __name__ == "__main__":
    main()
