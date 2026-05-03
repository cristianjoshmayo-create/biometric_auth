# ml/benchmark_session_replay.py
#
# Session-replay benchmark for the keystroke pipeline (RF + ProfileMatcher_GP).
#
# Why this exists
# ---------------
# benchmark_roc.py / benchmark_keystroke.py do LOOCV inside a single enrollment
# session and use synthetic + patched-CMU as the impostor pool. That tells us
# the algorithm can rank "genuine vs noise". It does NOT tell us what we see
# in production logs — that real users frequently fail the keystroke gate on
# day-2+ logins and are saved by the voice fusion step.
#
# This benchmark targets that operational gap directly:
#
#   train set     : enrollment samples only        (source == 'enrollment')
#   genuine test  : adaptive samples for that user (source != 'enrollment')   ← later-session drift
#   impostor test : OTHER users' samples (patched) ← real cross-user impostors, never seen during training
#
# It then reports
#   - ROC / AUC / EER at the threshold-free level
#   - Operational metrics at the production cutoffs:
#       deny    : score < 0.55
#       fusion  : 0.55 ≤ score < 0.80   ("saved by fusion" zone)
#       instant : score ≥ 0.80
#     for both genuine-adaptive and impostor populations.
#
# Outputs:
#   results/roc_data/sessrepl_<algo>_genuine.npz / _impostor.npz
#   results/figures/roc_session_replay.png
#   results/session_replay_summary.csv
#
# Run:
#   venv310/Scripts/python.exe ml/benchmark_session_replay.py
#   venv310/Scripts/python.exe ml/benchmark_session_replay.py --plot-only

import os
import sys
import csv
import argparse
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, 'results')
DATA_DIR = os.path.join(RESULTS, 'roc_data')
FIG_DIR = os.path.join(RESULTS, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'backend'))

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from database.db import SessionLocal
from database.models import User, KeystrokeTemplate
from utils.crypto import decrypt

from train_keystroke_rf import (
    FEATURE_NAMES,
    extract_feature_vector,
    load_cmu_impostors,
    generate_impostor_samples,
    get_active_digraphs,
    get_active_key_dwells,
    get_phrase_digraph_pairs,
    get_active_trigraphs,
    _is_quality_sample,
)
from keystroke_profile_matcher import compute_set_match_score
from benchmark_keystroke import patch_impostor_zeros, make_models


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading split by source
# ─────────────────────────────────────────────────────────────────────────────

def _load_split(db, user_id, extra_pairs, key_keys, flight_pair_keys, trigraph_keys):
    """Return (enrollment_vecs, adaptive_vecs) for a user, both filtered through
    the same quality gate as load_enrollment_samples."""
    rows = (
        db.query(KeystrokeTemplate)
        .filter(KeystrokeTemplate.user_id == user_id)
        .order_by(KeystrokeTemplate.sample_order.asc())
        .all()
    )
    enroll, adapt = [], []
    for t in rows:
        vec = extract_feature_vector(
            t, extra_keys=extra_pairs, key_keys=key_keys,
            flight_pair_keys=flight_pair_keys, trigraph_keys=trigraph_keys,
        )
        ok, _ = _is_quality_sample(vec)
        if not ok:
            continue
        if (t.source or 'enrollment') == 'enrollment':
            enroll.append(vec)
        else:
            adapt.append(vec)
    return enroll, adapt


def _load_other_user_samples(db, exclude_user_id):
    """Other users' samples (enrollment + adaptive), as base-length vectors.
    These are real cross-user impostors. They typed THEIR OWN phrases, so the
    target-phrase-specific columns will be zero and must be patched (same as
    the existing benchmark does) to avoid the 'zero == impostor' shortcut."""
    others = db.query(User).filter(User.id != exclude_user_id).all()
    out = []
    for u in others:
        rows = (
            db.query(KeystrokeTemplate)
            .filter(KeystrokeTemplate.user_id == u.id)
            .all()
        )
        for t in rows:
            vec = extract_feature_vector(t)  # base-length, no phrase expansion
            ok, _ = _is_quality_sample(vec)
            if ok:
                out.append(vec)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Per-user evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _pad(v, base_len, full_len):
    v = np.asarray(v, dtype=np.float64)
    if v.shape[0] >= full_len:
        return v[:full_len]
    out = np.zeros(full_len)
    out[:min(v.shape[0], base_len)] = v[:min(v.shape[0], base_len)]
    return out


def evaluate_user(db, user, cmu, other_raw):
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
    base_len = len(FEATURE_NAMES)
    full_len = len(full_feat_names)

    enroll, adapt = _load_split(
        db, user.id, extra_pairs, key_keys, flight_pair_keys, trigraph_keys,
    )
    if len(enroll) < 3 or len(adapt) < 1:
        return None  # not eligible for session-replay

    g_enroll = np.array([_pad(v, base_len, full_len) for v in enroll])
    g_adapt  = np.array([_pad(v, base_len, full_len) for v in adapt])
    p_mean = g_enroll.mean(axis=0)
    p_std  = g_enroll.std(axis=0) + 1e-9

    # Training impostors: CMU + other users' samples (patched), + synthetic to top up.
    real_imp_raw = np.array(
        [_pad(v, base_len, full_len) for v in cmu]
        + [_pad(v, base_len, full_len) for v in other_raw]
    ) if (cmu or other_raw) else np.empty((0, full_len))
    real_imp_arr = patch_impostor_zeros(real_imp_raw, full_feat_names, p_mean, p_std)

    # Test impostors = other users' samples ONLY, also patched. These ARE in the
    # training pool here; that biases the evaluation toward "easy" impostors.
    # To get a held-out impostor signal, we exclude other-users from training
    # and use them only at test time. CMU stays in training as background noise.
    cmu_only_raw = np.array([_pad(v, base_len, full_len) for v in cmu]) \
        if cmu else np.empty((0, full_len))
    cmu_only_arr = patch_impostor_zeros(cmu_only_raw, full_feat_names, p_mean, p_std)
    other_test_raw = np.array([_pad(v, base_len, full_len) for v in other_raw]) \
        if other_raw else np.empty((0, full_len))
    other_test_arr = patch_impostor_zeros(other_test_raw, full_feat_names, p_mean, p_std,
                                          seed=131)

    n_synth = max(0, max(400, len(g_enroll) * 80) - len(cmu_only_arr))
    syn = generate_impostor_samples(p_mean, p_std, n=n_synth,
                                    feat_names=full_feat_names) if n_synth > 0 else []
    syn_arr = np.array(syn) if syn else np.empty((0, full_len))

    X_train_imp = np.vstack(
        [a for a in (cmu_only_arr, syn_arr) if len(a)]
    ) if (len(cmu_only_arr) or len(syn_arr)) else np.empty((0, full_len))
    X_train = np.vstack([g_enroll, X_train_imp])
    y_train = np.array([1] * len(g_enroll) + [0] * len(X_train_imp))

    out = {'username': user.username,
           'n_enroll': len(g_enroll), 'n_adapt': len(g_adapt),
           'n_other': len(other_test_arr)}

    # ── RandomForest ──────────────────────────────────────────────────────
    rf = make_models()['RandomForest']
    rf.fit(X_train, y_train)
    rf_gen = rf.predict_proba(g_adapt)[:, 1] if len(g_adapt) else np.array([])
    rf_imp = rf.predict_proba(other_test_arr)[:, 1] if len(other_test_arr) else np.array([])
    out['rf_gen'] = rf_gen
    out['rf_imp'] = rf_imp

    # ── ProfileMatcher_GP ─────────────────────────────────────────────────
    profile_std = np.zeros(full_len)
    ref = [v for v in g_enroll]
    gp_gen = np.array([
        compute_set_match_score(v, full_feat_names, ref, profile_std)['score']
        for v in g_adapt
    ]) if len(g_adapt) else np.array([])
    gp_imp = np.array([
        compute_set_match_score(v, full_feat_names, ref, profile_std)['score']
        for v in other_test_arr
    ]) if len(other_test_arr) else np.array([])
    out['gp_gen'] = gp_gen
    out['gp_imp'] = gp_imp
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Run + save
# ─────────────────────────────────────────────────────────────────────────────

def run():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        cmu = load_cmu_impostors()

        rf_gen, rf_imp = [], []
        gp_gen, gp_imp = [], []
        per_user = []

        for u in users:
            others = _load_other_user_samples(db, u.id)
            res = evaluate_user(db, u, cmu, others)
            if res is None:
                print(f"  [skip] {u.username}  (insufficient enrollment/adaptive)")
                continue
            print(f"  [{u.username}]  enroll={res['n_enroll']}  adapt={res['n_adapt']}  "
                  f"other_imp={res['n_other']}  "
                  f"RF gen μ={res['rf_gen'].mean():.3f} imp μ={res['rf_imp'].mean():.3f}  "
                  f"GP gen μ={res['gp_gen'].mean():.3f} imp μ={res['gp_imp'].mean():.3f}")
            rf_gen.extend(res['rf_gen'].tolist())
            rf_imp.extend(res['rf_imp'].tolist())
            gp_gen.extend(res['gp_gen'].tolist())
            gp_imp.extend(res['gp_imp'].tolist())
            per_user.append(res)
    finally:
        db.close()

    def _save(name, gen, imp):
        y_t = np.array([1] * len(gen) + [0] * len(imp), dtype=np.int8)
        y_s = np.array(list(gen) + list(imp), dtype=np.float64)
        np.savez_compressed(os.path.join(DATA_DIR, f"{name}.npz"),
                            y_true=y_t, y_score=y_s)
        if len(np.unique(y_t)) >= 2:
            print(f"  saved {name}.npz  n_gen={len(gen)} n_imp={len(imp)}  "
                  f"AUC={roc_auc_score(y_t, y_s):.4f}")

    _save('sessrepl_RandomForest', rf_gen, rf_imp)
    _save('sessrepl_ProfileMatcher_GP', gp_gen, gp_imp)


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting + operational summary
# ─────────────────────────────────────────────────────────────────────────────

def _eer(y_t, y_s):
    fpr, tpr, _ = roc_curve(y_t, y_s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def _operational_split(scores, fusion_lo=0.55, fusion_hi=0.80):
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0
    deny = float(np.sum(np.asarray(scores) < fusion_lo)) / n
    fuse = float(np.sum((np.asarray(scores) >= fusion_lo)
                        & (np.asarray(scores) < fusion_hi))) / n
    inst = float(np.sum(np.asarray(scores) >= fusion_hi)) / n
    return deny, fuse, inst


def make_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, label, color in [
        ('sessrepl_RandomForest',     'RandomForest — session replay', '#1f77b4'),
        ('sessrepl_ProfileMatcher_GP','ProfileMatcher_GP — session replay', '#d62728'),
    ]:
        path = os.path.join(DATA_DIR, f"{name}.npz")
        if not os.path.exists(path):
            print(f"  [skip plot] missing {path}")
            continue
        z = np.load(path)
        y_t, y_s = z['y_true'], z['y_score']
        if len(np.unique(y_t)) < 2:
            print(f"  [skip plot] only one class in {name}")
            continue
        fpr, tpr, _ = roc_curve(y_t, y_s)
        auc = roc_auc_score(y_t, y_s)
        eer = _eer(y_t, y_s)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{label} (AUC={auc:.3f}, EER={eer:.3f})")

        gen = y_s[y_t == 1]
        imp = y_s[y_t == 0]
        gd, gf, gi = _operational_split(gen)
        id_, if_, ii_ = _operational_split(imp)
        rows.append({
            'algorithm': name.replace('sessrepl_', ''),
            'n_genuine': int((y_t == 1).sum()),
            'n_impostor': int((y_t == 0).sum()),
            'auc': auc, 'eer': eer,
            'gen_deny<0.55': gd, 'gen_fusion_0.55-0.80': gf, 'gen_instant>=0.80': gi,
            'imp_deny<0.55': id_, 'imp_fusion_0.55-0.80': if_, 'imp_instant>=0.80': ii_,
        })

    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, label='chance')
    ax.set_xlabel('False Acceptance Rate (impostor: other-users\' samples)')
    ax.set_ylabel('True Acceptance Rate (genuine: post-enrollment adaptive)')
    ax.set_title('ROC — Session-Replay\n'
                 'train=enrollment only, genuine test=adaptive, impostor test=other users')
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, 'roc_session_replay.png')
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"  -> {p}")

    # Operational summary
    p = os.path.join(RESULTS, 'session_replay_summary.csv')
    if rows:
        keys = list(rows[0].keys())
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"  -> {p}")
        print("\n=== Operational split (production thresholds) ===")
        print(f"{'algo':<22} {'AUC':>6} {'EER':>6}  "
              f"{'gen<0.55':>8} {'gen 0.55-0.80':>14} {'gen>=0.80':>10}  "
              f"{'imp<0.55':>8} {'imp 0.55-0.80':>14} {'imp>=0.80':>10}")
        for r in rows:
            print(f"{r['algorithm']:<22} {r['auc']:>6.3f} {r['eer']:>6.3f}  "
                  f"{r['gen_deny<0.55']:>8.2%} "
                  f"{r['gen_fusion_0.55-0.80']:>14.2%} "
                  f"{r['gen_instant>=0.80']:>10.2%}  "
                  f"{r['imp_deny<0.55']:>8.2%} "
                  f"{r['imp_fusion_0.55-0.80']:>14.2%} "
                  f"{r['imp_instant>=0.80']:>10.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    if not args.plot_only:
        run()
    print("\n=== Plotting ===")
    make_plots()
    print("\nDone.")


if __name__ == "__main__":
    main()
