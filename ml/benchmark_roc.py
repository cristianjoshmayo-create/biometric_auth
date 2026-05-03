# ml/benchmark_roc.py
#
# ROC + AUC plots for the production-deployed algorithms only:
#   - RandomForest        (keystroke, mature ≥11 samples)
#   - ProfileMatcher_GP   (keystroke, cold-start)
#   - ECAPA-TDNN + Cosine (voice)
#
# Each algorithm is evaluated on TWO datasets:
#   internal    — enrolled users (DB)
#   public      — CMU keystroke / LibriSpeech voice
#
# Outputs:
#   results/roc_data/<dataset>_<algo>.npz       (y_true, y_score, per_user_auc)
#   results/figures/roc_keystroke.png           (4 curves: RF/GP × internal/CMU)
#   results/figures/roc_voice.png               (2 curves: ECAPA × internal/LibriSpeech)
#   results/roc_summary.csv                     (algo, dataset, n_genuine, n_impostor, auc, eer)
#
# Usage:
#   venv310/Scripts/python.exe ml/benchmark_roc.py             # everything
#   venv310/Scripts/python.exe ml/benchmark_roc.py --skip-librispeech
#   venv310/Scripts/python.exe ml/benchmark_roc.py --plot-only # re-plot from cached .npz

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


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(fpr[idx]), float(tpr[idx])


def save_pool(name, y_true, y_score, per_user_auc):
    path = os.path.join(DATA_DIR, f"{name}.npz")
    np.savez_compressed(
        path,
        y_true=np.asarray(y_true, dtype=np.int8),
        y_score=np.asarray(y_score, dtype=np.float64),
        per_user_auc=np.asarray(per_user_auc, dtype=np.float64),
    )
    print(f"  saved -> {path}  (n={len(y_true)}, auc_pooled={roc_auc_score(y_true, y_score):.4f})")


# ─────────────────────────────────────────────────────────────────────────────
#  Keystroke INTERNAL  (RF + ProfileMatcher_GP)
# ─────────────────────────────────────────────────────────────────────────────

def run_keystroke_internal():
    print("\n=== Keystroke / INTERNAL ===")
    from database.db import SessionLocal
    from database.models import User
    from benchmark_keystroke import (
        build_user_dataset, evaluate_classifier, evaluate_profile_matcher, make_models,
    )

    rf_y_t, rf_y_s, rf_auc = [], [], []
    gp_y_t, gp_y_s, gp_auc = [], [], []

    db = SessionLocal()
    try:
        users = db.query(User).all()
        rf_pipe = make_models()['RandomForest']
        for u in users:
            ds = build_user_dataset(db, u)
            if ds is None:
                continue
            X, y, n_gen, n_real, n_syn, feat_names = ds
            print(f"  [{u.username}]  gen={n_gen}  imp={len(y)-n_gen}")

            try:
                yt, ys, _, _ = evaluate_classifier('RF', rf_pipe, X, y)
                if len(np.unique(yt)) >= 2:
                    rf_y_t.extend(yt.tolist()); rf_y_s.extend(ys.tolist())
                    rf_auc.append(roc_auc_score(yt, ys))
            except Exception as e:
                print(f"    RF failed: {e}")

            try:
                yt, ys, _, _ = evaluate_profile_matcher(X, y, feat_names)
                if len(yt) > 0 and len(np.unique(yt)) >= 2:
                    gp_y_t.extend(yt.tolist()); gp_y_s.extend(ys.tolist())
                    gp_auc.append(roc_auc_score(yt, ys))
            except Exception as e:
                print(f"    GP failed: {e}")
    finally:
        db.close()

    save_pool('keystroke_internal_RandomForest', rf_y_t, rf_y_s, rf_auc)
    save_pool('keystroke_internal_ProfileMatcher_GP', gp_y_t, gp_y_s, gp_auc)


# ─────────────────────────────────────────────────────────────────────────────
#  Keystroke PUBLIC (CMU)  (RF + ProfileMatcher_GP)
# ─────────────────────────────────────────────────────────────────────────────

def run_keystroke_cmu():
    print("\n=== Keystroke / CMU ===")
    from benchmark_keystroke_cmu import (
        load_cmu_dataset, build_subject_dataset, evaluate_classifier,
        evaluate_profile_matcher, make_models, FEAT_NAMES, CSV_PATH,
    )

    if not os.path.exists(CSV_PATH):
        print(f"  ❌ CMU CSV missing at {CSV_PATH}; skipping")
        return

    subjects = load_cmu_dataset()
    print(f"  loaded {len(subjects)} CMU subjects")

    rf_pipe = make_models()['RandomForest']
    rf_y_t, rf_y_s, rf_auc = [], [], []
    gp_y_t, gp_y_s, gp_auc = [], [], []

    for i, target in enumerate(sorted(subjects.keys()), start=1):
        X, y, n_gen, n_imp = build_subject_dataset(target, subjects)
        print(f"  [{i:>2}/{len(subjects)}] {target}  gen={n_gen}  imp={n_imp}")

        try:
            yt, ys, _, _ = evaluate_classifier(rf_pipe, X, y)
            if len(np.unique(yt)) >= 2:
                rf_y_t.extend(yt.tolist()); rf_y_s.extend(ys.tolist())
                rf_auc.append(roc_auc_score(yt, ys))
        except Exception as e:
            print(f"    RF failed: {e}")

        try:
            yt, ys, _, _ = evaluate_profile_matcher(X, y, FEAT_NAMES)
            if len(yt) > 0 and len(np.unique(yt)) >= 2:
                gp_y_t.extend(yt.tolist()); gp_y_s.extend(ys.tolist())
                gp_auc.append(roc_auc_score(yt, ys))
        except Exception as e:
            print(f"    GP failed: {e}")

    save_pool('keystroke_cmu_RandomForest', rf_y_t, rf_y_s, rf_auc)
    save_pool('keystroke_cmu_ProfileMatcher_GP', gp_y_t, gp_y_s, gp_auc)


# ─────────────────────────────────────────────────────────────────────────────
#  Voice INTERNAL  (ECAPA + Cosine)
# ─────────────────────────────────────────────────────────────────────────────

def run_voice_internal():
    print("\n=== Voice / INTERNAL ===")
    from database.db import SessionLocal
    from database.models import User
    from benchmark_voice import load_user_ecapa, loocv_cosine_to_mean

    db = SessionLocal()
    try:
        users = db.query(User).all()
        ecapa = {}
        for u in users:
            v = load_user_ecapa(u.username)
            if v is not None and len(v) >= 3:
                ecapa[u.username] = v
        print(f"  users with ≥3 ECAPA samples: {len(ecapa)}")

        y_t_all, y_s_all, per_auc = [], [], []
        for username, gen in ecapa.items():
            imp = np.vstack([v for k, v in ecapa.items() if k != username]) \
                if len(ecapa) > 1 else np.empty((0, gen.shape[1]))
            yt, ys = loocv_cosine_to_mean(gen, imp)
            if len(yt) > 0 and len(np.unique(yt)) >= 2:
                y_t_all.extend(yt.tolist()); y_s_all.extend(ys.tolist())
                per_auc.append(roc_auc_score(yt, ys))
                print(f"  [{username}]  AUC={per_auc[-1]:.4f}  n={len(yt)}")
    finally:
        db.close()

    save_pool('voice_internal_ECAPA', y_t_all, y_s_all, per_auc)


# ─────────────────────────────────────────────────────────────────────────────
#  Voice PUBLIC (LibriSpeech)  (ECAPA + Cosine)
# ─────────────────────────────────────────────────────────────────────────────

def run_voice_librispeech():
    print("\n=== Voice / LibriSpeech ===")
    DATASET_DIR = os.path.join(HERE, 'datasets', 'LibriSpeech', 'test-clean')
    if not os.path.isdir(DATASET_DIR):
        # also accept dev-clean as the existing script does
        DATASET_DIR = os.path.join(HERE, 'datasets', 'LibriSpeech', 'dev-clean')
    if not os.path.isdir(DATASET_DIR):
        print(f"  ❌ LibriSpeech not found under ml/datasets/LibriSpeech/{{test,dev}}-clean — skipping")
        return

    import soundfile as sf
    import librosa
    SAMPLE_RATE = 16000
    MAX_SECONDS = 8.0
    N_UTTERANCES = 10

    def load_audio(path):
        a, sr = sf.read(path, dtype='float32')
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SAMPLE_RATE:
            a = librosa.resample(a, orig_sr=sr, target_sr=SAMPLE_RATE)
        cap = int(MAX_SECONDS * SAMPLE_RATE)
        return a[:cap] if len(a) > cap else a

    speakers = sorted([
        s for s in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, s))
    ])
    print(f"  found {len(speakers)} speakers in {DATASET_DIR}")

    spk_audio = {}
    for si, spk in enumerate(speakers):
        sdir = os.path.join(DATASET_DIR, spk)
        flacs = []
        for chap in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, chap)
            if not os.path.isdir(cdir):
                continue
            for f in sorted(os.listdir(cdir)):
                if f.endswith('.flac'):
                    flacs.append(os.path.join(cdir, f))
                    if len(flacs) >= N_UTTERANCES:
                        break
            if len(flacs) >= N_UTTERANCES:
                break
        if len(flacs) < 3:
            continue
        audios = []
        for u in flacs:
            try:
                audios.append(load_audio(u))
            except Exception as e:
                print(f"    skip {os.path.basename(u)}: {e}")
        if len(audios) >= 3:
            spk_audio[spk] = audios
        if (si + 1) % 10 == 0:
            print(f"  audio loaded: {si+1}/{len(speakers)}")

    print(f"  loading ECAPA encoder ...")
    from voice_ecapa import _get_encoder, extract_embedding
    enc = _get_encoder()
    if enc is None:
        print("  ❌ ECAPA encoder unavailable — skipping")
        return

    spk_emb = {}
    for si, spk in enumerate(sorted(spk_audio.keys())):
        embs = []
        for a in spk_audio[spk]:
            try:
                e = extract_embedding(a, sr=SAMPLE_RATE)
                if e is not None:
                    embs.append(np.asarray(e, dtype=np.float64).flatten())
            except Exception as ex:
                print(f"    ECAPA skip {spk}: {ex}")
        if len(embs) >= 3:
            spk_emb[spk] = np.array(embs)
        if (si + 1) % 10 == 0:
            print(f"  embeddings: {si+1}/{len(spk_audio)}")
    spk_audio.clear()
    print(f"  speakers usable: {len(spk_emb)}")

    from benchmark_voice import loocv_cosine_to_mean
    y_t_all, y_s_all, per_auc = [], [], []
    for spk, gen in spk_emb.items():
        imp = np.vstack([v for k, v in spk_emb.items() if k != spk])
        yt, ys = loocv_cosine_to_mean(gen, imp)
        if len(yt) > 0 and len(np.unique(yt)) >= 2:
            y_t_all.extend(yt.tolist()); y_s_all.extend(ys.tolist())
            per_auc.append(roc_auc_score(yt, ys))

    save_pool('voice_librispeech_ECAPA', y_t_all, y_s_all, per_auc)


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _load(name):
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z['y_true'], z['y_score'], z['per_user_auc']


def _curve(ax, name, label, color, linestyle='-'):
    z = _load(name)
    if z is None:
        print(f"  [skip] no data for {name}")
        return None
    yt, ys, _ = z
    fpr, tpr, _ = roc_curve(yt, ys)
    auc = roc_auc_score(yt, ys)
    eer, eer_fpr, eer_tpr = compute_eer(yt, ys)
    ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2,
            label=f"{label}  (AUC={auc:.4f}, EER={eer:.4f})")
    ax.plot([eer_fpr], [eer_tpr], 'o', color=color, markersize=6,
            markeredgecolor='black', markeredgewidth=0.6)
    return auc, eer, len(yt)


def make_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary_rows = []

    # Keystroke figure
    fig, ax = plt.subplots(figsize=(7, 6))
    res = _curve(ax, 'keystroke_internal_RandomForest',
                 'RandomForest — internal', '#1f77b4', '-')
    if res: summary_rows.append(('RandomForest', 'internal', res[2], res[0], res[1]))
    res = _curve(ax, 'keystroke_cmu_RandomForest',
                 'RandomForest — CMU public', '#1f77b4', '--')
    if res: summary_rows.append(('RandomForest', 'CMU', res[2], res[0], res[1]))
    res = _curve(ax, 'keystroke_internal_ProfileMatcher_GP',
                 'ProfileMatcher_GP — internal', '#d62728', '-')
    if res: summary_rows.append(('ProfileMatcher_GP', 'internal', res[2], res[0], res[1]))
    res = _curve(ax, 'keystroke_cmu_ProfileMatcher_GP',
                 'ProfileMatcher_GP — CMU public', '#d62728', '--')
    if res: summary_rows.append(('ProfileMatcher_GP', 'CMU', res[2], res[0], res[1]))

    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, label='chance')
    ax.set_xlabel('False Acceptance Rate (FAR)')
    ax.set_ylabel('True Acceptance Rate (1 − FRR)')
    ax.set_title('ROC — Keystroke (RandomForest, ProfileMatcher_GP)')
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, 'roc_keystroke.png')
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"  -> {p}")

    # Voice figure
    fig, ax = plt.subplots(figsize=(7, 6))
    res = _curve(ax, 'voice_internal_ECAPA',
                 'ECAPA-TDNN — internal', '#2ca02c', '-')
    if res: summary_rows.append(('ECAPA', 'internal', res[2], res[0], res[1]))
    res = _curve(ax, 'voice_librispeech_ECAPA',
                 'ECAPA-TDNN — LibriSpeech public', '#2ca02c', '--')
    if res: summary_rows.append(('ECAPA', 'LibriSpeech', res[2], res[0], res[1]))

    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, label='chance')
    ax.set_xlabel('False Acceptance Rate (FAR)')
    ax.set_ylabel('True Acceptance Rate (1 − FRR)')
    ax.set_title('ROC — Voice (ECAPA-TDNN + Cosine)')
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, 'roc_voice.png')
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"  -> {p}")

    # Zoomed top-left view (keystroke) — useful when AUC ≈ 1
    fig, ax = plt.subplots(figsize=(7, 6))
    for nm, lbl, c, ls in [
        ('keystroke_internal_RandomForest',     'RF — internal',   '#1f77b4', '-'),
        ('keystroke_cmu_RandomForest',          'RF — CMU',        '#1f77b4', '--'),
        ('keystroke_internal_ProfileMatcher_GP','GP — internal',   '#d62728', '-'),
        ('keystroke_cmu_ProfileMatcher_GP',     'GP — CMU',        '#d62728', '--'),
        ('voice_internal_ECAPA',                'ECAPA — internal','#2ca02c', '-'),
        ('voice_librispeech_ECAPA',             'ECAPA — LibriSpeech','#2ca02c','--'),
    ]:
        z = _load(nm)
        if z is None: continue
        yt, ys, _ = z
        fpr, tpr, _ = roc_curve(yt, ys)
        ax.plot(fpr, tpr, color=c, linestyle=ls, linewidth=2,
                label=f"{lbl} (AUC={roc_auc_score(yt, ys):.4f})")
    ax.plot([0, 0.2], [0.8, 1.0], 'k:', linewidth=0.6)
    ax.set_xlabel('FAR (zoomed)')
    ax.set_ylabel('1 − FRR (zoomed)')
    ax.set_title('ROC — top-left zoom (FAR ≤ 0.2, TAR ≥ 0.8)')
    ax.set_xlim(0, 0.2); ax.set_ylim(0.8, 1.001)
    ax.grid(alpha=0.3); ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, 'roc_zoom.png')
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"  -> {p}")

    # Summary CSV
    p = os.path.join(RESULTS, 'roc_summary.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'dataset', 'n_trials', 'auc', 'eer'])
        for r in summary_rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.4f}", f"{r[4]:.4f}"])
    print(f"  -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--skip-internal-keystroke', action='store_true')
    ap.add_argument('--skip-cmu', action='store_true')
    ap.add_argument('--skip-internal-voice', action='store_true')
    ap.add_argument('--skip-librispeech', action='store_true')
    args = ap.parse_args()

    if not args.plot_only:
        if not args.skip_internal_keystroke:
            run_keystroke_internal()
        if not args.skip_cmu:
            run_keystroke_cmu()
        if not args.skip_internal_voice:
            run_voice_internal()
        if not args.skip_librispeech:
            run_voice_librispeech()

    print("\n=== Plotting ===")
    make_plots()
    print("\nDone.")


if __name__ == "__main__":
    main()
