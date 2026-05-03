# ml/benchmark_fused.py
#
# Fused end-to-end system benchmark — applies the production multimodal
# decision rule (backend/routers/auth.py /fuse) to paired cross-user
# (keystroke, voice) score pools and reports system-level FAR/FRR/EER.
#
# Why this exists
# ---------------
# Per-modality benchmarks (keystroke_benchmark_crossuser.csv, voice_benchmark.csv)
# tell us how each modality performs alone. They do NOT tell us how the
# DEPLOYED system performs end-to-end, because the deployed system applies a
# routing rule:
#   ks >= 0.80                  -> ACCEPT (voice skipped)
#   ks <  0.55 && hard-veto      -> DENY  (voice cannot recover)
#   else                         -> /fuse: weighted ks+voice with case A/B
#                                   thresholds, voice floor, ks hard-veto
#
# Methodology (Option A — score-pair Monte Carlo, paired by identity)
# -------------------------------------------------------------------
# For each enrolled user u with both keystroke and ECAPA voice data:
#   GENUINE attempts: cross-user-LOOCV ks genuine scores paired (random,
#       seeded) with u's LOOCV-cosine voice genuine scores. Number of pairs
#       = max(len_ks_g, len_v_g) using sampling-with-replacement on the
#       smaller pool.
#   IMPOSTOR attempts: for each impostor user v != u, ks scores of v's
#       samples against u's content-indep RF paired (random, seeded) with
#       cosine of v's ECAPA samples against u's ECAPA mean. Pair count per
#       impostor = max(n_ks, n_voice) using sampling-with-replacement on
#       the smaller pool. Pairs are paired-by-impostor-identity, not pooled
#       randomly across impostors — preserves the realistic constraint
#       that one attacker contributes both modalities of one attempt.
#
# Decision rule applied to each (ks, voice) pair (mirrors /fuse exactly):
#   - per_user_threshold = 0.55 (Ramp default; matches deployed value)
#   - if ks >= 0.80                              -> ACCEPT (instant grant)
#   - keystroke_passed = (ks >= per_user_threshold)
#   - voice_floor: voice must be >= 0.40 to fuse
#   - Case A (ks_passed):  weights 0.45/0.55, fused threshold 0.58
#   - Case B (ks_failed):  weights 0.35/0.65, fused threshold 0.65,
#                          plus ks hard-veto: ks < 0.45 -> DENY
#   - ks_reliability is treated as 1.0 (default; per-user calibration is a
#     secondary effect documented in the thesis as future work).
#
# Outputs:
#   results/fused_system_summary.csv  — operating-point FAR/FRR + EER curve
#   results/fused_system_pairs.csv    — every (user, label, ks, voice,
#                                       fused, decision) pair
#
# Run:
#   venv310/Scripts/python.exe ml/benchmark_fused.py

import os
import sys
import csv
import pickle
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'backend'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

from database.db import SessionLocal
from database.models import User, VoiceTemplate
from train_keystroke_rf import FEATURE_NAMES, load_enrollment_samples

# Same content-independent subset used by benchmark_keystroke_crossuser.py
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
KEEP_IDX = [FEATURE_NAMES.index(n) for n in CONTENT_INDEP_FEATURES]

# Production decision-rule constants (mirror backend/routers/auth.py)
PER_USER_THRESHOLD = 0.55  # Ramp default (deployed)
INSTANT_GRANT      = 0.80
VOICE_FLOOR        = 0.40
KS_HARD_VETO       = 0.45
CASE_A_WEIGHTS     = (0.45, 0.55)  # (ks, voice)
CASE_A_THRESHOLD   = 0.58
CASE_B_WEIGHTS     = (0.35, 0.65)
CASE_B_THRESHOLD   = 0.65

RNG = np.random.default_rng(42)


def project(vec):
    v = np.asarray(vec, dtype=np.float64)
    base_len = len(FEATURE_NAMES)
    if v.shape[0] < base_len:
        out = np.zeros(base_len)
        out[:v.shape[0]] = v
        v = out
    return v[KEEP_IDX]


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_user_ecapa(username):
    safe = username.replace("@", "_at_").replace(".", "_").replace(" ", "_")
    path = os.path.join(HERE, 'models', f"{safe}_voice_ecapa.pkl")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        d = pickle.load(f)
    embs = d.get('embeddings')
    if embs is None or len(embs) == 0:
        return None
    out = []
    for e in embs:
        v = e['vec'] if isinstance(e, dict) else e
        out.append(np.asarray(v, dtype=np.float64).flatten())
    return np.array(out)


def make_rf():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=3,
            class_weight={0: 2, 1: 1}, random_state=42, n_jobs=-1,
        )),
    ])


def ks_genuine_loocv(g_vec, imp_vec):
    """LOOCV: held-out genuine scored by RF trained on rest+imp."""
    out = []
    for i in range(len(g_vec)):
        keep = np.array([j for j in range(len(g_vec)) if j != i])
        X_tr = np.vstack([g_vec[keep], imp_vec])
        y_tr = np.array([1] * len(keep) + [0] * len(imp_vec))
        m = make_rf()
        m.fit(X_tr, y_tr)
        out.append(float(m.predict_proba(g_vec[i:i+1])[0, 1]))
    return np.array(out)


def ks_per_impostor_scores(g_vec, per_user, target_uid):
    """For each impostor user, return their per-sample RF scores against
    the target's content-indep RF (trained on full target genuine + all
    other-user impostor samples)."""
    target = per_user[target_uid]['vectors']
    imp_all = np.vstack([d['vectors'] for k, d in per_user.items() if k != target_uid])
    X_tr = np.vstack([target, imp_all])
    y_tr = np.array([1] * len(target) + [0] * len(imp_all))
    m = make_rf()
    m.fit(X_tr, y_tr)
    out = {}
    for uid, d in per_user.items():
        if uid == target_uid:
            continue
        s = m.predict_proba(d['vectors'])[:, 1]
        out[uid] = np.asarray(s, dtype=np.float64)
    return out


def voice_genuine_loocv(g_emb):
    out = []
    for i in range(len(g_emb)):
        ref = np.delete(g_emb, i, axis=0)
        if len(ref) == 0:
            continue
        out.append(cosine(g_emb[i], ref.mean(axis=0)))
    return np.clip(np.array(out), 0.0, 1.0)


def voice_per_impostor_scores(g_emb, per_user_voice, target_user):
    ref_mean = g_emb.mean(axis=0)
    out = {}
    for u, embs in per_user_voice.items():
        if u == target_user:
            continue
        scores = np.array([cosine(e, ref_mean) for e in embs])
        out[u] = np.clip(scores, 0.0, 1.0)
    return out


def pair_scores(a, b, n=None):
    """Random pairing with replacement on the smaller pool. Returns aligned arrays."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.empty(0), np.empty(0)
    if n is None:
        n = max(len(a), len(b))
    ai = RNG.integers(0, len(a), n) if len(a) < n else RNG.permutation(len(a))[:n]
    bi = RNG.integers(0, len(b), n) if len(b) < n else RNG.permutation(len(b))[:n]
    return a[ai], b[bi]


def production_decision(ks, voice):
    """Return (granted: bool, fused_score: float, case: str).
    Mirrors backend/routers/auth.py /fuse exactly. ks_reliability assumed 1.0."""
    ks = float(np.clip(ks, 0.0, 1.0))
    voice = float(np.clip(voice, 0.0, 1.0))
    if ks >= INSTANT_GRANT:
        return True, ks, "instant_grant"
    keystroke_passed = (ks >= PER_USER_THRESHOLD)
    if keystroke_passed:
        ks_w, v_w = CASE_A_WEIGHTS
        thr = CASE_A_THRESHOLD
        case = "A_passed"
        ks_vetoed = False
    else:
        ks_w, v_w = CASE_B_WEIGHTS
        thr = CASE_B_THRESHOLD
        case = "B_failed"
        ks_vetoed = (ks < KS_HARD_VETO)
    fused = ks_w * ks + v_w * voice
    voice_ok = (voice >= VOICE_FLOOR)
    granted = (fused >= thr) and voice_ok and (not ks_vetoed)
    return bool(granted), float(fused), case


def continuous_score(ks, voice):
    """Monotone-ish continuous score for an EER sweep: max of (ks, fused-A).
    Hard constraints (voice floor, ks hard-veto) zero the score so they
    cannot be undone by lowering the threshold.

    This is the score the parametric EER curve operates on. It is NOT the
    deployed decision rule; it is a single-threshold proxy for sweeping.
    """
    ks = float(np.clip(ks, 0.0, 1.0))
    voice = float(np.clip(voice, 0.0, 1.0))
    fused_a = CASE_A_WEIGHTS[0] * ks + CASE_A_WEIGHTS[1] * voice
    s = max(ks, fused_a)
    if voice < VOICE_FLOOR and ks < INSTANT_GRANT:
        s = 0.0
    if ks < KS_HARD_VETO and ks < PER_USER_THRESHOLD:
        s = 0.0
    return s


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(fpr[idx]), float(fnr[idx])


def main():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        print(f"Found {len(users)} users.")
        per_user = {}
        per_user_voice = {}
        for u in users:
            samples = load_enrollment_samples(db, u.id)
            ec = load_user_ecapa(u.username)
            if len(samples) >= 3 and ec is not None and len(ec) >= 2:
                per_user[u.id] = {
                    'username': u.username,
                    'vectors': np.array([project(v) for v in samples]),
                }
                per_user_voice[u.username] = ec
        print(f"{len(per_user)} users have BOTH >=3 keystroke and >=2 ECAPA samples.\n")
    finally:
        db.close()

    if len(per_user) < 2:
        print("Need at least 2 users.")
        return

    pair_rows = []  # full (user, label, ks, voice, fused, granted, case)

    for uid, info in per_user.items():
        u = info['username']
        g_ks_vec = info['vectors']
        imp_ks_vec = np.vstack([d['vectors'] for k, d in per_user.items() if k != uid])
        g_voice = per_user_voice[u]
        print(f"[{u}] ks_g={len(g_ks_vec)} ks_imp={len(imp_ks_vec)} voice_g={len(g_voice)}")

        # Genuine ks/voice score pools for u
        ks_g_scores    = ks_genuine_loocv(g_ks_vec, imp_ks_vec)
        voice_g_scores = voice_genuine_loocv(g_voice)

        # Pair genuine attempts (random with seed)
        ks_g_paired, v_g_paired = pair_scores(ks_g_scores, voice_g_scores)
        for ks, vs in zip(ks_g_paired, v_g_paired):
            granted, fused, case = production_decision(ks, vs)
            pair_rows.append({
                'user': u, 'label': 1, 'impostor_id': '',
                'ks': float(ks), 'voice': float(vs),
                'fused': fused, 'granted': int(granted), 'case': case,
                'cont_score': continuous_score(ks, vs),
            })

        # Impostor pools — paired by impostor identity
        ks_imp_per = ks_per_impostor_scores(g_ks_vec, per_user, uid)
        voice_imp_per = voice_per_impostor_scores(g_voice, per_user_voice, u)
        common = set(ks_imp_per.keys()) & {
            uid2 for uid2, d in per_user.items() if d['username'] in voice_imp_per
        }
        for v_uid in common:
            v_username = per_user[v_uid]['username']
            ks_v = ks_imp_per[v_uid]
            voice_v = voice_imp_per[v_username]
            ks_p, vs_p = pair_scores(ks_v, voice_v)
            for ks, vs in zip(ks_p, vs_p):
                granted, fused, case = production_decision(ks, vs)
                pair_rows.append({
                    'user': u, 'label': 0, 'impostor_id': v_username,
                    'ks': float(ks), 'voice': float(vs),
                    'fused': fused, 'granted': int(granted), 'case': case,
                    'cont_score': continuous_score(ks, vs),
                })

    if not pair_rows:
        print("No pairs produced.")
        return

    # Write per-pair detail
    pairs_path = os.path.join(RESULTS, 'fused_system_pairs.csv')
    with open(pairs_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        for r in pair_rows:
            r2 = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(r2)
    print(f"\nPer-pair detail -> {pairs_path}")

    # Aggregate: as-deployed FAR/FRR + parametric EER
    y_true = np.array([r['label'] for r in pair_rows])
    y_grant = np.array([r['granted'] for r in pair_rows])
    y_score = np.array([r['cont_score'] for r in pair_rows])

    n_g = int((y_true == 1).sum())
    n_i = int((y_true == 0).sum())
    far_dep = float(((y_true == 0) & (y_grant == 1)).sum() / max(n_i, 1))
    frr_dep = float(((y_true == 1) & (y_grant == 0)).sum() / max(n_g, 1))

    eer, far_eer, frr_eer = compute_eer(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    # Always-fused-A baseline EER (no routing): single linear score
    y_fa = np.array([
        CASE_A_WEIGHTS[0] * r['ks'] + CASE_A_WEIGHTS[1] * r['voice']
        for r in pair_rows
    ])
    eer_fa, far_fa, frr_fa = compute_eer(y_true, y_fa)
    auc_fa = float(roc_auc_score(y_true, y_fa))

    summary_path = os.path.join(RESULTS, 'fused_system_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value', 'notes'])
        w.writerow(['n_genuine_pairs', n_g, ''])
        w.writerow(['n_impostor_pairs', n_i, ''])
        w.writerow(['n_users', len(per_user), ''])
        w.writerow(['as_deployed_FAR', f"{far_dep:.6f}",
                    f"impostors granted by production rule "
                    f"(thresholds 0.55/0.80, /fuse case A T=0.58, B T=0.65)"])
        w.writerow(['as_deployed_FRR', f"{frr_dep:.6f}",
                    'genuine denied by production rule'])
        w.writerow(['parametric_EER',           f"{eer:.6f}",
                    'sweep on max(ks, 0.45*ks+0.55*voice) with floors/veto enforced'])
        w.writerow(['parametric_FAR_at_EER',    f"{far_eer:.6f}", ''])
        w.writerow(['parametric_FRR_at_EER',    f"{frr_eer:.6f}", ''])
        w.writerow(['parametric_AUC',           f"{auc:.6f}", ''])
        w.writerow(['always_fused_EER',         f"{eer_fa:.6f}",
                    'baseline: 0.45*ks+0.55*voice, no routing'])
        w.writerow(['always_fused_FAR_at_EER',  f"{far_fa:.6f}", ''])
        w.writerow(['always_fused_FRR_at_EER',  f"{frr_fa:.6f}", ''])
        w.writerow(['always_fused_AUC',         f"{auc_fa:.6f}", ''])

    print(f"Summary -> {summary_path}\n")
    print("=== FUSED END-TO-END SYSTEM ===")
    print(f"  users={len(per_user)}  n_genuine_pairs={n_g}  n_impostor_pairs={n_i}")
    print(f"  As-deployed (production rule):")
    print(f"      FAR = {far_dep*100:.3f}%   FRR = {frr_dep*100:.3f}%")
    print(f"  Parametric (sweep continuous score):")
    print(f"      EER = {eer*100:.3f}%   AUC = {auc:.4f}")
    print(f"  Always-fused-A baseline (no routing):")
    print(f"      EER = {eer_fa*100:.3f}%   AUC = {auc_fa:.4f}")


if __name__ == "__main__":
    main()
