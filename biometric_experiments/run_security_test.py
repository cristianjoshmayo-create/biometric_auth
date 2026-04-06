"""
run_security_tests.py
─────────────────────────────────────────────────────────────────────────────
BIOMETRIC SECURITY PENETRATION TESTING FRAMEWORK
Chapter 5 — Security Evaluation & Vulnerability Analysis

Usage
─────
  Single user : python run_security_tests.py <username>
  All users   : python run_security_tests.py --all-users

Options
───────
  --fusion and|or|weighted   Fusion strategy (default: and)
  --n N                      Attack samples per type (default: 50)
  --all-users                Run against every enrolled user in the DB
  --min-samples N            Min enrollment samples to include a user (default: 3)
  --no-csv                   Skip CSV export

Output  →  security_results/  subfolder
──────
  <user>_keystroke_attacks_<ts>.csv
  <user>_voice_attacks_<ts>.csv
  <user>_multimodal_attacks_<ts>.csv
  <user>_security_summary_<ts>.csv
  all_users_security_summary.csv      (--all-users only)

Security note
─────────────
  All attacks are simulated in FEATURE SPACE using stored model parameters.
  No keystrokes are recorded and no audio is captured.
  For academic security evaluation only.
"""

import sys
import os
import argparse
import warnings
import csv
import time
import numpy as np
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
BACKEND_PATH = os.path.join(ROOT_DIR, "backend")
ML_PATH      = os.path.join(ROOT_DIR, "ml")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "security_results")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, BACKEND_PATH)
sys.path.insert(0, ML_PATH)

from attack_simulator import KeystrokeAttacker, VoiceAttacker, MultimodalAttacker
from auth_tester      import (AuthenticationTester, KeystrokeModelWrapper,
                               VoiceModelWrapper)
from report_generator import (print_console_report, export_csv,
                               print_thesis_table,
                               print_aggregate_security_table)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_filename(u): return u.replace("@","_at_").replace(".","_").replace(" ","_")


def find_model_paths(username):
    d = os.path.join(ML_PATH, "models")
    s = _safe_filename(username)
    return (os.path.join(d, f"{s}_keystroke_rf.pkl"),
            os.path.join(d, f"{s}_voice_cnn.pkl"))


# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_all_users(min_ks=3, min_voice=1):
    try:
        from database.db import SessionLocal
        from database.models import User, KeystrokeTemplate, VoiceTemplate
        from sqlalchemy import func
        db = SessionLocal()
        try:
            ks_c = dict(db.query(KeystrokeTemplate.user_id,
                                  func.count(KeystrokeTemplate.id))
                          .group_by(KeystrokeTemplate.user_id).all())
            vc   = dict(db.query(VoiceTemplate.user_id,
                                  func.count(VoiceTemplate.id))
                          .group_by(VoiceTemplate.user_id).all())
            out  = []
            for u in db.query(User).order_by(User.id).all():
                kn, vn = ks_c.get(u.id, 0), vc.get(u.id, 0)
                kp, vp = find_model_paths(u.username)
                if kn >= min_ks and vn >= min_voice and os.path.exists(kp) and os.path.exists(vp):
                    out.append({"username": u.username,
                                "ks_samples": kn, "voice_samples": vn})
            return out
        finally:
            db.close()
    except Exception as e:
        print(f"  ❌ DB query failed: {e}")
        return []


def load_genuine_samples(username, ks_model_data, voice_model_data):
    try:
        from database.db import SessionLocal
        from database.models import User
        from train_keystroke_rf import (load_enrollment_samples,
                                        get_active_digraphs, FEATURE_NAMES)
        from train_voice_cnn import load_enrollment_samples as load_voice

        phrase     = ks_model_data.get("phrase", "")
        extra_keys = get_active_digraphs(phrase)[1] if phrase else []
        feat_names = ks_model_data.get("feature_names", [])
        inactive   = {f for f in FEATURE_NAMES
                      if f.startswith("digraph_") and f not in feat_names}
        drop_idx   = [i for i, n in enumerate(FEATURE_NAMES) if n in inactive]

        def _strip(v):
            b = np.delete(np.asarray(v)[:len(FEATURE_NAMES)], drop_idx)
            e = np.asarray(v)[len(FEATURE_NAMES):]
            return np.concatenate([b, e])

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                return [], []
            ks_vecs    = load_enrollment_samples(db, user.id, extra_keys=extra_keys)
            voice_vecs, _ = load_voice(db, user.id)
        finally:
            db.close()

        return [_strip(v) for v in ks_vecs], list(voice_vecs)
    except Exception as e:
        print(f"  ⚠  Genuine samples failed: {e}")
        return [], []


def load_cross_user_impostors(target_username, ks_model_data):
    """Load real other-user samples — the most realistic attack scenario."""
    try:
        from database.db import SessionLocal
        from database.models import User
        from train_keystroke_rf import (load_enrollment_samples,
                                        get_active_digraphs, FEATURE_NAMES)
        from train_voice_cnn import load_enrollment_samples as load_voice

        phrase     = ks_model_data.get("phrase", "")
        extra_keys = get_active_digraphs(phrase)[1] if phrase else []
        feat_names = ks_model_data.get("feature_names", [])
        inactive   = {f for f in FEATURE_NAMES
                      if f.startswith("digraph_") and f not in feat_names}
        drop_idx   = [i for i, n in enumerate(FEATURE_NAMES) if n in inactive]

        def _strip(v):
            b = np.delete(np.asarray(v)[:len(FEATURE_NAMES)], drop_idx)
            e = np.asarray(v)[len(FEATURE_NAMES):]
            return np.concatenate([b, e])

        db = SessionLocal()
        try:
            others = db.query(User).filter(User.username != target_username).all()
            ks_imp, v_imp, log = [], [], []
            for u in others:
                kv    = load_enrollment_samples(db, u.id, extra_keys=extra_keys)
                vv, _ = load_voice(db, u.id)
                if kv: ks_imp.extend([_strip(v) for v in kv])
                if vv: v_imp.extend(vv)
                if kv or vv:
                    log.append(f"    {u.username}: {len(kv)} ks / {len(vv)} voice")
        finally:
            db.close()

        if log:
            print(f"  Cross-user impostors from {len(log)} user(s):")
            for l in log: print(l)

        return ks_imp, v_imp
    except Exception as e:
        print(f"  ⚠  Cross-user load failed: {e}")
        return [], []


# ─────────────────────────────────────────────────────────────────────────────
#  ATTACK BATTERY
# ─────────────────────────────────────────────────────────────────────────────

def build_attack_battery(ks_att, v_att, mm_att,
                          genuine_ks, genuine_voice,
                          cross_ks, cross_voice, n=50):
    print(f"\n  Building attack battery ({n} samples/type) ...")
    rng = np.random.default_rng(99)

    # ── Keystroke ──────────────────────────────────────────────────────────────
    ks = {}
    if genuine_ks:
        ks["keystroke_replay"]        = ks_att.replay_attack(genuine_ks, n=n, jitter_pct=0.00)
        ks["keystroke_replay_jitter"] = ks_att.replay_attack(genuine_ks, n=n, jitter_pct=0.03)
    ks["keystroke_synthetic_variation"] = ks_att.synthetic_variation(n=n, noise_pct=0.10)
    ks["keystroke_near_miss"]           = ks_att.synthetic_variation(n=n, noise_pct=0.03)
    synth = ks_att.profile_mean + rng.normal(0, ks_att.profile_std * 1.5)
    ks["keystroke_morph_50"]  = ks_att.statistical_morph(synth, blend_ratio=0.50, n=n)
    ks["keystroke_morph_80"]  = ks_att.statistical_morph(synth, blend_ratio=0.80, n=n)
    ks["keystroke_random"]    = ks_att.random_impostor(n=n)
    if cross_ks:
        ks["keystroke_cross_user"]       = ks_att.replay_attack(cross_ks, n=min(n, len(cross_ks)))
        ks["keystroke_cross_user_morph"] = ks_att.statistical_morph(
            cross_ks[rng.integers(0, len(cross_ks))], blend_ratio=0.50, n=n)

    # ── Voice ──────────────────────────────────────────────────────────────────
    va = {}
    if genuine_voice:
        va["voice_replay"]        = v_att.replay_attack(genuine_voice, n=n, jitter_pct=0.00)
        va["voice_replay_jitter"] = v_att.replay_attack(genuine_voice, n=n, jitter_pct=0.03)
    sv = v_att.cmvn_mean + rng.normal(0, v_att.cmvn_std * 1.2)
    va["voice_pitch_shift"]      = v_att.pitch_shift_attack(sv, semitone_range=3.0, n=n)
    va["voice_time_stretch"]     = v_att.time_stretch_attack(sv, n=n)
    va["voice_synthetic_close"]  = v_att.synthetic_voice(n=n, proximity=0.7)
    va["voice_synthetic_far"]    = v_att.synthetic_voice(n=n, proximity=0.2)
    if cross_voice:
        va["voice_cross_user"] = v_att.replay_attack(cross_voice, n=min(n, len(cross_voice)))

    # ── Multimodal ─────────────────────────────────────────────────────────────
    mm = []
    if genuine_ks and genuine_voice:
        mm += mm_att.coordinated_best(genuine_ks, genuine_voice, n=n)
    mm += mm_att.independent_pair(genuine_ks or [], n=n)
    if "keystroke_near_miss" in ks and "voice_synthetic_close" in va:
        mm += list(zip(ks["keystroke_near_miss"][:n],
                       va["voice_synthetic_close"][:n]))
    if cross_ks and cross_voice and "keystroke_cross_user" in ks:
        mm += list(zip(ks["keystroke_cross_user"][:n],
                       va.get("voice_cross_user", va["voice_synthetic_close"])[:n]))

    tk = sum(len(v) for v in ks.values())
    tv = sum(len(v) for v in va.values())
    print(f"  ✅ {tk} keystroke attacks across {len(ks)} types")
    print(f"  ✅ {tv} voice attacks across {len(va)} types")
    print(f"  ✅ {len(mm)} multimodal attack pairs")
    return ks, va, mm


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE USER RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_for_user(username, fusion, n_per_type, no_csv):
    print(f"\n{'█'*72}")
    print(f"  PENETRATION TEST  —  {username}")
    print(f"  Fusion: {fusion.upper()}-gate  |  {n_per_type} attacks/type")
    print(f"{'█'*72}")

    ks_path, v_path = find_model_paths(username)
    ks_w = KeystrokeModelWrapper(username, model_path=ks_path)
    v_w  = VoiceModelWrapper(username, model_path=v_path)

    if ks_w.model_data is None and v_w.model_data is None:
        print(f"  ❌ No trained models found. Train first, then re-run.")
        return {"username": username, "status": "NO_MODEL"}

    # Profiles
    ks_mean   = ks_w.model_data["profile_mean"]
    ks_std    = ks_w.model_data["profile_std"]
    feat_names= ks_w.feat_names
    cm_mean   = v_w.model_data["profile_mean"]
    cm_std    = v_w.model_data["profile_std"]
    raw_mean  = v_w.model_data.get("raw_profile_mean", cm_mean[:36])
    raw_std   = v_w.model_data.get("raw_profile_std",  cm_std[:36])

    print(f"\n  KS features: {len(feat_names)}  |  Voice features: {len(cm_mean)}")

    print(f"\n  Loading genuine enrollment samples ...")
    g_ks, g_v = load_genuine_samples(username, ks_w.model_data, v_w.model_data)
    print(f"  Genuine KS: {len(g_ks)}  |  Voice: {len(g_v)}")

    print(f"\n  Loading cross-user real impostors ...")
    c_ks, c_v = load_cross_user_impostors(username, ks_w.model_data)
    print(f"  Cross-user KS: {len(c_ks)}  |  Voice: {len(c_v)}")

    ks_att = KeystrokeAttacker(ks_mean, ks_std, feat_names, rng_seed=42)
    v_att  = VoiceAttacker(raw_mean, raw_std, cm_mean, cm_std, rng_seed=43)
    mm_att = MultimodalAttacker(ks_att, v_att)

    ks_atks, v_atks, mm_pairs = build_attack_battery(
        ks_att, v_att, mm_att, g_ks, g_v, c_ks, c_v, n=n_per_type)

    tester  = AuthenticationTester(ks_model=ks_w, voice_model=v_w,
                                    fusion_strategy=fusion)
    print(f"\n  Running security test battery ...")
    results = tester.run_full_battery(
        ks_attacks_by_type    = ks_atks,
        voice_attacks_by_type = v_atks,
        multimodal_pairs      = mm_pairs,
        genuine_ks            = g_ks,
        genuine_voice         = g_v,
    )

    print_console_report(results, username)
    print_thesis_table(results)

    if not no_csv:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"\n  Exporting CSVs → {RESULTS_DIR}/")
        export_csv(results, username, output_dir=RESULTS_DIR)

    def _m(s, k): return results.get(s, {}).get("metrics", {}).get(k, 0)
    return {
        "username":       username,
        "status":         "OK",
        "ks_far":         f"{_m('keystroke','far')*100:.2f}%",
        "ks_frr":         f"{_m('keystroke','frr')*100:.2f}%",
        "ks_eer":         f"{_m('keystroke','eer')*100:.2f}%",
        "voice_far":      f"{_m('voice','far')*100:.2f}%",
        "voice_frr":      f"{_m('voice','frr')*100:.2f}%",
        "voice_eer":      f"{_m('voice','eer')*100:.2f}%",
        "mm_far":         f"{results.get('multimodal',{}).get('multimodal_far',0)*100:.2f}%",
        "mm_eer":         f"{_m('multimodal','eer')*100:.2f}%",
        "fusion":         fusion,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("username",      nargs="?", default=None)
    parser.add_argument("--fusion",      default="and",
                        choices=["and", "or", "weighted"])
    parser.add_argument("--n",           type=int, default=50)
    parser.add_argument("--all-users",   action="store_true")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--no-csv",      action="store_true")
    args = parser.parse_args()

    if args.all_users:
        print(f"\n{'█'*72}")
        print(f"  SECURITY FRAMEWORK — ALL USERS")
        print(f"  Fusion: {args.fusion.upper()}-gate  |  "
              f"Min samples: {args.min_samples}")
        print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'█'*72}")

        users = get_all_users(min_ks=args.min_samples, min_voice=1)
        if not users:
            print("  ❌ No eligible users with trained models found.")
            sys.exit(1)

        print(f"\n  {len(users)} eligible user(s):")
        print(f"  {'#':<4} {'Username':<36} {'KS':>5} {'Voice':>6}")
        print(f"  {'─'*4} {'─'*36} {'─'*5} {'─'*6}")
        for i, u in enumerate(users, 1):
            print(f"  {i:<4} {u['username']:<36} "
                  f"{u['ks_samples']:>5} {u['voice_samples']:>6}")

        summaries, failed = [], []
        for i, u in enumerate(users, 1):
            print(f"\n\n  ── USER {i}/{len(users)}: {u['username']} ──")
            try:
                s = run_for_user(u["username"], args.fusion,
                                 args.n, args.no_csv)
                summaries.append(s)
                if s.get("status") != "OK":
                    failed.append(u["username"])
            except Exception as e:
                print(f"  ❌ Error: {e}")
                failed.append(u["username"])
                summaries.append({"username": u["username"],
                                  "status": "ERROR"})

        # Save aggregate
        if summaries:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            path = os.path.join(RESULTS_DIR, "all_users_security_summary.csv")
            fieldnames = list(summaries[0].keys())
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for s in summaries:
                    w.writerow({k: s.get(k, "") for k in fieldnames})
            print(f"\n  📄 All-users summary → {path}")
            print_aggregate_security_table(summaries)

        print(f"\n{'█'*72}")
        print(f"  COMPLETE — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Processed: {len(users)}  |  Failed: {len(failed)}")
        print(f"  Results  : {RESULTS_DIR}/")
        print(f"{'█'*72}\n")

    else:
        username = args.username or input("Enter username: ").strip()
        run_for_user(username, args.fusion, args.n, args.no_csv)
        print(f"\n  Results saved to: {RESULTS_DIR}/\n")


if __name__ == "__main__":
    main()