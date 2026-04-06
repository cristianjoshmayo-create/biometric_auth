"""
report_generator.py
─────────────────────────────────────────────────────────────────────────────
Report Generation Module — Penetration & Vulnerability Testing Framework

Generates:
  1. Console report            — formatted text for immediate review
  2. CSV export                — per-attack logs saved to security_results/
  3. Vulnerability summary     — ranked weaknesses with severity & recommendations
  4. Thesis-ready tables       — Chapter 5 attack success rate and metric tables
  5. Aggregate cross-user table — summary across all users (--all-users mode)
"""

import os
import csv
import time
from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
#  SEVERITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rate_severity(far: float) -> str:
    if far == 0.0:       return "✅ SECURE"
    elif far < 0.01:     return "🟢 LOW"
    elif far < 0.05:     return "🟡 MODERATE"
    elif far < 0.15:     return "🟠 HIGH"
    else:                return "🔴 CRITICAL"


def rate_vulnerability(sr: float) -> str:
    if sr == 0.0:        return "Not Vulnerable"
    elif sr < 0.05:      return "Marginally Vulnerable"
    elif sr < 0.20:      return "Vulnerable"
    elif sr < 0.50:      return "Highly Vulnerable"
    else:                return "CRITICAL — System Compromised"


def _sev_label(far: float) -> str:
    """Short label without emoji for table columns."""
    s = rate_severity(far)
    return s.split(" ", 1)[1] if " " in s else s


# ─────────────────────────────────────────────────────────────────────────────
#  CONSOLE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_console_report(results: Dict, username: str):
    print(f"\n{'█'*72}")
    print(f"  BIOMETRIC SECURITY PENETRATION TEST REPORT")
    print(f"  User      : {username}")
    print(f"  Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█'*72}")

    # ── Section 1: Keystroke ──────────────────────────────────────────────────
    ks = results.get("keystroke", {})
    if "error" not in ks:
        m = ks.get("metrics", {})
        print(f"\n{'═'*72}")
        print(f"  SECTION 1 — KEYSTROKE DYNAMICS SECURITY")
        print(f"{'═'*72}")
        print(f"\n  Aggregate Metrics (all attack types combined):")
        print(f"  {'Metric':<28} {'Value':>10}  Severity")
        print(f"  {'─'*28} {'─'*10}  {'─'*20}")
        print(f"  {'False Accept Rate (FAR)':<28} {m.get('far',0)*100:>9.2f}%  "
              f"{rate_severity(m.get('far',0))}")
        print(f"  {'False Reject Rate (FRR)':<28} {m.get('frr',0)*100:>9.2f}%")
        print(f"  {'Equal Error Rate (EER)':<28} {m.get('eer',0)*100:>9.2f}%")
        print(f"  {'Accuracy':<28} {m.get('accuracy',0)*100:>9.2f}%")
        print(f"  {'Attack Samples Tested':<28} {m.get('n_impostor',0):>10}")
        print(f"  {'Genuine Samples':<28} {m.get('n_genuine',0):>10}")

        print(f"\n  Attack Type Breakdown:")
        print(f"  {'Attack Type':<38} {'Attempts':>9} {'Passed':>7} "
              f"{'Success':>8}  Vulnerability")
        print(f"  {'─'*38} {'─'*9} {'─'*7} {'─'*8}  {'─'*24}")
        for atype, stat in ks.get("attack_types", {}).items():
            sr = stat.get("success_rate", 0)
            print(f"  {atype:<38} {stat['total']:>9} {stat['passed']:>7} "
                  f"{sr*100:>7.1f}%  {rate_vulnerability(sr)}")

    # ── Section 2: Voice ──────────────────────────────────────────────────────
    voice = results.get("voice", {})
    if "error" not in voice:
        m = voice.get("metrics", {})
        print(f"\n{'═'*72}")
        print(f"  SECTION 2 — VOICE BIOMETRICS SECURITY")
        print(f"{'═'*72}")
        print(f"\n  Aggregate Metrics:")
        print(f"  {'Metric':<28} {'Value':>10}  Severity")
        print(f"  {'─'*28} {'─'*10}  {'─'*20}")
        print(f"  {'False Accept Rate (FAR)':<28} {m.get('far',0)*100:>9.2f}%  "
              f"{rate_severity(m.get('far',0))}")
        print(f"  {'False Reject Rate (FRR)':<28} {m.get('frr',0)*100:>9.2f}%")
        print(f"  {'Equal Error Rate (EER)':<28} {m.get('eer',0)*100:>9.2f}%")
        print(f"  {'Accuracy':<28} {m.get('accuracy',0)*100:>9.2f}%")
        print(f"  {'Attack Samples Tested':<28} {m.get('n_impostor',0):>10}")

        print(f"\n  Attack Type Breakdown:")
        print(f"  {'Attack Type':<38} {'Attempts':>9} {'Passed':>7} "
              f"{'Success':>8}  Vulnerability")
        print(f"  {'─'*38} {'─'*9} {'─'*7} {'─'*8}  {'─'*24}")
        for atype, stat in voice.get("attack_types", {}).items():
            sr = stat.get("success_rate", 0)
            print(f"  {atype:<38} {stat['total']:>9} {stat['passed']:>7} "
                  f"{sr*100:>7.1f}%  {rate_vulnerability(sr)}")

    # ── Section 3: Multimodal ─────────────────────────────────────────────────
    mm = results.get("multimodal", {})
    if "error" not in mm:
        m      = mm.get("metrics", {})
        mm_far = mm.get("multimodal_far", 0)
        print(f"\n{'═'*72}")
        print(f"  SECTION 3 — MULTIMODAL FUSED SYSTEM "
              f"({mm.get('fusion_strategy','AND').upper()}-GATE)")
        print(f"{'═'*72}")
        print(f"\n  {'Metric':<34} {'Value':>10}  Severity")
        print(f"  {'─'*34} {'─'*10}  {'─'*20}")
        print(f"  {'Coordinated Attack Success Rate':<34} {mm_far*100:>9.2f}%  "
              f"{rate_severity(mm_far)}")
        print(f"  {'System FAR (fused)':<34} {m.get('far',0)*100:>9.2f}%  "
              f"{rate_severity(m.get('far',0))}")
        print(f"  {'System FRR (fused)':<34} {m.get('frr',0)*100:>9.2f}%")
        print(f"  {'System EER (fused)':<34} {m.get('eer',0)*100:>9.2f}%")
        print(f"  {'Attacks Attempted':<34} {mm.get('attacks_attempted',0):>10}")
        print(f"  {'Attacks Passed Both Gates':<34} {mm.get('attacks_passed',0):>10}")

    # ── Section 4: Threshold sensitivity ─────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  SECTION 4 — THRESHOLD SENSITIVITY")
    print(f"{'═'*72}")
    for label, data in [("Keystroke", ks), ("Voice", voice)]:
        sens = data.get("sensitivity", [])
        if not sens:
            continue
        print(f"\n  {label} — FAR/FRR across threshold range:")
        print(f"  {'Threshold':>10}  {'FAR':>8}  {'FRR':>8}  {'EER':>8}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
        for s in sens[::5]:
            print(f"  {s['threshold']:>10.2f}  "
                  f"{s.get('far',0)*100:>7.2f}%  "
                  f"{s.get('frr',0)*100:>7.2f}%  "
                  f"{s.get('eer',0)*100:>7.2f}%")

    # ── Section 5: Vulnerability summary ─────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  SECTION 5 — VULNERABILITY SUMMARY & RECOMMENDATIONS")
    print(f"{'═'*72}")
    vulns = _extract_vulnerabilities(results)
    if not vulns:
        print("\n  ✅ No significant vulnerabilities detected.")
    else:
        print(f"\n  {'#':<4} {'Vulnerability':<40} {'Severity':<10}  Recommendation")
        print(f"  {'─'*4} {'─'*40} {'─'*10}  {'─'*35}")
        for i, v in enumerate(vulns, 1):
            print(f"  {i:<4} {v['name']:<40} {v['severity']:<10}  "
                  f"{v['recommendation']}")

    print(f"\n{'█'*72}")
    print(f"  END OF REPORT")
    print(f"{'█'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  VULNERABILITY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_vulnerabilities(results: Dict) -> List[Dict]:
    vulns = []

    for atype, stat in results.get("keystroke", {}).get("attack_types", {}).items():
        sr = stat.get("success_rate", 0)
        if sr > 0:
            vulns.append({
                "name":           f"KS: {atype}",
                "severity":       _sev_label(sr),
                "success_rate":   sr,
                "recommendation": _recommend_ks(atype),
            })

    for atype, stat in results.get("voice", {}).get("attack_types", {}).items():
        sr = stat.get("success_rate", 0)
        if sr > 0:
            vulns.append({
                "name":           f"Voice: {atype}",
                "severity":       _sev_label(sr),
                "success_rate":   sr,
                "recommendation": _recommend_voice(atype),
            })

    mm_far = results.get("multimodal", {}).get("multimodal_far", 0)
    if mm_far > 0:
        vulns.append({
            "name":           "Multimodal: Coordinated Attack",
            "severity":       _sev_label(mm_far),
            "success_rate":   mm_far,
            "recommendation": "Enforce AND-gate; add liveness detection",
        })

    vulns.sort(key=lambda x: x["success_rate"], reverse=True)
    return vulns


def _recommend_ks(atype: str) -> str:
    return {
        "keystroke_replay":              "Add session nonces to prevent template replay",
        "keystroke_replay_jitter":       "Add session nonces; tighten quality check",
        "keystroke_synthetic_variation": "Tighten genuine augmentation noise spread",
        "keystroke_near_miss":           "Raise threshold floor; increase impostors",
        "keystroke_morph_50":            "Increase FAR-weight in threshold calibration",
        "keystroke_morph_80":            "Increase FAR-weight; add more training data",
        "keystroke_random":              "Raise minimum threshold floor",
        "keystroke_cross_user":          "Increase enrolled-user impostor training set",
        "keystroke_cross_user_morph":    "Tighten morph resistance via stricter penalty",
    }.get(atype, "Review threshold and augmentation pipeline")


def _recommend_voice(atype: str) -> str:
    return {
        "voice_replay":             "Add challenge phrase rotation / liveness check",
        "voice_replay_jitter":      "Add challenge rotation; tighten quality check",
        "voice_pitch_shift":        "Add vocal tract length normalisation features",
        "voice_time_stretch":       "Add duration variance filter",
        "voice_synthetic_close":    "Deploy anti-spoofing classifier (AASIST/RawNet2)",
        "voice_synthetic_far":      "Increase synthetic impostor training diversity",
        "voice_cross_user":         "Increase real enrolled-user impostor training",
    }.get(atype, "Review voice model training data diversity")


# ─────────────────────────────────────────────────────────────────────────────
#  CSV EXPORT  →  security_results/ folder
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(results: Dict, username: str, output_dir: str = "."):
    """Export per-section attack logs and aggregate summary to output_dir."""
    safe_user = username.replace("@", "_at_").replace(".", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    for section, data in results.items():
        logs = data.get("attack_log", [])
        if not logs:
            continue
        all_keys   = set()
        for log in logs: all_keys.update(log.keys())
        fieldnames = sorted(all_keys)
        path = os.path.join(output_dir,
                            f"{safe_user}_{section}_attacks_{timestamp}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for log in logs:
                w.writerow({k: log.get(k, "") for k in fieldnames})
        print(f"  📄 {path}")

    # Aggregate metrics summary
    summary_path = os.path.join(output_dir,
                                f"{safe_user}_security_summary_{timestamp}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Section", "Accuracy", "FAR", "FRR", "EER",
                    "Attack_Success_Rate", "N_Impostor", "N_Genuine"])
        for section, data in results.items():
            m = data.get("metrics", {})
            w.writerow([
                section,
                f"{m.get('accuracy',0)*100:.2f}%",
                f"{m.get('far',0)*100:.2f}%",
                f"{m.get('frr',0)*100:.2f}%",
                f"{m.get('eer',0)*100:.2f}%",
                f"{data.get('multimodal_far', m.get('attack_success_rate',0))*100:.2f}%",
                m.get("n_impostor", 0),
                m.get("n_genuine", 0),
            ])
    print(f"  📄 {summary_path}")
    return summary_path


# ─────────────────────────────────────────────────────────────────────────────
#  THESIS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_thesis_table(results: Dict):
    print(f"\n{'═'*72}")
    print(f"  THESIS TABLE — Chapter 5 Security Evaluation")
    print(f"{'═'*72}")

    # Attack success rates
    print(f"\n  Table 5.X — Attack Success Rates by Type")
    print(f"\n  {'Attack Type':<38} {'Modality':<12} {'Success':>8}  Severity")
    print(f"  {'─'*38} {'─'*12} {'─'*8}  {'─'*20}")

    rows = []
    for atype, stat in results.get("keystroke", {}).get("attack_types", {}).items():
        rows.append((atype, "Keystroke", stat.get("success_rate", 0)))
    for atype, stat in results.get("voice", {}).get("attack_types", {}).items():
        rows.append((atype, "Voice", stat.get("success_rate", 0)))
    mm_far = results.get("multimodal", {}).get("multimodal_far", 0)
    rows.append(("Coordinated Multimodal", "Both", mm_far))

    for atype, modality, sr in sorted(rows, key=lambda x: x[2], reverse=True):
        print(f"  {atype:<38} {modality:<12} {sr*100:>7.1f}%  {rate_severity(sr)}")

    # Fused system metrics
    print(f"\n  Table 5.Y — Fused System Metrics")
    print(f"\n  {'Metric':<20} {'Keystroke':>12} {'Voice':>12} {'Multimodal':>12}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")

    def _m(s, k): return results.get(s, {}).get("metrics", {}).get(k, 0)
    for label, key in [("Accuracy","accuracy"),("FAR","far"),
                       ("FRR","frr"),("EER","eer")]:
        print(f"  {label:<20} {_m('keystroke',key)*100:>11.2f}% "
              f"{_m('voice',key)*100:>11.2f}% "
              f"{_m('multimodal',key)*100:>11.2f}%")

    print(f"{'═'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  AGGREGATE CROSS-USER TABLE  (--all-users mode)
# ─────────────────────────────────────────────────────────────────────────────

def print_aggregate_security_table(summaries: List[Dict]):
    print(f"\n{'═'*72}")
    print(f"  AGGREGATE SECURITY SUMMARY — ALL USERS")
    print(f"{'═'*72}")
    print(f"\n  {'Username':<32} {'Status':<8} "
          f"{'KS FAR':>8} {'KS EER':>8} "
          f"{'V FAR':>7} {'V EER':>7} "
          f"{'MM FAR':>8}")
    print(f"  {'─'*32} {'─'*8} "
          f"{'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7} "
          f"{'─'*8}")
    for s in summaries:
        status = "✅ OK" if s.get("status") == "OK" else "❌ ERR"
        print(f"  {s['username']:<32} {status:<8} "
              f"{s.get('ks_far','N/A'):>8} {s.get('ks_eer','N/A'):>8} "
              f"{s.get('voice_far','N/A'):>7} {s.get('voice_eer','N/A'):>7} "
              f"{s.get('mm_far','N/A'):>8}")
    print(f"{'═'*72}\n")