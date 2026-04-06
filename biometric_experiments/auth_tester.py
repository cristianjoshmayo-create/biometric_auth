"""
auth_tester.py
─────────────────────────────────────────────────────────────────────────────
Authentication Testing Module — Penetration & Vulnerability Testing Framework

Tests authentication models against attack vectors and computes security
metrics:  Accuracy, FAR, FRR, EER, and threshold sensitivity curves.

Supports:
  - Individual model testing (keystroke or voice)
  - Fused authentication (AND-gate, OR-gate)
  - Threshold sensitivity sweeps
  - Attack success rate by attack type
"""

import sys
import os
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Tuple, Optional, Callable, Union
from attack_simulator import AttackResult


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADER  — wraps production predict functions
# ─────────────────────────────────────────────────────────────────────────────

class KeystrokeModelWrapper:
    """
    Wraps the production predict_keystroke() function.
    Falls back to a sklearn Pipeline if the production model is unavailable.
    """

    def __init__(self, username: str, model_path: Optional[str] = None):
        self.username   = username
        self.model_data = None
        self.pipeline   = None
        self.feat_names = None
        self.threshold  = 0.55
        self._load(model_path)

    def _load(self, model_path: Optional[str]):
        import pickle
        if model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model_data = pickle.load(f)
            self.pipeline   = self.model_data["pipeline"]
            self.feat_names = self.model_data["feature_names"]
            self.threshold  = self.model_data["threshold"]
            print(f"  ✅ Keystroke model loaded from {model_path}")
        else:
            print(f"  ⚠  No keystroke model file — using production predict_keystroke()")

    def score(self, feature_vector: np.ndarray) -> float:
        """Return genuine probability score [0, 1]."""
        if self.pipeline is not None:
            vec = np.asarray(feature_vector, dtype=np.float64)
            if len(vec) != len(self.feat_names):
                vec = vec[:len(self.feat_names)]
            return float(self.pipeline.predict_proba(vec.reshape(1, -1))[0][1])
        # Fallback: use production function via feature dict
        try:
            from train_keystroke_rf import predict_keystroke
            feat_dict = {n: float(feature_vector[i])
                         for i, n in enumerate(self.feat_names or [])}
            result = predict_keystroke(self.username, feat_dict)
            return result.get("confidence", 0.0)
        except Exception as e:
            print(f"  ⚠  Keystroke score fallback failed: {e}")
            return 0.0

    def decide(self, feature_vector: np.ndarray) -> bool:
        return self.score(feature_vector) >= self.threshold


class VoiceModelWrapper:
    """
    Wraps the production voice model (GBM pipeline in .pkl).
    """

    def __init__(self, username: str, model_path: Optional[str] = None):
        self.username    = username
        self.model_data  = None
        self.pipeline    = None
        self.threshold   = 0.55
        self.raw_mean    = None
        self.raw_std     = None
        self._load(model_path)

    def _load(self, model_path: Optional[str]):
        import pickle
        if model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model_data = pickle.load(f)
            self.pipeline  = self.model_data["pipeline"]
            self.threshold = self.model_data["threshold"]
            self.raw_mean  = self.model_data.get("raw_profile_mean")
            self.raw_std   = self.model_data.get("raw_profile_std")
            print(f"  ✅ Voice model loaded from {model_path}")
        else:
            print(f"  ⚠  No voice model file found.")

    def _mahalanobis(self, vec: np.ndarray) -> float:
        if self.raw_mean is None:
            return 0.5
        raw_vec  = np.concatenate([vec[13:26], vec[26:39], vec[52:62]])
        var      = self.raw_std ** 2
        safe_var = np.where(var < 1e-10, 1e-10, var)
        diff     = raw_vec - self.raw_mean
        d_sq     = float(np.sum(diff ** 2 / safe_var)) / max(len(raw_vec), 1)
        return float(np.clip(1.0 / (1.0 + np.exp(2.5 * (d_sq - 1.0))), 0, 1))

    def score(self, feature_vector: np.ndarray) -> float:
        """Return fused genuine probability [0, 1]."""
        if self.pipeline is None:
            return 0.0
        vec  = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        n    = self.model_data["profile_mean"].shape[0]
        if vec.shape[1] < n:
            vec = np.hstack([vec, np.zeros((1, n - vec.shape[1]))])
        elif vec.shape[1] > n:
            vec = vec[:, :n]
        model_prob = float(self.pipeline.predict_proba(vec)[0][1])
        mah        = self._mahalanobis(feature_vector)
        return 0.70 * model_prob + 0.30 * mah

    def decide(self, feature_vector: np.ndarray) -> bool:
        return self.score(feature_vector) >= self.threshold


# ─────────────────────────────────────────────────────────────────────────────
#  FUSION STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def fuse_and_gate(
    ks_score: float,
    voice_score: float,
    ks_threshold: float,
    voice_threshold: float,
) -> Tuple[bool, float]:
    """
    AND-gate fusion: BOTH modalities must pass independently.
    Combined score = min(ks_score, voice_score) — the bottleneck score.
    This is the production system's strategy.
    """
    passed = (ks_score >= ks_threshold) and (voice_score >= voice_threshold)
    combined = min(ks_score, voice_score)
    return passed, combined


def fuse_or_gate(
    ks_score: float,
    voice_score: float,
    ks_threshold: float,
    voice_threshold: float,
) -> Tuple[bool, float]:
    """
    OR-gate fusion: EITHER modality passing is sufficient.
    Combined score = max(ks_score, voice_score).
    More permissive — lower security, higher usability.
    """
    passed   = (ks_score >= ks_threshold) or (voice_score >= voice_threshold)
    combined = max(ks_score, voice_score)
    return passed, combined


def fuse_weighted(
    ks_score: float,
    voice_score: float,
    ks_weight: float = 0.5,
    threshold: float = 0.55,
) -> Tuple[bool, float]:
    """
    Weighted sum fusion: linear combination of both scores.
    A single fused score is compared against one threshold.
    """
    combined = ks_weight * ks_score + (1.0 - ks_weight) * voice_score
    return combined >= threshold, combined


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> Dict:
    """
    Compute full biometric authentication metrics at a given threshold.

    Returns dict with: accuracy, far, frr, eer, threshold,
                       attack_success_rate, n_genuine, n_impostor
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

    y_pred = (y_scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return {"error": "Insufficient class diversity in predictions"}

    tn, fp, fn, tp = cm.ravel()

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)

    # EER via ROC sweep
    try:
        fpr, tpr, threshs = roc_curve(y_true, y_scores)
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
    except Exception:
        eer = (far + frr) / 2.0

    n_impostor = int(np.sum(y_true == 0))
    n_genuine  = int(np.sum(y_true == 1))

    return {
        "accuracy":            float(acc),
        "far":                 float(far),
        "frr":                 float(frr),
        "eer":                 float(eer),
        "threshold":           float(threshold),
        "attack_success_rate": float(far),   # FAR = fraction of attacks that succeeded
        "n_genuine":           n_genuine,
        "n_impostor":          n_impostor,
        "true_positives":      int(tp),
        "false_positives":     int(fp),
        "true_negatives":      int(tn),
        "false_negatives":     int(fn),
    }


def threshold_sensitivity(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    t_range: Tuple[float, float] = (0.30, 0.95),
    step: float = 0.02,
) -> List[Dict]:
    """
    Sweep threshold across a range and compute FAR/FRR at each point.
    Returns a list of metric dicts for plotting a DET / ROC curve.
    """
    results = []
    for t in np.arange(t_range[0], t_range[1], step):
        m = compute_metrics(y_true, y_scores, threshold=t)
        m["threshold"] = float(t)
        results.append(m)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  AUTHENTICATION TESTER
# ─────────────────────────────────────────────────────────────────────────────

class AuthenticationTester:
    """
    Main orchestrator for authentication security testing.

    Usage
    -----
    tester = AuthenticationTester(ks_model, voice_model)
    results = tester.run_attack_battery(genuine_ks, genuine_voice, attacker)
    report  = tester.generate_report(results)
    """

    def __init__(
        self,
        ks_model: Optional[KeystrokeModelWrapper] = None,
        voice_model: Optional[VoiceModelWrapper] = None,
        fusion_strategy: str = "and",   # "and" | "or" | "weighted"
        ks_weight: float = 0.5,
    ):
        self.ks_model        = ks_model
        self.voice_model     = voice_model
        self.fusion_strategy = fusion_strategy
        self.ks_weight       = ks_weight

    # ── Score individual samples ──────────────────────────────────────────────

    def score_keystroke(self, attack: AttackResult) -> float:
        if self.ks_model is None:
            return 0.0
        return self.ks_model.score(attack.feature_vector)

    def score_voice(self, attack: AttackResult) -> float:
        if self.voice_model is None:
            return 0.0
        return self.voice_model.score(attack.feature_vector)

    def fuse(self, ks_score: float, voice_score: float) -> Tuple[bool, float]:
        ks_thresh    = self.ks_model.threshold    if self.ks_model    else 0.55
        voice_thresh = self.voice_model.threshold if self.voice_model else 0.55

        if self.fusion_strategy == "and":
            return fuse_and_gate(ks_score, voice_score, ks_thresh, voice_thresh)
        elif self.fusion_strategy == "or":
            return fuse_or_gate(ks_score, voice_score, ks_thresh, voice_thresh)
        else:
            return fuse_weighted(ks_score, voice_score, self.ks_weight,
                                 (ks_thresh + voice_thresh) / 2)

    # ── Single-modality attack test ───────────────────────────────────────────

    def test_keystroke_attacks(
        self,
        attacks: List[AttackResult],
        genuine_vectors: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """
        Test a batch of keystroke attacks.
        Returns per-attack-type breakdown and aggregate metrics.
        """
        if self.ks_model is None:
            return {"error": "No keystroke model loaded"}

        scores     = []
        labels     = []
        attack_log = []

        # Score attacks (label=0, impostor)
        for atk in attacks:
            t0    = time.perf_counter()
            score = self.score_keystroke(atk)
            dt    = time.perf_counter() - t0
            scores.append(score)
            labels.append(0)
            attack_log.append({
                "attack_type": atk.attack_type,
                "score":       round(score, 4),
                "passed":      score >= self.ks_model.threshold,
                "latency_ms":  round(dt * 1000, 3),
                **atk.metadata,
            })

        # Score genuine samples (label=1) if provided
        if genuine_vectors:
            for vec in genuine_vectors:
                score = self.ks_model.score(np.asarray(vec))
                scores.append(score)
                labels.append(1)

        y_true   = np.array(labels)
        y_scores = np.array(scores)
        metrics  = compute_metrics(y_true, y_scores, self.ks_model.threshold)

        # Per-attack-type success rates
        type_stats = {}
        for log in attack_log:
            atype = log["attack_type"]
            if atype not in type_stats:
                type_stats[atype] = {"total": 0, "passed": 0}
            type_stats[atype]["total"]  += 1
            type_stats[atype]["passed"] += int(log["passed"])

        for atype, stat in type_stats.items():
            stat["success_rate"] = stat["passed"] / max(stat["total"], 1)

        return {
            "modality":      "keystroke",
            "metrics":       metrics,
            "attack_types":  type_stats,
            "attack_log":    attack_log,
            "sensitivity":   threshold_sensitivity(y_true, y_scores),
        }

    def test_voice_attacks(
        self,
        attacks: List[AttackResult],
        genuine_vectors: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """Test a batch of voice attacks."""
        if self.voice_model is None:
            return {"error": "No voice model loaded"}

        scores     = []
        labels     = []
        attack_log = []

        for atk in attacks:
            t0    = time.perf_counter()
            score = self.score_voice(atk)
            dt    = time.perf_counter() - t0
            scores.append(score)
            labels.append(0)
            attack_log.append({
                "attack_type": atk.attack_type,
                "score":       round(score, 4),
                "passed":      score >= self.voice_model.threshold,
                "latency_ms":  round(dt * 1000, 3),
                **atk.metadata,
            })

        if genuine_vectors:
            for vec in genuine_vectors:
                score = self.voice_model.score(np.asarray(vec))
                scores.append(score)
                labels.append(1)

        y_true   = np.array(labels)
        y_scores = np.array(scores)
        metrics  = compute_metrics(y_true, y_scores, self.voice_model.threshold)

        type_stats = {}
        for log in attack_log:
            atype = log["attack_type"]
            if atype not in type_stats:
                type_stats[atype] = {"total": 0, "passed": 0}
            type_stats[atype]["total"]  += 1
            type_stats[atype]["passed"] += int(log["passed"])
        for atype, stat in type_stats.items():
            stat["success_rate"] = stat["passed"] / max(stat["total"], 1)

        return {
            "modality":     "voice",
            "metrics":      metrics,
            "attack_types": type_stats,
            "attack_log":   attack_log,
            "sensitivity":  threshold_sensitivity(y_true, y_scores),
        }

    def test_multimodal_attacks(
        self,
        attack_pairs: List[Tuple[AttackResult, AttackResult]],
        genuine_ks: Optional[List[np.ndarray]] = None,
        genuine_voice: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """
        Test coordinated multimodal attacks against the fused system.

        Each pair (ks_attack, voice_attack) is evaluated through the
        configured fusion strategy (AND / OR / weighted).
        """
        ks_scores    = []
        voice_scores = []
        fused_scores = []
        labels       = []
        attack_log   = []

        for ks_atk, v_atk in attack_pairs:
            ks_s   = self.score_keystroke(ks_atk)
            v_s    = self.score_voice(v_atk)
            passed, fused = self.fuse(ks_s, v_s)

            ks_scores.append(ks_s)
            voice_scores.append(v_s)
            fused_scores.append(fused)
            labels.append(0)

            attack_log.append({
                "ks_attack_type":    ks_atk.attack_type,
                "voice_attack_type": v_atk.attack_type,
                "ks_score":          round(ks_s, 4),
                "voice_score":       round(v_s, 4),
                "fused_score":       round(fused, 4),
                "passed":            passed,
                "fusion_strategy":   self.fusion_strategy,
            })

        # Genuine samples
        if genuine_ks and genuine_voice:
            n = min(len(genuine_ks), len(genuine_voice))
            for i in range(n):
                ks_s  = self.ks_model.score(genuine_ks[i]) if self.ks_model else 0.8
                v_s   = self.voice_model.score(genuine_voice[i]) if self.voice_model else 0.8
                _, f  = self.fuse(ks_s, v_s)
                fused_scores.append(f)
                labels.append(1)

        y_true    = np.array(labels)
        y_fused   = np.array(fused_scores)

        ks_thresh    = self.ks_model.threshold    if self.ks_model    else 0.55
        voice_thresh = self.voice_model.threshold if self.voice_model else 0.55
        fused_thresh = min(ks_thresh, voice_thresh)

        metrics = compute_metrics(y_true, y_fused, fused_thresh)

        n_passed = sum(1 for log in attack_log if log["passed"])

        return {
            "modality":          "multimodal",
            "fusion_strategy":   self.fusion_strategy,
            "metrics":           metrics,
            "attacks_attempted": len(attack_pairs),
            "attacks_passed":    n_passed,
            "multimodal_far":    n_passed / max(len(attack_pairs), 1),
            "attack_log":        attack_log,
            "sensitivity":       threshold_sensitivity(y_true, y_fused),
        }

    # ── Full battery ──────────────────────────────────────────────────────────

    def run_full_battery(
        self,
        ks_attacks_by_type: Dict[str, List[AttackResult]],
        voice_attacks_by_type: Dict[str, List[AttackResult]],
        multimodal_pairs: List[Tuple[AttackResult, AttackResult]],
        genuine_ks: Optional[List[np.ndarray]] = None,
        genuine_voice: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """
        Run all attack types and return consolidated results.
        """
        print("\n  Running keystroke attack battery ...")
        all_ks_attacks = [a for attacks in ks_attacks_by_type.values() for a in attacks]
        ks_results     = self.test_keystroke_attacks(all_ks_attacks, genuine_ks)

        print("  Running voice attack battery ...")
        all_voice_attacks = [a for attacks in voice_attacks_by_type.values() for a in attacks]
        voice_results     = self.test_voice_attacks(all_voice_attacks, genuine_voice)

        print("  Running multimodal attack battery ...")
        mm_results = self.test_multimodal_attacks(
            multimodal_pairs, genuine_ks, genuine_voice
        )

        return {
            "keystroke":  ks_results,
            "voice":      voice_results,
            "multimodal": mm_results,
        }