"""
eval_utils.py
─────────────────────────────────────────────────────────────────────────────
Shared evaluation utilities for biometric authentication experiments.
Used by all keystroke and voice benchmark scripts in Chapter 4.

Computes:
  - FAR (False Accept Rate)
  - FRR (False Reject Rate)
  - EER (Equal Error Rate)
  - Accuracy
  - Average inference time
  - Performance summary table
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
)
from typing import Callable, Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_far_frr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Compute FAR and FRR from binary predictions.

    Parameters
    ----------
    y_true : array-like, shape (n,)  — ground truth: 1=genuine, 0=impostor
    y_pred : array-like, shape (n,)  — predicted labels

    Returns
    -------
    (far, frr) as floats in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        return 0.0, 0.0
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return float(far), float(frr)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute Equal Error Rate (EER) by sweeping threshold over the ROC curve.

    Returns
    -------
    (eer, best_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    # Find the operating point where FAR ≈ FRR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, thresh


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    far_weight: float = 3.0,
    min_thresh: float = 0.45,
) -> tuple:
    """
    Search for the threshold that minimises FAR-weighted cost.

    cost = (far_weight * FAR + FRR) / (far_weight + 1)

    Returns
    -------
    (best_threshold, far, frr, eer)
    """
    best_thresh = min_thresh
    best_cost   = 1.0
    best_eer    = 1.0

    for t in np.arange(0.30, 0.95, 0.01):
        y_t = (y_scores >= t).astype(int)
        if len(np.unique(y_t)) < 2:
            continue
        far_t, frr_t = compute_far_frr(y_true, y_t)
        eer_t = (far_t + frr_t) / 2.0
        cost_t = (far_weight * far_t + frr_t) / (far_weight + 1.0)

        if eer_t < best_eer:
            best_eer = eer_t
        if cost_t < best_cost:
            best_cost   = cost_t
            best_thresh = float(t)

    best_thresh = max(best_thresh, min_thresh)
    y_final = (y_scores >= best_thresh).astype(int)
    far_f, frr_f = compute_far_frr(y_true, y_final)
    return best_thresh, far_f, frr_f, best_eer


# ─────────────────────────────────────────────────────────────────────────────
#  TIMING UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def measure_inference_time(
    predict_fn: Callable,
    X_sample: np.ndarray,
    n_trials: int = 100,
) -> float:
    """
    Measure average per-sample inference time in seconds.

    Parameters
    ----------
    predict_fn : callable  — accepts a 2-D array, returns probabilities
    X_sample   : (n, d) array — samples to time
    n_trials   : number of timing runs

    Returns
    -------
    mean_time_per_sample (float, seconds)
    """
    times = []
    n = min(n_trials, len(X_sample))
    indices = np.random.choice(len(X_sample), size=n, replace=(len(X_sample) < n))
    for i in indices:
        x = X_sample[i : i + 1]
        t0 = time.perf_counter()
        predict_fn(x)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ─────────────────────────────────────────────────────────────────────────────
#  FULL EVALUATION BLOCK
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    inference_time: Optional[float] = None,
    threshold: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """
    Complete evaluation for one model.

    Parameters
    ----------
    model_name     : label for printing / table
    y_true         : ground-truth binary labels (1=genuine, 0=impostor)
    y_scores       : probability of genuine class (continuous)
    inference_time : seconds per sample (pre-measured) or None
    threshold      : fixed threshold override; if None, EER threshold is used
    verbose        : print formatted results

    Returns
    -------
    dict with keys: model, accuracy, far, frr, eer, threshold, time_s
    """
    eer, eer_thresh = compute_eer(y_true, y_scores)

    if threshold is None:
        threshold = eer_thresh

    y_pred = (y_scores >= threshold).astype(int)
    acc    = accuracy_score(y_true, y_pred)
    far, frr = compute_far_frr(y_true, y_pred)

    result = {
        "model":     model_name,
        "accuracy":  float(acc),
        "far":       float(far),
        "frr":       float(frr),
        "eer":       float(eer),
        "threshold": float(threshold),
        "time_s":    float(inference_time) if inference_time is not None else None,
    }

    if verbose:
        t_str = f"{inference_time*1000:.2f} ms" if inference_time is not None else "N/A"
        print(f"\n{'─'*55}")
        print(f"  Model: {model_name}")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  FAR       : {far*100:.2f}%")
        print(f"  FRR       : {frr*100:.2f}%")
        print(f"  EER       : {eer*100:.2f}%")
        print(f"  Threshold : {threshold:.3f}")
        print(f"  Avg Infer : {t_str}")
        print(f"{'─'*55}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list, title: str = "Performance Summary"):
    """
    Print a formatted comparison table from a list of evaluation dicts.

    Parameters
    ----------
    results : list of dicts returned by evaluate_model()
    title   : table header
    """
    header = f"\n{'═'*72}\n  {title}\n{'═'*72}"
    print(header)
    col = f"  {'Model':<26} {'Accuracy':>9} {'FAR':>7} {'FRR':>7} {'EER':>7} {'Time':>10}"
    print(col)
    print(f"  {'─'*26} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")
    for r in results:
        t_str = f"{r['time_s']*1000:.2f}ms" if r.get("time_s") is not None else "  N/A"
        print(
            f"  {r['model']:<26} "
            f"{r['accuracy']*100:>8.2f}% "
            f"{r['far']*100:>6.2f}% "
            f"{r['frr']*100:>6.2f}% "
            f"{r['eer']*100:>6.2f}% "
            f"{t_str:>10}"
        )
    print(f"{'═'*72}\n")


def print_literature_comparison(results: list):
    """
    Print a literature comparison table with placeholder published benchmarks.
    """
    benchmarks = [
        {"model": "Sharma et al. (SVM)",           "accuracy": 0.930, "eer": 0.068, "source": "2020"},
        {"model": "Alsultan & Warwick (KNN)",       "accuracy": 0.912, "eer": 0.082, "source": "2013"},
        {"model": "DEFT Dataset (RF baseline)",     "accuracy": 0.990, "eer": 0.009, "source": "2022"},
        {"model": "Morales et al. (LSTM)",          "accuracy": 0.941, "eer": 0.055, "source": "2019"},
    ]

    print(f"\n{'═'*72}")
    print(f"  Literature Comparison")
    print(f"{'═'*72}")
    print(f"  {'Source':<38} {'Accuracy':>9} {'EER':>8}")
    print(f"  {'─'*38} {'─'*9} {'─'*8}")

    print("  [This Study]")
    for r in results:
        eer_str = f"{r['eer']*100:.2f}%"
        print(f"    {r['model']:<36} {r['accuracy']*100:>8.2f}% {eer_str:>8}")

    print("\n  [Published Benchmarks]")
    for b in benchmarks:
        eer_str = f"{b['eer']*100:.2f}%"
        print(f"    {b['model']:<36} {b['accuracy']*100:>8.2f}% {eer_str:>8}")

    print(f"{'═'*72}\n")
    print("  NOTE: Direct comparisons are indicative only — dataset, phrase, and")
    print("  enrollment count differ from published studies.\n")