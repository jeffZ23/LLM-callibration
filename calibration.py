"""
Calibration metrics and disagreement measures.

ECE: Expected Calibration Error (Guo et al., 2017). Bin predictions by their
confidence; in each bin compare empirical accuracy to mean confidence.
Lower is better. Perfectly calibrated => ECE = 0.

Vote entropy: Shannon entropy (in nats) of the answer distribution across
agents on a single question. Higher => more disagreement.
"""

from collections import Counter
from math import log
from typing import Iterable

import numpy as np


def expected_calibration_error(confidences: Iterable[float],
                               correctness: Iterable[bool],
                               n_bins: int = 10) -> float:
    """ECE on confidences in [0, 100]. Returns a value in [0, 1]."""
    confs = np.asarray(list(confidences), dtype=float) / 100.0
    corr = np.asarray(list(correctness), dtype=float)
    n = len(confs)
    if n == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if not mask.any():
            continue
        bin_acc = corr[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_bins(confidences: Iterable[float],
                     correctness: Iterable[bool],
                     n_bins: int = 10):
    """Return (bin_centers, bin_accuracies, bin_counts) for plotting."""
    confs = np.asarray(list(confidences), dtype=float) / 100.0
    corr = np.asarray(list(correctness), dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        centers.append((lo + hi) / 2)
        if mask.any():
            accs.append(float(corr[mask].mean()))
        else:
            accs.append(float("nan"))
        counts.append(int(mask.sum()))
    return centers, accs, counts


def vote_entropy(answers: Iterable[str]) -> float:
    """Shannon entropy of the answer distribution, in nats."""
    answers = list(answers)
    if not answers:
        return 0.0
    counts = Counter(answers)
    n = len(answers)
    return -sum((c / n) * log(c / n) for c in counts.values())


def reward(correct: bool, confidence_pct: float, lam: float = 0.5) -> float:
    """Reward from the proposal: r = 1{correct} - lambda * |c - 1{correct}|.
    Penalizes confident-wrong and unconfident-correct, both."""
    c = confidence_pct / 100.0
    indicator = 1.0 if correct else 0.0
    return indicator - lam * abs(c - indicator)


def brier_score(confidences: Iterable[float],
                correctness: Iterable[bool]) -> float:
    """Mean squared error between confidence and outcome (Murphy, 1973).

    A strictly proper scoring rule: unlike ECE it cannot be gamed by bin
    placement, and it jointly rewards calibration AND sharpness. Lower is
    better; perfect prediction => 0.
    """
    confs = np.asarray(list(confidences), dtype=float) / 100.0
    corr = np.asarray(list(correctness), dtype=float)
    if len(confs) == 0:
        return float("nan")
    return float(np.mean((confs - corr) ** 2))


def _ece_metric(confs01: np.ndarray, corr: np.ndarray, n_bins: int = 10) -> float:
    """ECE on confidences already in [0, 1] (internal bootstrap helper)."""
    n = len(confs01)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs01 > lo) & (confs01 <= hi) if i > 0 else (confs01 >= lo) & (confs01 <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(corr[mask].mean() - confs01[mask].mean())
    return float(ece)


def bootstrap_metric(confidences: Iterable[float],
                     correctness: Iterable[bool],
                     metric: str = "ece",
                     n_bins: int = 10,
                     n_boot: int = 2000,
                     alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap CI for a calibration metric.

    Resamples (confidence, correctness) pairs with replacement and recomputes
    the metric each time. Returns (point_estimate, ci_low, ci_high) where the
    point estimate is the bootstrap mean.

    metric: "ece", "brier", or "acc".
    """
    confs = np.asarray(list(confidences), dtype=float) / 100.0
    corr = np.asarray(list(correctness), dtype=float)
    n = len(confs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    if metric == "ece":
        fn = lambda c, k: _ece_metric(c, k, n_bins)
    elif metric == "brier":
        fn = lambda c, k: float(np.mean((c - k) ** 2))
    elif metric == "acc":
        fn = lambda c, k: float(k.mean())
    else:
        raise ValueError(f"unknown metric {metric!r}")

    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(confs[idx], corr[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stats.mean()), float(lo), float(hi)
