"""Classification, calibration and full-curve file inspection metrics."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


def choose_threshold(labels, probability):
    """Maximize validation MCC; ties choose the largest threshold."""
    y, p = np.asarray(labels), np.asarray(probability, dtype=float)
    if len(np.unique(y)) != 2:
        return 0.5
    candidates = sorted(set(p.tolist() + [0.0, 0.5, float(np.nextafter(max(p), np.inf))]))
    return max(candidates, key=lambda t: (matthews_corrcoef(y, p >= t), t))


def inspection(labels, probability, loc, ids, probability_only=False):
    y, p, cost = np.asarray(labels), np.asarray(probability), np.maximum(np.asarray(loc), 1.0)
    if len(y) == 0 or len(set(ids)) != len(ids) or (np.asarray(loc) < 0).any():
        raise ValueError("Invalid inspection records")
    total = int(y.sum())
    rank = p if probability_only else p / cost
    order = sorted(range(len(y)), key=lambda i: (-float(rank[i]), ids[i]))

    def curve(sequence):
        effort = np.r_[0.0, np.cumsum(cost[sequence]) / cost.sum()]
        recall = np.r_[0.0, np.cumsum(y[sequence]) / max(1, total)]
        return effort, recall

    x, recall = curve(order)
    if not total:
        return {
            "popt": None,
            "recall_at_20_loc": None,
            "ifa": None,
            "curve": {"effort": x.tolist(), "recall": recall.tolist()},
        }
    ideal = sorted(range(len(y)), key=lambda i: (-float(y[i] / cost[i]), ids[i]))
    worst = sorted(range(len(y)), key=lambda i: (float(y[i] / cost[i]), ids[i]))
    best_area = float(np.trapz(*reversed(curve(ideal))))
    worst_area = float(np.trapz(*reversed(curve(worst))))
    area = float(np.trapz(recall, x))
    popt = (area - worst_area) / (best_area - worst_area) if best_area > worst_area else None
    return {
        "popt": popt,
        "recall_at_20_loc": float(np.interp(0.2, x, recall)),
        "ifa": next(k for k, i in enumerate(order) if y[i]),
        "curve": {"effort": x.tolist(), "recall": recall.tolist()},
    }


def evaluate(labels, probability, loc, ids, threshold=0.5, bins=10):
    y, p = np.asarray(labels, dtype=int), np.asarray(probability, dtype=float)
    if not len(y) or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid binary predictions")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite in [0,1]")
    predicted = p >= threshold
    two_classes = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    reliability, ece = [], 0.0
    assignments = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        mask = assignments == b
        count = int(mask.sum())
        accuracy, confidence = (
            (float(y[mask].mean()), float(p[mask].mean())) if count else (None, None)
        )
        reliability.append(
            {
                "lower": b / bins,
                "upper": (b + 1) / bins,
                "count": count,
                "observed": accuracy,
                "predicted": confidence,
            }
        )
        if count:
            ece += count / len(y) * abs(accuracy - confidence)
    effort = inspection(y, p, loc, ids)
    result = {
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)) if two_classes else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else None,
        "pr_auc": float(average_precision_score(y, p)) if two_classes else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if two_classes else None,
        "g_mean": float(np.sqrt(tp / (tp + fn) * tn / (tn + fp))) if two_classes else None,
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "threshold": float(threshold),
        "reliability": reliability,
        **effort,
        "probability_ranking": inspection(y, p, loc, ids, probability_only=True),
    }
    return result
