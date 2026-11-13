"""Fixed training-only stress transformations, never selected on test outcomes."""

import copy

import numpy as np


LEVELS = (0.05, 0.10, 0.20)


def flip_training_labels(labels, training_ids, rate, seed, class_conditional=None):
    if rate not in LEVELS:
        raise ValueError("Use preregistered 5%, 10%, or 20% label-noise levels")
    result = dict(labels)
    rng = np.random.default_rng(seed)
    eligible = sorted(
        i for i in training_ids if class_conditional is None or labels[i] == class_conditional
    )
    count = int(np.floor(rate * len(eligible)))
    changed = rng.choice(eligible, count, replace=False).tolist() if count else []
    for i in changed:
        result[i] = 1 - result[i]
    return result, changed


def metric_noise(features, training_ids, columns, scale, seed):
    if scale not in (0.05, 0.10, 0.20):
        raise ValueError("Use fixed metric-noise levels")
    result = copy.deepcopy(features)
    rng = np.random.default_rng(seed)
    for name in columns:
        values = [
            float(features[i]["metrics"][name])
            for i in training_ids
            if features[i]["metrics"].get(name) is not None
        ]
        std = np.std(values) if values else 0.0
        for i in sorted(training_ids):
            value = result[i]["metrics"].get(name)
            if value is not None:
                result[i]["metrics"][name] = float(value + rng.normal(0, std * scale))
    return result
