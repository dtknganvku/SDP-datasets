"""Project-level inference; releases and seeds are dependent observations."""

import numpy as np
from scipy.stats import wilcoxon, t


def seed_summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    delta = (
        float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
    )
    return {"mean": mean, "sd": sd, "seed_ci95": [mean - delta, mean + delta], "n": len(values)}


def project_means(rows, metric="mcc"):
    grouped = {}
    for row in rows:
        value = row[metric]
        if value is not None:
            grouped.setdefault((row["model"], row["project"], row["task"]), []).append(value)
    tasks = {}
    for (model, project, task), values in grouped.items():
        tasks.setdefault(model, {}).setdefault(project, []).append(float(np.mean(values)))
    return {
        model: {p: float(np.mean(v)) for p, v in projects.items()}
        for model, projects in tasks.items()
    }


def compare_projects(rows, reference, metric="mcc", bootstrap=10000, seed=10):
    grouped = project_means(rows, metric)
    if reference not in grouped:
        raise ValueError("Reference model missing")
    rng, results = np.random.default_rng(seed), []
    for model, projects in sorted(grouped.items()):
        if model == reference:
            continue
        common = sorted(set(projects) & set(grouped[reference]))
        if not common:
            continue
        diff = np.array([grouped[reference][p] - projects[p] for p in common])
        means = rng.choice(diff, (bootstrap, len(diff)), replace=True).mean(axis=1)
        pvalue = float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0
        results.append(
            {
                "reference": reference,
                "model": model,
                "projects": common,
                "mean_difference": float(diff.mean()),
                "cluster_ci95": np.quantile(means, [0.025, 0.975]).tolist(),
                "p": pvalue,
                "paired_a12": float(((diff > 0) + 0.5 * (diff == 0)).mean()),
            }
        )
    running = 0.0
    for rank, row in enumerate(sorted(results, key=lambda r: r["p"])):
        running = max(running, min(1.0, row["p"] * (len(results) - rank)))
        row["holm_p"] = running
    return results
