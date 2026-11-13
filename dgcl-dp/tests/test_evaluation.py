import numpy as np
import pytest

from dgcl_dp.evaluation import choose_threshold, evaluate, inspection
from dgcl_dp.statistics import compare_projects


def test_effort_fractional_boundary_zero_loc_and_ties():
    result = inspection([1, 0], [0.8, 0.1], [10, 10], ["a", "b"])
    assert result["recall_at_20_loc"] == pytest.approx(0.4)
    assert result["popt"] == pytest.approx(1.0) and result["ifa"] == 0
    result = inspection([0, 1], [0.1, 0.1], [0, 0], ["a", "b"])
    assert result["ifa"] == 1
    assert inspection([0, 0], [0.1, 0.2], [1, 2], ["a", "b"])["popt"] is None


def test_threshold_selection_and_single_class_behavior():
    y, p = [0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8]
    assert choose_threshold(y, p) == 0.7
    result = evaluate(y, p, [1] * 4, list("abcd"), 0.7)
    assert result["mcc"] == 1 and result["pr_auc"] == 1
    single = evaluate([0, 0], [0.1, 0.3], [1, 1], list("ab"))
    assert single["pr_auc"] is None and single["roc_auc"] is None
    assert choose_threshold([0, 0], [0.1, 0.3]) == 0.5


def test_statistics_aggregate_seeds_and_releases_within_project():
    rows = [
        {"project": project, "model": model, "task": task, "mcc": value}
        for project in ("a", "b", "c")
        for task in ("r1", "r2")
        for model, value in (("dgcl", 0.4), ("lr", 0.2))
        for _ in range(10)
    ]
    results = compare_projects(rows, "dgcl", bootstrap=100, seed=10)
    assert len(results[0]["projects"]) == 3
    np.testing.assert_allclose(results[0]["cluster_ci95"], [0.2, 0.2])
    assert results[0]["paired_a12"] == 1
