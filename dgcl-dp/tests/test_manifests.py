import copy
from datetime import timedelta

import pytest

from dgcl_dp.manifests import cross_version, remove_later_clones, seal_split, timestamp, validate


def test_sample_clone_project_separation(task):
    rows, labels, split = (
        list(task["samples"].values()),
        list(task["labels"].values()),
        task["split"],
    )
    validate(rows, labels, split, primary=False)
    bad = copy.deepcopy(split)
    bad["test"].append(bad["train"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate(rows, labels, seal_split(bad), primary=False)
    rows[-1]["clone_group"] = rows[0]["clone_group"]
    with pytest.raises(ValueError, match="Clone"):
        validate(rows, labels, split, primary=False)
    rows[-1]["clone_group"] = rows[-1]["sample_id"]
    bad = {**split, "protocol": "cross_project"}
    with pytest.raises(ValueError, match="projects"):
        validate(rows, labels, seal_split(bad), primary=False)


def mature_rows():
    samples, labels = [], []
    for release in ("2020-01-01", "2021-01-01", "2022-01-01"):
        for j in range(40):
            i = f"{release}-{j}"
            prediction = release + "T00:00:00Z"
            end = (timestamp(prediction) + timedelta(days=180)).isoformat()
            samples.append(
                {
                    "sample_id": i,
                    "project": "p",
                    "prediction_time": prediction,
                    "source_time": prediction,
                    "extraction_status": "ok",
                    "loc": 1,
                    "release_verified": True,
                    "production": True,
                    "top_level_classes": 1,
                    "clone_group": i,
                }
            )
            events = (
                [
                    {
                        "fixed_at": (timestamp(prediction) + timedelta(days=10)).isoformat(),
                        "ancestry_verified": True,
                        "confirmed_bug": True,
                        "fixing_commit": "fix",
                        "inducing_commit": "induce",
                        "issue": "BUG-1",
                        "szz_variant": "documented",
                    }
                ]
                if j % 2
                else []
            )
            labels.append(
                {
                    "sample_id": i,
                    "label": j % 2,
                    "window_end": end,
                    "available_at": end,
                    "mechanism": "issue-linked-szz",
                    "events": events,
                }
            )
    return samples, labels


def test_timestamps_label_maturation_and_source_snapshot():
    rows, labels = mature_rows()
    split = cross_version(rows, labels, "2026-10-31T00:00:00Z")
    validate(rows, labels, split)
    bad = copy.deepcopy(labels)
    bad[0]["available_at"] = "2023-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="unavailable"):
        validate(rows, bad, split)
    rows[0]["source_time"] = "2024-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="future"):
        validate(rows, labels, split)
    with pytest.raises(ValueError, match="checksum"):
        validate(rows, labels, {**split, "checksum": "wrong"})


def test_earliest_clone_kept_exclusion_recorded():
    samples = [
        {"sample_id": "a", "clone_group": "x"},
        {"sample_id": "b", "clone_group": "x"},
        {"sample_id": "c", "clone_group": "z"},
    ]
    parts, exclusions = remove_later_clones(
        {"train": ["a"], "validation": ["b"], "test": ["c"]}, samples
    )
    assert parts["train"] == ["a"] and not parts["validation"]
    assert exclusions == [{"sample_id": "b", "partition": "validation", "reason": "earlier_clone"}]


def test_declared_deployment_matches_actual_prediction_snapshot():
    rows, labels = mature_rows()
    split = cross_version(rows, labels, "2026-10-31T00:00:00Z")
    incorrect = seal_split({**split, "test_time": "2024-01-01T00:00:00Z"})
    with pytest.raises(ValueError, match="deployment snapshot"):
        validate(rows, labels, incorrect)
