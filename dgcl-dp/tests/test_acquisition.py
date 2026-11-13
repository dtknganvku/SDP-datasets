import pytest

from dgcl_dp.acquisition import select_half_year_releases, snapshot_source
from dgcl_dp.prepare import build_labels


def test_release_selection_is_fixed_half_year_metadata_only():
    releases = [
        {"project": "p", "released_at": date + "T00:00:00Z", "stable": stable, "commit": commit}
        for date, stable, commit in (
            ("2020-01-01", True, "a"),
            ("2020-05-01", True, "b"),
            ("2020-06-01", False, "c"),
            ("2020-08-01", True, "d"),
            ("2026-01-01", True, "e"),
        )
    ]
    assert [r["commit"] for r in select_half_year_releases(releases)] == ["b", "d"]


def test_label_window_excludes_late_fixes_and_waits_for_evidence():
    event = {
        "confirmed_bug": True,
        "ancestry_verified": True,
        "fixing_commit": "f",
        "inducing_commit": "i",
        "issue": "BUG-1",
        "szz_variant": "documented",
        "fixed_at": "2020-02-01T00:00:00Z",
        "available_at": "2020-08-01T00:00:00Z",
    }
    label = build_labels("a", "2020-01-01T00:00:00Z", [event], "2021-01-01T00:00:00Z")
    assert label["label"] == 1 and label["available_at"] == "2020-08-01T00:00:00+00:00"
    late = {**event, "fixed_at": "2020-12-01T00:00:00Z"}
    assert build_labels("a", "2020-01-01T00:00:00Z", [late], "2021-01-01T00:00:00Z")["label"] == 0
    with pytest.raises(ValueError, match="incomplete"):
        build_labels("a", "2020-01-01T00:00:00Z", [], "2020-02-01T00:00:00Z")
    with pytest.raises(ValueError, match="immutable"):
        snapshot_source(".", "main", "a.java")
