import numpy as np
import pytest

from dgcl_dp.features import MetricTransform
from dgcl_dp.source import Vocabulary, clone_families, parse_java


def test_training_only_metrics_missing_and_constants():
    transform = MetricTransform(["size", "coupling", "constant"]).fit(
        [
            {"metrics": {"size": 1, "coupling": None, "constant": 3}},
            {"metrics": {"size": 3, "coupling": 4, "constant": 3}},
        ]
    )
    before = transform.state()
    np.testing.assert_array_equal(transform.transform([{"metrics": {"size": None}}]), [[0.0]])
    assert transform.state() == before
    with pytest.raises(ValueError, match="Outcome"):
        MetricTransform(["bug"])
    with pytest.raises(ValueError, match="No nonconstant"):
        MetricTransform(["a"]).fit([{"metrics": {"a": None}}])


def test_java_unknown_parse_failure_and_method_partitions():
    good = parse_java(b"class A { int f() { return 1; } }")
    assert good.status == "ok" and good.top_level_classes == 1
    assert parse_java(b"class {").status == "parse_failure"
    assert parse_java(b"class A {} interface B {}").top_level_classes == 2
    vocab = Vocabulary()
    assert vocab.graphs(good)[0].token.unique().tolist() == [0]
    source = (
        b"class A {" + b"".join(f"int f{i}() {{ return {i}; }}".encode() for i in range(12)) + b"}"
    )
    large = parse_java(source, max_nodes=15)
    assert all(len(p["types"]) <= 15 for p in large.partitions)
    assert len([r for r in large.records if r["policy"] == "method"]) == 12
    assert all(not r["dropped"] for r in large.records)
    huge = parse_java(b"class A { void f() {" + b"int x=1;" * 100 + b"} }", max_nodes=20)
    assert any(r["dropped"] > 0 for r in huge.records)
    first = parse_java(source, first_nodes=5)
    assert first.records[0]["dropped"] > 0 and len(first.partitions[0]["types"]) == 5


def test_clone_fingerprints_ignore_comments_whitespace_but_not_token_boundaries():
    groups = clone_families(
        {
            "a": b"class A { int f(){return 1;} }",
            "b": b"class A { /* comment */ int f() { return 1; } }",
            "c": b"class B { String s; }",
        }
    )
    assert groups["a"] == groups["b"]
    assert groups["a"] != groups["c"]
