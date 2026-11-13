"""Prepare manifests from explicit, reviewed Git/source/metric/label mappings."""

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

from .acquisition import git, snapshot_source
from .io import digest, read_json, write_json
from .manifests import cross_version, timestamp
from .source import LANGUAGE, clone_families, parse_java
from tree_sitter import Parser


def build_labels(sample_id, release_time, events, collection_cutoff):
    start = timestamp(release_time)
    end = start + timedelta(days=180)
    if end > timestamp(collection_cutoff):
        raise ValueError("Observation window is incomplete")
    qualifying = []
    for event in events:
        if not event.get("confirmed_bug") or not event.get("ancestry_verified"):
            raise ValueError("Every supplied event must have confirmed issue/ancestry evidence")
        for key in ("fixing_commit", "inducing_commit", "issue", "szz_variant", "available_at"):
            if not event.get(key):
                raise ValueError(f"Missing label event field: {key}")
        if start < timestamp(event["fixed_at"]) <= end:
            if timestamp(event["available_at"]) > timestamp(collection_cutoff):
                raise ValueError("Fix evidence arrived after frozen collection cutoff")
            qualifying.append(event)
    available = max([end] + [timestamp(e["available_at"]) for e in qualifying])
    return {
        "sample_id": sample_id,
        "label": int(bool(qualifying)),
        "mechanism": "issue-linked-szz",
        "window_end": end.isoformat(),
        "available_at": available.isoformat(),
        "events": qualifying,
    }


def prepare(mapping_path, output):
    mapping_path, output = Path(mapping_path), Path(output)
    spec = read_json(mapping_path)
    if output.exists():
        raise ValueError("Preparation output must be new; manifests are immutable")
    output.mkdir(parents=True)
    samples, labels, features, sources, exclusions = [], [], [], {}, []
    seen = set()
    parsed_ok = metrics_ok = 0
    for row in spec["records"]:
        identity = {k: row[k] for k in ("project", "release", "commit", "path")}
        sample_id = digest(identity)
        if sample_id in seen:
            raise ValueError("Repeated release/source mapping")
        seen.add(sample_id)
        repo = (mapping_path.parent / row["repo"]).resolve()
        if git(repo, "rev-parse", f"{row['release_ref']}^{{commit}}") != row["commit"]:
            raise ValueError("Release reference does not identify the supplied commit")
        source, source_hash = snapshot_source(repo, row["commit"], row["path"])
        if source_hash != row["source_hash"]:
            raise ValueError("Reviewed source hash does not match Git snapshot")
        parsed = parse_java(source)
        parsed_ok += parsed.status == "ok"
        has_metrics = set(row["metrics"]) == set(spec["schema"]["metrics"])
        metrics_ok += has_metrics
        declarations = [
            n
            for n in Parser(LANGUAGE).parse(source).root_node.named_children
            if n.type.endswith("_declaration") and n.type != "package_declaration"
        ]
        reason = (
            "parse_failure"
            if parsed.status != "ok"
            else "missing_metrics"
            if not has_metrics
            else "not_production"
            if not row["production"]
            else "not_single_top_level_class"
            if len(declarations) != 1 or declarations[0].type != "class_declaration"
            else None
        )
        if reason:
            exclusions.append({"sample_id": sample_id, "reason": reason, **identity})
            continue
        sources[sample_id] = source
        destination = f"sources/{sample_id}.java"
        (output / "sources").mkdir(exist_ok=True)
        (output / destination).write_bytes(source)
        samples.append(
            {
                "sample_id": sample_id,
                **identity,
                "source_path": destination,
                "source_hash": source_hash,
                "prediction_time": row["released_at"],
                "source_time": git(repo, "show", "-s", "--format=%cI", row["commit"]),
                "unit": "file",
                "production": True,
                "top_level_classes": 1,
                "release_verified": True,
                "loc": len(source.splitlines()),
                "extraction_status": "ok",
                "licence": row["licence"],
            }
        )
        features.append({"sample_id": sample_id, "metrics": row["metrics"]})
        for event in row["events"]:
            git(repo, "merge-base", "--is-ancestor", event["inducing_commit"], row["commit"])
            git(repo, "merge-base", "--is-ancestor", row["commit"], event["fixing_commit"])
        labels.append(
            build_labels(sample_id, row["released_at"], row["events"], spec["collection_cutoff"])
        )
    groups = clone_families(sources)
    for row in samples:
        row["clone_group"] = groups[row["sample_id"]]
    label_by_id = {r["sample_id"]: r["label"] for r in labels}
    conflicts = []
    for group in sorted(set(groups.values())):
        members = [i for i, g in groups.items() if g == group]
        if len({label_by_id[i] for i in members}) > 1:
            conflicts.append(
                {
                    "clone_group": group,
                    "sample_ids": members,
                    "labels": [label_by_id[i] for i in members],
                }
            )
    for name, rows in (("samples", samples), ("labels", labels), ("features", features)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    write_json(output / "schema.json", spec["schema"], immutable=True)
    write_json(output / "exclusions.json", exclusions, immutable=True)
    write_json(output / "clone-conflicts.json", conflicts, immutable=True)
    write_json(
        output / "split.json",
        cross_version(samples, labels, spec["collection_cutoff"]),
        immutable=True,
    )
    total = len(spec["records"])
    report = {
        "candidate_samples": total,
        "included_samples": len(samples),
        "parse_coverage": parsed_ok / total if total else 0,
        "metric_coverage": metrics_ok / total if total else 0,
        "mapping_sha256": hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
        "human_label_audit": spec.get("human_label_audit", {"complete": False}),
    }
    write_json(output / "admission.json", report, immutable=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.mapping, args.output), indent=2))


if __name__ == "__main__":
    main()
