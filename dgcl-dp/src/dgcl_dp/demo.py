"""Synthetic integration fixture, explicitly excluded from research evidence."""

import hashlib
import json
from pathlib import Path

from .io import write_json
from .manifests import seal_split


def create_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    samples, labels, features = [], [], []
    split = {"train": [], "validation": [], "test": []}
    for part, size in (("train", 24), ("validation", 8), ("test", 8)):
        for j in range(size):
            i = f"synthetic-{part}-{j:03}"
            source = (
                f"class C{part}{j} {{ int f(int x) {{ return x {'+' if j % 2 else '-'} {j}; }} }}\n"
            )
            path = f"sources/{i}.java"
            (directory / "sources").mkdir(exist_ok=True)
            (directory / path).write_bytes(source.encode("utf-8"))
            samples.append(
                {
                    "sample_id": i,
                    "project": "synthetic",
                    "release": part,
                    "commit": "synthetic",
                    "source_path": path,
                    "source_hash": hashlib.sha256(source.encode()).hexdigest(),
                    "unit": "file",
                    "loc": 1 + j,
                    "extraction_status": "ok",
                    "clone_group": i,
                    "prediction_time": {
                        "train": "2022-01-01T00:00:00Z",
                        "validation": "2023-01-01T00:00:00Z",
                        "test": "2024-01-01T00:00:00Z",
                    }[part],
                }
            )
            labels.append({"sample_id": i, "label": j % 2, "mechanism": "synthetic"})
            features.append(
                {"sample_id": i, "metrics": {"synthetic_size": j + 1, "synthetic_branch": j % 3}}
            )
            split[part].append(i)
    split.update(
        {
            "protocol": "cross_version",
            "validation_time": "2023-01-01T00:00:00Z",
            "test_time": "2024-01-01T00:00:00Z",
            "collection_cutoff": "2026-10-31T23:59:59Z",
            "synthetic": True,
        }
    )
    for name, rows in (("samples", samples), ("labels", labels), ("features", features)):
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(directory / "split.json", seal_split(split))
    write_json(
        directory / "schema.json",
        {
            "extractor": "synthetic-integration-fixture",
            "metrics": ["synthetic_size", "synthetic_branch"],
        },
    )
