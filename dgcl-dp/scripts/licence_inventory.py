"""Inventory installed dependency licence metadata; missing metadata needs review."""

import importlib.metadata
import json


rows = []
for distribution in importlib.metadata.distributions():
    meta = distribution.metadata
    rows.append(
        {
            "name": meta["Name"],
            "version": distribution.version,
            "licence": meta.get("License-Expression") or meta.get("License") or "REVIEW_REQUIRED",
            "classifiers": [c for c in meta.get_all("Classifier", []) if "License" in c],
        }
    )
print(json.dumps(sorted(rows, key=lambda r: r["name"].lower()), indent=2))
