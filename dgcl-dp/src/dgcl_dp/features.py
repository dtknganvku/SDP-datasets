"""Explicit-schema, training-only metric preprocessing."""

import numpy as np


FORBIDDEN = {
    "bug",
    "bugs",
    "label",
    "realbug",
    "heubug",
    "sample_id",
    "project",
    "release",
    "commit",
    "source_hash",
    "prediction_time",
    "available_at",
    "clone_group",
}


class MetricTransform:
    def __init__(self, schema):
        if not schema or len(set(schema)) != len(schema):
            raise ValueError("Metric schema must be nonempty and unique")
        if any(name.lower() in FORBIDDEN for name in schema):
            raise ValueError("Outcome/provenance field in feature schema")
        self.schema = list(schema)

    def matrix(self, records):
        values = []
        for row in records:
            metrics = row["metrics"]
            if set(metrics) - set(self.schema):
                raise ValueError("Undeclared metric fields; do not infer schemas")
            values.append(
                [np.nan if metrics.get(k) is None else float(metrics[k]) for k in self.schema]
            )
        x = np.asarray(values, dtype=np.float64).reshape(-1, len(self.schema))
        if np.isinf(x).any():
            raise ValueError("Infinite metric value")
        return x

    def fit(self, records):
        x = self.matrix(records)
        if not len(x):
            raise ValueError("Cannot fit without training samples")
        self.medians = np.array(
            [np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0 for col in x.T]
        )
        x = np.where(np.isnan(x), self.medians, x)
        self.keep = x.std(axis=0) > 0
        if not self.keep.any():
            raise ValueError("No nonconstant training metrics")
        self.mean = x[:, self.keep].mean(axis=0)
        self.scale = x[:, self.keep].std(axis=0)
        return self

    def transform(self, records):
        x = self.matrix(records)
        return (
            (np.where(np.isnan(x), self.medians, x)[:, self.keep] - self.mean) / self.scale
        ).astype(np.float32)

    def state(self):
        return {
            "schema": self.schema,
            "medians": self.medians.tolist(),
            "keep": self.keep.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }
