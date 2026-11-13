"""Manifest validation and deployment-time split policies.

Features live in separate files; labels and provenance are never feature columns.
"""

from datetime import datetime, timedelta, timezone

from .io import digest


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Timestamp must include timezone: {value}")
    return dt.astimezone(timezone.utc)


def indexed(rows):
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate sample IDs")
    return result


def seal_split(split):
    body = {k: v for k, v in split.items() if k != "checksum"}
    return {**body, "checksum": digest(body)}


def validate(samples, labels, split, primary=True):
    s, y = indexed(samples), indexed(labels)
    expected = seal_split(split)["checksum"]
    if split.get("checksum") != expected:
        raise ValueError("Split checksum mismatch")
    partitions = [split[p] for p in ("train", "validation", "test")]
    if any(not p for p in partitions):
        raise ValueError("Every partition must be nonempty")
    flat = sum(partitions, [])
    if len(set(flat)) != len(flat):
        raise ValueError("Sample partitions overlap or repeat")
    if not set(flat) <= s.keys() or not set(flat) <= y.keys():
        raise ValueError("Missing sample/label records")
    clone_sets = [{s[i]["clone_group"] for i in p} for p in partitions]
    if any(clone_sets[a] & clone_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Clone families cross partitions")
    if split["protocol"] == "cross_project":
        projects = [{s[i]["project"] for i in p} for p in partitions]
        if any(projects[a] & projects[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Cross-project partitions share projects")
    elif split["protocol"] != "cross_version":
        raise ValueError("Unknown split protocol")
    elif len({s[i]["project"] for i in flat}) != 1:
        raise ValueError("A cross-version task must contain one project")
    val_time, test_time = timestamp(split["validation_time"]), timestamp(split["test_time"])
    if val_time >= test_time:
        raise ValueError("Validation must precede test deployment")
    cutoff = timestamp(split["collection_cutoff"])
    if test_time >= cutoff:
        raise ValueError("Test deployment must precede the collection cutoff")
    if primary:
        if any(timestamp(s[i]["prediction_time"]) != test_time for i in split["test"]):
            raise ValueError("Test samples do not match the declared deployment snapshot")
        if split["protocol"] == "cross_version" and any(
            timestamp(s[i]["prediction_time"]) != val_time for i in split["validation"]
        ):
            raise ValueError("Validation samples do not match the declared release snapshot")
        if split["protocol"] == "cross_project" and any(
            s[i]["project"] != split["target_project"] for i in split["test"]
        ):
            raise ValueError("Test project differs from held-out target")
    for part, boundary in zip(partitions, (val_time, test_time, cutoff)):
        for i in part:
            sample, label = s[i], y[i]
            if label["label"] not in (0, 1):
                raise ValueError("Only observed binary labels are allowed")
            if sample["loc"] < 0 or sample["extraction_status"] != "ok":
                raise ValueError(f"Invalid extraction/effort for {i}")
            if primary:
                if not sample.get("release_verified") or not sample.get("production"):
                    raise ValueError(f"Unverified release/production source for {i}")
                if sample.get("top_level_classes") != 1:
                    raise ValueError("Primary unit requires one top-level class")
                prediction = timestamp(sample["prediction_time"])
                if timestamp(sample["source_time"]) > prediction:
                    raise ValueError("Source comes from the future")
                end = timestamp(label["window_end"])
                if end != prediction + timedelta(days=180):
                    raise ValueError("Outcome window must be exactly 180 days")
                if timestamp(label["available_at"]) < end:
                    raise ValueError("Complete labels cannot precede window closure")
                if timestamp(label["available_at"]) > boundary:
                    raise ValueError("Label unavailable at deployment boundary")
                if prediction >= boundary:
                    raise ValueError("Prediction must precede its observation boundary")
                if label.get("mechanism") != "issue-linked-szz":
                    raise ValueError("Primary labels require documented issue-linked SZZ")
                if label["label"] and not label.get("events"):
                    raise ValueError("Positive label has no fixing/inducing evidence")
                if not label["label"] and label.get("events"):
                    raise ValueError("Negative label contains qualifying fix events")
                for event in label.get("events", []):
                    if not (prediction < timestamp(event["fixed_at"]) <= end):
                        raise ValueError("Fix outside observation window")
                    if not event.get("ancestry_verified"):
                        raise ValueError("Unverified inducing-commit ancestry")
                    if not event.get("confirmed_bug"):
                        raise ValueError("Issue is not a confirmed bug")
                    for field in ("fixing_commit", "inducing_commit", "issue", "szz_variant"):
                        if not event.get(field):
                            raise ValueError(f"Missing label provenance: {field}")
                    if event.get("available_at") and timestamp(event["available_at"]) > timestamp(
                        label["available_at"]
                    ):
                        raise ValueError("Label predates its fixing evidence")
    if primary:
        for part in partitions[1:]:
            counts = [sum(y[i]["label"] == c for i in part) for c in (0, 1)]
            if min(counts) < 20:
                raise ValueError("Confirmatory evaluation requires 20 samples per class")
    return {"samples": len(flat), "primary": primary, "checksum": expected}


def remove_later_clones(partitions, samples):
    s, seen, result, exclusions = indexed(samples), set(), {}, []
    for name in ("train", "validation", "test"):
        kept = []
        for i in sorted(partitions[name]):
            if s[i]["clone_group"] in seen:
                exclusions.append({"sample_id": i, "partition": name, "reason": "earlier_clone"})
            else:
                kept.append(i)
        seen.update(s[i]["clone_group"] for i in kept)
        result[name] = kept
    return result, exclusions


def cross_version(samples, labels, collection_cutoff):
    s, y = indexed(samples), indexed(labels)
    if len({row["project"] for row in samples}) != 1:
        raise ValueError("Build a cross-version task for one project at a time")
    releases = sorted({row["prediction_time"] for row in samples}, key=timestamp)
    mature = [
        t
        for t in releases
        if all(
            timestamp(y[i]["available_at"]) <= timestamp(collection_cutoff)
            for i in s
            if s[i]["prediction_time"] == t
        )
    ]
    if not mature:
        raise ValueError("No mature test release")
    test_time = mature[-1]
    candidates = [
        t
        for t in releases
        if timestamp(t) < timestamp(test_time)
        and all(
            timestamp(y[i]["available_at"]) <= timestamp(test_time)
            for i in s
            if s[i]["prediction_time"] == t
        )
    ]
    if not candidates:
        raise ValueError("No mature validation release")
    val_time = candidates[-1]
    train_times = {
        t
        for t in releases
        if timestamp(t) < timestamp(val_time)
        and all(
            timestamp(y[i]["available_at"]) <= timestamp(val_time)
            for i in s
            if s[i]["prediction_time"] == t
        )
    }
    partitions = {
        "train": [i for i in s if s[i]["prediction_time"] in train_times],
        "validation": [i for i in s if s[i]["prediction_time"] == val_time],
        "test": [i for i in s if s[i]["prediction_time"] == test_time],
    }
    partitions, exclusions = remove_later_clones(partitions, samples)
    return seal_split(
        {
            **partitions,
            "protocol": "cross_version",
            "validation_time": val_time,
            "test_time": test_time,
            "collection_cutoff": collection_cutoff,
            "exclusions": exclusions,
        }
    )


def cross_project(
    samples, labels, target, validation_project, validation_time, test_time, collection_cutoff
):
    if target == validation_project:
        raise ValueError("Target cannot be a source validation project")
    y = indexed(labels)
    parts = {"train": [], "validation": [], "test": []}
    for row in samples:
        i, project = row["sample_id"], row["project"]
        name = (
            "test"
            if project == target
            else ("validation" if project == validation_project else "train")
        )
        boundary = {"train": validation_time, "validation": test_time, "test": collection_cutoff}[
            name
        ]
        if timestamp(row["prediction_time"]) < timestamp(boundary) and (
            timestamp(y[i]["available_at"]) <= timestamp(boundary)
        ):
            if name != "test" or row["prediction_time"] == test_time:
                parts[name].append(i)
    parts, exclusions = remove_later_clones(parts, samples)
    return seal_split(
        {
            **parts,
            "protocol": "cross_project",
            "validation_time": validation_time,
            "test_time": test_time,
            "collection_cutoff": collection_cutoff,
            "target_project": target,
            "exclusions": exclusions,
        }
    )
