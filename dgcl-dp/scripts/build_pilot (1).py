"""Build a REDUCED HISTORICAL PILOT manifest from the seven local archives.

Each sample is one Java file from an archive, joined to one labelled row of the
PROMISE-style metric CSV that sits next to it. This is NOT the primary study:
release alignment, label provenance and release dates are not verified, and
prediction times are ordinal placeholders. Results must be called "pilot".

Run from dgcl-dp/:
    uv run python scripts/build_pilot.py --tasks cv   # within-project, old -> new release
    uv run python scripts/build_pilot.py --tasks cp   # cross-project, leave one project out
    uv run python scripts/build_pilot.py --stats      # only print the match table
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

import pandas as pd
import tree_sitter_java
from tree_sitter import Language, Parser

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dgcl_dp.io import write_json  # noqa: E402
from dgcl_dp.manifests import remove_later_clones, seal_split, validate  # noqa: E402
from dgcl_dp.source import clone_families  # noqa: E402

PARSER = Parser(Language(tree_sitter_java.language()))
METRICS = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3",
    "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc",
]  # fmt: skip
CUTOFF = "2026-10-31T23:59:59Z"
PLACEHOLDER_TIMES = {
    "train": "2001-01-01T00:00:00Z",
    "validation": "2002-01-01T00:00:00Z",
    "test": "2003-01-01T00:00:00Z",
}
EXTRACTOR = "historical PROMISE-style CK metric CSV; release alignment unverified (pilot only)"

# Cross-version tasks (train on an older release, test on a newer one of the same project).
CV_TASKS = [
    {"name": "xalan-cv", "protocol": "cross_version",
     "train": ["xalan-2.4"], "validation": ["xalan-2.5"], "test": ["xalan-2.6"]},
    {"name": "lucene-cv", "protocol": "cross_version",
     "train": ["lucene-2.0"], "validation": ["lucene-2.2"], "test": ["lucene-2.4"]},
    {"name": "ant-cv", "protocol": "cross_version",
     "train": ["ant-1.5"], "validation": ["ant-1.6"], "test": ["ant-1.7"]},
]  # fmt: skip

# Cross-project tasks: the target project is the test set, the next project in
# CP_TARGETS is the validation set, every other project is training data.
# One release per project. poi-3.0 is not used: its CSV sits in a folder named
# "poi 3.13 src", so the source may come from another release. jedit-4.3,
# log4j-1.2 and xalan-2.7 are not used either (almost one class only).
CP_VERSIONS = {
    "ant": "ant-1.7", "jedit": "jedit-4.0", "log4j": "log4j-1.1", "lucene": "lucene-2.4",
    "poi": "poi-1.5", "synapse": "synapse-1.2", "xalan": "xalan-2.6",
}  # fmt: skip
CP_TARGETS = ["ant", "jedit", "lucene", "poi", "synapse", "xalan"]  # log4j is too small to test on


def cp_tasks():
    tasks = []
    for i, target in enumerate(CP_TARGETS):
        val = CP_TARGETS[(i + 1) % len(CP_TARGETS)]
        train = [v for k, v in CP_VERSIONS.items() if k not in (target, val)]
        tasks.append(
            {"name": f"{target}-cp", "protocol": "cross_project", "target_project": target,
             "train": train, "validation": [CP_VERSIONS[val]], "test": [CP_VERSIONS[target]]}
        )  # fmt: skip
    return tasks


def version_key(zip_stem, csv_name):
    found = re.search(r"(\d+\.\d+)", Path(csv_name).name)
    return f"{zip_stem.lower()}-{found.group(1)}" if found else None


def parse_status(source):
    root = PARSER.parse(source).root_node
    top = sum(
        c.type in {"class_declaration", "interface_declaration", "enum_declaration"}
        for c in root.named_children
    )
    return ("parse_failure" if root.has_error else "ok"), top


def collect(archives):
    """Return {version_key: {"rows": [...], "exclusions": Counter, "csv_rows": int}}."""
    versions = {}
    for zip_path in sorted(Path(archives).glob("*.zip")):
        with ZipFile(zip_path) as z:
            names = z.namelist()
            for csv_path in sorted(n for n in names if n.lower().endswith(".csv")):
                key = version_key(zip_path.stem, csv_path)
                if key is None:
                    continue
                df = pd.read_csv(BytesIO(z.read(csv_path)), low_memory=False)
                prefix = csv_path.rsplit("/", 1)[0] + "/"
                by_base = defaultdict(list)
                for n in names:
                    if n.startswith(prefix) and n.lower().endswith(".java"):
                        by_base[n.rsplit("/", 1)[-1]].append(n)
                ident = "name.1" if "name.1" in df.columns else "name"
                bug = pd.to_numeric(df["bug"], errors="coerce")
                excl, picked = Counter(), defaultdict(list)
                for idx, row in df.iterrows():
                    if pd.isna(bug.loc[idx]):
                        excl["no_label_in_csv"] += 1  # e.g. SourceMeter-only rows in ant-1.5/1.6
                        continue
                    if pd.isna(row[ident]) or not str(row[ident]).strip():
                        excl["missing_identifier"] += 1
                        continue
                    fqcn = str(row[ident]).strip()
                    suffix = "/" + fqcn.replace(".", "/") + ".java"
                    cands = [p for p in by_base.get(suffix.rsplit("/", 1)[-1], []) if p.endswith(suffix)]
                    if not cands:
                        excl["no_java_file"] += 1
                        continue
                    hashes = {hashlib.sha256(z.read(p)).hexdigest() for p in cands}
                    if len(hashes) > 1:
                        excl["ambiguous_different_content"] += 1
                        continue
                    chosen = sorted(cands, key=lambda p: (len(p), p))[0]
                    if len(cands) > 1:
                        excl["_identical_copies_resolved"] += 1  # informational, row is kept
                    picked[chosen].append((idx, row, fqcn))
                rows = []
                for path, group in sorted(picked.items()):
                    if len(group) > 1:
                        excl["several_rows_same_file"] += len(group)
                        continue
                    idx, row, fqcn = group[0]
                    source = z.read(path)
                    status, top = parse_status(source)
                    if status != "ok":
                        excl["parse_failure"] += 1
                        continue
                    metrics = {
                        m: (None if pd.isna(row[m]) else float(row[m])) for m in METRICS
                    }
                    loc = metrics["loc"] if metrics["loc"] is not None else source.count(b"\n") + 1
                    rows.append(
                        {
                            "fqcn": fqcn,
                            "zip_path": path,
                            "source": source,
                            "label": int(bug.loc[idx] > 0),
                            "bug_count": float(bug.loc[idx]),
                            "metrics": metrics,
                            "loc": max(0.0, float(loc)),
                            "top_level_types": top,
                            "archive": zip_path.name,
                            "csv": csv_path,
                        }
                    )
                versions[key] = {"rows": rows, "exclusions": excl, "csv_rows": len(df)}
    return versions


def print_stats(versions):
    print(f"{'version':14} {'csv':>5} {'kept':>5} {'pos':>5} {'neg':>5} {'pos%':>6}  exclusions")
    for key in sorted(versions):
        v = versions[key]
        pos = sum(r["label"] for r in v["rows"])
        kept = len(v["rows"])
        ex = {k: n for k, n in v["exclusions"].items() if not k.startswith("_")}
        print(f"{key:14} {v['csv_rows']:5} {kept:5} {pos:5} {kept - pos:5} "
              f"{100 * pos / max(kept, 1):6.1f}  {ex}")


def build_task(task, versions, out):
    root = Path(out) / task["name"]
    (root / "sources").mkdir(parents=True, exist_ok=True)
    samples, labels, features, sources, part_of = [], [], [], {}, {}
    for part in ("train", "validation", "test"):
        for key in task[part]:
            for n, r in enumerate(versions[key]["rows"]):
                sid = f"{key}-{n:04d}"
                rel = f"sources/{sid}.java"
                (root / rel).write_bytes(r["source"])
                project, release = key.rsplit("-", 1)
                samples.append(
                    {
                        "sample_id": sid, "project": project, "release": release,
                        "commit": "historical-archive-unverified", "source_path": rel,
                        "source_hash": hashlib.sha256(r["source"]).hexdigest(),
                        "unit": "file", "loc": r["loc"], "extraction_status": "ok",
                        "prediction_time": PLACEHOLDER_TIMES[part],
                        "release_verified": False, "production": None,
                        "top_level_classes": r["top_level_types"],
                        "fqcn": r["fqcn"], "archive": r["archive"], "csv": r["csv"],
                        "archive_source_path": r["zip_path"],
                    }
                )  # fmt: skip
                labels.append({"sample_id": sid, "label": r["label"],
                               "mechanism": "historical-csv-bug-count-gt-0"})
                features.append({"sample_id": sid, "metrics": r["metrics"]})
                sources[sid], part_of[sid] = r["source"], part
    families = clone_families(sources)
    for s in samples:
        s["clone_group"] = families[s["sample_id"]]
    partitions = {p: [i for i in part_of if part_of[i] == p] for p in PLACEHOLDER_TIMES}
    partitions, exclusions = remove_later_clones(partitions, samples)
    keep = set(sum(partitions.values(), []))
    for i in set(part_of) - keep:
        (root / f"sources/{i}.java").unlink()
    samples = [s for s in samples if s["sample_id"] in keep]
    labels = [y for y in labels if y["sample_id"] in keep]
    features = [f for f in features if f["sample_id"] in keep]
    split = seal_split(
        {
            **partitions,
            "protocol": task["protocol"],
            "validation_time": PLACEHOLDER_TIMES["validation"],
            "test_time": PLACEHOLDER_TIMES["test"],
            "collection_cutoff": CUTOFF,
            "exclusions": exclusions,
            "pilot": True,
            "time_semantics": "ordinal placeholders, not real release dates",
            **({"target_project": task["target_project"]} if "target_project" in task else {}),
        }
    )
    for name, rows in (("samples", samples), ("labels", labels), ("features", features)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(root / "split.json", split)
    write_json(root / "schema.json", {"extractor": EXTRACTOR, "metrics": METRICS})
    validate(samples, labels, split, primary=False)
    by_id = {y["sample_id"]: y["label"] for y in labels}
    counts = {p: [sum(by_id[i] == c for i in ids) for c in (0, 1)] for p, ids in partitions.items()}
    write_json(
        root / "admission.json",
        {
            "pilot": True,
            "confirmatory": False,
            "warning": "Historical archives; release alignment, labels and dates unverified.",
            "versions": {p: task[p] for p in PLACEHOLDER_TIMES},
            "kept_per_partition": {p: len(v) for p, v in partitions.items()},
            "class_counts_neg_pos": counts,
            "later_clone_exclusions": dict(Counter(e["partition"] for e in exclusions)),
            "match_exclusions": {
                k: {x: n for x, n in versions[k]["exclusions"].items() if not x.startswith("_")}
                for p in PLACEHOLDER_TIMES for k in task[p]
            },
        },
    )
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", default="..")
    ap.add_argument("--tasks", choices=["cv", "cp"], default="cv")
    ap.add_argument("--out", default=None, help="default: manifests/pilot (cv) or manifests/pilot_cp (cp)")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    versions = collect(args.archives)
    print_stats(versions)
    if args.stats:
        return
    out = Path(args.out or ("manifests/pilot" if args.tasks == "cv" else "manifests/pilot_cp"))
    tasks = CV_TASKS if args.tasks == "cv" else cp_tasks()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; choose a new --out (manifests are immutable)")
    built = []
    for task in tasks:
        counts = build_task(task, versions, out)
        built.append(task["name"])
        print(task["name"], "[neg, pos] per partition:", counts)
    write_json(out / "catalog.json", {"tasks": built, "pilot": True})
    print("Wrote", out, "with tasks", built)


if __name__ == "__main__":
    main()
