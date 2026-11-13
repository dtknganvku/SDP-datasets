"""Archive provenance audit and manifest checks. No archive is implicitly admitted."""

import argparse
import csv
import io
import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from .io import file_hash, read_json, read_jsonl, write_json
from .manifests import validate


def audit_archives(directory):
    archives, total = [], 0
    for path in sorted(Path(directory).glob("*.zip")):
        files = []
        with ZipFile(path) as archive:
            names = archive.namelist()
            for name in sorted(n for n in names if n.lower().endswith(".csv")):
                rows = csv.reader(
                    io.StringIO(archive.read(name).decode("utf-8-sig", errors="replace"))
                )
                header = next(rows)
                duplicate = [k for k, n in Counter(header).items() if n > 1]
                indices = [i for i, col in enumerate(header) if col.lower() == "bug"]
                valid, missing, invalid, positive, count = 0, 0, 0, 0, 0
                for row in rows:
                    if not row:
                        continue
                    count += 1
                    value = (
                        row[indices[0]].strip()
                        if len(indices) == 1 and len(row) > indices[0]
                        else ""
                    )
                    if not value:
                        missing += 1
                        continue
                    try:
                        bug = float(value)
                        if not math.isfinite(bug) or bug < 0 or not bug.is_integer():
                            raise ValueError("Invalid defect count")
                        valid += 1
                        positive += bug > 0
                    except ValueError:
                        invalid += 1
                total += valid
                files.append(
                    {
                        "path": name,
                        "rows": count,
                        "labelled": valid,
                        "positive": positive,
                        "missing_label": missing,
                        "invalid_label": invalid,
                        "duplicate_headers": duplicate,
                        "columns": header,
                        "admission": "quarantined",
                        "reason": "Exact source release and label provenance unverified",
                    }
                )
            java = [n for n in names if n.endswith(".java")]
            roots = sorted({"/".join(n.split("/")[:2]) for n in java})
        archives.append(
            {
                "archive": path.name,
                "sha256": file_hash(path),
                "csvs": files,
                "java_files": len(java),
                "source_directory_hints": roots,
            }
        )
    return {
        "role": "historical provenance audit only",
        "archives": archives,
        "archive_count": len(archives),
        "csv_count": sum(len(a["csvs"]) for a in archives),
        "parseable_labelled_rows": total,
        "confirmatory_admitted": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--archives", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-archive-audit.json"))
    parser.add_argument("--historical", action="store_true")
    args = parser.parse_args()
    if args.archives:
        result = audit_archives(args.archives)
    else:
        overrides = dict(item.split("=", 1) for item in args.overrides)
        root = Path(overrides.get("manifest", "manifests/primary"))
        if not root.exists() and (Path("manifests") / root).exists():
            root = Path("manifests") / root
        result = validate(
            read_jsonl(root / "samples.jsonl"),
            read_jsonl(root / "labels.jsonl"),
            read_json(root / "split.json"),
            primary=not args.historical,
        )
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "archives"}, indent=2))


if __name__ == "__main__":
    main()
