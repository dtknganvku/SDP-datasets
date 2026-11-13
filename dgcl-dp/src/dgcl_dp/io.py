"""Canonical serialization, immutable records, and artifact hashes."""

import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, immutable=False):
    path = Path(path)
    content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to overwrite immutable artifact: {path}")
        return
    with path.open("x" if immutable else "w") as stream:
        stream.write(content)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
