"""Hash-verified acquisition and exact Git snapshot operations."""

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from .io import file_hash
from .manifests import timestamp


def download(url, sha256, destination):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("Acquisition requires a public HTTPS URL without embedded credentials")
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Expected SHA256 is required before downloading")
    destination = Path(destination)
    if destination.exists():
        if file_hash(destination) != sha256:
            raise ValueError("Existing raw artifact has wrong hash")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temp_name = stream.name
            with urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    stream.write(chunk)
        if file_hash(temp_name) != sha256:
            raise ValueError("Downloaded raw artifact has wrong hash")
        os.link(temp_name, destination)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return destination


def git(repo, *args, binary=False):
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode().strip()


def snapshot_source(repo, commit, path):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use a full immutable Git commit hash")
    if path.startswith(("/", "-")) or ".." in Path(path).parts:
        raise ValueError("Invalid repository-relative source path")
    if git(repo, "rev-parse", f"{commit}^{{commit}}") != commit:
        raise ValueError("Snapshot commit mismatch")
    source = git(repo, "show", f"{commit}:{path}", binary=True)
    return source, hashlib.sha256(source).hexdigest()


def select_half_year_releases(releases):
    """Selection inspects release metadata only, never outcome labels."""
    chosen = {}
    for release in releases:
        date = timestamp(release["released_at"])
        if not 2020 <= date.year <= 2025 or not release["stable"]:
            continue
        key = (release["project"], date.year, 1 if date.month <= 6 else 2)
        previous = chosen.get(key)
        if previous is None or (date, release["commit"]) > (
            timestamp(previous["released_at"]),
            previous["commit"],
        ):
            chosen[key] = release
    return [chosen[k] for k in sorted(chosen)]


def run_ck(jar, expected_sha256, source_directory, output_directory):
    if file_hash(jar) != expected_sha256:
        raise ValueError("CK JAR hash differs from recorded build")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            "java",
            "-jar",
            str(Path(jar).resolve()),
            str(Path(source_directory).resolve()),
            "false",
            "0",
            "true",
            str(output.resolve()) + os.sep,
        ],
        check=True,
    )
    return {p.name: file_hash(p) for p in sorted(output.glob("*.csv"))}
