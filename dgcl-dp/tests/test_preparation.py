import hashlib
import json
import os
import subprocess

from dgcl_dp.io import read_json, read_jsonl
from dgcl_dp.manifests import validate
from dgcl_dp.prepare import prepare


def test_preparation_uses_exact_release_commits_and_immutable_sources(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, date=None):
        env = dict(os.environ)
        if date:
            env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Synthetic test")
    git("config", "user.email", "synthetic@example.invalid")
    records = []
    for release in range(3):
        date = f"{2020 + release}-01-01T00:00:00Z"
        source = f"class R{release} {{ int f() {{ return {release}; }} }}\n".encode()
        (repo / "A.java").write_bytes(source)
        git("add", "A.java")
        git("commit", "-m", f"Synthetic release {release}", date=date)
        commit = git("rev-parse", "HEAD")
        git("tag", f"release-{release}")
        records.append(
            {
                "repo": "repo",
                "project": "synthetic",
                "release": str(release),
                "release_ref": f"release-{release}",
                "commit": commit,
                "path": "A.java",
                "source_hash": hashlib.sha256(source).hexdigest(),
                "released_at": date,
                "production": True,
                "licence": "synthetic-test-only",
                "events": [],
                "metrics": {"size": release + 1},
            }
        )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "collection_cutoff": "2026-10-31T00:00:00Z",
                "schema": {"extractor": "synthetic", "metrics": ["size"]},
                "records": records,
            }
        )
    )
    output = tmp_path / "task"
    result = prepare(mapping, output)
    assert result["included_samples"] == 3 and result["parse_coverage"] == 1
    split = read_json(output / "split.json")
    assert [len(split[p]) for p in ("train", "validation", "test")] == [1, 1, 1]
    validate(
        read_jsonl(output / "samples.jsonl"),
        read_jsonl(output / "labels.jsonl"),
        split,
        primary=False,
    )
