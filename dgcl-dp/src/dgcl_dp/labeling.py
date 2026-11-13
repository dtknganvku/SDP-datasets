"""Conservative issue-linked, whitespace-ignoring removed-line blame collector.

This is a documented SZZ variant, not ground truth. Renames, merge fixes and
insertion-only hunks are explicitly excluded and must enter sensitivity reports.
"""

import argparse
import re
import subprocess
from pathlib import Path

from .acquisition import git
from .io import read_json, write_json
from .manifests import timestamp


VARIANT = "dgcl-removed-line-blame-w-v1"


def is_ancestor(repo, older, newer):
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer], capture_output=True
    )
    if result.returncode not in (0, 1):
        raise ValueError("Unable to verify commit ancestry")
    return result.returncode == 0


def collect_release_events(repo, release, fixes, release_time, window_end, cutoff):
    if not re.fullmatch(r"[0-9a-f]{40}", release):
        raise ValueError("Use a full immutable release commit")
    events, exclusions = {}, []
    for fix in fixes:
        commit = fix["commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("Use full immutable fixing commits")
        if not fix.get("confirmed_bug") or not fix.get("issue"):
            raise ValueError("Reviewed confirmed issue links are required")
        fixed_at = git(repo, "show", "-s", "--format=%cI", commit)
        available = fix["available_at"]
        if timestamp(available) < timestamp(fixed_at):
            raise ValueError("Fix evidence availability precedes the fix")
        if timestamp(available) > timestamp(cutoff):
            exclusions.append({"commit": commit, "reason": "evidence_after_collection_cutoff"})
            continue
        if not (timestamp(release_time) < timestamp(fixed_at) <= timestamp(window_end)):
            exclusions.append({"commit": commit, "reason": "outside_outcome_window"})
            continue
        if not is_ancestor(repo, release, commit):
            exclusions.append({"commit": commit, "reason": "fix_not_descendant_of_release"})
            continue
        parents = git(repo, "show", "-s", "--format=%P", commit).split()
        if len(parents) != 1:
            exclusions.append({"commit": commit, "reason": "merge_or_root_fix"})
            continue
        parent = parents[0]
        diff = git(
            repo,
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--unified=0",
            parent,
            commit,
            "--",
            "*.java",
        )
        path, attributed = None, set()
        for line in diff.splitlines():
            if line.startswith("--- a/"):
                path = line[len("--- a/") :]
                if path.startswith('"') or "\\" in path:
                    exclusions.append({"commit": commit, "reason": "quoted_or_escaped_path"})
                    path = None
            elif line.startswith("--- /dev/null"):
                path = None
            elif line.startswith("@@"):
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+", line)
                if not match:
                    raise ValueError("Unexpected unified diff header")
                start, count = int(match[1]), int(match[2] or 1)
                if count == 0 or path is None:
                    exclusions.append(
                        {"commit": commit, "path": path, "reason": "insertion_only_or_new_file"}
                    )
                    continue
                if not git(repo, "ls-tree", release, "--", path):
                    exclusions.append(
                        {"commit": commit, "path": path, "reason": "no_exact_release_path"}
                    )
                    continue
                blame = git(
                    repo,
                    "blame",
                    "-w",
                    "--line-porcelain",
                    "-L",
                    f"{start},{start + count - 1}",
                    parent,
                    "--",
                    path,
                )
                for header in blame.splitlines():
                    found = re.match(r"^([0-9a-f]{40}) \d+ \d+(?: \d+)?$", header)
                    if found and is_ancestor(repo, found[1], release):
                        attributed.add((path, found[1]))
        for path, inducing in sorted(attributed):
            events.setdefault(path, []).append(
                {
                    "fixing_commit": commit,
                    "inducing_commit": inducing,
                    "issue": fix["issue"],
                    "confirmed_bug": True,
                    "ancestry_verified": True,
                    "fixed_at": fixed_at,
                    "available_at": available,
                    "szz_variant": VARIANT,
                }
            )
        if not attributed:
            exclusions.append({"commit": commit, "reason": "no_eligible_removed_line_attribution"})
    return {
        "variant": VARIANT,
        "events_by_release_path": events,
        "exclusions": exclusions,
        "requires_human_validation": True,
        "collection_completeness": "Provided issue/fix inventory must be independently audited",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = read_json(args.spec)
    result = collect_release_events(
        repo=args.spec.parent / spec["repo"],
        release=spec["release_commit"],
        fixes=spec["fixes"],
        release_time=spec["released_at"],
        window_end=spec["window_end"],
        cutoff=spec["collection_cutoff"],
    )
    write_json(args.output, result, immutable=True)


if __name__ == "__main__":
    main()
