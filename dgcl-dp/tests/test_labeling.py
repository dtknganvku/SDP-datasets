import os
import subprocess

from dgcl_dp.labeling import collect_release_events


def test_blame_attribution_links_only_release_ancestors_and_windowed_confirmed_fixes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, date=None):
        env = dict(os.environ)
        if date:
            env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
        return subprocess.run(
            ["git", "-C", str(repo), *args], env=env, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Synthetic test")
    git("config", "user.email", "synthetic@example.invalid")
    (repo / "A.java").write_text("class A {\n int f() { return 0; }\n}\n")
    git("add", "A.java")
    git("commit", "-m", "introduce synthetic bug", date="2019-12-01T00:00:00Z")
    inducing = git("rev-parse", "HEAD")
    git("commit", "--allow-empty", "-m", "release", date="2020-01-01T00:00:00Z")
    release = git("rev-parse", "HEAD")
    (repo / "A.java").write_text("class A {\n int f() { return 1; }\n}\n")
    git("add", "A.java")
    git("commit", "-m", "confirmed fix", date="2020-02-01T00:00:00Z")
    fixing = git("rev-parse", "HEAD")
    result = collect_release_events(
        repo,
        release,
        [
            {
                "commit": fixing,
                "confirmed_bug": True,
                "issue": "SYNTHETIC-1",
                "available_at": "2020-02-02T00:00:00Z",
            }
        ],
        "2020-01-01T00:00:00Z",
        "2020-06-29T00:00:00Z",
        "2020-12-31T00:00:00Z",
    )
    event = result["events_by_release_path"]["A.java"][0]
    assert event["inducing_commit"] == inducing and event["fixing_commit"] == fixing
    assert event["ancestry_verified"] and result["requires_human_validation"]
