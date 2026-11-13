# Local validation: 2026-11-13

Verified on this machine with CPython 3.11.11 and the hash-locked environment.

| Check | Observed outcome |
|---|---|
| `uv sync --frozen --all-extras` | All pinned components install together |
| `uv run pytest -q` | 25 tests pass, no warnings |
| `uv run ruff check src tests scripts` | Pass |
| `uv build --wheel` | Wheel builds; Hydra YAML and checkpoint registry included |
| Archive audit | 7 archives, 18 CSVs, 7,749 parseable labelled rows; zero confirmatory admissions |
| Synthetic full CPU campaign | All 32 implemented non-pretrained variants complete with seed 0 |
| Cached predictions | Tables regenerate from the locked campaign without retraining |
| Sequential tuning smoke | LR and DGCL each complete three development trials and confirmation runs |
| Local MLflow export | Completed campaign exports to its local tracking directory |
| Exact-source preparation | Temporary Git repository verifies release refs, commits, hashes and splits |
| SZZ collector | Temporary Git repository verifies issue-linked removed-line ancestry attribution |
| Dataset worktree | Existing SDP-datasets repository remains clean |

Inspect the final CPU smoke artifacts under `artifacts/verified-cpu-smoke/`.
These contain **synthetic** examples and cannot support research conclusions.
The smoke tuning budget is three trials; the confirmatory default remains 20.
Smoke pretraining uses two epochs; the default is 50. No real primary test
campaign or compute-budget estimate has been validated.

Pretrained adapters are tested with a local stub backbone/tokenizer; real
CodeBERT/UniXcoder/CodeT5+ checkpoint weights were not downloaded or trained.
GraphCodeBERT data-flow processing and exact recent-method replications remain
open. Human link review, GPU runs, a Docker build, second-machine reproduction,
licence clearance and archive publication remain unverified. Docker's daemon
was unavailable; only its client was present.
