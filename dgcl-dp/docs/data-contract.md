# Data contracts and preparation

Manifests are immutable JSON/JSONL records. All timestamps include a timezone;
all source commits and external model revisions are immutable hashes. Preserve
original historical labels in their own cohort; never rewrite them as SZZ labels.

## Task directory

- `samples.jsonl`: sample_id, project, release, commit, source_path (relative to
  task root), source_hash (SHA256), prediction_time, source_time, unit, loc,
  extraction_status, clone_group, release_verified, production, top_level_classes.
- `labels.jsonl`: sample_id, label (observed 0/1), mechanism, window_end,
  available_at, and events. Events contain fixing_commit, inducing_commit, issue,
  fixed_at, available_at, szz_variant and ancestry_verified. Window end is exactly
  prediction_time +180 days. Labels become available after the complete window
  and after all supporting evidence arrives.
- `features.jsonl`: sample_id and a `metrics` object only. No labels, release
  indicators, issue text, defect counts or future information are features.
- `schema.json`: extractor identity/version/commit and ordered `metrics` names.
  Feature definitions must be checked before combining extractors.
- `split.json`: train, validation and test ID lists; protocol, validation_time,
  test_time, collection_cutoff, exclusions and a SHA256 checksum of the canonical
  body. Construct with `seal_split`; edits invalidate the checksum.
- `admission.json`: candidate/included counts, parse_coverage, metric_coverage,
  mapping_sha256 and human_label_audit. Retain every exclusion separately.

A primary cross-project task also requires `target_project` in its split and
`source-folds.json`, a list of relative source-validation task directories. None
may contain the target project. These are source-only leave-one-project-out
folds with their own training-fitted preprocessing and temporal checks.

## Reviewed Git mappings

`python -m dgcl_dp.prepare mapping.json manifests/project-task` builds a
cross-version task from an explicit mapping. It checks the release ref against a
full Git commit, source SHA256, parse eligibility, production status, declared
metrics, and inducing/release/fixing ancestry. It computes label-independent exact
and five-token-gram clone families, removes later cross-partition clones, and
records conflicting labels and exclusions.

The mapping contains:

```json
{
  "collection_cutoff": "2026-10-31T23:59:59Z",
  "schema": {
    "extractor": "CK commit 8a1ef916d597d33ad619d748b616bb7dbc3f0c36; record JAR SHA256",
    "metrics": ["wmc", "cbo", "rfc"]
  },
  "human_label_audit": {"complete": false, "positive_links_reviewed": 0},
  "records": []
}
```

Each record supplies `project`, `release`, `release_ref`, `repo` (relative to the
mapping), full `commit`, `path`, reviewed `source_hash`, `released_at`, `production`,
`licence`, `metrics` and `events`. These values require real provenance; do not
populate records using filename similarity. An event also needs `confirmed_bug`.
For negatives, an empty event list is valid only after complete collection of all
qualifying issue-linked fixes through the entire window. The code cannot certify
the completeness or correctness of issue histories; record that collection audit.

The acquisition module provides public HTTPS downloads that require an expected
SHA256, pinned Git snapshot reads, fixed half-year release selection, and a CK JAR
runner that verifies the archived JAR hash. CK must be built from the pinned source
commit and its actual resulting JAR hash recorded before extraction. The extractor
was not built on this machine. The separate SZZ collector is available as
`python -m dgcl_dp.labeling spec.json events.json`. Its spec supplies `repo`, a
full `release_commit`, `released_at`, `window_end`, `collection_cutoff`, and
`fixes` containing full `commit`, `issue`, `confirmed_bug`, and `available_at`.
It uses `git blame -w` on removed lines of single-parent Java fixes, checks
inducing/release/fixing ancestry, and records every skipped fix/path/hunk.
Renames, merge fixes and insertion-only changes remain explicit limitations.
The returned events are grouped by exact release path for the reviewed mapping.
Issue confirmation, complete issue/fix enumeration, alternative attribution,
and independent human review remain external research steps.

## Primary selection and exclusions

Choose the last stable release of each half-year in 2020-2025 using only release
metadata. Target projects are ActiveMQ, Camel, Derby, HBase, Hive, Lucene, Wicket
and Ant. Freeze collection on 2026-10-31. Training labels must mature before
validation deployment; validation labels must mature before test deployment.
Test labels must be fully observed before the collection cutoff.

Near clones use fixed token-five-gram Jaccard >=0.9 and token length ratio >=0.8.
Comment and whitespace removal uses the Java lexer; raw/token hashes preserve
token boundaries. Clone grouping is transitive. Families belong to their earliest
partition; later members are excluded. This changes the estimand to novel-code
generalization, and retained population sizes must be reported.

Primary sample generation includes a single top-level class only; nested classes
remain inside it. Parse failures, multiple top-level types, missing metric
extractions and nonproduction code are explicit exclusions. Empty/missing numeric
metric cells may be imputed from training data; a completely failed extraction
must not be represented as a fabricated clean sample.

The local seven archives are quarantined historical inputs. Their audit is at
`artifacts/local-archive-audit.json`. A directory name, matching filename, or
archive README does not certify exact source/CSV release alignment.
