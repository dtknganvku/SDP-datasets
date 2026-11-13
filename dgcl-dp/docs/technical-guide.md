# DGCL-DP

An executable research package for the approved Java file/class defect-prediction
plan. The primary method and leakage controls are implemented. The journal study
is **not complete**: there is no admitted timestamped cohort, no confirmatory
result, and several required literature/pretrained comparators still need audited
implementations. See [implementation status](implementation-status.md) and
[local validation](validation.md).

## Run locally

From this directory, with `uv` installed:

```bash
uv sync --frozen --all-extras
uv run pytest -q
uv run python -m dgcl_dp.audit --archives ..
uv run python -m dgcl_dp.reproduce experiment=demo
uv run python -m dgcl_dp.reproduce experiment=demo mode=from_predictions
uv run python -m dgcl_dp.readiness
```

The demo uses generated Java files and synthetic labels. Its numbers only verify
software execution. It creates checkpoints, immutable predictions, a campaign lock,
run metadata, calibration/inspection curves, project comparisons and `table3.csv`
under `artifacts/demo/run/`. Use a new `output=...` for another campaign: locked
campaigns cannot be overwritten.

Run these commands from `SDP-datasets/dgcl-dp/` after cloning
`https://github.com/dtknganvku/SDP-datasets`. The ZIP archives live in the parent
directory. A short Vietnamese platform guide is in `dgcl-dp/README.md`.

After preparing and auditing real manifests:

```bash
uv run python -m dgcl_dp.prepare mapping.json manifests/project-task
uv run python -m dgcl_dp.audit manifest=primary
uv run python -m dgcl_dp.reproduce experiment=table3
uv run python -m dgcl_dp.reproduce experiment=table3 mode=from_predictions
```

`manifest=primary` resolves to `manifests/primary` in the audit command. A table3
catalog is `manifests/primary/catalog.json`, containing relative task directories:
`{"tasks": ["project-a", "project-b"]}`. The training runner defaults to
`manifests/primary`. It stops if manifests are missing, coverage is below 95%,
fewer than six projects qualify, chronology fails, or evaluation class support is
below 20 per class. Reduced historical pilots require `primary=false` and must be
reported as historical evidence. These software gates do not replace scientific
readiness checks or human label validation.

## Implemented method

- Tree-sitter Java named-node graph approximation, identifier/literal node types,
  child-order node features, forward/reverse parent-child edges, deterministic
  method partitions capped at 2,000 nodes, and explicit truncation records.
- Training-only metric median imputation, constant removal, scaling and symmetric
  Euclidean k-NN (default k=5), with deterministic sample-ID distance ties.
- Two 128-dimensional GraphSAGE encoders, file-level mean pooling, 64-dimensional
  projection heads, pre-projection concatenation and binary MLP classification.
- Weighted BCE plus symmetric masked InfoNCE; masks use direct neighbors,
  closed two-layer receptive-field Jaccard >=0.5, and clone families. Positive
  pairs remain in denominators; anchors without negatives contribute zero
  alignment loss and remain in classification training.
- Frozen training layer caches. A query aggregates incoming training messages;
  it cannot modify training nodes or communicate with another query.

Code graphs are minibatched using PyG `Batch`; the complete training metric graph
is recomputed on each optimizer step. Naturally shuffled batches contain unique
samples. Default Adam settings, validation PR-AUC early stopping, 20 sequential
Optuna trials, seed-11/12 confirmation of the top three, and final seeds 0-9 follow
the plan. Each checkpoint retains its own validation MCC threshold; ties select
the largest threshold. Threshold 0.5 is reported separately.

## Model/control names

Classical/historical metrics: `majority`, `loc`, `inverse_loc`, `lr`, `rf`, `svm`, `hgb`, `dbn`.

Neural: `dgcl`, `metric_mlp`, `metric_graph`, `ast_sage`, `ast_gcn`, `ast_gat`,
`fusion_mlp`, `dual_no_cl`, `gated_fusion`, `infonce`, `clone_mask`, `random_mask`,
`shuffled_graph`, `no_message_passing`, `first_500`, `supcon`, `balanced_supcon`,
`focal`, `progcl_weight_adaptation`, `token_cnn`, `token_bilstm`,
`contrastive_pretrain`, `graphcl_adaptation`, `graphmae_adaptation`.

External encoders: `codebert`, `unixcoder`, `codet5plus` and corresponding
`*_fusion` variants. These use the pinned external tokenizer/checkpoint,
nonoverlapping windows of at most 512 tokens, masked token mean pooling, mean
file-window pooling, and a learned projection. UniXcoder retains its encoder-only
prefix. All backbone parameters are fine-tuned. These are declared file-level
adaptations; full-weight runs and external-corpus exposure remain unvalidated.
The local tests use a stub backbone and do not download large checkpoints.

DBN uses two greedily fitted Bernoulli RBMs followed by balanced logistic
regression; a fixed sigmoid maps standardized metric inputs to [0,1]. The token
CNN uses kernels 3/5/7 and masked max pooling; BiLSTM uses packed sequences.
Both learn a frequency-ranked vocabulary on training Java tokens only, retain
all nonoverlapping 512-token windows, and mean-pool windows to a file.

`contrastive_pretrain` performs training-only masked alignment before supervised
fine-tuning. `graphcl_adaptation` contrasts two 10% masked code-node-type views.
`graphmae_adaptation` pretrains by reconstructing masked code-node categories
and metric features; its decoders are not used for classification. These SSL
conditions use 50 pretraining epochs by default (`+model.pretrain_epochs=...`).
The adaptations do not reproduce the original graph benchmark tasks.

`random_mask` preserves each anchor's actual minibatch negative count in both
directions. It must not be confused with matching counts only in the full graph.
`shuffled_graph` permutes graph identities while preserving graph structure.
Gated fusion is a control, not a parameter-matched claim. Inverse LOC is an
inspection heuristic, not a calibrated defect-probability estimator; calibration
scores for that row are descriptive only.

The ProGCL-style control is explicitly an **adaptation**: a two-component beta
mixture estimates negative probabilities on each training batch; weights are
detached, normalized per anchor, and applied to cross-view negatives. It omits
the original within-view denominator and fixed-epoch global probability fit.
It does not substitute for an audited ProGCL replication in a publication.

Use a selected subset with Hydra overrides:

```bash
uv run python -m dgcl_dp.reproduce experiment=demo output=artifacts/controls \
  'models=[lr,metric_mlp,fusion_mlp,dual_no_cl,infonce,random_mask,dgcl]' \
  'seeds=[0]'
```

## Metrics and provenance

Outputs include MCC, average precision (the declared PR-AUC convention), ROC-AUC,
F1, balanced accuracy, G-mean, Brier, 10 equal-width-bin ECE, full-curve Popt,
Recall@20%LOC and IFA. Default inspection rank is probability/LOC with stable
sample-ID ties; probability-only sensitivity curves are included. LOC <=0 uses
cost 1. Prepared-source effort is physical line count, including blank/comment
lines. Recall at a partial file boundary is linearly interpolated. If no files
are defective, recall/IFA/Popt are undefined (`null`). If optimal and worst areas
coincide, Popt is `null`. Single-class AUC/AP are `null`; neural training or model
selection with an ineligible single-class partition stops explicitly.

Seed intervals use Student t intervals. Statistical comparisons average seeds
within task, then tasks within project. Project bootstrap preserves those
dependencies; paired Wilcoxon, Holm p-values and paired A12 use projects as the
unit. The paired A12 is P(reference > comparator) + 0.5 P(tie).

Raw input files, source bytes, split checksums, code, checkpoints and predictions
are hashed. Configuration and all final checkpoints are frozen before the first
test prediction. Rebuilding tables verifies prediction hashes and never trains.
Outcome fields are kept in separate label manifests; fitting never receives test
labels. Checkpoints use joblib and are intended only for trusted local artifacts.

## Environment and release

Python 3.11.11 and the proposed library pins were installed and smoke-tested on
this Mac using CPU. `uv.lock` and `requirements.lock` contain transitive hashes.
`research` extras include Transformers, MLflow and PyDriller. Set `mlflow=true`
to export a completed campaign to its local MLflow directory. Published model
checkpoint revisions are recorded in `src/dgcl_dp/configs/pretrained.json`; their
weights were not downloaded or used for the reported smoke tests.

Deterministic algorithms are required; unsupported GPU operations fail rather
than silently relaxing determinism. Set `PYTHONHASHSEED=0` before starting Python.
GPU execution and second-machine reproducibility remain unverified.

The Docker base is pinned to a verified multi-platform image digest. The Docker
daemon was unavailable here, so the image has **not** been built or tested:

```bash
docker build -t dgcl-dp .
docker run --rm -v "$PWD/artifacts:/app/artifacts" dgcl-dp experiment=demo
```

The supplied image is the CPU reproduction path; a GPU image needs its own
validated CUDA base, digest and environment record. No cloud resources were
purchased. Before submission, complete data/metric licence review, baseline
audits, the locked study, human link review, second-machine reproduction and a
versioned archive with an actually issued DOI.
