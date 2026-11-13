# Implementation status

This file distinguishes executable software from the research activities in the
[approved plan](approved-plan.md). No completed journal results are claimed.

| Plan area | Implemented and exercised | Remaining work |
|---|---|---|
| Local data audit | Seven ZIPs, 18 CSVs, 7,749 parseable labels, missing/invalid labels, duplicate columns, source-directory hints and raw hashes | Resolve every release/source mismatch; permissions for metric redistribution |
| Data contracts | Immutable manifests, Git mapping builder, half-year selection, 180-day labels, conservative removed-line blame SZZ collector, clone detection/exclusion, chronology and class support checks | Collect issue histories; audit SZZ variant/alternative; human review of >=100 positive links |
| Primary model | Named-node Java graphs, method partitions, training-only transforms/k-NN, dual GraphSAGE, masked symmetric InfoNCE, BCE, isolated cache inference | Full data/hardware validation and development novelty gate |
| Baselines | Classical/DBN, token CNN/BiLSTM, graph/fusion controls; all/clone/random masks; first-500, SupCon/balanced SupCon, focal controls; training-only contrastive/GraphCL/GraphMAE adaptations | Distance-matched/random averaging controls; capacity matching; publication-level adaptation audits |
| ProGCL | Explicit beta-mixture cross-view weighting adaptation, with pinned inspected upstream source | Audit global fixed-epoch/within-view replication before treating as mandatory close-method control |
| Pretrained | Real immutable revisions and CodeBERT/UniXcoder/CodeT5+ code-only/fusion file-window adapters; stub-backbone tests | GraphCodeBERT data flow, actual pretrained-weight runs, exposure audit |
| Close methods | Explicit unavailable statuses with actionable errors | DP-CCL, GLA-SDP, DP-GCN full-text and implementation audits/reproductions |
| Training | Sequential Optuna, top-three confirmation, validation stopping/threshold, seeds, immutable checkpoints, local MLflow export | Equal-budget primary runs; published-baseline search-space audits |
| Evaluation | Discrimination, effort curves, calibration data, project bootstrap, paired Wilcoxon/Holm/A12, seed CIs | Journal figure generation, failure-inclusive analyses, population/retention tables |
| Robustness | Training-only fixed-level label flips and metric noise helpers | Full preregistered campaign, imbalance/rewiring/augmentation studies, explanation fidelity |
| Reproduction | Hash-locked packages and image base, synthetic campaign, cached prediction tables, failed-run records | Docker/GPU and second-machine validation, licence review, real Zenodo DOI |
| Manuscript | Approved plan preserved and result-driven outline | Complete citation screening, write results only after valid experiments, journal formatting |

`python -m dgcl_dp.readiness` reports these open gates. It deliberately returns
`confirmatory_ready: false` in this workspace. Running a reduced historical pilot
does not clear them. Adding an adapter requires meaningful validation and accurate
reimplementation/adaptation labeling; do not remove a pending status merely to
make the readiness flag green.

Tests cover all eight mandatory automated-test categories from the plan, plus
immutable campaign behavior, tamper detection, repeated table generation, Java
partitioning, deterministic matched random masks and fixed training-only stresses.

## Documentation evidence

Context7 CLI library resolution and docs retrieval were performed for PyG,
py-tree-sitter, Hydra, scikit-learn 1.7.1, Optuna and Transformers 4.51.3 on
2026-11-13. Hydra remains explicitly on 1.3 with `version_base="1.3"` despite
newer documentation covering 1.4. Implemented APIs were then tested in the pinned
environment. ProGCL source inspected at commit
`3580f640b3e0c73dde6757b9a4196d5dba008ebd`:
`https://github.com/junxia97/ProGCL/blob/3580f640b3e0c73dde6757b9a4196d5dba008ebd/pGRACE/model.py`.
