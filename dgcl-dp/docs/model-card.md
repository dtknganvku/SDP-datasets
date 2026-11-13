# Model card

Research prototype for binary defect risk at Java production-file release time.
The primary unit is one top-level class including its nested classes. Code and
metrics must describe the prediction snapshot. A positive indicates an implicated
confirmed fix within 180 days; a negative means none observed in that window.

No real-data checkpoint is released here. Generated demo checkpoints use synthetic
labels and must not be used to assess software quality. Model effectiveness,
fairness across projects, calibration under deployment shift and computational
utility are unestablished.

Known limitations: noisy issue/SZZ attribution, potentially unsupported metric
neighborhoods, lexical/structural approximations, oversized method truncation,
clone-disjoint population changes, external model exposure and untested GPU
determinism. Probability estimates require evaluation on the intended population.
Attributions, when added, must not be described as causal explanations.

Training and inference use only training graph context. Training labels influence
weighted classification; the primary instance alignment mask is label-independent.
The supervised contrastive alternatives deliberately use training labels and must
be reported separately. No test-time target adaptation is implemented.
