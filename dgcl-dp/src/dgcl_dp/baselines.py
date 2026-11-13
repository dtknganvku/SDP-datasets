"""Explicit implementation identities; unavailable replications fail closed."""

NEURAL = {
    "dgcl": {},
    "contrastive_pretrain": {"pretrain_kind": "contrastive", "contrastive_weight": 0.0},
    "graphmae_adaptation": {"pretrain_kind": "graphmae_adaptation", "contrastive_weight": 0.0},
    "graphcl_adaptation": {"contrastive_loss": "graphcl_adaptation"},
    "infonce": {"negative_mode": "all"},
    "clone_mask": {"negative_mode": "clone"},
    "random_mask": {"negative_mode": "random"},
    "dual_no_cl": {"contrastive_weight": 0.0},
    "fusion_mlp": {"metric_graph": False, "contrastive_weight": 0.0},
    "metric_mlp": {"metric_graph": False, "view": "metric", "contrastive_weight": 0.0},
    "metric_graph": {"view": "metric", "contrastive_weight": 0.0},
    "ast_sage": {"view": "code", "contrastive_weight": 0.0},
    "ast_gcn": {"view": "code", "code_kind": "gcn", "contrastive_weight": 0.0},
    "ast_gat": {"view": "code", "code_kind": "gat", "contrastive_weight": 0.0},
    "gated_fusion": {"fusion": "gate", "contrastive_weight": 0.0},
    "shuffled_graph": {"graph_control": "shuffled"},
    "no_message_passing": {"graph_control": "none"},
    "first_500": {"first_nodes": 500},
    "supcon": {"contrastive_loss": "supcon"},
    "balanced_supcon": {"contrastive_loss": "balanced_supcon"},
    "progcl_weight_adaptation": {"contrastive_loss": "progcl_weight_adaptation"},
    "focal": {"classification_loss": "focal"},
    "token_cnn": {"sequence_kind": "cnn", "view": "code", "contrastive_weight": 0.0},
    "token_bilstm": {"sequence_kind": "bilstm", "view": "code", "contrastive_weight": 0.0},
    "codebert": {"pretrained_key": "codebert", "view": "code", "contrastive_weight": 0.0},
    "codebert_fusion": {
        "pretrained_key": "codebert",
        "metric_graph": False,
        "contrastive_weight": 0.0,
    },
    "unixcoder": {"pretrained_key": "unixcoder", "view": "code", "contrastive_weight": 0.0},
    "unixcoder_fusion": {
        "pretrained_key": "unixcoder",
        "metric_graph": False,
        "contrastive_weight": 0.0,
    },
    "codet5plus": {"pretrained_key": "codet5plus", "view": "code", "contrastive_weight": 0.0},
    "codet5plus_fusion": {
        "pretrained_key": "codet5plus",
        "metric_graph": False,
        "contrastive_weight": 0.0,
    },
}

# These identities must never silently resolve to the proposed architecture.
REQUIRES_ADAPTER = {
    "dp_ccl": "Full-text protocol, pinned author implementation or declared adaptation",
    "gla_sdp": "Metric LSTM/attention specification and paper-based replication audit",
    "dp_gcn": "Global-dependency graph specification and GraphCodeBERT data-flow preprocessing",
    "graphcodebert": "Pinned checkpoint plus audited data-flow input adapter",
    "progcl": "Audited probability/negative-weight estimator, fitted on training graph only",
    "graphmae": "Audited masked reconstruction/pretraining adaptation",
}


def model_config(name, base):
    from .training import CLASSICAL

    if name in REQUIRES_ADAPTER:
        raise ValueError(f"{name} is not an exact reproduction: {REQUIRES_ADAPTER[name]}")
    if name not in NEURAL and name not in CLASSICAL:
        raise ValueError(f"Unknown model: {name}")
    return {**base, **NEURAL.get(name, {}), "name": name}
