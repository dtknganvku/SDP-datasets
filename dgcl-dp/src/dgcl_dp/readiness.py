"""Report study gates without presenting pilot software as completed research."""

import argparse
import json
from pathlib import Path

from .baselines import NEURAL, REQUIRES_ADAPTER
from .io import read_json
from .training import CLASSICAL


REQUIRED = sorted(
    CLASSICAL
    | {
        "metric_mlp",
        "ast_gcn",
        "ast_sage",
        "ast_gat",
        "metric_graph",
        "fusion_mlp",
        "dual_no_cl",
        "dgcl",
        "token_cnn",
        "token_bilstm",
        "codebert",
        "codebert_fusion",
        "unixcoder",
        "unixcoder_fusion",
        "codet5plus",
        "codet5plus_fusion",
        "infonce",
        "random_mask",
        "clone_mask",
    }
    | set(REQUIRES_ADAPTER)
)


def readiness(manifest):
    manifest = Path(manifest)
    cards = list(manifest.rglob("admission.json")) if manifest.exists() else []
    admitted = []
    positive_links = 0
    reviewed_projects = set()
    for path in cards:
        card = read_json(path)
        audit = card.get("human_label_audit", {})
        if min(card.get("parse_coverage", 0), card.get("metric_coverage", 0)) >= 0.95:
            admitted.append(str(path.parent))
        positive_links += audit.get("positive_links_reviewed", 0)
        if audit.get("independent_review_complete"):
            reviewed_projects.add(str(path.parent))
    pending = {name: detail for name, detail in REQUIRES_ADAPTER.items() if name in REQUIRED}
    return {
        "implemented_models": sorted(CLASSICAL | set(NEURAL)),
        "pending_adapters": pending,
        "candidate_admitted_tasks": admitted,
        "human_positive_links_reviewed": positive_links,
        "independently_reviewed_tasks": sorted(reviewed_projects),
        "confirmatory_ready": not pending
        and len(admitted) >= 6
        and positive_links >= 100
        and bool(reviewed_projects),
        "remaining_external_gates": [
            "Complete timestamped cohort and licence review",
            "At least six eligible projects, 20 positives/negatives per evaluation task",
            "Human audit of >=100 positive links with independent subset review",
            "Development mechanism gate against random masking and ProGCL",
            "Frozen 20-trial/ten-seed campaign on suitable hardware",
            "Second-machine container reproduction and issued archive DOI",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default="manifests/primary")
    args = parser.parse_args()
    print(json.dumps(readiness(args.manifest), indent=2))


if __name__ == "__main__":
    main()
