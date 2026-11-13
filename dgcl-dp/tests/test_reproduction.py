import json

import pytest

from dgcl_dp.baselines import model_config
from dgcl_dp.reproduce import generate_tables, run
from dgcl_dp.robustness import flip_training_labels, metric_noise


def test_locked_campaign_predictions_regenerate_and_detect_tampering(tmp_path, config):
    cfg = {
        **config,
        "output": str(tmp_path / "campaign"),
        "manifest": str(tmp_path / "fixture"),
        "models": ["lr", "dgcl"],
        "seeds": [0],
    }
    result = run(cfg)
    assert result["runs"] == 2 and result["synthetic"]
    output = tmp_path / "campaign"
    before = (output / "table3.csv").read_bytes()
    generate_tables(output)
    assert (output / "table3.csv").read_bytes() == before
    with pytest.raises(ValueError, match="already locked"):
        run(cfg)
    predictions = next((output / "predictions").glob("*.jsonl"))
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[0]["probability"] = 0.123456
    predictions.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="hash mismatch"):
        generate_tables(output)


def test_unavailable_reproductions_are_explicit(config):
    with pytest.raises(ValueError, match="not an exact reproduction"):
        model_config("dp_gcn", config["model"])


def test_stress_never_modifies_held_out_rows(task):
    labels = {i: r["label"] for i, r in task["labels"].items()}
    train = task["split"]["train"]
    noisy, changed = flip_training_labels(labels, train, 0.2, 4)
    assert set(changed) <= set(train)
    assert all(noisy[i] == labels[i] for i in task["split"]["test"])
    features = metric_noise(task["features"], train, ["synthetic_size"], 0.1, 5)
    assert all(features[i] == task["features"][i] for i in task["split"]["test"])
