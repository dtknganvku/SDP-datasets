"""Train a frozen campaign, or regenerate results from immutable predictions."""

import csv
import json
import time
import traceback
from pathlib import Path

import hydra
import joblib
from omegaconf import DictConfig, OmegaConf

from .baselines import model_config
from .demo import create_fixture
from .evaluation import evaluate
from .io import digest, file_hash, read_json, read_jsonl, write_json
from .manifests import indexed, validate
from .runtime import environment
from .search import select
from .source import parse_java
from .statistics import compare_projects, seed_summary
from .training import fit, predict


def load_task(root, cfg):
    root = Path(root)
    samples, labels = read_jsonl(root / "samples.jsonl"), read_jsonl(root / "labels.jsonl")
    split = read_json(root / "split.json")
    validate(samples, labels, split, primary=cfg["primary"])
    if cfg["primary"]:
        admission = read_json(root / "admission.json")
        if min(admission["parse_coverage"], admission["metric_coverage"]) < 0.95:
            raise ValueError("Primary extraction coverage must be at least 95%")
    features = indexed(read_jsonl(root / "features.jsonl"))
    schema = read_json(root / "schema.json")
    if not schema.get("extractor"):
        raise ValueError("An explicit metric extractor identity is required")
    sample_map, label_map = indexed(samples), indexed(labels)
    parsed = {}
    extraction = {}
    for i in sum([split[p] for p in ("train", "validation", "test")], []):
        row = sample_map[i]
        path = (root / row["source_path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Source path escapes manifest root")
        if file_hash(path) != row["source_hash"]:
            raise ValueError(f"Source hash mismatch: {i}")
        parsed[i] = parse_java(
            path.read_bytes(), cfg["model"]["max_nodes"], cfg["model"]["first_nodes"]
        )
        if parsed[i].status != "ok" or (cfg["primary"] and not parsed[i].primary_eligible):
            raise ValueError(f"Common evaluation manifest contains parse failure: {i}")
        extraction[i] = parsed[i].records
        if i not in features:
            raise ValueError(f"Missing metrics: {i}")
    fit_args = {
        "train_ids": split["train"],
        "validation_ids": split["validation"],
        "features": features,
        "parsed": parsed,
        "samples": sample_map,
        "train_labels": {i: label_map[i]["label"] for i in split["train"]},
        "validation_labels": {i: label_map[i]["label"] for i in split["validation"]},
        "schema": schema["metrics"],
    }
    hashes = {
        name: file_hash(root / name)
        for name in ("samples.jsonl", "labels.jsonl", "features.jsonl", "schema.json", "split.json")
    }
    return {
        "name": root.name,
        "root": root,
        "samples": sample_map,
        "labels": label_map,
        "features": features,
        "parsed": parsed,
        "split": split,
        "fit_args": fit_args,
        "hashes": hashes,
        "extraction": extraction,
    }


def checkpoint(fitted, directory):
    directory.mkdir(parents=True, exist_ok=False)
    # Checkpoints are trusted local artifacts; joblib/pickle files must never be loaded untrusted.
    joblib.dump(fitted, directory / "fitted.joblib", compress=3)
    write_json(directory / "metadata.json", fitted.metadata, immutable=True)
    return {
        str(p.relative_to(directory.parent.parent)): file_hash(p)
        for p in sorted(directory.iterdir())
    }


def generate_tables(output):
    output = Path(output)
    lock = read_json(output / "campaign-lock.json")
    completed = read_json(output / "completed.json")
    if completed["campaign_hash"] != digest(lock):
        raise ValueError("Prediction campaign does not match frozen lock")
    for name, checksum in completed["prediction_hashes"].items():
        if file_hash(output / name) != checksum:
            raise ValueError(f"Prediction hash mismatch: {name}")
    results = []
    for name in sorted(completed["prediction_hashes"]):
        rows = read_jsonl(output / name)
        if not rows:
            raise ValueError("Empty predictions")
        first = rows[0]
        metrics = evaluate(
            [r["label"] for r in rows],
            [r["probability"] for r in rows],
            [r["effort"] for r in rows],
            [r["sample_id"] for r in rows],
            first["threshold"],
        )
        fixed = evaluate(
            [r["label"] for r in rows],
            [r["probability"] for r in rows],
            [r["effort"] for r in rows],
            [r["sample_id"] for r in rows],
            0.5,
        )
        results.append(
            {
                "model": first["model"],
                "project": first["project"],
                "task": first["task"],
                "seed": first["seed"],
                **metrics,
                "at_0_5": fixed,
            }
        )
    write_json(output / "metrics.json", results)
    comparisons = (
        compare_projects(results, "dgcl") if any(r["model"] == "dgcl" for r in results) else []
    )
    write_json(output / "project-comparisons.json", comparisons)
    columns = ["task", "model", "metric", "mean", "sd", "seed_ci95_low", "seed_ci95_high", "n"]
    with (output / "table3.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for task, model in sorted({(r["task"], r["model"]) for r in results}):
            group = [r for r in results if (r["task"], r["model"]) == (task, model)]
            for metric in (
                "mcc",
                "pr_auc",
                "f1",
                "roc_auc",
                "recall_at_20_loc",
                "popt",
                "ifa",
                "brier",
                "ece",
            ):
                values = [r[metric] for r in group if r[metric] is not None]
                if values:
                    summary = seed_summary(values)
                    writer.writerow(
                        {
                            "task": task,
                            "model": model,
                            "metric": metric,
                            "mean": summary["mean"],
                            "sd": summary["sd"],
                            "n": summary["n"],
                            "seed_ci95_low": summary["seed_ci95"][0],
                            "seed_ci95_high": summary["seed_ci95"][1],
                        }
                    )
    return {
        "runs": len(results),
        "table": str(output / "table3.csv"),
        "synthetic": lock["synthetic"],
    }


def run(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    output = Path(cfg["output"])
    if cfg["mode"] == "from_predictions":
        return generate_tables(output)
    if cfg["mode"] != "train":
        raise ValueError("mode must be train or from_predictions")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "campaign-lock.json").exists():
        raise ValueError(
            "Campaign already locked; use mode=from_predictions or a new output directory"
        )
    if cfg["experiment"]["name"] == "demo":
        create_fixture(cfg["manifest"])
    manifest = Path(cfg["manifest"])
    if not manifest.exists():
        raise ValueError(
            f"Missing audited manifests: {manifest}. See docs/data-contract.md; run experiment=demo for a synthetic smoke test."
        )
    catalog = read_json(manifest / "catalog.json") if (manifest / "catalog.json").exists() else None
    roots = [manifest / task for task in catalog["tasks"]] if catalog else [manifest]
    tasks = [load_task(root, cfg) for root in roots]
    if len({task["name"] for task in tasks}) != len(tasks):
        raise ValueError("Task directory names must be unique")
    if cfg["primary"]:
        projects = {task["samples"][i]["project"] for task in tasks for i in task["split"]["test"]}
        if len(projects) < 6:
            raise ValueError("Fewer than six admitted projects: deployment claim is blocked")
        if (
            cfg["seeds"] != list(range(10))
            or not cfg["search"]["enabled"]
            or cfg["search"]["trials"] != 20
        ):
            raise ValueError("Confirmatory campaign requires seeds 0-9 and 20-trial selection")
        from .readiness import REQUIRED, readiness

        missing = set(REQUIRED) - set(cfg["models"])
        report = readiness(manifest)
        if missing or not report["confirmatory_ready"]:
            raise ValueError(
                "Confirmatory study gates remain open; run python -m dgcl_dp.readiness. "
                f"Missing mandatory comparisons: {sorted(missing)}"
            )
    pending, hashes, selected = [], {}, {}
    for task in tasks:
        write_json(
            output / "extraction" / f"{task['name']}.json", task["extraction"], immutable=True
        )
        for name in cfg["models"]:
            base = model_config(name, cfg["model"])
            active_task = task
            if base["first_nodes"] != cfg["model"]["first_nodes"]:
                active_task = load_task(task["root"], {**cfg, "model": base})
                write_json(
                    output / "extraction" / f"{task['name']}--{name}.json",
                    active_task["extraction"],
                    immutable=True,
                )
            args = active_task["fit_args"]
            folds = None
            if cfg["primary"] and task["split"]["protocol"] == "cross_project":
                fold_paths = read_json(task["root"] / "source-folds.json")
                fold_tasks = [load_task(task["root"] / p, cfg) for p in fold_paths]
                target = task["split"]["target_project"]
                for fold in fold_tasks:
                    if any(s["project"] == target for s in fold["samples"].values()):
                        raise ValueError("Target project in source-only selection fold")
                folds = [t["fit_args"] for t in fold_tasks]
                if len(folds) < 2:
                    raise ValueError("CPDP requires leave-one-source-project-out selection")
            chosen, study = select(base, cfg["search"], args, folds=folds)
            key = f"{task['name']}/{name}"
            selected[key] = chosen
            write_json(output / "studies" / task["name"] / f"{name}.json", study, immutable=True)
            for seed in cfg["seeds"]:
                run_name = f"{task['name']}--{name}--{seed}"
                fitted = fit(chosen, **args, seed=seed)
                files = checkpoint(fitted, output / "checkpoints" / run_name)
                hashes.update(files)
                pending.append((active_task, name, seed, run_name))
                del fitted
    lock = {
        "config": cfg,
        "selected": selected,
        "checkpoints": hashes,
        "manifests": {task["name"]: task["hashes"] for task in tasks},
        "environment": environment(),
        "implementation": {
            str(p.relative_to(Path(__file__).parent)): file_hash(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "synthetic": cfg["experiment"]["name"] == "demo",
    }
    write_json(output / "campaign-lock.json", lock, immutable=True)
    prediction_hashes = {}
    for task, name, seed, run_name in pending:
        checkpoint_path = Path("checkpoints") / run_name / "fitted.joblib"
        if file_hash(output / checkpoint_path) != lock["checkpoints"][str(checkpoint_path)]:
            raise ValueError("Checkpoint hash mismatch after campaign lock")
        fitted = joblib.load(output / checkpoint_path)
        ids = sorted(task["split"]["test"])
        start = time.perf_counter()
        p = predict(fitted, ids, task["features"], task["parsed"], task["samples"])
        seconds = time.perf_counter() - start
        filename = f"predictions/{run_name}.jsonl"
        path = output / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "sample_id": i,
                "model": name,
                "config_hash": fitted.metadata["config_hash"],
                "seed": seed,
                "probability": float(prob),
                "threshold": fitted.threshold,
                "prediction": int(prob >= fitted.threshold),
                "effort": task["samples"][i]["loc"],
                "runtime_seconds": seconds / len(ids),
                "label": task["labels"][i]["label"],
                "project": task["samples"][i]["project"],
                "task": task["name"],
                "campaign_hash": digest(lock),
            }
            for i, prob in zip(ids, p)
        ]
        with path.open("x") as stream:
            stream.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        prediction_hashes[filename] = file_hash(path)
    write_json(
        output / "completed.json",
        {"campaign_hash": digest(lock), "prediction_hashes": prediction_hashes},
        immutable=True,
    )
    result = generate_tables(output)
    if cfg.get("mlflow"):
        from .tracking import export_mlflow

        export_mlflow(output, cfg, result)
    return result


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try:
        print(json.dumps(run(cfg), indent=2))
    except Exception:
        output = Path(cfg.output)
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "exception": traceback.format_exc(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        write_json(output / f"failed-{time.time_ns()}.json", failure, immutable=True)
        raise


if __name__ == "__main__":
    main()
