"""Equal-completed-trial, sequential development selection."""

import numpy as np
import optuna

from .training import CLASSICAL, fit


def suggest(trial, base):
    cfg = dict(base)
    name = cfg["name"]
    if name in ("lr", "svm"):
        cfg["C"] = trial.suggest_float("C", 1e-3, 1e3, log=True)
    elif name == "rf":
        cfg["n_estimators"] = trial.suggest_categorical("n_estimators", [100, 300, 500])
        cfg["max_depth"] = trial.suggest_categorical("max_depth", [None, 5, 10, 20])
    elif name == "hgb":
        cfg["lr"] = trial.suggest_float("lr", 0.01, 0.3, log=True)
    elif name == "dbn":
        cfg["hidden"] = trial.suggest_categorical("hidden", [64, 128, 256])
        cfg["rbm_lr"] = trial.suggest_float("rbm_lr", 0.001, 0.1, log=True)
        cfg["rbm_epochs"] = trial.suggest_categorical("rbm_epochs", [20, 50])
    elif name not in CLASSICAL:
        cfg.update(
            {
                "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
                "dropout": trial.suggest_categorical("dropout", [0.1, 0.3, 0.5]),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            }
        )
        if cfg["metric_graph"]:
            cfg["k"] = trial.suggest_categorical("k", [3, 5, 10, 15])
        if cfg["contrastive_weight"]:
            cfg["temperature"] = trial.suggest_categorical("temperature", [0.05, 0.1, 0.2, 0.5])
            cfg["contrastive_weight"] = trial.suggest_categorical(
                "contrastive_weight", [0.1, 0.5, 1.0]
            )
    return cfg


def select(base, search, fit_args, folds=None):
    """folds supplies leave-one-source-project-out fitting arguments for CPDP."""
    studies = folds or [fit_args]

    def score(cfg, seed):
        return float(
            np.mean([fit(cfg, **args, seed=seed).metadata["validation_pr_auc"] for args in studies])
        )

    if not search["enabled"]:
        return base, {"enabled": False, "reason": "Explicit smoke/fixed-configuration condition"}
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=search["seed"])
    )

    def objective(trial):
        cfg = suggest(trial, base)
        trial.set_user_attr("config", cfg)
        return score(cfg, search["seed"])

    study.optimize(objective, n_trials=search["trials"], n_jobs=1)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) != search["trials"]:
        raise ValueError("Equal completed-trial budget not met")
    top = sorted(completed, key=lambda t: (-t.value, t.number))[: search["top"]]
    confirmation = []
    for trial in top:
        scores = [trial.value] + [
            score(trial.user_attrs["config"], seed) for seed in search["confirmation_seeds"]
        ]
        confirmation.append(
            {
                "trial": trial.number,
                "scores": scores,
                "mean": float(np.mean(scores)),
                "config": trial.user_attrs["config"],
            }
        )
    winner = max(confirmation, key=lambda row: (row["mean"], -row["trial"]))
    return winner["config"], {
        "completed": len(completed),
        "confirmation": confirmation,
        "trials": [{"number": t.number, "value": t.value, "params": t.params} for t in completed],
    }
