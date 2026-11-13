"""Optional local MLflow export; no remote tracking server by default."""

from pathlib import Path


def export_mlflow(output, cfg, result):
    import mlflow

    output = Path(output)
    mlflow.set_tracking_uri((output / "mlruns").resolve().as_uri())
    mlflow.set_experiment("dgcl-dp")
    with mlflow.start_run(run_name=cfg["experiment"]["name"]):
        mlflow.log_params(
            {
                "primary": cfg["primary"],
                "seeds": str(cfg["seeds"]),
                "synthetic": result["synthetic"],
            }
        )
        mlflow.log_metric("completed_runs", result["runs"])
        for name in (
            "campaign-lock.json",
            "completed.json",
            "metrics.json",
            "table3.csv",
            "project-comparisons.json",
        ):
            mlflow.log_artifact(str(output / name))
