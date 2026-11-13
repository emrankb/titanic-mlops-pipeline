"""MLflow experiment tracking and registered model versioning.

Only aggregate metrics and settings are recorded. Original passenger records,
input examples and individual predictions are never stored in tracking.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from .training import TrainingSummary, candidates

EXPERIMENT_NAME = "titanic-survival"
MODEL_NAME = "titanic-survival-classifier"
ALIAS = "champion"


def local_tracking_uri(path: str | Path) -> str:
    """Use a local SQLite database for both tracking and registry."""
    database = Path(path).expanduser().resolve()
    if database.suffix != ".db":
        raise ValueError("MLflow database path must end in .db.")
    database.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + database.as_posix()


def register_experiment(model, summary: TrainingSummary, *,
                        tracking_uri: str, experiment_name: str = EXPERIMENT_NAME,
                        registered_model_name: str = MODEL_NAME) -> dict:
    """Log three candidate runs and register only the CV-F1 winner."""
    if not tracking_uri.startswith("sqlite:///"):
        raise ValueError("Use an explicitly configured local SQLite tracking URI.")
    if not experiment_name.strip() or not registered_model_name.strip():
        raise ValueError("Experiment and registry names cannot be empty.")
    catalog = candidates()
    if set(summary.cross_validation) != set(catalog):
        raise ValueError("Evaluated candidates do not match available models.")
    winner = max(summary.cross_validation,
                 key=lambda name: summary.cross_validation[name]["f1"])
    if (summary.selection_metric != "mean_training_cv_f1"
            or winner != summary.selected_model):
        raise ValueError("Model selection must be based on training CV F1.")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run_ids = {}
    for name, estimator in catalog.items():
        with mlflow.start_run(run_name=name) as run:
            run_ids[name] = run.info.run_id
            mlflow.set_tags({
                "dataset": "local_titanic_csv",
                "validation": "stratified_cv_on_training_split",
                "selected": str(name == winner).lower(),
            })
            mlflow.log_params({
                "classifier": name,
                "seed": summary.seed,
                "cv_folds": summary.cv_folds,
                "train_rows": summary.train_rows,
                "holdout_rows": summary.holdout_rows,
            })
            mlflow.log_params({
                "estimator_" + key: str(value)
                for key, value in estimator.get_params(deep=False).items()
            })
            mlflow.log_metrics({
                "cv_" + metric: float(value)
                for metric, value in summary.cross_validation[name].items()
            })
            if name == winner:
                mlflow.log_metrics({
                    "holdout_" + metric: float(value)
                    for metric, value in summary.holdout_metrics.items()
                })
                # The pipeline contains custom feature classes. Cloudpickle
                # requires trusting the locally generated model before load.
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="survival_pipeline",
                    registered_model_name=registered_model_name,
                    serialization_format="cloudpickle",
                    await_registration_for=120,
                )

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    matching = [v for v in versions if v.run_id == run_ids[winner]]
    if len(matching) != 1:
        raise RuntimeError("Unable to resolve the registered version for the selected run.")
    version = str(matching[0].version)
    client.set_registered_model_alias(registered_model_name, ALIAS, version)
    return {
        "selected_model": winner,
        "selection_metric": "mean_training_cv_f1",
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
        "registered_version": version,
        "alias": ALIAS,
        "run_ids": run_ids,
        "selected_run_id": run_ids[winner],
        "model_uri": f"models:/{registered_model_name}/{version}",
        "alias_uri": f"models:/{registered_model_name}@{ALIAS}",
        "tracking_uri": tracking_uri,
    }
