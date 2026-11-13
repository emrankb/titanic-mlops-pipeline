"""MLflow registry tests on invented Titanic passenger data."""
from __future__ import annotations

import json

import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pytest

from titanic_mlops.tracking import (
    ALIAS, EXPERIMENT_NAME, MODEL_NAME, local_tracking_uri, register_experiment,
)
from titanic_mlops.training import train_and_evaluate
from titanic_mlops.data import split_features_target
from tests.test_pipeline import fictional


@pytest.fixture(scope="module")
def trained():
    return train_and_evaluate(fictional())


def test_sqlite_only(tmp_path):
    uri = local_tracking_uri(tmp_path / "models.db")
    assert uri.startswith("sqlite:////")
    with pytest.raises(ValueError, match=".db"):
        local_tracking_uri(tmp_path / "models.txt")


def test_real_tracking_and_registry_roundtrip(trained, tmp_path):
    model, scores = trained
    uri = local_tracking_uri(tmp_path / "registry.db")
    recorded = register_experiment(model, scores, tracking_uri=uri)
    assert recorded["selected_model"] == scores.selected_model
    assert recorded["registered_version"] == "1"
    assert len(recorded["run_ids"]) == 3
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    assert client.get_experiment_by_name(EXPERIMENT_NAME) is not None

    for name, run_id in recorded["run_ids"].items():
        run = client.get_run(run_id)
        assert run.data.params["classifier"] == name
        assert run.data.metrics["cv_f1"] == pytest.approx(
            scores.cross_validation[name]["f1"]
        )
        assert run.data.tags["selected"] == str(name == scores.selected_model).lower()
        if name == scores.selected_model:
            assert run.data.metrics["holdout_roc_auc"] == pytest.approx(
                scores.holdout_metrics["roc_auc"]
            )
        else:
            assert "holdout_f1" not in run.data.metrics
        logged = json.dumps({"params": run.data.params, "tags": run.data.tags})
        assert "Fictional Person" not in logged
        assert "Ticket0" not in logged

    selected = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    assert str(selected.version) == "1"
    loaded = mlflow.sklearn.load_model(recorded["model_uri"])
    X, _ = split_features_target(fictional())
    assert list(model.predict(X.iloc[:5])) == list(loaded.predict(X.iloc[:5]))

    newer = register_experiment(model, scores, tracking_uri=uri)
    assert newer["registered_version"] == "2"
    assert str(client.get_model_version_by_alias(MODEL_NAME, ALIAS).version) == "2"


def test_refuse_wrong_selected_candidate(trained, tmp_path):
    model, scores = trained
    with pytest.raises(ValueError, match="SQLite"):
        register_experiment(model, scores, tracking_uri="https://example.org")
    scores.selection_metric = "holdout_f1"
    with pytest.raises(ValueError, match="training CV F1"):
        register_experiment(
            model, scores, tracking_uri=local_tracking_uri(tmp_path / "test.db"),
        )
