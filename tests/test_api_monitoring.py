"""API and Evidently integration checks using invented passenger records."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_pipeline import fictional
from titanic_mlops.api import create_app
from titanic_mlops.monitoring import generate_monitoring_report
from titanic_mlops.training import train_and_evaluate


@pytest.fixture(scope="module")
def trained():
    return train_and_evaluate(fictional())


def example_payload():
    return {
        "Pclass": 2, "Sex": "female", "Age": 29.0,
        "SibSp": 1, "Parch": 0, "Fare": 22.5, "Embarked": "S",
    }


def test_health_and_probability(trained):
    app = TestClient(create_app(model=trained[0]))
    assert app.get("/health").json() == {"status": "ready", "model_loaded": True}
    response = app.post("/predict", json=example_payload())
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"survived", "survival_probability"}
    assert result["survived"] in (0, 1)
    assert 0 <= result["survival_probability"] <= 1


@pytest.mark.parametrize("extra", [
    {"PassengerId": 9}, {"Survived": 1}, {"Name": "private"},
    {"Ticket": "ticket"}, {"Age": -1}, {"Pclass": 4},
    {"Fare": -1}, {"Sex": "unknown"}, {"SibSp": -1},
])
def test_invalid_request_rejected(trained, extra):
    app = TestClient(create_app(model=trained[0]))
    assert app.post("/predict", json={**example_payload(), **extra}).status_code == 422


def test_missing_feature_rejected(trained):
    request = example_payload()
    del request["Pclass"]
    assert TestClient(create_app(model=trained[0])).post("/predict", json=request).status_code == 422


def test_saved_trusted_model_serves_predictions(tmp_path, trained):
    path = tmp_path / "trusted.joblib"
    joblib.dump(trained[0], path)
    assert TestClient(create_app(model_path=path)).post(
        "/predict", json=example_payload()
    ).status_code == 200


def test_missing_model_not_silently_ignored(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        create_app(model_path=tmp_path / "missing.joblib")


def test_evidently_report_artifacts_and_alert(tmp_path, trained):
    reference = fictional(72)
    current = reference.copy()
    current["Age"] = current["Age"] + 24
    current["Fare"] = current["Fare"] * 3
    current["Survived"] = 1 - current["Survived"]
    output = tmp_path / "evidently"
    report = generate_monitoring_report(
        reference, current, trained[0], output,
        reference_accuracy=1.0,
        accuracy_drop_alert=0.05,
    )
    assert report["reference_rows"] == 72
    assert report["current_rows"] == 72
    assert report["accuracy_alert"]
    assert (output / "data_drift.html").is_file()
    assert (output / "data_drift.html").stat().st_size > 1000
    assert (output / "evidently_results.json").is_file()
    assert (output / "evidently_results.json").stat().st_size > 100
    summary = json.loads((output / "monitoring_summary.json").read_text())
    assert summary["accuracy_alert"]
    assert "Fictional Person" not in json.dumps(summary)
    assert "Ticket0" not in json.dumps(summary)
    with pytest.raises(FileExistsError):
        generate_monitoring_report(reference, current, trained[0], output, reference_accuracy=0.8)


def test_monitoring_fails_on_insufficient_observations(tmp_path, trained):
    frame = fictional()
    with pytest.raises(ValueError, match="20"):
        generate_monitoring_report(
            frame, frame.iloc[:8], trained[0], tmp_path / "report",
            reference_accuracy=0.8,
        )


def test_monitoring_checks_accuracy_threshold(trained, tmp_path):
    frame = fictional()
    with pytest.raises(ValueError, match="accuracy"):
        generate_monitoring_report(
            frame, frame, trained[0], tmp_path / "report",
            reference_accuracy=1.5,
        )
