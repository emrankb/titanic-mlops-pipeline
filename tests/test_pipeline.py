"""Offline synthetic checks; no actual passenger data enters CI."""
import json
import numpy as np
import pandas as pd
import pytest
from titanic_mlops.data import RAW_FEATURES, QuantileClipper, split_features_target, validate_dataset
from titanic_mlops.training import train_and_evaluate, save_artifacts, candidates, METRICS
from titanic_mlops.eda import explore


def fictional(n=72):
    rng = np.random.default_rng(22)
    return pd.DataFrame({
        "PassengerId": np.arange(1, n + 1),
        "Survived": np.arange(n) % 2,
        "Pclass": np.tile([1, 2, 3], n // 3),
        "Name": [f"Fictional Person {i}" for i in range(n)],
        "Sex": np.tile(["female", "male"], n // 2),
        "Age": rng.integers(2, 78, size=n).astype(float),
        "SibSp": rng.integers(0, 4, size=n),
        "Parch": rng.integers(0, 3, size=n),
        "Ticket": [f"Ticket{i}" for i in range(n)],
        "Fare": rng.uniform(5, 200, size=n),
        "Cabin": [np.nan] * n,
        "Embarked": np.tile(["S", "C", "Q"], n // 3),
    })


def test_validate_and_features():
    frame = fictional()
    X, y = split_features_target(frame)
    assert len(X) == 72 and len(y) == 72
    assert set(X) == set(RAW_FEATURES)
    assert "PassengerId" not in X and "Name" not in X


def test_reject_duplicate_ids():
    d = fictional()
    d.loc[1, "PassengerId"] = 1
    with pytest.raises(ValueError, match="unique"):
        validate_dataset(d)


def test_reject_invalid_target():
    d = fictional()
    d.loc[0, "Survived"] = 3
    with pytest.raises(ValueError, match="0/1"):
        validate_dataset(d)


def test_missing_age_and_port_handled():
    d = fictional()
    d.loc[0, "Age"] = np.nan
    d.loc[1, "Embarked"] = np.nan
    trained, metrics = train_and_evaluate(d)
    X, _ = split_features_target(d)
    assert len(trained.predict(X.iloc[:2])) == 2
    assert all(0 <= n <= 1 for n in metrics.holdout_metrics.values())


def test_outlier_caps_fit_on_training_only():
    c = QuantileClipper(low=0.1, high=0.9)
    c.fit(np.array([[1.], [2.], [3.], [4.]]))
    transformed = c.transform(np.array([[999.]]))
    assert transformed[0, 0] < 10


@pytest.fixture(scope="module")
def result():
    return train_and_evaluate(fictional())


def test_select_by_cv_f1(result):
    _, summary = result
    assert set(summary.cross_validation) == set(candidates())
    assert summary.selected_model == max(summary.cross_validation, key=lambda k: summary.cross_validation[k]["f1"])
    assert summary.selection_metric == "mean_training_cv_f1"
    assert set(summary.holdout_metrics) == set(METRICS)
    assert summary.train_rows + summary.holdout_rows == 72


def test_reproducible_results(result):
    _, first = result
    _, second = train_and_evaluate(fictional())
    assert first.as_dict() == second.as_dict()


def test_save_and_aggregate_eda(result, tmp_path):
    model, summary = result
    save_artifacts(model, summary, tmp_path / "model")
    assert (tmp_path / "model" / "model.joblib").exists()
    metrics = json.loads((tmp_path / "model" / "metrics.json").read_text())
    assert "Fictional" not in json.dumps(metrics)
    assert metrics["selected_model"] == summary.selected_model
    with pytest.raises(FileExistsError):
        save_artifacts(model, summary, tmp_path / "model")
    report = explore(fictional(), tmp_path / "eda")
    assert report["rows"] == 72
    assert (tmp_path / "eda" / "survival_by_class.png").exists()
    assert "Fictional" not in (tmp_path / "eda" / "summary.json").read_text()
