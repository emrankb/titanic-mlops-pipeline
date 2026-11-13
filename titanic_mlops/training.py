"""Reproducible model comparison; the holdout never determines model selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .data import TitanicFeatureBuilder, make_preprocessor, split_features_target, RAW_FEATURES

SEED = 42
CV_FOLDS = 3
TEST_SIZE = 0.2
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def candidates() -> dict:
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, random_state=SEED, n_jobs=1
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("feature_engineering", TitanicFeatureBuilder()),
        ("preprocessing", make_preprocessor()),
        ("classifier", estimator),
    ])


@dataclass
class TrainingSummary:
    selected_model: str
    selection_metric: str
    cross_validation: dict
    holdout_metrics: dict
    train_rows: int
    holdout_rows: int
    seed: int
    cv_folds: int

    def as_dict(self) -> dict:
        return asdict(self)


def train_and_evaluate(frame):
    X, y = split_features_target(frame)
    if y.value_counts().min() < 8:
        raise ValueError("Each class needs at least eight examples.")
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv = {}
    for name, clf in candidates().items():
        scores = cross_validate(
            build_pipeline(clf), Xtrain, ytrain, cv=splitter,
            scoring={key: key for key in METRICS}, error_score="raise", n_jobs=1,
        )
        cv[name] = {key: float(np.mean(scores["test_" + key])) for key in METRICS}
    selected = max(cv, key=lambda name: cv[name]["f1"])
    fitted = build_pipeline(candidates()[selected])
    fitted.fit(Xtrain, ytrain)
    predictions = fitted.predict(Xtest)
    positive = fitted.predict_proba(Xtest)[:, list(fitted.classes_).index(1)]
    metrics = {
        "accuracy": float(accuracy_score(ytest, predictions)),
        "precision": float(precision_score(ytest, predictions, zero_division=0)),
        "recall": float(recall_score(ytest, predictions, zero_division=0)),
        "f1": float(f1_score(ytest, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(ytest, positive)),
    }
    return fitted, TrainingSummary(
        selected_model=selected,
        selection_metric="mean_training_cv_f1",
        cross_validation=cv,
        holdout_metrics=metrics,
        train_rows=len(Xtrain),
        holdout_rows=len(Xtest),
        seed=SEED, cv_folds=CV_FOLDS,
    )


def save_artifacts(model, summary: TrainingSummary, directory: str | Path) -> None:
    """Persist only trusted local artifacts. Never load untrusted joblib files."""
    target = Path(directory)
    if target.exists():
        raise FileExistsError("Output model directory already exists.")
    target.mkdir(parents=True)
    joblib.dump(model, target / "model.joblib")
    (target / "metrics.json").write_text(
        json.dumps({"features": list(RAW_FEATURES), **summary.as_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
