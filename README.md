# Titanic MLOps Pipeline

This project trains a passenger survival classifier from the supplied Titanic
dataset. The coursework scenario mentions hospital readmission, but that
outcome is not in this dataset.

## Current implementation

Data validation, feature engineering, and model comparison use a stratified
training/holdout split. Three classifiers are compared by training-fold F1.
The winner is evaluated using accuracy, precision, recall, F1, and ROC-AUC.

MLflow stores candidate parameters and aggregate metrics in a local SQLite
database. It registers the selected model with a numbered version and a
`champion` alias. DVC defines the training stage and tracks changes to its
inputs and cached outputs. FastAPI, CI/CD deployment and Evidently reports
are still to be added.

## Setup and training

Use Python 3.12 and place the supplied `titanic.csv` locally at
`data/raw/titanic.csv`. The dataset is not committed to Git.

```bash
python -m pip install -r requirements-dev.txt -r requirements-mlops.txt
python -m scripts.train
```

This creates `artifacts/survival-model/model.joblib`,
`artifacts/survival-model/metrics.json`, and EDA reports in
`reports/eda/`.

To track experiments in MLflow and register the selected model:

```bash
python -m scripts.track_experiments
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

To run the DVC pipeline instead of the standalone training command,
start with no existing model output directory, then run:

```bash
dvc repro
dvc status
```

The DVC training stage writes the same local outputs and generates
`dvc.lock` to record input and output fingerprints. If a dependency
changes, `dvc repro` reruns the stage. The repository has no external
DVC remote configured.

## Tests

```bash
python -m pytest -q
```

The GitHub Actions jobs test the classifier pipeline, MLflow registry
versioning, and actual DVC reproduction with synthetic Titanic records.
No original passenger rows or generated model artifacts are committed.
A locally generated model must be treated as trusted input when loaded:
joblib and MLflow cloudpickle artifacts can execute code on deserialization.

See `docs/data_and_model.md` and `docs/versioning.md` for details.
