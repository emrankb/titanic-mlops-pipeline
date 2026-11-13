# Titanic MLOps Pipeline

This project uses the supplied Titanic dataset to predict passenger survival.
The course description discusses hospital readmission, but the supplied data
has survival labels, not hospital readmission outcomes.

## Current implementation

The first milestone validates the dataset, builds features, trains and
compares three classifiers, and saves an aggregate evaluation. MLflow,
DVC, FastAPI, Evidently reports, and the release pipeline will be added
in the following milestones.

## Setup

Use Python 3.12. Put the supplied CSV in the local folder
`data/raw/titanic.csv`; this file is excluded from Git.

```bash
python -m pip install -r requirements-dev.txt
python -m scripts.train
python -m pytest -q
```

The training command saves the fitted pipeline to
`artifacts/survival-model/model.joblib`, aggregate metrics to
`artifacts/survival-model/metrics.json`, and exploratory reports to
`reports/eda/`. These outputs are ignored by Git.

Preprocessing and feature engineering are included inside every training
fold. The model is chosen using cross-validated F1, then evaluated on a
separate 20% holdout. Metrics are accuracy, precision, recall, F1 and
ROC-AUC. The GitHub Actions tests use fictional rows, not the actual dataset.

See `docs/data_and_model.md` for the data and model evaluation method.
