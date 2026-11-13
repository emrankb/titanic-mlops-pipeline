# Titanic MLOps Project Report

## Objective

The goal of this project is to build a reproducible machine learning pipeline that trains, tracks, serves, and monitors a classification model. The assignment describes predicting hospital readmission, but the supplied Titanic dataset has a `Survived` label rather than a hospital readmission outcome. For that reason, the model predicts Titanic passenger survival. It should not be used to make healthcare decisions.

## Dataset and exploration

The supplied `titanic.csv` contains 891 passengers and 12 columns. Of those passengers, 342 survived (38.4%). The missing fields are `Age` (177), `Cabin` (687), and `Embarked` (2).

The initial exploration showed different survival rates by passenger class: 63.0% in first class (216 passengers), 47.3% in second class (184 passengers), and 24.2% in third class (491 passengers). The survival rate was 74.2% among female passengers and 18.9% among male passengers. These are descriptive patterns, not causal conclusions.

The pipeline uses passenger class, sex, age, sibling/spouse count, parent/child count, fare, and embarkation port. It adds family-related features, handles missing values, limits outliers, and encodes categorical variables. Names, passenger IDs, ticket numbers, and cabin values are not model inputs.

Running `python -m scripts.train` generates an exploration summary and a survival-by-class chart under `reports/eda/`.

## Modeling and evaluation

The pipeline compares a majority-class baseline, logistic regression, and random forest. Training uses a stratified 80/20 split with a fixed seed. It selects the model using mean F1 score from three-fold cross-validation on the training portion only. The selected model is then evaluated once on the holdout portion.

The reported holdout measures are accuracy, precision, recall, F1, and ROC-AUC. The holdout results are saved to `artifacts/survival-model/metrics.json` when training finishes.

The complete preprocessing and classifier steps are stored together in a scikit-learn pipeline, so the same transformations are used during training and prediction.

## Architecture and tools

The workflow is:

1. Local Titanic CSV -> schema checks and exploration
2. Feature engineering and preprocessing -> model comparison -> selected classifier
3. Saved model and holdout metrics -> MLflow experiment tracking and model registry
4. Data and training dependencies -> DVC reproduction
5. Saved model -> FastAPI prediction endpoint -> Docker container
6. Reference and current labeled batches -> Evidently drift report and accuracy alert
7. GitHub pull requests and pushes -> automated tests, DVC checks, API checks, and container smoke test

Python and scikit-learn handle the model. MLflow records parameters, metrics, and registered models. DVC records data and code dependencies. FastAPI provides the REST interface. Docker packages the API service. Evidently produces data summary and drift reports. GitHub Actions automates verification.

## Setup and outputs

Use Python 3.12 and place the supplied CSV at `data/raw/titanic.csv`. From the repository root:

```bash
python -m pip install -r requirements-dev.txt -r requirements-mlops.txt
python -m scripts.train
python -m scripts.track_experiments
dvc repro
```

Training creates a model at `artifacts/survival-model/model.joblib`, metrics at `artifacts/survival-model/metrics.json`, and exploration outputs under `reports/eda/`. Model and data files remain local rather than being committed to Git.

For the prediction API:

```bash
python -m pip install -r requirements-api.txt
python -m scripts.serve_api
```

The locally served API exposes `GET /health`, `POST /predict`, and interactive documentation at `http://127.0.0.1:8000/docs`. For example, `POST /predict` accepts:

```json
{
  "Pclass": 2,
  "Sex": "female",
  "Age": 29,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 22.5,
  "Embarked": "S"
}
```

The response returns a survival classification and its estimated probability; the exact prediction depends on the trained model. The Docker workflow and its local model mount are documented in the README.

## Monitoring and alerts

The monitoring command compares the original feature distribution against a separate labeled current batch:

```bash
python -m pip install -r requirements-monitoring.txt
python -m scripts.monitor --reference data/raw/titanic.csv --current data/raw/recent_labeled.csv
```

It saves `data_drift.html`, `evidently_results.json`, and `monitoring_summary.json` under `reports/evidently/`. The HTML file is an interactive report of drift and descriptive feature statistics. A decrease in accuracy of at least five percentage points from the reference holdout accuracy raises the configured alert.

The supplied CSV is a single historical dataset. It does not contain a later production batch. This means there is no basis to claim genuine post-deployment drift, continuous live monitoring, or real-world accuracy changes. The GitHub Actions workflow generates an example Evidently report using synthetic data for checking the reporting code, and uploads it as a temporary workflow artifact.

## Automated testing and deployment checks

The GitHub Actions workflow runs on pull requests and updates to `main`. It checks data processing, model training, MLflow registration, DVC reproduction and change detection, API validation, monitoring report generation, and an HTTP prediction request against a running Docker container. Tests use fictional passenger data.

The checks demonstrate local container deployment and automated validation. They do **not** deploy a publicly hosted service. The API binds to a local address and has no authentication, so it should not be exposed directly to the internet.

## Evidence and example outputs

- `reports/eda/summary.json`: dataset counts, missing values, and grouped survival summaries, generated by the training script.
- `reports/eda/survival_by_class.png`: chart showing survival rate by passenger class.
- `artifacts/survival-model/metrics.json`: model selection information and holdout metrics.
- `http://127.0.0.1:8000/docs`: interactive API input and output documentation when the service is running.
- `reports/evidently/data_drift.html`: interactive drift report generated when a current labeled batch is supplied.
- GitHub Actions run: https://github.com/emrankb/titanic-mlops-pipeline/actions/runs/43365324096 (successful checks for PR #3).

The output files are created by running the pipeline locally. Screenshots of the charts, API responses, and monitoring report can be taken after running the corresponding commands.

## Lessons learned and possible improvements

Keeping preprocessing in the same pipeline as the model reduces differences between training and serving. Selecting models through cross-validation while keeping a holdout separate gives a clearer evaluation. Tracking data dependencies and model metadata makes it easier to reproduce a result. Testing the API inside Docker catches deployment issues that ordinary unit tests may miss.

For a further version, the project could add a secure hosting environment, authenticated API access, data validation for newly arriving batches, a scheduling service for periodic monitoring, and a review step before replacing a deployed model. Proper healthcare readmission modeling would require a separate patient dataset with an appropriate readmission label, a privacy review, and domain-specific validation.
