# Titanic MLOps Pipeline

This project uses the Titanic dataset to predict passenger survival. The course scenario mentions hospital readmission, but the supplied dataset does not contain readmission labels.

## Workflow

The training pipeline validates input data, engineers family-related features, imputes missing values and handles outliers. Three classifiers are compared using training-fold cross-validation. The chosen model is evaluated on an untouched 20% holdout using accuracy, precision, recall, F1 and ROC-AUC.

DVC tracks data and code dependencies for reproducible training. MLflow logs model parameters and evaluation metrics, registers the selected model, and assigns it the `champion` alias.

FastAPI serves predictions from a trusted locally saved model. Docker builds the serving image without embedding the dataset or fitted model. Evidently generates local HTML and JSON reports for feature drift and basic monitoring; a separate accuracy alert checks labeled current outcomes.

## Setup and training

Use Python 3.12. Place the supplied CSV at `data/raw/titanic.csv` locally. The dataset and model artifacts are excluded from Git.

```bash
python -m pip install -r requirements-dev.txt -r requirements-mlops.txt
python -m scripts.train
```

The model is saved at `artifacts/survival-model/model.joblib` and its aggregate evaluation at `artifacts/survival-model/metrics.json`. Exploratory summaries and charts are saved under `reports/eda/`.

To register a model or rerun the DVC training stage:

```bash
python -m scripts.track_experiments
dvc repro
```

The MLflow command writes a local SQLite registry. The DVC command requires the training output directory to be absent on first execution unless the stage was previously tracked. See `docs/versioning.md`.

## API and monitoring

```bash
python -m pip install -r requirements-api.txt -r requirements-monitoring.txt
python -m scripts.serve_api
```

The API serves `GET /health` and `POST /predict` at `127.0.0.1:8000`. Documentation is at `http://127.0.0.1:8000/docs`. Prediction inputs consist of seven non-identifying Titanic features.

To run the container with the local model mounted read-only:

```bash
docker build -t titanic-api .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$(pwd)/artifacts/survival-model:/app/artifacts/survival-model:ro" \
  titanic-api
```

To generate a drift and accuracy report from a separate labeled batch:

```bash
python -m scripts.monitor \
  --reference data/raw/titanic.csv \
  --current data/raw/recent_labeled.csv
```

The monitoring command saves an interactive Evidently HTML report and JSON results under `reports/evidently/`. It needs known labels to measure accuracy. The supplied Titanic data alone does not provide real post-deployment outcomes.

## CI/CD

GitHub Actions checks modeling, MLflow model registration, DVC reproduction, FastAPI prediction, Evidently monitoring, and a running Docker container. Workflow checks use fictional passenger rows and store no original passenger records. The Docker stage performs an HTTP prediction test but does not deploy to public hosting.

```bash
python -m pip install -r requirements-ci.txt -r requirements-mlops.txt
python -m pytest -q
```

The project is a local demonstration. The prediction service has no authentication and must not be exposed as a public production endpoint. Only load trusted model artifacts: joblib and MLflow cloudpickle files can execute code during deserialization.

See `docs/data_and_model.md`, `docs/versioning.md` and `docs/api_monitoring.md` for implementation details.
