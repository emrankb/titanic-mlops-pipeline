# API deployment and monitoring

The service predicts Titanic survival, not hospital readmission. The supplied dataset has no hospital-readmission labels.

## Prediction API

Train a local model first with `python -m scripts.train`. Install the API dependencies and start the service:

```bash
python -m pip install -r requirements-api.txt
python -m scripts.serve_api
```

The API listens on `127.0.0.1:8000`. `GET /health` returns a readiness status. `POST /predict` accepts passenger class, sex, age, siblings/spouses, parents/children, fare, and port of embarkation. It returns a binary survival prediction and the estimated probability of survival. Required predictors cannot be omitted, apart from optional missing Age and Embarked values, which are imputed by the fitted model pipeline. Requests containing names, IDs, tickets, or outcome labels are rejected. The API does not store request bodies.

The model is loaded once when the service starts. Loading a joblib file can execute code; only a trusted locally generated file should be used. The API has no authentication and is configured for local testing, not public deployment.

## Container test

Build a Docker image without putting the source CSV or trained model in it:

```bash
docker build -t titanic-api .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$(pwd)/artifacts/survival-model:/app/artifacts/survival-model:ro" \
  titanic-api
```

The image runs as a non-root user. The fitted model is mounted read-only when the container starts. The API can then be checked at `http://127.0.0.1:8000/docs`.

## Evidently monitoring

Evidently compares a reference Titanic feature batch against a current batch. Both datasets must have the original Titanic schema, including the known `Survived` outcome. The target is used only for measuring accuracy; it is excluded from drift features. Passenger identifiers, names, tickets, and cabins are not sent to the drift report.

```bash
python -m pip install -r requirements-monitoring.txt
python -m scripts.monitor \
  --reference data/raw/titanic.csv \
  --current data/raw/recent_labeled.csv \
  --output reports/evidently
```

The command creates an interactive Evidently HTML report, Evidently's raw JSON results, and a short JSON monitoring summary. An absolute drop in accuracy of at least five percentage points raises an accuracy alert in the summary. The HTML report provides data drift and data summary metrics. Monitoring only runs when the command is executed; it is not a hosted service or a real-time data feed.

Use a genuinely new labeled batch for operational monitoring. Reusing or modifying the original dataset to create an example shows the reporting mechanism, not real-world drift or production model stability.

## CI/CD workflow

GitHub Actions runs on pull requests and pushes to `main`. It verifies preprocessing and modeling, MLflow registration, DVC reproduction and changed-data triggers, FastAPI request validation, Evidently report generation, and a Dockerized prediction request with a synthetic model. A synthetic monitoring report is available as a temporary Actions artifact.

The Docker stage checks the built image and runs an actual HTTP prediction test. It does not publish an image or deploy a live service to the internet. A production deployment would need a hosting destination, authentication, secure model storage, version promotion, and a separate approval process. No passenger data is uploaded to CI.

## Limitations

The classification scores measure Titanic survival only. The original Titanic CSV does not provide new post-deployment batches or later confirmed outcomes, so ongoing accuracy and drift monitoring cannot be claimed from this source alone. The alert threshold is a configurable classroom demonstration, not a validated operational control.
