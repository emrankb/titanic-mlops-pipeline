# Model and dataset versioning

MLflow records parameters and metrics for the majority baseline, logistic
regression, and random forest. The model chosen by mean cross-validated F1
is registered as `titanic-survival-classifier`, assigned a numbered version,
and given the `champion` alias. Held-out metrics are logged only for the
winner.

Run this locally after installing the MLOps dependencies and placing
`titanic.csv` under `data/raw/`:

```bash
python -m pip install -r requirements-dev.txt -r requirements-mlops.txt
python -m scripts.track_experiments
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

MLflow uses a local SQLite registry. Both the registry database and the
saved model stay under ignored `artifacts/` or `mlruns/`. No passenger
records, input examples, or individual predictions are logged. The fitted
pipeline includes custom Python transformers, so MLflow uses cloudpickle.
Only load models generated in a trusted local environment: cloudpickle
deserialization can execute arbitrary code.

## DVC pipeline

The versioned `dvc.yaml` defines a training stage, including the CSV,
implementation files, dependencies, and generated model and EDA reports.
It tracks changed data and code through dependency checksums in
`dvc.lock` and caches the pipeline outputs locally.

```bash
dvc repro
dvc status
```

On a first run, DVC creates `dvc.lock` and locally cached artifacts.
After changing the CSV or code, `dvc repro` identifies changed dependencies
and reruns training. Do not commit the source CSV, the cache, or generated
model files. `dvc.lock` contains file hashes rather than original rows;
it may be versioned in Git after a trusted local run when the coursework
requires reproducibility. No private data or generated `dvc.lock` from
synthetic CI cases is included in the repository.

No DVC remote is configured. The local DVC cache is not an externally
accessible artifact store, and data sharing requires a separate approved
storage configuration. CI verifies DVC by generating an invented Titanic
dataset inside the runner, executing the stage, and checking the lock file.
The resulting synthetic lock and model stay within the CI runner.
