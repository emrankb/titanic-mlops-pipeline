"""Train Titanic classifiers and register the F1 winner with local MLflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from titanic_mlops.data import load_dataset
from titanic_mlops.tracking import local_tracking_uri, register_experiment
from titanic_mlops.training import save_artifacts, train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/titanic.csv"))
    parser.add_argument("--database", type=Path, default=Path("artifacts/mlflow.db"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/mlflow-selected"))
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(2, "Model output folder exists. Choose a new output path.\n")
    frame = load_dataset(args.csv)
    model, results = train_and_evaluate(frame)
    record = register_experiment(
        model, results, tracking_uri=local_tracking_uri(args.database),
    )
    save_artifacts(model, results, args.output)
    (args.output / "registry.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Selected: {results.selected_model}")
    print(f"CV F1: {results.cross_validation[results.selected_model]['f1']:.4f}")
    print(f"Holdout F1: {results.holdout_metrics['f1']:.4f}")
    print(f"Registered: {record['model_uri']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
