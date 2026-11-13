"""Compare labeled Titanic batches with Evidently and save an HTML report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from titanic_mlops.api import load_trusted_model
from titanic_mlops.data import load_dataset
from titanic_mlops.monitoring import generate_monitoring_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/survival-model/model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/survival-model/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/evidently"))
    parser.add_argument("--accuracy-drop", type=float, default=0.05)
    args = parser.parse_args()
    model = load_trusted_model(args.model)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = generate_monitoring_report(
        load_dataset(args.reference),
        load_dataset(args.current),
        model,
        args.output,
        reference_accuracy=float(metrics["holdout_metrics"]["accuracy"]),
        accuracy_drop_alert=args.accuracy_drop,
    )
    print(f"Observed accuracy: {result['observed_accuracy']:.4f}")
    print(f"Accuracy alert: {result['accuracy_alert']}")
    print(f"Reports: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
