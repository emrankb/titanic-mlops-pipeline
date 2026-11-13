"""Train survival classifiers from a local CSV without uploading passenger rows."""
from __future__ import annotations
import argparse
from pathlib import Path
from titanic_mlops.data import load_dataset
from titanic_mlops.eda import explore
from titanic_mlops.training import train_and_evaluate, save_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/titanic.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/survival-model"))
    parser.add_argument("--eda", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    frame = load_dataset(args.csv)
    explore(frame, args.eda)
    model, summary = train_and_evaluate(frame)
    save_artifacts(model, summary, args.output)
    print("Selected model:", summary.selected_model)
    print("CV F1:", round(summary.cross_validation[summary.selected_model]["f1"], 4))
    for metric, score in summary.holdout_metrics.items():
        print(f"Holdout {metric}: {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
