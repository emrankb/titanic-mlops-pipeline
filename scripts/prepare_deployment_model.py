"""Build the Docker deployment model from a fixed Titanic dataset revision."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from titanic_mlops.data import validate_dataset
from titanic_mlops.training import save_artifacts, train_and_evaluate

DATASET_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/"
    "f0ccab6a7ceafdff780052166fb6fab3311398eb/titanic.csv"
)


def main() -> None:
    with urlopen(DATASET_URL, timeout=30) as response:
        data = response.read()
    frame = validate_dataset(pd.read_csv(BytesIO(data)))
    if len(frame) != 891 or int(frame["Survived"].sum()) != 342:
        raise ValueError("Unexpected training data.")
    model, summary = train_and_evaluate(frame)
    save_artifacts(model, summary, Path("artifacts/survival-model"))
    print("Deployment model:", summary.selected_model)
    print("Holdout F1:", round(summary.holdout_metrics["f1"], 4))


if __name__ == "__main__":
    main()
