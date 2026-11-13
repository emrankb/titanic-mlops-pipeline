"""Produce group-level Titanic dataset summaries without passenger records."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import validate_dataset


def explore(frame, output_dir: str | Path):
    data = validate_dataset(frame)
    report = {
        "rows": len(data),
        "survivors": int(data["Survived"].sum()),
        "survival_rate": float(data["Survived"].mean()),
        "missing_values": {name: int(value) for name, value in data.isna().sum().items()},
        "groups": {},
    }
    for column in ("Pclass", "Sex", "Embarked"):
        grouped = data.groupby(column, dropna=False)["Survived"].agg(["count", "mean"])
        report["groups"][column] = [
            {"value": str(index), "passengers": int(row["count"]),
             "survival_rate": float(row["mean"])}
            for index, row in grouped.iterrows()
        ]
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    groups = report["groups"]["Pclass"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([item["value"] for item in groups], [100 * item["survival_rate"] for item in groups])
    ax.set(xlabel="Passenger class", ylabel="Survival rate (%)", title="Titanic survival by class")
    fig.tight_layout()
    fig.savefig(dest / "survival_by_class.png", dpi=140)
    plt.close(fig)
    return report
