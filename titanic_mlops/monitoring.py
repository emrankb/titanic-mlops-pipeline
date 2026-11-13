"""Generate Evidently drift reports from local Titanic feature batches.

Reports are saved locally; no passenger records or reports are sent to a
hosted service. Current labels are required for evaluating model accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from sklearn.metrics import accuracy_score
import pandas as pd

from .data import RAW_FEATURES, split_features_target


def generate_monitoring_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model,
    output_dir: str | Path,
    *,
    reference_accuracy: float,
    accuracy_drop_alert: float = 0.05,
) -> dict:
    if not 0 <= reference_accuracy <= 1:
        raise ValueError("Reference accuracy must be between zero and one.")
    if not 0 < accuracy_drop_alert <= 1:
        raise ValueError("Accuracy alert threshold must be positive and at most one.")
    reference_features, _ = split_features_target(reference)
    current_features, targets = split_features_target(current)
    if len(reference_features) < 20 or len(current_features) < 20:
        raise ValueError("Monitoring requires at least 20 reference and 20 current records.")
    predictions = model.predict(current_features)
    observed_accuracy = float(accuracy_score(targets, predictions))
    # Monitor input features without passenger identifiers, names, tickets, or labels.
    report = Report([DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(current_features, reference_features)
    dest = Path(output_dir)
    if dest.exists():
        raise FileExistsError("Monitoring output directory already exists.")
    dest.mkdir(parents=True)
    snapshot.save_html(str(dest / "data_drift.html"))
    (dest / "evidently_results.json").write_text(
        snapshot.json(), encoding="utf-8",
    )
    decline = reference_accuracy - observed_accuracy
    summary = {
        "reference_rows": len(reference_features),
        "current_rows": len(current_features),
        "reference_accuracy": reference_accuracy,
        "observed_accuracy": observed_accuracy,
        "accuracy_drop": float(decline),
        "accuracy_drop_alert": accuracy_drop_alert,
        "accuracy_alert": bool(decline >= accuracy_drop_alert),
        "report": "data_drift.html",
        "report_type": "evidently_data_drift_and_summary",
    }
    (dest / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary
