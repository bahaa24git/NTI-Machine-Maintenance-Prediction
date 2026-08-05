from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model_training import save_artifact, train_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save machine-maintenance models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("factory_sensor_simulator_2040.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("models") / "maintenance_models.joblib",
    )
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}")

    print(f"Loading {args.data} …")
    data = pd.read_csv(args.data, low_memory=False)
    print(f"Training on a reproducible sample from {len(data):,} rows …")
    artifact = train_artifact(data)
    save_artifact(artifact, args.output)
    print(f"Saved model version {artifact['version']} to {args.output}")
    for name, bundle in artifact["regression"].items():
        metrics = bundle["metrics"]
        print(f"{name}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R²={metrics['R²']:.4f}")
    classifier = artifact["failure_classifier"]["metrics"]
    print(f"Failure classifier: ROC AUC={classifier['ROC AUC']:.4f}, recall={classifier['Recall']:.4f}")


if __name__ == "__main__":
    main()
