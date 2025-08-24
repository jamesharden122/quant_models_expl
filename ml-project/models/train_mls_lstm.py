"""Command-line entry point to train the MLS-LSTM model."""

from __future__ import annotations

import argparse
import json

from mls_lstm_trainer import MLSLSTMTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLS-LSTM model from CSV data")
    parser.add_argument("csv", help="Path to CSV with Date and Close columns")
    args = parser.parse_args()

    trainer = MLSLSTMTrainer(args.csv)
    result = trainer.train()
    print(json.dumps({"model_dir": result.model_dir, "metrics": result.metrics}, indent=2))


if __name__ == "__main__":
    main()
