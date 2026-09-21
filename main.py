#!/usr/bin/env python3
"""Machine Health Degradation Monitoring via Mean-Shift Clustering.

Terminal CLI entry point supporting:
- python main.py generate-data [--samples N] [--output PATH]
- python main.py train [--config PATH] [--data PATH]
- python main.py analyze [--config PATH]
- python main.py predict --file PATH [--config PATH]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.pipeline import analyze_pipeline, predict_pipeline, train_pipeline
from src.synthetic_data import (
    generate_sample_monitoring_chunks,
    generate_synthetic_vibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Industrial Machine Health Degradation Monitoring using Mean-Shift Clustering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Generate realistic synthetic multi-sensor vibration dataset
  python main.py generate-data --samples 6000 --output data/synthetic_vibration.csv

  # 2. Train unsupervised Mean-Shift model and calibrate degradation baseline
  python main.py train --config config.yaml

  # 3. Perform offline degradation analysis and generate report visualizations
  python main.py analyze --config config.yaml

  # 4. Perform real-time health prediction on new vibration measurements
  python main.py predict --file data/samples/test_warning.csv
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: generate-data
    p_gen = subparsers.add_parser("generate-data", help="Generate synthetic multi-sensor vibration dataset for testing.")
    p_gen.add_argument("--samples", type=int, default=6000, help="Number of vibration samples (default: 6000)")
    p_gen.add_argument("--output", type=str, default="data/synthetic_vibration.csv", help="Output CSV path")
    p_gen.add_argument("--create-test-samples", action="store_true", default=True, help="Create test normal/warning/critical snippet CSVs in data/samples")

    # Command: train
    p_train = subparsers.add_parser("train", help="Train Mean-Shift clustering and calibrate health baseline.")
    p_train.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: config.yaml)")
    p_train.add_argument("--data", type=str, default=None, help="Optional override for training CSV path")

    # Command: analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze health degradation progression and generate visual plots.")
    p_analyze.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: config.yaml)")

    # Command: predict
    p_predict = subparsers.add_parser("predict", help="Predict machine health status and degradation score on new vibration data.")
    p_predict.add_argument("--file", type=str, required=True, help="Path to new multi-sensor vibration CSV file")
    p_predict.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml (default: config.yaml)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "generate-data":
            df = generate_synthetic_vibration(num_samples=args.samples, output_path=args.output)
            if args.create_test_samples:
                generate_sample_monitoring_chunks(df, output_dir="data/samples")

        elif args.command == "train":
            train_pipeline(config_path=args.config, data_path=args.data)

        elif args.command == "analyze":
            analyze_pipeline(config_path=args.config)

        elif args.command == "predict":
            predict_pipeline(file_path=args.file, config_path=args.config)

    except Exception as e:
        print(f"\n[Error] Command '{args.command}' failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
