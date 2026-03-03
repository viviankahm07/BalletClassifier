"""
run_training.py
---------------
Entry point: train all models and save the best one.

Usage:
    python run_training.py
    python run_training.py --splits data/splits --models models/saved
"""

import argparse
from train import train_all_models


def main():
    parser = argparse.ArgumentParser(
        description="Train ballet pose classifiers.")
    parser.add_argument("--splits", default="data/splits",
                        help="Path to train/val/test CSVs")
    parser.add_argument("--models", default="models/saved",
                        help="Where to save the best model")
    parser.add_argument(
        "--experiment", default="ballet_static_classifier", help="MLflow experiment name")
    args = parser.parse_args()

    results, best = train_all_models(
        splits_dir=args.splits,
        models_dir=args.models,
        experiment_name=args.experiment,
    )

    print("\nAll model results:")
    for name, metrics in results.items():
        print(
            f"  {name:<25} val_acc={metrics['val_acc']:.4f}  val_f1={metrics['val_f1']:.4f}")

    if best:
        print(f"\nBest model saved: {best[0]}")
        print("View all runs: mlflow ui")


if __name__ == "__main__":
    main()
