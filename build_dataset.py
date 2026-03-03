"""
build_dataset.py
----------------
Entry point: process raw images into train/val/test CSVs.

Usage:
    python build_dataset.py
    python build_dataset.py --images data/raw_images --splits data/splits
"""

import argparse
from src.preprocessing.dataset_builder import build_dataset


def main():
    parser = argparse.ArgumentParser(description="Build ballet pose dataset from raw images.")
    parser.add_argument("--images", default="data/raw_images", help="Path to raw images directory")
    parser.add_argument("--splits", default="data/splits", help="Where to save train/val/test CSVs")
    parser.add_argument("--config", default="data_config.yaml", help="Path to data config YAML")
    args = parser.parse_args()

    build_dataset(
        raw_images_dir=args.images,
        splits_dir=args.splits,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
