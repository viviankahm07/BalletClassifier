"""
dataset_builder.py
------------------
Runs the full preprocessing pipeline:
1. Walk raw_images/ directories
2. Extract pose keypoints from each image
3. Convert keypoints to joint angle features
4. Build train/val/test CSV splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.extraction.pose_extractor import PoseExtractor
from src.preprocessing.normalizer import extract_all_features, ALL_FEATURE_NAMES
from src.utils.config import load_data_config


def build_dataset(
    raw_images_dir: str = "data/raw_images",
    splits_dir: str = "data/splits",
    config_path: str = "data_config.yaml",
    min_visibility: float = 0.5,
) -> dict:
    """
    Full pipeline: images → keypoints → angles → train/val/test CSVs.

    Returns a summary dict with counts per class and split.
    """
    cfg = load_data_config(config_path)
    classes = cfg["classes"]
    split_cfg = cfg["splits"]
    valid_exts = set(cfg["image"]["valid_extensions"])
    min_images = cfg["image"]["min_images_per_class"]

    extractor = PoseExtractor()
    records = []
    skipped = 0

    print(f"Processing {len(classes)} classes from {raw_images_dir}/\n")

    for class_name in classes:
        class_dir = Path(raw_images_dir) / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} does not exist — skipping")
            continue

        image_paths = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in valid_exts
        ]

        if len(image_paths) < min_images:
            print(f"  WARNING: {class_name} has only {len(image_paths)} images (minimum recommended: {min_images})")

        print(f"  {class_name}: {len(image_paths)} images")

        for img_path in tqdm(image_paths, desc=f"    extracting", leave=False):
            result = extractor.extract(str(img_path))

            if not result.success:
                skipped += 1
                continue

            if result.visibility.mean() < min_visibility:
                skipped += 1
                continue

            angles = extract_all_features(result.keypoints)
            record = dict(zip(ALL_FEATURE_NAMES, angles))
            record["label"] = class_name
            records.append(record)

    print(f"\nExtracted {len(records)} valid samples ({skipped} skipped — pose not detected or too low confidence)\n")

    if not records:
        raise ValueError("No valid samples found. Check that images exist and bodies are clearly visible.")

    df = pd.DataFrame(records)
    label_classes = np.array(sorted(df["label"].unique()))

    # Use stratification only when there are enough samples per class
    min_per_class = df["label"].value_counts().min()
    use_stratify = min_per_class >= 3

    if not use_stratify:
        print("  NOTE: Too few images per class for stratified splits — splitting randomly instead.")

    train_val_df, test_df = train_test_split(
        df,
        test_size=split_cfg["test"],
        stratify=df["label"] if use_stratify else None,
        random_state=split_cfg["random_seed"],
    )

    val_size_adjusted = split_cfg["val"] / (split_cfg["train"] + split_cfg["val"])
    min_trainval_per_class = train_val_df["label"].value_counts().min()
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df["label"] if min_trainval_per_class >= 3 else None,
        random_state=split_cfg["random_seed"],
    )

    # Save splits
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(f"{splits_dir}/train.csv", index=False)
    val_df.to_csv(f"{splits_dir}/val.csv", index=False)
    test_df.to_csv(f"{splits_dir}/test.csv", index=False)
    np.save(f"{splits_dir}/label_classes.npy", label_classes)

    summary = {
        "total": len(df),
        "skipped": skipped,
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "classes": list(label_classes),
        "per_class": df["label"].value_counts().to_dict(),
    }

    print("Split summary:")
    print(f"  Train : {len(train_df)}")
    print(f"  Val   : {len(val_df)}")
    print(f"  Test  : {len(test_df)}")
    print(f"\nSaved to {splits_dir}/")

    return summary
