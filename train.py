"""
train.py
--------
Training loop that:
1. Loads train/val splits from CSVs
2. Trains all configured models
3. Evaluates on validation set
4. Logs everything to MLflow
5. Saves the best model
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

from src.models.classifier import ALL_MODELS
from src.models.grouped_classifier import GroupedClassifier
from src.utils.feature_groups import FEATURE_GROUPS


def load_splits(splits_dir: str = "data/splits"):
    """Load pre-built train/val/test CSVs."""
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    label_classes = np.load(os.path.join(splits_dir, "label_classes.npy"), allow_pickle=True)

    feature_names = list(train_df.drop("label", axis=1).columns)

    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values
    X_val = val_df.drop("label", axis=1).values
    y_val = val_df["label"].values
    X_test = test_df.drop("label", axis=1).values
    y_test = test_df["label"].values

    return X_train, X_val, X_test, y_train, y_val, y_test, label_classes, feature_names


def train_all_models(
    splits_dir: str = "data/splits",
    models_dir: str = "models/saved",
    experiment_name: str = "ballet_static_classifier"
):
    """Train all models, log to MLflow, save the best one."""

    X_train, X_val, X_test, y_train, y_val, y_test, label_classes, feature_names = load_splits(splits_dir)
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    mlflow.set_experiment(experiment_name)

    best_model = None
    best_val_acc = 0.0
    results = {}

    for model_name, ModelClass in ALL_MODELS.items():
        print(f"\n--- Training {model_name} ---")

        with mlflow.start_run(run_name=model_name):
            try:
                model = ModelClass()
                model.train(X_train, y_train)

                # Validation metrics
                y_val_pred = model.predict(X_val)
                val_acc = accuracy_score(y_val, y_val_pred)
                val_f1 = f1_score(y_val, y_val_pred, average="weighted")

                # MLflow logging
                mlflow.log_param("model_type", model_name)
                mlflow.log_metric("val_accuracy", val_acc)
                mlflow.log_metric("val_f1_weighted", val_f1)
                mlflow.log_metric("train_size", len(X_train))

                print(f"  Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")
                print(classification_report(y_val, y_val_pred,
                                            target_names=label_classes))

                results[model_name] = {"val_acc": val_acc, "val_f1": val_f1}

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = (model_name, model)

            except Exception as e:
                print(f"  SKIPPED — {e}")

    # Train GroupedClassifier (feature-group-aware model)
    print(f"\n--- Training grouped_classifier ---")
    with mlflow.start_run(run_name="grouped_classifier"):
        try:
            model = GroupedClassifier(
                feature_groups=FEATURE_GROUPS,
                feature_names=feature_names,
            )
            model.train(X_train, y_train)

            y_val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average="weighted")

            mlflow.log_param("model_type", "grouped_classifier")
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_f1_weighted", val_f1)
            mlflow.log_metric("train_size", len(X_train))

            print(f"  Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            print(classification_report(y_val, y_val_pred, target_names=label_classes))

            results["grouped_classifier"] = {"val_acc": val_acc, "val_f1": val_f1}

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = ("grouped_classifier", model)

        except Exception as e:
            print(f"  SKIPPED — {e}")

    # Save best model
    if best_model:
        name, model = best_model
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model.save(os.path.join(models_dir, f"best_model_{name}.pkl"))
        print(f"\nBest model: {name} (val accuracy: {best_val_acc:.4f})")

    return results, best_model
