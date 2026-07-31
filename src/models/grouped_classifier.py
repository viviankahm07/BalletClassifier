"""
grouped_classifier.py
---------------------
GroupedClassifier: trains separate sub-classifiers per feature group.

Each group defines which joint angles matter for its set of positions.
This prevents arm angles from adding noise when classifying leg-focused positions
like tendu, degage, and arabesque.

Example:
    leg_focused group  → trained on leg angles only, for tendu/arabesque/etc.
    full_body group    → trained on all directional angles, for 1st/2nd/5th position
"""

import numpy as np
from src.models.classifier import BalletClassifierBase, RandomForestModel


class GroupedClassifier(BalletClassifierBase):
    """
    Trains one sub-classifier per feature group, each using only the relevant
    joint angle features for its class group.

    At prediction time, probabilities from all sub-classifiers are assembled
    into a single probability vector over all classes.
    """

    def __init__(self, feature_groups: dict, feature_names: list,
                 base_model_class=RandomForestModel):
        """
        Args:
            feature_groups:   dict mapping group_name → {features: [...], classes: [...]}
            feature_names:    ordered list of feature names matching training columns
            base_model_class: which BalletClassifierBase subclass to use per group
        """
        self.feature_groups = feature_groups
        self.feature_names = feature_names
        self.base_model_class = base_model_class
        self.classifiers = {}
        self.all_classes = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.all_classes = sorted(np.unique(y_train))
        self.class_to_idx = {c: i for i, c in enumerate(self.all_classes)}

        # Warn about any classes not covered by any group
        all_grouped = {c for g in self.feature_groups.values() for c in g["classes"]}
        uncovered = [c for c in self.all_classes if c not in all_grouped]
        if uncovered:
            print(f"  WARNING: These classes are not in any feature group and will be ignored: {uncovered}")

        for group_name, group_cfg in self.feature_groups.items():
            group_features = [f for f in group_cfg["features"] if f in self.feature_names]
            group_classes = [c for c in group_cfg["classes"] if c in self.all_classes]

            if not group_classes:
                print(f"  Skipping group '{group_name}' — no training data for its classes")
                continue

            mask = np.isin(y_train, group_classes)
            if mask.sum() < 2:
                print(f"  Skipping group '{group_name}' — too few samples")
                continue

            feat_indices = [self.feature_names.index(f) for f in group_features]
            X_group = X_train[mask][:, feat_indices]
            y_group = y_train[mask]

            clf = self.base_model_class()
            clf.train(X_group, y_group)

            self.classifiers[group_name] = {
                "clf": clf,
                "feat_indices": feat_indices,
            }

            print(f"  Group '{group_name}': trained on {len(group_features)} features, "
                  f"{len(group_classes)} classes, {mask.sum()} samples")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        full_proba = np.zeros((n_samples, len(self.all_classes)))

        for group_name, group_info in self.classifiers.items():
            clf = group_info["clf"]
            feat_indices = group_info["feat_indices"]

            X_group = X[:, feat_indices]
            group_proba = clf.predict_proba(X_group)  # (n_samples, n_group_classes)

            # Map sub-classifier class order → global class indices
            for local_idx, cls in enumerate(clf.model.classes_):
                if cls in self.class_to_idx:
                    global_idx = self.class_to_idx[cls]
                    full_proba[:, global_idx] = group_proba[:, local_idx]

        # Normalise rows to sum to 1
        row_sums = full_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return full_proba / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array([self.all_classes[i] for i in proba.argmax(axis=1)])
