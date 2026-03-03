"""
classifier.py
-------------
Wrapper classes for all classifiers used in this project.
Each classifier exposes a consistent train() / predict() / predict_proba() interface.

Classifier progression (in order of complexity):
1. RandomForestClassifier  — fast, interpretable, good baseline
2. SVMClassifier           — strong on small datasets with normalized features
3. GradientBoostingClassifier — often best classical ML performance
4. MLPClassifier           — neural network, use when dataset is larger (500+ per class)
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import Optional


class BalletClassifierBase:
    """Shared interface for all classifiers."""

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


class RandomForestModel(BalletClassifierBase):

    def __init__(self, n_estimators: int = 200, max_depth=None, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    def feature_importances(self):
        return self.model.feature_importances_


class SVMModel(BalletClassifierBase):

    def __init__(self, C: float = 1.0, kernel: str = "rbf", random_state: int = 42):
        self.model = SVC(C=C, kernel=kernel, probability=True, random_state=random_state)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


class GradientBoostingModel(BalletClassifierBase):

    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.1,
                 max_depth: int = 4, random_state: int = 42):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


class MLPModel(BalletClassifierBase):
    """Scikit-learn MLP — sufficient for most cases. Swap for Keras if needed."""

    def __init__(self, hidden_layer_sizes=(128, 64, 32), max_iter: int = 300,
                 random_state: int = 42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))


ALL_MODELS = {
    "random_forest": RandomForestModel,
    "svm": SVMModel,
    "gradient_boosting": GradientBoostingModel,
    "mlp": MLPModel,
}
