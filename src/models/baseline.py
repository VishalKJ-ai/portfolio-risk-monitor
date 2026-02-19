"""
Baseline logistic regression model for market downturn prediction.

Provides a simple but interpretable baseline using StandardScaler
and L2-regularized logistic regression via scikit-learn.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LogisticRegressionModel:
    """Baseline logistic regression model with standardized features.

    Uses a scikit-learn Pipeline of StandardScaler and LogisticRegression
    with L2 regularization. Serves as a baseline for comparison against
    more complex models.

    Attributes:
        C: Inverse regularization strength.
        penalty: Regularization type ('l2').
        max_iter: Maximum iterations for solver convergence.
        solver: Optimization algorithm.
        model: The fitted sklearn Pipeline.
        name: Human-readable model name.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
    ) -> None:
        """Initialize the logistic regression baseline.

        Args:
            C: Inverse regularization strength (smaller = stronger reg).
            max_iter: Maximum solver iterations.
            solver: Optimization algorithm ('lbfgs', 'saga', etc.).
        """
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.name = "LogisticRegression"

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=self.C,
                l1_ratio=0,
                max_iter=self.max_iter,
                solver=self.solver,
                random_state=42,
                class_weight="balanced",
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        """Fit the model on training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target array of shape (n_samples,).

        Returns:
            self, for method chaining.
        """
        logger.info(
            "Training %s on %d samples with %d features",
            self.name, X.shape[0], X.shape[1],
        )
        self.model.fit(X, y)

        # Log coefficient statistics
        coefs = self.model.named_steps["classifier"].coef_[0]
        logger.info(
            "  Coefficients: mean=%.4f, std=%.4f, max=%.4f, min=%.4f",
            np.mean(coefs), np.std(coefs), np.max(coefs), np.min(coefs),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class.
        """
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model and return metrics.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Dictionary containing accuracy, precision, recall, F1, and ROC-AUC.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = {
            "model": self.name,
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, y_proba)),
        }

        logger.info(
            "%s evaluation -> Accuracy: %.3f, F1: %.3f, ROC-AUC: %.3f",
            self.name, metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
        )
        logger.info("\n%s", classification_report(y, y_pred, zero_division=0))

        return metrics

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save the model (e.g. 'models/baseline.pkl').
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path)
        logger.info("Saved %s to %s", self.name, output_path)

    def load(self, path: str) -> "LogisticRegressionModel":
        """Load a previously saved model.

        Args:
            path: File path to the saved model.

        Returns:
            self, with the loaded model.
        """
        self.model = joblib.load(path)
        logger.info("Loaded %s from %s", self.name, path)
        return self

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model coefficients.

        Args:
            feature_names: List of feature names matching the training data.

        Returns:
            Dictionary mapping feature names to absolute coefficient values.
        """
        coefs = np.abs(self.model.named_steps["classifier"].coef_[0])
        importance = dict(zip(feature_names, coefs))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
