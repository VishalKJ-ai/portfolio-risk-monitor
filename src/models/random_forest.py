"""
Random Forest model with time-series cross-validation.

Implements hyperparameter tuning using TimeSeriesSplit to respect
the temporal ordering of financial data.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest classifier with time-series aware cross-validation.

    Performs hyperparameter tuning using TimeSeriesSplit to prevent
    data leakage from future observations. Extracts feature importance
    for interpretability.

    Attributes:
        n_estimators_range: Range of n_estimators to search.
        max_depth_range: Range of max_depth values to search.
        min_samples_split_range: Range of min_samples_split to search.
        min_samples_leaf_range: Range of min_samples_leaf to search.
        n_cv_splits: Number of TimeSeriesSplit folds.
        random_state: Random seed for reproducibility.
        model: The fitted RandomForestClassifier.
        scaler: StandardScaler for feature normalization.
        best_params: Best hyperparameters found during tuning.
        name: Human-readable model name.
    """

    def __init__(
        self,
        n_estimators_range: List[int] = None,
        max_depth_range: List[Optional[int]] = None,
        min_samples_split_range: List[int] = None,
        min_samples_leaf_range: List[int] = None,
        n_cv_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        """Initialize the Random Forest model.

        Args:
            n_estimators_range: List of tree counts to try (default [100, 200, 300]).
            max_depth_range: List of max depths (default [5, 10, 15, None]).
            min_samples_split_range: List of min split sizes (default [2, 5, 10]).
            min_samples_leaf_range: List of min leaf sizes (default [1, 2, 4]).
            n_cv_splits: Number of folds for TimeSeriesSplit.
            random_state: Random seed.
        """
        self.n_estimators_range = n_estimators_range or [100, 200, 300]
        self.max_depth_range = max_depth_range or [5, 10, 15, None]
        self.min_samples_split_range = min_samples_split_range or [2, 5, 10]
        self.min_samples_leaf_range = min_samples_leaf_range or [1, 2, 4]
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.name = "RandomForest"

        self.model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler()
        self.best_params: Dict[str, Any] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tune: bool = True,
    ) -> "RandomForestModel":
        """Fit the model, optionally with hyperparameter tuning.

        When tune=True, uses a randomized search over the parameter
        grid with TimeSeriesSplit cross-validation. When tune=False,
        uses the first value from each range for a quick fit.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target array of shape (n_samples,).
            tune: Whether to perform hyperparameter tuning.

        Returns:
            self, for method chaining.
        """
        logger.info(
            "Training %s on %d samples with %d features (tune=%s)",
            self.name, X.shape[0], X.shape[1], tune,
        )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        if tune:
            self._tune_hyperparameters(X_scaled, y)
        else:
            self.best_params = {
                "n_estimators": self.n_estimators_range[0],
                "max_depth": self.max_depth_range[0],
                "min_samples_split": self.min_samples_split_range[0],
                "min_samples_leaf": self.min_samples_leaf_range[0],
            }

        self.model = RandomForestClassifier(
            n_estimators=self.best_params["n_estimators"],
            max_depth=self.best_params["max_depth"],
            min_samples_split=self.best_params["min_samples_split"],
            min_samples_leaf=self.best_params["min_samples_leaf"],
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)

        logger.info("  Best params: %s", self.best_params)
        return self

    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """Tune hyperparameters with TimeSeriesSplit cross-validation.

        Uses a simplified grid search (random subset) to balance
        thoroughness with training time.

        Args:
            X: Scaled feature matrix.
            y: Target array.
        """
        logger.info("  Tuning hyperparameters with %d-fold TimeSeriesSplit...",
                     self.n_cv_splits)

        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        best_score = -1.0
        best_params: Dict[str, Any] = {}

        # Generate parameter combinations (limit to avoid excessive training)
        rng = np.random.RandomState(self.random_state)
        param_combos = []
        for n_est in self.n_estimators_range:
            for depth in self.max_depth_range:
                for split in self.min_samples_split_range:
                    for leaf in self.min_samples_leaf_range:
                        param_combos.append({
                            "n_estimators": n_est,
                            "max_depth": depth,
                            "min_samples_split": split,
                            "min_samples_leaf": leaf,
                        })

        # Subsample if too many combinations
        max_combos = 20
        if len(param_combos) > max_combos:
            indices = rng.choice(len(param_combos), max_combos, replace=False)
            param_combos = [param_combos[i] for i in indices]

        logger.info("  Evaluating %d parameter combinations...", len(param_combos))

        for i, params in enumerate(param_combos):
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                rf = RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    random_state=self.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                y_proba = rf.predict_proba(X_val)[:, 1]

                try:
                    score = roc_auc_score(y_val, y_proba)
                except ValueError:
                    score = 0.5  # Default if only one class in fold
                cv_scores.append(score)

            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()

        self.best_params = best_params
        logger.info("  Best CV ROC-AUC: %.4f with params: %s", best_score, best_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of predicted labels (0 or 1).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model and return metrics.

        Args:
            X: Feature matrix.
            y: True labels.

        Returns:
            Dictionary containing accuracy, precision, recall, F1, ROC-AUC,
            and best hyperparameters.
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
            "best_params": self.best_params,
        }

        logger.info(
            "%s evaluation -> Accuracy: %.3f, F1: %.3f, ROC-AUC: %.3f",
            self.name, metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
        )
        logger.info("\n%s", classification_report(y, y_pred, zero_division=0))

        return metrics

    def save(self, path: str) -> None:
        """Save model and scaler to disk.

        Args:
            path: File path for saving (e.g. 'models/random_forest.pkl').
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "params": self.best_params}, output_path)
        logger.info("Saved %s to %s", self.name, output_path)

    def load(self, path: str) -> "RandomForestModel":
        """Load a previously saved model.

        Args:
            path: File path to the saved model.

        Returns:
            self, with loaded model and scaler.
        """
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.best_params = data["params"]
        logger.info("Loaded %s from %s", self.name, path)
        return self

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from the trained model.

        Args:
            feature_names: List of feature names matching the training data.

        Returns:
            Dictionary mapping feature names to importance scores, sorted descending.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        importances = self.model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
