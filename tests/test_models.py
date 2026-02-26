"""
Unit tests for model implementations.

Tests model fitting, prediction, evaluation, and serialization
for all three model types.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.data.preprocessor import DataPreprocessor


class TestLogisticRegressionModel:
    """Tests for the LogisticRegressionModel class."""

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5 > 0).astype(np.float32)
        return X, y

    def test_fit_predict(self, sample_data: tuple) -> None:
        """Test that the model can fit and predict."""
        X, y = sample_data
        model = LogisticRegressionModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba_shape(self, sample_data: tuple) -> None:
        """Test that predict_proba returns correct shape."""
        X, y = sample_data
        model = LogisticRegressionModel()
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_evaluate_returns_metrics(self, sample_data: tuple) -> None:
        """Test that evaluate returns expected metrics."""
        X, y = sample_data
        model = LogisticRegressionModel()
        model.fit(X, y)
        metrics = model.evaluate(X, y)

        expected_keys = {"model", "accuracy", "precision", "recall", "f1", "roc_auc"}
        assert expected_keys.issubset(set(metrics.keys()))
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_save_load(self, sample_data: tuple) -> None:
        """Test model serialization and deserialization."""
        X, y = sample_data
        model = LogisticRegressionModel()
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.pkl")
            model.save(path)

            loaded_model = LogisticRegressionModel()
            loaded_model.load(path)

            original_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)
            np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_feature_importance(self, sample_data: tuple) -> None:
        """Test feature importance extraction."""
        X, y = sample_data
        model = LogisticRegressionModel()
        model.fit(X, y)

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        importance = model.get_feature_importance(feature_names)

        assert len(importance) == X.shape[1]
        assert all(v >= 0 for v in importance.values())


class TestRandomForestModel:
    """Tests for the RandomForestModel class."""

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5 > 0).astype(np.float32)
        return X, y

    def test_fit_without_tuning(self, sample_data: tuple) -> None:
        """Test fitting without hyperparameter tuning."""
        X, y = sample_data
        model = RandomForestModel()
        model.fit(X, y, tune=False)

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_fit_with_tuning(self, sample_data: tuple) -> None:
        """Test fitting with TimeSeriesSplit hyperparameter tuning."""
        X, y = sample_data
        model = RandomForestModel(
            n_estimators_range=[50, 100],
            max_depth_range=[3, 5],
            min_samples_split_range=[2],
            min_samples_leaf_range=[1],
            n_cv_splits=3,
        )
        model.fit(X, y, tune=True)

        assert model.best_params is not None
        assert "n_estimators" in model.best_params


class TestDataPreprocessor:
    """Tests for the DataPreprocessor class."""

    @pytest.fixture
    def sample_prices(self) -> pd.DataFrame:
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.bdate_range("2021-01-01", periods=500)
        price = 100.0
        rows = []
        for date in dates:
            ret = np.random.normal(0.0005, 0.015)
            price *= (1 + ret)
            rows.append({
                "Date": date,
                "Ticker": "TEST",
                "Open": round(price, 2),
                "High": round(price * 1.01, 2),
                "Low": round(price * 0.99, 2),
                "Close": round(price, 2),
                "Volume": 1000000,
            })
        return pd.DataFrame(rows)

    def test_create_target_binary(self, sample_prices: pd.DataFrame) -> None:
        """Test that target variable is binary (0 or 1)."""
        preprocessor = DataPreprocessor(target_threshold=0.02, target_horizon=5)
        result = preprocessor.create_target(sample_prices)

        assert "target" in result.columns
        assert set(result["target"].unique()).issubset({0, 1})

    def test_time_based_split_no_leakage(self, sample_prices: pd.DataFrame) -> None:
        """Test that time-based split has no temporal leakage."""
        preprocessor = DataPreprocessor(
            train_end="2021-12-31",
            val_end="2022-06-30",
        )
        result = preprocessor.create_target(sample_prices)
        result["feature_1"] = np.random.randn(len(result))

        train, val, test = preprocessor.split_data(result)

        if not train.empty and not val.empty:
            assert train["Date"].max() < val["Date"].min(), \
                "Training data must end before validation data starts"

        if not val.empty and not test.empty:
            assert val["Date"].max() < test["Date"].min(), \
                "Validation data must end before test data starts"

    def test_target_no_future_at_end(self, sample_prices: pd.DataFrame) -> None:
        """Test that rows without enough future data are dropped."""
        preprocessor = DataPreprocessor(target_threshold=0.02, target_horizon=5)
        result = preprocessor.create_target(sample_prices)

        # The last 5 rows should have been dropped (no future data)
        assert len(result) < len(sample_prices)
