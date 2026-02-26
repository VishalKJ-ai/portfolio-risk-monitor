"""
Unit tests for feature engineering modules.

Tests technical indicator computation, sentiment aggregation,
and edge cases in feature calculation.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import TechnicalFeatureEngineer
from src.features.sentiment import SentimentFeatureEngineer


class TestTechnicalFeatureEngineer:
    """Tests for the TechnicalFeatureEngineer class."""

    @pytest.fixture
    def sample_prices(self) -> pd.DataFrame:
        """Create a sample price DataFrame for testing."""
        np.random.seed(42)
        dates = pd.bdate_range("2022-01-01", periods=300)
        price = 100.0
        rows = []
        for date in dates:
            ret = np.random.normal(0.0005, 0.015)
            price *= (1 + ret)
            rows.append({
                "Date": date,
                "Ticker": "TEST",
                "Open": round(price * (1 + np.random.uniform(-0.005, 0.005)), 2),
                "High": round(price * (1 + abs(np.random.normal(0, 0.01))), 2),
                "Low": round(price * (1 - abs(np.random.normal(0, 0.01))), 2),
                "Close": round(price, 2),
                "Volume": int(np.random.lognormal(17, 0.3)),
            })
        return pd.DataFrame(rows)

    @pytest.fixture
    def engineer(self) -> TechnicalFeatureEngineer:
        """Create a TechnicalFeatureEngineer with default settings."""
        return TechnicalFeatureEngineer()

    def test_compute_returns_dataframe(
        self, engineer: TechnicalFeatureEngineer, sample_prices: pd.DataFrame
    ) -> None:
        """Test that compute() returns a DataFrame with expected columns."""
        result = engineer.compute(sample_prices)
        assert isinstance(result, pd.DataFrame)
        assert "Date" in result.columns
        assert "Ticker" in result.columns
        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert "bb_upper" in result.columns

    def test_rsi_range(
        self, engineer: TechnicalFeatureEngineer, sample_prices: pd.DataFrame
    ) -> None:
        """Test that RSI values are within [0, 100]."""
        result = engineer.compute(sample_prices)
        rsi = result["rsi"].dropna()
        assert rsi.min() >= 0, f"RSI minimum {rsi.min()} is below 0"
        assert rsi.max() <= 100, f"RSI maximum {rsi.max()} is above 100"

    def test_bollinger_band_ordering(
        self, engineer: TechnicalFeatureEngineer, sample_prices: pd.DataFrame
    ) -> None:
        """Test that Bollinger Bands maintain upper > middle > lower ordering."""
        result = engineer.compute(sample_prices)
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_middle"]).all(), \
            "Upper band should be >= middle band"
        assert (valid["bb_middle"] >= valid["bb_lower"]).all(), \
            "Middle band should be >= lower band"

    def test_moving_average_windows(
        self, engineer: TechnicalFeatureEngineer, sample_prices: pd.DataFrame
    ) -> None:
        """Test that all configured moving averages are computed."""
        result = engineer.compute(sample_prices)
        for window in engineer.ma_windows:
            assert f"ma_{window}" in result.columns, \
                f"Missing MA column: ma_{window}"
            assert f"ma_{window}_ratio" in result.columns, \
                f"Missing MA ratio column: ma_{window}_ratio"

    def test_volume_ratio_positive(
        self, engineer: TechnicalFeatureEngineer, sample_prices: pd.DataFrame
    ) -> None:
        """Test that volume ratio is positive."""
        result = engineer.compute(sample_prices)
        vol_ratio = result["volume_ratio"].dropna()
        assert (vol_ratio > 0).all(), "Volume ratio should be positive"

    def test_feature_names_match_output(
        self, engineer: TechnicalFeatureEngineer, sample_prices: pd.DataFrame
    ) -> None:
        """Test that get_feature_names() matches the actual output columns."""
        result = engineer.compute(sample_prices)
        expected_names = engineer.get_feature_names()
        output_cols = [c for c in result.columns if c not in ["Date", "Ticker"]]
        for name in expected_names:
            assert name in output_cols, f"Expected feature '{name}' not in output"


class TestSentimentFeatureEngineer:
    """Tests for the SentimentFeatureEngineer class."""

    @pytest.fixture
    def sample_sentiment(self) -> pd.DataFrame:
        """Create sample pre-computed sentiment data."""
        return pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02"] * 3 + ["2023-01-03"] * 2),
            "headline": [
                "Stock surges on earnings beat",
                "Market drops on recession fears",
                "Trading volume steady",
                "Company announces buyback",
                "Analyst downgrades stock",
            ],
            "ticker": ["AAPL", "AAPL", "", "MSFT", "MSFT"],
            "source": ["Reuters"] * 5,
            "positive": [0.8, 0.1, 0.3, 0.7, 0.1],
            "negative": [0.05, 0.75, 0.2, 0.1, 0.8],
            "neutral": [0.15, 0.15, 0.5, 0.2, 0.1],
            "compound_score": [0.75, -0.65, 0.1, 0.6, -0.7],
            "label": ["positive", "negative", "neutral", "positive", "negative"],
        })

    def test_precomputed_validation(self, sample_sentiment: pd.DataFrame) -> None:
        """Test that pre-computed sentiment is validated correctly."""
        engineer = SentimentFeatureEngineer(use_precomputed=True)
        result = engineer.score_headlines(sample_sentiment)
        assert "compound_score" in result.columns
        assert len(result) == len(sample_sentiment)

    def test_daily_aggregation(self, sample_sentiment: pd.DataFrame) -> None:
        """Test daily sentiment aggregation produces correct features."""
        engineer = SentimentFeatureEngineer(use_precomputed=True)
        scored = engineer.score_headlines(sample_sentiment)
        daily = engineer.aggregate_daily(scored, tickers=["AAPL", "MSFT"])

        assert "sentiment_mean" in daily.columns
        assert "sentiment_min" in daily.columns
        assert "sentiment_max" in daily.columns
        assert "sentiment_count" in daily.columns
        assert "sentiment_std" in daily.columns
        assert "Ticker" in daily.columns

    def test_market_headlines_distributed(self, sample_sentiment: pd.DataFrame) -> None:
        """Test that market-wide headlines are included for all tickers."""
        engineer = SentimentFeatureEngineer(use_precomputed=True)
        scored = engineer.score_headlines(sample_sentiment)
        daily = engineer.aggregate_daily(scored, tickers=["AAPL", "MSFT"])

        # AAPL on 2023-01-02: 2 AAPL headlines + 1 market = 3 headlines
        aapl_jan2 = daily[
            (daily["Ticker"] == "AAPL")
            & (daily["Date"] == pd.Timestamp("2023-01-02"))
        ]
        if not aapl_jan2.empty:
            assert aapl_jan2["sentiment_count"].values[0] == 3
