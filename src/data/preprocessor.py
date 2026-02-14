"""
Data preprocessing module for the Portfolio Risk Monitor.

Handles cleaning, normalization, merging of price data with sentiment
features, and creation of the binary target variable.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses raw market and sentiment data for model training.

    Handles missing value imputation, feature merging, target variable
    creation, and time-based train/val/test splitting.

    Attributes:
        target_threshold: Percentage drop threshold for the binary target.
        target_horizon: Number of trading days to look ahead.
        train_end: End date for training data.
        val_end: End date for validation data.
        processed_path: Path to save processed data.
    """

    def __init__(
        self,
        target_threshold: float = 0.02,
        target_horizon: int = 5,
        train_end: str = "2022-12-31",
        val_end: str = "2023-06-30",
        processed_path: str = "data/processed",
    ) -> None:
        """Initialize the data preprocessor.

        Args:
            target_threshold: Fraction drop to label as positive (e.g. 0.02 = 2%).
            target_horizon: Number of trading days to look ahead.
            train_end: Last date (inclusive) of the training set.
            val_end: Last date (inclusive) of the validation set.
            processed_path: Directory to save processed CSV files.
        """
        self.target_threshold = target_threshold
        self.target_horizon = target_horizon
        self.train_end = pd.Timestamp(train_end)
        self.val_end = pd.Timestamp(val_end)
        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def create_target(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Create binary target variable: did the stock drop > threshold in horizon?

        For each row, looks ahead `target_horizon` trading days and checks
        whether the minimum close price represents a drop exceeding
        `target_threshold` from the current close.

        Args:
            prices_df: DataFrame with columns Date, Ticker, Close.

        Returns:
            DataFrame with additional 'target' column (0 or 1).
        """
        logger.info(
            "Creating target variable (threshold=%.1f%%, horizon=%d days)",
            self.target_threshold * 100,
            self.target_horizon,
        )

        df = prices_df.copy()
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        targets: List[Optional[int]] = []

        for ticker in df["Ticker"].unique():
            mask = df["Ticker"] == ticker
            closes = df.loc[mask, "Close"].values

            ticker_targets: List[Optional[int]] = []
            for i in range(len(closes)):
                if i + self.target_horizon >= len(closes):
                    ticker_targets.append(None)  # Not enough future data
                else:
                    future_closes = closes[i + 1: i + 1 + self.target_horizon]
                    min_future = np.min(future_closes)
                    drop = (closes[i] - min_future) / closes[i]
                    ticker_targets.append(1 if drop >= self.target_threshold else 0)

            targets.extend(ticker_targets)

        df["target"] = targets
        n_before = len(df)
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)
        n_after = len(df)
        logger.info(
            "Target created: %d positive (%.1f%%), %d negative (%.1f%%). "
            "Dropped %d rows without sufficient future data.",
            df["target"].sum(),
            df["target"].mean() * 100,
            (1 - df["target"]).sum(),
            (1 - df["target"].mean()) * 100,
            n_before - n_after,
        )
        return df

    def merge_features(
        self,
        prices_with_target: pd.DataFrame,
        technical_features: pd.DataFrame,
        sentiment_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge price data with technical and sentiment features.

        Args:
            prices_with_target: DataFrame with Date, Ticker, Close, target.
            technical_features: DataFrame with Date, Ticker, and technical cols.
            sentiment_features: DataFrame with Date, Ticker, and sentiment cols.

        Returns:
            Merged DataFrame ready for model training.
        """
        logger.info("Merging features...")

        # Merge technical features
        df = prices_with_target.merge(
            technical_features,
            on=["Date", "Ticker"],
            how="left",
            suffixes=("", "_tech"),
        )

        # Merge sentiment features
        df = df.merge(
            sentiment_features,
            on=["Date", "Ticker"],
            how="left",
        )

        # Fill missing sentiment (days with no headlines) with neutral values
        sentiment_cols = [
            "sentiment_mean", "sentiment_min", "sentiment_max",
            "sentiment_count", "sentiment_std",
        ]
        for col in sentiment_cols:
            if col in df.columns:
                if col == "sentiment_count":
                    df[col] = df[col].fillna(0)
                elif col == "sentiment_std":
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(0.0)

        # Drop rows where technical features are NaN (warmup period)
        feature_cols = [
            c for c in df.columns
            if c not in ["Date", "Ticker", "target", "Open", "High", "Low", "Close", "Volume"]
        ]
        n_before = len(df)
        df = df.dropna(subset=feature_cols)
        n_after = len(df)
        logger.info(
            "After merging and dropping NaNs: %d rows (dropped %d warmup rows)",
            n_after,
            n_before - n_after,
        )

        # Save processed data
        output_file = self.processed_path / "features.csv"
        df.to_csv(output_file, index=False)
        logger.info("Saved processed features to %s", output_file)

        return df

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform time-based train/validation/test split.

        CRITICAL: This is a time-based split, NOT random, to avoid
        data leakage when working with time-series financial data.

        Args:
            df: Complete feature DataFrame with Date column.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        df["Date"] = pd.to_datetime(df["Date"])

        train = df[df["Date"] <= self.train_end].copy()
        val = df[(df["Date"] > self.train_end) & (df["Date"] <= self.val_end)].copy()
        test = df[df["Date"] > self.val_end].copy()

        logger.info(
            "Time-based split: train=%d (until %s), val=%d (until %s), test=%d (after %s)",
            len(train), self.train_end.date(),
            len(val), self.val_end.date(),
            len(test), self.val_end.date(),
        )
        logger.info(
            "Target distribution -> train: %.1f%%, val: %.1f%%, test: %.1f%%",
            train["target"].mean() * 100 if len(train) > 0 else 0,
            val["target"].mean() * 100 if len(val) > 0 else 0,
            test["target"].mean() * 100 if len(test) > 0 else 0,
        )

        return train, val, test

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract feature column names from the DataFrame.

        Args:
            df: DataFrame with all columns.

        Returns:
            List of feature column names (excludes metadata and target).
        """
        exclude = {"Date", "Ticker", "target", "Open", "High", "Low", "Close", "Volume"}
        return [c for c in df.columns if c not in exclude]

    def prepare_xy(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix X and target vector y.

        Args:
            df: DataFrame containing features and target.
            feature_cols: List of feature column names.

        Returns:
            Tuple of (X, y) as numpy arrays.
        """
        X = df[feature_cols].values.astype(np.float32)
        y = df["target"].values.astype(np.float32)

        # Replace any remaining inf/nan with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y
