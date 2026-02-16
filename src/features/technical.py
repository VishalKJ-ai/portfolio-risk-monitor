"""
Technical indicator feature engineering module.

Computes a comprehensive set of technical analysis features from OHLCV data,
including momentum, volatility, volume, and trend indicators.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalFeatureEngineer:
    """Computes technical indicators from OHLCV price data.

    Generates features commonly used in quantitative finance:
    RSI, MACD, Bollinger Bands, moving averages, volume ratios,
    rolling volatility, Rate of Change (ROC), and On-Balance Volume (OBV).

    Attributes:
        rsi_period: Lookback period for RSI calculation.
        macd_fast: Fast EMA period for MACD.
        macd_slow: Slow EMA period for MACD.
        macd_signal: Signal line period for MACD.
        bollinger_window: Window for Bollinger Bands.
        bollinger_std: Number of standard deviations for Bollinger Bands.
        ma_windows: List of moving average window sizes.
        volume_avg_window: Window for average volume calculation.
        volatility_window: Window for rolling volatility.
        roc_period: Period for Rate of Change.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bollinger_window: int = 20,
        bollinger_std: int = 2,
        ma_windows: Optional[List[int]] = None,
        volume_avg_window: int = 20,
        volatility_window: int = 20,
        roc_period: int = 10,
    ) -> None:
        """Initialize the technical feature engineer.

        Args:
            rsi_period: RSI lookback period (default 14).
            macd_fast: MACD fast EMA period (default 12).
            macd_slow: MACD slow EMA period (default 26).
            macd_signal: MACD signal line period (default 9).
            bollinger_window: Bollinger Bands window (default 20).
            bollinger_std: Bollinger Bands std multiplier (default 2).
            ma_windows: Moving average windows (default [20, 50, 200]).
            volume_avg_window: Volume averaging window (default 20).
            volatility_window: Volatility calculation window (default 20).
            roc_period: Rate of Change period (default 10).
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std
        self.ma_windows = ma_windows or [20, 50, 200]
        self.volume_avg_window = volume_avg_window
        self.volatility_window = volatility_window
        self.roc_period = roc_period

    def compute(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features for all tickers.

        Args:
            prices_df: DataFrame with columns Date, Ticker, Open, High, Low,
                       Close, Volume.

        Returns:
            DataFrame with Date, Ticker, and all computed technical features.
        """
        logger.info("Computing technical indicators...")
        all_features: List[pd.DataFrame] = []

        tickers = prices_df["Ticker"].unique()
        for ticker in tickers:
            ticker_df = prices_df[prices_df["Ticker"] == ticker].copy()
            ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)

            features = self._compute_for_ticker(ticker_df)
            features["Ticker"] = ticker
            all_features.append(features)

        result = pd.concat(all_features, ignore_index=True)
        logger.info(
            "Computed %d technical features for %d tickers (%d total rows)",
            len([c for c in result.columns if c not in ["Date", "Ticker"]]),
            len(tickers),
            len(result),
        )
        return result

    def _compute_for_ticker(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features for a single ticker.

        Args:
            df: OHLCV data for a single ticker, sorted by date.

        Returns:
            DataFrame with Date and all technical feature columns.
        """
        features = pd.DataFrame()
        features["Date"] = df["Date"]

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"].astype(float)

        # Daily returns
        returns = close.pct_change()
        features["daily_return"] = returns

        # RSI
        features["rsi"] = self._compute_rsi(close)

        # MACD
        macd, macd_signal, macd_hist = self._compute_macd(close)
        features["macd"] = macd
        features["macd_signal"] = macd_signal
        features["macd_histogram"] = macd_hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width, bb_pct = self._compute_bollinger(close)
        features["bb_upper"] = bb_upper
        features["bb_middle"] = bb_middle
        features["bb_lower"] = bb_lower
        features["bb_width"] = bb_width
        features["bb_percent"] = bb_pct

        # Moving Averages
        for window in self.ma_windows:
            ma = close.rolling(window=window).mean()
            features[f"ma_{window}"] = ma
            features[f"ma_{window}_ratio"] = close / ma

        # Volume Ratio
        avg_volume = volume.rolling(window=self.volume_avg_window).mean()
        features["volume_ratio"] = volume / avg_volume

        # Rolling Volatility (annualized)
        features["volatility_20d"] = returns.rolling(
            window=self.volatility_window
        ).std() * np.sqrt(252)

        # Rate of Change
        features["roc"] = self._compute_roc(close)

        # On-Balance Volume
        features["obv"] = self._compute_obv(close, volume)

        # Normalize OBV to a ratio for comparability across tickers
        obv_ma = features["obv"].rolling(window=20).mean()
        features["obv_ratio"] = features["obv"] / obv_ma.replace(0, np.nan)

        return features

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        """Compute Relative Strength Index (RSI).

        Args:
            close: Series of closing prices.

        Returns:
            RSI values (0-100 scale).
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_macd(
        self, close: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD, signal line, and histogram.

        Args:
            close: Series of closing prices.

        Returns:
            Tuple of (macd_line, signal_line, histogram) as pd.Series.
        """
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _compute_bollinger(
        self, close: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Compute Bollinger Bands and derived features.

        Args:
            close: Series of closing prices.

        Returns:
            Tuple of (upper, middle, lower, width, percent_b) as pd.Series.
        """
        middle = close.rolling(window=self.bollinger_window).mean()
        std = close.rolling(window=self.bollinger_window).std()

        upper = middle + self.bollinger_std * std
        lower = middle - self.bollinger_std * std
        width = (upper - lower) / middle
        percent_b = (close - lower) / (upper - lower)

        return upper, middle, lower, width, percent_b

    def _compute_roc(self, close: pd.Series) -> pd.Series:
        """Compute Rate of Change.

        Args:
            close: Series of closing prices.

        Returns:
            Rate of Change as percentage.
        """
        return close.pct_change(periods=self.roc_period) * 100

    def _compute_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On-Balance Volume.

        Args:
            close: Series of closing prices.
            volume: Series of trading volumes.

        Returns:
            On-Balance Volume series.
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        obv = (direction * volume).cumsum()
        return obv

    def get_feature_names(self) -> List[str]:
        """Return the list of feature names this engineer produces.

        Returns:
            List of technical feature column names.
        """
        names = [
            "daily_return", "rsi",
            "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent",
            "volume_ratio", "volatility_20d", "roc", "obv", "obv_ratio",
        ]
        for w in self.ma_windows:
            names.extend([f"ma_{w}", f"ma_{w}_ratio"])
        return names
