"""
Sentiment feature engineering module using FinBERT.

Scores financial news headlines using a pretrained FinBERT model and
aggregates daily sentiment metrics per ticker.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentFeatureEngineer:
    """Computes sentiment features from financial news headlines.

    Uses ProsusAI/finbert for headline-level sentiment scoring, then
    aggregates to daily ticker-level features: mean, min, max, std,
    and count of sentiment scores.

    In sample mode, uses pre-computed sentiment scores to avoid
    requiring HuggingFace model downloads.

    Attributes:
        model_name: HuggingFace model identifier for FinBERT.
        max_length: Maximum token length for the tokenizer.
        batch_size: Batch size for model inference.
        use_precomputed: If True, skip model inference and use existing scores.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length: int = 512,
        batch_size: int = 32,
        use_precomputed: bool = False,
    ) -> None:
        """Initialize the sentiment feature engineer.

        Args:
            model_name: HuggingFace model name for FinBERT.
            max_length: Maximum sequence length for tokenization.
            batch_size: Batch size for model inference.
            use_precomputed: Whether to use pre-computed sentiment scores.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_precomputed = use_precomputed
        self._model = None
        self._tokenizer = None

    def score_headlines(self, headlines_df: pd.DataFrame) -> pd.DataFrame:
        """Score headlines with FinBERT sentiment.

        Args:
            headlines_df: DataFrame with columns: date, headline, ticker.
                          If pre-computed, also: positive, negative, neutral,
                          compound_score.

        Returns:
            DataFrame with sentiment scores for each headline.
        """
        if self.use_precomputed:
            logger.info("Using pre-computed sentiment scores")
            return self._validate_precomputed(headlines_df)

        return self._score_with_finbert(headlines_df)

    def _validate_precomputed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and return pre-computed sentiment data.

        Args:
            df: DataFrame with pre-computed sentiment columns.

        Returns:
            Validated DataFrame with required columns.
        """
        required_cols = ["date", "headline", "ticker", "compound_score"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Pre-computed sentiment data missing columns: {missing}"
            )

        # Ensure compound_score is numeric
        df = df.copy()
        df["compound_score"] = pd.to_numeric(df["compound_score"], errors="coerce")
        df["compound_score"] = df["compound_score"].fillna(0.0)

        return df

    def _score_with_finbert(self, headlines_df: pd.DataFrame) -> pd.DataFrame:
        """Score headlines using the FinBERT model.

        Args:
            headlines_df: DataFrame with columns: date, headline, ticker.

        Returns:
            DataFrame with added sentiment score columns.
        """
        if self._model is None:
            self._load_model()

        df = headlines_df.copy()
        headlines = df["headline"].tolist()

        logger.info("Scoring %d headlines with FinBERT...", len(headlines))

        all_scores: List[Dict[str, float]] = []

        for i in range(0, len(headlines), self.batch_size):
            batch = headlines[i: i + self.batch_size]
            scores = self._infer_batch(batch)
            all_scores.extend(scores)

            if (i + self.batch_size) % 100 == 0:
                logger.info("  Scored %d / %d headlines", min(i + self.batch_size, len(headlines)), len(headlines))

        scores_df = pd.DataFrame(all_scores)
        df["positive"] = scores_df["positive"]
        df["negative"] = scores_df["negative"]
        df["neutral"] = scores_df["neutral"]
        df["compound_score"] = scores_df["positive"] - scores_df["negative"]

        logger.info(
            "Sentiment scoring complete. Mean compound score: %.3f",
            df["compound_score"].mean(),
        )

        return df

    def _load_model(self) -> None:
        """Load FinBERT model and tokenizer from HuggingFace."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for FinBERT sentiment scoring. "
                "Install with: pip install transformers torch\n"
                "Alternatively, use --mode sample to run with pre-computed scores."
            )

        logger.info("Loading FinBERT model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        logger.info("FinBERT model loaded successfully")

    def _infer_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Run inference on a batch of texts.

        Args:
            texts: List of headline strings.

        Returns:
            List of score dictionaries with positive, negative, neutral.
        """
        import torch

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        results: List[Dict[str, float]] = []
        for prob in probs:
            results.append({
                "positive": float(prob[0]),
                "negative": float(prob[1]),
                "neutral": float(prob[2]),
            })

        return results

    def aggregate_daily(
        self,
        scored_df: pd.DataFrame,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate headline-level sentiment to daily ticker-level features.

        Creates per-ticker daily features: mean sentiment, min, max,
        standard deviation, and count of headlines.

        Args:
            scored_df: DataFrame with date, ticker, compound_score columns.
            tickers: List of tickers to include. If None, uses all tickers
                     found in the data.

        Returns:
            DataFrame with Date, Ticker, and aggregated sentiment features.
        """
        logger.info("Aggregating daily sentiment features...")

        df = scored_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Handle both ticker-specific and market-wide headlines
        # Market-wide headlines (empty ticker) are assigned to all tickers
        market_headlines = df[df["ticker"] == ""].copy()
        ticker_headlines = df[df["ticker"] != ""].copy()

        if tickers is None:
            tickers = ticker_headlines["ticker"].unique().tolist()

        all_daily: List[pd.DataFrame] = []

        for ticker in tickers:
            # Combine ticker-specific and market headlines
            ticker_data = pd.concat([
                ticker_headlines[ticker_headlines["ticker"] == ticker],
                market_headlines,
            ], ignore_index=True)

            if ticker_data.empty:
                continue

            daily_agg = ticker_data.groupby("date")["compound_score"].agg(
                sentiment_mean="mean",
                sentiment_min="min",
                sentiment_max="max",
                sentiment_std="std",
                sentiment_count="count",
            ).reset_index()

            daily_agg["sentiment_std"] = daily_agg["sentiment_std"].fillna(0)
            daily_agg["Ticker"] = ticker
            daily_agg = daily_agg.rename(columns={"date": "Date"})

            all_daily.append(daily_agg)

        if not all_daily:
            logger.warning("No sentiment data to aggregate")
            return pd.DataFrame(
                columns=["Date", "Ticker", "sentiment_mean", "sentiment_min",
                          "sentiment_max", "sentiment_std", "sentiment_count"]
            )

        result = pd.concat(all_daily, ignore_index=True)
        logger.info(
            "Aggregated sentiment: %d daily records for %d tickers",
            len(result),
            len(tickers),
        )
        return result
