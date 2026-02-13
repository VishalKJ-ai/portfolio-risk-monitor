"""
Data collection module for market data and financial news.

Collects historical OHLCV data via yfinance and financial headlines
from RSS feeds. Includes fallback mechanisms and sample data generation.
"""

import logging
import math
import random
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ticker parameters for synthetic sample data generation
# ---------------------------------------------------------------------------
_TICKER_PARAMS: Dict[str, Dict[str, float]] = {
    "SPY": {"start_price": 260.0, "annual_drift": 0.10, "annual_vol": 0.18},
    "QQQ": {"start_price": 165.0, "annual_drift": 0.14, "annual_vol": 0.22},
    "AAPL": {"start_price": 155.0, "annual_drift": 0.18, "annual_vol": 0.28},
    "MSFT": {"start_price": 105.0, "annual_drift": 0.20, "annual_vol": 0.26},
    "GOOGL": {"start_price": 1050.0, "annual_drift": 0.12, "annual_vol": 0.25},
}

_HEADLINE_TEMPLATES: Dict[str, List[str]] = {
    "positive": [
        "{ticker} beats earnings expectations, shares surge",
        "{ticker} reports record revenue in Q{q} results",
        "Analysts upgrade {ticker} to 'buy' after strong quarter",
        "{ticker} announces major share buyback program",
        "Wall Street bullish on {ticker} growth prospects",
        "{ticker} raises full-year guidance above consensus",
        "{ticker} expands into new market with strategic acquisition",
        "Institutional investors increase holdings in {ticker}",
        "{ticker} dividend increase signals management confidence",
        "Strong consumer demand drives {ticker} sales higher",
        "{ticker} launches innovative product to positive reviews",
        "{ticker} profit margins expand amid cost cutting efforts",
        "Market rally lifts {ticker} to new 52-week high",
        "{ticker} secures major government contract worth billions",
        "{ticker} cloud revenue growth accelerates in latest quarter",
    ],
    "negative": [
        "{ticker} misses revenue estimates, stock drops",
        "Concerns mount over {ticker} slowing growth trajectory",
        "Analysts downgrade {ticker} citing valuation concerns",
        "{ticker} faces regulatory scrutiny over business practices",
        "{ticker} warns of supply chain disruptions impacting margins",
        "{ticker} lowers guidance amid macroeconomic headwinds",
        "Short sellers increase bets against {ticker}",
        "{ticker} layoffs signal deeper problems in core business",
        "Competition intensifies as {ticker} loses market share",
        "{ticker} hit by data breach affecting millions of users",
        "Rising costs pressure {ticker} profit margins",
        "{ticker} faces class action lawsuit from shareholders",
        "Market selloff drags {ticker} below key support levels",
        "{ticker} CFO departure raises governance questions",
        "Trade tensions weigh on {ticker} international revenue",
    ],
    "neutral": [
        "{ticker} reports results in line with expectations",
        "{ticker} announces leadership transition plan",
        "Trading volume in {ticker} remains steady ahead of earnings",
        "{ticker} to present at upcoming investor conference",
        "Options activity in {ticker} suggests sideways movement",
        "{ticker} maintains dividend at current levels",
        "{ticker} files routine regulatory paperwork",
        "Sector rotation leaves {ticker} flat on the day",
        "{ticker} trading near 50-day moving average",
        "{ticker} board approves routine corporate matters",
        "Analysts maintain hold rating on {ticker}",
        "{ticker} completes previously announced restructuring",
        "Mixed signals for {ticker} as sector consolidates",
        "{ticker} volume picks up ahead of next week earnings",
        "{ticker} holds investor day with no major announcements",
    ],
    "market": [
        "Fed signals potential rate changes, markets react",
        "S&P 500 closes at session highs on broad rally",
        "Treasury yields climb as inflation data tops expectations",
        "Markets tumble on recession fears and weak economic data",
        "Oil prices surge on geopolitical tensions in Middle East",
        "Dollar strengthens as global growth concerns persist",
        "VIX spikes as volatility returns to equity markets",
        "Job report exceeds expectations, stocks mixed",
        "Consumer confidence drops to lowest level in six months",
        "Housing market shows signs of cooling as rates rise",
        "Tech sector leads market decline on growth concerns",
        "Retail investors drive unusual options activity",
        "Global markets rally on trade deal optimism",
        "Cryptocurrency volatility spills over into tech stocks",
        "GDP growth slows but beats revised expectations",
        "Manufacturing PMI signals contraction for third month",
        "Earnings season kicks off with mixed bank results",
        "Central banks globally signal tightening monetary policy",
        "Small caps outperform as risk appetite improves",
        "Bond market inversion deepens recession concerns",
    ],
}


class MarketDataCollector:
    """Collects historical market data from yfinance.

    Attributes:
        tickers: List of ticker symbols to collect data for.
        start_date: Start date for data collection.
        end_date: End date for data collection.
        raw_path: Directory to store raw data files.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        raw_path: str = "data/raw",
    ) -> None:
        """Initialize the market data collector.

        Args:
            tickers: List of ticker symbols (e.g., ['SPY', 'AAPL']).
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            raw_path: Path to store raw downloaded data.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_path = Path(raw_path)
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def collect(self) -> pd.DataFrame:
        """Download OHLCV data for all tickers using yfinance.

        Returns:
            DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume.

        Raises:
            RuntimeError: If data collection fails for all tickers.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error(
                "yfinance is not installed. Install with: pip install yfinance"
            )
            raise

        all_data: List[pd.DataFrame] = []
        failed_tickers: List[str] = []

        for ticker in self.tickers:
            try:
                logger.info("Downloading data for %s (%s to %s)",
                            ticker, self.start_date, self.end_date)
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                )
                if df.empty:
                    logger.warning("No data returned for %s", ticker)
                    failed_tickers.append(ticker)
                    continue

                df = df.reset_index()
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]
                df["Ticker"] = ticker
                df = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
                all_data.append(df)
                logger.info("  Collected %d rows for %s", len(df), ticker)

            except Exception as e:
                logger.error("Failed to download %s: %s", ticker, str(e))
                failed_tickers.append(ticker)

        if not all_data:
            raise RuntimeError(
                f"Data collection failed for all tickers: {failed_tickers}"
            )

        if failed_tickers:
            logger.warning("Failed tickers: %s", failed_tickers)

        combined = pd.concat(all_data, ignore_index=True)
        combined["Date"] = pd.to_datetime(combined["Date"])
        combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

        # Save raw data
        output_file = self.raw_path / "market_data.csv"
        combined.to_csv(output_file, index=False)
        logger.info("Saved raw market data to %s (%d rows)", output_file, len(combined))

        return combined


class NewsCollector:
    """Collects financial news headlines from RSS feeds.

    Attributes:
        rss_feeds: List of RSS feed URLs.
        max_headlines_per_day: Maximum headlines to retain per day.
        raw_path: Directory to store raw headline data.
    """

    def __init__(
        self,
        rss_feeds: List[str],
        max_headlines_per_day: int = 50,
        raw_path: str = "data/raw",
    ) -> None:
        """Initialize the news collector.

        Args:
            rss_feeds: List of RSS feed URLs.
            max_headlines_per_day: Maximum headlines to retain per day.
            raw_path: Path to store raw downloaded data.
        """
        self.rss_feeds = rss_feeds
        self.max_headlines_per_day = max_headlines_per_day
        self.raw_path = Path(raw_path)
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def collect(self) -> pd.DataFrame:
        """Fetch headlines from RSS feeds with fallback mechanisms.

        Returns:
            DataFrame with columns: date, headline, ticker, source.
        """
        all_headlines: List[Dict[str, str]] = []

        for feed_url in self.rss_feeds:
            try:
                headlines = self._fetch_rss(feed_url)
                all_headlines.extend(headlines)
                logger.info("Fetched %d headlines from %s", len(headlines), feed_url)
            except Exception as e:
                logger.warning("Failed to fetch from %s: %s", feed_url, str(e))

        if not all_headlines:
            logger.warning(
                "No headlines collected from RSS feeds. "
                "Consider using sample data (--mode sample)."
            )
            return pd.DataFrame(columns=["date", "headline", "ticker", "source"])

        df = pd.DataFrame(all_headlines)
        df["date"] = pd.to_datetime(df["date"])

        # Save raw headlines
        output_file = self.raw_path / "headlines.csv"
        df.to_csv(output_file, index=False)
        logger.info("Saved %d headlines to %s", len(df), output_file)

        return df

    def _fetch_rss(self, url: str) -> List[Dict[str, str]]:
        """Fetch and parse a single RSS feed.

        Args:
            url: RSS feed URL.

        Returns:
            List of headline dictionaries.
        """
        req = Request(url, headers={"User-Agent": "PortfolioRiskMonitor/1.0"})
        with urlopen(req, timeout=15) as response:
            content = response.read()

        root = ET.fromstring(content)
        headlines: List[Dict[str, str]] = []

        for item in root.iter("item"):
            title_elem = item.find("title")
            pub_date_elem = item.find("pubDate")

            if title_elem is not None and title_elem.text:
                headline_text = title_elem.text.strip()
                pub_date = ""
                if pub_date_elem is not None and pub_date_elem.text:
                    try:
                        dt = datetime.strptime(
                            pub_date_elem.text.strip()[:25],
                            "%a, %d %b %Y %H:%M:%S",
                        )
                        pub_date = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        pub_date = datetime.now().strftime("%Y-%m-%d")

                headlines.append({
                    "date": pub_date,
                    "headline": headline_text,
                    "ticker": "",
                    "source": url.split("/")[2] if "/" in url else "unknown",
                })

        return headlines


class SampleDataGenerator:
    """Generates realistic synthetic market data and headlines for testing.

    This allows reviewers to run the full pipeline without requiring
    API keys or internet access.

    Attributes:
        sample_path: Path to the sample data directory.
        tickers: List of tickers to generate data for.
        start_date: Start date for data generation.
        end_date: End date for data generation.
    """

    def __init__(
        self,
        sample_path: str = "data/sample",
        tickers: Optional[List[str]] = None,
        start_date: str = "2021-01-04",
        end_date: str = "2023-12-29",
    ) -> None:
        """Initialize the sample data generator.

        Args:
            sample_path: Directory to write sample CSV files.
            tickers: Tickers to generate. Defaults to SPY, QQQ, AAPL, MSFT, GOOGL.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
        """
        self.sample_path = Path(sample_path)
        self.tickers = tickers or list(_TICKER_PARAMS.keys())
        self.start_date = start_date
        self.end_date = end_date
        self.sample_path.mkdir(parents=True, exist_ok=True)
        np.random.seed(42)
        random.seed(42)

    def generate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate all sample data files and return DataFrames.

        Returns:
            Tuple of (prices_df, headlines_df, sentiment_df).
        """
        business_days = self._generate_business_days()
        logger.info("Generating sample data with %d business days", len(business_days))

        prices_df = self._generate_prices(business_days)
        headlines_df = self._generate_headlines(business_days)
        sentiment_df = self._generate_sentiment(headlines_df)

        return prices_df, headlines_df, sentiment_df

    def load_or_generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load existing sample data or generate if missing.

        Returns:
            Tuple of (prices_df, headlines_df, sentiment_df).
        """
        price_file = self.sample_path / "sample_prices.csv"
        headline_file = self.sample_path / "sample_headlines.csv"
        sentiment_file = self.sample_path / "sample_sentiment.csv"

        if price_file.exists() and headline_file.exists() and sentiment_file.exists():
            logger.info("Loading existing sample data from %s", self.sample_path)
            prices_df = pd.read_csv(price_file, parse_dates=["Date"])
            headlines_df = pd.read_csv(headline_file, parse_dates=["date"])
            sentiment_df = pd.read_csv(sentiment_file, parse_dates=["date"])
            return prices_df, headlines_df, sentiment_df

        logger.info("Sample data not found. Generating...")
        return self.generate_all()

    def _generate_business_days(self) -> List[datetime]:
        """Generate list of business days in the date range."""
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        days: List[datetime] = []
        current = start
        while current <= end:
            if current.weekday() < 5:
                days.append(current)
            current += timedelta(days=1)
        return days

    def _generate_prices(self, business_days: List[datetime]) -> pd.DataFrame:
        """Generate synthetic OHLCV data using geometric Brownian motion."""
        all_rows: List[Dict[str, Any]] = []
        dt = 1.0 / 252.0

        for ticker in self.tickers:
            params = _TICKER_PARAMS.get(
                ticker,
                {"start_price": 100.0, "annual_drift": 0.08, "annual_vol": 0.20},
            )
            price = params["start_price"]
            daily_drift = params["annual_drift"] * dt
            daily_vol = params["annual_vol"] * math.sqrt(dt)
            vol_state = 1.0

            for day in business_days:
                vol_state += 0.05 * (1.0 - vol_state) + 0.1 * np.random.randn()
                vol_state = max(0.5, min(2.0, vol_state))
                current_vol = daily_vol * vol_state
                ret = daily_drift + current_vol * np.random.randn()

                if np.random.random() < 0.03:
                    ret += np.random.choice([-1, 1]) * abs(np.random.randn()) * daily_vol * 2

                new_price = price * math.exp(ret)
                intraday_vol = abs(new_price - price) + new_price * 0.005 * abs(np.random.randn())
                high = max(price, new_price) + abs(np.random.randn()) * intraday_vol * 0.3
                low = min(price, new_price) - abs(np.random.randn()) * intraday_vol * 0.3
                low = max(low, new_price * 0.95)
                open_price = price + (new_price - price) * np.random.uniform(0.0, 0.4)
                high = max(high, open_price, new_price)
                low = min(low, open_price, new_price)

                base_volume = 50_000_000
                vol_mult = 1.0 + 0.5 * abs(ret / max(daily_vol, 1e-8))
                volume = int(base_volume * vol_mult * np.random.lognormal(0, 0.3))

                all_rows.append({
                    "Date": day.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Open": round(open_price, 2),
                    "High": round(high, 2),
                    "Low": round(low, 2),
                    "Close": round(new_price, 2),
                    "Volume": volume,
                })
                price = new_price

        df = pd.DataFrame(all_rows)
        df["Date"] = pd.to_datetime(df["Date"])
        output_file = self.sample_path / "sample_prices.csv"
        df.to_csv(output_file, index=False)
        logger.info("Generated %d price rows -> %s", len(df), output_file)
        return df

    def _generate_headlines(self, business_days: List[datetime]) -> pd.DataFrame:
        """Generate synthetic financial headlines."""
        headlines: List[Dict[str, str]] = []

        for day in business_days:
            n_headlines = int(np.random.choice([0, 1, 1, 1, 2, 2, 3]))
            for _ in range(n_headlines):
                category = str(np.random.choice(
                    ["positive", "negative", "neutral", "market"],
                    p=[0.25, 0.25, 0.30, 0.20],
                ))
                template = random.choice(_HEADLINE_TEMPLATES[category])

                if "{ticker}" in template:
                    ticker = random.choice(self.tickers)
                    headline_text = template.format(
                        ticker=ticker, q=random.choice([1, 2, 3, 4])
                    )
                else:
                    ticker = ""
                    headline_text = template

                headlines.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "headline": headline_text,
                    "ticker": ticker,
                    "source": random.choice([
                        "Yahoo Finance", "Reuters", "Bloomberg", "CNBC", "MarketWatch"
                    ]),
                })

        df = pd.DataFrame(headlines)
        df["date"] = pd.to_datetime(df["date"])
        output_file = self.sample_path / "sample_headlines.csv"
        df.to_csv(output_file, index=False)
        logger.info("Generated %d headlines -> %s", len(df), output_file)
        return df

    def _generate_sentiment(self, headlines_df: pd.DataFrame) -> pd.DataFrame:
        """Generate pre-computed sentiment scores for headlines."""
        positive_words = [
            "beats", "surge", "record", "upgrade", "bullish", "raises",
            "expands", "increase", "strong", "innovative", "rally", "high",
            "secures", "accelerates", "outperform", "optimism",
        ]
        negative_words = [
            "misses", "drops", "concerns", "downgrade", "scrutiny",
            "disruptions", "lowers", "layoffs", "loses", "breach",
            "pressure", "lawsuit", "selloff", "departure", "tumble",
            "recession", "fears", "contraction", "inversion",
        ]

        rows: List[Dict[str, Any]] = []
        for _, row in headlines_df.iterrows():
            headline_lower = str(row["headline"]).lower()
            pos_count = sum(1 for w in positive_words if w in headline_lower)
            neg_count = sum(1 for w in negative_words if w in headline_lower)

            if pos_count > neg_count:
                bp = np.random.uniform(0.55, 0.92)
                bn = np.random.uniform(0.02, 0.15)
                label = "positive"
            elif neg_count > pos_count:
                bn = np.random.uniform(0.55, 0.92)
                bp = np.random.uniform(0.02, 0.15)
                label = "negative"
            else:
                bneut = np.random.uniform(0.45, 0.75)
                bp = np.random.uniform(0.05, 0.30)
                bn = 1.0 - bneut - bp
                label = "neutral"

            if label != "neutral":
                bneut = 1.0 - bp - bn

            scores = np.array([max(0.01, bp), max(0.01, bn), max(0.01, bneut)])
            scores = scores / scores.sum()

            rows.append({
                "date": row["date"],
                "headline": row["headline"],
                "ticker": row["ticker"],
                "source": row["source"],
                "positive": round(float(scores[0]), 4),
                "negative": round(float(scores[1]), 4),
                "neutral": round(float(scores[2]), 4),
                "label": label,
                "compound_score": round(float(scores[0] - scores[1]), 4),
            })

        df = pd.DataFrame(rows)
        output_file = self.sample_path / "sample_sentiment.csv"
        df.to_csv(output_file, index=False)
        logger.info("Generated %d sentiment rows -> %s", len(df), output_file)
        return df
