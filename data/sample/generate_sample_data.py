"""
Script to generate realistic sample data for the Portfolio Risk Monitor.

This creates synthetic but realistic price data, headlines, and sentiment
scores so reviewers can test the pipeline without API keys.

Usage:
    python data/sample/generate_sample_data.py
"""

import csv
import logging
import math
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Realistic starting prices and annual drift/vol for each ticker
TICKER_PARAMS: Dict[str, Dict[str, float]] = {
    "SPY": {"start_price": 260.0, "annual_drift": 0.10, "annual_vol": 0.18},
    "QQQ": {"start_price": 165.0, "annual_drift": 0.14, "annual_vol": 0.22},
    "AAPL": {"start_price": 155.0, "annual_drift": 0.18, "annual_vol": 0.28},
    "MSFT": {"start_price": 105.0, "annual_drift": 0.20, "annual_vol": 0.26},
    "GOOGL": {"start_price": 1050.0, "annual_drift": 0.12, "annual_vol": 0.25},
}

# Headline templates for different sentiment categories
HEADLINE_TEMPLATES: Dict[str, List[str]] = {
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


def generate_business_days(
    start_date: str, end_date: str
) -> List[datetime]:
    """Generate list of business days between start and end dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday to Friday
            days.append(current)
        current += timedelta(days=1)
    return days


def generate_price_series(
    business_days: List[datetime],
    start_price: float,
    annual_drift: float,
    annual_vol: float,
) -> List[Dict[str, Any]]:
    """
    Generate realistic OHLCV data using geometric Brownian motion
    with mean-reverting volatility.
    """
    dt = 1.0 / 252.0  # Trading day fraction
    daily_drift = annual_drift * dt
    daily_vol = annual_vol * math.sqrt(dt)

    prices = []
    price = start_price
    vol_state = 1.0  # Volatility regime multiplier

    for i, day in enumerate(business_days):
        # Mean-reverting volatility regime
        vol_state += 0.05 * (1.0 - vol_state) + 0.1 * np.random.randn()
        vol_state = max(0.5, min(2.0, vol_state))

        # Daily return with regime-dependent vol
        current_vol = daily_vol * vol_state
        ret = daily_drift + current_vol * np.random.randn()

        # Occasional larger moves (fat tails)
        if np.random.random() < 0.03:
            ret += np.random.choice([-1, 1]) * abs(np.random.randn()) * daily_vol * 2

        new_price = price * math.exp(ret)

        # Generate OHLC around the close
        intraday_vol = abs(new_price - price) + new_price * 0.005 * abs(np.random.randn())
        high = max(price, new_price) + abs(np.random.randn()) * intraday_vol * 0.3
        low = min(price, new_price) - abs(np.random.randn()) * intraday_vol * 0.3
        low = max(low, new_price * 0.95)  # Prevent unrealistic lows
        open_price = price + (new_price - price) * np.random.uniform(0.0, 0.4)

        # Ensure OHLC consistency
        high = max(high, open_price, new_price)
        low = min(low, open_price, new_price)

        # Volume: base volume with random variation and trend
        base_volume = 50_000_000
        vol_mult = 1.0 + 0.5 * abs(ret / daily_vol)  # Higher vol on bigger moves
        volume = int(base_volume * vol_mult * np.random.lognormal(0, 0.3))

        prices.append({
            "Date": day.strftime("%Y-%m-%d"),
            "Open": round(open_price, 2),
            "High": round(high, 2),
            "Low": round(low, 2),
            "Close": round(new_price, 2),
            "Volume": volume,
        })

        price = new_price

    return prices


def generate_headlines(
    business_days: List[datetime],
    tickers: List[str],
) -> List[Dict[str, str]]:
    """Generate realistic financial headlines with dates and ticker associations."""
    headlines = []

    for day in business_days:
        # Not every day has headlines for every ticker
        # Generate 0-3 headlines per day
        n_headlines = int(np.random.choice([0, 1, 1, 1, 2, 2, 3]))

        for _ in range(n_headlines):
            # Pick category with weighted probability
            category = np.random.choice(
                ["positive", "negative", "neutral", "market"],
                p=[0.25, 0.25, 0.30, 0.20],
            )

            template = random.choice(HEADLINE_TEMPLATES[category])

            if "{ticker}" in template:
                ticker = random.choice(tickers)
                headline_text = template.format(
                    ticker=ticker,
                    q=random.choice([1, 2, 3, 4]),
                )
            else:
                ticker = ""  # Market-wide headline
                headline_text = template

            headlines.append({
                "date": day.strftime("%Y-%m-%d"),
                "headline": headline_text,
                "ticker": ticker,
                "source": random.choice(["Yahoo Finance", "Reuters", "Bloomberg", "CNBC", "MarketWatch"]),
            })

    return headlines


def compute_sentiment_scores(
    headlines: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Generate pre-computed sentiment scores that mimic FinBERT output.

    Scores are correlated with headline category for realism.
    """
    sentiment_data = []

    for item in headlines:
        headline = item["headline"].lower()

        # Determine likely sentiment from keywords
        positive_words = ["beats", "surge", "record", "upgrade", "bullish", "raises",
                          "expands", "increase", "strong", "innovative", "rally", "high",
                          "secures", "accelerates", "outperform", "optimism"]
        negative_words = ["misses", "drops", "concerns", "downgrade", "scrutiny",
                          "disruptions", "lowers", "layoffs", "loses", "breach",
                          "pressure", "lawsuit", "selloff", "departure", "tumble",
                          "recession", "fears", "contraction", "inversion"]

        pos_count = sum(1 for w in positive_words if w in headline)
        neg_count = sum(1 for w in negative_words if w in headline)

        if pos_count > neg_count:
            base_positive = np.random.uniform(0.55, 0.92)
            base_negative = np.random.uniform(0.02, 0.15)
            base_neutral = 1.0 - base_positive - base_negative
            label = "positive"
        elif neg_count > pos_count:
            base_negative = np.random.uniform(0.55, 0.92)
            base_positive = np.random.uniform(0.02, 0.15)
            base_neutral = 1.0 - base_positive - base_negative
            label = "negative"
        else:
            base_neutral = np.random.uniform(0.45, 0.75)
            base_positive = np.random.uniform(0.05, 0.30)
            base_negative = 1.0 - base_neutral - base_positive
            label = "neutral"

        # Ensure all scores are positive and sum to 1
        scores = np.array([max(0.01, base_positive), max(0.01, base_negative), max(0.01, base_neutral)])
        scores = scores / scores.sum()

        sentiment_data.append({
            "date": item["date"],
            "headline": item["headline"],
            "ticker": item["ticker"],
            "source": item["source"],
            "positive": round(float(scores[0]), 4),
            "negative": round(float(scores[1]), 4),
            "neutral": round(float(scores[2]), 4),
            "label": label,
            "compound_score": round(float(scores[0] - scores[1]), 4),
        })

    return sentiment_data


def main() -> None:
    """Generate all sample data files."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = os.path.dirname(os.path.abspath(__file__))

    start_date = "2021-01-04"
    end_date = "2023-12-29"
    tickers = list(TICKER_PARAMS.keys())

    logger.info("Generating sample data from %s to %s", start_date, end_date)
    business_days = generate_business_days(start_date, end_date)
    logger.info("Generated %d business days", len(business_days))

    # --- Price Data ---
    logger.info("Generating price data...")
    price_file = os.path.join(output_dir, "sample_prices.csv")
    with open(price_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])
        writer.writeheader()
        for ticker in tickers:
            params = TICKER_PARAMS[ticker]
            series = generate_price_series(
                business_days,
                params["start_price"],
                params["annual_drift"],
                params["annual_vol"],
            )
            for row in series:
                row["Ticker"] = ticker
                writer.writerow(row)

    logger.info("Wrote %d price rows to %s", len(business_days) * len(tickers), price_file)

    # --- Headlines ---
    logger.info("Generating headlines...")
    headlines = generate_headlines(business_days, tickers)
    headline_file = os.path.join(output_dir, "sample_headlines.csv")
    with open(headline_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "headline", "ticker", "source"])
        writer.writeheader()
        for row in headlines:
            writer.writerow({k: row[k] for k in ["date", "headline", "ticker", "source"]})

    logger.info("Wrote %d headlines to %s", len(headlines), headline_file)

    # --- Sentiment Scores ---
    logger.info("Generating sentiment scores...")
    sentiment = compute_sentiment_scores(headlines)
    sentiment_file = os.path.join(output_dir, "sample_sentiment.csv")
    with open(sentiment_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "headline", "ticker", "source",
                         "positive", "negative", "neutral", "label", "compound_score"],
        )
        writer.writeheader()
        for row in sentiment:
            writer.writerow(row)

    logger.info("Wrote %d sentiment rows to %s", len(sentiment), sentiment_file)
    logger.info("Sample data generated successfully.")


if __name__ == "__main__":
    main()
