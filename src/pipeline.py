"""
Main pipeline orchestrator for the Portfolio Risk Monitor.

Entry point for training, evaluation, and prediction. Supports three modes:
- sample: Run end-to-end with synthetic sample data (no API keys needed)
- train: Collect real data and train all models
- predict: Load trained models and generate predictions

Usage:
    python -m src.pipeline --mode sample
    python -m src.pipeline --mode train
    python -m src.pipeline --mode predict
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collector import MarketDataCollector, NewsCollector, SampleDataGenerator
from src.data.preprocessor import DataPreprocessor
from src.features.technical import TechnicalFeatureEngineer
from src.features.sentiment import SentimentFeatureEngineer
from src.models.baseline import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.lstm import LSTMModel
from src.evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing all configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration from %s", config_file)
    return config


def setup_logging(config: Dict[str, Any]) -> None:
    """Configure logging based on config settings.

    Args:
        config: Configuration dictionary with logging settings.
    """
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    fmt = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.basicConfig(level=level, format=fmt, force=True)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def run_sample_pipeline(config: Dict[str, Any]) -> None:
    """Run the full pipeline using sample data.

    This mode requires no API keys or internet access. Uses synthetic
    data and pre-computed sentiment scores for demonstration.

    Args:
        config: Configuration dictionary.
    """
    logger.info("=" * 60)
    logger.info("PORTFOLIO RISK MONITOR - Sample Mode")
    logger.info("=" * 60)

    # --- Step 1: Generate or load sample data ---
    logger.info("\n--- Step 1: Loading Sample Data ---")
    sample_gen = SampleDataGenerator(
        sample_path=str(PROJECT_ROOT / config["data"]["paths"]["sample"]),
        start_date="2021-01-04",
        end_date="2023-12-29",
    )
    prices_df, headlines_df, sentiment_df = sample_gen.load_or_generate()
    logger.info("Loaded %d price rows, %d headlines, %d sentiment scores",
                len(prices_df), len(headlines_df), len(sentiment_df))

    # --- Step 2: Feature engineering ---
    logger.info("\n--- Step 2: Feature Engineering ---")
    tech_config = config["features"]["technical"]
    tech_engineer = TechnicalFeatureEngineer(
        rsi_period=tech_config["rsi_period"],
        macd_fast=tech_config["macd_fast"],
        macd_slow=tech_config["macd_slow"],
        macd_signal=tech_config["macd_signal"],
        bollinger_window=tech_config["bollinger_window"],
        bollinger_std=tech_config["bollinger_std"],
        ma_windows=tech_config["ma_windows"],
        volume_avg_window=tech_config["volume_avg_window"],
        volatility_window=tech_config["volatility_window"],
        roc_period=tech_config.get("roc_period", 10),
    )
    technical_features = tech_engineer.compute(prices_df)

    # Sentiment features (using pre-computed scores)
    sent_engineer = SentimentFeatureEngineer(use_precomputed=True)
    scored_headlines = sent_engineer.score_headlines(sentiment_df)
    tickers = prices_df["Ticker"].unique().tolist()
    sentiment_features = sent_engineer.aggregate_daily(scored_headlines, tickers=tickers)

    # --- Step 3: Preprocessing ---
    logger.info("\n--- Step 3: Preprocessing & Target Creation ---")
    preprocessor = DataPreprocessor(
        target_threshold=config["target"]["threshold"],
        target_horizon=config["target"]["horizon"],
        train_end=config["split"]["train_end"],
        val_end=config["split"]["val_end"],
        processed_path=str(PROJECT_ROOT / config["data"]["paths"]["processed"]),
    )

    prices_with_target = preprocessor.create_target(prices_df)
    merged = preprocessor.merge_features(
        prices_with_target, technical_features, sentiment_features
    )

    feature_cols = preprocessor.get_feature_columns(merged)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # --- Step 4: Train/Val/Test split ---
    logger.info("\n--- Step 4: Time-Based Split ---")
    train_df, val_df, test_df = preprocessor.split_data(merged)

    X_train, y_train = preprocessor.prepare_xy(train_df, feature_cols)
    X_val, y_val = preprocessor.prepare_xy(val_df, feature_cols)
    X_test, y_test = preprocessor.prepare_xy(test_df, feature_cols)

    logger.info("Train: %s, Val: %s, Test: %s",
                X_train.shape, X_val.shape, X_test.shape)

    # --- Step 5: Train models ---
    logger.info("\n--- Step 5: Training Models ---")

    # Logistic Regression
    lr_config = config["models"]["logistic_regression"]
    lr_model = LogisticRegressionModel(
        C=lr_config["C"],
        max_iter=lr_config["max_iter"],
        solver=lr_config["solver"],
    )
    lr_model.fit(X_train, y_train)

    # Random Forest
    rf_config = config["models"]["random_forest"]
    rf_model = RandomForestModel(
        n_estimators_range=rf_config["n_estimators_range"],
        max_depth_range=rf_config["max_depth_range"],
        min_samples_split_range=rf_config["min_samples_split_range"],
        min_samples_leaf_range=rf_config["min_samples_leaf_range"],
        n_cv_splits=rf_config["n_cv_splits"],
        random_state=rf_config["random_state"],
    )
    rf_model.fit(X_train, y_train, tune=True)

    # LSTM
    lstm_config = config["models"]["lstm"]
    lstm_model = LSTMModel(
        sequence_length=lstm_config["sequence_length"],
        hidden_size=lstm_config["hidden_size"],
        num_layers=lstm_config["num_layers"],
        dropout=lstm_config["dropout"],
        learning_rate=lstm_config["learning_rate"],
        batch_size=lstm_config["batch_size"],
        epochs=lstm_config["epochs"],
        patience=lstm_config["patience"],
        random_state=lstm_config["random_state"],
    )
    lstm_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # --- Step 6: Evaluate ---
    logger.info("\n--- Step 6: Evaluation ---")
    evaluator = ModelEvaluator(
        figures_dir=str(PROJECT_ROOT / config["output"]["figures_dir"])
    )

    # Get predictions for each model
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    lstm_proba_full = lstm_model.predict_proba(X_test)
    lstm_pred = (lstm_proba_full[:, 1] >= 0.5).astype(int)
    lstm_proba = lstm_proba_full[:, 1]

    # Align LSTM predictions (sequences drop first N items)
    y_test_lstm = y_test[lstm_model.sequence_length:]
    min_len = min(len(y_test_lstm), len(lstm_pred))
    y_test_lstm = y_test_lstm[:min_len]
    lstm_pred = lstm_pred[:min_len]
    lstm_proba = lstm_proba[:min_len]

    # Build model predictions list for evaluator
    model_predictions = [
        {
            "name": "LogisticRegression",
            "y_true": y_test,
            "y_pred": lr_pred,
            "y_proba": lr_proba,
            "feature_importance": lr_model.get_feature_importance(feature_cols),
        },
        {
            "name": "RandomForest",
            "y_true": y_test,
            "y_pred": rf_pred,
            "y_proba": rf_proba,
            "feature_importance": rf_model.get_feature_importance(feature_cols),
        },
        {
            "name": "LSTM",
            "y_true": y_test_lstm,
            "y_pred": lstm_pred,
            "y_proba": lstm_proba,
        },
    ]

    generated_files = evaluator.generate_all_plots(model_predictions)

    # --- Step 7: Save models ---
    logger.info("\n--- Step 7: Saving Models ---")
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    lr_model.save(str(models_dir / "logistic_regression.pkl"))
    rf_model.save(str(models_dir / "random_forest.pkl"))
    lstm_model.save(str(models_dir / "lstm.pt"))

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    summary = evaluator.print_summary()
    logger.info("Generated files: %s", generated_files)
    logger.info("=" * 60)


def run_train_pipeline(config: Dict[str, Any]) -> None:
    """Run the full training pipeline with real data.

    Collects data from yfinance and RSS feeds, computes features,
    trains all models, and evaluates them.

    Args:
        config: Configuration dictionary.
    """
    logger.info("=" * 60)
    logger.info("PORTFOLIO RISK MONITOR - Training Mode")
    logger.info("=" * 60)

    # --- Step 1: Collect data ---
    logger.info("\n--- Step 1: Data Collection ---")
    market_collector = MarketDataCollector(
        tickers=config["data"]["tickers"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        raw_path=str(PROJECT_ROOT / config["data"]["paths"]["raw"]),
    )
    prices_df = market_collector.collect()

    news_collector = NewsCollector(
        rss_feeds=config["data"]["news"]["rss_feeds"],
        max_headlines_per_day=config["data"]["news"]["max_headlines_per_day"],
        raw_path=str(PROJECT_ROOT / config["data"]["paths"]["raw"]),
    )
    headlines_df = news_collector.collect()

    # --- Step 2: Feature engineering ---
    logger.info("\n--- Step 2: Feature Engineering ---")
    tech_config = config["features"]["technical"]
    tech_engineer = TechnicalFeatureEngineer(
        rsi_period=tech_config["rsi_period"],
        macd_fast=tech_config["macd_fast"],
        macd_slow=tech_config["macd_slow"],
        macd_signal=tech_config["macd_signal"],
        bollinger_window=tech_config["bollinger_window"],
        bollinger_std=tech_config["bollinger_std"],
        ma_windows=tech_config["ma_windows"],
        volume_avg_window=tech_config["volume_avg_window"],
        volatility_window=tech_config["volatility_window"],
        roc_period=tech_config.get("roc_period", 10),
    )
    technical_features = tech_engineer.compute(prices_df)

    # Sentiment: use FinBERT if headlines are available
    sent_config = config["features"]["sentiment"]
    if not headlines_df.empty:
        sent_engineer = SentimentFeatureEngineer(
            model_name=sent_config["model_name"],
            max_length=sent_config["max_length"],
            batch_size=sent_config["batch_size"],
            use_precomputed=False,
        )
        scored_headlines = sent_engineer.score_headlines(headlines_df)
    else:
        sent_engineer = SentimentFeatureEngineer(use_precomputed=True)
        scored_headlines = pd.DataFrame(
            columns=["date", "headline", "ticker", "compound_score"]
        )

    tickers = prices_df["Ticker"].unique().tolist()
    sentiment_features = sent_engineer.aggregate_daily(scored_headlines, tickers=tickers)

    # --- Steps 3-7: Same as sample pipeline ---
    logger.info("\n--- Step 3: Preprocessing & Target Creation ---")
    preprocessor = DataPreprocessor(
        target_threshold=config["target"]["threshold"],
        target_horizon=config["target"]["horizon"],
        train_end=config["split"]["train_end"],
        val_end=config["split"]["val_end"],
        processed_path=str(PROJECT_ROOT / config["data"]["paths"]["processed"]),
    )

    prices_with_target = preprocessor.create_target(prices_df)
    merged = preprocessor.merge_features(
        prices_with_target, technical_features, sentiment_features
    )

    feature_cols = preprocessor.get_feature_columns(merged)

    logger.info("\n--- Step 4: Time-Based Split ---")
    train_df, val_df, test_df = preprocessor.split_data(merged)
    X_train, y_train = preprocessor.prepare_xy(train_df, feature_cols)
    X_val, y_val = preprocessor.prepare_xy(val_df, feature_cols)
    X_test, y_test = preprocessor.prepare_xy(test_df, feature_cols)

    logger.info("\n--- Step 5: Training Models ---")
    lr_config = config["models"]["logistic_regression"]
    lr_model = LogisticRegressionModel(
        C=lr_config["C"],
        max_iter=lr_config["max_iter"],
        solver=lr_config["solver"],
    )
    lr_model.fit(X_train, y_train)

    rf_config = config["models"]["random_forest"]
    rf_model = RandomForestModel(
        n_estimators_range=rf_config["n_estimators_range"],
        max_depth_range=rf_config["max_depth_range"],
        min_samples_split_range=rf_config["min_samples_split_range"],
        min_samples_leaf_range=rf_config["min_samples_leaf_range"],
        n_cv_splits=rf_config["n_cv_splits"],
        random_state=rf_config["random_state"],
    )
    rf_model.fit(X_train, y_train, tune=True)

    lstm_config = config["models"]["lstm"]
    lstm_model = LSTMModel(
        sequence_length=lstm_config["sequence_length"],
        hidden_size=lstm_config["hidden_size"],
        num_layers=lstm_config["num_layers"],
        dropout=lstm_config["dropout"],
        learning_rate=lstm_config["learning_rate"],
        batch_size=lstm_config["batch_size"],
        epochs=lstm_config["epochs"],
        patience=lstm_config["patience"],
        random_state=lstm_config["random_state"],
    )
    lstm_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    logger.info("\n--- Step 6: Evaluation ---")
    evaluator = ModelEvaluator(
        figures_dir=str(PROJECT_ROOT / config["output"]["figures_dir"])
    )

    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    lstm_proba_full = lstm_model.predict_proba(X_test)
    lstm_pred = (lstm_proba_full[:, 1] >= 0.5).astype(int)
    lstm_proba = lstm_proba_full[:, 1]

    y_test_lstm = y_test[lstm_model.sequence_length:]
    min_len = min(len(y_test_lstm), len(lstm_pred))
    y_test_lstm = y_test_lstm[:min_len]
    lstm_pred = lstm_pred[:min_len]
    lstm_proba = lstm_proba[:min_len]

    model_predictions = [
        {
            "name": "LogisticRegression",
            "y_true": y_test,
            "y_pred": lr_pred,
            "y_proba": lr_proba,
            "feature_importance": lr_model.get_feature_importance(feature_cols),
        },
        {
            "name": "RandomForest",
            "y_true": y_test,
            "y_pred": rf_pred,
            "y_proba": rf_proba,
            "feature_importance": rf_model.get_feature_importance(feature_cols),
        },
        {
            "name": "LSTM",
            "y_true": y_test_lstm,
            "y_pred": lstm_pred,
            "y_proba": lstm_proba,
        },
    ]

    evaluator.generate_all_plots(model_predictions)

    logger.info("\n--- Step 7: Saving Models ---")
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    lr_model.save(str(models_dir / "logistic_regression.pkl"))
    rf_model.save(str(models_dir / "random_forest.pkl"))
    lstm_model.save(str(models_dir / "lstm.pt"))

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)
    evaluator.print_summary()


def run_predict_pipeline(config: Dict[str, Any]) -> None:
    """Load trained models and generate predictions on new data.

    Args:
        config: Configuration dictionary.
    """
    logger.info("=" * 60)
    logger.info("PORTFOLIO RISK MONITOR - Prediction Mode")
    logger.info("=" * 60)

    models_dir = PROJECT_ROOT / config["output"]["models_dir"]

    # Load models
    lr_model = LogisticRegressionModel()
    lr_path = models_dir / "logistic_regression.pkl"
    if not lr_path.exists():
        logger.error("Model file not found: %s. Run training first.", lr_path)
        return
    lr_model.load(str(lr_path))

    rf_model = RandomForestModel()
    rf_path = models_dir / "random_forest.pkl"
    if not rf_path.exists():
        logger.error("Model file not found: %s. Run training first.", rf_path)
        return
    rf_model.load(str(rf_path))

    logger.info("Models loaded. Ready for prediction.")
    logger.info("To generate predictions, provide new data through the API or extend this method.")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Portfolio Risk Monitor - ML Pipeline for Market Downturn Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline --mode sample    # Run with sample data (no API keys needed)
  python -m src.pipeline --mode train     # Train with real market data
  python -m src.pipeline --mode predict   # Generate predictions with trained models
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "sample"],
        default="sample",
        help="Pipeline mode: train, predict, or sample (default: sample)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(config)

    logger.info("Running pipeline in '%s' mode", args.mode)

    if args.mode == "sample":
        run_sample_pipeline(config)
    elif args.mode == "train":
        run_train_pipeline(config)
    elif args.mode == "predict":
        run_predict_pipeline(config)
    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
