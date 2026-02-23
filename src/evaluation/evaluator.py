"""
Model evaluation and comparison module.

Generates comprehensive evaluation metrics and visualizations including
ROC curves, precision-recall curves, confusion matrices, and model
comparison tables.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Consistent styling
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2196F3", "#FF5722", "#4CAF50"]


class ModelEvaluator:
    """Evaluates and compares multiple models on the same test set.

    Generates evaluation plots (ROC, PR, confusion matrix) for each
    model and a comparison table summarizing all metrics.

    Attributes:
        figures_dir: Directory to save evaluation plots.
        results: Collected evaluation results for comparison.
    """

    def __init__(self, figures_dir: str = "outputs/figures") -> None:
        """Initialize the model evaluator.

        Args:
            figures_dir: Directory path to save generated figures.
        """
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate a single model and store results.

        Args:
            model_name: Human-readable model name.
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            y_proba: Predicted probabilities for the positive class.

        Returns:
            Dictionary of computed metrics.
        """
        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
        }

        self.results.append(metrics)

        logger.info(
            "%s -> Acc: %.3f | Prec: %.3f | Rec: %.3f | F1: %.3f | AUC: %.3f",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )

        return metrics

    def plot_roc_curves(
        self,
        model_results: List[Tuple[str, np.ndarray, np.ndarray]],
    ) -> str:
        """Plot ROC curves for all models on a single figure.

        Args:
            model_results: List of (model_name, y_true, y_proba) tuples.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, (name, y_true, y_proba) in enumerate(model_results):
            RocCurveDisplay.from_predictions(
                y_true, y_proba,
                name=name,
                ax=ax,
                color=COLORS[i % len(COLORS)],
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
        ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = self.figures_dir / "roc_curves.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved ROC curves to %s", output_path)
        return str(output_path)

    def plot_precision_recall_curves(
        self,
        model_results: List[Tuple[str, np.ndarray, np.ndarray]],
    ) -> str:
        """Plot Precision-Recall curves for all models.

        Args:
            model_results: List of (model_name, y_true, y_proba) tuples.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, (name, y_true, y_proba) in enumerate(model_results):
            PrecisionRecallDisplay.from_predictions(
                y_true, y_proba,
                name=name,
                ax=ax,
                color=COLORS[i % len(COLORS)],
            )

        # Add baseline (class prevalence)
        prevalence = model_results[0][1].mean() if model_results else 0.5
        ax.axhline(y=prevalence, color="k", linestyle="--", alpha=0.5,
                    label=f"Baseline ({prevalence:.2f})")

        ax.set_title("Precision-Recall Curves - Model Comparison",
                      fontsize=14, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = self.figures_dir / "precision_recall_curves.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved precision-recall curves to %s", output_path)
        return str(output_path)

    def plot_confusion_matrix(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> str:
        """Plot confusion matrix for a single model.

        Args:
            model_name: Model name for the title.
            y_true: True binary labels.
            y_pred: Predicted binary labels.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            ax=ax,
            cmap="Blues",
            display_labels=["No Drop", "Drop >2%"],
        )

        ax.set_title(f"Confusion Matrix - {model_name}",
                      fontsize=14, fontweight="bold")

        safe_name = model_name.lower().replace(" ", "_")
        output_path = self.figures_dir / f"confusion_matrix_{safe_name}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved confusion matrix for %s to %s", model_name, output_path)
        return str(output_path)

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        model_name: str,
        top_n: int = 15,
    ) -> str:
        """Plot feature importance bar chart.

        Args:
            feature_importance: Dictionary mapping feature names to scores.
            model_name: Model name for the title.
            top_n: Number of top features to display.

        Returns:
            Path to the saved figure.
        """
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        names = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(names)), values, color="#2196F3", alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(f"Top {top_n} Features - {model_name}",
                      fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        safe_name = model_name.lower().replace(" ", "_")
        output_path = self.figures_dir / f"feature_importance_{safe_name}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved feature importance for %s to %s", model_name, output_path)
        return str(output_path)

    def comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of all evaluated models.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        if not self.results:
            logger.warning("No evaluation results to compare")
            return pd.DataFrame()

        df = pd.DataFrame(self.results)
        df = df.set_index("model")
        df = df.round(4)

        logger.info("\n=== Model Comparison ===\n%s\n", df.to_string())
        return df

    def print_summary(self) -> str:
        """Print a summary identifying the best model.

        Returns:
            Summary string describing the best model.
        """
        if not self.results:
            return "No models evaluated."

        df = pd.DataFrame(self.results)
        best_idx = df["f1"].idxmax()
        best_model = df.loc[best_idx]

        summary = (
            f"Based on evaluation, {best_model['model']} performs best "
            f"with F1 of {best_model['f1']:.3f} on the test set "
            f"(ROC-AUC: {best_model['roc_auc']:.3f}, "
            f"Precision: {best_model['precision']:.3f}, "
            f"Recall: {best_model['recall']:.3f})."
        )

        logger.info("\n%s", summary)
        return summary

    def save_comparison(self) -> str:
        """Save the comparison table as CSV.

        Returns:
            Path to the saved CSV file.
        """
        df = self.comparison_table()
        output_path = self.figures_dir / "model_comparison.csv"
        df.to_csv(output_path)
        logger.info("Saved comparison table to %s", output_path)
        return str(output_path)

    def generate_all_plots(
        self,
        model_predictions: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate all evaluation plots for a list of model results.

        Args:
            model_predictions: List of dicts, each with keys:
                - 'name': str, model name
                - 'y_true': array of true labels
                - 'y_pred': array of predicted labels
                - 'y_proba': array of positive class probabilities
                - 'feature_importance': optional dict of feature importances

        Returns:
            List of paths to generated figures.
        """
        generated_files: List[str] = []

        # Evaluate each model
        for mp in model_predictions:
            self.evaluate_model(
                mp["name"], mp["y_true"], mp["y_pred"], mp["y_proba"]
            )

        # ROC curves
        roc_data = [(mp["name"], mp["y_true"], mp["y_proba"]) for mp in model_predictions]
        generated_files.append(self.plot_roc_curves(roc_data))

        # PR curves
        generated_files.append(self.plot_precision_recall_curves(roc_data))

        # Confusion matrices
        for mp in model_predictions:
            generated_files.append(
                self.plot_confusion_matrix(mp["name"], mp["y_true"], mp["y_pred"])
            )

        # Feature importance (if available)
        for mp in model_predictions:
            if "feature_importance" in mp and mp["feature_importance"]:
                generated_files.append(
                    self.plot_feature_importance(mp["feature_importance"], mp["name"])
                )

        # Comparison table
        generated_files.append(self.save_comparison())

        # Summary
        summary = self.print_summary()

        return generated_files
