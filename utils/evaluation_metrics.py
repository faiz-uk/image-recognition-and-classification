"""
Comprehensive Evaluation Metrics Module
Implements complete evaluation strategy: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation following the specified strategy:
    - Accuracy: Overall classification performance
    - Precision, Recall, F1-Score: For multi-class and imbalanced tasks
    - Confusion Matrix: Visual inspection of class-specific performance
    - Loss Curves & Accuracy Curves: Learning dynamics tracking
    - Model Comparison: Performance comparison across models
    """

    def __init__(
        self, class_names: Optional[List[str]] = None, task_type: str = "multiclass"
    ):
        """
        Initialize comprehensive evaluator

        Args:
            class_names: List of class names for labeling
            task_type: 'binary', 'multiclass', or 'multilabel'
        """
        self.class_names = class_names
        self.task_type = task_type
        self.results_history = []

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        model_name: str = "Model",
        dataset_name: str = "Dataset",
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation following the evaluation strategy

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC curves)
            model_name: Name of the model being evaluated
            dataset_name: Name of the dataset

        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"Evaluating {model_name} on {dataset_name}")

        # Handle label format conversion
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            # One-hot encoded to label indices
            y_true_labels = np.argmax(y_true, axis=1).astype(int)
        else:
            y_true_labels = y_true.flatten().astype(int)

        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_labels = np.argmax(y_pred, axis=1).astype(int)
        else:
            y_pred_labels = y_pred.flatten().astype(int)

        # 1. ACCURACY: Overall classification performance
        accuracy = accuracy_score(y_true_labels, y_pred_labels)

        # 2. PRECISION, RECALL, F1-SCORE: Multi-class and imbalanced tasks
        if self.task_type == "binary":
            precision = precision_score(y_true_labels, y_pred_labels, average="binary")
            recall = recall_score(y_true_labels, y_pred_labels, average="binary")
            f1 = f1_score(y_true_labels, y_pred_labels, average="binary")
        else:
            precision = precision_score(
                y_true_labels, y_pred_labels, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_true_labels, y_pred_labels, average="weighted", zero_division=0
            )
            f1 = f1_score(
                y_true_labels, y_pred_labels, average="weighted", zero_division=0
            )

        # Per-class metrics
        precision_per_class = precision_score(
            y_true_labels, y_pred_labels, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true_labels, y_pred_labels, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true_labels, y_pred_labels, average=None, zero_division=0
        )

        # 3. CONFUSION MATRIX: Class-specific performance
        cm = confusion_matrix(y_true_labels, y_pred_labels)

        # Classification report
        class_report = classification_report(
            y_true_labels,
            y_pred_labels,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                if self.task_type == "binary":
                    # For binary classification, handle both 1D and 2D probability arrays
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                        # If 2D with 2 columns, use positive class (index 1)
                        proba_positive = y_pred_proba[:, 1]
                    else:
                        # If 1D or single column, use as-is (should be probability of positive class)
                        proba_positive = y_pred_proba.flatten()
                    roc_auc = roc_auc_score(y_true_labels, proba_positive)
                else:
                    # Multi-class ROC AUC
                    y_true_onehot = (
                        y_true
                        if y_true.ndim > 1
                        else tf.keras.utils.to_categorical(y_true_labels)
                    )
                    roc_auc = roc_auc_score(
                        y_true_onehot,
                        y_pred_proba,
                        multi_class="ovr",
                        average="weighted",
                    )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")

        # Compile comprehensive results
        results = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "task_type": self.task_type,
            # Overall metrics
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            # Per-class metrics
            "precision_per_class": (
                precision_per_class.tolist()
                if hasattr(precision_per_class, "tolist")
                else precision_per_class
            ),
            "recall_per_class": (
                recall_per_class.tolist()
                if hasattr(recall_per_class, "tolist")
                else recall_per_class
            ),
            "f1_per_class": (
                f1_per_class.tolist()
                if hasattr(f1_per_class, "tolist")
                else f1_per_class
            ),
            # Confusion matrix
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            # Class distribution (fix dtype issue)
            "class_distribution": {
                "true": np.bincount(y_true_labels.astype(int)).tolist(),
                "predicted": np.bincount(y_pred_labels.astype(int)).tolist(),
            },
        }

        # Store in history for model comparison
        self.results_history.append(results)

        logger.info(f"Evaluation completed: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        return results

    def plot_confusion_matrix(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot confusion matrix with visual inspection capabilities

        Args:
            results: Results from evaluate_model
            save_path: Path to save the plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        cm = np.array(results["confusion_matrix"])
        model_name = results["model_name"]
        dataset_name = results["dataset_name"]

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names or range(cm.shape[1]),
            yticklabels=self.class_names or range(cm.shape[0]),
            ax=ax,
        )

        ax.set_title(
            f"Confusion Matrix: {model_name} on {dataset_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted Labels", fontsize=12)
        ax.set_ylabel("True Labels", fontsize=12)

        # Add accuracy info
        accuracy = results["accuracy"]
        ax.text(
            0.02,
            0.98,
            f"Accuracy: {accuracy:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_classification_metrics(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Plot comprehensive classification metrics

        Args:
            results: Results from evaluate_model
            save_path: Path to save the plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        model_name = results["model_name"]
        dataset_name = results["dataset_name"]

        # 1. Overall metrics bar chart
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [
            results["accuracy"],
            results["precision"],
            results["recall"],
            results["f1_score"],
        ]

        bars = ax1.bar(
            metrics, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        )
        ax1.set_ylim(0, 1)
        ax1.set_title("Overall Performance Metrics", fontweight="bold")
        ax1.set_ylabel("Score")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. Per-class F1 scores
        if len(results["f1_per_class"]) > 1:
            class_labels = self.class_names or [
                f"Class {i}" for i in range(len(results["f1_per_class"]))
            ]
            ax2.bar(
                range(len(results["f1_per_class"])),
                results["f1_per_class"],
                color="skyblue",
            )
            ax2.set_xlabel("Classes")
            ax2.set_ylabel("F1-Score")
            ax2.set_title("Per-Class F1 Scores", fontweight="bold")
            ax2.set_xticks(range(len(class_labels)))
            ax2.set_xticklabels(class_labels, rotation=45, ha="right")
            ax2.set_ylim(0, 1)
        else:
            ax2.text(
                0.5,
                0.5,
                "Binary Classification\nSee Overall Metrics",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )
            ax2.set_title("Per-Class Analysis", fontweight="bold")

        # 3. Class distribution comparison
        true_dist = results["class_distribution"]["true"]
        pred_dist = results["class_distribution"]["predicted"]

        x = np.arange(len(true_dist))
        width = 0.35

        ax3.bar(
            x - width / 2, true_dist, width, label="True", alpha=0.8, color="lightcoral"
        )
        ax3.bar(
            x + width / 2,
            pred_dist,
            width,
            label="Predicted",
            alpha=0.8,
            color="lightblue",
        )

        ax3.set_xlabel("Classes")
        ax3.set_ylabel("Count")
        ax3.set_title("Class Distribution Comparison", fontweight="bold")
        ax3.legend()
        ax3.set_xticks(x)
        if self.class_names:
            ax3.set_xticklabels(
                self.class_names[: len(true_dist)], rotation=45, ha="right"
            )

        # 4. Performance summary text
        ax4.axis("off")
        summary_text = f"""
{model_name} on {dataset_name}
═══════════════════════════════════

Accuracy: {results['accuracy']:.3f}
Precision: {results['precision']:.3f}
Recall: {results['recall']:.3f}
F1-Score: {results['f1_score']:.3f}
{"ROC AUC: " + f"{results['roc_auc']:.3f}" if results['roc_auc'] else ""}

Task Type: {results['task_type'].title()}
Classes: {len(true_dist)}
Total Samples: {sum(true_dist)}
        """.strip()

        ax4.text(
            0.05,
            0.95,
            summary_text,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3),
        )

        plt.suptitle(
            f"Classification Analysis: {model_name}", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Classification metrics plot saved to {save_path}")

        return fig

    def compare_models(self, save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Model comparison following the evaluation strategy

        Args:
            save_path: Path to save comparison results

        Returns:
            DataFrame with model comparison results
        """
        if not self.results_history:
            logger.warning("No evaluation results available for comparison")
            return pd.DataFrame()

        # Create comparison DataFrame
        comparison_data = []
        for result in self.results_history:
            comparison_data.append(
                {
                    "Model": result["model_name"],
                    "Dataset": result["dataset_name"],
                    "Accuracy": result["accuracy"],
                    "Precision": result["precision"],
                    "Recall": result["recall"],
                    "F1-Score": result["f1_score"],
                    "ROC AUC": result["roc_auc"] if result["roc_auc"] else "N/A",
                }
            )

        df = pd.DataFrame(comparison_data)

        # Sort by F1-Score (comprehensive metric)
        df = df.sort_values("F1-Score", ascending=False)

        if save_path:
            df.to_csv(save_path, index=False)
            logger.info(f"Model comparison saved to {save_path}")

        logger.info("Model Comparison Results:")
        logger.info("\n" + df.to_string(index=False))

        return df

    def plot_learning_curves(
        self,
        history: Dict[str, List[float]],
        model_name: str,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """
        Plot loss curves and accuracy curves for learning dynamics

        Args:
            history: Training history with loss and accuracy curves
            model_name: Name of the model
            save_path: Path to save the plot
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        epochs = range(1, len(history["loss"]) + 1)

        # 1. Loss curves
        ax1.plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
        if "val_loss" in history:
            ax1.plot(
                epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2
            )
        ax1.set_title("Model Loss", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy curves
        ax2.plot(
            epochs, history["accuracy"], "b-", label="Training Accuracy", linewidth=2
        )
        if "val_accuracy" in history:
            ax2.plot(
                epochs,
                history["val_accuracy"],
                "r-",
                label="Validation Accuracy",
                linewidth=2,
            )
        ax2.set_title("Model Accuracy", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Learning rate (if available)
        if "lr" in history:
            ax3.plot(epochs, history["lr"], "g-", linewidth=2)
            ax3.set_title("Learning Rate", fontweight="bold")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_yscale("log")
            ax3.grid(True, alpha=0.3)
        else:
            # Training dynamics summary
            final_train_acc = history["accuracy"][-1]
            final_val_acc = (
                history.get("val_accuracy", [0])[-1]
                if history.get("val_accuracy")
                else 0
            )
            final_train_loss = history["loss"][-1]
            final_val_loss = (
                history.get("val_loss", [0])[-1] if history.get("val_loss") else 0
            )

            summary_text = f"""
Learning Dynamics Summary
════════════════════════

Final Training Accuracy: {final_train_acc:.3f}
Final Validation Accuracy: {final_val_acc:.3f}
Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}

Epochs Trained: {len(epochs)}
Convergence: {"Good" if abs(final_train_acc - final_val_acc) < 0.1 else "Check for overfitting"}
            """.strip()

            ax3.text(
                0.05,
                0.95,
                summary_text,
                transform=ax3.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3),
            )
            ax3.set_title("Training Summary", fontweight="bold")
            ax3.axis("off")

        plt.suptitle(f"Learning Dynamics: {model_name}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Learning curves saved to {save_path}")

        return fig


# Utility functions for easy integration
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    model_name: str = "Model",
    dataset_name: str = "Dataset",
    task_type: str = "multiclass",
) -> Dict[str, Any]:
    """
    Quick evaluation function following the comprehensive strategy

    Returns:
        Complete evaluation results
    """
    evaluator = ComprehensiveEvaluator(class_names=class_names, task_type=task_type)
    return evaluator.evaluate_model(
        y_true, y_pred, y_pred_proba, model_name, dataset_name
    )


def create_evaluation_report(
    results: Dict[str, Any], history: Dict[str, List[float]], save_dir: Path, timestamp: str = None
) -> Dict[str, str]:
    """
    Create comprehensive evaluation report with all visualizations

    Args:
        results: Evaluation results
        history: Training history
        save_dir: Directory to save reports
        timestamp: Optional timestamp for consistent naming

    Returns:
        Dictionary with paths to saved files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_name = results["model_name"]
    dataset_name = results["dataset_name"]

    evaluator = ComprehensiveEvaluator(task_type=results["task_type"])
    saved_files = {}

    # Create consistent base name with timestamp if provided
    if timestamp:
        base_name = f"{model_name}_{dataset_name}_{timestamp}"
    else:
        base_name = f"{model_name}_{dataset_name}"

    # 1. Confusion Matrix
    cm_path = save_dir / f"{base_name}_confusion_matrix.png"
    evaluator.plot_confusion_matrix(results, save_path=cm_path)
    saved_files["confusion_matrix"] = str(cm_path)

    # 2. Classification Metrics
    metrics_path = save_dir / f"{base_name}_classification_metrics.png"
    evaluator.plot_classification_metrics(results, save_path=metrics_path)
    saved_files["classification_metrics"] = str(metrics_path)

    # 3. Learning Curves
    curves_path = save_dir / f"{base_name}_learning_curves.png"
    evaluator.plot_learning_curves(history, model_name, save_path=curves_path)
    saved_files["learning_curves"] = str(curves_path)

    logger.info(f"Comprehensive evaluation report created in {save_dir}")
    return saved_files
