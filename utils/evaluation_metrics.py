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
        """Comprehensive model evaluation following the evaluation strategy"""
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1).astype(int)
        else:
            y_true_labels = y_true.flatten().astype(int)

        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_labels = np.argmax(y_pred, axis=1).astype(int)
        else:
            y_pred_labels = y_pred.flatten().astype(int)

        # Core metrics
        accuracy = accuracy_score(y_true_labels, y_pred_labels)

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

        precision_per_class = precision_score(
            y_true_labels, y_pred_labels, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true_labels, y_pred_labels, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true_labels, y_pred_labels, average=None, zero_division=0
        )

        cm = confusion_matrix(y_true_labels, y_pred_labels)

        class_report = classification_report(
            y_true_labels,
            y_pred_labels,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        roc_auc = None
        if y_pred_proba is not None:
            try:
                if self.task_type == "binary":
                    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                        proba_positive = y_pred_proba[:, 1]
                    else:
                        proba_positive = y_pred_proba.flatten()
                    roc_auc = roc_auc_score(y_true_labels, proba_positive)
                else:
                    y_true_onehot = (
                        y_true
                        if y_true.ndim > 1
                        else tf.keras.utils.to_categorical(y_true_labels)
                    )
                    roc_auc = roc_auc_score(
                        y_true_onehot, y_pred_proba, multi_class="ovr", average="weighted"
                    )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")

        results = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "roc_auc": roc_auc,
            "num_samples": len(y_true_labels),
            "num_classes": len(np.unique(y_true_labels)),
        }

        self.results_history.append(results)
        return results

    def plot_confusion_matrix(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Optional[str]:
        """Plot confusion matrix with proper formatting"""
        cm = np.array(results["confusion_matrix"])
        
        plt.figure(figsize=figsize)
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Count"},
        )
        
        plt.title(
            f"Confusion Matrix - {results['model_name']} on {results['dataset_name']}\n"
            f"Accuracy: {results['accuracy']:.4f}"
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return None

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        model_name: str = "Model",
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> Optional[str]:
        """Plot training curves for loss and accuracy"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        epochs = range(1, len(history["loss"]) + 1)

        # Loss curves
        axes[0].plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
        if "val_loss" in history:
            axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
        axes[0].set_title("Model Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[1].plot(epochs, history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
        if "val_accuracy" in history:
            axes[1].plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2)
        axes[1].set_title("Model Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate
        if "lr" in history:
            axes[2].plot(epochs, history["lr"], "g-", label="Learning Rate", linewidth=2)
            axes[2].set_title("Learning Rate Schedule")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Learning Rate")
            axes[2].set_yscale("log")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "Learning Rate\nNot Available", 
                        ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_title("Learning Rate")

        plt.suptitle(f"Training Curves - {model_name}", fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return None

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Optional[str]:
        """Plot ROC curves for multi-class classification"""
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true.flatten().astype(int)

        n_classes = len(np.unique(y_true_labels))

        if self.task_type == "binary":
            fpr, tpr, _ = roc_curve(y_true_labels, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
            roc_auc = roc_auc_score(y_true_labels, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)

            plt.figure(figsize=figsize)
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
        else:
            y_true_bin = label_binarize(y_true_labels, classes=range(n_classes))
            
            plt.figure(figsize=figsize)
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                
                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.4f})")

            plt.plot([0, 1], [0, 1], "k--", lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Multi-class ROC Curves - {model_name}")
            plt.legend(loc="lower right")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return None

    def compare_models(self, results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple model results"""
        comparison_data = []
        
        for result in results_list:
            comparison_data.append({
                "Model": result["model_name"],
                "Dataset": result["dataset_name"],
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1-Score": result["f1_score"],
                "ROC-AUC": result.get("roc_auc", "N/A"),
                "Samples": result["num_samples"],
                "Classes": result["num_classes"],
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values("Accuracy", ascending=False)


def create_evaluation_report(
    results_list: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
    title: str = "Model Evaluation Report",
) -> str:
    """Create comprehensive evaluation report"""
    report_lines = []
    report_lines.append(f"{title}")
    report_lines.append("=" * len(title))
    report_lines.append("")

    # Summary table
    report_lines.append("SUMMARY RESULTS")
    report_lines.append("-" * 50)
    
    for result in results_list:
        report_lines.append(f"Model: {result['model_name']} | Dataset: {result['dataset_name']}")
        report_lines.append(f"  Accuracy:  {result['accuracy']:.4f}")
        report_lines.append(f"  Precision: {result['precision']:.4f}")
        report_lines.append(f"  Recall:    {result['recall']:.4f}")
        report_lines.append(f"  F1-Score:  {result['f1_score']:.4f}")
        if result.get('roc_auc'):
            report_lines.append(f"  ROC-AUC:   {result['roc_auc']:.4f}")
        report_lines.append("")

    # Best performing model
    if results_list:
        best_model = max(results_list, key=lambda x: x['accuracy'])
        report_lines.append("BEST PERFORMING MODEL")
        report_lines.append("-" * 30)
        report_lines.append(f"Model: {best_model['model_name']}")
        report_lines.append(f"Dataset: {best_model['dataset_name']}")
        report_lines.append(f"Accuracy: {best_model['accuracy']:.4f}")
        report_lines.append("")

    report_text = "\n".join(report_lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report_text)
        return str(save_path)
    
    return report_text
