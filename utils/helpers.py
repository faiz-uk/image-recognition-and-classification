"""
Helper utilities for CNN Image Classification Project
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, SAVED_MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducible results across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    tf.config.experimental.enable_op_determinism()

    logger.info(f"Random seed set to {seed} for reproducibility")


def get_device_info() -> Dict[str, Any]:
    """Get information about available computing devices"""
    device_info = {
        "tensorflow_version": tf.__version__,
        "gpu_available": tf.config.list_physical_devices("GPU"),
        "gpu_count": len(tf.config.list_physical_devices("GPU")),
        "cpu_count": len(tf.config.list_physical_devices("CPU")),
        "mixed_precision": tf.keras.mixed_precision.global_policy().name,
    }

    if device_info["gpu_available"]:
        gpu_details = []
        for gpu in tf.config.list_physical_devices("GPU"):
            gpu_details.append({"name": gpu.name, "device_type": gpu.device_type})
        device_info["gpu_details"] = gpu_details

    logger.info("Device information gathered")
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")

    return device_info


def save_model(
    model: keras.Model,
    model_name: str,
    save_weights_only: bool = False,
    save_dir: Optional[Path] = None,
) -> str:
    """Save trained model with metadata"""
    if save_dir is None:
        save_dir = SAVED_MODELS_DIR

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_weights_only:
        save_path = save_dir / f"{model_name}_weights.h5"
        model.save_weights(save_path)
        logger.info(f"Model weights saved to {save_path}")
    else:
        save_path = save_dir / f"{model_name}.h5"
        model.save(save_path)
        logger.info(f"Full model saved to {save_path}")

    architecture_path = save_dir / f"{model_name}_architecture.json"
    with open(architecture_path, "w") as f:
        f.write(model.to_json())

    summary_path = save_dir / f"{model_name}_summary.txt"
    with open(summary_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    logger.info(f"Model metadata saved to {save_dir}")
    return str(save_path)


def load_model(
    model_path: str,
    weights_only: bool = False,
    custom_objects: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Load saved model"""
    model_path = Path(model_path)

    if weights_only:
        architecture_path = (
            model_path.parent
            / f"{model_path.stem.replace('_weights', '')}_architecture.json"
        )

        if not architecture_path.exists():
            raise FileNotFoundError(f"Architecture file not found: {architecture_path}")

        with open(architecture_path, "r") as f:
            model_json = f.read()

        model = keras.models.model_from_json(model_json, custom_objects=custom_objects)
        model.load_weights(model_path)
        logger.info(f"Model weights loaded from {model_path}")
    else:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info(f"Full model loaded from {model_path}")

    return model


def save_training_history(
    history: keras.callbacks.History, model_name: str, save_dir: Optional[Path] = None
) -> str:
    """Save training history to file"""
    if save_dir is None:
        save_dir = SAVED_MODELS_DIR

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history_path = save_dir / f"{model_name}_history.json"

    history_dict = {
        "history": history.history,
        "epoch": history.epoch,
        "params": history.params,
    }

    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    logger.info(f"Training history saved to {history_path}")
    return str(history_path)


def load_training_history(history_path: str) -> Dict[str, Any]:
    """Load training history from file"""
    with open(history_path, "r") as f:
        history_dict = json.load(f)

    logger.info(f"Training history loaded from {history_path}")
    return history_dict


def calculate_model_size(model: keras.Model) -> Dict[str, Union[int, float]]:
    """Calculate model size and parameter count"""
    trainable_params = model.count_params()
    non_trainable_params = sum(
        [layer.count_params() for layer in model.layers if not layer.trainable]
    )
    total_params = trainable_params + non_trainable_params

    model_size_mb = (total_params * 4) / (1024 * 1024)

    size_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "model_size_mb": model_size_mb,
    }

    logger.info(
        f"Model size calculated: {total_params:,} parameters ({model_size_mb:.2f} MB)"
    )
    return size_info


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"


def create_experiment_config(
    model_name: str, dataset_name: str, **kwargs
) -> Dict[str, Any]:
    """Create experiment configuration dictionary"""
    from datetime import datetime

    config = {
        "experiment_name": f"{model_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "model_name": model_name,
        "dataset_name": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "tensorflow_version": tf.__version__,
        "device_info": get_device_info(),
    }

    config.update(kwargs)

    logger.info(f"Experiment configuration created: {config['experiment_name']}")
    return config


def save_experiment_config(
    config: Dict[str, Any], save_dir: Optional[Path] = None
) -> str:
    """Save experiment configuration to file"""
    if save_dir is None:
        save_dir = SAVED_MODELS_DIR

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / f"{config['experiment_name']}_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Experiment configuration saved to {config_path}")
    return str(config_path)


def save_experiment_results(
    results: Dict[str, Any],
    history: keras.callbacks.History,
    save_dir: Path,
    include_plots: bool = True,
) -> Dict[str, str]:
    """Save comprehensive experiment results including metrics, history, and plots"""
    import matplotlib.pyplot as plt
    from datetime import datetime

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = save_dir / "plots"
    logs_dir = save_dir / "logs"
    tables_dir = save_dir / "tables"

    for dir_path in [plots_dir, logs_dir, tables_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    model_name = results.get("model_name", "model")
    dataset_name = results.get("dataset", "dataset")

    saved_files = {}

    results_path = tables_dir / f"{model_name}_{dataset_name}_{timestamp}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    saved_files["results"] = str(results_path)

    history_path = logs_dir / f"{model_name}_{dataset_name}_{timestamp}_history.json"
    history_dict = {
        "history": history.history,
        "epoch": history.epoch,
        "params": history.params,
    }
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    saved_files["history"] = str(history_path)

    if include_plots and hasattr(history, "history"):
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            ax1.plot(history.history["loss"], label="Training Loss")
            if "val_loss" in history.history:
                ax1.plot(history.history["val_loss"], label="Validation Loss")
            ax1.set_title("Model Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(history.history["accuracy"], label="Training Accuracy")
            if "val_accuracy" in history.history:
                ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
            ax2.set_title("Model Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True)

            if "lr" in history.history:
                ax3.plot(history.history["lr"], label="Learning Rate")
                ax3.set_title("Learning Rate")
                ax3.set_xlabel("Epoch")
                ax3.set_ylabel("Learning Rate")
                ax3.legend()
                ax3.grid(True)
                ax3.set_yscale("log")
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "Learning Rate\nNot Available",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                )

            final_metrics = {
                "Final Training Loss": history.history["loss"][-1],
                "Final Training Accuracy": history.history["accuracy"][-1],
                "Test Accuracy": results.get("test_accuracy", "N/A"),
                "Epochs Trained": len(history.history["loss"]),
            }

            if "val_loss" in history.history:
                final_metrics["Final Validation Loss"] = history.history["val_loss"][-1]
                final_metrics["Final Validation Accuracy"] = history.history[
                    "val_accuracy"
                ][-1]

            metrics_text = "\n".join(
                [
                    f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                    for k, v in final_metrics.items()
                ]
            )
            ax4.text(
                0.1,
                0.9,
                metrics_text,
                transform=ax4.transAxes,
                verticalalignment="top",
                fontsize=10,
                fontfamily="monospace",
            )
            ax4.set_title("Final Metrics")
            ax4.axis("off")

            plt.tight_layout()

            plot_path = (
                plots_dir
                / f"{model_name}_{dataset_name}_{timestamp}_training_curves.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            saved_files["training_curves"] = str(plot_path)

        except Exception as e:
            logger.error(f"Error generating plots: {e}")

    summary_path = tables_dir / f"{model_name}_{dataset_name}_{timestamp}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Experiment Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Timestamp: {timestamp}\n")

        test_acc = results.get("test_accuracy", "N/A")
        if isinstance(test_acc, (int, float)):
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
        else:
            f.write(f"Test Accuracy: {test_acc}\n")

        test_loss = results.get("test_loss", "N/A")
        if isinstance(test_loss, (int, float)):
            f.write(f"Test Loss: {test_loss:.4f}\n")
        else:
            f.write(f"Test Loss: {test_loss}\n")

        f.write(f"Epochs Trained: {results.get('epochs_trained', 'N/A')}\n")
        f.write(f"\nFinal Training Metrics:\n")
        f.write(f"  Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"  Accuracy: {history.history['accuracy'][-1]:.4f}\n")

        if "val_loss" in history.history:
            f.write(f"\nFinal Validation Metrics:\n")
            f.write(f"  Loss: {history.history['val_loss'][-1]:.4f}\n")
            f.write(f"  Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")

    saved_files["summary"] = str(summary_path)

    logger.info(f"Experiment results saved to {save_dir}")
    logger.info(f"Saved files: {list(saved_files.keys())}")

    return saved_files
