"""
Configuration file for CNN Image Classification Project
Contains all hyperparameters, paths, and settings
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
TABLES_DIR = RESULTS_DIR / "tables"

DATASETS = {
    "cifar10": {
        "name": "CIFAR-10",
        "num_classes": 10,
        "input_shape": (32, 32, 3),
        "class_names": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    },
    "cifar100": {
        "name": "CIFAR-100",
        "num_classes": 100,
        "input_shape": (32, 32, 3),
        "class_names": None,
    },
    "fashion_mnist": {
        "name": "Fashion-MNIST",
        "num_classes": 10,
        "input_shape": (28, 28, 1),
        "class_names": [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    },
    "celeba": {
        "name": "CelebA",
        "num_classes": 2,
        "input_shape": (64, 64, 3),
        "class_names": None,
        "attributes": [
            "Smiling",
            "Male",
            "Young",
            "Attractive",
            "Heavy_Makeup",
            "Eyeglasses",
            "Bald",
            "Mustache",
            "Goatee",
            "Pale_Skin",
        ],
    },
}

MODELS = {
    "baseline_cnn": {
        "name": "BaselineCNN",
        "architecture": "baseline_cnn",
        "input_shape": (32, 32, 3),
        "include_top": False,
        "weights": None,
        "trainable_params": "all",
    },
    "resnet50": {
        "name": "ResNet50",
        "architecture": "resnet50",
        "input_shape": (32, 32, 3),
        "include_top": False,
        "weights": "imagenet",
        "trainable_params": "top_layers",
    },
    "densenet121": {
        "name": "DenseNet121",
        "architecture": "densenet121",
        "input_shape": (32, 32, 3),
        "include_top": False,
        "weights": "imagenet",
        "trainable_params": "top_layers",
    },
    "inceptionv3": {
        "name": "InceptionV3",
        "architecture": "inceptionv3",
        "input_shape": (32, 32, 3),
        "include_top": False,
        "weights": "imagenet",
        "trainable_params": "top_layers",
        "min_input_size": (75, 75),
    },
    "mobilenet": {
        "name": "MobileNet",
        "architecture": "mobilenet",
        "input_shape": (32, 32, 3),
        "include_top": False,
        "weights": "imagenet",
        "trainable_params": "top_layers",
        "alpha": 1.0,
        "mobile_optimized": True,
    },
}

TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 100,
    "validation_split": 0.2,
    "learning_rate": 0.001,
    "patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-7,
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
}

MODEL_TRAINING_CONFIG = {
    "baseline_cnn": {
        "epochs": 80,
        "batch_size": 32,
        "learning_rate": 0.001,
        "patience": 8,
        "description": "Custom CNN trained from scratch",
    },
    "resnet50": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "patience": 10,
        "description": "ResNet50 with ImageNet pre-training",
    },
    "densenet121": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "patience": 10,
        "description": "DenseNet121 with ImageNet pre-training",
    },
    "inceptionv3": {
        "epochs": 120,
        "batch_size": 24,
        "learning_rate": 0.0003,
        "patience": 12,
        "description": "InceptionV3 with ImageNet pre-training",
    },
    "mobilenet": {
        "epochs": 80,
        "batch_size": 64,
        "learning_rate": 0.001,
        "patience": 8,
        "description": "MobileNet optimized for efficiency",
    },
}

DATASET_TRAINING_CONFIG = {
    "cifar10": {
        "difficulty": "medium",
        "recommended_epochs": 80,
        "batch_size_multiplier": 1.0,
        "augmentation_level": "standard",
    },
    "cifar100": {
        "difficulty": "hard",
        "recommended_epochs": 120,
        "batch_size_multiplier": 0.8,
        "augmentation_level": "aggressive",
    },
    "fashion_mnist": {
        "difficulty": "easy",
        "recommended_epochs": 50,
        "batch_size_multiplier": 1.2,
        "augmentation_level": "minimal",
    },
    "celeba": {
        "difficulty": "medium",
        "recommended_epochs": 60,
        "batch_size_multiplier": 0.7,
        "augmentation_level": "standard",
    },
}

PERFORMANCE_BENCHMARKS = {
    "baseline_cnn": {
        "cifar10": {"accuracy": 0.78, "range": (0.75, 0.82)},
        "cifar100": {"accuracy": 0.52, "range": (0.48, 0.58)},
        "fashion_mnist": {"accuracy": 0.89, "range": (0.87, 0.92)},
        "celeba": {"accuracy": 0.84, "range": (0.82, 0.87)},
    },
    "resnet50": {
        "cifar10": {"accuracy": 0.89, "range": (0.87, 0.92)},
        "cifar100": {"accuracy": 0.72, "range": (0.68, 0.76)},
        "fashion_mnist": {"accuracy": 0.93, "range": (0.91, 0.95)},
        "celeba": {"accuracy": 0.89, "range": (0.87, 0.91)},
    },
    "densenet121": {
        "cifar10": {"accuracy": 0.91, "range": (0.89, 0.93)},
        "cifar100": {"accuracy": 0.74, "range": (0.71, 0.77)},
        "fashion_mnist": {"accuracy": 0.94, "range": (0.92, 0.96)},
        "celeba": {"accuracy": 0.90, "range": (0.88, 0.92)},
    },
    "inceptionv3": {
        "cifar10": {"accuracy": 0.90, "range": (0.88, 0.92)},
        "cifar100": {"accuracy": 0.73, "range": (0.70, 0.76)},
        "fashion_mnist": {"accuracy": 0.94, "range": (0.92, 0.96)},
        "celeba": {"accuracy": 0.91, "range": (0.89, 0.93)},
    },
    "mobilenet": {
        "cifar10": {"accuracy": 0.87, "range": (0.85, 0.90)},
        "cifar100": {"accuracy": 0.69, "range": (0.66, 0.72)},
        "fashion_mnist": {"accuracy": 0.92, "range": (0.90, 0.94)},
        "celeba": {"accuracy": 0.88, "range": (0.86, 0.90)},
    },
}

AUGMENTATION_CONFIG = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "horizontal_flip": True,
    "zoom_range": 0.2,
    "shear_range": 0.15,
    "fill_mode": "nearest",
}

AUGMENTATION_LEVELS = {
    "none": {"enabled": False, "description": "No augmentation"},
    "minimal": {
        "enabled": True,
        "rotation_range": 10,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "horizontal_flip": True,
        "zoom_range": 0.05,
        "description": "Light augmentation for easy datasets",
    },
    "standard": {
        "enabled": True,
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "horizontal_flip": True,
        "zoom_range": 0.2,
        "shear_range": 0.15,
        "brightness_range": [0.9, 1.1],
        "description": "Standard augmentation for most datasets",
    },
    "aggressive": {
        "enabled": True,
        "rotation_range": 30,
        "width_shift_range": 0.3,
        "height_shift_range": 0.3,
        "horizontal_flip": True,
        "vertical_flip": False,
        "zoom_range": 0.3,
        "shear_range": 0.2,
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2],
        "saturation_range": [0.8, 1.2],
        "description": "Aggressive augmentation for difficult datasets",
    },
}

DATASET_AUGMENTATION_CONFIG = {
    "cifar10": {
        "default_level": "standard",
        "custom": {
            "horizontal_flip": True,
            "rotation_range": 15,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.1,
            "cutout": True,
            "cutout_size": 8,
        },
    },
    "cifar100": {
        "default_level": "aggressive",
        "custom": {
            "horizontal_flip": True,
            "rotation_range": 20,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "zoom_range": 0.2,
            "cutout": True,
            "cutout_size": 8,
            "mixup": True,
        },
    },
    "fashion_mnist": {
        "default_level": "minimal",
        "custom": {
            "horizontal_flip": False,
            "rotation_range": 10,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.05,
            "elastic_transform": True,
        },
    },
    "celeba": {
        "default_level": "standard",
        "custom": {
            "horizontal_flip": True,
            "rotation_range": 5,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.1,
            "brightness_range": [0.9, 1.1],
            "contrast_range": [0.9, 1.1],
        },
    },
}

PREPROCESSING_CONFIG = {
    "normalize": True,
    "rescale": 1.0 / 255.0,
    "channel_mean": [0.485, 0.456, 0.406],
    "channel_std": [0.229, 0.224, 0.225],
}

EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1_score"],
    "average": "weighted",
    "plot_confusion_matrix": True,
    "plot_training_curves": True,
    "save_predictions": True,
}

HARDWARE_CONFIG = {"use_gpu": True, "gpu_memory_growth": True, "mixed_precision": True}

RANDOM_SEED = 42

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "save_to_file": True,
}

OPTIONAL_DATASETS = {
    "celeba": {
        "url": "https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view",
        "num_classes": 2,
        "input_shape": (64, 64, 3),
    },
    "svhn": {
        "url": "http://ufldl.stanford.edu/housenumbers/",
        "num_classes": 10,
        "input_shape": (32, 32, 3),
    },
}

for directory in [
    DATA_DIR,
    SAVED_MODELS_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    LOGS_DIR,
    TABLES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def get_optimal_config(model_name: str, dataset_name: str) -> dict:
    """
    Get optimal configuration for a model-dataset combination

    Args:
        model_name: Name of the model ('baseline_cnn', 'resnet50', etc.)
        dataset_name: Name of the dataset ('cifar10', 'cifar100', etc.)

    Returns:
        Dictionary with optimal configuration
    """
    if model_name not in MODEL_TRAINING_CONFIG:
        raise ValueError(f"Model '{model_name}' not found in MODEL_TRAINING_CONFIG")
    if dataset_name not in DATASET_TRAINING_CONFIG:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in DATASET_TRAINING_CONFIG"
        )

    model_config = MODEL_TRAINING_CONFIG[model_name].copy()
    dataset_config = DATASET_TRAINING_CONFIG[dataset_name].copy()

    optimal_batch_size = int(
        model_config["batch_size"] * dataset_config["batch_size_multiplier"]
    )
    model_config["batch_size"] = max(8, optimal_batch_size)

    if dataset_config["difficulty"] == "hard":
        model_config["epochs"] = int(model_config["epochs"] * 1.2)
    elif dataset_config["difficulty"] == "easy":
        model_config["epochs"] = int(model_config["epochs"] * 0.8)

    model_config.update(
        {
            "dataset_difficulty": dataset_config["difficulty"],
            "recommended_augmentation": dataset_config["augmentation_level"],
        }
    )

    return model_config


def get_expected_performance(model_name: str, dataset_name: str) -> dict:
    """
    Get expected performance for a model-dataset combination

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset

    Returns:
        Dictionary with expected performance metrics
    """
    if model_name not in PERFORMANCE_BENCHMARKS:
        return {"accuracy": 0.5, "range": (0.4, 0.6), "note": "No benchmark available"}

    if dataset_name not in PERFORMANCE_BENCHMARKS[model_name]:
        return {"accuracy": 0.5, "range": (0.4, 0.6), "note": "No benchmark available"}

    return PERFORMANCE_BENCHMARKS[model_name][dataset_name]


def get_augmentation_config(dataset_name: str, level: str = None) -> dict:
    """
    Get augmentation configuration for a dataset

    Args:
        dataset_name: Name of the dataset
        level: Augmentation level ('minimal', 'standard', 'aggressive', or None for dataset default)

    Returns:
        Dictionary with augmentation configuration
    """
    if dataset_name not in DATASET_AUGMENTATION_CONFIG:
        return AUGMENTATION_LEVELS["standard"].copy()

    dataset_aug_config = DATASET_AUGMENTATION_CONFIG[dataset_name]

    if level is None:
        level = dataset_aug_config["default_level"]

    if level not in AUGMENTATION_LEVELS:
        level = "standard"

    aug_config = AUGMENTATION_LEVELS[level].copy()

    if "custom" in dataset_aug_config:
        aug_config.update(dataset_aug_config["custom"])

    return aug_config


def validate_config_combination(model_name: str, dataset_name: str) -> dict:
    """
    Validate and get warnings for a model-dataset combination

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset

    Returns:
        Dictionary with validation results and warnings
    """
    warnings = []
    recommendations = []

    if model_name not in MODELS:
        warnings.append(f"Model '{model_name}' not found in MODELS configuration")

    if dataset_name not in DATASETS:
        warnings.append(f"Dataset '{dataset_name}' not found in DATASETS configuration")

    if model_name == "inceptionv3" and dataset_name in ["fashion_mnist"]:
        warnings.append(
            "InceptionV3 on Fashion-MNIST: Input will be resized from 28x28 to 75x75"
        )

    if model_name == "mobilenet" and dataset_name == "celeba":
        recommendations.append(
            "Consider using alpha=0.75 for MobileNet on CelebA for better performance"
        )

    if model_name == "baseline_cnn" and dataset_name == "cifar100":
        warnings.append(
            "BaselineCNN on CIFAR-100: Expect lower performance due to high complexity"
        )

    memory_intensive = {
        ("inceptionv3", "celeba"): "High memory usage expected",
        ("resnet50", "celeba"): "High memory usage with 64x64 images",
        (
            "densenet121",
            "celeba",
        ): "Consider reducing batch size if memory issues occur",
    }

    if (model_name, dataset_name) in memory_intensive:
        warnings.append(memory_intensive[(model_name, dataset_name)])

    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def get_training_summary(model_name: str, dataset_name: str) -> str:
    """
    Get a comprehensive training summary for a model-dataset combination

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset

    Returns:
        Formatted string with training summary
    """
    try:
        optimal_config = get_optimal_config(model_name, dataset_name)
        expected_perf = get_expected_performance(model_name, dataset_name)
        validation = validate_config_combination(model_name, dataset_name)

        model_info = MODELS.get(model_name, {})
        dataset_info = DATASETS.get(dataset_name, {})

        summary = f"""
Training Configuration Summary
============================

Model: {model_info.get('name', model_name)}
Dataset: {dataset_info.get('name', dataset_name)}
Classes: {dataset_info.get('num_classes', 'Unknown')}

Optimal Training Settings:
  Epochs: {optimal_config['epochs']}
  Batch Size: {optimal_config['batch_size']}
  Learning Rate: {optimal_config['learning_rate']}
  Patience: {optimal_config['patience']}

Expected Performance:
  Target Accuracy: {expected_perf['accuracy']:.1%}
  Expected Range: {expected_perf['range'][0]:.1%} - {expected_perf['range'][1]:.1%}

Estimated Training Time: {estimate_training_time(model_name, dataset_name, optimal_config['epochs'])}

Validation Status: {'Valid' if validation['valid'] else 'Has Warnings'}
"""

        if validation["warnings"]:
            summary += "\nWarnings:\n"
            for warning in validation["warnings"]:
                summary += f"  WARNING: {warning}\n"

        if validation["recommendations"]:
            summary += "\nRecommendations:\n"
            for rec in validation["recommendations"]:
                summary += f"  TIP: {rec}\n"

        return summary

    except Exception as e:
        return f"Error generating summary: {e}"


def estimate_training_time(model_name: str, dataset_name: str, epochs: int) -> str:
    """
    Estimate training time for a model-dataset combination

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        epochs: Number of epochs

    Returns:
        Estimated training time as string
    """
    base_times = {
        "baseline_cnn": {
            "cifar10": 1,
            "cifar100": 1,
            "fashion_mnist": 0.5,
            "celeba": 2,
        },
        "resnet50": {"cifar10": 3, "cifar100": 3, "fashion_mnist": 2, "celeba": 8},
        "densenet121": {"cifar10": 2, "cifar100": 2, "fashion_mnist": 1.5, "celeba": 6},
        "inceptionv3": {"cifar10": 4, "cifar100": 4, "fashion_mnist": 3, "celeba": 10},
        "mobilenet": {"cifar10": 1.5, "cifar100": 1.5, "fashion_mnist": 1, "celeba": 3},
    }

    if model_name not in base_times or dataset_name not in base_times[model_name]:
        return "Unknown"

    total_minutes = base_times[model_name][dataset_name] * epochs

    if total_minutes < 60:
        return f"~{total_minutes:.0f} minutes"
    elif total_minutes < 1440:
        hours = total_minutes / 60
        return f"~{hours:.1f} hours"
    else:
        days = total_minutes / 1440
        return f"~{days:.1f} days"


ALL_MODEL_DATASET_COMBINATIONS = [
    (model, dataset) for model in MODELS.keys() for dataset in DATASETS.keys()
]

TOTAL_EXPERIMENTS = len(ALL_MODEL_DATASET_COMBINATIONS)

print(
    f"Configuration loaded: {TOTAL_EXPERIMENTS} total model-dataset combinations available"
)
