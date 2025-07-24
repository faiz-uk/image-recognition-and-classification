"""
CIFAR Data Loader for CNN Image Classification Project
Implements CIFAR-10 and CIFAR-100 dataset loading and preprocessing using local tar.gz files
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import tarfile
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, Any
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, PREPROCESSING_CONFIG, RANDOM_SEED

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_cifar_batch(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single CIFAR batch file

    Args:
        file_path: Path to the batch file

    Returns:
        Tuple of (data, labels)
    """
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    data = batch[b"data"]
    labels = batch[b"labels"]

    # Reshape data from (10000, 3072) to (10000, 32, 32, 3)
    data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)

    return np.array(data), np.array(labels)


def load_cifar10_from_local(
    datasets_dir: Path,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load CIFAR-10 dataset from local tar.gz file

    Args:
        datasets_dir: Directory containing the dataset files

    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    cifar10_path = datasets_dir / "cifar-10-python.tar.gz"

    if not cifar10_path.exists():
        raise FileNotFoundError(f"CIFAR-10 dataset not found at {cifar10_path}")

    logger.info(f"Loading CIFAR-10 from local file: {cifar10_path}")

    # Extract and load training batches
    X_train_list = []
    y_train_list = []

    with tarfile.open(cifar10_path, "r:gz") as tar:
        # Load training batches (data_batch_1 to data_batch_5)
        for i in range(1, 6):
            batch_name = f"cifar-10-batches-py/data_batch_{i}"
            batch_file = tar.extractfile(batch_name)
            if batch_file:
                batch = pickle.load(batch_file, encoding="bytes")
                data = batch[b"data"]
                labels = batch[b"labels"]

                # Reshape data from (10000, 3072) to (10000, 32, 32, 3)
                data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)

                X_train_list.append(data)
                y_train_list.extend(labels)

        # Load test batch
        test_batch_file = tar.extractfile("cifar-10-batches-py/test_batch")
        if test_batch_file:
            test_batch = pickle.load(test_batch_file, encoding="bytes")
            X_test = test_batch[b"data"]
            y_test = test_batch[b"labels"]

            # Reshape test data
            X_test = X_test.reshape(len(X_test), 3, 32, 32).transpose(0, 2, 3, 1)

    # Combine training batches
    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test)

    logger.info(f"CIFAR-10 loaded: Train {X_train.shape}, Test {X_test.shape}")
    return (X_train, y_train), (X_test, y_test)


def load_cifar100_from_local(
    datasets_dir: Path, label_mode: str = "fine"
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load CIFAR-100 dataset from local tar.gz file

    Args:
        datasets_dir: Directory containing the dataset files
        label_mode: 'fine' for 100 classes or 'coarse' for 20 superclasses

    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    cifar100_path = datasets_dir / "cifar-100-python.tar.gz"

    if not cifar100_path.exists():
        raise FileNotFoundError(f"CIFAR-100 dataset not found at {cifar100_path}")

    logger.info(f"Loading CIFAR-100 from local file: {cifar100_path}")

    label_key = b"fine_labels" if label_mode == "fine" else b"coarse_labels"

    with tarfile.open(cifar100_path, "r:gz") as tar:
        # Load training data
        train_file = tar.extractfile("cifar-100-python/train")
        if train_file:
            train_batch = pickle.load(train_file, encoding="bytes")
            X_train = train_batch[b"data"]
            y_train = train_batch[label_key]

            # Reshape training data
            X_train = X_train.reshape(len(X_train), 3, 32, 32).transpose(0, 2, 3, 1)

        # Load test data
        test_file = tar.extractfile("cifar-100-python/test")
        if test_file:
            test_batch = pickle.load(test_file, encoding="bytes")
            X_test = test_batch[b"data"]
            y_test = test_batch[label_key]

            # Reshape test data
            X_test = X_test.reshape(len(X_test), 3, 32, 32).transpose(0, 2, 3, 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    logger.info(f"CIFAR-100 loaded: Train {X_train.shape}, Test {X_test.shape}")
    return (X_train, y_train), (X_test, y_test)


class DataLoader:
    """Base class for data loading and preprocessing"""

    def __init__(self, dataset_name: str, validation_split: float = 0.2):
        """
        Initialize DataLoader

        Args:
            dataset_name: Name of the dataset ('cifar10', 'cifar100', 'fashion_mnist', 'celeba')
            validation_split: Fraction of training data to use for validation
        """
        self.dataset_name = dataset_name
        self.validation_split = validation_split

        # Try to get config from DATASETS, otherwise set defaults
        if hasattr(self, "_get_dataset_config"):
            self.dataset_config = self._get_dataset_config()
        else:
            self.dataset_config = DATASETS.get(dataset_name, {})

        # Set defaults if config not found
        self.num_classes = getattr(
            self, "num_classes", self.dataset_config.get("num_classes", 10)
        )
        self.input_shape = getattr(
            self, "input_shape", self.dataset_config.get("input_shape", (32, 32, 3))
        )
        self.class_names = getattr(
            self, "class_names", self.dataset_config.get("class_names", None)
        )

        # Set random seed for reproducibility
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

        dataset_display_name = self.dataset_config.get("name", dataset_name.upper())
        logger.info(f"Initialized {self.__class__.__name__} for {dataset_display_name}")

    def load_and_preprocess(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess dataset

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        raise NotImplementedError(
            "Subclasses must implement load_and_preprocess method"
        )

    def normalize_data(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize pixel values to [0, 1] range

        Args:
            X_train, X_val, X_test: Image arrays

        Returns:
            Normalized image arrays
        """
        X_train = X_train.astype("float32") / 255.0
        X_val = X_val.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

        logger.info("Data normalized to [0, 1] range")
        return X_train, X_val, X_test

    def standardize_data(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardize data using ImageNet statistics

        Args:
            X_train, X_val, X_test: Image arrays

        Returns:
            Standardized image arrays
        """
        mean = np.array(PREPROCESSING_CONFIG["channel_mean"])
        std = np.array(PREPROCESSING_CONFIG["channel_std"])

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        logger.info("Data standardized using ImageNet statistics")
        return X_train, X_val, X_test

    def encode_labels(
        self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One-hot encode labels

        Args:
            y_train, y_val, y_test: Label arrays

        Returns:
            One-hot encoded label arrays
        """

        y_train_onehot = tf.keras.utils.to_categorical(
            y_train, self.num_classes
        ).astype(np.float32)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, self.num_classes).astype(
            np.float32
        )
        y_test_onehot = tf.keras.utils.to_categorical(y_test, self.num_classes).astype(
            np.float32
        )

        logger.info(f"Labels one-hot encoded for {self.num_classes} classes")
        return y_train_onehot, y_val_onehot, y_test_onehot

    def get_data_info(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Get dataset information

        Returns:
            Dictionary with dataset statistics
        """
        dataset_name = self.dataset_config.get("name", self.dataset_name.upper())

        info = {
            "dataset_name": dataset_name,
            "loader_class": self.__class__.__name__,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "train_samples": X_train.shape[0],
            "val_samples": X_val.shape[0],
            "test_samples": X_test.shape[0],
            "total_samples": X_train.shape[0] + X_val.shape[0] + X_test.shape[0],
            "class_names": self.class_names,
            "pixel_range": f"[{X_train.min():.3f}, {X_train.max():.3f}]",
        }

        # Add dataset-specific info if available
        if hasattr(self, "_get_additional_info"):
            additional_info = self._get_additional_info(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            info.update(additional_info)

        logger.info(f"Dataset info gathered for {dataset_name}")
        return info

    def split_train_validation(
        self, X_train_full: np.ndarray, y_train_full: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split training data into train and validation sets

        Args:
            X_train_full: Full training images
            y_train_full: Full training labels

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=self.validation_split,
            random_state=RANDOM_SEED,
            stratify=y_train_full,
        )

        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
        return X_train, X_val, y_train, y_val

    def create_data_generators(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        augment_training: bool = True,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow data generators for training and validation

        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            batch_size: Batch size for training
            augment_training: Whether to apply data augmentation

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Create training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1000)

        if augment_training:
            # Apply dataset-specific augmentation
            augment_fn = self._get_augmentation_function()
            train_dataset = train_dataset.map(
                augment_fn, num_parallel_calls=tf.data.AUTOTUNE
            )

        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Create validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        dataset_name = self.dataset_config.get("name", self.dataset_name.upper())
        logger.info(
            f"{dataset_name} data generators created with batch size {batch_size}"
        )
        return train_dataset, val_dataset

    def _get_augmentation_function(self):
        """
        Get dataset-specific augmentation function
        Subclasses can override this for custom augmentation

        Returns:
            Augmentation function
        """

        def default_augment(image, label):
            # Basic augmentation suitable for most datasets
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            return image, label

        return default_augment


class CIFAR10Loader(DataLoader):
    """CIFAR-10 dataset loader and preprocessor using local files"""

    def __init__(self, validation_split: float = 0.2):
        super().__init__("cifar10", validation_split)

    def load_and_preprocess(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CIFAR-10 dataset from local files

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Loading CIFAR-10 dataset from local files...")

        # Load CIFAR-10 data from local files
        (X_train_full, y_train_full), (X_test, y_test) = load_cifar10_from_local(
            DATASETS_DIR
        )

        # Split training data into train and validation sets using parent method
        X_train, X_val, y_train, y_val = self.split_train_validation(
            X_train_full, y_train_full
        )

        # Normalize data using parent method
        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)

        # Encode labels using parent method
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)

        logger.info("CIFAR-10 data preprocessing completed")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _get_augmentation_function(self):
        """Get CIFAR-10 specific augmentation function"""

        def cifar10_augment(image, label):
            # CIFAR-10 specific augmentations
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            # Small crops for CIFAR (already small images)
            image = tf.image.random_crop(image, size=[28, 28, 3])
            image = tf.image.resize(image, [32, 32])
            return image, label

        return cifar10_augment


class CIFAR100Loader(DataLoader):
    """CIFAR-100 dataset loader and preprocessor using local files"""

    def __init__(self, validation_split: float = 0.2, label_mode: str = "fine"):
        """
        Initialize CIFAR-100 loader

        Args:
            validation_split: Fraction of training data for validation
            label_mode: 'fine' for 100 classes or 'coarse' for 20 superclasses
        """
        super().__init__("cifar100", validation_split)
        self.label_mode = label_mode

        if label_mode == "coarse":
            self.num_classes = 20
            logger.info("Using CIFAR-100 coarse labels (20 superclasses)")
        else:
            logger.info("Using CIFAR-100 fine labels (100 classes)")

    def load_and_preprocess(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CIFAR-100 dataset from local files

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Loading CIFAR-100 dataset from local files...")

        # Load CIFAR-100 data from local files
        (X_train_full, y_train_full), (X_test, y_test) = load_cifar100_from_local(
            DATASETS_DIR, self.label_mode
        )

        # Split training data into train and validation sets using parent method
        X_train, X_val, y_train, y_val = self.split_train_validation(
            X_train_full, y_train_full
        )

        # Normalize data using parent method
        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)

        # Encode labels using parent method
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)

        logger.info("CIFAR-100 data preprocessing completed")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _get_additional_info(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Get CIFAR-100 specific additional information"""
        return {"label_mode": self.label_mode, "effective_classes": self.num_classes}

    def _get_augmentation_function(self):
        """Get CIFAR-100 specific augmentation function"""

        def cifar100_augment(image, label):
            # CIFAR-100 specific augmentations (similar to CIFAR-10 but potentially more aggressive)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(
                image, max_delta=0.15
            )  # Slightly more aggressive
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            # Small crops for CIFAR (already small images)
            image = tf.image.random_crop(image, size=[28, 28, 3])
            image = tf.image.resize(image, [32, 32])
            return image, label

        return cifar100_augment


def create_data_generators(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    augment_training: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Legacy function for backward compatibility
    Create TensorFlow data generators for training and validation

    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        batch_size: Batch size for training
        augment_training: Whether to apply data augmentation

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000)

    if augment_training:
        # Add data augmentation to training set
        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            return image, label

        train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    logger.info(f"Data generators created with batch size {batch_size}")
    return train_dataset, val_dataset


# Example usage and testing
if __name__ == "__main__":
    # Test CIFAR-10 loader
    print("Testing CIFAR-10 loader...")
    try:
        cifar10_loader = CIFAR10Loader(validation_split=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            cifar10_loader.load_and_preprocess()
        )

        # Print dataset info
        info = cifar10_loader.get_data_info(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        print("\nCIFAR-10 Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test data generators
        train_ds, val_ds = cifar10_loader.create_data_generators(
            X_train, y_train, X_val, y_val, batch_size=32
        )
        print(f"\nTrain dataset: {train_ds}")
        print(f"Validation dataset: {val_ds}")

        print("\nCIFAR-10 data loading test completed successfully!")

    except Exception as e:
        print(f"Error testing CIFAR-10 loader: {e}")

    # Test CIFAR-100 loader
    print("\n" + "=" * 50)
    print("Testing CIFAR-100 loader...")
    try:
        cifar100_loader = CIFAR100Loader(validation_split=0.2, label_mode="fine")
        X_train, X_val, X_test, y_train, y_val, y_test = (
            cifar100_loader.load_and_preprocess()
        )

        # Print dataset info
        info = cifar100_loader.get_data_info(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        print("\nCIFAR-100 Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\nCIFAR-100 data loading test completed successfully!")

    except Exception as e:
        print(f"Error testing CIFAR-100 loader: {e}")
