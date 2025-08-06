"""
CIFAR Data Loader for CNN Image Classification Project
"""

import numpy as np
import tensorflow as tf
import pickle
import tarfile
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, Any
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, PREPROCESSING_CONFIG, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_cifar10_from_tar(datasets_dir: Path) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 dataset from tar.gz file"""
    cifar10_path = datasets_dir / "cifar-10-python.tar.gz"

    if not cifar10_path.exists():
        raise FileNotFoundError(f"CIFAR-10 dataset not found at {cifar10_path}")

    X_train_list = []
    y_train_list = []

    with tarfile.open(cifar10_path, "r:gz") as tar:
        for i in range(1, 6):
            batch_name = f"cifar-10-batches-py/data_batch_{i}"
            batch_file = tar.extractfile(batch_name)
            batch = pickle.load(batch_file, encoding="bytes")
            data = batch[b"data"]
            labels = batch[b"labels"]
            data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
            X_train_list.append(data)
            y_train_list.extend(labels)

        test_batch_file = tar.extractfile("cifar-10-batches-py/test_batch")
        test_batch = pickle.load(test_batch_file, encoding="bytes")
        X_test = test_batch[b"data"]
        y_test = test_batch[b"labels"]
        X_test = X_test.reshape(len(X_test), 3, 32, 32).transpose(0, 2, 3, 1)

    X_train = np.vstack(X_train_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test)

    return (X_train, y_train), (X_test, y_test)


def load_cifar100_from_tar(datasets_dir: Path, label_mode: str = "fine") -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-100 dataset from tar.gz file"""
    cifar100_path = datasets_dir / "cifar-100-python.tar.gz"

    if not cifar100_path.exists():
        raise FileNotFoundError(f"CIFAR-100 dataset not found at {cifar100_path}")

    label_key = b"fine_labels" if label_mode == "fine" else b"coarse_labels"

    with tarfile.open(cifar100_path, "r:gz") as tar:
        train_file = tar.extractfile("cifar-100-python/train")
        train_batch = pickle.load(train_file, encoding="bytes")
        X_train = train_batch[b"data"]
        y_train = train_batch[label_key]
        X_train = X_train.reshape(len(X_train), 3, 32, 32).transpose(0, 2, 3, 1)

        test_file = tar.extractfile("cifar-100-python/test")
        test_batch = pickle.load(test_file, encoding="bytes")
        X_test = test_batch[b"data"]
        y_test = test_batch[label_key]
        X_test = X_test.reshape(len(X_test), 3, 32, 32).transpose(0, 2, 3, 1)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return (X_train, y_train), (X_test, y_test)


class DataLoader:
    """Base class for data loading and preprocessing"""

    def __init__(self, dataset_name: str, validation_split: float = 0.2):
        self.dataset_name = dataset_name
        self.validation_split = validation_split
        self.dataset_config = DATASETS.get(dataset_name, {})

        self.num_classes = getattr(self, "num_classes", self.dataset_config.get("num_classes", 10))
        self.input_shape = getattr(self, "input_shape", self.dataset_config.get("input_shape", (32, 32, 3)))
        self.class_names = getattr(self, "class_names", self.dataset_config.get("class_names", None))

        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

        dataset_display_name = self.dataset_config.get("name", dataset_name.upper())
        logger.info(f"Initialized {self.__class__.__name__} for {dataset_display_name}")

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses must implement load_and_preprocess method")

    def normalize_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize pixel values to [0, 1] range"""
        X_train = X_train.astype("float32") / 255.0
        X_val = X_val.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0
        return X_train, X_val, X_test

    def standardize_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize data using ImageNet statistics"""
        mean = np.array(PREPROCESSING_CONFIG["channel_mean"])
        std = np.array(PREPROCESSING_CONFIG["channel_std"])

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        return X_train, X_val, X_test

    def encode_labels(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One-hot encode labels"""
        y_train_onehot = tf.keras.utils.to_categorical(y_train, self.num_classes).astype(np.float32)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, self.num_classes).astype(np.float32)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, self.num_classes).astype(np.float32)
        return y_train_onehot, y_val_onehot, y_test_onehot

    def get_data_info(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Get dataset information"""
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

        return info

    def split_train_validation(self, X_train_full: np.ndarray, y_train_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split training data into train and validation sets"""
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=self.validation_split, random_state=RANDOM_SEED, stratify=y_train_full
        )
        return X_train, X_val, y_train, y_val

    def create_data_generators(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 32, augment_training: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Create TensorFlow data generators for training and validation"""
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1000)

        if augment_training:
            augment_fn = self._get_augmentation_function()
            train_dataset = train_dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    def _get_augmentation_function(self):
        """Get dataset-specific augmentation function"""
        def default_augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            return image, label
        return default_augment


class CIFAR10Loader(DataLoader):
    """CIFAR-10 dataset loader and preprocessor"""

    def __init__(self, validation_split: float = 0.2):
        super().__init__("cifar10", validation_split)

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess CIFAR-10 dataset from tar.gz file"""
        (X_train_full, y_train_full), (X_test, y_test) = load_cifar10_from_tar(DATASETS_DIR)

        X_train, X_val, y_train, y_val = self.split_train_validation(X_train_full, y_train_full)
        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _get_augmentation_function(self):
        """Get CIFAR-10 specific augmentation function"""
        def cifar10_augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.image.random_crop(image, size=[28, 28, 3])
            image = tf.image.resize(image, [32, 32])
            return image, label
        return cifar10_augment


class CIFAR100Loader(DataLoader):
    """CIFAR-100 dataset loader and preprocessor"""

    def __init__(self, validation_split: float = 0.2, label_mode: str = "fine"):
        super().__init__("cifar100", validation_split)
        self.label_mode = label_mode

        if label_mode == "coarse":
            self.num_classes = 20

    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess CIFAR-100 dataset from tar.gz file"""
        (X_train_full, y_train_full), (X_test, y_test) = load_cifar100_from_tar(DATASETS_DIR, self.label_mode)

        X_train, X_val, y_train, y_val = self.split_train_validation(X_train_full, y_train_full)
        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_data_info(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Get dataset information with CIFAR-100 specific details"""
        info = super().get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
        info.update({"label_mode": self.label_mode, "effective_classes": self.num_classes})
        return info

    def _get_augmentation_function(self):
        """Get CIFAR-100 specific augmentation function"""
        def cifar100_augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.15)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            image = tf.image.random_crop(image, size=[28, 28, 3])
            image = tf.image.resize(image, [32, 32])
            return image, label
        return cifar100_augment


if __name__ == "__main__":
    print("Testing CIFAR-10 loader...")
    try:
        cifar10_loader = CIFAR10Loader(validation_split=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = cifar10_loader.load_and_preprocess()

        info = cifar10_loader.get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
        print("\nCIFAR-10 Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        train_ds, val_ds = cifar10_loader.create_data_generators(X_train, y_train, X_val, y_val, batch_size=32)
        print(f"\nTrain dataset: {train_ds}")
        print(f"Validation dataset: {val_ds}")
        print("\nCIFAR-10 test completed successfully!")

    except Exception as e:
        print(f"Error testing CIFAR-10 loader: {e}")

    print("\n" + "=" * 50)
    print("Testing CIFAR-100 loader...")
    try:
        cifar100_loader = CIFAR100Loader(validation_split=0.2, label_mode="fine")
        X_train, X_val, X_test, y_train, y_val, y_test = cifar100_loader.load_and_preprocess()

        info = cifar100_loader.get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
        print("\nCIFAR-100 Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\nCIFAR-100 test completed successfully!")

    except Exception as e:
        print(f"Error testing CIFAR-100 loader: {e}")
