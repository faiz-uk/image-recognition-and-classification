"""
Fashion-MNIST Data Loader for CNN Image Classification Project
"""

import struct
import tarfile
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .cifar_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_fashion_mnist_from_tar(datasets_dir: Path) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load Fashion-MNIST dataset from local tar.gz file"""
    fashion_tar_path = datasets_dir / "fashion-mnist.tar.gz"

    if not fashion_tar_path.exists():
        raise FileNotFoundError(f"Fashion-MNIST tar.gz file not found at {fashion_tar_path}")

    with tarfile.open(fashion_tar_path, "r:gz") as tar:
        train_images_file = tar.extractfile("train-images-idx3-ubyte")
        magic, num_images, rows, cols = struct.unpack(">IIII", train_images_file.read(16))
        X_train = np.frombuffer(train_images_file.read(), dtype=np.uint8).reshape(num_images, rows, cols)

        train_labels_file = tar.extractfile("train-labels-idx1-ubyte")
        magic, num_labels = struct.unpack(">II", train_labels_file.read(8))
        y_train = np.frombuffer(train_labels_file.read(), dtype=np.uint8)

        test_images_file = tar.extractfile("t10k-images-idx3-ubyte")
        magic, num_images, rows, cols = struct.unpack(">IIII", test_images_file.read(16))
        X_test = np.frombuffer(test_images_file.read(), dtype=np.uint8).reshape(num_images, rows, cols)

        test_labels_file = tar.extractfile("t10k-labels-idx1-ubyte")
        magic, num_labels = struct.unpack(">II", test_labels_file.read(8))
        y_test = np.frombuffer(test_labels_file.read(), dtype=np.uint8)

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return (X_train, y_train), (X_test, y_test)


class FashionMNISTLoader(DataLoader):
    """Fashion-MNIST dataset loader and preprocessor"""

    def __init__(
        self,
        validation_split: float = 0.2,
        resize_to: Tuple[int, int] = None,
    ):
        self.resize_to = resize_to

        self.num_classes = 10
        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        
        self.input_shape = self.resize_to + (1,) if self.resize_to else (28, 28, 1)
        super().__init__("fashion_mnist", validation_split)

    def load_and_preprocess(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess Fashion-MNIST dataset from tar.gz file"""
        (X_train_full, y_train_full), (X_test, y_test) = load_fashion_mnist_from_tar(DATASETS_DIR)

        X_train, X_val, y_train, y_val = self.split_train_validation(X_train_full, y_train_full)

        if self.resize_to:
            X_train = self._resize_images(X_train, self.resize_to)
            X_val = self._resize_images(X_val, self.resize_to)
            X_test = self._resize_images(X_test, self.resize_to)

        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _resize_images(self, images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize images to target size"""
        resized_images = tf.image.resize(images, target_size, method="bilinear")
        return resized_images.numpy()

    def get_data_info(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Get dataset information with Fashion-MNIST specific details"""
        info = super().get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
        info.update({"original_size": (28, 28), "resized_to": self.resize_to})
        return info

    def _get_augmentation_function(self):
        def fashion_mnist_augment(image, label):
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            return image, label

        return fashion_mnist_augment


if __name__ == "__main__":
    fashion_loader = FashionMNISTLoader(validation_split=0.2, resize_to=(32, 32))
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = fashion_loader.load_and_preprocess()
        info = fashion_loader.get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
        print("Fashion-MNIST Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
