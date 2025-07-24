"""
Fashion-MNIST Data Loader for CNN Image Classification Project
Implements Fashion-MNIST dataset loading and preprocessing using both local files and TensorFlow
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import struct
import gzip
import tarfile
import logging
from typing import Tuple, Dict, Any
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .cifar_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_fashion_mnist_from_tar(
    datasets_dir: Path,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load Fashion-MNIST dataset from local tar.gz file (consistent with CIFAR format)

    Args:
        datasets_dir: Directory containing the dataset files

    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    fashion_tar_path = datasets_dir / "fashion-mnist.tar.gz"

    if not fashion_tar_path.exists():
        raise FileNotFoundError(
            f"Fashion-MNIST tar.gz file not found at {fashion_tar_path}"
        )

    logger.info(f"Loading Fashion-MNIST from tar.gz file: {fashion_tar_path}")

    with tarfile.open(fashion_tar_path, "r:gz") as tar:
        # Load training images
        train_images_file = tar.extractfile("train-images-idx3-ubyte")
        if train_images_file:
            magic, num_images, rows, cols = struct.unpack(
                ">IIII", train_images_file.read(16)
            )
            X_train = np.frombuffer(train_images_file.read(), dtype=np.uint8).reshape(
                num_images, rows, cols
            )

        # Load training labels
        train_labels_file = tar.extractfile("train-labels-idx1-ubyte")
        if train_labels_file:
            magic, num_labels = struct.unpack(">II", train_labels_file.read(8))
            y_train = np.frombuffer(train_labels_file.read(), dtype=np.uint8)

        # Load test images
        test_images_file = tar.extractfile("t10k-images-idx3-ubyte")
        if test_images_file:
            magic, num_images, rows, cols = struct.unpack(
                ">IIII", test_images_file.read(16)
            )
            X_test = np.frombuffer(test_images_file.read(), dtype=np.uint8).reshape(
                num_images, rows, cols
            )

        # Load test labels
        test_labels_file = tar.extractfile("t10k-labels-idx1-ubyte")
        if test_labels_file:
            magic, num_labels = struct.unpack(">II", test_labels_file.read(8))
            y_test = np.frombuffer(test_labels_file.read(), dtype=np.uint8)

    # Add channel dimension for CNN compatibility (28, 28) -> (28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    logger.info(
        f"Fashion-MNIST loaded from tar.gz: Train {X_train.shape}, Test {X_test.shape}"
    )
    return (X_train, y_train), (X_test, y_test)


def load_fashion_mnist_from_local(
    datasets_dir: Path,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load Fashion-MNIST dataset from local IDX files

    Args:
        datasets_dir: Directory containing the dataset files

    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    fashion_dir = datasets_dir / "fashion"

    train_images_path = fashion_dir / "train-images-idx3-ubyte.gz"
    train_labels_path = fashion_dir / "train-labels-idx1-ubyte.gz"
    test_images_path = fashion_dir / "t10k-images-idx3-ubyte.gz"
    test_labels_path = fashion_dir / "t10k-labels-idx1-ubyte.gz"

    # Check for .gz files first, then plain files
    for path in [
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
    ]:
        if not path.exists():
            # Try without .gz extension
            plain_path = Path(str(path).replace(".gz", ""))
            if not plain_path.exists():
                raise FileNotFoundError(
                    f"Fashion-MNIST file not found: {path} or {plain_path}"
                )

    logger.info(f"Loading Fashion-MNIST from local files in: {fashion_dir}")

    # Helper function to read IDX file (gzipped or plain)
    def read_idx_file(file_path):
        if file_path.suffix == ".gz":
            with gzip.open(file_path, "rb") as f:
                return f.read()
        else:
            with open(file_path, "rb") as f:
                return f.read()

    # Load training images
    train_images_path = (
        train_images_path
        if train_images_path.exists()
        else Path(str(train_images_path).replace(".gz", ""))
    )
    data = read_idx_file(train_images_path)
    magic, num_images, rows, cols = struct.unpack(">IIII", data[:16])
    X_train = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, rows, cols)

    # Load training labels
    train_labels_path = (
        train_labels_path
        if train_labels_path.exists()
        else Path(str(train_labels_path).replace(".gz", ""))
    )
    data = read_idx_file(train_labels_path)
    magic, num_labels = struct.unpack(">II", data[:8])
    y_train = np.frombuffer(data[8:], dtype=np.uint8)

    # Load test images
    test_images_path = (
        test_images_path
        if test_images_path.exists()
        else Path(str(test_images_path).replace(".gz", ""))
    )
    data = read_idx_file(test_images_path)
    magic, num_images, rows, cols = struct.unpack(">IIII", data[:16])
    X_test = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, rows, cols)

    # Load test labels
    test_labels_path = (
        test_labels_path
        if test_labels_path.exists()
        else Path(str(test_labels_path).replace(".gz", ""))
    )
    data = read_idx_file(test_labels_path)
    magic, num_labels = struct.unpack(">II", data[:8])
    y_test = np.frombuffer(data[8:], dtype=np.uint8)

    # Add channel dimension for CNN compatibility (28, 28) -> (28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    logger.info(f"Fashion-MNIST loaded: Train {X_train.shape}, Test {X_test.shape}")
    return (X_train, y_train), (X_test, y_test)


def load_fashion_mnist_from_tensorflow() -> (
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
):
    """
    Load Fashion-MNIST dataset from TensorFlow/Keras

    Returns:
        Tuple of ((X_train, y_train), (X_test, y_test))
    """
    logger.info("Loading Fashion-MNIST from TensorFlow/Keras...")

    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Add channel dimension for CNN compatibility (28, 28) -> (28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    logger.info(
        f"Fashion-MNIST loaded from TensorFlow: Train {X_train.shape}, Test {X_test.shape}"
    )
    return (X_train, y_train), (X_test, y_test)


class FashionMNISTLoader(DataLoader):
    """Fashion-MNIST dataset loader and preprocessor"""

    def __init__(
        self,
        validation_split: float = 0.2,
        use_local: bool = True,
        prefer_tar: bool = True,
        resize_to: Tuple[int, int] = None,
    ):
        """
        Initialize Fashion-MNIST loader

        Args:
            validation_split: Fraction of training data to use for validation
            use_local: Whether to use local files (True) or TensorFlow (False)
            prefer_tar: Whether to prefer tar.gz format over individual files
            resize_to: Optional tuple (height, width) to resize images
        """
        # Set Fashion-MNIST specific attributes before calling parent
        self.use_local = use_local
        self.prefer_tar = prefer_tar
        self.resize_to = resize_to

        # Fashion-MNIST metadata
        self.num_classes = 10
        self.class_names = [
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
        ]

        # Set input shape based on resize option
        if resize_to:
            self.input_shape = (*resize_to, 1)
        else:
            self.input_shape = (28, 28, 1)

        # Initialize parent class
        super().__init__("fashion_mnist", validation_split)

        logger.info(
            f"Fashion-MNIST specific settings: local={use_local}, tar={prefer_tar}, resize={resize_to}"
        )

    def _get_dataset_config(self) -> Dict[str, Any]:
        """Get Fashion-MNIST specific configuration"""
        return {
            "name": "Fashion-MNIST",
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "class_names": self.class_names,
        }

    def load_and_preprocess(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess Fashion-MNIST dataset

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Loading Fashion-MNIST dataset...")

        # Load data based on preference
        try:
            if self.use_local:
                if self.prefer_tar:
                    # Try tar.gz first (consistent with CIFAR format)
                    try:
                        (X_train_full, y_train_full), (X_test, y_test) = (
                            load_fashion_mnist_from_tar(DATASETS_DIR)
                        )
                        logger.info("Loaded from tar.gz format (consistent with CIFAR)")
                    except FileNotFoundError:
                        logger.info(
                            "tar.gz not found, falling back to individual files..."
                        )
                        (X_train_full, y_train_full), (X_test, y_test) = (
                            load_fashion_mnist_from_local(DATASETS_DIR)
                        )
                else:
                    # Try individual files first
                    try:
                        (X_train_full, y_train_full), (X_test, y_test) = (
                            load_fashion_mnist_from_local(DATASETS_DIR)
                        )
                    except FileNotFoundError:
                        logger.info("Individual files not found, trying tar.gz...")
                        (X_train_full, y_train_full), (X_test, y_test) = (
                            load_fashion_mnist_from_tar(DATASETS_DIR)
                        )
            else:
                (X_train_full, y_train_full), (X_test, y_test) = (
                    load_fashion_mnist_from_tensorflow()
                )
        except Exception as e:
            logger.warning(f"Failed to load from preferred source: {e}")
            logger.info("Falling back to TensorFlow/Keras...")
            (X_train_full, y_train_full), (X_test, y_test) = (
                load_fashion_mnist_from_tensorflow()
            )

        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = self.split_train_validation(
            X_train_full, y_train_full
        )

        # Resize images if requested
        if self.resize_to:
            X_train = self._resize_images(X_train, self.resize_to)
            X_val = self._resize_images(X_val, self.resize_to)
            X_test = self._resize_images(X_test, self.resize_to)
            logger.info(f"Images resized to {self.resize_to}")

        # Normalize data using parent method
        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)

        # Encode labels using parent method
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)

        logger.info("Fashion-MNIST data preprocessing completed")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _resize_images(
        self, images: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize images to target size

        Args:
            images: Input images array
            target_size: Target (height, width)

        Returns:
            Resized images array
        """
        resized_images = tf.image.resize(images, target_size, method="bilinear")
        return resized_images.numpy()

    def _get_additional_info(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Get Fashion-MNIST specific additional information"""
        return {
            "original_size": (28, 28),
            "resized_to": self.resize_to,
            "use_local": self.use_local,
            "prefer_tar": self.prefer_tar,
        }

    def _get_augmentation_function(self):
        """Get Fashion-MNIST specific augmentation function"""

        def fashion_mnist_augment(image, label):
            # Fashion-MNIST specific augmentations
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            # Small rotation for fashion items (90-degree rotations work well for clothing)
            image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            return image, label

        return fashion_mnist_augment


def create_fashion_mnist_data_generators(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32,
    augment_training: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Legacy function for backward compatibility
    Create TensorFlow data generators for Fashion-MNIST training and validation

    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        batch_size: Batch size for training
        augment_training: Whether to apply data augmentation

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create a temporary loader instance for the method
    temp_loader = FashionMNISTLoader()
    return temp_loader.create_data_generators(
        X_train, y_train, X_val, y_val, batch_size, augment_training
    )


# Example usage and testing
if __name__ == "__main__":
    # Test Fashion-MNIST loader with local files
    print("Testing Fashion-MNIST loader with local files...")
    try:
        fashion_loader = FashionMNISTLoader(validation_split=0.2, use_local=True)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            fashion_loader.load_and_preprocess()
        )

        # Print dataset info
        info = fashion_loader.get_data_info(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        print("\nFashion-MNIST Dataset Info (Local Files):")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test data generators
        train_ds, val_ds = create_fashion_mnist_data_generators(
            X_train, y_train, X_val, y_val, batch_size=32
        )
        print(f"\nTrain dataset: {train_ds}")
        print(f"Validation dataset: {val_ds}")

        print("\nFashion-MNIST local files test completed successfully!")

    except Exception as e:
        print(f"Error testing Fashion-MNIST local files: {e}")

    # Test Fashion-MNIST loader with TensorFlow
    print("\n" + "=" * 50)
    print("Testing Fashion-MNIST loader with TensorFlow...")
    try:
        fashion_loader_tf = FashionMNISTLoader(validation_split=0.2, use_local=False)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            fashion_loader_tf.load_and_preprocess()
        )

        # Print dataset info
        info = fashion_loader_tf.get_data_info(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        print("\nFashion-MNIST Dataset Info (TensorFlow):")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\nFashion-MNIST TensorFlow test completed successfully!")

    except Exception as e:
        print(f"Error testing Fashion-MNIST TensorFlow: {e}")

    # Test Fashion-MNIST loader with resizing
    print("\n" + "=" * 50)
    print("Testing Fashion-MNIST loader with resizing to 32x32...")
    try:
        fashion_loader_resized = FashionMNISTLoader(
            validation_split=0.2, use_local=False, resize_to=(32, 32)
        )
        X_train, X_val, X_test, y_train, y_val, y_test = (
            fashion_loader_resized.load_and_preprocess()
        )

        # Print dataset info
        info = fashion_loader_resized.get_data_info(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        print("\nFashion-MNIST Dataset Info (Resized to 32x32):")
        for key, value in info.items():
            print(f"  {key}: {value}")

        print("\nFashion-MNIST resizing test completed successfully!")

    except Exception as e:
        print(f"Error testing Fashion-MNIST resizing: {e}")
