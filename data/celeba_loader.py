"""
CelebA Data Loader for CNN Image Classification Project
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Optional, Dict, Any, List
import os
import sys
from pathlib import Path
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED
from .cifar_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


class CelebALoader(DataLoader):
    """CelebA dataset loader and preprocessor for image classification"""

    def __init__(
        self,
        validation_split: float = 0.2,
        target_attribute: str = "Smiling",
        image_size: Tuple[int, int] = (64, 64),
        max_samples: Optional[int] = None,
    ):
        self.target_attribute = target_attribute
        self.image_size = image_size
        self.max_samples = max_samples

        self.num_classes = 2
        self.input_shape = (*image_size, 3)
        self.class_names = None

        self.celeba_dir = DATASETS_DIR / "celeba"
        self.images_dir = self.celeba_dir / "img_align_celeba"
        self.attr_file = self.celeba_dir / "list_attr_celeba.txt"

        super().__init__("celeba", validation_split)

    def load_attributes(self) -> pd.DataFrame:
        """Load CelebA attributes file"""
        if not self.attr_file.exists():
            raise FileNotFoundError(f"Attributes file not found: {self.attr_file}")

        with open(self.attr_file, "r") as f:
            lines = f.readlines()

        num_images = int(lines[0].strip())
        attr_names = lines[1].strip().split()

        data = []
        for i in range(2, min(len(lines), num_images + 2)):
            if self.max_samples and len(data) >= self.max_samples:
                break

            parts = lines[i].strip().split()
            if len(parts) >= len(attr_names) + 1:
                image_name = parts[0]
                attributes = [int(x) for x in parts[1 : len(attr_names) + 1]]
                data.append([image_name] + attributes)

        columns = ["image_id"] + attr_names
        df = pd.DataFrame(data, columns=columns)

        for col in attr_names:
            df[col] = (df[col] == 1).astype(int)

        logger.info(f"Loaded {len(df)} image attributes")
        return df

    def load_images(self, image_list: List[str]) -> np.ndarray:
        """Load and preprocess images"""
        images = []

        for i, image_name in enumerate(image_list):
            if i % 1000 == 0:
                logger.info(f"Loaded {i}/{len(image_list)} images")

            image_path = self.images_dir / image_name

            if image_path.exists():
                try:
                    img = Image.open(image_path).convert("RGB")
                    img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    images.append(img_array)
                except Exception as e:
                    logger.warning(f"Error loading image {image_name}: {e}")
                    dummy_img = np.random.randint(
                        0, 255, (*self.image_size, 3), dtype=np.uint8
                    )
                    images.append(dummy_img)
            else:
                dummy_img = np.random.randint(
                    0, 255, (*self.image_size, 3), dtype=np.uint8
                )
                images.append(dummy_img)

        return np.array(images)

    def load_and_preprocess(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess CelebA dataset"""
        logger.info(f"Loading CelebA dataset for '{self.target_attribute}' classification...")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"CelebA images directory not found: {self.images_dir}")

        df_attrs = self.load_attributes()

        if self.target_attribute not in df_attrs.columns:
            available_attrs = [col for col in df_attrs.columns if col != "image_id"]
            raise ValueError(
                f"Attribute '{self.target_attribute}' not found. "
                f"Available attributes: {available_attrs}"
            )

        if self.max_samples and len(df_attrs) > self.max_samples:
            df_attrs = df_attrs.head(self.max_samples)
            logger.info(f"Limited to {self.max_samples} samples")

        image_names = df_attrs["image_id"].tolist()
        labels = df_attrs[self.target_attribute].values

        images = self.load_images(image_names)

        min_len = min(len(images), len(labels))
        images = images[:min_len]
        labels = labels[:min_len]

        logger.info(f"Final dataset loaded: {len(images)} samples")

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
        )

        X_train, X_val, y_train, y_val = self.split_train_validation(
            X_train_full, y_train_full
        )

        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)

        if self.num_classes == 2:
            y_train = np.argmax(y_train, axis=1).astype(np.float32)
            y_val = np.argmax(y_val, axis=1).astype(np.float32)
            y_test = np.argmax(y_test, axis=1).astype(np.float32)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_data_info(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Get dataset information with CelebA specific details"""
        info = super().get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
        
        if y_train.ndim == 1:
            class_distribution = {
                "negative": np.sum(y_train == 0),
                "positive": np.sum(y_train == 1),
            }
        else:
            class_distribution = {
                "negative": np.sum(y_train[:, 0]),
                "positive": np.sum(y_train[:, 1]),
            }

        info.update({
            "target_attribute": self.target_attribute,
            "image_size": self.image_size,
            "max_samples": self.max_samples,
            "class_distribution_train": class_distribution,
        })
        
        return info

    def _get_augmentation_function(self):
        def celeba_augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            return image, label

        return celeba_augment

    def get_available_attributes(self) -> List[str]:
        try:
            df_attrs = self.load_attributes()
            return [col for col in df_attrs.columns if col != "image_id"]
        except Exception:
            return [
                "Smiling", "Male", "Young", "Attractive", "Heavy_Makeup",
                "Eyeglasses", "Bald", "Mustache", "Goatee", "Pale_Skin",
            ]


if __name__ == "__main__":
    celeba_loader = CelebALoader(
        validation_split=0.2,
        target_attribute="Smiling",
        image_size=(64, 64),
        max_samples=100,
    )

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = celeba_loader.load_and_preprocess()
        info = celeba_loader.get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
        print("CelebA Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
