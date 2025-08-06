"""
Base Model class for CNN Image Classification Project
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING_CONFIG
from utils.helpers import save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all CNN models"""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        weights: str = "imagenet",
        include_top: bool = False,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.weights = weights
        self.include_top = include_top

        self.model = None
        self.base_model = None

        self.training_config = TRAINING_CONFIG.copy()

        self.model_info = {
            "model_name": model_name,
            "num_classes": num_classes,
            "input_shape": input_shape,
            "weights": weights,
            "include_top": include_top,
        }

        logger.info(f"Initialized {model_name} for {num_classes} classes")

    @abstractmethod
    def build_model(self) -> Model:
        """Build the model architecture"""
        pass

    def add_classification_head(
        self,
        base_model: Model,
        dropout_rate: float = 0.5,
        hidden_units: int = 512,
        activation: str = "relu",
    ) -> Model:
        """Add classification head to base model"""
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dense(
            hidden_units, activation=activation, name="dense_hidden"
        )(x)
        x = layers.Dropout(dropout_rate, name="dropout")(x)

        if self.num_classes == 2:
            predictions = layers.Dense(
                1, activation="sigmoid", name="predictions"
            )(x)
        else:
            predictions = layers.Dense(
                self.num_classes, activation="softmax", name="predictions"
            )(x)

        model = Model(inputs=base_model.input, outputs=predictions, name=f"{self.model_name}_with_head")
        return model

    def compile_model(
        self,
        optimizer: str = "adam",
        learning_rate: float = None,
        loss: str = None,
        metrics: list = None,
    ) -> None:
        """Compile the model"""
        if self.model is None:
            raise ValueError("Model must be built before compilation")

        if learning_rate is None:
            learning_rate = self.training_config.get("learning_rate", 0.001)

        if loss is None:
            loss = "binary_crossentropy" if self.num_classes == 2 else "categorical_crossentropy"

        if metrics is None:
            metrics = ["accuracy"]

        if optimizer.lower() == "adam":
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == "sgd":
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == "rmsprop":
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        logger.info(f"Model compiled with {optimizer} (lr={learning_rate})")

    def freeze_base_layers(self, trainable: bool = False) -> None:
        """Freeze or unfreeze base model layers"""
        if self.base_model is None:
            logger.warning("No base model found to freeze/unfreeze")
            return

        self.base_model.trainable = trainable
        logger.info(f"Base model layers {'unfrozen' if trainable else 'frozen'}")

    def fine_tune(self, unfreeze_layers: int = 10, fine_tune_lr: float = None) -> None:
        """Fine-tune top layers of the model"""
        if self.base_model is None:
            raise ValueError("No base model available for fine-tuning")

        self.base_model.trainable = True

        for layer in self.base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        if fine_tune_lr is None:
            fine_tune_lr = self.training_config.get("fine_tune_lr", 1e-5)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss="binary_crossentropy" if self.num_classes == 2 else "categorical_crossentropy",
            metrics=["accuracy"],
        )

        logger.info(f"Fine-tuning enabled for top {unfreeze_layers} layers (lr={fine_tune_lr})")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information"""
        if self.model is None:
            raise ValueError("Model must be built before getting summary")

        total_params = self.model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
        }

    def save_model(self, filepath: str, save_format: str = "tf") -> None:
        """Save the model"""
        if self.model is None:
            raise ValueError("Model must be built before saving")

        save_model(self.model, filepath, save_format)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

    def predict_batch(self, X: tf.Tensor) -> tf.Tensor:
        """Make predictions on a batch of data"""
        if self.model is None:
            raise ValueError("Model must be built before making predictions")
        return self.model.predict(X)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "weights": self.weights,
            "include_top": self.include_top,
            "model_info": self.model_info,
        }
