"""
Baseline CNN Model implementation for CNN Image Classification Project
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineCNNModel(BaseModel):
    """Baseline CNN model implementation"""

    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        weights: str = None,
        include_top: bool = False,
        dropout_rate: float = 0.5,
        conv_filters: Tuple[int, ...] = (32, 64, 128, 256),
        dense_units: Tuple[int, ...] = (512, 256),
    ):
        super().__init__(
            model_name="BaselineCNN",
            num_classes=num_classes,
            input_shape=input_shape,
            weights=weights,
            include_top=include_top,
        )

        self.dropout_rate = dropout_rate
        self.conv_filters = conv_filters
        self.dense_units = dense_units

        self.model_info.update(
            {
                "architecture": "BaselineCNN",
                "depth": len(conv_filters) * 2 + len(dense_units) + 1,
                "dropout_rate": dropout_rate,
                "conv_filters": conv_filters,
                "dense_units": dense_units,
            }
        )

        logger.info(f"Baseline CNN model initialized for {num_classes} classes")

    def build_model(self) -> Model:
        """Build Baseline CNN model using configured architecture parameters"""
        logger.info(f"Building Baseline CNN model with {len(self.conv_filters)} conv layers...")

        inputs = keras.Input(shape=self.input_shape, name="input")
        x = inputs

        for i, filters in enumerate(self.conv_filters):
            x = layers.Conv2D(
                filters,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_initializer="he_normal",
                name=f"conv{i+1}",
            )(x)
            x = layers.MaxPooling2D((2, 2), name=f"pool{i+1}")(x)
            x = layers.Dropout(0.25, name=f"dropout{i+1}")(x)

        x = layers.Flatten(name="flatten")(x)

        for i, units in enumerate(self.dense_units):
            x = layers.Dense(
                units,
                activation="relu",
                kernel_initializer="he_normal",
                name=f"dense{i+1}",
            )(x)
            x = layers.Dropout(self.dropout_rate, name=f"dense_dropout{i+1}")(x)

        if self.num_classes == 2:
            outputs = layers.Dense(
                1,
                activation="sigmoid",
                kernel_initializer="glorot_uniform",
                name="predictions",
            )(x)
        else:
            outputs = layers.Dense(
                self.num_classes,
                activation="softmax",
                kernel_initializer="glorot_uniform",
                name="predictions",
            )(x)

        model = Model(
            inputs=inputs,
            outputs=outputs,
            name=f"BaselineCNN_{len(self.conv_filters)}Conv_{len(self.dense_units)}Dense",
        )

        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

        self.model_info.update(
            {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),
                "architecture_type": f"Configurable CNN ({len(self.conv_filters)} conv + {len(self.dense_units)} dense)",
            }
        )

        logger.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")

        self.model = model
        return model

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for saving/loading"""
        config = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "dropout_rate": self.dropout_rate,
            "conv_filters": self.conv_filters,
            "dense_units": self.dense_units,
            "model_info": self.model_info,
        }
        return config

    def compile_model(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        loss: str = None,
        metrics: list = None,
    ) -> None:
        """Compile the model with specified parameters"""
        if self.model is None:
            raise ValueError("Model must be built before compilation")

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
        logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")


def create_baseline_cnn(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    architecture: str = "small",
) -> BaselineCNNModel:
    """Factory function to create baseline CNN with predefined architectures"""
    architectures = {
        "small": {"conv_filters": (32, 64), "dense_units": (256,), "dropout_rate": 0.3},
        "medium": {
            "conv_filters": (32, 64, 128),
            "dense_units": (512, 256),
            "dropout_rate": 0.5,
        },
        "large": {
            "conv_filters": (32, 64, 128, 256),
            "dense_units": (1024, 512, 256),
            "dropout_rate": 0.6,
        },
    }

    config = architectures.get(architecture, architectures["medium"])
    return BaselineCNNModel(num_classes=num_classes, input_shape=input_shape, **config)


if __name__ == "__main__":
    print("Testing Baseline CNN model...")

    try:
        model = BaselineCNNModel(
            num_classes=10,
            input_shape=(32, 32, 3),
            dropout_rate=0.5,
            conv_filters=(32, 64, 128),
            dense_units=(512, 256),
        )

        keras_model = model.build_model()

        model.compile_model(
            optimizer="adam",
            learning_rate=0.001,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("\nModel configuration:")
        config = model.get_model_config()
        for key, value in config.items():
            if key != "model_info":
                print(f"  {key}: {value}")

        print("\nBaseline CNN model test completed successfully!")

        print("\nTesting factory function...")
        small_model = create_baseline_cnn(num_classes=10, architecture="small")
        medium_model = create_baseline_cnn(num_classes=10, architecture="medium")
        large_model = create_baseline_cnn(num_classes=10, architecture="large")

        print(f"Small model filters: {small_model.conv_filters}")
        print(f"Medium model filters: {medium_model.conv_filters}")
        print(f"Large model filters: {large_model.conv_filters}")

        print("Factory function test completed successfully!")

    except Exception as e:
        print(f"Error testing Baseline CNN model: {e}")
