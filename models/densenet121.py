"""
DenseNet121 Model implementation for CNN Image Classification Project
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseNet121Model(BaseModel):
    """DenseNet121 model implementation with transfer learning"""

    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        weights: str = "imagenet",
        include_top: bool = False,
        dropout_rate: float = 0.5,
        hidden_units: int = 512,
        freeze_base: bool = True,
    ):
        super().__init__(
            model_name="DenseNet121",
            num_classes=num_classes,
            input_shape=input_shape,
            weights=weights,
            include_top=include_top,
        )

        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.freeze_base = freeze_base

        self.model_info.update(
            {
                "architecture": "DenseNet121",
                "depth": 121,
                "growth_rate": 32,
                "dropout_rate": dropout_rate,
                "hidden_units": hidden_units,
                "freeze_base": freeze_base,
            }
        )

        logger.info(f"DenseNet121 model initialized for {num_classes} classes")

    def build_model(self) -> Model:
        """Build DenseNet121 model with custom classification head"""
        logger.info("Building DenseNet121 model...")

        if self.input_shape[2] == 1:
            inputs = keras.Input(shape=self.input_shape, name="input")
            x = layers.Lambda(
                lambda x: tf.repeat(x, 3, axis=-1), name="grayscale_to_rgb"
            )(inputs)

            self.base_model = DenseNet121(
                weights=self.weights,
                include_top=False,
                input_tensor=x,
                pooling=None,
            )
        else:
            self.base_model = DenseNet121(
                weights=self.weights,
                include_top=self.include_top,
                input_shape=self.input_shape,
                pooling=None,
            )

        self.model = self.add_classification_head(
            base_model=self.base_model,
            dropout_rate=self.dropout_rate,
            hidden_units=self.hidden_units,
        )

        if self.freeze_base:
            self.freeze_base_layers(trainable=False)

        logger.info(f"DenseNet121 model built successfully")
        logger.info(f"Total parameters: {self.model.count_params():,}")

        return self.model

    def build_and_compile(
        self,
        optimizer: str = "adam",
        learning_rate: float = None,
        loss: str = None,
        metrics: list = None,
    ) -> Model:
        """Build and compile the model in one step"""
        self.build_model()
        self.compile_model(
            optimizer=optimizer, learning_rate=learning_rate, loss=loss, metrics=metrics
        )
        return self.model

    def get_feature_extractor(self) -> Model:
        """Get the feature extractor part of the model"""
        if self.base_model is None:
            raise ValueError("Model must be built before getting feature extractor")

        feature_extractor = Model(
            inputs=self.base_model.input,
            outputs=self.base_model.output,
            name="densenet121_feature_extractor",
        )

        return feature_extractor

    def fine_tune_top_layers(
        self, num_layers: int = 10, fine_tune_lr: float = None
    ) -> None:
        """Fine-tune the top N layers of DenseNet121"""
        if self.base_model is None:
            raise ValueError("Model must be built before fine-tuning")

        self.fine_tune(unfreeze_layers=num_layers, fine_tune_lr=fine_tune_lr)
        logger.info(f"Fine-tuning enabled for top {num_layers} layers")


def create_densenet121_model(
    num_classes: int, input_shape: Tuple[int, int, int] = (32, 32, 3), **kwargs
) -> DenseNet121Model:
    """Factory function to create DenseNet121 model"""
    return DenseNet121Model(num_classes=num_classes, input_shape=input_shape, **kwargs)


if __name__ == "__main__":
    print("Testing DenseNet121 model...")

    model = create_densenet121_model(
        num_classes=10, input_shape=(32, 32, 3), dropout_rate=0.5, hidden_units=512
    )

    keras_model = model.build_and_compile(optimizer="adam", learning_rate=0.001)

    feature_extractor = model.get_feature_extractor()
    print(f"Feature extractor output shape: {feature_extractor.output_shape}")

    print("\nDenseNet121 model testing completed successfully!")
