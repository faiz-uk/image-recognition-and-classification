"""
MobileNet Model implementation for CNN Image Classification Project
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileNetModel(BaseModel):
    """MobileNet model implementation with transfer learning"""

    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        weights: str = "imagenet",
        include_top: bool = False,
        dropout_rate: float = 0.5,
        hidden_units: int = 512,
        freeze_base: bool = True,
        alpha: float = 1.0,
        depth_multiplier: int = 1,
        trainable_params: str = "top_layers",  # Added missing parameter
    ):
        super().__init__(
            model_name="MobileNet",
            num_classes=num_classes,
            input_shape=input_shape,
            weights=weights,
            include_top=include_top,
        )

        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.freeze_base = freeze_base
        self.alpha = alpha
        self.depth_multiplier = depth_multiplier
        self.trainable_params = trainable_params  # Store the parameter

        self.model_info.update(
            {
                "architecture": "MobileNet",
                "alpha": alpha,
                "depth_multiplier": depth_multiplier,
                "dropout_rate": dropout_rate,
                "hidden_units": hidden_units,
                "freeze_base": freeze_base,
            }
        )

        logger.info(f"MobileNet model initialized for {num_classes} classes")

    def build_model(self) -> Model:
        """Build MobileNet model with custom classification head"""
        logger.info("Building MobileNet model...")

        inputs = keras.Input(shape=self.input_shape, name="input")

        # Handle grayscale to RGB conversion without Lambda layers
        if self.input_shape[2] == 1:  # Grayscale input
            # Repeat the single channel 3 times using native operations
            x = layers.Conv2D(3, (1, 1), activation='linear', use_bias=False, 
                            kernel_initializer='ones', trainable=False)(inputs)
        else:
            x = inputs

        # Create base model with proper input handling
        if self.input_shape[2] == 1:
            # For grayscale, we need RGB input shape for pretrained weights
            self.base_model = MobileNet(
                weights=self.weights,
                include_top=False,
                input_shape=(self.input_shape[0], self.input_shape[1], 3),
                pooling=None,
                alpha=self.alpha,
            )
        else:
            self.base_model = MobileNet(
                weights=self.weights,
                include_top=False,
                input_shape=self.input_shape,
                pooling=None,
                alpha=self.alpha,
            )

        # Apply base model
        x = self.base_model(x)

        # Build complete model
        self.model = keras.Model(inputs=inputs, outputs=x, name="mobilenet_base")
        
        # Add custom classification head
        self.model = self.add_classification_head(
            base_model=self.model,
            dropout_rate=self.dropout_rate,
            hidden_units=self.hidden_units,
        )

        if self.freeze_base:
            self.freeze_base_layers(trainable=False)

        logger.info(f"MobileNet model built successfully")
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
            name="mobilenet_feature_extractor",
        )

        return feature_extractor

    def fine_tune_top_layers(
        self, num_layers: int = 10, fine_tune_lr: float = None
    ) -> None:
        """Fine-tune the top N layers of MobileNet"""
        if self.base_model is None:
            raise ValueError("Model must be built before fine-tuning")

        self.fine_tune(unfreeze_layers=num_layers, fine_tune_lr=fine_tune_lr)
        logger.info(f"Fine-tuning enabled for top {num_layers} layers")


def create_mobilenet(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    alpha: float = 1.0,
    **kwargs
) -> MobileNetModel:
    """Factory function to create MobileNet model"""
    return MobileNetModel(
        num_classes=num_classes, input_shape=input_shape, alpha=alpha, **kwargs
    )


def create_mobilenet_light(
    num_classes: int, input_shape: Tuple[int, int, int] = (32, 32, 3), **kwargs
) -> MobileNetModel:
    """Factory function to create lightweight MobileNet model"""
    return MobileNetModel(
        num_classes=num_classes,
        input_shape=input_shape,
        alpha=0.5,
        dropout_rate=0.3,
        hidden_units=256,
        **kwargs
    )


def create_mobilenet_standard(
    num_classes: int, input_shape: Tuple[int, int, int] = (32, 32, 3), **kwargs
) -> MobileNetModel:
    """Factory function to create standard MobileNet model"""
    return MobileNetModel(
        num_classes=num_classes,
        input_shape=input_shape,
        alpha=1.0,
        dropout_rate=0.5,
        hidden_units=512,
        **kwargs
    )


if __name__ == "__main__":
    print("Testing MobileNet model...")

    model = create_mobilenet(
        num_classes=10, input_shape=(32, 32, 3), alpha=1.0, dropout_rate=0.5
    )

    keras_model = model.build_and_compile(optimizer="adam", learning_rate=0.001)

    feature_extractor = model.get_feature_extractor()
    print(f"Feature extractor output shape: {feature_extractor.output_shape}")

    print("\nTesting factory functions...")
    light_model = create_mobilenet_light(num_classes=10)
    standard_model = create_mobilenet_standard(num_classes=10)

    print(f"Light model alpha: {light_model.alpha}")
    print(f"Standard model alpha: {standard_model.alpha}")

    print("\nMobileNet model testing completed successfully!")
