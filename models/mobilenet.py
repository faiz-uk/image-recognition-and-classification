"""
MobileNet Model implementation for CNN Image Classification Project
Implements MobileNet architecture with transfer learning capabilities
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel
from config import MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileNetModel(BaseModel):
    """
    MobileNet model implementation

    MobileNet features:
    - Depthwise separable convolutions for efficiency
    - Lightweight architecture suitable for mobile devices
    - Excellent accuracy-to-size ratio
    - Pre-trained on ImageNet
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int, int] = (32, 32, 3),
        weights: str = "imagenet",
        include_top: bool = False,
        dropout_rate: float = 0.5,
        hidden_units: int = 512,
        trainable_params: str = "top_layers",
        alpha: float = 1.0,
    ):
        """
        Initialize MobileNet model

        Args:
            num_classes: Number of output classes
            input_shape: Input image shape (H, W, C)
            weights: Pre-trained weights ('imagenet' or None)
            include_top: Whether to include top classification layer
            dropout_rate: Dropout rate for regularization
            hidden_units: Number of hidden units in classification head
            trainable_params: 'all', 'top_layers', or 'none'
            alpha: Width multiplier for controlling model size
        """
        super().__init__(
            model_name="MobileNet",
            num_classes=num_classes,
            input_shape=input_shape,
            weights=weights,
            include_top=include_top,
        )

        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.trainable_params = trainable_params
        self.alpha = alpha

        # Update model info with MobileNet specific parameters
        self.model_info.update(
            {
                "architecture": "MobileNet",
                "depth": "28 layers",
                "parameters": f"~{4.2 * alpha:.1f}M",
                "dropout_rate": dropout_rate,
                "hidden_units": hidden_units,
                "trainable_params": trainable_params,
                "alpha": alpha,
                "depthwise_separable": True,
                "mobile_optimized": True,
            }
        )

        logger.info(
            f"MobileNet model initialized for {num_classes} classes (alpha={alpha})"
        )

    def build_model(self) -> Model:
        """
        Build and compile the MobileNet model

        Returns:
            Compiled Keras model
        """
        logger.info("Building MobileNet model...")

        # Handle grayscale images (convert to 3 channels for ImageNet compatibility)
        if self.input_shape[2] == 1:
            logger.info(
                f"Converting grayscale input {self.input_shape} to 3-channel for MobileNet compatibility"
            )
            inputs = keras.Input(shape=self.input_shape, name="input")
            # Convert grayscale to 3-channel by repeating the channel
            x = layers.Lambda(
                lambda x: tf.repeat(x, 3, axis=-1), name="grayscale_to_rgb"
            )(inputs)

            # Handle size requirements
            if self.input_shape[0] < 32 or self.input_shape[1] < 32:
                x = layers.Lambda(
                    lambda x: tf.image.resize(x, [32, 32]), name="resize_input"
                )(x)

            # Create base model with converted input
            base_model = MobileNet(
                weights=self.weights,
                include_top=False,
                input_tensor=x,
                alpha=self.alpha,
                pooling=None,
            )
        elif self.input_shape[0] < 32 or self.input_shape[1] < 32:
            logger.info(
                f"Input shape {self.input_shape} too small for MobileNet, will resize internally"
            )
            # Create input layer for small images
            inputs = keras.Input(shape=self.input_shape, name="input")
            # Resize to minimum required size
            x = layers.Lambda(
                lambda x: tf.image.resize(x, [32, 32]), name="resize_input"
            )(inputs)

            # Create base model with resized input
            base_model = MobileNet(
                weights=self.weights,
                include_top=False,
                input_tensor=x,
                alpha=self.alpha,
                pooling=None,
            )
        else:
            # Direct input for appropriate sizes with 3 channels
            base_model = MobileNet(
                weights=self.weights,
                include_top=self.include_top,
                input_shape=self.input_shape,
                alpha=self.alpha,
                pooling=None,
            )
            inputs = base_model.input

        self.base_model = base_model

        # Configure trainable parameters
        if self.trainable_params == "none":
            self.freeze_base_layers(trainable=False)
        elif self.trainable_params == "top_layers":
            # Freeze most layers, unfreeze top layers for fine-tuning
            self.freeze_base_layers(trainable=False)
            # Unfreeze top 15 layers (MobileNet has fewer layers)
            for layer in self.base_model.layers[-15:]:
                layer.trainable = True
        elif self.trainable_params == "all":
            self.freeze_base_layers(trainable=True)

        # Add classification head
        if self.input_shape[0] < 32 or self.input_shape[1] < 32:
            # Use base_model output directly since we have custom input
            model = self.add_classification_head(
                base_model, self.num_classes, self.dropout_rate, self.hidden_units
            )
            # Reconstruct with original input
            model = Model(inputs=inputs, outputs=model.output, name="MobileNet")
        else:
            model = self.add_classification_head(
                base_model, self.num_classes, self.dropout_rate, self.hidden_units
            )

        # Log model summary
        logger.info("MobileNet model built successfully")

        # Calculate model size
        total_params = model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

        self.model_info.update(
            {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),
            }
        )

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        self.model = model
        return model

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration for saving/loading

        Returns:
            Dictionary with model configuration
        """
        config = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "weights": self.weights,
            "dropout_rate": self.dropout_rate,
            "hidden_units": self.hidden_units,
            "trainable_params": self.trainable_params,
            "alpha": self.alpha,
            "model_info": self.model_info,
        }
        return config


# Factory function for easier model creation
def create_mobilenet(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    weights: str = "imagenet",
    alpha: float = 1.0,
) -> MobileNetModel:
    """
    Factory function to create MobileNet model

    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        weights: Pre-trained weights to use
        alpha: Width multiplier (0.25, 0.5, 0.75, 1.0)

    Returns:
        MobileNetModel instance
    """
    return MobileNetModel(
        num_classes=num_classes,
        input_shape=input_shape,
        weights=weights,
        alpha=alpha,
        trainable_params="top_layers",
    )


# Preset configurations for different use cases
def create_mobilenet_light(
    num_classes: int, input_shape: Tuple[int, int, int] = (32, 32, 3)
) -> MobileNetModel:
    """Create lightweight MobileNet (alpha=0.5)"""
    return create_mobilenet(num_classes, input_shape, alpha=0.5)


def create_mobilenet_standard(
    num_classes: int, input_shape: Tuple[int, int, int] = (32, 32, 3)
) -> MobileNetModel:
    """Create standard MobileNet (alpha=1.0)"""
    return create_mobilenet(num_classes, input_shape, alpha=1.0)


# Example usage and testing
if __name__ == "__main__":
    print("Testing MobileNet model...")

    try:
        # Test model creation
        model = MobileNetModel(
            num_classes=10,
            input_shape=(32, 32, 3),
            weights=None,  # Skip download for testing
            dropout_rate=0.5,
            trainable_params="top_layers",
            alpha=1.0,
        )

        # Build model
        keras_model = model.build_model()

        # Compile model
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

        print("\nMobileNet model test completed successfully!")

        # Test factory functions
        print("\nTesting factory functions...")
        standard_model = create_mobilenet_standard(num_classes=10)
        light_model = create_mobilenet_light(num_classes=10)

        print(f"Standard model alpha: {standard_model.alpha}")
        print(f"Light model alpha: {light_model.alpha}")

        print("Factory function test completed successfully!")

    except Exception as e:
        print(f"Error testing MobileNet model: {e}")
        import traceback

        traceback.print_exc()
