"""
InceptionV3 Model implementation for CNN Image Classification Project
Implements InceptionV3 architecture with transfer learning capabilities
"""

import os
import sys
import logging
from typing import Tuple, Dict, Any, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionV3

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel
from config import MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InceptionV3Model(BaseModel):
    """
    InceptionV3 model implementation

    InceptionV3 features:
    - Inception modules with factorized convolutions
    - Efficient architecture with reduced parameters
    - Excellent performance on image classification
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
    ):
        """
        Initialize InceptionV3 model

        Args:
            num_classes: Number of output classes
            input_shape: Input image shape (H, W, C)
            weights: Pre-trained weights ('imagenet' or None)
            include_top: Whether to include top classification layer
            dropout_rate: Dropout rate for regularization
            hidden_units: Number of hidden units in classification head
            trainable_params: 'all', 'top_layers', or 'none'
        """
        super().__init__(
            model_name="InceptionV3",
            num_classes=num_classes,
            input_shape=input_shape,
            weights=weights,
            include_top=include_top,
        )

        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.trainable_params = trainable_params

        # Update model info with InceptionV3 specific parameters
        self.model_info.update(
            {
                "architecture": "InceptionV3",
                "depth": "48 layers",
                "parameters": "~23.8M",
                "dropout_rate": dropout_rate,
                "hidden_units": hidden_units,
                "trainable_params": trainable_params,
                "inception_modules": True,
                "factorized_convolutions": True,
            }
        )

        logger.info(f"InceptionV3 model initialized for {num_classes} classes")

    def build_model(self) -> Model:
        """
        Build and compile the InceptionV3 model

        Returns:
            Compiled Keras model
        """
        logger.info("Building InceptionV3 model...")

        # Handle grayscale images (convert to 3 channels for ImageNet compatibility)
        if self.input_shape[2] == 1:
            logger.info(
                f"Converting grayscale input {self.input_shape} to 3-channel for InceptionV3 compatibility"
            )
            inputs = keras.Input(shape=self.input_shape, name="input")
            # Convert grayscale to 3-channel by repeating the channel
            x = layers.Lambda(
                lambda x: tf.repeat(x, 3, axis=-1), name="grayscale_to_rgb"
            )(inputs)

            # InceptionV3 requires minimum 75x75 input
            if self.input_shape[0] < 75 or self.input_shape[1] < 75:
                x = layers.Lambda(
                    lambda x: tf.image.resize(x, [75, 75]), name="resize_input"
                )(x)

            # Create base model with converted input
            base_model = InceptionV3(
                weights=self.weights, include_top=False, input_tensor=x, pooling=None
            )
        elif self.input_shape[0] < 75 or self.input_shape[1] < 75:
            logger.info(
                f"Input shape {self.input_shape} too small for InceptionV3, will resize internally"
            )
            inputs = keras.Input(shape=self.input_shape, name="input")
            x = layers.Lambda(
                lambda x: tf.image.resize(x, [75, 75]), name="resize_input"
            )(inputs)
            base_model = InceptionV3(
                weights=self.weights, include_top=False, input_tensor=x, pooling=None
            )
        else:
            # Direct input for appropriate sizes with 3 channels
            base_model = InceptionV3(
                weights=self.weights,
                include_top=self.include_top,
                input_shape=self.input_shape,
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
            # Unfreeze top 20 layers
            for layer in self.base_model.layers[-20:]:
                layer.trainable = True
        elif self.trainable_params == "all":
            self.freeze_base_layers(trainable=True)

        # Add classification head
        if self.input_shape[0] < 75 or self.input_shape[1] < 75:
            # Use base_model output directly since we have custom input
            model = self.add_classification_head(
                base_model, self.num_classes, self.dropout_rate, self.hidden_units
            )
            # Reconstruct with original input
            model = Model(inputs=inputs, outputs=model.output, name="InceptionV3")
        else:
            model = self.add_classification_head(
                base_model, self.num_classes, self.dropout_rate, self.hidden_units
            )

        # Log model summary
        logger.info("InceptionV3 model built successfully")

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
            "model_info": self.model_info,
        }
        return config


# Factory function for easier model creation
def create_inceptionv3(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    weights: str = "imagenet",
) -> InceptionV3Model:
    """
    Factory function to create InceptionV3 model

    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        weights: Pre-trained weights to use

    Returns:
        InceptionV3Model instance
    """
    return InceptionV3Model(
        num_classes=num_classes,
        input_shape=input_shape,
        weights=weights,
        trainable_params="top_layers",
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing InceptionV3 model...")

    try:
        # Test model creation
        model = InceptionV3Model(
            num_classes=10,
            input_shape=(32, 32, 3),
            weights=None,  # Skip download for testing
            dropout_rate=0.5,
            trainable_params="top_layers",
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

        print("\nInceptionV3 model test completed successfully!")

        # Test factory function
        print("\nTesting factory function...")
        factory_model = create_inceptionv3(num_classes=10, weights=None)
        print(f"Factory model class: {factory_model.__class__.__name__}")

        print("Factory function test completed successfully!")

    except Exception as e:
        print(f"Error testing InceptionV3 model: {e}")
        import traceback

        traceback.print_exc()
