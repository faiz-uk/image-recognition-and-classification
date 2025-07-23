"""
ResNet50 Model implementation for CNN Image Classification Project
Implements ResNet50 architecture with transfer learning capabilities
"""

import os
import sys
import logging
from typing import Tuple, Optional, Union, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras import layers

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel
from config import MODELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResNet50Model(BaseModel):
    """
    ResNet50 model implementation with transfer learning
    
    ResNet50 is a deep convolutional neural network with 50 layers that uses
    residual connections to enable training of very deep networks.
    """
    
    def __init__(self, 
                 num_classes: int,
                 input_shape: Tuple[int, int, int] = (32, 32, 3),
                 weights: str = 'imagenet',
                 include_top: bool = False,
                 dropout_rate: float = 0.5,
                 hidden_units: int = 512,
                 freeze_base: bool = True):
        """
        Initialize ResNet50 model
        
        Args:
            num_classes: Number of output classes
            input_shape: Input image shape (H, W, C)
            weights: Pre-trained weights ('imagenet' or None)
            include_top: Whether to include top classification layer
            dropout_rate: Dropout rate for regularization
            hidden_units: Number of hidden units in dense layer
            freeze_base: Whether to freeze base model layers initially
        """
        super().__init__(
            model_name='ResNet50',
            num_classes=num_classes,
            input_shape=input_shape,
            weights=weights,
            include_top=include_top
        )
        
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.freeze_base = freeze_base
        
        # Update model info with ResNet50 specific parameters
        self.model_info.update({
            'architecture': 'ResNet50',
            'depth': 50,
            'dropout_rate': dropout_rate,
            'hidden_units': hidden_units,
            'freeze_base': freeze_base
        })
        
        logger.info(f"ResNet50 model initialized for {num_classes} classes")
    
    def build_model(self) -> Model:
        """
        Build ResNet50 model with custom classification head
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building ResNet50 model...")
        
        # Handle grayscale images (convert to 3 channels for ImageNet compatibility)
        if self.input_shape[2] == 1:
            logger.info(f"Converting grayscale input {self.input_shape} to 3-channel for ResNet50 compatibility")
            # Create custom input processing for grayscale images
            inputs = keras.Input(shape=self.input_shape, name='input')
            # Convert grayscale to 3-channel by repeating the channel
            x = layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1), name='grayscale_to_rgb')(inputs)
            
            # Create base model with converted input
            self.base_model = ResNet50(
                weights=self.weights,
                include_top=False,
                input_tensor=x,
                pooling=None  # We'll add our own pooling
            )
        else:
            # Standard 3-channel input
            self.base_model = ResNet50(
                weights=self.weights,
                include_top=self.include_top,
                input_shape=self.input_shape,
                pooling=None  # We'll add our own pooling
            )
        
        # Add custom classification head
        self.model = self.add_classification_head(
            base_model=self.base_model,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
            hidden_units=self.hidden_units
        )
        
        # Freeze base model layers if specified
        if self.freeze_base:
            self.freeze_base_layers(trainable=False)
        
        logger.info(f"ResNet50 model built successfully")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def build_and_compile(self, 
                         optimizer: str = 'adam',
                         learning_rate: float = None,
                         loss: str = None,
                         metrics: list = None) -> Model:
        """
        Build and compile the model in one step
        
        Args:
            optimizer: Optimizer to use
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics to track
            
        Returns:
            Compiled model
        """
        # Build model
        self.build_model()
        
        # Compile model
        self.compile_model(
            optimizer=optimizer,
            learning_rate=learning_rate,
            loss=loss,
            metrics=metrics
        )
        
        return self.model
    
    def get_feature_extractor(self) -> Model:
        """
        Get the feature extractor part of the model (without classification head)
        
        Returns:
            Feature extractor model
        """
        if self.base_model is None:
            raise ValueError("Model must be built before getting feature extractor")
        
        # Create feature extractor model
        feature_extractor = Model(
            inputs=self.base_model.input,
            outputs=self.base_model.output,
            name='resnet50_feature_extractor'
        )
        
        return feature_extractor
    
    def get_layer_outputs(self, layer_names: list) -> Dict[str, tf.Tensor]:
        """
        Get outputs from specific layers
        
        Args:
            layer_names: List of layer names to get outputs from
            
        Returns:
            Dictionary mapping layer names to their outputs
        """
        if self.model is None:
            raise ValueError("Model must be built before getting layer outputs")
        
        layer_outputs = {}
        for layer_name in layer_names:
            try:
                layer = self.model.get_layer(layer_name)
                layer_outputs[layer_name] = layer.output
            except ValueError:
                logger.warning(f"Layer '{layer_name}' not found in model")
        
        return layer_outputs
    
    def fine_tune_top_layers(self, 
                            num_layers: int = 10,
                            fine_tune_lr: float = None) -> None:
        """
        Fine-tune the top N layers of ResNet50
        
        Args:
            num_layers: Number of top layers to unfreeze
            fine_tune_lr: Learning rate for fine-tuning
        """
        if self.base_model is None:
            raise ValueError("Model must be built before fine-tuning")
        
        # Unfreeze top layers
        self.fine_tune(
            unfreeze_layers=num_layers,
            fine_tune_lr=fine_tune_lr
        )
        
        logger.info(f"Fine-tuning enabled for top {num_layers} layers")
    
    def get_resnet_blocks(self) -> Dict[str, Model]:
        """
        Get individual ResNet blocks for analysis
        
        Returns:
            Dictionary of ResNet blocks
        """
        if self.base_model is None:
            raise ValueError("Model must be built before getting blocks")
        
        blocks = {}
        
        # Define ResNet50 block names
        block_names = [
            'conv1_relu',
            'conv2_block1_out',
            'conv2_block2_out',
            'conv2_block3_out',
            'conv3_block1_out',
            'conv3_block2_out',
            'conv3_block3_out',
            'conv3_block4_out',
            'conv4_block1_out',
            'conv4_block2_out',
            'conv4_block3_out',
            'conv4_block4_out',
            'conv4_block5_out',
            'conv4_block6_out',
            'conv5_block1_out',
            'conv5_block2_out',
            'conv5_block3_out'
        ]
        
        # Create models for each block
        for block_name in block_names:
            try:
                layer = self.base_model.get_layer(block_name)
                block_model = Model(
                    inputs=self.base_model.input,
                    outputs=layer.output,
                    name=f'resnet50_{block_name}'
                )
                blocks[block_name] = block_model
            except ValueError:
                logger.warning(f"Block '{block_name}' not found in ResNet50")
        
        return blocks
    
    def analyze_model_complexity(self) -> Dict[str, Any]:
        """
        Analyze model complexity and computational requirements
        
        Returns:
            Dictionary with complexity analysis
        """
        if self.model is None:
            raise ValueError("Model must be built before complexity analysis")
        
        # Get basic model info
        summary = self.get_model_summary()
        
        # Calculate FLOPs (approximate)
        # ResNet50 typically requires ~4.1 GFLOPs for ImageNet input
        # Scale based on input size
        input_pixels = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        imagenet_pixels = 224 * 224 * 3
        flops_scale = input_pixels / imagenet_pixels
        estimated_flops = 4.1e9 * flops_scale  # Approximate FLOPs
        
        complexity_info = {
            **summary,
            'estimated_flops': estimated_flops,
            'flops_per_parameter': estimated_flops / summary['total_parameters'],
            'memory_footprint_mb': summary['model_size_mb'],
            'inference_complexity': 'High' if estimated_flops > 1e9 else 'Medium' if estimated_flops > 1e8 else 'Low'
        }
        
        return complexity_info
    
    def print_architecture_summary(self) -> None:
        """Print detailed architecture summary"""
        print(f"\n{'='*70}")
        print(f"RESNET50 ARCHITECTURE SUMMARY")
        print(f"{'='*70}")
        
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        # Print basic info
        complexity = self.analyze_model_complexity()
        print(f"Input Shape: {self.input_shape}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Total Parameters: {complexity['total_parameters']:,}")
        print(f"Trainable Parameters: {complexity['trainable_parameters']:,}")
        print(f"Model Size: {complexity['model_size_mb']:.2f} MB")
        print(f"Estimated FLOPs: {complexity['estimated_flops']:.2e}")
        print(f"Inference Complexity: {complexity['inference_complexity']}")
        
        # Print layer distribution
        layer_info = self.get_layer_info()
        layer_types = {}
        for layer in layer_info:
            layer_type = layer['type']
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        print(f"\nLayer Distribution:")
        for layer_type, count in sorted(layer_types.items()):
            print(f"  {layer_type}: {count}")
        
        print(f"\nBase Model Frozen: {not self.base_model.layers[0].trainable}")
        print(f"Dropout Rate: {self.dropout_rate}")
        print(f"Hidden Units: {self.hidden_units}")
        
        print(f"{'='*70}\n")


# Factory function for easy model creation
def create_resnet50_model(num_classes: int,
                         input_shape: Tuple[int, int, int] = (32, 32, 3),
                         **kwargs) -> ResNet50Model:
    """
    Factory function to create ResNet50 model
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        **kwargs: Additional arguments for ResNet50Model
        
    Returns:
        ResNet50Model instance
    """
    return ResNet50Model(
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing ResNet50 model...")
    
    # Create model for CIFAR-10
    model = create_resnet50_model(
        num_classes=10,
        input_shape=(32, 32, 3),
        dropout_rate=0.5,
        hidden_units=512
    )
    
    # Build and compile model
    keras_model = model.build_and_compile(
        optimizer='adam',
        learning_rate=0.001
    )
    
    # Print architecture summary
    model.print_architecture_summary()
    
    # Test feature extractor
    feature_extractor = model.get_feature_extractor()
    print(f"Feature extractor output shape: {feature_extractor.output_shape}")
    
    # Test complexity analysis
    complexity = model.analyze_model_complexity()
    print(f"Model complexity: {complexity['inference_complexity']}")
    
    print("\nResNet50 model testing completed successfully!") 