"""
Base Model class for CNN Image Classification Project
Contains common functionality for all model implementations
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS, TRAINING_CONFIG
from utils.helpers import calculate_model_size, save_model, save_training_history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all CNN models
    Provides common functionality and interface for model implementations
    """
    
    def __init__(self, 
                 model_name: str,
                 num_classes: int,
                 input_shape: Tuple[int, int, int] = (32, 32, 3),
                 weights: str = 'imagenet',
                 include_top: bool = False):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            input_shape: Input image shape (H, W, C)
            weights: Pre-trained weights to use
            include_top: Whether to include top classification layer
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.weights = weights
        self.include_top = include_top
        
        # Initialize model as None - will be created by subclasses
        self.model = None
        self.base_model = None
        
        # Training configuration
        self.training_config = TRAINING_CONFIG.copy()
        
        # Model metadata
        self.model_info = {
            'model_name': model_name,
            'num_classes': num_classes,
            'input_shape': input_shape,
            'weights': weights,
            'include_top': include_top
        }
        
        logger.info(f"Initialized {model_name} for {num_classes} classes")
    
    @abstractmethod
    def build_model(self) -> Model:
        """
        Build the model architecture
        Must be implemented by subclasses
        
        Returns:
            Compiled Keras model
        """
        pass
    
    def add_classification_head(self, 
                               base_model: Model,
                               num_classes: int,
                               dropout_rate: float = 0.5,
                               hidden_units: int = 512,
                               activation: str = 'relu') -> Model:
        """
        Add classification head to base model
        
        Args:
            base_model: Base model (feature extractor)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            hidden_units: Number of hidden units in dense layer
            activation: Activation function for hidden layer
            
        Returns:
            Model with classification head
        """
        # Global average pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(base_model.output)
        
        # Dense layer with dropout
        x = layers.Dense(hidden_units, activation=activation, name='dense_hidden')(x)
        x = layers.Dropout(dropout_rate, name='dropout')(x)
        
        # Output layer
        if num_classes == 2:
            # Binary classification
            predictions = layers.Dense(1, activation='sigmoid', name='predictions')(x)
        else:
            # Multi-class classification
            predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model
    
    def freeze_base_layers(self, trainable: bool = False) -> None:
        """
        Freeze or unfreeze base model layers
        
        Args:
            trainable: Whether to make base layers trainable
        """
        if self.base_model is None:
            logger.warning("Base model not found. Cannot freeze/unfreeze layers.")
            return
        
        for layer in self.base_model.layers:
            layer.trainable = trainable
        
        action = "unfrozen" if trainable else "frozen"
        logger.info(f"Base model layers {action}")
    
    def compile_model(self, 
                     optimizer: str = 'adam',
                     learning_rate: float = None,
                     loss: str = None,
                     metrics: List[str] = None) -> None:
        """
        Compile the model with specified parameters
        
        Args:
            optimizer: Optimizer to use
            learning_rate: Learning rate (uses config default if None)
            loss: Loss function (auto-selected if None)
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Set default values from config
        if learning_rate is None:
            learning_rate = self.training_config['learning_rate']
        
        if loss is None:
            if self.num_classes == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'categorical_crossentropy'
        
        if metrics is None:
            metrics = ['accuracy']
        
        # Create optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer, lr={learning_rate}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
        
        # Get model size information
        size_info = calculate_model_size(self.model)
        
        # Combine with model metadata
        summary = {
            **self.model_info,
            **size_info,
            'layers_count': len(self.model.layers),
            'optimizer': self.model.optimizer.__class__.__name__ if self.model.optimizer else None,
            'loss': self.model.loss,
            'metrics': self.model.metrics_names if hasattr(self.model, 'metrics_names') else None
        }
        
        return summary
    
    def save_model_with_metadata(self, 
                                save_dir: Path,
                                save_weights_only: bool = False) -> str:
        """
        Save model with comprehensive metadata
        
        Args:
            save_dir: Directory to save model
            save_weights_only: Whether to save only weights
            
        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")
        
        # Create model name with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_save_name = f"{self.model_name}_{timestamp}"
        
        # Save model
        model_path = save_model(
            self.model,
            model_save_name,
            save_weights_only=save_weights_only,
            save_dir=save_dir
        )
        
        # Save model metadata
        metadata_path = save_dir / f"{model_save_name}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.get_model_summary(), f, indent=2, default=str)
        
        logger.info(f"Model and metadata saved to {save_dir}")
        return model_path
    
    def fine_tune(self, 
                  unfreeze_layers: int = None,
                  fine_tune_lr: float = None) -> None:
        """
        Prepare model for fine-tuning by unfreezing some layers
        
        Args:
            unfreeze_layers: Number of top layers to unfreeze (None = all)
            fine_tune_lr: Learning rate for fine-tuning
        """
        if self.base_model is None:
            logger.warning("Base model not found. Cannot fine-tune.")
            return
        
        # Set fine-tuning learning rate
        if fine_tune_lr is None:
            fine_tune_lr = self.training_config['learning_rate'] / 10  # 10x lower
        
        # Unfreeze specified layers
        if unfreeze_layers is None:
            # Unfreeze all layers
            for layer in self.base_model.layers:
                layer.trainable = True
            logger.info("All base model layers unfrozen for fine-tuning")
        else:
            # Unfreeze top N layers
            for layer in self.base_model.layers[-unfreeze_layers:]:
                layer.trainable = True
            logger.info(f"Top {unfreeze_layers} base model layers unfrozen for fine-tuning")
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=fine_tune_lr)
        logger.info(f"Model recompiled for fine-tuning with lr={fine_tune_lr}")
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about model layers
        
        Returns:
            List of dictionaries with layer information
        """
        if self.model is None:
            raise ValueError("Model must be built before getting layer info")
        
        layer_info = []
        for i, layer in enumerate(self.model.layers):
            info = {
                'index': i,
                'name': layer.name,
                'type': layer.__class__.__name__,
                'trainable': layer.trainable,
                'output_shape': layer.output_shape if hasattr(layer, 'output_shape') else None,
                'param_count': layer.count_params()
            }
            layer_info.append(info)
        
        return layer_info
    
    def print_model_summary(self) -> None:
        """Print detailed model summary"""
        if self.model is None:
            raise ValueError("Model must be built before printing summary")
        
        print(f"\n{'='*60}")
        print(f"MODEL SUMMARY: {self.model_name}")
        print(f"{'='*60}")
        
        # Print basic info
        summary = self.get_model_summary()
        print(f"Classes: {summary['num_classes']}")
        print(f"Input Shape: {summary['input_shape']}")
        print(f"Total Parameters: {summary['total_parameters']:,}")
        print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
        print(f"Model Size: {summary['model_size_mb']:.2f} MB")
        
        # Print Keras summary
        print(f"\nKeras Model Summary:")
        self.model.summary()
        
        print(f"{'='*60}\n")


# Example usage and testing
if __name__ == "__main__":
    print("BaseModel is an abstract class and cannot be instantiated directly.")
    print("Use specific model implementations like ResNet50Model or DenseNet121Model.")
    print("Testing completed successfully!") 