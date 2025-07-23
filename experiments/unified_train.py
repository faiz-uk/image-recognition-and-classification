#!/usr/bin/env python3
"""
Unified training script using predefined configurations
Supports all 5 models and 4 datasets with ConfigManager
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from models import (
    BaselineCNNModel, ResNet50Model, DenseNet121Model, InceptionV3Model, MobileNetModel,
    create_baseline_cnn
)
from data import CIFAR10Loader, CIFAR100Loader, FashionMNISTLoader, CelebALoader
from utils.device_utils import setup_gpu, get_optimal_batch_size
from utils.helpers import save_experiment_results, set_random_seed
from config_manager import ConfigManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_model_class(architecture: str):
    """Get the appropriate model class for the given architecture"""
    model_classes = {
        'baseline_cnn': BaselineCNNModel,
        'resnet50': ResNet50Model,
        'densenet121': DenseNet121Model,
        'inceptionv3': InceptionV3Model,
        'mobilenet': MobileNetModel
    }
    
    if architecture.lower() not in model_classes:
        raise ValueError(f"Unsupported architecture: {architecture}. Available: {list(model_classes.keys())}")
    
    return model_classes[architecture.lower()]


def get_data_loader(dataset_type: str, validation_split: float = 0.2, **kwargs):
    """Get the appropriate data loader for the given dataset type"""
    loaders = {
        'cifar10': CIFAR10Loader,
        'cifar100': CIFAR100Loader,
        'fashion_mnist': FashionMNISTLoader,
        'celeba': CelebALoader
    }
    
    if dataset_type not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_type}. Available: {list(loaders.keys())}")
    
    loader_class = loaders[dataset_type]
    
    if dataset_type == 'cifar100':
        return loader_class(validation_split=validation_split, 
                           label_mode=kwargs.get('label_mode', 'fine'))
    elif dataset_type == 'fashion_mnist':
        return loader_class(validation_split=validation_split,
                           use_local=kwargs.get('use_local', True),
                           resize_to=kwargs.get('resize_to', None))
    elif dataset_type == 'celeba':
        return loader_class(validation_split=validation_split,
                           target_attribute=kwargs.get('target_attribute', 'Smiling'),
                           image_size=kwargs.get('image_size', (64, 64)),
                           max_samples=kwargs.get('max_samples', None))
    else:
        return loader_class(validation_split=validation_split)


def train_with_config(config_name: str, custom_overrides: dict = None):
    """Train model using predefined configuration with optional custom overrides"""
    
    config_manager = ConfigManager()
    
    quick_configs = config_manager.create_quick_configs()
    if config_name in quick_configs:
        experiment_config = quick_configs[config_name]
    else:
        parts = config_name.split('_')
        if len(parts) >= 2:
            model_key = parts[0]
            dataset_key = parts[1]
            training_key = parts[2] if len(parts) > 2 else 'standard'
            
            model_dataset_key = f"{model_key}_{dataset_key}"
            
            experiment_config = config_manager.create_experiment_config(
                experiment_name=config_name,
                model_key=model_dataset_key,
                dataset_key=dataset_key,
                training_key=training_key,
                custom_overrides=custom_overrides
            )
        else:
            raise ValueError(f"Invalid config name: {config_name}")
    
    if custom_overrides:
        experiment_config = config_manager._apply_overrides(experiment_config, custom_overrides)
    
    logger.info("Configuration Summary:")
    logger.info(config_manager.get_config_summary(experiment_config))
    
    set_random_seed(42)
    
    gpu_info = setup_gpu()
    logger.info(f"GPU setup: {gpu_info}")
    
    if experiment_config.training.batch_size == 'auto':
        optimal_batch_size = get_optimal_batch_size(
            model_input_shape=experiment_config.model.input_shape,
            model_name=experiment_config.model.architecture
        )
        experiment_config.training.batch_size = optimal_batch_size
        logger.info(f"Auto-selected batch size for {experiment_config.model.architecture}: {optimal_batch_size}")
    
    logger.info(f"Loading {experiment_config.dataset.name} dataset...")
    
    dataset_params = {
        'validation_split': experiment_config.training.validation_split
    }
    
    if experiment_config.dataset.dataset_type == 'cifar100':
        dataset_params['label_mode'] = getattr(experiment_config.dataset, 'label_mode', 'fine')
    elif experiment_config.dataset.dataset_type == 'fashion_mnist':
        dataset_params['use_local'] = True
        dataset_params['resize_to'] = (32, 32) if experiment_config.model.input_shape[:2] == (32, 32) else None
    elif experiment_config.dataset.dataset_type == 'celeba':
        dataset_params['target_attribute'] = 'Smiling'
        dataset_params['image_size'] = experiment_config.model.input_shape[:2]
        dataset_params['max_samples'] = 10000
    
    data_loader = get_data_loader(
        dataset_type=experiment_config.dataset.dataset_type,
        **dataset_params
    )
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_preprocess()
    
    logger.info(f"Creating {experiment_config.model.name} model...")
    
    if experiment_config.model.architecture == 'baseline_cnn':
        num_classes = experiment_config.model.num_classes
        if num_classes <= 10:
            arch_size = 'medium'
        elif num_classes <= 100:
            arch_size = 'large'
        else:
            arch_size = 'large'
        
        model = create_baseline_cnn(
            num_classes=experiment_config.model.num_classes,
            input_shape=experiment_config.model.input_shape,
            architecture=arch_size
        )
        keras_model = model
    else:
        model_class = get_model_class(experiment_config.model.architecture)
        
        model_params = {
            'num_classes': experiment_config.model.num_classes,
            'input_shape': experiment_config.model.input_shape,
            'weights': experiment_config.model.weights,
            'dropout_rate': experiment_config.model.dropout_rate,
            'hidden_units': experiment_config.model.hidden_units
        }
        
        if hasattr(experiment_config.model, 'freeze_base'):
            if experiment_config.model.architecture in ['resnet50', 'densenet121', 'inceptionv3', 'mobilenet']:
                model_params['trainable_params'] = 'top_layers' if experiment_config.model.freeze_base else 'all'
        
        if experiment_config.model.architecture == 'mobilenet':
            model_params['alpha'] = getattr(experiment_config.model, 'alpha', 1.0)
        
        model = model_class(**model_params)
        keras_model = model.build_model()
    
    optimizer_map = {
        'adam': tf.keras.optimizers.Adam(learning_rate=experiment_config.training.learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=experiment_config.training.learning_rate),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=experiment_config.training.learning_rate)
    }
    
    optimizer = optimizer_map.get(experiment_config.training.optimizer.lower(), 
                                 tf.keras.optimizers.Adam(learning_rate=experiment_config.training.learning_rate))
    
    keras_model.compile(
        optimizer=optimizer,
        loss=experiment_config.training.loss,
        metrics=experiment_config.training.metrics
    )
    
    saved_models_dir = Path("saved_models")
    results_dir = Path("results")
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name_clean = experiment_config.model.architecture.lower()
    dataset_name_clean = experiment_config.dataset.dataset_type.lower()
    
    callbacks = [
        ModelCheckpoint(
            filepath=saved_models_dir / f'{model_name_clean}_{dataset_name_clean}_{timestamp}.h5',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=experiment_config.training.patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=experiment_config.training.reduce_lr_factor,
            patience=experiment_config.training.reduce_lr_patience,
            min_lr=experiment_config.training.min_lr,
            verbose=1
        )
    ]
    
    logger.info(f"Starting training with configuration: {experiment_config.experiment_name}")
    logger.info(f"Batch size: {experiment_config.training.batch_size}")
    logger.info(f"Epochs: {experiment_config.training.epochs}")
    
    history = keras_model.fit(
        X_train, y_train,
        batch_size=experiment_config.training.batch_size,
        epochs=experiment_config.training.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy = keras_model.evaluate(X_test, y_test, verbose=0)
    
    config_path = config_manager.save_config(experiment_config)
    logger.info(f"Configuration saved to: {config_path}")
    
    results = {
        'experiment_name': experiment_config.experiment_name,
        'model_name': experiment_config.model.name,
        'dataset': experiment_config.dataset.name,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'epochs_trained': len(history.history['loss']),
        'timestamp': timestamp,
        'config_path': config_path,
        'model_architecture': experiment_config.model.architecture,
        'batch_size': experiment_config.training.batch_size,
        'learning_rate': experiment_config.training.learning_rate
    }
    
    save_experiment_results(results, history, results_dir)
    
    logger.info(f"Training completed! Test accuracy: {test_accuracy:.4f}")
    return keras_model, history, results, experiment_config


def main():
    parser = argparse.ArgumentParser(description='Unified CNN Training Script with ConfigManager')
    parser.add_argument('--config', type=str,
                       help='Configuration name or model_dataset combination (e.g., resnet50_cifar10_standard)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--list-configs', action='store_true',
                       help='List all available predefined configurations')
    parser.add_argument('--show-matrix', action='store_true',
                       help='Show model-dataset compatibility matrix')
    parser.add_argument('--analyze-configs', action='store_true',
                       help='Show comprehensive configuration analysis')
    
    args = parser.parse_args()
    
    # Initialize ConfigManager
    config_manager = ConfigManager()
    
    if args.list_configs:
        # List all available configurations
        logger.info("Available Predefined Configurations:")
        logger.info("=" * 60)
        
        quick_configs = config_manager.create_quick_configs()
        
        # Group by model architecture
        config_groups = {}
        for config_name in quick_configs.keys():
            parts = config_name.split('_')
            model = parts[0]
            if model not in config_groups:
                config_groups[model] = []
            config_groups[model].append(config_name)
        
        for model, configs in sorted(config_groups.items()):
            logger.info(f"\n{model.upper()} Model:")
            for config in sorted(configs):
                experiment_config = quick_configs[config]
                logger.info(f"  • {config:<35} | {experiment_config.dataset.name:<12} | {experiment_config.training.epochs:3d} epochs")
        
        logger.info(f"\nTotal: {len(quick_configs)} predefined configurations available")
        logger.info("\nUsage Examples:")
        logger.info("  python unified_train.py --config resnet50_cifar10_standard")
        logger.info("  python unified_train.py --config mobilenet_fashion_mnist_fast")
        logger.info("  python unified_train.py --config densenet121_cifar100_finetune")
        return
    
    if args.show_matrix:
        # Show compatibility matrix
        logger.info(config_manager.get_configuration_matrix())
        return
    
    if args.analyze_configs:
        # Show comprehensive analysis
        logger.info(config_manager.analyze_all_configurations())
        return
    
    if not args.config:
        # Show help if no config specified
        logger.error("No configuration specified!")
        logger.error("\nUsage:")
        logger.error("  python unified_train.py --config <config_name>")
        logger.error("  python unified_train.py --list-configs         # See all available configs")
        logger.error("  python unified_train.py --show-matrix          # See model-dataset matrix")
        logger.error("  python unified_train.py --analyze-configs      # See detailed analysis")
        logger.error("\nExamples:")
        logger.error("  python unified_train.py --config resnet50_cifar10_standard")
        logger.error("  python unified_train.py --config baseline_cnn_fashion_mnist_fast")
        logger.error("  python unified_train.py --config inceptionv3_celeba_finetune")
        parser.print_help()
        return 1
    
    # Prepare custom overrides
    custom_overrides = {}
    if args.epochs:
        custom_overrides['training'] = custom_overrides.get('training', {})
        custom_overrides['training']['epochs'] = args.epochs
    if args.batch_size:
        custom_overrides['training'] = custom_overrides.get('training', {})
        custom_overrides['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        custom_overrides['training'] = custom_overrides.get('training', {})
        custom_overrides['training']['learning_rate'] = args.learning_rate
    
    try:
        # Train with configuration
        logger.info(f"Starting training with configuration: {args.config}")
        
        if custom_overrides:
            logger.info("Applied custom overrides:")
            for category, overrides in custom_overrides.items():
                for key, value in overrides.items():
                    logger.info(f"  {category}.{key}: {value}")
        
        model, history, results, config = train_with_config(args.config, custom_overrides)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final Results:")
        logger.info(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"  Test Loss: {results['test_loss']:.4f}")
        logger.info(f"  Epochs Trained: {results['epochs_trained']}")
        logger.info(f"  Model: {results['model_architecture']} on {results['dataset']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 