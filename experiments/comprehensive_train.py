"""
Comprehensive training script supporting 5 CNN models on 4 datasets
With intelligent parameter optimization and performance prediction
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

from config import (
    get_optimal_config, get_expected_performance, get_training_summary,
    estimate_training_time, validate_config_combination, MODELS, DATASETS
)

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


def get_data_loader(dataset_name: str, **kwargs):
    """Get the appropriate data loader for the given dataset"""
    loaders = {
        'cifar10': CIFAR10Loader,
        'cifar100': CIFAR100Loader,
        'fashion_mnist': FashionMNISTLoader,
        'celeba': CelebALoader
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    loader_class = loaders[dataset_name.lower()]
    
    if dataset_name.lower() == 'cifar100':
        return loader_class(validation_split=kwargs.get('validation_split', 0.2),
                           label_mode=kwargs.get('label_mode', 'fine'))
    elif dataset_name.lower() == 'fashion_mnist':
        resize_to = kwargs.get('resize_to', (32, 32)) if kwargs.get('standardize_size', True) else None
        return loader_class(validation_split=kwargs.get('validation_split', 0.2),
                           use_local=kwargs.get('use_local', True),
                           resize_to=resize_to)
    elif dataset_name.lower() == 'celeba':
        return loader_class(validation_split=kwargs.get('validation_split', 0.2),
                           target_attribute=kwargs.get('target_attribute', 'Smiling'),
                           image_size=kwargs.get('image_size', (64, 64)),
                           max_samples=kwargs.get('max_samples', None))
    else:
        return loader_class(validation_split=kwargs.get('validation_split', 0.2))


def create_model(architecture: str, dataset_info: dict, dataset_name: str, **model_kwargs):
    """Create model with intelligent configuration based on architecture and dataset"""
    
    num_classes = dataset_info['num_classes']
    input_shape = dataset_info['input_shape']
    
    try:
        optimal_config = get_optimal_config(architecture, dataset_name)
        logger.info(f"Using optimal configuration for {architecture} on {dataset_name}")
        logger.info(f"Optimal settings: {optimal_config}")
    except Exception as e:
        logger.warning(f"Could not get optimal config: {e}. Using defaults.")
        optimal_config = {}
    
    if architecture.lower() == 'baseline_cnn':
        if num_classes <= 10:
            arch_size = 'medium'
        elif num_classes <= 100:
            arch_size = 'large'
        else:
            arch_size = 'large'
        
        model = create_baseline_cnn(
            num_classes=num_classes,
            input_shape=input_shape,
            architecture=arch_size
        )
    else:
        model_class = get_model_class(architecture)
        
        model_params = {
            'num_classes': num_classes,
            'input_shape': input_shape,
            'weights': model_kwargs.get('weights', 'imagenet'),
            'dropout_rate': model_kwargs.get('dropout_rate', 0.5),
            'hidden_units': model_kwargs.get('hidden_units', 512)
        }
        
        # Handle trainable parameters for transfer learning models
        trainable_mode = model_kwargs.get('trainable_params', 'top_layers')
        
        if architecture.lower() in ['resnet50', 'densenet121']:
            # These models use freeze_base parameter
            model_params['freeze_base'] = (trainable_mode == 'top_layers')
        elif architecture.lower() == 'inceptionv3':
            # InceptionV3 uses trainable_params parameter
            model_params['trainable_params'] = trainable_mode
        elif architecture.lower() == 'mobilenet':
            # MobileNet uses trainable_params parameter and has alpha
            model_params['trainable_params'] = trainable_mode
            model_params['alpha'] = model_kwargs.get('alpha', 1.0)
        
        model = model_class(**model_params)
    
    return model


def train_model(architecture: str, 
                dataset: str,
                epochs: int = None,
                batch_size: int = None,
                learning_rate: float = None,
                validation_split: float = 0.2,
                save_results: bool = True,
                use_optimal_config: bool = True,
                **kwargs):
    """
    Train a model on a specific dataset with intelligent configuration
    
    Args:
        architecture: Model architecture ('baseline_cnn', 'resnet50', 'densenet121', 'inceptionv3', 'mobilenet')
        dataset: Dataset name ('cifar10', 'cifar100', 'fashion_mnist', 'celeba')
        epochs: Number of training epochs (None = use optimal)
        batch_size: Training batch size (None = use optimal)
        learning_rate: Learning rate (None = use optimal)
        validation_split: Validation split ratio
        save_results: Whether to save results
        use_optimal_config: Whether to use intelligent configuration optimization
        **kwargs: Additional dataset and model specific arguments
    """
    
    logger.info(f"Starting training: {architecture} on {dataset}")
    
    try:
        validation = validate_config_combination(architecture, dataset)
        if validation['warnings']:
            logger.warning("Configuration warnings:")
            for warning in validation['warnings']:
                logger.warning(f"  WARNING: {warning}")
        if validation['recommendations']:
            logger.info("Configuration recommendations:")
            for rec in validation['recommendations']:
                logger.info(f"  TIP: {rec}")
    except Exception as e:
        logger.warning(f"Could not validate configuration: {e}")
    
    if use_optimal_config:
        try:
            optimal_config = get_optimal_config(architecture, dataset)
            
            if epochs is None:
                epochs = optimal_config.get('epochs', 100)
            if batch_size is None:
                batch_size = optimal_config.get('batch_size', 32)
            if learning_rate is None:
                learning_rate = optimal_config.get('learning_rate', 0.001)
                
            logger.info(f"Applied optimal configuration:")
            logger.info(f"  Epochs: {epochs}")
            logger.info(f"  Batch size: {batch_size}")
            logger.info(f"  Learning rate: {learning_rate}")
            
            expected_perf = get_expected_performance(architecture, dataset)
            logger.info(f"Expected performance: {expected_perf['accuracy']:.1%} accuracy")
            
            time_estimate = estimate_training_time(architecture, dataset, epochs)
            logger.info(f"Estimated training time: {time_estimate}")
            
        except Exception as e:
            logger.warning(f"Could not get optimal configuration: {e}. Using provided/default values.")
            epochs = epochs or 100
            batch_size = batch_size or 32
            learning_rate = learning_rate or 0.001
    else:
        epochs = epochs or 100
        batch_size = batch_size or 32
        learning_rate = learning_rate or 0.001
    
    logger.info(f"Final parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    set_random_seed(42)
    
    gpu_info = setup_gpu()
    logger.info(f"GPU setup: {gpu_info}")
    
    if batch_size == 'auto':
        batch_size = get_optimal_batch_size(
            model_input_shape=kwargs.get('input_shape', (32, 32, 3)),
            model_name=architecture
        )
        logger.info(f"Auto-selected batch size for {architecture}: {batch_size}")
    
    logger.info(f"Loading {dataset} dataset...")
    data_loader = get_data_loader(dataset, validation_split=validation_split, **kwargs)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_preprocess()
    
    dataset_info = data_loader.get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
    logger.info(f"Dataset loaded: {dataset_info}")
    
    logger.info(f"Creating {architecture} model...")
    model = create_model(architecture, dataset_info, dataset, **kwargs)
    keras_model = model.build_model()
    
    optimizer_map = {
        'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    }
    
    optimizer = optimizer_map.get(kwargs.get('optimizer', 'adam'), 
                                 tf.keras.optimizers.Adam(learning_rate=learning_rate))
    
    if dataset_info['num_classes'] == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
    else:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    
    keras_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    saved_models_dir = Path("saved_models")
    results_dir = Path("results")
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name_clean = f"{architecture}_{dataset}_{timestamp}"
    
    callbacks = [
        ModelCheckpoint(
            filepath=saved_models_dir / f"{model_name_clean}_best.keras",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=kwargs.get('patience', 15),
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=kwargs.get('lr_patience', 5),
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    logger.info("Starting model training...")
    history = keras_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    test_results = keras_model.evaluate(X_test, y_test, verbose=0)
    test_metrics = dict(zip(keras_model.metrics_names, test_results))
    
    test_accuracy = test_metrics.get('accuracy', test_metrics.get('compile_metrics', 0.0))
    test_loss = test_metrics.get('loss', 0.0)
    
    logger.info(f"Test Results: {test_metrics}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
    
    if save_results:
        experiment_results = {
            'model_architecture': architecture,
            'dataset': dataset,
            'dataset_info': dataset_info,
            'model_config': model.get_model_config() if hasattr(model, 'get_model_config') else {},
            'training_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'validation_split': validation_split,
                'optimizer': kwargs.get('optimizer', 'adam')
            },
            'test_metrics': {
                'accuracy': test_accuracy,
                'loss': test_loss,
                'raw_metrics': test_metrics
            },
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            },
            'timestamp': timestamp
        }
        
        save_experiment_results(experiment_results, history, results_dir)
        logger.info(f"Results saved for experiment: {model_name_clean}")
    
    return keras_model, {'accuracy': test_accuracy, 'loss': test_loss, 'raw_metrics': test_metrics}, history


def run_comprehensive_experiments(use_optimal_config: bool = True):
    """Run experiments across all model-dataset combinations with intelligent configuration"""
    
    models = ['baseline_cnn', 'resnet50', 'densenet121', 'inceptionv3', 'mobilenet']
    datasets = ['cifar10', 'fashion_mnist', 'cifar100', 'celeba']
    
    logger.info(f"Starting comprehensive experiments: {len(models)} models × {len(datasets)} datasets = {len(models) * len(datasets)} combinations")
    
    results_summary = []
    
    for model in models:
        for dataset in datasets:
            try:
                logger.info(f"\n{'='*80}")
                logger.info(f"Training {model.upper()} on {dataset.upper()}")
                logger.info(f"{'='*80}")
                
                try:
                    summary = get_training_summary(model, dataset)
                    logger.info("Training Summary:")
                    for line in summary.split('\n'):
                        if line.strip():
                            logger.info(line)
                except Exception as e:
                    logger.warning(f"Could not get training summary: {e}")
                
                if use_optimal_config:
                    trained_model, test_metrics, history = train_model(
                        architecture=model,
                        dataset=dataset,
                        use_optimal_config=True
                    )
                else:
                    dataset_configs = {
                        'cifar10': {'epochs': 50, 'batch_size': 32},
                        'cifar100': {'epochs': 100, 'batch_size': 32, 'label_mode': 'fine'},
                        'fashion_mnist': {'epochs': 30, 'batch_size': 64, 'standardize_size': True},
                        'celeba': {'epochs': 25, 'batch_size': 16, 'target_attribute': 'Smiling', 'max_samples': 10000}
                    }
                    
                    config = dataset_configs.get(dataset, {'epochs': 50, 'batch_size': 32})
                    
                    trained_model, test_metrics, history = train_model(
                        architecture=model,
                        dataset=dataset,
                        use_optimal_config=False,
                        **config
                    )
                
                results_summary.append({
                    'model': model,
                    'dataset': dataset,
                    'test_accuracy': test_metrics.get('accuracy', test_metrics.get('compile_metrics', 0.0)),
                    'test_loss': test_metrics.get('loss', 0.0),
                    'epochs_trained': len(history.history['loss'])
                })
                
                logger.info(f"Completed {model} on {dataset}: Test Accuracy = {test_metrics.get('accuracy', 0.0):.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model} on {dataset}: {e}")
                results_summary.append({
                    'model': model,
                    'dataset': dataset,
                    'test_accuracy': 0.0,
                    'test_loss': float('inf'),
                    'epochs_trained': 0,
                    'error': str(e)
                })
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPREHENSIVE EXPERIMENT RESULTS")
    logger.info(f"{'='*80}")
    
    for result in results_summary:
        if 'error' not in result:
            logger.info(f"{result['model']:12} | {result['dataset']:12} | "
                       f"Accuracy: {result['test_accuracy']:.4f} | "
                       f"Loss: {result['test_loss']:.4f} | "
                       f"Epochs: {result['epochs_trained']:3d}")
        else:
            logger.info(f"{result['model']:12} | {result['dataset']:12} | "
                       f"FAILED: {result['error']}")
    
    return results_summary


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Comprehensive CNN Training Script with Intelligent Configuration')
    
    parser.add_argument('--model', type=str, choices=['baseline_cnn', 'resnet50', 'densenet121', 'inceptionv3', 'mobilenet'],
                       help='Model architecture to train')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'fashion_mnist', 'celeba'],
                       help='Dataset to train on')
    parser.add_argument('--epochs', type=int, help='Number of epochs (auto-optimized if not specified)')
    parser.add_argument('--batch-size', type=int, help='Batch size (auto-optimized if not specified)')
    parser.add_argument('--learning-rate', type=float, help='Learning rate (auto-optimized if not specified)')
    
    # Experiment types
    parser.add_argument('--comprehensive', action='store_true', 
                       help='Run comprehensive experiments on all model-dataset combinations')
    parser.add_argument('--comprehensive-models', action='store_true',
                       help='Run all models on specified dataset for comparison')
    
    # Configuration options
    parser.add_argument('--use-optimal-config', action='store_true', default=True,
                       help='Use intelligent configuration optimization (default: True)')
    parser.add_argument('--disable-optimal-config', action='store_true',
                       help='Disable intelligent configuration optimization')
    parser.add_argument('--show-config-summary', action='store_true',
                       help='Show configuration summary before training')
    
    # Dataset-specific arguments
    parser.add_argument('--label-mode', type=str, choices=['fine', 'coarse'], default='fine',
                       help='CIFAR-100 label mode')
    parser.add_argument('--target-attribute', type=str, default='Smiling',
                       help='CelebA target attribute')
    parser.add_argument('--max-samples', type=int, help='Maximum samples for CelebA')
    parser.add_argument('--standardize-size', action='store_true', default=True,
                       help='Resize Fashion-MNIST to 32x32')
    
    # Model-specific arguments
    parser.add_argument('--alpha', type=float, default=1.0, help='MobileNet width multiplier')
    parser.add_argument('--trainable-params', type=str, choices=['all', 'top_layers', 'none'], default='top_layers',
                       help='Which parameters to train for transfer learning models')
    
    args = parser.parse_args()
    
    use_optimal_config = args.use_optimal_config and not args.disable_optimal_config
    
    if args.comprehensive:
        logger.info("Starting comprehensive experiments: 5 models × 4 datasets = 20 combinations")
        results = run_comprehensive_experiments(use_optimal_config=use_optimal_config)
        logger.info("Comprehensive experiments completed!")
    
    elif args.comprehensive_models and args.dataset:
        logger.info(f"Running all 5 models on {args.dataset.upper()}")
        
        results_summary = []
        for model in ['baseline_cnn', 'resnet50', 'densenet121', 'inceptionv3', 'mobilenet']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model.upper()} on {args.dataset.upper()}")
            logger.info(f"{'='*60}")
            
            if args.show_config_summary:
                try:
                    summary = get_training_summary(model, args.dataset)
                    logger.info(summary)
                except Exception as e:
                    logger.warning(f"Could not get configuration summary: {e}")
            
            try:
                kwargs = {
                    'label_mode': args.label_mode,
                    'target_attribute': args.target_attribute,
                    'max_samples': args.max_samples,
                    'standardize_size': args.standardize_size,
                    'alpha': args.alpha,
                    'trainable_params': args.trainable_params
                }
                
                trained_model, test_metrics, history = train_model(
                    architecture=model,
                    dataset=args.dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    use_optimal_config=use_optimal_config,
                    **kwargs
                )
                
                results_summary.append({
                    'model': model,
                    'dataset': args.dataset,
                    'test_accuracy': test_metrics.get('accuracy', 0.0),
                    'test_loss': test_metrics.get('loss', 0.0)
                })
                
            except Exception as e:
                logger.error(f"Error training {model} on {args.dataset}: {e}")
                results_summary.append({
                    'model': model,
                    'dataset': args.dataset,
                    'test_accuracy': 0.0,
                    'test_loss': float('inf'),
                    'error': str(e)
                })
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL COMPARISON RESULTS ON {args.dataset.upper()}")
        logger.info(f"{'='*60}")
        
        valid_results = [r for r in results_summary if 'error' not in r]
        valid_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        for i, result in enumerate(valid_results, 1):
            logger.info(f"{i}. {result['model']:12} | Accuracy: {result['test_accuracy']:.4f} | Loss: {result['test_loss']:.4f}")
    
    elif args.model and args.dataset:
        logger.info(f"Training {args.model.upper()} on {args.dataset.upper()}")
        
        if args.show_config_summary:
            try:
                summary = get_training_summary(args.model, args.dataset)
                logger.info(summary)
            except Exception as e:
                logger.warning(f"Could not get configuration summary: {e}")
        
        kwargs = {
            'label_mode': args.label_mode,
            'target_attribute': args.target_attribute,
            'max_samples': args.max_samples,
            'standardize_size': args.standardize_size,
            'alpha': args.alpha,
            'trainable_params': args.trainable_params
        }
        
        trained_model, test_metrics, history = train_model(
            architecture=args.model,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_optimal_config=use_optimal_config,
            **kwargs
        )
        
        logger.info(f"Training completed!")
        logger.info(f"Final Test Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
        logger.info(f"Final Test Loss: {test_metrics.get('loss', 0.0):.4f}")
    
    else:
        logger.error("Please specify either:")
        logger.error("  1. Single experiment: --model <model> --dataset <dataset>")
        logger.error("  2. Model comparison: --dataset <dataset> --comprehensive-models")
        logger.error("  3. Full experiments: --comprehensive")
        logger.error("")
        logger.error("Usage Examples:")
        logger.error("  # Single training with optimal config")
        logger.error("  python comprehensive_train.py --model resnet50 --dataset cifar10")
        logger.error("")
        logger.error("  # All models on one dataset")
        logger.error("  python comprehensive_train.py --dataset cifar10 --comprehensive-models")
        logger.error("")
        logger.error("  # Full comprehensive experiments")
        logger.error("  python comprehensive_train.py --comprehensive")
        logger.error("")
        logger.error("  # Custom parameters (overrides optimal config)")
        logger.error("  python comprehensive_train.py --model densenet121 --dataset cifar100 --epochs 50 --batch-size 16")
        logger.error("")
        logger.error("  # Show configuration summary before training")
        logger.error("  python comprehensive_train.py --model inceptionv3 --dataset celeba --show-config-summary")
        
        parser.print_help()
        return 1
    
    logger.info("All experiments completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main()) 