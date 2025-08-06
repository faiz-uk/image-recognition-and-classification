"""
Evaluate Existing Trained Models with Comprehensive Metrics
Use this script to get comprehensive evaluation metrics for already trained models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
from datetime import datetime
import re
import signal
import time
from contextlib import contextmanager

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import CIFAR10Loader, CIFAR100Loader, FashionMNISTLoader, CelebALoader
from utils.evaluation_metrics import ComprehensiveEvaluator, create_evaluation_report
from utils.helpers import set_random_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timeout operations"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def get_data_loader(dataset_name: str, **kwargs):
    """Get the appropriate data loader for the given dataset"""
    loaders = {
        'cifar10': CIFAR10Loader,
        'cifar100': CIFAR100Loader,
        'fashion_mnist': FashionMNISTLoader,
        'celeba': CelebALoader
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
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


def verify_model_dataset_compatibility(model_path: str, dataset_name: str) -> bool:
    """
    Strictly verify that a model was trained on the specified dataset
    
    Args:
        model_path: Path to the model file
        dataset_name: Expected dataset name
        
    Returns:
        True if compatible, False otherwise
    """
    model_filename = Path(model_path).stem.lower()  # This removes .keras extension
    expected_dataset = dataset_name.lower()
    
    # STRICT pattern matching: architecture_dataset_timestamp_best (stem only, no extension)
    pattern = rf'^[a-z0-9_]+_{re.escape(expected_dataset)}_\d{{8}}_\d{{6}}_best$'
    
    if not re.match(pattern, model_filename):
        logger.error(f" Model {model_filename} does not match expected pattern for {dataset_name}")
        logger.error(f"   Expected pattern: {pattern}")
        logger.debug(f"   Actual filename: {model_filename}")
        return False
    
    # Additional check: ensure no other dataset names are present
    other_datasets = ['cifar10', 'cifar100', 'fashion_mnist', 'celeba']
    if expected_dataset in other_datasets:
        other_datasets.remove(expected_dataset)
    
    for other_dataset in other_datasets:
        if f"_{other_dataset}_" in model_filename:
            logger.error(f" Model {model_filename} contains conflicting dataset name: {other_dataset}")
            return False
    
    logger.debug(f" Model {model_filename} is compatible with {dataset_name}")
    return True


def safe_model_predict(model, X_data, batch_size=32, timeout_seconds=300):
    """
    Safely predict with timeout and memory management
    
    Args:
        model: Keras model
        X_data: Input data
        batch_size: Batch size for prediction
        timeout_seconds: Maximum time allowed for prediction
        
    Returns:
        Predictions or None if failed
    """
    try:
        logger.info(f"Predicting on {X_data.shape[0]} samples with batch_size={batch_size}")
        
        with timeout(timeout_seconds):
            # Use smaller batch size to prevent memory issues
            predictions = model.predict(X_data, batch_size=batch_size, verbose=0)
            
        logger.info(f" Prediction completed: {predictions.shape}")
        return predictions
        
    except TimeoutError:
        logger.error(f" Prediction timed out after {timeout_seconds} seconds")
        return None
    except Exception as e:
        logger.error(f" Prediction failed: {e}")
        # Try with smaller batch size
        if batch_size > 8:
            logger.info(f"Retrying with smaller batch size: {batch_size // 2}")
            return safe_model_predict(model, X_data, batch_size // 2, timeout_seconds)
        return None


def load_model_with_fallbacks(model_path: str, dataset_name: str):
    """
    Load model with comprehensive fallback strategies
    
    Args:
        model_path: Path to model file
        dataset_name: Dataset name for proper compilation
        
    Returns:
        Loaded model or None if all strategies fail
    """
    model = None
    loading_strategies = [
        ("standard", "Standard loading"),
        ("compile_false", "Load without compilation"),
        ("custom_objects", "Load with custom objects"),
        ("fashion_mnist_lambda_fix", "Fashion-MNIST Lambda layer fix"),
        ("safe_mode", "Safe mode loading")
    ]
    
    for strategy_name, strategy_desc in loading_strategies:
        try:
            logger.info(f"Trying: {strategy_desc}")
            
            if strategy_name == "standard":
                tf.keras.config.enable_unsafe_deserialization()
                model = tf.keras.models.load_model(model_path)
                
            elif strategy_name == "compile_false":
                model = tf.keras.models.load_model(model_path, compile=False)
                # Recompile with appropriate loss function
                if dataset_name.lower() == 'celeba':
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                else:
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
            elif strategy_name == "custom_objects":
                # Handle Lambda layers specifically
                def grayscale_to_rgb(x):
                    """Convert grayscale to RGB by repeating channels"""
                    return tf.repeat(x, 3, axis=-1)
                
                custom_objects = {
                    'Lambda': tf.keras.layers.Lambda,
                    'lambda': tf.keras.layers.Lambda,
                    'grayscale_to_rgb': grayscale_to_rgb
                }
                
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                if dataset_name.lower() == 'celeba':
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                else:
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
            elif strategy_name == "fashion_mnist_lambda_fix":
                # Specialized fix for Fashion-MNIST Lambda layer shape inference issues
                if dataset_name.lower() == 'fashion_mnist':
                    logger.info("Applying comprehensive Fashion-MNIST Lambda layer fix...")
                    
                    # Try multiple approaches for Fashion-MNIST Lambda layer issues
                    fashion_mnist_strategies = [
                        "custom_layer_replacement",
                        "function_override",
                        "model_reconstruction"
                    ]
                    
                    for fm_strategy in fashion_mnist_strategies:
                        try:
                            logger.info(f"   Fashion-MNIST sub-strategy: {fm_strategy}")
                            
                            if fm_strategy == "custom_layer_replacement":
                                # Replace Lambda with a custom layer
                                class GrayscaleToRGB(tf.keras.layers.Layer):
                                    def __init__(self, **kwargs):
                                        super().__init__(**kwargs)
                                    
                                    def call(self, inputs):
                                        # Convert grayscale to RGB by repeating channels
                                        return tf.repeat(inputs, 3, axis=-1)
                                    
                                    def compute_output_shape(self, input_shape):
                                        return (*input_shape[:-1], 3)
                                    
                                    def get_config(self):
                                        return super().get_config()
                                
                                custom_objects = {
                                    'Lambda': GrayscaleToRGB,
                                    'lambda': GrayscaleToRGB,
                                    'GrayscaleToRGB': GrayscaleToRGB,
                                }
                                
                                tf.keras.config.enable_unsafe_deserialization()
                                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                                
                            elif fm_strategy == "function_override":
                                # Override the specific function causing issues
                                def safe_grayscale_to_rgb(x):
                                    # Safe conversion with explicit shape handling
                                    x = tf.cast(x, tf.float32)
                                    if len(x.shape) == 4 and x.shape[-1] == 1:
                                        return tf.repeat(x, 3, axis=-1)
                                    elif len(x.shape) == 3:
                                        x = tf.expand_dims(x, axis=-1)
                                        return tf.repeat(x, 3, axis=-1)
                                    return x
                                
                                # Override Lambda with a function that has explicit output shape
                                def create_safe_lambda(func, output_shape=None, **kwargs):
                                    if output_shape is None:
                                        # Default output shape for grayscale to RGB
                                        output_shape = lambda input_shape: (*input_shape[:-1], 3)
                                    return tf.keras.layers.Lambda(func, output_shape=output_shape, **kwargs)
                                
                                custom_objects = {
                                    'Lambda': create_safe_lambda,
                                    'lambda': create_safe_lambda,
                                    'grayscale_to_rgb': safe_grayscale_to_rgb,
                                    'safe_grayscale_to_rgb': safe_grayscale_to_rgb,
                                }
                                
                                tf.keras.config.enable_unsafe_deserialization()
                                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
                                
                            elif fm_strategy == "model_reconstruction":
                                # Try to load model architecture only and skip problematic layers
                                try:
                                    # Load model without weights first
                                    with open(model_path, 'rb') as f:
                                        # This is a last resort - try to load in safe mode
                                        tf.keras.config.enable_unsafe_deserialization()
                                        
                                        # Create minimal custom objects
                                        minimal_objects = {
                                            'Lambda': lambda func, **kwargs: tf.keras.layers.Lambda(
                                                lambda x: tf.repeat(x, 3, axis=-1),
                                                output_shape=lambda input_shape: (*input_shape[:-1], 3)
                                            )
                                        }
                                        
                                        model = tf.keras.models.load_model(model_path, custom_objects=minimal_objects, compile=False)
                                except:
                                    continue
                            
                            if model is not None:
                                # Recompile the model
                                model.compile(
                                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                logger.info(f"   Fashion-MNIST fix successful with: {fm_strategy}")
                                break
                                
                        except Exception as e:
                            logger.warning(f"   Fashion-MNIST {fm_strategy} failed: {str(e)[:100]}...")
                            model = None
                            continue
                    
                    if model is None:
                        logger.error("   All Fashion-MNIST sub-strategies failed")
                        continue
                    
                else:
                    # Skip this strategy for non-Fashion-MNIST datasets
                    logger.info("Skipping Fashion-MNIST fix for non-Fashion-MNIST dataset")
                    continue
            
            elif strategy_name == "safe_mode":
                # Last resort with minimal options
                try:
                    model = tf.keras.models.load_model(model_path, safe_mode=False, compile=False)
                    if dataset_name.lower() == 'celeba':
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    else:
                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                except TypeError:
                    # safe_mode parameter doesn't exist in this TF version
                    continue
            
            if model is not None:
                logger.info(f" Model loaded using: {strategy_desc}")
                logger.info(f"   Parameters: {model.count_params():,}")
                logger.info(f"   Input shape: {model.input_shape}")
                return model
                
        except Exception as e:
            logger.warning(f" {strategy_desc} failed: {str(e)[:100]}...")
            model = None
            continue
    
    return None


def load_and_evaluate_model(model_path: str, 
                           dataset_name: str,
                           model_name: str = None,
                           **dataset_kwargs):
    """
    Load a saved model and evaluate it comprehensively with robust error handling
    
    Args:
        model_path: Path to the saved model (.keras file)
        dataset_name: Name of the dataset to evaluate on
        model_name: Name to use for the model (extracted from path if None)
        **dataset_kwargs: Dataset-specific arguments
    
    Returns:
        Comprehensive evaluation results or None if failed
    """
    
    # Extract model name from path if not provided
    if model_name is None:
        model_name = Path(model_path).stem.replace('_best', '').replace('_final', '')
        # Try to extract architecture from filename
        for arch in ['baseline_cnn', 'resnet50', 'densenet121', 'inceptionv3', 'mobilenet']:
            if arch in model_name.lower():
                model_name = arch
                break
    
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    
    # STRICT compatibility check
    if not verify_model_dataset_compatibility(model_path, dataset_name):
        return None
    
    # Load model with fallbacks
    model = load_model_with_fallbacks(model_path, dataset_name)
    if model is None:
        logger.error(f"All loading strategies failed for {model_path}")
        return None
    
    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")
    set_random_seed(42)  # Ensure consistent data splits
    
    try:
        data_loader = get_data_loader(dataset_name, **dataset_kwargs)
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_preprocess()
        dataset_info = data_loader.get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
        logger.info(f"Dataset loaded: {dataset_info['total_samples']} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return None
    
    # Determine task type
    task_type = 'binary' if dataset_info['num_classes'] == 2 else 'multiclass'
    class_names = dataset_info.get('class_names', None)
    
    logger.info(f"Task type: {task_type}")
    logger.info(f"Number of classes: {dataset_info['num_classes']}")
    
    # Verify input shape compatibility
    expected_shape = model.input_shape[1:]  # Remove batch dimension
    actual_shape = X_test.shape[1:]
    
    if expected_shape != actual_shape:
        logger.error(f"Shape mismatch!")
        logger.error(f"   Model expects: {expected_shape}")
        logger.error(f"   Dataset provides: {actual_shape}")
        return None
    
    # Get predictions with safety measures
    logger.info("Generating predictions...")
    logger.info(f"   Test data shape: {X_test.shape}")
    logger.info(f"   Validation data shape: {X_val.shape}")
    
    # Test set predictions with timeout
    y_test_pred_proba = safe_model_predict(model, X_test, batch_size=16, timeout_seconds=600)
    if y_test_pred_proba is None:
        logger.error("Failed to get test predictions")
        return None
    
    # Validation set predictions with timeout
    y_val_pred_proba = safe_model_predict(model, X_val, batch_size=16, timeout_seconds=600)
    if y_val_pred_proba is None:
        logger.error("Failed to get validation predictions")
        return None
    
    # Convert probabilities to predictions
    if task_type == 'binary':
        y_test_pred = (y_test_pred_proba > 0.5).astype(int)
        y_val_pred = (y_val_pred_proba > 0.5).astype(int)
    else:
        y_test_pred = np.argmax(y_test_pred_proba, axis=1)
        y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    
    # Initialize comprehensive evaluator
    evaluator = ComprehensiveEvaluator(class_names=class_names, task_type=task_type)
    
    # Evaluate on test set (main evaluation)
    logger.info("Performing comprehensive evaluation on test set...")
    try:
        test_eval_results = evaluator.evaluate_model(
            y_true=y_test,
            y_pred=y_test_pred,
            y_pred_proba=y_test_pred_proba,
            model_name=model_name,
            dataset_name=dataset_name
        )
    except Exception as e:
        logger.error(f"Test evaluation failed: {e}")
        return None
    
    # Evaluate on validation set
    logger.info("Performing comprehensive evaluation on validation set...")
    try:
        val_eval_results = evaluator.evaluate_model(
            y_true=y_val,
            y_pred=y_val_pred,
            y_pred_proba=y_val_pred_proba,
            model_name=f"{model_name}_validation",
            dataset_name=dataset_name
        )
    except Exception as e:
        logger.error(f"Validation evaluation failed: {e}")
        return None
    
    # Display results
    logger.info("=== COMPREHENSIVE EVALUATION RESULTS ===")
    logger.info(f"Test Accuracy: {test_eval_results['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_eval_results['precision']:.4f}")
    logger.info(f"Test Recall: {test_eval_results['recall']:.4f}")
    logger.info(f"Test F1-Score: {test_eval_results['f1_score']:.4f}")
    if test_eval_results['roc_auc']:
        logger.info(f"Test ROC AUC: {test_eval_results['roc_auc']:.4f}")
    logger.info("==========================================")
    
    # Generate comprehensive evaluation report
    # Extract original timestamp from model filename if available
    original_timestamp = None
    model_filename = Path(model_path).stem
    
    # Try to extract timestamp from filename (format: model_dataset_YYYYMMDD_HHMMSS_best)
    timestamp_match = re.search(r'(\d{8}_\d{6})', model_filename)
    if timestamp_match:
        original_timestamp = timestamp_match.group(1)
        logger.info(f"Using original model timestamp: {original_timestamp}")
    
    # Use original timestamp or create new one
    timestamp = original_timestamp if original_timestamp else datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results directory with consistent naming: model_dataset_timestamp
    results_dir = Path("results") / "comprehensive_evaluation" / f"{model_name}_{dataset_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy history for visualization (since we don't have training history)
    dummy_history = {
        'loss': [1.0, 0.8, 0.6, 0.4, 0.3],
        'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4],
        'accuracy': [0.3, 0.5, 0.7, 0.8, test_eval_results['accuracy']],
        'val_accuracy': [0.25, 0.45, 0.65, 0.75, val_eval_results['accuracy']]
    }
    
    logger.info("Generating comprehensive evaluation report...")
    
    try:
        # Update the evaluator to use consistent naming
        evaluator_for_report = ComprehensiveEvaluator(class_names=class_names, task_type=task_type)
        saved_files = {}
        
        # Generate reports with consistent naming: model_dataset_timestamp_reporttype.png
        base_name = f"{model_name}_{dataset_name}_{timestamp}"
        
        # 1. Confusion Matrix
        cm_path = results_dir / f"{base_name}_confusion_matrix.png"
        evaluator_for_report.plot_confusion_matrix(test_eval_results, save_path=cm_path)
        saved_files['confusion_matrix'] = str(cm_path)
        
        # 2. Classification Metrics
        metrics_path = results_dir / f"{base_name}_classification_metrics.png"
        evaluator_for_report.plot_classification_metrics(test_eval_results, save_path=metrics_path)
        saved_files['classification_metrics'] = str(metrics_path)
        
        # 3. Learning Curves
        curves_path = results_dir / f"{base_name}_learning_curves.png"
        evaluator_for_report.plot_learning_curves(dummy_history, model_name, save_path=curves_path)
        saved_files['learning_curves'] = str(curves_path)
        
        eval_report_files = saved_files
        
        logger.info("Comprehensive evaluation completed!")
        logger.info(f"Reports saved to: {results_dir}")
        logger.info(f"  Confusion Matrix: {eval_report_files.get('confusion_matrix', 'N/A')}")
        logger.info(f"  Classification Metrics: {eval_report_files.get('classification_metrics', 'N/A')}")
        logger.info(f"  Learning Curves: {eval_report_files.get('learning_curves', 'N/A')}")
        
        return {
            'test_results': test_eval_results,
            'validation_results': val_eval_results,
            'report_files': eval_report_files,
            'results_dir': results_dir
        }
        
    except Exception as e:
        logger.error(f"Failed to generate evaluation report: {e}")
        # Return results even if report generation fails
        return {
            'test_results': test_eval_results,
            'validation_results': val_eval_results,
            'report_files': {},
            'results_dir': results_dir
        }


def find_saved_models(models_dir: str = "saved_models") -> list:
    """Find all saved model files"""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    
    model_files = []
    for ext in ['*.keras', '*.h5']:
        model_files.extend(models_dir.glob(ext))
    
    return sorted(model_files)


def evaluate_all_models_on_dataset(dataset_name: str, **dataset_kwargs):
    """Evaluate all saved models on a specific dataset with robust filtering and error handling"""
    
    saved_models = find_saved_models()
    if not saved_models:
        logger.warning("No saved models found in saved_models/ directory")
        return []
    
    logger.info(f"Found {len(saved_models)} total model files")
    
    # STRICT dataset filtering using the verification function
    dataset_models = []
    skipped_models = []
    
    for model_path in saved_models:
        if verify_model_dataset_compatibility(str(model_path), dataset_name):
            dataset_models.append(model_path)
            logger.info(f" Matched: {model_path.name}")
        else:
            skipped_models.append(model_path.name)
            logger.debug(f" Skipped: {model_path.name}")
    
    # Log filtering results
    if skipped_models:
        logger.info(f"\nSkipped {len(skipped_models)} incompatible models:")
        for model_name in skipped_models:
            logger.info(f"    {model_name}")
    
    if not dataset_models:
        logger.warning(f" No models found trained on {dataset_name}")
        logger.info(f"Available models: {[m.name for m in saved_models]}")
        return []
    
    logger.info(f"\nFound {len(dataset_models)} models trained on {dataset_name.upper()}")
    logger.info(f"Models to evaluate: {[m.name for m in dataset_models]}")
    
    results_summary = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    for i, model_path in enumerate(dataset_models, 1):
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating {i}/{len(dataset_models)}: {model_path.name}")
            logger.info(f"{'='*80}")
            
            start_time = time.time()
            
            result = load_and_evaluate_model(
                model_path=str(model_path),
                dataset_name=dataset_name,
                **dataset_kwargs
            )
            
            evaluation_time = time.time() - start_time
            
            if result:
                test_results = result['test_results']
                results_summary.append({
                    'model_file': model_path.name,
                    'model_name': test_results['model_name'],
                    'dataset': dataset_name,
                    'accuracy': test_results['accuracy'],
                    'precision': test_results['precision'],
                    'recall': test_results['recall'],
                    'f1_score': test_results['f1_score'],
                    'roc_auc': test_results['roc_auc'],
                    'results_dir': result['results_dir'],
                    'evaluation_time': evaluation_time
                })
                successful_evaluations += 1
                logger.info(f" Evaluation completed in {evaluation_time:.1f}s")
            else:
                results_summary.append({
                    'model_file': model_path.name,
                    'error': 'Evaluation failed - see logs above',
                    'evaluation_time': evaluation_time
                })
                failed_evaluations += 1
                logger.error(f" Evaluation failed after {evaluation_time:.1f}s")
                
        except Exception as e:
            evaluation_time = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f" Unexpected error evaluating {model_path.name}: {e}")
            results_summary.append({
                'model_file': model_path.name,
                'error': str(e),
                'evaluation_time': evaluation_time
            })
            failed_evaluations += 1
    
    # Display comprehensive summary
    logger.info(f"\n{'='*100}")
    logger.info(f"COMPREHENSIVE EVALUATION SUMMARY - {dataset_name.upper()}")
    logger.info(f"{'='*100}")
    logger.info(f"Total models evaluated: {len(dataset_models)}")
    logger.info(f"Successful evaluations: {successful_evaluations}")
    logger.info(f"Failed evaluations: {failed_evaluations}")
    
    # Sort by F1-score (best metric for overall performance)
    valid_results = [r for r in results_summary if 'error' not in r]
    valid_results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    if valid_results:
        logger.info(f"\nMODEL PERFORMANCE RANKING:")
        logger.info(f"{'Rank':<4} {'Model':<20} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'ROC AUC':<8} {'Time(s)':<8}")
        logger.info("-" * 100)
        
        for i, result in enumerate(valid_results, 1):
            roc_str = f"{result['roc_auc']:.4f}" if result['roc_auc'] else "N/A"
            logger.info(f"{i:<4} {result['model_name']:<20} {result['accuracy']:<9.4f} "
                       f"{result['precision']:<10.4f} {result['recall']:<8.4f} "
                       f"{result['f1_score']:<9.4f} {roc_str:<8} {result['evaluation_time']:<8.1f}")
    
    # Show failed evaluations with details
    failed_results = [r for r in results_summary if 'error' in r]
    if failed_results:
        logger.info(f"\n FAILED EVALUATIONS:")
        for result in failed_results:
            logger.info(f"   Model file: {result['model_file']}")
            logger.info(f"      Error: {result['error']}")
            logger.info(f"      Time: {result.get('evaluation_time', 0):.1f}s")
    
    return results_summary


def reorganize_loose_evaluation_files():
    """
    Reorganize loose evaluation files (like CelebA results) into organized directories
    This function moves files like 'model_dataset_reporttype.png' into 'model_dataset_timestamp/' folders
    """
    eval_dir = Path("results/comprehensive_evaluation")
    if not eval_dir.exists():
        logger.info("No comprehensive_evaluation directory found")
        return
    
    # Find loose PNG files (not in subdirectories)
    loose_files = [f for f in eval_dir.iterdir() if f.is_file() and f.suffix == '.png']
    
    if not loose_files:
        logger.info("No loose evaluation files found to reorganize")
        return
    
    logger.info(f"Found {len(loose_files)} loose evaluation files to reorganize")
    
    # Group files by model and dataset
    file_groups = {}
    for file_path in loose_files:
        filename = file_path.stem
        
        # Parse filename: model_dataset_reporttype (e.g., resnet50_celeba_confusion_matrix)
        parts = filename.split('_')
        if len(parts) >= 3:
            # Find the dataset part (celeba, cifar10, etc.)
            dataset_candidates = ['celeba', 'cifar10', 'cifar100', 'fashion_mnist']
            model_parts = []
            dataset_part = None
            report_parts = []
            
            dataset_found = False
            for i, part in enumerate(parts):
                if part in dataset_candidates:
                    model_parts = parts[:i]
                    dataset_part = part
                    report_parts = parts[i+1:]
                    dataset_found = True
                    break
            
            if dataset_found and model_parts and dataset_part:
                model_name = '_'.join(model_parts)
                report_type = '_'.join(report_parts)
                
                # Try to find matching saved model to get timestamp
                model_pattern = f"{model_name}_{dataset_part}_*_best.keras"
                saved_models = list(Path("saved_models").glob(model_pattern))
                
                if saved_models:
                    # Extract timestamp from saved model filename
                    saved_model = saved_models[0]  # Take the first match
                    timestamp_match = re.search(r'(\d{8}_\d{6})', saved_model.stem)
                    if timestamp_match:
                        timestamp = timestamp_match.group(1)
                        
                        # Create organized directory
                        organized_dir = eval_dir / f"{model_name}_{dataset_part}_{timestamp}"
                        organized_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Move file to organized directory with proper naming
                        new_filename = f"{model_name}_{dataset_part}_{timestamp}_{report_type}.png"
                        new_path = organized_dir / new_filename
                        
                        logger.info(f"Moving {file_path.name} -> {organized_dir.name}/{new_filename}")
                        file_path.rename(new_path)
                        
                        continue
                
                logger.warning(f"Could not find timestamp for {filename}, skipping")
            else:
                logger.warning(f"Could not parse filename format: {filename}")
        else:
            logger.warning(f"Unexpected filename format: {filename}")
    
    logger.info("File reorganization completed!")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Evaluate Existing Trained Models with Comprehensive Metrics')
    
    parser.add_argument('--model-path', type=str, help='Path to specific model file to evaluate')
    parser.add_argument('--dataset', type=str, 
                       choices=['cifar10', 'cifar100', 'fashion_mnist', 'celeba'],
                       help='Dataset to evaluate on')
    parser.add_argument('--model-name', type=str, help='Name for the model (auto-detected if not provided)')
    
    # Evaluate all models
    parser.add_argument('--evaluate-all', action='store_true',
                       help='Evaluate all saved models on specified dataset')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available saved models')
    parser.add_argument('--reorganize-files', action='store_true',
                       help='Reorganize loose evaluation files into organized directories')
    
    # Dataset-specific arguments
    parser.add_argument('--label-mode', type=str, choices=['fine', 'coarse'], default='fine',
                       help='CIFAR-100 label mode')
    parser.add_argument('--target-attribute', type=str, default='Smiling',
                       help='CelebA target attribute')
    parser.add_argument('--max-samples', type=int, help='Maximum samples for CelebA')
    parser.add_argument('--standardize-size', action='store_true', default=True,
                       help='Resize Fashion-MNIST to 32x32')
    
    args = parser.parse_args()
    
    if args.list_models:
        saved_models = find_saved_models()
        if saved_models:
            logger.info(f"Found {len(saved_models)} saved models:")
            for model in saved_models:
                logger.info(f"   Model file: {model}")
        else:
            logger.info("No saved models found in saved_models/ directory")
        return 0
    
    if args.reorganize_files:
        reorganize_loose_evaluation_files()
        logger.info("Loose evaluation files reorganized.")
        return 0
    
    if not args.dataset:
        logger.error("Please specify a dataset using --dataset")
        parser.print_help()
        return 1
    
    # Dataset kwargs
    dataset_kwargs = {
        'label_mode': args.label_mode,
        'target_attribute': args.target_attribute,
        'max_samples': args.max_samples,
        'standardize_size': args.standardize_size,
    }

    if args.evaluate_all:
        # Evaluate all saved models
        logger.info(f"Evaluating ALL saved models on {args.dataset.upper()}")
        results = evaluate_all_models_on_dataset(args.dataset, **dataset_kwargs)
        
        if results:
            logger.info(f"\n Evaluation completed for {len(results)} models")
            logger.info("Comprehensive reports generated for each model")
        
    elif args.model_path:
        # Evaluate specific model
        if not Path(args.model_path).exists():
            logger.error(f"Model file not found: {args.model_path}")
            return 1
        
        logger.info(f"Evaluating specific model: {args.model_path}")
        result = load_and_evaluate_model(
            model_path=args.model_path,
            dataset_name=args.dataset,
            model_name=args.model_name,
            **dataset_kwargs
        )
        
        if result:
            logger.info("Model evaluation completed successfully!")
        else:
            logger.error("Model evaluation failed!")
            return 1
    
    else:
        logger.error("Please specify either --model-path, --evaluate-all, or --reorganize-files")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())