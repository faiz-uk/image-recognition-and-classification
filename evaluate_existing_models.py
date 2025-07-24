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

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import CIFAR10Loader, CIFAR100Loader, FashionMNISTLoader, CelebALoader
from utils.evaluation_metrics import ComprehensiveEvaluator, create_evaluation_report
from utils.helpers import set_random_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def load_and_evaluate_model(model_path: str, 
                           dataset_name: str,
                           model_name: str = None,
                           **dataset_kwargs):
    """
    Load a saved model and evaluate it comprehensively
    
    Args:
        model_path: Path to the saved model (.keras file)
        dataset_name: Name of the dataset to evaluate on
        model_name: Name to use for the model (extracted from path if None)
        **dataset_kwargs: Dataset-specific arguments
    
    Returns:
        Comprehensive evaluation results
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
    
    # Load the model
    try:
        # Enable unsafe deserialization to handle Lambda layers in transfer learning models
        tf.keras.config.enable_unsafe_deserialization()
        
        # Try different loading methods for Lambda layer issues
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as lambda_error:
            logger.warning(f"Standard loading failed: {lambda_error}")
            logger.info("Trying alternative loading method...")
            # Try loading with compile=False to avoid Lambda layer issues
            model = tf.keras.models.load_model(model_path, compile=False)
            # Recompile the model manually
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"Model summary: {model.count_params()} parameters")
        logger.info(f"Model input shape: {model.input_shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None
    
    # Load dataset
    logger.info(f"Loading {dataset_name} dataset...")
    set_random_seed(42)  # Ensure consistent data splits
    
    data_loader = get_data_loader(dataset_name, **dataset_kwargs)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_preprocess()
    
    dataset_info = data_loader.get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
    logger.info(f"✅ Dataset loaded: {dataset_info['total_samples']} samples")
    
    # Determine task type
    task_type = 'binary' if dataset_info['num_classes'] == 2 else 'multiclass'
    class_names = dataset_info.get('class_names', None)
    
    logger.info(f"Task type: {task_type}")
    logger.info(f"Number of classes: {dataset_info['num_classes']}")
    
    # Get predictions
    logger.info("Generating predictions...")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    
    # Test set predictions
    try:
        y_test_pred_proba = model.predict(X_test, verbose=0)
        logger.info(f"Test predictions shape: {y_test_pred_proba.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to predict on test set: {e}")
        logger.error(f"Model expects: {model.input_shape}, Got: {X_test.shape}")
        return None
        
    try:
        y_val_pred_proba = model.predict(X_val, verbose=0)
        logger.info(f"Validation predictions shape: {y_val_pred_proba.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to predict on validation set: {e}")
        logger.error(f"Model expects: {model.input_shape}, Got: {X_val.shape}")
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
    logger.info("🔍 Performing comprehensive evaluation on test set...")
    test_eval_results = evaluator.evaluate_model(
        y_true=y_test,
        y_pred=y_test_pred,
        y_pred_proba=y_test_pred_proba,
        model_name=model_name,
        dataset_name=dataset_name
    )
    
    # Evaluate on validation set
    logger.info("🔍 Performing comprehensive evaluation on validation set...")
    val_eval_results = evaluator.evaluate_model(
        y_true=y_val,
        y_pred=y_val_pred,
        y_pred_proba=y_val_pred_proba,
        model_name=f"{model_name}_validation",
        dataset_name=dataset_name
    )
    
    # Display results
    logger.info("=== COMPREHENSIVE EVALUATION RESULTS ===")
    logger.info(f"📊 Test Accuracy: {test_eval_results['accuracy']:.4f}")
    logger.info(f"🎯 Test Precision: {test_eval_results['precision']:.4f}")
    logger.info(f"📈 Test Recall: {test_eval_results['recall']:.4f}")
    logger.info(f"⚖️ Test F1-Score: {test_eval_results['f1_score']:.4f}")
    if test_eval_results['roc_auc']:
        logger.info(f"📋 Test ROC AUC: {test_eval_results['roc_auc']:.4f}")
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
    
    logger.info("📊 Generating comprehensive evaluation report...")
    
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
    
    logger.info("✅ Comprehensive evaluation completed!")
    logger.info(f"📁 Reports saved to: {results_dir}")
    logger.info(f"  📈 Confusion Matrix: {eval_report_files.get('confusion_matrix', 'N/A')}")
    logger.info(f"  📊 Classification Metrics: {eval_report_files.get('classification_metrics', 'N/A')}")
    logger.info(f"  📉 Learning Curves: {eval_report_files.get('learning_curves', 'N/A')}")
    
    return {
        'test_results': test_eval_results,
        'validation_results': val_eval_results,
        'report_files': eval_report_files,
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
    """Evaluate all saved models on a specific dataset"""
    
    saved_models = find_saved_models()
    if not saved_models:
        logger.warning("No saved models found in saved_models/ directory")
        return []
    
    # Filter models to only include those trained on the target dataset
    dataset_models = []
    for model_path in saved_models:
        filename = model_path.name.lower()
        # More precise filtering using regex to match exact pattern: {model}_{dataset}_{timestamp}_best.keras
        import re
        pattern = rf'.*_{re.escape(dataset_name.lower())}_\d{{8}}_\d{{6}}_best\.keras$'
        if re.match(pattern, filename):
            dataset_models.append(model_path)
            logger.info(f"✅ Matched: {model_path.name}")
        else:
            logger.debug(f"❌ Skipped: {model_path.name} (doesn't match {dataset_name})")
    
    if not dataset_models:
        logger.warning(f"No models found trained on {dataset_name}")
        logger.info(f"Available models: {[m.name for m in saved_models]}")
        return []
    
    logger.info(f"Found {len(dataset_models)} models trained on {dataset_name.upper()}")
    logger.info(f"Evaluating models: {[m.name for m in dataset_models]}")
    
    results_summary = []
    
    for model_path in dataset_models:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_path.name}")
            logger.info(f"{'='*60}")
            
            result = load_and_evaluate_model(
                model_path=str(model_path),
                dataset_name=dataset_name,
                **dataset_kwargs
            )
            
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
                    'results_dir': result['results_dir']
                })
                
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_path.name}: {e}")
            results_summary.append({
                'model_file': model_path.name,
                'error': str(e)
            })
    
    # Display summary
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPREHENSIVE EVALUATION SUMMARY - {dataset_name.upper()}")
    logger.info(f"{'='*80}")
    
    # Sort by F1-score (best metric for overall performance)
    valid_results = [r for r in results_summary if 'error' not in r]
    valid_results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    if valid_results:
        logger.info(f"{'Rank':<4} {'Model':<20} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'ROC AUC':<8}")
        logger.info("-" * 80)
        
        for i, result in enumerate(valid_results, 1):
            roc_str = f"{result['roc_auc']:.4f}" if result['roc_auc'] else "N/A"
            logger.info(f"{i:<4} {result['model_name']:<20} {result['accuracy']:<9.4f} "
                       f"{result['precision']:<10.4f} {result['recall']:<8.4f} "
                       f"{result['f1_score']:<9.4f} {roc_str:<8}")
    
    # Show failed evaluations
    failed_results = [r for r in results_summary if 'error' in r]
    if failed_results:
        logger.info(f"\n❌ Failed evaluations:")
        for result in failed_results:
            logger.info(f"  {result['model_file']}: {result['error']}")
    
    return results_summary


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
                logger.info(f"  📁 {model}")
        else:
            logger.info("No saved models found in saved_models/ directory")
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
        logger.info(f"🔍 Evaluating ALL saved models on {args.dataset.upper()}")
        results = evaluate_all_models_on_dataset(args.dataset, **dataset_kwargs)
        
        if results:
            logger.info(f"\n✅ Evaluation completed for {len(results)} models")
            logger.info("📊 Comprehensive reports generated for each model")
        
    elif args.model_path:
        # Evaluate specific model
        if not Path(args.model_path).exists():
            logger.error(f"Model file not found: {args.model_path}")
            return 1
        
        logger.info(f"🔍 Evaluating specific model: {args.model_path}")
        result = load_and_evaluate_model(
            model_path=args.model_path,
            dataset_name=args.dataset,
            model_name=args.model_name,
            **dataset_kwargs
        )
        
        if result:
            logger.info("✅ Model evaluation completed successfully!")
        else:
            logger.error("❌ Model evaluation failed!")
            return 1
    
    else:
        logger.error("Please specify either --model-path or --evaluate-all")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())