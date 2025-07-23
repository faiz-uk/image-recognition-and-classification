#!/usr/bin/env python3
"""
System test script to verify all components work correctly
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        from models import BaseModel, BaselineCNNModel, ResNet50Model, DenseNet121Model, InceptionV3Model, MobileNetModel
        logger.info("✓ Model imports successful (all 5 models)")
        
        from data import CIFAR10Loader, CIFAR100Loader, FashionMNISTLoader, CelebALoader
        logger.info("✓ All data loader imports successful (4 datasets)")
        
        from utils.device_utils import setup_gpu, get_optimal_batch_size
        from utils.helpers import save_experiment_results, set_random_seed
        logger.info("✓ Utility imports successful")
        
        from config_manager import ConfigManager
        from config import DATASETS, MODELS, TRAINING_CONFIG
        logger.info("✓ Configuration imports successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_all_data_loaders():
    """Test initialization of all 4 data loaders"""
    logger.info("Testing all data loaders with unified inheritance...")
    
    try:
        from data import CIFAR10Loader, CIFAR100Loader, FashionMNISTLoader, CelebALoader
        
        loaders = [
            (CIFAR10Loader(), "CIFAR-10", 10, (32, 32, 3)),
            (CIFAR100Loader(label_mode='coarse'), "CIFAR-100", 20, (32, 32, 3)),
            (FashionMNISTLoader(use_local=True), "Fashion-MNIST", 10, (28, 28, 1)),
            (CelebALoader(target_attribute='Smiling', image_size=(64, 64)), "CelebA", 2, (64, 64, 3))
        ]
        
        for loader, name, expected_classes, expected_shape in loaders:
            info = loader.get_basic_info()
            logger.info(f"✓ {name}: {info['num_classes']} classes, {info['input_shape']}")
            
            if info['num_classes'] != expected_classes or info['input_shape'] != expected_shape:
                logger.error(f"✗ {name}: Unexpected configuration")
                return False
        
        test_loader = CIFAR10Loader()
        train_gen, val_gen = test_loader.create_data_generators(batch_size=4)
        logger.info("CIFAR-10 data generators created with batch size 4")
        logger.info("✓ Unified data generator works")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loader test failed: {e}")
        return False


def test_all_models():
    """Test initialization of all 5 model architectures"""
    logger.info("Testing all model architectures...")
    
    try:
        from models import BaselineCNNModel, ResNet50Model, DenseNet121Model, InceptionV3Model, MobileNetModel
        
        models = [
            (BaselineCNNModel(num_classes=10), "Baseline CNN"),
            (ResNet50Model(num_classes=10), "ResNet50"),
            (DenseNet121Model(num_classes=10), "DenseNet121"),
            (InceptionV3Model(num_classes=10), "InceptionV3"),
            (MobileNetModel(num_classes=10), "MobileNet")
        ]
        
        for model, name in models:
            keras_model = model.build_model()
            if keras_model is None:
                logger.error(f"✗ {name} model creation failed")
                return False
            logger.info(f"✓ {name} model creation successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return False


def test_configuration_system():
    """Test configuration management system"""
    logger.info("Testing configuration system...")
    
    try:
        from config import DATASETS, MODELS
        from config_manager import ConfigManager
        
        if len(DATASETS) != 4:
            logger.error(f"✗ Expected 4 datasets, found {len(DATASETS)}")
            return False
        logger.info("✓ All 4 datasets in configuration")
        
        if len(MODELS) != 5:
            logger.error(f"✗ Expected 5 models, found {len(MODELS)}")
            return False
        logger.info("✓ All 5 models in configuration")
        
        config_manager = ConfigManager()
        quick_configs = config_manager.create_quick_configs()
        
        if len(quick_configs) < 20:
            logger.error(f"✗ Expected at least 20 configurations, found {len(quick_configs)}")
            return False
        logger.info("✓ Predefined configurations loaded")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_comprehensive_training_compatibility():
    """Test compatibility between all components"""
    logger.info("Testing comprehensive training script compatibility...")
    
    try:
        from data import CIFAR10Loader, FashionMNISTLoader, CelebALoader, CIFAR100Loader
        from models import BaselineCNNModel, ResNet50Model, DenseNet121Model, InceptionV3Model, MobileNetModel
        
        test_combinations = [
            (BaselineCNNModel, CIFAR10Loader(), "baseline_cnn + cifar10"),
            (ResNet50Model, FashionMNISTLoader(use_local=True), "resnet50 + fashion_mnist"),
            (DenseNet121Model, CelebALoader(target_attribute='Smiling', image_size=(64, 64)), "densenet121 + celeba"),
            (InceptionV3Model, CIFAR100Loader(label_mode='coarse'), "inceptionv3 + cifar100"),
            (MobileNetModel, FashionMNISTLoader(use_local=True), "mobilenet + fashion_mnist")
        ]
        
        for model_class, data_loader, description in test_combinations:
            info = data_loader.get_basic_info()
            model = model_class(num_classes=info['num_classes'], input_shape=info['input_shape'])
            keras_model = model.build_model()
            
            if keras_model is None:
                logger.error(f"✗ {description} compatibility failed")
                return False
            
            logger.info(f"✓ {description} compatibility confirmed")
        
        logger.info("✓ Comprehensive training compatibility confirmed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Comprehensive training test failed: {e}")
        return False


def test_dataset_availability():
    """Test dataset file availability"""
    logger.info("Testing dataset availability...")
    
    datasets_dir = Path("datasets")
    expected_files = [
        ("cifar-10-python.tar.gz", "CIFAR-10"),
        ("cifar-100-python.tar.gz", "CIFAR-100"),
        ("fashion-mnist.tar.gz", "Fashion-MNIST"),
        ("celeba.tar.gz", "CelebA")
    ]
    
    try:
        for filename, name in expected_files:
            file_path = datasets_dir / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"✓ {filename} available ({size_mb:.0f} MB)")
            else:
                logger.error(f"✗ {filename} not found")
                return False
        
        logger.info(f"✓ {len(expected_files)} datasets available")
        return True
        
    except Exception as e:
        logger.error(f"✗ Dataset availability test failed: {e}")
        return False


def test_device_utils():
    """Test device utilities"""
    logger.info("Testing device utilities...")
    
    try:
        from utils.device_utils import setup_gpu, get_optimal_batch_size, get_system_info
        
        gpu_info = setup_gpu()
        logger.info("✓ GPU setup successful")
        
        batch_size = get_optimal_batch_size((32, 32, 3))
        logger.info(f"✓ Optimal batch size calculation: {batch_size}")
        
        system_info = get_system_info()
        logger.info("✓ System info retrieval successful")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Device utilities test failed: {e}")
        return False


def test_helpers():
    """Test helper functions"""
    logger.info("Testing helper functions...")
    
    try:
        from utils.helpers import set_random_seed, get_device_info, format_time, create_experiment_config
        
        set_random_seed(42)
        logger.info("✓ Random seed setting successful")
        
        device_info = get_device_info()
        logger.info("✓ Device info retrieval successful")
        
        time_formatted = format_time(125.7)
        if not time_formatted:
            logger.error("✗ Time formatting failed")
            return False
        
        config = create_experiment_config('TestModel', 'TestDataset')
        if not config or 'experiment_name' not in config:
            logger.error("✗ Experiment config creation failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Helper functions test failed: {e}")
        return False


def test_directory_structure():
    """Test that required directories exist or can be created"""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        Path("models"),
        Path("data"),
        Path("utils"),
        Path("experiments"),
        Path("saved_models"),
        Path("results"),
        Path("configs")
    ]
    
    try:
        for directory in required_dirs:
            if directory.exists():
                logger.info(f"✓ Directory {directory} exists")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Directory {directory} created/exists")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Directory structure test failed: {e}")
        return False


def run_all_tests():
    """Run all system tests"""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE SYSTEM TESTS")
    logger.info("Testing: 5 Models × 4 Datasets + Unified Training System")
    logger.info("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("All Data Loaders (4 datasets)", test_all_data_loaders),
        ("All Models (5 architectures)", test_all_models),
        ("Configuration System", test_configuration_system),
        ("Comprehensive Training Compatibility", test_comprehensive_training_compatibility),
        ("Dataset Availability", test_dataset_availability),
        ("Device Utilities", test_device_utils),
        ("Helper Functions", test_helpers),
        ("Directory Structure", test_directory_structure)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("ALL TESTS PASSED!")
        logger.info("System is ready for comprehensive training!")
        logger.info("All 5 models × 4 datasets = 20 combinations available")
        return True
    else:
        logger.error("SOME TESTS FAILED! Please fix issues before training.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 