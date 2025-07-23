"""
Comprehensive Configuration Demo
Showcases all configuration management capabilities for the CNN Image Classification Project
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import *
from config_manager import ConfigManager
import argparse


def demo_basic_configurations():
    """Demo basic configuration access"""
    print("BASIC CONFIGURATIONS")
    print("=" * 30)
    
    print(f"Available Models: {len(MODELS)}")
    for model in MODELS:
        print(f"  - {model}: {MODELS[model]['name']}")
    
    print(f"\nAvailable Datasets: {len(DATASETS)}")
    for dataset in DATASETS:
        print(f"  - {dataset}: {DATASETS[dataset]['name']}")
    
    print(f"\nTotal Combinations: {TOTAL_EXPERIMENTS}")


def demo_optimal_configurations():
    """Demo optimal configuration generation"""
    print("\nOPTIMAL CONFIGURATIONS")
    print("=" * 50)
    
    # Test different model-dataset combinations
    test_combinations = [
        ('resnet50', 'cifar10'),
        ('densenet121', 'cifar100'),
        ('mobilenet', 'fashion_mnist'),
        ('inceptionv3', 'celeba'),
        ('baseline_cnn', 'cifar10')
    ]
    
    for model, dataset in test_combinations:
        print(f"\n{model.upper()} on {dataset.upper()}:")
        try:
            config = get_optimal_config(model, dataset)
            perf = get_expected_performance(model, dataset)
            validation = validate_config_combination(model, dataset)
            
            print(f"  Epochs: {config['epochs']}")
            print(f"  Batch Size: {config['batch_size']}")
            print(f"  Learning Rate: {config['learning_rate']}")
            print(f"  Expected Accuracy: {perf['accuracy']:.1%}")
            print(f"  Estimated Time: {estimate_training_time(model, dataset, config['epochs'])}")
            
            if validation['warnings']:
                print(f"  WARNING: {len(validation['warnings'])} warnings")
            if validation['recommendations']:
                print(f"  TIP: {len(validation['recommendations'])} recommendations")
                
        except Exception as e:
            print(f"  ERROR: {e}")


def demo_augmentation_configs():
    """Demo augmentation configuration system"""
    print("\nAUGMENTATION CONFIGURATIONS")
    print("=" * 40)
    
    for dataset in ['cifar10', 'cifar100', 'fashion_mnist', 'celeba']:
        print(f"\n{dataset.upper()}:")
        config = get_augmentation_config(dataset)
        print(f"  Default Level: {config.get('description', 'Unknown')}")
        print(f"  Enabled: {config.get('enabled', False)}")
        if config.get('enabled'):
            print(f"  Rotation: {config.get('rotation_range', 0)}°")
            print(f"  Shift: {config.get('width_shift_range', 0)}")
            print(f"  Flip: {config.get('horizontal_flip', False)}")


def demo_training_summaries():
    """Demo training summary generation"""
    print("\nTRAINING SUMMARIES")
    print("=" * 30)
    
    # Show detailed summaries for a few combinations
    test_cases = [
        ('resnet50', 'cifar10'),
        ('mobilenet', 'fashion_mnist'),
        ('densenet121', 'cifar100')
    ]
    
    for model, dataset in test_cases:
        print(f"\n{'-' * 40}")
        summary = get_training_summary(model, dataset)
        print(summary)


def demo_config_manager():
    """Demo ConfigManager functionality"""
    print("\nCONFIG MANAGER DEMO")
    print("=" * 30)
    
    config_manager = ConfigManager()
    
    # Show available models and datasets
    print("Available model-dataset pairs:")
    for model_key in list(config_manager.predefined_models.keys())[:5]:
        print(f"  - {model_key}")
    
    print(f"\nAvailable Configurations:")
    quick_configs = config_manager.create_quick_configs()
    for i, (config_name, config) in enumerate(list(quick_configs.items())[:5]):
        print(f"  {i+1}. {config_name}: {config.model.name} on {config.dataset.name}")
    
    print(f"\nQuick Configurations:")
    quick_config_names = list(quick_configs.keys())[:3]
    for config_name in quick_config_names:
        config = quick_configs[config_name]
        print(f"  {config_name}:")
        print(f"    Model: {config.model.name}")
        print(f"    Dataset: {config.dataset.name}")
        print(f"    Epochs: {config.training.epochs}")
        print(f"    Batch Size: {config.training.batch_size}")


def demo_recommended_configs():
    """Demo recommended configurations for different use cases"""
    print("\nRECOMMENDED CONFIGURATIONS")
    print("=" * 40)
    
    use_cases = {
        'quick_test': ['mobilenet_fashion_mnist_fast', 'baseline_cnn_cifar10_fast'],
        'research_quality': ['densenet121_cifar100_standard', 'resnet50_cifar10_standard'],
        'transfer_learning': ['inceptionv3_celeba_finetune', 'resnet50_fashion_mnist_finetune']
    }
    
    for use_case, configs in use_cases.items():
        print(f"\n{use_case.upper()} Use Case:")
        for config_name in configs:
            try:
                parts = config_name.split('_')
                if len(parts) >= 2:
                    model = parts[0]
                    dataset = parts[1]
                    print(f"  - {config_name}")
                    print(f"    {model.title()} on {dataset.upper()}")
                    
                    optimal_config = get_optimal_config(model, dataset)
                    print(f"    Epochs: {optimal_config['epochs']}, Batch: {optimal_config['batch_size']}")
                    
                    expected = get_expected_performance(model, dataset)
                    print(f"    Expected Accuracy: {expected['accuracy']:.1%}")
            except Exception as e:
                print(f"  - {config_name}: Error - {e}")


def demo_comprehensive_analysis():
    """Demo comprehensive configuration analysis"""
    print("\nCOMPREHENSIVE ANALYSIS")
    print("=" * 35)
    
    config_manager = ConfigManager()
    
    print("Configuration Matrix:")
    matrix = config_manager.get_configuration_matrix()
    print(matrix[:500] + "..." if len(matrix) > 500 else matrix)
    
    print("\nDetailed Analysis:")
    analysis = config_manager.analyze_all_configurations()
    print(analysis[:800] + "..." if len(analysis) > 800 else analysis)


def demo_performance_benchmarks():
    """Demo performance benchmark data"""
    print("\nPERFORMANCE BENCHMARKS")
    print("=" * 35)
    
    from config import PERFORMANCE_BENCHMARKS
    
    models = ['baseline_cnn', 'resnet50', 'densenet121']
    datasets = ['cifar10', 'fashion_mnist']
    
    print(f"{'Model':<15} {'Dataset':<15} {'Accuracy':<10} {'Range':<15}")
    print("-" * 55)
    
    for model in models:
        for dataset in datasets:
            if model in PERFORMANCE_BENCHMARKS and dataset in PERFORMANCE_BENCHMARKS[model]:
                perf = PERFORMANCE_BENCHMARKS[model][dataset]
                accuracy = f"{perf['accuracy']:.1%}"
                range_str = f"{perf['range'][0]:.1%}-{perf['range'][1]:.1%}"
                print(f"{model:<15} {dataset:<15} {accuracy:<10} {range_str:<15}")


def demo_save_load_configs():
    """Demo saving and loading configurations"""
    print("\nSAVE & LOAD CONFIGURATIONS")
    print("=" * 40)
    
    config_manager = ConfigManager()
    
    # Create a test configuration
    test_config = config_manager.create_experiment_config(
        experiment_name="demo_test_config",
        model_key="resnet50_cifar10",
        dataset_key="cifar10",
        training_key="standard"
    )
    
    # Save the configuration
    config_path = config_manager.save_config(test_config)
    print(f"Configuration saved to: {config_path}")
    
    # Load the configuration
    loaded_config = config_manager.load_config(config_path)
    print(f"Configuration loaded successfully: {loaded_config.experiment_name}")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Configuration System Demo')
    parser.add_argument('--section', choices=[
        'basic', 'optimal', 'augmentation', 'summaries', 'manager', 
        'recommended', 'analysis', 'benchmarks', 'save-load', 'all'
    ], default='all', help='Which section to demo')
    
    args = parser.parse_args()
    
    print("CNN IMAGE CLASSIFICATION - CONFIGURATION DEMO")
    print("=" * 60)
    print(f"Total Models: {len(MODELS)} | Total Datasets: {len(DATASETS)} | Total Combinations: {TOTAL_EXPERIMENTS}")
    print("=" * 60)
    
    if args.section in ['basic', 'all']:
        demo_basic_configurations()
    
    if args.section in ['optimal', 'all']:
        demo_optimal_configurations()
    
    if args.section in ['augmentation', 'all']:
        demo_augmentation_configs()
    
    if args.section in ['summaries', 'all']:
        demo_training_summaries()
    
    if args.section in ['manager', 'all']:
        demo_config_manager()
    
    if args.section in ['recommended', 'all']:
        demo_recommended_configs()
    
    if args.section in ['analysis', 'all']:
        demo_comprehensive_analysis()
    
    if args.section in ['benchmarks', 'all']:
        demo_performance_benchmarks()
    
    if args.section in ['save-load', 'all']:
        demo_save_load_configs()
    
    print("\nCONFIGURATION DEMO COMPLETED!")
    print("=" * 60)
    print("Next Steps:")
    print("  • Run experiments with: python experiments/comprehensive_train.py")
    print("  • Test system with: python test_system.py")
    print("  • View all configs with: python demo_configurations.py --section analysis")


if __name__ == "__main__":
    main()