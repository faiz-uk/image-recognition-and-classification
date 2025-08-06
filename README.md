# CNN Image Classification Research Pipeline

A comprehensive deep learning pipeline for image classification research using Convolutional Neural Networks (CNNs) with TensorFlow/Keras. This project implements **5 model architectures** across **4 datasets** with **intelligent configuration management** for comparative analysis.

## Project Overview

This research pipeline enables comprehensive CNN experiments with:
- **5 Model Architectures**: BaselineCNN, ResNet50, DenseNet121, InceptionV3, MobileNet
- **4 Datasets**: CIFAR-10, CIFAR-100, Fashion-MNIST, CelebA
- **Comprehensive Evaluation**: Multi-metric analysis with visualizations
- **Intelligent Configuration**: Automated hyperparameter optimization

## Model Architectures

### 1. **BaselineCNN** (`BaselineCNNModel`)
- Custom CNN architecture trained from scratch
- Configurable layers and parameters
- Perfect for comparative baseline analysis
- **Best Performance**: 93.10% (Fashion-MNIST), 80.36% (CIFAR-10)

### 2. **ResNet50** (`ResNet50Model`)
- ImageNet pre-trained weights with transfer learning
- Residual connections for deep network training
- Freezable base layers for fine-tuning
- Optimized for small images (32×32)

### 3. **DenseNet121** (`DenseNet121Model`)
- Dense connectivity pattern for efficient parameter usage
- ImageNet pre-trained weights
- State-of-the-art performance with fewer parameters
- Excellent gradient flow and feature reuse

### 4. **InceptionV3** (`InceptionV3Model`)
- Multi-scale feature extraction with factorized convolutions
- ImageNet pre-trained weights
- Handles input resizing automatically
- Complex architecture for challenging datasets

### 5. **MobileNet** (`MobileNetModel`)
- Depthwise separable convolutions for mobile optimization
- Lightweight architecture with configurable width multiplier
- Fast training and inference
- Mobile-optimized deployment ready

## Supported Datasets

- **CIFAR-10**: 10 classes, 32×32 RGB natural images
- **CIFAR-100**: 100 classes, 32×32 RGB natural images  
- **Fashion-MNIST**: 10 classes, 28×28 grayscale fashion items
- **CelebA**: Binary classification, 64×64 RGB celebrity faces

## Quick Start

### **1. Environment Setup**
```bash
# Clone/navigate to project directory
cd image-recognition-and-classification

# Create virtual environment
python3 -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate
# OR Windows
# venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### **2. Verify Installation**
```bash
# Test environment setup
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU Available: {tf.config.list_physical_devices(\"GPU\")}')"
```

## Training Commands

### **Single Model Training**

#### **Basic Training**
```bash
# Train ResNet50 on CIFAR-10 with optimal settings
python experiments/comprehensive_train.py --model resnet50 --dataset cifar10

# Train DenseNet121 on Fashion-MNIST
python experiments/comprehensive_train.py --model densenet121 --dataset fashion_mnist

# Train BaselineCNN on CIFAR-100
python experiments/comprehensive_train.py --model baseline_cnn --dataset cifar100
```

#### **Custom Configuration Training**
```bash
# Custom epochs and learning rate
python experiments/comprehensive_train.py --model mobilenet --dataset cifar10 --epochs 150 --learning-rate 0.0001

# Custom batch size and dropout
python experiments/comprehensive_train.py --model inceptionv3 --dataset celeba --batch-size 16 --dropout-rate 0.3

# Fast training for testing (5 minutes)
python experiments/comprehensive_train.py --model mobilenet --dataset fashion_mnist --epochs 5 --batch-size 64
```

### **Comprehensive Model Comparison**

#### **All Models on Single Dataset**
```bash
# Compare all 5 models on CIFAR-10
python experiments/comprehensive_train.py --dataset cifar10 --comprehensive-models

# Compare all models on Fashion-MNIST
python experiments/comprehensive_train.py --dataset fashion_mnist --comprehensive-models

# Compare all models on CIFAR-100 (long training)
python experiments/comprehensive_train.py --dataset cifar100 --comprehensive-models --epochs 200
```

#### **Single Model on All Datasets**
```bash
# Train ResNet50 on all datasets
python experiments/comprehensive_train.py --model resnet50 --comprehensive-datasets

# Train BaselineCNN on all datasets
python experiments/comprehensive_train.py --model baseline_cnn --comprehensive-datasets
```

## Evaluation Commands

### **Evaluate Existing Models**

#### **Single Model Evaluation**
```bash
# Evaluate specific model
python evaluate_existing_models.py --model-path saved_models/baseline_cnn_cifar10_20250723_114007_best.keras --dataset cifar10

# Evaluate with detailed metrics
python evaluate_existing_models.py --model-path saved_models/densenet121_fashion_mnist_20250724_063907_best.keras --dataset fashion_mnist --detailed-metrics
```

#### **Batch Evaluation**
```bash
# Evaluate all CIFAR-10 models
python evaluate_existing_models.py --dataset cifar10 --evaluate-all

# Evaluate all Fashion-MNIST models
python evaluate_existing_models.py --dataset fashion_mnist --evaluate-all

# Evaluate all models across all datasets
python evaluate_existing_models.py --evaluate-all-combinations
```

#### **Model Comparison**
```bash
# Compare models on specific dataset
python evaluate_existing_models.py --dataset cifar10 --compare-models --generate-report

# Generate comprehensive comparison report
python evaluate_existing_models.py --comprehensive-evaluation --output-format html
```

#### **File Organization**
```bash
# Reorganize loose evaluation files into organized directories
python evaluate_existing_models.py --reorganize-files
```

## Configuration Management

### **View Available Configurations**
```bash
# List all available model configurations
python -c "from config import MODELS; print('Available Models:', list(MODELS.keys()))"

# List all dataset configurations  
python -c "from config import DATASETS; print('Available Datasets:', list(DATASETS.keys()))"

# View optimal configuration for model-dataset combination
python -c "from config import get_optimal_config; print(get_optimal_config('resnet50', 'cifar10'))"
```

### **Performance Predictions**
```bash
# Get expected performance for model-dataset combination
python -c "from config import get_expected_performance; print(get_expected_performance('densenet121', 'fashion_mnist'))"

# Estimate training time
python -c "from config import estimate_training_time; print(estimate_training_time('inceptionv3', 'cifar100', 100))"
```

## Results and Analysis

### **View Training Results**
```bash
# List all trained models
ls -la saved_models/

# View training history
python -c "
import json
with open('results/logs/model_cifar10_20250723_114007_history.json') as f:
    history = json.load(f)
    print(f'Final Accuracy: {history[\"accuracy\"][-1]:.4f}')
    print(f'Final Val Accuracy: {history[\"val_accuracy\"][-1]:.4f}')
"
```

### **Generate Reports**
```bash
# Generate comprehensive evaluation report
python utils/evaluation_metrics.py --generate-report --dataset cifar10

# Create performance comparison charts
python utils/visualization.py --create-comparison-charts --output results/plots/

# Export results to CSV
python utils/export_results.py --format csv --output results/tables/summary.csv
```

## Model Inspection

### **Model Architecture Analysis**
```bash
# View model summary
python -c "
from models.resnet50 import ResNet50Model
model = ResNet50Model(num_classes=10, input_shape=(32, 32, 3))
built_model = model.build_model()
built_model.summary()
"

# Count parameters for all models
python -c "
from models import *
models = ['baseline_cnn', 'resnet50', 'densenet121', 'inceptionv3', 'mobilenet']
for name in models:
    print(f'{name}: Parameters will be shown during training')
"
```

### **Data Pipeline Inspection**
```bash
# Test data loading
python -c "
from data import CIFAR10Loader
loader = CIFAR10Loader()
X_train, X_val, X_test, y_train, y_val, y_test = loader.load_and_preprocess()
print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
"

# View dataset information
python -c "
from data import FashionMNISTLoader
loader = FashionMNISTLoader()
info = loader.get_data_info()
print(info)
"
```

## Debugging and Troubleshooting

### **Check GPU Availability**
```bash
# Verify GPU setup
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))
print('Built with CUDA:', tf.test.is_built_with_cuda())
"
```

### **Test Individual Components**
```bash
# Test model loading
python -c "
from models.baseline_cnn import BaselineCNNModel
model = BaselineCNNModel(num_classes=10)
print('BaselineCNN initialized successfully')
"

# Test data loading
python -c "
from data.cifar10_loader import CIFAR10Loader
loader = CIFAR10Loader()
print('CIFAR-10 loader initialized successfully')
"

# Test evaluation metrics
python -c "
from utils.evaluation_metrics import ComprehensiveEvaluator
evaluator = ComprehensiveEvaluator()
print('Evaluator initialized successfully')
"
```

### **Lambda Layer Issue Resolution**
```bash
# For models with Lambda layer issues (Fashion-MNIST transfer learning models)
python evaluate_existing_models.py --model-path saved_models/resnet50_fashion_mnist_*.keras --dataset fashion_mnist --fix-lambda-layers

# Alternative: Recreate models without Lambda layers
python -c "
# See LAMBDA_ISSUE_SOLUTIONS.md for detailed fix instructions
print('Refer to LAMBDA_ISSUE_SOLUTIONS.md for Lambda layer fixes')
"
```

## Project Structure Navigation

```bash
# View project structure
tree -I '__pycache__|*.pyc|venv' -L 3

# Key directories:
# models/          - Model implementations
# data/           - Dataset loaders  
# experiments/    - Training scripts
# utils/          - Helper utilities
# results/        - Training results and logs
# saved_models/   - Trained model files
# documentation/  - Project documentation
```

## Advanced Usage

### **Custom Model Development**
```bash
# Create custom model based on BaseModel
python -c "
from models.base_model import BaseModel
# Inherit from BaseModel and implement build_model()
print('See models/baseline_cnn.py for implementation example')
"
```

### **Hyperparameter Optimization**
```bash
# Grid search over hyperparameters (implement custom script)
python experiments/hyperparameter_search.py --model resnet50 --dataset cifar10 --param-grid config/hyperparam_grid.json
```

### **Distributed Training** (Future Enhancement)
```bash
# Multi-GPU training (when implemented)
python experiments/distributed_train.py --model densenet121 --dataset cifar100 --gpus 2
```

## Performance Benchmarks

### **Current Best Results**
- **Fashion-MNIST**: BaselineCNN (93.10% accuracy)
- **CIFAR-10**: BaselineCNN (80.36% accuracy)  
- **CIFAR-100**: InceptionV3 (44.36% accuracy)
- **CelebA**: Various models (~52% accuracy)

### **Training Time Estimates**
```bash
# Quick training (5-10 minutes)
python experiments/comprehensive_train.py --model mobilenet --dataset fashion_mnist --epochs 10

# Medium training (30-60 minutes)  
python experiments/comprehensive_train.py --model densenet121 --dataset cifar10 --epochs 50

# Long training (2-4 hours)
python experiments/comprehensive_train.py --model inceptionv3 --dataset cifar100 --epochs 200
```

## Research Features

### **Experimental Analysis**
```bash
# Learning curve analysis
python utils/plot_learning_curves.py --model-history results/logs/model_*_history.json

# Confusion matrix generation
python utils/generate_confusion_matrix.py --model-path saved_models/*.keras --dataset cifar10

# Feature visualization (when implemented)
python utils/visualize_features.py --model-path saved_models/densenet121_*.keras --layer-name conv2d_5
```

### **Model Comparison Studies**
```bash
# Statistical significance testing
python utils/statistical_analysis.py --results-dir results/comprehensive_evaluation/

# Performance correlation analysis  
python utils/correlation_analysis.py --metrics-file results/tables/all_results.csv
```

## Maintenance Commands

### **Clean Up**
```bash
# Remove temporary files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Clean old model checkpoints
find saved_models/ -name "*checkpoint*" -delete

# Archive old results
mkdir -p results/archive/$(date +%Y%m%d)
mv results/logs/* results/archive/$(date +%Y%m%d)/
```

### **Backup Important Files**
```bash
# Backup trained models
tar -czf backup_models_$(date +%Y%m%d).tar.gz saved_models/

# Backup results
tar -czf backup_results_$(date +%Y%m%d).tar.gz results/

# Backup configuration
cp config.py config_backup_$(date +%Y%m%d).py
```

## Documentation

- **Project Overview**: `documentation/project_overview.md`
- **Results Analysis**: `documentation/results.md`  
- **Conclusions**: `documentation/conclusion.md`
- **Lambda Issue**: `LAMBDA_ISSUE_SOLUTIONS.md`
- **Code Documentation**: Inline docstrings in all modules

## Contributing

```bash
# Set up development environment
git clone <repository-url>
cd image-recognition-and-classification
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Run tests (when implemented)
python -m pytest tests/

# Code formatting
black . --line-length 88
isort . --profile black
```

## License

This project follows academic research guidelines and open-source principles. Refer to individual model licenses for pre-trained weights usage.

---

## Quick Command Reference

| Task | Command |
|------|---------|
| **Train single model** | `python experiments/comprehensive_train.py --model resnet50 --dataset cifar10` |
| **Compare all models** | `python experiments/comprehensive_train.py --dataset cifar10 --comprehensive-models` |
| **Evaluate model** | `python evaluate_existing_models.py --model-path saved_models/*.keras --dataset cifar10` |
| **Quick test** | `python experiments/comprehensive_train.py --model mobilenet --dataset fashion_mnist --epochs 5` |
| **View results** | `ls results/comprehensive_evaluation/` |
| **Check GPU** | `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` |
| **Reorganize files** | `python evaluate_existing_models.py --reorganize-files` |

For detailed usage and advanced features, refer to the documentation in the `documentation/` directory.
