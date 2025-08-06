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

# Activate virtual environment Windows
venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
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

#### **Batch Evaluation**
```bash
# Evaluate all CIFAR-10 models
python evaluate_existing_models.py --dataset cifar10 --evaluate-all

# Evaluate all Fashion-MNIST models
python evaluate_existing_models.py --dataset fashion_mnist --evaluate-all

```

# Key directories:
# models/          - Model implementations
# data/           - Dataset loaders  
# experiments/    - Training scripts
# utils/          - Helper utilities
# results/        - Training results and logs
# saved_models/   - Trained model files
