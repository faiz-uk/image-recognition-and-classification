# CNN Image Classification Research Pipeline

A comprehensive deep learning pipeline for image classification research using Convolutional Neural Networks (CNNs) with TensorFlow/Keras. This project implements **5 model architectures** across **4 datasets** with **intelligent configuration management** for comparative analysis.

## Project Overview

This research pipeline enables comprehensive CNN experiments with:
- **5 Model Architectures**: BaselineCNN, ResNet50, DenseNet121, InceptionV3, MobileNet
- **4 Datasets**: CIFAR-10, CIFAR-100, Fashion-MNIST, CelebA

#### **Five Model Architectures**

1. **BaselineCNN** (`BaselineCNNModel`)
   - Custom CNN architecture
   - Configurable layers and parameters
   - No pre-trained weights (trained from scratch)
   - Perfect for comparative analysis

2. **ResNet50** (`ResNet50Model`)
   - ImageNet pre-trained weights
   - Transfer learning capabilities
   - Freezable base layers
   - Optimized for small images (32×32)

3. **DenseNet121** (`DenseNet121Model`)
   - Dense connectivity pattern
   - ImageNet pre-trained weights
   - Efficient parameter usage
   - State-of-the-art performance

4. **InceptionV3** (`InceptionV3Model`)
   - Multi-scale feature extraction
   - ImageNet pre-trained weights
   - Handles input resizing automatically
   - Complex architecture for challenging datasets

5. **MobileNet** (`MobileNetModel`)
   - Depthwise separable convolutions
   - Mobile-optimized architecture
   - Configurable width multiplier (alpha)
   - Fast training and inference


## Getting Started

### **1. Environment Setup**
```bash
# Clone/navigate to project directory
cd image-recognition-and-classification

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # or source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```



## Training & Experiments

### **Intelligent Training with Optimal Configurations**

The system automatically applies optimal configurations for each model-dataset combination:

```bash
# Automatic optimization - system chooses best parameters
python experiments/comprehensive_train.py --model resnet50 --dataset cifar10

# The system automatically applies:
# - Epochs: 100 (optimized for ResNet50)
# - Batch size: 32 (balanced for CIFAR-10)
# - Learning rate: 0.0005 (transfer learning optimized)
# - Augmentation: Standard level for CIFAR-10
```

### **Quick Start Examples**

```bash
# Fast validation (5 minutes)
python experiments/comprehensive_train.py --model mobilenet --dataset fashion_mnist --epochs 5

# Research-quality training (2-3 hours)
python experiments/comprehensive_train.py --model densenet121 --dataset cifar100 --epochs 100

# All models on one dataset for comparison
python experiments/comprehensive_train.py --dataset cifar10 --comprehensive-models
```
