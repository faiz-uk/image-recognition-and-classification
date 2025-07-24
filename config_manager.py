"""
Configuration Management System for CNN Image Classification Project
Provides centralized configuration management for training different datasets on different models
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    name: str
    architecture: str
    input_shape: tuple
    num_classes: int
    weights: str = 'imagenet'
    include_top: bool = False
    dropout_rate: float = 0.5
    hidden_units: int = 512
    freeze_base: bool = True


@dataclass
class DatasetConfig:
    """Configuration for dataset"""
    name: str
    dataset_type: str  # 'cifar10', 'cifar100', 'custom'
    num_classes: int
    input_shape: tuple
    validation_split: float = 0.2
    label_mode: str = 'fine'  # For CIFAR-100: 'fine' or 'coarse'
    class_names: Optional[List[str]] = None


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss: str = 'categorical_crossentropy'
    metrics: List[str] = None
    validation_split: float = 0.2
    patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy']


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    enabled: bool = True
    rotation_range: float = 20.0
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = False
    zoom_range: float = 0.2
    shear_range: float = 0.15
    fill_mode: str = 'nearest'
    brightness_range: Optional[tuple] = None
    channel_shift_range: float = 0.0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    timestamp: str
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ConfigManager:
    """
    Configuration Manager for handling all experiment configurations
    """
    
    def __init__(self, config_dir: Path = None):
        """
        Initialize Configuration Manager
        
        Args:
            config_dir: Directory to store configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Predefined configurations
        self.predefined_models = self._load_predefined_models()
        self.predefined_datasets = self._load_predefined_datasets()
        self.predefined_training = self._load_predefined_training()
        self.predefined_augmentation = self._load_predefined_augmentation()
        
        logger.info(f"ConfigManager initialized with config directory: {self.config_dir}")
    
    def _load_predefined_models(self) -> Dict[str, ModelConfig]:
        """Load predefined model configurations"""
        return {
            # Baseline CNN configurations
            'baseline_cnn_cifar10': ModelConfig(
                name='BaselineCNN_CIFAR10',
                architecture='baseline_cnn',
                input_shape=(32, 32, 3),
                num_classes=10,
                weights=None,  # No pre-trained weights
                dropout_rate=0.5,
                hidden_units=512,
                freeze_base=False  # Not applicable for baseline
            ),
            'baseline_cnn_cifar100': ModelConfig(
                name='BaselineCNN_CIFAR100',
                architecture='baseline_cnn',
                input_shape=(32, 32, 3),
                num_classes=100,
                weights=None,
                dropout_rate=0.6,  # Higher dropout for more classes
                hidden_units=1024,
                freeze_base=False
            ),
            'baseline_cnn_fashion_mnist': ModelConfig(
                name='BaselineCNN_FashionMNIST',
                architecture='baseline_cnn',
                input_shape=(28, 28, 1),
                num_classes=10,
                weights=None,
                dropout_rate=0.4,  # Lower dropout for easier dataset
                hidden_units=256,
                freeze_base=False
            ),
            'baseline_cnn_celeba': ModelConfig(
                name='BaselineCNN_CelebA',
                architecture='baseline_cnn',
                input_shape=(64, 64, 3),
                num_classes=2,  # Binary classification
                weights=None,
                dropout_rate=0.5,
                hidden_units=512,
                freeze_base=False
            ),
            
            # ResNet50 configurations
            'resnet50_cifar10': ModelConfig(
                name='ResNet50_CIFAR10',
                architecture='resnet50',
                input_shape=(32, 32, 3),
                num_classes=10,
                weights='imagenet',
                dropout_rate=0.5,
                hidden_units=512,
                freeze_base=True
            ),
            'resnet50_cifar100': ModelConfig(
                name='ResNet50_CIFAR100',
                architecture='resnet50',
                input_shape=(32, 32, 3),
                num_classes=100,
                weights='imagenet',
                dropout_rate=0.6,
                hidden_units=1024,
                freeze_base=True
            ),
            'resnet50_fashion_mnist': ModelConfig(
                name='ResNet50_FashionMNIST',
                architecture='resnet50',
                input_shape=(28, 28, 1),
                num_classes=10,
                weights='imagenet',
                dropout_rate=0.4,
                hidden_units=512,
                freeze_base=True
            ),
            'resnet50_celeba': ModelConfig(
                name='ResNet50_CelebA',
                architecture='resnet50',
                input_shape=(64, 64, 3),
                num_classes=2,
                weights='imagenet',
                dropout_rate=0.5,
                hidden_units=512,
                freeze_base=True
            ),
            
            # DenseNet121 configurations
            'densenet121_cifar10': ModelConfig(
                name='DenseNet121_CIFAR10',
                architecture='densenet121',
                input_shape=(32, 32, 3),
                num_classes=10,
                weights='imagenet',
                dropout_rate=0.5,
                hidden_units=512,
                freeze_base=True
            ),
            'densenet121_cifar100': ModelConfig(
                name='DenseNet121_CIFAR100',
                architecture='densenet121',
                input_shape=(32, 32, 3),
                num_classes=100,
                weights='imagenet',
                dropout_rate=0.6,
                hidden_units=1024,
                freeze_base=True
            ),
            'densenet121_fashion_mnist': ModelConfig(
                name='DenseNet121_FashionMNIST',
                architecture='densenet121',
                input_shape=(28, 28, 1),
                num_classes=10,
                weights='imagenet',
                dropout_rate=0.4,
                hidden_units=512,
                freeze_base=True
            ),
            'densenet121_celeba': ModelConfig(
                name='DenseNet121_CelebA',
                architecture='densenet121',
                input_shape=(64, 64, 3),
                num_classes=2,
                weights='imagenet',
                dropout_rate=0.5,
                hidden_units=512,
                freeze_base=True
            ),
            
            # InceptionV3 configurations
            'inceptionv3_cifar10': ModelConfig(
                name='InceptionV3_CIFAR10',
                architecture='inceptionv3',
                input_shape=(32, 32, 3),
                num_classes=10,
                weights='imagenet',
                dropout_rate=0.5,
                hidden_units=512,
                freeze_base=True
            ),
            'inceptionv3_cifar100': ModelConfig(
                name='InceptionV3_CIFAR100',
                architecture='inceptionv3',
                input_shape=(32, 32, 3),
                num_classes=100,
                weights='imagenet',
                dropout_rate=0.6,
                hidden_units=1024,
                freeze_base=True
            ),
            'inceptionv3_fashion_mnist': ModelConfig(
                name='InceptionV3_FashionMNIST',
                architecture='inceptionv3',
                input_shape=(28, 28, 1),
                num_classes=10,
                weights='imagenet',
                dropout_rate=0.4,
                hidden_units=512,
                freeze_base=True
            ),
            'inceptionv3_celeba': ModelConfig(
                name='InceptionV3_CelebA',
                architecture='inceptionv3',
                input_shape=(64, 64, 3),
                num_classes=2,
                weights='imagenet',
                dropout_rate=0.5,
                hidden_units=512,
                freeze_base=True
            ),
            
            # MobileNet configurations
            'mobilenet_cifar10': ModelConfig(
                name='MobileNet_CIFAR10',
                architecture='mobilenet',
                input_shape=(32, 32, 3),
                num_classes=10,
                weights='imagenet',
                dropout_rate=0.4,  # Lower dropout for efficient model
                hidden_units=256,  # Smaller hidden layer
                freeze_base=True
            ),
            'mobilenet_cifar100': ModelConfig(
                name='MobileNet_CIFAR100',
                architecture='mobilenet',
                input_shape=(32, 32, 3),
                num_classes=100,
                weights='imagenet',
                dropout_rate=0.5,
                hidden_units=512,
                freeze_base=True
            ),
            'mobilenet_fashion_mnist': ModelConfig(
                name='MobileNet_FashionMNIST',
                architecture='mobilenet',
                input_shape=(28, 28, 1),
                num_classes=10,
                weights='imagenet',
                dropout_rate=0.3,
                hidden_units=256,
                freeze_base=True
            ),
            'mobilenet_celeba': ModelConfig(
                name='MobileNet_CelebA',
                architecture='mobilenet',
                input_shape=(64, 64, 3),
                num_classes=2,
                weights='imagenet',
                dropout_rate=0.4,
                hidden_units=256,
                freeze_base=True
            )
        }
    
    def _load_predefined_datasets(self) -> Dict[str, DatasetConfig]:
        """Load predefined dataset configurations"""
        return {
            'cifar10': DatasetConfig(
                name='CIFAR-10',
                dataset_type='cifar10',
                num_classes=10,
                input_shape=(32, 32, 3),
                validation_split=0.2,
                class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
            ),
            'cifar100_fine': DatasetConfig(
                name='CIFAR-100-Fine',
                dataset_type='cifar100',
                num_classes=100,
                input_shape=(32, 32, 3),
                validation_split=0.2,
                label_mode='fine'
            ),
            'cifar100_coarse': DatasetConfig(
                name='CIFAR-100-Coarse',
                dataset_type='cifar100',
                num_classes=20,
                input_shape=(32, 32, 3),
                validation_split=0.2,
                label_mode='coarse'
            ),
            'fashion_mnist': DatasetConfig(
                name='Fashion MNIST',
                dataset_type='fashion_mnist',
                num_classes=10,
                input_shape=(28, 28, 1),
                validation_split=0.2
            ),
            'celeba': DatasetConfig(
                name='CelebA',
                dataset_type='celeba',
                num_classes=2,
                input_shape=(64, 64, 3),
                validation_split=0.2
            )
        }
    
    def _load_predefined_training(self) -> Dict[str, TrainingConfig]:
        """Load predefined training configurations"""
        return {
            'standard': TrainingConfig(
                batch_size=32,
                epochs=100,
                learning_rate=0.001,
                optimizer='adam',
                patience=10,
                reduce_lr_patience=5
            ),
            'fast': TrainingConfig(
                batch_size=64,
                epochs=50,
                learning_rate=0.01,
                optimizer='sgd',
                patience=5,
                reduce_lr_patience=3
            ),
            'fine_tuning': TrainingConfig(
                batch_size=16,
                epochs=50,
                learning_rate=0.0001,
                optimizer='adam',
                patience=15,
                reduce_lr_patience=7
            ),
            'large_dataset': TrainingConfig(
                batch_size=128,
                epochs=200,
                learning_rate=0.001,
                optimizer='adam',
                patience=20,
                reduce_lr_patience=10
            )
        }
    
    def _load_predefined_augmentation(self) -> Dict[str, AugmentationConfig]:
        """Load predefined augmentation configurations"""
        return {
            'standard': AugmentationConfig(
                enabled=True,
                rotation_range=20.0,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.15
            ),
            'aggressive': AugmentationConfig(
                enabled=True,
                rotation_range=30.0,
                width_shift_range=0.3,
                height_shift_range=0.3,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.3,
                shear_range=0.2,
                brightness_range=(0.8, 1.2),
                channel_shift_range=0.2
            ),
            'minimal': AugmentationConfig(
                enabled=True,
                rotation_range=10.0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1
            ),
            'none': AugmentationConfig(
                enabled=False
            )
        }
    
    def create_experiment_config(self,
                               experiment_name: str,
                               model_key: str,
                               dataset_key: str,
                               training_key: str = 'standard',
                               augmentation_key: str = 'standard',
                               description: str = "",
                               tags: List[str] = None,
                               custom_overrides: Dict[str, Any] = None) -> ExperimentConfig:
        """
        Create experiment configuration from predefined components
        
        Args:
            experiment_name: Name for the experiment
            model_key: Key for predefined model configuration
            dataset_key: Key for predefined dataset configuration
            training_key: Key for predefined training configuration
            augmentation_key: Key for predefined augmentation configuration
            description: Description of the experiment
            tags: Tags for the experiment
            custom_overrides: Custom overrides for any configuration
            
        Returns:
            ExperimentConfig object
        """
        # Get predefined configurations
        model_config = self.predefined_models.get(model_key)
        dataset_config = self.predefined_datasets.get(dataset_key)
        training_config = self.predefined_training.get(training_key)
        augmentation_config = self.predefined_augmentation.get(augmentation_key)
        
        if not all([model_config, dataset_config, training_config, augmentation_config]):
            missing = []
            if not model_config: missing.append(f"model '{model_key}'")
            if not dataset_config: missing.append(f"dataset '{dataset_key}'")
            if not training_config: missing.append(f"training '{training_key}'")
            if not augmentation_config: missing.append(f"augmentation '{augmentation_key}'")
            raise ValueError(f"Missing predefined configurations: {', '.join(missing)}")
        
        # Apply custom overrides if provided
        if custom_overrides:
            model_config = self._apply_overrides(model_config, custom_overrides.get('model', {}))
            dataset_config = self._apply_overrides(dataset_config, custom_overrides.get('dataset', {}))
            training_config = self._apply_overrides(training_config, custom_overrides.get('training', {}))
            augmentation_config = self._apply_overrides(augmentation_config, custom_overrides.get('augmentation', {}))
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(
            experiment_name=experiment_name,
            model=model_config,
            dataset=dataset_config,
            training=training_config,
            augmentation=augmentation_config,
            timestamp=datetime.now().isoformat(),
            description=description,
            tags=tags or []
        )
        
        logger.info(f"Created experiment configuration: {experiment_name}")
        return experiment_config
    
    def _apply_overrides(self, config_obj, overrides: Dict[str, Any]):
        """Apply overrides to a configuration object"""
        if not overrides:
            return config_obj
        
        # Convert to dict, apply overrides, convert back
        config_dict = asdict(config_obj)
        config_dict.update(overrides)
        
        # Recreate object with updated values
        return type(config_obj)(**config_dict)
    
    def save_config(self, config: ExperimentConfig, filename: str = None) -> str:
        """
        Save experiment configuration to file
        
        Args:
            config: ExperimentConfig to save
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved configuration file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{config.experiment_name}_{timestamp}.yaml"
        
        config_path = self.config_dir / filename
        
        # Convert to dictionary for serialization
        config_dict = asdict(config)
        
        # Save as YAML for human readability
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
        return str(config_path)
    
    def load_config(self, config_path: str) -> ExperimentConfig:
        """
        Load experiment configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ExperimentConfig object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML file
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested objects
        model_config = ModelConfig(**config_dict['model'])
        dataset_config = DatasetConfig(**config_dict['dataset'])
        training_config = TrainingConfig(**config_dict['training'])
        augmentation_config = AugmentationConfig(**config_dict['augmentation'])
        
        # Create experiment configuration
        experiment_config = ExperimentConfig(
            experiment_name=config_dict['experiment_name'],
            model=model_config,
            dataset=dataset_config,
            training=training_config,
            augmentation=augmentation_config,
            timestamp=config_dict['timestamp'],
            description=config_dict.get('description', ''),
            tags=config_dict.get('tags', [])
        )
        
        logger.info(f"Configuration loaded from: {config_path}")
        return experiment_config
    
    def list_predefined_configs(self) -> Dict[str, List[str]]:
        """
        List all available predefined configurations
        
        Returns:
            Dictionary with lists of available configuration keys
        """
        return {
            'models': list(self.predefined_models.keys()),
            'datasets': list(self.predefined_datasets.keys()),
            'training': list(self.predefined_training.keys()),
            'augmentation': list(self.predefined_augmentation.keys())
        }
    
    def get_config_summary(self, config: ExperimentConfig) -> str:
        """
        Get a human-readable summary of the configuration
        
        Args:
            config: ExperimentConfig to summarize
            
        Returns:
            Formatted summary string
        """
        summary = f"""
Experiment Configuration Summary
================================

Experiment Name: {config.experiment_name}
Description: {config.description}
Tags: {', '.join(config.tags) if config.tags else 'None'}
Timestamp: {config.timestamp}

Model Configuration:
  Architecture: {config.model.architecture}
  Input Shape: {config.model.input_shape}
  Number of Classes: {config.model.num_classes}
  Dropout Rate: {config.model.dropout_rate}
  Hidden Units: {config.model.hidden_units}
  Freeze Base: {config.model.freeze_base}

Dataset Configuration:
  Name: {config.dataset.name}
  Type: {config.dataset.dataset_type}
  Number of Classes: {config.dataset.num_classes}
  Validation Split: {config.dataset.validation_split}
  Label Mode: {config.dataset.label_mode}

Training Configuration:
  Batch Size: {config.training.batch_size}
  Epochs: {config.training.epochs}
  Learning Rate: {config.training.learning_rate}
  Optimizer: {config.training.optimizer}
  Patience: {config.training.patience}
  Reduce LR Patience: {config.training.reduce_lr_patience}

Augmentation Configuration:
  Enabled: {config.augmentation.enabled}
  Rotation Range: {config.augmentation.rotation_range}
  Width Shift Range: {config.augmentation.width_shift_range}
  Height Shift Range: {config.augmentation.height_shift_range}
  Horizontal Flip: {config.augmentation.horizontal_flip}
  Zoom Range: {config.augmentation.zoom_range}
"""
        return summary
    
    def create_quick_configs(self) -> Dict[str, ExperimentConfig]:
        """
        Create a set of quick-start experiment configurations
        
        Returns:
            Dictionary of ready-to-use experiment configurations
        """
        quick_configs = {}
        
        # Standard configurations for all model-dataset combinations
        model_dataset_pairs = [
            # Baseline CNN
            ('baseline_cnn_cifar10', 'cifar10', 'baseline_cnn', 'cifar10'),
            ('baseline_cnn_cifar100', 'cifar100_fine', 'baseline_cnn', 'cifar100'),
            ('baseline_cnn_fashion_mnist', 'fashion_mnist', 'baseline_cnn', 'fashion_mnist'),
            ('baseline_cnn_celeba', 'celeba', 'baseline_cnn', 'celeba'),
            
            # ResNet50
            ('resnet50_cifar10', 'cifar10', 'resnet50', 'cifar10'),
            ('resnet50_cifar100', 'cifar100_fine', 'resnet50', 'cifar100'),
            ('resnet50_fashion_mnist', 'fashion_mnist', 'resnet50', 'fashion_mnist'),
            ('resnet50_celeba', 'celeba', 'resnet50', 'celeba'),
            
            # DenseNet121
            ('densenet121_cifar10', 'cifar10', 'densenet121', 'cifar10'),
            ('densenet121_cifar100', 'cifar100_fine', 'densenet121', 'cifar100'),
            ('densenet121_fashion_mnist', 'fashion_mnist', 'densenet121', 'fashion_mnist'),
            ('densenet121_celeba', 'celeba', 'densenet121', 'celeba'),
            
            # InceptionV3
            ('inceptionv3_cifar10', 'cifar10', 'inceptionv3', 'cifar10'),
            ('inceptionv3_cifar100', 'cifar100_fine', 'inceptionv3', 'cifar100'),
            ('inceptionv3_fashion_mnist', 'fashion_mnist', 'inceptionv3', 'fashion_mnist'),
            ('inceptionv3_celeba', 'celeba', 'inceptionv3', 'celeba'),
            
            # MobileNet
            ('mobilenet_cifar10', 'cifar10', 'mobilenet', 'cifar10'),
            ('mobilenet_cifar100', 'cifar100_fine', 'mobilenet', 'cifar100'),
            ('mobilenet_fashion_mnist', 'fashion_mnist', 'mobilenet', 'fashion_mnist'),
            ('mobilenet_celeba', 'celeba', 'mobilenet', 'celeba')
        ]
        
        for model_key, dataset_key, model_short, dataset_short in model_dataset_pairs:
            # Standard configuration
            config_name = f"{model_short}_{dataset_short}_standard"
            quick_configs[config_name] = self.create_experiment_config(
                experiment_name=f'{model_short.title()}_{dataset_short.upper()}_Standard',
                model_key=model_key,
                dataset_key=dataset_key,
                training_key='standard',
                augmentation_key='standard',
                description=f'Standard {model_short.title()} training on {dataset_short.upper()}',
                tags=[model_short, dataset_short, 'standard']
            )
        
        # Fast configurations for testing
        fast_pairs = [
            ('baseline_cnn_cifar10', 'cifar10', 'baseline_cnn', 'cifar10'),
            ('resnet50_cifar10', 'cifar10', 'resnet50', 'cifar10'),
            ('mobilenet_fashion_mnist', 'fashion_mnist', 'mobilenet', 'fashion_mnist')
        ]
        
        for model_key, dataset_key, model_short, dataset_short in fast_pairs:
            config_name = f"{model_short}_{dataset_short}_fast"
            quick_configs[config_name] = self.create_experiment_config(
                experiment_name=f'{model_short.title()}_{dataset_short.upper()}_Fast',
                model_key=model_key,
                dataset_key=dataset_key,
                training_key='fast',
                augmentation_key='minimal',
                description=f'Fast {model_short.title()} training on {dataset_short.upper()} for quick testing',
                tags=[model_short, dataset_short, 'fast', 'testing']
            )
        
        # Fine-tuning configurations for transfer learning models
        finetune_pairs = [
            ('resnet50_cifar100', 'cifar100_fine', 'resnet50', 'cifar100'),
            ('densenet121_cifar100', 'cifar100_fine', 'densenet121', 'cifar100'),
            ('inceptionv3_celeba', 'celeba', 'inceptionv3', 'celeba')
        ]
        
        for model_key, dataset_key, model_short, dataset_short in finetune_pairs:
            config_name = f"{model_short}_{dataset_short}_finetune"
            quick_configs[config_name] = self.create_experiment_config(
                experiment_name=f'{model_short.title()}_{dataset_short.upper()}_FineTune',
                model_key=model_key,
                dataset_key=dataset_key,
                training_key='fine_tuning',
                augmentation_key='standard',
                description=f'Fine-tuning {model_short.title()} on {dataset_short.upper()}',
                tags=[model_short, dataset_short, 'fine_tuning']
            )
        
        return quick_configs
    
    def analyze_all_configurations(self) -> str:
        """
        Provide comprehensive analysis of all available configurations
        
        Returns:
            Detailed analysis string
        """
        models = self.list_predefined_configs()['models']
        datasets = self.list_predefined_configs()['datasets']
        training_configs = self.list_predefined_configs()['training']
        augmentation_configs = self.list_predefined_configs()['augmentation']
        
        analysis = f"""
Available Configurations:
{'=' * 60}

Model Architectures:
  BaselineCNN: Custom CNN architecture
  ResNet50: ImageNet pre-trained ResNet
  DenseNet121: Dense connectivity ResNet
  InceptionV3: Multi-scale feature extraction
  MobileNet: Efficient mobile-optimized architecture

Dataset Information:
  CIFAR-10: 10 classes, 32x32 RGB images
  CIFAR-100: 100 classes, 32x32 RGB images  
  Fashion-MNIST: 10 classes, 28x28 grayscale
  CelebA: 2 classes (binary), 64x64 RGB images

Training Strategies:
  Standard: Balanced approach (100 epochs, lr=0.001)
  Fast: Quick testing (50 epochs, lr=0.01) 
  Fine-tuning: Transfer learning (50 epochs, lr=0.0001)
  Large Dataset: Extended training (200 epochs)

Augmentation Levels:
  None: No augmentation
  Minimal: Light augmentation (rotation=10°, shift=0.1)
  Standard: Balanced augmentation (rotation=20°, shift=0.2)
  Aggressive: Heavy augmentation (rotation=30°, shift=0.3)

Recommendations:
  Easy Datasets (Fashion-MNIST): Use minimal augmentation
  Medium Datasets (CIFAR-10, CelebA): Use standard augmentation  
  Hard Datasets (CIFAR-100): Use aggressive augmentation
  
  Fast Models (MobileNet): Higher batch sizes, fewer epochs
  Complex Models (InceptionV3): Lower batch sizes, more epochs
  
  Transfer Learning: Use fine-tuning strategy with pre-trained weights
  From Scratch: Use standard strategy with longer training

Total Possible Experiments: {len(models) * len(training_configs) * len(augmentation_configs)} combinations
"""
        
        return analysis
    
    def get_configuration_matrix(self) -> str:
        """
        Generate a matrix showing all model-dataset combinations
        
        Returns:
            Formatted matrix string
        """
        models = ['baseline_cnn', 'resnet50', 'densenet121', 'inceptionv3', 'mobilenet']
        datasets = ['cifar10', 'cifar100', 'fashion_mnist', 'celeba']
        
        matrix = """
Model-Dataset Compatibility Matrix
=================================

"""
        
        # Header
        matrix += f"{'Model':<15}"
        for dataset in datasets:
            matrix += f"{dataset:<15}"
        matrix += "Total\n"
        matrix += "=" * (15 + 15 * len(datasets) + 5) + "\n"
        
        # Rows
        for model in models:
            matrix += f"{model:<15}"
            count = 0
            for dataset_key in datasets:
                config_key = f"{model}_{dataset_key}"
                if any(config_key in key for key in self.predefined_models.keys()):
                    matrix += f"{'Available':<15}"
                    count += 1
                else:
                    matrix += f"{'Missing':<15}"
            matrix += f"{count}\n"
        
        # Footer
        matrix += "=" * (15 + 15 * len(datasets) + 5) + "\n"
        matrix += f"{'Total':<15}"
        for dataset_key in datasets:
            dataset_count = sum(1 for key in self.predefined_models.keys() if dataset_key in key)
            matrix += f"{dataset_count:<15}"
        matrix += f"{len(self.predefined_models)}\n"
        
        return matrix
    
    def get_recommended_configs(self, use_case: str = "research") -> Dict[str, ExperimentConfig]:
        """
        Get recommended configurations based on use case
        
        Args:
            use_case: Type of use case ('research', 'testing', 'production', 'comparison')
            
        Returns:
            Dictionary of recommended configurations
        """
        recommendations = {}
        
        if use_case == "research":
            # Best performing models on each dataset
            research_configs = [
                ('densenet121_cifar10', 'cifar10', 'standard', 'standard'),
                ('densenet121_cifar100', 'cifar100_fine', 'standard', 'aggressive'),
                ('resnet50_fashion_mnist', 'fashion_mnist', 'standard', 'minimal'),
                ('inceptionv3_celeba', 'celeba', 'fine_tuning', 'standard')
            ]
            
            for model_key, dataset_key, training_key, aug_key in research_configs:
                name = f"research_{model_key}"
                recommendations[name] = self.create_experiment_config(
                    experiment_name=f"Research_{model_key.upper()}",
                    model_key=model_key,
                    dataset_key=dataset_key,
                    training_key=training_key,
                    augmentation_key=aug_key,
                    description=f"Research-grade configuration for {model_key}",
                    tags=['research', 'high_performance']
                )
        
        elif use_case == "testing":
            # Fast configurations for system testing
            test_configs = [
                ('baseline_cnn_cifar10', 'cifar10', 'fast', 'minimal'),
                ('mobilenet_fashion_mnist', 'fashion_mnist', 'fast', 'none'),
                ('resnet50_celeba', 'celeba', 'fast', 'minimal')
            ]
            
            for model_key, dataset_key, training_key, aug_key in test_configs:
                name = f"test_{model_key}"
                recommendations[name] = self.create_experiment_config(
                    experiment_name=f"Test_{model_key.upper()}",
                    model_key=model_key,
                    dataset_key=dataset_key,
                    training_key=training_key,
                    augmentation_key=aug_key,
                    description=f"Fast testing configuration for {model_key}",
                    tags=['testing', 'fast', 'validation']
                )
        
        elif use_case == "comparison":
            # All models on same dataset for comparison
            comparison_configs = [
                ('baseline_cnn_cifar10', 'cifar10', 'standard', 'standard'),
                ('resnet50_cifar10', 'cifar10', 'standard', 'standard'),
                ('densenet121_cifar10', 'cifar10', 'standard', 'standard'),
                ('inceptionv3_cifar10', 'cifar10', 'standard', 'standard'),
                ('mobilenet_cifar10', 'cifar10', 'standard', 'standard')
            ]
            
            for model_key, dataset_key, training_key, aug_key in comparison_configs:
                name = f"compare_{model_key}"
                recommendations[name] = self.create_experiment_config(
                    experiment_name=f"Compare_{model_key.upper()}",
                    model_key=model_key,
                    dataset_key=dataset_key,
                    training_key=training_key,
                    augmentation_key=aug_key,
                    description=f"Comparison configuration for {model_key} on CIFAR-10",
                    tags=['comparison', 'cifar10', 'standardized']
                )
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    print("Testing Configuration Manager...")
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # List available configurations
    available_configs = config_manager.list_predefined_configs()
    print(f"Available configurations: {available_configs}")
    
    # Create a custom experiment configuration
    experiment_config = config_manager.create_experiment_config(
        experiment_name='Test_ResNet50_CIFAR10',
        model_key='resnet50_cifar10',
        dataset_key='cifar10',
        training_key='standard',
        augmentation_key='standard',
        description='Test configuration for ResNet50 on CIFAR-10',
        tags=['test', 'resnet50', 'cifar10'],
        custom_overrides={
            'training': {'batch_size': 64, 'epochs': 50},
            'model': {'dropout_rate': 0.3}
        }
    )
    
    # Print configuration summary
    print(config_manager.get_config_summary(experiment_config))
    
    # Save configuration
    config_path = config_manager.save_config(experiment_config)
    print(f"Configuration saved to: {config_path}")
    
    # Load configuration
    loaded_config = config_manager.load_config(config_path)
    print(f"Configuration loaded successfully: {loaded_config.experiment_name}")
    
    # Create quick configurations
    quick_configs = config_manager.create_quick_configs()
    print(f"Created {len(quick_configs)} quick-start configurations")
    
    # Analyze all configurations
    print("\n--- Comprehensive Analysis ---")
    print(config_manager.analyze_all_configurations())

    # Get configuration matrix
    print("\n--- Configuration Matrix ---")
    print(config_manager.get_configuration_matrix())

    # Get recommended configurations
    print("\n--- Recommended Configurations ---")
    recommended_configs = config_manager.get_recommended_configs()
    for name, config in recommended_configs.items():
        print(f"Recommended Config: {name}")
        print(config_manager.get_config_summary(config))
    
    print("\nConfiguration Manager testing completed successfully!") 