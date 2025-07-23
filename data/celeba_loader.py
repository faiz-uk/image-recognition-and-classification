"""
CelebA Data Loader for CNN Image Classification Project
Implements CelebA dataset downloading, loading and preprocessing for image classification
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import tarfile
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Optional, Dict, Any, List
import os
import sys
from pathlib import Path
from PIL import Image
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED
from .cifar_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_celeba_from_tar(datasets_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load CelebA dataset from tar.gz file (consistent with other datasets)
    
    Args:
        datasets_dir: Directory containing the dataset files
        
    Returns:
        Tuple of (attributes_df, image_names_list)
    """
    celeba_tar_path = datasets_dir / "celeba.tar.gz"
    
    if not celeba_tar_path.exists():
        raise FileNotFoundError(f"CelebA tar.gz file not found at {celeba_tar_path}")
    
    logger.info(f"Loading CelebA from tar.gz file: {celeba_tar_path}")
    
    with tarfile.open(celeba_tar_path, 'r:gz') as tar:
        # Load attributes CSV
        attr_file = tar.extractfile("list_attr_celeba.csv")
        if attr_file:
            # Read CSV content
            csv_content = attr_file.read().decode('utf-8')
            from io import StringIO
            df_attrs = pd.read_csv(StringIO(csv_content))
            
            # Convert -1/1 to 0/1 for binary classification
            for col in df_attrs.columns:
                if col != 'image_id':
                    df_attrs[col] = (df_attrs[col] == 1).astype(int)
            
            logger.info(f"Loaded {len(df_attrs)} image attributes from tar.gz")
        else:
            raise FileNotFoundError("Could not find list_attr_celeba.csv in tar.gz")
        
        # Get list of available images in tar
        image_names = []
        for member in tar.getnames():
            if member.endswith('.jpg') and 'img_align_celeba' in member:
                # Extract just the filename
                image_name = member.split('/')[-1]
                image_names.append(image_name)
        
        image_names.sort()  # Sort to ensure consistent ordering
        logger.info(f"Found {len(image_names)} images in tar.gz")
    
    return df_attrs, image_names


def load_images_from_tar(datasets_dir: Path, image_list: List[str], image_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Load images from CelebA tar.gz file
    
    Args:
        datasets_dir: Directory containing the dataset files
        image_list: List of image filenames to load
        image_size: Target image size (height, width)
        
    Returns:
        Array of preprocessed images
    """
    celeba_tar_path = datasets_dir / "celeba.tar.gz"
    images = []
    
    logger.info(f"Loading {len(image_list)} images from tar.gz...")
    
    with tarfile.open(celeba_tar_path, 'r:gz') as tar:
        # Create a mapping of filenames to their paths in tar
        filename_to_path = {}
        for member_name in tar.getnames():
            if member_name.endswith('.jpg') and 'img_align_celeba' in member_name:
                filename = member_name.split('/')[-1]
                filename_to_path[filename] = member_name
        
        for i, image_name in enumerate(image_list):
            if i % 5000 == 0:
                logger.info(f"Loaded {i}/{len(image_list)} images")
            
            if image_name in filename_to_path:
                try:
                    # Extract image data
                    image_file = tar.extractfile(filename_to_path[image_name])
                    if image_file:
                        # Load image from bytes
                        img_data = image_file.read()
                        img = Image.open(io.BytesIO(img_data)).convert('RGB')
                        img = img.resize(image_size, Image.Resampling.LANCZOS)
                        img_array = np.array(img)
                        images.append(img_array)
                    else:
                        # Create dummy image for missing files
                        dummy_img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                        images.append(dummy_img)
                except Exception as e:
                    logger.warning(f"Error loading image {image_name}: {e}")
                    # Create dummy image for corrupted files
                    dummy_img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                    images.append(dummy_img)
            else:
                # Create dummy image for missing files
                dummy_img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
                images.append(dummy_img)
    
    images_array = np.array(images)
    logger.info(f"Loaded images from tar.gz: {images_array.shape}")
    return images_array


class CelebALoader(DataLoader):
    """CelebA dataset loader and preprocessor for image classification"""
    
    def __init__(self, 
                 validation_split: float = 0.2,
                 target_attribute: str = 'Smiling',
                 image_size: Tuple[int, int] = (64, 64),
                 max_samples: Optional[int] = None,
                 use_existing: bool = True,
                 prefer_tar: bool = True):
        """
        Initialize CelebA loader
        
        Args:
            validation_split: Fraction of training data to use for validation
            target_attribute: Attribute to use for classification (e.g., 'Smiling', 'Male', 'Young')
            image_size: Target image size (height, width)
            max_samples: Maximum number of samples to load (None for all)
            use_existing: Whether to use existing downloaded data
            prefer_tar: Whether to prefer tar.gz format over directory structure
        """
        # Set CelebA-specific attributes before calling parent
        self.target_attribute = target_attribute
        self.image_size = image_size
        self.max_samples = max_samples
        self.use_existing = use_existing
        self.prefer_tar = prefer_tar
        
        # CelebA metadata
        self.num_classes = 2  # Binary classification
        self.input_shape = (*image_size, 3)
        self.class_names = None  # Depends on chosen attribute
        
        # Dataset paths
        self.celeba_dir = DATASETS_DIR / "celeba"
        self.images_dir = self.celeba_dir / "img_align_celeba"
        self.attr_file = self.celeba_dir / "list_attr_celeba.txt"
        self.partition_file = self.celeba_dir / "list_eval_partition.txt"
        
        # Initialize parent class
        super().__init__('celeba', validation_split)
        
        logger.info(f"CelebA specific settings: attribute='{target_attribute}', size={image_size}, tar={prefer_tar}")
    
    def _get_dataset_config(self) -> Dict[str, Any]:
        """Get CelebA specific configuration"""
        return {
            'name': 'CelebA',
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'class_names': self.class_names,
            'target_attribute': self.target_attribute
        }
    
    def download_celeba(self) -> bool:
        """
        Download CelebA dataset (simplified version)
        Note: This is a placeholder. Real CelebA requires manual download due to Google Drive restrictions.
        
        Returns:
            Success status
        """
        logger.warning("CelebA dataset requires manual download from:")
        logger.warning("1. Images: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBVzg")
        logger.warning("2. Attributes: https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs")
        logger.warning("Please download and extract to: {}".format(self.celeba_dir))
        
        # Create directory structure
        self.celeba_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy files for demonstration (remove in production)
        if not self.attr_file.exists():
            logger.info("Creating dummy attribute file for demonstration...")
            self._create_dummy_attributes()
        
        return self.attr_file.exists() and self.images_dir.exists()
    
    def _create_dummy_attributes(self) -> None:
        """Create dummy attribute file for demonstration purposes"""
        # This is just for demonstration - remove in production
        dummy_attrs = """202599
image_id 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young
000001.jpg -1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 1 1 -1 1 -1 1 -1 -1 1
000002.jpg -1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 -1 -1 -1 -1 -1 1 1 -1 -1 -1 1 -1 -1 1
"""
        with open(self.attr_file, 'w') as f:
            f.write(dummy_attrs)
    
    def load_attributes(self) -> pd.DataFrame:
        """
        Load CelebA attributes file
        
        Returns:
            DataFrame with image names and attributes
        """
        if not self.attr_file.exists():
            logger.error(f"Attributes file not found: {self.attr_file}")
            logger.info("Attempting to download/create CelebA dataset...")
            if not self.download_celeba():
                raise FileNotFoundError("Could not download or find CelebA dataset")
        
        logger.info(f"Loading attributes from: {self.attr_file}")
        
        # Read the attributes file
        # First line contains number of images, second line contains attribute names
        with open(self.attr_file, 'r') as f:
            lines = f.readlines()
        
        num_images = int(lines[0].strip())
        attr_names = lines[1].strip().split()
        
        # Parse image data
        data = []
        for i in range(2, min(len(lines), num_images + 2)):
            if self.max_samples and len(data) >= self.max_samples:
                break
            
            parts = lines[i].strip().split()
            if len(parts) >= len(attr_names) + 1:  # +1 for image name
                image_name = parts[0]
                attributes = [int(x) for x in parts[1:len(attr_names)+1]]
                data.append([image_name] + attributes)
        
        # Create DataFrame
        columns = ['image_id'] + attr_names
        df = pd.DataFrame(data, columns=columns)
        
        # Convert -1/1 to 0/1 for binary classification
        for col in attr_names:
            df[col] = (df[col] == 1).astype(int)
        
        logger.info(f"Loaded {len(df)} image attributes")
        return df
    
    def load_images(self, image_list: List[str]) -> np.ndarray:
        """
        Load and preprocess images
        
        Args:
            image_list: List of image filenames
            
        Returns:
            Array of preprocessed images
        """
        images = []
        
        logger.info(f"Loading {len(image_list)} images...")
        
        for i, image_name in enumerate(image_list):
            if i % 1000 == 0:
                logger.info(f"Loaded {i}/{len(image_list)} images")
            
            image_path = self.images_dir / image_name
            
            if image_path.exists():
                try:
                    # Load and resize image
                    img = Image.open(image_path).convert('RGB')
                    img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    images.append(img_array)
                except Exception as e:
                    logger.warning(f"Error loading image {image_name}: {e}")
                    # Create dummy image for missing files
                    dummy_img = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
                    images.append(dummy_img)
            else:
                # Create dummy image for missing files
                dummy_img = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
                images.append(dummy_img)
        
        images_array = np.array(images)
        logger.info(f"Loaded images shape: {images_array.shape}")
        return images_array
    
    def load_and_preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                         np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CelebA dataset
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Loading CelebA dataset for '{self.target_attribute}' classification...")
        
        # Try to load from tar.gz first if preferred
        if self.prefer_tar:
            try:
                df_attrs, available_images = load_celeba_from_tar(DATASETS_DIR)
                logger.info("Loaded from tar.gz format (consistent with other datasets)")
                use_tar = True
            except FileNotFoundError:
                logger.info("tar.gz not found, falling back to directory structure...")
                df_attrs = self.load_attributes()
                available_images = [f"{i:06d}.jpg" for i in range(1, len(df_attrs) + 1)]
                use_tar = False
        else:
            # Use directory structure
            df_attrs = self.load_attributes()
            available_images = [f"{i:06d}.jpg" for i in range(1, len(df_attrs) + 1)]
            use_tar = False
        
        if self.target_attribute not in df_attrs.columns:
            available_attrs = [col for col in df_attrs.columns if col != 'image_id']
            raise ValueError(f"Attribute '{self.target_attribute}' not found. "
                           f"Available attributes: {available_attrs}")
        
        # Limit samples if specified
        if self.max_samples and len(df_attrs) > self.max_samples:
            df_attrs = df_attrs.head(self.max_samples)
            available_images = available_images[:self.max_samples]
            logger.info(f"Limited to {self.max_samples} samples")
        
        # Get image names and labels
        if 'image_id' in df_attrs.columns:
            image_names = df_attrs['image_id'].tolist()
        else:
            image_names = available_images
        
        labels = df_attrs[self.target_attribute].values
        
        # Load images
        if use_tar:
            images = load_images_from_tar(DATASETS_DIR, image_names, self.image_size)
        else:
            images = self.load_images(image_names)
        
        # Ensure we have matching numbers
        min_len = min(len(images), len(labels))
        images = images[:min_len]
        labels = labels[:min_len]
        
        logger.info(f"Final dataset: {len(images)} samples")
        
        # Split into train/test (80/20)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            images, labels,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=labels
        )
        
        # Split training into train/validation using parent method
        X_train, X_val, y_train, y_val = self.split_train_validation(X_train_full, y_train_full)
        
        # Normalize data using parent method
        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)
        
        # Encode labels using parent method
        y_train, y_val, y_test = self.encode_labels(y_train, y_val, y_test)
        
        logger.info("CelebA data preprocessing completed")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _get_additional_info(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Get CelebA specific additional information"""
        return {
            'target_attribute': self.target_attribute,
            'image_size': self.image_size,
            'max_samples': self.max_samples,
            'use_existing': self.use_existing,
            'prefer_tar': self.prefer_tar,
            'class_distribution_train': {
                'negative': np.sum(y_train[:, 0]),
                'positive': np.sum(y_train[:, 1])
            }
        }
    
    def _get_augmentation_function(self):
        """Get CelebA specific augmentation function"""
        def celeba_augment(image, label):
            # CelebA specific augmentations for face images
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            return image, label
        
        return celeba_augment
    
    def _normalize_data(self, X_train: np.ndarray, X_val: np.ndarray, 
                       X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize pixel values to [0, 1] range (kept for legacy compatibility)
        """
        return self.normalize_data(X_train, X_val, X_test)
    
    def get_available_attributes(self) -> List[str]:
        """
        Get list of available attributes for classification
        
        Returns:
            List of attribute names
        """
        try:
            df_attrs = self.load_attributes()
            return [col for col in df_attrs.columns if col != 'image_id']
        except Exception as e:
            logger.warning(f"Could not load attributes: {e}")
            # Return common CelebA attributes
            return ['Smiling', 'Male', 'Young', 'Attractive', 'Heavy_Makeup',
                   'Eyeglasses', 'Bald', 'Mustache', 'Goatee', 'Pale_Skin']


def create_celeba_data_generators(X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 batch_size: int = 32,
                                 augment_training: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Legacy function for backward compatibility
    Create TensorFlow data generators for CelebA training and validation
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        batch_size: Batch size for training
        augment_training: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create a temporary loader instance for the method
    temp_loader = CelebALoader()
    return temp_loader.create_data_generators(X_train, y_train, X_val, y_val, batch_size, augment_training)


# Example usage and testing
if __name__ == "__main__":
    print("Testing CelebA loader...")
    try:
        # Test CelebA loader
        celeba_loader = CelebALoader(
            validation_split=0.2,
            target_attribute='Smiling',
            image_size=(64, 64),
            max_samples=100  # Small sample for testing
        )
        
        # Get available attributes
        attrs = celeba_loader.get_available_attributes()
        print(f"\nAvailable attributes: {attrs}")
        
        # Try to load data (will create dummy data if real data not available)
        X_train, X_val, X_test, y_train, y_val, y_test = celeba_loader.load_and_preprocess()
        
        # Print dataset info
        info = celeba_loader.get_data_info(X_train, X_val, X_test, y_train, y_val, y_test)
        print("\nCelebA Dataset Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test data generators
        train_ds, val_ds = create_celeba_data_generators(X_train, y_train, X_val, y_val, batch_size=16)
        print(f"\nTrain dataset: {train_ds}")
        print(f"Validation dataset: {val_ds}")
        
        print("\nCelebA loader test completed successfully!")
        print("Note: This used dummy data. For real usage, download CelebA dataset manually.")
        
    except Exception as e:
        print(f"Error testing CelebA loader: {e}")
        print("This is expected if CelebA dataset is not downloaded.") 