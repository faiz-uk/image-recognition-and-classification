"""
Device utilities for GPU setup, memory management, and batch size optimization
"""

import logging
import tensorflow as tf
from typing import Optional, Dict, Any, Union
import psutil
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_gpu(
    memory_growth: bool = True, mixed_precision: bool = True
) -> Dict[str, Any]:
    """Setup GPU configuration for optimal training performance"""
    gpu_info = {
        "gpu_available": False,
        "gpu_count": 0,
        "memory_growth_enabled": False,
        "mixed_precision_enabled": False,
        "gpu_devices": [],
    }

    gpus = tf.config.list_physical_devices("GPU")
    gpu_info["gpu_available"] = len(gpus) > 0
    gpu_info["gpu_count"] = len(gpus)

    if gpus:
        try:
            if memory_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                gpu_info["memory_growth_enabled"] = True

            for i, gpu in enumerate(gpus):
                gpu_details = tf.config.experimental.get_device_details(gpu)
                gpu_info["gpu_devices"].append(
                    {
                        "id": i,
                        "name": gpu.name,
                        "device_type": gpu.device_type,
                        "details": gpu_details,
                    }
                )

            if mixed_precision:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                gpu_info["mixed_precision_enabled"] = True

        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
            gpu_info["error"] = str(e)

    return gpu_info


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory information"""
    memory_info = {
        "gpu_available": False,
        "total_memory": 0,
        "free_memory": 0,
        "used_memory": 0,
        "memory_utilization": 0.0,
    }

    try:
        if tf.config.list_physical_devices("GPU"):
            memory_info["gpu_available"] = True

            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                memory_info["total_memory"] = info.total
                memory_info["free_memory"] = info.free
                memory_info["used_memory"] = info.used
                memory_info["memory_utilization"] = (info.used / info.total) * 100

            except ImportError:
                memory_info["total_memory"] = "Unknown"
                memory_info["free_memory"] = "Unknown"
                memory_info["used_memory"] = "Unknown"
                memory_info["memory_utilization"] = "Unknown"

    except Exception as e:
        logger.error(f"Error getting GPU memory info: {e}")
        memory_info["error"] = str(e)

    return memory_info


def get_optimal_batch_size(
    model_input_shape: tuple,
    available_memory_gb: Optional[float] = None,
    model_complexity: str = "medium",
    model_name: Optional[str] = None,
) -> int:
    """
    Calculate optimal batch size based on available memory and model complexity

    Model Complexity Settings:
    - low: BaselineCNN, MobileNet (efficient architectures)
    - medium: ResNet50, DenseNet121 (standard models)  
    - high: InceptionV3 (complex architecture)
    - very_high: Future very large models
    """
    model_complexity_map = {
        "baseline_cnn": "low",
        "baselinecnn": "low",
        "mobilenet": "low",
        "resnet50": "medium",
        "densenet121": "medium",
        "inceptionv3": "high",
        "inceptionv3model": "high",
    }

    if model_name and model_name.lower() in model_complexity_map:
        model_complexity = model_complexity_map[model_name.lower()]

    complexity_multipliers = {
        "low": 1.8,
        "medium": 1.0,
        "high": 0.6,
        "very_high": 0.4,
    }

    model_batch_adjustments = {
        "baseline_cnn": 1.2,
        "mobilenet": 1.5,
        "resnet50": 1.0,
        "densenet121": 0.9,
        "inceptionv3": 0.7,
    }

    if available_memory_gb is None:
        memory_info = get_gpu_memory_info()
        if memory_info["gpu_available"] and memory_info["total_memory"] != "Unknown":
            available_memory_gb = memory_info["free_memory"] / (1024**3)
        else:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            available_memory_gb = min(available_memory_gb, 8.0)

    input_size = np.prod(model_input_shape)
    memory_per_sample_mb = (input_size * 4) / (1024**2)

    memory_multiplier = 4  # Conservative estimate for gradients, activations
    total_memory_per_sample_mb = memory_per_sample_mb * memory_multiplier

    available_memory_mb = available_memory_gb * 1024 * 0.8  # Use 80% of available
    base_batch_size = int(available_memory_mb / total_memory_per_sample_mb)

    complexity_multiplier = complexity_multipliers.get(model_complexity, 1.0)
    optimal_batch_size = int(base_batch_size * complexity_multiplier)

    if model_name and model_name.lower() in model_batch_adjustments:
        model_adjustment = model_batch_adjustments[model_name.lower()]
        optimal_batch_size = int(optimal_batch_size * model_adjustment)

    optimal_batch_size = max(1, min(optimal_batch_size, 512))
    optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))  # Round to power of 2

    return optimal_batch_size


def configure_tensorflow_for_performance() -> Dict[str, Any]:
    """Configure TensorFlow for optimal performance"""
    config_info = {
        "inter_op_parallelism_threads": 0,
        "intra_op_parallelism_threads": 0,
        "allow_soft_placement": True,
        "log_device_placement": False,
    }

    try:
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all cores
        tf.config.set_soft_device_placement(True)
        tf.debugging.set_log_device_placement(False)

    except Exception as e:
        logger.error(f"Error configuring TensorFlow: {e}")
        config_info["error"] = str(e)

    return config_info


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    system_info = {
        "tensorflow_version": tf.__version__,
        "python_version": tf.version.VERSION,
        "gpu_available": tf.config.list_physical_devices("GPU"),
        "cpu_count": psutil.cpu_count(),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "available_memory_gb": psutil.virtual_memory().available / (1024**3),
        "memory_usage_percent": psutil.virtual_memory().percent,
    }

    if system_info["gpu_available"]:
        system_info["gpu_count"] = len(tf.config.list_physical_devices("GPU"))
        system_info["gpu_memory_info"] = get_gpu_memory_info()

    return system_info


# Example usage and testing
if __name__ == "__main__":
    print("Testing device utilities...")

    # Test GPU setup
    gpu_info = setup_gpu()
    print(f"GPU setup completed: {gpu_info}")

    # Test system info
    system_info = get_system_info()
    print(f"\nSystem Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Test optimal batch size calculation
    batch_size = get_optimal_batch_size((32, 32, 3), model_complexity="medium")
    print(f"\nOptimal batch size for CIFAR input: {batch_size}")

    # Test model-specific batch size calculation
    for model_name in [
        "baseline_cnn",
        "mobilenet",
        "resnet50",
        "densenet121",
        "inceptionv3",
    ]:
        batch_size = get_optimal_batch_size((32, 32, 3), model_name=model_name)
        print(f"Optimal batch size for {model_name}: {batch_size}")

    # Test TensorFlow configuration
    tf_config = configure_tensorflow_for_performance()
    print(f"\nTensorFlow configuration: {tf_config}")

    print("\nDevice utilities test completed successfully!")
