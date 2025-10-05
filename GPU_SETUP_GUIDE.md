# 🚀 GPU Setup Guide for AutoML

## Overview

This guide helps you set up GPU acceleration for the AutoML pipeline to significantly speed up model training.

## GPU Requirements

### Minimum Requirements
- **NVIDIA GPU** with CUDA support
- **CUDA 11.x or 12.x** installed
- **8GB+ GPU memory** (recommended)
- **16GB+ system RAM** (recommended)

### Supported GPU Libraries
- **XGBoost GPU** - For gradient boosting
- **LightGBM GPU** - For fast gradient boosting  
- **CuPy** - For GPU-accelerated NumPy operations
- **cuML** - For GPU-accelerated scikit-learn algorithms

## Installation Steps

### 1. Install CUDA Toolkit

#### For CUDA 11.x
```bash
# Download and install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### For CUDA 12.x
```bash
# Download and install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install GPU-Accelerated Libraries

#### Option A: Using pip (Recommended)
```bash
# Install CuPy for CUDA 11.x
pip install cupy-cuda11x

# Install cuML for CUDA 11.x
pip install cuml-cu11

# Or for CUDA 12.x
pip install cupy-cuda12x
pip install cuml-cu12
```

#### Option B: Using conda (Alternative)
```bash
# Create conda environment with GPU support
conda create -n automl-gpu python=3.10
conda activate automl-gpu

# Install CUDA toolkit
conda install -c nvidia cuda-toolkit=11.8

# Install GPU libraries
conda install -c rapidsai -c conda-forge -c nvidia cuml=22.12 python=3.10 cudatoolkit=11.8
conda install -c conda-forge cupy
```

### 4. Install XGBoost with GPU Support

```bash
# XGBoost with GPU support
pip install xgboost[gpu]

# Or build from source with GPU support
git clone --recursive https://github.com/dmlc/xgboost.git
cd xgboost
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
make -j4
pip install -e .
```

### 5. Install LightGBM with GPU Support

```bash
# LightGBM with GPU support
pip install lightgbm --config-settings=cmake.define.USE_GPU=ON

# Or using conda
conda install -c conda-forge lightgbm-gpu
```

## Verification

### Test GPU Availability
```python
import torch
import cupy as cp
import xgboost as xgb
import lightgbm as lgb

print("🔍 GPU Verification:")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Devices: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

print(f"\nCuPy Available: {cp.cuda.is_available()}")
print(f"XGBoost GPU: {'gpu_hist' in xgb.__version__}")
print(f"LightGBM GPU: hasattr(lgb.LGBMClassifier, 'device')")
```

### Test AutoML with GPU
```python
from train_automl import AutoMLPipeline, CONFIG

# Enable GPU acceleration
CONFIG['use_gpu'] = True
CONFIG['parallel_trials'] = True
CONFIG['max_parallel_trials'] = 4

# Run AutoML
automl = AutoMLPipeline(CONFIG)
results, best_model, best_params = automl.run_automl()
```

## Performance Optimization

### GPU Memory Management
```python
# Limit GPU memory usage
CONFIG.update({
    'gpu_memory_fraction': 0.8,  # Use 80% of GPU memory
    'use_mixed_precision': True,  # Use FP16 for faster training
    'max_parallel_trials': 2,     # Reduce parallel trials for GPU
})
```

### CPU-GPU Hybrid Setup
```python
# Use GPU for tree-based models, CPU for others
CONFIG.update({
    'use_gpu': True,
    'n_jobs': 4,  # Use 4 CPU cores for non-GPU models
    'parallel_trials': True,
    'max_parallel_trials': 2,  # Limit to avoid GPU memory issues
})
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce batch size or model complexity
CONFIG.update({
    'max_parallel_trials': 1,  # Run one trial at a time
    'gpu_memory_fraction': 0.5,  # Use less GPU memory
})
```

#### 2. Library Import Errors
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall with correct CUDA version
pip uninstall cupy cuml
pip install cupy-cuda11x cuml-cu11
```

#### 3. XGBoost GPU Not Working
```python
# Check XGBoost GPU support
import xgboost as xgb
print("XGBoost version:", xgb.__version__)

# Test GPU training
import numpy as np
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
model.fit(X, y)
print("✅ XGBoost GPU working!")
```

#### 4. LightGBM GPU Not Working
```python
# Test LightGBM GPU
import lightgbm as lgb
import numpy as np

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

model = lgb.LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0)
model.fit(X, y)
print("✅ LightGBM GPU working!")
```

## Performance Benchmarks

### Expected Speedups
- **XGBoost GPU**: 2-5x faster than CPU
- **LightGBM GPU**: 3-8x faster than CPU
- **Parallel Processing**: 2-4x faster with multiple cores
- **Combined**: 5-20x faster overall

### Memory Requirements
- **CPU Only**: 8GB RAM minimum
- **GPU Training**: 16GB RAM + 8GB GPU memory
- **Large Datasets**: 32GB RAM + 16GB GPU memory

## Configuration Examples

### High-Performance Setup
```python
CONFIG = {
    'use_gpu': True,
    'n_jobs': -1,  # Use all CPU cores
    'parallel_trials': True,
    'max_parallel_trials': 4,
    'gpu_memory_fraction': 0.9,
    'use_mixed_precision': True,
    'n_trials': 100,  # More trials for better results
}
```

### Memory-Constrained Setup
```python
CONFIG = {
    'use_gpu': True,
    'n_jobs': 2,  # Limit CPU cores
    'parallel_trials': False,  # Sequential training
    'gpu_memory_fraction': 0.5,
    'n_trials': 20,  # Fewer trials
}
```

### CPU-Only Fallback
```python
CONFIG = {
    'use_gpu': False,
    'n_jobs': -1,  # Use all CPU cores
    'parallel_trials': True,
    'max_parallel_trials': 8,
    'n_trials': 50,
}
```

## Next Steps

1. **Install GPU dependencies** following the steps above
2. **Run verification tests** to ensure everything works
3. **Configure AutoML** for your hardware
4. **Run AutoML pipeline** with GPU acceleration
5. **Monitor performance** and adjust settings as needed

---

**Happy GPU-accelerated AutoML! 🚀**
