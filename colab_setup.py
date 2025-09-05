# Google Colab A100 Optimization Setup

# Enable A100 high performance mode
!nvidia-smi -pm 1
!nvidia-smi -ac 1215,1410

# Install JAX with CUDA support
!pip install --upgrade pip
!pip install jax[cuda11_pip]==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
!pip install flax optax chex gymnax
!pip install wandb

# Install additional GPU-accelerated packages
!pip install cupy-cuda11x  # For NumPy replacement
!pip install numba        # For custom kernels if needed

# Set JAX memory preallocation (important for A100)
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'  # Use 95% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'

# Enable JIT compilation caching
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/content/jax_cache'
!mkdir -p /content/jax_cache

# Verify GPU setup
import jax
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"GPU memory: {jax.devices()[0].memory_stats()}")

# Set up mixed precision if needed (A100 supports bfloat16)
jax.config.update('jax_enable_x64', False)  # Use float32
jax.config.update('jax_default_matmul_precision', 'high')  # Use tensor cores
