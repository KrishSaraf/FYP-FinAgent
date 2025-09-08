#!/usr/bin/env python3
"""
Dependency installation script for FYP-FinAgent
Handles GPU-optimized package installation with error handling
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Installing: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing {description}: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def test_gpu_availability():
    """Test GPU availability after installation"""
    print(f"\n{'='*50}")
    print("Testing GPU Availability")
    print(f"{'='*50}")
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        import jax
        print(f"✅ JAX imported successfully")
        print(f"JAX Backend: {jax.default_backend()}")
        print(f"JAX Devices: {jax.devices()}")
    except ImportError:
        print("❌ JAX not available")

def main():
    """Main installation function"""
    print("=== FYP-FinAgent Dependency Installation ===")
    print("This script will install all required packages for GPU-accelerated training")
    
    # Create jax_cache directory
    jax_cache_dir = "./jax_cache"
    os.makedirs(jax_cache_dir, exist_ok=True)
    os.environ['JAX_COMPILATION_CACHE_DIR'] = jax_cache_dir
    print(f"✅ JAX compilation cache directory set to: {jax_cache_dir}")
    
    # Update pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        print("Warning: Failed to upgrade pip, continuing...")
    
    # Installation packages in order of dependency
    packages = [
        # Core ML libraries
        ("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", "PyTorch with CUDA"),
        ("pip install jaxlib==0.4.20+cuda12 --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html", "JAXLib with CUDA"),
        ("pip install jax==0.4.20", "JAX"),
        ("pip install stable-baselines3[extra] gymnasium tensorboard", "Core ML libraries"),
        
        # JAX ecosystem
        ("pip install flax optax orbax distrax chex gymnax", "JAX ecosystem"),
        
        # Data processing
        ("pip install pandas numpy scikit-learn scikit-learn-extra pyarrow fastparquet", "Data processing"),
        
        # Financial data
        ("pip install yfinance pandas_market_calendars polygon-api-client finnhub-python ta mplfinance", "Financial data"),
        
        # Environment and config
        ("pip install python-dotenv json5", "Environment and configuration"),
        
        # Web scraping
        ("pip install praw==7.8.1 selenium playwright twscrape", "Web scraping"),
        
        # Visualization
        ("pip install pyecharts wandb", "Visualization"),
        
        # NLP
        ("pip install openai langchain langchain_community transformers tiktoken", "NLP libraries"),
        
        # Utilities
        ("pip install unstructured mmengine fuzzywuzzy snapshot_selenium", "Data utilities"),
        ("pip install h5py matplotlib seaborn plotly", "Additional utilities"),
    ]
    
    # Install packages
    failed_packages = []
    for command, description in packages:
        if not run_command(command, description):
            failed_packages.append(description)
    
    # Test GPU availability
    test_gpu_availability()
    
    # Summary
    print(f"\n{'='*50}")
    print("Installation Summary")
    print(f"{'='*50}")
    
    if failed_packages:
        print(f"❌ Failed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually or check for version conflicts")
    else:
        print("✅ All packages installed successfully!")
    
    print("\nYou can now run: python train_ppo.py")

if __name__ == "__main__":
    main()
"""
Dependency installation script for FYP-FinAgent
Handles GPU-optimized package installation with error handling
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Installing: {description}")
    print(f"Command: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing {description}: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def test_gpu_availability():
    """Test GPU availability after installation"""
    print(f"\n{'='*50}")
    print("Testing GPU Availability")
    print(f"{'='*50}")
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        import jax
        print(f"✅ JAX imported successfully")
        print(f"JAX Backend: {jax.default_backend()}")
        print(f"JAX Devices: {jax.devices()}")
    except ImportError:
        print("❌ JAX not available")

def main():
    """Main installation function"""
    print("=== FYP-FinAgent Dependency Installation ===")
    print("This script will install all required packages for GPU-accelerated training")
    
    # Create jax_cache directory
    jax_cache_dir = "./jax_cache"
    os.makedirs(jax_cache_dir, exist_ok=True)
    os.environ['JAX_COMPILATION_CACHE_DIR'] = jax_cache_dir
    print(f"✅ JAX compilation cache directory set to: {jax_cache_dir}")
    
    # Update pip first
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        print("Warning: Failed to upgrade pip, continuing...")
    
    # Installation packages in order of dependency
    packages = [
        # Core ML libraries
        ("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", "PyTorch with CUDA"),
        ("pip install jaxlib==0.4.20+cuda12 --extra-index-url https://storage.googleapis.com/jax-releases/jax_cuda_releases.html", "JAXLib with CUDA"),
        ("pip install jax==0.4.20", "JAX"),
        ("pip install stable-baselines3[extra] gymnasium tensorboard", "Core ML libraries"),
        
        # JAX ecosystem
        ("pip install flax optax orbax distrax chex gymnax", "JAX ecosystem"),
        
        # Data processing
        ("pip install pandas numpy scikit-learn scikit-learn-extra pyarrow fastparquet", "Data processing"),
        
        # Financial data
        ("pip install yfinance pandas_market_calendars polygon-api-client finnhub-python ta mplfinance", "Financial data"),
        
        # Environment and config
        ("pip install python-dotenv json5", "Environment and configuration"),
        
        # Web scraping
        ("pip install praw==7.8.1 selenium playwright twscrape", "Web scraping"),
        
        # Visualization
        ("pip install pyecharts wandb", "Visualization"),
        
        # NLP
        ("pip install openai langchain langchain_community transformers tiktoken", "NLP libraries"),
        
        # Utilities
        ("pip install unstructured mmengine fuzzywuzzy snapshot_selenium", "Data utilities"),
        ("pip install h5py matplotlib seaborn plotly", "Additional utilities"),
    ]
    
    # Install packages
    failed_packages = []
    for command, description in packages:
        if not run_command(command, description):
            failed_packages.append(description)
    
    # Test GPU availability
    test_gpu_availability()
    
    # Summary
    print(f"\n{'='*50}")
    print("Installation Summary")
    print(f"{'='*50}")
    
    if failed_packages:
        print(f"❌ Failed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually or check for version conflicts")
    else:
        print("✅ All packages installed successfully!")
    
    print("\nYou can now run: python train_ppo.py")

if __name__ == "__main__":
    main()