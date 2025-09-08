#!/bin/bash

echo "=== Installing Dependencies for FYP-FinAgent ==="
echo "This script will install all required packages for GPU-accelerated training"

# Update pip first
echo "Updating pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install JAX with CUDA support
echo "Installing JAX with CUDA support..."
pip install jax[cuda12] jaxlib

# Install core ML libraries
echo "Installing core ML libraries..."
pip install stable-baselines3[extra] gymnasium tensorboard

# Install JAX ecosystem
echo "Installing JAX ecosystem..."
pip install flax optax orbax distrax chex gymnax

# Install data processing libraries
echo "Installing data processing libraries..."
pip install pandas numpy scikit-learn scikit-learn-extra pyarrow fastparquet

# Install financial data libraries
echo "Installing financial data libraries..."
pip install yfinance pandas_market_calendars polygon-api-client finnhub-python ta mplfinance

# Install environment and configuration
echo "Installing environment and configuration..."
pip install python-dotenv json5

# Install web scraping libraries
echo "Installing web scraping libraries..."
pip install praw==7.8.1 selenium playwright twscrape

# Install visualization libraries
echo "Installing visualization libraries..."
pip install pyecharts wandb

# Install NLP libraries
echo "Installing NLP libraries..."
pip install openai langchain langchain_community transformers tiktoken

# Install data processing utilities
echo "Installing data processing utilities..."
pip install unstructured mmengine fuzzywuzzy snapshot_selenium

# Install additional utilities
echo "Installing additional utilities..."
pip install h5py matplotlib seaborn plotly

echo "=== Installation Complete ==="
echo "Testing GPU availability..."

python3 -c "
import torch
import jax
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'PyTorch CUDA device: {torch.cuda.get_device_name()}')
print(f'JAX backend: {jax.default_backend()}')
print(f'JAX devices: {jax.devices()}')
"

echo "All dependencies installed successfully!"

echo "=== Installing Dependencies for FYP-FinAgent ==="
echo "This script will install all required packages for GPU-accelerated training"

# Update pip first
echo "Updating pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install JAX with CUDA support
echo "Installing JAX with CUDA support..."
pip install jax[cuda12] jaxlib

# Install core ML libraries
echo "Installing core ML libraries..."
pip install stable-baselines3[extra] gymnasium tensorboard

# Install JAX ecosystem
echo "Installing JAX ecosystem..."
pip install flax optax orbax distrax chex gymnax

# Install data processing libraries
echo "Installing data processing libraries..."
pip install pandas numpy scikit-learn scikit-learn-extra pyarrow fastparquet

# Install financial data libraries
echo "Installing financial data libraries..."
pip install yfinance pandas_market_calendars polygon-api-client finnhub-python ta mplfinance

# Install environment and configuration
echo "Installing environment and configuration..."
pip install python-dotenv json5

# Install web scraping libraries
echo "Installing web scraping libraries..."
pip install praw==7.8.1 selenium playwright twscrape

# Install visualization libraries
echo "Installing visualization libraries..."
pip install pyecharts wandb

# Install NLP libraries
echo "Installing NLP libraries..."
pip install openai langchain langchain_community transformers tiktoken

# Install data processing utilities
echo "Installing data processing utilities..."
pip install unstructured mmengine fuzzywuzzy snapshot_selenium

# Install additional utilities
echo "Installing additional utilities..."
pip install h5py matplotlib seaborn plotly

echo "=== Installation Complete ==="
echo "Testing GPU availability..."

python3 -c "
import torch
import jax
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'PyTorch CUDA device: {torch.cuda.get_device_name()}')
print(f'JAX backend: {jax.default_backend()}')
print(f'JAX devices: {jax.devices()}')
"

echo "All dependencies installed successfully!"