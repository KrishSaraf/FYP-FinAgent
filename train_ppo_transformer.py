import os
import time
import jax
import jax.numpy as jnp
from jax import random, vmap, lax
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from flax import serialization
import chex
from typing import Tuple, Dict, Any, NamedTuple, List, Optional
import wandb
import pickle
import distrax
from pathlib import Path
from functools import partial
import json
import logging

# Import the JAX environment
from finagent.environment.portfolio_env import JAXVectorizedPortfolioEnv, EnvState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable JAX optimizations
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_compilation_cache_dir', './jax_cache')
jax.config.update('jax_debug_nans', False) # We'll handle NaN detection manually

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Trajectory(NamedTuple):
    """Trajectory data structure for PPO"""
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array

class LSTMState(NamedTuple):
    """Dummy LSTMState for compatibility, not used in Transformer"""
    h: chex.Array
    c: chex.Array

def safe_normalize(x: chex.Array, eps: float = 1e-8) -> chex.Array:
    """Safely normalize array, handling edge cases"""
    std = jnp.std(x)
    mean = jnp.mean(x)

    # If standard deviation is too small, use a minimum std to avoid zero advantages
    std = jnp.maximum(std, eps)
    normalized = (x - mean) / (std + eps)
    return jnp.clip(normalized, -10.0, 10.0)

def check_for_nans(x: chex.Array, name: str = "array") -> bool:
    """Check for NaN values in array"""
    has_nan = jnp.any(jnp.isnan(x))
    return has_nan


# Defining nn.Sequential module manually
class MLPHead(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


# ============================================================================
# NEURAL NETWORK ARCHITECTURE (Transformer-based Actor-Critic)
# ============================================================================

class TransformerEncoderBlock(nn.Module):
    """Single Transformer Encoder Layer"""
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: chex.Array, training: bool):
        # Self-attention
        norm_x = nn.LayerNorm()(x)
        attn_output = nn.SelfAttention(num_heads=self.nhead, qkv_features=self.d_model)(norm_x)
        attn_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn_output)
        x = x + attn_output # Residual connection

        # Feedforward
        norm_x = nn.LayerNorm()(x)
        ff_output = nn.Dense(self.dim_feedforward)(norm_x)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dense(self.d_model)(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_output)
        x = x + ff_output # Residual connection
        return x

# Updated Transformer network to handle flattened observations correctly

class ActorCriticTransformerFlat(nn.Module):
    """Transformer-based Actor-Critic network that handles flattened observations"""
    action_dim: int
    obs_dim: int  # Full observation dimension (flattened)
    window_size: int
    n_stocks: int
    n_features: int
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout_rate: float = 0.1

    def setup(self):
        # Calculate sizes for different parts of the observation based on actual portfolio_env structure
        self.historical_size = self.window_size * self.n_stocks * self.n_features
        self.current_info_size = self.n_stocks * 2  # current_open + current_gap
        self.portfolio_weights_size = self.action_dim  # portfolio weights (n_stocks + 1)
        self.short_positions_size = self.n_stocks  # short position flags
        self.market_state_size = 8  # market state indicators (8 elements)
        
        # Verify observation structure matches environment
        expected_obs_size = (self.historical_size + self.current_info_size + 
                           self.portfolio_weights_size + self.short_positions_size + self.market_state_size)
        assert self.obs_dim == expected_obs_size, (
            f"Observation dimension mismatch: expected {expected_obs_size}, got {self.obs_dim}"
        )

        # Historical data processing (main Transformer path)
        self.historical_proj = nn.Dense(self.d_model, kernel_init=nn.initializers.he_normal())
        
        # Positional encoding for historical sequence
        max_seq_len = self.window_size * self.n_stocks * 2
        self.pos_embedding = self.param('pos_embedding', 
                                       nn.initializers.zeros, 
                                       (1, max_seq_len, self.d_model))

        # Transformer blocks for historical data
        self.transformer_blocks = [
            TransformerEncoderBlock(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=4 * self.d_model,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_layers)
        ]
        self.transformer_norm = nn.LayerNorm()

        # Current market info processing
        self.current_info_proj = nn.Dense(self.d_model // 2, 
                                        kernel_init=nn.initializers.he_normal())
        
        # Portfolio state processing
        self.portfolio_proj = nn.Dense(self.d_model // 2, 
                                     kernel_init=nn.initializers.he_normal())
        
        # Short positions processing
        self.short_positions_proj = nn.Dense(self.d_model // 4, 
                                           kernel_init=nn.initializers.he_normal())
        
        # Market metrics processing
        self.market_state_proj = nn.Dense(self.d_model // 4, 
                                        kernel_init=nn.initializers.he_normal())

        # Final fusion layer
        fusion_input_size = self.d_model + self.d_model // 2 + self.d_model // 2 + self.d_model // 4 + self.d_model // 4
        self.fusion_layer = nn.Dense(self.d_model, kernel_init=nn.initializers.he_normal())

        # Actor (policy) head
        self.actor_head = MLPHead(self.d_model // 2, self.action_dim)

        # Critic (value) head  
        self.critic_head = MLPHead(self.d_model // 2, 1)

    @nn.compact
    def __call__(self, x: chex.Array, training: bool = True):
        """Forward pass through the network with flattened observations"""
        batch_size = x.shape[0]

        # Clean inputs
        x = jnp.where(jnp.isnan(x), 0.0, x)
        x = jnp.where(jnp.isinf(x), jnp.sign(x) * 10.0, x)

        # Split the flattened observation into its components based on actual portfolio_env structure
        historical_data = x[:, :self.historical_size]
        start_idx = self.historical_size
        
        current_info = x[:, start_idx:start_idx + self.current_info_size]  # Shape: (batch, n_stocks*2)
        start_idx += self.current_info_size
        
        portfolio_weights = x[:, start_idx:start_idx + self.portfolio_weights_size]  # Shape: (batch, action_dim)
        start_idx += self.portfolio_weights_size
        
        short_positions = x[:, start_idx:start_idx + self.short_positions_size]  # Shape: (batch, n_stocks)
        start_idx += self.short_positions_size
        
        market_state = x[:, start_idx:start_idx + self.market_state_size]  # Shape: (batch, 8)

        # Check if any component is empty before processing
        if current_info.size == 0:
            current_features = jnp.zeros((batch_size, self.d_model // 2))
        else:
            current_info = current_info.reshape(batch_size, -1)
            current_features = self.current_info_proj(current_info)
        
        if portfolio_weights.size == 0:
            portfolio_features = jnp.zeros((batch_size, self.d_model // 2))
        else:
            portfolio_weights = portfolio_weights.reshape(batch_size, -1)
            portfolio_features = self.portfolio_proj(portfolio_weights)
        
        if short_positions.size == 0:
            short_features = jnp.zeros((batch_size, self.d_model // 4))
        else:
            short_positions = short_positions.reshape(batch_size, -1)
            short_features = self.short_positions_proj(short_positions)
        
        if market_state.size == 0:
            market_features = jnp.zeros((batch_size, self.d_model // 4))
        else:
            market_state = market_state.reshape(batch_size, -1)
            market_features = self.market_state_proj(market_state)

        # Process historical data with Transformer
        # Reshape: [batch, window_size * n_stocks, n_features]
        historical_reshaped = historical_data.reshape(batch_size, 
                                                    -1, 
                                                    self.n_features)
        
        # Normalize historical data
        hist_mean = jnp.mean(historical_reshaped, axis=(-1, -2), keepdims=True)
        hist_std = jnp.std(historical_reshaped, axis=(-1, -2), keepdims=True)
        hist_std = jnp.maximum(hist_std, 1e-8)
        historical_normalized = (historical_reshaped - hist_mean) / hist_std
        historical_normalized = jnp.clip(historical_normalized, -5.0, 5.0)

        # Project historical data
        historical_encoded = self.historical_proj(historical_normalized)
        
        # Add positional encoding
        seq_len = historical_encoded.shape[1]
        historical_encoded = historical_encoded + self.pos_embedding[:, :seq_len, :]

        # Pass through Transformer blocks
        for block in self.transformer_blocks:
            historical_encoded = block(historical_encoded, training=training)
        
        historical_encoded = self.transformer_norm(historical_encoded)
        
        # Mean pooling across sequence dimension
        historical_features = historical_encoded.mean(axis=1)  # [batch, d_model]

        # Process current market info
        current_info_normalized = jnp.clip(current_info, -10.0, 10.0)
        current_features = self.current_info_proj(current_info_normalized)
        current_features = nn.relu(current_features)

        # Process portfolio weights
        portfolio_normalized = jnp.clip(portfolio_weights, 0.0, 1.0)  # Portfolio weights should be [0,1]
        portfolio_features = self.portfolio_proj(portfolio_normalized)
        portfolio_features = nn.relu(portfolio_features)

        # Process short positions
        short_positions_normalized = jnp.clip(short_positions, 0.0, 1.0)  # Short positions are binary flags
        short_features = self.short_positions_proj(short_positions_normalized)
        short_features = nn.relu(short_features)

        # Process market state
        market_state_normalized = jnp.clip(market_state, -10.0, 10.0)
        market_features = self.market_state_proj(market_state_normalized)
        market_features = nn.relu(market_features)

        # Fuse all features
        combined_features = jnp.concatenate([
            historical_features,
            current_features, 
            portfolio_features,
            short_features,
            market_features
        ], axis=-1)
        
        fused_features = self.fusion_layer(combined_features)
        fused_features = nn.relu(fused_features)
        
        # Final cleaning
        fused_features = jnp.where(jnp.isnan(fused_features), 0.0, fused_features)
        fused_features = jnp.clip(fused_features, -10.0, 10.0)

        # Actor head (policy)
        logits = self.actor_head(fused_features)
        logits = jnp.where(jnp.isnan(logits), 0.0, logits)
        logits = jnp.clip(logits, -10.0, 10.0)

        # Critic head (value)
        values = self.critic_head(fused_features).squeeze(-1)
        values = jnp.where(jnp.isnan(values), 0.0, values)
        values = jnp.clip(values, -100.0, 100.0)

        # Dummy carry for compatibility
        new_carry = [LSTMState(h=jnp.zeros((batch_size, 1)), c=jnp.zeros((batch_size, 1)))]

        return logits, values, new_carry

# ============================================================================
# PPO TRAINER WITH COMPREHENSIVE ERROR HANDLING
# ============================================================================

# Key fixes for the PPOTrainer class in train_ppo_transformer.py

class PPOTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nan_count = 0
        self.max_nan_resets = 5

        logger.info("Initializing PPO Trainer...")

        # Initialize environment with error handling
        try:
            env_config = self._get_env_config()
            self.env = JAXVectorizedPortfolioEnv(**env_config)
            logger.info(f"Environment initialized: obs_dim={self.env.obs_dim}, action_dim={self.env.action_dim}")
            
            # FIXED: Get the correct dimensions from environment
            self.window_size = config.get('window_size', 30)
            self.n_stocks = self.env.n_stocks  # Use n_stocks, not n_assets
            self.n_features = self.env.n_features  # Use the actual n_features from env
            
            # Calculate the historical data portion of the observation
            self.historical_obs_size = self.window_size * self.n_stocks * self.n_features
            
            logger.info(f"Environment dimensions: window_size={self.window_size}, "
                       f"n_stocks={self.n_stocks}, n_features={self.n_features}")
            logger.info(f"Historical obs size: {self.historical_obs_size}, "
                       f"Total obs dim: {self.env.obs_dim}")

        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            raise

        # Rest of initialization...
        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0,))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0))
        
        # FIXED: Initialize network with correct input handling
        self.network = ActorCriticTransformerFlat(
            action_dim=self.env.action_dim,
            obs_dim=self.env.obs_dim,  # Use full observation dimension
            window_size=self.window_size,
            n_stocks=self.n_stocks,
            n_features=self.n_features,
            d_model=config.get('d_model', 64),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 2),
            dropout_rate=config.get('dropout_rate', 0.1)
        )

        # Initialize parameters with correct dummy input
        self._initialize_parameters()
        # ... rest of init

    def _initialize_parameters(self):
        """Initialize network parameters with correct input shape"""
        logger.info("Initializing network parameters...")

        self.rng = random.PRNGKey(self.config.get('seed', 42))
        self.rng, init_rng = random.split(self.rng)

        # FIXED: Create dummy input with correct flattened observation shape
        dummy_obs = jnp.ones((self.config.get('n_envs', 8), self.env.obs_dim))

        try:
            self.params = self.network.init(init_rng, dummy_obs)
            logger.info("Network parameters initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize network parameters: {e}")
            raise
        
        # Validation and testing...
        if self._has_nan_params(self.params):
            logger.warning("NaN detected in initial parameters, reinitializing...")
            self.rng, init_rng = random.split(self.rng)
            self.params = self.network.init(init_rng, dummy_obs)

            if self._has_nan_params(self.params):
                raise RuntimeError("Failed to initialize parameters without NaN")

        self._test_network_forward_pass(dummy_obs)
        
        # Initialize optimizer and train state
        self.optimizer = optax.adam(learning_rate=self.config.get('learning_rate', 1e-4))
        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=self.params,
            tx=self.optimizer
        )
        logger.info("Train state initialized successfully")

    def _initialize_environment_state(self):
        """Initialize environment state with correct observation handling"""
        logger.info("Initializing environment state...")

        try:
            self.rng, *reset_keys = random.split(self.rng, self.config.get('n_envs', 8) + 1)
            reset_keys = jnp.array(reset_keys)
            self.env_states, self.obs = self.vmap_reset(reset_keys)
            
            # FIXED: Keep observations flattened - don't reshape
            # The observations are already in the correct shape from the environment
            
            # Clean environment observations
            self.obs = jnp.where(jnp.isnan(self.obs), 0.0, self.obs)
            self.obs = jnp.where(jnp.isinf(self.obs), 0.0, self.obs)
            
            if check_for_nans(self.obs, "initial observations"):
                raise RuntimeError("NaN detected in initial environment observations after cleaning")
            
            # Debug: Check observation statistics
            obs_mean = jnp.mean(self.obs)
            obs_std = jnp.std(self.obs)
            obs_max = jnp.max(self.obs)
            obs_min = jnp.min(self.obs)
            logger.info(f"Initial observations - mean: {obs_mean:.6f}, std: {obs_std:.6f}, "
                       f"range: [{obs_min:.6f}, {obs_max:.6f}], shape: {self.obs.shape}")
            
            if obs_std < 1e-8:
                logger.warning("⚠️ Initial observations have very low variance - this may cause training issues")

            logger.info("✅ Environment state initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize environment state: {e}")
            raise

    def _create_empty_trajectory(self) -> Trajectory:
        """Create empty trajectory with correct observation shape"""
        n_steps = self.config.get('n_steps', 128)
        n_envs = self.config.get('n_envs', 8)

        return Trajectory(
            obs=jnp.zeros((n_steps, n_envs, self.env.obs_dim)),  # FIXED: Use correct obs_dim
            actions=jnp.zeros((n_steps, n_envs, self.env.action_dim)),
            rewards=jnp.zeros((n_steps, n_envs)),
            values=jnp.zeros((n_steps, n_envs)),
            log_probs=jnp.zeros((n_steps, n_envs)),
            dones=jnp.zeros((n_steps, n_envs), dtype=bool),
        )

    def _get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration"""
        return {
            'data_root': self.config['data_root'],
            'stocks': self.config.get('stocks', None),
            'start_date': self.config['train_start_date'],
            'end_date': self.config['train_end_date'],
            'window_size': self.config.get('window_size', 30),
            'transaction_cost_rate': self.config.get('transaction_cost_rate', 0.005),
            'sharpe_window': self.config.get('sharpe_window', 252),
            'use_all_features': self.config.get('use_all_features', True),
            # 'fill_missing_features_with': self.config.get('fill_missing_features_with', 'interpolate'),
            # 'save_cache': self.config.get('save_cache', True),
            # 'cache_format': self.config.get('cache_format', 'hdf5'),
            # 'force_reload': self.config.get('force_reload', False),
            # 'preload_to_gpu': self.config.get('preload_to_gpu', True)
            'hdf5_file': self.config.get('hdf5_file', None)
        }

    def _initialize_parameters(self):
        """Initialize network parameters with comprehensive error handling"""
        logger.info("Initializing network parameters...")

        # Initialize RNG
        self.rng = random.PRNGKey(self.config.get('seed', 42))
        self.rng, init_rng = random.split(self.rng)

        # Create dummy inputs for initialization - use flattened observation shape
        dummy_obs = jnp.ones((self.config.get('n_envs', 8), self.env.obs_dim))

        # Initialize parameters with error handling
        try:
            # Initialize network parameters. Note: Transformer does not use LSTM carry
            self.params = self.network.init(init_rng, dummy_obs)
            logger.info("Network parameters initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize network parameters: {e}")
            raise
        
        # Validate initial parameters
        if self._has_nan_params(self.params):
            logger.warning("NaN detected in initial parameters, reinitializing...")
            self.rng, init_rng = random.split(self.rng)
            self.params = self.network.init(init_rng, dummy_obs)

            if self._has_nan_params(self.params):
                raise RuntimeError("Failed to initialize parameters without NaN")

        # Test forward pass
        self._test_network_forward_pass(dummy_obs)

    def _has_nan_params(self, params) -> bool:
        """Check for NaN values in parameters"""
        def check_nan(x):
            return jnp.any(jnp.isnan(x)) if jnp.issubdtype(x.dtype, jnp.floating) else False

        has_nan = jax.tree_util.tree_reduce(
            lambda acc, x: acc | check_nan(x),
            params, False
        )
        return has_nan

    def _test_network_forward_pass(self, obs: chex.Array):
        """Test network forward pass for NaN issues"""
        logger.info("Testing network forward pass...")

        try:
            # Split PRNG key for dropout
            self.rng, new_rng = random.split(self.rng)

            # Test network forward pass. Note: Transformer does not use LSTM carry
            logits, values, _ = self.network.apply(self.params, obs, rngs={"dropout": new_rng})

            # Check for NaN in outputs
            if check_for_nans(logits, "logits"):
                raise RuntimeError("NaN detected in logits output")
            if check_for_nans(values, "values"):
                raise RuntimeError("NaN detected in values output")

            logger.info("✅ Network forward pass test passed")

        except Exception as e:
            logger.error(f"❌ Network forward pass test failed: {e}")
            raise

    def _initialize_environment_state(self):
        """Initialize environment and 'LSTM' states (dummy for Transformer)"""
        logger.info("Initializing environment state...")

        try:
            # Initialize environment
            self.rng, *reset_keys = random.split(self.rng, self.config.get('n_envs', 8) + 1)
            reset_keys = jnp.array(reset_keys)
            self.env_states, self.obs = self.vmap_reset(reset_keys)

            # Debug: Log the shape of self.obs
            logger.info(f"Initial obs shape: {self.obs.shape}")

            # FIXED: Keep observations flattened - don't reshape
            # The observations are already in the correct shape from the environment

            # Clean environment observations (handle inf/nan)
            self.obs = jnp.where(jnp.isnan(self.obs), 0.0, self.obs)
            self.obs = jnp.where(jnp.isinf(self.obs), 0.0, self.obs)

            # Validate environment outputs
            if check_for_nans(self.obs, "initial observations"):
                raise ValueError("NaN detected in initial observations")

            # Debug: Check observation statistics
            obs_mean = jnp.mean(self.obs)
            obs_std = jnp.std(self.obs)
            obs_max = jnp.max(self.obs)
            obs_min = jnp.min(self.obs)
            logger.info(f"Initial observations - mean: {obs_mean:.6f}, std: {obs_std:.6f}, range: [{obs_min:.6f}, {obs_max:.6f}]")

            if obs_std < 1e-8:
                raise ValueError("Standard deviation of observations is too small, indicating potential issues with the data")

            logger.info("✅ Environment state initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize environment state: {e}")
            raise


    def collect_trajectory(self, train_state: TrainState, env_states: List[EnvState],
                        initial_obs: chex.Array, rng_key: chex.PRNGKey) -> Tuple[Trajectory, List[EnvState], chex.Array]:
        """Collect trajectory using current policy. No LSTM state management for Transformer."""

        def step_fn(carry_step, _):
            """Single step in trajectory collection"""
            env_states, obs, rng_key = carry_step

            # Lightweight observation cleaning (only essential)
            obs = jnp.where(jnp.isnan(obs), 0.0, obs)
            obs = jnp.clip(obs, -50.0, 50.0)  # Reduced clipping range

            # Get action from policy
            rng_key, action_rng = random.split(rng_key)

            # Apply network with error handling. No LSTM carry for Transformer.
            try:
                logits, values, _ = train_state.apply_fn(
                    train_state.params, obs, rngs={"dropout": action_rng}
                )
            except Exception as e:
                logger.error(f"Network forward pass failed: {e}")
                # Return safe defaults
                logits = jnp.zeros((obs.shape[0], self.env.action_dim))
                values = jnp.zeros(obs.shape[0])

            # Lightweight network output cleaning
            logits = jnp.where(jnp.isnan(logits), 0.0, logits)
            values = jnp.where(jnp.isnan(values), 0.0, values)
            logits = jnp.clip(logits, -5.0, 5.0)  # Reduced clipping
            values = jnp.clip(values, -50.0, 50.0)  # Reduced clipping

            # Sample actions
            action_std = self.config.get('action_std', 1.0)
            action_std = jnp.maximum(action_std, 1e-6)  # Ensure positive std

            action_distribution = distrax.Normal(loc=logits, scale=action_std)
            actions = action_distribution.sample(seed=action_rng)
            actions = jnp.clip(actions, -5.0, 5.0)

            # Calculate log probabilities
            log_probs = action_distribution.log_prob(actions).sum(axis=-1)
            log_probs = jnp.where(jnp.isnan(log_probs), -10.0, log_probs)
            log_probs = jnp.clip(log_probs, -50.0, 10.0)

            # Step environment
            new_env_states, next_obs, rewards, dones, info = self.vmap_step(env_states, actions)

            # FIXED: Keep next_obs flattened - no reshaping needed
            next_obs = jnp.where(jnp.isnan(next_obs), 0.0, next_obs)
            next_obs = jnp.clip(next_obs, -50.0, 50.0)

            # Lightweight environment output cleaning
            next_obs = jnp.where(jnp.isnan(next_obs), 0.0, next_obs)
            next_obs = jnp.clip(next_obs, -50.0, 50.0)  # Reduced clipping

            rewards = jnp.where(jnp.isnan(rewards), 0.0, rewards)
            rewards = jnp.clip(rewards, -50.0, 50.0)  # Reduced clipping

            # Create trajectory transition
            transition = Trajectory(
                obs=obs,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                dones=dones,
            )

            return (new_env_states, next_obs, rng_key), transition

        # Roll out trajectory using lax.scan (JIT-compiled, much faster)
        n_steps = self.config.get('n_steps', 64)
        initial_carry = (env_states, initial_obs, rng_key)

        try:
            # Use regular Python loop for stability (no JIT compilation)
            current_carry = initial_carry
            trajectory_list = []

            for step in range(n_steps):
                current_carry, transition = step_fn(current_carry, None)
                trajectory_list.append(transition)

            # Convert to trajectory format
            trajectory = jax.tree_util.tree_map(
                lambda *args: jnp.stack(args, axis=0), *trajectory_list
            )

        except Exception as e:
            logger.error(f"Trajectory collection failed: {e}")
            # Return safe defaults
            return self._create_empty_trajectory(), env_states, initial_obs

        final_env_states, final_obs, _ = current_carry

        return trajectory, final_env_states, final_obs

    def _create_empty_trajectory(self) -> Trajectory:
        """Create empty trajectory for error recovery"""
        n_steps = self.config.get('n_steps', 128)
        n_envs = self.config.get('n_envs', 8)

        return Trajectory(
            obs=jnp.zeros((n_steps, n_envs, self.env.obs_dim)),  # FIXED: Use correct flattened obs_dim
            actions=jnp.zeros((n_steps, n_envs, self.env.action_dim)),
            rewards=jnp.zeros((n_steps, n_envs)),
            values=jnp.zeros((n_steps, n_envs)),
            log_probs=jnp.zeros((n_steps, n_envs)),
            dones=jnp.zeros((n_steps, n_envs), dtype=bool),
        )
        
    @partial(jax.jit, static_argnums=(0,), device=jax.devices('gpu')[0])
    def compute_gae(self, trajectory: Trajectory, last_values: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Compute Generalized Advantage Estimation with robust numerical stability"""
        gamma = self.config.get('gamma', 0.99)
        gae_lambda = self.config.get('gae_lambda', 0.95)

        # Clean inputs
        rewards = jnp.where(jnp.isnan(trajectory.rewards), 0.0, trajectory.rewards)
        values = jnp.where(jnp.isnan(trajectory.values), 0.0, trajectory.values)
        last_values = jnp.where(jnp.isnan(last_values), 0.0, last_values)
        
        # Clip inputs to reasonable ranges
        rewards = jnp.clip(rewards, -100.0, 100.0)
        values = jnp.clip(values, -100.0, 100.0)
        last_values = jnp.clip(last_values, -100.0, 100.0)

        # Extend values for GAE calculation
        extended_values = jnp.concatenate([values, last_values[None, :]], axis=0)

        def gae_step(gae_carry, inputs):
            """Single GAE computation step"""
            current_gae = gae_carry
            reward, value, next_value, done = inputs

            # Compute TD error
            delta = reward + gamma * next_value * (1 - done) - value

            # Compute advantage
            advantage = delta + gamma * gae_lambda * (1 - done) * current_gae

            # Clean advantage
            advantage = jnp.where(jnp.isnan(advantage), 0.0, advantage)
            advantage = jnp.clip(advantage, -100.0, 100.0)

            return advantage, advantage

        # Prepare inputs for reverse scan (JIT-safe with static shapes)
        gae_inputs = (
            rewards[::-1], # Reverse rewards
            values[::-1], # Reverse values
            extended_values[1:][::-1], # Reverse next values
            trajectory.dones[::-1] # Reverse dones
        )

        # Initial GAE (zero for all environments)
        init_gae = jnp.zeros_like(last_values)

        # Compute advantages in reverse order using lax.scan (JIT-safe)
        _, advantages_reversed = lax.scan(gae_step, init_gae, gae_inputs)

        # Reverse back to original order
        advantages = advantages_reversed[::-1]

        # Final cleaning
        advantages = jnp.where(jnp.isnan(advantages), 0.0, advantages)
        advantages = jnp.clip(advantages, -100.0, 100.0)
        
        # Compute returns
        returns = advantages + values
        returns = jnp.where(jnp.isnan(returns), 0.0, returns)
        returns = jnp.clip(returns, -100.0, 100.0)
        
        return advantages, returns

    @partial(jax.jit, static_argnums=(0,), device=jax.devices('gpu')[0])
    def ppo_loss(self, params: chex.Array, train_batch: Trajectory, gae_advantages: chex.Array,
                 returns: chex.Array, rng_key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """Compute PPO loss with comprehensive numerical stability"""

        # Clean batch data
        obs_batch = jnp.where(jnp.isnan(train_batch.obs), 0.0, train_batch.obs)
        actions_batch = jnp.where(jnp.isnan(train_batch.actions), 0.0, train_batch.actions)
        old_log_probs_batch = jnp.where(jnp.isnan(train_batch.log_probs), -10.0, train_batch.log_probs)
        
        batch_size = obs_batch.shape[0]

        # Get current policy outputs. No LSTM carry for Transformer.
        try:
            logits, values, _ = self.network.apply(params, obs_batch)
        except Exception as e:
            logger.error(f"Network forward pass failed in PPO loss: {e}")
            # Return safe defaults
            logits = jnp.zeros((batch_size, self.env.action_dim))
            values = jnp.zeros(batch_size)

        # Clean network outputs
        logits = jnp.where(jnp.isnan(logits), 0.0, logits)
        values = jnp.where(jnp.isnan(values), 0.0, values)
        logits = jnp.clip(logits, -10.0, 10.0)
        values = jnp.clip(values, -100.0, 100.0)
        
        # Clean targets
        returns = jnp.where(jnp.isnan(returns), 0.0, returns)
        gae_advantages = jnp.where(jnp.isnan(gae_advantages), 0.0, gae_advantages)
        returns = jnp.clip(returns, -100.0, 100.0)
        
        # Safe advantage normalization
        gae_advantages = safe_normalize(gae_advantages)

        # Value loss (clipped)
        old_values = jnp.where(jnp.isnan(train_batch.values), 0.0, train_batch.values)
        old_values = jnp.clip(old_values, -100.0, 100.0)
        
        value_pred_clipped = old_values + (values - old_values).clip(
            -self.config.get('clip_eps', 0.2),
            self.config.get('clip_eps', 0.2)
        )

        value_losses = jnp.square(values - returns)
        value_losses_clipped = jnp.square(value_pred_clipped - returns)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        
        # Policy loss
        action_std = self.config.get('action_std', 1.0)
        action_std = jnp.maximum(action_std, 1e-6)

        action_distribution = distrax.Normal(loc=logits, scale=action_std)
        new_log_probs = action_distribution.log_prob(actions_batch).sum(axis=-1)
        new_log_probs = jnp.where(jnp.isnan(new_log_probs), -10.0, new_log_probs)
        new_log_probs = jnp.clip(new_log_probs, -50.0, 10.0)
        
        # Compute importance sampling ratio with numerical stability
        log_ratio = new_log_probs - old_log_probs_batch
        log_ratio = jnp.clip(log_ratio, -10.0, 10.0)
        
        # Use log-sum-exp trick for numerical stability
        ratio = jnp.exp(jnp.clip(log_ratio, -10.0, 10.0))
        ratio = jnp.clip(ratio, 1e-8, 10.0) # Prevent division by zero

        # PPO policy loss with clipping
        clip_eps = self.config.get('clip_eps', 0.2)
        pg_losses1 = ratio * gae_advantages
        pg_losses2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_advantages
        policy_loss = -jnp.minimum(pg_losses1, pg_losses2).mean()
        
        # Entropy bonus
        entropy = action_distribution.entropy().sum(axis=-1)
        entropy = jnp.where(jnp.isnan(entropy), 0.0, entropy)
        entropy = jnp.clip(entropy, 0.0, 10.0)
        entropy_coeff = self.config.get('entropy_coeff', 0.01)
        entropy_loss = -entropy_coeff * entropy.mean()
        
        # Total loss
        value_coeff = self.config.get('value_coeff', 0.5)
        total_loss = policy_loss + value_coeff * value_loss + entropy_loss
        total_loss = jnp.where(jnp.isnan(total_loss), 0.0, total_loss)
        
        # Compute metrics
        approx_kl = jnp.where(ratio == 0.0, 0.0, (ratio - 1) - jnp.log(ratio))
        approx_kl = jnp.where(jnp.isnan(approx_kl), 0.0, approx_kl)
        approx_kl = jnp.clip(approx_kl, -10.0, 10.0)
        
        clip_fraction = (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)
        
        metrics = {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'approx_kl': approx_kl.mean(),
            'clip_fraction': clip_fraction.mean(),
            'mean_ratio': ratio.mean(),
            'mean_advantage': gae_advantages.mean()
        }
        
        return total_loss, metrics

    @partial(jax.jit, static_argnums=(0,), device=jax.devices('gpu')[0])
    def train_step(self, train_state: TrainState, train_batch: Trajectory, gae_advantages: chex.Array,
               returns: chex.Array, rng_key: chex.PRNGKey) -> Tuple[TrainState, Dict[str, chex.Array]]:
        """Performs a single PPO training step (compute gradients and update parameters)"""

        # Define the loss function to be differentiated
        grad_fn = jax.value_and_grad(self.ppo_loss, has_aux=True)

        # Compute loss and gradients
        (total_loss, metrics), grads = grad_fn(
            train_state.params, train_batch, gae_advantages, returns, rng_key
        )

        # Apply gradients
        train_state = train_state.apply_gradients(grads=grads)

        # Log NaN in gradients
        if self._has_nan_params(grads):
            logger.warning("NaN detected in gradients during train step!")
            metrics['nan_in_gradients'] = 1.0
        else:
            metrics['nan_in_gradients'] = 0.0

        return train_state, metrics

    def train(self):
        """Main training loop for PPO"""
        logger.info("Starting PPO training...")

        n_envs = self.config.get('n_envs', 8)
        n_steps = self.config.get('n_steps', 64)
        total_timesteps = self.config.get('total_timesteps', 1_000_000)
        n_updates = total_timesteps // (n_envs * n_steps)
        n_minibatch = self.config.get('n_minibatch', 4)
        update_epochs = self.config.get('update_epochs', 4)
    
        batch_size = n_envs * n_steps
        minibatch_size = batch_size // n_minibatch

        if batch_size % n_minibatch != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by n_minibatch ({n_minibatch})")

        global_step = 0
        start_time = time.time()
    
        # Initial environment state and observations are already set in __init__
        env_states = self.env_states
        obs = self.obs
        #current_lstm_state = [LSTMState(h=jnp.zeros((n_envs, 1)), c=jnp.zeros((n_envs, 1)))] # Dummy for transformer

        for update in range(n_updates):
            self.rng, collect_rng = random.split(self.rng)
        
            # Collect trajectory with error handling
            try:
                trajectory, env_states, obs = self.collect_trajectory(
                    self.train_state, env_states, obs, collect_rng
                )
            
                # Check for NaNs in collected trajectory
                if any(check_for_nans(getattr(trajectory, field), field) for field in Trajectory._fields):
                    logger.error(f"NaN detected in trajectory after collection at update {update}, resetting environment and re-collecting.")
                    self.nan_count += 1
                    if self.nan_count > self.max_nan_resets:
                        raise RuntimeError("Too many NaN resets, stopping training.")
                    self._initialize_environment_state() # Re-initialize env state
                    env_states = self.env_states
                    obs = self.obs
                    #current_lstm_state = [LSTMState(h=jnp.zeros((n_envs, 1)), c=jnp.zeros((n_envs, 1)))] # Reset dummy LSTM state
                    continue # Skip to next update

            except Exception as e:
                logger.error(f"Trajectory collection failed at update {update}: {e}, resetting and continuing.")
                self.nan_count += 1
                if self.nan_count > self.max_nan_resets:
                    raise RuntimeError("Too many NaN resets, stopping training.")
                self._initialize_environment_state() # Re-initialize env state
                env_states = self.env_states
                obs = self.obs
                #current_lstm_state = [LSTMState(h=jnp.zeros((n_envs, 1)), c=jnp.zeros((n_envs, 1)))] # Reset dummy LSTM state
                continue # Skip to next update

            # Calculate last_values for GAE
            self.rng, last_value_rng = random.split(self.rng)
            try:
                _, last_values, _ = self.network.apply(self.train_state.params, obs)
                last_values = jnp.where(jnp.isnan(last_values), 0.0, last_values)
                last_values = jnp.clip(last_values, -100.0, 100.0)
            except Exception as e:
                logger.error(f"Failed to compute last values for GAE at update {update}: {e}")
                last_values = jnp.zeros(n_envs) # Default to zeros

            # Compute GAE advantages and returns
            gae_advantages, returns = self.compute_gae(trajectory, last_values)

            # Flatten the trajectory for training
            flat_trajectory = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), trajectory
            )
            flat_gae_advantages = gae_advantages.reshape(-1)
            flat_returns = returns.reshape(-1)

            # Check for NaNs after GAE computation
            if check_for_nans(flat_gae_advantages, "flat_gae_advantages") or check_for_nans(flat_returns, "flat_returns"):
                logger.error(f"NaN detected in GAE advantages or returns at update {update}, skipping update.")
                self.nan_count += 1
                if self.nan_count > self.max_nan_resets:
                    raise RuntimeError("Too many NaN resets, stopping training.")
                continue # Skip to next update

            # PPO epochs
            for epoch in range(update_epochs):
                self.rng, shuffle_rng = random.split(self.rng)
                permutation = jax.random.permutation(shuffle_rng, batch_size)

                # Shuffle and split into minibatches
                for i in range(n_minibatch):
                    batch_indices = permutation[i * minibatch_size : (i + 1) * minibatch_size]
                
                    minibatch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, batch_indices, axis=0), flat_trajectory
                    )
                    minibatch_advantages = jnp.take(flat_gae_advantages, batch_indices, axis=0)
                    minibatch_returns = jnp.take(flat_returns, batch_indices, axis=0)

                    # Perform training step
                    self.rng, train_rng = random.split(self.rng)
                    try:
                        self.train_state, metrics = self.train_step(
                            self.train_state, minibatch, minibatch_advantages, minibatch_returns, train_rng
                        )
                        # Aggregate metrics
                        if self.config.get('use_wandb', False):
                            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)
                    except Exception as e:
                        logger.error(f"Training step failed at update {update}, epoch {epoch}, minibatch {i}: {e}, skipping minibatch.")
                        self.nan_count += 1
                        if self.nan_count > self.max_nan_resets:
                            raise RuntimeError("Too many NaN resets, stopping training.")
                        break # Break from minibatch loop, try next epoch or update

            global_step += batch_size
            if update % 10 == 0:
                elapsed_time = time.time() - start_time
                logger.info(f"Update: {update}/{n_updates}, Global Steps: {global_step}, Time: {elapsed_time:.2f}s")
                if self.config.get('use_wandb', False):
                    wandb.log({"chart/SPS": global_step / elapsed_time}, step=global_step)

            # Early stopping based on KL divergence
            if metrics['approx_kl'] > self.config.get('target_kl', 0.015) * 1.5:
                logger.warning(f"KL divergence {metrics['approx_kl']:.4f} exceeded target_kl * 1.5. Early stopping.")
                break

            # Save model periodically
            if update % self.config.get('save_interval', 100) == 0:
                self.save_model(f"checkpoint_{update}.pkl")

        self.save_model("final_model.pkl")
        if self.config.get('use_wandb', False):
            wandb.finish()
        logger.info("PPO training finished!")

    def save_model(self, filename: str = "model.pkl"):
        """Saves the current train state parameters to a file."""
        model_dir = Path(self.config.get('model_dir', './models'))
        model_dir.mkdir(parents=True, exist_ok=True)
        filepath = model_dir / filename
        with open(filepath, "wb") as f:
            f.write(serialization.to_bytes(self.train_state.params))
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filename: str = "model.pkl"):
        """Loads model parameters from a file."""
        model_dir = Path(self.config.get('model_dir', './models'))
        filepath = model_dir / filename
        with open(filepath, "rb") as f:
            params_bytes = f.read()
    
        # We need to re-initialize the network to get a "template" for deserialization
        dummy_obs = jnp.ones((1, self.env.obs_dim))  # FIXED: Use flattened observation shape
        temp_params = self.network.init(random.PRNGKey(0), dummy_obs) # Use a dummy key
        loaded_params = serialization.from_bytes(temp_params, params_bytes)
        self.train_state = self.train_state.replace(params=loaded_params)
        logger.info(f"Model loaded from {filepath}")

    def evaluate(self, eval_start_date: str, eval_end_date: str, n_eval_envs: int = 1):
        """
        Evaluates the trained model on a separate dataset.
        Returns a dictionary of evaluation metrics.
        """
        logger.info("Starting evaluation...")

        # Create a separate evaluation environment
        eval_env_config = self._get_env_config()
        eval_env_config['start_date'] = eval_start_date
        eval_env_config['end_date'] = eval_end_date
        eval_env_config['preload_to_gpu'] = False # For potentially larger eval data

        try:
            eval_env = JAXVectorizedPortfolioEnv(**eval_env_config)
            logger.info(f"Evaluation environment initialized: obs_dim={eval_env.obs_dim}, action_dim={eval_env.action_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize evaluation environment: {e}")
            raise

        vmap_eval_reset = jax.vmap(eval_env.reset, in_axes=(0,))
        vmap_eval_step = jax.vmap(eval_env.step, in_axes=(0, 0))

        rng_key = random.PRNGKey(self.config.get('seed', 42) + 123) # Different seed for eval
        rng_key, *reset_keys = random.split(rng_key, n_eval_envs + 1)
        reset_keys = jnp.array(reset_keys)
        eval_env_states, eval_obs = vmap_eval_reset(reset_keys)
    
        # FIXED: Keep eval_obs flattened - no reshaping needed
        eval_obs = jnp.where(jnp.isnan(eval_obs), 0.0, eval_obs)
        eval_obs = jnp.where(jnp.isinf(eval_obs), 0.0, eval_obs)

        # Track total rewards, portfolio values, etc.
        total_rewards = jnp.zeros(n_eval_envs)
        episode_lengths = jnp.zeros(n_eval_envs)
        is_done = jnp.zeros(n_eval_envs, dtype=bool)

        eval_steps = 0
        max_eval_steps = eval_env.total_steps # Or a fixed number for evaluation

        logger.info(f"Starting evaluation rollout for {max_eval_steps} steps...")
    
        while not jnp.all(is_done) and eval_steps < max_eval_steps:
            rng_key, action_rng = random.split(rng_key)

            # Get action from policy (no training mode)
            logits, _, _ = self.network.apply(self.train_state.params, eval_obs, training=False)
            logits = jnp.where(jnp.isnan(logits), 0.0, logits)
            logits = jnp.clip(logits, -5.0, 5.0)

            action_std = self.config.get('action_std', 1.0) # Use the same std as training
            action_std = jnp.maximum(action_std, 1e-6)

            action_distribution = distrax.Normal(loc=logits, scale=action_std)
            actions = action_distribution.sample(seed=action_rng)
            actions = jnp.clip(actions, -5.0, 5.0)

            # Step environment
            new_eval_env_states, next_eval_obs, rewards, dones, info = vmap_eval_step(eval_env_states, actions)

            # Update observations and states
            eval_env_states = new_eval_env_states
            # FIXED: Keep next_eval_obs flattened - no reshaping needed
            eval_obs = jnp.where(jnp.isnan(next_eval_obs), 0.0, next_eval_obs)
            eval_obs = jnp.where(jnp.isinf(eval_obs), 0.0, eval_obs)

            # Accumulate rewards for episodes not yet done
            total_rewards += rewards * (1 - is_done.astype(float))
            episode_lengths += (1 - is_done.astype(float)) # Increment for active episodes
            is_done = is_done | dones # Mark episodes as done

            eval_steps += 1
            if eval_steps % 100 == 0:
                logger.info(f"Eval Step: {eval_steps}/{max_eval_steps}, Mean Reward: {jnp.mean(total_rewards):.4f}")

        # Final evaluation metrics
        mean_total_reward = jnp.mean(total_rewards)
        mean_episode_length = jnp.mean(episode_lengths)

        # Retrieve final portfolio values and calculate other metrics if available
        # This assumes the environment's info dict contains relevant evaluation metrics
        # The info dict from env.step is typically per-environment, so we'd need to aggregate
    
        # For simplicity, let's assume portfolio_values are directly accessible in info
        # This part might need adjustment based on your specific EnvState structure
        final_portfolio_values = jnp.array([state.portfolio_value for state in eval_env_states])
        mean_final_portfolio_value = jnp.mean(final_portfolio_values)

        # Example of how you might get sharpe ratio if the env provides it
        # This will depend on the `info` structure from your JAXVectorizedPortfolioEnv
        # For a truly robust evaluation, you'd collect a full history of portfolio values
        # and calculate metrics outside the environment.
    
        metrics = {
            "eval/mean_total_reward": mean_total_reward,
            "eval/mean_episode_length": mean_episode_length,
            "eval/mean_final_portfolio_value": mean_final_portfolio_value,
            # Add other metrics like Sharpe Ratio, Max Drawdown if available from info or computed from history
        }

        logger.info("Evaluation complete!")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.6f}")
            if self.config.get('use_wandb', False):
                wandb.log({k: v}, step=global_step if 'global_step' in locals() else 0)

        return metrics
# Example Usage

if __name__ == '__main__':
    # Define a configuration dictionary
    config = {
        'seed': 42,
        'n_envs': 8,
        'n_steps': 128, # Number of steps to collect in each environment per update
        'total_timesteps': 500_000,
        'learning_rate': 1e-4,
        'max_grad_norm': 0.5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'entropy_coeff': 0.01,
        'value_coeff': 0.5,
        'update_epochs': 4,
        'n_minibatch': 4,
        'target_kl': 0.01,
        'action_std': 0.5, # Initial action standard deviation
        'use_wandb': True, # Set to True to enable Weights & Biases logging
        'model_dir': './models',
        'save_interval': 100, # Save model every X updates

        # Environment specific configurations
        'data_root': './processed_data', # Path to your financial data
        'stocks': None, # e.g., ['AAPL', 'MSFT', 'GOOG'] or None for all
        'train_start_date': '2024-06-06',
        'train_end_date': '2025-03-06',
        'window_size': 30, # Lookback window for observations
        'transaction_cost_rate': 0.001,
        'sharpe_window': 252,
        'use_all_features': True,
        # 'fill_missing_features_with': 'interpolate',
        'save_cache': True,
        'cache_format': 'parquet',
        'force_reload': False,
        'preload_to_gpu': True,
        'hdf5_file': None, # Path to HDF5 file if using

        # Transformer specific configurations
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dropout_rate': 0.1
    }

    # Ensure data directory exists
    Path(config['data_root']).mkdir(parents=True, exist_ok=True)

    # Initialize and train the PPO agent
    trainer = PPOTrainer(config)
    trainer.train()

    # After training, evaluate the model
    eval_start_date = '2021-01-01'
    eval_end_date = '2023-12-31'
    eval_metrics = trainer.evaluate(eval_start_date, eval_end_date)
    print("\nEvaluation Metrics:", eval_metrics)

# Example of loading a saved model and evaluating again
# new_trainer = PPOTrainer(config) # Create a new trainer instance
# new_trainer.load_model("final_model.pkl")
# new_eval_metrics = new_trainer.evaluate(eval_start_date, eval_end_date)
# print("\nEvaluation Metrics (Loaded Model):", new_eval_metrics)
