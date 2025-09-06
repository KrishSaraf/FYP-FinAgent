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
from typing import Tuple, Dict, Any, NamedTuple, List
import wandb
import pickle
import distrax
from pathlib import Path
from functools import partial
import json

# Import the JAX environment (assume it's in the same directory or installed)
from finagent.environment.portfolio_env import JAXVectorizedPortfolioEnv, EnvState

# Enable JAX optimizations
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_compilation_cache_dir', './jax_cache')

# Trajectory storage
class Trajectory(NamedTuple):
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array
    # Storing initial_carry (List[LSTMCarry]) for each step, so we can re-evaluate
    # This will be (n_steps, n_envs, n_lstm_layers, h/c dim)
    initial_lstm_h: chex.Array 
    initial_lstm_c: chex.Array


class LSTMCarry(NamedTuple):
    h: chex.Array
    c: chex.Array

# LSTM Policy Network
class LSTMActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 512
    n_lstm_layers: int = 2
    
    def setup(self):
        # Initialize with proper scaling to prevent NaN
        self.input_dense1 = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.input_dense2 = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.actor_dense = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.actor_output = nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.critic_dense = nn.Dense(256, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
        self.critic_output = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform(), bias_init=nn.initializers.zeros)
    
    @nn.compact
    def __call__(self, x, carry: List[LSTMCarry], training=True):
        batch_size = x.shape[0]
        
        # Input preprocessing with NaN checking
        x = self.input_dense1(x)
        x = jnp.where(jnp.isnan(x), 0.0, x)  # Replace NaN with 0
        x = nn.relu(x)
        
        x = self.input_dense2(x)
        x = jnp.where(jnp.isnan(x), 0.0, x)  # Replace NaN with 0
        x = nn.relu(x)
        
        # LSTM layers - fixed implementation
        lstm_input = x
        new_carry = []
        
        for i in range(self.n_lstm_layers):
            if carry[i] is None:
                # Initialize carry with zeros - this is the correct way
                h = jnp.zeros((batch_size, self.hidden_size), dtype=jnp.float32)
                c = jnp.zeros((batch_size, self.hidden_size), dtype=jnp.float32)
                layer_carry = LSTMCarry(h=h, c=c)
            else:
                layer_carry = carry[i]
                # Check for NaN in carry states and reset if found
                h = jnp.where(jnp.isnan(layer_carry.h), 0.0, layer_carry.h)
                c = jnp.where(jnp.isnan(layer_carry.c), 0.0, layer_carry.c)
                layer_carry = LSTMCarry(h=h, c=c)
            
            # LSTM cell - create properly
            lstm_cell = nn.OptimizedLSTMCell(features=self.hidden_size)
            new_h_val, lstm_input = lstm_cell(layer_carry, lstm_input)
            
            # Check for NaN in LSTM output and reset if found
            lstm_input = jnp.where(jnp.isnan(lstm_input), 0.0, lstm_input)
            new_h_val = LSTMCarry(
                h=jnp.where(jnp.isnan(new_h_val[0]), 0.0, new_h_val[0]),
                c=jnp.where(jnp.isnan(new_h_val[1]), 0.0, new_h_val[1])
            )
            new_carry.append(new_h_val)
        
        # Actor head (policy) with NaN checking
        actor_hidden = self.actor_dense(lstm_input)
        actor_hidden = jnp.where(jnp.isnan(actor_hidden), 0.0, actor_hidden)
        actor_hidden = nn.relu(actor_hidden)
        
        logits = self.actor_output(actor_hidden)
        logits = jnp.where(jnp.isnan(logits), 0.0, logits)
        # Clip logits to prevent extreme values
        logits = jnp.clip(logits, -10.0, 10.0)
        
        # Critic head (value function) with NaN checking
        critic_hidden = self.critic_dense(lstm_input)
        critic_hidden = jnp.where(jnp.isnan(critic_hidden), 0.0, critic_hidden)
        critic_hidden = nn.relu(critic_hidden)
        
        values = self.critic_output(critic_hidden).squeeze(-1)
        values = jnp.where(jnp.isnan(values), 0.0, values)
        # Clip values to prevent extreme values
        values = jnp.clip(values, -100.0, 100.0)
        
        return logits, values, new_carry

class PPOTrainer:
    def __init__(self, config: dict):
        self.config = config
        
        # Add NaN checking utility
        self.nan_count = 0
        self.max_nan_resets = 3  # Maximum number of parameter resets allowed (reduced for debugging)
        
        # Initialize environment
        env_config = {
            'data_root': config['data_root'],
            'stocks': config.get('stocks', None),
            'start_date': config['train_start_date'],
            'end_date': config['train_end_date'],
            'window_size': config['window_size'],
            'transaction_cost_rate': config['transaction_cost_rate'],
            'sharpe_window': config['sharpe_window'],
            'use_all_features': True  # Use all available features
        }
        
        self.env = JAXVectorizedPortfolioEnv(**env_config)

        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0,))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0,0))
        
        # Network initialization
        self.network = LSTMActorCritic(
            action_dim=self.env.action_dim,
            hidden_size=config['hidden_size'],
            n_lstm_layers=config['n_lstm_layers']
        )
        
        # Initialize network parameters
        self.rng = random.PRNGKey(config['seed'])
        self.rng, init_rng = random.split(self.rng)
        
        dummy_obs = jnp.ones((config['n_envs'], self.env.obs_dim))
        
        # Initialize LSTM carry for dummy init
        dummy_carry = [
            LSTMCarry(
                h=jnp.zeros((config['n_envs'], config['hidden_size'])),
                c=jnp.zeros((config['n_envs'], config['hidden_size']))
            ) for _ in range(config['n_lstm_layers'])
        ]
        
        # Initialize network with proper error handling
        try:
            self.params = self.network.init(init_rng, dummy_obs, tuple(dummy_carry))
            print("Network initialized successfully")
        except Exception as e:
            print(f"Error during network initialization: {e}")
            raise
        
        # Check for NaN in initial parameters
        def has_nan_params(params):
            return jax.tree_util.tree_reduce(
                lambda acc, x: acc | jnp.any(jnp.isnan(x)) if jnp.issubdtype(x.dtype, jnp.floating) else acc,
                params, False
            )
        
        if has_nan_params(self.params):
            print("WARNING: NaN detected in initial parameters!")
            # Try reinitializing with different seed
            self.rng, init_rng = random.split(self.rng)
            self.params = self.network.init(init_rng, dummy_obs, tuple(dummy_carry))
        
        # Test network forward pass
        print("Testing network forward pass...")
        try:
            test_logits, test_values, test_carry = self.network.apply(self.params, dummy_obs, tuple(dummy_carry))
            if jnp.any(jnp.isnan(test_logits)) or jnp.any(jnp.isnan(test_values)):
                print("❌ Network produces NaN outputs during test!")
                print(f"   NaN in logits: {jnp.any(jnp.isnan(test_logits))}")
                print(f"   NaN in values: {jnp.any(jnp.isnan(test_values))}")
            else:
                print("✅ Network test passed - no NaN outputs")
        except Exception as e:
            print(f"❌ Network test failed: {e}")
            raise
        
        # Test environment
        print("Testing environment...")
        try:
            test_key = random.PRNGKey(42)
            test_env_state, test_obs = self.env.reset(test_key)
            if jnp.any(jnp.isnan(test_obs)):
                print("❌ Environment produces NaN observations!")
            else:
                print("✅ Environment test passed - no NaN observations")
                
            # Test environment step
            test_action = jnp.zeros(self.env.action_dim)
            test_env_state, test_next_obs, test_reward, test_done, test_info = self.env.step(test_env_state, test_action)
            if jnp.any(jnp.isnan(test_next_obs)) or jnp.any(jnp.isnan(test_reward)):
                print("❌ Environment produces NaN in step!")
                print(f"   NaN in obs: {jnp.any(jnp.isnan(test_next_obs))}")
                print(f"   NaN in reward: {jnp.any(jnp.isnan(test_reward))}")
            else:
                print("✅ Environment step test passed - no NaN outputs")
        except Exception as e:
            print(f"❌ Environment test failed: {e}")
            raise
        
        # Optimizer with gradient clipping and better numerical stability
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']),
            optax.adam(learning_rate=config['learning_rate'], eps=1e-8, b1=0.9, b2=0.999) # Better eps and beta values
        )
        
        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=self.params,
            tx=self.optimizer
        )
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(project="jax-ppo-portfolio", config=config)
            
        self.rng, *reset_keys = random.split(self.rng, config['n_envs'] + 1)
        reset_keys = jnp.array(reset_keys)
        self.env_states, self.obs = self.vmap_reset(reset_keys)
        # Initialize the LSTM carry for the collector (will be updated at each step)
        self.collector_carry = [
            LSTMCarry(
                h=jnp.zeros((config['n_envs'], config['hidden_size'])),
                c=jnp.zeros((config['n_envs'], config['hidden_size']))
            ) for _ in range(config['n_lstm_layers'])
        ]
    
    @partial(jax.jit, static_argnums=(0,))
    def collect_trajectory(self, train_state: TrainState, env_states: List[EnvState], initial_obs: chex.Array, initial_carry: List[LSTMCarry], rng_key: chex.PRNGKey) -> Tuple[Trajectory, List[EnvState], chex.Array, List[LSTMCarry]]:
        """Collect trajectory using current policy"""
    
        def step_fn(carry_step, _):
            env_states, obs, lstm_carry_tuple, rng_key = carry_step
        
            # Get action from policy
            rng_key, action_rng = random.split(rng_key)
        
            # Apply network
            logits, values, new_carry_list = train_state.apply_fn(train_state.params, obs, lstm_carry_tuple)
        
            # Sample actions from the distribution with NaN protection
            action_std = self.config['action_std']
            
            # Ensure logits and action_std are valid
            logits = jnp.where(jnp.isnan(logits), 0.0, logits)
            logits = jnp.clip(logits, -10.0, 10.0)
            action_std = jnp.maximum(action_std, 1e-6)  # Ensure positive std
            
            action_distribution = distrax.Normal(loc=logits, scale=action_std)
            actions = action_distribution.sample(seed=action_rng)
            actions = jnp.clip(actions, -5.0, 5.0)
            
            # Calculate log probabilities with NaN protection
            log_probs = action_distribution.log_prob(actions).sum(axis=-1)
            log_probs = jnp.where(jnp.isnan(log_probs), -10.0, log_probs)  # Replace NaN with large negative value
            log_probs = jnp.clip(log_probs, -50.0, 10.0)  # Clip to reasonable range
        
            # Step environment
            new_env_states, next_obs, rewards, dones, info = self.vmap_step(env_states, actions)

            new_carry_tuple = new_carry_list[0]
            # Handle LSTM state resets on episode boundaries using vmap
            def reset_carry_fn(layer_carry, done):
                h_state = layer_carry[0]
                c_state = layer_carry[1]
                h_zeros = jnp.zeros_like(h_state)
                c_zeros = jnp.zeros_like(c_state)
                reset_h = jnp.where(done, h_zeros, h_state)
                reset_c = jnp.where(done, c_zeros, c_state)
                return LSTMCarry(h=reset_h, c=reset_c)

            # Vectorize the reset function over the list of LSTM layers
            reset_carry_list = jax.vmap(reset_carry_fn, in_axes=(0, 0))(new_carry_tuple, dones)
        
            # Stack LSTM carry for trajectory storage
            initial_lstm_h_stacked = jnp.stack([c.h for c in lstm_carry_tuple], axis=1)
            initial_lstm_c_stacked = jnp.stack([c.c for c in lstm_carry_tuple], axis=1)
        
            transition = Trajectory(
                obs=obs,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                dones=dones,
                initial_lstm_h=initial_lstm_h_stacked,
                initial_lstm_c=initial_lstm_c_stacked
            )
        
            return (new_env_states, next_obs, (reset_carry_list, ), rng_key), transition
    
        # Roll out trajectory
        n_steps = self.config['n_steps']
        init_carry_scan = (env_states, initial_obs, tuple(initial_carry), rng_key)
    
        final_carry_scan, trajectory = lax.scan(step_fn, init_carry_scan, None, length=n_steps)
    
        final_env_states, final_obs, final_lstm_carry_tuple, _ = final_carry_scan
    
        return trajectory, final_env_states, final_obs, list(final_lstm_carry_tuple)
        
    @partial(jax.jit, static_argnums=(0,))
    def compute_gae(self, trajectory: Trajectory, last_values: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Compute Generalized Advantage Estimation with NaN protection"""
        gamma = self.config['gamma']
        gae_lambda = self.config['gae_lambda']
        
        # Clean inputs for NaN values
        rewards = jnp.where(jnp.isnan(trajectory.rewards), 0.0, trajectory.rewards)
        values = jnp.where(jnp.isnan(trajectory.values), 0.0, trajectory.values)
        last_values = jnp.where(jnp.isnan(last_values), 0.0, last_values)
        
        # Add last values to the end of the trajectory values for GAE calculation
        # Values are (n_steps, n_envs)
        # last_values is (n_envs,)
        extended_values = jnp.concatenate([values, last_values[None, :]], axis=0) # (n_steps + 1, n_envs)
        
        def gae_step(gae_and_advantage, inputs):
            current_reward, current_value, next_value, current_done = inputs
            
            # The 'gae' in the carry is the GAE from the previous step (t+1)
            # The 'advantage' in the carry is the list of advantages collected so far
            current_gae, _ = gae_and_advantage # We only need the current_gae for calculation
            
            # Delta_t = R_t + gamma * V(S_{t+1}) * (1 - D_t) - V(S_t)
            delta = current_reward + gamma * next_value * (1 - current_done) - current_value
            
            # A_t = delta_t + gamma * lambda * (1 - D_t) * A_{t+1}
            current_gae = delta + gamma * gae_lambda * (1 - current_done) * current_gae
            
            # Check for NaN in GAE calculation and reset if found
            current_gae = jnp.where(jnp.isnan(current_gae), 0.0, current_gae)
            current_gae = jnp.clip(current_gae, -100.0, 100.0)  # Clip to prevent extreme values
            
            return (current_gae, None), current_gae # We return current_gae to be the new carry's gae
        
        # Prepare inputs for scan: (reward, value_t, value_{t+1}, done)
        # We need to process in reverse order for GAE
        # rewards: (n_steps, n_envs)
        # values: (n_steps, n_envs)
        # extended_values: (n_steps + 1, n_envs)
        # trajectory.dones: (n_steps, n_envs)
        
        # Inputs for gae_step: (rewards, values_t, values_{t+1}, dones)
        gae_inputs = (
            rewards,
            values,
            extended_values[1:], # next_values
            trajectory.dones
        )
        
        # Initial GAE is 0 for all environments
        # For scan, we need to pass a tuple that represents the carry state.
        # The first element of the carry will be the running GAE, the second (None) is a placeholder.
        init_gae_carry = (jnp.zeros_like(last_values), None) 
        
        # lax.scan processes from left to right. For GAE, we need to go backwards.
        # So we reverse the inputs, and the outputs will be in reversed order.
        (_, _), advantages = lax.scan(
            gae_step, 
            init_gae_carry, 
            jax.tree_util.tree_map(lambda x: x[::-1], gae_inputs), # Reverse all inputs
            length=self.config['n_steps']
        )
        
        advantages = advantages[::-1] # Reverse advantages back to original order
        
        # Final NaN check and clipping
        advantages = jnp.where(jnp.isnan(advantages), 0.0, advantages)
        advantages = jnp.clip(advantages, -100.0, 100.0)
        
        # Target = Advantages + Values
        returns = advantages + values
        returns = jnp.where(jnp.isnan(returns), 0.0, returns)
        returns = jnp.clip(returns, -100.0, 100.0)
        
        return advantages, returns

    @partial(jax.jit, static_argnums=(0,))
    def ppo_loss(self, params: chex.Array, train_batch: Trajectory, gae_advantages: chex.Array, returns: chex.Array, rng_key: chex.PRNGKey):
        """Compute PPO loss for a given batch with NaN protection"""
        
        # Unpack batch data and clean for NaN values
        obs_batch = jnp.where(jnp.isnan(train_batch.obs), 0.0, train_batch.obs) # (batch_size, obs_dim)
        actions_batch = jnp.where(jnp.isnan(train_batch.actions), 0.0, train_batch.actions) # (batch_size, action_dim)
        old_log_probs_batch = jnp.where(jnp.isnan(train_batch.log_probs), -10.0, train_batch.log_probs) # (batch_size,)
        # Stacked initial LSTM carries for this batch (batch_size, n_lstm_layers, hidden_size)
        initial_lstm_h_batch = jnp.where(jnp.isnan(train_batch.initial_lstm_h), 0.0, train_batch.initial_lstm_h)
        initial_lstm_c_batch = jnp.where(jnp.isnan(train_batch.initial_lstm_c), 0.0, train_batch.initial_lstm_c)

        # Reconstruct LSTM carry from stacked h and c
        lstm_carry_batch = tuple([
            LSTMCarry(h=initial_lstm_h_batch[:, i, :], c=initial_lstm_c_batch[:, i, :])
            for i in range(self.config['n_lstm_layers'])
        ])
        
        # Get current policy outputs
        logits, values, _ = self.network.apply(params, obs_batch, tuple(lstm_carry_batch))
        
        # Clean network outputs for NaN values
        logits = jnp.where(jnp.isnan(logits), 0.0, logits)
        values = jnp.where(jnp.isnan(values), 0.0, values)
        logits = jnp.clip(logits, -10.0, 10.0)
        values = jnp.clip(values, -100.0, 100.0)
        
        # Clean returns and advantages
        returns = jnp.where(jnp.isnan(returns), 0.0, returns)
        gae_advantages = jnp.where(jnp.isnan(gae_advantages), 0.0, gae_advantages)
        returns = jnp.clip(returns, -100.0, 100.0)
        gae_advantages = jnp.clip(gae_advantages, -100.0, 100.0)
        
        # Value loss with NaN protection
        old_values = jnp.where(jnp.isnan(train_batch.values), 0.0, train_batch.values)
        old_values = jnp.clip(old_values, -100.0, 100.0)
        
        value_pred_clipped = old_values + (values - old_values).clip(-self.config['clip_eps'], self.config['clip_eps'])
        value_losses = jnp.square(values - returns)
        value_losses_clipped = jnp.square(value_pred_clipped - returns)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        
        # Policy loss with NaN protection
        action_std = jnp.maximum(self.config['action_std'], 1e-6)  # Ensure positive std
        action_distribution = distrax.Normal(loc=logits, scale=action_std)
        new_log_probs = action_distribution.log_prob(actions_batch).sum(axis=-1)
        new_log_probs = jnp.where(jnp.isnan(new_log_probs), -10.0, new_log_probs)
        new_log_probs = jnp.clip(new_log_probs, -50.0, 10.0)
        
        # Calculate ratio with numerical stability
        log_ratio = new_log_probs - old_log_probs_batch
        log_ratio = jnp.clip(log_ratio, -10.0, 10.0)  # Prevent extreme values
        ratio = jnp.exp(log_ratio)
        ratio = jnp.clip(ratio, 0.0, 10.0)  # Prevent extreme ratios
        
        # Normalize advantages for stability
        gae_advantages = (gae_advantages - gae_advantages.mean()) / (jnp.std(gae_advantages) + 1e-8)
        gae_advantages = jnp.clip(gae_advantages, -10.0, 10.0)
        
        pg_losses1 = ratio * gae_advantages
        pg_losses2 = jnp.clip(ratio, 1.0 - self.config['clip_eps'], 1.0 + self.config['clip_eps']) * gae_advantages
        policy_loss = -jnp.minimum(pg_losses1, pg_losses2).mean()
        
        # Entropy loss with NaN protection
        entropy = action_distribution.entropy().sum(axis=-1)
        entropy = jnp.where(jnp.isnan(entropy), 0.0, entropy)
        entropy = jnp.clip(entropy, 0.0, 10.0)
        entropy_loss = -self.config['entropy_coeff'] * entropy.mean()
        
        # Calculate total loss with NaN protection
        total_loss = policy_loss + value_loss * self.config['value_coeff'] + entropy_loss
        total_loss = jnp.where(jnp.isnan(total_loss), 0.0, total_loss)
        
        # Calculate metrics with NaN protection
        approx_kl = (ratio - 1) - jnp.log(ratio + 1e-8)
        approx_kl = jnp.where(jnp.isnan(approx_kl), 0.0, approx_kl)
        approx_kl = jnp.clip(approx_kl, -10.0, 10.0)
        
        clip_fraction = (jnp.abs(ratio - 1.0) > self.config['clip_eps']).astype(jnp.float32)
        
        metrics = {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'approx_kl': approx_kl.mean(),
            'clip_fraction': clip_fraction.mean()
        }
        
        return total_loss, metrics

    def check_and_reset_nan_params(self, train_state: TrainState, rng_key: chex.PRNGKey) -> Tuple[TrainState, chex.PRNGKey]:
        """Check for NaN values in parameters and reset if necessary"""
        def has_nan_params(params):
            return jax.tree_util.tree_reduce(
                lambda acc, x: acc | jnp.any(jnp.isnan(x)) if jnp.issubdtype(x.dtype, jnp.floating) else acc,
                params, False
            )
        
        if has_nan_params(train_state.params):
            print(f"WARNING: NaN detected in parameters at step {self.nan_count}. Resetting parameters...")
            self.nan_count += 1
            
            if self.nan_count > self.max_nan_resets:
                raise RuntimeError(f"Too many NaN resets ({self.nan_count}). Training stopped.")
            
            # Reinitialize parameters
            rng_key, init_rng = random.split(rng_key)
            dummy_obs = jnp.ones((self.config['n_envs'], self.env.obs_dim))
            dummy_carry = [
                LSTMCarry(
                    h=jnp.zeros((self.config['n_envs'], self.config['hidden_size'])),
                    c=jnp.zeros((self.config['n_envs'], self.config['hidden_size']))
                ) for _ in range(self.config['n_lstm_layers'])
            ]
            new_params = self.network.init(init_rng, dummy_obs, tuple(dummy_carry))
            
            # Create new train state with reset parameters
            train_state = train_state.replace(params=new_params)
            print("Parameters reset successfully.")
        
        return train_state, rng_key

    def debug_nan_sources(self, obs, actions, rewards, values, logits):
        """Debug function to identify NaN sources (non-JIT)"""
        print("=== NaN Debugging ===")
        
        # Check observations
        if jnp.any(jnp.isnan(obs)):
            print("❌ NaN detected in observations")
            print(f"   NaN count: {jnp.sum(jnp.isnan(obs))}")
        else:
            print("✅ Observations are clean")
        
        # Check actions
        if jnp.any(jnp.isnan(actions)):
            print("❌ NaN detected in actions")
            print(f"   NaN count: {jnp.sum(jnp.isnan(actions))}")
        else:
            print("✅ Actions are clean")
        
        # Check rewards
        if jnp.any(jnp.isnan(rewards)):
            print("❌ NaN detected in rewards")
            print(f"   NaN count: {jnp.sum(jnp.isnan(rewards))}")
        else:
            print("✅ Rewards are clean")
        
        # Check values
        if jnp.any(jnp.isnan(values)):
            print("❌ NaN detected in values")
            print(f"   NaN count: {jnp.sum(jnp.isnan(values))}")
        else:
            print("✅ Values are clean")
        
        # Check logits
        if jnp.any(jnp.isnan(logits)):
            print("❌ NaN detected in logits")
            print(f"   NaN count: {jnp.sum(jnp.isnan(logits))}")
        else:
            print("✅ Logits are clean")
        
        print("===================")

    @partial(jax.jit, static_argnums=(0,5))
    def train_step(self, train_state: TrainState, trajectory: Trajectory, last_values: chex.Array, rng_key: chex.PRNGKey, num_minibatches: int) -> Tuple[TrainState, Dict[str, Any]]:
        """Perform one PPO training step (multiple epochs over the trajectory)"""
        
        advantages, returns = self.compute_gae(trajectory, last_values)
        
        # Flatten trajectory data from (n_steps, n_envs, ...) to (n_steps * n_envs, ...)
        # Need to handle namedtuple flattening
        flat_trajectory = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), trajectory
        )
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        
        total_loss_sum = jnp.array(0.0)
        policy_loss_sum = jnp.array(0.0)
        value_loss_sum = jnp.array(0.0)
        entropy_loss_sum = jnp.array(0.0)
        approx_kl_sum = jnp.array(0.0)
        clip_fraction_sum = jnp.array(0.0)
        
        # Shuffle indices for mini-batch sampling
        rng_key, shuffle_rng = random.split(rng_key)
        permutation = random.permutation(shuffle_rng, flat_trajectory.obs.shape[0])
        
        shuffled_flat_trajectory = jax.tree_util.tree_map(
            lambda x: x[permutation], flat_trajectory
        )
        shuffled_flat_advantages = flat_advantages[permutation]
        shuffled_flat_returns = flat_returns[permutation]
        
        # Define a single PPO epoch function
        def ppo_epoch(carry, i):
            current_train_state, current_metrics_sums, current_rng_key = carry
            batch_size = self.config['ppo_batch_size']
            # Extract mini-batch
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            
            # Use dynamic_slice to handle batching within JIT
            mini_batch_trajectory = jax.tree_util.tree_map(
                lambda x: lax.dynamic_slice(x, (start_idx,) + (0,) * (x.ndim - 1), (batch_size,) + x.shape[1:]),
                shuffled_flat_trajectory
            )
            mini_batch_advantages = lax.dynamic_slice(shuffled_flat_advantages, (start_idx,), (batch_size,))
            mini_batch_returns = lax.dynamic_slice(shuffled_flat_returns, (start_idx,), (batch_size,))
            
            # Compute gradients and update train state
            grad_fn = jax.value_and_grad(self.ppo_loss, has_aux=True)
            (loss, metrics), grads = grad_fn(current_train_state.params, mini_batch_trajectory, mini_batch_advantages, mini_batch_returns, current_rng_key)
            current_train_state = current_train_state.apply_gradients(grads=grads)
            
            # Accumulate metrics
            current_metrics_sums = jax.tree_util.tree_map(lambda s, m: s + m, current_metrics_sums, metrics)
            
            return (current_train_state, current_metrics_sums, current_rng_key), None
        
        # Initial metrics sum (all zeros)
        init_metrics_sums = {k: jnp.array(0.0) for k in self.ppo_loss(train_state.params, flat_trajectory, flat_advantages, flat_returns, rng_key)[1].keys()}
        
        # Loop for PPO epochs
        for _ in range(self.config['ppo_epochs']):
            (train_state, init_metrics_sums, rng_key), _ = lax.scan(
                ppo_epoch,
                (train_state, init_metrics_sums, rng_key),
                jnp.arange(num_minibatches)
            )
            
        # Average metrics
        avg_metrics = jax.tree_util.tree_map(lambda s: s / (self.config['ppo_epochs'] * num_minibatches), init_metrics_sums)
        
        return train_state, avg_metrics

    def train(self):
        """Main training loop"""
        
        print("Starting PPO training...")

        num_minibatches = self.config['n_steps'] * self.config['n_envs'] // self.config['ppo_batch_size']
        
        for update in range(self.config['num_updates']):
            start_time = time.time()
            self.rng, collect_rng = random.split(self.rng)
            
            # Collect trajectory
            trajectory, self.env_states, self.obs, self.collector_carry = self.collect_trajectory(
                self.train_state, 
                self.env_states, 
                self.obs, 
                self.collector_carry, 
                collect_rng
            )
            
            # Debug NaN sources in trajectory
            if update % 5 == 0:  # Debug every 5 updates
                # Get network outputs for debugging
                _, last_values, _ = self.train_state.apply_fn(self.train_state.params, self.obs, tuple(self.collector_carry))
                
                self.debug_nan_sources(
                    trajectory.obs[0],  # First step observations
                    trajectory.actions[0],  # First step actions
                    trajectory.rewards[0],  # First step rewards
                    trajectory.values[0],  # First step values
                    last_values  # Network values output
                )
            
            # Get last values (for GAE)
            # Make sure to pass the carry of the *last* step to predict the last_values
            # The obs are the ones *after* the last step of the trajectory.
            _, last_values, _ = self.train_state.apply_fn(self.train_state.params, self.obs, tuple(self.collector_carry))

            
            
            # Check for NaN values before training
            self.train_state, self.rng = self.check_and_reset_nan_params(self.train_state, self.rng)
            
            # Perform PPO training step
            self.rng, train_rng = random.split(self.rng)
            self.train_state, metrics = self.train_step(self.train_state, trajectory, last_values, train_rng, num_minibatches)
            
            # Check for NaN values after training
            self.train_state, self.rng = self.check_and_reset_nan_params(self.train_state, self.rng)
            
            end_time = time.time()
            
            # Logging
            if update % self.config['log_interval'] == 0:
                print(f"Update {update}/{self.config['num_updates']} | Time: {end_time - start_time:.2f}s")
                print(f"  Total Loss: {metrics['total_loss']:.4f}, Policy Loss: {metrics['policy_loss']:.4f}, Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Avg Reward: {trajectory.rewards.mean():.4f}, Max Return: {trajectory.rewards.sum(axis=0).max():.4f}")
                
                # Log to wandb
                if self.config.get('use_wandb', True):
                    wandb_log = {
                        "charts/learning_rate": self.config['learning_rate'], # if using fixed LR
                        "losses/total_loss": metrics['total_loss'],
                        "losses/policy_loss": metrics['policy_loss'],
                        "losses/value_loss": metrics['value_loss'],
                        "losses/entropy_loss": metrics['entropy_loss'],
                        "losses/approx_kl": metrics['approx_kl'],
                        "losses/clip_fraction": metrics['clip_fraction'],
                        "rollout/avg_reward": trajectory.rewards.mean(),
                        "rollout/max_reward": trajectory.rewards.max(),
                        "rollout/min_reward": trajectory.rewards.min(),
                        "rollout/avg_episode_return": trajectory.rewards.sum(axis=0).mean(),
                        "rollout/max_episode_return": trajectory.rewards.sum(axis=0).max(),
                        "rollout/avg_portfolio_value": self.env_states.portfolio_value.mean(), # Access from final env_states
                        "global_step": update
                    }
                    # Also log the portfolio values and sharpe ratios from the last step's info
                    # The `info` dict from `step` contains 'portfolio_values', 'sharpe_ratios', etc.
                    # You'd need to extend `collect_trajectory` to return `all_infos` or derive from `env_states`.
                    # For simplicity, let's assume we can get some from `self.env_states` which are the final states.
                    # For logging portfolio value and sharpe ratio from the final state of each env
                    wandb_log['final_env/avg_portfolio_value'] = self.env_states.portfolio_value.mean()
                    # Note: Sharpe calculation needs to happen on the full buffer, not just its mean
                    # A more accurate Sharpe calculation would be needed here, e.g., if env_states has info
                    # sharpe_means = jnp.mean(self.env_states.sharpe_buffer, axis=1)
                    # sharpe_stds = jnp.std(self.env_states.sharpe_buffer, axis=1)
                    # avg_sharpe = jnp.mean((sharpe_means - self.env.risk_free_rate_daily) / (sharpe_stds + 1e-8) * jnp.sqrt(252.0))
                    # wandb_log['final_env/avg_sharpe_ratio'] = avg_sharpe
                    wandb_log['final_env/avg_sharpe_ratio'] = jnp.array(0.0) # Placeholder for now to avoid error
                    
                    wandb.log(wandb_log)
            
            # Save model
            if update % self.config['save_interval'] == 0 and update > 0:
                self.save_model(f"ppo_model_update_{update}")
                
        print("Training complete!")
        if self.config.get('use_wandb', True):
            wandb.finish()

    def save_model(self, filename: str):
        """Save the training state (parameters and optimizer state)"""
        save_path = Path(self.config['model_dir']) / filename
        save_path = save_path.resolve()
        os.makedirs(save_path.parent, exist_ok=True)
        state_dict = serialization.to_state_dict(self.train_state)
        with open(save_path.with_suffix('.json'), 'w') as f:
            json_safe_dict = jax.tree_util.tree_map(
                lambda x: x.tolist() if hasattr(x, 'tolist') else x,
                state_dict
            )
            json.dump(json_safe_dict, f)
        print(f"Model saved to {save_path}")

    def load_model(self, filename: str):
        """Load a saved training state"""
        load_path = Path(self.config['model_dir']) / filename
        with open(load_path, 'rb') as f:
            self.train_state = pickle.load(f)
        print(f"Model loaded from {load_path}")

if __name__ == "__main__":
    # Example Configuration
    config = {
        'seed': 0,
        'data_root': 'processed_data/',
        'stocks': None, 
        'train_start_date': '2024-06-06', # Extended for more data
        'train_end_date': '2025-03-06',
        'window_size': 30,
        'transaction_cost_rate': 0.005,
        'sharpe_window': 252,
        'n_envs' : 16,
        
        # PPO parameters
        'num_updates': 100000, # Number of PPO updates
        'n_steps': 128,      # Number of steps to collect per environment
        'gamma': 0.99,       # Discount factor
        'gae_lambda': 0.95,  # GAE lambda parameter
        'clip_eps': 0.2,     # PPO clipping epsilon
        'ppo_epochs': 4,     # Number of PPO epochs per update
        'ppo_batch_size': 256, # Mini-batch size for PPO
        'learning_rate': 1e-4,  # Reduced learning rate for stability
        'max_grad_norm': 0.5,
        'value_coeff': 0.5,
        'entropy_coeff': 0.01,
        'action_std': 1.0,  # Increased action std for better exploration
        
        # LSTM parameters
        'hidden_size': 256, # Reduced for initial testing
        'n_lstm_layers': 1, # Reduced for initial testing
        
        # Logging and saving
        'use_wandb': True,
        'log_interval': 10,
        'save_interval': 50,
        'model_dir': 'models',
    }
    
    # Create and run the trainer
    trainer = PPOTrainer(config)
    trainer.train()

    # !git add .
    # !git commit -m "Trained PPO LSTM model"
    # !git push origin gpu-training-scripts
