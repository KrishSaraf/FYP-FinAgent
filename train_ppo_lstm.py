import os
import time
import jax
import jax.numpy as jnp
from jax import random, vmap, lax
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
import chex
from typing import Tuple, Dict, Any, NamedTuple, List
import wandb
import pickle
import distrax
from pathlib import Path
from functools import partial
import orbax.checkpoint as ocp

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
    
    @nn.compact
    def __call__(self, x, carry: List[LSTMCarry], training=True):
        batch_size = x.shape[0]
        
        # Input preprocessing
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x) 
        x = nn.relu(x)
        
        # LSTM layers
        lstm_input = x
        new_carry = []
        
        for i in range(self.n_lstm_layers):
            if carry[i] is None:
                # Initialize carry if None for this layer or not provided
                h = jnp.zeros((batch_size, self.hidden_size), dtype = jnp.float32)
                c = jnp.zeros((batch_size, self.hidden_size), dtype = jnp.float32)
                layer_carry = LSTMCarry(h=h, c=c)
            else:
                layer_carry = carry[i]
            
            # LSTM cell
            lstm_cell = nn.OptimizedLSTMCell(features=self.hidden_size)
            new_h_val, lstm_input = lstm_cell(layer_carry, lstm_input) # new_h_val is the new LSTMCarry, lstm_input is the output
            new_carry.append(new_h_val)
        
        # Actor head (policy)
        actor_hidden = nn.Dense(256)(lstm_input)
        actor_hidden = nn.relu(actor_hidden)
        logits = nn.Dense(self.action_dim)(actor_hidden)
        
        # Critic head (value function)
        critic_hidden = nn.Dense(256)(lstm_input)
        critic_hidden = nn.relu(critic_hidden)
        values = nn.Dense(1)(critic_hidden).squeeze(-1)
        
        return logits, values, new_carry

class PPOTrainer:
    def __init__(self, config: dict):
        self.config = config
        
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
        self.params = self.network.init(init_rng, dummy_obs, tuple(dummy_carry))
        
        # Optimizer with gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']),
            optax.adam(learning_rate=config['learning_rate'], eps=1e-5) # Changed to optax.adam for common use, was adamw
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
        
            # Sample actions from the distribution
            action_std = self.config['action_std']
            action_distribution = distrax.Normal(loc=logits, scale=action_std)
            actions = action_distribution.sample(seed=action_rng)
            actions = jnp.clip(actions, -5.0, 5.0)
            log_probs = action_distribution.log_prob(actions).sum(axis=-1)
        
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
        """Compute Generalized Advantage Estimation"""
        gamma = self.config['gamma']
        gae_lambda = self.config['gae_lambda']
        
        # Add last values to the end of the trajectory values for GAE calculation
        # Values are (n_steps, n_envs)
        # last_values is (n_envs,)
        extended_values = jnp.concatenate([trajectory.values, last_values[None, :]], axis=0) # (n_steps + 1, n_envs)
        
        def gae_step(gae_and_advantage, inputs):
            current_reward, current_value, next_value, current_done = inputs
            
            # The 'gae' in the carry is the GAE from the previous step (t+1)
            # The 'advantage' in the carry is the list of advantages collected so far
            current_gae, _ = gae_and_advantage # We only need the current_gae for calculation
            
            # Delta_t = R_t + gamma * V(S_{t+1}) * (1 - D_t) - V(S_t)
            delta = current_reward + gamma * next_value * (1 - current_done) - current_value
            
            # A_t = delta_t + gamma * lambda * (1 - D_t) * A_{t+1}
            current_gae = delta + gamma * gae_lambda * (1 - current_done) * current_gae
            
            return (current_gae, None), current_gae # We return current_gae to be the new carry's gae
        
        # Prepare inputs for scan: (reward, value_t, value_{t+1}, done)
        # We need to process in reverse order for GAE
        # trajectory.rewards: (n_steps, n_envs)
        # trajectory.values: (n_steps, n_envs)
        # extended_values: (n_steps + 1, n_envs)
        # trajectory.dones: (n_steps, n_envs)
        
        # Inputs for gae_step: (rewards, values_t, values_{t+1}, dones)
        gae_inputs = (
            trajectory.rewards,
            trajectory.values,
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
        
        # Target = Advantages + Values
        returns = advantages + trajectory.values
        
        return advantages, returns

    @partial(jax.jit, static_argnums=(0,))
    def ppo_loss(self, params: chex.Array, train_batch: Trajectory, gae_advantages: chex.Array, returns: chex.Array, rng_key: chex.PRNGKey):
        """Compute PPO loss for a given batch"""
        
        # Unpack batch data
        obs_batch = train_batch.obs # (batch_size, obs_dim)
        actions_batch = train_batch.actions # (batch_size, action_dim)
        old_log_probs_batch = train_batch.log_probs # (batch_size,)
        # Stacked initial LSTM carries for this batch (batch_size, n_lstm_layers, hidden_size)
        initial_lstm_h_batch = train_batch.initial_lstm_h
        initial_lstm_c_batch = train_batch.initial_lstm_c

        # Reconstruct LSTM carry from stacked h and c
        lstm_carry_batch = tuple([
            LSTMCarry(h=initial_lstm_h_batch[:, i, :], c=initial_lstm_c_batch[:, i, :])
            for i in range(self.config['n_lstm_layers'])
        ])
        
        # Get current policy outputs
        logits, values, _ = self.network.apply(params, obs_batch, tuple(lstm_carry_batch))
        
        # Value loss
        value_pred_clipped = train_batch.values + (values - train_batch.values).clip(-self.config['clip_eps'], self.config['clip_eps'])
        value_losses = jnp.square(values - returns)
        value_losses_clipped = jnp.square(value_pred_clipped - returns)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        
        # Policy loss
        # Use the same continuous action distribution assumption as in collect_trajectory
        action_std = self.config['action_std']
        action_distribution = distrax.Normal(loc=logits, scale=action_std)
        new_log_probs = action_distribution.log_prob(actions_batch).sum(axis=-1)
        
        ratio = jnp.exp(new_log_probs - old_log_probs_batch)
        
        # Normalize advantages (optional, but common)
        # gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)
        
        pg_losses1 = ratio * gae_advantages
        pg_losses2 = jnp.clip(ratio, 1.0 - self.config['clip_eps'], 1.0 + self.config['clip_eps']) * gae_advantages
        policy_loss = -jnp.minimum(pg_losses1, pg_losses2).mean()
        
        # Entropy loss
        # For continuous actions, entropy of Gaussian: 0.5 * log(2*pi*e*sigma^2)
        # Sum across action dimensions
        entropy = action_distribution.entropy().sum(axis=-1)
        entropy_loss = -self.config['entropy_coeff'] * entropy.mean() # Maximize entropy, so negative coefficient
        
        total_loss = policy_loss + value_loss * self.config['value_coeff'] + entropy_loss
        
        metrics = {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'approx_kl': ((ratio - 1) - jnp.log(ratio)).mean(), # for debugging
            'clip_fraction': (jnp.abs(ratio - 1.0) > self.config['clip_eps']).mean()
        }
        
        return total_loss, metrics

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
            
            # Get last values (for GAE)
            # Make sure to pass the carry of the *last* step to predict the last_values
            # The obs are the ones *after* the last step of the trajectory.
            _, last_values, _ = self.train_state.apply_fn(self.train_state.params, self.obs, tuple(self.collector_carry))

            
            
            # Perform PPO training step
            self.rng, train_rng = random.split(self.rng)
            self.train_state, metrics = self.train_step(self.train_state, trajectory, last_values, train_rng, num_minibatches)
            
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
        os.makedirs(save_path.parent, exist_ok=True)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(save_path, self.train_state)
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
        'data_root': 'FYP-FinAgent/processed_data/',
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
        'learning_rate': 3e-4,
        'max_grad_norm': 0.5,
        'value_coeff': 0.5,
        'entropy_coeff': 0.01,
        'action_std': 0.5,
        
        # LSTM parameters
        'hidden_size': 256, # Reduced for initial testing
        'n_lstm_layers': 1, # Reduced for initial testing
        
        # Logging and saving
        'use_wandb': True,
        'log_interval': 10,
        'save_interval': 5000,
        'model_dir': 'models',
    }
    
    # Create and run the trainer
    trainer = PPOTrainer(config)
    trainer.train()

    # !git add .
    # !git commit -m "Trained PPO LSTM model"
    # !git push origin gpu-training-scripts
