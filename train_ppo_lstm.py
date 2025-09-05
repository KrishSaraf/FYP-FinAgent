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
from functools import partial

# Import the JAX environment (assume it's in the same directory or installed)
from jax_portfolio_env import JAXPortfolioEnvWrapper, EnvState

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
    def __call__(self, x, carry: Optional[List[LSTMCarry]] = None, training=True):
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
            if carry is None or len(carry) <= i or carry[i] is None:
                # Initialize carry if None for this layer or not provided
                h = jnp.zeros((batch_size, self.hidden_size))
                c = jnp.zeros((batch_size, self.hidden_size))
                layer_carry = LSTMCarry(h=h, c=c)
            else:
                layer_carry = carry[i]
            
            # LSTM cell
            lstm_cell = nn.OptimizedLSTMCell()
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
            'n_envs': config['n_envs'],
            'window_size': config['window_size'],
            'transaction_cost': config['transaction_cost'],
            'sharpe_window': config['sharpe_window']
        }
        
        self.env = JAXPortfolioEnvWrapper(env_config)
        
        # Network initialization
        self.network = LSTMActorCritic(
            action_dim=self.env.env.action_dim,
            hidden_size=config['hidden_size'],
            n_lstm_layers=config['n_lstm_layers']
        )
        
        # Initialize network parameters
        self.rng = random.PRNGKey(config['seed'])
        self.rng, init_rng = random.split(self.rng)
        
        dummy_obs = jnp.ones((config['n_envs'], self.env.env.obs_dim))
        
        # Initialize LSTM carry for dummy init
        dummy_carry = [
            LSTMCarry(
                h=jnp.zeros((config['n_envs'], config['hidden_size'])),
                c=jnp.zeros((config['n_envs'], config['hidden_size']))
            ) for _ in range(config['n_lstm_layers'])
        ]
        self.params = self.network.init(init_rng, dummy_obs, dummy_carry)
        
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
            
        self.env_states, self.obs = self.env.reset()
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
            env_states, obs, lstm_carry, rng_key = carry_step
            
            # Get action from policy
            rng_key, action_rng = random.split(rng_key)
            
            # Apply network
            # Ensure lstm_carry is treated as a list of LSTMCarry, even if it's the initial one (could be None initially if not handled in network init)
            logits, values, new_carry_list = train_state.apply_fn(train_state.params, obs, lstm_carry)
            
            # Sample actions
            # For continuous actions, you'd typically sample from a distribution (e.g., Normal)
            # For discrete actions (as implied by random.categorical), ensure the action space aligns.
            # Here, we assume continuous actions are directly the logits, which are then softmax-normalized by the env.
            # We'll treat `logits` as direct actions or the means of a distribution.
            # Let's assume for now `actions` are directly the continuous values passed to the env.
            # For PPO with continuous actions, we often output mean and std.
            # Given the environment expects weights (softmax-normalized), let's assume logits are the raw outputs.
            # The environment will then softmax these logits.
            
            # Let's assume actions are the raw logits (which will be softmaxed by the env)
            actions = logits # Pass raw logits, env will softmax
            
            # For log_probs, we need to calculate them based on the *actual* actions taken.
            # If the env softmaxes, the agent isn't directly choosing discrete categories.
            # For PPO with continuous actions:
            # - Policy outputs mean and log_std (or std).
            # - Actions are sampled from N(mean, std).
            # - Log_probs are then calculated for those sampled actions.
            # Given the current setup, `logits` are *not* directly probabilities.
            # Let's *re-interpret* `actions` as continuous outputs that the env then processes.
            # For PPO on continuous actions, it's common to use a TanhNormal or directly output mean/log_std for Normal.
            # For simplicity, if `actions` are directly continuous, and the environment uses softmax,
            # then `log_probs` would refer to the log-likelihood of the continuous action.
            # A common approach for this is to use a Beta distribution or Normal distribution
            # and then apply a squash function like Tanh.
            # For now, let's assume `logits` are the raw outputs, and we need to define how `actions` and `log_probs` are derived.
            # If the environment `step` function applies `jax.nn.softmax(action)`, then `action` here are the unnormalized log-probabilities.
            # A common approach for continuous actions is to sample from a Gaussian parameterized by the actor output.
            # For simplicity for now, let's assume 'logits' are the direct continuous action values.
            # The PPO formulation usually works with a policy that outputs distribution parameters.
            # Let's make an explicit choice: `logits` are the means of a Gaussian policy, and we'll use a fixed `action_std`.
            # This is a simplification. A more robust PPO would have the network also output `log_std`.
            
            # Simplified continuous action sampling for now:
            # We'll use the logits as means and a fixed standard deviation
            action_std = self.config['action_std'] # Add action_std to config
            action_distribution = jax.scipy.stats.norm(loc=logits, scale=action_std)
            actions = action_distribution.sample(key=action_rng)
            
            # Clip actions to reasonable range if necessary before environment
            actions = jnp.clip(actions, -5.0, 5.0) # Example clipping
            
            log_probs = action_distribution.logpdf(actions).sum(axis=-1) # Sum log_probs across action_dim
            
            # Step environment
            new_env_states, next_obs, rewards, dones, info = self.env.batch_step(env_states, actions)
            
            # Handle LSTM state resets on episode boundaries
            # new_carry_list is a list of LSTMCarry objects
            reset_carry_list = []
            for i, layer_carry in enumerate(new_carry_list):
                batch_size_h = layer_carry.h.shape[0]
                h_zeros = jnp.zeros((batch_size_h, self.config['hidden_size']))
                c_zeros = jnp.zeros((batch_size_h, self.config['hidden_size']))
                zero_carry_layer = LSTMCarry(h=h_zeros, c=c_zeros)
                
                reset_h = jnp.where(dones[:, None], zero_carry_layer.h, layer_carry.h)
                reset_c = jnp.where(dones[:, None], zero_carry_layer.c, layer_carry.c)
                reset_carry_list.append(LSTMCarry(h=reset_h, c=reset_c))

            # Stack LSTM carry for trajectory storage
            # (n_lstm_layers, n_envs, hidden_size) -> (n_envs, n_lstm_layers, hidden_size)
            # Need to store the *initial* carry of the step, not the *new* carry for the transition.
            # So, `lstm_carry` are the states used to produce `logits, values`
            initial_lstm_h_stacked = jnp.stack([c.h for c in lstm_carry], axis=1) # (n_envs, n_lstm_layers, hidden_size)
            initial_lstm_c_stacked = jnp.stack([c.c for c in lstm_carry], axis=1) # (n_envs, n_lstm_layers, hidden_size)
            
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
            
            return (new_env_states, next_obs, reset_carry_list, rng_key), transition
        
        # Roll out trajectory
        n_steps = self.config['n_steps']
        init_carry_scan = (env_states, initial_obs, initial_carry, rng_key)
        
        # trajectory will be a Trajectory object where each field has shape (n_steps, n_envs, ...)
        final_carry_scan, trajectory = lax.scan(step_fn, init_carry_scan, None, length=n_steps)
        
        final_env_states, final_obs, final_lstm_carry, _ = final_carry_scan
        
        return trajectory, final_env_states, final_obs, final_lstm_carry
    
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
        lstm_carry_batch = [
            LSTMCarry(h=initial_lstm_h_batch[:, i, :], c=initial_lstm_c_batch[:, i, :])
            for i in range(self.config['n_lstm_layers'])
        ]
        
        # Get current policy outputs
        logits, values, _ = self.network.apply(params, obs_batch, lstm_carry_batch)
        
        # Value loss
        value_pred_clipped = train_batch.values + (values - train_batch.values).clip(-self.config['clip_eps'], self.config['clip_eps'])
        value_losses = jnp.square(values - returns)
        value_losses_clipped = jnp.square(value_pred_clipped - returns)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        
        # Policy loss
        # Use the same continuous action distribution assumption as in collect_trajectory
        action_std = self.config['action_std']
        action_distribution = jax.scipy.stats.norm(loc=logits, scale=action_std)
        new_log_probs = action_distribution.logpdf(actions_batch).sum(axis=-1)
        
        ratio = jnp.exp(new_log_probs - old_log_probs_batch)
        
        # Normalize advantages (optional, but common)
        # gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)
        
        pg_losses1 = ratio * gae_advantages
        pg_losses2 = jnp.clip(ratio, 1.0 - self.config['clip_eps'], 1.0 + self.config['clip_eps']) * gae_advantages
        policy_loss = -jnp.minimum(pg_losses1, pg_losses2).mean()
        
        # Entropy loss
        # For continuous actions, entropy of Gaussian: 0.5 * log(2*pi*e*sigma^2)
        # Sum across action dimensions
        entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * jnp.square(action_std)) * self.network.action_dim
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

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, train_state: TrainState, trajectory: Trajectory, last_values: chex.Array, rng_key: chex.PRNGKey) -> Tuple[TrainState, Dict[str, Any]]:
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
        
        batch_size = self.config['ppo_batch_size']
        num_minibatches = flat_trajectory.obs.shape[0] // batch_size
        
        # Ensure num_minibatches is at least 1, even if flat_trajectory.obs.shape[0] < batch_size
        num_minibatches = jnp.maximum(1, num_minibatches)
        
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
            
            # Extract mini-batch
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            
            # Use dynamic_slice to handle batching within JIT
            mini_batch_trajectory = jax.tree_util.tree_map(
                lambda x: lax.dynamic_slice(x, (start_idx, 0) if x.ndim > 1 else (start_idx,), (batch_size, x.shape[1]) if x.ndim > 1 else (batch_size,)),
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
            _, last_values, _ = self.train_state.apply_fn(self.train_state.params, self.obs, self.collector_carry)
            
            # Perform PPO training step
            self.rng, train_rng = random.split(self.rng)
            self.train_state, metrics = self.train_step(self.train_state, trajectory, last_values, train_rng)
            
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
                    if isinstance(self.env_states, EnvState): # Check if EnvState is a NamedTuple/single instance
                        wandb_log['final_env/avg_portfolio_value'] = self.env_states.portfolio_value.mean()
                        wandb_log['final_env/avg_sharpe_ratio'] = self.env_states.sharpe_buffer.mean() # This is crude, better calculate from buffer
                    
                    wandb.log(wandb_log)
            
            # Save model
            if update % self.config['save_interval'] == 0 and update > 0:
                self.save_model(f"ppo_model_update_{update}.pkl")
                
        print("Training complete!")
        if self.config.get('use_wandb', True):
            wandb.finish()

    def save_model(self, filename: str):
        """Save the training state (parameters and optimizer state)"""
        save_path = Path(self.config['model_dir']) / filename
        os.makedirs(save_path.parent, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.train_state, f)
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
        'train_start_date': '2010-01-01', # Extended for more data
        'train_end_date': '2020-12-31', 
        'n_envs': 16, # Reduced for initial testing on Colab
        'window_size': 30,
        'transaction_cost': 0.001,
        'sharpe_window': 252,
        
        # PPO parameters
        'num_updates': 1000, # Number of PPO updates
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
        'action_std': 0.5, # Fixed standard deviation for continuous action sampling (simplified)
                           # For more robust PPO, the network should learn this.
        
        # LSTM parameters
        'hidden_size': 256, # Reduced for initial testing
        'n_lstm_layers': 1, # Reduced for initial testing
        
        # Logging and saving
        'use_wandb': True,
        'log_interval': 10,
        'save_interval': 100,
        'model_dir': 'ppo_models',
    }
    
    # Create and run the trainer
    trainer = PPOTrainer(config)
    trainer.train()

    !git add .
    !git commit -m "Trained PPO LSTM model"
    !git push origin gpu-training-scripts
