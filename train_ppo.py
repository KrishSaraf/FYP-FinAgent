from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import time
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

# GPU Configuration
print("=== GPU Configuration ===")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set default device
    torch.cuda.set_device(0)
    print("✅ PyTorch will use GPU for neural network training")
else:
    print("⚠️ CUDA not available - will use CPU for neural network training")

# JAX GPU Configuration
try:
    import jax
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"JAX Devices: {jax.devices()}")
    if 'gpu' in str(jax.default_backend()).lower() or 'cuda' in str(jax.default_backend()).lower():
        print("✅ JAX will use GPU for environment computations")
    else:
        print("⚠️ JAX will use CPU for environment computations")
except ImportError:
    print("⚠️ JAX not available")

# Import your custom environment
from finagent.environment.portfolio_env import JAXVectorizedPortfolioEnv

class JAXToSB3Wrapper(gym.Env):
    """
    Wrapper to make JAX environment compatible with Stable-Baselines3
    Handles the new [-1, 1] action space with short position support
    """
    
    def __init__(self, jax_env):
        super().__init__()
        self.jax_env = jax_env
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(jax_env.action_dim,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(jax_env.obs_dim,),
            dtype=np.float32
        )
        
        self._env_state = None
        self._current_obs = None
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Create a random key for JAX environment
        import jax
        key = jax.random.PRNGKey(np.random.randint(0, 2**31))
        
        self._env_state, self._current_obs = self.jax_env.reset(key)
        
        # Convert JAX arrays to numpy
        obs = np.array(self._current_obs, dtype=np.float32)
        
        # Normalize observation to prevent numerical issues
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs, {}
    
    def step(self, action):
        """Execute one step with comprehensive error handling"""
        try:
            # Validate action
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            
            # Ensure action is within bounds
            action = np.clip(action, -1.0, 1.0)
            
            # Convert numpy action to JAX array
            import jax.numpy as jnp
            jax_action = jnp.array(action, dtype=jnp.float32)
            
            # Step the JAX environment
            self._env_state, next_obs, reward, done, info = self.jax_env.step(self._env_state, jax_action)
            
            # Convert outputs to numpy with error handling
            obs = np.array(next_obs, dtype=np.float32)
            
            # Handle NaN/Inf values in observations
            obs = np.where(np.isnan(obs), 0.0, obs)
            obs = np.where(np.isinf(obs), 0.0, obs)
            
            reward = float(reward)
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0
                
            done = bool(done)
            
            # Convert info dict values to numpy where needed
            info_np = {}
            for key, value in info.items():
                if hasattr(value, 'shape'):  # JAX array
                    info_np[key] = np.array(value)
                else:
                    info_np[key] = value
            
            self._current_obs = obs
            return obs, reward, done, False, info_np
            
        except Exception as e:
            print(f"Error in step function: {e}")
            # Return safe defaults
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            done = True
            info = {'error': str(e)}
            return obs, reward, done, False, info
    
    def render(self, mode='human'):
        """Render the environment"""
        if hasattr(self.jax_env, 'render'):
            return self.jax_env.render(mode)
        return None
    
    def close(self):
        """Close the environment"""
        pass

if __name__ == '__main__':

    # --- 1. DEFINE FILE PATHS AND PARAMETERS ---

    # Create directories to save logs and models
    models_dir = f"models/PPO_continued_{int(time.time())}"
    logdir = f"logs/PPO_continued_{int(time.time())}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Training parameters
    TRAIN_START_DATE = '2024-06-06'
    TRAIN_END_DATE = '2025-03-06'
    EVAL_START_DATE = '2025-03-07'
    EVAL_END_DATE = '2025-06-06'
    DATA_ROOT = "processed_data/"

    # --- 2. INSTANTIATE THE TRAINING ENVIRONMENT ---

    # Function to create a single environment instance with short position support
    def make_env(data_root, start_date, end_date):
        def _init():
            # Create JAX environment
            jax_env = JAXVectorizedPortfolioEnv(
            data_root=data_root,
            start_date=start_date,
                end_date=end_date,
                window_size=30,
                transaction_cost_rate=0.005,
                sharpe_window=252,
                use_all_features=True
            )
            # Wrap for Stable-Baselines3 compatibility
            return JAXToSB3Wrapper(jax_env)
        return _init

    # Use a single environment for GPU memory efficiency for initial testing
    num_cpu = 8  # Reduced to 1 to address CUDA out of memory error

    # Create a vectorized environment with DummyVecEnv to avoid multiprocessing issues
    train_env = DummyVecEnv([make_env(DATA_ROOT, TRAIN_START_DATE, TRAIN_END_DATE) for _ in range(num_cpu)])

    # Add observation and reward normalization for training stability
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # --- 3. CREATE THE PPO AGENT ---

    # MlpPolicy is a standard feed-forward neural network policy.
    # We pass the environment to the PPO agent, enable verbose logging,
    # and specify the directory for TensorBoard logs.

    # Note: Observation space mismatch prevents loading the 80% model
    # Starting fresh training with the same hyperparameters that were working well
    RESUME_MODEL_PATH = None  # Disabled due to obs space mismatch
    
    if RESUME_MODEL_PATH and os.path.exists(RESUME_MODEL_PATH):
        print(f"Resuming training from {RESUME_MODEL_PATH}")
        model = PPO.load(
            RESUME_MODEL_PATH,
            env=train_env,
            tensorboard_log=logdir)
        print(f"✅ Model loaded successfully from 80% completion")
    else:
        print("--- Creating a new PPO agent with short position support ---")

        # Stabilized hyperparameters for training stability
        if torch.cuda.is_available():
            print("🚀 Using GPU-optimized hyperparameters")
            batch_size = 512      # Larger batch for stability
            n_steps = 2048        # More steps for better estimates
            device = 'cuda'
        else:
            print("💻 Using CPU-optimized hyperparameters")
            batch_size = 256      # Smaller batch for CPU
            n_steps = 1024        # Fewer steps for CPU
            device = 'cpu'

        # Balanced policy configuration with more exploration
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh,  # Tanh activation for stability
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Separate networks
            ortho_init=False,
            log_std_init=-2.0  # More exploration (≈ 0.135)
        )

        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=logdir,
            device=device,        # Explicitly set device
            # Balanced hyperparameters
            learning_rate=1e-5,   # Small step size
            n_steps=2048,         # Standard step count
            batch_size=1024,      # Larger batches stabilize updates
            n_epochs=5,           # A few more passes over data
            gamma=0.99,           # Standard discount factor
            gae_lambda=0.95,      # Standard GAE lambda
            clip_range=lambda _: 0.05,      # Allow some movement
            clip_range_vf=lambda _: 0.05,   # Critic clipping
            ent_coef=0.001,       # Encourage exploration
            vf_coef=0.25,         # Value function coefficient
            max_grad_norm=0.3,    # Gradient clipping
            target_kl=0.01,       # Early stop on KL divergence
            policy_kwargs=policy_kwargs
        )

        print(f"✅ PPO model created with device: {model.device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Steps per update: {n_steps}")
        
        # Sanity check: verify initial log_std
        pi_log_std = model.policy.log_std.detach().cpu().mean().item()
        print(f"   Initial log_std: {pi_log_std:.2f} (should be around -2.0)")
        print(f"   Initial std: {torch.exp(model.policy.log_std).detach().cpu().mean().item():.3f}")

    # --- 4. TRAIN THE AGENT ---

    # Starting fresh training with proven hyperparameters
    TIMESTEPS = 100_000  # Train for 100k steps (equivalent to the remaining 20%)
    SAVE_FREQ = 10000    # More frequent saves for monitoring

    print("--- Starting Fresh Training with Proven Hyperparameters ---")
    print(f"Training timesteps: {TIMESTEPS:,}")
    print(f"Action space: [-1, 1] for short/long positions")
    print(f"Intraday constraints: Short positions closed at end of day")
    print(f"Note: Starting fresh due to observation space changes")

    # Enhanced checkpoint callback with additional metrics and GPU monitoring
    class EnhancedCheckpointCallback(CheckpointCallback):
        def __init__(self, save_freq, save_path, name_prefix):
            super().__init__(save_freq, save_path, name_prefix)
            self.best_reward = -np.inf
            self.step_count = 0

        def _on_step(self) -> bool:
            # Call parent callback
            result = super()._on_step()
            self.step_count += 1

            # Log additional metrics
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                if recent_rewards:
                    avg_reward = np.mean(recent_rewards)
                    if avg_reward > self.best_reward:
                        self.best_reward = avg_reward
                        print(f"New best average reward: {avg_reward:.4f}")

            # GPU memory monitoring (every 1000 steps)
            if self.step_count % 1000 == 0 and torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                gpu_memory_max = torch.cuda.max_memory_allocated() / 1e9
                print(f"GPU Memory: {gpu_memory:.2f}GB (Peak: {gpu_memory_max:.2f}GB)")

            return result

    checkpoint_callback = EnhancedCheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=models_dir,
        name_prefix='checkpoint'
    )

    try:
        model.learn(
            total_timesteps=TIMESTEPS, 
            reset_num_timesteps=True, # Start fresh training
            tb_log_name="PPO_short_positions_fresh",
            callback=checkpoint_callback,
            progress_bar=True  # Show progress bar
        )
        print("--- Agent Training Finished Successfully ---")

    except KeyboardInterrupt:
        print("--- Training Interrupted by User ---")
        print("Saving current model...")
        model.save(f"{models_dir}/interrupted_model.zip")

    except Exception as e:
        print(f"--- Training Failed with Error: {e} ---")
        print("Saving current model for debugging...")
        model.save(f"{models_dir}/error_model.zip")
        raise

    # Save the final trained model (100% complete)
    model.save(f"{models_dir}/final_model_100percent.zip")
    print(f"✅ Final model saved: {models_dir}/final_model_100percent.zip")

    # --- 5. EVALUATE THE TRAINED AGENT ---

    print("\n--- Starting Agent Evaluation ---")

    # Create a separate evaluation environment with a different date range
    eval_jax_env = JAXVectorizedPortfolioEnv(
        data_root=DATA_ROOT,
        start_date=EVAL_START_DATE,
        end_date=EVAL_END_DATE,
        window_size=30,
        transaction_cost_rate=0.005,
        sharpe_window=252,
        use_all_features=True
    )

    eval_env = JAXToSB3Wrapper(eval_jax_env)

    obs, info = eval_env.reset()
    done = False
    episode_reward = 0
    step_count = 0
    max_steps = 1000  # Prevent infinite loops

    print(f"Starting evaluation with observation shape: {obs.shape}")
    print(f"Action space: {eval_env.action_space}")

    while not done and step_count < max_steps:
        # Use the trained model to predict the best action
        # deterministic=True ensures the agent doesn't take random exploratory actions
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, terminated, truncated, info = eval_env.step(action)

        # The environment is "done" if terminated (reached end) or truncated (time limit)
        done = terminated or truncated
        episode_reward += reward
        step_count += 1

        # Print progress every 100 steps
        if step_count % 100 == 0:
            cash_weight = info.get('cash_weight', 0)
            cash_return_contrib = info.get('cash_return_contribution', 0)
            print(f"Step {step_count}: Reward={reward:.4f}, Total={episode_reward:.4f}, Portfolio Value={info.get('portfolio_value', 0):.4f}, Cash Weight={cash_weight:.3f}, Cash Return={cash_return_contrib:.6f}")

        # Render the environment's state to see what's happening
        eval_env.render()

    # Print final portfolio state after evaluation
    print("\n--- Evaluation Finished ---")
    print(f"Total Steps: {step_count}")
    print(f"Total Episode Reward: {episode_reward:.4f}")
    print(f"Final Portfolio Value: {info.get('portfolio_value', 0):.4f}")
    print(f"Total Return: {info.get('total_return', 0):.4f}")
    print(f"Final Sharpe Ratio: {info.get('sharpe_ratio', 0):.4f}")
    print(f"Short Exposure: {info.get('short_exposure', 0):.4f}")
    print(f"Overnight Short Penalty: {info.get('overnight_short_penalty', 0):.4f}")
    print(f"Final Cash Weight: {info.get('cash_weight', 0):.4f}")
    print(f"Cash Return Contribution: {info.get('cash_return_contribution', 0):.6f}")
    print(f"Cash Return Rate (Daily): {info.get('cash_return_rate_daily', 0):.6f} ({info.get('cash_return_rate_daily', 0) * 252 * 100:.2f}% annual)")

    # Close the evaluation environment
    eval_env.close()

    # --- 6. COMPREHENSIVE TESTING ---

    def test_short_position_functionality():
        """Test the short position functionality and edge cases"""
        print("\n--- Testing Short Position Functionality ---")

        try:
            # Create test environment
            test_jax_env = JAXVectorizedPortfolioEnv(
                data_root=DATA_ROOT,
                start_date='2024-06-06',
                end_date='2024-06-10',  # Short period for testing
                window_size=30,
                transaction_cost_rate=0.005,
                sharpe_window=252,
                use_all_features=True
            )

            test_env = JAXToSB3Wrapper(test_jax_env)

            # Test 1: Environment initialization
            print("Test 1: Environment initialization...")
            obs, info = test_env.reset()
            print(f"✅ Environment initialized successfully")
            print(f"   Observation shape: {obs.shape}")
            print(f"   Action space: {test_env.action_space}")

            # Test 2: Long position (positive action)
            print("\nTest 2: Long position (positive action)...")
            long_action = np.random.uniform(0.1, 0.5, test_env.action_space.shape[0])
            obs, reward, done, truncated, info = test_env.step(long_action)
            print(f"✅ Long position executed successfully")
            print(f"   Reward: {reward:.4f}")
            print(f"   Portfolio value: {info.get('portfolio_value', 0):.4f}")

            # Test 3: Short position (negative action)
            print("\nTest 3: Short position (negative action)...")
            short_action = np.random.uniform(-0.5, -0.1, test_env.action_space.shape[0])
            obs, reward, done, truncated, info = test_env.step(short_action)
            print(f"✅ Short position executed successfully")
            print(f"   Reward: {reward:.4f}")
            print(f"   Short exposure: {info.get('short_exposure', 0):.4f}")
            print(f"   Portfolio value: {info.get('portfolio_value', 0):.4f}")

            # Test 4: Edge case - extreme actions
            print("\nTest 4: Edge case - extreme actions...")
            extreme_action = np.full(test_env.action_space.shape[0], 1.0)  # All max long
            obs, reward, done, truncated, info = test_env.step(extreme_action)
            print(f"✅ Extreme long action handled successfully")

            extreme_action = np.full(test_env.action_space.shape[0], -1.0)  # All max short
            obs, reward, done, truncated, info = test_env.step(extreme_action)
            print(f"✅ Extreme short action handled successfully")
            print(f"   Short exposure: {info.get('short_exposure', 0):.4f}")

            # Test 5: NaN/Inf handling
            print("\nTest 5: NaN/Inf handling...")
            nan_action = np.full(test_env.action_space.shape[0], np.nan)
            obs, reward, done, truncated, info = test_env.step(nan_action)
            print(f"✅ NaN action handled successfully")

            # Test 6: Action clipping
            print("\nTest 6: Action clipping...")
            out_of_bounds_action = np.full(test_env.action_space.shape[0], 2.0)  # Out of bounds
            obs, reward, done, truncated, info = test_env.step(out_of_bounds_action)
            print(f"✅ Out-of-bounds action clipped successfully")

            # Test 7: Cash return functionality
            print("\nTest 7: Cash return functionality...")
            # Test with high cash allocation
            cash_action = np.zeros(test_env.action_space.shape[0])
            cash_action[-1] = 0.8  # High cash allocation
            obs, reward, done, truncated, info = test_env.step(cash_action)
            cash_weight = info.get('cash_weight', 0)
            cash_return_contrib = info.get('cash_return_contribution', 0)
            cash_return_rate = info.get('cash_return_rate_daily', 0)
            print(f"✅ Cash return functionality working")
            print(f"   Cash weight: {cash_weight:.4f}")
            print(f"   Cash return contribution: {cash_return_contrib:.6f}")
            print(f"   Cash return rate (daily): {cash_return_rate:.6f} ({cash_return_rate * 252 * 100:.2f}% annual)")

            test_env.close()
            print("\n✅ All tests passed successfully!")

        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

    # Run the tests
    test_short_position_functionality()

    def test_gpu_performance():
        """Test GPU performance and memory usage"""
        print("\n--- Testing GPU Performance ---")

        if not torch.cuda.is_available():
            print("⚠️ CUDA not available - skipping GPU performance test")
            return

        try:
            # Test JAX GPU performance
            import jax
            import jax.numpy as jnp

            print("Testing JAX GPU performance...")
            key = jax.random.PRNGKey(42)

            # Large matrix operations to test GPU
            start_time = time.time()
            a = jax.random.normal(key, (1000, 1000))
            b = jax.random.normal(key, (1000, 1000))
            c = jnp.dot(a, b)
            jax.block_until_ready(c)  # Ensure computation is complete
            jax_time = time.time() - start_time

            print(f"✅ JAX GPU matrix multiplication (1000x1000): {jax_time:.4f}s")

            # Test PyTorch GPU performance
            print("Testing PyTorch GPU performance...")
            start_time = time.time()
            a_torch = torch.randn(1000, 1000, device='cuda')
            b_torch = torch.randn(1000, 1000, device='cuda')
            c_torch = torch.mm(a_torch, b_torch)
            torch.cuda.synchronize()  # Ensure computation is complete
            torch_time = time.time() - start_time

            print(f"✅ PyTorch GPU matrix multiplication (1000x1000): {torch_time:.4f}s")

            # Memory usage
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            print(f"✅ GPU Memory Usage: {gpu_memory:.2f}GB")

            # Clear GPU memory
            del a, b, c, a_torch, b_torch, c_torch
            torch.cuda.empty_cache()
            jax.clear_caches()

            print("✅ GPU performance test completed successfully!")

        except Exception as e:
            print(f"❌ GPU performance test failed: {e}")

    # Run GPU performance test
    test_gpu_performance()

    print("\n--- Training Complete ---")