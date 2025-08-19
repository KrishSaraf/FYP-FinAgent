import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import time
import os

# Import your custom environment
from finagent.environment.portfolio_env import PortfolioEnv

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---

# Create directories to save logs and models
models_dir = f"models/PPO-{int(time.time())}"
logdir = f"logs/PPO-{int(time.time())}"

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

# Important: Stable Baselines3 requires the environment to be "vectorized"
# We use DummyVecEnv to wrap our single environment.
train_env = DummyVecEnv([lambda: PortfolioEnv(
    data_root=DATA_ROOT,
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE
)])

# --- 3. CREATE THE PPO AGENT ---

# MlpPolicy is a standard feed-forward neural network policy.
# We pass the environment to the PPO agent, enable verbose logging,
# and specify the directory for TensorBoard logs.

RESUME_MODEL_PATH = "models/PPO-1755516322/checkpoint_655000_steps.zip"  # Path to a saved model to resume training, if any

if RESUME_MODEL_PATH and os.path.exists(RESUME_MODEL_PATH):
    print(f"Resuming training from {RESUME_MODEL_PATH}")
    model = PPO.load(
        RESUME_MODEL_PATH,
        env=train_env,
        tensorboard_log=logdir)
else:
    print("--- Creating a new PPO agent ---")
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        tensorboard_log=logdir,
        # You can tune these hyperparameters
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

# --- 4. TRAIN THE AGENT ---

# The total number of steps the agent will be trained for.
# Start with a smaller number (e.g., 20,000) for testing, 
# then increase significantly (e.g., 1,000,000+) for real training.
TIMESTEPS = 5000
SAVE_FREQ = 50000

print("--- Starting Agent Training ---")

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=models_dir,
    name_prefix='checkpoint'
)

model.learn(
    total_timesteps=TIMESTEPS, 
    reset_num_timesteps=False, # Set to False to continue training from a saved model
    tb_log_name="PPO_run",
    callback=checkpoint_callback
)
print("--- Agent Training Finished ---")

# Save the final trained model
model.save(f"{models_dir}/final_model.zip")

# --- 5. EVALUATE THE TRAINED AGENT ---

print("\n--- Starting Agent Evaluation ---")

# Create a separate evaluation environment with a different date range
eval_env = PortfolioEnv(
    data_root=DATA_ROOT,
    start_date=EVAL_START_DATE,
    end_date=EVAL_END_DATE
)

obs, info = eval_env.reset()
done = False
episode_reward = 0

while not done:
    # Use the trained model to predict the best action
    # deterministic=True ensures the agent doesn't take random exploratory actions
    action, _states = model.predict(obs, deterministic=True)
    
    # Take the action in the environment
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    # The environment is "done" if terminated (reached end) or truncated (time limit)
    done = terminated or truncated

    # Render the environment's state to see what's happening
    eval_env.render()
    
# Print final portfolio state after evaluation
print("\n--- Evaluation Finished ---")
print(f"Final Portfolio Value: {info.get('portfolio_value'):.2f}")
print(f"Total Return: {info.get('total_return'):.2f}%")
print(f"Final Sharpe Ratio: {info.get('sharpe_ratio'):.4f}")