import os
import time
import multiprocessing
import numpy as np
import pandas as pd

# Recurrent PPO from sb3_contrib
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your custom environment
from finagent.environment.portfolio_env import PortfolioEnv

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---

# Create directories to save logs and models
models_dir = f"models/PPO-LSTM-{int(time.time())}"
logdir = f"logs/PPO-LSTM-{int(time.time())}"

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

def make_env(data_root, start_date, end_date):
    # return a factory for SubprocVecEnv
    return lambda: PortfolioEnv(
        data_root=data_root,
        start_date=start_date,
        end_date=end_date
    )

# Choose number of parallel envs (keep small for recurrent policies)
num_cpu = multiprocessing.cpu_count()
n_envs = min(4, num_cpu)  # Recurrent policies maintain LSTM state per-env; 4 is a reasonable default

train_env = SubprocVecEnv([make_env(DATA_ROOT, TRAIN_START_DATE, TRAIN_END_DATE) for _ in range(n_envs)])

# --- 3. CREATE THE RecurrentPPO AGENT (LSTM policy) ---

RESUME_MODEL_PATH = "models/PPO-1756379121/checkpoint_950000_steps.zip"  # path to a saved model to resume (optional)

# LSTM policy kwargs:
policy_kwargs = dict(
    lstm_hidden_size=256,   # size of LSTM hidden state
    n_lstm_layers=1,        # number of LSTM layers
    shared_lstm=False,      # whether actor+critic share LSTM
    enable_critic_lstm=True,
    # net_arch defines the MLP(s) for pi and vf before/after LSTM
    net_arch=[dict(pi=[128], vf=[128])],
)

if RESUME_MODEL_PATH and os.path.exists(RESUME_MODEL_PATH):
    print(f"Resuming training from {RESUME_MODEL_PATH}")
    # Try loading with RecurrentPPO (model saved must be compatible)
    model = RecurrentPPO.load(RESUME_MODEL_PATH, env=train_env, tensorboard_log=logdir)
else:
    print("--- Creating a new RecurrentPPO (LSTM) agent ---")
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=logdir,
        # Important recurrent/hyperparameter choices
        n_steps=256,          # rollout length per env (smaller than large values for recurrent)
        batch_size=256,       # minibatch size (should divide n_steps * n_envs)
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=42,
    )

# --- 4. TRAIN THE AGENT ---

TIMESTEPS = 1_000_000      # change to desired total timesteps for real runs
SAVE_FREQ = 50000

print("--- Starting Agent Training ---")

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=models_dir,
    name_prefix='checkpoint'
)

model.learn(
    total_timesteps=TIMESTEPS,
    reset_num_timesteps=False,
    tb_log_name="PPO_LSTM_run",
    callback=checkpoint_callback
)
print("--- Agent Training Finished ---")

# Save the final trained model
model.save(f"{models_dir}/final_model.zip")

# --- 5. EVALUATE THE TRAINED AGENT ---

print("\n--- Starting Agent Evaluation ---")

# Use a single env for clear evaluation and to simplify LSTM state handling
eval_env = DummyVecEnv([make_env(DATA_ROOT, EVAL_START_DATE, EVAL_END_DATE)])

# Reset env and initialize LSTM states for prediction
obs = eval_env.reset()
lstm_states = None  # None means zeros initial state for RecurrentPPO
episode_starts = np.ones((eval_env.num_envs,), dtype=bool)

done = False
episode_reward = 0.0
step_count = 0
max_steps = 10_000_000  # safety cap to avoid infinite loops

while not done and step_count < max_steps:
    # pass LSTM state and episode_starts mask to predict
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, infos = eval_env.step(action)

    # Update mask for next model.predict call
    episode_starts = dones

    # Check if the single env finished
    done = bool(dones[0])
    step_count += 1

    # Render (if implemented in your env)
    try:
        eval_env.render()
    except Exception:
        pass

# After completion, infos is a list of one dict (for DummyVecEnv)
final_info = infos[0] if isinstance(infos, (list, tuple)) else infos

print("\n--- Evaluation Finished ---")
# Defensive prints if keys exist
print(f"Final Portfolio Value: {final_info.get('portfolio_value', float('nan')):.2f}")
print(f"Total Return: {final_info.get('total_return', float('nan')):.2f}%")
print(f"Final Sharpe Ratio: {final_info.get('sharpe_ratio', float('nan')):.4f}")
