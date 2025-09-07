import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import time
import os
import multiprocessing
import json

# Import your custom environment and feature engineering
from finagent.environment.portfolio_env import PortfolioEnv
from feature_engineering_utils import enhance_training_data

# --- 0. ENHANCED FEATURE ENGINEERING ---

print("🚀 Starting Enhanced Training with Advanced Feature Engineering...")

# Run feature engineering if enhanced data doesn't exist
ENHANCED_DATA_PATH = "processed_data/enhanced"
if not os.path.exists(ENHANCED_DATA_PATH):
    print("📊 Creating enhanced features...")
    enhance_training_data("processed_data/", ENHANCED_DATA_PATH)
else:
    print("✅ Enhanced features already exist, skipping feature engineering")

# --- 1. DEFINE FILE PATHS AND PARAMETERS ---

# Create directories to save logs and models
models_dir = f"models/PPO-Enhanced-{int(time.time())}"
logdir = f"logs/PPO-Enhanced-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Training parameters - Enhanced
TRAIN_START_DATE = '2024-06-06'
TRAIN_END_DATE = '2025-03-06'
EVAL_START_DATE = '2025-03-07'  
EVAL_END_DATE = '2025-06-06'

# Use enhanced data if available, otherwise fall back to regular processed data
DATA_ROOT = ENHANCED_DATA_PATH if os.path.exists(ENHANCED_DATA_PATH) else "processed_data/"

print(f"📁 Using data from: {DATA_ROOT}")

# --- 2. INSTANTIATE THE ENHANCED TRAINING ENVIRONMENT ---

def make_enhanced_env(data_root, start_date, end_date):
    """Create environment with enhanced feature engineering"""
    return lambda: PortfolioEnv(
        data_root=data_root,
        start_date=start_date,
        end_date=end_date,
        use_all_features=True,  # Use all available features including engineered ones
    )

# Use all available CPU cores for parallel environments
num_cpu = multiprocessing.cpu_count()
print(f"🖥️  Using {num_cpu} CPU cores for parallel training")

# Create a vectorized environment with enhanced features
train_env = SubprocVecEnv([
    make_enhanced_env(DATA_ROOT, TRAIN_START_DATE, TRAIN_END_DATE) 
    for _ in range(num_cpu)
])

# --- 3. CREATE THE ENHANCED PPO AGENT ---

# Enhanced hyperparameters for better performance with more features
RESUME_MODEL_PATH = None  # Set to path if resuming training

if RESUME_MODEL_PATH and os.path.exists(RESUME_MODEL_PATH):
    print(f"🔄 Resuming training from {RESUME_MODEL_PATH}")
    model = PPO.load(
        RESUME_MODEL_PATH,
        env=train_env,
        tensorboard_log=logdir
    )
else:
    print("🆕 Creating a new Enhanced PPO agent...")
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        tensorboard_log=logdir,
        
        # Enhanced hyperparameters for complex feature space
        n_steps=4096,          # Increased from 2048 for better feature learning
        batch_size=128,        # Increased from 64 for more stable gradients
        n_epochs=15,           # Increased from 10 for better feature utilization
        gamma=0.995,           # Slightly higher for longer-term planning
        gae_lambda=0.98,       # Slightly higher for better advantage estimation
        ent_coef=0.01,         # Small entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=5e-4,    # Slightly lower for stable learning
        
        # Policy network architecture for handling more features
        policy_kwargs=dict(
            net_arch=[512, 512, 256],  # Larger network for complex features
            activation_fn=torch.nn.ReLU,
        )
    )

# --- 4. ENHANCED TRAINING SETUP ---

# Increased timesteps for enhanced feature learning
TIMESTEPS = 2_000_000      # Increased from 1M for better feature learning
SAVE_FREQ = 100000         # More frequent saves for enhanced model

print("🎯 Starting Enhanced Agent Training...")
print(f"⏱️  Training for {TIMESTEPS:,} timesteps")
print(f"💾 Saving every {SAVE_FREQ:,} steps")

# Enhanced callback with more frequent checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=models_dir,
    name_prefix='enhanced_checkpoint'
)

# Custom callback for logging enhanced metrics
class EnhancedMetricsCallback:
    def __init__(self, log_freq=10000):
        self.log_freq = log_freq
        self.last_log = 0
        
    def __call__(self, locals_, globals_):
        if locals_['self'].num_timesteps - self.last_log >= self.log_freq:
            # Log additional metrics here if needed
            self.last_log = locals_['self'].num_timesteps
        return True

enhanced_callback = EnhancedMetricsCallback()

model.learn(
    total_timesteps=TIMESTEPS,
    reset_num_timesteps=False,
    tb_log_name="PPO_Enhanced_run", 
    callback=[checkpoint_callback]  # Can add enhanced_callback if needed
)

print("✅ Enhanced Agent Training Finished!")

# Save the final enhanced model
final_model_path = f"{models_dir}/final_enhanced_model.zip"
model.save(final_model_path)
print(f"💾 Final enhanced model saved to: {final_model_path}")

# --- 5. EVALUATE THE ENHANCED TRAINED AGENT ---

print("\n🔬 Starting Enhanced Agent Evaluation...")

# Create evaluation environment with enhanced features
eval_env = PortfolioEnv(
    data_root=DATA_ROOT,
    start_date=EVAL_START_DATE,
    end_date=EVAL_END_DATE,
    use_all_features=True
)

# Evaluation loop
obs, info = eval_env.reset()
done = False
episode_reward = 0
step_count = 0

print("📊 Running evaluation...")

while not done:
    # Use the enhanced trained model to predict
    action, _states = model.predict(obs, deterministic=True)
    
    # Take action
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated
    episode_reward += reward
    step_count += 1
    
    # Periodic logging during evaluation
    if step_count % 50 == 0:
        print(f"Step {step_count}: Reward={reward:.4f}, Portfolio Value={info.get('portfolio_value', 0):.4f}")
    
    # Render occasionally (not every step to save time)
    if step_count % 20 == 0:
        eval_env.render()

# Final evaluation results
print("\n🎉 Enhanced Evaluation Complete!")
print("=" * 50)
print(f"📈 Final Portfolio Value: {info.get('portfolio_value', 0):.4f}")
print(f"💰 Total Return: {info.get('total_return', 0):.2f}%")
print(f"📊 Sharpe Ratio: {info.get('sharpe_ratio', 0):.4f}")
print(f"📉 Max Drawdown: {info.get('max_drawdown', 0):.2f}%")
print(f"🎯 Total Steps: {step_count}")
print(f"🏆 Total Episode Reward: {episode_reward:.4f}")
print("=" * 50)

# Save evaluation results
eval_results = {
    'final_portfolio_value': info.get('portfolio_value', 0),
    'total_return_pct': info.get('total_return', 0), 
    'sharpe_ratio': info.get('sharpe_ratio', 0),
    'max_drawdown_pct': info.get('max_drawdown', 0),
    'total_steps': step_count,
    'episode_reward': episode_reward,
    'model_path': final_model_path,
    'data_path': DATA_ROOT,
    'features_used': 'enhanced'
}

results_path = f"{models_dir}/evaluation_results.json"
import json
with open(results_path, 'w') as f:
    json.dump(eval_results, f, indent=2)

print(f"📝 Evaluation results saved to: {results_path}")
print("\n🚀 Enhanced training and evaluation complete!")