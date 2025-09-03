import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Tuple
import time
import os
import multiprocessing
from finagent.environment.portfolio_env import PortfolioEnv

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer encoder over (window_size, n_assets, n_features).
    Automatically infers dimensions from observation_space.
    """

    def __init__(
        self,
        observation_space,
        d_model=64,
        nhead=4,
        num_layers=2,
    ):
        # observation_space.shape expected = (window_size, n_assets, n_features)
        # Might need to change portfolio_env for this to work
        shape = observation_space.shape
        if len(shape) != 3:
            raise ValueError(
                f"Expected observation shape (window_size, n_assets, n_features), got {shape}"
            )

        self.window_size, self.n_assets, self.n_features = shape
        features_dim = d_model
        super().__init__(observation_space, features_dim=features_dim)

        # Project raw features into d_model space
        self.input_proj = nn.Linear(self.n_features, d_model)

        # Positional encoding (learnable)
        seq_len = self.window_size * self.n_assets
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: [batch, window_size, n_assets, n_features]
        """
        batch_size = observations.shape[0]
        # Flatten window & assets into sequence
        obs = observations.view(
            batch_size, self.window_size * self.n_assets, self.n_features
        )

        x = self.input_proj(obs)  # [batch, seq_len, d_model]
        seq_len = x.size(1)

        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Mean pooling across sequence
        x = x.mean(dim=1)

        return self.fc_out(x)


class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(
                d_model=64,
                nhead=4,
                num_layers=2,
            ),
        )

# --- 1. FILE PATHS ---
models_dir = f"models/PPO-Transformer-{int(time.time())}"
logdir = f"logs/PPO-Transformer-{int(time.time())}"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# --- 2. DATA RANGES ---
TRAIN_START_DATE = '2024-06-06'
TRAIN_END_DATE = '2025-03-06'
EVAL_START_DATE = '2025-03-07'
EVAL_END_DATE = '2025-06-06'
DATA_ROOT = "processed_data/"

# --- 3. ENV FACTORY ---
def make_env(data_root, start_date, end_date):
    return lambda: PortfolioEnv(
        data_root=data_root,
        start_date=start_date,
        end_date=end_date
    )

num_cpu = min(4, multiprocessing.cpu_count())
train_env = SubprocVecEnv([make_env(DATA_ROOT, TRAIN_START_DATE, TRAIN_END_DATE) for _ in range(num_cpu)])

# --- 4. PPO with Transformer ---
RESUME_MODEL_PATH = None  # update if resuming

if RESUME_MODEL_PATH and os.path.exists(RESUME_MODEL_PATH):
    print(f"Resuming from {RESUME_MODEL_PATH}")
    model = PPO.load(RESUME_MODEL_PATH, env=train_env, tensorboard_log=logdir)
else:
    print("--- Creating new PPO+Transformer agent ---")
    model = PPO(
        policy=TransformerPolicy,
        env=train_env,
        verbose=1,
        tensorboard_log=logdir,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

# --- 5. TRAIN ---
TIMESTEPS = 1_000_000
SAVE_FREQ = 50000
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=models_dir,
    name_prefix='checkpoint'
)

print("--- Training Started ---")
model.learn(
    total_timesteps=TIMESTEPS,
    reset_num_timesteps=False,
    tb_log_name="PPO_Transformer_run",
    callback=checkpoint_callback
)
print("--- Training Finished ---")

model.save(f"{models_dir}/final_model.zip")

# --- 6. EVALUATE ---
eval_env = DummyVecEnv([make_env(DATA_ROOT, EVAL_START_DATE, EVAL_END_DATE)])
obs = eval_env.reset()

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    try:
        eval_env.render()
    except Exception:
        pass

print("\n--- Evaluation Finished ---")
print(info[0])  # final portfolio stats if your env returns them