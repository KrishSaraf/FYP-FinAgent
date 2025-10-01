import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import deque
import copy
import time
import math
import sys
import multiprocessing as mp
import cloudpickle
import glob
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------------------------
# Utilities / Config
# -------------------------
SEED = 42
np.random.seed(SEED)


# -------------------------
# Feature Engineering
# -------------------------
class FeatureSelector:
    """Feature selector for different feature categories"""

    def __init__(self):
        self.feature_categories = {
            'basic': ['close'],
            'ohlcv': ['open', 'high', 'low', 'close', 'volume'],
            'technical': ['close_lag_1', 'close_diff_pct_lag_1', 'sma_5', 'sma_20', 'rsi_14', 'volatility_5'],
            'sentiment': ['news_sentiment_mean']
        }

    def get_features_for_combination(self, combination: str) -> List[str]:
        """Get list of features for a given combination string."""
        if combination.lower() == 'all':
            selected_features = []
            for features in self.feature_categories.values():
                selected_features.extend(features)
            return list(set(selected_features))

        if combination.lower() == 'basic':
            return self.feature_categories['basic']

        categories = [cat.strip().lower() for cat in combination.split('+')]
        valid_categories = set(self.feature_categories.keys())
        invalid_categories = set(categories) - valid_categories
        if invalid_categories:
            raise ValueError(f"Invalid feature categories: {invalid_categories}. Valid categories are: {valid_categories}")

        selected_features = []
        for category in categories:
            selected_features.extend(self.feature_categories[category])

        return list(set(selected_features))

    def engineer_features(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """Engineer features based on the selected feature set."""
        df_engineered = pd.DataFrame(index=df.index)

        # Basic features
        if 'close' in selected_features:
            df_engineered['close'] = df['close']

        # OHLCV features
        if 'open' in selected_features and 'open' in df.columns:
            df_engineered['open'] = df['open']
        if 'high' in selected_features and 'high' in df.columns:
            df_engineered['high'] = df['high']
        if 'low' in selected_features and 'low' in df.columns:
            df_engineered['low'] = df['low']
        if 'volume' in selected_features and 'volume' in df.columns:
            df_engineered['volume'] = df['volume']

        # Technical features
        if 'close_lag_1' in selected_features:
            df_engineered['close_lag_1'] = df['close'].shift(1)
        if 'close_diff_pct_lag_1' in selected_features:
            df_engineered['close_diff_pct_lag_1'] = df['close'].pct_change(1)
        if 'sma_5' in selected_features:
            df_engineered['sma_5'] = df['close'].rolling(window=5).mean()
        if 'sma_20' in selected_features:
            df_engineered['sma_20'] = df['close'].rolling(window=20).mean()
        if 'rsi_14' in selected_features:
            df_engineered['rsi_14'] = self._calculate_rsi(df['close'], 14)
        if 'volatility_5' in selected_features:
            df_engineered['volatility_5'] = df['close'].pct_change().rolling(window=5).std()

        # Sentiment features
        if 'news_sentiment_mean' in selected_features and 'news_sentiment_mean' in df.columns:
            df_engineered['news_sentiment_mean'] = df['news_sentiment_mean']

        return df_engineered.fillna(0.0)

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


# -------------------------
# Enhanced Environment with Feature Engineering
# -------------------------
class EnhancedTradingEnv:
    def __init__(
        self,
        data_dir: str,
        feature_combination: str = 'basic',
        lookback: int = 5,
        transaction_cost: float = 0.001,
        short_cost_rate: float = 0.001,
        max_leverage: float = 1.0,
        reward_scaling: float = 100.0,
        turnover_penalty_scale: float = 0.0,
        clip_return: float = 0.2,
        sharpe_window: int = 30,
        train_start_date: str = None,
        train_end_date: str = None,
    ):
        self.data_dir = data_dir
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        self.short_cost_rate = short_cost_rate
        self.max_leverage = max_leverage
        self.reward_scaling = reward_scaling
        self.turnover_penalty_scale = turnover_penalty_scale
        self.clip_return = clip_return
        self.sharpe_window = sharpe_window
        self.train_start_date = pd.Timestamp(train_start_date) if train_start_date else None
        self.train_end_date = pd.Timestamp(train_end_date) if train_end_date else None

        # Initialize feature selector
        self.feature_selector = FeatureSelector()
        self.selected_features = self.feature_selector.get_features_for_combination(feature_combination)
        logger.info(f"Selected features: {self.selected_features}")

        # Load and engineer features
        self.data, self.tickers = self._load_and_engineer_data()
        self.T, self.N, self.F = self.data.shape  # Time, Assets, Features
        logger.info(f"Data shape: {self.data.shape} (Time x Assets x Features)")

        # state
        self.current_step = None
        self.portfolio_value = None
        self.prev_weights = None
        self.returns_history = None

    def _load_and_engineer_data(self):
        """Load data and apply feature engineering."""
        files = sorted(glob.glob(os.path.join(self.data_dir, "*_aligned.csv")))
        if not files:
            raise ValueError(f"No *_aligned.csv files found in {self.data_dir}")
        
        engineered_data = []
        tickers = []
        
        for f in files:
            ticker = os.path.basename(f).replace("_aligned.csv", "")
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()

            # Filter by date range
            if self.train_start_date:
                df = df[df.index >= self.train_start_date]
            if self.train_end_date:
                df = df[df.index <= self.train_end_date]
            
            # Apply feature engineering
            df_engineered = self.feature_selector.engineer_features(df, self.selected_features)
            
            # Align the shape of the engineered data
            if len(engineered_data) > 0:
                df_engineered = df_engineered.reindex(engineered_data[0].index, fill_value=0.0)
            engineered_data.append(df_engineered)
            tickers.append(ticker)

        # Stack data: (time, assets, features)
        data = np.stack([df.values for df in engineered_data], axis=1)
        # Forward fill and backward fill NaN values
        for asset_idx in range(data.shape[1]):
            for feature_idx in range(data.shape[2]):
                series = pd.Series(data[:, asset_idx, feature_idx])
                series = series.fillna(method='ffill').fillna(method='bfill').fillna(0)
                data[:, asset_idx, feature_idx] = series.values

        logger.info(f"Loaded {len(files)} tickers with engineered features")
        return data, tickers

    def reset(self):
        self.current_step = self.lookback  # Start after lookback period
        self.portfolio_value = 1.0
        self.prev_weights = np.zeros(self.N, dtype=float)
        self.returns_history = deque(maxlen=self.sharpe_window)
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        """Build observation from engineered features."""
        t = self.current_step
        if t < self.lookback:
            # Pad with zeros if not enough history
            obs_data = np.zeros((self.lookback, self.N, self.F))
            if t > 0:
                obs_data[-t:] = self.data[:t]
        else:
            obs_data = self.data[t - self.lookback:t]
        
        # Flatten the observation
        obs = obs_data.flatten()
        
        # Add previous portfolio weights to observation
        obs = np.concatenate([obs, self.prev_weights])
        return obs

    def step(self, raw_action_scores):
        """Execute one step in the environment."""
        assert len(raw_action_scores) == self.N
        
        # Map to signed allocations in (-1,1)
        weights = np.tanh(np.asarray(raw_action_scores, dtype=float))
        
        # Normalize to target leverage
        abs_sum = np.sum(np.abs(weights)) + 1e-9
        if abs_sum > 0:
            weights = weights / abs_sum * self.max_leverage
        else:
            weights = np.zeros_like(weights)

        t = self.current_step
        if t >= self.T - 1:
            return self._get_obs(), 0.0, True, {"portfolio_value": self.portfolio_value}

        # Calculate returns based on close prices (assuming close is first feature)
        current_prices = self.data[t, :, 0]  # Close price feature
        next_prices = self.data[t + 1, :, 0]
        asset_returns = (next_prices - current_prices) / (current_prices + 1e-9)
        asset_returns = np.clip(asset_returns, -self.clip_return, self.clip_return)

        # Portfolio return
        portfolio_return = float(np.sum(weights * asset_returns))

        # Transaction costs
        turnover = float(np.sum(np.abs(weights - self.prev_weights)))
        transaction_cost = turnover * self.transaction_cost

        # Short borrow costs
        short_exposure = float(np.sum(np.abs(weights[weights < 0])))
        short_borrow_cost = short_exposure * self.short_cost_rate

        # Net return
        net_return = portfolio_return - transaction_cost - short_borrow_cost

        # Update portfolio value
        self.portfolio_value = max(self.portfolio_value * (1.0 + net_return), 1e-6)

        # Record return for Sharpe calculation
        self.returns_history.append(net_return)

        # Sharpe-like bonus
        sharpe_like = 0.0
        if len(self.returns_history) >= 2:
            arr = np.array(self.returns_history)
            mu = arr.mean()
            sigma = arr.std(ddof=0) + 1e-9
            sharpe_like = mu / sigma

        # Reward calculation
        reward = net_return * self.reward_scaling + 0.5 * sharpe_like - self.turnover_penalty_scale * turnover

        # Step forward
        self.prev_weights = weights.copy()
        self.current_step += 1
        next_obs = self._get_obs()
        done = (self.current_step >= self.T - 1)

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "net_return": net_return,
            "transaction_cost": transaction_cost,
            "short_borrow_cost": short_borrow_cost,
            "turnover": turnover,
            "short_exposure": short_exposure,
        }
        return next_obs, float(reward), done, info


# -------------------------
# Enhanced Policy with Better Architecture
# -------------------------
class EnhancedLGBMPolicy:
    def __init__(
        self,
        n_assets,
        feature_dim,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=SEED,
    ):
        self.n_assets = n_assets
        self.feature_dim = feature_dim
        self.base_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": min(2**max_depth, 31),
            "min_data_in_leaf": 5,
            "min_gain_to_split": 0.01,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "max_bin": 255,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": 4,
            "force_col_wise": True,
            "verbose": -1
        }
        
        self.models = [lgb.LGBMRegressor(**self.base_params) for _ in range(n_assets)]
        self.is_fitted = False

    def predict(self, X):
        """Predict action scores for given observations."""
        single = False
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single = True
        
        preds = []
        for i in range(self.n_assets):
            model = self.models[i]
            if self.is_fitted:
                pred = model.predict(X).reshape(-1, 1)
            else:
                pred = np.zeros((X.shape[0], 1))
            preds.append(pred)
        
        preds = np.hstack(preds)
        return preds[0] if single else preds

    def fit(self, X, Y, sample_weight=None):
        """Fit the models."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples = X.shape[0]
        assert Y.shape[0] == n_samples and Y.shape[1] == self.n_assets

        for i in range(self.n_assets):
            y_i = Y[:, i]
            try:
                if sample_weight is not None:
                    self.models[i].fit(X, y_i, sample_weight=sample_weight)
                else:
                    self.models[i].fit(X, y_i)
            except Exception as e:
                logger.warning(f"Failed to fit model for asset {i}: {e}")
                # Continue with unfitted model for this asset
                pass
        
        self.is_fitted = True


def worker_run_episode(env_config_bytes, policy_bytes, gamma, exploration_std, temp, epsilon_uniform, seed=None):
    """Worker function for parallel episode execution."""
    import random
    np.random.seed(seed if seed is not None else int(time.time()) % 1000000)
    random.seed(seed if seed is not None else int(time.time()) % 1000000)

    env_config = cloudpickle.loads(env_config_bytes)
    policy = cloudpickle.loads(policy_bytes)

    env = EnhancedTradingEnv(**env_config)

    obs = env.reset()
    done = False
    obs_list, act_list, rew_list, info_list = [], [], [], []
    
    while not done:
        base_scores = policy.predict(obs)
        noise = np.random.randn(*base_scores.shape) * exploration_std * temp
        scores = base_scores + noise
        
        if np.random.rand() < epsilon_uniform:
            scores = np.random.randn(*base_scores.shape) * exploration_std * 2.0
            
        next_obs, reward, done, info = env.step(scores)
        obs_list.append(obs.copy())
        act_list.append(scores.copy())
        rew_list.append(reward)
        info_list.append(info)
        obs = next_obs

    ep = {
        "obs": np.vstack(obs_list) if obs_list else np.array([]),
        "actions": np.vstack(act_list) if act_list else np.array([]),
        "rewards": np.array(rew_list, dtype=float),
        "infos": info_list,
        "final_value": float(env.portfolio_value),
    }
    return ep


# -------------------------
# Enhanced REINFORCE Trainer
# -------------------------
class EnhancedREINFORCETrainer:
    def __init__(self, env_config, policy, gamma=0.99,
                 exploration_std=0.05, temp=1.0, epsilon_uniform=0.01,
                 episodes_per_update=4, max_buffer_episodes=20,
                 log_every=1, n_workers=4, reward_scaling=100, save_dir="models/lgbm"):
        self.env_config = env_config
        self.policy = policy
        self.gamma = gamma
        self.exploration_std = exploration_std
        self.temp = temp
        self.epsilon_uniform = epsilon_uniform
        self.episodes_per_update = episodes_per_update
        self.max_buffer_episodes = max_buffer_episodes
        self.log_every = log_every
        self.episode_buffer = []
        self.n_workers = n_workers
        self.reward_scaling = reward_scaling
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.training_history = []

    def train_with_curriculum(self, curriculum_stages, n_epochs=50):
        """
        Train the policy using a curriculum learning approach.

        Args:
            curriculum_stages (list of dict): Each stage defines parameters like:
                - 'exploration_std': Exploration noise
                - 'reward_scaling': Reward scaling factor
                - 'epochs': Number of epochs for this stage
                - 'epsilon_uniform': Probability of uniform random actions
            n_epochs (int): Total number of epochs to train across all stages.
        """
        history = []
        env_bytes = cloudpickle.dumps(self.env_config)
        pool = mp.Pool(processes=self.n_workers)

        try:
            total_epochs = 0
            for stage_idx, stage in enumerate(curriculum_stages):
                logger.info(f"Starting curriculum stage {stage_idx + 1}/{len(curriculum_stages)}: {stage}")

                # Update trainer parameters for this stage
                self.exploration_std = stage.get('exploration_std', self.exploration_std)
                self.reward_scaling = stage.get('reward_scaling', self.reward_scaling)
                self.epsilon_uniform = stage.get('epsilon_uniform', self.epsilon_uniform)
                stage_epochs = stage.get('epochs', 10)

                for epoch in range(1, stage_epochs + 1):
                    if total_epochs >= n_epochs:
                        logger.info("Reached the total number of epochs specified. Stopping training.")
                        return history

                    policy_bytes = cloudpickle.dumps(self.policy)
                    tasks = [
                        (env_bytes, policy_bytes, self.gamma, self.exploration_std,
                         self.temp, self.epsilon_uniform, SEED + k)
                        for k in range(self.episodes_per_update)
                    ]

                    eps = pool.starmap(worker_run_episode, tasks)

                    # Add episodes to buffer
                    self.episode_buffer = (self.episode_buffer + eps)[-self.max_buffer_episodes:]

                    # Update policy
                    self.update_policy_from_buffer()

                    # Calculate metrics
                    final_values = [ep["final_value"] for ep in eps]
                    avg_final = np.mean(final_values)
                    std_final = np.std(final_values)

                    # Log progress
                    if epoch % self.log_every == 0:
                        logger.info(f"[Stage {stage_idx + 1}, Epoch {epoch}/{stage_epochs}] "
                                    f"avg_final={avg_final:.4f} ± {std_final:.4f}, buffer={len(self.episode_buffer)}")
                        sys.stdout.flush()

                    # Store history
                    epoch_info = {
                        "stage": stage_idx + 1,
                        "epoch": total_epochs + 1,
                        "avg_final_value": float(avg_final),
                        "std_final_value": float(std_final),
                        "min_final_value": float(np.min(final_values)),
                        "max_final_value": float(np.max(final_values))
                    }
                    history.append(epoch_info)
                    self.training_history.append(epoch_info)

                    # Save model periodically and at the end of the stage
                    if epoch % 10 == 0 or epoch == stage_epochs:
                        self.save_model(f"stage_{stage_idx + 1}_epoch_{epoch}", final_values)

                    total_epochs += 1

                # Dynamic adjustments after each stage
                self._adjust_parameters_dynamically(history)

        finally:
            pool.close()
            pool.join()

        return history

    def _adjust_parameters_dynamically(self, history):
        """
        Adjust training parameters dynamically based on training progress.
        """
        if len(history) < 20:
            return  # Not enough data to adjust

        recent_history = history[-20:]
        avg_portfolio_values = [h['avg_final_value'] for h in recent_history]
        variance = np.var(avg_portfolio_values)

        # Adjust exploration_std if portfolio value is dropping
        if avg_portfolio_values[-1] < avg_portfolio_values[0]:
            self.exploration_std = min(self.exploration_std * 1.2, 0.5)
            logger.info(f"Increasing exploration_std to {self.exploration_std:.4f} due to portfolio value drop.")

        # Adjust epsilon_uniform if variance is dropping quickly
        if variance < 0.01:
            self.epsilon_uniform = min(self.epsilon_uniform * 1.5, 0.1)
            logger.info(f"Increasing epsilon_uniform to {self.epsilon_uniform:.4f} due to low variance.")

        # Adjust reward_scaling if consistent small losses are observed
        if np.mean(avg_portfolio_values) < 1.0:
            self.reward_scaling = max(self.reward_scaling * 0.8, 1.0)
            logger.info(f"Reducing reward_scaling to {self.reward_scaling:.4f} due to consistent small losses.")

    def _discounted_returns(self, rewards):
        """Calculate discounted returns."""
        R, out = 0.0, []
        for r in reversed(rewards):
            R = r + self.gamma * R
            out.append(R)
        return np.array(out[::-1])

    def update_policy_from_buffer(self):
        """Update policy using episodes in buffer."""
        if not self.episode_buffer:
            return

        X_list, Y_list, A_list = [], [], []
        for ep in self.episode_buffer:
            if len(ep["obs"]) == 0:
                continue
                
            returns = self._discounted_returns(ep["rewards"])
            adv = returns - returns.mean()
            X_list.append(ep["obs"])
            Y_list.append(ep["actions"])
            A_list.append(adv)

        if not X_list:
            return

        X = np.vstack(X_list)
        Y = np.vstack(Y_list)
        A = np.concatenate(A_list)
        
        # Normalize advantages
        A = (A - A.mean()) / (A.std() + 1e-9)
        A_shift = A - A.min() + 1e-6
        max_w = np.percentile(A_shift, 99.5)
        sample_weight = np.minimum(A_shift, max_w)
        
        # Fit policy
        self.policy.fit(X, Y, sample_weight=sample_weight)

    def save_model(self, epoch, final_values):
        """Save the trained model and training history."""
        # Save policy
        policy_path = self.save_dir / f"policy_epoch_{epoch}.pkl"
        with open(policy_path, 'wb') as f:
            pickle.dump(self.policy, f)
        
        # Save training history
        history_path = self.save_dir / "training_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        # Save training plots
        self.plot_training_progress(epoch, final_values)
        
        logger.info(f"Model saved to {policy_path}")

    def plot_training_progress(self, epoch, final_values):
        """Plot and save training progress."""
        if not self.training_history:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot final values over epochs
        epochs = [h['epoch'] for h in self.training_history]
        avg_finals = [h['avg_final_value'] for h in self.training_history]
        
        axes[0, 0].plot(epochs, avg_finals, 'b-', linewidth=2)
        axes[0, 0].set_title('Average Final Portfolio Value')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot current episode final values
        axes[0, 1].hist(final_values, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title(f'Final Values Distribution (Epoch {epoch})')
        axes[0, 1].set_xlabel('Portfolio Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot rolling average
        if len(avg_finals) >= 5:
            rolling_avg = pd.Series(avg_finals).rolling(window=5, center=True).mean()
            axes[1, 0].plot(epochs, avg_finals, 'b-', alpha=0.5, label='Actual')
            axes[1, 0].plot(epochs, rolling_avg, 'r-', linewidth=2, label='5-epoch avg')
            axes[1, 0].set_title('Portfolio Value Trend')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Portfolio Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot improvement over time
        if len(avg_finals) >= 2:
            improvements = np.diff(avg_finals)
            axes[1, 1].plot(epochs[1:], improvements, 'g-', linewidth=2)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Epoch-to-Epoch Improvement')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Change in Portfolio Value')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.save_dir / f"training_progress_epoch_{epoch}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, n_epochs=50):
        """Train the policy."""
        history = []
        env_bytes = cloudpickle.dumps(self.env_config)
        pool = mp.Pool(processes=self.n_workers)

        try:
            for epoch in range(1, n_epochs + 1):
                policy_bytes = cloudpickle.dumps(self.policy)
                tasks = [
                    (env_bytes, policy_bytes, self.gamma, self.exploration_std,
                     self.temp, self.epsilon_uniform, SEED + k)
                    for k in range(self.episodes_per_update)
                ]
                
                eps = pool.starmap(worker_run_episode, tasks)
                
                # Add episodes to buffer
                self.episode_buffer = (self.episode_buffer + eps)[-self.max_buffer_episodes:]
                
                # Update policy
                self.update_policy_from_buffer()
                
                # Calculate metrics
                final_values = [ep["final_value"] for ep in eps]
                avg_final = np.mean(final_values)
                std_final = np.std(final_values)
                
                # Log progress
                if epoch % self.log_every == 0:
                    logger.info(f"[Epoch {epoch}/{n_epochs}] avg_final={avg_final:.4f} ± {std_final:.4f}, buffer={len(self.episode_buffer)}")
                    sys.stdout.flush()
                
                # Store history
                epoch_info = {
                    "epoch": epoch,
                    "avg_final_value": float(avg_final),
                    "std_final_value": float(std_final),
                    "min_final_value": float(np.min(final_values)),
                    "max_final_value": float(np.max(final_values))
                }
                history.append(epoch_info)
                self.training_history.append(epoch_info)
                
                # Save model periodically and at the end
                if epoch % 10 == 0 or epoch == n_epochs:
                    self.save_model(epoch, final_values)

        finally:
            pool.close()
            pool.join()

        return history


# -------------------------
# Main function with enhanced features
# -------------------------
def main(args):
    logger.info("Starting enhanced RL training with feature engineering...")
    
    # Environment configuration
    env_config = {
        "data_dir": args.data_dir,
        "feature_combination": args.feature_combination,
        "lookback": args.lookback,
        "transaction_cost": args.transaction_cost,
        "short_cost_rate": args.short_cost_rate,
        "max_leverage": args.max_leverage,
        "reward_scaling": args.reward_scaling,
        "turnover_penalty_scale": args.turnover_penalty_scale,
        "clip_return": args.clip_return,
        "sharpe_window": args.sharpe_window,
        "train_start_date": args.train_start_date,
        "train_end_date": args.train_end_date,
    }
    
    # Create environment to get dimensions
    env = EnhancedTradingEnv(**env_config)
    
    # Calculate feature dimension
    obs = env.reset()
    feature_dim = len(obs)
    N = env.N
    
    logger.info(f"Environment initialized: {N} assets, {feature_dim} features")
    
    # Initialize policy
    policy = EnhancedLGBMPolicy(
        n_assets=N,
        feature_dim=feature_dim,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth
    )
    
    # Initialize trainer
    trainer = EnhancedREINFORCETrainer(
        env_config, policy,
        gamma=args.gamma,
        exploration_std=args.exploration_std,
        temp=args.temp,
        epsilon_uniform=args.epsilon_uniform,
        episodes_per_update=args.episodes_per_update,
        max_buffer_episodes=args.max_buffer_episodes,
        log_every=args.log_every,
        n_workers=args.n_workers,
        save_dir=args.save_dir
    )
    
    # Train
    t0 = time.time()
    logger.info(f"Starting training for {args.epochs} epochs...")
    curriculum_stages = [
        {"exploration_std": 0.4, "reward_scaling": 5.0, "epochs": 50},
        {"exploration_std": 0.2, "reward_scaling": 25.0, "epochs": 75},
        {"exploration_std": 0.1, "reward_scaling": 100.0, "epochs": 100},
    ]
    hist = trainer.train_with_curriculum(curriculum_stages=curriculum_stages, n_epochs=args.epochs)
    t1 = time.time()
    
    logger.info(f"Training finished in {t1-t0:.1f}s. Final: {hist[-1]}")
    
    # Save final summary
    summary = {
        'args': vars(args),
        'training_time': t1 - t0,
        'final_performance': hist[-1],
        'selected_features': env.selected_features,
        'data_shape': env.data.shape
    }
    
    summary_path = Path(args.save_dir) / "training_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    logger.info(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced RL Trading with Feature Engineering")
    
    # Data and features
    parser.add_argument("--data_dir", type=str, default="processed_data")
    parser.add_argument("--feature_combination", type=str, default="basic", 
                       help="Feature combination: 'basic', 'ohlcv', 'technical', 'all', or 'ohlcv+technical'")
    parser.add_argument("--save_dir", type=str, default="models/lgbm",
                       help="Directory to save trained models")
    
    # Environment parameters
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--short_cost_rate", type=float, default=0.001)
    parser.add_argument("--max_leverage", type=float, default=1.0)
    parser.add_argument("--reward_scaling", type=float, default=100.0)
    parser.add_argument("--turnover_penalty_scale", type=float, default=0.02)
    parser.add_argument("--clip_return", type=float, default=0.2)
    parser.add_argument("--sharpe_window", type=int, default=30)
    
    # Model parameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--max_depth", type=int, default=5)
    
    # Training parameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exploration_std", type=float, default=0.05)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--epsilon_uniform", type=float, default=0.1)
    parser.add_argument("--episodes_per_update", type=int, default=4)
    parser.add_argument("--max_buffer_episodes", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=4)

    # Training date range
    parser.add_argument("--train_start_date", type=str, default="2024-06-06", 
                        help="Start date for training data (YYYY-MM-DD)")
    parser.add_argument("--train_end_date", type=str, default="2025-03-06", 
                        help="End date for training data (YYYY-MM-DD)")
    
    args = parser.parse_args()
    main(args)