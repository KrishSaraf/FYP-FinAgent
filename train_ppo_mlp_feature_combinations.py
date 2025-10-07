#!/usr/bin/env python3
"""Train PPO+MLP with different feature combinations using Stable-Baselines3"""

import argparse
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from finagent.environment.portfolio_env import JAXVectorizedPortfolioEnv
from train_ppo import JAXToSB3Wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None


class FeatureSelector:
    """Feature selector for different feature categories"""
    def __init__(self):
        self.feature_categories = {
            'ohlcv': {
                'description': 'Basic OHLCV price data and derived features',
                'features': [
                    'open', 'high', 'low', 'close', 'volume', 'vwap',
                    'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d', 'returns_20d',
                    'log_returns_1d', 'log_returns_5d', 'overnight_gap', 'daily_range', 'close_position'
                ]
            },
            'technical': {
                'description': 'Technical indicators and trading signals',
                'features': [
                    'dma_50', 'dma_200', 'rsi_14',
                    'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_30d', 'volatility_60d',
                    'vol_ratio_short_long', 'vol_ratio_5_20',
                    'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
                    'momentum_acceleration_10d', 'ma_convergence', 'ma_trend_strength', 'price_position_20d',
                    'price_above_ma50', 'price_above_ma200',
                    'rsi_oversold', 'rsi_overbought', 'bb_position', 'bb_squeeze',
                    'price_zscore_20d', 'price_zscore_60d', 'volume_zscore_20d',
                    'price_deviation_50d', 'price_breakout_20d', 'volume_spike',
                    'ma_cross_bullish', 'ma_cross_bearish', 'high_vol_regime', 'low_vol_regime',
                    'body_ratio', 'doji_pattern', 'hammer_pattern', 'shooting_star_pattern',
                    'volume_ratio_20d', 'bullish_signals', 'bearish_signals', 'net_signal_strength',
                    'risk_adjusted_momentum', 'trend_regime'
                ]
            },
            'financial': {
                'description': 'Financial statement metrics',
                'features': [
                    'metric_Revenue', 'metric_GrossProfit', 'metric_OperatingIncome',
                    'metric_NetIncome', 'metric_Cash', 'metric_TotalAssets', 'metric_TotalEquity',
                    'metric_CashfromOperatingActivities', 'metric_freeCashFlowtrailing12Month',
                    'metric_marketCap', 'metric_beta'
                ]
            },
            'sentiment': {
                'description': 'News and social media sentiment',
                'features': [
                    'reddit_title_sentiments_mean', 'reddit_body_sentiments', 'reddit_score_mean',
                    'reddit_posts_count', 'news_sentiment_mean', 'news_articles_count'
                ]
            }
        }

    def get_features_for_combination(self, combination: str) -> List[str]:
        if combination.lower() == 'all':
            all_features = []
            for cat_info in self.feature_categories.values():
                all_features.extend(cat_info['features'])
            return list(dict.fromkeys(all_features))

        categories = [c.strip().lower() for c in combination.split('+')]
        valid_cats = set(self.feature_categories.keys())
        invalid = set(categories) - valid_cats
        if invalid:
            raise ValueError(f"Invalid categories: {invalid}. Valid: {valid_cats}")

        features = []
        for cat in categories:
            features.extend(self.feature_categories[cat]['features'])
        return list(dict.fromkeys(features))

    def print_available_combinations(self):
        print("\nAvailable feature combinations:")
        print("  'all' - All features")
        for cat, info in self.feature_categories.items():
            print(f"  '{cat}' - {info['description']} ({len(info['features'])} features)")
        print("\nCombine with '+' (e.g., 'ohlcv+technical')")


class CurriculumConfig:
    """Curriculum learning configuration for progressive training"""
    def __init__(self):
        self.stages = {
            1: {
                "name": "Exploration",
                "description": "High exploration, basic learning",
                "total_timesteps": 100_000,
                "learning_rate": 3e-4,
                "ent_coef": 0.01,
                "clip_range": 0.2,
                "n_steps": 2048,
                "batch_size": 512,
            },
            2: {
                "name": "Refinement",
                "description": "Balanced exploration-exploitation",
                "total_timesteps": 150_000,
                "learning_rate": 1e-4,
                "ent_coef": 0.005,
                "clip_range": 0.15,
                "n_steps": 2048,
                "batch_size": 1024,
            },
            3: {
                "name": "Optimization",
                "description": "Fine-tuning and convergence",
                "total_timesteps": 100_000,
                "learning_rate": 5e-5,
                "ent_coef": 0.001,
                "clip_range": 0.1,
                "n_steps": 2048,
                "batch_size": 1024,
            }
        }

    def get_stage(self, stage_num: int) -> Dict[str, Any]:
        return self.stages.get(stage_num, self.stages[1])


def create_env(data_root: str, start_date: str, end_date: str,
               selected_features: List[str], window_size: int = 30):
    """Create environment with selected features"""
    def _init():
        jax_env = JAXVectorizedPortfolioEnv(
            data_root=data_root,
            features=selected_features,
            use_all_features=False,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            transaction_cost_rate=0.005,
            sharpe_window=252
        )
        return JAXToSB3Wrapper(jax_env)
    return _init


def train_stage(stage_num: int, stage_config: Dict, selected_features: List[str],
                args: argparse.Namespace, model=None):
    """Train a curriculum stage"""
    logger.info(f"=== Stage {stage_num}: {stage_config['name']} ===")
    logger.info(f"Description: {stage_config['description']}")

    # Create environment
    train_env = DummyVecEnv([create_env(
        args.data_root, args.train_start_date, args.train_end_date,
        selected_features, args.window_size
    )])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create or update model
    if model is None:
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ortho_init=False,
            log_std_init=-2.0
        )

        model = PPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            device=device,
            learning_rate=stage_config['learning_rate'],
            n_steps=stage_config['n_steps'],
            batch_size=stage_config['batch_size'],
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=stage_config['clip_range'],
            clip_range_vf=0.05,
            ent_coef=stage_config['ent_coef'],
            vf_coef=0.25,
            max_grad_norm=0.3,
            target_kl=0.01,
            policy_kwargs=policy_kwargs
        )
    else:
        # Update hyperparameters
        model.learning_rate = stage_config['learning_rate']
        model.ent_coef = stage_config['ent_coef']
        model.clip_range = stage_config['clip_range']
        model.set_env(train_env)

    # Setup callbacks
    checkpoint_dir = Path(args.model_dir) / f"stage_{stage_num}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(checkpoint_dir),
        name_prefix='checkpoint'
    )

    # Train
    logger.info(f"Training for {stage_config['total_timesteps']:,} timesteps...")
    model.learn(
        total_timesteps=stage_config['total_timesteps'],
        reset_num_timesteps=False,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save stage model
    model.save(checkpoint_dir / f"curriculum_stage_{stage_num}.zip")
    logger.info(f"Stage {stage_num} completed and saved")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO+MLP with feature combinations")

    # Feature selection
    valid_combinations = ['ohlcv', 'technical', 'financial', 'sentiment',
                         'ohlcv+technical', 'ohlcv+technical+sentiment', 'all']
    parser.add_argument('--feature_combination', type=str, default='ohlcv+technical',
                       choices=valid_combinations)
    parser.add_argument('--list_combinations', action='store_true',
                       help='List available feature combinations and exit')

    # Curriculum learning
    parser.add_argument('--curriculum_stage', type=int, choices=[1, 2, 3],
                       help='Train specific stage')
    parser.add_argument('--auto_curriculum', action='store_true',
                       help='Run full curriculum (all stages)')
    parser.add_argument('--start_stage', type=int, default=1, choices=[1, 2, 3])

    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=100_000)
    parser.add_argument('--data_root', type=str, default='processed_data/')
    parser.add_argument('--train_start_date', type=str, default='2024-06-06')
    parser.add_argument('--train_end_date', type=str, default='2025-03-06')
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--model_dir', type=str, default='models/ppo_mlp')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # List combinations if requested
    if args.list_combinations:
        FeatureSelector().print_available_combinations()
        return

    # Setup
    feature_selector = FeatureSelector()
    selected_features = feature_selector.get_features_for_combination(args.feature_combination)
    logger.info(f"Selected {len(selected_features)} features for '{args.feature_combination}'")

    model_dir = Path(args.model_dir) / args.feature_combination.replace('+', '_')
    model_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir = str(model_dir)

    # Wandb
    if args.use_wandb and wandb:
        run_name = f"ppo_mlp_{args.feature_combination}_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project="finagent-ppo-mlp", name=run_name, config=vars(args))

    # Train
    if args.auto_curriculum:
        logger.info("Running full curriculum training")
        curriculum = CurriculumConfig()
        model = None
        for stage_num in range(args.start_stage, 4):
            stage_config = curriculum.get_stage(stage_num)
            model = train_stage(stage_num, stage_config, selected_features, args, model)
    elif args.curriculum_stage:
        logger.info(f"Training curriculum stage {args.curriculum_stage}")
        curriculum = CurriculumConfig()
        stage_config = curriculum.get_stage(args.curriculum_stage)
        train_stage(args.curriculum_stage, stage_config, selected_features, args)
    else:
        logger.info("Training single model (no curriculum)")
        train_env = DummyVecEnv([create_env(
            args.data_root, args.train_start_date, args.train_end_date,
            selected_features, args.window_size
        )])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        policy_kwargs = dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )

        model = PPO('MlpPolicy', train_env, verbose=1, device=device, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=args.total_timesteps, progress_bar=True)
        model.save(model_dir / f"final_model_{args.feature_combination.replace('+', '_')}.zip")

    if wandb and wandb.run:
        wandb.finish()

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
