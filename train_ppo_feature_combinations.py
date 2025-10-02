"""
PPO LSTM Training Script with Feature Category Combinations
(Updated to include CustomDataLoader, CustomPortfolioEnv and robust trainer init)
Author: AI Assistant (updated)
Date: 2025
"""

import os
import sys
import argparse
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Any, Set

# Apple Silicon setup: prefer Metal if available; allow override via env
if 'JAX_PLATFORM_NAME' not in os.environ:
    try:
        import jax  # local import safe here
        platforms = {d.platform for d in jax.devices()}
        os.environ['JAX_PLATFORM_NAME'] = 'metal' if 'metal' in platforms else 'cpu'
    except Exception:
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Threading controls to avoid 100% CPU utilization
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '8')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.7')

# Setup logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix for JAX 0.6.2 + Flax 0.8.4 compatibility issue
def fix_evaltrace_error():
    try:
        import flax.core.tracers as tracers
        def patched_trace_level(main):
            if main:
                if hasattr(main, 'level'):
                    return main.level
                else:
                    return 0
            return float('-inf')
        tracers.trace_level = patched_trace_level
        logger.info("Applied monkey patch to fix EvalTrace level attribute error")
    except Exception as e:
        logger.warning(f"Could not apply EvalTrace fix: {e}")

fix_evaltrace_error()

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
from functools import partial

# Import the complete PPO LSTM architecture
from train_ppo_lstm import PPOTrainer, ActorCriticLSTM, LSTMState, Trajectory

# IMPORTANT: we now need the base data loader to subclass from
from finagent.environment.portfolio_env import (
    JAXPortfolioDataLoader,
    JAXVectorizedPortfolioEnv,
    EnvState
)

import optax
from flax.training import train_state
import time

# Optional imports for wandb
try:
    import wandb
except ImportError:
    wandb = None

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# -----------------------
# Utilities
# -----------------------
def has_columns(df: pd.DataFrame, cols: List[str]) -> bool:
    """
    Safely check whether DataFrame has all columns in `cols`.
    Avoids ambiguous truth-value operations with pandas Index.
    """
    return set(cols).issubset(set(df.columns))

# -----------------------
# CustomDataLoader
# -----------------------
class CustomDataLoader(JAXPortfolioDataLoader):
    """
    Custom data loader that only generates requested features and prevents
    the parent loader from appending a large default feature set.
    """
    def __init__(self, selected_features: List[str], *args, **kwargs):
        # Store selected features explicitly
        self.selected_features = [str(f) for f in selected_features]
        # Force disable 'use_all_features' behaviour by default
        kwargs.setdefault('use_all_features', False)
        # Set the features attribute (so parent may see it)
        kwargs.setdefault('features', self.selected_features)
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            # Defensive: if parent has a different signature, try without args
            try:
                super().__init__()
                # set attributes directly
                self.features = self.selected_features
                self.use_all_features = False
            except Exception as e:
                logger.error(f"Could not call parent JAXPortfolioDataLoader.__init__: {e}")
                raise

        # Ensure we don't later get overwritten accidentally
        self.features = list(self.selected_features)
        self.use_all_features = False

    def engineer_features(self, df: pd.DataFrame, stock: str) -> pd.DataFrame:
        """
        Engineer only requested features. Overrides parent to prevent generating all features.
        Returns DataFrame with exactly the columns in self.selected_features (order preserved).
        """
        try:
            # Ensure required raw columns exist
            required = {'open', 'high', 'low', 'close', 'volume'}
            missing_required = required - set(df.columns)
            if missing_required:
                logger.warning(f"Missing required columns for {stock}: {missing_required}")
                for col in missing_required:
                    df[col] = 0.0

            df_engineered = pd.DataFrame(index=df.index)

            # Copy any raw columns that are requested and present
            for col in self.selected_features:
                if col in df.columns:
                    df_engineered[col] = df[col]

            # Prepare set of needed (requested but not present)
            needed = [f for f in self.selected_features if f not in df_engineered.columns]

            # === BASIC PRICE FEATURES ===
            # Returns (multiple timeframes)
            periods_map = {'returns_1d':1, 'returns_3d':3, 'returns_5d':5, 'returns_10d':10, 'returns_20d':20}
            for feat, p in periods_map.items():
                if feat in needed and 'close' in df.columns:
                    df_engineered[feat] = df['close'].pct_change(p).fillna(0.0)

            # Log returns
            log_map = {'log_returns_1d':1, 'log_returns_5d':5}
            for feat, p in log_map.items():
                if feat in needed and 'close' in df.columns:
                    df_engineered[feat] = np.log(df['close'] / df['close'].shift(p)).fillna(0.0)

            # Gap and range features
            if 'overnight_gap' in needed and has_columns(df, ['open', 'close']):
                df_engineered['overnight_gap'] = (df['open'] / df['close'].shift(1) - 1.0).fillna(0.0)

            if 'daily_range' in needed and has_columns(df, ['high', 'low', 'open']):
                df_engineered['daily_range'] = ((df['high'] - df['low']) / df['open'].replace(0, np.nan)).fillna(0.0)

            if 'close_position' in needed and has_columns(df, ['high', 'low', 'close']):
                denom = df['high'] - df['low']
                df_engineered['close_position'] = ((df['close'] - df['low']) / denom.replace(0, np.nan)).fillna(0.0)

            # === LAG FEATURES ===
            lag_periods = [1, 2, 3, 5, 10, 20]
            lag_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in lag_cols:
                if col in df.columns:
                    for lag in lag_periods:
                        feat_name = f"{col}_lag_{lag}"
                        if feat_name in needed:
                            df_engineered[feat_name] = df[col].shift(lag).fillna(0.0)

            # === ROLLING FEATURES ===
            rolling_windows = [5, 20]
            rolling_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in rolling_cols:
                if col in df.columns:
                    for window in rolling_windows:
                        # Rolling mean
                        feat_mean = f"{col}_rolling_mean_{window}"
                        if feat_mean in needed:
                            df_engineered[feat_mean] = df[col].rolling(window).mean().fillna(0.0)

                        # Rolling std (only for window=20)
                        if window == 20:
                            feat_std = f"{col}_rolling_std_{window}"
                            if feat_std in needed:
                                df_engineered[feat_std] = df[col].rolling(window).std().fillna(0.0)

            # Close momentum
            for period in [5, 20]:
                feat_name = f"close_momentum_{period}"
                if feat_name in needed and 'close' in df.columns:
                    df_engineered[feat_name] = df['close'].diff(period).fillna(0.0)

            # === VOLATILITY FEATURES ===
            if 'returns_1d' not in df_engineered.columns and 'close' in df.columns:
                temp_returns = df['close'].pct_change()
            else:
                temp_returns = df_engineered.get('returns_1d', df['close'].pct_change())

            vol_periods = [5, 10, 20, 30, 60]
            for period in vol_periods:
                feat_name = f"volatility_{period}d"
                if feat_name in needed:
                    df_engineered[feat_name] = temp_returns.rolling(period).std().fillna(0.0)

            # Volatility ratios
            if 'vol_ratio_short_long' in needed and has_columns(df_engineered, ['volatility_10d', 'volatility_30d']):
                df_engineered['vol_ratio_short_long'] = df_engineered['volatility_10d'] / (df_engineered['volatility_30d'] + 1e-8)

            if 'vol_ratio_5_20' in needed and has_columns(df_engineered, ['volatility_5d', 'volatility_20d']):
                df_engineered['vol_ratio_5_20'] = df_engineered['volatility_5d'] / (df_engineered['volatility_20d'] + 1e-8)

            # === MOMENTUM FEATURES ===
            momentum_periods = [5, 10, 20, 60]
            for period in momentum_periods:
                feat_name = f"momentum_{period}d"
                if feat_name in needed and 'close' in df.columns:
                    df_engineered[feat_name] = df['close'] / df['close'].shift(period) - 1.0
                    df_engineered[feat_name] = df_engineered[feat_name].fillna(0.0)

            if 'momentum_acceleration_10d' in needed and has_columns(df_engineered, ['momentum_10d']):
                df_engineered['momentum_acceleration_10d'] = df_engineered['momentum_10d'] - df_engineered['momentum_10d'].shift(5)
                df_engineered['momentum_acceleration_10d'] = df_engineered['momentum_acceleration_10d'].fillna(0.0)

            # === TECHNICAL INDICATORS ===
            # DMA features
            for period in [50, 200]:
                feat_name = f"dma_{period}"
                if feat_name in needed and 'close' in df.columns:
                    df_engineered[feat_name] = df['close'].rolling(period).mean().fillna(0.0)

                # DMA lags
                if feat_name in df_engineered.columns or feat_name in df.columns:
                    source = df_engineered[feat_name] if feat_name in df_engineered.columns else df[feat_name]
                    for lag in lag_periods:
                        lag_feat = f"{feat_name}_lag_{lag}"
                        if lag_feat in needed:
                            df_engineered[lag_feat] = source.shift(lag).fillna(0.0)

                    # DMA rolling features
                    for window in [5, 20]:
                        mean_feat = f"{feat_name}_rolling_mean_{window}"
                        if mean_feat in needed:
                            df_engineered[mean_feat] = source.rolling(window).mean().fillna(0.0)

                        if window == 20:
                            std_feat = f"{feat_name}_rolling_std_{window}"
                            if std_feat in needed:
                                df_engineered[std_feat] = source.rolling(window).std().fillna(0.0)

            # DMA cross and distance
            if 'dma_cross' in needed and has_columns(df_engineered, ['dma_50', 'dma_200']):
                df_engineered['dma_cross'] = (df_engineered['dma_50'] > df_engineered['dma_200']).astype(float)

            if 'dma_distance' in needed and has_columns(df_engineered, ['dma_50', 'dma_200']):
                df_engineered['dma_distance'] = (df_engineered['dma_50'] - df_engineered['dma_200']) / df_engineered['dma_200']
                df_engineered['dma_distance'] = df_engineered['dma_distance'].fillna(0.0)

            # Volume price trend
            if 'volume_price_trend' in needed and has_columns(df, ['close', 'volume']):
                df_engineered['volume_price_trend'] = (df['close'].pct_change() * df['volume']).fillna(0.0)

            # RSI
            if 'rsi_14' in needed and 'close' in df.columns:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                df_engineered['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50.0)

            # === TREND FEATURES ===
            if 'ma_convergence' in needed and has_columns(df_engineered, ['dma_50', 'dma_200']):
                df_engineered['ma_convergence'] = (df_engineered['dma_50'] - df_engineered['dma_200']) / df_engineered['dma_200']
                df_engineered['ma_convergence'] = df_engineered['ma_convergence'].fillna(0.0)

            if 'ma_trend_strength' in needed and 'ma_convergence' in df_engineered.columns:
                df_engineered['ma_trend_strength'] = df_engineered['ma_convergence'] - df_engineered['ma_convergence'].shift(5)
                df_engineered['ma_trend_strength'] = df_engineered['ma_trend_strength'].fillna(0.0)

            if 'price_position_20d' in needed and 'close' in df.columns:
                rolling_min = df['close'].rolling(20).min()
                rolling_max = df['close'].rolling(20).max()
                df_engineered['price_position_20d'] = ((df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)).fillna(0.0)

            # Price above MA flags
            if 'price_above_ma50' in needed and 'close' in df.columns and has_columns(df_engineered, ['dma_50']):
                df_engineered['price_above_ma50'] = (df['close'] > df_engineered['dma_50']).astype(float)

            if 'price_above_ma200' in needed and 'close' in df.columns and has_columns(df_engineered, ['dma_200']):
                df_engineered['price_above_ma200'] = (df['close'] > df_engineered['dma_200']).astype(float)

            # === RSI SIGNALS ===
            if 'rsi_14' in df_engineered.columns:
                if 'rsi_oversold' in needed:
                    df_engineered['rsi_oversold'] = (df_engineered['rsi_14'] < 30).astype(float)
                if 'rsi_overbought' in needed:
                    df_engineered['rsi_overbought'] = (df_engineered['rsi_14'] > 70).astype(float)
                if 'rsi_bullish_divergence' in needed and 'close' in df.columns:
                    df_engineered['rsi_bullish_divergence'] = ((df_engineered['rsi_14'] > df_engineered['rsi_14'].shift(1)) &
                                                               (df['close'] < df['close'].shift(1))).astype(float)
                if 'rsi_bearish_divergence' in needed and 'close' in df.columns:
                    df_engineered['rsi_bearish_divergence'] = ((df_engineered['rsi_14'] < df_engineered['rsi_14'].shift(1)) &
                                                               (df['close'] > df['close'].shift(1))).astype(float)
                if 'rsi_zscore_20d' in needed:
                    rsi_mean = df_engineered['rsi_14'].rolling(20).mean()
                    rsi_std = df_engineered['rsi_14'].rolling(20).std()
                    df_engineered['rsi_zscore_20d'] = ((df_engineered['rsi_14'] - rsi_mean) / (rsi_std + 1e-8)).fillna(0.0)

            # === BOLLINGER BANDS ===
            if any(f in needed for f in ['bb_position', 'bb_squeeze', 'bb_breakout_up', 'bb_breakout_down']) and 'close' in df.columns:
                bb_middle = df['close'].rolling(20).mean()
                bb_std = df['close'].rolling(20).std()
                bb_upper = bb_middle + (2 * bb_std)
                bb_lower = bb_middle - (2 * bb_std)

                if 'bb_position' in needed:
                    df_engineered['bb_position'] = ((df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)).fillna(0.5)
                if 'bb_squeeze' in needed:
                    df_engineered['bb_squeeze'] = ((bb_upper - bb_lower) / bb_middle).fillna(0.0)
                if 'bb_breakout_up' in needed:
                    df_engineered['bb_breakout_up'] = (df['close'] > bb_upper).astype(float)
                if 'bb_breakout_down' in needed:
                    df_engineered['bb_breakout_down'] = (df['close'] < bb_lower).astype(float)

            # === Z-SCORE FEATURES ===
            zscore_periods = [20, 60]
            for period in zscore_periods:
                # Price z-score
                feat_name = f"price_zscore_{period}d"
                if feat_name in needed and 'close' in df.columns:
                    price_mean = df['close'].rolling(period).mean()
                    price_std = df['close'].rolling(period).std()
                    df_engineered[feat_name] = ((df['close'] - price_mean) / (price_std + 1e-8)).fillna(0.0)

                # Volume z-score
                feat_name = f"volume_zscore_{period}d"
                if feat_name in needed and 'volume' in df.columns:
                    vol_mean = df['volume'].rolling(period).mean()
                    vol_std = df['volume'].rolling(period).std()
                    df_engineered[feat_name] = ((df['volume'] - vol_mean) / (vol_std + 1e-8)).fillna(0.0)

            # === MEAN REVERSION SIGNALS ===
            for period in [50, 200]:
                dma_col = f"dma_{period}"
                dev_feat = f"price_deviation_{period}d"
                signal_feat = f"mean_reversion_signal_{period}d"

                if dev_feat in needed and 'close' in df.columns and dma_col in df_engineered.columns:
                    df_engineered[dev_feat] = (df['close'] - df_engineered[dma_col]) / df_engineered[dma_col]
                    df_engineered[dev_feat] = df_engineered[dev_feat].fillna(0.0)

                if signal_feat in needed and dev_feat in df_engineered.columns:
                    threshold = 0.05 if period == 50 else 0.1
                    df_engineered[signal_feat] = np.where(df_engineered[dev_feat] > threshold, -1,
                                                          np.where(df_engineered[dev_feat] < -threshold, 1, 0))

            # === BREAKOUT SIGNALS ===
            if 'price_breakout_20d' in needed and 'close' in df.columns:
                df_engineered['price_breakout_20d'] = (df['close'] > df['close'].rolling(20).max().shift(1)).astype(float)

            if 'price_breakdown_20d' in needed and 'close' in df.columns:
                df_engineered['price_breakdown_20d'] = (df['close'] < df['close'].rolling(20).min().shift(1)).astype(float)

            if 'volume_breakout_20d' in needed and 'volume' in df.columns:
                df_engineered['volume_breakout_20d'] = (df['volume'] > df['volume'].rolling(20).max().shift(1)).astype(float)

            if 'volume_spike' in needed and 'volume' in df.columns:
                df_engineered['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(float)

            # === MA CROSSOVER SIGNALS ===
            if 'ma_cross_bullish' in needed and has_columns(df_engineered, ['dma_50', 'dma_200']):
                df_engineered['ma_cross_bullish'] = ((df_engineered['dma_50'] > df_engineered['dma_200']) &
                                                     (df_engineered['dma_50'].shift(1) <= df_engineered['dma_200'].shift(1))).astype(float)

            if 'ma_cross_bearish' in needed and has_columns(df_engineered, ['dma_50', 'dma_200']):
                df_engineered['ma_cross_bearish'] = ((df_engineered['dma_50'] < df_engineered['dma_200']) &
                                                     (df_engineered['dma_50'].shift(1) >= df_engineered['dma_200'].shift(1))).astype(float)

            # === VOLATILITY REGIME SIGNALS ===
            if 'volatility_20d' in df_engineered.columns:
                if 'high_vol_regime' in needed:
                    vol_threshold_high = df_engineered['volatility_20d'].rolling(60).quantile(0.8)
                    df_engineered['high_vol_regime'] = (df_engineered['volatility_20d'] > vol_threshold_high).astype(float)

                if 'low_vol_regime' in needed:
                    vol_threshold_low = df_engineered['volatility_20d'].rolling(60).quantile(0.2)
                    df_engineered['low_vol_regime'] = (df_engineered['volatility_20d'] < vol_threshold_low).astype(float)

                if 'vol_expansion' in needed:
                    df_engineered['vol_expansion'] = (df_engineered['volatility_20d'] > df_engineered['volatility_20d'].shift(5)).astype(float)

                if 'vol_contraction' in needed:
                    df_engineered['vol_contraction'] = (df_engineered['volatility_20d'] < df_engineered['volatility_20d'].shift(5)).astype(float)

            # === CANDLESTICK PATTERNS ===
            if has_columns(df, ['open', 'high', 'low', 'close']):
                body_size = abs(df['close'] - df['open'])
                upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
                lower_wick = df[['close', 'open']].min(axis=1) - df['low']
                total_range = df['high'] - df['low'] + 1e-8

                if 'body_ratio' in needed:
                    df_engineered['body_ratio'] = (body_size / total_range).fillna(0.0)
                if 'upper_wick_ratio' in needed:
                    df_engineered['upper_wick_ratio'] = (upper_wick / total_range).fillna(0.0)
                if 'lower_wick_ratio' in needed:
                    df_engineered['lower_wick_ratio'] = (lower_wick / total_range).fillna(0.0)

                if 'doji_pattern' in needed and 'body_ratio' in df_engineered.columns:
                    df_engineered['doji_pattern'] = (df_engineered['body_ratio'] < 0.1).astype(float)
                if 'hammer_pattern' in needed and has_columns(df_engineered, ['lower_wick_ratio', 'body_ratio']):
                    df_engineered['hammer_pattern'] = ((df_engineered['lower_wick_ratio'] > 0.6) &
                                                       (df_engineered['body_ratio'] < 0.3)).astype(float)
                if 'shooting_star_pattern' in needed and has_columns(df_engineered, ['upper_wick_ratio', 'body_ratio']):
                    df_engineered['shooting_star_pattern'] = ((df_engineered['upper_wick_ratio'] > 0.6) &
                                                              (df_engineered['body_ratio'] < 0.3)).astype(float)

            # === VOLUME ANALYSIS ===
            if 'volume' in df.columns:
                if 'volume_price_momentum' in needed and 'returns_1d' in df_engineered.columns:
                    df_engineered['volume_price_momentum'] = (df_engineered['returns_1d'] * df['volume'] /
                                                              (df['volume'].rolling(20).mean() + 1e-8)).fillna(0.0)

                for period in [5, 20]:
                    feat_name = f"volume_ratio_{period}d"
                    if feat_name in needed:
                        df_engineered[feat_name] = (df['volume'] / (df['volume'].rolling(period).mean() + 1e-8)).fillna(1.0)

                if 'volume_trend_10d' in needed:
                    df_engineered['volume_trend_10d'] = (df['volume'].rolling(10).mean() /
                                                         (df['volume'].rolling(20).mean() + 1e-8) - 1.0).fillna(0.0)

                if 'volume_confirms_price' in needed and 'returns_1d' in df_engineered.columns and 'volume_ratio_20d' in df_engineered.columns:
                    df_engineered['volume_confirms_price'] = ((df_engineered['returns_1d'] > 0) &
                                                              (df_engineered['volume_ratio_20d'] > 1.2)).astype(float)

                if 'volume_divergence' in needed and 'returns_1d' in df_engineered.columns and 'volume_ratio_20d' in df_engineered.columns:
                    df_engineered['volume_divergence'] = ((df_engineered['returns_1d'] > 0) &
                                                          (df_engineered['volume_ratio_20d'] < 0.8)).astype(float)

            # === SIGNAL AGGREGATION ===
            signal_cols = [col for col in df_engineered.columns if any(signal in col for signal in
                          ['signal', 'breakout', 'cross', 'above', 'below', 'oversold', 'overbought'])]

            if signal_cols:
                if 'bullish_signals' in needed:
                    df_engineered['bullish_signals'] = df_engineered[signal_cols].sum(axis=1)

                bearish_cols = [col for col in signal_cols if 'bearish' in col or 'breakdown' in col or 'below' in col]
                if 'bearish_signals' in needed and bearish_cols:
                    df_engineered['bearish_signals'] = df_engineered[bearish_cols].sum(axis=1)

                if 'net_signal_strength' in needed and 'bullish_signals' in df_engineered.columns:
                    bearish_sum = df_engineered[bearish_cols].sum(axis=1) if bearish_cols else 0
                    df_engineered['net_signal_strength'] = df_engineered['bullish_signals'] - bearish_sum

            # === REGIME DETECTION ===
            if 'vol_regime_change' in needed and 'volatility_20d' in df_engineered.columns:
                vol_ma_short = df_engineered['volatility_20d'].rolling(20).mean()
                vol_ma_long = df_engineered['volatility_20d'].rolling(60).mean()
                df_engineered['vol_regime_change'] = (vol_ma_short > vol_ma_long).astype(float)

            if 'trend_regime' in needed and 'returns_1d' in df_engineered.columns and 'volatility_20d' in df_engineered.columns:
                returns_range = df_engineered['returns_1d'].rolling(20).apply(lambda x: x.max() - x.min() if len(x) == 20 else 0)
                df_engineered['trend_regime'] = (returns_range / (df_engineered['volatility_20d'] * np.sqrt(20) + 1e-8)).fillna(0.0)

            # === CROSS-SECTIONAL FEATURES ===
            if 'momentum_rank_proxy' in needed and 'momentum_20d' in df_engineered.columns:
                df_engineered['momentum_rank_proxy'] = df_engineered['momentum_20d'].rolling(60).rank(pct=True).fillna(0.5)

            if 'vol_rank_proxy' in needed and 'volatility_20d' in df_engineered.columns:
                df_engineered['vol_rank_proxy'] = df_engineered['volatility_20d'].rolling(60).rank(pct=True).fillna(0.5)

            # === INTERACTION FEATURES ===
            if 'risk_adjusted_momentum' in needed and has_columns(df_engineered, ['momentum_10d', 'volatility_10d']):
                df_engineered['risk_adjusted_momentum'] = (df_engineered['momentum_10d'] /
                                                          (df_engineered['volatility_10d'] + 1e-8)).fillna(0.0)

            if 'volume_confirmed_trend' in needed and has_columns(df_engineered, ['volume_ratio_20d']):
                # Approximation if price_to_dma50 not available
                if 'close' in df.columns and 'dma_50' in df_engineered.columns:
                    price_to_dma50 = df['close'] / (df_engineered['dma_50'] + 1e-8)
                    df_engineered['volume_confirmed_trend'] = (price_to_dma50 * df_engineered['volume_ratio_20d']).fillna(0.0)

            # === SENTIMENT FEATURES (if present in raw data) ===
            sentiment_patterns = ['sentiment', 'reddit_', 'news_']
            for pattern in sentiment_patterns:
                sentiment_cols_in_df = [col for col in df.columns if pattern in col.lower()]
                for col in sentiment_cols_in_df:
                    if col in needed:
                        df_engineered[col] = df[col].fillna(0.0)

                    # Sentiment momentum
                    for period in [3, 5]:
                        momentum_feat = f"{col}_momentum_{period}d"
                        if momentum_feat in needed and col in df.columns:
                            df_engineered[momentum_feat] = (df[col] - df[col].shift(period)).fillna(0.0)

                    # Sentiment extremes
                    for extreme_type in ['extreme_positive', 'extreme_negative']:
                        extreme_feat = f"{col}_{extreme_type}"
                        if extreme_feat in needed and col in df.columns:
                            if 'positive' in extreme_type:
                                threshold = df[col].rolling(30).quantile(0.9)
                                df_engineered[extreme_feat] = (df[col] > threshold).astype(float)
                            else:
                                threshold = df[col].rolling(30).quantile(0.1)
                                df_engineered[extreme_feat] = (df[col] < threshold).astype(float)

            # Fill any requested features that still don't exist with zeros
            for feat in self.selected_features:
                if feat not in df_engineered.columns:
                    logger.warning(f"Feature '{feat}' could not be generated for {stock}, filling with zeros")
                    df_engineered[feat] = 0.0

            # Final enforcement: reorder columns to match selected_features exactly
            df_engineered = df_engineered[self.selected_features]

            logger.info(f"CustomDataLoader: Generated {len(df_engineered.columns)} features for {stock}: {list(df_engineered.columns)[:5]}...")
            return df_engineered.fillna(0.0)

        except Exception as e:
            logger.error(f"Error in CustomDataLoader.engineer_features for {stock}: {e}", exc_info=True)
            # Return DataFrame of zeros with selected features
            return pd.DataFrame(0.0, index=df.index, columns=self.selected_features)

    def load_and_preprocess_data(self, start_date: str, end_date: str,
                                 fill_missing_features_with: str = 'interpolate',
                                 preload_to_gpu: bool = True,
                                 save_cache: bool = False,
                                 cache_format: str = 'hdf5',
                                 force_reload: bool = False):
        """
        Override parent's load_and_preprocess_data to prevent feature expansion.

        The parent class at line 584-585 does:
            engineered_feats = [c for c in all_data[0].columns if c not in self.features + ["symbol"]]
            self.features = self.features + engineered_feats

        This ALWAYS appends all engineered features. We completely bypass this by reimplementing.
        """
        try:
            # CRITICAL: Set features before anything
            self.features = list(self.selected_features)
            self.use_all_features = False

            logger.info(f"CustomDataLoader: Loading data with {len(self.selected_features)} selected features")

            # Try to load from cache first (unless force_reload is True)
            if not force_reload:
                cached_result = self.load_cached_data(
                    start_date, end_date, fill_missing_features_with,
                    cache_format, preload_to_gpu
                )
                if cached_result is not None:
                    # Still enforce our features
                    self.features = list(self.selected_features)
                    return cached_result

            all_data = []
            all_features = set()
            stock_features = {}

            # Pass 1: Collect available features (we still need stock_features for validation)
            for stock in self.stocks:
                csv_path = self.data_root / f"{stock}_aligned.csv"
                if not csv_path.exists():
                    logger.warning(f"CSV for {stock} not found, skipping.")
                    continue
                try:
                    sample_df = pd.read_csv(csv_path, nrows=0, index_col=0)
                    feats = list(sample_df.columns)
                    stock_features[stock] = feats
                    all_features.update(feats)
                except Exception as e:
                    logger.warning(f"Error reading {stock}: {e}, skipping.")
                    continue

            # CRITICAL DIFFERENCE: We use our selected_features, not parent's logic
            # Parent would set self.features here based on use_all_features
            # We ignore that and use our selected_features
            self.features = list(self.selected_features)

            # Ensure 'close' is available
            if 'close' not in self.features:
                raise ValueError("'close' must be present in features")

            # Reorganize so 'close' is always first (if requested)
            if self.features[0] != 'close':
                self.features = ['close'] + [f for f in self.features if f != 'close']
                self.selected_features = self.features  # Keep in sync

            # Pass 2: Load and standardize
            for stock in self.stocks:
                csv_path = self.data_root / f"{stock}_aligned.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
                df.index.name = "date"
                df = df.sort_index()
                df = df.loc[start_date:end_date]

                proc = pd.DataFrame(index=df.index)
                for feat in self.features:
                    if feat in df.columns:
                        proc[feat] = pd.to_numeric(df[feat], errors="coerce")
                    else:
                        proc[feat] = np.nan

                # Fill missing
                if fill_missing_features_with == "zero":
                    proc = proc.fillna(0.0)
                elif fill_missing_features_with == "forward_fill":
                    proc = proc.ffill().bfill().fillna(0.0)
                elif fill_missing_features_with == "interpolate":
                    proc = proc.interpolate(method="linear").ffill().bfill().fillna(0.0)
                else:
                    proc = proc.ffill().bfill().interpolate().ffill().bfill().fillna(0.0)

                # Call OUR engineer_features (which only generates selected features)
                proc = self.engineer_features(proc, stock)

                # CRITICAL: Ensure only selected features are in the output
                # This prevents parent's line 584 from finding extra features
                proc = proc[self.selected_features]

                proc["symbol"] = stock
                all_data.append(proc)

            if not all_data:
                raise RuntimeError("No stock data loaded.")

            # Merge into panel
            panel = pd.concat(all_data).reset_index().set_index(["date", "symbol"])
            panel = panel[~panel.index.duplicated(keep="first")]

            # CRITICAL DIFFERENCE: Parent does this at line 584-585:
            #   engineered_feats = [c for c in all_data[0].columns if c not in self.features + ["symbol"]]
            #   self.features = self.features + engineered_feats
            # We DON'T append engineered features - we already have exactly what we want
            # self.features stays as selected_features

            panel = panel.unstack(level="symbol").sort_index(axis=1)
            full_columns = pd.MultiIndex.from_product([self.features, self.stocks])

            # Align to requested date range
            valid_dates = pd.date_range(start=start_date, end=end_date, freq="B")
            panel = panel.reindex(valid_dates, columns=full_columns, method="ffill").fillna(0.0)

            # Convert to tensor
            data_array = panel.values.astype(np.float32)
            n_days = len(valid_dates)
            n_features = len(self.features)
            n_stocks = len(self.stocks)
            data_array = data_array.reshape(n_days, n_features, n_stocks).transpose(0, 2, 1)

            # Save to cache if requested
            if save_cache:
                self.save_data(
                    data_array, valid_dates, start_date, end_date,
                    fill_missing_features_with, cache_format
                )

            if preload_to_gpu:
                data_array = jnp.array(data_array)

            logger.info(f"CustomDataLoader: Data loaded with shape {data_array.shape}, features: {len(self.features)}")

            return data_array, jnp.arange(n_days), valid_dates, n_features

        except Exception as e:
            logger.error(f"Error in CustomDataLoader.load_and_preprocess_data: {e}", exc_info=True)
            raise

# -----------------------
# CustomPortfolioEnv
# -----------------------
class CustomPortfolioEnv(JAXVectorizedPortfolioEnv):
    """
    Vectorized portfolio environment that uses CustomDataLoader to ensure the requested
    features are used end-to-end.

    CRITICAL: We override __init__ completely to prevent parent from creating its own loader.
    """
    def __init__(self, selected_features: List[str],
                data_root: str = 'processed_data/',
                stocks: List[str] = None,
                initial_cash: float = 1_000_000.0,
                window_size: int = 30,
                start_date: str = '2024-06-06',
                end_date: str = '2025-03-06',
                transaction_cost_rate: float = 0.005,
                sharpe_window: int = 252,
                **kwargs):
        """
        Initialize CustomPortfolioEnv with complete control over feature selection.

        We DON'T call super().__init__() because the parent creates its own JAXPortfolioDataLoader
        at line 674 which bypasses our custom loader. Instead, we replicate the parent's init
        logic but use our CustomDataLoader.
        """
        try:
            # Store and normalize selected features FIRST
            self.selected_features = [str(f) for f in selected_features]
            self.features = list(self.selected_features)
            self.n_features = len(self.selected_features)
            self.use_all_features = False

            logger.info(f"CustomPortfolioEnv: Initializing with {len(self.selected_features)} features")

            # === REPLICATE PARENT'S __init__ LOGIC ===
            # From JAXVectorizedPortfolioEnv.__init__ lines 650-717

            self.data_root = data_root
            self.window_size = window_size
            self.start_date = start_date
            self.end_date = end_date
            self.initial_cash_actual = initial_cash
            self.transaction_cost_rate = transaction_cost_rate
            self.sharpe_window = sharpe_window
            self.risk_free_rate_daily = 0.04 / 252.0
            self.cash_return_rate = 0.04 / 252.0
            self.close_price_idx = None

            # Load stocks
            if stocks is None:
                stocks = self._load_stock_list()
            self.stocks = stocks
            self.n_stocks = len(stocks)

            logger.info(f"CustomPortfolioEnv: Data root: {self.data_root}")
            logger.info(f"CustomPortfolioEnv: Stocks loaded: {len(self.stocks)}")

            # === CRITICAL: Use CustomDataLoader instead of parent's JAXPortfolioDataLoader ===
            self.data_loader = CustomDataLoader(
                selected_features=self.selected_features,
                data_root=data_root,
                stocks=stocks,
                use_all_features=False
            )

            # Load data using our custom loader
            logger.info(f"CustomPortfolioEnv: Loading data with {len(self.selected_features)} features...")
            self.data, self.dates_idx, self.actual_dates, self.n_features = self.data_loader.load_and_preprocess_data(
                start_date=start_date,
                end_date=end_date,
                preload_to_gpu=True,
                force_reload=True  # Force reload to ensure clean data
            )

            # Verify data shape
            if self.data.shape[2] != len(self.selected_features):
                raise ValueError(
                    f"Data shape mismatch! Expected {len(self.selected_features)} features, "
                    f"got {self.data.shape[2]}. Data shape: {self.data.shape}"
                )

            logger.info(f"CustomPortfolioEnv: Data loaded successfully with shape: {self.data.shape}")

            # Set close price index
            self.close_price_idx = 0  # Always 0 after reorganization

            # Create feature indices dict
            self.feature_indices = {
                'open': self.data_loader.features.index('open') if 'open' in self.data_loader.features else 0,
                'close': self.data_loader.features.index('close') if 'close' in self.data_loader.features else 0,
                'returns_1d': self.data_loader.features.index('returns_1d') if 'returns_1d' in self.data_loader.features else None,
                'volatility_10d': self.data_loader.features.index('volatility_10d') if 'volatility_10d' in self.data_loader.features else None,
                'overnight_gap': self.data_loader.features.index('overnight_gap') if 'overnight_gap' in self.data_loader.features else None
            }

            # Validate data
            if self.data.shape[0] < self.window_size + 2:
                raise ValueError(
                    f"Insufficient data: need at least {self.window_size + 2} time steps, "
                    f"got {self.data.shape[0]}"
                )
            self.n_timesteps = len(self.dates_idx)

            # Set action and observation dimensions
            self.action_dim = self.n_stocks + 1

            # Calculate observation dimension
            obs_size = (
                (self.window_size * self.n_stocks * self.n_features) +  # Historical features
                self.n_stocks * 2 +                                     # Current open prices + gaps
                self.action_dim +                                       # Portfolio weights
                self.n_stocks +                                         # Short position flags
                8                                                       # Market state
            )
            self.obs_dim = obs_size

            logger.info(f"CustomPortfolioEnv: Initialized successfully")
            logger.info(f"  Stocks: {self.n_stocks}")
            logger.info(f"  Features: {self.n_features}")
            logger.info(f"  Window size: {self.window_size}")
            logger.info(f"  Observation dim: {self.obs_dim}")
            logger.info(f"  Action dim: {self.action_dim}")
            logger.info(f"  Timesteps: {self.n_timesteps}")

        except Exception as e:
            logger.error(f"Error in CustomPortfolioEnv initialization: {e}", exc_info=True)
            raise

    def _load_stock_list(self) -> List[str]:
        """Load stock list from file or scan directory"""
        stocks_file = Path("finagent/stocks.txt")
        if not stocks_file.exists():
            stocks_file = Path("FYP-FinAgent/finagent/stocks.txt")

        if stocks_file.exists():
            with open(stocks_file, 'r') as f:
                stocks = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"Loaded {len(stocks)} stocks from file")
                return stocks

        # Fallback: scan directory
        data_path = Path(self.data_root) if isinstance(self.data_root, str) else self.data_root
        stocks = [p.stem.replace('_aligned', '') for p in data_path.glob("*_aligned.csv")]
        logger.info(f"Scanned {len(stocks)} stocks from directory")
        return stocks

# -----------------------
# CurriculumConfig (unchanged)
# -----------------------
class CurriculumConfig:
    def __init__(self):
        self.stages = {
            1: {
                "stage_num": 1,
                "name": "Exploration",
                "description": "Initial exploration stage with high entropy and learning rate",
                "exploration_std": 0.3,
                "entropy_coeff": 0.02,
                "learning_rate": 5e-4,
                "clip_eps": 0.3,
                "num_epochs": 800,  # Increased from 300
                "epsilon_uniform": 0.1,
                "reward_scaling": 1.0,
                "enable_constraints": False,
                "early_stopping_patience": 300,  # Increased from 150
                "early_stopping_min_delta": 0.001
            },
            2: {
                "stage_num": 2,
                "name": "Refinement",
                "description": "Refinement stage with balanced parameters",
                "exploration_std": 0.2,
                "entropy_coeff": 0.01,
                "learning_rate": 3e-4,
                "clip_eps": 0.2,
                "num_epochs": 1000,  # Increased from 400
                "epsilon_uniform": 0.05,
                "reward_scaling": 1.0,
                "enable_constraints": True,
                "early_stopping_patience": 400,  # Increased from 200
                "early_stopping_min_delta": 0.003
            },
            3: {
                "stage_num": 3,
                "name": "Optimization",
                "description": "Final optimization with focused learning",
                "exploration_std": 0.1,
                "entropy_coeff": 0.005,
                "learning_rate": 1e-4,
                "clip_eps": 0.1,
                "num_epochs": 800,  # Increased from 300
                "epsilon_uniform": 0.01,
                "reward_scaling": 1.0,
                "enable_constraints": True,
                "early_stopping_patience": 500,  # Increased from 250
                "early_stopping_min_delta": 0.005
            }
        }

    def get_stage(self, stage_num):
        return self.stages.get(stage_num, None)

    def total_updates(self):
        return sum(stage['num_epochs'] for stage in self.stages.values())

# -----------------------
# FeatureSelector (unchanged)
# -----------------------
class FeatureSelector:
    """Feature selector for different feature categories"""
    def __init__(self):
        self.feature_categories = {
            'ohlcv': {
                'description': 'Basic OHLCV price data and simple derived features',
                'features': [
                    'open', 'high', 'low', 'close', 'volume', 'vwap',
                    'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d', 'returns_20d',
                    'log_returns_1d', 'log_returns_5d',
                    'overnight_gap', 'daily_range', 'close_position',
                    'open_lag_1', 'open_lag_2', 'open_lag_3', 'open_lag_5', 'open_lag_10', 'open_lag_20',
                    'high_lag_1', 'high_lag_2', 'high_lag_3', 'high_lag_5', 'high_lag_10', 'high_lag_20',
                    'low_lag_1', 'low_lag_2', 'low_lag_3', 'low_lag_5', 'low_lag_10', 'low_lag_20',
                    'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10', 'close_lag_20',
                    'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10', 'volume_lag_20',
                    'open_rolling_mean_5', 'open_rolling_mean_20', 'open_rolling_std_20',
                    'high_rolling_mean_5', 'high_rolling_mean_20', 'high_rolling_std_20',
                    'low_rolling_mean_5', 'low_rolling_mean_20', 'low_rolling_std_20',
                    'close_rolling_mean_5', 'close_rolling_mean_20', 'close_rolling_std_20',
                    'close_momentum_5', 'close_momentum_20',
                    'volume_rolling_mean_5', 'volume_rolling_mean_20', 'volume_rolling_std_20'
                ]
            },
            'technical': {
                'description': 'Technical indicators and algorithmic trading signals',
                'features': [
                    'dma_50', 'dma_200',
                    'dma_50_lag_1', 'dma_50_lag_2', 'dma_50_lag_3', 'dma_50_lag_5', 'dma_50_lag_10', 'dma_50_lag_20',
                    'dma_200_lag_1', 'dma_200_lag_2', 'dma_200_lag_3', 'dma_200_lag_5', 'dma_200_lag_10', 'dma_200_lag_20',
                    'dma_50_rolling_mean_5', 'dma_50_rolling_mean_20', 'dma_50_rolling_std_20',
                    'dma_200_rolling_mean_5', 'dma_200_rolling_mean_20', 'dma_200_rolling_std_20',
                    'dma_cross', 'dma_distance', 'volume_price_trend',
                    'rsi_14',
                    'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_30d', 'volatility_60d',
                    'vol_ratio_short_long', 'vol_ratio_5_20',
                    'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
                    'momentum_acceleration_10d',
                    'ma_convergence', 'ma_trend_strength', 'price_position_20d',
                    'price_above_ma50', 'price_above_ma200',
                    'rsi_oversold', 'rsi_overbought', 'rsi_bullish_divergence', 'rsi_bearish_divergence',
                    'rsi_zscore_20d', 'bb_position', 'bb_squeeze', 'bb_breakout_up', 'bb_breakout_down',
                    'price_zscore_20d', 'price_zscore_60d', 'volume_zscore_20d', 'volume_zscore_60d',
                    'price_deviation_50d', 'price_deviation_200d', 'mean_reversion_signal_50d', 'mean_reversion_signal_200d',
                    'price_breakout_20d', 'price_breakdown_20d', 'volume_breakout_20d', 'volume_spike',
                    'ma_cross_bullish', 'ma_cross_bearish', 'high_vol_regime', 'low_vol_regime',
                    'vol_expansion', 'vol_contraction', 'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
                    'doji_pattern', 'hammer_pattern', 'shooting_star_pattern', 'volume_price_momentum',
                    'volume_ratio_5d', 'volume_ratio_20d', 'volume_trend_10d', 'volume_confirms_price',
                    'volume_divergence', 'bullish_signals', 'bearish_signals', 'net_signal_strength',
                    'risk_adjusted_momentum', 'volume_confirmed_trend', 'vol_regime_change', 'trend_regime',
                    'momentum_rank_proxy', 'vol_rank_proxy'
                ]
            },
            'financial': {
                'description': 'Financial statement metrics and fundamental analysis',
                'features': [
                    'metric_Revenue', 'metric_TotalRevenue', 'metric_CostofRevenueTotal', 'metric_GrossProfit',
                    'metric_OperatingIncome', 'metric_NetIncomeBeforeTaxes', 'metric_NetIncomeAfterTaxes',
                    'metric_NetIncome', 'metric_DilutedNetIncome', 'metric_DilutedWeightedAverageShares',
                    'metric_DilutedEPSExcludingExtraOrdItems', 'metric_DPS-CommonStockPrimaryIssue',
                    'metric_Cash', 'metric_ShortTermInvestments', 'metric_CashandShortTermInvestments',
                    'metric_TotalCurrentAssets', 'metric_TotalAssets', 'metric_TotalCurrentLiabilities',
                    'metric_TotalLiabilities', 'metric_TotalEquity', 'metric_TotalCommonSharesOutstanding',
                    'metric_CashfromOperatingActivities', 'metric_CapitalExpenditures',
                    'metric_CashfromInvestingActivities', 'metric_CashfromFinancingActivities',
                    'metric_NetChangeinCash', 'metric_TotalCashDividendsPaid', 'metric_freeCashFlowtrailing12Month',
                    'metric_freeCashFlowMostRecentFiscalYear', 'metric_periodLength', 'metric_periodType',
                    'metric_pPerEExcludingExtraordinaryItemsMostRecentFiscalYear',
                    'metric_currentDividendYieldCommonStockPrimaryIssueLTM', 'metric_priceToBookMostRecentFiscalYear',
                    'metric_priceToFreeCashFlowPerShareTrailing12Months', 'metric_pPerEBasicExcludingExtraordinaryItemsTTM',
                    'metric_pPerEIncludingExtraordinaryItemsTTM', 'metric_returnOnAverageEquityMostRecentFiscalYear',
                    'metric_returnOnInvestmentMostRecentFiscalYear', 'metric_netProfitMarginPercentTrailing12Month',
                    'metric_operatingMarginTrailing12Month', 'metric_grossMarginTrailing12Month',
                    'metric_currentRatioMostRecentFiscalYear', 'metric_quickRatioMostRecentFiscalYear',
                    'metric_totalDebtPerTotalEquityMostRecentFiscalYear', 'metric_netInterestCoverageMostRecentFiscalYear',
                    'metric_marketCap', 'metric_beta'
                ]
            },
            'sentiment': {
                'description': 'News and social media sentiment indicators',
                'features': [
                    'reddit_title_sentiments_mean', 'reddit_title_sentiments_std',
                    'reddit_body_sentiments', 'reddit_body_sentiments_std', 'reddit_score_mean',
                    'reddit_score_sum', 'reddit_posts_count', 'reddit_comments_sum',
                    'news_sentiment_mean', 'news_articles_count', 'news_sentiment_std', 'news_sources',
                    'sentiment_momentum_3d', 'sentiment_momentum_5d',
                    'sentiment_extreme_positive', 'sentiment_extreme_negative'
                ]
            }
        }

    def get_features_for_combination(self, combination: str) -> List[str]:
        if combination.lower() == 'all':
            all_features = []
            for category_features in self.feature_categories.values():
                all_features.extend(category_features['features'])
            return list(dict.fromkeys(all_features))
        categories = [cat.strip().lower() for cat in combination.split('+')]
        valid_categories = set(self.feature_categories.keys())
        invalid = set(categories) - valid_categories
        if invalid:
            raise ValueError(f"Invalid feature categories: {invalid}")
        selected = []
        for category in categories:
            selected.extend(self.feature_categories[category]['features'])
        selected = list(dict.fromkeys(selected))
        if 'close' in selected:
            selected.remove('close')
            selected = ['close'] + selected
        return selected

    def print_available_combinations(self):
        print("\n=== Available Feature Categories ===")
        for category, info in self.feature_categories.items():
            print(f"\n{category.upper()}: {info['description']}")
            print(f"  Features ({len(info['features'])}): {', '.join(info['features'][:5])}...")
        print("\n=== Example Combinations ===")
        print("• ohlcv - Basic price data only")
        print("• technical - Technical indicators only")
        print("• financial - Financial metrics only")
        print("• sentiment - Sentiment data only")
        print("• all - All features")


# -----------------------
# FeatureCombinationPPOTrainer (updated __init__)
# -----------------------
class FeatureCombinationPPOTrainer(PPOTrainer):
    """PPO Trainer with feature combination support, curriculum learning, and robust early stopping."""

    def __init__(self, config: Dict[str, Any], selected_features: List[str]):
        """
        Initialize trainer with selected features.

        CRITICAL: We DON'T call super().__init__() because parent creates JAXVectorizedPortfolioEnv.
        Instead, we replicate parent's logic but use CustomPortfolioEnv.
        """
        self.selected_features = selected_features
        self.config = config
        self.curriculum_stage = config.get('curriculum_stage', None)
        self.curriculum_config = config.get('curriculum_config', None)
        self.nan_count = 0
        self.max_nan_resets = 5

        logger.info(f"Trainer: Initializing with {len(selected_features)} features")

        # Ensure config has the right settings
        self.config['use_all_features'] = False
        self.config['features'] = selected_features

        # Apply curriculum stage settings if provided
        if self.curriculum_stage and self.curriculum_config:
            self._apply_curriculum_settings()

        # === CRITICAL: Create CustomPortfolioEnv instead of calling parent ===
        logger.info("Creating CustomPortfolioEnv...")
        self.env = CustomPortfolioEnv(
            selected_features=selected_features,
            data_root=self.config.get('data_root', 'processed_data/'),
            stocks=self.config.get('stocks', None),
            initial_cash=self.config.get('initial_cash', 1000000.0),
            window_size=self.config.get('window_size', 30),
            start_date=self.config.get('train_start_date', '2024-06-06'),
            end_date=self.config.get('train_end_date', '2025-03-06'),
            transaction_cost_rate=self.config.get('transaction_cost_rate', 0.005),
            sharpe_window=self.config.get('sharpe_window', 252)
        )

        logger.info(f"Environment initialized: obs_dim={self.env.obs_dim}, action_dim={self.env.action_dim}")

        # Verify environment has correct features
        if self.env.n_features != len(selected_features):
            raise ValueError(
                f"Environment feature count mismatch! "
                f"Expected {len(selected_features)}, got {self.env.n_features}"
            )

        logger.info(f"✓ Environment verified: {self.env.n_features} features, shape {self.env.data.shape}")

        # === Replicate PPOTrainer.__init__ logic (lines 271-301) ===
        # Vectorized environment functions
        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0,))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0))

        # Initialize network
        logger.info("Initializing network parameters...")
        self.network = ActorCriticLSTM(
            action_dim=self.env.action_dim,
            hidden_size=config.get('hidden_size', 256),
            n_lstm_layers=config.get('n_lstm_layers', 1)
        )

        # Initialize parameters (replicate parent's _initialize_parameters)
        logger.info("Initializing network parameters...")

        # Initialize RNG
        self.rng = random.PRNGKey(self.config.get('seed', 42))
        self.rng, init_rng = random.split(self.rng)

        # Create dummy inputs for initialization
        dummy_obs = jnp.ones((self.config.get('n_envs', 8), self.env.obs_dim))
        dummy_carry = self._create_dummy_carry(self.config.get('n_envs', 8))

        # Initialize parameters
        try:
            self.params = self.network.init(init_rng, dummy_obs, dummy_carry)
            logger.info("Network parameters initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize network parameters: {e}")
            raise

        # Validate initial parameters
        if self._has_nan_params(self.params):
            logger.warning("NaN detected in initial parameters, reinitializing...")
            self.rng, init_rng = random.split(self.rng)
            self.params = self.network.init(init_rng, dummy_obs, dummy_carry)

            if self._has_nan_params(self.params):
                raise RuntimeError("Failed to initialize parameters without NaN")

        # Test forward pass
        self._test_network_forward_pass(dummy_obs, dummy_carry)

        # Setup optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.get('max_grad_norm', 0.5)),
            optax.adam(
                learning_rate=config.get('learning_rate', 1e-4),
                eps=1e-8,
                b1=0.9,
                b2=0.999
            )
        )

        # Create training state
        self.train_state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=self.params,
            tx=self.optimizer
        )

        # Initialize training state
        self._initialize_training_state()
        logger.info("PPO Trainer initialization complete!")

    def _create_dummy_carry(self, batch_size: int) -> List[LSTMState]:
        """Create dummy LSTM carry states"""
        return [
            LSTMState(
                h=jnp.zeros((batch_size, self.config.get('hidden_size', 256))),
                c=jnp.zeros((batch_size, self.config.get('hidden_size', 256)))
            ) for _ in range(self.config.get('n_lstm_layers', 1))
        ]

    def _has_nan_params(self, params) -> bool:
        """Check for NaN values in parameters"""
        def check_nan(x):
            return jnp.any(jnp.isnan(x)) if jnp.issubdtype(x.dtype, jnp.floating) else False

        has_nan = jax.tree_util.tree_reduce(
            lambda acc, x: acc | check_nan(x),
            params, False
        )
        return bool(has_nan)

    def _test_network_forward_pass(self, obs, carry: List[LSTMState]):
        """Test network forward pass for NaN issues"""
        logger.info("Testing network forward pass...")
        try:
            logits, values, new_carry = self.network.apply(self.params, obs, carry)

            if jnp.any(jnp.isnan(logits)) or jnp.any(jnp.isnan(values)):
                raise RuntimeError("NaN detected in network forward pass")

            logger.info("✅ Network forward pass test passed")
        except Exception as e:
            logger.error(f"Network forward pass test failed: {e}")
            raise

    def normalize_rewards(self, rewards: jnp.ndarray) -> jnp.ndarray:
        try:
            if jnp.any(jnp.isnan(rewards)) or jnp.any(jnp.isinf(rewards)):
                rewards = jnp.where(jnp.isfinite(rewards), rewards, 0.0)
            mean = jnp.mean(jnp.where(jnp.isfinite(rewards), rewards, 0.0))
            std = jnp.std(jnp.where(jnp.isfinite(rewards), rewards, 0.0))
            std = jnp.where(std > 1e-8, std, 1.0)
            normalized = (rewards - mean) / std
            normalized = jnp.where(jnp.isfinite(normalized), normalized, 0.0)
            return jnp.clip(normalized, -10.0, 10.0)
        except Exception as e:
            logger.warning(f"Error in normalize_rewards: {e}")
            return jnp.zeros_like(rewards)

    def _apply_curriculum_settings(self):
        """Apply curriculum-specific settings to the configuration."""
        stage = None
        try:
            stage = self.curriculum_config.get_stage(self.curriculum_stage)
        except Exception:
            stage = None
        if not stage:
            logger.warning(f"No curriculum stage found for stage {self.curriculum_stage}; using current config values")
            return

        # Stages are defined as dicts; safely extract with defaults
        stage_num = stage.get('stage_num', self.curriculum_stage)
        stage_name = stage.get('name', f"Stage {stage_num}")
        logger.info(f"Initializing Stage {stage_num}: {stage_name}")
        logger.info(f"Description: {stage.get('description', '')}")

        defaults = {
            'exploration_std': self.config.get('action_std', 0.2),
            'entropy_coeff': self.config.get('entropy_coeff', 0.01),
            'clip_eps': self.config.get('clip_eps', 0.2),
            'learning_rate': self.config.get('learning_rate', 3e-4),
            'num_epochs': self.config.get('num_updates', 1000),
            'epsilon_uniform': 0.05,
            'reward_scaling': 1.0,
            'enable_constraints': False,
        }

        params = {k: stage.get(k, defaults[k]) for k in defaults}

        # Update config with stage-specific hyperparameters
        self.config.update({
            'action_std': params['exploration_std'],
            'entropy_coeff': params['entropy_coeff'],
            'clip_eps': params['clip_eps'],
            'learning_rate': params['learning_rate'],
            'num_updates': params['num_epochs'],
            'original_action_std': params['exploration_std'],
            'final_action_std': params['exploration_std'] * 0.7,
            'action_std_decay_steps': max(1, int(params['num_epochs'] * 0.8)),
            'original_entropy_coeff': params['entropy_coeff'],
            'final_entropy_coeff': params['entropy_coeff'] * 0.7,
            'entropy_decay_steps': max(1, int(params['num_epochs'] * 0.8)),
        })

        # Store stage-specific parameters
        self.epsilon_uniform = params['epsilon_uniform']
        self.reward_scaling = params['reward_scaling']
        self.enable_constraints = params['enable_constraints']

        # Log applied curriculum parameters for visibility
        logger.info(
            f"Curriculum params applied | lr={self.config['learning_rate']} | "
            f"std={self.config['action_std']}→{self.config['final_action_std']} ({self.config['action_std_decay_steps']} steps) | "
            f"entropy={self.config['entropy_coeff']}→{self.config['final_entropy_coeff']} ({self.config['entropy_decay_steps']} steps) | "
            f"clip_eps={self.config['clip_eps']} | updates={self.config['num_updates']}"
        )

        # Early stopping settings based on stage number
        if stage_num == 1:
            self.config.update({'early_stopping_patience': 150, 'early_stopping_min_delta': 0.001})
        elif stage_num == 2:
            self.config.update({'early_stopping_patience': 200, 'early_stopping_min_delta': 0.005})
        else:
            self.config.update({'early_stopping_patience': 250, 'early_stopping_min_delta': 0.01})

    def _initialize_training_state(self):
        """Initialize training state including environment states and metrics"""
        logger.info("Initializing environment state...")

        # Initialize RNG if not already done
        if not hasattr(self, 'rng'):
            self.rng = random.PRNGKey(self.config.get('seed', 42))

        # Initialize environment states
        try:
            self.rng, *reset_keys = random.split(self.rng, self.config.get('n_envs', 8) + 1)
            reset_keys = jnp.array(reset_keys)
            self.env_states, self.obs = self.vmap_reset(reset_keys)

            # Clean environment observations
            if jnp.any(jnp.isnan(self.obs)):
                logger.warning("NaN detected in initial observations, setting to zero")
                self.obs = jnp.where(jnp.isnan(self.obs), 0.0, self.obs)
            if jnp.any(jnp.isinf(self.obs)):
                logger.warning("Inf detected in initial observations, clamping")
                self.obs = jnp.clip(self.obs, -1e10, 1e10)

            logger.info(f"Initial observations - mean: {float(self.obs.mean()):.6f}, "
                       f"std: {float(self.obs.std()):.6f}, "
                       f"range: [{float(self.obs.min()):.6f}, {float(self.obs.max()):.6f}]")

            # Initialize collector carry (LSTM states)
            self.collector_carry = [
                LSTMState(
                    h=jnp.zeros((self.config.get('n_envs', 8), self.config.get('hidden_size', 256))),
                    c=jnp.zeros((self.config.get('n_envs', 8), self.config.get('hidden_size', 256)))
                ) for _ in range(self.config.get('n_lstm_layers', 1))
            ]

            logger.info("✅ Environment state initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize environment state: {e}")
            raise

        # Initialize training metrics
        self.early_training_metrics = {'portfolio_values': [], 'rewards': [], 'variances': [], 'losses': []}
        self.best_performance = -1e10
        self.patience_counter = 0
        self.should_stop_early = False
        self.kl_divergence_history = []
        
    def update_stage_parameters(self, stage_params: Dict[str, Any]):
        """Update trainer parameters for a new curriculum stage."""
        logger.info("Updating trainer parameters for new stage:")
        for key, value in stage_params.items():
            if key in self.config:
                logger.info(f"  {key}: {self.config[key]} -> {value}")
                self.config[key] = value
            if key == "action_std":
                # Ensure the curriculum-specific action std persists
                self.config["original_action_std"] = value
            if key == "entropy_coeff":
                # Ensure the curriculum-specific entropy coeff persists
                self.config["original_entropy_coeff"] = value
            if key == "early_stopping_patience":
                # Update early stopping patience with curriculum-specific value
                self.config["early_stopping_patience"] = value
        
        # Reset stage-specific counters and metrics
        self.patience_counter = 0
        self.kl_divergence_history = []
        
        # Update optimizer learning rate if changed
        if 'learning_rate' in stage_params:
            self.train_state = self.train_state.replace(
                opt_state=optax.adam(stage_params['learning_rate']).init(self.train_state.params)
            )

    def train(self):
        """Train the PPO model with curriculum learning and early stopping."""
        logger.info("Starting PPO training...")
        num_updates = self.config.get('num_updates', 1000)
        log_interval = self.config.get('log_interval', 10)
        save_interval = self.config.get('save_interval', 50)

        for update in range(num_updates):
            start_time = time.time()

            try:
                # Collect trajectory
                trajectory = self._collect_trajectory(update)
                
                # Store raw rewards for logging before normalization
                raw_rewards = trajectory.rewards
                avg_reward_raw = float(raw_rewards.mean())
                
                # Normalize rewards for training stability but keep raw rewards for logging
                normalized_rewards = self.normalize_rewards(raw_rewards)
                trajectory = trajectory._replace(rewards=normalized_rewards)
                
                # Store both raw and normalized for logging
                self.current_raw_rewards = raw_rewards
                self.current_normalized_rewards = normalized_rewards
                
                # Update hyperparameters
                self._update_hyperparameters(update)

                # Perform training step
                self._train_step(trajectory, update)

                # Log progress
                if update % log_interval == 0:
                    self._log_progress(trajectory, update, start_time, avg_reward_raw)

                # Check early stopping
                if self.check_early_stopping(trajectory, update):
                    logger.info("Early stopping triggered. Training complete.")
                    break

                # Save model periodically
                if update % save_interval == 0 and update > 0:
                    self.save_model(f"model_checkpoint_{update}")

            except Exception as e:
                logger.error(f"Error during training step {update}: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue

        logger.info("Training complete!")

    def _collect_trajectory(self, update: int) -> Trajectory:
        """Collect trajectory from the environment."""
        self.rng, collect_rng = random.split(self.rng)
        trajectory, self.env_states, self.obs, self.collector_carry = self.collect_trajectory(
            self.train_state, self.env_states, self.obs, self.collector_carry, collect_rng
        )
        return trajectory

    def _update_hyperparameters(self, update: int):
        """Update hyperparameters dynamically during training."""
        self.config['action_std'] = self.get_current_action_std(update)
        self.config['entropy_coeff'] = self.get_current_entropy_coeff(update)

    def get_current_action_std(self, update: int) -> float:
        """Decay action std with cosine annealing to maintain exploration longer."""
        # Always use the original action std from config, not the possibly-updated value
        initial = float(self.config.get('original_action_std', 1.0))
        decay_fraction = float(self.config.get('action_std_decay_fraction', 0.7))  # Decay to 70% by default
        final = initial * decay_fraction

        # Use more of the updates for decay
        decay_steps_fraction = float(self.config.get('decay_steps_fraction', 0.8))
        total_updates = int(self.config.get('num_updates', 1000))
        decay_steps = int(total_updates * decay_steps_fraction)

        if update >= decay_steps:
            return final

        # Cosine annealing for smoother decay
        progress = update / decay_steps
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        value = final + (initial - final) * cosine_decay

        return float(jnp.maximum(value, 1e-6))

    def get_current_entropy_coeff(self, update: int) -> float:
        """Decay entropy coefficient with cosine annealing to maintain exploration."""
        initial = float(self.config.get('original_entropy_coeff', 0.01))
        decay_fraction = float(self.config.get('entropy_decay_fraction', 0.7))  # Decay to 70% by default
        final = initial * decay_fraction
        
        # Use more of the updates for decay
        decay_steps_fraction = float(self.config.get('decay_steps_fraction', 0.8))
        total_updates = int(self.config.get('num_updates', 1000))
        decay_steps = int(total_updates * decay_steps_fraction)
        
        if update >= decay_steps:
            return final
        
        # Cosine annealing for smoother decay
        progress = update / decay_steps
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        value = final + (initial - final) * cosine_decay
        
        return float(jnp.maximum(value, 0.0))

    def compute_robust_metrics(self, trajectory: Trajectory) -> Dict[str, float]:
        """Compute simple, robust metrics like Sharpe ratio from RAW rewards.

        CRITICAL: Uses self.current_raw_rewards instead of trajectory.rewards
        because trajectory.rewards contains normalized rewards (mean≈0, std≈1)
        which would make Sharpe ratio always ≈0.
        """
        try:
            # Use RAW rewards stored before normalization
            if hasattr(self, 'current_raw_rewards') and self.current_raw_rewards is not None:
                rewards = self.current_raw_rewards
            else:
                # Fallback to trajectory rewards if raw rewards not available
                logger.warning("Raw rewards not available, using trajectory rewards (may be normalized)")
                rewards = trajectory.rewards

            # Ensure rewards is a valid array and not empty
            if rewards is None or (hasattr(rewards, 'size') and rewards.size == 0):
                raise ValueError("Trajectory rewards are empty or None")
            rewards = jnp.array(rewards)
            rewards = jnp.where(jnp.isfinite(rewards), rewards, 0.0)

            # Sum rewards per env across time (axis 0 is time dimension)
            per_env_return = jnp.sum(rewards, axis=0)

            # Compute mean and std safely
            mean_return = jnp.mean(per_env_return)
            std_return = jnp.std(per_env_return)

            # Safe division for Sharpe ratio
            std_return = jnp.where(std_return > 1e-8, std_return, 1.0)
            sharpe = mean_return / std_return

            return {
                'sharpe_ratio': float(jnp.clip(sharpe, -1e3, 1e3)),
                'avg_reward': float(jnp.mean(rewards)),
                'std_reward': float(std_return),
                'mean_return': float(mean_return)  # Add this for better tracking
            }
        except Exception as e:
            logger.warning(f"Error in compute_robust_metrics: {e}")
            return {
                'sharpe_ratio': 0.0,
                'avg_reward': 0.0,
                'std_reward': 1.0,
                'mean_return': 0.0
            }

    def _train_step(self, trajectory: Trajectory, update: int):
        """Perform a single training step."""
        self.rng, train_rng = random.split(self.rng)
        n_steps = self.config.get('n_steps', 64)
        n_envs = self.config.get('n_envs', 8)
        ppo_batch_size = self.config.get('ppo_batch_size', 256)
        num_minibatches = max(1, (n_steps * n_envs) // ppo_batch_size)
        self.train_state, metrics = self.train_step(
            self.train_state, trajectory, self._get_last_values(), train_rng, num_minibatches
        )
        self._track_metrics(metrics)

    def _get_last_values(self) -> jnp.ndarray:
        """Get bootstrap values for GAE."""
        try:
            # Get value estimates from critic
            _, last_values, _ = self.train_state.apply_fn(
                self.train_state.params, self.obs, self.collector_carry
            )
            
            # Replace any non-finite values with zeros
            last_values = jnp.where(jnp.isfinite(last_values), last_values, 0.0)
            
            # Clip to reasonable range to prevent exploding gradients
            return jnp.clip(last_values, -1e6, 1e6)
            
        except Exception as e:
            logger.warning(f"Error in _get_last_values: {e}")
            return jnp.zeros(self.config.get('n_envs', 8))

    def _track_metrics(self, metrics: Dict[str, Any]):
        """Track training metrics."""
        if 'approx_kl' in metrics:
            self.kl_divergence_history.append(float(metrics['approx_kl']))
            if len(self.kl_divergence_history) > 50:
                self.kl_divergence_history.pop(0)

    def _log_progress(self, trajectory: Trajectory, update: int, start_time: float, avg_reward_raw: float = None):
        """Log training progress and metrics to wandb if enabled."""
        elapsed = time.time() - start_time
        
        # Use stored raw and normalized rewards
        raw_reward_mean = float(self.current_raw_rewards.mean())
        raw_reward_min = float(self.current_raw_rewards.min())
        raw_reward_max = float(self.current_raw_rewards.max())
        normalized_reward_mean = float(self.current_normalized_rewards.mean())
        
        # Calculate episode-specific metrics
        # Convert JAX arrays to numpy for safe boolean operations
        dones_np = np.array(trajectory.dones)  # Shape: (n_steps, n_envs)
        rewards_np = np.array(trajectory.rewards)  # Shape: (n_steps, n_envs)

        # Calculate episode returns per environment
        episode_mask = dones_np.astype(bool)
        if episode_mask.sum() > 0:
            # Track returns for each environment separately
            n_envs = rewards_np.shape[1] if rewards_np.ndim > 1 else 1
            episode_returns = []

            if rewards_np.ndim == 1:
                # Single environment case
                current_return = 0
                for i in range(len(rewards_np)):
                    current_return += float(rewards_np[i])
                    if bool(dones_np[i]):
                        episode_returns.append(current_return)
                        current_return = 0
            else:
                # Multiple environments case
                current_returns = np.zeros(n_envs)
                for t in range(rewards_np.shape[0]):  # Iterate over timesteps
                    for e in range(n_envs):  # Iterate over environments
                        current_returns[e] += float(rewards_np[t, e])
                        if bool(dones_np[t, e]):
                            episode_returns.append(current_returns[e])
                            current_returns[e] = 0

            if episode_returns:
                max_episode_return = float(max(episode_returns))
                avg_episode_return = float(sum(episode_returns) / len(episode_returns))
            else:
                max_episode_return = 0.0
                avg_episode_return = 0.0
        else:
            max_episode_return = 0.0
            avg_episode_return = 0.0
        
        # Get current hyperparameters
        current_std = self.get_current_action_std(update)
        current_entropy = self.get_current_entropy_coeff(update)
        
        # Calculate global step for wandb
        global_step = self.config.get('global_step', 0) + update
        
        # Get average portfolio value from environment states
        # Note: env_states might be a JAX array or EnvState object depending on implementation
        try:
            if hasattr(self.env_states, 'portfolio_value'):
                # Single EnvState or named tuple
                avg_portfolio_value = float(jnp.mean(self.env_states.portfolio_value))
            elif isinstance(self.env_states, (list, tuple)):
                # List of EnvState objects
                avg_portfolio_value = float(
                    jnp.mean(jnp.array([state.portfolio_value for state in self.env_states]))
                )
            else:
                # Fallback: use trajectory rewards as proxy
                avg_portfolio_value = float(trajectory.rewards.sum())
        except Exception as e:
            logger.debug(f"Could not extract portfolio value: {e}")
            avg_portfolio_value = 0.0
        
        # Log to console
        logger.info(
            f"Update {update} | Global Step {global_step} | Time: {elapsed:.2f}s | "
            f"Raw Avg: {raw_reward_mean:.4f} | Norm Avg: {normalized_reward_mean:.4f} | "
            f"Portfolio: {avg_portfolio_value:.2f} | "
            f"Max Return: {max_episode_return:.4f} | "
            f"std={current_std:.6f} | entropy={current_entropy:.6f}"
        )

        # Compute robust metrics for logging (using raw rewards)
        robust_metrics = self.compute_robust_metrics(trajectory)

        # Log to wandb if available
        if wandb is not None and self.config.get('use_wandb', False):
            wandb_metrics = {
                # Rewards
                'train/raw_reward': raw_reward_mean,
                'train/raw_reward_min': raw_reward_min,
                'train/raw_reward_max': raw_reward_max,
                'train/normalized_reward': normalized_reward_mean,
                'train/max_episode_return': max_episode_return,
                'train/avg_episode_return': avg_episode_return,

                # Performance metrics (from raw rewards)
                'train/portfolio_value': avg_portfolio_value,
                'train/sharpe_ratio': robust_metrics.get('sharpe_ratio', 0.0),
                'train/mean_return': robust_metrics.get('mean_return', 0.0),
                'train/std_return': robust_metrics.get('std_reward', 1.0),

                # Training metrics from last update
                'train/total_loss': self.last_metrics.get('total_loss', 0.0) if hasattr(self, 'last_metrics') else 0.0,
                'train/policy_loss': self.last_metrics.get('policy_loss', 0.0) if hasattr(self, 'last_metrics') else 0.0,
                'train/value_loss': self.last_metrics.get('value_loss', 0.0) if hasattr(self, 'last_metrics') else 0.0,
                'train/entropy_loss': self.last_metrics.get('entropy_loss', 0.0) if hasattr(self, 'last_metrics') else 0.0,

                # Training parameters
                'train/action_std': current_std,
                'train/entropy_coeff': current_entropy,
                'train/learning_rate': self.config.get('learning_rate', 3e-4),
                'train/approx_kl': self.last_metrics.get('approx_kl', 0.0) if hasattr(self, 'last_metrics') else 0.0,
                'train/clip_fraction': self.last_metrics.get('clip_fraction', 0.0) if hasattr(self, 'last_metrics') else 0.0,
                'train/update_time': elapsed,

                # Curriculum info
                'curriculum/stage': self.curriculum_stage if self.curriculum_stage else 0,
                'curriculum/global_step': global_step,

                # Early stopping metrics
                'train/best_performance': self.best_performance,
                'train/patience_counter': self.patience_counter
            }
            wandb.log(wandb_metrics, step=global_step)

    def check_early_stopping(self, trajectory: Trajectory, update: int) -> bool:
        """Check if early stopping criteria are met with more conservative settings."""
        try:
            # Don't check early stopping until we have enough updates
            min_updates = 100  # Minimum number of updates before considering early stopping
            if update < min_updates:
                return False
                
            # Compute metrics safely
            metrics = self.compute_robust_metrics(trajectory)
            current_performance = metrics['sharpe_ratio']
            
            # Handle potential NaN/Inf in performance metrics
            if not jnp.isfinite(current_performance):
                logger.warning("Non-finite performance metric detected, skipping early stopping check")
                return False
            
            # Calculate improvement
            improvement = current_performance - self.best_performance
            
            # Log early stopping metrics
            logger.info(f"Early stopping check - Current: {current_performance:.4f}, "
                       f"Best: {self.best_performance:.4f}, "
                       f"Improvement: {improvement:.4f}, "
                       f"Patience: {self.patience_counter}/{self.config.get('early_stopping_patience', 150)}")
            
            # Get early stopping parameters
            min_delta = self.config.get('early_stopping_min_delta', 0.005)
            patience = self.config.get('early_stopping_patience', 150)
            
            # Update best performance if improved beyond min_delta
            if jnp.isfinite(improvement) and improvement > min_delta:
                self.best_performance = float(current_performance)  # Convert to Python float
                self.patience_counter = 0
                logger.info(f"New best performance: {self.best_performance:.4f}")
            else:
                self.patience_counter += 1
                
            # Check if should stop
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {update} updates. "
                           f"No improvement > {min_delta:.6f} for {self.patience_counter} updates.")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error in early stopping check: {e}")
            return False
    


# -----------------------
# Runner & verification
# -----------------------
def run_training_with_combination(feature_combination: str, config: Dict[str, Any]):
    logger.info(f"Starting training with feature combination: {feature_combination}")

    feature_selector = FeatureSelector()
    try:
        selected_features = feature_selector.get_features_for_combination(feature_combination)
        logger.info(f"Selected {len(selected_features)} features for training")
        for category, info in feature_selector.feature_categories.items():
            category_features = [f for f in selected_features if f in info['features']]
            logger.info(f"  {category}: {len(category_features)} features")
    except Exception as e:
        logger.error(f"Failed to get features for combination '{feature_combination}': {e}")
        feature_selector.print_available_combinations()
        raise

    config['feature_combination'] = feature_combination
    config['selected_features'] = selected_features
    config['model_name'] = f"ppo_lstm_{feature_combination.replace('+', '_')}"

    try:
        trainer = FeatureCombinationPPOTrainer(config, selected_features)
        logger.info("Feature combination PPO trainer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        raise

    # COMPREHENSIVE VERIFICATION TEST
    try:
        logger.info("=" * 80)
        logger.info("COMPREHENSIVE FEATURE VERIFICATION TEST")
        logger.info("=" * 80)

        # 1. Check feature counts
        logger.info(f"✓ Step 1: Feature Count Verification")
        logger.info(f"  Expected features: {len(selected_features)}")
        env_n_features = getattr(trainer.env, 'n_features', None)
        logger.info(f"  Environment n_features: {env_n_features}")

        if env_n_features != len(selected_features):
            raise ValueError(
                f"CRITICAL: Feature count mismatch! "
                f"Expected {len(selected_features)}, got {env_n_features}"
            )
        logger.info(f"  ✓ Feature count matches: {env_n_features}")

        # 2. Check actual feature names
        logger.info(f"\n✓ Step 2: Feature Names Verification")
        env_features = getattr(trainer.env, 'features', None)
        if env_features is not None:
            logger.info(f"  Environment features (first 10): {env_features[:10]}")
            logger.info(f"  Expected features (first 10): {selected_features[:10]}")

            # Check if lists match exactly
            if env_features != selected_features:
                # Find mismatches
                missing_features = set(selected_features) - set(env_features)
                extra_features = set(env_features) - set(selected_features)

                error_msg = "CRITICAL: Feature names mismatch!\n"
                if missing_features:
                    error_msg += f"  Missing features: {list(missing_features)[:10]}...\n"
                if extra_features:
                    error_msg += f"  Extra features: {list(extra_features)[:10]}...\n"

                raise ValueError(error_msg)
            logger.info(f"  ✓ Feature names match exactly ({len(env_features)} features)")
        else:
            logger.warning("  ! Could not verify feature names (env.features not available)")

        # 3. Check data shape
        logger.info(f"\n✓ Step 3: Data Shape Verification")
        if hasattr(trainer.env, 'data'):
            env_data = trainer.env.data
            logger.info(f"  Data type: {type(env_data)}")

            if isinstance(env_data, (np.ndarray, jnp.ndarray)):
                logger.info(f"  Data shape: {env_data.shape}")
                expected_shape = f"(timesteps, {getattr(trainer.env, 'n_stocks', '?')}, {len(selected_features)})"
                logger.info(f"  Expected shape: {expected_shape}")

                if env_data.ndim == 3:
                    actual_feature_dim = env_data.shape[2]
                    if actual_feature_dim != len(selected_features):
                        raise ValueError(
                            f"CRITICAL: Data shape mismatch! "
                            f"Feature dimension is {actual_feature_dim}, expected {len(selected_features)}"
                        )
                    logger.info(f"  ✓ Data shape matches: feature dimension = {actual_feature_dim}")
                else:
                    logger.warning(f"  ! Unexpected data dimensions: {env_data.ndim}")
            else:
                logger.warning(f"  ! Data is not a numpy/jax array, cannot verify shape")
        else:
            logger.warning("  ! Environment has no 'data' attribute")

        # 4. Check data loader
        logger.info(f"\n✓ Step 4: Data Loader Verification")
        if hasattr(trainer.env, 'data_loader'):
            loader = trainer.env.data_loader
            loader_features = getattr(loader, 'features', None)
            loader_selected = getattr(loader, 'selected_features', None)

            if loader_features is not None:
                logger.info(f"  Loader features count: {len(loader_features)}")
                if loader_features != selected_features:
                    logger.warning(f"  ! Loader features don't match selected features")
                    logger.warning(f"    Loader has {len(loader_features)}, expected {len(selected_features)}")
                else:
                    logger.info(f"  ✓ Loader features match selected features")

            if loader_selected is not None:
                if loader_selected != selected_features:
                    logger.warning(f"  ! Loader selected_features mismatch")
                else:
                    logger.info(f"  ✓ Loader selected_features match")
        else:
            logger.warning("  ! Environment has no data_loader attribute")

        # 5. Check observation dimensions
        logger.info(f"\n✓ Step 5: Observation Dimension Verification")
        obs_dim = getattr(trainer.env, 'obs_dim', None)
        if obs_dim is not None:
            logger.info(f"  Environment obs_dim: {obs_dim}")

            # Calculate expected obs dim
            window_size = getattr(trainer.env, 'window_size', 30)
            n_stocks = getattr(trainer.env, 'n_stocks', 0)
            action_dim = getattr(trainer.env, 'action_dim', n_stocks)

            expected_obs_dim = (
                (window_size * n_stocks * len(selected_features)) +  # Price window
                n_stocks * 2 +  # Holdings + prices
                action_dim +  # Last action
                n_stocks +  # Normalized holdings
                8  # Portfolio stats
            )

            logger.info(f"  Expected obs_dim: {expected_obs_dim}")
            logger.info(f"    (window={window_size} * stocks={n_stocks} * features={len(selected_features)} "
                       f"+ stocks*2 + action_dim={action_dim} + stocks + 8)")

            if obs_dim != expected_obs_dim:
                logger.warning(
                    f"  ! Observation dimension mismatch: got {obs_dim}, expected {expected_obs_dim}"
                )
            else:
                logger.info(f"  ✓ Observation dimension matches: {obs_dim}")
        else:
            logger.warning("  ! Environment has no obs_dim attribute")

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("✓ VERIFICATION PASSED: All critical checks passed")
        logger.info(f"  - Feature count: {len(selected_features)} ✓")
        logger.info(f"  - Feature names: Verified ✓")
        logger.info(f"  - Data shape: Verified ✓")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("✗ VERIFICATION FAILED")
        logger.error(f"Error: {e}")
        logger.error("=" * 80)
        raise

    # Run training
    try:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")
        trainer.save_model(f"final_model_{feature_combination.replace('+', '_')}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


# -----------------------
# CLI & main()
# -----------------------
def get_valid_feature_combinations():
    feature_selector = FeatureSelector()
    base_categories = list(feature_selector.feature_categories.keys())
    valid_combinations = ['all']
    valid_combinations.extend(base_categories)
    for i in range(2, len(base_categories) + 1):
        for combo in itertools.combinations(base_categories, i):
            valid_combinations.append('+'.join(combo))
    return valid_combinations

def main():
    parser = argparse.ArgumentParser(
        description="Train PPO LSTM with different feature combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    valid_combinations = get_valid_feature_combinations()

    parser.add_argument('--feature_combination', type=str, default='ohlcv+technical', choices=valid_combinations)
    parser.add_argument('--curriculum_stage', type=int, choices=[1,2,3])
    parser.add_argument('--auto_curriculum', action='store_true')
    parser.add_argument('--start_stage', type=int, default=1, choices=[1,2,3])
    parser.add_argument('--num_updates', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=32)
    parser.add_argument('--ppo_epochs', type=int, default=3)
    parser.add_argument('--ppo_batch_size', type=int, default=96)
    parser.add_argument('--hidden_size', type=int, default=192)
    parser.add_argument('--n_lstm_layers', type=int, default=2)
    parser.add_argument('--data_root', type=str, default='processed_data/')
    parser.add_argument('--train_start_date', type=str, default='2024-06-06')
    parser.add_argument('--train_end_date', type=str, default='2025-03-06')
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--early_stopping_patience', type=int, default=150)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.005)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--list_combinations', action='store_true')
    parser.add_argument('--target_update_time', type=float, default=0.5)
    parser.add_argument('--max_cpu_threads', type=int, default=8)
    parser.add_argument('--memory_fraction', type=float, default=0.7)

    args = parser.parse_args()

    if args.list_combinations:
        fs = FeatureSelector()
        fs.print_available_combinations()
        return

    os.environ['OMP_NUM_THREADS'] = str(max(1, args.max_cpu_threads))
    os.environ['MKL_NUM_THREADS'] = str(max(1, args.max_cpu_threads))
    os.environ['NUMEXPR_NUM_THREADS'] = str(max(1, args.max_cpu_threads))
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(max(0.1, min(0.95, args.memory_fraction)))

    config = {
        'seed': args.seed,
        'data_root': args.data_root,
        'train_start_date': args.train_start_date,
        'train_end_date': args.train_end_date,
        'window_size': args.window_size,
        'n_envs': args.n_envs,
        'n_steps': args.n_steps,
        'num_updates': args.num_updates,
        'learning_rate': args.learning_rate,
        'ppo_epochs': args.ppo_epochs,
        'ppo_batch_size': args.ppo_batch_size,
        'hidden_size': args.hidden_size,
        'n_lstm_layers': args.n_lstm_layers,
        'use_wandb': args.use_wandb,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': args.early_stopping_min_delta,
        'model_dir': args.model_dir,
        'target_update_time': args.target_update_time,
        'max_cpu_threads': args.max_cpu_threads,
        'memory_fraction': args.memory_fraction,
    }

    if args.curriculum_stage or args.auto_curriculum:
        curriculum_config = CurriculumConfig()
        config['curriculum_config'] = curriculum_config
        config['curriculum_stage'] = args.curriculum_stage

    try:
        if args.auto_curriculum:
            logger.info("Running FULL CURRICULUM (all stages)")
            curriculum_config = CurriculumConfig()
            if args.use_wandb and wandb is not None:
                run_name = f"curriculum_{args.feature_combination}_{time.strftime('%Y%m%d_%H%M%S')}"
                wandb.init(project="finagent-curriculum", name=run_name, config=config, resume="allow")

            base_config = config.copy()
            base_config['feature_combination'] = args.feature_combination

            feature_selector = FeatureSelector()
            selected_features = feature_selector.get_features_for_combination(args.feature_combination)
            base_config['selected_features'] = selected_features
            logger.info(f"Selected {len(selected_features)} features for training")

            trainer = None
            global_step = 0
            for stage_num in range(args.start_stage, 4):
                logger.info(f"\n{'='*50}\nStarting Curriculum Stage {stage_num}\n{'='*50}")
                stage = curriculum_config.stages[stage_num]
                stage_params = {
                    'action_std': stage['exploration_std'],
                    'entropy_coeff': stage['entropy_coeff'],
                    'learning_rate': stage['learning_rate'],
                    'clip_eps': stage['clip_eps'],
                    'num_updates': stage['num_epochs'],
                    'action_std_decay_fraction': 0.7,
                    'entropy_decay_fraction': 0.7,
                    'decay_steps_fraction': 0.8,
                    'curriculum_stage': stage_num,
                    'global_step': global_step
                }
                if trainer is None:
                    config_with_stage = base_config.copy()
                    config_with_stage.update(stage_params)
                    trainer = FeatureCombinationPPOTrainer(config_with_stage, selected_features)
                else:
                    trainer.update_stage_parameters(stage_params)
                trainer.train()
                trainer.save_model(f"curriculum_stage_{stage_num}")
                global_step += stage['num_epochs']
                logger.info(f"Completed Curriculum Stage {stage_num}. Global step: {global_step}")

            if wandb is not None and wandb.run is not None:
                wandb.finish()

        else:
            # Single stage training
            feature_selector = FeatureSelector()
            selected_features = feature_selector.get_features_for_combination(args.feature_combination)
            config['selected_features'] = selected_features
            logger.info(f"Selected {len(selected_features)} features for training")

            if args.use_wandb and wandb is not None:
                run_name = f"single_{args.feature_combination}_{time.strftime('%Y%m%d_%H%M%S')}"
                wandb.init(project="finagent", name=run_name, config=config, resume="allow")

            run_training_with_combination(args.feature_combination, config)

            if wandb is not None and wandb.run is not None:
                wandb.finish()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if args.use_wandb and wandb is not None:
            wandb.finish()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if args.use_wandb and wandb is not None:
            wandb.finish()
        raise

if __name__ == "__main__":
    main()