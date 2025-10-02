import os
import time
import jax
import jax.numpy as jnp
from jax import random, vmap, lax
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from flax import serialization
import chex
from typing import Tuple, Dict, Any, NamedTuple, List, Optional
import wandb
import pickle
import distrax
from pathlib import Path
from functools import partial
import json
import logging
import argparse
import pandas as pd

# Import the JAX environment
from finagent.environment.portfolio_env import (
    JAXPortfolioDataLoader,
    JAXVectorizedPortfolioEnv,
    EnvState
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable JAX optimizations
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_compilation_cache_dir', './jax_cache')
jax.config.update('jax_debug_nans', False)

# ============================================================================
# FEATURE SELECTOR (from train_ppo_feature_combinations.py)
# ============================================================================

class FeatureSelector:
    """Feature selector for different feature categories"""
    
    def __init__(self):
        # Define feature categories based on actual CSV data
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
                    'rsi_zscore_20d',
                    'bb_position', 'bb_squeeze', 'bb_breakout_up', 'bb_breakout_down',
                    'price_zscore_20d', 'price_zscore_60d', 'volume_zscore_20d', 'volume_zscore_60d',
                    'price_deviation_50d', 'price_deviation_200d',
                    'mean_reversion_signal_50d', 'mean_reversion_signal_200d',
                    'price_breakout_20d', 'price_breakdown_20d', 'volume_breakout_20d', 'volume_spike',
                    'ma_cross_bullish', 'ma_cross_bearish',
                    'high_vol_regime', 'low_vol_regime', 'vol_expansion', 'vol_contraction',
                    'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
                    'doji_pattern', 'hammer_pattern', 'shooting_star_pattern',
                    'volume_price_momentum', 'volume_ratio_5d', 'volume_ratio_20d',
                    'volume_trend_10d', 'volume_confirms_price', 'volume_divergence',
                    'bullish_signals', 'bearish_signals', 'net_signal_strength',
                    'risk_adjusted_momentum', 'volume_confirmed_trend',
                    'vol_regime_change', 'trend_regime',
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
                    'metric_NetChangeinCash', 'metric_TotalCashDividendsPaid',
                    'metric_freeCashFlowtrailing12Month', 'metric_freeCashFlowMostRecentFiscalYear',
                    'metric_periodLength', 'metric_periodType',
                    'metric_pPerEExcludingExtraordinaryItemsMostRecentFiscalYear',
                    'metric_currentDividendYieldCommonStockPrimaryIssueLTM',
                    'metric_priceToBookMostRecentFiscalYear',
                    'metric_priceToFreeCashFlowPerShareTrailing12Months',
                    'metric_pPerEBasicExcludingExtraordinaryItemsTTM',
                    'metric_pPerEIncludingExtraordinaryItemsTTM',
                    'metric_returnOnAverageEquityMostRecentFiscalYear',
                    'metric_returnOnInvestmentMostRecentFiscalYear',
                    'metric_netProfitMarginPercentTrailing12Month',
                    'metric_operatingMarginTrailing12Month',
                    'metric_grossMarginTrailing12Month',
                    'metric_currentRatioMostRecentFiscalYear',
                    'metric_quickRatioMostRecentFiscalYear',
                    'metric_totalDebtPerTotalEquityMostRecentFiscalYear',
                    'metric_netInterestCoverageMostRecentFiscalYear',
                    'metric_marketCap',
                    'metric_beta'
                ]
            },
            'sentiment': {
                'description': 'News and social media sentiment indicators',
                'features': [
                    'reddit_title_sentiments_mean', 'reddit_title_sentiments_std',
                    'reddit_body_sentiments', 'reddit_body_sentiments_std',
                    'reddit_score_mean', 'reddit_score_sum', 'reddit_posts_count', 'reddit_comments_sum',
                    'news_sentiment_mean', 'news_articles_count', 'news_sentiment_std', 'news_sources',
                    'sentiment_momentum_3d', 'sentiment_momentum_5d',
                    'sentiment_extreme_positive', 'sentiment_extreme_negative'
                ]
            }
        }
    
    def get_features_for_combination(self, combination: str) -> List[str]:
        """Get list of features for a given combination string"""
        if combination.lower() == 'all':
            all_features = []
            for category_features in self.feature_categories.values():
                all_features.extend(category_features['features'])
            return list(set(all_features))
        
        categories = [cat.strip().lower() for cat in combination.split('+')]
        valid_categories = set(self.feature_categories.keys())
        invalid_categories = set(categories) - valid_categories
        if invalid_categories:
            raise ValueError(f"Invalid feature categories: {invalid_categories}. "
                           f"Valid categories are: {list(valid_categories)}")
        
        selected_features = []
        for category in categories:
            selected_features.extend(self.feature_categories[category]['features'])
        
        selected_features = list(set(selected_features))
        if 'close' in selected_features:
            selected_features.remove('close')
            selected_features = ['close'] + selected_features
        
        return selected_features
    
    def print_available_combinations(self):
        """Print available feature combinations"""
        print("\n=== Available Feature Categories ===")
        for category, info in self.feature_categories.items():
            print(f"\n{category.upper()}: {info['description']}")
            print(f"  Features ({len(info['features'])}): {', '.join(info['features'][:5])}...")
        
        print(f"\n=== Example Combinations ===")
        print("• ohlcv - Basic price data only")
        print("• technical - Technical indicators only")
        print("• ohlcv+technical - Price data + technical indicators")
        print("• all - All available features")


# ============================================================================
# CURRICULUM LEARNING CONFIG
# ============================================================================

class CurriculumConfig:
    def __init__(self):
        # Updated for longer training duration
        self.stages = {
            1: {
                "stage_num": 1,
                "name": "Exploration",
                "description": "Initial exploration stage with high entropy",
                "exploration_std": 1.0,
                "entropy_coeff": 0.02,
                "clip_eps": 0.3,
                "learning_rate": 3e-4,
                "num_epochs": 800,  # Increased from 300
                "epsilon_uniform": 0.2,
                "reward_scaling": 1.0,
                "enable_constraints": False,
                "early_stopping_patience": 300,
                "early_stopping_min_delta": 0.001
            },
            2: {
                "stage_num": 2,
                "name": "Refinement",
                "description": "Refinement of learned policies with balanced parameters",
                "exploration_std": 0.7,
                "entropy_coeff": 0.01,
                "clip_eps": 0.2,
                "learning_rate": 1e-4,
                "num_epochs": 1000,  # Increased from 500
                "epsilon_uniform": 0.1,
                "reward_scaling": 1.5,
                "enable_constraints": True,
                "early_stopping_patience": 400,
                "early_stopping_min_delta": 0.003
            },
            3: {
                "stage_num": 3,
                "name": "Optimization",
                "description": "Final optimization stage with focused learning",
                "exploration_std": 0.5,
                "entropy_coeff": 0.005,
                "clip_eps": 0.1,
                "learning_rate": 5e-5,
                "num_epochs": 800,  # Increased from 700
                "epsilon_uniform": 0.0,
                "reward_scaling": 2.0,
                "enable_constraints": True,
                "early_stopping_patience": 500,
                "early_stopping_min_delta": 0.005
            }
        }

    def get_stage(self, stage_num):
        return self.stages.get(stage_num, None)


# ============================================================================
# CUSTOM PORTFOLIO ENVIRONMENT
# ============================================================================

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
        Engineer only requested features. Returns DataFrame with exactly
        the columns in self.selected_features (order preserved).
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

            # Fill any requested features that still don't exist with zeros
            for feat in self.selected_features:
                if feat not in df_engineered.columns:
                    logger.warning(f"Feature '{feat}' could not be generated for {stock}, filling with zeros")
                    df_engineered[feat] = 0.0

            # Final enforcement: reorder columns to match selected_features exactly
            df_engineered = df_engineered[self.selected_features]

            logger.info(f"CustomDataLoader: Generated {len(df_engineered.columns)} features for {stock}")
            return df_engineered.fillna(0.0)

        except Exception as e:
            logger.error(f"Error in CustomDataLoader.engineer_features for {stock}: {e}", exc_info=True)
            # Return DataFrame of zeros with selected features
            return pd.DataFrame(0.0, index=df.index, columns=self.selected_features)


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
        which bypasses our custom loader. Instead, we replicate the parent's init logic
        but use our CustomDataLoader.
        """
        try:
            # Store and normalize selected features FIRST
            self.selected_features = [str(f) for f in selected_features]
            self.features = list(self.selected_features)
            self.n_features = len(self.selected_features)
            self.use_all_features = False

            logger.info(f"CustomPortfolioEnv: Initializing with {len(self.selected_features)} features")

            # === REPLICATE PARENT'S __init__ LOGIC ===
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
                force_reload=False  # Use cache when available
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
        from pathlib import Path
        stocks_file = Path("finagent/stocks.txt")
        if not stocks_file.exists():
            stocks_file = Path("FYP-FinAgent/finagent/stocks.txt")

        if stocks_file.exists():
            with open(stocks_file, 'r') as f:
                stocks = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"Loaded {len(stocks)} stocks from file")
                return stocks

        # Fallback: scan directory
        from pathlib import Path
        data_path = Path(self.data_root) if isinstance(self.data_root, str) else self.data_root
        stocks = [p.stem.replace('_aligned', '') for p in data_path.glob("*_aligned.csv")]
        logger.info(f"Scanned {len(stocks)} stocks from directory")
        return stocks


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Trajectory(NamedTuple):
    """Trajectory data structure for PPO"""
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array

class LSTMState(NamedTuple):
    """Dummy LSTMState for compatibility"""
    h: chex.Array
    c: chex.Array

def safe_normalize(x: chex.Array, eps: float = 1e-8) -> chex.Array:
    """Safely normalize array"""
    std = jnp.std(x)
    mean = jnp.mean(x)
    std = jnp.maximum(std, eps)
    normalized = (x - mean) / (std + eps)
    return jnp.clip(normalized, -10.0, 10.0)

def check_for_nans(x: chex.Array, name: str = "array") -> bool:
    """Check for NaN values in array"""
    return jnp.any(jnp.isnan(x))


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class MLPHead(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Single Transformer Encoder Layer"""
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: chex.Array, training: bool):
        norm_x = nn.LayerNorm()(x)
        attn_output = nn.SelfAttention(num_heads=self.nhead, qkv_features=self.d_model)(norm_x)
        attn_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(attn_output)
        x = x + attn_output

        norm_x = nn.LayerNorm()(x)
        ff_output = nn.Dense(self.dim_feedforward)(norm_x)
        ff_output = nn.relu(ff_output)
        ff_output = nn.Dense(self.d_model)(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_output)
        x = x + ff_output
        return x


class ActorCriticTransformerFlat(nn.Module):
    """Transformer-based Actor-Critic network"""
    action_dim: int
    obs_dim: int
    window_size: int
    n_stocks: int
    n_features: int
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout_rate: float = 0.1

    def setup(self):
        self.historical_size = self.window_size * self.n_stocks * self.n_features
        self.current_info_size = self.n_stocks * 2
        self.portfolio_weights_size = self.action_dim
        self.short_positions_size = self.n_stocks
        self.market_state_size = 8
        
        expected_obs_size = (self.historical_size + self.current_info_size + 
                           self.portfolio_weights_size + self.short_positions_size + self.market_state_size)
        assert self.obs_dim == expected_obs_size

        self.historical_proj = nn.Dense(self.d_model, kernel_init=nn.initializers.he_normal())
        
        max_seq_len = self.window_size * self.n_stocks * 2
        self.pos_embedding = self.param('pos_embedding', 
                                       nn.initializers.zeros, 
                                       (1, max_seq_len, self.d_model))

        self.transformer_blocks = [
            TransformerEncoderBlock(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=4 * self.d_model,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_layers)
        ]
        self.transformer_norm = nn.LayerNorm()

        self.current_info_proj = nn.Dense(self.d_model // 2, kernel_init=nn.initializers.he_normal())
        self.portfolio_proj = nn.Dense(self.d_model // 2, kernel_init=nn.initializers.he_normal())
        self.short_positions_proj = nn.Dense(self.d_model // 4, kernel_init=nn.initializers.he_normal())
        self.market_state_proj = nn.Dense(self.d_model // 4, kernel_init=nn.initializers.he_normal())

        fusion_input_size = self.d_model + self.d_model // 2 + self.d_model // 2 + self.d_model // 4 + self.d_model // 4
        self.fusion_layer = nn.Dense(self.d_model, kernel_init=nn.initializers.he_normal())

        self.actor_head = MLPHead(self.d_model // 2, self.action_dim)
        self.critic_head = MLPHead(self.d_model // 2, 1)

    @nn.compact
    def __call__(self, x: chex.Array, training: bool = True):
        batch_size = x.shape[0]
        x = jnp.where(jnp.isnan(x), 0.0, x)
        x = jnp.where(jnp.isinf(x), jnp.sign(x) * 10.0, x)

        historical_data = x[:, :self.historical_size]
        start_idx = self.historical_size
        
        current_info = x[:, start_idx:start_idx + self.current_info_size]
        start_idx += self.current_info_size
        
        portfolio_weights = x[:, start_idx:start_idx + self.portfolio_weights_size]
        start_idx += self.portfolio_weights_size
        
        short_positions = x[:, start_idx:start_idx + self.short_positions_size]
        start_idx += self.short_positions_size
        
        market_state = x[:, start_idx:start_idx + self.market_state_size]

        historical_reshaped = historical_data.reshape(batch_size, -1, self.n_features)
        
        hist_mean = jnp.mean(historical_reshaped, axis=(-1, -2), keepdims=True)
        hist_std = jnp.std(historical_reshaped, axis=(-1, -2), keepdims=True)
        hist_std = jnp.maximum(hist_std, 1e-8)
        historical_normalized = (historical_reshaped - hist_mean) / hist_std
        historical_normalized = jnp.clip(historical_normalized, -5.0, 5.0)

        historical_encoded = self.historical_proj(historical_normalized)
        seq_len = historical_encoded.shape[1]
        historical_encoded = historical_encoded + self.pos_embedding[:, :seq_len, :]

        for block in self.transformer_blocks:
            historical_encoded = block(historical_encoded, training=training)
        
        historical_encoded = self.transformer_norm(historical_encoded)
        historical_features = historical_encoded.mean(axis=1)

        current_features = self.current_info_proj(jnp.clip(current_info, -10.0, 10.0))
        current_features = nn.relu(current_features)

        portfolio_features = self.portfolio_proj(jnp.clip(portfolio_weights, 0.0, 1.0))
        portfolio_features = nn.relu(portfolio_features)

        short_features = self.short_positions_proj(jnp.clip(short_positions, 0.0, 1.0))
        short_features = nn.relu(short_features)

        market_features = self.market_state_proj(jnp.clip(market_state, -10.0, 10.0))
        market_features = nn.relu(market_features)

        combined_features = jnp.concatenate([
            historical_features,
            current_features, 
            portfolio_features,
            short_features,
            market_features
        ], axis=-1)
        
        fused_features = self.fusion_layer(combined_features)
        fused_features = nn.relu(fused_features)
        fused_features = jnp.where(jnp.isnan(fused_features), 0.0, fused_features)
        fused_features = jnp.clip(fused_features, -10.0, 10.0)

        logits = self.actor_head(fused_features)
        logits = jnp.where(jnp.isnan(logits), 0.0, logits)
        logits = jnp.clip(logits, -10.0, 10.0)

        values = self.critic_head(fused_features).squeeze(-1)
        values = jnp.where(jnp.isnan(values), 0.0, values)
        values = jnp.clip(values, -100.0, 100.0)

        new_carry = [LSTMState(h=jnp.zeros((batch_size, 1)), c=jnp.zeros((batch_size, 1)))]

        return logits, values, new_carry


# ============================================================================
# PPO TRAINER WITH FEATURE COMBINATIONS AND CURRICULUM LEARNING
# ============================================================================

class PPOTrainer:
    def __init__(self, config: Dict[str, Any], selected_features: List[str]):
        self.config = config
        self.selected_features = selected_features
        self.nan_count = 0
        self.max_nan_resets = 5
        
        # Apply curriculum settings if provided
        self.curriculum_stage = config.get('curriculum_stage', None)
        self.curriculum_config = config.get('curriculum_config', None)
        
        if self.curriculum_stage and self.curriculum_config:
            self._apply_curriculum_settings()
        
        logger.info("Initializing PPO Trainer with Transformer...")

        # Initialize environment with custom feature selection
        try:
            env_config = self._get_env_config()
            self.env = CustomPortfolioEnv(selected_features=selected_features, **env_config)
            logger.info(f"Environment initialized: obs_dim={self.env.obs_dim}, action_dim={self.env.action_dim}")
            
            self.window_size = config.get('window_size', 30)
            self.n_stocks = self.env.n_stocks
            self.n_features = self.env.n_features
            self.historical_obs_size = self.window_size * self.n_stocks * self.n_features
            
            logger.info(f"Environment dimensions: window_size={self.window_size}, "
                       f"n_stocks={self.n_stocks}, n_features={self.n_features}")

        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            raise

        self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0,))
        self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0))
        
        self.network = ActorCriticTransformerFlat(
            action_dim=self.env.action_dim,
            obs_dim=self.env.obs_dim,
            window_size=self.window_size,
            n_stocks=self.n_stocks,
            n_features=self.n_features,
            d_model=config.get('d_model', 64),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 2),
            dropout_rate=config.get('dropout_rate', 0.1)
        )

        self._initialize_parameters()
        self._initialize_environment_state()
        
        # Initialize training metrics
        self.best_performance = -1e10
        self.patience_counter = 0
        self.global_step = 0
        self.current_raw_rewards = None

    def _apply_curriculum_settings(self):
        """Apply curriculum-specific settings"""
        stage = self.curriculum_config.get_stage(self.curriculum_stage)
        logger.info(f"Initializing Stage {stage['stage_num']}: {stage['name']}")
        logger.info(f"Description: {stage['description']}")

        self.config.update({
            'action_std': stage['exploration_std'],
            'entropy_coeff': stage['entropy_coeff'],
            'clip_eps': stage['clip_eps'],
            'learning_rate': stage['learning_rate'],
            'total_timesteps': stage['num_epochs'] * self.config.get('n_envs', 8) * self.config.get('n_steps', 128),
        })

        # Early stopping settings from curriculum config
        early_stopping_patience = stage.get('early_stopping_patience', 300)
        early_stopping_min_delta = stage.get('early_stopping_min_delta', 0.001)
        self.config.update({
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta
        })
        logger.info(f"Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

    def _get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration"""
        return {
            'data_root': self.config['data_root'],
            'stocks': self.config.get('stocks', None),
            'start_date': self.config['train_start_date'],
            'end_date': self.config['train_end_date'],
            'window_size': self.config.get('window_size', 30),
            'transaction_cost_rate': self.config.get('transaction_cost_rate', 0.005),
            'sharpe_window': self.config.get('sharpe_window', 252),
            'use_all_features': False,  # We're using custom feature selection
            'hdf5_file': self.config.get('hdf5_file', None)
        }

    def _initialize_parameters(self):
        """Initialize network parameters"""
        logger.info("Initializing network parameters...")
        
        self.rng = random.PRNGKey(self.config.get('seed', 42))
        self.rng, init_rng = random.split(self.rng)
        
        dummy_obs = jnp.ones((self.config.get('n_envs', 8), self.env.obs_dim))
        
        try:
            self.params = self.network.init(init_rng, dummy_obs)
            logger.info("Network parameters initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize network parameters: {e}")
            raise
        
        if self._has_nan_params(self.params):
            logger.warning("NaN detected in initial parameters, reinitializing...")
            self.rng, init_rng = random.split(self.rng)
            self.params = self.network.init(init_rng, dummy_obs)
            
            if self._has_nan_params(self.params):
                raise RuntimeError("Failed to initialize parameters without NaN")
        
        self._test_network_forward_pass(dummy_obs)
        
        self.optimizer = optax.adam(learning_rate=self.config.get('learning_rate', 1e-4))
        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=self.params,
            tx=self.optimizer
        )
        logger.info("Train state initialized successfully")

    def _has_nan_params(self, params) -> bool:
        """Check for NaN values in parameters"""
        def check_nan(x):
            return jnp.any(jnp.isnan(x)) if jnp.issubdtype(x.dtype, jnp.floating) else False
        
        has_nan = jax.tree_util.tree_reduce(
            lambda acc, x: acc | check_nan(x),
            params, False
        )
        return has_nan

    def _test_network_forward_pass(self, obs: chex.Array):
        """Test network forward pass"""
        logger.info("Testing network forward pass...")
        
        try:
            self.rng, new_rng = random.split(self.rng)
            logits, values, _ = self.network.apply(self.params, obs, rngs={"dropout": new_rng})
            
            if check_for_nans(logits, "logits"):
                raise RuntimeError("NaN detected in logits output")
            if check_for_nans(values, "values"):
                raise RuntimeError("NaN detected in values output")
            
            logger.info("Network forward pass test passed")
        
        except Exception as e:
            logger.error(f"Network forward pass test failed: {e}")
            raise

    def _initialize_environment_state(self):
        """Initialize environment state"""
        logger.info("Initializing environment state...")
        
        try:
            self.rng, *reset_keys = random.split(self.rng, self.config.get('n_envs', 8) + 1)
            reset_keys = jnp.array(reset_keys)
            self.env_states, self.obs = self.vmap_reset(reset_keys)
            
            self.obs = jnp.where(jnp.isnan(self.obs), 0.0, self.obs)
            self.obs = jnp.where(jnp.isinf(self.obs), 0.0, self.obs)
            
            if check_for_nans(self.obs, "initial observations"):
                raise RuntimeError("NaN detected in initial environment observations")
            
            logger.info("Environment state initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize environment state: {e}")
            raise

    def collect_trajectory(self, train_state: TrainState, env_states: List[EnvState],
                          initial_obs: chex.Array, rng_key: chex.PRNGKey) -> Tuple[Trajectory, List[EnvState], chex.Array]:
        """Collect trajectory using current policy"""
        
        def step_fn(carry_step, _):
            env_states, obs, rng_key = carry_step
            
            obs = jnp.where(jnp.isnan(obs), 0.0, obs)
            obs = jnp.clip(obs, -50.0, 50.0)
            
            rng_key, action_rng = random.split(rng_key)
            
            try:
                logits, values, _ = train_state.apply_fn(
                    train_state.params, obs, rngs={"dropout": action_rng}
                )
            except Exception as e:
                logger.error(f"Network forward pass failed: {e}")
                logits = jnp.zeros((obs.shape[0], self.env.action_dim))
                values = jnp.zeros(obs.shape[0])
            
            logits = jnp.where(jnp.isnan(logits), 0.0, logits)
            values = jnp.where(jnp.isnan(values), 0.0, values)
            logits = jnp.clip(logits, -5.0, 5.0)
            values = jnp.clip(values, -50.0, 50.0)
            
            action_std = self.config.get('action_std', 1.0)
            action_std = jnp.maximum(action_std, 1e-6)
            
            action_distribution = distrax.Normal(loc=logits, scale=action_std)
            actions = action_distribution.sample(seed=action_rng)
            actions = jnp.clip(actions, -5.0, 5.0)
            
            log_probs = action_distribution.log_prob(actions).sum(axis=-1)
            log_probs = jnp.where(jnp.isnan(log_probs), -10.0, log_probs)
            log_probs = jnp.clip(log_probs, -50.0, 10.0)
            
            new_env_states, next_obs, rewards, dones, info = self.vmap_step(env_states, actions)
            
            next_obs = jnp.where(jnp.isnan(next_obs), 0.0, next_obs)
            next_obs = jnp.clip(next_obs, -50.0, 50.0)
            
            rewards = jnp.where(jnp.isnan(rewards), 0.0, rewards)
            rewards = jnp.clip(rewards, -50.0, 50.0)
            
            transition = Trajectory(
                obs=obs,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                dones=dones,
            )
            
            return (new_env_states, next_obs, rng_key), transition
        
        n_steps = self.config.get('n_steps', 64)
        initial_carry = (env_states, initial_obs, rng_key)
        
        try:
            current_carry = initial_carry
            trajectory_list = []
            
            for step in range(n_steps):
                current_carry, transition = step_fn(current_carry, None)
                trajectory_list.append(transition)
            
            trajectory = jax.tree_util.tree_map(
                lambda *args: jnp.stack(args, axis=0), *trajectory_list
            )
        
        except Exception as e:
            logger.error(f"Trajectory collection failed: {e}")
            return self._create_empty_trajectory(), env_states, initial_obs
        
        final_env_states, final_obs, _ = current_carry
        return trajectory, final_env_states, final_obs

    def _create_empty_trajectory(self) -> Trajectory:
        """Create empty trajectory for error recovery"""
        n_steps = self.config.get('n_steps', 128)
        n_envs = self.config.get('n_envs', 8)
        
        return Trajectory(
            obs=jnp.zeros((n_steps, n_envs, self.env.obs_dim)),
            actions=jnp.zeros((n_steps, n_envs, self.env.action_dim)),
            rewards=jnp.zeros((n_steps, n_envs)),
            values=jnp.zeros((n_steps, n_envs)),
            log_probs=jnp.zeros((n_steps, n_envs)),
            dones=jnp.zeros((n_steps, n_envs), dtype=bool),
        )

    @partial(jax.jit, static_argnums=(0,))
    def compute_gae(self, trajectory: Trajectory, last_values: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Compute Generalized Advantage Estimation"""
        gamma = self.config.get('gamma', 0.99)
        gae_lambda = self.config.get('gae_lambda', 0.95)
        
        rewards = jnp.where(jnp.isnan(trajectory.rewards), 0.0, trajectory.rewards)
        values = jnp.where(jnp.isnan(trajectory.values), 0.0, trajectory.values)
        last_values = jnp.where(jnp.isnan(last_values), 0.0, last_values)
        
        rewards = jnp.clip(rewards, -100.0, 100.0)
        values = jnp.clip(values, -100.0, 100.0)
        last_values = jnp.clip(last_values, -100.0, 100.0)
        
        extended_values = jnp.concatenate([values, last_values[None, :]], axis=0)
        
        def gae_step(gae_carry, inputs):
            current_gae = gae_carry
            reward, value, next_value, done = inputs
            
            delta = reward + gamma * next_value * (1 - done) - value
            advantage = delta + gamma * gae_lambda * (1 - done) * current_gae
            
            advantage = jnp.where(jnp.isnan(advantage), 0.0, advantage)
            advantage = jnp.clip(advantage, -100.0, 100.0)
            
            return advantage, advantage
        
        gae_inputs = (
            rewards[::-1],
            values[::-1],
            extended_values[1:][::-1],
            trajectory.dones[::-1]
        )
        
        init_gae = jnp.zeros_like(last_values)
        _, advantages_reversed = lax.scan(gae_step, init_gae, gae_inputs)
        advantages = advantages_reversed[::-1]
        
        advantages = jnp.where(jnp.isnan(advantages), 0.0, advantages)
        advantages = jnp.clip(advantages, -100.0, 100.0)
        
        returns = advantages + values
        returns = jnp.where(jnp.isnan(returns), 0.0, returns)
        returns = jnp.clip(returns, -100.0, 100.0)
        
        return advantages, returns

    @partial(jax.jit, static_argnums=(0,))
    def ppo_loss(self, params: chex.Array, train_batch: Trajectory, gae_advantages: chex.Array,
                 returns: chex.Array, rng_key: chex.PRNGKey) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """Compute PPO loss"""
        
        obs_batch = jnp.where(jnp.isnan(train_batch.obs), 0.0, train_batch.obs)
        actions_batch = jnp.where(jnp.isnan(train_batch.actions), 0.0, train_batch.actions)
        old_log_probs_batch = jnp.where(jnp.isnan(train_batch.log_probs), -10.0, train_batch.log_probs)
        
        batch_size = obs_batch.shape[0]
        
        try:
            logits, values, _ = self.network.apply(params, obs_batch)
        except Exception as e:
            logger.error(f"Network forward pass failed in PPO loss: {e}")
            logits = jnp.zeros((batch_size, self.env.action_dim))
            values = jnp.zeros(batch_size)
        
        logits = jnp.where(jnp.isnan(logits), 0.0, logits)
        values = jnp.where(jnp.isnan(values), 0.0, values)
        logits = jnp.clip(logits, -10.0, 10.0)
        values = jnp.clip(values, -100.0, 100.0)
        
        returns = jnp.where(jnp.isnan(returns), 0.0, returns)
        gae_advantages = jnp.where(jnp.isnan(gae_advantages), 0.0, gae_advantages)
        returns = jnp.clip(returns, -100.0, 100.0)
        
        gae_advantages = safe_normalize(gae_advantages)
        
        old_values = jnp.where(jnp.isnan(train_batch.values), 0.0, train_batch.values)
        old_values = jnp.clip(old_values, -100.0, 100.0)
        
        value_pred_clipped = old_values + (values - old_values).clip(
            -self.config.get('clip_eps', 0.2),
            self.config.get('clip_eps', 0.2)
        )
        
        value_losses = jnp.square(values - returns)
        value_losses_clipped = jnp.square(value_pred_clipped - returns)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        
        action_std = self.config.get('action_std', 1.0)
        action_std = jnp.maximum(action_std, 1e-6)
        
        action_distribution = distrax.Normal(loc=logits, scale=action_std)
        new_log_probs = action_distribution.log_prob(actions_batch).sum(axis=-1)
        new_log_probs = jnp.where(jnp.isnan(new_log_probs), -10.0, new_log_probs)
        new_log_probs = jnp.clip(new_log_probs, -50.0, 10.0)
        
        log_ratio = new_log_probs - old_log_probs_batch
        log_ratio = jnp.clip(log_ratio, -10.0, 10.0)
        
        ratio = jnp.exp(jnp.clip(log_ratio, -10.0, 10.0))
        ratio = jnp.clip(ratio, 1e-8, 10.0)
        
        clip_eps = self.config.get('clip_eps', 0.2)
        pg_losses1 = ratio * gae_advantages
        pg_losses2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_advantages
        policy_loss = -jnp.minimum(pg_losses1, pg_losses2).mean()
        
        entropy = action_distribution.entropy().sum(axis=-1)
        entropy = jnp.where(jnp.isnan(entropy), 0.0, entropy)
        entropy = jnp.clip(entropy, 0.0, 10.0)
        entropy_coeff = self.config.get('entropy_coeff', 0.01)
        entropy_loss = -entropy_coeff * entropy.mean()
        
        value_coeff = self.config.get('value_coeff', 0.5)
        total_loss = policy_loss + value_coeff * value_loss + entropy_loss
        total_loss = jnp.where(jnp.isnan(total_loss), 0.0, total_loss)
        
        approx_kl = jnp.where(ratio == 0.0, 0.0, (ratio - 1) - jnp.log(ratio))
        approx_kl = jnp.where(jnp.isnan(approx_kl), 0.0, approx_kl)
        approx_kl = jnp.clip(approx_kl, -10.0, 10.0)
        
        clip_fraction = (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)
        
        metrics = {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'approx_kl': approx_kl.mean(),
            'clip_fraction': clip_fraction.mean(),
            'mean_ratio': ratio.mean(),
            'mean_advantage': gae_advantages.mean()
        }
        
        return total_loss, metrics

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, train_state: TrainState, train_batch: Trajectory, gae_advantages: chex.Array,
                   returns: chex.Array, rng_key: chex.PRNGKey) -> Tuple[TrainState, Dict[str, chex.Array]]:
        """Perform a single PPO training step"""
        
        grad_fn = jax.value_and_grad(self.ppo_loss, has_aux=True)
        (total_loss, metrics), grads = grad_fn(
            train_state.params, train_batch, gae_advantages, returns, rng_key
        )
        
        train_state = train_state.apply_gradients(grads=grads)
        
        if self._has_nan_params(grads):
            logger.warning("NaN detected in gradients!")
            metrics['nan_in_gradients'] = 1.0
        else:
            metrics['nan_in_gradients'] = 0.0

        return train_state, metrics

    def compute_robust_metrics(self, rewards: chex.Array) -> Dict[str, float]:
        """Compute simple, robust metrics like Sharpe ratio from RAW rewards.

        CRITICAL: This should be called with RAW rewards (before normalization)
        not normalized rewards which would make Sharpe ratio always ≈0.
        """
        try:
            # Ensure rewards is a valid array and not empty
            if rewards is None or (hasattr(rewards, 'size') and rewards.size == 0):
                raise ValueError("Rewards are empty or None")
            rewards = jnp.array(rewards)
            rewards = jnp.where(jnp.isfinite(rewards), rewards, 0.0)

            # Sum rewards per env across time (axis 0 is time dimension)
            if rewards.ndim > 1:
                per_env_return = jnp.sum(rewards, axis=0)
            else:
                per_env_return = rewards

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

    def check_early_stopping(self, avg_reward: float) -> bool:
        """Check if early stopping criteria are met"""
        improvement = avg_reward - self.best_performance
        
        if improvement > self.config.get('early_stopping_min_delta', 0.005):
            self.best_performance = avg_reward
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.config.get('early_stopping_patience', 150):
            logger.info(f"Early stopping triggered after {self.patience_counter} steps without improvement")
            return True
        
        return False

    def update_stage_parameters(self, stage_params: Dict[str, Any]):
        """Update trainer parameters for a new curriculum stage."""
        logger.info("Updating trainer parameters for new stage:")
        for key, value in stage_params.items():
            if key in self.config:
                logger.info(f"  {key}: {self.config[key]} -> {value}")
                self.config[key] = value
            if key == "curriculum_stage":
                # Update the curriculum stage tracker
                self.curriculum_stage = value
            if key == "global_step":
                # Track global step across stages (for logging)
                self.global_step = value

        # Reset stage-specific counters and metrics
        self.patience_counter = 0

        # Update optimizer learning rate if changed
        if 'learning_rate' in stage_params:
            import optax
            from flax.training.train_state import TrainState
            new_optimizer = optax.adam(stage_params['learning_rate'])
            self.train_state = TrainState.create(
                apply_fn=self.network.apply,
                params=self.train_state.params,  # Preserve parameters
                tx=new_optimizer
            )

    def train(self):
        """Main training loop"""
        logger.info("Starting PPO training with Transformer...")
        
        n_envs = self.config.get('n_envs', 8)
        n_steps = self.config.get('n_steps', 64)
        total_timesteps = self.config.get('total_timesteps', 1_000_000)
        n_updates = total_timesteps // (n_envs * n_steps)
        n_minibatch = self.config.get('n_minibatch', 4)
        update_epochs = self.config.get('update_epochs', 4)
        
        batch_size = n_envs * n_steps
        minibatch_size = batch_size // n_minibatch
        
        if batch_size % n_minibatch != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by n_minibatch ({n_minibatch})")
        
        global_step = 0
        start_time = time.time()
        
        env_states = self.env_states
        obs = self.obs
        
        for update in range(n_updates):
            self.rng, collect_rng = random.split(self.rng)
            
            try:
                trajectory, env_states, obs = self.collect_trajectory(
                    self.train_state, env_states, obs, collect_rng
                )

                # Store raw rewards BEFORE any normalization for metrics computation
                self.current_raw_rewards = trajectory.rewards

                if any(check_for_nans(getattr(trajectory, field), field) for field in Trajectory._fields):
                    logger.error(f"NaN detected in trajectory at update {update}")
                    self.nan_count += 1
                    if self.nan_count > self.max_nan_resets:
                        raise RuntimeError("Too many NaN resets")
                    self._initialize_environment_state()
                    env_states = self.env_states
                    obs = self.obs
                    continue
            
            except Exception as e:
                logger.error(f"Trajectory collection failed at update {update}: {e}")
                self.nan_count += 1
                if self.nan_count > self.max_nan_resets:
                    raise RuntimeError("Too many NaN resets")
                self._initialize_environment_state()
                env_states = self.env_states
                obs = self.obs
                continue
            
            self.rng, last_value_rng = random.split(self.rng)
            try:
                _, last_values, _ = self.network.apply(self.train_state.params, obs)
                last_values = jnp.where(jnp.isnan(last_values), 0.0, last_values)
                last_values = jnp.clip(last_values, -100.0, 100.0)
            except Exception as e:
                logger.error(f"Failed to compute last values: {e}")
                last_values = jnp.zeros(n_envs)
            
            gae_advantages, returns = self.compute_gae(trajectory, last_values)
            
            flat_trajectory = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), trajectory
            )
            flat_gae_advantages = gae_advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            
            if check_for_nans(flat_gae_advantages, "advantages") or check_for_nans(flat_returns, "returns"):
                logger.error(f"NaN detected in GAE at update {update}")
                self.nan_count += 1
                if self.nan_count > self.max_nan_resets:
                    raise RuntimeError("Too many NaN resets")
                continue
            
            for epoch in range(update_epochs):
                self.rng, shuffle_rng = random.split(self.rng)
                permutation = jax.random.permutation(shuffle_rng, batch_size)
                
                for i in range(n_minibatch):
                    batch_indices = permutation[i * minibatch_size : (i + 1) * minibatch_size]
                    
                    minibatch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, batch_indices, axis=0), flat_trajectory
                    )
                    minibatch_advantages = jnp.take(flat_gae_advantages, batch_indices, axis=0)
                    minibatch_returns = jnp.take(flat_returns, batch_indices, axis=0)
                    
                    self.rng, train_rng = random.split(self.rng)
                    try:
                        self.train_state, metrics = self.train_step(
                            self.train_state, minibatch, minibatch_advantages, minibatch_returns, train_rng
                        )
                        
                        if self.config.get('use_wandb', False):
                            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)
                    
                    except Exception as e:
                        logger.error(f"Training step failed: {e}")
                        self.nan_count += 1
                        if self.nan_count > self.max_nan_resets:
                            raise RuntimeError("Too many NaN resets")
                        break
            
            global_step += batch_size

            # Initialize metrics if not set (in case of error)
            if 'metrics' not in locals():
                metrics = {'approx_kl': 0.0}

            if update % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_reward = float(trajectory.rewards.mean())

                # Compute robust metrics from RAW rewards (before normalization)
                robust_metrics = self.compute_robust_metrics(self.current_raw_rewards)

                logger.info(f"Update: {update}/{n_updates}, Steps: {global_step}, "
                           f"Avg Reward: {avg_reward:.4f}, "
                           f"Sharpe Ratio: {robust_metrics['sharpe_ratio']:.4f}, "
                           f"Time: {elapsed_time:.2f}s")

                if self.config.get('use_wandb', False):
                    wandb.log({
                        "chart/SPS": global_step / elapsed_time,
                        "chart/avg_reward": avg_reward,
                        # Performance metrics (from RAW rewards)
                        "performance/sharpe_ratio": robust_metrics.get('sharpe_ratio', 0.0),
                        "performance/mean_return": robust_metrics.get('mean_return', 0.0),
                        "performance/std_return": robust_metrics.get('std_reward', 1.0),
                        # Early stopping
                        "early_stopping/patience_counter": self.patience_counter,
                        "early_stopping/best_performance": self.best_performance
                    }, step=global_step)

                # Early stopping check using Sharpe ratio from raw rewards
                if self.check_early_stopping(robust_metrics['sharpe_ratio']):
                    logger.info("Early stopping triggered")
                    break
            
            if metrics['approx_kl'] > self.config.get('target_kl', 0.015) * 1.5:
                logger.warning(f"KL divergence {metrics['approx_kl']:.4f} exceeded target. Early stopping.")
                break
            
            if update % self.config.get('save_interval', 100) == 0 and update > 0:
                self.save_model(f"checkpoint_{update}.pkl")
        
        self.save_model("final_model.pkl")
        if self.config.get('use_wandb', False):
            wandb.finish()
        logger.info("Training complete!")

    def save_model(self, filename: str = "model.pkl"):
        """Save model parameters"""
        model_dir = Path(self.config.get('model_dir', './models'))
        model_dir.mkdir(parents=True, exist_ok=True)
        filepath = model_dir / filename
        with open(filepath, "wb") as f:
            f.write(serialization.to_bytes(self.train_state.params))
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filename: str = "model.pkl"):
        """Load model parameters"""
        model_dir = Path(self.config.get('model_dir', './models'))
        filepath = model_dir / filename
        with open(filepath, "rb") as f:
            params_bytes = f.read()
        
        dummy_obs = jnp.ones((1, self.env.obs_dim))
        temp_params = self.network.init(random.PRNGKey(0), dummy_obs)
        loaded_params = serialization.from_bytes(temp_params, params_bytes)
        self.train_state = self.train_state.replace(params=loaded_params)
        logger.info(f"Model loaded from {filepath}")


# ============================================================================
# MAIN FUNCTION WITH CLI
# ============================================================================

def run_training_with_combination(feature_combination: str, config: Dict[str, Any]):
    """Run training with a specific feature combination"""
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
    config['model_name'] = f"ppo_transformer_{feature_combination.replace('+', '_')}"
    
    try:
        trainer = PPOTrainer(config, selected_features)
        logger.info("Transformer PPO trainer initialized successfully")
        
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")
        
        trainer.save_model(f"final_model_{feature_combination.replace('+', '_')}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(
        description="Train PPO Transformer with different feature combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ppo_transformer_with_features.py --feature_combination ohlcv+technical
  python train_ppo_transformer_with_features.py --feature_combination all --total_timesteps 500000
  python train_ppo_transformer_with_features.py --feature_combination ohlcv+technical --curriculum_stage 1
  python train_ppo_transformer_with_features.py --feature_combination ohlcv+technical --auto_curriculum
        """
    )
    
    parser.add_argument('--feature_combination', type=str, default='ohlcv+technical',
                       help='Feature combination to use (e.g., ohlcv+technical, all)')
    
    # Curriculum learning
    parser.add_argument('--curriculum_stage', type=int, choices=[1, 2, 3],
                       help='Specific curriculum stage (1=exploration, 2=refinement, 3=optimization)')
    parser.add_argument('--auto_curriculum', action='store_true',
                       help='Automatically progress through all curriculum stages')
    parser.add_argument('--start_stage', type=int, default=1, choices=[1, 2, 3],
                       help='Stage to start from in auto mode')
    
    # Training config
    parser.add_argument('--total_timesteps', type=int, default=2_000_000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=128)
    parser.add_argument('--update_epochs', type=int, default=4)
    parser.add_argument('--n_minibatch', type=int, default=4)
    
    # Transformer config
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    # Data config
    parser.add_argument('--data_root', type=str, default='processed_data/')
    parser.add_argument('--train_start_date', type=str, default='2024-06-06')
    parser.add_argument('--train_end_date', type=str, default='2025-03-06')
    parser.add_argument('--window_size', type=int, default=30)
    
    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=150)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.005)
    
    # Other
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--list_combinations', action='store_true')
    
    args = parser.parse_args()
    
    if args.list_combinations:
        feature_selector = FeatureSelector()
        feature_selector.print_available_combinations()
        return
    
    config = {
        'seed': args.seed,
        'data_root': args.data_root,
        'train_start_date': args.train_start_date,
        'train_end_date': args.train_end_date,
        'window_size': args.window_size,
        'n_envs': args.n_envs,
        'n_steps': args.n_steps,
        'total_timesteps': args.total_timesteps,
        'learning_rate': args.learning_rate,
        'update_epochs': args.update_epochs,
        'n_minibatch': args.n_minibatch,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate,
        'use_wandb': args.use_wandb,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_min_delta': args.early_stopping_min_delta,
        'model_dir': args.model_dir,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'entropy_coeff': 0.01,
        'value_coeff': 0.5,
        'target_kl': 0.01,
        'action_std': 0.5,
        'transaction_cost_rate': 0.001,
        'sharpe_window': 252,
        'save_interval': 100,
    }
    
    # Curriculum learning configuration
    if args.curriculum_stage or args.auto_curriculum:
        curriculum_config = CurriculumConfig()
        config['curriculum_config'] = curriculum_config
        config['curriculum_stage'] = args.curriculum_stage
    
    try:
        if args.auto_curriculum:
            logger.info("Running FULL CURRICULUM (all stages)")
            curriculum_config = CurriculumConfig()

            feature_selector = FeatureSelector()
            selected_features = feature_selector.get_features_for_combination(args.feature_combination)
            config['selected_features'] = selected_features
            config['curriculum_config'] = curriculum_config
            logger.info(f"Selected {len(selected_features)} features for training")

            trainer = None
            global_step = 0
            for stage_num in range(args.start_stage, 4):
                logger.info(f"\n{'='*50}\nStarting Curriculum Stage {stage_num}\n{'='*50}")
                stage = curriculum_config.get_stage(stage_num)
                stage_params = {
                    'action_std': stage['exploration_std'],
                    'entropy_coeff': stage['entropy_coeff'],
                    'clip_eps': stage['clip_eps'],
                    'learning_rate': stage['learning_rate'],
                    'total_timesteps': stage['num_epochs'] * config.get('n_envs', 8) * config.get('n_steps', 128),
                    'early_stopping_patience': stage.get('early_stopping_patience', 300),
                    'early_stopping_min_delta': stage.get('early_stopping_min_delta', 0.001),
                    'curriculum_stage': stage_num,
                    'global_step': global_step
                }

                if trainer is None:
                    config['curriculum_stage'] = stage_num
                    trainer = PPOTrainer(config, selected_features)
                else:
                    trainer.update_stage_parameters(stage_params)

                # Train for this stage
                trainer.train()
                trainer.save_model(f"curriculum_stage_{stage_num}")
                global_step += stage['num_epochs']
                logger.info(f"Completed Curriculum Stage {stage_num}. Global step: {global_step}")
        else:
            run_training_with_combination(args.feature_combination, config)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()