import jax
import jax.numpy as jnp
from jax import random, vmap, lax
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, NamedTuple, Union
from pathlib import Path
import chex
from functools import partial
import h5py
from concurrent.futures import ThreadPoolExecutor
import os # Added for os.cpu_count()
import distrax
import json
import pickle

# JAX environment state
class EnvState(NamedTuple):
    """JAX-compatible environment state with short position support"""
    current_step: int
    portfolio_weights: chex.Array  # Current portfolio weights (stocks + cash, sums to 1)
    short_positions: chex.Array  # Track short positions: 1 if short, 0 if long/none
    short_entry_steps: chex.Array  # Track when short positions were entered (step numbers)
    done: bool
    total_return: float # Cumulative simple return, normalized to initial value of 1.0
    portfolio_value: float # Normalized to 1.0 at start, represents the multiplier of initial value
    sharpe_buffer: chex.Array  # Rolling buffer for Sharpe calculation (stores *daily_returns*)
    sharpe_buffer_idx: int
    portfolio_volatility: float
    max_drawdown: float
    rolling_correlation: chex.Array # Portfolio correlation with market


class JAXPortfolioDataLoader:
    """Optimized data loader for portfolio environment (CSV-based, flexible feature handling)"""

    def __init__(self, data_root: str, stocks: List[str], features: Optional[List[str]] = None,
                 use_all_features: bool = False, cache_dir: Optional[str] = None):
        self.data_root = Path(data_root)
        self.stocks = stocks
        self.features = features  # can be None, will be inferred
        self.use_all_features = use_all_features
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_root / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    def engineer_features(self, df: pd.DataFrame, stock: str) -> pd.DataFrame:
        """Add engineered features to data - Enhanced with algorithmic trading signals"""
        
        # === REMOVE TIME-INVARIANT FEATURES ===
        # Keep only the specific financial statement features that were previously chosen
        # These are the core financial statement features that change with quarterly/annual reports
        financial_statement_features = {
            # Core Income Statement Features
            'metric_Revenue', 'metric_TotalRevenue', 'metric_CostofRevenueTotal', 'metric_GrossProfit',
            'metric_OperatingIncome', 'metric_NetIncomeBeforeTaxes', 'metric_NetIncomeAfterTaxes',
            'metric_NetIncome', 'metric_DilutedNetIncome', 'metric_DilutedWeightedAverageShares',
            'metric_DilutedEPSExcludingExtraOrdItems', 'metric_DPS-CommonStockPrimaryIssue',
            
            # Core Balance Sheet Features  
            'metric_Cash', 'metric_ShortTermInvestments', 'metric_CashandShortTermInvestments',
            'metric_TotalCurrentAssets', 'metric_TotalAssets', 'metric_TotalCurrentLiabilities',
            'metric_TotalLiabilities', 'metric_TotalEquity', 'metric_TotalCommonSharesOutstanding',
            
            # Core Cash Flow Features
            'metric_CashfromOperatingActivities', 'metric_CapitalExpenditures', 
            'metric_CashfromInvestingActivities', 'metric_CashfromFinancingActivities',
            'metric_NetChangeinCash', 'metric_TotalCashDividendsPaid',
            
            # Key Financial Metrics
            'metric_freeCashFlowtrailing12Month', 'metric_freeCashFlowMostRecentFiscalYear',
            'metric_periodLength', 'metric_periodType'
        }
        
        # Remove truly time-invariant features (static fundamental metrics and corporate data)
        # Keep only the specific financial statement features we've chosen
        time_invariant_cols = [col for col in df.columns if 
                              (col.startswith('metric_') and col not in financial_statement_features) or 
                              col in ['period_end_date', 'dividend_amount', 'dividend_type', 'bonus_remarks']]
        
        # Keep only time-variant features
        df = df.drop(columns=time_invariant_cols, errors='ignore')
        
        # === BASIC PRICE FEATURES ===
        # Multiple timeframe returns
        df['returns_1d'] = df['close'].pct_change()
        df['returns_3d'] = df['close'].pct_change(periods=3)
        df['returns_5d'] = df['close'].pct_change(periods=5)
        df['returns_10d'] = df['close'].pct_change(periods=10)
        df['returns_20d'] = df['close'].pct_change(periods=20)
        
        # Log returns (more stationary)
        df['log_returns_1d'] = np.log(df['close'] / df['close'].shift(1))
        df['log_returns_5d'] = np.log(df['close'] / df['close'].shift(5))
        
        # === VOLATILITY & RISK FEATURES ===
        # Multiple timeframe volatility
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_10d'] = df['returns_1d'].rolling(10).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['volatility_30d'] = df['returns_1d'].rolling(30).std()
        df['volatility_60d'] = df['returns_1d'].rolling(60).std()
        
        # Volatility ratios (regime change detection)
        df['vol_ratio_short_long'] = df['volatility_10d'] / (df['volatility_30d'] + 1e-8)
        df['vol_ratio_5_20'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-8)
        
        # === MOMENTUM FEATURES ===
        # Price momentum at different horizons
        for period in [5, 10, 20, 60]:
            if len(df) > period:
                df[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1.0
        
        # Acceleration (momentum of momentum)
        df['momentum_acceleration_10d'] = df['momentum_10d'] - df['momentum_10d'].shift(5)
        
        # === TREND STRENGTH ===
        # Moving average convergence/divergence
        if all(col in df.columns for col in ['dma_50', 'dma_200']):
            df['ma_convergence'] = (df['dma_50'] - df['dma_200']) / df['dma_200']
            df['ma_trend_strength'] = df['ma_convergence'] - df['ma_convergence'].shift(5)
        
        # Price position within recent range
        df['price_position_20d'] = ((df['close'] - df['close'].rolling(20).min()) / 
                                   (df['close'].rolling(20).max() - df['close'].rolling(20).min() + 1e-8))
        
        # === ALGORITHMIC TRADING SIGNALS ===
        
        # 1. Z-SCORE MEAN REVERSION SIGNALS
        # Price z-score (mean reversion signal)
        df['price_zscore_20d'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-8)
        df['price_zscore_60d'] = (df['close'] - df['close'].rolling(60).mean()) / (df['close'].rolling(60).std() + 1e-8)
        
        # Volume z-score
        if 'volume' in df.columns:
            df['volume_zscore_20d'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-8)
            df['volume_zscore_60d'] = (df['volume'] - df['volume'].rolling(60).mean()) / (df['volume'].rolling(60).std() + 1e-8)
        
        # RSI z-score (if available)
        if 'rsi_14' in df.columns:
            df['rsi_zscore_20d'] = (df['rsi_14'] - df['rsi_14'].rolling(20).mean()) / (df['rsi_14'].rolling(20).std() + 1e-8)
        
        # 2. BOLLINGER BANDS SIGNALS
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        df['bb_std'] = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std'])
        
        # Bollinger Band position and signals
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']  # Low volatility signal
        df['bb_breakout_up'] = (df['close'] > df['bb_upper']).astype(float)
        df['bb_breakout_down'] = (df['close'] < df['bb_lower']).astype(float)
        
        # 3. MEAN REVERSION SIGNALS
        # Price deviation from moving averages
        if 'dma_50' in df.columns:
            df['price_deviation_50d'] = (df['close'] - df['dma_50']) / df['dma_50']
            df['mean_reversion_signal_50d'] = np.where(df['price_deviation_50d'] > 0.05, -1,  # Overbought
                                                      np.where(df['price_deviation_50d'] < -0.05, 1, 0))  # Oversold
        
        if 'dma_200' in df.columns:
            df['price_deviation_200d'] = (df['close'] - df['dma_200']) / df['dma_200']
            df['mean_reversion_signal_200d'] = np.where(df['price_deviation_200d'] > 0.1, -1,  # Overbought
                                                       np.where(df['price_deviation_200d'] < -0.1, 1, 0))  # Oversold
        
        # 4. MOMENTUM BREAKOUT SIGNALS
        # Price breakout from recent ranges
        df['price_breakout_20d'] = (df['close'] > df['close'].rolling(20).max().shift(1)).astype(float)
        df['price_breakdown_20d'] = (df['close'] < df['close'].rolling(20).min().shift(1)).astype(float)
        
        # Volume breakout signals
        if 'volume' in df.columns:
            df['volume_breakout_20d'] = (df['volume'] > df['volume'].rolling(20).max().shift(1)).astype(float)
            df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(float)
        
        # 5. TREND FOLLOWING SIGNALS
        # Moving average crossover signals
        if all(col in df.columns for col in ['dma_50', 'dma_200']):
            df['ma_cross_bullish'] = ((df['dma_50'] > df['dma_200']) & (df['dma_50'].shift(1) <= df['dma_200'].shift(1))).astype(float)
            df['ma_cross_bearish'] = ((df['dma_50'] < df['dma_200']) & (df['dma_50'].shift(1) >= df['dma_200'].shift(1))).astype(float)
        
        # Price above/below moving averages
        if 'dma_50' in df.columns:
            df['price_above_ma50'] = (df['close'] > df['dma_50']).astype(float)
        if 'dma_200' in df.columns:
            df['price_above_ma200'] = (df['close'] > df['dma_200']).astype(float)
        
        # 6. VOLATILITY REGIME SIGNALS
        # High/low volatility regimes
        vol_threshold_high = df['volatility_20d'].rolling(60).quantile(0.8)
        vol_threshold_low = df['volatility_20d'].rolling(60).quantile(0.2)
        df['high_vol_regime'] = (df['volatility_20d'] > vol_threshold_high).astype(float)
        df['low_vol_regime'] = (df['volatility_20d'] < vol_threshold_low).astype(float)
        
        # Volatility expansion/contraction
        df['vol_expansion'] = (df['volatility_20d'] > df['volatility_20d'].shift(5)).astype(float)
        df['vol_contraction'] = (df['volatility_20d'] < df['volatility_20d'].shift(5)).astype(float)
        
        # 7. TECHNICAL INDICATOR SIGNALS
        # RSI signals (if available)
        if 'rsi_14' in df.columns:
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(float)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(float)
            df['rsi_bullish_divergence'] = ((df['rsi_14'] > df['rsi_14'].shift(1)) & 
                                          (df['close'] < df['close'].shift(1))).astype(float)
            df['rsi_bearish_divergence'] = ((df['rsi_14'] < df['rsi_14'].shift(1)) & 
                                           (df['close'] > df['close'].shift(1))).astype(float)
        
        # 8. COMBINED SIGNAL STRENGTH
        # Aggregate signal strength
        signal_cols = [col for col in df.columns if any(signal in col for signal in 
                      ['signal', 'breakout', 'cross', 'above', 'below', 'oversold', 'overbought'])]
        
        if signal_cols:
            # Count positive signals
            df['bullish_signals'] = df[signal_cols].sum(axis=1)
            # Count negative signals  
            df['bearish_signals'] = df[[col for col in signal_cols if 'bearish' in col or 'breakdown' in col or 'below' in col]].sum(axis=1)
            # Net signal strength
            df['net_signal_strength'] = df['bullish_signals'] - df['bearish_signals']
        
        # 9. PRICE ACTION PATTERNS
        # Gap analysis
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df['overnight_gap'] = df['open'] / df['close'].shift(1) - 1.0
            df['gap_fade'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
            
            # Intraday range and position
            df['daily_range'] = (df['high'] - df['low']) / df['open']
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
            
            # Body vs wick ratios (candlestick patterns)
            body_size = abs(df['close'] - df['open'])
            upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
            lower_wick = df[['close', 'open']].min(axis=1) - df['low']
            total_range = df['high'] - df['low'] + 1e-8
            
            df['body_ratio'] = body_size / total_range
            df['upper_wick_ratio'] = upper_wick / total_range  
            df['lower_wick_ratio'] = lower_wick / total_range
            
            # Doji pattern (small body)
            df['doji_pattern'] = (df['body_ratio'] < 0.1).astype(float)
            
            # Hammer pattern (long lower wick, small body)
            df['hammer_pattern'] = ((df['lower_wick_ratio'] > 0.6) & (df['body_ratio'] < 0.3)).astype(float)
            
            # Shooting star pattern (long upper wick, small body)
            df['shooting_star_pattern'] = ((df['upper_wick_ratio'] > 0.6) & (df['body_ratio'] < 0.3)).astype(float)
        
        # 10. VOLUME ANALYSIS SIGNALS
        if 'volume' in df.columns:
            # Volume-price relationships
            df['volume_price_momentum'] = (df['returns_1d'] * df['volume'] / df['volume'].rolling(20).mean()).fillna(0)
            
            # Volume relative to recent average
            df['volume_ratio_5d'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-8)
            df['volume_ratio_20d'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
            
            # Volume trend
            df['volume_trend_10d'] = df['volume'].rolling(10).mean() / df['volume'].rolling(20).mean() - 1.0
            
            # Volume confirmation signals
            df['volume_confirms_price'] = ((df['returns_1d'] > 0) & (df['volume_ratio_20d'] > 1.2)).astype(float)
            df['volume_divergence'] = ((df['returns_1d'] > 0) & (df['volume_ratio_20d'] < 0.8)).astype(float)
        
        # 11. SENTIMENT FEATURES (if available) ===
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
        for col in sentiment_cols:
            if col in df.columns and df[col].std() > 1e-8:
                # Sentiment momentum
                df[f'{col}_momentum_3d'] = df[col] - df[col].shift(3)
                df[f'{col}_momentum_5d'] = df[col] - df[col].shift(5)
                
                # Sentiment extremes (potential reversal signals)
                rolling_quantile_90 = df[col].rolling(30).quantile(0.9)
                rolling_quantile_10 = df[col].rolling(30).quantile(0.1)
                df[f'{col}_extreme_positive'] = (df[col] > rolling_quantile_90).astype(float)
                df[f'{col}_extreme_negative'] = (df[col] < rolling_quantile_10).astype(float)
        
        # 12. REGIME DETECTION ===
        # Market regime indicators based on volatility and trend
        if 'volatility_20d' in df.columns:
            vol_ma_short = df['volatility_20d'].rolling(20).mean()
            vol_ma_long = df['volatility_20d'].rolling(60).mean()
            df['vol_regime_change'] = (vol_ma_short > vol_ma_long).astype(float)
        
        # Trending vs mean-reverting regime
        if 'returns_1d' in df.columns:
            # Hurst exponent approximation
            returns_cumsum = df['returns_1d'].rolling(20).apply(lambda x: np.cumsum(x.values)[-1] if len(x) == 20 else 0)
            returns_range = df['returns_1d'].rolling(20).apply(lambda x: x.max() - x.min() if len(x) == 20 else 0)
            df['trend_regime'] = returns_range / (df['volatility_20d'] * np.sqrt(20) + 1e-8)
        
        # 13. CROSS-SECTIONAL FEATURES ===
        # These would ideally use market-wide data, but we approximate with single-stock features
        df['momentum_rank_proxy'] = df['momentum_20d'].rolling(60).rank(pct=True)
        df['vol_rank_proxy'] = df['volatility_20d'].rolling(60).rank(pct=True)
        
        # 14. INTERACTION FEATURES ===
        # Combine different signal types
        if all(col in df.columns for col in ['momentum_10d', 'volatility_10d']):
            df['risk_adjusted_momentum'] = df['momentum_10d'] / (df['volatility_10d'] + 1e-8)
        
        if all(col in df.columns for col in ['price_to_dma50', 'volume_ratio_20d']):
            df['volume_confirmed_trend'] = df['price_to_dma50'] * df['volume_ratio_20d']

        return df.fillna(0.0)

    def _generate_cache_key(self, start_date: str, end_date: str, 
                           fill_missing_features_with: str) -> str:
        """Generate a unique cache key based on parameters"""
        stocks_hash = hash(tuple(sorted(self.stocks)))
        features_hash = hash(tuple(sorted(self.features or [])))
        
        key_components = [
            f"stocks_{stocks_hash}",
            f"features_{features_hash}",
            f"start_{start_date}",
            f"end_{end_date}",
            f"fill_{fill_missing_features_with}",
            f"use_all_{self.use_all_features}"
        ]
        return "_".join(key_components).replace("/", "-").replace(":", "")

    def save_data(self, data_array: Union[np.ndarray, chex.Array], 
                  valid_dates: pd.DatetimeIndex, 
                  start_date: str, end_date: str,
                  fill_missing_features_with: str,
                  file_format: str = 'hdf5') -> str:
        """
        Save processed data to disk in specified format.
        
        Args:
            data_array: The processed data tensor
            valid_dates: DatetimeIndex of valid dates
            start_date: Start date string
            end_date: End date string
            fill_missing_features_with: Missing data fill method
            file_format: Format to save ('hdf5', 'npz', or 'pickle')
            
        Returns:
            Path to saved file
        """
        cache_key = self._generate_cache_key(start_date, end_date, fill_missing_features_with)
        
        # Convert JAX array to numpy for saving
        if hasattr(data_array, 'device'):  # JAX array
            data_np = np.array(data_array)
        else:
            data_np = data_array
            
        metadata = {
            'stocks': self.stocks,
            'features': self.features,
            'start_date': start_date,
            'end_date': end_date,
            'fill_method': fill_missing_features_with,
            'use_all_features': self.use_all_features,
            'shape': data_np.shape,
            'n_days': len(valid_dates),
            'n_stocks': len(self.stocks),
            'n_features': len(self.features),
            'dates': valid_dates.strftime('%Y-%m-%d').tolist()
        }
        
        if file_format == 'hdf5':
            filepath = self.cache_dir / f"{cache_key}.h5"
            with h5py.File(filepath, 'w') as f:
                # Save data array
                f.create_dataset('data', data=data_np, compression='gzip', compression_opts=9)
                f.create_dataset('day_indices', data=np.arange(len(valid_dates)))
                
                # Save metadata as JSON string attribute
                f.attrs['metadata'] = json.dumps(metadata)
                
                # Save dates as string dataset
                f.create_dataset('dates', data=valid_dates.strftime('%Y-%m-%d').astype('S10'))
                
        elif file_format == 'npz':
            filepath = self.cache_dir / f"{cache_key}.npz"
            np.savez_compressed(
                filepath,
                data=data_np,
                day_indices=np.arange(len(valid_dates)),
                dates=valid_dates.strftime('%Y-%m-%d'),
                metadata=json.dumps(metadata)
            )
            
        elif file_format == 'pickle':
            filepath = self.cache_dir / f"{cache_key}.pkl"
            save_dict = {
                'data': data_np,
                'day_indices': np.arange(len(valid_dates)),
                'valid_dates': valid_dates,
                'metadata': metadata
            }
            with open(filepath, 'wb') as f:
                pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
        print(f"Data saved to: {filepath}")
        print(f"Data shape: {data_np.shape}")
        print(f"File size: {filepath.stat().st_size / (1024**2):.2f} MB")
        
        return str(filepath)

    def load_cached_data(self, start_date: str, end_date: str,
                        fill_missing_features_with: str = 'interpolate',
                        file_format: str = 'hdf5',
                        preload_to_gpu: bool = True) -> Optional[Tuple[chex.Array, chex.Array, pd.DatetimeIndex, int]]:
        """
        Try to load cached data if it exists.
        
        Returns:
            Cached data tuple if found, None otherwise
        """
        cache_key = self._generate_cache_key(start_date, end_date, fill_missing_features_with)
        
        if file_format == 'hdf5':
            filepath = self.cache_dir / f"{cache_key}.h5"
            if not filepath.exists():
                return None
                
            with h5py.File(filepath, 'r') as f:
                data_array = f['data'][:]
                day_indices = f['day_indices'][:]
                dates_str = f['dates'][:].astype(str)
                metadata = json.loads(f.attrs['metadata'])
                
        elif file_format == 'npz':
            filepath = self.cache_dir / f"{cache_key}.npz"
            if not filepath.exists():
                return None
                
            with np.load(filepath, allow_pickle=True) as f:
                data_array = f['data']
                day_indices = f['day_indices']
                dates_str = f['dates']
                metadata = json.loads(str(f['metadata']))
                
        elif file_format == 'pickle':
            filepath = self.cache_dir / f"{cache_key}.pkl"
            if not filepath.exists():
                return None
                
            with open(filepath, 'rb') as f:
                cached = pickle.load(f)
                data_array = cached['data']
                day_indices = cached['day_indices']
                valid_dates = cached['valid_dates']
                metadata = cached['metadata']
                
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
        if file_format != 'pickle':
            valid_dates = pd.to_datetime(dates_str)
            
        # Restore features list
        self.features = metadata['features']
        
        if preload_to_gpu:
            data_array = jnp.array(data_array)
            day_indices = jnp.array(day_indices)
        else:
            day_indices = jnp.array(day_indices)
            
        print(f"Loaded cached data from: {filepath}")
        return data_array, day_indices, valid_dates, metadata['n_features']

    def load_and_preprocess_data(self,
                                 start_date: str,
                                 end_date: str,
                                 fill_missing_features_with: str = 'interpolate',
                                 preload_to_gpu: bool = True,
                                 save_cache: bool = False,
                                 cache_format: str = 'hdf5',
                                 force_reload: bool = False) -> Tuple[chex.Array, chex.Array, pd.DatetimeIndex, int]:
        """
        Load and preprocess CSV data into a consistent tensor for JAX.
        
        Args:
            start_date: Start date string
            end_date: End date string
            fill_missing_features_with: How to fill missing features
            preload_to_gpu: Whether to convert to JAX arrays
            save_cache: Whether to save processed data to cache
            cache_format: Format for caching ('hdf5', 'npz', 'pickle')
            force_reload: Whether to ignore existing cache

        Returns:
            data_array: jnp.array or np.array with shape (T, n_stocks, n_features)
            day_idx: jnp.array of indices (0..T-1)
            valid_dates: pd.DatetimeIndex
            n_features: number of features
        """
        
        # Try to load from cache first (unless force_reload is True)
        if not force_reload:
            cached_result = self.load_cached_data(
                start_date, end_date, fill_missing_features_with, 
                cache_format, preload_to_gpu
            )
            if cached_result is not None:
                return cached_result

        all_data = []
        all_features = set()
        stock_features = {}

        # Pass 1: Collect available features
        for stock in self.stocks:
            csv_path = self.data_root / f"{stock}_aligned.csv"
            if not csv_path.exists():
                print(f"Warning: CSV for {stock} not found, skipping.")
                continue
            try:
                sample_df = pd.read_csv(csv_path, nrows=0, index_col=0)
                feats = list(sample_df.columns)
                stock_features[stock] = feats
                all_features.update(feats)
            except Exception as e:
                print(f"Warning: Error reading {stock}: {e}, skipping.")
                continue

        if self.features is None:
            if self.use_all_features:
                self.features = sorted(list(all_features))
            else:
                common = set.intersection(*[set(f) for f in stock_features.values()])
                self.features = sorted(list(common)) if common else sorted(list(all_features))

        # Ensure 'close' is available
        if 'close' not in self.features:
            raise ValueError("'close' must be present in features")

        # Reorganize so 'close' is always first
        self.features = ['close'] + [f for f in self.features if f != 'close']

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

            proc = self.engineer_features(proc, stock)
            proc["symbol"] = stock
            all_data.append(proc)

        if not all_data:
            raise RuntimeError("No stock data loaded.")

        # Merge into panel
        panel = pd.concat(all_data).reset_index().set_index(["date", "symbol"])
        panel = panel[~panel.index.duplicated(keep="first")]

        # Update feature list to include engineered features
        engineered_feats = [c for c in all_data[0].columns if c not in self.features + ["symbol"]]
        self.features = self.features + engineered_feats

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

        return data_array, jnp.arange(n_days), valid_dates, n_features

    def clear_cache(self, pattern: str = "*") -> int:
        """
        Clear cached files matching pattern.
        
        Args:
            pattern: Glob pattern to match files (default: all cache files)
            
        Returns:
            Number of files deleted
        """
        import glob
        cache_files = list(self.cache_dir.glob(pattern))
        count = 0
        for file_path in cache_files:
            if file_path.is_file():
                file_path.unlink()
                count += 1
        print(f"Cleared {count} cache files")
        return count



class JAXVectorizedPortfolioEnv:
    """Vectorized Portfolio Environment optimized for JAX training"""

    def __init__(self,
                 data_root: str = "processed_data/",
                 stocks: List[str] = None,
                 features: List[str] = None,
                 initial_cash: float = 1000000.0,
                 window_size: int = 30,
                 start_date: str = '2024-06-06',
                 end_date: str = '2025-03-06',
                 transaction_cost_rate: float = 0.005,
                 sharpe_window: int = 252,
                 use_all_features: bool = True):

        self.data_root = data_root
        self.window_size = window_size
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash_actual = initial_cash
        self.transaction_cost_rate = transaction_cost_rate
        self.sharpe_window = sharpe_window
        self.risk_free_rate_daily = 0.04 / 252.0  # 4% annual risk-free rate
        self.cash_return_rate = 0.04 / 252.0  # 4% annual return for cash holdings
        self.use_all_features = use_all_features
        self.features = None
        self.close_price_idx = None  # Track close price index

        if stocks is None:
            # Example stock list - you should replace with your actual stocks
            # For demonstration, ensure these CSVs exist in 'processed_data'
            stocks = self._load_stock_list() 
        self.stocks = stocks
        self.n_stocks = len(stocks)
        
        print(f"DEBUG ENV: Data root: {self.data_root}")
        print(f"DEBUG ENV: Stocks loaded: {len(self.stocks)} - {self.stocks[:5]}...")
        print(f"DEBUG ENV: First stock CSV path: {self.data_root}/{self.stocks[0]}_aligned.csv" if self.stocks else "No stocks loaded")

        self.data_loader = JAXPortfolioDataLoader(data_root, stocks, self.features, self.use_all_features)
        self.data, self.dates_idx, self.actual_dates, self.n_features = self.data_loader.load_and_preprocess_data(
            start_date, end_date, preload_to_gpu=True
        )
        #jax.debug.print("Number of nans in {}: {}", self.data, jnp.sum(jnp.isnan(self.data)))
        # Set close price index (should be 0 after reorganization)
        self.close_price_idx = 0

        self.feature_indices = {
            'open': self.data_loader.features.index('open'),
            'close': self.data_loader.features.index('close'),
            'returns_1d': self.data_loader.features.index('returns_1d') if 'returns_1d' in self.data_loader.features else None,
            'volatility_10d': self.data_loader.features.index('volatility_10d') if 'volatility_10d' in self.data_loader.features else None,
            'overnight_gap': self.data_loader.features.index('overnight_gap') if 'overnight_gap' in self.data_loader.features else None
        }
        
        # Validate that we have valid price data
        if self.data.shape[0] < self.window_size + 2:
            raise ValueError(f"Insufficient data: need at least {self.window_size + 2} time steps, got {self.data.shape[0]}")
        self.n_timesteps = len(self.dates_idx)

        self.action_dim = self.n_stocks + 1
        chex.assert_trees_all_equal(self.data.shape[1], self.n_stocks)

        # Updated observation size calculation with short position support
        # Historical features + current prices + gaps + portfolio weights + short positions + market state
        obs_size = (
            (self.window_size * self.n_stocks * self.n_features) +  # Historical features
            self.n_stocks * 2 +                                     # Current open prices + gaps
            self.action_dim +                                       # Portfolio weights
            self.n_stocks +                                         # Short position flags
            8                                                       # Market state (8 elements now)
        )

        self.obs_dim = obs_size

        print(f"Environment initialized:")
        print(f"  Stocks: {self.n_stocks}")
        print(f"  Features: {self.n_features}")
        print(f"  Window size: {self.window_size}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim (stocks+cash): {self.action_dim}")
        print(f"  Timesteps available: {self.n_timesteps}")

    def _load_stock_list(self) -> List[str]:
        """Loads stock list from a file if not provided."""
        stocks_file = Path("finagent/stocks.txt")
        # Try current directory first, then relative to script location
        if not stocks_file.exists():
            stocks_file = Path("FYP-FinAgent/finagent/stocks.txt")
        
        print(f"DEBUG: Looking for stocks file at: {stocks_file}")
        print(f"DEBUG: File exists: {stocks_file.exists()}")
        
        if stocks_file.exists():
            with open(stocks_file, 'r') as f:
                stocks = [line.strip() for line in f.readlines() if line.strip()]
                print(f"DEBUG: Loaded {len(stocks)} stocks from file")
                return stocks
        # Fallback to scanning directory if file doesn't exist
        data_path = Path(self.data_root) if isinstance(self.data_root, str) else self.data_root
        return [p.stem.replace('_aligned', '') for p in data_path.glob("*_aligned.csv")]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, chex.Array]:
        """Reset environment state with short position support"""
        initial_weights = jnp.zeros(self.n_stocks)
        initial_cash_weight = 1.0

        initial_portfolio_weights = jnp.append(initial_weights, initial_cash_weight)
        
        # Initialize short position tracking
        initial_short_positions = jnp.zeros(self.n_stocks)  # 0 = no short position
        initial_short_entry_steps = jnp.full(self.n_stocks, -1)  # -1 = no entry step
    
        sharpe_buffer = jnp.zeros(self.sharpe_window)

        min_start_step = self.window_size - 1
        max_start_step = self.n_timesteps - 2

        start_step = random.randint(key, (), min_start_step, max_start_step + 1)

        env_state = EnvState(
            current_step=start_step,
            portfolio_weights=initial_portfolio_weights,
            short_positions=initial_short_positions,
            short_entry_steps=initial_short_entry_steps,
            done=False,
            total_return=0.0,
            portfolio_value=1.0,
            sharpe_buffer=sharpe_buffer,
            sharpe_buffer_idx=0,
            portfolio_volatility=0.0,
            max_drawdown=0.0,
            rolling_correlation=jnp.zeros(30)  # 30-day correlation buffer
        )

        obs = self._get_observation(env_state)
        return env_state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, env_state: EnvState, action: chex.Array) -> Tuple[EnvState, chex.Array, float, bool, dict]:
        """Execute one environment step with short position support and intraday constraints"""
        
        # Convert action from [-1, 1] to position weights
        # Action represents desired position size: -1 = max short, +1 = max long, 0 = no position
        raw_stock_actions = jnp.tanh(action[:-1])  # Stock positions in [-1, 1]
        raw_cash_action = jnp.tanh(action[-1])     # Cash position in [-1, 1]
        
        # Apply position size constraints (max 50% per stock for risk management)
        max_position_size = 0.5
        constrained_stock_actions = jnp.clip(raw_stock_actions, -max_position_size, max_position_size)
        
        # Calculate cash weight: 1 - sum(abs(stock_weights))
        total_stock_exposure = jnp.sum(jnp.abs(constrained_stock_actions))
        new_cash_weight = jnp.maximum(0.0, 1.0 - total_stock_exposure)
        
        # Normalize to ensure portfolio weights sum to 1
        total_weight = total_stock_exposure + new_cash_weight
        normalized_stock_weights = constrained_stock_actions / jnp.maximum(total_weight, 1e-8)
        normalized_cash_weight = new_cash_weight / jnp.maximum(total_weight, 1e-8)
        
        # Combine into portfolio weights
        new_portfolio_weights = jnp.append(normalized_stock_weights, normalized_cash_weight)
        
        # Get previous state
        prev_stock_weights = env_state.portfolio_weights[:-1]
        prev_cash_weight = env_state.portfolio_weights[-1]
        prev_portfolio_value = env_state.portfolio_value
        current_step = env_state.current_step
        
        # Handle intraday short position constraints
        # Check if we need to close short positions (end of trading day simulation)
        # For simplicity, we'll close shorts every 6 steps (simulating end of day)
        steps_per_day = 6  # Adjust based on your data frequency
        is_end_of_day = (current_step % steps_per_day) == (steps_per_day - 1)
        
        # Force close all short positions at end of day
        new_short_positions = jnp.where(is_end_of_day, 0.0, env_state.short_positions)
        new_short_entry_steps = jnp.where(is_end_of_day, -1, env_state.short_entry_steps)
        
        # Update short position tracking for new positions
        new_short_positions = jnp.where(
            (normalized_stock_weights < 0) & (env_state.short_positions == 0),
            1.0,  # Mark as short position
            new_short_positions
        )
        
        # Update entry steps for new short positions
        new_short_entry_steps = jnp.where(
            (normalized_stock_weights < 0) & (env_state.short_positions == 0),
            current_step,  # Record entry step
            new_short_entry_steps
        )
        
        # Apply penalty for holding short positions overnight (simulated)
        overnight_short_penalty = jnp.where(
            is_end_of_day & (jnp.sum(env_state.short_positions) > 0),
            -0.01,  # 1% penalty for overnight shorts
            0.0
        )
        
        # Calculate transaction costs
        weight_change_total = jnp.sum(jnp.abs(normalized_stock_weights - prev_stock_weights))
        transaction_cost_rate_applied = weight_change_total * self.transaction_cost_rate
        
        # Get market returns
        current_daily_returns = self._get_daily_returns_from_data(current_step + 1)
        
        # Calculate portfolio returns with short position handling
        # For short positions, returns are inverted
        stock_returns = jnp.where(
            normalized_stock_weights < 0,  # Short positions
            -current_daily_returns,        # Inverted returns
            current_daily_returns          # Normal returns for long positions
        )
        
        # Portfolio return calculation
        daily_portfolio_return_before_costs = (
            jnp.sum(normalized_stock_weights * stock_returns) +
            (normalized_cash_weight * self.cash_return_rate)
        )
        daily_portfolio_return_before_costs = jnp.where(
            jnp.isnan(daily_portfolio_return_before_costs), 0.0, daily_portfolio_return_before_costs
        )
        
        # Apply transaction costs and overnight penalty
        net_daily_portfolio_return = (
            daily_portfolio_return_before_costs - 
            transaction_cost_rate_applied + 
            overnight_short_penalty
        )
        net_daily_portfolio_return = jnp.where(
            jnp.isnan(net_daily_portfolio_return), 0.0, net_daily_portfolio_return
        )
        net_daily_portfolio_return = jnp.clip(net_daily_portfolio_return, -0.5, 0.5)
        
        # Update portfolio value
        new_portfolio_value = prev_portfolio_value * (1.0 + net_daily_portfolio_return)
        new_portfolio_value = jnp.where(
            jnp.isnan(new_portfolio_value), prev_portfolio_value, new_portfolio_value
        )
        new_portfolio_value = jnp.maximum(new_portfolio_value, 1e-6)
        
        new_total_return = (new_portfolio_value - 1.0)
        
        # Update Sharpe buffer
        new_sharpe_buffer = env_state.sharpe_buffer.at[env_state.sharpe_buffer_idx].set(net_daily_portfolio_return)
        new_sharpe_buffer_idx = (env_state.sharpe_buffer_idx + 1) % self.sharpe_window
        
        # Calculate drawdown
        mask = jnp.arange(self.sharpe_window) < (env_state.sharpe_buffer_idx + 1)
        rolling_returns = jnp.where(mask, env_state.sharpe_buffer, 0.0)
        rolling_returns = jnp.where(jnp.isnan(rolling_returns), 0.0, rolling_returns)
        cumulative_returns = jnp.cumsum(rolling_returns)
        running_max = lax.cummax(cumulative_returns, axis=0)
        drawdown = (cumulative_returns - running_max).min()
        drawdown = jnp.where(jnp.isnan(drawdown), 0.0, drawdown)
        new_max_drawdown = jnp.minimum(env_state.max_drawdown, drawdown)
        
        # Calculate volatility and Sharpe ratio
        portfolio_volatility = jnp.std(rolling_returns) * jnp.sqrt(252.0)
        portfolio_volatility = jnp.where(jnp.isnan(portfolio_volatility), 0.0, portfolio_volatility)
        
        sharpe_mean = jnp.mean(new_sharpe_buffer)
        sharpe_std = jnp.std(new_sharpe_buffer)
        sharpe_mean = jnp.where(jnp.isnan(sharpe_mean), 0.0, sharpe_mean)
        sharpe_std = jnp.where(jnp.isnan(sharpe_std), 1e-6, sharpe_std)
        sharpe_std = jnp.maximum(sharpe_std, 1e-6)
        
        sharpe_ratio = jnp.where(
            sharpe_std < 1e-6,
            0.0,
            (sharpe_mean - self.cash_return_rate) / (sharpe_std + 1e-8)
        )
        sharpe_ratio = jnp.where(jnp.isnan(sharpe_ratio), 0.0, sharpe_ratio)
        sharpe_ratio = jnp.clip(sharpe_ratio, -5.0, 5.0)
        
        # Simplified reward: Focus on portfolio returns
        # The environment already handles transaction costs (in portfolio_value update)
        # and short position constraints (intraday close, overnight penalties)

        log_portfolio_return = jnp.log(new_portfolio_value / prev_portfolio_value)

        # Better NaN handling: use small negative instead of large penalty
        log_portfolio_return = jnp.where(jnp.isnan(log_portfolio_return), -0.01, log_portfolio_return)

        # Primary reward: actual wealth growth (already accounts for costs and risks)
        reward = log_portfolio_return

        # Clip to reasonable daily return bounds (±50% per day)
        reward = jnp.clip(reward, -0.5, 0.5)

        
        # Check if episode is done
        next_step = current_step + 1
        done = (next_step >= self.n_timesteps - 1) | (new_portfolio_value <= 0.5)
        
        # Create new environment state
        new_env_state = EnvState(
            current_step=next_step,
            portfolio_weights=new_portfolio_weights,
            short_positions=new_short_positions,
            short_entry_steps=new_short_entry_steps,
            done=done,
            total_return=new_total_return,
            portfolio_value=new_portfolio_value,
            sharpe_buffer=new_sharpe_buffer,
            sharpe_buffer_idx=new_sharpe_buffer_idx,
            portfolio_volatility=portfolio_volatility,
            max_drawdown=new_max_drawdown,
            rolling_correlation=env_state.rolling_correlation
        )
        
        next_obs = self._get_observation(new_env_state)
        
        info = {
            'date_idx': next_step,
            'portfolio_value': new_portfolio_value,
            'total_return': new_total_return,
            'sharpe_ratio': sharpe_ratio,
            'daily_portfolio_return': net_daily_portfolio_return,
            'transaction_cost_value': transaction_cost_rate_applied,
            'overnight_short_penalty': overnight_short_penalty,
            'short_exposure': jnp.sum(jnp.abs(normalized_stock_weights * (normalized_stock_weights < 0))),
            'is_end_of_day': is_end_of_day,
            'new_stock_weights': normalized_stock_weights,
            'new_cash_weight': normalized_cash_weight,
            'prev_stock_weights': prev_stock_weights,
            'prev_cash_weight': prev_cash_weight
        }

        return new_env_state, next_obs, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, env_state: EnvState) -> chex.Array:
        """
        Constructs the observation for the current step with short position support.
        - Historical features: t-window_size to t-1 (ALL features)
        - Current partial info: t (ONLY open price and overnight gap)
        - Short position tracking information
        """
        # Historical data (t-window_size to t-1)
        hist_start_idx = env_state.current_step - self.window_size
        hist_end_idx = env_state.current_step  # Exclusive, so gets up to t-1
    
        historical_data = lax.dynamic_slice(
            self.data,
            (hist_start_idx, 0, 0),
            (self.window_size, self.n_stocks, self.n_features)
        )
    
        # Current partial information (time t)
        current_step_data = self.data[env_state.current_step, :, :]  # (n_stocks, n_features)
        current_open = current_step_data[:, self.feature_indices['open']]
        current_gap = current_step_data[:, self.feature_indices['overnight_gap']]
    
        # Flatten historical data
        historical_flat = historical_data.flatten()
    
        # Portfolio risk metrics
        mask = jnp.arange(self.sharpe_window) < (env_state.sharpe_buffer_idx + 1)
        returns_buffer = jnp.where(mask, env_state.sharpe_buffer, 0.0)
        portfolio_vol = jnp.std(returns_buffer) * jnp.sqrt(252.0)  # Annualized
        
        # Short position metrics
        short_exposure = jnp.sum(jnp.abs(env_state.portfolio_weights[:-1] * (env_state.portfolio_weights[:-1] < 0)))
        num_short_positions = jnp.sum(env_state.short_positions)
        
        # Time-based features for intraday constraints
        steps_per_day = 6  # Should match the value in step function
        time_in_day = env_state.current_step % steps_per_day
        is_near_close = (time_in_day >= steps_per_day - 2).astype(jnp.float32)  # Last 2 steps of day
    
        # Market state indicators with short position information
        market_state = jnp.array([
            env_state.portfolio_value,
            env_state.total_return,
            portfolio_vol,
            env_state.max_drawdown,
            short_exposure,
            num_short_positions,
            time_in_day / steps_per_day,  # Normalized time in day
            is_near_close
        ])
    
        obs = jnp.concatenate([
            historical_flat,           # Historical features (all)
            current_open,             # Current open prices
            current_gap,              # Overnight gaps
            env_state.portfolio_weights,  # Current allocation
            env_state.short_positions,    # Short position flags
            market_state              # Portfolio metrics + short position info
        ])
    
        return obs


    @partial(jax.jit, static_argnums=(0,))
    def _get_daily_returns_from_data(self, step: int) -> chex.Array:
        """
        Calculates daily returns for all stocks for a given step using CLOSE prices.
        Returns array of shape (n_stocks,).
        """
        # Use close price index (guaranteed to be 0 after reorganization)
        price_t = self.data[step, :, self.close_price_idx]
        price_t_minus_1 = self.data[step - 1, :, self.close_price_idx]
        
        # Clean prices for NaN values
        price_t = jnp.where(jnp.isnan(price_t), 1.0, price_t)
        price_t_minus_1 = jnp.where(jnp.isnan(price_t_minus_1), 1.0, price_t_minus_1)
        
        # Safety check for zero/negative prices with better handling
        price_t_minus_1_safe = jnp.where(price_t_minus_1 <= 1e-8, 1.0, price_t_minus_1)
        price_t_safe = jnp.where(price_t <= 1e-8, 1.0, price_t)

        # Calculate returns with numerical stability
        daily_returns = (price_t_safe / price_t_minus_1_safe) - 1.0
        
        # Clean and cap extreme returns
        daily_returns = jnp.where(jnp.isnan(daily_returns), 0.0, daily_returns)
        daily_returns = jnp.where(jnp.isinf(daily_returns), 0.0, daily_returns)
        daily_returns = jnp.clip(daily_returns, -0.3, 0.3)  # Cap at ±30% daily moves (more conservative)
        
        return daily_returns
