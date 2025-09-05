import jax
import jax.numpy as jnp
from jax import random, vmap, lax
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from pathlib import Path
import chex
from functools import partial
import h5py
from concurrent.futures import ThreadPoolExecutor

# JAX environment state
class EnvState(NamedTuple):
    """JAX-compatible environment state"""
    current_step: int
    portfolio_weights: chex.Array  # Current portfolio weights
    cash_weight: float
    done: bool
    total_return: float
    portfolio_value: float
    sharpe_buffer: chex.Array  # Rolling buffer for Sharpe calculation
    sharpe_buffer_idx: int

class JAXPortfolioDataLoader:
    """Optimized data loader for portfolio environment"""
    
    def __init__(self, data_root: str, stocks: List[str], features: List[str]):
        self.data_root = Path(data_root) 
        self.stocks = stocks
        self.features = features
        self.n_stocks = len(stocks)
        self.n_features = len(features)
        
    def load_and_preprocess_data(self, 
                                start_date: str, 
                                end_date: str,
                                preload_to_gpu: bool = True) -> Tuple[chex.Array, chex.Array]:
        """
        Load and preprocess data optimized for JAX training
        Returns: (data_array, dates_array)
        """
        print(f"Loading data for {self.n_stocks} stocks, {self.n_features} features...")
        
        # Check if HDF5 file exists, if not convert from CSV
        h5_path = self.data_root / "stocks_data.h5"
        if not h5_path.exists():
            print("HDF5 file not found. Converting from CSV...")
            self._convert_csv_to_hdf5()
        
        # Load data from HDF5
        data_arrays = []
        valid_dates = None
        
        with h5py.File(h5_path, 'r') as h5f:
            for stock in self.stocks:
                if stock not in h5f:
                    print(f"Warning: {stock} not found in HDF5 file")
                    continue
                
                stock_group = h5f[stock]
                
                # Get date range
                dates = pd.to_datetime(stock_group['dates'][:])
                mask = (dates >= start_date) & (dates <= end_date)
                
                if valid_dates is None:
                    valid_dates = dates[mask]
                
                # Load features for this stock
                stock_data = []
                for feature in self.features:
                    if feature in stock_group:
                        feature_data = stock_group[feature][:][mask]
                    else:
                        print(f"Feature {feature} not found for {stock}, filling with zeros")
                        feature_data = np.zeros(mask.sum())
                    
                    stock_data.append(feature_data)
                
                stock_array = np.stack(stock_data, axis=-1)  # Shape: (time, features)
                data_arrays.append(stock_array)
        
        if not data_arrays:
            raise ValueError("No valid stock data loaded")
        
        # Combine all stocks: Shape (time, stocks, features)
        data_array = np.stack(data_arrays, axis=1)
        
        # Handle NaNs and preprocessing
        data_array = self._preprocess_array(data_array)
        
        # Convert to JAX arrays
        if preload_to_gpu:
            data_array = jnp.array(data_array, dtype=jnp.float32)
        
        dates_array = np.arange(len(valid_dates))  # Use indices instead of actual dates
        
        print(f"Loaded data shape: {data_array.shape}")
        print(f"Date range: {valid_dates[0]} to {valid_dates[-1]} ({len(valid_dates)} days)")
        
        return data_array, dates_array, valid_dates
    
    def _convert_csv_to_hdf5(self):
        """Convert CSV files to HDF5 format for faster loading"""
        h5_path = self.data_root / "stocks_data.h5"
        
        print(f"Converting CSV files to {h5_path}...")
        
        with h5py.File(h5_path, 'w') as h5f:
            for stock in self.stocks:
                csv_path = self.data_root / f"{stock}_aligned.csv"
                if not csv_path.exists():
                    print(f"Warning: CSV file for {stock} not found")
                    continue
                
                try:
                    # Load CSV in chunks to handle large files
                    df_chunks = pd.read_csv(csv_path, chunksize=10000)
                    df = pd.concat(df_chunks, ignore_index=True)
                    
                    # Parse dates
                    date_col = df.columns[0]
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.sort_values(date_col)
                    
                    # Create group for this stock
                    stock_group = h5f.create_group(stock)
                    
                    # Store dates
                    dates_str = df[date_col].dt.strftime('%Y-%m-%d').values
                    stock_group.create_dataset('dates', data=dates_str.astype('S10'))
                    
                    # Store each feature
                    for feature in self.features:
                        if feature in df.columns:
                            data = pd.to_numeric(df[feature], errors='coerce').fillna(0).values
                        else:
                            data = np.zeros(len(df))
                        
                        stock_group.create_dataset(
                            feature, 
                            data=data.astype(np.float32),
                            compression='gzip'
                        )
                    
                    print(f"Converted {stock}: {len(df)} records")
                    
                except Exception as e:
                    print(f"Error converting {stock}: {e}")
                    continue
        
        print(f"Conversion complete: {h5_path}")
    
    @staticmethod
    def _preprocess_array(data_array: np.ndarray) -> np.ndarray:
        """Preprocess the data array (handle NaNs, normalize, etc.)"""
        # Fill NaNs
        mask = np.isnan(data_array)
        if mask.any():
            print(f"Filling {mask.sum()} NaN values...")
            # Forward fill along time axis
            for stock_idx in range(data_array.shape[1]):
                for feature_idx in range(data_array.shape[2]):
                    series = data_array[:, stock_idx, feature_idx]
                    # Forward fill
                    mask_series = np.isnan(series)
                    if mask_series.any():
                        series = pd.Series(series).ffill().bfill().fillna(0).values
                        data_array[:, stock_idx, feature_idx] = series
        
        # Normalize features (z-score normalization along time axis)
        mean = np.nanmean(data_array, axis=0, keepdims=True)
        std = np.nanstd(data_array, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        data_array = (data_array - mean) / std
        
        # Clip extreme values
        data_array = np.clip(data_array, -5, 5)
        
        return data_array

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
                 n_envs: int = 64,
                 transaction_cost: float = 0.001,
                 sharpe_window: int = 252):  # ~1 year for Sharpe calculation
        
        self.data_root = data_root
        self.window_size = window_size
        self.start_date = start_date
        self.end_date = end_date
        self.n_envs = n_envs
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.sharpe_window = sharpe_window
        
        # Load stock list if not provided
        if stocks is None:
            stocks = self._load_stock_list()
        self.stocks = stocks
        self.n_stocks = len(stocks)
        
        # Define features if not provided
        if features is None:
            features = ['close', 'open', 'high', 'low', 'volume', 'returns_1d', 
                       'volatility_10d', 'rsi_14', 'ma_20', 'ma_50']
        self.features = features
        self.n_features = len(features)
        
        # Load and preprocess data
        self.data_loader = JAXPortfolioDataLoader(data_root, stocks, features)
        self.data, self.dates, self.actual_dates = self.data_loader.load_and_preprocess_data(
            start_date, end_date, preload_to_gpu=True
        )
        
        self.n_timesteps = len(self.dates)
        
        # Action space: continuous weights for each stock (will be softmax normalized)
        self.action_dim = self.n_stocks
        
        # Observation space: (window_size * n_stocks * n_features) + portfolio_state
        obs_size = self.window_size * self.n_stocks * self.n_features + self.n_stocks + 2
        self.obs_dim = obs_size
        
        print(f"Environment initialized:")
        print(f"  Stocks: {self.n_stocks}")
        print(f"  Features: {self.n_features}")  
        print(f"  Window size: {self.window_size}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Timesteps: {self.n_timesteps}")
        print(f"  Parallel envs: {self.n_envs}")
    
    def _load_stock_list(self) -> List[str]:
        """Load stock list from file or directory"""
        stocks_file = Path("finagent/stocks.txt")
        if stocks_file.exists():
            with open(stocks_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # Fallback to scanning directory
        data_path = Path(self.data_root)
        return [p.stem.replace('_aligned', '') for p in data_path.glob("*_aligned.csv")]
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, chex.Array]:
        """Reset environment state"""
        # Initialize portfolio with equal cash allocation
        initial_weights = jnp.zeros(self.n_stocks)
        
        # Initialize Sharpe calculation buffer
        sharpe_buffer = jnp.zeros(self.sharpe_window)
        
        env_state = EnvState(
            current_step=self.window_size - 1,  # Start after window
            portfolio_weights=initial_weights,
            cash_weight=1.0,
            done=False,
            total_return=0.0,
            portfolio_value=1.0,  # Normalized to 1.0
            sharpe_buffer=sharpe_buffer,
            sharpe_buffer_idx=0
        )
        
        obs = self._get_observation(env_state)
        return env_state, obs
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, env_state: EnvState, action: chex.Array) -> Tuple[EnvState, chex.Array, float, bool, dict]:
        """Execute one environment step"""
        # Normalize actions to valid portfolio weights using softmax
        portfolio_weights = jax.nn.softmax(action)
        cash_weight = 0.0  # Assume fully invested for simplicity
        
        # Calculate transaction costs
        weight_changes = jnp.abs(portfolio_weights - env_state.portfolio_weights)
        transaction_costs = jnp.sum(weight_changes) * self.transaction_cost
        
        # Get returns for current step
        returns = self._get_returns(env_state.current_step)
        
        # Calculate portfolio return
        portfolio_return = jnp.sum(portfolio_weights * returns) - transaction_costs
        
        # Update portfolio value and total return
        new_portfolio_value = env_state.portfolio_value * (1 + portfolio_return)
        new_total_return = env_state.total_return + portfolio_return
        
        # Update Sharpe buffer (rolling window of returns)
        new_sharpe_buffer = env_state.sharpe_buffer.at[env_state.sharpe_buffer_idx].set(portfolio_return)
        new_sharpe_buffer_idx = (env_state.sharpe_buffer_idx + 1) % self.sharpe_window
        
        # Calculate reward (Sharpe ratio)
        returns_mean = jnp.mean(new_sharpe_buffer)
        returns_std = jnp.std(new_sharpe_buffer)
        sharpe_ratio = jnp.where(returns_std > 0, returns_mean / returns_std, 0.0)
        reward = sharpe_ratio
        
        # Check if done
        next_step = env_state.current_step + 1
        done = next_step >= self.n_timesteps - 1
        
        # Create new state
        new_env_state = EnvState(
            current_step=next_step,
            portfolio_weights=portfolio_weights,
            cash_weight=cash_weight,
            done=done,
            total_return=new_total_return,
            portfolio_value=new_portfolio_value,
            sharpe_buffer=new_sharpe_buffer,
            sharpe_buffer_idx=new_sharpe_buffer_idx
        )
        
        # Get next observation
        obs = self._get_observation(new_env_state)
        
        # Info dict
        info = {
            'portfolio_return': portfolio_return,
            'portfolio_value': new_portfolio_value,
            'total_return': new_total_return,
            'sharpe_ratio': sharpe_ratio,
            'transaction_costs': transaction_costs
        }
        
        return new_env_state, obs, reward, done, info
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, env_state: EnvState) -> chex.Array:
        """Get observation from current environment state"""
        current_step = env_state.current_step
        
        # Get market data window
        start_idx = jnp.maximum(0, current_step - self.window_size + 1)
        end_idx = current_step + 1
        
        # Extract market window and reshape
        market_window = lax.dynamic_slice(
            self.data, 
            (start_idx, 0, 0),
            (self.window_size, self.n_stocks, self.n_features)
        )
        
        # Flatten market data: (window_size * n_stocks * n_features,)
        market_obs = market_window.flatten()
        
        # Portfolio state: current weights + cash weight + total return
        portfolio_obs = jnp.concatenate([
            env_state.portfolio_weights,
            jnp.array([env_state.cash_weight, env_state.total_return])
        ])
        
        # Combine observations
        obs = jnp.concatenate([market_obs, portfolio_obs])
        
        return obs
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_returns(self, step: int) -> chex.Array:
        """Get returns for all stocks at given step"""
        # Get current and previous prices (assuming first feature is price)
        current_prices = self.data[step, :, 0]  # Current step, all stocks, first feature
        prev_step = jnp.maximum(0, step - 1)
        prev_prices = self.data[prev_step, :, 0]
        
        # Calculate returns, handling division by zero
        returns = jnp.where(
            prev_prices > 0,
            (current_prices - prev_prices) / prev_prices,
            0.0
        )
        
        return returns
    
    # Vectorized environment operations
    @partial(jax.jit, static_argnums=(0,))
    def batch_reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        """Reset multiple environments in parallel"""
        keys = random.split(key, self.n_envs)
        
        def single_reset(key):
            return self.reset(key)
        
        env_states, obs = vmap(single_reset)(keys)
        return env_states, obs
    
    @partial(jax.jit, static_argnums=(0,))  
    def batch_step(self, env_states: chex.Array, actions: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, dict]:
        """Step multiple environments in parallel"""
        def single_step(env_state, action):
            return self.step(env_state, action)
        
        new_env_states, obs, rewards, dones, infos = vmap(single_step)(env_states, actions)
        
        # Aggregate info dict
        info = {
            'portfolio_returns': infos['portfolio_return'],
            'portfolio_values': infos['portfolio_value'], 
            'total_returns': infos['total_return'],
            'sharpe_ratios': infos['sharpe_ratio'],
            'transaction_costs': infos['transaction_costs']
        }
        
        return new_env_states, obs, rewards, dones, info

# Wrapper for training integration
class JAXPortfolioEnvWrapper:
    """Wrapper to make JAX environment compatible with training loops"""
    
    def __init__(self, env_config: dict):
        self.env = JAXVectorizedPortfolioEnv(**env_config)
        self.rng_key = random.PRNGKey(42)
    
    def reset(self):
        self.rng_key, reset_key = random.split(self.rng_key)
        self.env_states, obs = self.env.batch_reset(reset_key)
        return obs
    
    def step(self, actions):
        self.env_states, obs, rewards, dones, info = self.env.batch_step(self.env_states, actions)
        
        # Reset environments that are done
        reset_mask = dones
        if jnp.any(reset_mask):
            self.rng_key, reset_key = random.split(self.rng_key)
            reset_keys = random.split(reset_key, self.env.n_envs)
            
            def maybe_reset(env_state, done, key):
                new_state, new_obs = self.env.reset(key)
                return lax.cond(done, lambda: (new_state, new_obs), lambda: (env_state, obs))
            
            reset_states, reset_obs = vmap(maybe_reset)(self.env_states, reset_mask, reset_keys)
            
            # Update states and observations for reset environments
            self.env_states = lax.select(reset_mask, reset_states, self.env_states)
            obs = jnp.where(reset_mask[:, None], reset_obs, obs)
        
        return obs, rewards, dones, info
    
    @property
    def observation_space(self):
        return {'shape': (self.env.obs_dim,), 'dtype': jnp.float32}
    
    @property  
    def action_space(self):
        return {'shape': (self.env.action_dim,), 'dtype': jnp.float32}

# Usage example
if __name__ == "__main__":
    # Example configuration
    env_config = {
        'data_root': 'processed_data/',
        'stocks': None,  # Will auto-load
        'features': None,  # Will use defaults
        'start_date': '2024-06-06',
        'end_date': '2025-03-06', 
        'n_envs': 64,
        'window_size': 30
    }
    
    # Create environment
    wrapper = JAXPortfolioEnvWrapper(env_config)
    
    # Test reset and step
    obs = wrapper.reset()
    print(f"Reset observation shape: {obs.shape}")
    
    # Random actions
    actions = random.normal(random.PRNGKey(0), (wrapper.env.n_envs, wrapper.env.action_dim))
    obs, rewards, dones, info = wrapper.step(actions)
    
    print(f"Step observation shape: {obs.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Average reward: {jnp.mean(rewards)}")
    print(f"Done environments: {jnp.sum(dones)}")
