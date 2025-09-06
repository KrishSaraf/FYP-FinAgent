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
import os # Added for os.cpu_count()
import distrax

# JAX environment state
class EnvState(NamedTuple):
    """JAX-compatible environment state"""
    current_step: int
    portfolio_weights: chex.Array  # Current portfolio weights (stocks + cash, sums to 1)
    done: bool
    total_return: float # Cumulative simple return, normalized to initial value of 1.0
    portfolio_value: float # Normalized to 1.0 at start, represents the multiplier of initial value
    sharpe_buffer: chex.Array  # Rolling buffer for Sharpe calculation (stores *daily_returns*)
    sharpe_buffer_idx: int


class JAXPortfolioDataLoader:
    """Optimized data loader for portfolio environment (CSV-based, flexible feature handling)"""

    def __init__(self, data_root: str, stocks: List[str], features: Optional[List[str]] = None,
                 use_all_features: bool = False):
        self.data_root = Path(data_root)
        self.stocks = stocks
        self.features = features  # can be None, will be inferred
        self.use_all_features = use_all_features

    def load_and_preprocess_data(self,
                                 start_date: str,
                                 end_date: str,
                                 fill_missing_features_with: str = 'interpolate',
                                 preload_to_gpu: bool = True) -> Tuple[chex.Array, chex.Array, pd.DatetimeIndex]:
        """
        Load and preprocess CSV data into a consistent tensor for JAX.

        Args:
            fill_missing_features_with: 'zero', 'nan', 'forward_fill', 'interpolate'
            preload_to_gpu: whether to return jnp.array instead of np.array
        """

        all_data = []
        all_features = set()
        stock_features = {}

        # Pass 1: Collect available features (ONLY from actual stock files)
        for stock in self.stocks:
            csv_path = self.data_root / f"{stock}_aligned.csv"
            if not csv_path.exists():
                print(f"Warning: CSV for {stock} not found, skipping.")
                continue
            try:
                sample_df = pd.read_csv(csv_path, nrows=0, index_col=0)
                feats = list(sample_df.columns)  # All columns are features after setting index
                stock_features[stock] = feats
                all_features.update(feats)
                print(f"Stock {stock}: {len(feats)} features, has 'close': {'close' in feats}")
            except Exception as e:
                print(f"Warning: Error reading {stock}: {e}, skipping.")
                continue

        print(f"DEBUG: Total stocks processed: {len(stock_features)}")
        print(f"DEBUG: All features found: {sorted(list(all_features))}")
        print(f"DEBUG: 'close' in all_features: {'close' in all_features}")

        if self.features is None:
            if self.use_all_features:
                self.features = sorted(list(all_features))
                print(f"DEBUG: Using all features: {len(self.features)} features")
            else:
                common = set.intersection(*[set(f) for f in stock_features.values()])
                self.features = sorted(list(common)) if common else sorted(list(all_features))
                print(f"DEBUG: Using common features: {len(self.features)} features")
        
        # CRITICAL: Ensure 'close' price is available and prioritize it
        if 'close' not in self.features:
            raise ValueError("'close' price must be available in the data for portfolio returns calculation")
        
        # Reorganize features to put 'close' first for reliable indexing
        if 'close' in self.features:
            self.features = ['close'] + [f for f in self.features if f != 'close']
        
        print(f"Using features: {self.features}")
        print(f"Close price will be at index 0 for reliable returns calculation")

        # Pass 2: Load and standardize
        for stock in self.stocks:
            csv_path = self.data_root / f"{stock}_aligned.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            df.index.name = "date"  # Ensure index is named 'date'
            df = df.sort_index()
            df = df.loc[start_date:end_date]

            proc = pd.DataFrame(index=df.index)
            for feat in self.features:
                if feat in df.columns:
                    proc[feat] = pd.to_numeric(df[feat], errors="coerce")
                else:
                    print(f"{feat} missing for {stock}, filling with {fill_missing_features_with}")
                    proc[feat] = np.nan

            # Fill missing values
            if fill_missing_features_with == "zero":
                proc = proc.fillna(0.0)
            elif fill_missing_features_with == "forward_fill":
                proc = proc.ffill().bfill().fillna(0.0)
            elif fill_missing_features_with == "interpolate":
                proc = proc.interpolate(method="linear").ffill().bfill().fillna(0.0)
            else:  # default nan then robust pipeline
                proc = proc.ffill().bfill().interpolate().ffill().bfill().fillna(0.0)

            proc["symbol"] = stock
            all_data.append(proc)

        if not all_data:
            raise RuntimeError("No stock data loaded.")

        panel = pd.concat(all_data).reset_index().set_index(["date", "symbol"])
        panel = panel[~panel.index.duplicated(keep="first")]
        panel = panel.unstack(level="symbol").sort_index(axis=1)
        full_columns = pd.MultiIndex.from_product([self.features, self.stocks])
        
        # Align to requested date range
        valid_dates = pd.date_range(start=start_date, end=end_date, freq="B")
        panel = panel.reindex(valid_dates, columns=full_columns, method="ffill").fillna(0.0)

        # Convert to numpy/jax
        data_array = panel.values.astype(np.float32)  # shape (T, features*stocks)
        n_days = len(valid_dates)
        n_features = len(self.features)
        n_stocks = len(self.stocks)
        data_array = data_array.reshape(n_days, n_features, n_stocks).transpose(0, 2, 1)

        if preload_to_gpu:
            data_array = jnp.array(data_array)

        return data_array, jnp.arange(n_days), valid_dates, n_features


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
                 risk_free_rate: float = 0.05,
                 use_all_features: bool = True):

        self.data_root = data_root
        self.window_size = window_size
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash_actual = initial_cash
        self.transaction_cost_rate = transaction_cost_rate
        self.sharpe_window = sharpe_window
        self.risk_free_rate_daily = risk_free_rate / 252.0
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
        
        # Set close price index (should be 0 after reorganization)
        self.close_price_idx = 0
        
        # Validate that we have valid price data
        if self.data.shape[0] < self.window_size + 2:
            raise ValueError(f"Insufficient data: need at least {self.window_size + 2} time steps, got {self.data.shape[0]}")
        self.n_timesteps = len(self.dates_idx)

        self.action_dim = self.n_stocks + 1
        chex.assert_trees_all_equal(self.data.shape[1], self.n_stocks)

        obs_size = (self.window_size * self.n_stocks * self.n_features) + self.action_dim + 2

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
        """Reset environment state"""
        initial_weights = jnp.zeros(self.n_stocks)
        initial_cash_weight = 1.0

        initial_portfolio_weights = jnp.append(initial_weights, initial_cash_weight)
    
        sharpe_buffer = jnp.zeros(self.sharpe_window)

        min_start_step = self.window_size - 1
        max_start_step = self.n_timesteps - 2

        start_step = random.randint(key, (), min_start_step, max_start_step + 1)

        env_state = EnvState(
            current_step=start_step,
            portfolio_weights=initial_portfolio_weights,
            done=False,
            total_return=0.0,
            portfolio_value=1.0,
            sharpe_buffer=sharpe_buffer,
            sharpe_buffer_idx=0
        )

        obs = self._get_observation(env_state)
        return env_state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, env_state: EnvState, action: chex.Array) -> Tuple[EnvState, chex.Array, float, bool, dict]:
        """Execute one environment step"""
        normalized_action_weights = jax.nn.softmax(action)
        chex.assert_trees_all_equal(normalized_action_weights.shape, (self.action_dim,))

        new_stock_weights = normalized_action_weights[:-1]
        new_cash_weight = normalized_action_weights[-1]

        prev_stock_weights = env_state.portfolio_weights[:-1]
        prev_cash_weight = env_state.portfolio_weights[-1]
        
        prev_portfolio_value = env_state.portfolio_value

        # Calculate transaction cost as proportional to weight changes
        # Transaction cost should be: sum(|new_weight - old_weight|) * cost_rate * portfolio_value
        weight_change_total = jnp.sum(jnp.abs(new_stock_weights - prev_stock_weights))
        transaction_cost_rate_applied = weight_change_total * self.transaction_cost_rate

        current_daily_returns = self._get_daily_returns_from_data(env_state.current_step + 1)

        # Portfolio return from holdings (before transaction costs)
        daily_portfolio_return_before_costs = (
            jnp.sum(prev_stock_weights * current_daily_returns) +
            (prev_cash_weight * self.risk_free_rate_daily)
        )

        # Apply transaction costs as a percentage reduction of portfolio value
        # Net return = gross return - transaction cost rate
        net_daily_portfolio_return = daily_portfolio_return_before_costs - transaction_cost_rate_applied

        new_portfolio_value = prev_portfolio_value * (1.0 + net_daily_portfolio_return)

        new_total_return = (new_portfolio_value - 1.0)

        new_sharpe_buffer = env_state.sharpe_buffer.at[env_state.sharpe_buffer_idx].set(net_daily_portfolio_return)
        new_sharpe_buffer_idx = (env_state.sharpe_buffer_idx + 1) % self.sharpe_window

        sharpe_mean = jnp.mean(new_sharpe_buffer)
        sharpe_std = jnp.std(new_sharpe_buffer)

        sharpe_ratio = lax.cond(
            sharpe_std == 0,
            lambda: 0.0,
            lambda: (sharpe_mean - self.risk_free_rate_daily) / sharpe_std * jnp.sqrt(252.0)
        )
        
        reward = sharpe_ratio

        next_step = env_state.current_step + 1
        done = (next_step >= self.n_timesteps - 1) | (new_portfolio_value <= 0.5)

        new_env_state = EnvState(
            current_step=next_step,
            portfolio_weights=normalized_action_weights,
            done=done,
            total_return=new_total_return,
            portfolio_value=new_portfolio_value,
            sharpe_buffer=new_sharpe_buffer,
            sharpe_buffer_idx=new_sharpe_buffer_idx
        )

        next_obs = self._get_observation(new_env_state)

        info = {
            'date_idx': next_step,
            'portfolio_value': new_portfolio_value,
            'total_return': new_total_return,
            'sharpe_ratio': sharpe_ratio,
            'daily_portfolio_return': net_daily_portfolio_return,
            'transaction_cost_value': transaction_cost_rate_applied,
            'new_stock_weights': new_stock_weights,
            'new_cash_weight': new_cash_weight,
            'prev_stock_weights': prev_stock_weights,
            'prev_cash_weight': prev_cash_weight
        }

        return new_env_state, next_obs, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, env_state: EnvState) -> chex.Array:
        """
        Constructs the observation for the current step.
        """
        start_idx = env_state.current_step - self.window_size + 1
        end_idx = env_state.current_step + 1
        
        data_slice = lax.dynamic_slice(
            self.data,
            (start_idx, 0, 0),
            (self.window_size, self.n_stocks, self.n_features)
        )
        
        market_data_flat = data_slice.flatten()

        portfolio_state_flat = jnp.concatenate([
            env_state.portfolio_weights,
            jnp.array([env_state.portfolio_value]),
            jnp.array([env_state.total_return])
        ])

        obs = jnp.concatenate([market_data_flat, portfolio_state_flat])
        chex.assert_trees_all_equal(obs.shape, (self.obs_dim,))

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
        
        # Safety check for zero/negative prices
        price_t_minus_1_safe = jnp.where(price_t_minus_1 <= 0, 1e-8, price_t_minus_1)
        price_t_safe = jnp.where(price_t <= 0, 1e-8, price_t)

        daily_returns = (price_t_safe / price_t_minus_1_safe) - 1.0
        
        # Sanity check: cap extreme returns (likely data errors)
        daily_returns = jnp.clip(daily_returns, -0.5, 0.5)  # Cap at ±50% daily moves
        
        return daily_returns

# # Example usage (for testing)
# if __name__ == "__main__":
#     # Create a dummy 'processed_data' directory and some CSVs for testing
#     # In a real scenario, you'd have your actual stock data here.
#     data_dir = Path("processed_data")
#     data_dir.mkdir(exist_ok=True)

#     # Generate dummy CSV files for AAPL, MSFT, GOOG
#     stocks_for_test = ['AAPL', 'MSFT', 'GOOG']
#     dates = pd.date_range(start='2023-12-01', end='2024-12-31', freq='B') # Business days
    
#     for stock in stocks_for_test:
#         dummy_data = {
#             'Date': dates,
#             'close': np.random.rand(len(dates)) * 100 + 100, # Base price 100-200
#             'open': np.random.rand(len(dates)) * 100 + 95,
#             'high': np.random.rand(len(dates)) * 100 + 105,
#             'low': np.random.rand(len(dates)) * 100 + 90,
#             'volume': np.random.randint(100000, 10000000, len(dates)),
#             'returns_1d': np.random.randn(len(dates)) * 0.01,
#             'volatility_10d': np.random.rand(len(dates)) * 0.05,
#             'rsi_14': np.random.rand(len(dates)) * 100,
#             'ma_20': np.random.rand(len(dates)) * 100 + 98,
#             'ma_50': np.random.rand(len(dates)) * 100 + 95,
#         }
#         df = pd.DataFrame(dummy_data)
#         df.to_csv(data_dir / f"{stock}_aligned.csv", index=False)
    
#     print("\n--- Dummy CSVs created for testing ---")
    
#     # Initialize the environment
#     try:
#         env = JAXVectorizedPortfolioEnv(
#             data_root="processed_data/",
#             stocks=stocks_for_test,
#             features=['close', 'volume', 'rsi_14'],
#             window_size=10,
#             start_date='2024-01-01',
#             end_date='2024-12-31',
#             transaction_cost_rate=0.001
#         )

#         key = random.PRNGKey(0)

#         # Test reset
#         key, reset_key = random.split(key)
#         env_state, obs = env.reset(reset_key)
#         print(f"\n--- Environment Reset ---")
#         print(f"Initial State: {env_state}")
#         print(f"Initial Observation shape: {obs.shape}")

#         # Test step
#         print(f"\n--- Stepping through the Environment ---")
#         num_steps_to_take = 5
#         for i in range(num_steps_to_take):
#             key, action_key = random.split(key)
#             # Example action: random weights for stocks and cash
#             # action.shape must be (self.action_dim,)
#             random_action = random.normal(action_key, (env.action_dim,))

#             env_state, next_obs, reward, done, info = env.step(env_state, random_action)
#             print(f"Step {i+1}:")
#             print(f"  Date: {env.actual_dates[env_state.current_step].strftime('%Y-%m-%d')}")
#             print(f"  Portfolio Value: {env_state.portfolio_value:.4f}")
#             print(f"  Total Return: {env_state.total_return:.4f}")
#             print(f"  Reward (Sharpe): {reward:.4f}")
#             print(f"  Done: {done}")
#             print(f"  Info (Daily Return): {info['daily_portfolio_return']:.4f}")
#             print(f"  Info (Transaction Cost): {info['transaction_cost_value']:.6f}")
#             if done:
#                 print("Environment terminated early!")
#                 break
        
#         # Example of vmap for parallel environments
#         print(f"\n--- Testing vmap for multiple environments ---")
#         num_parallel_envs = 4
        
#         # JIT-compile the reset and step functions for vmap
#         vmap_reset = jax.vmap(env.reset, in_axes=(0,))
#         vmap_step = jax.vmap(env.step, in_axes=(0, 0))

#         # Generate separate keys for each parallel environment
#         key, *reset_keys = random.split(key, num_parallel_envs + 1)
#         reset_keys = jnp.array(reset_keys)

#         # Reset all parallel environments
#         env_states, obs_batch = vmap_reset(reset_keys)
#         print(f"Parallel reset: Initial state batch (first env portfolio value): {env_states.portfolio_value[0]:.4f}")
#         print(f"Parallel reset: Obs batch shape: {obs_batch.shape}")

#         # Take a step in all parallel environments
#         key, *action_keys = random.split(key, num_parallel_envs + 1)
#         action_keys = jnp.array(action_keys)
#         random_action_batch = random.normal(action_keys, (num_parallel_envs, env.action_dim))

#         env_states, next_obs_batch, rewards_batch, dones_batch, infos_batch = vmap_step(env_states, random_action_batch)
#         print(f"Parallel step: Next state batch (first env portfolio value): {env_states.portfolio_value[0]:.4f}")
#         print(f"Parallel step: Rewards batch: {rewards_batch}")
#         print(f"Parallel step: Dones batch: {dones_batch}")

#     except Exception as e:
#         print(f"\nAn error occurred during environment testing: {e}")
#         # Optionally, you might want to delete the dummy data if an error occurs
#         # for stock in stocks_for_test:
#         #     (data_dir / f"{stock}_aligned.csv").unlink(missing_ok=True)
#         # (data_dir / "stocks_data.h5").unlink(missing_ok=True)
#         # data_dir.rmdir()

#     print("\n--- Example usage complete ---")
