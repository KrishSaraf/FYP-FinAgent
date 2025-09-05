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
                                 preload_to_gpu: bool = True) -> Tuple[chex.Array, chex.Array, pd.DatetimeIndex]:
        """
        Load and preprocess data optimized for JAX training
        Returns: (data_array, dates_array, valid_dates)
        """
        print(f"Loading data for {self.n_stocks} stocks, {self.n_features} features...")

        h5_path = self.data_root / "stocks_data.h5"
        if not h5_path.exists():
            print("HDF5 file not found. Converting from CSV...")
            self.data_root.mkdir(parents=True, exist_ok=True)
            self._convert_csv_to_hdf5()

        data_arrays = []
        
        with h5py.File(h5_path, 'r') as h5f:
            if not h5f:
                raise ValueError(f"HDF5 file {h5_path} is empty or corrupted.")

            first_stock_group_name = next(iter(h5f.keys()), None)
            if first_stock_group_name is None:
                raise ValueError("No stock groups found in HDF5 file.")
            first_stock_group = h5f[first_stock_group_name]

            all_h5_dates = pd.to_datetime(first_stock_group['dates'][:].astype(str))

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            date_mask = (all_h5_dates >= start_dt) & (all_h5_dates <= end_dt)
            valid_dates = all_h5_dates[date_mask].to_numpy()

            if len(valid_dates) == 0:
                raise ValueError(f"No valid dates found between {start_date} and {end_date}.")

            for stock in self.stocks:
                if stock not in h5f:
                    print(f"Warning: {stock} not found in HDF5 file. Skipping.")
                    continue

                stock_group = h5f[stock]

                stock_data = []
                for feature in self.features:
                    if feature in stock_group:
                        feature_data = stock_group[feature][:][date_mask]
                    else:
                        print(f"Feature {feature} not found for {stock}, filling with zeros for the selected date range.")
                        feature_data = np.zeros(date_mask.sum())

                    stock_data.append(feature_data)

                if stock_data:
                    stock_array = np.stack(stock_data, axis=-1)
                    data_arrays.append(stock_array)
                else:
                    print(f"Warning: No features loaded for {stock}. Skipping.")

        if not data_arrays:
            raise ValueError("No valid stock data loaded after filtering. Check stock list and date range.")

        data_array = np.stack(data_arrays, axis=1)

        data_array = self._preprocess_array(data_array)

        data_array = jnp.array(data_array, dtype=jnp.float32)

        dates_array = jnp.arange(len(valid_dates))

        print(f"Loaded data shape: {data_array.shape}")
        print(f"Date range: {pd.to_datetime(valid_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(valid_dates[-1]).strftime('%Y-%m-%d')} ({len(valid_dates)} days)")

        return data_array, dates_array, pd.DatetimeIndex(valid_dates)

    def _convert_csv_to_hdf5(self):
        """Convert CSV files to HDF5 format for faster loading"""
        h5_path = self.data_root / "stocks_data.h5"

        print(f"Converting CSV files to {h5_path}...")

        max_workers = os.cpu_count() or 1
        with h5py.File(h5_path, 'w') as h5f:
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for stock in self.stocks:
                    csv_path = self.data_root / f"{stock}_aligned.csv"
                    if not csv_path.exists():
                        print(f"Warning: CSV file for {stock} not found at {csv_path}. Skipping conversion for {stock}.")
                        continue
                    futures.append(executor.submit(self._process_single_csv, h5f, stock, csv_path, self.features))

                for future in futures:
                    future.result()

        print(f"Conversion complete: {h5_path}")

    def _process_single_csv(self, h5f_obj, stock_symbol: str, csv_path: Path, features_to_save: List[str]):
        """Helper to process a single CSV into HDF5, called by ThreadPoolExecutor"""
        try:
            print(f"Processing {stock_symbol} from {csv_path}...")
            df_chunks = pd.read_csv(csv_path, chunksize=10000)
            df = pd.concat(df_chunks, ignore_index=True)

            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)

            stock_group = h5f_obj.create_group(stock_symbol)

            dates_str = df[date_col].dt.strftime('%Y-%m-%d').values
            stock_group.create_dataset('dates', data=dates_str.astype('S10'), compression='gzip')

            for feature in features_to_save:
                if feature in df.columns:
                    data = pd.to_numeric(df[feature], errors='coerce').fillna(0).values.astype(np.float32)
                else:
                    data = np.zeros(len(df), dtype=np.float32)

                stock_group.create_dataset(
                    feature,
                    data=data,
                    compression='gzip'
                )
            print(f"Converted {stock_symbol}: {len(df)} records")
        except Exception as e:
            print(f"Error converting {stock_symbol}: {e}")
            raise

    @staticmethod
    def _preprocess_array(data_array: np.ndarray) -> np.ndarray:
        """
        Preprocess the data array (handle NaNs, normalize, etc.) using NumPy.
        Input: (time, stocks, features)
        Output: (time, stocks, features)
        """
        if np.isnan(data_array).any():
            print(f"Filling {np.isnan(data_array).sum()} NaN values in data_array...")
            for s_idx in range(data_array.shape[1]):
                for f_idx in range(data_array.shape[2]):
                    series = data_array[:, s_idx, f_idx]

                    mask = np.isnan(series)
                    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
                    np.maximum.accumulate(idx, out=idx)
                    series = series[idx]

                    mask = np.isnan(series)
                    idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0] - 1)
                    np.minimum.accumulate(idx[::-1], out=idx[::-1])
                    series = series[idx]

                    series = np.nan_to_num(series, nan=0.0)

                    data_array[:, s_idx, f_idx] = series
        
        mean = np.mean(data_array, axis=0, keepdims=True)
        std = np.std(data_array, axis=0, keepdims=True)

        std = np.where(std == 0, 1e-8, std)

        data_array = (data_array - mean) / std

        data_array = np.clip(data_array, -5.0, 5.0)

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
                 transaction_cost_rate: float = 0.005,
                 sharpe_window: int = 252,
                 risk_free_rate: float = 0.05):

        self.data_root = data_root
        self.window_size = window_size
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash_actual = initial_cash
        self.transaction_cost_rate = transaction_cost_rate
        self.sharpe_window = sharpe_window
        self.risk_free_rate_daily = risk_free_rate / 252.0

        if stocks is None:
            # Example stock list - you should replace with your actual stocks
            # For demonstration, ensure these CSVs exist in 'processed_data'
            stocks = ['AAPL', 'MSFT', 'GOOG'] 
        self.stocks = stocks
        self.n_stocks = len(stocks)

        if features is None:
            default_candidate_features = ['close', 'open', 'high', 'low', 'volume', 'returns_1d',
                                          'volatility_10d', 'rsi_14', 'ma_20', 'ma_50']
            self.features = ['close'] + [f for f in default_candidate_features if f != 'close']
        else:
            if 'close' not in features:
                 features = ['close'] + features
            elif features[0] != 'close':
                 features.remove('close')
                 features = ['close'] + features
            self.features = features

        self.n_features = len(self.features)

        self.data_loader = JAXPortfolioDataLoader(data_root, stocks, self.features)
        self.data, self.dates_idx, self.actual_dates = self.data_loader.load_and_preprocess_data(
            start_date, end_date, preload_to_gpu=True
        )
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

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, chex.Array]:
        """Reset environment state"""
        initial_weights = jnp.zeros(self.n_stocks)
        initial_cash_weight = 1.0

        initial_portfolio_weights = jnp.append(initial_weights, initial_cash_weight)
        chex.assert_trees_all_equal(jnp.sum(initial_portfolio_weights), 1.0)

        sharpe_buffer = jnp.zeros(self.sharpe_window)

        min_start_step = self.window_size - 1
        max_start_step = self.n_timesteps - 2

        chex.assert_scalar_in(min_start_step, max_start_step)

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

        transaction_cost_value = jnp.sum(jnp.abs(new_stock_weights - prev_stock_weights)) * self.transaction_cost_rate

        current_daily_returns = self._get_daily_returns_from_data(env_state.current_step + 1)

        daily_portfolio_return_before_costs = (
            jnp.sum(prev_stock_weights * current_daily_returns) +
            (prev_cash_weight * self.risk_free_rate_daily)
        )

        net_daily_portfolio_return = daily_portfolio_return_before_costs - transaction_cost_value

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
            'transaction_cost_value': transaction_cost_value,
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
        Calculates daily returns for all stocks for a given step.
        Returns array of shape (n_stocks,).
        """
        price_t = self.data[step, :, 0]
        price_t_minus_1 = self.data[step - 1, :, 0]
        
        price_t_minus_1_safe = jnp.where(price_t_minus_1 == 0, 1e-8, price_t_minus_1)

        daily_returns = (price_t / price_t_minus_1_safe) - 1.0
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
