# In this version of the project, PortfolioManager is obsolete. This is because of issues with the vectorisation of tax calculation which is more complex.

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
# Removed ThreadPoolExecutor as it's not used in the provided JAX code for the env logic itself.

# JAX environment state
class EnvState(NamedTuple):
    """JAX-compatible environment state"""
    current_step: int
    portfolio_weights: chex.Array  # Current portfolio weights (including cash if managed directly by agent, but here we assume fully invested)
    cash_weight: float # Represents the portion of total portfolio value held in cash
    done: bool
    total_return: float # Cumulative log return for simplicity, or simple return as currently
    portfolio_value: float # Normalized to 1.0 at start
    sharpe_buffer: chex.Array  # Rolling buffer for Sharpe calculation (stores *returns*)
    sharpe_buffer_idx: int
    # Potentially add other state variables if needed for observations, e.g., current prices
    # current_prices: chex.Array # If the agent needs access to specific prices not in features.
                               # For now, it's implicitly derived from features.


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
                                preload_to_gpu: bool = True) -> Tuple[chex.Array, chex.Array, pd.DatetimeIndex]: # Return valid_dates for info
        """
        Load and preprocess data optimized for JAX training
        Returns: (data_array, dates_array, valid_dates)
        """
        print(f"Loading data for {self.n_stocks} stocks, {self.n_features} features...")
        
        # Check if HDF5 file exists, if not convert from CSV
        h5_path = self.data_root / "stocks_data.h5"
        if not h5_path.exists():
            print("HDF5 file not found. Converting from CSV...")
            self._convert_csv_to_hdf5()
        
        # Load data from HDF5
        data_arrays = []
        full_dates_df = None # To get the master date index
        
        with h5py.File(h5_path, 'r') as h5f:
            if not h5f:
                raise ValueError(f"HDF5 file {h5_path} is empty or corrupted.")

            # Load dates from one stock to establish a master date index
            # This assumes all stock CSVs are aligned by date.
            first_stock_group = next(iter(h5f.values()), None)
            if first_stock_group is None:
                raise ValueError("No stock groups found in HDF5 file.")

            all_h5_dates = pd.to_datetime(first_stock_group['dates'][:].astype(str))
            
            # Filter dates based on start_date and end_date
            date_mask = (all_h5_dates >= start_date) & (all_h5_dates <= end_date)
            valid_dates = all_h5_dates[date_mask].to_numpy() # Use numpy array for consistent output type
            
            if len(valid_dates) == 0:
                raise ValueError(f"No valid dates found between {start_date} and {end_date}.")

            # Now load data for all stocks, filtered by the common valid_dates
            for stock in self.stocks:
                if stock not in h5f:
                    print(f"Warning: {stock} not found in HDF5 file. Skipping.")
                    continue
                
                stock_group = h5f[stock]
                
                stock_data = []
                for feature in self.features:
                    if feature in stock_group:
                        feature_data = stock_group[feature][:][date_mask] # Apply the date mask
                    else:
                        print(f"Feature {feature} not found for {stock}, filling with zeros for the selected date range.")
                        feature_data = np.zeros(date_mask.sum())
                    
                    stock_data.append(feature_data)
                
                # Check if stock_data is empty before stacking
                if stock_data:
                    stock_array = np.stack(stock_data, axis=-1)  # Shape: (time, features)
                    data_arrays.append(stock_array)
                else:
                    print(f"Warning: No features loaded for {stock}. Skipping.")

        if not data_arrays:
            raise ValueError("No valid stock data loaded after filtering. Check stock list and date range.")
        
        # Combine all stocks: Shape (time, stocks, features)
        data_array = np.stack(data_arrays, axis=1)
        
        # Handle NaNs and preprocessing
        data_array = self._preprocess_array(data_array)
        
        # Convert to JAX arrays
        if preload_to_gpu:
            data_array = jnp.array(data_array, dtype=jnp.float32)
        
        # dates_array should just be indices for JAX processing
        dates_array = jnp.arange(len(valid_dates)) 
        
        print(f"Loaded data shape: {data_array.shape}")
        print(f"Date range: {pd.to_datetime(valid_dates[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(valid_dates[-1]).strftime('%Y-%m-%d')} ({len(valid_dates)} days)")
        
        return data_array, dates_array, pd.DatetimeIndex(valid_dates) # Return pd.DatetimeIndex for easy external use
    
    def _convert_csv_to_hdf5(self):
        """Convert CSV files to HDF5 format for faster loading"""
        h5_path = self.data_root / "stocks_data.h5"
        
        print(f"Converting CSV files to {h5_path}...")
        
        # Use ThreadPoolExecutor for parallel processing of CSV files
        # This can speed up conversion if many large CSVs
        max_workers = os.cpu_count() or 1
        with h5py.File(h5_path, 'w') as h5f:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for stock in self.stocks:
                    csv_path = self.data_root / f"{stock}_aligned.csv"
                    if not csv_path.exists():
                        print(f"Warning: CSV file for {stock} not found. Skipping conversion for {stock}.")
                        continue
                    futures.append(executor.submit(self._process_single_csv, h5f, stock, csv_path))
                
                # Wait for all conversions to complete
                for future in futures:
                    future.result() # This will re-raise any exceptions from the worker threads
        
        print(f"Conversion complete: {h5_path}")

    def _process_single_csv(self, h5f_obj, stock_symbol: str, csv_path: Path):
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
            stock_group.create_dataset('dates', data=dates_str.astype('S10'))
            
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
            print(f"Converted {stock_symbol}: {len(df)} records")
        except Exception as e:
            print(f"Error converting {stock_symbol}: {e}")
            # Re-raise to ensure main thread sees it, or handle as per desired error strategy


    @staticmethod
    def _preprocess_array(data_array: np.ndarray) -> np.ndarray:
        """Preprocess the data array (handle NaNs, normalize, etc.)"""
        # Fill NaNs
        mask = np.isnan(data_array)
        if mask.any():
            print(f"Filling {mask.sum()} NaN values...")
            # Forward fill along time axis
            # Use a more efficient vectorized ffill/bfill if possible, or JAX's equivalent if this becomes a JAX operation
            # For NumPy, pandas' methods are robust
            
            # Temporary convert to pandas for ffill/bfill, then convert back
            # This can be memory intensive for very large arrays.
            # An alternative is custom numpy loop, but pandas is easier.
            for s_idx in range(data_array.shape[1]):
                for f_idx in range(data_array.shape[2]):
                    series = pd.Series(data_array[:, s_idx, f_idx])
                    data_array[:, s_idx, f_idx] = series.ffill().bfill().fillna(0).values

        # Normalize features (z-score normalization along time axis)
        # Avoid division by zero when std is 0
        mean = np.mean(data_array, axis=0, keepdims=True)
        std = np.std(data_array, axis=0, keepdims=True)
        std = np.where(std == 0, 1e-8, std) # Add a small epsilon to avoid NaN/inf
        
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
                 initial_cash: float = 1000000.0, # This initial_cash is mostly for logging/conceptual value now
                 window_size: int = 30,
                 start_date: str = '2024-06-06',
                 end_date: str = '2025-03-06',
                 n_envs: int = 64,
                 transaction_cost: float = 0.001,
                 sharpe_window: int = 252, # ~1 year for Sharpe calculation
                 # Added risk_free_rate, aligning with PortfolioManager's Sharpe
                 risk_free_rate: float = 0.05):
        
        self.data_root = data_root
        self.window_size = window_size
        self.start_date = start_date
        self.end_date = end_date
        self.n_envs = n_envs
        self.initial_cash_actual = initial_cash # Keep actual initial cash for value translation
        self.transaction_cost = transaction_cost
        self.sharpe_window = sharpe_window
        self.risk_free_rate = risk_free_rate # For Sharpe ratio calculation
        
        # Load stock list if not provided
        if stocks is None:
            stocks = self._load_stock_list()
        self.stocks = stocks
        self.n_stocks = len(stocks)
        
        # Define features if not provided
        if features is None:
            # Ensuring 'close' is always the first feature for returns calculation
            default_features = ['close', 'open', 'high', 'low', 'volume', 'returns_1d', 
                               'volatility_10d', 'rsi_14', 'ma_20', 'ma_50']
            # Make sure 'close' is included and at the front if specified otherwise
            if 'close' not in features:
                 features = ['close'] + default_features
            elif features[0] != 'close':
                 features.remove('close')
                 features = ['close'] + features
            else:
                 features = default_features

        self.features = features
        self.n_features = len(features)
        
        # Load and preprocess data
        self.data_loader = JAXPortfolioDataLoader(data_root, stocks, features)
        self.data, self.dates, self.actual_dates = self.data_loader.load_and_preprocess_data(
            start_date, end_date, preload_to_gpu=True
        )
        
        self.n_timesteps = len(self.dates)
        
        # Action space: continuous weights for each stock (will be softmax normalized)
        # Plus one for cash if the agent explicitly manages cash.
        # Your current setup implicitly assumes full investment in stocks (cash_weight=0.0 in step).
        # If agent should manage cash, action_dim should be self.n_stocks + 1, and the last element is cash.
        self.action_dim = self.n_stocks # Agent output is stock weights, cash is derived or zero.
        
        # Observation space: (window_size * n_stocks * n_features) + portfolio_state
        # portfolio_state: current portfolio weights (n_stocks), cash_weight (1), total_return (1)
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
        """Load stock list from file or directory, similar to JAXPortfolioDataLoader"""
        stocks_file = Path("finagent/stocks.txt")
        if stocks_file.exists():
            with open(stocks_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # Fallback to scanning directory for CSV files
        data_path = Path(self.data_root)
        if not data_path.is_dir():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")
        
        # Look for existing HDF5 file first for stock list
        h5_path = self.data_root / "stocks_data.h5"
        if h5_path.exists():
            try:
                with h5py.File(h5_path, 'r') as h5f:
                    stocks_in_h5 = list(h5f.keys())
                    if stocks_in_h5:
                        print(f"Loaded {len(stocks_in_h5)} stock symbols from {h5_path}.")
                        return stocks_in_h5
            except Exception as e:
                print(f"Error reading stocks from HDF5: {e}. Falling back to CSV scan.")

        # Fallback to CSV files if HDF5 fails or doesn't exist
        csv_stocks = [p.stem.replace('_aligned', '') for p in data_path.glob("*_aligned.csv")]
        if csv_stocks:
            print(f"Loaded {len(csv_stocks)} stock symbols by scanning CSV files.")
            return csv_stocks
        
        raise ValueError(f"No stocks found in {stocks_file} or in {self.data_root} directory.")

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, chex.Array]:
        """Reset environment state"""
        # Initialize portfolio with equal cash allocation, normalized value 1.0
        # Agent starts with a cash position. First action will dictate how to invest.
        initial_weights = jnp.zeros(self.n_stocks) # No shares initially
        initial_cash_weight = 1.0 # 100% cash
        
        # Initialize Sharpe calculation buffer
        sharpe_buffer = jnp.zeros(self.sharpe_window)
        
        # Randomize start step for each environment within the data range
        # Ensure enough data for window_size
        min_start_step = self.window_size - 1 
        max_start_step = self.n_timesteps - 1 - self.window_size 
        
        if max_start_step <= min_start_step:
            raise ValueError(f"Not enough timesteps ({self.n_timesteps}) for window size ({self.window_size}).")

        # Each environment starts at a random time step
        start_step = random.randint(key, (), min_start_step, max_start_step)
        
        env_state = EnvState(
            current_step=start_step,  
            portfolio_weights=initial_weights, # Initial weights are all cash, so stock weights are zero
            cash_weight=initial_cash_weight,
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
        # Actions are raw policy outputs, expected to be continuous values (e.g., means of a Gaussian)
        # We need to convert these to valid portfolio weights (sum to 1, including cash)
        # Assuming action is (n_stocks,) and represents target weights for stocks.
        # We need to add a cash component explicitly if the agent is to manage it.
        # For now, let's assume `action` represents desired weights for stocks, and the remaining
        # will be implicitly cash or a fixed cash component.
        
        # Method 1: Agent directly outputs target weights for N stocks. Cash weight is derived.
        # We apply softmax to ensure they sum to 1.
        # The agent's action dim would be N_stocks + 1 if it explicitly manages cash and N_stocks if it implicitly allocates remaining to cash.
        
        # Let's assume `action` (shape `self.n_stocks`) are the unnormalized target weights for stocks.
        # The agent controls N_stocks and the remaining becomes cash.
        # We need to decide if cash is explicitly managed or derived.
        # Current design `cash_weight = 0.0` suggests fully invested. Let's make it flexible.
        
        # If `action_dim` is `n_stocks`: The agent outputs N values. We need to normalize these including cash.
        # Common approach: add a 'cash' dimension to the action logits before softmax.
        # `action` should be `(self.n_stocks + 1,)` if agent directly controls cash weight.
        # For now, let's stick to the previous interpretation: `action` are just target stock weights,
        # and we use softmax on them to normalize. This implies the agent is always fully invested.
        # If cash is to be managed, `action_dim` in `LSTMActorCritic` should be `self.n_stocks + 1`.
        # Let's update `action_dim` in __init__ to `self.n_stocks + 1` for explicit cash management.
        
        # Re-evaluating `action_dim` and `action` interpretation:
        # If `self.action_dim` is `self.n_stocks`, then `action` has `self.n_stocks` elements.
        # Agent's policy outputs logits for N stocks.
        # To incorporate cash, we need `self.n_stocks + 1` logits.
        
        # Option 1 (current interpretation): Agent only chooses stock weights, cash is ignored or implicitly 0.
        # portfolio_weights = jax.nn.softmax(action) # Sums to 1. No explicit cash weight.
        # cash_weight = 0.0
        
        # Option 2 (better for real portfolio management): Agent outputs logits for N stocks + 1 cash.
        # Then softmax gives (N_stocks + 1) portfolio_weights.
        # Let's modify action_dim in __init__ to `self.n_stocks + 1` for this.
        # So `action` will be `(self.n_stocks + 1,)`
        # The last element of `portfolio_weights` will be cash.
        
        # Updated action interpretation: action is raw output for (stocks + cash)
        # If your agent outputs only `n_stocks`, then you need to decide how cash is handled.
        # For now, I'll assume `action` comes from a policy with `action_dim = self.n_stocks + 1`.
        # So `action` will have `self.n_stocks + 1` elements.
        # The last element is the unnormalized 'cash' component.

        # Normalize actions to valid portfolio weights (stocks + cash)
        all_weights_raw = action # Policy output for N stocks + 1 cash
        all_weights_softmax = jax.nn.softmax(all_weights_raw)
        
        portfolio_weights = all_weights_softmax[:-1] # Weights for stocks
        cash_weight = all_weights_softmax[-1]       # Weight for cash
        
        # Calculate transaction costs
        # Only apply costs to *changes* in stock weights, not cash weight.
        # The 'cash_weight' from env_state.portfolio_weights might be used if we want to include cash in transaction cost calculations.
        # For simplicity, let's assume `env_state.portfolio_weights` only stores stock weights.
        # This means we need to adjust `env_state.portfolio_weights` to also include cash weight or only track stock weights.
        
        # Let's assume `env_state.portfolio_weights` stores *stock* weights.
        # If the agent wants to reduce a stock position to increase cash, it's still a transaction.
        
        # Current transaction cost calculation:
        # weight_changes = jnp.abs(portfolio_weights - env_state.portfolio_weights) # This is only for stocks
        
        # Revised transaction cost: consider total portfolio value and amount of reallocation.
        # The `transaction_cost` from your PortfolioManager is proportional to `transaction_value`.
        # The JAX env currently applies `transaction_cost` to changes in weights.
        # It's a simplification, `sum(weight_changes) * self.transaction_cost` acts like a "slippage" or rebalancing cost.
        
        # Let's keep the current JAX transaction cost calculation for simplicity in RL.
        # The agent's `portfolio_weights` (for stocks) and `env_state.portfolio_weights` (previous stock weights)
        # are what matter for calculating transaction costs on *stock* rebalancing.
        
        # Recalculate `weight_changes` and `transaction_costs` based on *stock* weights only.
        # The agent's `action` determines the *target* stock weights.
        # The transaction costs should be proportional to the value of assets being bought/sold.
        # In a normalized environment, this is (change in weight) * portfolio_value.
        
        # Value of new stock positions: portfolio_weights * new_portfolio_value
        # Value of old stock positions: env_state.portfolio_weights * env_state.portfolio_value
        # The `portfolio_return` below will already incorporate price changes.
        
        # Transaction costs for rebalancing:
        # The cost applies to the *amount* of asset bought/sold.
        # Here `env_state.portfolio_value` is normalized to 1.0.
        # `portfolio_weights` are also normalized to 1 (excluding cash in this context)
        # or sum to 1 with cash if `action_dim = n_stocks + 1`.
        
        # Let's assume `portfolio_weights` are the target stock weights (sum to < 1 if cash > 0).
        # We need previous stock weights from `env_state.portfolio_weights`.
        
        # Compute changes in *value* of stock positions.
        # Value change due to agent's action: (new_weight - old_weight) * current_portfolio_value
        # This needs to be applied to the *previous* portfolio value.
        # However, it is simpler for PPO to just calculate costs on *weight changes*.
        
        # Keep the existing transaction cost logic in the JAX env for simplicity.
        # It assumes a proportional cost on the absolute sum of weight changes.
        weight_changes = jnp.abs(portfolio_weights - env_state.portfolio_weights) # Only for stock weights
        transaction_costs_value = jnp.sum(weight_changes) * self.transaction_cost * env_state.portfolio_value
        
        # Get returns for current step (next day's returns based on prices at `current_step + 1`)
        # returns is (n_stocks,)
        returns = self._get_returns(env_state.current_step + 1) # Returns for moving to next day
        
        # Calculate portfolio return from stock positions and cash
        # Return on stocks: sum(weights_stocks * returns_stocks)
        # Return on cash: cash_weight * (risk_free_rate / 252) if you want cash to earn interest
        
        # For simplicity, assume cash earns 0 or is part of a risk-free asset in the `sharpe_ratio`.
        # Here, the 'portfolio_return' is just from the invested stock portion.
        # The `portfolio_value` will then reflect overall value.
        
        stock_portfolio_return = jnp.sum(portfolio_weights * returns)
        
        # Total portfolio return including cash, and accounting for transaction costs.
        # (Weight in stocks * Stock Returns) + (Weight in Cash * Cash Returns) - Transaction Costs
        # Assuming cash has 0 returns for now in the daily update, as it's typically risk-free.
        # Or, cash return could be a small daily risk-free rate. For now, let's simplify.
        
        # The agent's *action* dictates the split between stocks and cash.
        # `portfolio_weights` are stock weights, `cash_weight` is the cash weight.
        
        # New value of portfolio:
        # Initial value of stocks: env_state.portfolio_weights * env_state.portfolio_value
        # Value change from returns on old positions: sum(env_state.portfolio_weights * returns) * env_state.portfolio_value
        # This is essentially `env_state.portfolio_value * (1 + stock_portfolio_return_on_old_weights)`
        
        # A more direct way:
        # The `portfolio_value` is updated based on `portfolio_return`.
        # `portfolio_return` should be the *net* return of the entire portfolio *before* new allocation.
        # It needs to reflect the change in value of *previous* holdings.
        
        # Let's adjust `portfolio_return` calculation. It should be the return *of the current holdings*.
        # The `env_state.portfolio_weights` are the holdings *from the previous step*.
        # The `action` decides the *new* holdings.
        
        # Return from holding *previous* portfolio weights for one day
        # `env_state.portfolio_weights` sum to (1 - previous cash_weight)
        return_on_old_stock_positions = jnp.sum(env_state.portfolio_weights * returns)
        return_on_old_cash_position = env_state.cash_
