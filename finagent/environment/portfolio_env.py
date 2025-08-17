import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

from finagent.environment.portfolio_manager import PortfolioManager, PositionSide

class PortfolioEnv(gym.Env):
    """
    A custom reinforcement learning environment for portfolio management.

    This environment integrates the PortfolioManager to simulate trading.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, 
                 data_root: str = "market_data/", 
                 stocks: List[str] = None,
                 initial_cash: float = 1000000.0,
                 window_size: int = 30, # Number of past days of market data to include in state
                 start_date: str = '2022-01-01',
                 end_date: str = '2023-01-01',
                 fill_missing_features_with: str = 'forward_fill',  # 'zero', 'nan', 'forward_fill', or 'interpolate'
                 standardize_features: bool = True,
                 use_all_features: bool = True):
        
        super(PortfolioEnv, self).__init__()

        self.data_root = Path(data_root)
        self.window_size = window_size
        self.start_date_str = start_date
        self.end_date_str = end_date
        self.fill_missing_features_with = fill_missing_features_with
        self.standardize_features = standardize_features
        self.use_all_features = use_all_features
        
        # Load and preprocess data
        self.stocks = stocks if stocks else self._load_stock_list()

        self.features = []
        self.num_features = 0
        self.stocks_features = 0

        self.data = self._load_and_preprocess_data(
            fill_missing_features_with=self.fill_missing_features_with,
            standardize_features=self.standardize_features
        )
        self.dates = self.data.index.unique()

        # Initialize Portfolio Manager
        self.portfolio_manager = PortfolioManager(initial_cash=initial_cash, stocks=self.stocks)

        # Define the Action Space
        # Continuous action space: one value per stock, representing the target weight.
        # The values can range from 0 (sell all/don't buy) to 1 (allocate max to this stock).
        self.action_space = spaces.Box(
            low=0, high=1, shape=(len(self.stocks),), dtype=np.float32
        )

        # Define the State Space (Observation Space)
        # It's a combination of market data and portfolio state.
        # Shape: (number of stocks, window_size + 2) -> (window_size prices + current holdings + current price)
        self._update_observation_space_size()

        # Initialize state
        self.current_step = 0
        self.done = False

    def _load_stock_list(self) -> List[str]:
        """Loads stock list from a file if not provided."""
        stocks_file = Path("finagent/stocks.txt")
        if stocks_file.exists():
            with open(stocks_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        # Fallback to scanning directory if file doesn't exist
        return [p.stem.replace('_aligned', '') for p in self.data_root.glob("*_aligned.csv")]

    def _load_and_preprocess_data(self, 
                               fill_missing_features_with: str = 'zero',  # 'zero', 'nan', 'forward_fill', or 'interpolate'
                               standardize_features: bool = True) -> pd.DataFrame:
        """
        Loads and combines data for all stocks into a single DataFrame.
        Handles missing features gracefully instead of dropping stocks.
        
        Args:
            fill_missing_features_with: How to handle missing features
                - 'zero': Fill with zeros
                - 'nan': Fill with NaNs (will be handled later)
                - 'forward_fill': Forward fill from available data
                - 'interpolate': Linear interpolation
            standardize_features: Whether to ensure all stocks have the same feature set
        """
        all_data = []
        all_features = set()  # Track all unique features across all stocks
        stock_features = {}   # Track which features each stock has
        
        print(f"Loading data for stocks: {self.stocks}")

        # First pass: collect all available features across all stocks
        for stock_symbol in self.stocks:
            file_path = self.data_root / f"{stock_symbol}_aligned.csv"
            if not file_path.exists():
                print(f"Warning: Data file for {stock_symbol} not found at {file_path}. Skipping.")
                continue
            
            try:
                # Just read headers to get feature list
                sample_df = pd.read_csv(file_path, nrows=0)
                date_col = sample_df.columns[0]
                stock_features_list = [f for f in sample_df.columns if f != date_col]
                stock_features[stock_symbol] = stock_features_list
                all_features.update(stock_features_list)
                
            except Exception as e:
                print(f"Error reading headers for {stock_symbol}: {e}")
                continue
        
        # Determine the standardized feature set
        if standardize_features:
            # Use the intersection of all features (features present in ALL stocks)
            common_features = set.intersection(*[set(features) for features in stock_features.values()])
            if not common_features:
                print("Warning: No common features found across all stocks. Using union of all features.")
                self.features = sorted(list(all_features))
            else:
                self.features = sorted(list(common_features))
                print(f"Using common features across all stocks: {self.features}")
        else:
            # Use all features found across any stock
            self.features = sorted(list(all_features))
            print(f"Using all features found across stocks: {self.features}")
        
        self.num_features = len(self.features)

        # Second pass: load actual data with standardized feature handling
        for stock_symbol in self.stocks:
            file_path = self.data_root / f"{stock_symbol}_aligned.csv"
            if not file_path.exists():
                continue
                
            try:
                df = pd.read_csv(file_path, parse_dates=[0])
                date_col = df.columns[0]
                
                # Set date as index first
                df.set_index(date_col, inplace=True)
                df.index.name = 'date'
                
                # Ensure datetime index
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)
                
                # Filter by date range
                df = df.loc[self.start_date_str:self.end_date_str]
                
                if df.empty:
                    print(f"Warning: No data for {stock_symbol} in date range. Creating empty DataFrame with correct structure.")
                    # Create empty DataFrame with correct date range and features
                    date_range = pd.date_range(start=self.start_date_str, end=self.end_date_str, freq='D')
                    df = pd.DataFrame(index=date_range, columns=self.features)
                    df.index.name = 'date'
                
                # Handle missing and extra features
                processed_df = pd.DataFrame(index=df.index)
                
                for feature in self.features:
                    if feature in df.columns:
                        # Feature exists, convert to numeric
                        feature_data = pd.to_numeric(df[feature], errors='coerce')
                        processed_df[feature] = feature_data
                    else:
                        # Feature missing, handle according to strategy
                        print(f"Feature '{feature}' missing for {stock_symbol}, filling with {fill_missing_features_with}")
                        
                        if fill_missing_features_with == 'zero':
                            processed_df[feature] = 0.0
                        elif fill_missing_features_with == 'nan':
                            processed_df[feature] = np.nan
                        elif fill_missing_features_with == 'forward_fill':
                            # Create series with NaN and forward fill
                            processed_df[feature] = np.nan
                        elif fill_missing_features_with == 'interpolate':
                            # Create series with NaN for later interpolation
                            processed_df[feature] = np.nan
                        else:
                            processed_df[feature] = 0.0
                
                # Apply additional filling strategies if needed
                if fill_missing_features_with == 'forward_fill':
                    processed_df = processed_df.ffill()
                    processed_df = processed_df.fillna(0)  # Fill remaining NaNs with 0
                elif fill_missing_features_with == 'interpolate':
                    processed_df = processed_df.interpolate(method='linear')
                    processed_df = processed_df.fillna(method='bfill')  # Backward fill for start
                    processed_df = processed_df.fillna(0)  # Fill any remaining NaNs
                
                # Add symbol column
                processed_df['symbol'] = stock_symbol
                all_data.append(processed_df)
                
                print(f"Loaded {stock_symbol}: {len(processed_df)} records, {len([f for f in self.features if f in df.columns])}/{len(self.features)} features present")
                
            except Exception as e:
                print(f"Error loading data for {stock_symbol}: {e}")
                # Create empty DataFrame for this stock to maintain consistency
                print(f"Creating empty DataFrame for {stock_symbol} to maintain structure")
                date_range = pd.date_range(start=self.start_date_str, end=self.end_date_str, freq='D')
                empty_df = pd.DataFrame(index=date_range, columns=self.features)
                empty_df.index.name = 'date'
                
                if fill_missing_features_with == 'zero':
                    empty_df = empty_df.fillna(0.0)
                else:
                    empty_df = empty_df.fillna(np.nan)
                
                empty_df['symbol'] = stock_symbol
                all_data.append(empty_df)
                continue
        
        if not all_data:
            raise FileNotFoundError("No valid stock data found. Please check the data files.")
        
        # Combine all data
        panel_data = pd.concat(all_data, axis=0)
        
        # Reset index to make 'date' a column, then set multi-index
        panel_data = panel_data.reset_index()
        
        # Remove any duplicate date-symbol combinations (keep first)
        panel_data = panel_data.drop_duplicates(subset=['date', 'symbol'], keep='first')
        
        # Set multi-index
        panel_data = panel_data.set_index(['date', 'symbol'])
        
        # Convert to wide format (stocks as columns)
        panel_data = panel_data.unstack(level='symbol')
        
        # Handle MultiIndex columns
        if isinstance(panel_data.columns, pd.MultiIndex):
            panel_data = panel_data.swaplevel(0, 1, axis=1).sort_index(axis=1)
        
        # Handle NaN values based on strategy
        if fill_missing_features_with in ['nan', 'forward_fill', 'interpolate']:
            # Forward fill temporal gaps
            panel_data = panel_data.ffill()
            
            # Handle remaining NaN values
            nan_count_before = panel_data.isna().sum().sum()
            if nan_count_before > 0:
                print(f"Handling {nan_count_before} remaining NaN values...")
                
                # Backward fill
                panel_data = panel_data.bfill()
                
                # Fill any remaining NaNs with 0
                panel_data = panel_data.fillna(0)
                
                nan_count_after = panel_data.isna().sum().sum()
                print(f"NaN values after processing: {nan_count_after}")
        
        # Update stocks_features for MultiIndex compatibility
        if isinstance(panel_data.columns, pd.MultiIndex):
            self.stocks_features = panel_data.columns.tolist()
            # Ensure stocks list matches what we actually have
            available_stocks = panel_data.columns.get_level_values(0).unique().tolist()
            self.stocks = [stock for stock in self.stocks if stock in available_stocks]
        else:
            self.stocks_features = panel_data.columns.tolist()
        
        print(f"Successfully processed data for {len(self.stocks)} stocks from {self.start_date_str} to {self.end_date_str}.")
        print(f"Final data shape: {panel_data.shape}")
        print(f"Features per stock: {self.num_features}")
        print(f"Available stocks: {self.stocks}")
        
        # Optional: Add data quality report
        self._print_data_quality_report(panel_data)
        
        return panel_data

    def _print_data_quality_report(self, data):
        """Print a summary of data quality issues."""
        print("\n=== Data Quality Report ===")
        
        if isinstance(data.columns, pd.MultiIndex):
            for stock in self.stocks:
                stock_data = data.xs(stock, level=0, axis=1)
                total_values = len(stock_data) * len(self.features)
                zero_values = (stock_data == 0).sum().sum()
                nan_values = stock_data.isna().sum().sum()
                
                if zero_values > 0 or nan_values > 0:
                    print(f"{stock}: {zero_values}/{total_values} zeros, {nan_values}/{total_values} NaNs")
        
        print("=== End Report ===\n")

    def _get_observation(self):
        """Constructs the observation for the current step."""
        # Calculate the actual observation size
        n_stocks = len(self.stocks)
        n_features = self.num_features
        # Each stock contributes: (features * window_size) + holdings + current_price
        obs_size = n_stocks * (n_features * self.window_size + 2)
        
        obs = np.zeros(obs_size, dtype=np.float32)
        
        # Get historical market data
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        market_window = self.data.iloc[start_idx:end_idx]
        
        # Get portfolio state
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        total_value = portfolio_state['total_value']
        
        # Process each stock
        obs_idx = 0
        
        for stock_idx, symbol in enumerate(self.stocks):
            # --- Historical market data for ALL features of this stock ---
            for feature_idx, feature in enumerate(self.features):
                if isinstance(self.data.columns, pd.MultiIndex):
                    # Handle MultiIndex columns: (symbol, feature)
                    if (symbol, feature) in self.data.columns:
                        feature_data = market_window[(symbol, feature)].values
                    else:
                        print(f"Warning: Column ({symbol}, {feature}) not found in data")
                        feature_data = np.zeros(len(market_window))
                else:
                    # Handle regular columns (fallback)
                    col_name = f"{symbol}_{feature}" if f"{symbol}_{feature}" in self.data.columns else feature
                    if col_name in self.data.columns:
                        feature_data = market_window[col_name].values
                    else:
                        print(f"Warning: Column {col_name} not found in data")
                        feature_data = np.zeros(len(market_window))
                
                # Pad with zeros if window is smaller than window_size (at the beginning)
                padded_data = np.zeros(self.window_size)
                data_len = min(len(feature_data), self.window_size)
                padded_data[-data_len:] = feature_data[-data_len:]
                
                # Add to observation
                obs[obs_idx:obs_idx + self.window_size] = padded_data
                obs_idx += self.window_size
            
            # --- Current holdings (as percentage of total value) ---
            if symbol in portfolio_state['positions']:
                pos = portfolio_state['positions'][symbol]
                holding_ratio = pos['market_value'] / total_value if total_value > 0 else 0
            else:
                holding_ratio = 0
            
            obs[obs_idx] = holding_ratio
            obs_idx += 1
            
            # --- Current price (use the first feature as price, typically 'close' or 'price') ---
            current_price = 0
            if len(market_window) > 0:
                # Use the first feature as the main price indicator
                main_feature = self.features[0]  # Assuming first feature is price-related
                
                if isinstance(self.data.columns, pd.MultiIndex):
                    if (symbol, main_feature) in self.data.columns:
                        current_price = market_window[(symbol, main_feature)].iloc[-1]
                else:
                    col_name = f"{symbol}_{main_feature}" if f"{symbol}_{main_feature}" in self.data.columns else main_feature
                    if col_name in self.data.columns:
                        current_price = market_window[col_name].iloc[-1]
            
            obs[obs_idx] = current_price
            obs_idx += 1
        
        # Debug information
        expected_size = n_stocks * (n_features * self.window_size + 2)
        if obs_idx != expected_size:
            print(f"Warning: Observation size mismatch. Expected: {expected_size}, Got: {obs_idx}")
            print(f"Stocks: {n_stocks}, Features: {n_features}, Window: {self.window_size}")
        
        return obs
    
    def _update_observation_space_size(self):
        """
        Helper method to update the observation space size after data loading.
        Call this after _load_and_preprocess_data() in __init__.
        """
        n_stocks = len(self.stocks)

        if self.use_all_features:
            # If you want to use all features for all stocks
            obs_space_size = n_stocks * self.num_features * self.window_size + n_stocks * 2
        else:
            # If you want to use only one main feature (like price) per stock
            obs_space_size = n_stocks * (self.window_size + 2)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_space_size,), 
            dtype=np.float32
        )

        print(f"Observation space updated: {self.observation_space.shape} (size: {obs_space_size})")

    def _take_action(self, action: np.ndarray):
        """Translates target weights from the agent into buy/sell orders."""
        target_weights = action / np.sum(action) if np.sum(action) > 0 else np.zeros(len(self.stocks))
        
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        current_weights = np.array([portfolio_state['position_weights'].get(s, 0) for s in self.stocks])
        total_value = portfolio_state['total_value']
        
        weight_diff = target_weights - current_weights
        
        current_date = self.dates[self.current_step]
        current_prices = self._get_current_prices(current_date)

        # First, process sell orders to free up cash
        for i, symbol in enumerate(self.stocks):
            if weight_diff[i] < 0:
                current_market_value = current_weights[i] * total_value
                target_market_value = target_weights[i] * total_value
                amount_to_sell = current_market_value - target_market_value
                
                if symbol in current_prices:
                    price = current_prices[symbol]
                    quantity_to_sell = amount_to_sell / price
                    self.portfolio_manager.execute_sell(symbol, quantity_to_sell, price, timestamp=current_date)

        # Then, process buy orders with the available cash
        for i, symbol in enumerate(self.stocks):
            if weight_diff[i] > 0:
                # Re-check total value as it might have changed slightly due to fees
                current_total_value = self.portfolio_manager.get_total_portfolio_value()
                target_market_value = target_weights[i] * current_total_value
                
                if symbol in current_prices:
                    price = current_prices[symbol]
                    quantity_to_buy = target_market_value / price
                    self.portfolio_manager.execute_buy(symbol, quantity_to_buy, price, timestamp=current_date)

    def _get_current_prices(self, date):
        """Helper method to extract current prices for all stocks at a given date."""
        current_data = self.data.loc[date]
        prices = {}
        
        # Use the first feature as the main price (typically 'close', 'price', etc.)
        main_feature = self.features[0]
        
        for symbol in self.stocks:
            if isinstance(self.data.columns, pd.MultiIndex):
                # Handle MultiIndex columns
                if (symbol, main_feature) in current_data.index:
                    prices[symbol] = current_data[(symbol, main_feature)]
                else:
                    print(f"Warning: Price for {symbol} not found at {date}")
            else:
                # Handle regular columns (fallback)
                col_name = f"{symbol}_{main_feature}" if f"{symbol}_{main_feature}" in current_data.index else main_feature
                if col_name in current_data.index:
                    prices[symbol] = current_data[col_name]
                else:
                    print(f"Warning: Price for {symbol} not found at {date}")
        
        return prices

    def step(self, action: np.ndarray):
        """
        Executes one time step within the environment.
        """
        # Get portfolio value before taking action
        prev_portfolio_value = self.portfolio_manager.get_total_portfolio_value()

        # Execute the action (rebalance portfolio based on target weights)
        self._take_action(action)
        
        # Move to the next day
        self.current_step += 1

        # Update market prices in the portfolio manager
        current_date = self.dates[self.current_step]
        latest_prices = self.data.loc[current_date].to_dict()
        self.portfolio_manager.update_market_prices(latest_prices, timestamp=current_date)
        
        # Get the new portfolio value
        current_portfolio_value = self.portfolio_manager.get_total_portfolio_value()

        # Calculate the reward
        reward = self.portfolio_manager.get_sharpe_ratio()
        
        # Check if the simulation is done
        if self.current_step >= len(self.dates) - 1:
            self.done = True

        # Get the next observation
        obs = self._get_observation()

        # `truncated` is for episodes that end due to a time limit, `terminated` for a terminal state.
        # Here, we only have a time limit.
        terminated = self.done
        truncated = False
        
        # Additional info (optional, but good for debugging)
        info = {
            'date': current_date,
            'portfolio_value': current_portfolio_value,
            'total_return': self.portfolio_manager.get_total_return(),
            'sharpe_ratio': reward
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        self.portfolio_manager.reset_portfolio()
        self.current_step = self.window_size # Start with enough data for the first observation
        self.done = False

        # Update portfolio with initial prices
        initial_date = self.dates[self.current_step]
        initial_prices = self.data.loc[initial_date].to_dict()
        self.portfolio_manager.update_market_prices(initial_prices, timestamp=initial_date)
        
        obs = self._get_observation()
        info = {'date': initial_date, 'portfolio_value': self.portfolio_manager.initial_cash}

        return obs, info

    def render(self, mode='human'):
        """Renders the environment's state."""
        if mode == 'human':
            portfolio_state = self.portfolio_manager.get_portfolio_state()
            print(f"Date: {self.dates[self.current_step]}")
            print(f"Portfolio Value: {portfolio_state['total_value']:.2f}")
            print(f"Total Return: {portfolio_state['total_return']:.2f}%")
            print(f"Sharpe Ratio: {portfolio_state['sharpe_ratio']:.4f}")
            print("Positions:")
            summary = self.portfolio_manager.get_position_summary()
            if not summary.empty:
                print(summary)
            else:
                print("No open positions.")
            print("-" * 30)