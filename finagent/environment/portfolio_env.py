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
                 end_date: str = '2023-01-01'):
        
        super(PortfolioEnv, self).__init__()

        self.data_root = Path(data_root)
        self.window_size = window_size
        self.start_date_str = start_date
        self.end_date_str = end_date
        
        # Load and preprocess data
        self.stocks = stocks if stocks else self._load_stock_list()
        self.data = self._load_and_preprocess_data()
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
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.stocks), self.window_size + 2), 
            dtype=np.float32
        )

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

    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Loads and combines data for all stocks into a single DataFrame."""
        all_data = []
        print(f"Loading data for stocks: {self.stocks}")

        for stock_symbol in self.stocks:
            file_path = self.data_root / f"{stock_symbol}_aligned.csv"
            if not file_path.exists():
                print(f"Warning: Data file for {stock_symbol} not found at {file_path}. Skipping.")
                continue
            try:
                df = pd.read_csv(file_path, parse_dates=[0], index_col=0)
                if df.empty:
                    print(f"Warning: Data for {stock_symbol} is empty. Skipping.")
                    continue
                df['symbol'] = stock_symbol
                all_data.append(df)
            except Exception as e:
                print(f"Error loading data for {stock_symbol}: {e}")
                continue
        
        if not all_data:
            raise FileNotFoundError("No valid stock data found. Please check the data files.")
        
        panel_data = pd.concat(all_data, axis=0)
        panel_data.index.rename("date", inplace=True)
        panel_data = panel_data.pivot_table(index='date', columns='symbol', values='price')
        panel_data = panel_data.loc[self.start_date_str:self.end_date_str]
        panel_data.ffill(inplace=True) # Forward fill missing values
        panel_data.dropna(axis=1, how='any', inplace=True) # Drop stocks with missing data at start
        self.stocks = panel_data.columns.tolist() # Update stock list based on available data
        print(f"Loaded data for {len(self.stocks)} stocks from {self.start_date_str} to {self.end_date_str}.")
        return panel_data

    def _get_observation(self):
        """Constructs the observation for the current step."""
        obs = np.zeros((len(self.stocks), self.window_size + 2), dtype=np.float32)
        
        # Get historical market data
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        market_window = self.data.iloc[start_idx:end_idx].values.T
        
        # Pad with zeros if window is smaller than window_size (at the beginning)
        obs[:, -market_window.shape[1]:] = market_window

        # Get portfolio state
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        total_value = portfolio_state['total_value']
        
        # Add current holdings (as a percentage of total value) and current prices
        for i, symbol in enumerate(self.stocks):
            if symbol in portfolio_state['positions']:
                pos = portfolio_state['positions'][symbol]
                obs[i, 0] = pos['market_value'] / total_value if total_value > 0 else 0
            else:
                obs[i, 0] = 0 # No position in this stock
            
            # Add the most recent price as a feature
            obs[i, 1] = market_window[i, -1] if market_window.shape[1] > 0 else 0

        return obs

    def _take_action(self, action: np.ndarray):
        """Translates target weights from the agent into buy/sell orders."""
        target_weights = action / np.sum(action) if np.sum(action) > 0 else np.zeros(len(self.stocks))
        
        portfolio_state = self.portfolio_manager.get_portfolio_state()
        current_weights = np.array([portfolio_state['position_weights'].get(s, 0) for s in self.stocks])
        total_value = portfolio_state['total_value']
        
        weight_diff = target_weights - current_weights
        
        current_date = self.dates[self.current_step]
        current_prices = self.data.loc[current_date]

        # First, process sell orders to free up cash
        for i, symbol in enumerate(self.stocks):
            if weight_diff[i] < 0:
                current_market_value = current_weights[i] * total_value
                target_market_value = target_weights[i] * total_value
                amount_to_sell = current_market_value - target_market_value
                
                price = current_prices[symbol]
                quantity_to_sell = amount_to_sell / price
                
                self.portfolio_manager.execute_sell(symbol, quantity_to_sell, price, timestamp=current_date)

        # Then, process buy orders with the available cash
        for i, symbol in enumerate(self.stocks):
            if weight_diff[i] > 0:
                # Re-check total value as it might have changed slightly due to fees
                current_total_value = self.portfolio_manager.get_total_portfolio_value()
                target_market_value = target_weights[i] * current_total_value
                
                price = current_prices[symbol]
                quantity_to_buy = target_market_value / price

                self.portfolio_manager.execute_buy(symbol, quantity_to_buy, price, timestamp=current_date)

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