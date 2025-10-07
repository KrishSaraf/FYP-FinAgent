"""
45-stock portfolio environment with shorting capabilities.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from ..utils.weights_projection import project_weights


class Portfolio45ShortEnv(gym.Env):
    """
    45-stock portfolio environment with shorting capabilities.
    
    Features:
    - Continuous action space for portfolio weights
    - Shorting allowed with borrowing costs
    - Realistic transaction costs (commissions + slippage)
    - Daily rebalancing at close prices
    - Comprehensive cost model
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        initial_capital: float = 1_000_000.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 2.0,
        borrow_rate_annual: float = 0.03,
        rebate_rate_annual: float = 0.00,
        w_max: float = 0.10,
        gross_cap: float = 1.5,
        target_net: float = 1.0,
        lookback_window: int = 20,
        feature_columns: Optional[List[str]] = None,
        random_start: bool = False,
    ):
        """
        Initialize the portfolio environment.
        
        Args:
            data: DataFrame with columns ['date', 'ticker', ...features...]
            tickers: List of 45 stock tickers
            initial_capital: Initial portfolio value
            commission_bps: Commission in basis points
            slippage_bps: Slippage in basis points
            borrow_rate_annual: Annual borrowing rate for shorts
            rebate_rate_annual: Annual rebate rate for cash
            w_max: Maximum absolute weight per stock
            gross_cap: Maximum gross exposure
            target_net: Target net exposure
            lookback_window: Number of days for feature lookback
            feature_columns: List of feature columns to use
            random_start: Whether to start at random date
        """
        super().__init__()
        
        # Validate inputs
        assert len(tickers) == 45, f"Expected 45 tickers, got {len(tickers)}"
        assert initial_capital > 0, "Initial capital must be positive"
        
        # Store parameters
        self.data = data.copy()
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps / 10000  # Convert to decimal
        self.slippage_bps = slippage_bps / 10000
        self.borrow_rate_daily = borrow_rate_annual / 252
        self.rebate_rate_daily = rebate_rate_annual / 252
        self.w_max = w_max
        self.gross_cap = gross_cap
        self.target_net = target_net
        self.lookback_window = lookback_window
        self.random_start = random_start
        
        # Define feature columns (exclude static columns)
        if feature_columns is None:
            static_columns = ['date', 'ticker', 'period_end_date']
            self.feature_columns = [col for col in data.columns if col not in static_columns]
        else:
            self.feature_columns = feature_columns
        
        # Prepare data
        self._prepare_data()
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(tickers),), dtype=np.float32
        )
        
        # Observation space: features for all tickers at current time
        n_features = len(self.feature_columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(tickers), n_features), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _prepare_data(self):
        """Prepare data for environment."""
        # Pivot data to have tickers as columns
        self.data_pivot = self.data.pivot_table(
            index='date', columns='ticker', values=self.feature_columns
        )
        
        # Get unique dates and ensure they're sorted
        self.dates = sorted(self.data['date'].unique())
        self.n_days = len(self.dates)
        
        # Validate data completeness
        for ticker in self.tickers:
            if ticker not in self.data_pivot.columns.levels[1]:
                raise ValueError(f"Ticker {ticker} not found in data")
        
        # Create price arrays for easy access
        self.close_prices = np.zeros((self.n_days, len(self.tickers)))
        self.open_prices = np.zeros((self.n_days, len(self.tickers)))
        
        for i, ticker in enumerate(self.tickers):
            self.close_prices[:, i] = self.data_pivot[('close', ticker)].values
            self.open_prices[:, i] = self.data_pivot[('open', ticker)].values
        
        # Calculate daily returns
        self.daily_returns = np.diff(self.close_prices, axis=0) / self.close_prices[:-1]
        
        # Set start and end indices
        self.start_idx = self.lookback_window
        self.end_idx = self.n_days - 1
        
        if self.start_idx >= self.end_idx:
            raise ValueError("Not enough data for lookback window")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Set current day
        if self.random_start:
            self.current_day = self.np_random.integers(self.start_idx, self.end_idx)
        else:
            self.current_day = self.start_idx
        
        # Initialize portfolio state
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.shares = np.zeros(len(self.tickers))
        
        # Equal-weight initial portfolio (long-only)
        initial_weights = np.ones(len(self.tickers)) / len(self.tickers)
        initial_prices = self.close_prices[self.current_day - 1]
        
        # Calculate initial shares
        for i, (ticker, price) in enumerate(zip(self.tickers, initial_prices)):
            if price > 0:
                target_value = initial_weights[i] * self.initial_capital
                self.shares[i] = target_value / price
                self.cash -= target_value
        
        # Initialize tracking variables
        self.trades_log = []
        self.portfolio_values = [self.portfolio_value]
        self.weights_history = [initial_weights.copy()]
        self.returns_history = []
        self.costs_history = []
        
        # Calculate initial portfolio value
        self._update_portfolio_value()
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Project action to valid weights
        target_weights = project_weights(
            action, self.w_max, self.target_net, self.gross_cap
        )
        
        # Get current prices
        current_prices = self.close_prices[self.current_day]
        
        # Execute rebalancing
        trades, costs = self._rebalance(target_weights, current_prices)
        
        # Update day
        self.current_day += 1
        
        # Check if episode is done
        done = self.current_day >= self.end_idx
        
        # Update portfolio value with new prices
        if not done:
            new_prices = self.close_prices[self.current_day]
            self._update_portfolio_value(new_prices)
        
        # Calculate reward (daily return)
        if len(self.portfolio_values) > 1:
            daily_return = (self.portfolio_values[-1] - self.portfolio_values[-2]) / self.portfolio_values[-2]
        else:
            daily_return = 0.0
        
        self.returns_history.append(daily_return)
        
        # Get next observation
        obs = self._get_observation()
        
        # Prepare info
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'daily_return': daily_return,
            'costs': costs,
            'trades': trades,
            'target_weights': target_weights,
            'actual_weights': self._get_current_weights(),
            'gross_exposure': np.sum(np.abs(self._get_current_weights())),
            'net_exposure': np.sum(self._get_current_weights()),
        }
        
        return obs, daily_return, done, False, info
    
    def _rebalance(self, target_weights: np.ndarray, prices: np.ndarray) -> Tuple[List[Dict], Dict]:
        """Execute portfolio rebalancing."""
        current_weights = self._get_current_weights()
        target_values = target_weights * self.portfolio_value
        
        trades = []
        total_commission = 0.0
        total_slippage = 0.0
        total_borrow_fee = 0.0
        
        for i, ticker in enumerate(self.tickers):
            if prices[i] <= 0:
                continue
            
            current_value = self.shares[i] * prices[i]
            target_value = target_values[i]
            trade_value = target_value - current_value
            
            if abs(trade_value) < 1.0:  # Ignore tiny trades
                continue
            
            # Calculate trade quantity
            if trade_value > 0:  # Buy
                # Apply slippage (buy at higher price)
                exec_price = prices[i] * (1 + self.slippage_bps)
                quantity = trade_value / exec_price
                side = 'buy'
            else:  # Sell/Short
                # Apply slippage (sell at lower price)
                exec_price = prices[i] * (1 - self.slippage_bps)
                quantity = trade_value / exec_price
                side = 'sell'
            
            # Calculate costs
            notional = abs(quantity * exec_price)
            commission = notional * self.commission_bps
            
            # Update shares and cash
            self.shares[i] += quantity
            self.cash -= quantity * exec_price - commission
            
            # Calculate borrow fee for shorts
            borrow_fee = 0.0
            if self.shares[i] < 0:  # Short position
                short_value = abs(self.shares[i] * exec_price)
                borrow_fee = short_value * self.borrow_rate_daily
            
            # Record trade
            trade = {
                'date': self.dates[self.current_day],
                'ticker': ticker,
                'quantity': quantity,
                'exec_price': exec_price,
                'side': side,
                'notional': notional,
                'commission': commission,
                'borrow_fee': borrow_fee,
            }
            trades.append(trade)
            
            total_commission += commission
            total_slippage += abs(trade_value) * self.slippage_bps
            total_borrow_fee += borrow_fee
        
        # Apply cash rebate
        if self.cash > 0:
            cash_rebate = self.cash * self.rebate_rate_daily
            self.cash += cash_rebate
        else:
            cash_rebate = 0.0
        
        # Store trades
        self.trades_log.extend(trades)
        
        # Calculate total costs
        costs = {
            'commission': total_commission,
            'slippage': total_slippage,
            'borrow_fee': total_borrow_fee,
            'cash_rebate': cash_rebate,
            'total': total_commission + total_slippage + total_borrow_fee - cash_rebate,
        }
        
        self.costs_history.append(costs)
        
        return trades, costs
    
    def _update_portfolio_value(self, prices: Optional[np.ndarray] = None):
        """Update portfolio value with current prices."""
        if prices is None:
            prices = self.close_prices[self.current_day]
        
        # Calculate portfolio value
        stock_value = np.sum(self.shares * prices)
        self.portfolio_value = self.cash + stock_value
        
        # Store portfolio value and weights
        self.portfolio_values.append(self.portfolio_value)
        self.weights_history.append(self._get_current_weights())
    
    def _get_current_weights(self) -> np.ndarray:
        """Get current portfolio weights."""
        if self.portfolio_value <= 0:
            return np.zeros(len(self.tickers))
        
        current_prices = self.close_prices[self.current_day]
        stock_values = self.shares * current_prices
        weights = stock_values / self.portfolio_value
        
        return weights
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_day < self.lookback_window:
            # Use available data
            start_idx = 0
            end_idx = self.current_day + 1
        else:
            start_idx = self.current_day - self.lookback_window + 1
            end_idx = self.current_day + 1
        
        # Extract features for all tickers
        obs = np.zeros((len(self.tickers), len(self.feature_columns)))
        
        for i, ticker in enumerate(self.tickers):
            for j, feature in enumerate(self.feature_columns):
                try:
                    values = self.data_pivot[(feature, ticker)].iloc[start_idx:end_idx].values
                    # Use the most recent value
                    obs[i, j] = values[-1] if len(values) > 0 else 0.0
                except (KeyError, IndexError):
                    obs[i, j] = 0.0
        
        return obs.astype(np.float32)
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics."""
        if len(self.portfolio_values) < 2:
            return {}
        
        returns = np.array(self.returns_history)
        portfolio_values = np.array(self.portfolio_values[1:])  # Exclude initial value
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annualized_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
        
        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Trading metrics
        total_trades = len(self.trades_log)
        total_commission = sum(cost['commission'] for cost in self.costs_history)
        total_borrow_fee = sum(cost['borrow_fee'] for cost in self.costs_history)
        
        # Exposure metrics
        weights_array = np.array(self.weights_history)
        avg_gross_exposure = np.mean([np.sum(np.abs(w)) for w in weights_array])
        avg_net_exposure = np.mean([np.sum(w) for w in weights_array])
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'total_commission': total_commission,
            'total_borrow_fee': total_borrow_fee,
            'avg_gross_exposure': avg_gross_exposure,
            'avg_net_exposure': avg_net_exposure,
            'final_portfolio_value': portfolio_values[-1],
        }
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades_log)
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Day: {self.current_day}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Current Weights: {self._get_current_weights()}")
            print(f"Gross Exposure: {np.sum(np.abs(self._get_current_weights())):.3f}")
            print(f"Net Exposure: {np.sum(self._get_current_weights()):.3f}")
    
    def close(self):
        """Close the environment."""
        pass
