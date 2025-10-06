"""
FinRL-inspired Environment for Indian Stock Market Trading
Adapted from FinRL framework for Nifty 50 stocks
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class IndianStockTradingEnv(gym.Env):
    """
    Custom Gym environment for Indian stock trading inspired by FinRL
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int = 100,
        initial_amount: float = 1000000,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        reward_scaling: float = 1e-4,
        state_space: int = 30,
        action_space: int = 30,
        tech_indicator_list: List[str] = None,
        turbulence_threshold: float = 140,
        make_plots: bool = False
    ):
        """
        Initialize the trading environment
        
        Args:
            df: DataFrame with stock data
            stock_dim: Number of stocks
            hmax: Maximum number of shares to buy/sell
            initial_amount: Initial capital
            buy_cost_pct: Transaction cost for buying
            sell_cost_pct: Transaction cost for selling
            reward_scaling: Scaling factor for rewards
            state_space: Dimension of state space
            action_space: Dimension of action space
            tech_indicator_list: List of technical indicators to use
            turbulence_threshold: Threshold for turbulence-based risk management
        """
        super(IndianStockTradingEnv, self).__init__()
        
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list or []
        self.turbulence_threshold = turbulence_threshold
        self.make_plots = make_plots
        
        # Get unique dates and stocks
        self.dates = self.df['date'].unique()
        self.stock_list = self.df['tic'].unique() if 'tic' in self.df.columns else [f'stock_{i}' for i in range(stock_dim)]
        
        # Initialize state
        self.state = None
        self.terminal = False
        self.day = 0
        self.data = self.df.copy()
        
        # Portfolio state
        self.asset_memory = [self.initial_amount]
        self.reward_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self.dates[0]]
        
        # Action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_space,)
        )
        
    def _get_observation(self):
        """Get current observation/state with proper dimension handling"""
        if self.day == 0:
            # First day - use initial state
            state = np.zeros(self.state_space)
            state[:self.stock_dim] = 0  # No holdings initially
            state[self.stock_dim:2*self.stock_dim] = 1  # Equal weights
            return state
        
        # Get current day data
        current_data = self.data[self.data['date'] == self.dates[self.day]]
        
        if len(current_data) == 0:
            return np.zeros(self.state_space)
        
        # Normalize prices
        prices = current_data['close'].values
        prices_normalized = prices / prices[0] if len(prices) > 0 else np.ones(self.stock_dim)
        
        # Get technical indicators if available
        tech_indicators = []
        for indicator in self.tech_indicator_list:
            if indicator in current_data.columns:
                values = current_data[indicator].values
                # Take values for all stocks, not just first 5
                tech_indicators.extend(values[:self.stock_dim])
        
        # Calculate how much space we have for tech indicators
        price_holdings_space = 2 * self.stock_dim
        available_tech_space = self.state_space - price_holdings_space
        
        # Truncate or pad tech indicators to fit available space
        if len(tech_indicators) > available_tech_space:
            tech_indicators = tech_indicators[:available_tech_space]
        else:
            tech_indicators.extend([0] * (available_tech_space - len(tech_indicators)))
        
        # Combine state components
        state = np.concatenate([
            prices_normalized[:self.stock_dim],  # Ensure correct size
            np.ones(self.stock_dim),  # Placeholder for holdings
            tech_indicators
        ])
        
        # Ensure state has exactly the right dimension
        if len(state) < self.state_space:
            # Pad with zeros if too short
            state = np.pad(state, (0, self.state_space - len(state)), 'constant')
        elif len(state) > self.state_space:
            # Truncate if too long
            state = state[:self.state_space]
        
        return state
    
    def _get_reward(self, actions):
        """Calculate reward based on actions and portfolio performance"""
        # Simple reward based on portfolio value change
        if self.day == 0:
            return 0
        
        # Get current and previous portfolio values
        current_value = self._get_portfolio_value()
        previous_value = self.asset_memory[-1] if len(self.asset_memory) > 1 else self.initial_amount
        
        # Calculate return
        portfolio_return = (current_value - previous_value) / previous_value
        
        # Add transaction cost penalty
        transaction_cost = np.sum(np.abs(actions)) * 0.001
        
        # Reward is return minus transaction costs
        reward = portfolio_return - transaction_cost
        
        return reward * self.reward_scaling
    
    def _get_portfolio_value(self):
        """Calculate current portfolio value with proper PnL logic and equal allocation"""
        if self.day == 0:
            return self.initial_amount
        
        current_data = self.data[self.data['date'] == self.dates[self.day]]
        if len(current_data) == 0:
            return self.asset_memory[-1] if len(self.asset_memory) > 0 else self.initial_amount
        
        # Portfolio value calculation with equal allocation across all stocks
        if len(self.asset_memory) > 1:
            # Calculate return from previous day
            prev_data = self.data[self.data['date'] == self.dates[self.day-1]]
            if len(prev_data) > 0 and len(current_data) > 0:
                # Get prices for stocks that exist in both days
                prev_prices = prev_data['close'].values
                curr_prices = current_data['close'].values
                
                # Ensure same number of stocks in both days
                min_stocks = min(len(prev_prices), len(curr_prices))
                if min_stocks > 0:
                    prev_prices = prev_prices[:min_stocks]
                    curr_prices = curr_prices[:min_stocks]
                    
                    # Calculate returns for each stock
                    returns = (curr_prices - prev_prices) / prev_prices
                    
                    # Equal allocation: each stock gets 1/num_stocks of the portfolio
                    # Calculate weighted average return (equal weights)
                    equal_weight = 1.0 / self.stock_dim
                    weighted_return = np.sum(returns * equal_weight)
                    
                    # Apply return to previous portfolio value
                    portfolio_value = self.asset_memory[-1] * (1 + weighted_return)
                else:
                    portfolio_value = self.asset_memory[-1]
            else:
                portfolio_value = self.asset_memory[-1]
        else:
            portfolio_value = self.initial_amount
        
        return portfolio_value
    
    def step(self, actions):
        """Execute one step in the environment"""
        self.terminal = self.day >= len(self.dates) - 1
        
        if self.terminal:
            # Calculate final reward
            reward = self._get_reward(actions)
            self.reward_memory.append(reward)
            self.actions_memory.append(actions)
            self.state_memory.append(self.state)
            
            return self.state, reward, self.terminal, {}
        
        else:
            # Get current state
            self.state = self._get_observation()
            
            # Calculate reward
            reward = self._get_reward(actions)
            
            # Update memory
            self.reward_memory.append(reward)
            self.actions_memory.append(actions)
            self.state_memory.append(self.state)
            self.asset_memory.append(self._get_portfolio_value())
            self.date_memory.append(self.dates[self.day])
            
            # Move to next day
            self.day += 1
            
            return self.state, reward, self.terminal, {}
    
    def reset(self):
        """Reset environment to initial state"""
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.copy()
        self.state = self._get_observation()
        self.terminal = False
        self.reward_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self.dates[0]]
        
        return self.state
    
    def render(self, mode='human'):
        """Render the environment"""
        return self.state
    
    def _seed(self, seed=None):
        """Set random seed"""
        np.random.seed(seed)
        return [seed]
    
    def save_asset_memory(self):
        """Save asset memory to file"""
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({
            'date': date_list,
            'account_value': asset_list
        })
        return df_account_value
    
    def save_action_memory(self):
        """Save action memory to file"""
        if len(self.dates) != len(self.actions_memory):
            # Pad actions if needed
            actions_list = self.actions_memory + [np.zeros(self.action_space)] * (len(self.dates) - len(self.actions_memory))
        else:
            actions_list = self.actions_memory
        
        df_actions = pd.DataFrame({
            'date': self.dates[:len(actions_list)],
            'actions': actions_list
        })
        return df_actions


class PortfolioOptimizationEnv(IndianStockTradingEnv):
    """
    Extended environment for portfolio optimization with risk management
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.max_weight = 0.1  # Maximum 10% weight per stock
        
    def _get_reward(self, actions):
        """Enhanced reward function with risk-adjusted returns"""
        if self.day == 0:
            return 0
        
        # Get portfolio return
        current_value = self._get_portfolio_value()
        previous_value = self.asset_memory[-1] if len(self.asset_memory) > 1 else self.initial_amount
        portfolio_return = (current_value - previous_value) / previous_value
        
        # Calculate Sharpe ratio if we have enough history
        if len(self.asset_memory) > 10:
            recent_values = self.asset_memory[-10:]
            if len(recent_values) > 1:
                returns = np.diff(recent_values) / recent_values[:-1]
                if len(returns) > 0 and np.std(returns) > 1e-8:
                    sharpe_ratio = (np.mean(returns) - self.risk_free_rate/252) / np.std(returns)
                    reward = sharpe_ratio * 0.1  # Scale Sharpe ratio
                else:
                    reward = portfolio_return
            else:
                reward = portfolio_return
        else:
            reward = portfolio_return
        
        # Add diversification bonus
        if len(actions) > 0:
            diversification = 1 - np.std(actions)  # Reward for balanced allocation
            reward += diversification * 0.01
        
        return reward * self.reward_scaling


def create_env(
    data: pd.DataFrame,
    env_type: str = "trading",
    **kwargs
) -> gym.Env:
    """
    Factory function to create different types of environments
    
    Args:
        data: Stock data DataFrame
        env_type: Type of environment ("trading" or "portfolio")
        **kwargs: Additional arguments for environment initialization
    
    Returns:
        Configured gym environment
    """
    if env_type == "portfolio":
        return PortfolioOptimizationEnv(data, **kwargs)
    else:
        return IndianStockTradingEnv(data, **kwargs)
