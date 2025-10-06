"""
Custom FinRL Trading Environment for Single Stock Trading
Implements a sophisticated trading environment with rich state space
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class SingleStockTradingEnv(gym.Env):
    """
    Custom FinRL environment for single stock trading with rich feature set
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 stock_dim: int = 1,
                 hmax: int = 100,
                 initial_amount: float = 1000000.0,
                 buy_cost_pct: float = 0.001,
                 sell_cost_pct: float = 0.001,
                 reward_scaling: float = 1e-4,
                 state_space: int = 50,
                 action_space: int = 3,
                 tech_indicator_list: List[str] = None,
                 turbulence_threshold: float = None,
                 make_plots: bool = False):
        """
        Initialize the trading environment
        
        Args:
            df: DataFrame with stock data
            stock_dim: Number of stocks (1 for single stock)
            hmax: Maximum number of shares to buy/sell
            initial_amount: Initial capital
            buy_cost_pct: Transaction cost for buying
            sell_cost_pct: Transaction cost for selling
            reward_scaling: Scaling factor for rewards
            state_space: Dimension of state space
            action_space: Dimension of action space (3 for buy/hold/sell)
            tech_indicator_list: List of technical indicators
            turbulence_threshold: Threshold for turbulence
            make_plots: Whether to make plots
        """
        super(SingleStockTradingEnv, self).__init__()
        
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
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(action_space)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32
        )
        
        # Initialize environment state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state
        
        Returns:
            Initial observation
        """
        self.day = 0
        self.data = self.df.copy()
        self.data = self.data.reset_index(drop=True)
        
        # Initialize portfolio state
        self.state = np.array([
            self.initial_amount,  # cash
            0,  # shares held
            0,  # total assets
            0,  # total cost
            0,  # total shares
            0,  # total trades
            0,  # total buy trades
            0,  # total sell trades
            0,  # total buy amount
            0,  # total sell amount
        ])
        
        # Initialize performance tracking
        self.asset_memory = [self.initial_amount]
        self.reward_memory = []
        self.actions_memory = []
        self.date_memory = [self.data.iloc[0]['date'] if 'date' in self.data.columns else 0]
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def step(self, actions: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            actions: Action to take (0: hold, 1: buy, 2: sell)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current price
        current_price = self.data.iloc[self.day]['close']
        
        # Execute action
        reward = self._execute_action(actions, current_price)
        
        # Update day
        self.day += 1
        
        # Check if episode is done
        done = self.day >= len(self.data) - 1
        
        # Get next observation
        obs = self._get_observation()
        
        # Store information
        info = {
            'total_assets': self.state[2],
            'cash': self.state[0],
            'shares': self.state[1],
            'current_price': current_price,
            'action': actions,
            'reward': reward
        }
        
        return obs, reward, done, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """
        Execute the trading action
        
        Args:
            action: Action to execute
            current_price: Current stock price
            
        Returns:
            Reward for the action
        """
        # Calculate current portfolio value
        current_portfolio_value = self.state[0] + self.state[1] * current_price
        
        # Execute action based on action type
        if action == 1:  # Buy
            # Calculate number of shares to buy
            available_cash = self.state[0]
            shares_to_buy = min(
                self.hmax - self.state[1],
                int(available_cash * 0.1 / current_price)  # Use 10% of cash
            )
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.buy_cost_pct)
                if cost <= available_cash:
                    self.state[0] -= cost
                    self.state[1] += shares_to_buy
                    self.state[4] += shares_to_buy
                    self.state[5] += 1
                    self.state[6] += 1
                    self.state[8] += cost
        
        elif action == 2:  # Sell
            # Calculate number of shares to sell
            shares_to_sell = min(self.state[1], self.hmax)
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.sell_cost_pct)
                self.state[0] += proceeds
                self.state[1] -= shares_to_sell
                self.state[4] -= shares_to_sell
                self.state[5] += 1
                self.state[7] += 1
                self.state[9] += proceeds
        
        # Update portfolio state
        self.state[2] = self.state[0] + self.state[1] * current_price
        self.state[3] = self.state[8] - self.state[9]
        
        # Calculate reward
        reward = self._calculate_reward(current_portfolio_value, current_price)
        
        # Store memory
        self.asset_memory.append(self.state[2])
        self.reward_memory.append(reward)
        self.actions_memory.append(action)
        
        return reward
    
    def _calculate_reward(self, previous_portfolio_value: float, current_price: float) -> float:
        """
        Calculate reward for the current action
        
        Args:
            previous_portfolio_value: Previous portfolio value
            current_price: Current stock price
            
        Returns:
            Reward value
        """
        current_portfolio_value = self.state[0] + self.state[1] * current_price
        
        # Basic return-based reward
        portfolio_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        
        # Risk-adjusted reward (Sharpe ratio approximation)
        if len(self.asset_memory) > 1:
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
                risk_adjusted_reward = sharpe_ratio * 0.1
            else:
                risk_adjusted_reward = 0
        else:
            risk_adjusted_reward = 0
        
        # Transaction cost penalty
        transaction_penalty = -0.001 * self.state[5]  # Penalty for excessive trading
        
        # Combine rewards
        total_reward = portfolio_return + risk_adjusted_reward + transaction_penalty
        
        return total_reward * self.reward_scaling
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state)
        
        Returns:
            Current observation vector
        """
        if self.day >= len(self.data):
            return np.zeros(self.state_space, dtype=np.float32)
        
        # Get current row data
        current_data = self.data.iloc[self.day]
        
        # Portfolio state (10 features)
        portfolio_state = self.state.copy()
        
        # Price and volume features (6 features)
        price_features = [
            current_data['open'],
            current_data['high'], 
            current_data['low'],
            current_data['close'],
            current_data['volume'],
            current_data['vwap']
        ]
        
        # Technical indicators (5 features)
        technical_features = [
            current_data.get('dma_50', 0),
            current_data.get('dma_200', 0),
            current_data.get('rsi_14', 50),
            current_data.get('dma_cross', 0),
            current_data.get('dma_distance', 0)
        ]
        
        # Fundamental features (10 features)
        fundamental_features = [
            current_data.get('metric_pPerEExcludingExtraordinaryItemsMostRecentFiscalYear', 0),
            current_data.get('metric_priceToBookMostRecentFiscalYear', 0),
            current_data.get('metric_returnOnAverageEquityTrailing12Month', 0),
            current_data.get('metric_operatingMarginTrailing12Month', 0),
            current_data.get('metric_grossMarginTrailing12Month', 0),
            current_data.get('metric_currentRatioMostRecentFiscalYear', 0),
            current_data.get('metric_totalDebtPerTotalEquityMostRecentFiscalYear', 0),
            current_data.get('metric_revenueGrowthRate5Year', 0),
            current_data.get('metric_marketCap', 0),
            current_data.get('metric_beta', 1)
        ]
        
        # Sentiment features (5 features)
        sentiment_features = [
            current_data.get('reddit_title_sentiments_mean', 0),
            current_data.get('reddit_body_sentiments', 0),
            current_data.get('news_sentiment_mean', 0),
            current_data.get('reddit_posts_count', 0),
            current_data.get('news_articles_count', 0)
        ]
        
        # Lag features (7 features)
        lag_features = [
            current_data.get('close_lag_1', current_data['close']),
            current_data.get('close_lag_5', current_data['close']),
            current_data.get('volume_lag_1', current_data['volume']),
            current_data.get('volume_lag_5', current_data['volume']),
            current_data.get('dma_50_lag_1', current_data.get('dma_50', 0)),
            current_data.get('dma_50_lag_5', current_data.get('dma_50', 0)),
            current_data.get('close_momentum_5', 0)
        ]
        
        # Rolling features (6 features)
        rolling_features = [
            current_data.get('close_rolling_mean_5', current_data['close']),
            current_data.get('close_rolling_mean_20', current_data['close']),
            current_data.get('close_rolling_std_20', 0),
            current_data.get('volume_rolling_mean_5', current_data['volume']),
            current_data.get('volume_rolling_mean_20', current_data['volume']),
            current_data.get('close_momentum_5', 0)
        ]
        
        # Combine all features
        observation = np.concatenate([
            portfolio_state,
            price_features,
            technical_features,
            fundamental_features,
            sentiment_features,
            lag_features,
            rolling_features
        ])
        
        # Ensure observation has correct size
        if len(observation) > self.state_space:
            observation = observation[:self.state_space]
        elif len(observation) < self.state_space:
            observation = np.pad(observation, (0, self.state_space - len(observation)))
        
        return observation.astype(np.float32)
    
    def render(self, mode='human'):
        """
        Render the environment
        """
        if mode == 'human':
            print(f"Day: {self.day}")
            print(f"Portfolio Value: {self.state[2]:.2f}")
            print(f"Cash: {self.state[0]:.2f}")
            print(f"Shares: {self.state[1]}")
            print(f"Total Trades: {self.state[5]}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.asset_memory) < 2:
            return {}
        
        # Calculate returns
        returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
        
        # Performance metrics
        total_return = (self.asset_memory[-1] - self.asset_memory[0]) / self.asset_memory[0]
        annualized_return = (1 + total_return) ** (252 / len(self.asset_memory)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / (volatility + 1e-8)
        
        # Drawdown
        peak = np.maximum.accumulate(self.asset_memory)
        drawdown = (self.asset_memory - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        positive_returns = np.sum(returns > 0)
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.state[5],
            'final_portfolio_value': self.asset_memory[-1]
        }

# Example usage
if __name__ == "__main__":
    from data_loader import FinancialDataLoader
    
    # Load and preprocess data
    loader = FinancialDataLoader()
    reliance_data = loader.load_stock_data("RELIANCE")
    processed_data = loader.preprocess_data(reliance_data)
    
    # Create environment
    env = SingleStockTradingEnv(
        df=processed_data,
        state_space=50,
        action_space=3,
        initial_amount=1000000.0
    )
    
    # Test environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Portfolio={info['total_assets']:.2f}")
        
        if done:
            break
    
    # Get performance metrics
    metrics = env.get_performance_metrics()
    print(f"\nPerformance Metrics: {metrics}")
