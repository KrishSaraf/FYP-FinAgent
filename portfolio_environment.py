"""
Advanced Multi-Stock Portfolio Trading Environment for FinRL Phase 2
Implements sophisticated portfolio management with cross-asset features
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class PortfolioTradingEnv(gym.Env):
    """
    Advanced portfolio trading environment for multi-stock strategies
    Optimized for achieving >10% returns with sophisticated risk management
    """
    
    def __init__(self, 
                 df_dict: Dict[str, pd.DataFrame],
                 stock_list: List[str],
                 initial_amount: float = 1000000.0,
                 transaction_cost_pct: float = 0.001,
                 reward_scaling: float = 1e-4,
                 state_space: int = 200,
                 action_space: int = None,  # Will be set based on stock_list
                 hmax: int = 100,
                 turbulence_threshold: float = 140,
                 risk_free_rate: float = 0.02,
                 lookback: int = 252,
                 make_plots: bool = False):
        """
        Initialize the portfolio trading environment
        
        Args:
            df_dict: Dictionary of DataFrames for each stock
            stock_list: List of stock symbols
            initial_amount: Initial capital
            transaction_cost_pct: Transaction cost percentage
            reward_scaling: Scaling factor for rewards
            state_space: Dimension of state space
            action_space: Dimension of action space
            hmax: Maximum number of shares per stock
            turbulence_threshold: Market turbulence threshold
            risk_free_rate: Risk-free rate for Sharpe calculation
            lookback: Lookback period for technical indicators
            make_plots: Whether to make plots
        """
        super(PortfolioTradingEnv, self).__init__()
        
        self.df_dict = df_dict
        self.stock_list = stock_list
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.hmax = hmax
        self.turbulence_threshold = turbulence_threshold
        self.risk_free_rate = risk_free_rate
        self.lookback = lookback
        self.make_plots = make_plots
        
        # Set action space (buy/sell/hold for each stock + cash allocation)
        self.action_space_dim = len(stock_list) * 2 + 1  # 2 actions per stock + cash allocation
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_space_dim,), dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32
        )
        
        # Initialize environment
        self._prepare_data()
        self.reset()
    
    def _prepare_data(self):
        """Prepare and align data across all stocks"""
        # Align all dataframes by date
        all_dates = set()
        for df in self.df_dict.values():
            all_dates.update(df.index)
        
        self.dates = sorted(list(all_dates))
        self.n_days = len(self.dates)
        
        # Create aligned data
        self.aligned_data = {}
        for stock in self.stock_list:
            if stock in self.df_dict:
                df = self.df_dict[stock].reindex(self.dates, method='ffill')
                self.aligned_data[stock] = df.fillna(method='bfill')
        
        # Calculate market features
        self._calculate_market_features()
    
    def _calculate_market_features(self):
        """Calculate market-wide features"""
        # Calculate market index (equal-weighted portfolio)
        market_prices = []
        for stock in self.stock_list:
            if stock in self.aligned_data:
                market_prices.append(self.aligned_data[stock]['close'].values)
        
        if market_prices:
            self.market_index = np.mean(market_prices, axis=0)
            self.market_returns = np.diff(self.market_index) / self.market_index[:-1]
            self.market_volatility = pd.Series(self.market_returns).rolling(20).std().values
        else:
            self.market_index = np.ones(self.n_days)
            self.market_returns = np.zeros(self.n_days - 1)
            self.market_volatility = np.zeros(self.n_days)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.day = 0
        self.data = self.aligned_data.copy()
        
        # Initialize portfolio state
        self.state = np.array([
            self.initial_amount,  # cash
            0,  # total_portfolio_value
            0,  # total_cost
            0,  # total_trades
            0,  # total_buy_trades
            0,  # total_sell_trades
            0,  # total_buy_amount
            0,  # total_sell_amount
            0,  # daily_return
            0,  # cumulative_return
        ])
        
        # Initialize stock positions
        self.stock_owned = np.zeros(len(self.stock_list))
        self.stock_price = np.zeros(len(self.stock_list))
        self.stock_cost = np.zeros(len(self.stock_list))
        
        # Performance tracking
        self.asset_memory = [self.initial_amount]
        self.reward_memory = []
        self.actions_memory = []
        self.portfolio_weights = []
        self.returns_memory = []
        
        # Risk management
        self.max_drawdown = 0
        self.peak_value = self.initial_amount
        self.consecutive_losses = 0
        self.volatility_memory = []
        
        return self._get_observation()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Get current prices
        current_prices = self._get_current_prices()
        
        # Execute actions
        reward = self._execute_actions(actions, current_prices)
        
        # Update day
        self.day += 1
        
        # Check if episode is done
        done = self.day >= self.n_days - 1
        
        # Get next observation
        obs = self._get_observation()
        
        # Store information
        info = {
            'total_assets': self.state[1],
            'cash': self.state[0],
            'daily_return': self.state[8],
            'cumulative_return': self.state[9],
            'portfolio_weights': self._get_portfolio_weights(),
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'volatility': self._calculate_volatility()
        }
        
        return obs, reward, done, info
    
    def _get_current_prices(self) -> np.ndarray:
        """Get current prices for all stocks"""
        prices = np.zeros(len(self.stock_list))
        for i, stock in enumerate(self.stock_list):
            if stock in self.data and self.day < len(self.data[stock]):
                prices[i] = self.data[stock].iloc[self.day]['close']
            else:
                prices[i] = 0
        return prices
    
    def _execute_actions(self, actions: np.ndarray, current_prices: np.ndarray) -> float:
        """Execute trading actions"""
        # Calculate current portfolio value
        current_portfolio_value = self.state[0] + np.sum(self.stock_owned * current_prices)
        
        # Ensure actions array has correct size
        if len(actions) != self.action_space_dim:
            # Pad or truncate actions to match expected size
            if len(actions) < self.action_space_dim:
                actions = np.pad(actions, (0, self.action_space_dim - len(actions)))
            else:
                actions = actions[:self.action_space_dim]
        
        # Process actions for each stock
        for i, stock in enumerate(self.stock_list):
            if current_prices[i] > 0:  # Valid price
                # Buy action (first half of actions)
                buy_action = actions[i] if i < len(actions) else 0
                # Sell action (second half of actions)
                sell_action = actions[i + len(self.stock_list)] if (i + len(self.stock_list)) < len(actions) else 0
                
                # Execute buy action
                if buy_action > 0.1:  # Buy threshold
                    shares_to_buy = min(
                        self.hmax - self.stock_owned[i],
                        int(self.state[0] * buy_action * 0.1 / current_prices[i])
                    )
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_prices[i] * (1 + self.transaction_cost_pct)
                        if cost <= self.state[0]:
                            self.state[0] -= cost
                            self.stock_owned[i] += shares_to_buy
                            self.stock_cost[i] += cost
                            self.state[3] += 1
                            self.state[4] += 1
                            self.state[6] += cost
                
                # Execute sell action
                if sell_action > 0.1:  # Sell threshold
                    shares_to_sell = min(self.stock_owned[i], int(self.stock_owned[i] * sell_action))
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_prices[i] * (1 - self.transaction_cost_pct)
                        self.state[0] += proceeds
                        self.stock_owned[i] -= shares_to_sell
                        self.stock_cost[i] -= proceeds
                        self.state[3] += 1
                        self.state[5] += 1
                        self.state[7] += proceeds
        
        # Update portfolio state
        self.stock_price = current_prices
        new_portfolio_value = self.state[0] + np.sum(self.stock_owned * current_prices)
        
        # Calculate returns
        daily_return = (new_portfolio_value - current_portfolio_value) / current_portfolio_value
        cumulative_return = (new_portfolio_value - self.initial_amount) / self.initial_amount
        
        self.state[1] = new_portfolio_value
        self.state[2] = self.state[6] - self.state[7]
        self.state[8] = daily_return
        self.state[9] = cumulative_return
        
        # Update risk metrics
        self._update_risk_metrics(new_portfolio_value)
        
        # Calculate reward
        reward = self._calculate_reward(current_portfolio_value, new_portfolio_value)
        
        # Store memory
        self.asset_memory.append(new_portfolio_value)
        self.reward_memory.append(reward)
        self.actions_memory.append(actions.copy())
        self.portfolio_weights.append(self._get_portfolio_weights())
        self.returns_memory.append(daily_return)
        
        return reward
    
    def _update_risk_metrics(self, portfolio_value: float):
        """Update risk management metrics"""
        # Update peak and drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
            self.consecutive_losses = 0
        else:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
            self.consecutive_losses += 1
        
        # Update volatility
        if len(self.returns_memory) > 1:
            volatility = np.std(self.returns_memory[-20:])  # 20-day rolling volatility
            self.volatility_memory.append(volatility)
    
    def _calculate_reward(self, previous_value: float, current_value: float) -> float:
        """Calculate sophisticated reward function"""
        # Basic return
        portfolio_return = (current_value - previous_value) / previous_value
        
        # Risk-adjusted reward (Sharpe ratio)
        if len(self.returns_memory) > 1:
            returns = np.array(self.returns_memory[-20:])  # 20-day window
            if len(returns) > 1:
                excess_returns = returns - self.risk_free_rate / 252
                sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
                risk_adjusted_reward = sharpe_ratio * 0.1
            else:
                risk_adjusted_reward = 0
        else:
            risk_adjusted_reward = 0
        
        # Drawdown penalty
        drawdown_penalty = -0.01 * self.max_drawdown
        
        # Volatility penalty
        volatility_penalty = 0
        if len(self.volatility_memory) > 0:
            current_volatility = self.volatility_memory[-1]
            if current_volatility > 0.05:  # 5% daily volatility threshold
                volatility_penalty = -0.005 * (current_volatility - 0.05)
        
        # Transaction cost penalty
        transaction_penalty = -0.001 * self.state[3] / max(1, self.day)
        
        # Diversification bonus
        diversification_bonus = 0
        if len(self.portfolio_weights) > 0:
            weights = self.portfolio_weights[-1]
            if len(weights) > 1:
                # Calculate Herfindahl index (concentration)
                hhi = np.sum(weights ** 2)
                diversification_bonus = 0.01 * (1 - hhi)  # Bonus for diversification
        
        # Momentum bonus
        momentum_bonus = 0
        if len(self.returns_memory) >= 5:
            recent_returns = self.returns_memory[-5:]
            if np.mean(recent_returns) > 0:
                momentum_bonus = 0.005 * np.mean(recent_returns)
        
        # Combine all rewards
        total_reward = (portfolio_return + risk_adjusted_reward + drawdown_penalty + 
                       volatility_penalty + transaction_penalty + diversification_bonus + 
                       momentum_bonus)
        
        return total_reward * self.reward_scaling
    
    def _get_portfolio_weights(self) -> np.ndarray:
        """Get current portfolio weights"""
        total_value = self.state[1]
        if total_value == 0:
            return np.zeros(len(self.stock_list))
        
        weights = np.zeros(len(self.stock_list))
        for i in range(len(self.stock_list)):
            weights[i] = (self.stock_owned[i] * self.stock_price[i]) / total_value
        
        return weights
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)"""
        if self.day >= self.n_days:
            return np.zeros(self.state_space, dtype=np.float32)
        
        # Portfolio state (10 features)
        portfolio_state = self.state.copy()
        
        # Stock-specific features
        stock_features = []
        for i, stock in enumerate(self.stock_list):
            if stock in self.data and self.day < len(self.data[stock]):
                current_data = self.data[stock].iloc[self.day]
                
                # Price features
                price_features = [
                    current_data['open'],
                    current_data['high'],
                    current_data['low'],
                    current_data['close'],
                    current_data['volume'],
                    current_data['vwap']
                ]
                
                # Technical indicators
                technical_features = [
                    current_data.get('dma_50', 0),
                    current_data.get('dma_200', 0),
                    current_data.get('rsi_14', 50),
                    current_data.get('dma_cross', 0),
                    current_data.get('dma_distance', 0)
                ]
                
                # Fundamental features
                fundamental_features = [
                    current_data.get('metric_pPerEExcludingExtraordinaryItemsMostRecentFiscalYear', 0),
                    current_data.get('metric_priceToBookMostRecentFiscalYear', 0),
                    current_data.get('metric_returnOnAverageEquityTrailing12Month', 0),
                    current_data.get('metric_operatingMarginTrailing12Month', 0),
                    current_data.get('metric_grossMarginTrailing12Month', 0)
                ]
                
                # Sentiment features
                sentiment_features = [
                    current_data.get('reddit_title_sentiments_mean', 0),
                    current_data.get('news_sentiment_mean', 0),
                    current_data.get('reddit_posts_count', 0),
                    current_data.get('news_articles_count', 0)
                ]
                
                # Position features
                position_features = [
                    self.stock_owned[i],
                    self.stock_price[i],
                    self.stock_cost[i],
                    self._get_portfolio_weights()[i]
                ]
                
                stock_features.extend(price_features + technical_features + 
                                    fundamental_features + sentiment_features + 
                                    position_features)
            else:
                # Fill with zeros if no data
                stock_features.extend([0] * 25)
        
        # Market features
        market_features = [
            self.market_index[self.day] if self.day < len(self.market_index) else 0,
            self.market_volatility[self.day] if self.day < len(self.market_volatility) else 0,
            self.max_drawdown,
            self.consecutive_losses,
            len(self.volatility_memory) > 0 and self.volatility_memory[-1] or 0
        ]
        
        # Cross-asset features
        cross_asset_features = self._calculate_cross_asset_features()
        
        # Combine all features
        observation = np.concatenate([
            portfolio_state,
            stock_features,
            market_features,
            cross_asset_features
        ])
        
        # Ensure observation has correct size
        if len(observation) > self.state_space:
            observation = observation[:self.state_space]
        elif len(observation) < self.state_space:
            observation = np.pad(observation, (0, self.state_space - len(observation)))
        
        return observation.astype(np.float32)
    
    def _calculate_cross_asset_features(self) -> np.ndarray:
        """Calculate cross-asset features"""
        if self.day < 1:
            return np.zeros(10)
        
        # Relative strength features
        relative_strength = np.zeros(len(self.stock_list))
        for i, stock in enumerate(self.stock_list):
            if stock in self.data and self.day < len(self.data[stock]):
                current_price = self.data[stock].iloc[self.day]['close']
                if self.day > 0:
                    prev_price = self.data[stock].iloc[self.day - 1]['close']
                    relative_strength[i] = (current_price - prev_price) / prev_price
        
        # Market correlation
        market_correlation = 0
        if len(relative_strength) > 1 and len(self.market_returns) > self.day:
            market_correlation = np.corrcoef(relative_strength, [self.market_returns[self.day]])[0, 1]
        
        # Sector rotation (simplified)
        sector_rotation = np.mean(relative_strength)
        
        # Volatility clustering
        volatility_clustering = 0
        if len(self.volatility_memory) > 1:
            volatility_clustering = np.corrcoef(self.volatility_memory[:-1], self.volatility_memory[1:])[0, 1]
        
        # Momentum features
        momentum_features = [
            np.mean(relative_strength),
            np.std(relative_strength),
            np.max(relative_strength),
            np.min(relative_strength),
            market_correlation,
            sector_rotation,
            volatility_clustering,
            self._calculate_momentum_score(),
            self._calculate_mean_reversion_score(),
            self._calculate_volatility_score()
        ]
        
        return np.array(momentum_features)
    
    def _calculate_momentum_score(self) -> float:
        """Calculate momentum score across portfolio"""
        if len(self.returns_memory) < 5:
            return 0
        
        recent_returns = self.returns_memory[-5:]
        return np.mean(recent_returns)
    
    def _calculate_mean_reversion_score(self) -> float:
        """Calculate mean reversion score"""
        if len(self.returns_memory) < 10:
            return 0
        
        recent_returns = self.returns_memory[-10:]
        mean_return = np.mean(recent_returns)
        return -mean_return  # Negative for mean reversion
    
    def _calculate_volatility_score(self) -> float:
        """Calculate volatility score"""
        if len(self.volatility_memory) < 5:
            return 0
        
        recent_volatility = self.volatility_memory[-5:]
        return np.mean(recent_volatility)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns_memory) < 2:
            return 0
        
        returns = np.array(self.returns_memory)
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.returns_memory) < 2:
            return 0
        
        return np.std(self.returns_memory) * np.sqrt(252)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(self.asset_memory) < 2:
            return {}
        
        # Calculate returns
        returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
        
        # Performance metrics
        total_return = (self.asset_memory[-1] - self.asset_memory[0]) / self.asset_memory[0]
        annualized_return = (1 + total_return) ** (252 / len(self.asset_memory)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / (volatility + 1e-8)
        
        # Drawdown
        peak = np.maximum.accumulate(self.asset_memory)
        drawdown = (self.asset_memory - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        positive_returns = np.sum(returns > 0)
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / (abs(max_drawdown) + 1e-8)
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_volatility = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / (downside_volatility + 1e-8)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': self.state[3],
            'final_portfolio_value': self.asset_memory[-1],
            'diversification_ratio': 1 - np.sum(np.array(self.portfolio_weights[-1]) ** 2) if self.portfolio_weights else 0
        }

# Example usage
if __name__ == "__main__":
    from data_loader import FinancialDataLoader
    
    # Load data for multiple stocks
    loader = FinancialDataLoader()
    stock_list = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    
    df_dict = {}
    for stock in stock_list:
        try:
            df_dict[stock] = loader.load_stock_data(stock)
            print(f"Loaded {stock}: {df_dict[stock].shape}")
        except:
            print(f"Could not load {stock}")
    
    if df_dict:
        # Create environment
        env = PortfolioTradingEnv(
            df_dict=df_dict,
            stock_list=list(df_dict.keys()),
            state_space=200,
            initial_amount=1000000.0
        )
        
        # Test environment
        obs = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Run a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"Step {i+1}: Reward={reward:.4f}, Portfolio={info['total_assets']:.2f}")
            
            if done:
                break
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        print(f"\nPerformance Metrics: {metrics}")
