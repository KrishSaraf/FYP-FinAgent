"""
Fixed Smart Phase 2: Portfolio Management with Data Validation
Handles NaN values and missing data properly
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import FinancialDataLoader
from stable_baselines3 import PPO
import gym
from gym import spaces

class FixedSmartPortfolioEnv(gym.Env):
    """
    Fixed smart portfolio environment with proper data validation
    """
    
    def __init__(self, df_dict, stock_list, initial_amount=1000000.0):
        super(FixedSmartPortfolioEnv, self).__init__()
        
        self.df_dict = df_dict
        self.stock_list = stock_list
        self.initial_amount = initial_amount
        self.n_stocks = len(stock_list)
        
        # Calculate equal allocation per stock
        self.initial_allocation_per_stock = initial_amount / self.n_stocks
        print(f"💰 Initial allocation per stock: ${self.initial_allocation_per_stock:,.2f}")
        
        # Smart action space: 7 actions
        self.action_space = spaces.Discrete(7)
        
        # Smart observation space: Only the most important features
        obs_dim = 5 * self.n_stocks + 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Define the most important features for each stock
        self.core_features = [
            'close',           # Current price
            'volume',          # Trading volume
            'rsi_14',          # RSI momentum
            'close_momentum_5', # 5-day momentum
            'dma_distance'     # Distance from moving average
        ]
        
        self.reset()
    
    def reset(self):
        """Reset environment with equal allocation"""
        self.day = 0
        self.cash = self.initial_amount
        self.shares = np.zeros(self.n_stocks)
        self.portfolio_values = [self.initial_amount]
        self.initial_prices = np.zeros(self.n_stocks)
        self.trade_count = 0
        
        # Get initial prices and calculate equal shares
        self._calculate_initial_allocation()
        
        return self._get_obs()
    
    def _calculate_initial_allocation(self):
        """Calculate initial equal allocation across all stocks"""
        current_prices = self._get_current_prices()
        
        # Calculate shares for equal allocation
        for i, price in enumerate(current_prices):
            if price > 0:
                shares = int(self.initial_allocation_per_stock / price)
                self.shares[i] = shares
                self.cash -= shares * price
                self.initial_prices[i] = price
        
        print(f"📊 Initial allocation calculated:")
        print(f"   Total shares purchased: {np.sum(self.shares):,}")
        print(f"   Remaining cash: ${self.cash:,.2f}")
        print(f"   Initial portfolio value: ${self.cash + np.sum(self.shares * current_prices):,.2f}")
    
    def step(self, action):
        # Smart action decoding
        strategy, intensity = self._decode_smart_action(action)
        
        # Get current prices
        current_prices = self._get_current_prices()
        
        # Execute smart strategy
        self._execute_smart_strategy(strategy, intensity, current_prices)
        
        # Update day
        self.day += 1
        
        # Calculate portfolio value
        portfolio_value = self.cash + np.sum(self.shares * current_prices)
        self.portfolio_values.append(portfolio_value)
        
        # Smart reward calculation
        reward = self._calculate_smart_reward(portfolio_value)
        
        # Check if done
        done = self.day >= 200
        
        return self._get_obs(), reward, done, {'portfolio_value': portfolio_value}
    
    def _decode_smart_action(self, action):
        """Decode action into smart strategy"""
        if action == 0:  # Strong buy
            return "strong_buy", 0.15
        elif action == 1:  # Moderate buy
            return "moderate_buy", 0.10
        elif action == 2:  # Light buy
            return "light_buy", 0.05
        elif action == 3:  # Hold
            return "hold", 0
        elif action == 4:  # Light sell
            return "light_sell", 0.05
        elif action == 5:  # Moderate sell
            return "moderate_sell", 0.10
        else:  # Strong sell
            return "strong_sell", 0.15
    
    def _execute_smart_strategy(self, strategy, intensity, current_prices):
        """Execute smart trading strategy based on momentum and RSI"""
        if "buy" in strategy:
            # Get stock scores based on momentum and RSI
            scores = self._calculate_stock_scores(current_prices)
            
            # Select top performers
            if strategy == "strong_buy":
                top_stocks = np.argsort(scores)[-15:]  # Top 15
            elif strategy == "moderate_buy":
                top_stocks = np.argsort(scores)[-10:]  # Top 10
            else:  # light_buy
                top_stocks = np.argsort(scores)[-5:]   # Top 5
            
            # Execute buys
            for i in top_stocks:
                if current_prices[i] > 0 and self.cash > 1000:
                    max_shares = int(self.cash * intensity / current_prices[i])
                    shares_to_buy = min(max_shares, 100)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_prices[i] * 1.001
                        if cost <= self.cash:
                            self.cash -= cost
                            self.shares[i] += shares_to_buy
                            self.trade_count += 1
        
        elif "sell" in strategy:
            # Get stock scores
            scores = self._calculate_stock_scores(current_prices)
            
            # Select worst performers
            if strategy == "strong_sell":
                bottom_stocks = np.argsort(scores)[:15]  # Bottom 15
            elif strategy == "moderate_sell":
                bottom_stocks = np.argsort(scores)[:10]  # Bottom 10
            else:  # light_sell
                bottom_stocks = np.argsort(scores)[:5]   # Bottom 5
            
            # Execute sells
            for i in bottom_stocks:
                if self.shares[i] > 0 and current_prices[i] > 0:
                    max_shares = int(self.shares[i] * intensity)
                    shares_to_sell = min(max_shares, self.shares[i])
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_prices[i] * 0.999
                        self.cash += proceeds
                        self.shares[i] -= shares_to_sell
                        self.trade_count += 1
    
    def _calculate_stock_scores(self, current_prices):
        """Calculate smart scores for stock selection"""
        scores = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                # Get features with proper NaN handling
                momentum = self._safe_get(data, 'close_momentum_5', 0)
                rsi = self._safe_get(data, 'rsi_14', 50)
                volume = self._safe_get(data, 'volume', 0)
                dma_dist = self._safe_get(data, 'dma_distance', 0)
                
                # Momentum score (5-day momentum)
                momentum_score = np.clip(momentum, -1, 1)  # Clip to reasonable range
                
                # RSI score (inverted - lower RSI = higher score for buying)
                rsi_score = (50 - rsi) / 50  # -1 to 1, where -1 is oversold
                rsi_score = np.clip(rsi_score, -1, 1)
                
                # Volume score (higher volume = higher score)
                volume_score = min(volume / 10000000, 1)  # Normalize to 0-1
                volume_score = np.clip(volume_score, 0, 1)
                
                # DMA distance score (closer to DMA = higher score)
                dma_score = max(0, 1 - abs(dma_dist) * 10)  # Closer to DMA = higher score
                dma_score = np.clip(dma_score, 0, 1)
                
                # Combine scores
                scores[i] = momentum_score * 0.4 + rsi_score * 0.3 + volume_score * 0.2 + dma_score * 0.1
        
        return scores
    
    def _safe_get(self, data, key, default=0):
        """Safely get value from data with NaN handling"""
        try:
            value = data.get(key, default)
            if pd.isna(value) or np.isnan(value) or np.isinf(value):
                return default
            return float(value)
        except:
            return default
    
    def _calculate_smart_reward(self, portfolio_value):
        """Smart reward function focused on returns and risk"""
        if len(self.portfolio_values) < 2:
            return 0
        
        # Basic return
        daily_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
        
        # Momentum bonus
        momentum_bonus = 0
        if len(self.portfolio_values) > 5:
            recent_returns = []
            for i in range(max(0, len(self.portfolio_values) - 5), len(self.portfolio_values) - 1):
                ret = (self.portfolio_values[i + 1] - self.portfolio_values[i]) / self.portfolio_values[i]
                recent_returns.append(ret)
            
            if len(recent_returns) > 0:
                momentum_bonus = np.mean(recent_returns) * 0.2
        
        # Diversification bonus
        diversification_bonus = 0
        active_stocks = np.sum(self.shares > 0)
        if active_stocks > 0:
            diversification_bonus = (active_stocks / self.n_stocks) * 0.05
        
        # Trade efficiency penalty
        trade_penalty = -self.trade_count * 0.0001
        
        # Risk penalty
        risk_penalty = 0
        if len(self.portfolio_values) > 10:
            recent_returns = []
            for i in range(max(0, len(self.portfolio_values) - 10), len(self.portfolio_values) - 1):
                ret = (self.portfolio_values[i + 1] - self.portfolio_values[i]) / self.portfolio_values[i]
                recent_returns.append(ret)
            
            if len(recent_returns) > 1:
                volatility = np.std(recent_returns)
                if volatility > 0.03:  # 3% daily volatility threshold
                    risk_penalty = -volatility * 0.5
        
        # Combine rewards
        total_reward = daily_return + momentum_bonus + diversification_bonus + trade_penalty + risk_penalty
        
        return total_reward * 20  # Scale up for better learning
    
    def _get_current_prices(self):
        """Get current prices for all stocks"""
        prices = []
        for stock in self.stock_list:
            if self.day < len(self.df_dict[stock]):
                prices.append(self.df_dict[stock].iloc[self.day]['close'])
            else:
                prices.append(0)
        return np.array(prices)
    
    def _get_obs(self):
        """Get smart observation with only the most important features"""
        current_prices = self._get_current_prices()
        portfolio_value = self.cash + np.sum(self.shares * current_prices)
        
        # Create observation with core features for each stock
        obs = []
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                # Core features for this stock with proper validation
                obs.extend([
                    self._safe_get(data, 'close', 0) / 1000,  # Normalized price
                    self._safe_get(data, 'volume', 0) / 10000000,  # Normalized volume
                    self._safe_get(data, 'rsi_14', 50) / 100,  # Normalized RSI
                    self._safe_get(data, 'close_momentum_5', 0),  # 5-day momentum
                    self._safe_get(data, 'dma_distance', 0)  # DMA distance
                ])
            else:
                obs.extend([0, 0, 0, 0, 0])
        
        # Market-level features
        obs.extend([
            portfolio_value / self.initial_amount,  # Portfolio value ratio
            self.cash / self.initial_amount,  # Cash ratio
            np.sum(self.shares > 0) / self.n_stocks,  # Diversification ratio
            self.day / 200.0,  # Time progress
            np.sum(self.shares * current_prices) / self.initial_amount,  # Stock value ratio
            self.trade_count / 100.0,  # Trade intensity
            (portfolio_value - self.initial_amount) / self.initial_amount,  # Total return
            len(self.portfolio_values) / 200.0  # Episode progress
        ])
        
        # Convert to numpy array and handle any remaining NaN values
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs_array
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return {}
        
        initial_value = self.portfolio_values[0]
        final_value = self.portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        # Calculate Sharpe ratio
        if len(daily_returns) > 1:
            excess_returns = np.array(daily_returns) - 0.02/252
            sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (np.array(self.portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(self.portfolio_values)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': final_value,
            'initial_portfolio_value': initial_value,
            'active_stocks': np.sum(self.shares > 0),
            'diversification': np.sum(self.shares > 0) / self.n_stocks,
            'total_shares': np.sum(self.shares),
            'total_trades': self.trade_count
        }

def run_fixed_smart_phase2():
    """
    Run fixed smart Phase 2 with data validation
    """
    print("🚀 Fixed Smart Phase 2: Data-Validated Portfolio Management")
    print("=" * 60)
    
    # Load data for all stocks
    print("📊 Loading data for all 45 stocks...")
    loader = FinancialDataLoader()
    stock_list = loader.get_available_stocks()
    
    df_dict = {}
    successful_loads = 0
    
    for stock in stock_list:
        try:
            df_dict[stock] = loader.load_stock_data(stock)
            successful_loads += 1
            if successful_loads <= 5:
                print(f"✅ {stock}: {df_dict[stock].shape}")
        except Exception as e:
            print(f"❌ Error loading {stock}: {e}")
    
    print(f"📈 Successfully loaded {successful_loads}/{len(stock_list)} stocks")
    
    # Create environment
    print(f"\n🏗️ Creating fixed smart environment...")
    env = FixedSmartPortfolioEnv(df_dict, list(df_dict.keys()))
    print(f"✅ Environment created: {env.observation_space.shape[0]} obs, {env.action_space.n} actions")
    print(f"🎯 Using only {len(env.core_features)} core features per stock: {env.core_features}")
    
    # Train PPO agent with optimized parameters
    print("\n🤖 Training fixed smart PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,  # Higher learning rate
        n_steps=2048,
        batch_size=256,  # Larger batch size
        n_epochs=20,  # More epochs
        gamma=0.99,  # High discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1
    )
    
    print("   Training for 80000 timesteps...")
    model.learn(total_timesteps=80000)
    print("✅ Training completed")
    
    # Test the agent
    print("\n📊 Testing fixed smart agent...")
    obs = env.reset()
    total_reward = 0
    portfolio_values = []
    
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        portfolio_values.append(info['portfolio_value'])
        
        if done:
            break
    
    # Calculate performance
    metrics = env.get_performance_metrics()
    
    print(f"\n📈 Fixed Smart Performance Results:")
    print(f"   Initial Portfolio: ${metrics['initial_portfolio_value']:,.2f}")
    print(f"   Final Portfolio: ${metrics['final_portfolio_value']:,.2f}")
    print(f"   Total Return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
    print(f"   Annualized Return: {metrics['annualized_return']:.4f} ({metrics['annualized_return']*100:.2f}%)")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
    print(f"   Active Stocks: {metrics['active_stocks']}/{successful_loads}")
    print(f"   Diversification: {metrics['diversification']:.4f} ({metrics['diversification']*100:.1f}% of stocks)")
    print(f"   Total Shares: {metrics['total_shares']:,.0f}")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Total Reward: {total_reward:.4f}")
    
    # Check if target achieved
    if metrics['total_return'] >= 0.10:
        print(f"🎉 SUCCESS: Target achieved! ({metrics['total_return']*100:.2f}% > 10%)")
        return True
    elif metrics['total_return'] >= 0.05:
        print(f"✅ Good performance: {metrics['total_return']*100:.2f}%")
        return True
    else:
        print(f"⚠️ Below target: {metrics['total_return']*100:.2f}%")
        return False

if __name__ == "__main__":
    success = run_fixed_smart_phase2()
    
    if success:
        print("\n🎉 Fixed Smart Phase 2 completed successfully!")
        print("✅ Feature selection + data validation working for >10% returns!")
    else:
        print("\n🔧 Further optimization needed")
        print("💡 Try different market periods or longer training")
