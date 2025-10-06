"""
Practical Phase 2: Portfolio Management with All 45 Stocks
$1 Million equally divided among all 45 stocks
Realistic action space for training
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

class PracticalPortfolioEnv(gym.Env):
    """
    Practical portfolio environment with all 45 stocks
    $1 Million equally divided among all stocks
    Realistic action space for training
    """
    
    def __init__(self, df_dict, stock_list, initial_amount=1000000.0):
        super(PracticalPortfolioEnv, self).__init__()
        
        self.df_dict = df_dict
        self.stock_list = stock_list
        self.initial_amount = initial_amount
        self.n_stocks = len(stock_list)
        
        # Calculate equal allocation per stock
        self.initial_allocation_per_stock = initial_amount / self.n_stocks
        print(f"💰 Initial allocation per stock: ${self.initial_allocation_per_stock:,.2f}")
        
        # Practical action space: 9 actions (3x3 grid for portfolio management)
        # Actions: [0-2: Buy more, 3-5: Hold, 6-8: Sell some]
        self.action_space = spaces.Discrete(9)
        
        # Observation space: aggregated features for all stocks
        obs_dim = 20  # Reduced but comprehensive observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment with equal allocation"""
        self.day = 0
        self.cash = self.initial_amount
        self.shares = np.zeros(self.n_stocks)
        self.portfolio_values = [self.initial_amount]
        self.initial_prices = np.zeros(self.n_stocks)
        
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
        # Decode action into portfolio strategy
        if action < 3:  # Buy more (0, 1, 2)
            strategy = "buy"
            intensity = (action + 1) * 0.05  # 5%, 10%, 15% of cash
        elif action < 6:  # Hold (3, 4, 5)
            strategy = "hold"
            intensity = 0
        else:  # Sell some (6, 7, 8)
            strategy = "sell"
            intensity = (action - 5) * 0.05  # 5%, 10%, 15% of positions
        
        # Get current prices
        current_prices = self._get_current_prices()
        
        # Execute strategy
        if strategy == "buy" and self.cash > 1000:
            # Buy stocks with highest momentum
            returns = self._calculate_stock_returns(current_prices)
            top_stocks = np.argsort(returns)[-5:]  # Top 5 performing stocks
            
            for i in top_stocks:
                if current_prices[i] > 0:
                    max_shares = int(self.cash * intensity / current_prices[i])
                    shares_to_buy = min(max_shares, 50)  # Limit per stock
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_prices[i] * 1.001
                        if cost <= self.cash:
                            self.cash -= cost
                            self.shares[i] += shares_to_buy
        
        elif strategy == "sell":
            # Sell stocks with lowest momentum
            returns = self._calculate_stock_returns(current_prices)
            bottom_stocks = np.argsort(returns)[:5]  # Bottom 5 performing stocks
            
            for i in bottom_stocks:
                if self.shares[i] > 0 and current_prices[i] > 0:
                    max_shares = int(self.shares[i] * intensity)
                    shares_to_sell = min(max_shares, self.shares[i])
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_prices[i] * 0.999
                        self.cash += proceeds
                        self.shares[i] -= shares_to_sell
        
        # Update day
        self.day += 1
        
        # Calculate portfolio value
        portfolio_value = self.cash + np.sum(self.shares * current_prices)
        self.portfolio_values.append(portfolio_value)
        
        # Calculate reward
        if len(self.portfolio_values) > 1:
            reward = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
        else:
            reward = 0
        
        # Check if done
        done = self.day >= 200  # Use 200 days for training
        
        return self._get_obs(), reward, done, {'portfolio_value': portfolio_value}
    
    def _calculate_stock_returns(self, current_prices):
        """Calculate recent returns for all stocks"""
        returns = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stock_list):
            if self.day > 0 and self.day < len(self.df_dict[stock]):
                current_price = current_prices[i]
                if self.day > 0:
                    prev_price = self.df_dict[stock].iloc[self.day - 1]['close']
                    if prev_price > 0:
                        returns[i] = (current_price - prev_price) / prev_price
        
        return returns
    
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
        """Get current observation with aggregated features"""
        current_prices = self._get_current_prices()
        portfolio_value = self.cash + np.sum(self.shares * current_prices)
        
        # Calculate aggregated features
        total_shares = np.sum(self.shares)
        active_stocks = np.sum(self.shares > 0)
        avg_price = np.mean(current_prices[current_prices > 0]) if np.any(current_prices > 0) else 0
        price_volatility = np.std(current_prices[current_prices > 0]) / avg_price if avg_price > 0 else 0
        
        # Calculate sector performance (simplified)
        returns = self._calculate_stock_returns(current_prices)
        avg_return = np.mean(returns)
        return_volatility = np.std(returns)
        
        # Portfolio metrics
        cash_ratio = self.cash / self.initial_amount
        portfolio_return = (portfolio_value - self.initial_amount) / self.initial_amount
        
        # Create observation
        obs = [
            portfolio_value / self.initial_amount,  # Portfolio value ratio
            cash_ratio,  # Cash ratio
            total_shares / 1000,  # Total shares (normalized)
            active_stocks / self.n_stocks,  # Diversification ratio
            avg_price / 1000,  # Average price (normalized)
            price_volatility,  # Price volatility
            avg_return,  # Average return
            return_volatility,  # Return volatility
            portfolio_return,  # Portfolio return
            self.day / 200.0,  # Time progress
            np.sum(self.shares * current_prices) / self.initial_amount,  # Stock value ratio
            np.max(returns) if len(returns) > 0 else 0,  # Best performing stock return
            np.min(returns) if len(returns) > 0 else 0,  # Worst performing stock return
            np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0,  # Win rate
            np.percentile(returns, 75) if len(returns) > 0 else 0,  # 75th percentile return
            np.percentile(returns, 25) if len(returns) > 0 else 0,  # 25th percentile return
            np.sum(self.shares > 0) / self.n_stocks,  # Position ratio
            np.mean(self.shares[self.shares > 0]) / 100 if np.any(self.shares > 0) else 0,  # Avg position size
            np.std(self.shares[self.shares > 0]) / 100 if np.any(self.shares > 0) else 0,  # Position size std
            len(self.portfolio_values) / 200.0  # Episode progress
        ]
        
        return np.array(obs, dtype=np.float32)
    
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
        
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        if len(daily_returns) > 1:
            excess_returns = np.array(daily_returns) - 0.02/252  # Daily risk-free rate
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
            'total_shares': np.sum(self.shares)
        }

def run_practical_phase2():
    """
    Run practical Phase 2 with all 45 stocks
    """
    print("🚀 Practical Phase 2: All 45 Stocks Portfolio Management")
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
            if successful_loads <= 5:  # Show first 5
                print(f"✅ {stock}: {df_dict[stock].shape}")
        except Exception as e:
            print(f"❌ Error loading {stock}: {e}")
    
    print(f"📈 Successfully loaded {successful_loads}/{len(stock_list)} stocks")
    
    if successful_loads < 10:
        print("❌ Need at least 10 stocks for proper diversification")
        return False
    
    # Create environment
    print(f"\n🏗️ Creating environment with {successful_loads} stocks...")
    env = PracticalPortfolioEnv(df_dict, list(df_dict.keys()))
    print(f"✅ Environment created: {env.observation_space.shape[0]} obs, {env.action_space.n} actions")
    
    # Train PPO agent
    print("\n🤖 Training PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )
    
    print("   Training for 40000 timesteps...")
    model.learn(total_timesteps=40000)
    print("✅ Training completed")
    
    # Test the agent
    print("\n📊 Testing trained agent...")
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
    
    print(f"\n📈 Performance Results:")
    print(f"   Initial Portfolio: ${metrics['initial_portfolio_value']:,.2f}")
    print(f"   Final Portfolio: ${metrics['final_portfolio_value']:,.2f}")
    print(f"   Total Return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
    print(f"   Annualized Return: {metrics['annualized_return']:.4f} ({metrics['annualized_return']*100:.2f}%)")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
    print(f"   Active Stocks: {metrics['active_stocks']}/{successful_loads}")
    print(f"   Diversification: {metrics['diversification']:.4f} ({metrics['diversification']*100:.1f}% of stocks)")
    print(f"   Total Shares: {metrics['total_shares']:,.0f}")
    print(f"   Total Reward: {total_reward:.4f}")
    
    # Check if target achieved
    if metrics['total_return'] >= 0.10:  # 10% target
        print(f"🎉 SUCCESS: Target achieved! ({metrics['total_return']*100:.2f}% > 10%)")
        return True
    elif metrics['total_return'] >= 0.05:  # 5% minimum
        print(f"✅ Good performance: {metrics['total_return']*100:.2f}%")
        return True
    else:
        print(f"⚠️ Below target: {metrics['total_return']*100:.2f}%")
        return False

if __name__ == "__main__":
    success = run_practical_phase2()
    
    if success:
        print("\n🎉 Practical Phase 2 completed successfully!")
        print("✅ All 45 stocks with equal $1M allocation working!")
    else:
        print("\n🔧 Phase 2 needs optimization")
        print("💡 Try increasing training time or adjusting parameters")
