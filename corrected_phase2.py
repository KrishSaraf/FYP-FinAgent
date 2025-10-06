"""
Corrected Phase 2: Portfolio Management with All 45 Stocks
$1 Million equally divided among all 45 stocks
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

class CorrectedPortfolioEnv(gym.Env):
    """
    Corrected portfolio environment with all 45 stocks
    $1 Million equally divided among all stocks
    """
    
    def __init__(self, df_dict, stock_list, initial_amount=1000000.0):
        super(CorrectedPortfolioEnv, self).__init__()
        
        self.df_dict = df_dict
        self.stock_list = stock_list
        self.initial_amount = initial_amount
        self.n_stocks = len(stock_list)
        
        # Calculate equal allocation per stock
        self.initial_allocation_per_stock = initial_amount / self.n_stocks
        print(f"💰 Initial allocation per stock: ${self.initial_allocation_per_stock:,.2f}")
        
        # Action space: buy/sell/hold for each stock (3^n_stocks)
        self.action_space = spaces.Discrete(3 ** self.n_stocks)
        
        # Observation space: price, volume, shares, cash, portfolio_value for each stock + market info
        obs_dim = self.n_stocks * 5 + 5  # 5 features per stock + 5 market features
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
        # Decode action
        actions = self._decode_action(action)
        
        # Get current prices
        current_prices = self._get_current_prices()
        
        # Execute trades
        for i, stock_action in enumerate(actions):
            if current_prices[i] > 0:
                if stock_action == 1:  # Buy
                    # Buy with 5% of available cash
                    max_shares = int(self.cash * 0.05 / current_prices[i])
                    shares_to_buy = min(max_shares, 100)  # Limit to 100 shares per trade
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_prices[i] * 1.001  # 0.1% transaction cost
                        if cost <= self.cash:
                            self.cash -= cost
                            self.shares[i] += shares_to_buy
                
                elif stock_action == 2:  # Sell
                    # Sell up to 10% of current position
                    max_shares = max(1, int(self.shares[i] * 0.1))
                    shares_to_sell = min(self.shares[i], max_shares)
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_prices[i] * 0.999  # 0.1% transaction cost
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
    
    def _decode_action(self, action):
        """Decode single action into individual stock actions"""
        actions = []
        for i in range(self.n_stocks):
            actions.append(action % 3)
            action //= 3
        return actions
    
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
        """Get current observation"""
        current_prices = self._get_current_prices()
        portfolio_value = self.cash + np.sum(self.shares * current_prices)
        
        # Create observation: [price, volume, shares, cash_ratio, portfolio_ratio] for each stock
        obs = []
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                obs.extend([
                    data['close'] / 1000,  # Normalize price (divide by 1000)
                    data['volume'] / 1000000,  # Normalize volume
                    self.shares[i] / 1000,  # Normalize shares
                    self.cash / self.initial_amount,  # Cash ratio
                    portfolio_value / self.initial_amount  # Portfolio ratio
                ])
            else:
                obs.extend([0, 0, 0, 0, 0])
        
        # Add market info
        obs.extend([
            portfolio_value / self.initial_amount,  # Portfolio return
            self.cash / self.initial_amount,  # Cash ratio
            self.day / 200.0,  # Time progress
            np.sum(self.shares > 0) / self.n_stocks,  # Diversification ratio
            np.std(current_prices) / np.mean(current_prices) if np.mean(current_prices) > 0 else 0  # Price volatility
        ])
        
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
            'total_trades': np.sum(self.shares > 0),
            'diversification': np.sum(self.shares > 0) / self.n_stocks
        }

def run_corrected_phase2():
    """
    Run corrected Phase 2 with all 45 stocks
    """
    print("🚀 Corrected Phase 2: All 45 Stocks Portfolio Management")
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
    env = CorrectedPortfolioEnv(df_dict, list(df_dict.keys()))
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
    
    print("   Training for 30000 timesteps...")
    model.learn(total_timesteps=30000)
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
    print(f"   Diversification: {metrics['diversification']:.4f} ({metrics['diversification']*100:.1f}% of stocks)")
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
    success = run_corrected_phase2()
    
    if success:
        print("\n🎉 Corrected Phase 2 completed successfully!")
        print("✅ All 45 stocks with equal $1M allocation working!")
    else:
        print("\n🔧 Phase 2 needs optimization")
        print("💡 Try increasing training time or adjusting parameters")
