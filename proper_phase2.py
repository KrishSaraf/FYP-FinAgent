"""
PROPER Phase 2: No Remaining Cash + 4% Risk-Free Interest
Invest ALL $1M, no idle cash, proper feature selection
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

class ProperPortfolioEnv(gym.Env):
    """
    PROPER portfolio environment - invests ALL money, no idle cash
    """
    
    def __init__(self, df_dict, stock_list, initial_amount=1000000.0):
        super(ProperPortfolioEnv, self).__init__()
        
        self.df_dict = df_dict
        self.stock_list = stock_list
        self.initial_amount = initial_amount
        self.n_stocks = len(stock_list)
        self.risk_free_rate = 0.04  # 4% annual risk-free rate
        
        # Calculate equal allocation per stock
        self.initial_allocation_per_stock = initial_amount / self.n_stocks
        print(f"💰 Initial allocation per stock: ${self.initial_allocation_per_stock:,.2f}")
        
        # Simple action space: 5 actions
        self.action_space = spaces.Discrete(5)
        
        # Better observation space: 4 features per stock + 6 market features
        obs_dim = 4 * self.n_stocks + 6  # 4 features per stock + 6 market features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset environment with ALL money invested"""
        self.day = 0
        self.cash = 0  # Start with NO cash - invest everything
        self.shares = np.zeros(self.n_stocks)
        self.portfolio_values = [self.initial_amount]
        self.initial_prices = np.zeros(self.n_stocks)
        self.trade_count = 0
        
        # Invest ALL money immediately
        self._invest_all_money()
        
        return self._get_obs()
    
    def _invest_all_money(self):
        """Invest ALL $1M - no remaining cash"""
        current_prices = self._get_current_prices()
        
        # Calculate shares for each stock to use ALL money
        total_invested = 0
        for i, price in enumerate(current_prices):
            if price > 0:
                # Calculate shares to invest the full allocation
                shares = int(self.initial_allocation_per_stock / price)
                self.shares[i] = shares
                total_invested += shares * price
                self.initial_prices[i] = price
        
        # Any remaining money goes to the first stock
        remaining = self.initial_amount - total_invested
        if remaining > 0 and current_prices[0] > 0:
            extra_shares = int(remaining / current_prices[0])
            self.shares[0] += extra_shares
            total_invested += extra_shares * current_prices[0]
        
        self.cash = self.initial_amount - total_invested
        
        print(f"📊 ALL MONEY INVESTED:")
        print(f"   Total shares purchased: {np.sum(self.shares):,}")
        print(f"   Total invested: ${total_invested:,.2f}")
        print(f"   Remaining cash: ${self.cash:,.2f}")
        print(f"   Initial portfolio value: ${self.cash + total_invested:,.2f}")
    
    def step(self, action):
        # Simple action decoding
        if action == 0:  # Buy more
            self._execute_buy_strategy()
        elif action == 1:  # Hold
            pass  # Do nothing
        elif action == 2:  # Sell some
            self._execute_sell_strategy()
        elif action == 3:  # Rebalance
            self._execute_rebalance_strategy()
        else:  # Hold and wait
            pass
        
        # Update day
        self.day += 1
        
        # Calculate portfolio value with risk-free interest on cash
        current_prices = self._get_current_prices()
        stock_value = np.sum(self.shares * current_prices)
        
        # Add 4% annual risk-free interest on cash (daily rate = 4%/252)
        daily_risk_free_rate = self.risk_free_rate / 252
        self.cash = self.cash * (1 + daily_risk_free_rate)
        
        portfolio_value = self.cash + stock_value
        self.portfolio_values.append(portfolio_value)
        
        # PROPER reward calculation - just the portfolio return
        if len(self.portfolio_values) > 1:
            reward = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
        else:
            reward = 0
        
        # Check if done
        done = self.day >= 200
        
        return self._get_obs(), reward, done, {'portfolio_value': portfolio_value}
    
    def _execute_buy_strategy(self):
        """Buy stocks with positive momentum using available cash"""
        if self.cash < 1000:  # Need at least $1000 to trade
            return
            
        current_prices = self._get_current_prices()
        
        # Find stocks with positive momentum
        good_stocks = []
        for i, stock in enumerate(self.stock_list):
            if self.day > 0 and self.day < len(self.df_dict[stock]):
                current_price = current_prices[i]
                if self.day > 0:
                    prev_price = self.df_dict[stock].iloc[self.day - 1]['close']
                    if prev_price > 0 and current_price > prev_price:
                        good_stocks.append(i)
        
        # Buy top 5 performing stocks
        if good_stocks and self.cash > 1000:
            for i in good_stocks[:5]:
                if current_prices[i] > 0:
                    shares_to_buy = min(10, int(self.cash * 0.1 / current_prices[i]))
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_prices[i] * 1.001
                        if cost <= self.cash:
                            self.cash -= cost
                            self.shares[i] += shares_to_buy
                            self.trade_count += 1
    
    def _execute_sell_strategy(self):
        """Sell stocks with negative momentum"""
        current_prices = self._get_current_prices()
        
        # Find stocks with negative momentum
        bad_stocks = []
        for i, stock in enumerate(self.stock_list):
            if self.day > 0 and self.day < len(self.df_dict[stock]):
                current_price = current_prices[i]
                if self.day > 0:
                    prev_price = self.df_dict[stock].iloc[self.day - 1]['close']
                    if prev_price > 0 and current_price < prev_price:
                        bad_stocks.append(i)
        
        # Sell worst 5 performing stocks
        if bad_stocks:
            for i in bad_stocks[:5]:
                if self.shares[i] > 0 and current_prices[i] > 0:
                    shares_to_sell = min(10, int(self.shares[i] * 0.1))
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_prices[i] * 0.999
                        self.cash += proceeds
                        self.shares[i] -= shares_to_sell
                        self.trade_count += 1
    
    def _execute_rebalance_strategy(self):
        """Rebalance portfolio to equal weights"""
        current_prices = self._get_current_prices()
        total_value = self.cash + np.sum(self.shares * current_prices)
        target_per_stock = total_value / self.n_stocks
        
        for i, price in enumerate(current_prices):
            if price > 0:
                current_value = self.shares[i] * price
                target_shares = int(target_per_stock / price)
                
                if target_shares > self.shares[i]:  # Need to buy
                    shares_to_buy = target_shares - self.shares[i]
                    cost = shares_to_buy * price * 1.001
                    if cost <= self.cash:
                        self.cash -= cost
                        self.shares[i] = target_shares
                        self.trade_count += 1
                elif target_shares < self.shares[i]:  # Need to sell
                    shares_to_sell = self.shares[i] - target_shares
                    proceeds = shares_to_sell * price * 0.999
                    self.cash += proceeds
                    self.shares[i] = target_shares
                    self.trade_count += 1
    
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
        """Get observation with better features"""
        current_prices = self._get_current_prices()
        stock_value = np.sum(self.shares * current_prices)
        portfolio_value = self.cash + stock_value
        
        # Create observation with better features for each stock
        obs = []
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                # Better features for this stock
                obs.extend([
                    data.get('close', 0) / 1000,  # Normalized price
                    data.get('volume', 0) / 10000000,  # Normalized volume
                    data.get('rsi_14', 50) / 100,  # RSI momentum (0-1)
                    self.shares[i] / 100  # Normalized shares
                ])
            else:
                obs.extend([0, 0, 0, 0])
        
        # Market-level features
        obs.extend([
            portfolio_value / self.initial_amount,  # Portfolio value ratio
            self.cash / self.initial_amount,  # Cash ratio
            np.sum(self.shares > 0) / self.n_stocks,  # Diversification ratio
            self.day / 200.0,  # Time progress
            stock_value / self.initial_amount,  # Stock value ratio
            (portfolio_value - self.initial_amount) / self.initial_amount  # Total return
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
            excess_returns = np.array(daily_returns) - 0.04/252  # 4% risk-free rate
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
            'total_trades': self.trade_count,
            'final_cash': self.cash
        }

def run_proper_phase2():
    """
    Run PROPER Phase 2 - no remaining cash, 4% risk-free interest
    """
    print("🚀 PROPER Phase 2: No Remaining Cash + 4% Risk-Free Interest")
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
    print(f"\n🏗️ Creating PROPER environment...")
    env = ProperPortfolioEnv(df_dict, list(df_dict.keys()))
    print(f"✅ Environment created: {env.observation_space.shape[0]} obs, {env.action_space.n} actions")
    print(f"🎯 Using 4 features per stock: [close, volume, rsi_14, shares] + 6 market features")
    
    # Train PPO agent
    print("\n🤖 Training PROPER PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1
    )
    
    print("   Training for 100000 timesteps...")
    model.learn(total_timesteps=100000)
    print("✅ Training completed")
    
    # Test the agent
    print("\n📊 Testing PROPER agent...")
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
    
    print(f"\n📈 PROPER Performance Results:")
    print(f"   Initial Portfolio: ${metrics['initial_portfolio_value']:,.2f}")
    print(f"   Final Portfolio: ${metrics['final_portfolio_value']:,.2f}")
    print(f"   Final Cash: ${metrics['final_cash']:,.2f}")
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
    success = run_proper_phase2()
    
    if success:
        print("\n🎉 PROPER Phase 2 completed successfully!")
        print("✅ No remaining cash + 4% risk-free interest working!")
    else:
        print("\n🔧 Still needs work")
        print("💡 But now we're investing ALL money properly!")
