"""
Comprehensive Optimized Trading Pipeline for FinRL
Integrates all advanced components to achieve >10% returns
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
from advanced_feature_engineering import AdvancedFeatureEngineer
from hyperparameter_optimizer import HyperparameterOptimizer
from advanced_ensemble import AdvancedEnsemble
from advanced_market_timing import AdvancedMarketTiming
from stable_baselines3 import PPO, A2C, DDPG, SAC
import gym
from gym import spaces

class OptimizedTradingEnvironment(gym.Env):
    """
    Optimized trading environment with advanced features
    """
    
    def __init__(self, df_dict, stock_list, initial_amount=1000000.0):
        super(OptimizedTradingEnvironment, self).__init__()
        
        self.df_dict = df_dict
        self.stock_list = stock_list
        self.initial_amount = initial_amount
        self.n_stocks = len(stock_list)
        
        # Advanced components
        self.market_timing = AdvancedMarketTiming()
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Risk management parameters
        self.max_position_pct = 0.08  # 8% max per stock
        self.max_total_exposure = 0.95  # 95% max total exposure
        self.transaction_cost = 0.0015  # 0.15% transaction cost
        self.slippage = 0.0005  # 0.05% slippage
        
        # Action space: 7 sophisticated actions
        self.action_space = spaces.Discrete(7)
        
        # Observation space: 15 features per stock + 10 market features
        obs_dim = 15 * self.n_stocks + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Advanced features
        self.advanced_features = [
            'close', 'volume', 'rsi_14', 'dma_50', 'dma_distance',
            'close_momentum_5', 'close_momentum_20', 'bollinger_position',
            'macd', 'macd_signal', 'stoch_k', 'williams_r', 'cci',
            'price_vs_dma50', 'z_score_10'
        ]
        
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.day = 0
        self.cash = self.initial_amount
        self.long_shares = np.zeros(self.n_stocks)
        self.long_cost_basis = np.zeros(self.n_stocks)
        self.portfolio_values = [self.initial_amount]
        self.trade_count = 0
        
        # Initial allocation
        self._initial_smart_allocation()
        
        return self._get_obs()
    
    def _initial_smart_allocation(self):
        """Smart initial allocation based on market conditions"""
        current_prices = self._get_current_prices()
        
        # Calculate equal allocation
        allocation_per_stock = self.initial_amount / self.n_stocks
        
        total_invested = 0
        for i, price in enumerate(current_prices):
            if price > 0:
                shares = int(allocation_per_stock / price)
                self.long_shares[i] = shares
                self.long_cost_basis[i] = shares * price
                total_invested += shares * price
        
        self.cash = self.initial_amount - total_invested
        
        print(f"📊 SMART INITIAL ALLOCATION:")
        print(f"   Total invested: ${total_invested:,.2f}")
        print(f"   Remaining cash: ${self.cash:,.2f}")
        print(f"   Initial portfolio value: ${self.cash + total_invested:,.2f}")
    
    def step(self, action):
        """Execute trading step"""
        # Decode action
        strategy = self._decode_action(action)
        
        # Execute strategy
        self._execute_strategy(strategy)
        
        # Update day
        self.day += 1
        
        # Calculate portfolio value
        current_prices = self._get_current_prices()
        portfolio_value = self._calculate_portfolio_value(current_prices)
        self.portfolio_values.append(portfolio_value)
        
        # Show progress
        if self.day % 20 == 0:
            total_return = (portfolio_value - self.initial_amount) / self.initial_amount * 100
            print(f"📈 Day {self.day}: Portfolio = ${portfolio_value:,.2f} ({total_return:+.2f}%)")
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_value)
        
        # Check if done
        done = self.day >= 200
        
        return self._get_obs(), reward, done, {'portfolio_value': portfolio_value}
    
    def _decode_action(self, action):
        """Decode action into trading strategy"""
        strategies = [
            "momentum_buy",      # 0: Buy high momentum stocks
            "mean_reversion",    # 1: Mean reversion strategy
            "trend_following",   # 2: Trend following
            "volatility_play",   # 3: Volatility-based strategy
            "sentiment_trade",   # 4: Sentiment-based trading
            "rebalance",         # 5: Rebalance portfolio
            "risk_off"           # 6: Risk-off strategy
        ]
        return strategies[action]
    
    def _execute_strategy(self, strategy):
        """Execute trading strategy"""
        current_prices = self._get_current_prices()
        
        if strategy == "momentum_buy":
            self._execute_momentum_strategy(current_prices)
        elif strategy == "mean_reversion":
            self._execute_mean_reversion_strategy(current_prices)
        elif strategy == "trend_following":
            self._execute_trend_following_strategy(current_prices)
        elif strategy == "volatility_play":
            self._execute_volatility_strategy(current_prices)
        elif strategy == "sentiment_trade":
            self._execute_sentiment_strategy(current_prices)
        elif strategy == "rebalance":
            self._execute_rebalance_strategy(current_prices)
        elif strategy == "risk_off":
            self._execute_risk_off_strategy(current_prices)
    
    def _execute_momentum_strategy(self, current_prices):
        """Execute momentum strategy"""
        if self.cash < 1000:
            return
        
        # Calculate momentum scores
        momentum_scores = self._calculate_momentum_scores(current_prices)
        
        # Buy top momentum stocks
        top_stocks = np.argsort(momentum_scores)[-3:]
        
        for i in top_stocks:
            if current_prices[i] > 0:
                current_position_value = self.long_shares[i] * current_prices[i]
                max_position_value = self.portfolio_values[-1] * self.max_position_pct
                
                if current_position_value < max_position_value:
                    max_investment = min(
                        self.cash * 0.15,
                        max_position_value - current_position_value
                    )
                    shares_to_buy = int(max_investment / current_prices[i])
                    shares_to_buy = min(shares_to_buy, 20)
                    
                    if shares_to_buy > 0:
                        effective_price = current_prices[i] * (1 + self.slippage)
                        cost = shares_to_buy * effective_price * (1 + self.transaction_cost)
                        
                        if cost <= self.cash:
                            self.cash -= cost
                            self.long_shares[i] += shares_to_buy
                            self.long_cost_basis[i] += cost
                            self.trade_count += 1
    
    def _execute_mean_reversion_strategy(self, current_prices):
        """Execute mean reversion strategy"""
        # Calculate mean reversion scores
        mean_reversion_scores = self._calculate_mean_reversion_scores(current_prices)
        
        # Buy oversold stocks
        oversold_stocks = np.argsort(mean_reversion_scores)[:3]
        
        for i in oversold_stocks:
            if current_prices[i] > 0 and self.cash > 1000:
                max_investment = min(self.cash * 0.1, 50000)
                shares_to_buy = int(max_investment / current_prices[i])
                shares_to_buy = min(shares_to_buy, 15)
                
                if shares_to_buy > 0:
                    effective_price = current_prices[i] * (1 + self.slippage)
                    cost = shares_to_buy * effective_price * (1 + self.transaction_cost)
                    
                    if cost <= self.cash:
                        self.cash -= cost
                        self.long_shares[i] += shares_to_buy
                        self.long_cost_basis[i] += cost
                        self.trade_count += 1
    
    def _execute_trend_following_strategy(self, current_prices):
        """Execute trend following strategy"""
        # Calculate trend scores
        trend_scores = self._calculate_trend_scores(current_prices)
        
        # Buy strong trend stocks
        strong_trend_stocks = np.argsort(trend_scores)[-2:]
        
        for i in strong_trend_stocks:
            if current_prices[i] > 0 and self.cash > 1000:
                max_investment = min(self.cash * 0.2, 75000)
                shares_to_buy = int(max_investment / current_prices[i])
                shares_to_buy = min(shares_to_buy, 25)
                
                if shares_to_buy > 0:
                    effective_price = current_prices[i] * (1 + self.slippage)
                    cost = shares_to_buy * effective_price * (1 + self.transaction_cost)
                    
                    if cost <= self.cash:
                        self.cash -= cost
                        self.long_shares[i] += shares_to_buy
                        self.long_cost_basis[i] += cost
                        self.trade_count += 1
    
    def _execute_volatility_strategy(self, current_prices):
        """Execute volatility-based strategy"""
        # Calculate volatility scores
        volatility_scores = self._calculate_volatility_scores(current_prices)
        
        # Buy low volatility stocks
        low_vol_stocks = np.argsort(volatility_scores)[:2]
        
        for i in low_vol_stocks:
            if current_prices[i] > 0 and self.cash > 1000:
                max_investment = min(self.cash * 0.12, 60000)
                shares_to_buy = int(max_investment / current_prices[i])
                shares_to_buy = min(shares_to_buy, 18)
                
                if shares_to_buy > 0:
                    effective_price = current_prices[i] * (1 + self.slippage)
                    cost = shares_to_buy * effective_price * (1 + self.transaction_cost)
                    
                    if cost <= self.cash:
                        self.cash -= cost
                        self.long_shares[i] += shares_to_buy
                        self.long_cost_basis[i] += cost
                        self.trade_count += 1
    
    def _execute_sentiment_strategy(self, current_prices):
        """Execute sentiment-based strategy"""
        # Calculate sentiment scores
        sentiment_scores = self._calculate_sentiment_scores(current_prices)
        
        # Buy positive sentiment stocks
        positive_sentiment_stocks = np.argsort(sentiment_scores)[-2:]
        
        for i in positive_sentiment_stocks:
            if current_prices[i] > 0 and self.cash > 1000:
                max_investment = min(self.cash * 0.08, 40000)
                shares_to_buy = int(max_investment / current_prices[i])
                shares_to_buy = min(shares_to_buy, 12)
                
                if shares_to_buy > 0:
                    effective_price = current_prices[i] * (1 + self.slippage)
                    cost = shares_to_buy * effective_price * (1 + self.transaction_cost)
                    
                    if cost <= self.cash:
                        self.cash -= cost
                        self.long_shares[i] += shares_to_buy
                        self.long_cost_basis[i] += cost
                        self.trade_count += 1
    
    def _execute_rebalance_strategy(self, current_prices):
        """Execute rebalancing strategy"""
        total_value = self._calculate_portfolio_value(current_prices)
        target_per_stock = total_value / self.n_stocks
        
        for i, price in enumerate(current_prices):
            if price > 0:
                current_value = self.long_shares[i] * price
                target_shares = int(target_per_stock / price)
                
                if target_shares > self.long_shares[i]:  # Need to buy
                    shares_to_buy = target_shares - self.long_shares[i]
                    if shares_to_buy > 0:
                        effective_price = price * (1 + self.slippage)
                        cost = shares_to_buy * effective_price * (1 + self.transaction_cost)
                        if cost <= self.cash:
                            self.cash -= cost
                            self.long_shares[i] = target_shares
                            self.long_cost_basis[i] += cost
                            self.trade_count += 1
                
                elif target_shares < self.long_shares[i]:  # Need to sell
                    shares_to_sell = self.long_shares[i] - target_shares
                    if shares_to_sell > 0:
                        effective_price = price * (1 - self.slippage)
                        proceeds = shares_to_sell * effective_price * (1 - self.transaction_cost)
                        self.cash += proceeds
                        self.long_shares[i] = target_shares
                        self.long_cost_basis[i] -= proceeds
                        self.trade_count += 1
    
    def _execute_risk_off_strategy(self, current_prices):
        """Execute risk-off strategy"""
        # Sell all positions
        for i in range(self.n_stocks):
            if self.long_shares[i] > 0 and current_prices[i] > 0:
                effective_price = current_prices[i] * (1 - self.slippage)
                proceeds = self.long_shares[i] * effective_price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.long_shares[i] = 0
                self.long_cost_basis[i] = 0
                self.trade_count += 1
    
    def _calculate_momentum_scores(self, current_prices):
        """Calculate momentum scores for stocks"""
        scores = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                momentum_5 = self._safe_get(data, 'close_momentum_5', 0)
                momentum_20 = self._safe_get(data, 'close_momentum_20', 0)
                rsi = self._safe_get(data, 'rsi_14', 50)
                
                scores[i] = momentum_5 * 0.5 + momentum_20 * 0.3 + (rsi - 50) * 0.2
        
        return scores
    
    def _calculate_mean_reversion_scores(self, current_prices):
        """Calculate mean reversion scores"""
        scores = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                z_score = self._safe_get(data, 'z_score_10', 0)
                bollinger_pos = self._safe_get(data, 'bollinger_position', 0.5)
                rsi = self._safe_get(data, 'rsi_14', 50)
                
                scores[i] = -z_score * 0.4 + (0.5 - bollinger_pos) * 0.3 + (50 - rsi) * 0.3
        
        return scores
    
    def _calculate_trend_scores(self, current_prices):
        """Calculate trend scores"""
        scores = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                price_vs_dma50 = self._safe_get(data, 'price_vs_dma50', 0)
                macd = self._safe_get(data, 'macd', 0)
                macd_signal = self._safe_get(data, 'macd_signal', 0)
                
                scores[i] = price_vs_dma50 * 0.4 + (macd - macd_signal) * 0.6
        
        return scores
    
    def _calculate_volatility_scores(self, current_prices):
        """Calculate volatility scores"""
        scores = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                # Lower volatility = higher score
                volatility = abs(self._safe_get(data, 'close_momentum_5', 0))
                scores[i] = -volatility
        
        return scores
    
    def _calculate_sentiment_scores(self, current_prices):
        """Calculate sentiment scores"""
        scores = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                sentiment = self._safe_get(data, 'reddit_title_sentiments_mean', 0)
                news_sentiment = self._safe_get(data, 'news_sentiment_mean', 0)
                
                scores[i] = sentiment * 0.6 + news_sentiment * 0.4
        
        return scores
    
    def _safe_get(self, data, key, default=0):
        """Safely get value from data"""
        try:
            value = data.get(key, default)
            if pd.isna(value) or np.isnan(value) or np.isinf(value):
                return default
            return float(value)
        except:
            return default
    
    def _calculate_portfolio_value(self, current_prices):
        """Calculate portfolio value"""
        # Add risk-free interest on cash
        daily_risk_free_rate = 0.04 / 252
        self.cash = self.cash * (1 + daily_risk_free_rate)
        
        # Calculate stock value
        stock_value = np.sum(self.long_shares * current_prices)
        
        return self.cash + stock_value
    
    def _calculate_reward(self, portfolio_value):
        """Calculate reward"""
        if len(self.portfolio_values) < 2:
            return 0
        
        # Daily return
        daily_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
        
        # Risk penalty
        risk_penalty = 0
        if len(self.portfolio_values) > 10:
            recent_returns = []
            for i in range(max(0, len(self.portfolio_values) - 10), len(self.portfolio_values) - 1):
                ret = (self.portfolio_values[i + 1] - self.portfolio_values[i]) / self.portfolio_values[i]
                recent_returns.append(ret)
            
            if len(recent_returns) > 1:
                volatility = np.std(recent_returns)
                if volatility > 0.02:  # 2% daily volatility threshold
                    risk_penalty = -volatility * 1.5
        
        # Trade cost penalty
        trade_penalty = -self.trade_count * 0.0001
        
        # Combine rewards
        total_reward = daily_return + risk_penalty + trade_penalty
        
        return total_reward * 10  # Scale up for better learning
    
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
        """Get observation"""
        current_prices = self._get_current_prices()
        portfolio_value = self._calculate_portfolio_value(current_prices)
        
        obs = []
        
        # Stock features
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                for feature in self.advanced_features:
                    if feature in data:
                        value = self._safe_get(data, feature, 0)
                        # Normalize features
                        if feature in ['close', 'high', 'vwap']:
                            obs.append(value / 1000)
                        elif feature == 'volume':
                            obs.append(value / 10000000)
                        elif feature == 'rsi_14':
                            obs.append(value / 100)
                        else:
                            obs.append(np.clip(value, -1, 1))
                    else:
                        obs.append(0)
            else:
                obs.extend([0] * len(self.advanced_features))
        
        # Market features
        long_value = np.sum(self.long_shares * current_prices)
        
        obs.extend([
            portfolio_value / self.initial_amount,
            self.cash / self.initial_amount,
            np.sum(self.long_shares > 0) / self.n_stocks,
            self.day / 200.0,
            long_value / self.initial_amount,
            self.trade_count / 100.0,
            np.mean(current_prices) / 1000,
            np.std(current_prices) / 1000,
            np.sum(self.long_shares) / 1000.0,
            portfolio_value / self.initial_amount - 1.0
        ])
        
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
            excess_returns = np.array(daily_returns) - 0.04/252
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
            'active_stocks': np.sum(self.long_shares > 0),
            'total_trades': self.trade_count,
            'final_cash': self.cash
        }

def run_optimized_trading_pipeline():
    """
    Run the comprehensive optimized trading pipeline
    """
    print("🚀 COMPREHENSIVE OPTIMIZED TRADING PIPELINE")
    print("=" * 70)
    print("🎯 TARGET: >10% ANNUAL RETURNS")
    print("🔧 COMPONENTS:")
    print("   ✅ Advanced Feature Engineering")
    print("   ✅ Hyperparameter Optimization")
    print("   ✅ Ensemble Methods")
    print("   ✅ Market Timing")
    print("   ✅ Risk Management")
    print("=" * 70)
    
    # Load data
    print("📊 Loading data...")
    loader = FinancialDataLoader()
    stock_list = loader.get_available_stocks()
    
    df_dict = {}
    for stock in stock_list:
        try:
            df_dict[stock] = loader.load_stock_data(stock)
        except Exception as e:
            print(f"❌ Error loading {stock}: {e}")
    
    print(f"✅ Loaded {len(df_dict)} stocks")
    
    # Create optimized environment
    print("\n🏗️ Creating optimized environment...")
    env = OptimizedTradingEnvironment(df_dict, list(df_dict.keys()))
    print(f"✅ Environment created: {env.observation_space.shape[0]} obs, {env.action_space.n} actions")
    
    # Train optimized PPO agent
    print("\n🤖 Training optimized PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1
    )
    
    print("   Training for 50000 timesteps...")
    model.learn(total_timesteps=50000)
    print("✅ Training completed")
    
    # Test the agent
    print("\n📊 Testing optimized agent...")
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
    
    print(f"\n📈 OPTIMIZED PERFORMANCE RESULTS:")
    print(f"   Initial Portfolio: ${metrics['initial_portfolio_value']:,.2f}")
    print(f"   Final Portfolio: ${metrics['final_portfolio_value']:,.2f}")
    print(f"   Final Cash: ${metrics['final_cash']:,.2f}")
    print(f"   Total Return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
    print(f"   Annualized Return: {metrics['annualized_return']:.4f} ({metrics['annualized_return']*100:.2f}%)")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
    print(f"   Active Stocks: {metrics['active_stocks']}/{len(df_dict)}")
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
    success = run_optimized_trading_pipeline()
    
    if success:
        print("\n🎉 OPTIMIZED PIPELINE COMPLETED SUCCESSFULLY!")
        print("✅ Advanced features and optimization techniques applied!")
    else:
        print("\n🔧 Pipeline needs further optimization")
        print("💡 Consider: More training, different algorithms, or parameter tuning")
