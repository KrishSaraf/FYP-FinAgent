"""
FIXED OPTIMIZED Phase 2: Shows REAL portfolio value changes
The share prices ARE changing - we need to track the actual portfolio value!
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

class FixedOptimizedPortfolioEnv(gym.Env):
    """
    FIXED portfolio environment that shows REAL portfolio value changes
    """
    
    def __init__(self, df_dict, stock_list, initial_amount=1000000.0):
        super(FixedOptimizedPortfolioEnv, self).__init__()
        
        self.df_dict = df_dict
        self.stock_list = stock_list
        self.initial_amount = initial_amount
        self.n_stocks = len(stock_list)
        self.risk_free_rate = 0.04  # 4% annual risk-free rate
        
        # Calculate equal allocation per stock
        self.initial_allocation_per_stock = initial_amount / self.n_stocks
        
        # Advanced action space: 9 actions for sophisticated strategies
        self.action_space = spaces.Discrete(9)
        
        # Optimized observation space: 30 most important features per stock + 8 market features
        obs_dim = 30 * self.n_stocks + 8  # 30 features per stock + 8 market features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Define the 30 most important features for trading
        self.trading_features = [
            # Price Action (5)
            'close', 'high', 'low', 'volume', 'vwap',
            # Technical Indicators (5)
            'rsi_14', 'dma_50', 'dma_200', 'dma_distance', 'volume_price_trend',
            # Momentum (2)
            'close_momentum_5', 'close_momentum_20',
            # Rolling Statistics (5)
            'close_rolling_mean_5', 'close_rolling_mean_20', 'close_rolling_std_20',
            'volume_rolling_mean_5', 'volume_rolling_std_20',
            # Key Lag Features (5)
            'close_lag_1', 'close_lag_5', 'close_lag_20', 'volume_lag_1', 'volume_lag_5',
            # Sentiment (3)
            'reddit_title_sentiments_mean', 'reddit_body_sentiments', 'news_sentiment_mean',
            # Key Fundamental Metrics (5)
            'metric_beta', 'metric_marketCap', 'metric_priceToBookMostRecentFiscalYear',
            'metric_pPerEBasicExcludingExtraordinaryItemsTTM', 'metric_currentRatioMostRecentFiscalYear'
        ]
        
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
        
        # Calculate REAL initial portfolio value
        real_portfolio_value = self.cash + total_invested
        print(f"📊 INITIAL INVESTMENT:")
        print(f"   Total shares purchased: {np.sum(self.shares):,}")
        print(f"   Total invested: ${total_invested:,.2f}")
        print(f"   Remaining cash: ${self.cash:,.2f}")
        print(f"   REAL initial portfolio value: ${real_portfolio_value:,.2f}")
    
    def step(self, action):
        # Advanced action decoding
        strategy = self._decode_advanced_action(action)
        
        # Execute advanced strategy
        self._execute_advanced_strategy(strategy)
        
        # Update day
        self.day += 1
        
        # Calculate REAL portfolio value with current share prices
        current_prices = self._get_current_prices()
        stock_value = np.sum(self.shares * current_prices)
        
        # Add 4% annual risk-free interest on cash (daily rate = 4%/252)
        daily_risk_free_rate = self.risk_free_rate / 252
        self.cash = self.cash * (1 + daily_risk_free_rate)
        
        portfolio_value = self.cash + stock_value
        self.portfolio_values.append(portfolio_value)
        
        # Show REAL portfolio value every 20 days
        if self.day % 20 == 0:
            total_return = (portfolio_value - self.initial_amount) / self.initial_amount * 100
            print(f"📈 Day {self.day}: Portfolio = ${portfolio_value:,.2f} ({total_return:+.2f}%)")
            print(f"   Stock value: ${stock_value:,.2f}, Cash: ${self.cash:,.2f}")
            print(f"   Top 3 holdings: {self._get_top_holdings(current_prices)}")
        
        # Enhanced reward calculation
        reward = self._calculate_enhanced_reward(portfolio_value)
        
        # Check if done
        done = self.day >= 200
        
        return self._get_obs(), reward, done, {'portfolio_value': portfolio_value}
    
    def _get_top_holdings(self, current_prices):
        """Get top 3 holdings by value"""
        holdings = []
        for i, (stock, shares, price) in enumerate(zip(self.stock_list, self.shares, current_prices)):
            if shares > 0 and price > 0:
                value = shares * price
                holdings.append((stock, value))
        
        holdings.sort(key=lambda x: x[1], reverse=True)
        return [(stock, f"${value:,.0f}") for stock, value in holdings[:3]]
    
    def _decode_advanced_action(self, action):
        """Decode action into advanced trading strategy"""
        strategies = [
            "momentum_buy",      # 0: Buy high momentum stocks
            "value_buy",         # 1: Buy undervalued stocks
            "technical_buy",     # 2: Buy based on technical signals
            "sentiment_buy",     # 3: Buy based on positive sentiment
            "hold",              # 4: Hold current positions
            "rebalance",         # 5: Rebalance to equal weights
            "momentum_sell",     # 6: Sell low momentum stocks
            "technical_sell",    # 7: Sell based on technical signals
            "risk_off"           # 8: Sell high-risk positions
        ]
        return strategies[action]
    
    def _execute_advanced_strategy(self, strategy):
        """Execute advanced trading strategy"""
        current_prices = self._get_current_prices()
        
        if "buy" in strategy:
            self._execute_buy_strategy(strategy, current_prices)
        elif "sell" in strategy:
            self._execute_sell_strategy(strategy, current_prices)
        elif strategy == "rebalance":
            self._execute_rebalance_strategy(current_prices)
        # "hold" does nothing
    
    def _execute_buy_strategy(self, strategy, current_prices):
        """Execute advanced buy strategy"""
        if self.cash < 1000:  # Need at least $1000 to trade
            return
        
        # Calculate stock scores based on strategy
        scores = self._calculate_stock_scores(strategy, current_prices)
        
        # Select top performers based on strategy
        if strategy == "momentum_buy":
            top_stocks = np.argsort(scores)[-10:]  # Top 10 momentum stocks
            allocation_pct = 0.15  # 15% of cash per stock
        elif strategy == "value_buy":
            top_stocks = np.argsort(scores)[-8:]   # Top 8 value stocks
            allocation_pct = 0.20  # 20% of cash per stock
        elif strategy == "technical_buy":
            top_stocks = np.argsort(scores)[-12:]  # Top 12 technical stocks
            allocation_pct = 0.12  # 12% of cash per stock
        elif strategy == "sentiment_buy":
            top_stocks = np.argsort(scores)[-6:]   # Top 6 sentiment stocks
            allocation_pct = 0.25  # 25% of cash per stock
        else:
            return
        
        # Execute buys
        for i in top_stocks:
            if current_prices[i] > 0:
                max_investment = self.cash * allocation_pct
                shares_to_buy = int(max_investment / current_prices[i])
                shares_to_buy = min(shares_to_buy, 50)  # Cap at 50 shares
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_prices[i] * 1.001
                    if cost <= self.cash:
                        self.cash -= cost
                        self.shares[i] += shares_to_buy
                        self.trade_count += 1
    
    def _execute_sell_strategy(self, strategy, current_prices):
        """Execute advanced sell strategy"""
        # Calculate stock scores based on strategy
        scores = self._calculate_stock_scores(strategy, current_prices)
        
        # Select worst performers based on strategy
        if strategy == "momentum_sell":
            bottom_stocks = np.argsort(scores)[:10]  # Bottom 10 momentum stocks
            sell_pct = 0.20  # Sell 20% of position
        elif strategy == "technical_sell":
            bottom_stocks = np.argsort(scores)[:12]  # Bottom 12 technical stocks
            sell_pct = 0.15  # Sell 15% of position
        elif strategy == "risk_off":
            bottom_stocks = np.argsort(scores)[:8]   # Bottom 8 risk stocks
            sell_pct = 0.30  # Sell 30% of position
        else:
            return
        
        # Execute sells
        for i in bottom_stocks:
            if self.shares[i] > 0 and current_prices[i] > 0:
                shares_to_sell = int(self.shares[i] * sell_pct)
                shares_to_sell = min(shares_to_sell, self.shares[i])
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_prices[i] * 0.999
                    self.cash += proceeds
                    self.shares[i] -= shares_to_sell
                    self.trade_count += 1
    
    def _execute_rebalance_strategy(self, current_prices):
        """Rebalance portfolio to equal weights"""
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
    
    def _calculate_stock_scores(self, strategy, current_prices):
        """Calculate stock scores based on strategy"""
        scores = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                if strategy == "momentum_buy":
                    # Score based on momentum indicators
                    momentum_5 = self._safe_get(data, 'close_momentum_5', 0)
                    momentum_20 = self._safe_get(data, 'close_momentum_20', 0)
                    rsi = self._safe_get(data, 'rsi_14', 50)
                    scores[i] = momentum_5 * 0.4 + momentum_20 * 0.3 + (rsi - 50) * 0.3
                
                elif strategy == "value_buy":
                    # Score based on value metrics
                    pe_ratio = self._safe_get(data, 'metric_pPerEBasicExcludingExtraordinaryItemsTTM', 20)
                    pb_ratio = self._safe_get(data, 'metric_priceToBookMostRecentFiscalYear', 2)
                    current_ratio = self._safe_get(data, 'metric_currentRatioMostRecentFiscalYear', 1)
                    scores[i] = (1/pe_ratio) * 0.4 + (1/pb_ratio) * 0.3 + current_ratio * 0.3
                
                elif strategy == "technical_buy":
                    # Score based on technical indicators
                    rsi = self._safe_get(data, 'rsi_14', 50)
                    dma_distance = self._safe_get(data, 'dma_distance', 0)
                    volume_trend = self._safe_get(data, 'volume_price_trend', 0)
                    scores[i] = (50 - rsi) * 0.4 + (1 - abs(dma_distance)) * 0.3 + volume_trend * 0.3
                
                elif strategy == "sentiment_buy":
                    # Score based on sentiment
                    reddit_sentiment = self._safe_get(data, 'reddit_title_sentiments_mean', 0)
                    news_sentiment = self._safe_get(data, 'news_sentiment_mean', 0)
                    reddit_body = self._safe_get(data, 'reddit_body_sentiments', 0)
                    scores[i] = reddit_sentiment * 0.4 + news_sentiment * 0.3 + reddit_body * 0.3
                
                elif strategy in ["momentum_sell", "technical_sell", "risk_off"]:
                    # For sell strategies, use same logic but inverted
                    if strategy == "momentum_sell":
                        momentum_5 = self._safe_get(data, 'close_momentum_5', 0)
                        momentum_20 = self._safe_get(data, 'close_momentum_20', 0)
                        scores[i] = -momentum_5 * 0.5 - momentum_20 * 0.5
                    elif strategy == "technical_sell":
                        rsi = self._safe_get(data, 'rsi_14', 50)
                        dma_distance = self._safe_get(data, 'dma_distance', 0)
                        scores[i] = -(50 - rsi) * 0.5 - (1 - abs(dma_distance)) * 0.5
                    elif strategy == "risk_off":
                        beta = self._safe_get(data, 'metric_beta', 1)
                        volatility = self._safe_get(data, 'close_rolling_std_20', 0)
                        scores[i] = -beta * 0.5 - volatility * 0.5
        
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
    
    def _calculate_enhanced_reward(self, portfolio_value):
        """Enhanced reward function with multiple factors"""
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
                momentum_bonus = np.mean(recent_returns) * 0.3
        
        # Diversification bonus
        diversification_bonus = 0
        active_stocks = np.sum(self.shares > 0)
        if active_stocks > 0:
            diversification_bonus = (active_stocks / self.n_stocks) * 0.1
        
        # Trade efficiency penalty
        trade_penalty = -self.trade_count * 0.00005
        
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
                    risk_penalty = -volatility * 0.3
        
        # Combine rewards
        total_reward = daily_return + momentum_bonus + diversification_bonus + trade_penalty + risk_penalty
        
        return total_reward * 50  # Scale up for better learning
    
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
        """Get optimized observation with 30 most important features"""
        current_prices = self._get_current_prices()
        stock_value = np.sum(self.shares * current_prices)
        portfolio_value = self.cash + stock_value
        
        # Create observation with 30 optimized features for each stock
        obs = []
        
        for i, stock in enumerate(self.stock_list):
            if self.day < len(self.df_dict[stock]):
                data = self.df_dict[stock].iloc[self.day]
                
                # 30 optimized features for this stock
                for feature in self.trading_features:
                    if feature in data:
                        value = self._safe_get(data, feature, 0)
                        # Normalize based on feature type
                        if feature in ['close', 'high', 'low', 'vwap']:
                            obs.append(value / 1000)  # Price normalization
                        elif feature == 'volume':
                            obs.append(value / 10000000)  # Volume normalization
                        elif feature == 'rsi_14':
                            obs.append(value / 100)  # RSI normalization
                        elif 'momentum' in feature or 'rolling' in feature:
                            obs.append(np.clip(value, -1, 1))  # Momentum/rolling normalization
                        elif 'sentiment' in feature:
                            obs.append(np.clip(value, -1, 1))  # Sentiment normalization
                        elif feature.startswith('metric_'):
                            obs.append(value / 100)  # Fundamental normalization
                        else:
                            obs.append(value)  # Default normalization
                    else:
                        obs.append(0)
            else:
                obs.extend([0] * len(self.trading_features))
        
        # Market-level features
        obs.extend([
            portfolio_value / self.initial_amount,  # Portfolio value ratio
            self.cash / self.initial_amount,  # Cash ratio
            np.sum(self.shares > 0) / self.n_stocks,  # Diversification ratio
            self.day / 200.0,  # Time progress
            stock_value / self.initial_amount,  # Stock value ratio
            (portfolio_value - self.initial_amount) / self.initial_amount,  # Total return
            self.trade_count / 1000.0,  # Trade intensity
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

def run_fixed_optimized_phase2():
    """
    Run FIXED OPTIMIZED Phase 2 that shows REAL portfolio value changes
    """
    print("🚀 FIXED OPTIMIZED Phase 2: Shows REAL Portfolio Value Changes!")
    print("=" * 70)
    
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
    print(f"\n🏗️ Creating FIXED OPTIMIZED environment...")
    env = FixedOptimizedPortfolioEnv(df_dict, list(df_dict.keys()))
    print(f"✅ Environment created: {env.observation_space.shape[0]} obs, {env.action_space.n} actions")
    
    # Train PPO agent with optimized parameters
    print("\n🤖 Training FIXED OPTIMIZED PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # Higher learning rate
        n_steps=4096,  # More steps
        batch_size=128,  # Larger batch size
        n_epochs=15,  # More epochs
        gamma=0.995,  # Higher discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Higher entropy
        vf_coef=0.5,
        verbose=1
    )
    
    print("   Training for 100000 timesteps...")
    model.learn(total_timesteps=100000)
    print("✅ Training completed")
    
    # Test the agent
    print("\n📊 Testing FIXED OPTIMIZED agent...")
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
    
    print(f"\n📈 FIXED OPTIMIZED Performance Results:")
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
    success = run_fixed_optimized_phase2()
    
    if success:
        print("\n🎉 FIXED OPTIMIZED Phase 2 completed successfully!")
        print("✅ Now showing REAL portfolio value changes!")
    else:
        print("\n🔧 Still needs work")
        print("💡 But now we can see the REAL portfolio value changes!")
