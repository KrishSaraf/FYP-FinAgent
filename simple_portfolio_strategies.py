"""
Simple Portfolio Strategies Implementation

This script implements various non-RL, non-neural network portfolio strategies
that output portfolio weights compatible with the existing portfolio environment.

Strategies implemented:
1. Equal Weight Portfolio
2. Volatility-Adjusted Equal Weight Portfolio  
3. Moving Average Crossover Strategy
4. Momentum-Based Portfolio
5. Black-Litterman Model
6. Minimum Variance Portfolio
7. Average of Technical Analysis Models (RSI, Bollinger Bands, MACD)
8. Average of Sentiment-Based Models (News, Social Media)

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePortfolioStrategies:
    """
    Implementation of various simple portfolio strategies that output portfolio weights.
    Compatible with the existing portfolio environment structure.
    """
    
    def __init__(self, 
                 data_root: str = "processed_data/",
                 stocks: Optional[List[str]] = None,
                 start_date: str = '2024-06-06',
                 end_date: str = '2025-03-06',
                 transaction_cost_rate: float = 0.005,
                 risk_free_rate: float = 0.04):
        """
        Initialize the portfolio strategies.
        
        Args:
            data_root: Path to processed data directory
            stocks: List of stock symbols (if None, loads from directory)
            start_date: Start date for backtesting
            end_date: End date for backtesting
            transaction_cost_rate: Transaction cost rate (0.5% default)
            risk_free_rate: Annual risk-free rate (4% default)
        """
        self.data_root = Path(data_root)
        self.start_date = start_date
        self.end_date = end_date
        self.transaction_cost_rate = transaction_cost_rate
        self.risk_free_rate_daily = risk_free_rate / 252.0
        
        # Load stock list
        if stocks is None:
            self.stocks = self._load_stock_list()
        else:
            self.stocks = stocks
            
        self.n_stocks = len(self.stocks)
        logger.info(f"Loaded {self.n_stocks} stocks: {self.stocks[:5]}...")
        
        # Load market data
        self.data = self._load_market_data()
        self.market_data = self._load_market_features()
        
        # Portfolio state tracking
        self.portfolio_history = []
        self.current_weights = None
        
    def _load_stock_list(self) -> List[str]:
        """Load stock list from processed data directory."""
        csv_files = list(self.data_root.glob("*_aligned.csv"))
        stocks = [f.stem.replace('_aligned', '') for f in csv_files]
        return sorted(stocks)
    
    def _load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for all stocks."""
        data = {}
        for stock in self.stocks:
            csv_path = self.data_root / f"{stock}_aligned.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
                df = df.loc[self.start_date:self.end_date]
                data[stock] = df
            else:
                logger.warning(f"Data file not found for {stock}")
        return data
    
    def _load_market_features(self) -> pd.DataFrame:
        """Load market-level features."""
        market_path = self.data_root / "market_features.csv"
        if market_path.exists():
            df = pd.read_csv(market_path, parse_dates=[0], index_col=0)
            return df.loc[self.start_date:self.end_date]
        else:
            logger.warning("Market features file not found")
            return pd.DataFrame()
    
    def _get_price_data(self, date: str) -> np.ndarray:
        """Get price data for all stocks on a given date."""
        prices = []
        for stock in self.stocks:
            if stock in self.data and date in self.data[stock].index:
                try:
                    # Get the row for this date
                    df_row = self.data[stock].loc[date]
                    
                    # Handle both Series and scalar cases
                    if hasattr(df_row, 'close'):
                        price_val = df_row['close']
                    else:
                        # If df_row is a Series, get the close value
                        price_val = df_row['close'] if 'close' in df_row.index else np.nan
                    
                    # Convert to float, handling Series case
                    if hasattr(price_val, 'iloc'):
                        price_val = price_val.iloc[0] if len(price_val) > 0 else np.nan
                    
                    if pd.isna(price_val):
                        prices.append(np.nan)
                    else:
                        prices.append(float(price_val))
                except Exception as e:
                    logger.warning(f"Error getting price for {stock} on {date}: {e}")
                    prices.append(np.nan)
            else:
                prices.append(np.nan)
        return np.array(prices)
    
    def _get_returns_data(self, window: int = 20) -> np.ndarray:
        """Get returns data for all stocks over a window."""
        returns_data = []
        for stock in self.stocks:
            if stock in self.data:
                df = self.data[stock]
                returns = df['close'].pct_change().dropna()
                if len(returns) >= window:
                    returns_data.append(returns.tail(window).values)
                else:
                    # Pad with zeros if not enough data
                    padded_returns = np.zeros(window)
                    padded_returns[-len(returns):] = returns.values
                    returns_data.append(padded_returns)
            else:
                returns_data.append(np.zeros(window))
        
        # Ensure all arrays have the same length
        min_length = min(len(arr) for arr in returns_data)
        returns_data = [arr[:min_length] for arr in returns_data]
        
        return np.array(returns_data)
    
    def _get_volatility_data(self, window: int = 20) -> np.ndarray:
        """Get volatility data for all stocks."""
        volatilities = []
        for stock in self.stocks:
            if stock in self.data:
                df = self.data[stock]
                returns = df['close'].pct_change().dropna()
                vol = returns.rolling(window).std().iloc[-1]
                volatilities.append(vol if not pd.isna(vol) else 0.01)
            else:
                volatilities.append(0.01)
        return np.array(volatilities)
    
    def _get_technical_indicators(self, date: str) -> Dict[str, np.ndarray]:
        """Get technical indicators for all stocks on a given date."""
        indicators = {
            'rsi': [],
            'bb_position': [],
            'macd_signal': []
        }
        
        for stock in self.stocks:
            if stock in self.data and date in self.data[stock].index:
                df = self.data[stock].loc[:date]
                
                # RSI calculation
                if 'rsi_14' in df.columns:
                    rsi = df['rsi_14'].iloc[-1]
                else:
                    # Calculate RSI if not available
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                indicators['rsi'].append(rsi)
                
                # Bollinger Bands position
                if len(df) >= 20:
                    bb_middle = df['close'].rolling(20).mean().iloc[-1]
                    bb_std = df['close'].rolling(20).std().iloc[-1]
                    bb_upper = bb_middle + 2 * bb_std
                    bb_lower = bb_middle - 2 * bb_std
                    current_price = df['close'].iloc[-1]
                    bb_pos = (current_price - bb_lower) / (bb_upper - bb_lower)
                    indicators['bb_position'].append(bb_pos if not pd.isna(bb_pos) else 0.5)
                else:
                    indicators['bb_position'].append(0.5)
                
                # MACD signal (simplified)
                if len(df) >= 26:
                    ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
                    ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
                    macd = ema_12 - ema_26
                    indicators['macd_signal'].append(1 if macd > 0 else -1)
                else:
                    indicators['macd_signal'].append(0)
            else:
                indicators['rsi'].append(50)
                indicators['bb_position'].append(0.5)
                indicators['macd_signal'].append(0)
        
        return {k: np.array(v) for k, v in indicators.items()}
    
    def _get_sentiment_data(self, date: str) -> Dict[str, np.ndarray]:
        """Get sentiment data for all stocks on a given date."""
        sentiments = {
            'news_sentiment': [],
            'reddit_sentiment': []
        }
        
        for stock in self.stocks:
            if stock in self.data and date in self.data[stock].index:
                try:
                    df_row = self.data[stock].loc[date]
                    
                    # News sentiment - handle both Series and scalar cases
                    if 'news_sentiment_mean' in df_row.index:
                        news_val = df_row['news_sentiment_mean']
                        if pd.isna(news_val):
                            sentiments['news_sentiment'].append(0.0)
                        else:
                            sentiments['news_sentiment'].append(float(news_val))
                    else:
                        sentiments['news_sentiment'].append(0.0)
                    
                    # Reddit sentiment - handle both Series and scalar cases
                    if 'reddit_title_sentiments_mean' in df_row.index:
                        reddit_val = df_row['reddit_title_sentiments_mean']
                        if pd.isna(reddit_val):
                            sentiments['reddit_sentiment'].append(0.0)
                        else:
                            sentiments['reddit_sentiment'].append(float(reddit_val))
                    else:
                        sentiments['reddit_sentiment'].append(0.0)
                        
                except Exception as e:
                    logger.warning(f"Error getting sentiment data for {stock} on {date}: {e}")
                    sentiments['news_sentiment'].append(0.0)
                    sentiments['reddit_sentiment'].append(0.0)
            else:
                sentiments['news_sentiment'].append(0.0)
                sentiments['reddit_sentiment'].append(0.0)
        
        return {k: np.array(v) for k, v in sentiments.items()}
    
    def equal_weight_strategy(self) -> np.ndarray:
        """Equal weight portfolio strategy."""
        weights = np.ones(self.n_stocks) / self.n_stocks
        return weights
    
    def volatility_adjusted_equal_weight_strategy(self) -> np.ndarray:
        """Volatility-adjusted equal weight portfolio strategy."""
        volatilities = self._get_volatility_data()
        
        # Avoid division by zero
        volatilities = np.maximum(volatilities, 0.001)
        
        # Inverse volatility weights
        inv_vol_weights = 1.0 / volatilities
        weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        return weights
    
    def ma_crossover_strategy(self, date: str, short_window: int = 10, long_window: int = 20) -> np.ndarray:
        """Moving average crossover strategy."""
        weights = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stocks):
            if stock in self.data and date in self.data[stock].index:
                try:
                    df = self.data[stock].loc[:date]
                    
                    if len(df) >= long_window:
                        short_ma = df['close'].rolling(short_window).mean().iloc[-1]
                        long_ma = df['close'].rolling(long_window).mean().iloc[-1]
                        
                        # Signal: 1 if short MA > long MA, 0 otherwise
                        if not pd.isna(short_ma) and not pd.isna(long_ma) and short_ma > long_ma:
                            weights[i] = 1.0
                except Exception as e:
                    logger.warning(f"Error in MA crossover for {stock} on {date}: {e}")
                    continue
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Fallback to equal weights if no signals
            weights = np.ones(self.n_stocks) / self.n_stocks
        
        return weights
    
    def momentum_strategy(self, date: str, lookback_window: int = 20) -> np.ndarray:
        """Momentum-based portfolio strategy."""
        weights = np.zeros(self.n_stocks)
        
        for i, stock in enumerate(self.stocks):
            if stock in self.data and date in self.data[stock].index:
                try:
                    df = self.data[stock].loc[:date]
                    
                    if len(df) >= lookback_window:
                        # Calculate momentum as return over lookback period
                        momentum = (df['close'].iloc[-1] / df['close'].iloc[-lookback_window]) - 1
                        
                        # Only positive momentum gets weight
                        if not pd.isna(momentum) and momentum > 0:
                            weights[i] = momentum
                except Exception as e:
                    logger.warning(f"Error in momentum for {stock} on {date}: {e}")
                    continue
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Fallback to equal weights if no momentum
            weights = np.ones(self.n_stocks) / self.n_stocks
        
        return weights
    
    def black_litterman_strategy(self, date: str, confidence_level: float = 0.25) -> np.ndarray:
        """Black-Litterman portfolio strategy."""
        try:
            # Get returns data
            returns_data = self._get_returns_data(window=60)  # Use 60 days for estimation
            
            # Check if we have valid data
            if np.all(np.isnan(returns_data)) or np.all(returns_data == 0):
                logger.warning("No valid returns data for Black-Litterman strategy")
                return self.equal_weight_strategy()
            
            # Clean the data - replace NaN with 0
            returns_data = np.nan_to_num(returns_data, nan=0.0)
            
            # Calculate market equilibrium returns (implied returns)
            # Using equal weight as market portfolio proxy
            market_weights = np.ones(self.n_stocks) / self.n_stocks
            
            # Estimate covariance matrix using Ledoit-Wolf shrinkage
            cov_estimator = LedoitWolf()
            cov_matrix = cov_estimator.fit(returns_data.T).covariance_
            
            # Check if covariance matrix is valid
            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                logger.warning("Invalid covariance matrix for Black-Litterman strategy")
                return self.equal_weight_strategy()
            
            # Risk aversion parameter (lambda)
            risk_aversion = 3.0
            
            # Market equilibrium returns
            pi = risk_aversion * np.dot(cov_matrix, market_weights)
            
            # Views (simplified: momentum-based views)
            momentum_returns = self.momentum_strategy(date)
            views = momentum_returns * 0.1  # Scale down views
            
            # View confidence matrix (diagonal)
            omega = np.diag(np.diag(cov_matrix)) * confidence_level
            
            # Black-Litterman formula
            tau = 0.05  # Scaling factor
            
            # P matrix (identity for absolute views)
            P = np.eye(self.n_stocks)
            
            # Q vector (views)
            Q = views
            
            # Black-Litterman expected returns
            M1 = np.linalg.inv(tau * cov_matrix)
            M2 = np.dot(P.T, np.linalg.inv(omega))
            M3 = np.dot(M2, P)
            M4 = np.dot(M2, Q)
            
            expected_returns = np.linalg.inv(M1 + M3).dot(np.dot(M1, pi) + M4)
            
            # Check if expected returns are valid
            if np.any(np.isnan(expected_returns)) or np.any(np.isinf(expected_returns)):
                logger.warning("Invalid expected returns for Black-Litterman strategy")
                return self.equal_weight_strategy()
            
            # Optimize portfolio weights
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(self.n_stocks)]
            
            # Initial guess
            x0 = np.ones(self.n_stocks) / self.n_stocks
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success and not np.any(np.isnan(result.x)):
                # Normalize weights to ensure they sum to 1
                weights = result.x / np.sum(result.x)
                return weights
            else:
                logger.warning(f"Black-Litterman optimization failed: {result.message}")
                return self.equal_weight_strategy()
                
        except Exception as e:
            logger.warning(f"Black-Litterman strategy failed: {e}, using equal weights")
            return self.equal_weight_strategy()
    
    def minimum_variance_strategy(self, date: str) -> np.ndarray:
        """Minimum variance portfolio strategy."""
        try:
            # Get returns data
            returns_data = self._get_returns_data(window=60)
            
            # Check if we have valid data
            if np.all(np.isnan(returns_data)) or np.all(returns_data == 0):
                logger.warning("No valid returns data for minimum variance strategy")
                return self.equal_weight_strategy()
            
            # Clean the data - replace NaN with 0
            returns_data = np.nan_to_num(returns_data, nan=0.0)
            
            # Calculate individual volatilities first
            individual_vols = np.std(returns_data, axis=1)
            
            # If volatilities are too similar, add some differentiation
            if np.max(individual_vols) - np.min(individual_vols) < 0.001:
                # Add small random noise to create differentiation
                np.random.seed(42)  # For reproducibility
                individual_vols += np.random.normal(0, 0.001, len(individual_vols))
                individual_vols = np.maximum(individual_vols, 0.001)  # Ensure positive
            
            # Use a simpler covariance matrix estimation
            # Create diagonal covariance matrix based on individual volatilities
            cov_matrix = np.diag(individual_vols ** 2)
            
            # Add some correlation structure (simplified)
            # Use average correlation of 0.3
            avg_correlation = 0.3
            for i in range(self.n_stocks):
                for j in range(i+1, self.n_stocks):
                    cov_value = avg_correlation * individual_vols[i] * individual_vols[j]
                    cov_matrix[i, j] = cov_value
                    cov_matrix[j, i] = cov_value
            
            # Check if covariance matrix is valid
            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                logger.warning("Invalid covariance matrix for minimum variance strategy")
                return self.equal_weight_strategy()
            
            # Add regularization to ensure matrix is invertible
            cov_matrix += np.eye(self.n_stocks) * 1e-6
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(self.n_stocks)]
            
            # Initial guess - try inverse volatility weights
            inv_vol_weights = 1.0 / individual_vols
            inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
            x0 = inv_vol_weights
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success and not np.any(np.isnan(result.x)):
                # Normalize weights to ensure they sum to 1
                weights = result.x / np.sum(result.x)
                return weights
            else:
                logger.warning(f"Minimum variance optimization failed: {result.message}")
                return self.equal_weight_strategy()
                
        except Exception as e:
            logger.warning(f"Minimum variance strategy failed: {e}, using equal weights")
            return self.equal_weight_strategy()
    
    def technical_analysis_strategy(self, date: str) -> np.ndarray:
        """Combined technical analysis strategy (RSI + Bollinger Bands + MACD)."""
        try:
            indicators = self._get_technical_indicators(date)
            
            weights = np.zeros(self.n_stocks)
            
            for i in range(self.n_stocks):
                rsi = indicators['rsi'][i]
                bb_pos = indicators['bb_position'][i]
                macd = indicators['macd_signal'][i]
                
                # Check for NaN values
                if pd.isna(rsi) or pd.isna(bb_pos) or pd.isna(macd):
                    continue
                
                # RSI signal: buy when oversold (< 30), sell when overbought (> 70)
                rsi_signal = 0
                if rsi < 30:
                    rsi_signal = 1  # Buy signal
                elif rsi > 70:
                    rsi_signal = -1  # Sell signal
                
                # Bollinger Bands signal: buy when near lower band, sell when near upper band
                bb_signal = 0
                if bb_pos < 0.2:
                    bb_signal = 1  # Buy signal
                elif bb_pos > 0.8:
                    bb_signal = -1  # Sell signal
                
                # MACD signal: already calculated as 1 or -1
                
                # Combine signals (simple average)
                combined_signal = (rsi_signal + bb_signal + macd) / 3
                
                # Only positive signals get weight
                if combined_signal > 0:
                    weights[i] = combined_signal
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                # Fallback to equal weights if no signals
                weights = np.ones(self.n_stocks) / self.n_stocks
            
            return weights
        except Exception as e:
            logger.warning(f"Technical analysis strategy failed: {e}")
            return np.ones(self.n_stocks) / self.n_stocks
    
    def sentiment_strategy(self, date: str) -> np.ndarray:
        """Combined sentiment-based strategy (News + Social Media)."""
        try:
            sentiments = self._get_sentiment_data(date)
            
            weights = np.zeros(self.n_stocks)
            
            for i in range(self.n_stocks):
                news_sentiment = sentiments['news_sentiment'][i]
                reddit_sentiment = sentiments['reddit_sentiment'][i]
                
                # Check for NaN values
                if pd.isna(news_sentiment) or pd.isna(reddit_sentiment):
                    continue
                
                # Combine sentiments (simple average)
                combined_sentiment = (news_sentiment + reddit_sentiment) / 2
                
                # Only positive sentiment gets weight
                if combined_sentiment > 0:
                    weights[i] = combined_sentiment
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                # Fallback to equal weights if no positive sentiment
                weights = np.ones(self.n_stocks) / self.n_stocks
            
            return weights
        except Exception as e:
            logger.warning(f"Sentiment strategy failed: {e}")
            return np.ones(self.n_stocks) / self.n_stocks
    
    def backtest_strategy(self, strategy_func, strategy_name: str) -> Dict:
        """Backtest a strategy and return performance metrics."""
        logger.info(f"Backtesting {strategy_name} strategy...")
        
        # Get all dates
        all_dates = []
        for stock in self.stocks:
            if stock in self.data:
                all_dates.extend(self.data[stock].index.tolist())
        
        unique_dates = sorted(list(set(all_dates)))
        
        portfolio_values = [1.0]  # Start with normalized value of 1
        returns = []
        weights_history = []
        
        for i, date in enumerate(unique_dates[1:], 1):  # Skip first date
            try:
                # Get portfolio weights
                if strategy_name in ['equal_weight', 'volatility_adjusted_equal_weight']:
                    weights = strategy_func()
                else:
                    weights = strategy_func(date)
                
                # Ensure weights is a valid numpy array
                if not isinstance(weights, np.ndarray) or len(weights) != self.n_stocks:
                    logger.warning(f"Invalid weights on {date}, using equal weights")
                    weights = np.ones(self.n_stocks) / self.n_stocks
                
                # Ensure weights sum to 1 and are non-negative
                weights = np.maximum(weights, 0.0)  # Ensure non-negative
                weight_sum = np.sum(weights)
                if weight_sum > 0:
                    weights = weights / weight_sum  # Normalize to sum to 1
                else:
                    weights = np.ones(self.n_stocks) / self.n_stocks  # Fallback to equal weights
                
                weights_history.append(weights)
                
                # Get returns for this period
                current_prices = self._get_price_data(date)
                prev_prices = self._get_price_data(unique_dates[i-1])
                
                # Ensure we have valid arrays
                if len(current_prices) != self.n_stocks or len(prev_prices) != self.n_stocks:
                    logger.warning(f"Price data length mismatch on {date}, skipping")
                    returns.append(0.0)
                    portfolio_values.append(portfolio_values[-1])
                    weights_history.append(np.ones(self.n_stocks) / self.n_stocks)
                    continue
                
                # Calculate stock returns
                stock_returns = np.zeros(self.n_stocks)
                for j in range(self.n_stocks):
                    if not (np.isnan(current_prices[j]) or np.isnan(prev_prices[j])):
                        if prev_prices[j] > 0:  # Avoid division by zero
                            stock_returns[j] = (current_prices[j] / prev_prices[j]) - 1
                
                # Calculate portfolio return
                portfolio_return = np.dot(weights, stock_returns)
                
                # Apply transaction costs
                if i > 1:
                    prev_weights = weights_history[-2]
                    turnover = np.sum(np.abs(weights - prev_weights))
                    transaction_cost = turnover * self.transaction_cost_rate
                    portfolio_return -= transaction_cost
                
                returns.append(portfolio_return)
                
                # Update portfolio value
                new_value = portfolio_values[-1] * (1 + portfolio_return)
                portfolio_values.append(new_value)
                
            except Exception as e:
                logger.warning(f"Error processing date {date}: {e}")
                returns.append(0.0)
                portfolio_values.append(portfolio_values[-1])
                weights_history.append(np.ones(self.n_stocks) / self.n_stocks)
        
        # Calculate performance metrics
        returns = np.array(returns)
        portfolio_values = np.array(portfolio_values)
        
        total_return = portfolio_values[-1] - 1
        annualized_return = (portfolio_values[-1] ** (252 / len(returns))) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate_daily) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        results = {
            'strategy_name': strategy_name,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'returns': returns,
            'weights_history': weights_history,
            'dates': unique_dates[1:]
        }
        
        logger.info(f"{strategy_name} - Return: {total_return:.4f}, Sharpe: {sharpe_ratio:.4f}, Max DD: {max_drawdown:.4f}")
        
        return results
    
    def run_all_strategies(self) -> Dict[str, Dict]:
        """Run all strategies and return results."""
        strategies = {
            'equal_weight': self.equal_weight_strategy,
            'volatility_adjusted_equal_weight': self.volatility_adjusted_equal_weight_strategy,
            'ma_crossover': self.ma_crossover_strategy,
            'momentum': self.momentum_strategy,
            'black_litterman': self.black_litterman_strategy,
            'minimum_variance': self.minimum_variance_strategy,
            'technical_analysis': self.technical_analysis_strategy,
            'sentiment': self.sentiment_strategy
        }
        
        results = {}
        for name, strategy_func in strategies.items():
            results[name] = self.backtest_strategy(strategy_func, name)
        
        return results
    
    def save_results(self, results: Dict[str, Dict], output_dir: str = "evaluation_results/simple_strategy_results"):
        """Save backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save summary results
        summary_data = []
        for strategy_name, result in results.items():
            summary_data.append({
                'Strategy': strategy_name,
                'Total Return': result['total_return'],
                'Annualized Return': result['annualized_return'],
                'Volatility': result['volatility'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown': result['max_drawdown']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / "strategy_summary.csv", index=False)
        
        # Save detailed results for each strategy
        for strategy_name, result in results.items():
            strategy_df = pd.DataFrame({
                'Date': result['dates'],
                'Portfolio Value': result['portfolio_values'][1:],  # Skip initial value
                'Returns': result['returns']
            })
            strategy_df.to_csv(output_path / f"{strategy_name}_detailed.csv", index=False)
        
        logger.info(f"Results saved to {output_path}")
        
        return summary_df

def main():
    """Main function to run all strategies."""
    logger.info("Starting Simple Portfolio Strategies Backtest")
    
    # Initialize strategies
    strategies = SimplePortfolioStrategies(
        data_root="processed_data/",
        start_date='2024-06-06',
        end_date='2025-03-06'
    )
    
    # Run all strategies
    results = strategies.run_all_strategies()
    
    # Save results
    summary_df = strategies.save_results(results)
    
    # Print summary
    print("\n" + "="*80)
    print("SIMPLE PORTFOLIO STRATEGIES BACKTEST RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    # Find best performing strategy
    best_strategy = summary_df.loc[summary_df['Sharpe Ratio'].idxmax()]
    print(f"\nBest performing strategy: {best_strategy['Strategy']}")
    print(f"Sharpe Ratio: {best_strategy['Sharpe Ratio']:.4f}")
    print(f"Total Return: {best_strategy['Total Return']:.4f}")
    
    return results, summary_df

if __name__ == "__main__":
    results, summary = main()
