#!/usr/bin/env python3
"""
LLM Enhanced Portfolio Strategy - FIXED VERSION 2.0

Key Fix: Transaction costs only applied when positions actually change,
not every single day!
"""

import numpy as np
import pandas as pd
import requests
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import time
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import the fixed components from the previous version
from llm_enhanced_strategy_fixed import (
    LLMSignal, PortfolioMetrics, SentimentAnalyzer, 
    MistralAPIClient, RealDataLoader
)

logger = logging.getLogger(__name__)

class LLMEnhancedStrategyFixedV2:
    """Fixed version that only applies transaction costs when positions change"""
    
    def __init__(self, 
                 api_key: str,
                 stocks: List[str],
                 data_root: str = "processed_data/",
                 initial_capital: float = 1000000.0,
                 max_position_size: float = 0.05,
                 max_sector_exposure: float = 0.20,
                 transaction_cost: float = 0.0005):  # Reduced to 0.05% (5 basis points)
        
        self.api_key = api_key
        self.stocks = stocks
        self.data_root = data_root
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.transaction_cost = transaction_cost  # Now reasonable 0.05%
        
        # Initialize components
        self.mistral_client = MistralAPIClient(api_key)
        self.sentiment_analyzer = SentimentAnalyzer(data_root)
        self.data_loader = RealDataLoader(data_root, stocks)
        
        logger.info(f"Initialized Fixed LLM Enhanced Strategy V2 with {len(stocks)} stocks")
        logger.info(f"Transaction cost: {transaction_cost:.3%} (applied only when positions change)")
    
    def run_pure_llm_strategy(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run Pure LLM strategy with FIXED transaction cost application"""
        
        logger.info(f"Running Pure LLM Strategy V2 from {start_date} to {end_date}")
        
        # Load market data
        market_data = self.data_loader.load_market_data(start_date, end_date)
        
        if market_data.empty:
            logger.error("No market data available")
            return {"error": "No market data available"}
        
        # Initialize portfolio
        equal_weight = 1.0 / len(self.stocks)
        portfolio_values = [self.initial_capital]
        trade_log = []
        
        # Track previous positions to detect changes
        previous_positions = {}
        
        # Get trading dates (last 10 days for quick demo)
        trading_dates = market_data.index[-10:]
        
        for i, date in enumerate(trading_dates):
            try:
                current_capital = portfolio_values[-1]
                
                # Generate LLM signals for all stocks
                signals = {}
                for stock in self.stocks:
                    stock_data = self._get_stock_data(market_data, stock, date)
                    if stock_data:
                        market_context = self._get_market_context(market_data, date)
                        sentiment_data = self._get_sentiment_data(stock, date)
                        
                        signal = self.mistral_client.generate_signal(
                            stock_data, market_context, sentiment_data
                        )
                        signals[stock] = signal
                        
                        logger.info(f"{date.strftime('%Y-%m-%d')} {stock}: {signal.signal} ({signal.confidence}%)")
                        
                        # Add delay to respect API limits
                        time.sleep(0.5)
                
                # Calculate current positions based on signals
                current_positions = {}
                for stock in self.stocks:
                    if stock in signals:
                        signal = signals[stock]
                        
                        if signal.signal == 'BUY':
                            position = equal_weight  # Full position
                        elif signal.signal == 'SELL':
                            position = 0.0  # No position
                        else:  # HOLD
                            position = equal_weight * 0.5  # Half position
                        
                        current_positions[stock] = position
                    else:
                        current_positions[stock] = equal_weight * 0.5  # Default to half
                
                # Calculate portfolio return from market movements
                portfolio_return = 0.0
                for stock in self.stocks:
                    if f"{stock}_returns_1d" in market_data.columns:
                        stock_return = market_data.loc[date, f"{stock}_returns_1d"]
                        position = current_positions.get(stock, 0.0)
                        portfolio_return += position * stock_return
                
                # Check if positions changed (transaction costs only if they did)
                positions_changed = (previous_positions != current_positions) and i > 0
                
                if positions_changed:
                    # Calculate the magnitude of position changes
                    total_change = 0.0
                    for stock in self.stocks:
                        old_pos = previous_positions.get(stock, 0.0)
                        new_pos = current_positions.get(stock, 0.0)
                        total_change += abs(new_pos - old_pos)
                    
                    # Apply transaction costs proportional to changes
                    transaction_drag = self.transaction_cost * total_change
                    portfolio_return -= transaction_drag
                    
                    logger.info(f"  Position changes detected: {total_change:.2f}, Transaction cost: {transaction_drag:.4f} ({transaction_drag:.3%})")
                else:
                    transaction_drag = 0.0
                    logger.info(f"  No position changes: No transaction costs")
                
                # Update portfolio value
                new_value = current_capital * (1 + portfolio_return)
                portfolio_values.append(new_value)
                
                trade_log.append({
                    'date': date,
                    'portfolio_value': new_value,
                    'daily_return': portfolio_return,
                    'transaction_costs': transaction_drag,
                    'positions_changed': positions_changed,
                    'current_positions': current_positions.copy(),
                    'signals': {stock: signal.signal for stock, signal in signals.items()}
                })
                
                # Update previous positions
                previous_positions = current_positions.copy()
                
                daily_return_pct = portfolio_return * 100
                logger.info(f"Pure LLM V2 - {date.strftime('%Y-%m-%d')}: ₹{new_value:,.0f} ({daily_return_pct:+.2f}%)")
                
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue
        
        # Calculate final metrics
        returns = pd.Series([log['daily_return'] for log in trade_log])
        metrics = self._calculate_metrics(returns)
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate total transaction costs paid
        total_transaction_costs = sum([log['transaction_costs'] for log in trade_log])
        
        results = {
            'strategy': 'Pure LLM V2 (Fixed)',
            'initial_capital': self.initial_capital,
            'final_capital': final_value,
            'total_return': total_return,
            'portfolio_values': portfolio_values,
            'trade_log': trade_log,
            'metrics': metrics,
            'total_transaction_costs': total_transaction_costs,
            'n_trades': sum([1 for log in trade_log if log['positions_changed']]),
            'n_days': len(trade_log)
        }
        
        logger.info(f"Pure LLM V2 Strategy completed: {total_return:.2%} return")
        logger.info(f"Total transaction costs: {total_transaction_costs:.4f} ({total_transaction_costs:.3%})")
        logger.info(f"Number of days with position changes: {results['n_trades']}/{results['n_days']}")
        
        return results
    
    def _get_stock_data(self, market_data: pd.DataFrame, stock: str, date: datetime) -> Optional[Dict[str, Any]]:
        """Extract stock data for LLM analysis"""
        
        try:
            if date not in market_data.index:
                return None
            
            row = market_data.loc[date]
            
            stock_data = {
                'symbol': stock,
                'close': row.get(f'{stock}_close', 0),
                'returns_1d': row.get(f'{stock}_returns_1d', 0),
                'volatility_20d': row.get(f'{stock}_volatility_20d', 0),
                'momentum_5d': row.get(f'{stock}_momentum_5d', 0),
                'momentum_20d': row.get(f'{stock}_momentum_20d', 0),
                'rsi_14': row.get(f'{stock}_rsi_14', 50),
                'volume_ratio_20d': row.get(f'{stock}_volume_ratio_20d', 1.0),
                'dma_50': row.get(f'{stock}_dma_50', 0),
                'dma_200': row.get(f'{stock}_dma_200', 0),
            }
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error getting stock data for {stock}: {e}")
            return None
    
    def _get_market_context(self, market_data: pd.DataFrame, date: datetime) -> Dict[str, Any]:
        """Get market context for LLM analysis"""
        
        try:
            returns_cols = [col for col in market_data.columns if col.endswith('_returns_1d')]
            
            if date not in market_data.index or not returns_cols:
                return {'regime': 'Normal', 'market_sentiment': 'Neutral'}
            
            returns = market_data.loc[date, returns_cols]
            market_return = returns.mean()
            market_volatility = returns.std()
            
            # Determine regime
            if market_volatility > 0.03:
                regime = "High Volatility"
            elif market_return > 0.02:
                regime = "Bull Market"
            elif market_return < -0.02:
                regime = "Bear Market"
            else:
                regime = "Normal"
            
            # Determine sentiment
            positive_ratio = (returns > 0).mean()
            if positive_ratio > 0.6:
                sentiment = 'Positive'
            elif positive_ratio < 0.4:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            return {
                'regime': regime,
                'market_sentiment': sentiment,
                'market_return': float(market_return),
                'market_volatility': float(market_volatility)
            }
            
        except Exception as e:
            logger.error(f"Error calculating market context: {e}")
            return {'regime': 'Normal', 'market_sentiment': 'Neutral'}
    
    def _get_sentiment_data(self, stock: str, date: datetime) -> Dict[str, Any]:
        """Get sentiment data for a stock"""
        
        return {
            'reddit': self.sentiment_analyzer.get_reddit_sentiment(stock, date),
            'news': self.sentiment_analyzer.get_news_sentiment(stock, date)
        }
    
    def _calculate_metrics(self, returns: pd.Series) -> PortfolioMetrics:
        """Calculate portfolio performance metrics"""
        
        if len(returns) < 2:
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0)
        
        total_return = (1 + returns).prod() - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.05
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Win/Loss ratio
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        if len(losing_returns) > 0 and len(winning_returns) > 0:
            avg_win = winning_returns.mean()
            avg_loss = abs(losing_returns.mean())
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            win_loss_ratio = 0
        
        # Calmar ratio
        annual_return = total_return * (252 / len(returns))
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return PortfolioMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            volatility=volatility,
            win_loss_ratio=win_loss_ratio,
            calmar_ratio=calmar_ratio
        )

def test_fixed_strategy():
    """Test the fixed strategy"""
    
    print("🔧 TESTING FIXED LLM STRATEGY V2")
    print("=" * 50)
    
    # Load stocks
    stocks_file = Path("finagent/stocks.txt")
    if stocks_file.exists():
        with open(stocks_file, 'r') as f:
            stocks = [line.strip() for line in f.readlines() if line.strip()][:3]  # 3 stocks for quick test
    else:
        stocks = ["RELIANCE", "TCS", "INFY"]
    
    print(f"Testing with: {stocks}")
    
    # Initialize fixed strategy
    strategy = LLMEnhancedStrategyFixedV2(
        api_key="5cqXuAMrvlEapMQjZMlJfChoH5npmMs8",
        stocks=stocks,
        transaction_cost=0.0005  # 0.05% instead of 0.5%
    )
    
    # Run test
    results = strategy.run_pure_llm_strategy("2024-08-25", "2024-08-30")
    
    if "error" not in results:
        print("\n📊 RESULTS:")
        print(f"Initial Capital: ₹{results['initial_capital']:,.0f}")
        print(f"Final Capital: ₹{results['final_capital']:,.0f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Total Transaction Costs: {results['total_transaction_costs']:.4f} ({results['total_transaction_costs']:.3%})")
        print(f"Days with Position Changes: {results['n_trades']}/{results['n_days']}")
        
        print("\n✅ Key Improvements:")
        print("- Transaction costs only applied when positions change")
        print("- Reduced transaction cost from 0.5% to 0.05%")
        print("- No more artificial -0.5% daily drag")
        print("- Portfolio returns now reflect actual market performance")
    else:
        print(f"❌ Error: {results['error']}")

if __name__ == "__main__":
    test_fixed_strategy()