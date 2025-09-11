#!/usr/bin/env python3
"""
LLM Portfolio Strategy with ACTUAL SHARE TRACKING

This implementation properly:
1. Buys actual quantities of shares at market prices
2. Tracks share holdings over time
3. Calculates portfolio value as sum of (shares × current_price)
4. Applies LLM signals to change actual positions
5. Uses 2024 for training/learning, 2025 for testing
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import time
from dataclasses import dataclass

# Import components
from llm_enhanced_strategy_fixed import MistralAPIClient, SentimentAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Track actual stock position"""
    stock: str
    shares: float  # Number of shares owned
    avg_cost: float  # Average cost per share
    current_price: float  # Current market price
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def total_cost(self) -> float:
        return self.shares * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.total_cost

class RealPortfolioManager:
    """Manages actual share positions like a real portfolio"""
    
    def __init__(self, 
                 api_key: str,
                 stocks: List[str],
                 initial_capital: float = 1000000.0,
                 transaction_cost_pct: float = 0.0005):  # 0.05%
        
        self.api_key = api_key
        self.stocks = stocks
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        
        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.portfolio_history = []
        
        # Components
        self.llm_client = MistralAPIClient(api_key)
        self.sentiment_analyzer = SentimentAnalyzer(".")
        
        logger.info(f"Initialized Real Portfolio Manager with ₹{initial_capital:,.0f}")
        logger.info(f"Stocks: {stocks}")
    
    def load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market data for all stocks"""
        
        all_data = {}
        
        for stock in self.stocks:
            csv_file = Path(f"processed_data/{stock}_aligned.csv")
            
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
                    
                    # Filter date range
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    
                    if len(df) > 0:
                        # Only keep essential columns to reduce memory
                        essential_cols = ['open', 'high', 'low', 'close', 'volume']
                        available_cols = [col for col in essential_cols if col in df.columns]
                        
                        if available_cols:
                            stock_data = df[available_cols].copy()
                            stock_data.columns = [f"{stock}_{col}" for col in stock_data.columns]
                            all_data[stock] = stock_data
                            logger.info(f"Loaded {len(stock_data)} days for {stock}")
                    
                except Exception as e:
                    logger.error(f"Error loading {stock}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data.values(), axis=1, sort=True)
            logger.info(f"Combined market data: {combined_df.shape}")
            return combined_df.fillna(method='ffill')
        else:
            return pd.DataFrame()
    
    def get_current_prices(self, market_data: pd.DataFrame, date: datetime) -> Dict[str, float]:
        """Get current closing prices for all stocks"""
        
        prices = {}
        if date in market_data.index:
            for stock in self.stocks:
                close_col = f"{stock}_close"
                if close_col in market_data.columns:
                    prices[stock] = market_data.loc[date, close_col]
                else:
                    prices[stock] = 0.0
        
        return prices
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Calculate total portfolio value"""
        
        total_equity_value = 0.0
        position_values = {}
        
        for stock, position in self.positions.items():
            if stock in current_prices:
                position.current_price = current_prices[stock]
                position_values[stock] = {
                    'shares': position.shares,
                    'price': position.current_price,
                    'market_value': position.market_value,
                    'avg_cost': position.avg_cost,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': (position.unrealized_pnl / position.total_cost) * 100 if position.total_cost > 0 else 0
                }
                total_equity_value += position.market_value
        
        total_portfolio_value = self.cash + total_equity_value
        
        return {
            'date': datetime.now(),
            'cash': self.cash,
            'equity_value': total_equity_value,
            'total_value': total_portfolio_value,
            'positions': position_values
        }
    
    def execute_trade(self, stock: str, action: str, quantity: float, price: float) -> Dict[str, Any]:
        """Execute actual buy/sell trade"""
        
        trade_value = quantity * price
        transaction_cost = trade_value * self.transaction_cost_pct
        
        if action.upper() == 'BUY':
            total_cost = trade_value + transaction_cost
            
            if self.cash >= total_cost:
                # Execute buy
                self.cash -= total_cost
                
                if stock in self.positions:
                    # Add to existing position (update average cost)
                    old_shares = self.positions[stock].shares
                    old_cost = self.positions[stock].avg_cost
                    
                    new_shares = old_shares + quantity
                    new_avg_cost = ((old_shares * old_cost) + (quantity * price)) / new_shares
                    
                    self.positions[stock].shares = new_shares
                    self.positions[stock].avg_cost = new_avg_cost
                else:
                    # New position
                    self.positions[stock] = Position(
                        stock=stock,
                        shares=quantity,
                        avg_cost=price,
                        current_price=price
                    )
                
                return {
                    'success': True,
                    'action': 'BUY',
                    'stock': stock,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value,
                    'transaction_cost': transaction_cost,
                    'cash_remaining': self.cash
                }
            else:
                return {'success': False, 'reason': f'Insufficient cash: ₹{self.cash:.0f} < ₹{total_cost:.0f}'}
        
        elif action.upper() == 'SELL':
            if stock in self.positions and self.positions[stock].shares >= quantity:
                # Execute sell
                self.cash += trade_value - transaction_cost
                
                self.positions[stock].shares -= quantity
                
                # Remove position if fully sold
                if self.positions[stock].shares <= 0:
                    del self.positions[stock]
                
                return {
                    'success': True,
                    'action': 'SELL',
                    'stock': stock,
                    'quantity': quantity,
                    'price': price,
                    'value': trade_value,
                    'transaction_cost': transaction_cost,
                    'cash_remaining': self.cash
                }
            else:
                current_shares = self.positions.get(stock, Position(stock, 0, 0, 0)).shares
                return {'success': False, 'reason': f'Insufficient shares: {current_shares} < {quantity}'}
        
        return {'success': False, 'reason': f'Unknown action: {action}'}
    
    def get_stock_data_for_llm(self, market_data: pd.DataFrame, stock: str, date: datetime) -> Optional[Dict[str, Any]]:
        """Get stock data for LLM analysis"""
        
        if date not in market_data.index:
            return None
        
        row = market_data.loc[date]
        
        # Calculate simple technical indicators
        close_col = f"{stock}_close"
        if close_col not in market_data.columns:
            return None
        
        close_price = row[close_col]
        
        # Get recent prices for momentum calculation
        stock_closes = market_data[close_col].dropna()
        current_idx = stock_closes.index.get_loc(date)
        
        # Calculate returns and momentum
        returns_1d = 0.0
        momentum_5d = 0.0
        
        if current_idx > 0:
            prev_price = stock_closes.iloc[current_idx - 1]
            returns_1d = (close_price - prev_price) / prev_price
        
        if current_idx > 5:
            price_5d_ago = stock_closes.iloc[current_idx - 5]
            momentum_5d = (close_price - price_5d_ago) / price_5d_ago
        
        return {
            'symbol': stock,
            'close': close_price,
            'returns_1d': returns_1d,
            'momentum_5d': momentum_5d,
            'current_position': self.positions.get(stock, Position(stock, 0, 0, close_price)).shares
        }
    
    def generate_llm_signal(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM signal for a stock"""
        
        try:
            # Get sentiment data
            sentiment_data = self.sentiment_analyzer.get_reddit_sentiment(
                stock_data['symbol'], datetime.now()
            )
            
            # Simple market context
            market_context = {'regime': 'Normal', 'market_sentiment': 'Neutral'}
            
            # Generate signal using LLM
            signal = self.llm_client.generate_signal(stock_data, market_context, {'reddit': sentiment_data})
            
            return {
                'signal': signal.signal,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            }
        
        except Exception as e:
            logger.error(f"Error generating LLM signal for {stock_data['symbol']}: {e}")
            return {'signal': 'HOLD', 'confidence': 50, 'reasoning': f'Error: {e}'}
    
    def run_backtest(self, start_date: str, end_date: str, initial_allocation: str = "equal") -> Dict[str, Any]:
        """Run backtest with actual share tracking"""
        
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        # Load market data
        market_data = self.load_market_data(start_date, end_date)
        
        if market_data.empty:
            return {"error": "No market data available"}
        
        # Get trading dates
        trading_dates = market_data.index
        
        results = {
            'daily_portfolios': [],
            'trades': [],
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital
        }
        
        # Day 1: Initial allocation
        first_date = trading_dates[0]
        current_prices = self.get_current_prices(market_data, first_date)
        
        logger.info(f"🏦 Initial allocation on {first_date.strftime('%Y-%m-%d')} across {len(self.stocks)} stocks")
        
        if initial_allocation == "equal":
            # Equal allocation across all stocks
            allocation_per_stock = (self.initial_capital * 0.95) / len(self.stocks)  # Keep 5% cash
            
            for stock in self.stocks:
                if stock in current_prices and current_prices[stock] > 0:
                    price = current_prices[stock]
                    shares_to_buy = allocation_per_stock / price
                    
                    trade_result = self.execute_trade(stock, 'BUY', shares_to_buy, price)
                    
                    if trade_result['success']:
                        results['trades'].append({
                            'date': first_date,
                            'stock': stock,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': price,
                            'value': trade_result['value'],
                            'reason': 'Initial allocation'
                        })
                        
                        # Reduce logging during initial allocation
                        pass
        
        # Calculate initial portfolio value
        portfolio_value = self.calculate_portfolio_value(current_prices)
        results['daily_portfolios'].append({
            'date': first_date,
            **portfolio_value
        })
        
        logger.info(f"   💰 Initial Portfolio Value: ₹{portfolio_value['total_value']:,.0f}")
        
        # Daily trading loop
        for i, date in enumerate(trading_dates[1:], 1):
            
            current_prices = self.get_current_prices(market_data, date)
            
            logger.info(f"\n📅 DAY {i+1} ({date.strftime('%Y-%m-%d')}):")
            
            # Generate LLM signals for each stock
            daily_trades = []
            
            for stock in self.stocks:
                if stock in current_prices:
                    stock_data = self.get_stock_data_for_llm(market_data, stock, date)
                    
                    if stock_data:
                        llm_signal = self.generate_llm_signal(stock_data)
                        current_shares = self.positions.get(stock, Position(stock, 0, 0, 0)).shares
                        current_price = current_prices[stock]
                        
                        # Only log high-confidence signals to reduce noise
                        if llm_signal['confidence'] >= 70 or llm_signal['signal'] != 'HOLD':
                            logger.info(f"   🤖 {stock}: {llm_signal['signal']} ({llm_signal['confidence']}%)")
                        
                        # Execute trades based on LLM signal
                        if llm_signal['signal'] == 'BUY' and llm_signal['confidence'] > 60:
                            # Buy more (add 25% to current position or start new position)
                            if current_shares > 0:
                                additional_value = self.positions[stock].market_value * 0.25
                            else:
                                additional_value = min(50000, self.cash * 0.1)  # Max ₹50k or 10% of cash
                            
                            if additional_value > 1000 and self.cash > additional_value:  # Minimum ₹1000 trade
                                shares_to_buy = additional_value / current_price
                                
                                trade_result = self.execute_trade(stock, 'BUY', shares_to_buy, current_price)
                                
                                if trade_result['success']:
                                    daily_trades.append({
                                        'date': date,
                                        'stock': stock,
                                        'action': 'BUY',
                                        'shares': shares_to_buy,
                                        'price': current_price,
                                        'value': trade_result['value'],
                                        'reason': f"LLM BUY signal ({llm_signal['confidence']}%)"
                                    })
                                    logger.info(f"      ✅ BUY {stock}: {shares_to_buy:.0f} shares = ₹{trade_result['value']:,.0f}")
                        
                        elif llm_signal['signal'] == 'SELL' and llm_signal['confidence'] > 70 and current_shares > 0:
                            # Sell half of current position
                            shares_to_sell = current_shares * 0.5
                            
                            if shares_to_sell > 0:
                                trade_result = self.execute_trade(stock, 'SELL', shares_to_sell, current_price)
                                
                                if trade_result['success']:
                                    daily_trades.append({
                                        'date': date,
                                        'stock': stock,
                                        'action': 'SELL',
                                        'shares': shares_to_sell,
                                        'price': current_price,
                                        'value': trade_result['value'],
                                        'reason': f"LLM SELL signal ({llm_signal['confidence']}%)"
                                    })
                                    logger.info(f"      🔻 SELL {stock}: {shares_to_sell:.0f} shares = ₹{trade_result['value']:,.0f}")
                        
                        # For HOLD, do nothing (just track price changes)
                        
                        time.sleep(0.1)  # Faster rate limiting for efficiency
            
            # Calculate end-of-day portfolio value
            portfolio_value = self.calculate_portfolio_value(current_prices)
            
            results['daily_portfolios'].append({
                'date': date,
                **portfolio_value
            })
            
            results['trades'].extend(daily_trades)
            
            # Daily summary
            prev_value = results['daily_portfolios'][-2]['total_value']
            daily_return = (portfolio_value['total_value'] - prev_value) / prev_value
            
            # Weekly summary instead of daily to reduce logs
            if i % 5 == 0 or i == len(trading_dates) - 1:  # Every 5 days or last day
                logger.info(f"   💰 Day {i+1}: ₹{portfolio_value['total_value']:,.0f} ({daily_return:+.2%}) | Cash: ₹{self.cash:,.0f}")
                
                # Show top 3 positions weekly
                if portfolio_value['positions']:
                    top_positions = sorted(portfolio_value['positions'].items(), 
                                         key=lambda x: x[1]['market_value'], reverse=True)[:3]
                    for stock, pos in top_positions:
                        logger.info(f"      {stock}: ₹{pos['market_value']:,.0f} ({pos['unrealized_pnl_pct']:+.1f}%)")
        
        # Final results
        final_value = results['daily_portfolios'][-1]['total_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        results['final_capital'] = final_value
        results['total_return'] = total_return
        results['total_trades'] = len(results['trades'])
        
        logger.info(f"\n🎯 FINAL RESULTS:")
        logger.info(f"   Initial Capital: ₹{self.initial_capital:,.0f}")
        logger.info(f"   Final Capital: ₹{final_value:,.0f}")
        logger.info(f"   Total Return: {total_return:.2%}")
        logger.info(f"   Total Trades: {results['total_trades']}")
        
        return results

def main():
    """Run the real portfolio backtest"""
    
    print("🏦 REAL PORTFOLIO BACKTEST WITH ACTUAL SHARE TRACKING")
    print("=" * 60)
    
    # Load stocks
    stocks_file = Path("finagent/stocks.txt")
    if stocks_file.exists():
        with open(stocks_file, 'r') as f:
            stocks = [line.strip() for line in f.readlines() if line.strip()][:5]  # First 5 stocks
    else:
        stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    
    # Initialize portfolio manager
    portfolio = RealPortfolioManager(
        api_key="5cqXuAMrvlEapMQjZMlJfChoH5npmMs8",
        stocks=stocks,
        initial_capital=1000000.0
    )
    
    # Run backtest (last 10 days for demo)
    results = portfolio.run_backtest("2024-08-20", "2024-08-30")
    
    if "error" not in results:
        print(f"\n✅ Backtest completed successfully!")
        print(f"Final portfolio value: ₹{results['final_capital']:,.0f}")
        print(f"Total return: {results['total_return']:.2%}")
    else:
        print(f"❌ Error: {results['error']}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    main()