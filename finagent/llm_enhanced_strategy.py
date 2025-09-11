"""
LLM-Enhanced Portfolio Strategy Implementation

This module implements a portfolio strategy that combines:
1. Inverse volatility weighting for smarter capital allocation
2. LLM-based signals with confidence scoring using Mistral API
3. Position sizing with risk constraints
4. Rolling 3-month backtesting framework
5. Enhanced evaluation metrics
"""

import numpy as np
import pandas as pd
import requests
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMSignal:
    """Structure for LLM-generated trading signals"""
    stock_symbol: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-100
    reasoning: str
    timestamp: datetime

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_return: float
    volatility: float
    win_loss_ratio: float
    calmar_ratio: float

class MistralAPIClient:
    """Client for Mistral API integration"""
    
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def generate_signal(self, stock_data: Dict[str, Any], market_context: Dict[str, Any]) -> LLMSignal:
        """Generate trading signal for a stock using LLM"""
        
        prompt = self._create_trading_prompt(stock_data, market_context)
        
        try:
            response = self._call_api(prompt)
            signal_data = self._parse_response(response)
            
            return LLMSignal(
                stock_symbol=stock_data['symbol'],
                signal=signal_data['signal'],
                confidence=signal_data['confidence'],
                reasoning=signal_data['reasoning'],
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error generating signal for {stock_data['symbol']}: {e}")
            return LLMSignal(
                stock_symbol=stock_data['symbol'],
                signal='HOLD',
                confidence=50.0,
                reasoning=f"Error in LLM processing: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _create_trading_prompt(self, stock_data: Dict[str, Any], market_context: Dict[str, Any]) -> str:
        """Create a structured prompt for trading signal generation"""
        
        symbol = stock_data['symbol']
        
        # Extract key metrics
        current_price = stock_data.get('close', 0)
        price_change = stock_data.get('returns_1d', 0) * 100
        volatility = stock_data.get('volatility_20d', 0) * 100
        momentum_5d = stock_data.get('momentum_5d', 0) * 100
        momentum_20d = stock_data.get('momentum_20d', 0) * 100
        rsi = stock_data.get('rsi_14', 50)
        volume_ratio = stock_data.get('volume_ratio_20d', 1.0)
        
        # Technical indicators
        ma_50 = stock_data.get('dma_50', current_price)
        ma_200 = stock_data.get('dma_200', current_price)
        price_to_ma50 = ((current_price / ma_50) - 1) * 100 if ma_50 > 0 else 0
        price_to_ma200 = ((current_price / ma_200) - 1) * 100 if ma_200 > 0 else 0
        
        prompt = f"""
You are a professional quantitative analyst analyzing stock {symbol} for investment decisions.

CURRENT MARKET DATA:
- Stock: {symbol}
- Current Price: ₹{current_price:.2f}
- 1-Day Return: {price_change:.2f}%
- 5-Day Momentum: {momentum_5d:.2f}%
- 20-Day Momentum: {momentum_20d:.2f}%
- 20-Day Volatility: {volatility:.2f}%
- RSI (14): {rsi:.1f}
- Volume Ratio (20D): {volume_ratio:.2f}x
- Price vs 50-Day MA: {price_to_ma50:.2f}%
- Price vs 200-Day MA: {price_to_ma200:.2f}%

MARKET CONTEXT:
- Market Regime: {market_context.get('regime', 'Normal')}
- VIX Level: {market_context.get('vix', 'N/A')}
- Sector Performance: {market_context.get('sector_performance', 'Mixed')}

ANALYSIS FRAMEWORK:
1. Technical Analysis: Evaluate price action, momentum, and technical indicators
2. Risk Assessment: Consider volatility and position in trading range
3. Market Environment: Factor in current market conditions
4. Volume Analysis: Assess trading volume patterns

Please provide a trading recommendation in the following JSON format:
{{
    "signal": "BUY" | "SELL" | "HOLD",
    "confidence": <integer from 0-100>,
    "reasoning": "<detailed explanation in 2-3 sentences>"
}}

IMPORTANT GUIDELINES:
- BUY: Strong positive signals with good risk-reward ratio
- SELL: Strong negative signals or high risk conditions
- HOLD: Mixed signals or insufficient conviction
- Confidence should reflect signal strength and certainty
- Focus on risk-adjusted returns, not just returns
"""
        
        return prompt
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to Mistral"""
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,  # Lower temperature for more consistent responses
            "max_tokens": 500
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")
        
        return response.json()['choices'][0]['message']['content']
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract signal data"""
        
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validate required fields
                signal = parsed.get('signal', 'HOLD').upper()
                if signal not in ['BUY', 'SELL', 'HOLD']:
                    signal = 'HOLD'
                
                confidence = float(parsed.get('confidence', 50))
                confidence = max(0, min(100, confidence))  # Clamp to 0-100
                
                reasoning = parsed.get('reasoning', 'No reasoning provided')
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'reasoning': reasoning
                }
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            
            # Fallback parsing
            response_upper = response_text.upper()
            if 'BUY' in response_upper and 'SELL' not in response_upper:
                signal = 'BUY'
                confidence = 60.0
            elif 'SELL' in response_upper and 'BUY' not in response_upper:
                signal = 'SELL'
                confidence = 60.0
            else:
                signal = 'HOLD'
                confidence = 50.0
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasoning': f"Fallback parsing: {response_text[:100]}"
            }

class LLMEnhancedStrategy:
    """Main strategy class implementing LLM-enhanced portfolio management"""
    
    def __init__(self, 
                 api_key: str,
                 stocks: List[str],
                 initial_capital: float = 1000000.0,
                 max_position_size: float = 0.05,  # 5% max per stock
                 max_sector_exposure: float = 0.20,  # 20% max per sector
                 transaction_cost: float = 0.005):  # 0.5% transaction cost
        
        self.api_key = api_key
        self.stocks = stocks
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.transaction_cost = transaction_cost
        
        # Initialize components
        self.mistral_client = MistralAPIClient(api_key)
        self.current_positions = {}
        self.portfolio_history = []
        self.trade_history = []
        
        logger.info(f"Initialized LLM Enhanced Strategy with {len(stocks)} stocks")
    
    def calculate_inverse_volatility_weights(self, price_data: pd.DataFrame, 
                                           lookback_period: int = 60) -> Dict[str, float]:
        """Calculate inverse volatility weights for capital allocation"""
        
        weights = {}
        volatilities = {}
        
        # Calculate volatilities for each stock
        for stock in self.stocks:
            if stock in price_data.columns:
                returns = price_data[stock].pct_change().dropna()
                if len(returns) >= lookback_period:
                    vol = returns.tail(lookback_period).std() * np.sqrt(252)  # Annualized
                    volatilities[stock] = vol
                else:
                    volatilities[stock] = 0.20  # Default 20% volatility
            else:
                volatilities[stock] = 0.20
        
        # Calculate inverse volatility weights
        if volatilities:
            inv_vols = {stock: 1.0 / max(vol, 0.01) for stock, vol in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            weights = {stock: inv_vol / total_inv_vol for stock, inv_vol in inv_vols.items()}
        else:
            # Equal weights if no volatility data
            equal_weight = 1.0 / len(self.stocks)
            weights = {stock: equal_weight for stock in self.stocks}
        
        logger.info(f"Calculated inverse volatility weights for {len(weights)} stocks")
        return weights
    
    def generate_llm_signals(self, market_data: pd.DataFrame, 
                           current_date: datetime) -> Dict[str, LLMSignal]:
        """Generate LLM signals for all stocks"""
        
        signals = {}
        market_context = self._get_market_context(market_data, current_date)
        
        # Process stocks in parallel for efficiency
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_stock = {}
            
            for stock in self.stocks:
                stock_data = self._prepare_stock_data(market_data, stock, current_date)
                if stock_data:
                    future = executor.submit(
                        self.mistral_client.generate_signal, 
                        stock_data, 
                        market_context
                    )
                    future_to_stock[future] = stock
                    
                    # Add small delay to avoid rate limiting
                    time.sleep(0.1)
            
            # Collect results
            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    signal = future.result()
                    signals[stock] = signal
                except Exception as e:
                    logger.error(f"Error getting signal for {stock}: {e}")
                    signals[stock] = LLMSignal(
                        stock_symbol=stock,
                        signal='HOLD',
                        confidence=50.0,
                        reasoning=f"Error: {str(e)}",
                        timestamp=datetime.now()
                    )
        
        logger.info(f"Generated LLM signals for {len(signals)} stocks")
        return signals
    
    def _prepare_stock_data(self, market_data: pd.DataFrame, 
                          stock: str, current_date: datetime) -> Optional[Dict[str, Any]]:
        """Prepare stock data for LLM analysis"""
        
        try:
            # Get stock data for current date
            if current_date in market_data.index:
                stock_row = market_data.loc[current_date]
                
                # Extract relevant features
                stock_data = {
                    'symbol': stock,
                    'close': stock_row.get(f'{stock}_close', 0),
                    'returns_1d': stock_row.get(f'{stock}_returns_1d', 0),
                    'volatility_20d': stock_row.get(f'{stock}_volatility_20d', 0),
                    'momentum_5d': stock_row.get(f'{stock}_momentum_5d', 0),
                    'momentum_20d': stock_row.get(f'{stock}_momentum_20d', 0),
                    'rsi_14': stock_row.get(f'{stock}_rsi_14', 50),
                    'volume_ratio_20d': stock_row.get(f'{stock}_volume_ratio_20d', 1.0),
                    'dma_50': stock_row.get(f'{stock}_dma_50', 0),
                    'dma_200': stock_row.get(f'{stock}_dma_200', 0),
                }
                
                return stock_data
            else:
                logger.warning(f"No data available for {stock} on {current_date}")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing data for {stock}: {e}")
            return None
    
    def _get_market_context(self, market_data: pd.DataFrame, 
                          current_date: datetime) -> Dict[str, Any]:
        """Get market context for LLM analysis"""
        
        try:
            # Calculate market-wide metrics
            market_returns = market_data.filter(regex='_returns_1d$').loc[current_date]
            market_volatility = market_data.filter(regex='_volatility_20d$').loc[current_date].mean()
            
            avg_return = market_returns.mean()
            return_volatility = market_returns.std()
            
            # Determine market regime
            if return_volatility > 0.02:  # High volatility threshold
                regime = "High Volatility"
            elif avg_return > 0.01:  # Strong positive returns
                regime = "Bull Market"
            elif avg_return < -0.01:  # Strong negative returns
                regime = "Bear Market"
            else:
                regime = "Normal"
            
            context = {
                'regime': regime,
                'avg_return': avg_return,
                'market_volatility': market_volatility,
                'sector_performance': 'Mixed'  # Simplified for now
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {'regime': 'Normal', 'sector_performance': 'Mixed'}
    
    def calculate_position_sizes(self, 
                               signals: Dict[str, LLMSignal],
                               base_weights: Dict[str, float],
                               current_capital: float) -> Dict[str, float]:
        """Calculate position sizes based on LLM signals and constraints"""
        
        position_sizes = {}
        
        for stock in self.stocks:
            if stock in signals and stock in base_weights:
                signal = signals[stock]
                base_weight = base_weights[stock]
                
                # Adjust weight based on LLM signal and confidence
                if signal.signal == 'BUY':
                    confidence_multiplier = signal.confidence / 100.0
                    adjusted_weight = base_weight * (1 + confidence_multiplier)
                elif signal.signal == 'SELL':
                    confidence_multiplier = signal.confidence / 100.0
                    adjusted_weight = base_weight * (1 - confidence_multiplier)
                else:  # HOLD
                    adjusted_weight = base_weight
                
                # Apply position size constraints
                adjusted_weight = min(adjusted_weight, self.max_position_size)
                position_sizes[stock] = adjusted_weight
            else:
                position_sizes[stock] = 0.0
        
        # Normalize to ensure total doesn't exceed 1.0
        total_weight = sum(position_sizes.values())
        if total_weight > 1.0:
            position_sizes = {stock: weight / total_weight 
                            for stock, weight in position_sizes.items()}
        
        # Calculate dollar amounts
        dollar_positions = {stock: weight * current_capital 
                          for stock, weight in position_sizes.items()}
        
        logger.info(f"Calculated position sizes for {len(dollar_positions)} stocks")
        return dollar_positions
    
    def calculate_portfolio_metrics(self, returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None) -> PortfolioMetrics:
        """Calculate comprehensive portfolio performance metrics"""
        
        if len(returns) < 2:
            return PortfolioMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        risk_free_rate = 0.05  # 5% annual risk-free rate
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
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
        
        if len(losing_returns) > 0:
            avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
            avg_loss = abs(losing_returns.mean())
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            win_loss_ratio = float('inf') if len(winning_returns) > 0 else 0
        
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
    
    def run_backtest(self, 
                    start_date: str,
                    end_date: str,
                    data_path: str,
                    scenario: str = "hybrid") -> Dict[str, Any]:
        """Run backtesting with rolling 3-month windows"""
        
        logger.info(f"Starting {scenario} backtest from {start_date} to {end_date}")
        
        # Load market data
        market_data = self._load_market_data(data_path, start_date, end_date)
        
        # Initialize portfolio
        current_capital = self.initial_capital
        portfolio_values = []
        daily_returns = []
        trade_log = []
        
        # Generate date range for 3-month rolling windows
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start_dt
        window_size = timedelta(days=90)  # 3 months
        
        while current_date <= end_dt:
            try:
                # Define window
                window_start = current_date
                window_end = min(current_date + window_size, end_dt)
                
                # Get data for current window
                window_data = market_data[window_start:window_end]
                
                if len(window_data) > 0:
                    if scenario == "pure_llm":
                        portfolio_value = self._run_pure_llm_scenario(
                            window_data, current_capital, trade_log
                        )
                    else:  # hybrid
                        portfolio_value = self._run_hybrid_scenario(
                            window_data, current_capital, trade_log
                        )
                    
                    portfolio_values.append(portfolio_value)
                    
                    # Calculate daily return
                    if len(portfolio_values) > 1:
                        daily_return = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
                        daily_returns.append(daily_return)
                    
                    current_capital = portfolio_value
                
                # Move to next day
                current_date += timedelta(days=1)
                
            except Exception as e:
                logger.error(f"Error in backtest at {current_date}: {e}")
                current_date += timedelta(days=1)
                continue
        
        # Calculate metrics
        returns_series = pd.Series(daily_returns)
        metrics = self.calculate_portfolio_metrics(returns_series)
        
        results = {
            'scenario': scenario,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_capital': current_capital,
            'total_return': (current_capital - self.initial_capital) / self.initial_capital,
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'trade_log': trade_log,
            'metrics': metrics,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_loss_ratio': metrics.win_loss_ratio,
            'calmar_ratio': metrics.calmar_ratio
        }
        
        logger.info(f"Backtest completed. Final capital: ₹{current_capital:,.2f}")
        return results
    
    def _run_pure_llm_scenario(self, window_data: pd.DataFrame, 
                              current_capital: float, trade_log: List) -> float:
        """Run pure LLM signals scenario - equal allocation + LLM signals"""
        
        # Equal allocation across all stocks
        equal_weight = 1.0 / len(self.stocks)
        base_weights = {stock: equal_weight for stock in self.stocks}
        
        # Get LLM signals for the last date in window
        last_date = window_data.index[-1]
        signals = self.generate_llm_signals(window_data, last_date)
        
        # Calculate positions based on LLM signals only
        positions = {}
        for stock in self.stocks:
            if stock in signals:
                signal = signals[stock]
                if signal.signal == 'BUY':
                    positions[stock] = equal_weight * current_capital
                elif signal.signal == 'SELL':
                    positions[stock] = 0.0
                else:  # HOLD
                    positions[stock] = equal_weight * current_capital * 0.5  # Half position
            else:
                positions[stock] = equal_weight * current_capital
        
        # Simulate portfolio performance (simplified)
        portfolio_return = 0.0
        for stock in self.stocks:
            if f'{stock}_returns_1d' in window_data.columns:
                stock_return = window_data[f'{stock}_returns_1d'].iloc[-1]
                weight = positions[stock] / current_capital
                portfolio_return += weight * stock_return
        
        # Apply transaction costs
        portfolio_return -= self.transaction_cost
        
        return current_capital * (1 + portfolio_return)
    
    def _run_hybrid_scenario(self, window_data: pd.DataFrame, 
                           current_capital: float, trade_log: List) -> float:
        """Run LLM + quant hybrid scenario"""
        
        # Calculate inverse volatility weights
        price_data = window_data.filter(regex='_close$')
        price_data.columns = [col.replace('_close', '') for col in price_data.columns]
        base_weights = self.calculate_inverse_volatility_weights(price_data)
        
        # Get LLM signals
        last_date = window_data.index[-1]
        signals = self.generate_llm_signals(window_data, last_date)
        
        # Calculate position sizes combining both approaches
        positions = self.calculate_position_sizes(signals, base_weights, current_capital)
        
        # Simulate portfolio performance
        portfolio_return = 0.0
        for stock in self.stocks:
            if f'{stock}_returns_1d' in window_data.columns:
                stock_return = window_data[f'{stock}_returns_1d'].iloc[-1]
                weight = positions[stock] / current_capital
                portfolio_return += weight * stock_return
        
        # Apply transaction costs
        portfolio_return -= self.transaction_cost
        
        return current_capital * (1 + portfolio_return)
    
    def _load_market_data(self, data_path: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load market data for backtesting"""
        
        # This is a placeholder - implement based on your data structure
        logger.info(f"Loading market data from {data_path}")
        
        # For now, return dummy data structure
        # You should replace this with actual data loading logic
        dates = pd.date_range(start_date, end_date, freq='D')
        data = pd.DataFrame(index=dates)
        
        for stock in self.stocks:
            # Add dummy columns for each stock
            data[f'{stock}_close'] = 100 * (1 + np.random.normal(0, 0.02, len(dates)).cumsum())
            data[f'{stock}_returns_1d'] = np.random.normal(0, 0.02, len(dates))
            data[f'{stock}_volatility_20d'] = np.random.uniform(0.15, 0.35, len(dates))
            data[f'{stock}_momentum_5d'] = np.random.normal(0, 0.05, len(dates))
            data[f'{stock}_momentum_20d'] = np.random.normal(0, 0.10, len(dates))
            data[f'{stock}_rsi_14'] = np.random.uniform(20, 80, len(dates))
            data[f'{stock}_volume_ratio_20d'] = np.random.uniform(0.5, 2.0, len(dates))
            data[f'{stock}_dma_50'] = data[f'{stock}_close'] * np.random.uniform(0.95, 1.05, len(dates))
            data[f'{stock}_dma_200'] = data[f'{stock}_close'] * np.random.uniform(0.90, 1.10, len(dates))
        
        return data

def main():
    """Example usage of the LLM Enhanced Strategy"""
    
    # Configuration
    MISTRAL_API_KEY = "5cqXuAMrvlEapMQjZMlJfChoH5npmMs8"
    
    # Sample stocks (replace with your actual stock list)
    stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", 
              "KOTAKBANK", "BHARTIARTL", "ITC", "SBIN", "LT"]
    
    # Initialize strategy
    strategy = LLMEnhancedStrategy(
        api_key=MISTRAL_API_KEY,
        stocks=stocks,
        initial_capital=1000000.0,
        max_position_size=0.05,
        max_sector_exposure=0.20,
        transaction_cost=0.005
    )
    
    # Run backtests
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    data_path = "processed_data/"
    
    logger.info("Running Pure LLM Scenario...")
    pure_llm_results = strategy.run_backtest(
        start_date=start_date,
        end_date=end_date,
        data_path=data_path,
        scenario="pure_llm"
    )
    
    logger.info("Running Hybrid Scenario...")
    hybrid_results = strategy.run_backtest(
        start_date=start_date,
        end_date=end_date,
        data_path=data_path,
        scenario="hybrid"
    )
    
    # Print results
    print("\n" + "="*50)
    print("BACKTEST RESULTS COMPARISON")
    print("="*50)
    
    print(f"\nPure LLM Strategy:")
    print(f"Total Return: {pure_llm_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {pure_llm_results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {pure_llm_results['max_drawdown']:.2%}")
    print(f"Sortino Ratio: {pure_llm_results['metrics'].sortino_ratio:.3f}")
    
    print(f"\nLLM + Quant Hybrid Strategy:")
    print(f"Total Return: {hybrid_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {hybrid_results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {hybrid_results['max_drawdown']:.2%}")
    print(f"Sortino Ratio: {hybrid_results['metrics'].sortino_ratio:.3f}")
    
    return pure_llm_results, hybrid_results

if __name__ == "__main__":
    main()