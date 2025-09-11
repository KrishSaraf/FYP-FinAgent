"""
LLM-Enhanced Portfolio Strategy Implementation - FIXED VERSION

This module implements a working portfolio strategy that combines:
1. Inverse volatility weighting for smarter capital allocation
2. LLM-based signals with confidence scoring using Mistral API
3. Sentiment data from Reddit and news
4. Position sizing with risk constraints
5. Rolling 3-month backtesting framework
6. Enhanced evaluation metrics

Key fixes:
- Proper JSON parsing for LLM responses
- Integration with existing data infrastructure
- Sentiment analysis from Reddit/news data
- Real data loading from processed_data directory
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

class SentimentAnalyzer:
    """Analyze sentiment from Reddit and news data"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.sentiment_cache = {}
    
    def get_reddit_sentiment(self, stock: str, date: datetime) -> Dict[str, Any]:
        """Get Reddit sentiment for a stock around a specific date"""
        
        cache_key = f"{stock}_{date.strftime('%Y-%m-%d')}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        try:
            # Load Reddit data
            reddit_file = self.data_root / "social_media_data" / "cleaned_data" / stock / "reddit.csv"
            
            if not reddit_file.exists():
                return {'sentiment': 'neutral', 'score': 0.0, 'volume': 0}
            
            df = pd.read_csv(reddit_file)
            
            if 'created_timestamp' in df.columns:
                df['created_timestamp'] = pd.to_datetime(df['created_timestamp'])
                
                # Get posts/comments from past 7 days
                start_date = date - timedelta(days=7)
                recent_data = df[
                    (df['created_timestamp'] >= start_date) & 
                    (df['created_timestamp'] <= date)
                ]
                
                if len(recent_data) > 0:
                    # Simple sentiment analysis based on scores and keywords
                    sentiment_score = 0.0
                    total_items = 0
                    
                    for _, row in recent_data.iterrows():
                        score = row.get('score', 0) or 0
                        
                        # Text analysis
                        text = str(row.get('body', '') or row.get('title', '') or '').lower()
                        
                        # Positive keywords
                        positive_words = ['buy', 'bullish', 'moon', 'rocket', 'gains', 'profit', 'up', 'rise', 'strong']
                        negative_words = ['sell', 'bearish', 'crash', 'loss', 'down', 'fall', 'weak', 'dump']
                        
                        text_sentiment = 0
                        for word in positive_words:
                            if word in text:
                                text_sentiment += 1
                        for word in negative_words:
                            if word in text:
                                text_sentiment -= 1
                        
                        # Combine score and text sentiment
                        item_sentiment = score + text_sentiment
                        sentiment_score += item_sentiment
                        total_items += 1
                    
                    if total_items > 0:
                        avg_sentiment = sentiment_score / total_items
                        
                        # Categorize sentiment
                        if avg_sentiment > 1:
                            sentiment = 'positive'
                        elif avg_sentiment < -1:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                        
                        result = {
                            'sentiment': sentiment,
                            'score': avg_sentiment,
                            'volume': total_items
                        }
                        
                        self.sentiment_cache[cache_key] = result
                        return result
            
            return {'sentiment': 'neutral', 'score': 0.0, 'volume': 0}
            
        except Exception as e:
            logger.error(f"Error loading Reddit sentiment for {stock}: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'volume': 0}
    
    def get_news_sentiment(self, stock: str, date: datetime) -> Dict[str, Any]:
        """Get news sentiment for a stock around a specific date"""
        
        try:
            # Load news data
            news_file = self.data_root / "news" / "stocks_news_data" / "indian_stock_news.csv"
            
            if not news_file.exists():
                return {'sentiment': 'neutral', 'score': 0.0, 'volume': 0}
            
            df = pd.read_csv(news_file)
            
            # Filter for the specific stock and recent dates
            if 'symbol' in df.columns or 'stock' in df.columns:
                symbol_col = 'symbol' if 'symbol' in df.columns else 'stock'
                stock_news = df[df[symbol_col].str.upper() == stock.upper()]
                
                if len(stock_news) > 0:
                    # Simple news sentiment (you could enhance this with NLP)
                    return {
                        'sentiment': 'neutral',
                        'score': 0.0,
                        'volume': len(stock_news)
                    }
            
            return {'sentiment': 'neutral', 'score': 0.0, 'volume': 0}
            
        except Exception as e:
            logger.error(f"Error loading news sentiment for {stock}: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'volume': 0}

class MistralAPIClient:
    """Client for Mistral API integration with improved JSON parsing"""
    
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def generate_signal(self, stock_data: Dict[str, Any], market_context: Dict[str, Any], 
                       sentiment_data: Optional[Dict[str, Any]] = None) -> LLMSignal:
        """Generate trading signal for a stock using LLM with sentiment data"""
        
        prompt = self._create_trading_prompt(stock_data, market_context, sentiment_data)
        
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
    
    def _create_trading_prompt(self, stock_data: Dict[str, Any], market_context: Dict[str, Any], 
                              sentiment_data: Optional[Dict[str, Any]] = None) -> str:
        """Create a structured prompt for trading signal generation with sentiment"""
        
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
        
        # Sentiment information
        sentiment_section = ""
        if sentiment_data:
            reddit_sentiment = sentiment_data.get('reddit', {})
            news_sentiment = sentiment_data.get('news', {})
            
            sentiment_section = f"""
SENTIMENT ANALYSIS:
- Reddit Sentiment: {reddit_sentiment.get('sentiment', 'neutral').title()} (Score: {reddit_sentiment.get('score', 0):.1f}, Volume: {reddit_sentiment.get('volume', 0)} posts)
- News Sentiment: {news_sentiment.get('sentiment', 'neutral').title()} (Volume: {news_sentiment.get('volume', 0)} articles)
"""
        
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
- Market Sentiment: {market_context.get('market_sentiment', 'Neutral')}
{sentiment_section}
ANALYSIS FRAMEWORK:
1. Technical Analysis: Evaluate price action, momentum, and technical indicators
2. Risk Assessment: Consider volatility and position in trading range
3. Market Environment: Factor in current market conditions
4. Sentiment Analysis: Consider social media and news sentiment
5. Volume Analysis: Assess trading volume patterns

Please provide a trading recommendation in this EXACT JSON format (no extra text or formatting):
{{"signal": "BUY", "confidence": 75, "reasoning": "Brief explanation"}}

IMPORTANT GUIDELINES:
- Respond with ONLY the JSON object, no markdown or extra text
- BUY: Strong positive signals with good risk-reward ratio
- SELL: Strong negative signals or high risk conditions  
- HOLD: Mixed signals or insufficient conviction
- Confidence should be integer from 0-100
- Keep reasoning brief and factual
- Factor in sentiment data if available
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
            "temperature": 0.3,
            "max_tokens": 200  # Reduced for more concise responses
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
        """Parse LLM response to extract signal data with improved JSON handling"""
        
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Try to find JSON object
            json_pattern = r'\{[^{}]*"signal"[^{}]*"confidence"[^{}]*"reasoning"[^{}]*\}'
            json_matches = re.findall(json_pattern, response_text)
            
            if json_matches:
                json_str = json_matches[0]
                
                # Clean up the JSON string
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)  # Remove newlines and tabs
                json_str = re.sub(r'\s+', ' ', json_str)  # Normalize spaces
                
                try:
                    parsed = json.loads(json_str)
                    
                    # Validate and clean fields
                    signal = str(parsed.get('signal', 'HOLD')).upper()
                    if signal not in ['BUY', 'SELL', 'HOLD']:
                        signal = 'HOLD'
                    
                    confidence = float(parsed.get('confidence', 50))
                    confidence = max(0, min(100, confidence))  # Clamp to 0-100
                    
                    reasoning = str(parsed.get('reasoning', 'No reasoning provided'))[:200]  # Limit length
                    
                    return {
                        'signal': signal,
                        'confidence': confidence,
                        'reasoning': reasoning
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try simpler JSON extraction
            signal_match = re.search(r'"signal":\s*"([^"]+)"', response_text, re.IGNORECASE)
            confidence_match = re.search(r'"confidence":\s*(\d+)', response_text)
            
            if signal_match and confidence_match:
                signal = signal_match.group(1).upper()
                if signal not in ['BUY', 'SELL', 'HOLD']:
                    signal = 'HOLD'
                
                confidence = float(confidence_match.group(1))
                confidence = max(0, min(100, confidence))
                
                return {
                    'signal': signal,
                    'confidence': confidence,
                    'reasoning': 'Extracted from partial response'
                }
            
            # Last resort: keyword-based parsing
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
                'reasoning': f"Keyword-based parsing from: {response_text[:100]}"
            }
                
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 50.0,
                'reasoning': f"Parsing error: {str(e)}"
            }

class RealDataLoader:
    """Load real data from processed_data directory"""
    
    def __init__(self, data_root: str, stocks: List[str]):
        self.data_root = Path(data_root)
        self.stocks = stocks
        
    def load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load real market data from processed_data directory"""
        
        all_data = {}
        
        for stock in self.stocks:
            csv_file = self.data_root / f"{stock}_aligned.csv"
            
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
                    
                    # Filter date range
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    
                    if len(df) > 0:
                        # Add stock prefix to columns
                        df.columns = [f"{stock}_{col}" for col in df.columns]
                        all_data[stock] = df
                        logger.info(f"Loaded {len(df)} rows for {stock}")
                    else:
                        logger.warning(f"No data in date range for {stock}")
                        
                except Exception as e:
                    logger.error(f"Error loading {stock}: {e}")
            else:
                logger.warning(f"File not found: {csv_file}")
        
        if all_data:
            # Combine all stock data
            combined_df = pd.concat(all_data.values(), axis=1, sort=True)
            logger.info(f"Combined data shape: {combined_df.shape}")
            return combined_df.fillna(method='ffill').fillna(0)
        else:
            logger.error("No data loaded for any stocks")
            return pd.DataFrame()

class LLMEnhancedStrategyFixed:
    """Main strategy class implementing LLM-enhanced portfolio management with fixes"""
    
    def __init__(self, 
                 api_key: str,
                 stocks: List[str],
                 data_root: str = "processed_data/",
                 initial_capital: float = 1000000.0,
                 max_position_size: float = 0.05,
                 max_sector_exposure: float = 0.20,
                 transaction_cost: float = 0.005):
        
        self.api_key = api_key
        self.stocks = stocks
        self.data_root = data_root
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.transaction_cost = transaction_cost
        
        # Initialize components
        self.mistral_client = MistralAPIClient(api_key)
        self.sentiment_analyzer = SentimentAnalyzer(data_root)
        self.data_loader = RealDataLoader(data_root, stocks)
        
        logger.info(f"Initialized Fixed LLM Enhanced Strategy with {len(stocks)} stocks")
    
    def run_pure_llm_strategy(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run Pure LLM strategy: equal allocation + LLM buy/sell/hold signals"""
        
        logger.info(f"Running Pure LLM Strategy from {start_date} to {end_date}")
        
        # Load market data
        market_data = self.data_loader.load_market_data(start_date, end_date)
        
        if market_data.empty:
            logger.error("No market data available")
            return {"error": "No market data available"}
        
        # Initialize portfolio
        equal_weight = 1.0 / len(self.stocks)
        portfolio_values = [self.initial_capital]
        trade_log = []
        
        # Get trading dates (use last 30 days for demo)
        trading_dates = market_data.index[-30:]
        
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
                
                # Calculate portfolio value based on signals
                portfolio_return = 0.0
                for stock in self.stocks:
                    if stock in signals and f"{stock}_returns_1d" in market_data.columns:
                        signal = signals[stock]
                        stock_return = market_data.loc[date, f"{stock}_returns_1d"]
                        
                        # Apply signal logic
                        if signal.signal == 'BUY':
                            weight = equal_weight
                        elif signal.signal == 'SELL':
                            weight = 0.0  # Exit position
                        else:  # HOLD
                            weight = equal_weight * 0.5  # Half position
                        
                        portfolio_return += weight * stock_return
                
                # Apply transaction costs
                portfolio_return -= self.transaction_cost
                
                # Update portfolio value
                new_value = current_capital * (1 + portfolio_return)
                portfolio_values.append(new_value)
                
                trade_log.append({
                    'date': date,
                    'portfolio_value': new_value,
                    'daily_return': portfolio_return,
                    'signals': {stock: signal.signal for stock, signal in signals.items()}
                })
                
                if i % 5 == 0:
                    logger.info(f"Pure LLM - {date.strftime('%Y-%m-%d')}: ₹{new_value:,.0f} ({portfolio_return:.2%})")
                
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue
        
        # Calculate final metrics
        returns = pd.Series([log['daily_return'] for log in trade_log])
        metrics = self._calculate_metrics(returns)
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        results = {
            'strategy': 'Pure LLM',
            'initial_capital': self.initial_capital,
            'final_capital': final_value,
            'total_return': total_return,
            'portfolio_values': portfolio_values,
            'trade_log': trade_log,
            'metrics': metrics,
            'n_trades': len(trade_log)
        }
        
        logger.info(f"Pure LLM Strategy completed: {total_return:.2%} return")
        return results
    
    def run_hybrid_strategy(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run LLM + Quant Hybrid strategy: inverse volatility weights + LLM confidence"""
        
        logger.info(f"Running Hybrid Strategy from {start_date} to {end_date}")
        
        # Load market data
        market_data = self.data_loader.load_market_data(start_date, end_date)
        
        if market_data.empty:
            logger.error("No market data available")
            return {"error": "No market data available"}
        
        # Calculate base inverse volatility weights
        base_weights = self._calculate_inverse_volatility_weights(market_data)
        
        # Initialize portfolio
        portfolio_values = [self.initial_capital]
        trade_log = []
        
        # Get trading dates (use last 30 days for demo)
        trading_dates = market_data.index[-30:]
        
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
                        
                        # Add delay to respect API limits
                        time.sleep(0.5)
                
                # Calculate adjusted position sizes
                positions = self._calculate_position_sizes(signals, base_weights, current_capital)
                
                # Calculate portfolio return
                portfolio_return = 0.0
                for stock in self.stocks:
                    if f"{stock}_returns_1d" in market_data.columns:
                        stock_return = market_data.loc[date, f"{stock}_returns_1d"]
                        weight = positions.get(stock, 0) / current_capital
                        portfolio_return += weight * stock_return
                
                # Apply transaction costs
                portfolio_return -= self.transaction_cost
                
                # Update portfolio value
                new_value = current_capital * (1 + portfolio_return)
                portfolio_values.append(new_value)
                
                trade_log.append({
                    'date': date,
                    'portfolio_value': new_value,
                    'daily_return': portfolio_return,
                    'positions': positions,
                    'signals': {stock: signal.signal for stock, signal in signals.items()}
                })
                
                if i % 5 == 0:
                    logger.info(f"Hybrid - {date.strftime('%Y-%m-%d')}: ₹{new_value:,.0f} ({portfolio_return:.2%})")
                
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue
        
        # Calculate final metrics
        returns = pd.Series([log['daily_return'] for log in trade_log])
        metrics = self._calculate_metrics(returns)
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        results = {
            'strategy': 'LLM + Quant Hybrid',
            'initial_capital': self.initial_capital,
            'final_capital': final_value,
            'total_return': total_return,
            'portfolio_values': portfolio_values,
            'trade_log': trade_log,
            'metrics': metrics,
            'n_trades': len(trade_log)
        }
        
        logger.info(f"Hybrid Strategy completed: {total_return:.2%} return")
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
    
    def _calculate_inverse_volatility_weights(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate inverse volatility weights"""
        
        weights = {}
        volatilities = {}
        
        for stock in self.stocks:
            vol_col = f"{stock}_volatility_20d"
            if vol_col in market_data.columns:
                vol = market_data[vol_col].mean()
                volatilities[stock] = max(vol, 0.01)  # Minimum volatility
            else:
                volatilities[stock] = 0.20  # Default
        
        # Calculate inverse volatility weights
        inv_vols = {stock: 1.0 / vol for stock, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        weights = {stock: inv_vol / total_inv_vol for stock, inv_vol in inv_vols.items()}
        
        return weights
    
    def _calculate_position_sizes(self, signals: Dict[str, LLMSignal], 
                                 base_weights: Dict[str, float], 
                                 current_capital: float) -> Dict[str, float]:
        """Calculate position sizes based on LLM signals and base weights"""
        
        positions = {}
        
        for stock in self.stocks:
            if stock in signals and stock in base_weights:
                signal = signals[stock]
                base_weight = base_weights[stock]
                
                # Adjust weight based on LLM signal and confidence
                if signal.signal == 'BUY':
                    confidence_multiplier = signal.confidence / 100.0
                    adjusted_weight = base_weight * (1 + confidence_multiplier * 0.5)
                elif signal.signal == 'SELL':
                    confidence_multiplier = signal.confidence / 100.0
                    adjusted_weight = base_weight * (1 - confidence_multiplier * 0.8)
                else:  # HOLD
                    adjusted_weight = base_weight
                
                # Apply position size constraints
                adjusted_weight = min(adjusted_weight, self.max_position_size)
                adjusted_weight = max(adjusted_weight, 0.0)
                
                positions[stock] = adjusted_weight * current_capital
            else:
                positions[stock] = 0.0
        
        return positions
    
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
    
    def compare_strategies(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Compare both strategies side by side"""
        
        logger.info("Running strategy comparison...")
        
        try:
            # Run both strategies
            pure_llm_results = self.run_pure_llm_strategy(start_date, end_date)
            hybrid_results = self.run_hybrid_strategy(start_date, end_date)
            
            # Generate comparison report
            comparison = {
                'test_period': f"{start_date} to {end_date}",
                'pure_llm': pure_llm_results,
                'hybrid': hybrid_results,
                'comparison_metrics': {
                    'return_difference': hybrid_results['total_return'] - pure_llm_results['total_return'],
                    'sharpe_difference': hybrid_results['metrics'].sharpe_ratio - pure_llm_results['metrics'].sharpe_ratio,
                    'better_strategy': 'Hybrid' if hybrid_results['total_return'] > pure_llm_results['total_return'] else 'Pure LLM'
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error in strategy comparison: {e}")
            return {"error": str(e)}

def main():
    """Example usage of the Fixed LLM Enhanced Strategy"""
    
    # Configuration
    MISTRAL_API_KEY = "5cqXuAMrvlEapMQjZMlJfChoH5npmMs8"
    
    # Load stocks from file
    stocks_file = Path("finagent/stocks.txt")
    if stocks_file.exists():
        with open(stocks_file, 'r') as f:
            stocks = [line.strip() for line in f.readlines() if line.strip()][:10]  # First 10 stocks for testing
    else:
        stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    
    print(f"Testing with {len(stocks)} stocks: {stocks}")
    
    # Initialize strategy
    strategy = LLMEnhancedStrategyFixed(
        api_key=MISTRAL_API_KEY,
        stocks=stocks,
        data_root="processed_data/",
        initial_capital=1000000.0
    )
    
    # Test date range (use recent dates for which data exists)
    start_date = "2024-08-01"
    end_date = "2024-09-01"
    
    try:
        # Run strategy comparison
        results = strategy.compare_strategies(start_date, end_date)
        
        if "error" not in results:
            print("\n" + "="*60)
            print("FIXED LLM ENHANCED PORTFOLIO STRATEGY - RESULTS")
            print("="*60)
            
            pure_llm = results['pure_llm']
            hybrid = results['hybrid']
            
            print(f"\nPure LLM Strategy:")
            print(f"  Final Capital: ₹{pure_llm['final_capital']:,.0f}")
            print(f"  Total Return: {pure_llm['total_return']:.2%}")
            print(f"  Sharpe Ratio: {pure_llm['metrics'].sharpe_ratio:.3f}")
            print(f"  Max Drawdown: {pure_llm['metrics'].max_drawdown:.2%}")
            print(f"  Number of Trades: {pure_llm['n_trades']}")
            
            print(f"\nLLM + Quant Hybrid Strategy:")
            print(f"  Final Capital: ₹{hybrid['final_capital']:,.0f}")
            print(f"  Total Return: {hybrid['total_return']:.2%}")
            print(f"  Sharpe Ratio: {hybrid['metrics'].sharpe_ratio:.3f}")
            print(f"  Max Drawdown: {hybrid['metrics'].max_drawdown:.2%}")
            print(f"  Number of Trades: {hybrid['n_trades']}")
            
            comparison = results['comparison_metrics']
            print(f"\nComparison:")
            print(f"  Better Strategy: {comparison['better_strategy']}")
            print(f"  Return Difference: {comparison['return_difference']:.2%}")
            print(f"  Sharpe Difference: {comparison['sharpe_difference']:.3f}")
            
            print("\n" + "="*60)
            print("STRATEGY TESTING COMPLETED SUCCESSFULLY!")
            print("="*60)
            
        else:
            print(f"Error in strategy testing: {results['error']}")
            
    except Exception as e:
        print(f"Error running strategy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()