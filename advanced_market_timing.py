"""
Advanced Market Timing and Risk Management for FinRL
Implements sophisticated entry/exit strategies and risk controls
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class AdvancedMarketTiming:
    """
    Advanced market timing and risk management system
    """
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.market_regime = MarketRegime.SIDEWAYS
        self.risk_level = RiskLevel.MEDIUM
        self.volatility_regime = "normal"
        
        # Risk management parameters
        self.max_position_size = 0.1  # 10% max per position
        self.max_portfolio_risk = 0.2  # 20% max portfolio risk
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.15  # 15% take profit
        
        # Market timing parameters
        self.momentum_threshold = 0.02  # 2% momentum threshold
        self.volatility_threshold = 0.03  # 3% volatility threshold
        self.correlation_threshold = 0.7  # 70% correlation threshold
        
        # Performance tracking
        self.performance_history = []
        self.drawdown_history = []
        self.volatility_history = []
    
    def detect_market_regime(self, price_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        if len(price_data) < self.lookback_period:
            return MarketRegime.SIDEWAYS
        
        # Calculate key metrics
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        momentum = (price_data['close'].iloc[-1] / price_data['close'].iloc[-20] - 1)
        trend_strength = self._calculate_trend_strength(price_data)
        
        # Regime classification
        if momentum > self.momentum_threshold and trend_strength > 0.6:
            regime = MarketRegime.BULL
        elif momentum < -self.momentum_threshold and trend_strength > 0.6:
            regime = MarketRegime.BEAR
        elif volatility > self.volatility_threshold:
            regime = MarketRegime.VOLATILE
        else:
            regime = MarketRegime.SIDEWAYS
        
        self.market_regime = regime
        return regime
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple timeframes"""
        if len(price_data) < 50:
            return 0.5
        
        # Short-term trend (20 days)
        short_trend = (price_data['close'].iloc[-1] / price_data['close'].iloc[-20] - 1)
        
        # Medium-term trend (50 days)
        medium_trend = (price_data['close'].iloc[-1] / price_data['close'].iloc[-50] - 1)
        
        # Long-term trend (200 days)
        long_trend = (price_data['close'].iloc[-1] / price_data['close'].iloc[-200] - 1)
        
        # Combine trends with weights
        trend_strength = (0.5 * short_trend + 0.3 * medium_trend + 0.2 * long_trend)
        
        # Normalize to 0-1 range
        return np.clip(abs(trend_strength), 0, 1)
    
    def assess_risk_level(self, portfolio_data: Dict, market_data: Dict) -> RiskLevel:
        """Assess current risk level"""
        # Portfolio risk metrics
        portfolio_volatility = self._calculate_portfolio_volatility(portfolio_data)
        portfolio_drawdown = self._calculate_portfolio_drawdown(portfolio_data)
        concentration_risk = self._calculate_concentration_risk(portfolio_data)
        
        # Market risk metrics
        market_volatility = market_data.get('volatility', 0)
        market_correlation = market_data.get('correlation', 0)
        
        # Risk score calculation
        risk_score = (
            0.3 * portfolio_volatility +
            0.2 * abs(portfolio_drawdown) +
            0.2 * concentration_risk +
            0.2 * market_volatility +
            0.1 * market_correlation
        )
        
        # Risk level classification
        if risk_score < 0.1:
            risk_level = RiskLevel.LOW
        elif risk_score < 0.2:
            risk_level = RiskLevel.MEDIUM
        elif risk_score < 0.3:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.EXTREME
        
        self.risk_level = risk_level
        return risk_level
    
    def _calculate_portfolio_volatility(self, portfolio_data: Dict) -> float:
        """Calculate portfolio volatility"""
        returns = portfolio_data.get('returns', [])
        if len(returns) < 20:
            return 0.1
        
        return np.std(returns[-20:]) * np.sqrt(252)
    
    def _calculate_portfolio_drawdown(self, portfolio_data: Dict) -> float:
        """Calculate current portfolio drawdown"""
        portfolio_values = portfolio_data.get('portfolio_values', [])
        if len(portfolio_values) < 20:
            return 0
        
        peak = np.maximum.accumulate(portfolio_values)
        current_drawdown = (portfolio_values[-1] - peak[-1]) / peak[-1]
        return current_drawdown
    
    def _calculate_concentration_risk(self, portfolio_data: Dict) -> float:
        """Calculate portfolio concentration risk"""
        position_weights = portfolio_data.get('position_weights', [])
        if not position_weights:
            return 0
        
        # Herfindahl index for concentration
        concentration = sum(w**2 for w in position_weights)
        return concentration
    
    def generate_market_timing_signals(self, price_data: pd.DataFrame, 
                                     technical_indicators: Dict) -> Dict[str, float]:
        """Generate market timing signals"""
        signals = {}
        
        # Trend signals
        signals['trend_signal'] = self._calculate_trend_signal(price_data)
        
        # Momentum signals
        signals['momentum_signal'] = self._calculate_momentum_signal(price_data)
        
        # Mean reversion signals
        signals['mean_reversion_signal'] = self._calculate_mean_reversion_signal(price_data)
        
        # Volatility signals
        signals['volatility_signal'] = self._calculate_volatility_signal(price_data)
        
        # Sentiment signals
        signals['sentiment_signal'] = self._calculate_sentiment_signal(technical_indicators)
        
        # Combined signal
        signals['combined_signal'] = self._combine_signals(signals)
        
        return signals
    
    def _calculate_trend_signal(self, price_data: pd.DataFrame) -> float:
        """Calculate trend following signal"""
        if len(price_data) < 50:
            return 0
        
        # Multiple timeframe trend analysis
        short_ma = price_data['close'].rolling(20).mean().iloc[-1]
        long_ma = price_data['close'].rolling(50).mean().iloc[-1]
        current_price = price_data['close'].iloc[-1]
        
        # Trend strength
        trend_strength = (current_price - long_ma) / long_ma
        
        # Trend direction
        trend_direction = 1 if short_ma > long_ma else -1
        
        return trend_direction * min(abs(trend_strength), 1.0)
    
    def _calculate_momentum_signal(self, price_data: pd.DataFrame) -> float:
        """Calculate momentum signal"""
        if len(price_data) < 20:
            return 0
        
        # Price momentum
        price_momentum = (price_data['close'].iloc[-1] / price_data['close'].iloc[-20] - 1)
        
        # Volume momentum
        volume_momentum = (price_data['volume'].iloc[-1] / price_data['volume'].rolling(20).mean().iloc[-1] - 1)
        
        # Combined momentum
        momentum_signal = 0.7 * price_momentum + 0.3 * volume_momentum
        
        return np.clip(momentum_signal, -1, 1)
    
    def _calculate_mean_reversion_signal(self, price_data: pd.DataFrame) -> float:
        """Calculate mean reversion signal"""
        if len(price_data) < 20:
            return 0
        
        # Bollinger Bands
        sma = price_data['close'].rolling(20).mean().iloc[-1]
        std = price_data['close'].rolling(20).std().iloc[-1]
        current_price = price_data['close'].iloc[-1]
        
        # Z-score
        z_score = (current_price - sma) / std
        
        # Mean reversion signal (inverse of z-score)
        mean_reversion_signal = -z_score / 2  # Scale down
        
        return np.clip(mean_reversion_signal, -1, 1)
    
    def _calculate_volatility_signal(self, price_data: pd.DataFrame) -> float:
        """Calculate volatility-based signal"""
        if len(price_data) < 20:
            return 0
        
        # Current volatility
        current_vol = price_data['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Historical volatility
        historical_vol = price_data['close'].pct_change().rolling(100).std().iloc[-1]
        
        # Volatility ratio
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
        
        # Volatility signal (higher volatility = more cautious)
        volatility_signal = -min(vol_ratio - 1, 1.0)
        
        return np.clip(volatility_signal, -1, 1)
    
    def _calculate_sentiment_signal(self, technical_indicators: Dict) -> float:
        """Calculate sentiment-based signal"""
        # RSI sentiment
        rsi = technical_indicators.get('rsi_14', 50)
        rsi_signal = (rsi - 50) / 50  # Normalize to -1 to 1
        
        # Sentiment data
        sentiment = technical_indicators.get('sentiment', 0)
        
        # Combined sentiment
        sentiment_signal = 0.6 * rsi_signal + 0.4 * sentiment
        
        return np.clip(sentiment_signal, -1, 1)
    
    def _combine_signals(self, signals: Dict[str, float]) -> float:
        """Combine all signals into final signal"""
        # Weighted combination based on market regime
        if self.market_regime == MarketRegime.BULL:
            weights = {'trend_signal': 0.4, 'momentum_signal': 0.3, 'sentiment_signal': 0.2, 'volatility_signal': 0.1}
        elif self.market_regime == MarketRegime.BEAR:
            weights = {'trend_signal': 0.3, 'mean_reversion_signal': 0.3, 'volatility_signal': 0.2, 'sentiment_signal': 0.2}
        elif self.market_regime == MarketRegime.VOLATILE:
            weights = {'volatility_signal': 0.4, 'mean_reversion_signal': 0.3, 'trend_signal': 0.2, 'sentiment_signal': 0.1}
        else:  # SIDEWAYS
            weights = {'mean_reversion_signal': 0.4, 'trend_signal': 0.2, 'momentum_signal': 0.2, 'volatility_signal': 0.2}
        
        # Calculate weighted signal
        combined_signal = sum(weights.get(signal, 0) * value for signal, value in signals.items())
        
        return np.clip(combined_signal, -1, 1)
    
    def calculate_position_size(self, signal_strength: float, volatility: float, 
                              current_position: float) -> float:
        """Calculate optimal position size using Kelly criterion and risk management"""
        # Base position size from signal
        base_size = signal_strength * self.max_position_size
        
        # Adjust for volatility
        volatility_adjustment = 1 / (1 + volatility * 10)  # Reduce size in high volatility
        
        # Adjust for risk level
        risk_adjustment = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.5,
            RiskLevel.EXTREME: 0.2
        }[self.risk_level]
        
        # Final position size
        position_size = base_size * volatility_adjustment * risk_adjustment
        
        # Ensure within limits
        position_size = np.clip(position_size, -self.max_position_size, self.max_position_size)
        
        return position_size
    
    def check_stop_loss(self, entry_price: float, current_price: float, 
                       position_type: str = 'long') -> bool:
        """Check if stop loss should be triggered"""
        if position_type == 'long':
            loss_pct = (entry_price - current_price) / entry_price
        else:  # short
            loss_pct = (current_price - entry_price) / entry_price
        
        return loss_pct >= self.stop_loss_pct
    
    def check_take_profit(self, entry_price: float, current_price: float, 
                         position_type: str = 'long') -> bool:
        """Check if take profit should be triggered"""
        if position_type == 'long':
            profit_pct = (current_price - entry_price) / entry_price
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price
        
        return profit_pct >= self.take_profit_pct
    
    def get_risk_management_actions(self, portfolio_data: Dict, 
                                  market_data: Dict) -> Dict[str, Any]:
        """Get risk management actions"""
        actions = {
            'reduce_positions': False,
            'increase_cash': False,
            'hedge_positions': False,
            'stop_trading': False
        }
        
        # Check risk level
        risk_level = self.assess_risk_level(portfolio_data, market_data)
        
        if risk_level == RiskLevel.HIGH:
            actions['reduce_positions'] = True
            actions['increase_cash'] = True
        elif risk_level == RiskLevel.EXTREME:
            actions['reduce_positions'] = True
            actions['increase_cash'] = True
            actions['hedge_positions'] = True
            actions['stop_trading'] = True
        
        return actions
    
    def update_performance_history(self, portfolio_value: float, 
                                 drawdown: float, volatility: float):
        """Update performance history for dynamic adjustments"""
        self.performance_history.append(portfolio_value)
        self.drawdown_history.append(drawdown)
        self.volatility_history.append(volatility)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
            self.drawdown_history = self.drawdown_history[-100:]
            self.volatility_history = self.volatility_history[-100:]

# Example usage
if __name__ == "__main__":
    print("Advanced market timing and risk management system ready!")
    print("Use with: timing = AdvancedMarketTiming()")
    print("Then: signals = timing.generate_market_timing_signals(price_data, indicators)")
    print("And: position_size = timing.calculate_position_size(signal, volatility, current_pos)")
