"""
Portfolio Management System with Risk Controls
Inspired by FinRL's portfolio management capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class PortfolioManager:
    """
    Portfolio management system with risk controls and position sizing
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000,
        max_position_size: float = 0.1,  # Max 10% per stock
        transaction_cost: float = 0.001,  # 0.1% transaction cost
        risk_free_rate: float = 0.05,  # 5% risk-free rate
        max_leverage: float = 1.0,  # No leverage
        rebalance_frequency: str = "daily"
    ):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.max_leverage = max_leverage
        self.rebalance_frequency = rebalance_frequency
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}  # {stock: shares}
        self.portfolio_value = initial_capital
        self.trade_history = []
        self.portfolio_history = []
        
        # Risk management
        self.var_limit = 0.05  # 5% VaR limit
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.current_drawdown = 0.0
        self.peak_value = initial_capital
        
    def get_portfolio_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Get current portfolio weights"""
        total_value = self.cash
        for stock, shares in self.positions.items():
            if stock in prices:
                total_value += shares * prices[stock]
        
        weights = {}
        for stock, shares in self.positions.items():
            if stock in prices and total_value > 0:
                weights[stock] = (shares * prices[stock]) / total_value
            else:
                weights[stock] = 0.0
        
        return weights
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.cash
        for stock, shares in self.positions.items():
            if stock in prices:
                total_value += shares * prices[stock]
        return total_value
    
    def rebalance_portfolio(
        self, 
        target_weights: Dict[str, float], 
        prices: Dict[str, float],
        date: str = None
    ) -> Dict[str, float]:
        """
        Rebalance portfolio to target weights
        
        Args:
            target_weights: Target weights for each stock
            prices: Current prices for each stock
            date: Current date for record keeping
        
        Returns:
            Actual weights after rebalancing
        """
        # Calculate current portfolio value
        current_value = self.calculate_portfolio_value(prices)
        
        # Apply risk controls
        target_weights = self._apply_risk_controls(target_weights, prices, current_value)
        
        # Calculate target positions
        target_positions = {}
        total_target_value = 0
        
        for stock, weight in target_weights.items():
            if stock in prices and weight > 0:
                target_value = current_value * weight
                target_shares = target_value / prices[stock]
                target_positions[stock] = target_shares
                total_target_value += target_value
        
        # Calculate trades needed
        trades = {}
        total_trade_value = 0
        
        for stock in set(list(self.positions.keys()) + list(target_positions.keys())):
            current_shares = self.positions.get(stock, 0)
            target_shares = target_positions.get(stock, 0)
            
            if abs(target_shares - current_shares) > 1e-6:  # Avoid tiny trades
                trade_shares = target_shares - current_shares
                trade_value = trade_shares * prices.get(stock, 0)
                trades[stock] = trade_shares
                total_trade_value += abs(trade_value)
        
        # Check if we have enough cash for purchases
        cash_needed = sum(
            trade_shares * prices[stock] 
            for stock, trade_shares in trades.items() 
            if trade_shares > 0 and stock in prices
        )
        
        if cash_needed > self.cash:
            # Scale down purchases proportionally
            scale_factor = self.cash / (cash_needed + 1e-8)
            for stock in trades:
                if trades[stock] > 0:
                    trades[stock] *= scale_factor
        
        # Execute trades
        total_transaction_cost = 0
        for stock, trade_shares in trades.items():
            if stock in prices and abs(trade_shares) > 1e-6:
                trade_value = trade_shares * prices[stock]
                transaction_cost = abs(trade_value) * self.transaction_cost
                total_transaction_cost += transaction_cost
                
                # Update positions
                self.positions[stock] = self.positions.get(stock, 0) + trade_shares
                
                # Update cash
                self.cash -= trade_value + transaction_cost
                
                # Record trade
                self.trade_history.append({
                    'date': date,
                    'stock': stock,
                    'shares': trade_shares,
                    'price': prices[stock],
                    'value': trade_value,
                    'transaction_cost': transaction_cost
                })
        
        # Update portfolio value
        self.portfolio_value = self.calculate_portfolio_value(prices)
        
        # Update drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # Record portfolio state
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'drawdown': self.current_drawdown,
            'transaction_cost': total_transaction_cost
        })
        
        return self.get_portfolio_weights(prices)
    
    def _apply_risk_controls(
        self, 
        target_weights: Dict[str, float], 
        prices: Dict[str, float],
        portfolio_value: float
    ) -> Dict[str, float]:
        """Apply risk controls to target weights"""
        # Limit position sizes
        controlled_weights = {}
        for stock, weight in target_weights.items():
            controlled_weights[stock] = min(weight, self.max_position_size)
        
        # Normalize weights
        total_weight = sum(controlled_weights.values())
        if total_weight > 0:
            for stock in controlled_weights:
                controlled_weights[stock] /= total_weight
        
        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown_limit:
            # Reduce position sizes during high drawdown
            reduction_factor = 0.5
            for stock in controlled_weights:
                controlled_weights[stock] *= reduction_factor
        
        return controlled_weights
    
    def get_portfolio_metrics(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        current_value = self.calculate_portfolio_value(prices)
        
        # Calculate returns
        if len(self.portfolio_history) > 1:
            previous_value = self.portfolio_history[-2]['portfolio_value']
            daily_return = (current_value - previous_value) / previous_value
        else:
            daily_return = 0.0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.portfolio_history) > 20:
            returns = []
            for i in range(1, min(21, len(self.portfolio_history))):
                prev_val = self.portfolio_history[-i-1]['portfolio_value']
                curr_val = self.portfolio_history[-i]['portfolio_value']
                ret = (curr_val - prev_val) / prev_val
                returns.append(ret)
            
            if len(returns) > 0:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (avg_return - self.risk_free_rate/252) / (std_return + 1e-8)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        return {
            'portfolio_value': current_value,
            'cash': self.cash,
            'daily_return': daily_return,
            'sharpe_ratio': sharpe_ratio,
            'current_drawdown': self.current_drawdown,
            'num_positions': len([p for p in self.positions.values() if abs(p) > 1e-6])
        }
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades"""
        if not self.trade_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trade_history)
        return df
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history"""
        if not self.portfolio_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.portfolio_history)
        return df


class RiskManager:
    """
    Risk management system for portfolio protection
    """
    
    def __init__(
        self,
        var_limit: float = 0.05,
        max_drawdown_limit: float = 0.15,
        position_limit: float = 0.1,
        correlation_limit: float = 0.7
    ):
        self.var_limit = var_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.position_limit = position_limit
        self.correlation_limit = correlation_limit
    
    def calculate_var(
        self, 
        returns: np.ndarray, 
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_expected_shortfall(
        self, 
        returns: np.ndarray, 
        confidence_level: float = 0.05
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) == 0:
            return 0.0
        var = self.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    def check_risk_limits(
        self, 
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame = None
    ) -> Dict[str, bool]:
        """Check if portfolio violates risk limits"""
        violations = {}
        
        # Check position limits
        max_weight = max(portfolio_weights.values()) if portfolio_weights else 0
        violations['position_limit'] = max_weight > self.position_limit
        
        # Check if we have returns data for VaR
        if returns_data is not None and len(returns_data) > 20:
            portfolio_returns = self._calculate_portfolio_returns(
                portfolio_weights, returns_data
            )
            var = self.calculate_var(portfolio_returns, self.var_limit)
            violations['var_limit'] = var < -self.var_limit
        else:
            violations['var_limit'] = False
        
        return violations
    
    def _calculate_portfolio_returns(
        self, 
        weights: Dict[str, float], 
        returns_data: pd.DataFrame
    ) -> np.ndarray:
        """Calculate portfolio returns from individual stock returns"""
        portfolio_returns = np.zeros(len(returns_data))
        
        for stock, weight in weights.items():
            if stock in returns_data.columns:
                stock_returns = returns_data[stock].values
                portfolio_returns += weight * stock_returns
        
        return portfolio_returns
    
    def suggest_risk_adjustments(
        self, 
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """Suggest adjustments to reduce risk"""
        violations = self.check_risk_limits(portfolio_weights, returns_data)
        adjusted_weights = portfolio_weights.copy()
        
        # Adjust for position limit violations
        if violations.get('position_limit', False):
            max_weight = max(adjusted_weights.values())
            if max_weight > self.position_limit:
                # Scale down all weights proportionally
                scale_factor = self.position_limit / max_weight
                for stock in adjusted_weights:
                    adjusted_weights[stock] *= scale_factor
        
        # Adjust for VaR violations
        if violations.get('var_limit', False) and returns_data is not None:
            # Reduce weights of high-risk stocks
            portfolio_returns = self._calculate_portfolio_returns(
                adjusted_weights, returns_data
            )
            
            # Calculate individual stock contributions to portfolio risk
            risk_contributions = {}
            for stock, weight in adjusted_weights.items():
                if stock in returns_data.columns:
                    stock_returns = returns_data[stock].values
                    correlation = np.corrcoef(portfolio_returns, stock_returns)[0, 1]
                    risk_contributions[stock] = weight * abs(correlation)
            
            # Reduce weights of highest risk contributors
            if risk_contributions:
                max_risk_stock = max(risk_contributions, key=risk_contributions.get)
                adjusted_weights[max_risk_stock] *= 0.8
        
        return adjusted_weights


class PositionSizer:
    """
    Position sizing strategies for portfolio management
    """
    
    @staticmethod
    def equal_weight(n_stocks: int) -> Dict[str, float]:
        """Equal weight allocation"""
        weight = 1.0 / n_stocks
        return {f'stock_{i}': weight for i in range(n_stocks)}
    
    @staticmethod
    def market_cap_weight(market_caps: Dict[str, float]) -> Dict[str, float]:
        """Market capitalization weighted allocation"""
        total_mcap = sum(market_caps.values())
        if total_mcap == 0:
            return {}
        
        weights = {}
        for stock, mcap in market_caps.items():
            weights[stock] = mcap / total_mcap
        
        return weights
    
    @staticmethod
    def risk_parity(
        returns_data: pd.DataFrame, 
        target_vol: float = 0.1
    ) -> Dict[str, float]:
        """Risk parity allocation"""
        if len(returns_data) == 0:
            return {}
        
        # Calculate individual stock volatilities
        volatilities = {}
        for stock in returns_data.columns:
            vol = returns_data[stock].std()
            volatilities[stock] = vol if vol > 0 else 0.01
        
        # Calculate inverse volatility weights
        inv_vol_weights = {}
        total_inv_vol = 0
        
        for stock, vol in volatilities.items():
            inv_vol = 1.0 / vol
            inv_vol_weights[stock] = inv_vol
            total_inv_vol += inv_vol
        
        # Normalize weights
        weights = {}
        for stock, inv_vol in inv_vol_weights.items():
            weights[stock] = inv_vol / total_inv_vol
        
        return weights
    
    @staticmethod
    def kelly_criterion(
        returns_data: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Kelly criterion for optimal position sizing"""
        if len(returns_data) == 0:
            return {}
        
        weights = {}
        for stock in returns_data.columns:
            stock_returns = returns_data[stock].dropna()
            if len(stock_returns) == 0:
                weights[stock] = 0.0
                continue
            
            # Calculate win rate and average win/loss
            positive_returns = stock_returns[stock_returns > 0]
            negative_returns = stock_returns[stock_returns < 0]
            
            if len(positive_returns) == 0 or len(negative_returns) == 0:
                weights[stock] = 0.0
                continue
            
            win_rate = len(positive_returns) / len(stock_returns)
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss > 0:
                b = avg_win / avg_loss
                kelly_fraction = (b * win_rate - (1 - win_rate)) / b
                # Cap at reasonable levels
                weights[stock] = max(0, min(kelly_fraction, 0.25))
            else:
                weights[stock] = 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for stock in weights:
                weights[stock] /= total_weight
        
        return weights
