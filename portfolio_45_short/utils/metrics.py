"""
Performance metrics and evaluation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def calculate_sharpe_ratio(
    returns: np.ndarray, 
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Daily returns array
        risk_free_rate: Annual risk-free rate
        annualization_factor: Days per year for annualization
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization_factor)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation).
    
    Args:
        returns: Daily returns array
        risk_free_rate: Annual risk-free rate
        annualization_factor: Days per year for annualization
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(annualization_factor)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        portfolio_values: Portfolio value time series
        
    Returns:
        Maximum drawdown (negative value)
    """
    if len(portfolio_values) == 0:
        return 0.0
    
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown)


def calculate_calmar_ratio(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
    annualization_factor: int = 252
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Daily returns array
        portfolio_values: Portfolio value time series
        annualization_factor: Days per year for annualization
        
    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    annualized_return = np.mean(returns) * annualization_factor
    max_dd = abs(calculate_max_drawdown(portfolio_values))
    
    if max_dd == 0:
        return np.inf if annualized_return > 0 else 0.0
    
    return annualized_return / max_dd


def calculate_turnover(weights_history: np.ndarray) -> float:
    """
    Calculate average daily turnover.
    
    Args:
        weights_history: Weight matrix (n_days, n_stocks)
        
    Returns:
        Average daily turnover
    """
    if len(weights_history) < 2:
        return 0.0
    
    weight_changes = np.diff(weights_history, axis=0)
    daily_turnover = np.sum(np.abs(weight_changes), axis=1)
    return np.mean(daily_turnover)


def calculate_gross_exposure(weights_history: np.ndarray) -> float:
    """
    Calculate average gross exposure.
    
    Args:
        weights_history: Weight matrix (n_days, n_stocks)
        
    Returns:
        Average gross exposure
    """
    if len(weights_history) == 0:
        return 0.0
    
    gross_exposure = np.sum(np.abs(weights_history), axis=1)
    return np.mean(gross_exposure)


def calculate_net_exposure(weights_history: np.ndarray) -> float:
    """
    Calculate average net exposure.
    
    Args:
        weights_history: Weight matrix (n_days, n_stocks)
        
    Returns:
        Average net exposure
    """
    if len(weights_history) == 0:
        return 0.0
    
    net_exposure = np.sum(weights_history, axis=1)
    return np.mean(net_exposure)


def calculate_short_notional(weights_history: np.ndarray, portfolio_values: np.ndarray) -> float:
    """
    Calculate average short notional value.
    
    Args:
        weights_history: Weight matrix (n_days, n_stocks)
        portfolio_values: Portfolio value time series
        
    Returns:
        Average short notional value
    """
    if len(weights_history) == 0:
        return 0.0
    
    short_weights = np.minimum(weights_history, 0)
    short_notional = np.sum(np.abs(short_weights), axis=1) * portfolio_values
    return np.mean(short_notional)


def calculate_portfolio_metrics(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
    weights_history: np.ndarray,
    risk_free_rate: float = 0.02,
    annualization_factor: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio performance metrics.
    
    Args:
        returns: Daily returns array
        portfolio_values: Portfolio value time series
        weights_history: Weight matrix (n_days, n_stocks)
        risk_free_rate: Annual risk-free rate
        annualization_factor: Days per year for annualization
        
    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'turnover': 0.0,
            'gross_exposure': 0.0,
            'net_exposure': 0.0,
            'short_notional': 0.0,
        }
    
    # Basic return metrics
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    annualized_return = np.mean(returns) * annualization_factor
    volatility = np.std(returns) * np.sqrt(annualization_factor)
    
    # Risk-adjusted metrics
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, annualization_factor)
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, annualization_factor)
    max_drawdown = calculate_max_drawdown(portfolio_values)
    calmar_ratio = calculate_calmar_ratio(returns, portfolio_values, annualization_factor)
    
    # Trading metrics
    turnover = calculate_turnover(weights_history)
    gross_exposure = calculate_gross_exposure(weights_history)
    net_exposure = calculate_net_exposure(weights_history)
    short_notional = calculate_short_notional(weights_history, portfolio_values)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'turnover': turnover,
        'gross_exposure': gross_exposure,
        'net_exposure': net_exposure,
        'short_notional': short_notional,
    }


def calculate_rolling_metrics(
    returns: np.ndarray,
    portfolio_values: np.ndarray,
    window: int = 252,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        returns: Daily returns array
        portfolio_values: Portfolio value time series
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with rolling metrics
    """
    if len(returns) < window:
        return pd.DataFrame()
    
    n_periods = len(returns) - window + 1
    rolling_metrics = []
    
    for i in range(n_periods):
        start_idx = i
        end_idx = i + window
        
        period_returns = returns[start_idx:end_idx]
        period_values = portfolio_values[start_idx:end_idx]
        
        metrics = calculate_portfolio_metrics(
            period_returns, period_values, 
            np.array([]), risk_free_rate
        )
        
        rolling_metrics.append(metrics)
    
    return pd.DataFrame(rolling_metrics)


def calculate_trade_analysis(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze trading activity from trades ledger.
    
    Args:
        trades_df: DataFrame with columns ['date', 'ticker', 'quantity', 'exec_price', 'side', 'notional', 'commission', 'borrow_fee']
        
    Returns:
        Dictionary with trade analysis metrics
    """
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'total_volume': 0.0,
            'total_commissions': 0.0,
            'total_borrow_fees': 0.0,
            'avg_trade_size': 0.0,
            'buy_trades': 0,
            'sell_trades': 0,
            'short_trades': 0,
        }
    
    total_trades = len(trades_df)
    total_volume = trades_df['notional'].abs().sum()
    total_commissions = trades_df['commission'].sum()
    total_borrow_fees = trades_df['borrow_fee'].sum()
    avg_trade_size = trades_df['notional'].abs().mean()
    
    buy_trades = len(trades_df[trades_df['side'] == 'buy'])
    sell_trades = len(trades_df[trades_df['side'] == 'sell'])
    short_trades = len(trades_df[trades_df['quantity'] < 0])
    
    return {
        'total_trades': total_trades,
        'total_volume': total_volume,
        'total_commissions': total_commissions,
        'total_borrow_fees': total_borrow_fees,
        'avg_trade_size': avg_trade_size,
        'buy_trades': buy_trades,
        'sell_trades': sell_trades,
        'short_trades': short_trades,
    }
