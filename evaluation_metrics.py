"""
Comprehensive Evaluation Metrics and Backtesting Framework
Inspired by FinRL's evaluation capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class PerformanceMetrics:
    """
    Comprehensive performance evaluation metrics
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate portfolio returns"""
        return portfolio_values.pct_change().dropna()
    
    def calculate_cumulative_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate cumulative returns"""
        return (portfolio_values / portfolio_values.iloc[0]) - 1
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = None
    ) -> float:
        """Calculate Sharpe ratio"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = None
    ) -> float:
        """Calculate Sortino ratio"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() * np.sqrt(252) / downside_deviation
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_drawdown = self.calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return np.inf if annual_return > 0 else 0.0
        
        return annual_return / max_drawdown
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_volatility(self, returns: pd.Series, annualized: bool = True) -> float:
        """Calculate volatility"""
        if len(returns) == 0:
            return 0.0
        
        vol = returns.std()
        if annualized:
            vol *= np.sqrt(252)
        return vol
    
    def calculate_beta(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate beta relative to benchmark"""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_returns) < 2:
            return 0.0
        
        portfolio_aligned = aligned_returns.iloc[:, 0]
        benchmark_aligned = aligned_returns.iloc[:, 1]
        
        covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        
        if benchmark_variance == 0:
            return 0.0
        
        return covariance / benchmark_variance
    
    def calculate_alpha(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series,
        risk_free_rate: float = None
    ) -> float:
        """Calculate alpha relative to benchmark"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)
        portfolio_annual_return = portfolio_returns.mean() * 252
        benchmark_annual_return = benchmark_returns.mean() * 252
        
        return portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
    
    def calculate_information_ratio(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> float:
        """Calculate information ratio"""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_returns) < 2:
            return 0.0
        
        portfolio_aligned = aligned_returns.iloc[:, 0]
        benchmark_aligned = aligned_returns.iloc[:, 1]
        
        excess_returns = portfolio_aligned - benchmark_aligned
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return excess_returns.mean() / tracking_error * np.sqrt(252)
    
    def calculate_treynor_ratio(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series,
        risk_free_rate: float = None
    ) -> float:
        """Calculate Treynor ratio"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)
        if beta == 0:
            return 0.0
        
        portfolio_annual_return = portfolio_returns.mean() * 252
        excess_return = portfolio_annual_return - risk_free_rate
        
        return excess_return / beta
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0.0
        return (returns > 0).sum() / len(returns)
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        if len(returns) == 0:
            return 0.0
        
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        
        if negative_returns == 0:
            return np.inf if positive_returns > 0 else 0.0
        
        return positive_returns / negative_returns
    
    def calculate_recovery_factor(self, returns: pd.Series) -> float:
        """Calculate recovery factor"""
        if len(returns) == 0:
            return 0.0
        
        total_return = returns.sum()
        max_drawdown = self.calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return np.inf if total_return > 0 else 0.0
        
        return total_return / max_drawdown
    
    def get_comprehensive_metrics(
        self, 
        portfolio_values: pd.Series,
        benchmark_values: pd.Series = None
    ) -> Dict[str, float]:
        """Get comprehensive performance metrics"""
        returns = self.calculate_returns(portfolio_values)
        cumulative_returns = self.calculate_cumulative_returns(portfolio_values)
        
        metrics = {
            'total_return': cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0.0,
            'annualized_return': returns.mean() * 252 if len(returns) > 0 else 0.0,
            'volatility': self.calculate_volatility(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'var_95': self.calculate_var(returns, 0.05),
            'cvar_95': self.calculate_cvar(returns, 0.05),
            'win_rate': self.calculate_win_rate(returns),
            'profit_factor': self.calculate_profit_factor(returns),
            'recovery_factor': self.calculate_recovery_factor(returns)
        }
        
        # Add benchmark-relative metrics if benchmark provided
        if benchmark_values is not None:
            benchmark_returns = self.calculate_returns(benchmark_values)
            metrics.update({
                'beta': self.calculate_beta(returns, benchmark_returns),
                'alpha': self.calculate_alpha(returns, benchmark_returns),
                'information_ratio': self.calculate_information_ratio(returns, benchmark_returns),
                'treynor_ratio': self.calculate_treynor_ratio(returns, benchmark_returns)
            })
        
        return metrics


class Backtester:
    """
    Backtesting framework for trading strategies
    """
    
    def __init__(
        self,
        initial_capital: float = 1000000,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.05
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.metrics_calculator = PerformanceMetrics(risk_free_rate)
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_weights: pd.DataFrame,
        benchmark_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data
        
        Args:
            data: Historical price data
            strategy_weights: Strategy weight allocations over time
            benchmark_data: Benchmark data for comparison
        
        Returns:
            Backtest results including performance metrics
        """
        # Initialize portfolio
        portfolio_values = []
        portfolio_returns = []
        dates = []
        
        current_value = self.initial_capital
        portfolio_values.append(current_value)
        dates.append(strategy_weights.index[0])
        
        # Run backtest
        for i in range(1, len(strategy_weights)):
            current_date = strategy_weights.index[i]
            previous_date = strategy_weights.index[i-1]
            
            # Get current and previous weights
            current_weights = strategy_weights.iloc[i]
            previous_weights = strategy_weights.iloc[i-1]
            
            # Calculate portfolio return
            if current_date in data.index and previous_date in data.index:
                # Get price returns
                current_prices = data.loc[current_date]
                previous_prices = data.loc[previous_date]
                
                # Calculate asset returns
                asset_returns = (current_prices / previous_prices) - 1
                
                # Calculate portfolio return
                portfolio_return = (current_weights * asset_returns).sum()
                
                # Apply transaction costs
                weight_change = abs(current_weights - previous_weights).sum()
                transaction_cost = weight_change * self.transaction_cost
                
                # Net portfolio return
                net_return = portfolio_return - transaction_cost
                
                # Update portfolio value
                current_value *= (1 + net_return)
                
                portfolio_values.append(current_value)
                portfolio_returns.append(net_return)
                dates.append(current_date)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values,
            'portfolio_return': [0] + portfolio_returns
        })
        results_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        portfolio_series = pd.Series(portfolio_values, index=dates)
        metrics = self.metrics_calculator.get_comprehensive_metrics(portfolio_series)
        
        # Add benchmark comparison if provided
        benchmark_metrics = {}
        if benchmark_data is not None:
            benchmark_series = benchmark_data.reindex(dates).fillna(method='ffill')
            benchmark_metrics = self.metrics_calculator.get_comprehensive_metrics(
                portfolio_series, benchmark_series
            )
        
        return {
            'portfolio_values': portfolio_series,
            'portfolio_returns': pd.Series(portfolio_returns, index=dates[1:]),
            'metrics': metrics,
            'benchmark_metrics': benchmark_metrics,
            'results_df': results_df
        }
    
    def plot_performance(
        self, 
        backtest_results: Dict[str, Any],
        benchmark_data: pd.Series = None,
        save_path: str = None
    ):
        """Plot backtest performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        portfolio_values = backtest_results['portfolio_values']
        portfolio_returns = backtest_results['portfolio_returns']
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_values.index, portfolio_values.values, label='Portfolio', linewidth=2)
        if benchmark_data is not None:
            benchmark_aligned = benchmark_data.reindex(portfolio_values.index).fillna(method='ffill')
            benchmark_normalized = benchmark_aligned / benchmark_aligned.iloc[0] * portfolio_values.iloc[0]
            axes[0, 0].plot(benchmark_normalized.index, benchmark_normalized.values, label='Benchmark', linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative_returns = (portfolio_values / portfolio_values.iloc[0]) - 1
        axes[0, 1].plot(cumulative_returns.index, cumulative_returns.values, linewidth=2)
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[1, 1].hist(portfolio_returns.values, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Returns Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(
        self, 
        backtest_results: Dict[str, Any],
        benchmark_data: pd.Series = None
    ) -> str:
        """Generate comprehensive backtest report"""
        metrics = backtest_results['metrics']
        portfolio_values = backtest_results['portfolio_values']
        
        report = f"""
        BACKTEST REPORT
        ===============
        
        Portfolio Performance:
        ---------------------
        Total Return: {metrics['total_return']:.2%}
        Annualized Return: {metrics['annualized_return']:.2%}
        Volatility: {metrics['volatility']:.2%}
        Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
        Sortino Ratio: {metrics['sortino_ratio']:.3f}
        Calmar Ratio: {metrics['calmar_ratio']:.3f}
        Maximum Drawdown: {metrics['max_drawdown']:.2%}
        VaR (95%): {metrics['var_95']:.2%}
        CVaR (95%): {metrics['cvar_95']:.2%}
        Win Rate: {metrics['win_rate']:.2%}
        Profit Factor: {metrics['profit_factor']:.3f}
        Recovery Factor: {metrics['recovery_factor']:.3f}
        
        """
        
        if benchmark_data is not None and 'benchmark_metrics' in backtest_results:
            benchmark_metrics = backtest_results['benchmark_metrics']
            report += f"""
        Benchmark Comparison:
        ---------------------
        Beta: {benchmark_metrics.get('beta', 'N/A'):.3f}
        Alpha: {benchmark_metrics.get('alpha', 'N/A'):.2%}
        Information Ratio: {benchmark_metrics.get('information_ratio', 'N/A'):.3f}
        Treynor Ratio: {benchmark_metrics.get('treynor_ratio', 'N/A'):.3f}
        
        """
        
        report += f"""
        Risk Metrics:
        -------------
        Risk-Free Rate: {self.risk_free_rate:.2%}
        Transaction Cost: {self.transaction_cost:.3%}
        Initial Capital: ${self.initial_capital:,.0f}
        Final Portfolio Value: ${portfolio_values.iloc[-1]:,.0f}
        
        """
        
        return report


class StrategyComparator:
    """
    Compare multiple trading strategies
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.metrics_calculator = PerformanceMetrics(risk_free_rate)
    
    def compare_strategies(
        self, 
        strategy_results: Dict[str, Dict[str, Any]],
        benchmark_data: pd.Series = None
    ) -> pd.DataFrame:
        """Compare multiple strategies"""
        comparison_data = []
        
        for strategy_name, results in strategy_results.items():
            portfolio_values = results['portfolio_values']
            metrics = self.metrics_calculator.get_comprehensive_metrics(portfolio_values)
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': metrics['total_return'],
                'Annualized Return': metrics['annualized_return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Sortino Ratio': metrics['sortino_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Win Rate': metrics['win_rate'],
                'Profit Factor': metrics['profit_factor']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.set_index('Strategy', inplace=True)
        
        return comparison_df
    
    def plot_strategy_comparison(
        self, 
        strategy_results: Dict[str, Dict[str, Any]],
        benchmark_data: pd.Series = None,
        save_path: str = None
    ):
        """Plot comparison of multiple strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio values comparison
        for strategy_name, results in strategy_results.items():
            portfolio_values = results['portfolio_values']
            normalized_values = portfolio_values / portfolio_values.iloc[0]
            axes[0, 0].plot(normalized_values.index, normalized_values.values, 
                           label=strategy_name, linewidth=2)
        
        if benchmark_data is not None:
            benchmark_normalized = benchmark_data / benchmark_data.iloc[0]
            axes[0, 0].plot(benchmark_normalized.index, benchmark_normalized.values, 
                           label='Benchmark', linewidth=2, linestyle='--')
        
        axes[0, 0].set_title('Strategy Comparison - Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Risk-Return scatter
        returns = []
        volatilities = []
        strategy_names = []
        
        for strategy_name, results in strategy_results.items():
            portfolio_values = results['portfolio_values']
            portfolio_returns = self.metrics_calculator.calculate_returns(portfolio_values)
            annual_return = portfolio_returns.mean() * 252
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            
            returns.append(annual_return)
            volatilities.append(annual_vol)
            strategy_names.append(strategy_name)
        
        axes[0, 1].scatter(volatilities, returns, s=100, alpha=0.7)
        for i, name in enumerate(strategy_names):
            axes[0, 1].annotate(name, (volatilities[i], returns[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_xlabel('Volatility')
        axes[0, 1].set_ylabel('Annual Return')
        axes[0, 1].set_title('Risk-Return Profile')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe ratio comparison
        sharpe_ratios = []
        for strategy_name, results in strategy_results.items():
            portfolio_values = results['portfolio_values']
            portfolio_returns = self.metrics_calculator.calculate_returns(portfolio_values)
            sharpe = self.metrics_calculator.calculate_sharpe_ratio(portfolio_returns)
            sharpe_ratios.append(sharpe)
        
        axes[1, 0].bar(strategy_names, sharpe_ratios, alpha=0.7)
        axes[1, 0].set_title('Sharpe Ratio Comparison')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Drawdown comparison
        for strategy_name, results in strategy_results.items():
            portfolio_values = results['portfolio_values']
            portfolio_returns = self.metrics_calculator.calculate_returns(portfolio_values)
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max
            axes[1, 1].plot(drawdown.index, drawdown.values, label=strategy_name, linewidth=2)
        
        axes[1, 1].set_title('Drawdown Comparison')
        axes[1, 1].set_ylabel('Drawdown')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
