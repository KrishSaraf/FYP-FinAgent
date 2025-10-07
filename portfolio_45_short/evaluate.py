"""
Evaluation and backtesting script for trained portfolio models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Custom imports
from envs import Portfolio45ShortEnv
from utils.metrics import (
    calculate_portfolio_metrics,
    calculate_rolling_metrics,
    calculate_trade_analysis
)
from data.synthetic_data import generate_synthetic_data


class PortfolioEvaluator:
    """Portfolio model evaluator and backtester."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        model_path: str,
        vec_normalize_path: str,
        test_start_date: str,
        test_end_date: str,
        results_save_path: str = "evaluation_results",
        **env_kwargs
    ):
        """
        Initialize evaluator.
        
        Args:
            data: Full dataset
            tickers: List of stock tickers
            model_path: Path to trained model
            vec_normalize_path: Path to VecNormalize parameters
            test_start_date: Test start date
            test_end_date: Test end date
            results_save_path: Path to save evaluation results
            **env_kwargs: Additional environment parameters
        """
        self.data = data
        self.tickers = tickers
        self.model_path = model_path
        self.vec_normalize_path = vec_normalize_path
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.results_save_path = results_save_path
        self.env_kwargs = env_kwargs
        
        # Create directories
        os.makedirs(results_save_path, exist_ok=True)
        
        # Prepare test data
        self._prepare_test_data()
        
        # Initialize model and environment
        self.model = None
        self.test_env = None
    
    def _prepare_test_data(self):
        """Prepare test data."""
        self.test_data = self.data[
            (self.data['date'] >= self.test_start_date) & 
            (self.data['date'] <= self.test_end_date)
        ].copy()
        
        print(f"Test data: {len(self.test_data)} records")
        print(f"Test period: {self.test_data['date'].min()} to {self.test_data['date'].max()}")
    
    def load_model(self):
        """Load trained model and environment."""
        print("📥 Loading trained model...")
        
        # Load model
        self.model = PPO.load(self.model_path)
        
        # Create test environment
        self.test_env = Portfolio45ShortEnv(
            data=self.test_data,
            tickers=self.tickers,
            random_start=False,
            **self.env_kwargs
        )
        
        # Wrap with Monitor and VecEnv
        from stable_baselines3.common.monitor import Monitor
        self.test_env = Monitor(self.test_env)
        self.test_env = DummyVecEnv([lambda: self.test_env])
        
        # Load normalization parameters
        self.test_env = VecNormalize.load(self.vec_normalize_path, self.test_env)
        self.test_env.training = False  # Disable training mode
        
        print("✅ Model loaded successfully!")
    
    def run_backtest(
        self, 
        deterministic: bool = True,
        save_trades: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest.
        
        Args:
            deterministic: Whether to use deterministic actions
            save_trades: Whether to save detailed trades
            
        Returns:
            Backtest results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("📈 Running comprehensive backtest...")
        
        # Reset environment
        obs = self.test_env.reset()
        done = False
        
        # Initialize tracking
        portfolio_values = []
        returns = []
        actions = []
        weights_history = []
        trades_log = []
        costs_log = []
        info_log = []
        
        step = 0
        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=deterministic)
            actions.append(action[0].copy())
            
            # Step environment
            obs, reward, done, info = self.test_env.step(action)
            
            # Store results
            portfolio_values.append(info[0]['portfolio_value'])
            returns.append(reward[0])
            weights_history.append(info[0]['actual_weights'])
            costs_log.append(info[0]['costs'])
            info_log.append(info[0])
            
            if 'trades' in info[0]:
                trades_log.extend(info[0]['trades'])
            
            step += 1
            if step % 50 == 0:
                print(f"  Step {step}: Portfolio Value = ${info[0]['portfolio_value']:,.2f}")
        
        # Convert to arrays
        portfolio_values = np.array(portfolio_values)
        returns = np.array(returns)
        weights_array = np.array(weights_history)
        
        # Calculate comprehensive metrics
        metrics = calculate_portfolio_metrics(
            returns, portfolio_values, weights_array
        )
        
        # Calculate rolling metrics
        rolling_metrics = calculate_rolling_metrics(
            returns, portfolio_values, window=60  # 60-day rolling window
        )
        
        # Analyze trades
        if trades_log:
            trades_df = pd.DataFrame(trades_log)
            trade_analysis = calculate_trade_analysis(trades_df)
        else:
            trades_df = pd.DataFrame()
            trade_analysis = {}
        
        # Calculate additional metrics
        additional_metrics = self._calculate_additional_metrics(
            portfolio_values, returns, weights_array, costs_log
        )
        
        # Combine all results
        backtest_results = {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'actions': actions,
            'weights_history': weights_history,
            'trades_log': trades_log,
            'trades_df': trades_df,
            'costs_log': costs_log,
            'info_log': info_log,
            'metrics': metrics,
            'rolling_metrics': rolling_metrics,
            'trade_analysis': trade_analysis,
            'additional_metrics': additional_metrics,
        }
        
        print("✅ Backtest completed!")
        self._print_summary(metrics, trade_analysis)
        
        return backtest_results
    
    def _calculate_additional_metrics(
        self, 
        portfolio_values: np.ndarray,
        returns: np.ndarray,
        weights_array: np.ndarray,
        costs_log: List[Dict]
    ) -> Dict[str, float]:
        """Calculate additional performance metrics."""
        # Calculate daily costs
        daily_costs = [cost['total'] for cost in costs_log]
        total_costs = sum(daily_costs)
        
        # Calculate cost-adjusted returns
        cost_adjusted_returns = returns - np.array(daily_costs) / portfolio_values[:-1]
        
        # Calculate exposure statistics
        gross_exposures = [np.sum(np.abs(w)) for w in weights_array]
        net_exposures = [np.sum(w) for w in weights_array]
        
        # Calculate position concentration
        concentrations = [np.sum(w**2) for w in weights_array]  # Herfindahl index
        
        # Calculate turnover
        if len(weights_array) > 1:
            weight_changes = np.diff(weights_array, axis=0)
            daily_turnover = np.sum(np.abs(weight_changes), axis=1)
            avg_turnover = np.mean(daily_turnover)
        else:
            avg_turnover = 0.0
        
        return {
            'total_costs': total_costs,
            'avg_daily_costs': np.mean(daily_costs),
            'cost_ratio': total_costs / portfolio_values[0],
            'avg_gross_exposure': np.mean(gross_exposures),
            'avg_net_exposure': np.mean(net_exposures),
            'max_gross_exposure': np.max(gross_exposures),
            'min_net_exposure': np.min(net_exposures),
            'max_net_exposure': np.max(net_exposures),
            'avg_concentration': np.mean(concentrations),
            'avg_turnover': avg_turnover,
            'cost_adjusted_sharpe': calculate_portfolio_metrics(
                cost_adjusted_returns, portfolio_values, weights_array
            )['sharpe_ratio'],
        }
    
    def _print_summary(self, metrics: Dict, trade_analysis: Dict):
        """Print backtest summary."""
        print("\n📊 Backtest Summary")
        print("=" * 50)
        
        print("📈 Performance Metrics:")
        print(f"  Total Return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
        print(f"  Annualized Return: {metrics['annualized_return']:.4f} ({metrics['annualized_return']*100:.2f}%)")
        print(f"  Volatility: {metrics['volatility']:.4f} ({metrics['volatility']*100:.2f}%)")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        
        print("\n📊 Exposure Metrics:")
        print(f"  Avg Gross Exposure: {metrics['gross_exposure']:.4f}")
        print(f"  Avg Net Exposure: {metrics['net_exposure']:.4f}")
        print(f"  Avg Short Notional: ${metrics['short_notional']:,.2f}")
        
        print("\n📊 Trading Metrics:")
        print(f"  Avg Turnover: {metrics['turnover']:.4f}")
        
        if trade_analysis:
            print(f"  Total Trades: {trade_analysis.get('total_trades', 0)}")
            print(f"  Total Volume: ${trade_analysis.get('total_volume', 0):,.2f}")
            print(f"  Total Commissions: ${trade_analysis.get('total_commissions', 0):,.2f}")
            print(f"  Total Borrow Fees: ${trade_analysis.get('total_borrow_fees', 0):,.2f}")
    
    def create_plots(self, backtest_results: Dict[str, Any]):
        """Create comprehensive evaluation plots."""
        print("📊 Creating evaluation plots...")
        
        portfolio_values = backtest_results['portfolio_values']
        returns = backtest_results['returns']
        weights_history = backtest_results['weights_history']
        rolling_metrics = backtest_results['rolling_metrics']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Portfolio value over time
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(portfolio_values)
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 2. Daily returns distribution
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.4f}')
        ax2.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = plt.subplot(3, 3, 3)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        ax3.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
        ax3.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Trading Days')
        ax3.set_ylabel('Drawdown')
        ax3.grid(True, alpha=0.3)
        
        # 4. Gross exposure over time
        ax4 = plt.subplot(3, 3, 4)
        gross_exposure = [np.sum(np.abs(w)) for w in weights_history]
        ax4.plot(gross_exposure, color='green')
        ax4.axhline(y=1.5, color='red', linestyle='--', label='Gross Cap (1.5)')
        ax4.set_title('Gross Exposure Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Trading Days')
        ax4.set_ylabel('Gross Exposure')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Net exposure over time
        ax5 = plt.subplot(3, 3, 5)
        net_exposure = [np.sum(w) for w in weights_history]
        ax5.plot(net_exposure, color='blue')
        ax5.axhline(y=1.0, color='red', linestyle='--', label='Target Net (1.0)')
        ax5.set_title('Net Exposure Over Time', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Trading Days')
        ax5.set_ylabel('Net Exposure')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Rolling Sharpe ratio
        ax6 = plt.subplot(3, 3, 6)
        if not rolling_metrics.empty:
            ax6.plot(rolling_metrics['sharpe_ratio'], color='purple')
            ax6.set_title('Rolling Sharpe Ratio (60-day)', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Trading Days')
            ax6.set_ylabel('Sharpe Ratio')
            ax6.grid(True, alpha=0.3)
        
        # 7. Position weights heatmap (top 10 stocks)
        ax7 = plt.subplot(3, 3, 7)
        weights_array = np.array(weights_history)
        # Get top 10 stocks by average absolute weight
        avg_abs_weights = np.mean(np.abs(weights_array), axis=0)
        top_10_indices = np.argsort(avg_abs_weights)[-10:]
        top_10_weights = weights_array[:, top_10_indices]
        top_10_tickers = [self.tickers[i] for i in top_10_indices]
        
        im = ax7.imshow(top_10_weights.T, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        ax7.set_title('Top 10 Stock Weights Over Time', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Trading Days')
        ax7.set_ylabel('Stocks')
        ax7.set_yticks(range(len(top_10_tickers)))
        ax7.set_yticklabels(top_10_tickers)
        plt.colorbar(im, ax=ax7, label='Weight')
        
        # 8. Cumulative returns
        ax8 = plt.subplot(3, 3, 8)
        cumulative_returns = np.cumprod(1 + returns) - 1
        ax8.plot(cumulative_returns, color='green', linewidth=2)
        ax8.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Trading Days')
        ax8.set_ylabel('Cumulative Return')
        ax8.grid(True, alpha=0.3)
        
        # 9. Risk-return scatter
        ax9 = plt.subplot(3, 3, 9)
        if not rolling_metrics.empty:
            ax9.scatter(rolling_metrics['volatility'], rolling_metrics['annualized_return'], 
                       alpha=0.6, s=20)
            ax9.set_title('Risk-Return Profile (Rolling)', fontsize=14, fontweight='bold')
            ax9.set_xlabel('Volatility')
            ax9.set_ylabel('Annualized Return')
            ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.results_save_path}/evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_path}")
        
        plt.show()
    
    def save_results(self, backtest_results: Dict[str, Any]):
        """Save evaluation results to files."""
        print("💾 Saving evaluation results...")
        
        # Save portfolio values and returns
        portfolio_df = pd.DataFrame({
            'portfolio_value': backtest_results['portfolio_values'],
            'daily_return': backtest_results['returns']
        })
        portfolio_df.to_csv(f"{self.results_save_path}/portfolio_values.csv", index=False)
        
        # Save weights history
        weights_df = pd.DataFrame(
            backtest_results['weights_history'],
            columns=self.tickers
        )
        weights_df.to_csv(f"{self.results_save_path}/weights_history.csv", index=False)
        
        # Save trades log
        if not backtest_results['trades_df'].empty:
            backtest_results['trades_df'].to_csv(
                f"{self.results_save_path}/trades_log.csv", index=False
            )
        
        # Save metrics
        metrics_df = pd.DataFrame([backtest_results['metrics']])
        metrics_df.to_csv(f"{self.results_save_path}/metrics.csv", index=False)
        
        # Save additional metrics
        additional_metrics_df = pd.DataFrame([backtest_results['additional_metrics']])
        additional_metrics_df.to_csv(f"{self.results_save_path}/additional_metrics.csv", index=False)
        
        # Save rolling metrics
        if not backtest_results['rolling_metrics'].empty:
            backtest_results['rolling_metrics'].to_csv(
                f"{self.results_save_path}/rolling_metrics.csv", index=False
            )
        
        # Save trade analysis
        if backtest_results['trade_analysis']:
            trade_analysis_df = pd.DataFrame([backtest_results['trade_analysis']])
            trade_analysis_df.to_csv(f"{self.results_save_path}/trade_analysis.csv", index=False)
        
        print(f"✅ Results saved to: {self.results_save_path}")


def main():
    """Main evaluation function."""
    print("🔍 Portfolio Model Evaluation")
    print("=" * 50)
    
    # Load real processed data
    print("📊 Loading data...")
    data, tickers = generate_synthetic_data(
        n_stocks=45,
        n_days=252,  # Ignored - uses actual data range
        start_date='2023-01-01'  # Ignored - uses actual data range
    )
    
    # Define test period
    test_start = '2023-09-01'
    test_end = '2023-12-31'
    
    # Initialize evaluator
    evaluator = PortfolioEvaluator(
        data=data,
        tickers=tickers,
        model_path="models/best_model.zip",  # Path to trained model
        vec_normalize_path="models/vec_normalize.pkl",
        test_start_date=test_start,
        test_end_date=test_end,
        results_save_path="evaluation_results",
        # Environment parameters (should match training)
        initial_capital=1_000_000.0,
        commission_bps=1.0,
        slippage_bps=2.0,
        borrow_rate_annual=0.03,
        w_max=0.10,
        gross_cap=1.5,
        target_net=1.0,
    )
    
    # Load model
    evaluator.load_model()
    
    # Run backtest
    backtest_results = evaluator.run_backtest(deterministic=True)
    
    # Create plots
    evaluator.create_plots(backtest_results)
    
    # Save results
    evaluator.save_results(backtest_results)
    
    print("🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
