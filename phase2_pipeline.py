"""
Phase 2: Advanced Portfolio Management Pipeline
Comprehensive training and evaluation pipeline for multi-stock strategies
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Custom imports
from data_loader import FinancialDataLoader
from portfolio_environment import PortfolioTradingEnv
from ensemble_agents import EnsembleAgent
from hyperparameter_optimizer import HyperparameterOptimizer

class Phase2Pipeline:
    """
    Comprehensive Phase 2 pipeline for advanced portfolio management
    Optimized to achieve >10% returns with sophisticated strategies
    """
    
    def __init__(self, 
                 stock_list: List[str] = None,
                 data_path: str = "processed_data",
                 model_save_path: str = "phase2_models",
                 results_path: str = "phase2_results",
                 target_return: float = 0.10):
        """
        Initialize Phase 2 pipeline
        
        Args:
            stock_list: List of stock symbols (if None, uses top stocks)
            data_path: Path to processed data
            model_save_path: Path to save models
            results_path: Path to save results
            target_return: Target annual return (10%)
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.results_path = results_path
        self.target_return = target_return
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = FinancialDataLoader(data_path)
        
        # Set stock list
        if stock_list is None:
            self.stock_list = self._select_top_stocks()
        else:
            self.stock_list = stock_list
        
        # Initialize components
        self.df_dict = {}
        self.ensemble_agent = None
        self.optimizer = None
        
        # Performance tracking
        self.training_history = []
        self.evaluation_results = {}
        self.optimization_results = {}
    
    def _select_top_stocks(self, n_stocks: int = 10) -> List[str]:
        """
        Select top performing stocks based on data availability and quality
        
        Args:
            n_stocks: Number of stocks to select
            
        Returns:
            List of selected stock symbols
        """
        available_stocks = self.data_loader.get_available_stocks()
        
        # Priority stocks (large cap, liquid stocks)
        priority_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "ITC", "KOTAKBANK", "BHARTIARTL", "LT",
            "SBIN", "ASIANPAINT", "MARUTI", "AXISBANK", "NESTLEIND"
        ]
        
        # Select stocks that are available and in priority list
        selected_stocks = []
        for stock in priority_stocks:
            if stock in available_stocks and len(selected_stocks) < n_stocks:
                selected_stocks.append(stock)
        
        # Fill remaining slots with other available stocks
        for stock in available_stocks:
            if stock not in selected_stocks and len(selected_stocks) < n_stocks:
                selected_stocks.append(stock)
        
        print(f"Selected {len(selected_stocks)} stocks: {selected_stocks}")
        return selected_stocks
    
    def load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and prepare data for all selected stocks
        
        Returns:
            Dictionary of processed DataFrames
        """
        print("📊 Loading and Preparing Multi-Stock Data")
        print("=" * 50)
        
        successful_loads = 0
        
        for stock in self.stock_list:
            try:
                print(f"Loading {stock}...")
                
                # Load raw data
                raw_data = self.data_loader.load_stock_data(stock)
                
                # Preprocess data
                processed_data = self.data_loader.preprocess_data(raw_data)
                
                # Store processed data
                self.df_dict[stock] = processed_data
                successful_loads += 1
                
                print(f"✅ {stock}: {processed_data.shape}")
                
            except Exception as e:
                print(f"❌ Failed to load {stock}: {e}")
        
        print(f"\n📈 Successfully loaded {successful_loads}/{len(self.stock_list)} stocks")
        
        if successful_loads < 3:
            raise ValueError("Need at least 3 stocks for portfolio management")
        
        # Update stock list to only include successfully loaded stocks
        self.stock_list = list(self.df_dict.keys())
        
        return self.df_dict
    
    def run_hyperparameter_optimization(self, 
                                      optimization_level: str = 'comprehensive',
                                      n_trials: int = 50) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Args:
            optimization_level: 'quick', 'standard', or 'comprehensive'
            n_trials: Number of optimization trials
            
        Returns:
            Optimization results
        """
        print("🎯 Hyperparameter Optimization")
        print("=" * 40)
        
        # Initialize optimizer
        self.optimizer = HyperparameterOptimizer(
            stock_list=self.stock_list,
            df_dict=self.df_dict,
            optimization_target='sharpe_ratio',
            n_trials=n_trials
        )
        
        if optimization_level == 'quick':
            # Quick optimization
            print("🚀 Quick Optimization Mode")
            env_params = self.optimizer.optimize_environment_parameters(n_trials=10)
            ppo_params = self.optimizer.optimize_ppo_hyperparameters(
                total_timesteps=5000, n_eval_episodes=2
            )
            
            optimization_results = {
                'environment': env_params,
                'ppo': ppo_params,
                'ensemble_weights': None
            }
            
        elif optimization_level == 'standard':
            # Standard optimization
            print("⚡ Standard Optimization Mode")
            env_params = self.optimizer.optimize_environment_parameters(n_trials=20)
            ppo_params = self.optimizer.optimize_ppo_hyperparameters(
                total_timesteps=10000, n_eval_episodes=3
            )
            
            optimization_results = {
                'environment': env_params,
                'ppo': ppo_params,
                'ensemble_weights': None
            }
            
        else:
            # Comprehensive optimization
            print("🔥 Comprehensive Optimization Mode")
            optimization_results = self.optimizer.comprehensive_optimization(
                total_timesteps=15000
            )
        
        self.optimization_results = optimization_results
        
        print("✅ Hyperparameter optimization completed!")
        return optimization_results
    
    def train_ensemble_agents(self, 
                            total_timesteps: int = 50000,
                            use_optimized_params: bool = True) -> EnsembleAgent:
        """
        Train ensemble of agents
        
        Args:
            total_timesteps: Total training timesteps
            use_optimized_params: Whether to use optimized parameters
            
        Returns:
            Trained ensemble agent
        """
        print("🤖 Training Ensemble Agents")
        print("=" * 40)
        
        # Initialize ensemble agent
        self.ensemble_agent = EnsembleAgent(
            stock_list=self.stock_list,
            df_dict=self.df_dict,
            model_save_path=self.model_save_path,
            ensemble_size=5
        )
        
        # Train ensemble
        training_results = self.ensemble_agent.train_ensemble(
            total_timesteps=total_timesteps,
            eval_freq=10000,
            n_eval_episodes=5
        )
        
        # Store training results
        self.training_history.append(training_results)
        
        print("✅ Ensemble training completed!")
        return self.ensemble_agent
    
    def evaluate_ensemble_performance(self, 
                                    n_episodes: int = 10,
                                    methods: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate ensemble performance with different methods
        
        Args:
            n_episodes: Number of evaluation episodes
            methods: List of ensemble methods to evaluate
            
        Returns:
            Evaluation results
        """
        if methods is None:
            methods = ['weighted_average', 'majority_vote', 'best_agent']
        
        print("📊 Evaluating Ensemble Performance")
        print("=" * 40)
        
        evaluation_results = {}
        
        for method in methods:
            print(f"\n🔍 Evaluating {method} method...")
            
            try:
                # Run backtest
                backtest_results = self.ensemble_agent.backtest_ensemble(
                    method=method,
                    initial_amount=1000000.0
                )
                
                # Store results
                evaluation_results[method] = {
                    'backtest_results': backtest_results,
                    'performance_metrics': backtest_results['final_metrics'],
                    'outperformance': (backtest_results['final_metrics'].get('total_return', 0) - 
                                     backtest_results['buy_and_hold_return'])
                }
                
                # Display results
                metrics = backtest_results['final_metrics']
                print(f"   Total Return: {metrics.get('total_return', 0):.4f}")
                print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
                print(f"   Outperformance: {evaluation_results[method]['outperformance']:.4f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {method}: {e}")
                evaluation_results[method] = {'error': str(e)}
        
        self.evaluation_results = evaluation_results
        
        # Find best method
        best_method = None
        best_performance = -np.inf
        
        for method, results in evaluation_results.items():
            if 'performance_metrics' in results:
                sharpe_ratio = results['performance_metrics'].get('sharpe_ratio', 0)
                if sharpe_ratio > best_performance:
                    best_performance = sharpe_ratio
                    best_method = method
        
        print(f"\n🏆 Best Method: {best_method} (Sharpe: {best_performance:.4f})")
        
        return evaluation_results
    
    def generate_performance_report(self, 
                                  save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            save_plots: Whether to save performance plots
            
        Returns:
            Performance report
        """
        print("📈 Generating Performance Report")
        print("=" * 40)
        
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return {}
        
        # Create performance report
        report = {
            'summary': {},
            'detailed_results': {},
            'plots': {}
        }
        
        # Summary statistics
        all_returns = []
        all_sharpe_ratios = []
        all_drawdowns = []
        
        for method, results in self.evaluation_results.items():
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                all_returns.append(metrics.get('total_return', 0))
                all_sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                all_drawdowns.append(abs(metrics.get('max_drawdown', 0)))
        
        report['summary'] = {
            'avg_return': np.mean(all_returns),
            'max_return': np.max(all_returns),
            'avg_sharpe_ratio': np.mean(all_sharpe_ratios),
            'max_sharpe_ratio': np.max(all_sharpe_ratios),
            'avg_drawdown': np.mean(all_drawdowns),
            'min_drawdown': np.min(all_drawdowns),
            'target_achieved': np.max(all_returns) >= self.target_return
        }
        
        # Detailed results
        report['detailed_results'] = self.evaluation_results
        
        # Generate plots
        if save_plots:
            report['plots'] = self._create_performance_plots()
        
        # Save report
        self._save_performance_report(report)
        
        print("✅ Performance report generated!")
        return report
    
    def _create_performance_plots(self) -> Dict[str, str]:
        """Create performance visualization plots"""
        plots = {}
        
        try:
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Phase 2: Portfolio Management Performance', fontsize=16)
            
            # Extract data for plotting
            methods = []
            returns = []
            sharpe_ratios = []
            drawdowns = []
            
            for method, results in self.evaluation_results.items():
                if 'performance_metrics' in results:
                    methods.append(method)
                    metrics = results['performance_metrics']
                    returns.append(metrics.get('total_return', 0))
                    sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                    drawdowns.append(abs(metrics.get('max_drawdown', 0)))
            
            # Returns comparison
            axes[0, 0].bar(methods, returns, color='green', alpha=0.7)
            axes[0, 0].axhline(y=self.target_return, color='red', linestyle='--', 
                              label=f'Target ({self.target_return:.1%})')
            axes[0, 0].set_title('Total Returns by Method')
            axes[0, 0].set_ylabel('Total Return')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Sharpe ratio comparison
            axes[0, 1].bar(methods, sharpe_ratios, color='blue', alpha=0.7)
            axes[0, 1].set_title('Sharpe Ratios by Method')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Drawdown comparison
            axes[1, 0].bar(methods, drawdowns, color='red', alpha=0.7)
            axes[1, 0].set_title('Maximum Drawdowns by Method')
            axes[1, 0].set_ylabel('Max Drawdown')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Risk-Return scatter
            axes[1, 1].scatter(drawdowns, returns, s=100, alpha=0.7)
            for i, method in enumerate(methods):
                axes[1, 1].annotate(method, (drawdowns[i], returns[i]), 
                                   xytext=(5, 5), textcoords='offset points')
            axes[1, 1].set_xlabel('Max Drawdown')
            axes[1, 1].set_ylabel('Total Return')
            axes[1, 1].set_title('Risk-Return Profile')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.results_path, 'phase2_performance_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots['comparison'] = plot_path
            
            plt.show()
            
            # Create detailed backtest plot for best method
            best_method = max(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x].get('performance_metrics', {}).get('sharpe_ratio', -np.inf))
            
            if 'backtest_results' in self.evaluation_results[best_method]:
                self._create_detailed_backtest_plot(
                    self.evaluation_results[best_method]['backtest_results'],
                    best_method
                )
                plots['detailed_backtest'] = os.path.join(self.results_path, f'{best_method}_detailed_backtest.png')
            
        except Exception as e:
            print(f"❌ Error creating plots: {e}")
        
        return plots
    
    def _create_detailed_backtest_plot(self, backtest_results: Dict[str, Any], method: str):
        """Create detailed backtest plot"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Detailed Backtest Results - {method}', fontsize=16)
            
            # Portfolio value over time
            axes[0, 0].plot(backtest_results['portfolio_values'])
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_xlabel('Trading Days')
            axes[0, 0].set_ylabel('Portfolio Value')
            axes[0, 0].grid(True)
            
            # Daily returns distribution
            returns = np.diff(backtest_results['portfolio_values']) / backtest_results['portfolio_values'][:-1]
            axes[0, 1].hist(returns, bins=30, alpha=0.7, color='blue')
            axes[0, 1].set_title('Daily Returns Distribution')
            axes[0, 1].set_xlabel('Daily Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
            
            # Drawdown
            portfolio_values = np.array(backtest_results['portfolio_values'])
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
            axes[1, 0].set_title('Drawdown Over Time')
            axes[1, 0].set_xlabel('Trading Days')
            axes[1, 0].set_ylabel('Drawdown')
            axes[1, 0].grid(True)
            
            # Performance metrics
            metrics = backtest_results['final_metrics']
            metric_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
            metric_values = [
                metrics.get('total_return', 0),
                metrics.get('sharpe_ratio', 0),
                abs(metrics.get('max_drawdown', 0)),
                metrics.get('win_rate', 0)
            ]
            
            bars = axes[1, 1].bar(metric_names, metric_values)
            axes[1, 1].set_title('Performance Metrics')
            axes[1, 1].set_ylabel('Value')
            
            # Color bars based on performance
            colors = ['green' if v > 0 else 'red' for v in metric_values]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.results_path, f'{method}_detailed_backtest.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating detailed backtest plot: {e}")
    
    def _save_performance_report(self, report: Dict[str, Any]):
        """Save performance report to files"""
        try:
            # Save summary
            summary_df = pd.DataFrame([report['summary']])
            summary_path = os.path.join(self.results_path, 'performance_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            
            # Save detailed results
            detailed_results = []
            for method, results in report['detailed_results'].items():
                if 'performance_metrics' in results:
                    row = {'method': method}
                    row.update(results['performance_metrics'])
                    detailed_results.append(row)
            
            if detailed_results:
                detailed_df = pd.DataFrame(detailed_results)
                detailed_path = os.path.join(self.results_path, 'detailed_performance.csv')
                detailed_df.to_csv(detailed_path, index=False)
            
            print(f"📁 Performance report saved to {self.results_path}")
            
        except Exception as e:
            print(f"❌ Error saving performance report: {e}")
    
    def run_complete_pipeline(self, 
                            optimization_level: str = 'standard',
                            total_timesteps: int = 30000) -> Dict[str, Any]:
        """
        Run complete Phase 2 pipeline
        
        Args:
            optimization_level: Optimization level ('quick', 'standard', 'comprehensive')
            total_timesteps: Total training timesteps
            
        Returns:
            Complete pipeline results
        """
        print("🚀 Phase 2: Advanced Portfolio Management Pipeline")
        print("=" * 60)
        
        pipeline_results = {}
        
        try:
            # Step 1: Load and prepare data
            print("\n1️⃣ Loading and Preparing Data")
            df_dict = self.load_and_prepare_data()
            pipeline_results['data_loaded'] = True
            
            # Step 2: Hyperparameter optimization
            print("\n2️⃣ Hyperparameter Optimization")
            optimization_results = self.run_hyperparameter_optimization(
                optimization_level=optimization_level,
                n_trials=30 if optimization_level == 'quick' else 50
            )
            pipeline_results['optimization'] = optimization_results
            
            # Step 3: Train ensemble agents
            print("\n3️⃣ Training Ensemble Agents")
            ensemble_agent = self.train_ensemble_agents(
                total_timesteps=total_timesteps,
                use_optimized_params=True
            )
            pipeline_results['ensemble_trained'] = True
            
            # Step 4: Evaluate performance
            print("\n4️⃣ Evaluating Performance")
            evaluation_results = self.evaluate_ensemble_performance(n_episodes=10)
            pipeline_results['evaluation'] = evaluation_results
            
            # Step 5: Generate report
            print("\n5️⃣ Generating Performance Report")
            performance_report = self.generate_performance_report(save_plots=True)
            pipeline_results['report'] = performance_report
            
            # Final summary
            print("\n🎉 Phase 2 Pipeline Completed Successfully!")
            print("=" * 60)
            
            # Display final results
            if 'summary' in performance_report:
                summary = performance_report['summary']
                print(f"📊 Final Results:")
                print(f"   Average Return: {summary.get('avg_return', 0):.4f}")
                print(f"   Maximum Return: {summary.get('max_return', 0):.4f}")
                print(f"   Average Sharpe Ratio: {summary.get('avg_sharpe_ratio', 0):.4f}")
                print(f"   Maximum Sharpe Ratio: {summary.get('max_sharpe_ratio', 0):.4f}")
                print(f"   Target Achieved: {'✅' if summary.get('target_achieved', False) else '❌'}")
            
            pipeline_results['success'] = True
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
        
        return pipeline_results

# Example usage
if __name__ == "__main__":
    # Initialize Phase 2 pipeline
    pipeline = Phase2Pipeline(
        stock_list=None,  # Will auto-select top stocks
        target_return=0.10  # 10% target return
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        optimization_level='standard',
        total_timesteps=20000
    )
    
    print("Phase 2 pipeline completed!")
