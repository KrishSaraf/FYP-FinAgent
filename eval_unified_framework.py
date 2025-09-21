"""
Unified Model Evaluation Framework

This script provides a unified framework for evaluating different types of trained models
(PPO, Plain RL) with different feature combinations on out-of-sample data.

Supported Models:
- PPO LSTM with feature combinations
- Plain RL LSTM with feature combinations

Features:
- Comprehensive performance analysis
- Feature combination comparison
- Model comparison across algorithms
- Batch evaluation of multiple models
- Visual analysis and reports

Author: AI Assistant
Date: 2024
"""

import os
import time
import jax
import jax.numpy as jnp
from jax import random, vmap, lax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from functools import partial
import argparse

# Configure JAX to use CPU only
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import evaluation classes
from eval_ppo_feature_combinations import PPOFeatureCombinationEvaluator
from eval_plain_rl_lstm import PlainRLLSTMEvaluator

class UnifiedModelEvaluator:
    """Unified framework for evaluating different model types and feature combinations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the unified evaluator
        
        Args:
            config: Configuration dictionary for evaluation
        """
        self.config = config
        self.results = {}
        self.model_results = {}
        self.model_features = {}  # Store features for each model
        
        logger.info("Unified Model Evaluator initialized")
    
    def evaluate_single_model(self, model_path: str, model_type: str = 'auto', selected_features: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single model
        
        Args:
            model_path: Path to the trained model file
            model_type: Type of model ('ppo', 'plain_rl', or 'auto' for auto-detection)
            selected_features: List of features to use for this model
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating model: {model_path}")
        
        # Auto-detect model type if not specified
        if model_type == 'auto':
            model_type = self._detect_model_type(model_path)
        
        # Store features for this model
        if selected_features is not None:
            self.model_features[model_path] = selected_features
            logger.info(f"Using provided features for {model_path}: {len(selected_features)} features")
        
        # Initialize appropriate evaluator
        if model_type == 'ppo':
            evaluator = PPOFeatureCombinationEvaluator(model_path, self.config, selected_features)
        elif model_type == 'plain_rl':
            evaluator = PlainRLLSTMEvaluator(model_path, self.config, selected_features)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Run evaluation
        results = evaluator.evaluate_model(
            num_episodes=self.config.get('num_episodes', 20),
            deterministic=self.config.get('deterministic', True)
        )
        
        # Store results
        self.model_results[model_path] = {
            'results': results,
            'model_type': model_type,
            'evaluator': evaluator,
            'selected_features': selected_features
        }
        
        return results
    
    def _detect_model_type(self, model_path: str) -> str:
        """Auto-detect model type from file path or contents"""
        # Try to detect from filename
        if 'ppo' in model_path.lower():
            return 'ppo'
        elif 'plain' in model_path.lower() or 'reinforce' in model_path.lower():
            return 'plain_rl'
        
        # Try to detect from model contents
        try:
            with open(model_path, 'rb') as f:
                model_state = pickle.load(f)
            
            # Check for algorithm-specific keys
            if 'algorithm' in model_state.get('config', {}):
                algo = model_state['config']['algorithm'].lower()
                if 'ppo' in algo:
                    return 'ppo'
                elif 'reinforce' in algo or 'plain' in algo:
                    return 'plain_rl'
            
            # Default to PPO if uncertain
            return 'ppo'
            
        except Exception as e:
            logger.warning(f"Could not auto-detect model type: {e}")
            return 'ppo'
    
    def evaluate_multiple_models(self, model_paths: List[str], model_types: List[str] = None, model_features: List[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate multiple models for comparison
        
        Args:
            model_paths: List of paths to trained model files
            model_types: List of model types (auto-detected if None)
            model_features: List of feature lists for each model (optional)
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Evaluating {len(model_paths)} models for comparison")
        
        if model_types is None:
            model_types = ['auto'] * len(model_paths)
        elif len(model_types) != len(model_paths):
            raise ValueError("Number of model types must match number of model paths")
        
        if model_features is not None and len(model_features) != len(model_paths):
            raise ValueError("Number of model features must match number of model paths")
        
        # Evaluate each model
        for i, (model_path, model_type) in enumerate(zip(model_paths, model_types)):
            try:
                features = model_features[i] if model_features else None
                self.evaluate_single_model(model_path, model_type, features)
                logger.info(f"Successfully evaluated: {model_path}")
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path}: {e}")
        
        # Generate comparison analysis
        comparison_results = self._generate_comparison_analysis()
        
        return comparison_results
    
    def _generate_comparison_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive comparison analysis across models"""
        if not self.model_results:
            return {}
        
        logger.info("Generating comparison analysis...")
        
        comparison = {
            'model_summaries': {},
            'performance_comparison': {},
            'feature_analysis': {},
            'rankings': {}
        }
        
        # Extract performance metrics for each model
        performance_metrics = []
        model_names = []
        
        for model_path, model_data in self.model_results.items():
            results = model_data['results']
            model_name = Path(model_path).stem
            
            model_names.append(model_name)
            
            # Store model summary
            comparison['model_summaries'][model_name] = {
                'model_type': model_data['model_type'],
                'feature_combination': results.get('feature_combination', 'unknown'),
                'num_features': results.get('num_features', 0),
                'mean_return': results['mean_return'],
                'mean_sharpe': results['mean_sharpe'],
                'mean_volatility': results['mean_volatility'],
                'mean_max_drawdown': results['mean_max_drawdown'],
                'success_rate': results['success_rate']
            }
            
            # Collect metrics for comparison
            performance_metrics.append([
                results['mean_return'],
                results['mean_sharpe'],
                results['mean_volatility'],
                results['mean_max_drawdown'],
                results['success_rate']
            ])
        
        # Create performance comparison DataFrame
        metrics_df = pd.DataFrame(
            performance_metrics,
            index=model_names,
            columns=['Return', 'Sharpe', 'Volatility', 'Max_Drawdown', 'Success_Rate']
        )
        
        comparison['performance_comparison'] = metrics_df.to_dict()
        
        # Generate rankings
        comparison['rankings'] = {
            'by_return': metrics_df['Return'].rank(ascending=False).to_dict(),
            'by_sharpe': metrics_df['Sharpe'].rank(ascending=False).to_dict(),
            'by_volatility': metrics_df['Volatility'].rank(ascending=True).to_dict(),  # Lower is better
            'by_max_drawdown': metrics_df['Max_Drawdown'].rank(ascending=True).to_dict(),  # Lower is better
            'by_success_rate': metrics_df['Success_Rate'].rank(ascending=False).to_dict()
        }
        
        # Feature analysis
        feature_combinations = {}
        for model_path, model_data in self.model_results.items():
            model_name = Path(model_path).stem
            results = model_data['results']
            feature_combo = results.get('feature_combination', 'unknown')
            
            if feature_combo not in feature_combinations:
                feature_combinations[feature_combo] = []
            
            feature_combinations[feature_combo].append({
                'model': model_name,
                'algorithm': model_data['model_type'],
                'performance': results['mean_return'],
                'sharpe': results['mean_sharpe']
            })
        
        comparison['feature_analysis'] = feature_combinations
        
        return comparison
    
    def create_comparison_visualizations(self, save_dir: Optional[str] = None):
        """Create comprehensive comparison visualizations"""
        if not self.model_results:
            raise ValueError("No model results available. Run evaluate_multiple_models() first.")
        
        logger.info("Creating comparison visualizations...")
        
        # Create save directory
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Generate comparison data
        comparison = self._generate_comparison_analysis()
        
        # Set up plotting
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        model_names = list(comparison['model_summaries'].keys())
        returns = [comparison['model_summaries'][name]['mean_return'] for name in model_names]
        sharpes = [comparison['model_summaries'][name]['mean_sharpe'] for name in model_names]
        volatilities = [comparison['model_summaries'][name]['mean_volatility'] for name in model_names]
        max_drawdowns = [comparison['model_summaries'][name]['mean_max_drawdown'] for name in model_names]
        success_rates = [comparison['model_summaries'][name]['success_rate'] for name in model_names]
        
        # 1. Return Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(model_names)), returns, alpha=0.7)
        ax1.set_title('Return Comparison')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Mean Return')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # 2. Sharpe Ratio Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(model_names)), sharpes, alpha=0.7)
        ax2.set_title('Sharpe Ratio Comparison')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Mean Sharpe Ratio')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # 3. Risk-Return Scatter
        ax3 = axes[0, 2]
        scatter = ax3.scatter(volatilities, returns, s=100, alpha=0.7)
        ax3.set_title('Risk-Return Profile')
        ax3.set_xlabel('Volatility')
        ax3.set_ylabel('Return')
        ax3.grid(True, alpha=0.3)
        
        # Add model labels
        for i, name in enumerate(model_names):
            ax3.annotate(name, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Max Drawdown Comparison
        ax4 = axes[1, 0]
        bars4 = ax4.bar(range(len(model_names)), max_drawdowns, alpha=0.7, color='red')
        ax4.set_title('Max Drawdown Comparison')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Mean Max Drawdown')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # 5. Success Rate Comparison
        ax5 = axes[1, 1]
        bars5 = ax5.bar(range(len(model_names)), success_rates, alpha=0.7, color='green')
        ax5.set_title('Success Rate Comparison')
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Success Rate')
        ax5.set_xticks(range(len(model_names)))
        ax5.set_xticklabels(model_names, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
        
        for bar in bars5:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 6. Feature Combination Analysis
        ax6 = axes[1, 2]
        feature_data = comparison['feature_analysis']
        feature_combos = list(feature_data.keys())
        combo_performances = []
        
        for combo in feature_combos:
            combo_returns = [model['performance'] for model in feature_data[combo]]
            combo_performances.append(np.mean(combo_returns))
        
        bars6 = ax6.bar(range(len(feature_combos)), combo_performances, alpha=0.7)
        ax6.set_title('Feature Combination Performance')
        ax6.set_xlabel('Feature Combinations')
        ax6.set_ylabel('Average Return')
        ax6.set_xticks(range(len(feature_combos)))
        ax6.set_xticklabels([combo.replace('+', '+\n') for combo in feature_combos], 
                           rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        for bar in bars6:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot if directory provided
        if save_dir:
            plt.savefig(save_path / 'model_comparison_analysis.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Comparison visualizations saved to: {save_path / 'model_comparison_analysis.png'}")
        
        plt.show()
    
    def generate_comparison_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive comparison report"""
        if not self.model_results:
            raise ValueError("No model results available. Run evaluate_multiple_models() first.")
        
        logger.info("Generating comparison report...")
        
        comparison = self._generate_comparison_analysis()
        
        report = f"""
Model Comparison Report
======================

Evaluation Configuration:
- Number of Models: {len(self.model_results)}
- Evaluation Period: {self.config['eval_start_date']} to {self.config['eval_end_date']}
- Number of Episodes per Model: {self.config.get('num_episodes', 20)}

Model Performance Summary:
========================
"""
        
        # Add performance summary for each model
        for model_name, summary in comparison['model_summaries'].items():
            report += f"""
{model_name}:
- Algorithm: {summary['model_type'].upper()}
- Feature Combination: {summary['feature_combination']}
- Number of Features: {summary['num_features']}
- Mean Return: {summary['mean_return']:.4f} ({summary['mean_return']*100:.2f}%)
- Mean Sharpe Ratio: {summary['mean_sharpe']:.4f}
- Mean Volatility: {summary['mean_volatility']:.4f} ({summary['mean_volatility']*100:.2f}%)
- Mean Max Drawdown: {summary['mean_max_drawdown']:.4f} ({summary['mean_max_drawdown']*100:.2f}%)
- Success Rate: {summary['success_rate']:.2f} ({summary['success_rate']*100:.1f}%)
"""
        
        # Add rankings
        report += f"""
Model Rankings:
==============

By Return:
{chr(10).join([f"{i+1}. {name}: {comparison['model_summaries'][name]['mean_return']:.4f}" 
               for i, name in enumerate(sorted(comparison['model_summaries'].keys(), 
                                              key=lambda x: comparison['model_summaries'][x]['mean_return'], 
                                              reverse=True))])}

By Sharpe Ratio:
{chr(10).join([f"{i+1}. {name}: {comparison['model_summaries'][name]['mean_sharpe']:.4f}" 
               for i, name in enumerate(sorted(comparison['model_summaries'].keys(), 
                                              key=lambda x: comparison['model_summaries'][x]['mean_sharpe'], 
                                              reverse=True))])}

By Volatility (Lower is Better):
{chr(10).join([f"{i+1}. {name}: {comparison['model_summaries'][name]['mean_volatility']:.4f}" 
               for i, name in enumerate(sorted(comparison['model_summaries'].keys(), 
                                              key=lambda x: comparison['model_summaries'][x]['mean_volatility']))])}
"""
        
        # Add feature combination analysis
        if comparison['feature_analysis']:
            report += f"""
Feature Combination Analysis:
============================
"""
            for combo, models in comparison['feature_analysis'].items():
                avg_return = np.mean([model['performance'] for model in models])
                avg_sharpe = np.mean([model['sharpe'] for model in models])
                report += f"""
{combo}:
- Average Return: {avg_return:.4f}
- Average Sharpe: {avg_sharpe:.4f}
- Models: {len(models)}
"""
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Comparison report saved to: {save_path}")
        
        return report
    
    def save_comparison_results(self, save_path: str):
        """Save comparison results to file"""
        if not self.model_results:
            raise ValueError("No model results available. Run evaluate_multiple_models() first.")
        
        try:
            # Generate comparison data
            comparison = self._generate_comparison_analysis()
            
            # Prepare results for JSON serialization
            results_to_save = {
                'comparison_analysis': comparison,
                'evaluation_config': self.config,
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_paths': list(self.model_results.keys())
            }
            
            with open(save_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            
            logger.info(f"Comparison results saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save comparison results: {e}")
            raise


def main():
    """Main evaluation script with command line interface"""
    
    parser = argparse.ArgumentParser(description='Unified Model Evaluation Framework')
    parser.add_argument('--models', nargs='+', required=True, 
                       help='Paths to model files to evaluate')
    parser.add_argument('--model_types', nargs='+', 
                       help='Model types (ppo, plain_rl, auto) for each model')
    parser.add_argument('--config', type=str, 
                       help='Path to evaluation configuration JSON file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/unified',
                       help='Output directory for results')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of episodes per model')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy evaluation')
    parser.add_argument('--model_features', nargs='+', 
                       help='Feature combinations to evaluate')
    
    args = parser.parse_args()
    
    # Default evaluation configuration
    eval_config = {
        'data_root': 'processed_data/',
        'stocks': None,
        'eval_start_date': '2025-03-07',
        'eval_end_date': '2025-06-06',
        'window_size': 30,
        'transaction_cost_rate': 0.005,
        'sharpe_window': 252,
        'n_eval_envs': 1,
        'eval_seed': 123,
        'fill_missing_features_with': 'interpolate',
        'preload_to_gpu': True,
        'num_episodes': args.episodes,
        'deterministic': args.deterministic,
        'save_results': True,
        'save_visualizations': True,
        'results_dir': args.output_dir
    }
    
    # Load custom config if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
            eval_config.update(custom_config)
        except Exception as e:
            logger.warning(f"Failed to load custom config: {e}")
    
    # Load stocks from file if not specified
    if eval_config['stocks'] is None:
        try:
            with open('finagent/stocks.txt', 'r') as f:
                eval_config['stocks'] = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(eval_config['stocks'])} stocks from stocks.txt")
        except FileNotFoundError:
            logger.warning("stocks.txt not found, using default stocks")
            eval_config['stocks'] = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR']
    
    try:
        # Initialize unified evaluator
        evaluator = UnifiedModelEvaluator(eval_config)
        
        # Parse feature combinations if provided
        model_features = None
        if args.model_features:
            from train_ppo_feature_combinations import FeatureSelector
            feature_selector = FeatureSelector()
            model_features = []
            
            for feature_combo in args.model_features:
                if feature_combo.lower() == 'all':
                    # Get all features
                    all_features = []
                    for category_info in feature_selector.feature_categories.values():
                        all_features.extend(category_info['features'])
                    model_features.append(all_features)
                else:
                    # Parse feature combination
                    selected_features = feature_selector.get_features_for_combination(feature_combo)
                    model_features.append(selected_features)
                
                logger.info(f"Feature combination '{feature_combo}' -> {len(model_features[-1])} features")
        
        # Evaluate multiple models
        logger.info("Starting unified model evaluation...")
        comparison_results = evaluator.evaluate_multiple_models(
            args.models, 
            args.model_types,
            model_features
        )
        
        # Generate comparison report
        report = evaluator.generate_comparison_report()
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        # Create visualizations
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator.create_comparison_visualizations(save_dir=str(output_dir))
        
        # Save results
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        evaluator.save_comparison_results(str(output_dir / f'comparison_results_{timestamp}.json'))
        
        # Save comparison report
        with open(output_dir / f'comparison_report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
        logger.info("Unified evaluation completed successfully!")
        
        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"Models evaluated: {len(args.models)}")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Unified evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
