"""
PPO Feature Combinations Model Evaluation Script

This script evaluates trained PPO models with different feature combinations
on out-of-sample data and provides comprehensive performance analysis including:
- Portfolio performance metrics
- Risk analysis
- Comparison with benchmarks
- Feature combination analysis
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
from typing import Dict, List, Tuple, Any, Optional
import warnings
from functools import partial

# Configure JAX to use CPU only (fixes Metal GPU compatibility issues)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import your modules
from finagent.environment.portfolio_env import JAXVectorizedPortfolioEnv, EnvState
from train_ppo_feature_combinations import (
    FeatureCombinationPPOTrainer, 
    CustomPortfolioEnv, 
    FeatureSelector,
    ActorCriticLSTM,
    LSTMState
)

class PPOFeatureCombinationEvaluator:
    """Comprehensive evaluation class for PPO models with feature combinations"""
    
    def __init__(self, model_path: str, config: Dict[str, Any], selected_features: List[str] = None):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the trained model file
            config: Configuration dictionary for evaluation
            selected_features: List of features to use (overrides model's features if provided)
        """
        self.model_path = model_path
        self.config = config
        self.results = {}
        self.provided_features = selected_features
        # Load the trained model
        self._load_model()
        
        # Initialize the evaluation environment
        self._setup_evaluation_environment()
        
        logger.info("PPO Feature Combination Evaluator initialized successfully")
    
    def _load_model(self):
        """Load the trained model parameters and configuration"""
        logger.info(f"Loading model from: {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                model_state = pickle.load(f)
            
            self.model_params = model_state['params']
            self.training_config = model_state['config']
            self.training_step = model_state.get('training_step', 0)
            
            # Try to get features from model state, fallback to provided features or config
            self.selected_features = model_state.get('selected_features', [])
            if not self.selected_features:
                if self.provided_features:
                    self.selected_features = self.provided_features
                    logger.info(f"Using provided features: {len(self.selected_features)} features")
                else:
                    # Try to get from config
                    self.selected_features = self.config.get('selected_features', [])
                    logger.warning("No selected_features found in model state or provided, using from config")
            
            self.feature_combination = model_state.get('feature_combination', 'unknown')
            
            # Initialize network architecture
            self.network = ActorCriticLSTM(
                action_dim=self.training_config.get('action_dim', len(self.config['stocks']) + 1),
                hidden_size=self.training_config.get('hidden_size', 256),
                n_lstm_layers=self.training_config.get('n_lstm_layers', 1)
            )
            
            logger.info("Model loaded successfully")
            logger.info(f"Training step: {self.training_step}")
            logger.info(f"Feature combination: {self.feature_combination}")
            logger.info(f"Selected features: {len(self.selected_features)} features")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_evaluation_environment(self):
        """Setup the evaluation environment with out-of-sample data"""
        logger.info("Setting up evaluation environment...")
        
        # Use evaluation dates (out-of-sample)
        env_config = {
            'data_root': self.config['data_root'],
            'stocks': self.config['stocks'],
            'start_date': self.config['eval_start_date'],
            'end_date': self.config['eval_end_date'],
            'window_size': self.config.get('window_size', 30),
            'transaction_cost_rate': self.config.get('transaction_cost_rate', 0.005),
            'sharpe_window': self.config.get('sharpe_window', 252),
            'use_all_features': False,  # Use custom feature selection
            'selected_features': self.selected_features
        }
        
        try:
            self.env = CustomPortfolioEnv(**env_config)
            self.n_envs = self.config.get('n_eval_envs', 1)
            
            # Vectorized environment functions
            self.vmap_reset = jax.vmap(self.env.reset, in_axes=(0,))
            self.vmap_step = jax.vmap(self.env.step, in_axes=(0, 0))
            
            logger.info(f"Evaluation environment created: {self.env.n_timesteps} timesteps")
            logger.info(f"Observation dimension: {self.env.obs_dim}")
            logger.info(f"Features per stock: {self.env.n_features}")
            
        except Exception as e:
            logger.error(f"Failed to setup evaluation environment: {e}")
            raise
    
    def _create_initial_carry(self, batch_size: int) -> List[LSTMState]:
        """Create initial LSTM carry states"""
        return [
            LSTMState(
                h=jnp.zeros((batch_size, self.training_config.get('hidden_size', 256))),
                c=jnp.zeros((batch_size, self.training_config.get('hidden_size', 256)))
            ) for _ in range(self.training_config.get('n_lstm_layers', 1))
        ]
    
    @partial(jax.jit, static_argnums=(0, 5))
    def _get_policy_action(self, params, obs: jnp.ndarray, carry: List[LSTMState], 
                          rng_key: jnp.ndarray, deterministic: bool = True):
        """Get action from the trained policy"""
        # Get network outputs
        logits, values, new_carry = self.network.apply(params, obs, carry)
        
        if deterministic:
            # Use mean of the policy (deterministic evaluation)
            actions = logits
        else:
            # Sample from the policy distribution
            import distrax
            action_std = self.training_config.get('action_std', 0.5)
            action_distribution = distrax.Normal(loc=logits, scale=action_std)
            actions = action_distribution.sample(seed=rng_key)
        
        # Clip actions to reasonable range
        actions = jnp.clip(actions, -5.0, 5.0)
        
        return actions, values, new_carry
    
    def evaluate_model(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model over multiple episodes
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation: {num_episodes} episodes, deterministic={deterministic}")
        
        # Initialize random key
        rng_key = random.PRNGKey(self.config.get('eval_seed', 123))
        
        # Storage for results
        episode_returns = []
        episode_sharpe_ratios = []
        episode_max_drawdowns = []
        episode_volatilities = []
        portfolio_values_history = []
        weights_history = []
        rewards_history = []
        actions_history = []
        
        for episode in range(num_episodes):
            logger.info(f"Running episode {episode + 1}/{num_episodes}")
            
            # Reset environment
            rng_key, reset_key = random.split(rng_key)
            env_state, obs = self.env.reset(reset_key)
            
            # Initialize LSTM carry state
            lstm_carry = self._create_initial_carry(1)
            
            # Episode tracking
            episode_portfolio_values = [float(env_state.portfolio_value)]
            episode_weights = [env_state.portfolio_weights.tolist()]
            episode_rewards = []
            episode_actions = []
            
            done = False
            step_count = 0
            
            while not done and step_count < self.env.n_timesteps - 1:
                # Get action from policy
                rng_key, action_key = random.split(rng_key)
                obs_batch = obs[None, :]  # Add batch dimension
                
                action, value, lstm_carry = self._get_policy_action(
                    self.model_params, obs_batch, lstm_carry, action_key, deterministic
                )
                
                action = action[0]  # Remove batch dimension
                episode_actions.append(action.tolist())
                
                # Step environment
                env_state, obs, reward, done, info = self.env.step(env_state, action)
                
                # Track episode data
                episode_portfolio_values.append(float(env_state.portfolio_value))
                episode_weights.append(env_state.portfolio_weights.tolist())
                episode_rewards.append(float(reward))
                
                step_count += 1
            
            # Calculate episode metrics
            final_return = float(env_state.total_return)
            final_portfolio_value = float(env_state.portfolio_value)
            
            # Calculate Sharpe ratio from episode
            episode_returns_array = np.diff(episode_portfolio_values) / np.array(episode_portfolio_values[:-1])
            episode_returns_array = episode_returns_array[~np.isnan(episode_returns_array)]
            
            if len(episode_returns_array) > 1:
                sharpe_ratio = np.mean(episode_returns_array) / (np.std(episode_returns_array) + 1e-8) * np.sqrt(252)
                volatility = np.std(episode_returns_array) * np.sqrt(252)
                
                # Calculate max drawdown
                cumulative_returns = np.cumprod(1 + episode_returns_array)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdown)
            else:
                sharpe_ratio = 0.0
                volatility = 0.0
                max_drawdown = 0.0
            
            # Store episode results
            episode_returns.append(final_return)
            episode_sharpe_ratios.append(sharpe_ratio)
            episode_max_drawdowns.append(max_drawdown)
            episode_volatilities.append(volatility)
            portfolio_values_history.append(episode_portfolio_values)
            weights_history.append(episode_weights)
            rewards_history.append(episode_rewards)
            actions_history.append(episode_actions)
            
            logger.info(f"Episode {episode + 1} completed: Return={final_return:.4f}, "
                       f"Sharpe={sharpe_ratio:.4f}, MaxDD={max_drawdown:.4f}")
        
        # Compile results
        results = {
            'episode_returns': episode_returns,
            'episode_sharpe_ratios': episode_sharpe_ratios,
            'episode_max_drawdowns': episode_max_drawdowns,
            'episode_volatilities': episode_volatilities,
            'portfolio_values_history': portfolio_values_history,
            'weights_history': weights_history,
            'rewards_history': rewards_history,
            'actions_history': actions_history,
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_sharpe': np.mean(episode_sharpe_ratios),
            'mean_volatility': np.mean(episode_volatilities),
            'mean_max_drawdown': np.mean(episode_max_drawdowns),
            'success_rate': np.mean([r > 0 for r in episode_returns]),
            'num_episodes': num_episodes,
            'feature_combination': self.feature_combination,
            'selected_features': self.selected_features,
            'num_features': len(self.selected_features)
        }
        
        self.results = results
        logger.info("Evaluation completed successfully")
        
        return results
    
    def calculate_benchmark_comparison(self) -> Dict[str, Any]:
        """Calculate benchmark comparisons (buy-and-hold, equal weight)"""
        logger.info("Calculating benchmark comparisons...")
        
        try:
            # Get market data for benchmark calculation
            eval_env = CustomPortfolioEnv(
                selected_features=['close'],  # Only need close prices for benchmark
                data_root=self.config['data_root'],
                stocks=self.config['stocks'],
                start_date=self.config['eval_start_date'],
                end_date=self.config['eval_end_date'],
                window_size=self.config.get('window_size', 30),
                transaction_cost_rate=0.0  # No transaction costs for benchmark
            )
            
            # Extract price data
            close_prices = eval_env.data[:, :, 0]  # Assuming close is first feature
            
            # Buy-and-hold benchmark (equal weight)
            equal_weights = jnp.ones(eval_env.n_stocks) / eval_env.n_stocks
            daily_returns = jnp.diff(close_prices, axis=0) / close_prices[:-1]
            daily_returns = jnp.where(jnp.isnan(daily_returns), 0.0, daily_returns)
            
            portfolio_returns = jnp.sum(equal_weights * daily_returns, axis=1)
            cumulative_returns = jnp.cumprod(1 + portfolio_returns)
            
            # Calculate benchmark metrics
            total_return = float(cumulative_returns[-1] - 1)
            sharpe_ratio = float(jnp.mean(portfolio_returns) / jnp.std(portfolio_returns) * jnp.sqrt(252))
            volatility = float(jnp.std(portfolio_returns) * jnp.sqrt(252))
            
            # Max drawdown
            running_max = lax.cummax(cumulative_returns, axis=0)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = float(jnp.min(drawdown))
            
            benchmark_results = {
                'buy_and_hold_return': total_return,
                'buy_and_hold_sharpe': sharpe_ratio,
                'buy_and_hold_volatility': volatility,
                'buy_and_hold_max_drawdown': max_drawdown,
                'buy_and_hold_cumulative_returns': cumulative_returns.tolist()
            }
            
            logger.info(f"Benchmark - Return: {total_return:.4f}, Sharpe: {sharpe_ratio:.4f}")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Failed to calculate benchmarks: {e}")
            return {}
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze the importance of different feature categories"""
        logger.info("Analyzing feature importance...")
        
        # Initialize feature selector to get categories
        feature_selector = FeatureSelector()
        
        feature_analysis = {
            'feature_combination': self.feature_combination,
            'total_features': len(self.selected_features),
            'feature_categories': {},
            'feature_breakdown': {}
        }
        
        # Analyze features by category
        for category, category_info in feature_selector.feature_categories.items():
            category_features = [f for f in self.selected_features if f in category_info['features']]
            feature_analysis['feature_categories'][category] = {
                'count': len(category_features),
                'features': category_features,
                'percentage': len(category_features) / len(self.selected_features) * 100 if self.selected_features else 0
            }
        
        # Create feature breakdown
        feature_analysis['feature_breakdown'] = {
            'ohlcv_features': feature_analysis['feature_categories'].get('ohlcv', {}).get('count', 0),
            'technical_features': feature_analysis['feature_categories'].get('technical', {}).get('count', 0),
            'financial_features': feature_analysis['feature_categories'].get('financial', {}).get('count', 0),
            'sentiment_features': feature_analysis['feature_categories'].get('sentiment', {}).get('count', 0)
        }
        
        return feature_analysis
    
    def generate_performance_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive performance report"""
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        logger.info("Generating performance report...")
        
        # Calculate benchmark comparison and feature analysis
        benchmark_results = self.calculate_benchmark_comparison()
        feature_analysis = self.analyze_feature_importance()
        
        # Generate report
        report = f"""
PPO Feature Combination Portfolio Model Evaluation Report
========================================================

Model Information:
- Model Path: {self.model_path}
- Training Step: {self.training_step}
- Feature Combination: {self.feature_combination}
- Evaluation Period: {self.config['eval_start_date']} to {self.config['eval_end_date']}
- Number of Stocks: {len(self.config['stocks'])}
- Number of Episodes: {self.results['num_episodes']}

Feature Analysis:
----------------
Total Features Used: {self.results['num_features']}
Feature Combination: {self.feature_combination}

Feature Category Breakdown:
- OHLCV Features: {feature_analysis['feature_breakdown']['ohlcv_features']} 
- Technical Features: {feature_analysis['feature_breakdown']['technical_features']}
- Financial Features: {feature_analysis['feature_breakdown']['financial_features']}
- Sentiment Features: {feature_analysis['feature_breakdown']['sentiment_features']}

Selected Features:
{', '.join(self.selected_features[:10])}{'...' if len(self.selected_features) > 10 else ''}

Portfolio Performance Metrics:
-----------------------------
Mean Return: {self.results['mean_return']:.4f} ({self.results['mean_return']*100:.2f}%)
Standard Deviation: {self.results['std_return']:.4f}
Mean Sharpe Ratio: {self.results['mean_sharpe']:.4f}
Mean Volatility: {self.results['mean_volatility']:.4f} ({self.results['mean_volatility']*100:.2f}% annualized)
Mean Max Drawdown: {self.results['mean_max_drawdown']:.4f} ({self.results['mean_max_drawdown']*100:.2f}%)
Success Rate: {self.results['success_rate']:.2f} ({self.results['success_rate']*100:.1f}% positive returns)

Episode Statistics:
------------------
Best Return: {max(self.results['episode_returns']):.4f}
Worst Return: {min(self.results['episode_returns']):.4f}
Best Sharpe: {max(self.results['episode_sharpe_ratios']):.4f}
Worst Sharpe: {min(self.results['episode_sharpe_ratios']):.4f}
"""
        
        # Add benchmark comparison if available
        if benchmark_results:
            report += f"""
Benchmark Comparison (Buy-and-Hold Equal Weight):
------------------------------------------------
Benchmark Return: {benchmark_results['buy_and_hold_return']:.4f} ({benchmark_results['buy_and_hold_return']*100:.2f}%)
Benchmark Sharpe: {benchmark_results['buy_and_hold_sharpe']:.4f}
Benchmark Volatility: {benchmark_results['buy_and_hold_volatility']:.4f} ({benchmark_results['buy_and_hold_volatility']*100:.2f}%)
Benchmark Max Drawdown: {benchmark_results['buy_and_hold_max_drawdown']:.4f} ({benchmark_results['buy_and_hold_max_drawdown']*100:.2f}%)

Relative Performance:
- Return Difference: {(self.results['mean_return'] - benchmark_results['buy_and_hold_return']):.4f}
- Sharpe Difference: {(self.results['mean_sharpe'] - benchmark_results['buy_and_hold_sharpe']):.4f}
- Volatility Difference: {(self.results['mean_volatility'] - benchmark_results['buy_and_hold_volatility']):.4f}
"""
        
        report += f"""
Risk Analysis:
--------------
Information Ratio: {self.results['mean_return'] / (self.results['std_return'] + 1e-8):.4f}
Risk-Adjusted Return: {self.results['mean_return'] / (self.results['mean_volatility'] + 1e-8):.4f}
Calmar Ratio: {self.results['mean_return'] / (abs(self.results['mean_max_drawdown']) + 1e-8):.4f}

Stock Universe:
--------------
{', '.join(self.config['stocks'])}

Configuration:
-------------
Transaction Cost Rate: {self.config.get('transaction_cost_rate', 'N/A')}
Window Size: {self.config.get('window_size', 'N/A')}
Evaluation Seed: {self.config.get('eval_seed', 'N/A')}
"""
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {save_path}")
        
        return report
    
    def create_visualizations(self, save_dir: Optional[str] = None):
        """Create comprehensive visualization plots"""
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        logger.info("Creating visualizations...")
        
        # Create save directory
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'PPO Feature Combination Model Evaluation Results\n{self.feature_combination}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Portfolio Value Evolution
        ax1 = axes[0, 0]
        for i, pv_history in enumerate(self.results['portfolio_values_history'][:5]):  # Plot first 5 episodes
            ax1.plot(pv_history, alpha=0.7, label=f'Episode {i+1}')
        ax1.set_title('Portfolio Value Evolution')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Return Distribution
        ax2 = axes[0, 1]
        ax2.hist(self.results['episode_returns'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(self.results['mean_return'], color='red', linestyle='--', 
                   label=f'Mean: {self.results["mean_return"]:.4f}')
        ax2.set_title('Distribution of Episode Returns')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Sharpe Ratio Distribution
        ax3 = axes[0, 2]
        ax3.hist(self.results['episode_sharpe_ratios'], bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(self.results['mean_sharpe'], color='red', linestyle='--',
                   label=f'Mean: {self.results["mean_sharpe"]:.4f}')
        ax3.set_title('Distribution of Sharpe Ratios')
        ax3.set_xlabel('Sharpe Ratio')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Max Drawdown Distribution
        ax4 = axes[1, 0]
        ax4.hist(self.results['episode_max_drawdowns'], bins=20, alpha=0.7, edgecolor='black')
        ax4.axvline(self.results['mean_max_drawdown'], color='red', linestyle='--',
                   label=f'Mean: {self.results["mean_max_drawdown"]:.4f}')
        ax4.set_title('Distribution of Max Drawdowns')
        ax4.set_xlabel('Max Drawdown')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Feature Category Breakdown
        ax5 = axes[1, 1]
        feature_analysis = self.analyze_feature_importance()
        categories = list(feature_analysis['feature_breakdown'].keys())
        counts = list(feature_analysis['feature_breakdown'].values())
        
        bars = ax5.bar(categories, counts, alpha=0.7)
        ax5.set_title('Feature Category Breakdown')
        ax5.set_xlabel('Feature Categories')
        ax5.set_ylabel('Number of Features')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 6. Risk-Return Scatter
        ax6 = axes[1, 2]
        ax6.scatter(self.results['episode_volatilities'], self.results['episode_returns'], 
                   alpha=0.7, s=50)
        ax6.set_title('Risk-Return Profile')
        ax6.set_xlabel('Volatility (Annualized)')
        ax6.set_ylabel('Return')
        ax6.grid(True, alpha=0.3)
        
        # Add trend line
        if len(self.results['episode_volatilities']) > 1:
            z = np.polyfit(self.results['episode_volatilities'], 
                          self.results['episode_returns'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(self.results['episode_volatilities']), 
                                 max(self.results['episode_volatilities']), 100)
            ax6.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        # Save plot if directory provided
        if save_dir:
            plt.savefig(save_path / f'evaluation_results_{self.feature_combination.replace("+", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Visualizations saved to: {save_path / f'evaluation_results_{self.feature_combination.replace("+", "_")}.png'}")
        
        plt.show()
    
    def save_results(self, save_path: str):
        """Save evaluation results to file"""
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = {}
            for key, value in self.results.items():
                if isinstance(value, np.ndarray):
                    results_to_save[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    results_to_save[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
                else:
                    results_to_save[key] = value
            
            # Add metadata
            results_to_save['evaluation_config'] = self.config
            results_to_save['model_path'] = self.model_path
            results_to_save['timestamp'] = pd.Timestamp.now().isoformat()
            results_to_save['feature_analysis'] = self.analyze_feature_importance()
            
            with open(save_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            
            logger.info(f"Results saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise


def main():
    """Main evaluation script"""
    
    # Evaluation configuration
    eval_config = {
        # Data configuration
        'data_root': 'processed_data/',
        'stocks': None,  # Will be loaded from stocks.txt or inferred
        'eval_start_date': '2025-03-07',  # Out-of-sample period
        'eval_end_date': '2025-06-06',
        'window_size': 30,
        'transaction_cost_rate': 0.005,
        'sharpe_window': 252,
        
        # Evaluation configuration
        'n_eval_envs': 1,
        'eval_seed': 123,
        
        # Data loading settings
        'fill_missing_features_with': 'interpolate',
        'preload_to_gpu': True,
        
        # Output settings
        'save_results': True,
        'save_visualizations': True,
        'results_dir': 'evaluation_results/PPO_feature_combinations'
    }
    
    # Load stocks from file if not specified
    if eval_config['stocks'] is None:
        try:
            with open('finagent/stocks.txt', 'r') as f:
                eval_config['stocks'] = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(eval_config['stocks'])} stocks from stocks.txt")
        except FileNotFoundError:
            logger.warning("stocks.txt not found, using default stocks")
            eval_config['stocks'] = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR']
    
    # Model path - adjust this to your trained model
    model_path = 'models/final_model_ohlcv.pkl'  # Update this path
    
    try:
        # Initialize evaluator
        evaluator = PPOFeatureCombinationEvaluator(model_path, eval_config)
        
        # Run evaluation
        logger.info("Starting model evaluation...")
        results = evaluator.evaluate_model(
            num_episodes=20,  # Number of evaluation episodes
            deterministic=True  # Use deterministic policy
        )
        
        # Generate performance report
        report = evaluator.generate_performance_report()
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        # Create visualizations
        if eval_config['save_visualizations']:
            results_dir = Path(eval_config['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            evaluator.create_visualizations(save_dir=str(results_dir))
        
        # Save results
        if eval_config['save_results']:
            results_dir = Path(eval_config['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            evaluator.save_results(str(results_dir / f'evaluation_results_{timestamp}.json'))
            
            # Save performance report
            with open(results_dir / f'performance_report_{timestamp}.txt', 'w') as f:
                f.write(report)
        
        logger.info("Evaluation completed successfully!")
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Feature Combination: {results['feature_combination']}")
        print(f"Total Features: {results['num_features']}")
        print(f"Mean Return: {results['mean_return']:.4f} ({results['mean_return']*100:.2f}%)")
        print(f"Mean Sharpe Ratio: {results['mean_sharpe']:.4f}")
        print(f"Success Rate: {results['success_rate']*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
