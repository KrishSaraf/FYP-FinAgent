"""
RL LightGBM Model Evaluation Script

This script evaluates a trained LightGBM-based RL agent on out-of-sample data
and provides comprehensive performance analysis, including:
- Portfolio performance metrics
- Risk analysis
- Comparison with benchmarks
- Visual analysis and reports

Author: AI Assistant
Date: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from train_plain_rl_lgbm import EnhancedTradingEnv, EnhancedLGBMPolicy
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set matplotlib style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RLLightGBMEvaluator:
    """Evaluation class for RL LightGBM-based agent"""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the trained model file
            config: Configuration dictionary for evaluation
        """
        self.model_path = model_path
        self.config = config
        self.results = {}

        # Load the trained model
        self._load_model()

        # Initialize the evaluation environment
        self._setup_evaluation_environment()

        logger.info("RL LightGBM Evaluator initialized successfully")

    def _load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from: {self.model_path}")

        try:
            with open(self.model_path, 'rb') as f:
                self.policy = pickle.load(f)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _setup_evaluation_environment(self):
        """Setup the evaluation environment with out-of-sample data"""
        logger.info("Setting up evaluation environment...")

        try:
            self.env = EnhancedTradingEnv(
                data_dir=self.config['data_dir'],
                feature_combination=self.config['feature_combination'],
                lookback=self.config['lookback'],
                transaction_cost=self.config['transaction_cost'],
                short_cost_rate=self.config['short_cost_rate'],
                max_leverage=self.config['max_leverage'],
                reward_scaling=self.config['reward_scaling'],
                turnover_penalty_scale=self.config['turnover_penalty_scale'],
                clip_return=self.config['clip_return'],
                sharpe_window=self.config['sharpe_window'],
                train_start_date=self.config['eval_start_date'],
                train_end_date=self.config['eval_end_date']
            )
            logger.info(f"Evaluation environment created: {self.env.T} timesteps, {self.env.N} assets")

        except Exception as e:
            logger.error(f"Failed to setup evaluation environment: {e}")
            raise

    def evaluate_model(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the model over multiple episodes
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation: {num_episodes} episodes")

        # Storage for results
        episode_returns = []
        episode_sharpe_ratios = []
        episode_max_drawdowns = []
        episode_volatilities = []
        portfolio_values_history = []

        for episode in range(num_episodes):
            logger.info(f"Running episode {episode + 1}/{num_episodes}")

            # Reset environment
            obs = self.env.reset()

            # Episode tracking
            episode_portfolio_values = [self.env.portfolio_value]
            done = False

            while not done:
                # Select actions
                actions = self.policy.predict(obs)
                obs, reward, done, info = self.env.step(actions)

                # Track portfolio value
                episode_portfolio_values.append(info['portfolio_value'])

            # Calculate episode metrics
            final_return = episode_portfolio_values[-1] - 1.0
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

            logger.info(f"Episode {episode + 1} completed: Return={final_return:.4f}, "
                        f"Sharpe={sharpe_ratio:.4f}, MaxDD={max_drawdown:.4f}")

        # Compile results
        results = {
            'episode_returns': episode_returns,
            'episode_sharpe_ratios': episode_sharpe_ratios,
            'episode_max_drawdowns': episode_max_drawdowns,
            'episode_volatilities': episode_volatilities,
            'portfolio_values_history': portfolio_values_history,
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_sharpe': np.mean(episode_sharpe_ratios),
            'mean_volatility': np.mean(episode_volatilities),
            'mean_max_drawdown': np.mean(episode_max_drawdowns),
            'success_rate': np.mean([r > 0 for r in episode_returns]),
            'num_episodes': num_episodes
        }

        self.results = results
        logger.info("Evaluation completed successfully")

        return results

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
        fig.suptitle('RL LightGBM Model Evaluation Results', fontsize=16, fontweight='bold')

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

        # 5. Risk-Return Scatter
        ax5 = axes[1, 1]
        ax5.scatter(self.results['episode_volatilities'], self.results['episode_returns'],
                    alpha=0.7, s=50)
        ax5.set_title('Risk-Return Profile')
        ax5.set_xlabel('Volatility (Annualized)')
        ax5.set_ylabel('Return')
        ax5.grid(True, alpha=0.3)

        # Add trend line
        if len(self.results['episode_volatilities']) > 1:
            z = np.polyfit(self.results['episode_volatilities'],
                           self.results['episode_returns'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(self.results['episode_volatilities']),
                                  max(self.results['episode_volatilities']), 100)
            ax5.plot(x_trend, p(x_trend), "r--", alpha=0.8)

        plt.tight_layout()

        # Save plot if directory provided
        if save_dir:
            plt.savefig(save_path / 'evaluation_results_rl_lgbm.png',
                        dpi=300, bbox_inches='tight')
            logger.info(f"Visualizations saved to: {save_path / 'evaluation_results_rl_lgbm.png'}")

        plt.show()


def main():
    """Main evaluation script"""
    # Evaluation configuration
    eval_config = {
        'data_dir': 'processed_data/',
        'feature_combination': 'all',
        'lookback': 5,
        'transaction_cost': 0.001,
        'short_cost_rate': 0.001,
        'max_leverage': 1.0,
        'reward_scaling': 100.0,
        'turnover_penalty_scale': 0.0,
        'clip_return': 0.2,
        'sharpe_window': 30,
        'eval_start_date': '2025-03-07',
        'eval_end_date': '2025-06-06'
    }

    # Model path - adjust this to your trained model
    model_path = 'models/lgbm/policy_epoch_stage_3_epoch_70.pkl'

    try:
        # Initialize evaluator
        evaluator = RLLightGBMEvaluator(model_path, eval_config)

        # Run evaluation
        logger.info("Starting model evaluation...")
        results = evaluator.evaluate_model(num_episodes=20)

        # Save results as JSON
        save_path = Path('evaluation_results/')
        save_path.mkdir(parents=True, exist_ok=True)
        results_file = save_path / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results saved to: {results_file}")

        # Create visualizations
        evaluator.create_visualizations(save_dir='evaluation_results/')

        # Print summary
        print("\nEvaluation Summary:")
        print(f"Mean Return: {results['mean_return']:.4f} ({results['mean_return'] * 100:.2f}%)")
        print(f"Mean Sharpe Ratio: {results['mean_sharpe']:.4f}")
        print(f"Success Rate: {results['success_rate'] * 100:.1f}%")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()