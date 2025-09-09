import os
import tarfile
import tempfile
import time
import numpy as np
import pandas as pd
import torch
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from stable_baselines3 import PPO
from typing import Any, Dict, List, Optional

# Import your custom environment components
from finagent.environment.portfolio_env import JAXVectorizedPortfolioEnv
from train_ppo import JAXToSB3Wrapper  # Import the wrapper from your training script

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PortfolioModelEvaluator:
    """
    Evaluator for trained portfolio management models
    """

    def __init__(self, model_path: str, start_date: str, end_date:str, data_root: str = "processed_data/", **env_kwargs):
        self.model_path = model_path
        self.data_root = data_root
        self.model = None
        self.start_date = start_date
        self.end_date = end_date
        self.results = {}

        self.load_model_from_tar()
        self.eval_env = self.create_evaluation_environment(**env_kwargs)

    def load_model_from_tar(self):
        """Load PPO model from .tar.gz file"""
        logger.info(f"Loading model from: {self.model_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(self.model_path, 'r:gz') as tar:
                tar.extractall(temp_dir)

            model_files = list(Path(temp_dir).rglob("*.zip"))
            if not model_files:
                raise FileNotFoundError("No .zip model file found in the tar.gz archive")

            model_file = model_files[0]
            logger.info(f"Found model file: {model_file}")
            self.model = PPO.load(str(model_file))

        logger.info("Model loaded successfully!")

    def create_evaluation_environment(self, **env_kwargs):
        """Create evaluation environment for out-of-sample testing"""
        default_params = {
            'data_root': self.data_root,
            'window_size': 30,
            'transaction_cost_rate': 0.005,
            'sharpe_window': 252,
            'use_all_features': True
        }
        default_params.update(env_kwargs)
        default_params.update({'start_date': self.start_date, 'end_date': self.end_date})
        jax_env = JAXVectorizedPortfolioEnv(**default_params)
        return JAXToSB3Wrapper(jax_env)

    def evaluate_model(
      self,
      max_steps: int = 10000,
      deterministic: bool = True,
      num_episodes: int = 20
    ) -> Dict[str, Any]:
        """
        Evaluate the loaded model over multiple episodes on out-of-sample data.

        Args:
            start_date: Evaluation start date
            end_date: Evaluation end date
            max_steps: Max steps per episode
            deterministic: Whether to use deterministic policy
            num_episodes: Number of evaluation episodes
            env_kwargs: Extra args for env creation

        Returns:
            Dictionary with aggregated + per-episode evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_from_tar() first.")

        logger.info(f"Starting evaluation: {num_episodes} episodes from {start_date} to {end_date}")

        # Storage for aggregated stats
        episode_returns = []
        episode_sharpes = []
        episode_vols = []
        episode_drawdowns = []

        portfolio_values_history = []
        returns_history = []
        rewards_history = []
        actions_history = []
        daily_results_history = []

        start_time_global = time.time()

        for episode in range(num_episodes):
            logger.info(f"Running episode {episode + 1}/{num_episodes}")

            
            obs, info = self.eval_env.reset()
            done = False
            step_count = 0

            daily_results = []
            actions_taken = []
            portfolio_values = []
            rewards = []
            returns_series = []

            start_time = time.time()

            while not done and step_count < max_steps:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated

                daily_results.append({
                    "step": int(step_count),
                    "reward": float(reward),
                    "portfolio_value": float(info.get("portfolio_value", 1.0)),
                    "total_return": float(info.get("total_return", 0.0)),
                    "sharpe_ratio": float(info.get("sharpe_ratio", 0.0)),
                    "daily_return": float(info.get("daily_portfolio_return", 0.0)),
                    "transaction_cost": float(info.get("transaction_cost_value", 0.0)),
                    "cash_weight": float(info.get("new_cash_weight", 0.0)),
                    "short_exposure": float(info.get("short_exposure", 0.0)),
                })

                actions_taken.append(action.tolist() if hasattr(action, "tolist") else float(action))
                portfolio_values.append(float(info.get("portfolio_value", 1.0)))
                rewards.append(float(reward))
                returns_series.append(float(info.get("daily_portfolio_return", 0.0)))

                step_count += 1

            eval_time = time.time() - start_time
            returns_array = np.array(returns_series)
            returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) > 1:
                volatility = float(np.std(returns_array) * np.sqrt(252))
                sharpe = float(np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252))
                cumulative_returns = np.cumprod(1 + returns_array)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = float(np.min(drawdowns))
            else:
                volatility, sharpe, max_drawdown = 0.0, 0.0, 0.0

            final_portfolio_value = portfolio_values[-1] if portfolio_values else 1.0
            final_return = final_portfolio_value - 1.0

            # Store episode-level metrics
            episode_returns.append(final_return)
            episode_sharpes.append(sharpe)
            episode_vols.append(volatility)
            episode_drawdowns.append(max_drawdown)

            portfolio_values_history.append(portfolio_values)
            returns_history.append(returns_series)
            rewards_history.append(rewards)
            actions_history.append(actions_taken)
            daily_results_history.append(daily_results)

            logger.info(
                f"Episode {episode + 1} finished: "
                f"Return={final_return:.4f}, Sharpe={sharpe:.4f}, "
                f"Vol={volatility:.4f}, MaxDD={max_drawdown:.4f}, "
                f"Steps={step_count}, Time={eval_time:.2f}s"
            )

            self.eval_env.close()

        total_eval_time = time.time() - start_time_global

        # Compile aggregated results
        results = {
            "evaluation_period": f"{start_date} to {end_date}",
            "num_episodes": num_episodes,
            "total_evaluation_time_seconds": total_eval_time,
            "episode_returns": episode_returns,
            "episode_sharpe_ratios": episode_sharpes,
            "episode_volatilities": episode_vols,
            "episode_max_drawdowns": episode_drawdowns,
            "portfolio_values_history": portfolio_values_history,
            "returns_history": returns_history,
            "rewards_history": rewards_history,
            "actions_history": actions_history,
            "daily_results_history": daily_results_history,
            "mean_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
            "mean_sharpe": float(np.mean(episode_sharpes)),
            "mean_volatility": float(np.mean(episode_vols)),
            "mean_max_drawdown": float(np.mean(episode_drawdowns)),
            "success_rate": float(np.mean([r > 0 for r in episode_returns])),
        }

        self.results = results
        logger.info("Evaluation completed successfully")

        return results

    def save_results(self, save_path: str):
        """Save evaluation results to JSON"""
        if not self.results:
            raise ValueError("No evaluation results to save")
        results_copy = {k: v for k, v in self.results.items()}
        results_copy['model_path'] = self.model_path
        results_copy['timestamp'] = pd.Timestamp.now().isoformat()
        with open(save_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        logger.info(f"Results saved to {save_path}")

    def generate_performance_report(self, save_path: str = None) -> str:
        """Generate a text report"""
        if not self.results:
            raise ValueError("No results available")
        r = self.results
        report = f"""
Portfolio Evaluation Report
===========================

Evaluation Period: {r['evaluation_period']}
Number of Episodes: {r['num_episodes']}

Portfolio Performance Metrics:
-----------------------------
Mean Return: {r['mean_return']:.4f} ({r['mean_return']*100:.2f}%)
Standard Deviation: {r.get('std_return', 0.0):.4f}
Mean Sharpe Ratio: {r['mean_sharpe']:.4f}
Mean Volatility: {r['mean_volatility']:.4f} ({r['mean_volatility']*100:.2f}% annualized)
Mean Max Drawdown: {r['mean_max_drawdown']:.4f} ({r['mean_max_drawdown']*100:.2f}%)
Success Rate: {r['success_rate']:.4f} ({r['success_rate']*100:.1f}% positive returns)

Episode Statistics:
------------------
Best Return: {max(r['episode_returns']):.4f}
Worst Return: {min(r['episode_returns']):.4f}
Best Sharpe: {max(r['episode_sharpe_ratios']):.4f}
Worst Sharpe: {min(r['episode_sharpe_ratios']):.4f}

Risk Analysis:
--------------
Information Ratio: {r['mean_return'] / (r.get('std_return', 1e-8)):.4f}
Risk-Adjusted Return: {r['mean_return'] / (r['mean_volatility'] + 1e-8):.4f}
Calmar Ratio: {r['mean_return'] / (abs(r['mean_max_drawdown']) + 1e-8):.4f}

Configuration:
-------------
Transaction Cost Rate: {self.eval_env.jax_env.transaction_cost_rate}
Window Size: {self.eval_env.jax_env.window_size}
"""
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
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
        fig.suptitle('PPO Portfolio Model Evaluation Results', fontsize=16, fontweight='bold')
        
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
        
        # 5. Average Portfolio Weights (first episode)
        ax5 = axes[1, 1]
        if self.results['actions_history']:
            weights_array = np.array(self.results['actions_history'][0])
            avg_weights = np.mean(weights_array, axis=0)
            stock_names = self.eval_env.jax_env.stocks + ['Cash']
            
            bars = ax5.bar(range(len(avg_weights)), avg_weights, alpha=0.7)
            ax5.set_title('Average Portfolio Weights (Episode 1)')
            ax5.set_xlabel('Assets')
            ax5.set_ylabel('Weight')
            ax5.set_xticks(range(len(stock_names)))
            ax5.set_xticklabels(stock_names, rotation=45)
            ax5.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
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
            plt.savefig(save_path / 'ppo_evaluation_results.png', dpi=300, bbox_inches='tight')
            logger.info(f"Visualizations saved to: {save_path / 'ppo_evaluation_results.png'}")
        
        plt.show()


if __name__ == "__main__":
    MODEL_PATH = "model_weights_80percent.tar.gz"
    DATA_ROOT = "processed_data/"
    EVALUATION_PERIODS = [('2025-03-07', '2025-06-06')]
    start_date, end_date = EVALUATION_PERIODS[0]
    evaluator = PortfolioModelEvaluator(MODEL_PATH, start_date, end_date, DATA_ROOT)
    
    results = evaluator.evaluate_model(max_steps=5000)

    report = evaluator.generate_performance_report("evaluation_results/performance_report_ppo.txt")
    print(report)

    evaluator.create_visualizations("evaluation_results")
    evaluator.save_results("evaluation_results/evaluation_results_ppo.json")
