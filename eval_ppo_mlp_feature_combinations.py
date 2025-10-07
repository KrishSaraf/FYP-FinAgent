#!/usr/bin/env python3
"""Evaluate PPO+MLP models trained with different feature combinations"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from finagent.environment.portfolio_env import JAXVectorizedPortfolioEnv
from train_ppo import JAXToSB3Wrapper
from train_ppo_mlp_feature_combinations import FeatureSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOMLPEvaluator:
    """Evaluator for PPO+MLP models with feature combinations"""

    def __init__(self, model_path: str, feature_combination: str,
                 start_date: str, end_date: str, data_root: str = "processed_data/"):
        self.model_path = model_path
        self.feature_combination = feature_combination
        self.data_root = data_root
        self.start_date = start_date
        self.end_date = end_date

        # Load feature selector and get features
        feature_selector = FeatureSelector()
        self.selected_features = feature_selector.get_features_for_combination(feature_combination)
        logger.info(f"Using {len(self.selected_features)} features for '{feature_combination}'")

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = PPO.load(model_path)

        # Create evaluation environment
        self.eval_env = self._create_eval_env()

    def _create_eval_env(self):
        """Create evaluation environment with selected features"""
        jax_env = JAXVectorizedPortfolioEnv(
            data_root=self.data_root,
            features=self.selected_features,
            use_all_features=False,
            start_date=self.start_date,
            end_date=self.end_date,
            window_size=30,
            transaction_cost_rate=0.005,
            sharpe_window=252
        )
        return JAXToSB3Wrapper(jax_env)

    def evaluate(self, num_episodes: int = 20, max_steps: int = 10000,
                deterministic: bool = True) -> Dict[str, Any]:
        """Evaluate model over multiple episodes"""
        logger.info(f"Evaluating for {num_episodes} episodes from {self.start_date} to {self.end_date}")

        episode_returns = []
        episode_sharpes = []
        episode_vols = []
        episode_drawdowns = []
        portfolio_values_history = []
        returns_history = []
        rewards_history = []
        daily_results_history = []

        start_time = time.time()

        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")

            obs, info = self.eval_env.reset()
            done = False
            step_count = 0

            daily_results = []
            portfolio_values = []
            rewards = []
            returns_series = []

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
                })

                portfolio_values.append(float(info.get("portfolio_value", 1.0)))
                rewards.append(float(reward))
                returns_series.append(float(info.get("daily_portfolio_return", 0.0)))
                step_count += 1

            # Calculate metrics
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

            final_value = portfolio_values[-1] if portfolio_values else 1.0
            final_return = final_value - 1.0

            episode_returns.append(final_return)
            episode_sharpes.append(sharpe)
            episode_vols.append(volatility)
            episode_drawdowns.append(max_drawdown)
            portfolio_values_history.append(portfolio_values)
            returns_history.append(returns_series)
            rewards_history.append(rewards)
            daily_results_history.append(daily_results)

            logger.info(f"Episode {episode + 1}: Return={final_return:.4f}, Sharpe={sharpe:.4f}, "
                       f"Vol={volatility:.4f}, MaxDD={max_drawdown:.4f}, Steps={step_count}")

        total_time = time.time() - start_time

        results = {
            "model_path": self.model_path,
            "feature_combination": self.feature_combination,
            "num_features": len(self.selected_features),
            "evaluation_period": f"{self.start_date} to {self.end_date}",
            "num_episodes": num_episodes,
            "total_evaluation_time_seconds": total_time,
            "episode_returns": episode_returns,
            "episode_sharpe_ratios": episode_sharpes,
            "episode_volatilities": episode_vols,
            "episode_max_drawdowns": episode_drawdowns,
            "portfolio_values_history": portfolio_values_history,
            "returns_history": returns_history,
            "rewards_history": rewards_history,
            "daily_results_history": daily_results_history,
            "mean_return": float(np.mean(episode_returns)),
            "std_return": float(np.std(episode_returns)),
            "mean_sharpe": float(np.mean(episode_sharpes)),
            "mean_volatility": float(np.mean(episode_vols)),
            "mean_max_drawdown": float(np.mean(episode_drawdowns)),
            "success_rate": float(np.mean([r > 0 for r in episode_returns])),
        }

        logger.info("Evaluation completed successfully")
        return results

    def save_results(self, results: Dict, save_path: str):
        """Save evaluation results to JSON"""
        results_copy = results.copy()
        results_copy['timestamp'] = pd.Timestamp.now().isoformat()
        with open(save_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        logger.info(f"Results saved to {save_path}")

    def generate_report(self, results: Dict, save_path: str = None) -> str:
        """Generate text report"""
        r = results
        report = f"""
PPO+MLP Feature Combinations Evaluation Report
==============================================

Feature Combination: {r['feature_combination']}
Number of Features: {r['num_features']}
Evaluation Period: {r['evaluation_period']}
Number of Episodes: {r['num_episodes']}

Portfolio Performance Metrics:
-----------------------------
Mean Return: {r['mean_return']:.4f} ({r['mean_return']*100:.2f}%)
Std Return: {r['std_return']:.4f}
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
Information Ratio: {r['mean_return'] / (r['std_return'] + 1e-8):.4f}
Risk-Adjusted Return: {r['mean_return'] / (r['mean_volatility'] + 1e-8):.4f}
Calmar Ratio: {r['mean_return'] / (abs(r['mean_max_drawdown']) + 1e-8):.4f}
"""
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO+MLP with feature combinations")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--feature_combination', type=str, required=True,
                       help='Feature combination used for training')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--eval_start_date', type=str, default='2025-03-07',
                       help='Evaluation start date')
    parser.add_argument('--eval_end_date', type=str, default='2025-06-06',
                       help='Evaluation end date')
    parser.add_argument('--data_root', type=str, default='processed_data/',
                       help='Data root directory')
    parser.add_argument('--results_dir', type=str, default='evaluation_results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluator
    evaluator = PPOMLPEvaluator(
        args.model_path,
        args.feature_combination,
        args.eval_start_date,
        args.eval_end_date,
        args.data_root
    )

    # Run evaluation
    results = evaluator.evaluate(num_episodes=args.num_episodes)

    # Save results
    feature_name = args.feature_combination.replace('+', '_')
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    json_path = results_dir / f"ppo_mlp_{feature_name}_evaluation_{timestamp}.json"
    evaluator.save_results(results, str(json_path))

    report_path = results_dir / f"ppo_mlp_{feature_name}_report_{timestamp}.txt"
    report = evaluator.generate_report(results, str(report_path))
    print(report)

    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
