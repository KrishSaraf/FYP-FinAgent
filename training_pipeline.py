"""
Training Pipeline for FinRL Single Stock Trading
Implements PPO agent training with comprehensive evaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# FinRL imports
from finrl import config
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.model.models import DRLEnsembleAgent
from finrl.model.models import DRLEnsembleAgent

# Stable Baselines3 imports
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure

# Custom imports
from data_loader import FinancialDataLoader
from trading_environment import SingleStockTradingEnv

class FinRLTrainingPipeline:
    """
    Comprehensive training pipeline for FinRL single stock trading
    """
    
    def __init__(self, 
                 stock_symbol: str = "RELIANCE",
                 data_path: str = "processed_data",
                 model_save_path: str = "trained_models",
                 results_path: str = "results"):
        """
        Initialize the training pipeline
        
        Args:
            stock_symbol: Stock symbol to train on
            data_path: Path to processed data
            model_save_path: Path to save trained models
            results_path: Path to save results
        """
        self.stock_symbol = stock_symbol
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.results_path = results_path
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = FinancialDataLoader(data_path)
        
        # Initialize training data
        self.train_data = None
        self.test_data = None
        self.env = None
        self.model = None
        
        # Performance tracking
        self.training_history = []
        self.evaluation_results = {}
    
    def load_and_prepare_data(self, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for training
        
        Args:
            train_ratio: Ratio of data for training
            
        Returns:
            Tuple of (train_data, test_data)
        """
        print(f"Loading data for {self.stock_symbol}...")
        
        # Load raw data
        raw_data = self.data_loader.load_stock_data(self.stock_symbol)
        print(f"Raw data shape: {raw_data.shape}")
        
        # Preprocess data
        processed_data = self.data_loader.preprocess_data(raw_data)
        print(f"Processed data shape: {processed_data.shape}")
        
        # Create train/test split
        self.train_data, self.test_data = self.data_loader.create_train_test_split(
            processed_data, train_ratio
        )
        
        # Normalize features
        self.train_data, self.test_data = self.data_loader.normalize_features(
            self.train_data, self.test_data
        )
        
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        return self.train_data, self.test_data
    
    def create_environment(self, 
                          data: pd.DataFrame,
                          initial_amount: float = 1000000.0,
                          state_space: int = 50,
                          action_space: int = 3) -> SingleStockTradingEnv:
        """
        Create trading environment
        
        Args:
            data: Training or testing data
            initial_amount: Initial capital
            state_space: State space dimension
            action_space: Action space dimension
            
        Returns:
            Trading environment
        """
        env = SingleStockTradingEnv(
            df=data,
            stock_dim=1,
            hmax=100,
            initial_amount=initial_amount,
            buy_cost_pct=0.001,
            sell_cost_pct=0.001,
            reward_scaling=1e-4,
            state_space=state_space,
            action_space=action_space
        )
        
        return env
    
    def train_ppo_agent(self, 
                       total_timesteps: int = 50000,
                       learning_rate: float = 3e-4,
                       n_steps: int = 2048,
                       batch_size: int = 64,
                       n_epochs: int = 10,
                       gamma: float = 0.99,
                       gae_lambda: float = 0.95,
                       clip_range: float = 0.2,
                       ent_coef: float = 0.0,
                       vf_coef: float = 0.5,
                       max_grad_norm: float = 0.5) -> PPO:
        """
        Train PPO agent
        
        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Number of steps per update
            batch_size: Batch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            
        Returns:
            Trained PPO model
        """
        print("Creating training environment...")
        self.env = self.create_environment(self.train_data)
        
        print("Initializing PPO agent...")
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=1,
            tensorboard_log=f"{self.results_path}/tensorboard_logs"
        )
        
        print(f"Training PPO agent for {total_timesteps} timesteps...")
        
        # Set up logging
        logger = configure(f"{self.results_path}/training_logs", ["stdout", "csv", "tensorboard"])
        self.model.set_logger(logger)
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save the model
        model_path = os.path.join(self.model_save_path, f"ppo_{self.stock_symbol}")
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return self.model
    
    def evaluate_model(self, 
                      model: PPO = None,
                      data: pd.DataFrame = None,
                      n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained model
        
        Args:
            model: Trained model (if None, uses self.model)
            data: Test data (if None, uses self.test_data)
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model is None:
            model = self.model
        if data is None:
            data = self.test_data
        
        print(f"Evaluating model on {n_episodes} episodes...")
        
        # Create evaluation environment
        eval_env = self.create_environment(data)
        
        # Run evaluation episodes
        episode_rewards = []
        episode_returns = []
        episode_sharpe_ratios = []
        episode_max_drawdowns = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
            
            # Get performance metrics
            metrics = eval_env.get_performance_metrics()
            episode_rewards.append(episode_reward)
            episode_returns.append(metrics.get('total_return', 0))
            episode_sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            episode_max_drawdowns.append(metrics.get('max_drawdown', 0))
        
        # Calculate average metrics
        evaluation_results = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'avg_sharpe_ratio': np.mean(episode_sharpe_ratios),
            'std_sharpe_ratio': np.std(episode_sharpe_ratios),
            'avg_max_drawdown': np.mean(episode_max_drawdowns),
            'std_max_drawdown': np.std(episode_max_drawdowns),
            'win_rate': np.mean([r > 0 for r in episode_returns])
        }
        
        self.evaluation_results = evaluation_results
        
        print("Evaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"  {metric}: {value:.4f}")
        
        return evaluation_results
    
    def backtest_strategy(self, 
                         model: PPO = None,
                         data: pd.DataFrame = None,
                         initial_amount: float = 1000000.0) -> Dict[str, any]:
        """
        Backtest the trading strategy
        
        Args:
            model: Trained model
            data: Test data
            initial_amount: Initial capital
            
        Returns:
            Dictionary with backtest results
        """
        if model is None:
            model = self.model
        if data is None:
            data = self.test_data
        
        print("Running backtest...")
        
        # Create backtest environment
        backtest_env = self.create_environment(data, initial_amount)
        
        # Run backtest
        obs = backtest_env.reset()
        done = False
        actions_taken = []
        portfolio_values = []
        dates = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = backtest_env.step(action)
            
            actions_taken.append(action)
            portfolio_values.append(info['total_assets'])
            dates.append(backtest_env.day)
        
        # Get final performance metrics
        final_metrics = backtest_env.get_performance_metrics()
        
        # Create backtest results
        backtest_results = {
            'portfolio_values': portfolio_values,
            'actions_taken': actions_taken,
            'dates': dates,
            'final_metrics': final_metrics,
            'buy_and_hold_return': self._calculate_buy_and_hold_return(data, initial_amount)
        }
        
        return backtest_results
    
    def _calculate_buy_and_hold_return(self, data: pd.DataFrame, initial_amount: float) -> float:
        """
        Calculate buy and hold return for comparison
        
        Args:
            data: Test data
            initial_amount: Initial capital
            
        Returns:
            Buy and hold return
        """
        if len(data) < 2:
            return 0.0
        
        initial_price = data.iloc[0]['close']
        final_price = data.iloc[-1]['close']
        
        shares_bought = initial_amount / initial_price
        final_value = shares_bought * final_price
        
        return (final_value - initial_amount) / initial_amount
    
    def plot_results(self, backtest_results: Dict[str, any], save_path: str = None):
        """
        Plot backtest results
        
        Args:
            backtest_results: Results from backtest_strategy
            save_path: Path to save plots
        """
        if save_path is None:
            save_path = self.results_path
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'FinRL Trading Strategy Results - {self.stock_symbol}', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(backtest_results['portfolio_values'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Trading Days')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].grid(True)
        
        # Actions taken
        action_names = ['Hold', 'Buy', 'Sell']
        action_counts = [backtest_results['actions_taken'].count(i) for i in range(3)]
        axes[0, 1].bar(action_names, action_counts)
        axes[0, 1].set_title('Actions Taken')
        axes[0, 1].set_ylabel('Count')
        
        # Returns distribution
        returns = np.diff(backtest_results['portfolio_values']) / backtest_results['portfolio_values'][:-1]
        axes[1, 0].hist(returns, bins=30, alpha=0.7)
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
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
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            if i == 0:  # Total return
                color = 'green' if value > 0 else 'red'
            elif i == 1:  # Sharpe ratio
                color = 'green' if value > 1 else 'orange' if value > 0 else 'red'
            elif i == 2:  # Max drawdown
                color = 'green' if value < 0.1 else 'orange' if value < 0.2 else 'red'
            else:  # Win rate
                color = 'green' if value > 0.5 else 'orange' if value > 0.4 else 'red'
            bar.set_color(color)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_path, f'{self.stock_symbol}_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {plot_path}")
        
        plt.show()
    
    def save_results(self, backtest_results: Dict[str, any], save_path: str = None):
        """
        Save backtest results to files
        
        Args:
            backtest_results: Results from backtest_strategy
            save_path: Path to save results
        """
        if save_path is None:
            save_path = self.results_path
        
        # Save portfolio values
        portfolio_df = pd.DataFrame({
            'day': backtest_results['dates'],
            'portfolio_value': backtest_results['portfolio_values'],
            'action': backtest_results['actions_taken']
        })
        
        portfolio_path = os.path.join(save_path, f'{self.stock_symbol}_portfolio.csv')
        portfolio_df.to_csv(portfolio_path, index=False)
        
        # Save performance metrics
        metrics_df = pd.DataFrame([backtest_results['final_metrics']])
        metrics_path = os.path.join(save_path, f'{self.stock_symbol}_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        # Save evaluation results
        if self.evaluation_results:
            eval_df = pd.DataFrame([self.evaluation_results])
            eval_path = os.path.join(save_path, f'{self.stock_symbol}_evaluation.csv')
            eval_df.to_csv(eval_path, index=False)
        
        print(f"Results saved to {save_path}")

# Example usage
if __name__ == "__main__":
    # Initialize training pipeline
    pipeline = FinRLTrainingPipeline(
        stock_symbol="RELIANCE",
        model_save_path="trained_models",
        results_path="results"
    )
    
    # Load and prepare data
    train_data, test_data = pipeline.load_and_prepare_data()
    
    # Train PPO agent
    model = pipeline.train_ppo_agent(total_timesteps=10000)
    
    # Evaluate model
    evaluation_results = pipeline.evaluate_model(n_episodes=5)
    
    # Run backtest
    backtest_results = pipeline.backtest_strategy()
    
    # Plot and save results
    pipeline.plot_results(backtest_results)
    pipeline.save_results(backtest_results)
    
    print("Training pipeline completed successfully!")
