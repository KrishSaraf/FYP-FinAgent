"""
PPO training script for 45-stock portfolio with shorting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

# Custom imports
from envs import Portfolio45ShortEnv
from utils.metrics import calculate_portfolio_metrics
from data.synthetic_data import generate_synthetic_data


class PortfolioTrainer:
    """PPO trainer for portfolio management."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        train_start_date: str,
        train_end_date: str,
        test_start_date: str,
        test_end_date: str,
        model_save_path: str = "models",
        results_save_path: str = "results",
        **env_kwargs
    ):
        """
        Initialize trainer.
        
        Args:
            data: Full dataset
            tickers: List of stock tickers
            train_start_date: Training start date
            train_end_date: Training end date
            test_start_date: Test start date
            test_end_date: Test end date
            model_save_path: Path to save models
            results_save_path: Path to save results
            **env_kwargs: Additional environment parameters
        """
        self.data = data
        self.tickers = tickers
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.model_save_path = model_save_path
        self.results_save_path = results_save_path
        self.env_kwargs = env_kwargs
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(results_save_path, exist_ok=True)
        
        # Prepare data splits
        self._prepare_data_splits()
        
        # Initialize environments
        self.train_env = None
        self.test_env = None
        self.model = None
    
    def _prepare_data_splits(self):
        """Prepare training and test data splits."""
        # Filter data by date ranges
        self.train_data = self.data[
            (self.data['date'] >= self.train_start_date) & 
            (self.data['date'] <= self.train_end_date)
        ].copy()
        
        self.test_data = self.data[
            (self.data['date'] >= self.test_start_date) & 
            (self.data['date'] <= self.test_end_date)
        ].copy()
        
        print(f"Training data: {len(self.train_data)} records")
        print(f"Test data: {len(self.test_data)} records")
        print(f"Training period: {self.train_data['date'].min()} to {self.train_data['date'].max()}")
        print(f"Test period: {self.test_data['date'].min()} to {self.test_data['date'].max()}")
    
    def create_environments(self):
        """Create training and test environments."""
        # Training environment
        self.train_env = Portfolio45ShortEnv(
            data=self.train_data,
            tickers=self.tickers,
            random_start=True,  # Random start for training
            **self.env_kwargs
        )
        
        # Test environment
        self.test_env = Portfolio45ShortEnv(
            data=self.test_data,
            tickers=self.tickers,
            random_start=False,  # Fixed start for testing
            **self.env_kwargs
        )
        
        # Wrap with Monitor for logging
        self.train_env = Monitor(self.train_env)
        self.test_env = Monitor(self.test_env)
        
        # Create vectorized environments
        self.train_env = DummyVecEnv([lambda: self.train_env])
        self.test_env = DummyVecEnv([lambda: self.test_env])
        
        # Apply normalization
        self.train_env = VecNormalize(
            self.train_env, 
            norm_obs=True, 
            norm_reward=True,
            clip_obs=10.0
        )
        
        self.test_env = VecNormalize(
            self.test_env,
            norm_obs=True,
            norm_reward=False,  # Don't normalize rewards during evaluation
            clip_obs=10.0
        )
        
        # Copy normalization parameters from training to test
        self.test_env.set_attr('obs_rms', self.train_env.obs_rms)
        self.test_env.set_attr('ret_rms', self.train_env.ret_rms)
    
    def train(
        self,
        total_timesteps: int = 100_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1
    ):
        """
        Train PPO agent.
        
        Args:
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            n_steps: Steps per update
            batch_size: Batch size
            n_epochs: Epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            verbose: Verbosity level
        """
        print("🚀 Starting PPO Training")
        print("=" * 50)
        
        # Create environments if not already created
        if self.train_env is None:
            self.create_environments()
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self.train_env,
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
            verbose=verbose,
            tensorboard_log=f"{self.model_save_path}/tensorboard_logs"
        )
        
        # Set up evaluation callback
        eval_callback = EvalCallback(
            self.test_env,
            best_model_save_path=f"{self.model_save_path}/best_model",
            log_path=f"{self.model_save_path}/eval_logs",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        print(f"Training for {total_timesteps:,} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save final model
        self.model.save(f"{self.model_save_path}/final_model")
        self.train_env.save(f"{self.model_save_path}/vec_normalize.pkl")
        
        print("✅ Training completed!")
        print(f"Model saved to: {self.model_save_path}")
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print(f"🔍 Evaluating model on {n_episodes} episodes...")
        
        episode_returns = []
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs = self.test_env.reset()
            done = False
            episode_return = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.test_env.step(action)
                episode_return += reward[0]
            
            # Get episode metrics
            env = self.test_env.envs[0].env
            metrics = env.get_portfolio_metrics()
            episode_metrics.append(metrics)
            episode_returns.append(episode_return)
        
        # Calculate average metrics
        avg_metrics = {}
        for key in episode_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in episode_metrics])
        
        avg_metrics['episode_return'] = np.mean(episode_returns)
        avg_metrics['episode_return_std'] = np.std(episode_returns)
        
        print("📊 Evaluation Results:")
        for key, value in avg_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return avg_metrics
    
    def backtest(self, deterministic: bool = True) -> Dict[str, Any]:
        """
        Run backtest on test data.
        
        Args:
            deterministic: Whether to use deterministic actions
            
        Returns:
            Backtest results
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print("📈 Running backtest...")
        
        # Reset test environment
        obs = self.test_env.reset()
        done = False
        
        # Track performance
        portfolio_values = []
        returns = []
        actions = []
        weights_history = []
        trades_log = []
        
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
            
            if 'trades' in info[0]:
                trades_log.extend(info[0]['trades'])
        
        # Get final metrics
        env = self.test_env.envs[0].env
        final_metrics = env.get_portfolio_metrics()
        
        # Calculate additional metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.array(returns)
        weights_array = np.array(weights_history)
        
        # Calculate comprehensive metrics
        comprehensive_metrics = calculate_portfolio_metrics(
            returns, portfolio_values, weights_array
        )
        
        # Combine metrics
        backtest_results = {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'actions': actions,
            'weights_history': weights_history,
            'trades_log': trades_log,
            'final_metrics': final_metrics,
            'comprehensive_metrics': comprehensive_metrics,
        }
        
        print("📊 Backtest Results:")
        for key, value in comprehensive_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return backtest_results
    
    def plot_results(self, backtest_results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot backtest results."""
        portfolio_values = backtest_results['portfolio_values']
        returns = backtest_results['returns']
        weights_history = backtest_results['weights_history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Backtest Results', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_values)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Trading Days')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Daily returns distribution
        axes[0, 1].hist(returns, bins=30, alpha=0.7, color='blue')
        axes[0, 1].set_title('Daily Returns Distribution')
        axes[0, 1].set_xlabel('Daily Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color='red')
        axes[1, 0].set_title('Drawdown Over Time')
        axes[1, 0].set_xlabel('Trading Days')
        axes[1, 0].set_ylabel('Drawdown')
        axes[1, 0].grid(True)
        
        # Gross exposure over time
        gross_exposure = [np.sum(np.abs(w)) for w in weights_history]
        axes[1, 1].plot(gross_exposure)
        axes[1, 1].set_title('Gross Exposure Over Time')
        axes[1, 1].set_xlabel('Trading Days')
        axes[1, 1].set_ylabel('Gross Exposure')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, backtest_results: Dict[str, Any]):
        """Save backtest results to files."""
        # Save portfolio values
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
        if backtest_results['trades_log']:
            trades_df = pd.DataFrame(backtest_results['trades_log'])
            trades_df.to_csv(f"{self.results_save_path}/trades_log.csv", index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([backtest_results['comprehensive_metrics']])
        metrics_df.to_csv(f"{self.results_save_path}/metrics.csv", index=False)
        
        print(f"Results saved to: {self.results_save_path}")


def main():
    """Main training function."""
    print("🎯 45-Stock Portfolio Training with Shorting")
    print("=" * 60)
    
    # Load real processed data
    print("📊 Loading data...")
    data, tickers = generate_synthetic_data(
        n_stocks=45,
        n_days=252,  # Ignored - uses actual data range
        start_date='2023-01-01'  # Ignored - uses actual data range
    )
    
    print(f"Data shape: {data.shape}")
    print(f"Tickers: {tickers[:5]}... (showing first 5)")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Define date splits (8 months train, 4 months test)
    train_start = '2023-01-01'
    train_end = '2023-08-31'
    test_start = '2023-09-01'
    test_end = '2023-12-31'
    
    # Initialize trainer
    trainer = PortfolioTrainer(
        data=data,
        tickers=tickers,
        train_start_date=train_start,
        train_end_date=train_end,
        test_start_date=test_start,
        test_end_date=test_end,
        model_save_path="models",
        results_save_path="results",
        # Environment parameters
        initial_capital=1_000_000.0,
        commission_bps=1.0,
        slippage_bps=2.0,
        borrow_rate_annual=0.03,
        w_max=0.10,
        gross_cap=1.5,
        target_net=1.0,
    )
    
    # Train model
    trainer.train(
        total_timesteps=50_000,  # Reduced for demo
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1
    )
    
    # Evaluate model
    eval_metrics = trainer.evaluate(n_episodes=5)
    
    # Run backtest
    backtest_results = trainer.backtest(deterministic=True)
    
    # Plot results
    trainer.plot_results(backtest_results, "results/backtest_plots.png")
    
    # Save results
    trainer.save_results(backtest_results)
    
    print("🎉 Training and evaluation completed!")


if __name__ == "__main__":
    main()
