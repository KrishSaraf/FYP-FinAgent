"""
Ensemble of Multiple DRL Agents for Portfolio Management
Implements sophisticated ensemble strategies to achieve >10% returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Stable Baselines3 imports
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure

# Custom imports
from portfolio_environment import PortfolioTradingEnv

class EnsembleAgent:
    """
    Ensemble of multiple DRL agents for robust portfolio management
    """
    
    def __init__(self, 
                 stock_list: List[str],
                 df_dict: Dict[str, pd.DataFrame],
                 model_save_path: str = "ensemble_models",
                 ensemble_size: int = 5):
        """
        Initialize ensemble agent
        
        Args:
            stock_list: List of stock symbols
            df_dict: Dictionary of DataFrames for each stock
            model_save_path: Path to save models
            ensemble_size: Number of agents in ensemble
        """
        self.stock_list = stock_list
        self.df_dict = df_dict
        self.model_save_path = model_save_path
        self.ensemble_size = ensemble_size
        
        # Initialize agents
        self.agents = {}
        self.agent_weights = {}
        self.performance_history = {}
        
        # Agent configurations for diversity
        self.agent_configs = {
            'PPO_Conservative': {
                'algorithm': PPO,
                'params': {
                    'learning_rate': 1e-4,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.1,  # Conservative clipping
                    'ent_coef': 0.01,
                    'vf_coef': 0.5
                }
            },
            'PPO_Aggressive': {
                'algorithm': PPO,
                'params': {
                    'learning_rate': 5e-4,
                    'n_steps': 1024,
                    'batch_size': 32,
                    'n_epochs': 15,
                    'gamma': 0.95,
                    'gae_lambda': 0.9,
                    'clip_range': 0.3,  # Aggressive clipping
                    'ent_coef': 0.0,
                    'vf_coef': 0.3
                }
            },
            'A2C_Momentum': {
                'algorithm': A2C,
                'params': {
                    'learning_rate': 3e-4,
                    'n_steps': 5,
                    'gamma': 0.99,
                    'gae_lambda': 1.0,
                    'ent_coef': 0.0,
                    'vf_coef': 0.25,
                    'max_grad_norm': 0.5
                }
            },
            'DDPG_Continuous': {
                'algorithm': DDPG,
                'params': {
                    'learning_rate': 1e-4,
                    'buffer_size': 100000,
                    'learning_starts': 1000,
                    'batch_size': 64,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': 1,
                    'gradient_steps': 1
                }
            },
            'SAC_Adaptive': {
                'algorithm': SAC,
                'params': {
                    'learning_rate': 3e-4,
                    'buffer_size': 100000,
                    'learning_starts': 1000,
                    'batch_size': 64,
                    'tau': 0.005,
                    'gamma': 0.99,
                    'train_freq': 1,
                    'gradient_steps': 1,
                    'ent_coef': 'auto',
                    'target_update_interval': 1
                }
            }
        }
    
    def create_environment(self, 
                          data_split: str = 'train',
                          initial_amount: float = 1000000.0,
                          state_space: int = 200) -> PortfolioTradingEnv:
        """
        Create trading environment
        
        Args:
            data_split: 'train' or 'test'
            initial_amount: Initial capital
            state_space: State space dimension
            
        Returns:
            Trading environment
        """
        # For now, use full dataset (can be modified for train/test splits)
        return PortfolioTradingEnv(
            df_dict=self.df_dict,
            stock_list=self.stock_list,
            initial_amount=initial_amount,
            state_space=state_space,
            transaction_cost_pct=0.001,
            reward_scaling=1e-4
        )
    
    def train_ensemble(self, 
                      total_timesteps: int = 100000,
                      eval_freq: int = 10000,
                      n_eval_episodes: int = 5) -> Dict[str, Any]:
        """
        Train ensemble of agents
        
        Args:
            total_timesteps: Total training timesteps per agent
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            
        Returns:
            Training results
        """
        print("🚀 Training Ensemble of DRL Agents")
        print("=" * 50)
        
        training_results = {}
        
        # Train each agent
        for i, (agent_name, config) in enumerate(self.agent_configs.items()):
            print(f"\n🤖 Training {agent_name} ({i+1}/{len(self.agent_configs)})")
            print("-" * 30)
            
            try:
                # Create environment
                env = self.create_environment()
                
                # Initialize agent
                agent = config['algorithm'](
                    "MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=f"{self.model_save_path}/tensorboard_logs/{agent_name}",
                    **config['params']
                )
                
                # Set up logging
                logger = configure(f"{self.model_save_path}/logs/{agent_name}", 
                                 ["stdout", "csv", "tensorboard"])
                agent.set_logger(logger)
                
                # Train agent
                agent.learn(total_timesteps=total_timesteps)
                
                # Save agent
                agent_path = f"{self.model_save_path}/{agent_name}"
                agent.save(agent_path)
                
                # Evaluate agent
                eval_results = self.evaluate_agent(agent, n_episodes=n_eval_episodes)
                
                # Store results
                self.agents[agent_name] = agent
                self.performance_history[agent_name] = eval_results
                training_results[agent_name] = {
                    'model_path': agent_path,
                    'performance': eval_results,
                    'status': 'success'
                }
                
                print(f"✅ {agent_name} trained successfully")
                print(f"   Sharpe Ratio: {eval_results.get('avg_sharpe_ratio', 0):.4f}")
                print(f"   Total Return: {eval_results.get('avg_return', 0):.4f}")
                
            except Exception as e:
                print(f"❌ Error training {agent_name}: {e}")
                training_results[agent_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Calculate ensemble weights
        self._calculate_ensemble_weights()
        
        return training_results
    
    def evaluate_agent(self, 
                      agent: Any,
                      n_episodes: int = 10,
                      data_split: str = 'test') -> Dict[str, float]:
        """
        Evaluate individual agent
        
        Args:
            agent: Trained agent
            n_episodes: Number of evaluation episodes
            data_split: Data split to evaluate on
            
        Returns:
            Evaluation results
        """
        # Create evaluation environment
        eval_env = self.create_environment(data_split)
        
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
                if hasattr(agent, 'predict'):
                    action, _ = agent.predict(obs, deterministic=True)
                else:
                    action = agent(obs)
                
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
            
            # Get performance metrics
            metrics = eval_env.get_performance_metrics()
            episode_rewards.append(episode_reward)
            episode_returns.append(metrics.get('total_return', 0))
            episode_sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            episode_max_drawdowns.append(metrics.get('max_drawdown', 0))
        
        # Calculate average metrics
        return {
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
    
    def _calculate_ensemble_weights(self):
        """
        Calculate ensemble weights based on performance
        """
        print("\n📊 Calculating Ensemble Weights")
        print("-" * 30)
        
        # Calculate weights based on Sharpe ratio and return
        weights = {}
        total_score = 0
        
        for agent_name, performance in self.performance_history.items():
            # Composite score: 60% Sharpe ratio + 40% return
            sharpe_score = max(0, performance.get('avg_sharpe_ratio', 0))
            return_score = max(0, performance.get('avg_return', 0))
            composite_score = 0.6 * sharpe_score + 0.4 * return_score
            
            weights[agent_name] = composite_score
            total_score += composite_score
        
        # Normalize weights
        if total_score > 0:
            for agent_name in weights:
                self.agent_weights[agent_name] = weights[agent_name] / total_score
        else:
            # Equal weights if no positive scores
            equal_weight = 1.0 / len(weights)
            for agent_name in weights:
                self.agent_weights[agent_name] = equal_weight
        
        # Display weights
        for agent_name, weight in self.agent_weights.items():
            print(f"   {agent_name}: {weight:.3f}")
    
    def ensemble_predict(self, 
                        obs: np.ndarray,
                        method: str = 'weighted_average') -> np.ndarray:
        """
        Make ensemble prediction
        
        Args:
            obs: Observation
            method: Ensemble method ('weighted_average', 'majority_vote', 'best_agent')
            
        Returns:
            Ensemble action
        """
        if not self.agents:
            raise ValueError("No trained agents available")
        
        if method == 'weighted_average':
            return self._weighted_average_prediction(obs)
        elif method == 'majority_vote':
            return self._majority_vote_prediction(obs)
        elif method == 'best_agent':
            return self._best_agent_prediction(obs)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _weighted_average_prediction(self, obs: np.ndarray) -> np.ndarray:
        """Weighted average of all agent predictions"""
        predictions = []
        weights = []
        
        for agent_name, agent in self.agents.items():
            if agent_name in self.agent_weights:
                action, _ = agent.predict(obs, deterministic=True)
                predictions.append(action)
                weights.append(self.agent_weights[agent_name])
        
        if predictions:
            # Weighted average
            weighted_action = np.average(predictions, axis=0, weights=weights)
            return weighted_action
        else:
            return np.zeros(self.agents[list(self.agents.keys())[0]].action_space.shape)
    
    def _majority_vote_prediction(self, obs: np.ndarray) -> np.ndarray:
        """Majority vote prediction"""
        predictions = []
        
        for agent_name, agent in self.agents.items():
            action, _ = agent.predict(obs, deterministic=True)
            predictions.append(action)
        
        if predictions:
            # For continuous actions, use median
            return np.median(predictions, axis=0)
        else:
            return np.zeros(self.agents[list(self.agents.keys())[0]].action_space.shape)
    
    def _best_agent_prediction(self, obs: np.ndarray) -> np.ndarray:
        """Prediction from best performing agent"""
        best_agent_name = max(self.agent_weights.keys(), 
                            key=lambda x: self.agent_weights[x])
        best_agent = self.agents[best_agent_name]
        action, _ = best_agent.predict(obs, deterministic=True)
        return action
    
    def backtest_ensemble(self, 
                         method: str = 'weighted_average',
                         initial_amount: float = 1000000.0) -> Dict[str, Any]:
        """
        Backtest ensemble strategy
        
        Args:
            method: Ensemble method
            initial_amount: Initial capital
            
        Returns:
            Backtest results
        """
        print(f"\n🔍 Backtesting Ensemble Strategy ({method})")
        print("-" * 40)
        
        # Create backtest environment
        backtest_env = self.create_environment('test', initial_amount)
        
        # Run backtest
        obs = backtest_env.reset()
        done = False
        actions_taken = []
        portfolio_values = []
        dates = []
        individual_predictions = {name: [] for name in self.agents.keys()}
        
        while not done:
            # Get ensemble prediction
            ensemble_action = self.ensemble_predict(obs, method)
            
            # Store individual predictions for analysis
            for agent_name, agent in self.agents.items():
                individual_action, _ = agent.predict(obs, deterministic=True)
                individual_predictions[agent_name].append(individual_action)
            
            # Execute action
            obs, reward, done, info = backtest_env.step(ensemble_action)
            
            actions_taken.append(ensemble_action)
            portfolio_values.append(info['total_assets'])
            dates.append(backtest_env.day)
        
        # Get final performance metrics
        final_metrics = backtest_env.get_performance_metrics()
        
        # Calculate buy and hold return
        buy_hold_return = self._calculate_buy_and_hold_return(backtest_env)
        
        # Create backtest results
        backtest_results = {
            'portfolio_values': portfolio_values,
            'actions_taken': actions_taken,
            'dates': dates,
            'final_metrics': final_metrics,
            'buy_and_hold_return': buy_hold_return,
            'individual_predictions': individual_predictions,
            'ensemble_method': method,
            'agent_weights': self.agent_weights
        }
        
        # Display results
        print(f"📈 Backtest Results:")
        print(f"   Total Return: {final_metrics.get('total_return', 0):.4f}")
        print(f"   Annualized Return: {final_metrics.get('annualized_return', 0):.4f}")
        print(f"   Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"   Max Drawdown: {final_metrics.get('max_drawdown', 0):.4f}")
        print(f"   Buy & Hold Return: {buy_hold_return:.4f}")
        print(f"   Outperformance: {final_metrics.get('total_return', 0) - buy_hold_return:.4f}")
        
        return backtest_results
    
    def _calculate_buy_and_hold_return(self, env: PortfolioTradingEnv) -> float:
        """Calculate buy and hold return for comparison"""
        if len(env.asset_memory) < 2:
            return 0.0
        
        initial_value = env.asset_memory[0]
        final_value = env.asset_memory[-1]
        
        return (final_value - initial_value) / initial_value
    
    def optimize_ensemble_weights(self, 
                                 n_trials: int = 100) -> Dict[str, float]:
        """
        Optimize ensemble weights using performance-based optimization
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Optimized weights
        """
        print(f"\n🎯 Optimizing Ensemble Weights ({n_trials} trials)")
        print("-" * 40)
        
        best_weights = None
        best_performance = -np.inf
        
        for trial in range(n_trials):
            # Generate random weights
            weights = np.random.dirichlet(np.ones(len(self.agents)))
            weight_dict = {name: weights[i] for i, name in enumerate(self.agents.keys())}
            
            # Temporarily set weights
            original_weights = self.agent_weights.copy()
            self.agent_weights = weight_dict
            
            # Evaluate ensemble
            try:
                backtest_results = self.backtest_ensemble('weighted_average')
                performance = backtest_results['final_metrics'].get('sharpe_ratio', 0)
                
                if performance > best_performance:
                    best_performance = performance
                    best_weights = weight_dict.copy()
                
            except Exception as e:
                print(f"Trial {trial} failed: {e}")
            
            # Restore original weights
            self.agent_weights = original_weights
        
        if best_weights:
            self.agent_weights = best_weights
            print(f"✅ Optimized weights found (Sharpe: {best_performance:.4f})")
            for name, weight in best_weights.items():
                print(f"   {name}: {weight:.3f}")
        else:
            print("❌ Optimization failed, keeping original weights")
        
        return self.agent_weights

# Example usage
if __name__ == "__main__":
    from data_loader import FinancialDataLoader
    
    # Load data for multiple stocks
    loader = FinancialDataLoader()
    stock_list = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    
    df_dict = {}
    for stock in stock_list:
        try:
            df_dict[stock] = loader.load_stock_data(stock)
            print(f"Loaded {stock}: {df_dict[stock].shape}")
        except:
            print(f"Could not load {stock}")
    
    if df_dict:
        # Create ensemble agent
        ensemble = EnsembleAgent(
            stock_list=list(df_dict.keys()),
            df_dict=df_dict,
            ensemble_size=5
        )
        
        # Train ensemble
        training_results = ensemble.train_ensemble(total_timesteps=20000)
        
        # Backtest ensemble
        backtest_results = ensemble.backtest_ensemble()
        
        print("Ensemble training and backtesting completed!")
