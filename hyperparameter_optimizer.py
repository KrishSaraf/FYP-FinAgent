"""
Advanced Hyperparameter Optimization for FinRL
Uses Optuna for Bayesian optimization to achieve >10% returns
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from stable_baselines3 import PPO, A2C, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import gym
from gym import spaces
import warnings
warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Optuna
    """
    
    def __init__(self, env_class, env_kwargs: Dict, n_trials: int = 100):
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.n_trials = n_trials
        self.best_params = {}
        self.optimization_results = []
        
        # Define optimization objectives
        self.objectives = {
            'total_return': 'maximize',
            'sharpe_ratio': 'maximize', 
            'max_drawdown': 'minimize',
            'win_rate': 'maximize',
            'calmar_ratio': 'maximize'
        }
    
    def create_environment(self) -> gym.Env:
        """Create environment instance"""
        return self.env_class(**self.env_kwargs)
    
    def optimize_ppo(self, study_name: str = "ppo_optimization") -> Dict:
        """Optimize PPO hyperparameters"""
        print("🚀 Starting PPO hyperparameter optimization...")
        
        def objective(trial):
            # PPO hyperparameters to optimize
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'n_epochs': trial.suggest_int('n_epochs', 3, 20),
                'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
                'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0)
            }
            
            # Create environment
            env = self.create_environment()
            vec_env = DummyVecEnv([lambda: env])
            
            # Create model
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=0,
                **params
            )
            
            # Train model
            model.learn(total_timesteps=20000)
            
            # Evaluate model
            metrics = self.evaluate_model(model, env)
            
            # Return primary objective (total return)
            return metrics['total_return']
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials)
        
        # Store results
        self.best_params['PPO'] = study.best_params
        self.optimization_results.append({
            'algorithm': 'PPO',
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        })
        
        print(f"✅ PPO optimization completed. Best return: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_a2c(self, study_name: str = "a2c_optimization") -> Dict:
        """Optimize A2C hyperparameters"""
        print("🚀 Starting A2C hyperparameter optimization...")
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [5, 10, 20, 50]),
                'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
                'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
                'use_rms_prop': trial.suggest_categorical('use_rms_prop', [True, False])
            }
            
            env = self.create_environment()
            vec_env = DummyVecEnv([lambda: env])
            
            model = A2C(
                "MlpPolicy",
                vec_env,
                verbose=0,
                **params
            )
            
            model.learn(total_timesteps=20000)
            metrics = self.evaluate_model(model, env)
            
            return metrics['total_return']
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['A2C'] = study.best_params
        self.optimization_results.append({
            'algorithm': 'A2C',
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        })
        
        print(f"✅ A2C optimization completed. Best return: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_ddpg(self, study_name: str = "ddpg_optimization") -> Dict:
        """Optimize DDPG hyperparameters"""
        print("🚀 Starting DDPG hyperparameter optimization...")
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000]),
                'learning_starts': trial.suggest_int('learning_starts', 100, 1000),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'tau': trial.suggest_float('tau', 0.001, 0.02),
                'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
                'gradient_steps': trial.suggest_int('gradient_steps', 1, 5),
                'noise_type': trial.suggest_categorical('noise_type', ['ornstein-uhlenbeck', 'normal']),
                'noise_std': trial.suggest_float('noise_std', 0.1, 1.0)
            }
            
            env = self.create_environment()
            vec_env = DummyVecEnv([lambda: env])
            
            model = DDPG(
                "MlpPolicy",
                vec_env,
                verbose=0,
                **params
            )
            
            model.learn(total_timesteps=20000)
            metrics = self.evaluate_model(model, env)
            
            return metrics['total_return']
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['DDPG'] = study.best_params
        self.optimization_results.append({
            'algorithm': 'DDPG',
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        })
        
        print(f"✅ DDPG optimization completed. Best return: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_sac(self, study_name: str = "sac_optimization") -> Dict:
        """Optimize SAC hyperparameters"""
        print("🚀 Starting SAC hyperparameter optimization...")
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000]),
                'learning_starts': trial.suggest_int('learning_starts', 100, 1000),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'tau': trial.suggest_float('tau', 0.001, 0.02),
                'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
                'gradient_steps': trial.suggest_int('gradient_steps', 1, 5),
                'ent_coef': trial.suggest_categorical('ent_coef', ['auto', 0.1, 0.2, 0.5]),
                'target_update_interval': trial.suggest_int('target_update_interval', 1, 10)
            }
            
            env = self.create_environment()
            vec_env = DummyVecEnv([lambda: env])
            
            model = SAC(
                "MlpPolicy",
                vec_env,
                verbose=0,
                **params
            )
            
            model.learn(total_timesteps=20000)
            metrics = self.evaluate_model(model, env)
            
            return metrics['total_return']
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['SAC'] = study.best_params
        self.optimization_results.append({
            'algorithm': 'SAC',
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        })
        
        print(f"✅ SAC optimization completed. Best return: {study.best_value:.4f}")
        return study.best_params
    
    def evaluate_model(self, model, env: gym.Env, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate model performance"""
        total_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_returns = []
            portfolio_values = [env.initial_amount]
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_returns.append(reward)
                portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
            
            # Calculate metrics
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            total_returns.append(total_return)
            
            # Sharpe ratio
            if len(episode_returns) > 1:
                sharpe = np.mean(episode_returns) / (np.std(episode_returns) + 1e-8) * np.sqrt(252)
                sharpe_ratios.append(sharpe)
            
            # Max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak
            max_drawdowns.append(np.min(drawdown))
            
            # Win rate
            win_rate = np.mean([r > 0 for r in episode_returns])
            win_rates.append(win_rate)
        
        return {
            'total_return': np.mean(total_returns),
            'sharpe_ratio': np.mean(sharpe_ratios),
            'max_drawdown': np.mean(max_drawdowns),
            'win_rate': np.mean(win_rates),
            'calmar_ratio': np.mean(total_returns) / (abs(np.mean(max_drawdowns)) + 1e-8)
        }
    
    def optimize_all_algorithms(self) -> Dict[str, Dict]:
        """Optimize all algorithms and return best parameters"""
        print("🚀 Starting comprehensive hyperparameter optimization...")
        
        # Optimize each algorithm
        self.optimize_ppo()
        self.optimize_a2c()
        self.optimize_ddpg()
        self.optimize_sac()
        
        # Find best overall algorithm
        best_algorithm = max(self.optimization_results, key=lambda x: x['best_value'])
        
        print(f"\n🏆 Best Algorithm: {best_algorithm['algorithm']}")
        print(f"   Best Return: {best_algorithm['best_value']:.4f}")
        print(f"   Best Parameters: {best_algorithm['best_params']}")
        
        return {
            'best_algorithm': best_algorithm['algorithm'],
            'best_params': self.best_params,
            'all_results': self.optimization_results
        }
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of optimization results"""
        summary_data = []
        for result in self.optimization_results:
            summary_data.append({
                'Algorithm': result['algorithm'],
                'Best_Return': result['best_value'],
                'N_Trials': result['n_trials'],
                'Best_Params': str(result['best_params'])
            })
        
        return pd.DataFrame(summary_data).sort_values('Best_Return', ascending=False)

# Example usage
if __name__ == "__main__":
    # This would be used with actual environment
    print("Hyperparameter optimization system ready!")
    print("Use with: optimizer = HyperparameterOptimizer(env_class, env_kwargs)")
    print("Then: results = optimizer.optimize_all_algorithms()")