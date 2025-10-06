"""
Hyperparameter Optimization Pipeline for FinRL Phase 2
Implements advanced optimization techniques to achieve >10% returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

# Custom imports
from portfolio_environment import PortfolioTradingEnv
from ensemble_agents import EnsembleAgent

class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization for FinRL agents
    """
    
    def __init__(self, 
                 stock_list: List[str],
                 df_dict: Dict[str, pd.DataFrame],
                 optimization_target: str = 'sharpe_ratio',
                 n_trials: int = 100):
        """
        Initialize hyperparameter optimizer
        
        Args:
            stock_list: List of stock symbols
            df_dict: Dictionary of DataFrames for each stock
            optimization_target: Target metric for optimization
            n_trials: Number of optimization trials
        """
        self.stock_list = stock_list
        self.df_dict = df_dict
        self.optimization_target = optimization_target
        self.n_trials = n_trials
        
        # Optimization results
        self.best_params = {}
        self.optimization_history = []
        self.best_performance = -np.inf
    
    def optimize_ppo_hyperparameters(self, 
                                   total_timesteps: int = 20000,
                                   n_eval_episodes: int = 3) -> Dict[str, Any]:
        """
        Optimize PPO hyperparameters using Optuna
        
        Args:
            total_timesteps: Training timesteps per trial
            n_eval_episodes: Evaluation episodes per trial
            
        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available, using grid search instead")
            return self._grid_search_ppo_hyperparameters(total_timesteps, n_eval_episodes)
        
        print("🎯 Optimizing PPO Hyperparameters with Optuna")
        print("=" * 50)
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'n_epochs': trial.suggest_int('n_epochs', 5, 20),
                'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
                'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0)
            }
            
            try:
                # Create environment
                env = PortfolioTradingEnv(
                    df_dict=self.df_dict,
                    stock_list=self.stock_list,
                    state_space=200,
                    initial_amount=1000000.0
                )
                
                # Create and train agent
                from stable_baselines3 import PPO
                agent = PPO("MlpPolicy", env, verbose=0, **params)
                agent.learn(total_timesteps=total_timesteps)
                
                # Evaluate agent
                performance = self._evaluate_agent(agent, env, n_eval_episodes)
                
                # Return target metric
                return performance.get(self.optimization_target, 0)
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return -np.inf
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials)
        
        # Store results
        self.best_params['PPO'] = study.best_params
        self.best_performance = study.best_value
        
        print(f"✅ Best PPO parameters found:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value}")
        print(f"   Best {self.optimization_target}: {study.best_value:.4f}")
        
        return study.best_params
    
    def _grid_search_ppo_hyperparameters(self, 
                                       total_timesteps: int = 20000,
                                       n_eval_episodes: int = 3) -> Dict[str, Any]:
        """
        Grid search for PPO hyperparameters (fallback when Optuna not available)
        """
        print("🔍 Grid Search for PPO Hyperparameters")
        print("=" * 40)
        
        # Define search space
        search_space = {
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'n_steps': [1024, 2048],
            'batch_size': [32, 64],
            'n_epochs': [10, 15],
            'gamma': [0.95, 0.99],
            'gae_lambda': [0.9, 0.95],
            'clip_range': [0.1, 0.2],
            'ent_coef': [0.0, 0.01],
            'vf_coef': [0.25, 0.5]
        }
        
        best_params = {}
        best_performance = -np.inf
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(search_space)
        
        for i, params in enumerate(param_combinations):
            print(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Create environment
                env = PortfolioTradingEnv(
                    df_dict=self.df_dict,
                    stock_list=self.stock_list,
                    state_space=200,
                    initial_amount=1000000.0
                )
                
                # Create and train agent
                from stable_baselines3 import PPO
                agent = PPO("MlpPolicy", env, verbose=0, **params)
                agent.learn(total_timesteps=total_timesteps)
                
                # Evaluate agent
                performance = self._evaluate_agent(agent, env, n_eval_episodes)
                target_metric = performance.get(self.optimization_target, 0)
                
                if target_metric > best_performance:
                    best_performance = target_metric
                    best_params = params.copy()
                
                print(f"   Performance: {target_metric:.4f}")
                
            except Exception as e:
                print(f"   Trial failed: {e}")
        
        self.best_params['PPO'] = best_params
        self.best_performance = best_performance
        
        print(f"✅ Best PPO parameters found:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"   Best {self.optimization_target}: {best_performance:.4f}")
        
        return best_params
    
    def _generate_parameter_combinations(self, search_space: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search"""
        import itertools
        
        keys = list(search_space.keys())
        values = list(search_space.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_agent(self, 
                       agent: Any,
                       env: PortfolioTradingEnv,
                       n_episodes: int = 3) -> Dict[str, float]:
        """
        Evaluate agent performance
        
        Args:
            agent: Trained agent
            env: Trading environment
            n_episodes: Number of evaluation episodes
            
        Returns:
            Performance metrics
        """
        episode_returns = []
        episode_sharpe_ratios = []
        episode_max_drawdowns = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            
            # Get performance metrics
            metrics = env.get_performance_metrics()
            episode_returns.append(metrics.get('total_return', 0))
            episode_sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            episode_max_drawdowns.append(metrics.get('max_drawdown', 0))
        
        return {
            'total_return': np.mean(episode_returns),
            'sharpe_ratio': np.mean(episode_sharpe_ratios),
            'max_drawdown': np.mean(episode_max_drawdowns),
            'volatility': np.std(episode_returns)
        }
    
    def optimize_ensemble_weights(self, 
                                 ensemble_agent: EnsembleAgent,
                                 n_trials: int = 50) -> Dict[str, float]:
        """
        Optimize ensemble weights using Bayesian optimization
        
        Args:
            ensemble_agent: Trained ensemble agent
            n_trials: Number of optimization trials
            
        Returns:
            Optimized weights
        """
        if not OPTUNA_AVAILABLE:
            return ensemble_agent.optimize_ensemble_weights(n_trials)
        
        print("🎯 Optimizing Ensemble Weights with Optuna")
        print("=" * 40)
        
        def objective(trial):
            # Generate weights using Dirichlet distribution
            alpha = trial.suggest_float('alpha', 0.1, 2.0)
            weights = np.random.dirichlet(np.full(len(ensemble_agent.agents), alpha))
            weight_dict = {name: weights[i] for i, name in enumerate(ensemble_agent.agents.keys())}
            
            # Temporarily set weights
            original_weights = ensemble_agent.agent_weights.copy()
            ensemble_agent.agent_weights = weight_dict
            
            try:
                # Evaluate ensemble
                backtest_results = ensemble_agent.backtest_ensemble('weighted_average')
                performance = backtest_results['final_metrics'].get(self.optimization_target, 0)
                
                return performance
                
            except Exception as e:
                return -np.inf
            finally:
                # Restore original weights
                ensemble_agent.agent_weights = original_weights
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Apply best weights
        best_alpha = study.best_params['alpha']
        best_weights = np.random.dirichlet(np.full(len(ensemble_agent.agents), best_alpha))
        weight_dict = {name: best_weights[i] for i, name in enumerate(ensemble_agent.agents.keys())}
        
        ensemble_agent.agent_weights = weight_dict
        
        print(f"✅ Optimized ensemble weights found:")
        for name, weight in weight_dict.items():
            print(f"   {name}: {weight:.3f}")
        print(f"   Best {self.optimization_target}: {study.best_value:.4f}")
        
        return weight_dict
    
    def optimize_environment_parameters(self, 
                                      n_trials: int = 30) -> Dict[str, Any]:
        """
        Optimize environment parameters
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Best environment parameters
        """
        if not OPTUNA_AVAILABLE:
            return self._grid_search_environment_parameters()
        
        print("🎯 Optimizing Environment Parameters")
        print("=" * 40)
        
        def objective(trial):
            # Define environment parameter search space
            params = {
                'transaction_cost_pct': trial.suggest_float('transaction_cost_pct', 0.0005, 0.005),
                'reward_scaling': trial.suggest_float('reward_scaling', 1e-5, 1e-3, log=True),
                'state_space': trial.suggest_categorical('state_space', [150, 200, 250]),
                'hmax': trial.suggest_categorical('hmax', [50, 100, 200]),
                'turbulence_threshold': trial.suggest_float('turbulence_threshold', 100, 200)
            }
            
            try:
                # Create environment with parameters
                env = PortfolioTradingEnv(
                    df_dict=self.df_dict,
                    stock_list=self.stock_list,
                    **params
                )
                
                # Create and train a simple agent for evaluation
                from stable_baselines3 import PPO
                agent = PPO("MlpPolicy", env, verbose=0, 
                          learning_rate=3e-4, n_steps=2048, batch_size=64)
                agent.learn(total_timesteps=10000)
                
                # Evaluate environment
                performance = self._evaluate_agent(agent, env, n_episodes=2)
                
                return performance.get(self.optimization_target, 0)
                
            except Exception as e:
                return -np.inf
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        print(f"✅ Best environment parameters found:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value}")
        print(f"   Best {self.optimization_target}: {study.best_value:.4f}")
        
        return study.best_params
    
    def _grid_search_environment_parameters(self) -> Dict[str, Any]:
        """Grid search for environment parameters"""
        print("🔍 Grid Search for Environment Parameters")
        print("=" * 40)
        
        search_space = {
            'transaction_cost_pct': [0.001, 0.002, 0.005],
            'reward_scaling': [1e-4, 5e-4, 1e-3],
            'state_space': [150, 200],
            'hmax': [100, 200],
            'turbulence_threshold': [140, 180]
        }
        
        best_params = {}
        best_performance = -np.inf
        
        param_combinations = self._generate_parameter_combinations(search_space)
        
        for i, params in enumerate(param_combinations):
            print(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Create environment
                env = PortfolioTradingEnv(
                    df_dict=self.df_dict,
                    stock_list=self.stock_list,
                    **params
                )
                
                # Create and train agent
                from stable_baselines3 import PPO
                agent = PPO("MlpPolicy", env, verbose=0, 
                          learning_rate=3e-4, n_steps=2048, batch_size=64)
                agent.learn(total_timesteps=10000)
                
                # Evaluate
                performance = self._evaluate_agent(agent, env, n_episodes=2)
                target_metric = performance.get(self.optimization_target, 0)
                
                if target_metric > best_performance:
                    best_performance = target_metric
                    best_params = params.copy()
                
                print(f"   Performance: {target_metric:.4f}")
                
            except Exception as e:
                print(f"   Trial failed: {e}")
        
        print(f"✅ Best environment parameters found:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        print(f"   Best {self.optimization_target}: {best_performance:.4f}")
        
        return best_params
    
    def comprehensive_optimization(self, 
                                 total_timesteps: int = 20000) -> Dict[str, Any]:
        """
        Comprehensive optimization of all parameters
        
        Args:
            total_timesteps: Training timesteps per trial
            
        Returns:
            Best parameters for all components
        """
        print("🚀 Comprehensive Hyperparameter Optimization")
        print("=" * 60)
        
        optimization_results = {}
        
        # 1. Optimize environment parameters
        print("\n1️⃣ Optimizing Environment Parameters")
        env_params = self.optimize_environment_parameters()
        optimization_results['environment'] = env_params
        
        # 2. Optimize PPO hyperparameters
        print("\n2️⃣ Optimizing PPO Hyperparameters")
        ppo_params = self.optimize_ppo_hyperparameters(total_timesteps)
        optimization_results['ppo'] = ppo_params
        
        # 3. Create optimized ensemble
        print("\n3️⃣ Creating Optimized Ensemble")
        ensemble_agent = EnsembleAgent(
            stock_list=self.stock_list,
            df_dict=self.df_dict,
            ensemble_size=5
        )
        
        # Train ensemble with optimized parameters
        training_results = ensemble_agent.train_ensemble(
            total_timesteps=total_timesteps // 2  # Reduced for ensemble
        )
        
        # 4. Optimize ensemble weights
        print("\n4️⃣ Optimizing Ensemble Weights")
        ensemble_weights = self.optimize_ensemble_weights(ensemble_agent)
        optimization_results['ensemble_weights'] = ensemble_weights
        
        # 5. Final evaluation
        print("\n5️⃣ Final Evaluation")
        final_backtest = ensemble_agent.backtest_ensemble('weighted_average')
        
        optimization_results['final_performance'] = final_backtest['final_metrics']
        optimization_results['ensemble_agent'] = ensemble_agent
        
        print(f"\n🎉 Optimization Complete!")
        print(f"   Final Sharpe Ratio: {final_backtest['final_metrics'].get('sharpe_ratio', 0):.4f}")
        print(f"   Final Total Return: {final_backtest['final_metrics'].get('total_return', 0):.4f}")
        print(f"   Final Annualized Return: {final_backtest['final_metrics'].get('annualized_return', 0):.4f}")
        
        return optimization_results

# Example usage
if __name__ == "__main__":
    from data_loader import FinancialDataLoader
    
    # Load data
    loader = FinancialDataLoader()
    stock_list = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    
    df_dict = {}
    for stock in stock_list:
        try:
            df_dict[stock] = loader.load_stock_data(stock)
        except:
            print(f"Could not load {stock}")
    
    if df_dict:
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            stock_list=list(df_dict.keys()),
            df_dict=df_dict,
            optimization_target='sharpe_ratio',
            n_trials=20  # Reduced for demo
        )
        
        # Run comprehensive optimization
        results = optimizer.comprehensive_optimization(total_timesteps=10000)
        
        print("Optimization completed!")
