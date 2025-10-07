"""
Advanced Ensemble Methods for FinRL
Implements sophisticated ensemble strategies to achieve >10% returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from stable_baselines3 import PPO, A2C, DDPG, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsemble:
    """
    Advanced ensemble methods for maximum trading performance
    """
    
    def __init__(self, env_class, env_kwargs: Dict):
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        
        # Ensemble strategies
        self.strategies = {
            'equal_weight': self._equal_weight_ensemble,
            'performance_weight': self._performance_weight_ensemble,
            'dynamic_weight': self._dynamic_weight_ensemble,
            'voting': self._voting_ensemble,
            'stacking': self._stacking_ensemble
        }
    
    def create_environment(self) -> gym.Env:
        """Create environment instance"""
        return self.env_class(**self.env_kwargs)
    
    def train_individual_models(self, algorithms: List[str], params: Dict[str, Dict]) -> Dict:
        """Train individual models with optimized parameters"""
        print("🚀 Training individual models for ensemble...")
        
        trained_models = {}
        
        for algorithm in algorithms:
            print(f"   Training {algorithm}...")
            
            env = self.create_environment()
            vec_env = DummyVecEnv([lambda: env])
            
            # Create model with optimized parameters
            if algorithm == 'PPO':
                model = PPO("MlpPolicy", vec_env, verbose=0, **params.get('PPO', {}))
            elif algorithm == 'A2C':
                model = A2C("MlpPolicy", vec_env, verbose=0, **params.get('A2C', {}))
            elif algorithm == 'DDPG':
                model = DDPG("MlpPolicy", vec_env, verbose=0, **params.get('DDPG', {}))
            elif algorithm == 'SAC':
                model = SAC("MlpPolicy", vec_env, verbose=0, **params.get('SAC', {}))
            else:
                continue
            
            # Train model
            model.learn(total_timesteps=50000)  # Longer training for ensemble
            
            # Evaluate performance
            performance = self._evaluate_model(model, env)
            
            trained_models[algorithm] = {
                'model': model,
                'performance': performance
            }
            
            print(f"   ✅ {algorithm} trained. Return: {performance['total_return']:.4f}")
        
        self.models = trained_models
        return trained_models
    
    def _evaluate_model(self, model, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate individual model performance"""
        total_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        
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
        
        return {
            'total_return': np.mean(total_returns),
            'sharpe_ratio': np.mean(sharpe_ratios),
            'max_drawdown': np.mean(max_drawdowns),
            'std_return': np.std(total_returns)
        }
    
    def _equal_weight_ensemble(self, observations: np.ndarray) -> np.ndarray:
        """Equal weight ensemble prediction"""
        predictions = []
        weights = []
        
        for algorithm, model_data in self.models.items():
            model = model_data['model']
            action, _ = model.predict(observations, deterministic=True)
            predictions.append(action)
            weights.append(1.0 / len(self.models))
        
        # Weighted average of actions
        ensemble_action = np.average(predictions, axis=0, weights=weights)
        return ensemble_action
    
    def _performance_weight_ensemble(self, observations: np.ndarray) -> np.ndarray:
        """Performance-weighted ensemble prediction"""
        predictions = []
        weights = []
        
        for algorithm, model_data in self.models.items():
            model = model_data['model']
            performance = model_data['performance']
            
            action, _ = model.predict(observations, deterministic=True)
            predictions.append(action)
            
            # Weight based on total return and Sharpe ratio
            weight = max(0, performance['total_return']) * (1 + performance['sharpe_ratio'])
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(self.models)] * len(self.models)
        
        ensemble_action = np.average(predictions, axis=0, weights=weights)
        return ensemble_action
    
    def _dynamic_weight_ensemble(self, observations: np.ndarray) -> np.ndarray:
        """Dynamic weight ensemble based on recent performance"""
        predictions = []
        weights = []
        
        for algorithm, model_data in self.models.items():
            model = model_data['model']
            action, _ = model.predict(observations, deterministic=True)
            predictions.append(action)
            
            # Dynamic weight based on recent performance
            if algorithm in self.performance_history:
                recent_performance = np.mean(self.performance_history[algorithm][-10:])  # Last 10 episodes
                weight = max(0, recent_performance)
            else:
                weight = model_data['performance']['total_return']
            
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(self.models)] * len(self.models)
        
        ensemble_action = np.average(predictions, axis=0, weights=weights)
        return ensemble_action
    
    def _voting_ensemble(self, observations: np.ndarray) -> np.ndarray:
        """Voting ensemble prediction"""
        predictions = []
        
        for algorithm, model_data in self.models.items():
            model = model_data['model']
            action, _ = model.predict(observations, deterministic=True)
            predictions.append(action)
        
        # Majority vote (for discrete actions)
        if len(predictions[0].shape) == 0:  # Single action
            ensemble_action = np.argmax(np.bincount(predictions))
        else:  # Multiple actions
            ensemble_action = np.round(np.mean(predictions, axis=0))
        
        return ensemble_action
    
    def _stacking_ensemble(self, observations: np.ndarray) -> np.ndarray:
        """Stacking ensemble with meta-learner"""
        # Get base predictions
        base_predictions = []
        for algorithm, model_data in self.models.items():
            model = model_data['model']
            action, _ = model.predict(observations, deterministic=True)
            base_predictions.append(action)
        
        # Simple meta-learner: weighted combination based on confidence
        # For now, use equal weights (can be enhanced with actual meta-learner)
        ensemble_action = np.mean(base_predictions, axis=0)
        
        return ensemble_action
    
    def predict_ensemble(self, observations: np.ndarray, strategy: str = 'performance_weight') -> np.ndarray:
        """Make ensemble prediction using specified strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self.strategies[strategy](observations)
    
    def evaluate_ensemble(self, strategy: str = 'performance_weight', n_episodes: int = 20) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        print(f"🔍 Evaluating {strategy} ensemble...")
        
        total_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        
        for episode in range(n_episodes):
            env = self.create_environment()
            obs = env.reset()
            episode_returns = []
            portfolio_values = [env.initial_amount]
            
            done = False
            while not done:
                # Get ensemble prediction
                action = self.predict_ensemble(obs, strategy)
                
                # Execute action
                obs, reward, done, info = env.step(action)
                episode_returns.append(reward)
                portfolio_values.append(info.get('portfolio_value', portfolio_values[-1]))
                
                # Update performance history for dynamic weighting
                if strategy == 'dynamic_weight':
                    for algorithm in self.models.keys():
                        if algorithm not in self.performance_history:
                            self.performance_history[algorithm] = []
                        # Simple performance tracking (can be enhanced)
                        self.performance_history[algorithm].append(reward)
            
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
            'strategy': strategy,
            'total_return': np.mean(total_returns),
            'sharpe_ratio': np.mean(sharpe_ratios),
            'max_drawdown': np.mean(max_drawdowns),
            'win_rate': np.mean(win_rates),
            'calmar_ratio': np.mean(total_returns) / (abs(np.mean(max_drawdowns)) + 1e-8),
            'std_return': np.std(total_returns)
        }
    
    def evaluate_all_strategies(self) -> pd.DataFrame:
        """Evaluate all ensemble strategies"""
        print("🚀 Evaluating all ensemble strategies...")
        
        results = []
        for strategy in self.strategies.keys():
            try:
                result = self.evaluate_ensemble(strategy)
                results.append(result)
                print(f"   ✅ {strategy}: {result['total_return']:.4f} return")
            except Exception as e:
                print(f"   ❌ {strategy}: Error - {e}")
        
        return pd.DataFrame(results).sort_values('total_return', ascending=False)
    
    def get_ensemble_summary(self) -> Dict:
        """Get summary of ensemble performance"""
        individual_performance = {}
        for algorithm, model_data in self.models.items():
            individual_performance[algorithm] = model_data['performance']
        
        return {
            'individual_models': individual_performance,
            'ensemble_strategies': list(self.strategies.keys()),
            'best_individual': max(individual_performance.items(), key=lambda x: x[1]['total_return'])
        }

# Example usage
if __name__ == "__main__":
    print("Advanced ensemble system ready!")
    print("Use with: ensemble = AdvancedEnsemble(env_class, env_kwargs)")
    print("Then: ensemble.train_individual_models(algorithms, params)")
    print("And: results = ensemble.evaluate_all_strategies()")
