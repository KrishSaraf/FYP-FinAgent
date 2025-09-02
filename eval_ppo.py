import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import json
import os
from datetime import datetime
import pickle
from typing import List, Dict, Any

# Import your custom environment
from finagent.environment.portfolio_env import PortfolioEnv

def evaluate_ppo_models(
    model_paths: List[str],
    data_root: str = "processed_data/",
    eval_start_date: str = '2025-03-07',
    eval_end_date: str = '2025-06-06',
    output_dir: str = "evaluation_results",
    deterministic: bool = True,
    save_actions: bool = True,
    save_observations: bool = False
):
    """
    Evaluate multiple PPO models and save their outcomes and actions.
    
    Args:
        model_paths: List of paths to saved PPO model files (.zip)
        data_root: Root directory for market data
        eval_start_date: Start date for evaluation period
        eval_end_date: End date for evaluation period
        output_dir: Directory to save evaluation results
        deterministic: Whether to use deterministic actions during evaluation
        save_actions: Whether to save action sequences
        save_observations: Whether to save observation sequences (can be large)
    
    Returns:
        DataFrame with summary results for all models
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize results storage
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting evaluation of {len(model_paths)} models...")
    print(f"Evaluation period: {eval_start_date} to {eval_end_date}")
    print(f"Results will be saved to: {output_dir}")
    
    for i, model_path in enumerate(model_paths):
        print(f"\n--- Evaluating Model {i+1}/{len(model_paths)}: {model_path} ---")
        
        try:
            # Load the model
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found: {model_path}")
                continue
                
            model = PPO.load(model_path)
            model_name = os.path.basename(model_path).replace('.zip', '')
            
            # Create evaluation environment
            eval_env = PortfolioEnv(
                data_root=data_root,
                start_date=eval_start_date,
                end_date=eval_end_date
            )
            
            # Initialize episode tracking
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            # Storage for episode data
            actions_taken = []
            rewards_received = []
            observations_list = []
            info_history = []
            
            print(f"Starting evaluation for {model_name}...")
            
            # Run evaluation episode
            while not done:
                # Get action from model
                action, _states = model.predict(obs, deterministic=deterministic)
                
                # Store data before step
                if save_actions:
                    actions_taken.append(action.tolist() if hasattr(action, 'tolist') else action)
                if save_observations:
                    observations_list.append(obs.tolist() if hasattr(obs, 'tolist') else obs)
                
                # Take action in environment
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                # Track episode statistics
                episode_reward += reward
                step_count += 1
                rewards_received.append(reward)
                info_history.append(info.copy())
                
                # Optional: Print progress every N steps
                if step_count % 100 == 0:
                    print(f"  Step {step_count}, Portfolio Value: {info.get('portfolio_value', 'N/A'):.2f}")
            
            # Collect final results
            final_info = info_history[-1] if info_history else {}
            
            result = {
                'model_name': model_name,
                'model_path': model_path,
                'evaluation_date': timestamp,
                'eval_start_date': eval_start_date,
                'eval_end_date': eval_end_date,
                'total_steps': step_count,
                'total_reward': episode_reward,
                'final_portfolio_value': final_info.get('portfolio_value', 0),
                'total_return_pct': final_info.get('total_return', 0),
                'sharpe_ratio': final_info.get('sharpe_ratio', 0),
                'max_drawdown': final_info.get('max_drawdown', 0),
                'volatility': final_info.get('volatility', 0),
                'avg_reward_per_step': episode_reward / step_count if step_count > 0 else 0,
                'deterministic': deterministic
            }
            
            # Add any additional metrics from the final info
            for key, value in final_info.items():
                if key not in result:
                    result[key] = value
            
            all_results.append(result)
            
            # Save detailed episode data for this model
            episode_data = {
                'model_name': model_name,
                'model_path': model_path,
                'evaluation_metadata': {
                    'eval_start_date': eval_start_date,
                    'eval_end_date': eval_end_date,
                    'total_steps': step_count,
                    'deterministic': deterministic,
                    'evaluation_timestamp': timestamp
                },
                'summary_metrics': result,
                'step_rewards': rewards_received,
                'info_history': info_history
            }
            
            if save_actions:
                episode_data['actions'] = actions_taken
            if save_observations:
                episode_data['observations'] = observations_list
            
            # Save individual model results
            model_output_file = os.path.join(output_dir, f"{model_name}_evaluation_{timestamp}.json")
            with open(model_output_file, 'w') as f:
                json.dump(episode_data, f, indent=2, default=str)
            
            print(f"✓ Model {model_name} evaluation completed")
            print(f"  Final Portfolio Value: {result['final_portfolio_value']:.2f}")
            print(f"  Total Return: {result['total_return_pct']:.2f}%")
            print(f"  Volatility: {result['volatility']:.4f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.4f}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print(f"  Results saved to: {model_output_file}")
            
        except Exception as e:
            print(f"ERROR evaluating {model_path}: {str(e)}")
            # Add error entry to results
            all_results.append({
                'model_name': os.path.basename(model_path).replace('.zip', ''),
                'model_path': model_path,
                'evaluation_date': timestamp,
                'error': str(e),
                'total_steps': 0,
                'total_reward': 0,
                'final_portfolio_value': 0,
                'total_return_pct': 0,
                'sharpe_ratio': 0
            })
            continue
    
    # Create summary results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save summary results
    summary_file = os.path.join(output_dir, f"model_comparison_summary_{timestamp}.csv")
    results_df.to_csv(summary_file, index=False)
    
    # Save summary as JSON as well for easier programmatic access
    summary_json_file = os.path.join(output_dir, f"model_comparison_summary_{timestamp}.json")
    with open(summary_json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Summary results saved to: {summary_file}")
    print(f"JSON summary saved to: {summary_json_file}")
    
    # Print quick comparison
    if len(results_df) > 0:
        print(f"\n--- Model Performance Comparison ---")
        comparison_cols = ['model_name', 'final_portfolio_value', 'total_return_pct', 'sharpe_ratio']
        available_cols = [col for col in comparison_cols if col in results_df.columns]
        print(results_df[available_cols].to_string(index=False))
        
        # Find best performing model
        if 'total_return_pct' in results_df.columns:
            best_model_idx = results_df['total_return_pct'].idxmax()
            best_model = results_df.loc[best_model_idx]
            print(f"\nBest performing model: {best_model['model_name']}")
            print(f"  Return: {best_model['total_return_pct']:.2f}%")
            if 'sharpe_ratio' in best_model:
                print(f"  Sharpe Ratio: {best_model['sharpe_ratio']:.4f}")
    
    return results_df

def evaluate_models_from_directory(
    models_directory: str,
    model_pattern: str = "*.zip",
    **kwargs
):
    """
    Convenience function to evaluate all models in a directory.
    
    Args:
        models_directory: Directory containing model files
        model_pattern: File pattern to match (default: *.zip)
        **kwargs: Additional arguments passed to evaluate_ppo_models
    """
    import glob
    
    # Find all model files matching pattern
    search_pattern = os.path.join(models_directory, model_pattern)
    model_paths = glob.glob(search_pattern)
    
    if not model_paths:
        print(f"No model files found in {models_directory} matching pattern {model_pattern}")
        return None
    
    print(f"Found {len(model_paths)} model files in {models_directory}")
    model_paths.sort()  # Sort for consistent ordering
    
    return evaluate_ppo_models(model_paths, **kwargs)

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    
    # Option 1: Evaluate specific models
    model_list = [
        "models/PPO-1756197233/checkpoint_50000_steps.zip",
        "models/PPO-1756216174/checkpoint_100000_steps.zip",
        "models/PPO-1756216174/checkpoint_150000_steps.zip",
        "models/PPO-1756216174/checkpoint_200000_steps.zip",
        "models/PPO-1756216174/checkpoint_250000_steps.zip",
        "models/PPO-1756216174/checkpoint_300000_steps.zip",
        "models/PPO-1756270453/checkpoint_350000_steps.zip",
        "models/PPO-1756270453/checkpoint_400000_steps.zip",
        "models/PPO-1756270453/checkpoint_450000_steps.zip",
        "models/PPO-1756270453/checkpoint_500000_steps.zip",
        "models/PPO-1756270453/checkpoint_550000_steps.zip",
        "models/PPO-1756270453/checkpoint_600000_steps.zip",
        "models/PPO-1756270453/checkpoint_650000_steps.zip",
        "models/PPO-1756270453/checkpoint_700000_steps.zip",
        "models/PPO-1756270453/checkpoint_750000_steps.zip",
        "models/PPO-1756379121/checkpoint_800000_steps.zip",
        "models/PPO-1756379121/checkpoint_850000_steps.zip",
        "models/PPO-1756379121/checkpoint_900000_steps.zip",
        "models/PPO-1756379121/checkpoint_950000_steps.zip",
        "models/PPO-1756379121/checkpoint_1000000_steps.zip",
        "models/PPO-1756379121/final_model.zip"
    ]
    
    # Evaluate the models
    results = evaluate_ppo_models(
        model_paths=model_list,
        data_root="processed_data/",
        eval_start_date='2025-03-07',
        eval_end_date='2025-06-06',
        output_dir="evaluation_results",
        deterministic=True,
        save_actions=True,
        save_observations=False  # Set to True if you want to save observations (can be large)
    )
    
    # Option 2: Evaluate all models in a directory
    # results = evaluate_models_from_directory(
    #     models_directory="models/PPO-1756379121/",
    #     model_pattern="*.zip",
    #     data_root="processed_data/",
    #     eval_start_date='2025-03-07',
    #     eval_end_date='2025-06-06',
    #     output_dir="evaluation_results",
    #     deterministic=True,
    #     save_actions=True,
    #     save_observations=False
    # )
    
    print("\nEvaluation script completed!")