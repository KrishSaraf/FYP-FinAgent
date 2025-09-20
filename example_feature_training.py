"""
Example script demonstrating how to use the feature combination PPO trainer.

This script shows how to train models with different feature combinations
and compare their performance.
"""

import logging
from train_ppo_feature_combinations import run_training_with_combination, FeatureSelector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Example training with different feature combinations"""
    
    # Base configuration
    base_config = {
        # Environment settings
        'seed': 42,
        'data_root': 'processed_data/',
        'stocks': None,  # Will be loaded from stocks.txt
        'train_start_date': '2024-06-06',
        'train_end_date': '2025-03-06',
        'window_size': 30,
        'transaction_cost_rate': 0.005,
        'sharpe_window': 252,
        
        # Data loading settings
        'use_all_features': False,  # We'll use custom feature selection
        'fill_missing_features_with': 'interpolate',
        'save_cache': True,
        'cache_format': 'hdf5',
        'force_reload': False,
        'preload_to_gpu': True,
        
        # Training environment (smaller for demo)
        'n_envs': 4,
        'n_steps': 16,
        
        # PPO hyperparameters (smaller for demo)
        'num_updates': 100,  # Reduced for demo
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_eps': 0.2,
        'ppo_epochs': 2,
        'ppo_batch_size': 64,
        'learning_rate': 3e-4,
        'max_grad_norm': 1.0,
        'value_coeff': 0.5,
        'entropy_coeff': 0.02,
        'action_std': 0.5,
        
        # Network architecture (smaller for demo)
        'hidden_size': 128,
        'n_lstm_layers': 1,
        
        # GPU optimizations
        'use_mixed_precision': True,
        'compile_mode': 'default',
        'memory_efficient': True,
        'gradient_checkpointing': True,
        
        # Logging and monitoring
        'use_wandb': False,  # Disabled for demo
        'log_interval': 10,
        'save_interval': 50,
        'model_dir': 'models',
    }
    
    # Feature combinations to test
    feature_combinations = [
        'ohlcv',                    # Basic price data only
        'ohlcv+technical',          # Price + technical indicators
        'ohlcv+financial',          # Price + financial metrics
        'ohlcv+sentiment',          # Price + sentiment
        'ohlcv+technical+financial', # Price + technical + financial
        'all'                       # All features
    ]
    
    logger.info("Starting feature combination training examples...")
    
    # Show available combinations
    feature_selector = FeatureSelector()
    feature_selector.print_available_combinations()
    
    # Train models with different feature combinations
    for combination in feature_combinations:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training with feature combination: {combination}")
            logger.info(f"{'='*60}")
            
            # Update config for this combination
            config = base_config.copy()
            config['model_name'] = f"demo_{combination.replace('+', '_')}"
            config['num_updates'] = 50  # Even smaller for demo
            
            # Run training
            run_training_with_combination(combination, config)
            
            logger.info(f"✅ Successfully trained model with {combination} features")
            
        except Exception as e:
            logger.error(f"❌ Failed to train model with {combination} features: {e}")
            continue
    
    logger.info("\n🎉 Feature combination training examples completed!")
    logger.info("Check the 'models' directory for saved models.")


if __name__ == "__main__":
    main()
