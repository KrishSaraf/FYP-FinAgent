"""
Phase 1: Single Stock Deep Dive - RELIANCE
Main execution script for FinRL training and evaluation
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import FinancialDataLoader
from trading_environment import SingleStockTradingEnv
from training_pipeline import FinRLTrainingPipeline
import matplotlib.pyplot as plt

def main():
    """
    Main execution function for Phase 1
    """
    print("=" * 60)
    print("🚀 FinRL Phase 1: Single Stock Deep Dive - RELIANCE")
    print("=" * 60)
    
    # Initialize training pipeline
    pipeline = FinRLTrainingPipeline(
        stock_symbol="RELIANCE",
        model_save_path="trained_models",
        results_path="results"
    )
    
    print("\n📊 Step 1: Loading and Preparing Data")
    print("-" * 40)
    
    # Load and prepare data
    try:
        train_data, test_data = pipeline.load_and_prepare_data(train_ratio=0.8)
        print("✅ Data loaded and prepared successfully!")
        
        # Display data summary
        print(f"\n📈 Data Summary:")
        print(f"   Training data shape: {train_data.shape}")
        print(f"   Testing data shape: {test_data.shape}")
        print(f"   Date range: {train_data.index[0]} to {train_data.index[-1]}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print("\n🤖 Step 2: Training PPO Agent")
    print("-" * 40)
    
    # Train PPO agent
    try:
        model = pipeline.train_ppo_agent(
            total_timesteps=20000,  # Reduced for faster execution
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )
        print("✅ PPO agent trained successfully!")
        
    except Exception as e:
        print(f"❌ Error training agent: {e}")
        return
    
    print("\n📊 Step 3: Evaluating Model")
    print("-" * 40)
    
    # Evaluate model
    try:
        evaluation_results = pipeline.evaluate_model(n_episodes=5)
        print("✅ Model evaluation completed!")
        
        # Display evaluation results
        print(f"\n📊 Evaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"   {metric}: {value:.4f}")
            
    except Exception as e:
        print(f"❌ Error evaluating model: {e}")
        return
    
    print("\n🔍 Step 4: Running Backtest")
    print("-" * 40)
    
    # Run backtest
    try:
        backtest_results = pipeline.backtest_strategy()
        print("✅ Backtest completed!")
        
        # Display backtest results
        final_metrics = backtest_results['final_metrics']
        print(f"\n📈 Backtest Results:")
        print(f"   Total Return: {final_metrics.get('total_return', 0):.4f}")
        print(f"   Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"   Max Drawdown: {final_metrics.get('max_drawdown', 0):.4f}")
        print(f"   Win Rate: {final_metrics.get('win_rate', 0):.4f}")
        print(f"   Total Trades: {final_metrics.get('total_trades', 0)}")
        
        # Compare with buy and hold
        buy_hold_return = backtest_results['buy_and_hold_return']
        strategy_return = final_metrics.get('total_return', 0)
        print(f"\n📊 Strategy vs Buy & Hold:")
        print(f"   Strategy Return: {strategy_return:.4f}")
        print(f"   Buy & Hold Return: {buy_hold_return:.4f}")
        print(f"   Outperformance: {strategy_return - buy_hold_return:.4f}")
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return
    
    print("\n📊 Step 5: Generating Results and Plots")
    print("-" * 40)
    
    # Plot and save results
    try:
        pipeline.plot_results(backtest_results)
        pipeline.save_results(backtest_results)
        print("✅ Results and plots generated successfully!")
        
    except Exception as e:
        print(f"❌ Error generating results: {e}")
        return
    
    print("\n🎉 Phase 1 Completed Successfully!")
    print("=" * 60)
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"   Stock: RELIANCE")
    print(f"   Model: PPO")
    print(f"   Training Data: {train_data.shape[0]} days")
    print(f"   Test Data: {test_data.shape[0]} days")
    print(f"   Final Portfolio Value: {final_metrics.get('final_portfolio_value', 0):.2f}")
    print(f"   Strategy Return: {strategy_return:.4f}")
    print(f"   Buy & Hold Return: {buy_hold_return:.4f}")
    
    print(f"\n📁 Files Generated:")
    print(f"   Trained Model: trained_models/ppo_RELIANCE")
    print(f"   Results: results/RELIANCE_*.csv")
    print(f"   Plots: results/RELIANCE_results.png")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Analyze the results and plots")
    print(f"   2. Experiment with different hyperparameters")
    print(f"   3. Try other algorithms (A2C, DDPG, SAC)")
    print(f"   4. Move to Phase 2: Portfolio Management")

if __name__ == "__main__":
    main()
