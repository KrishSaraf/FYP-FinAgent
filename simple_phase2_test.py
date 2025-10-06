"""
Simplified Phase 2 Test - Working Version
Tests the core functionality without complex optimization
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import FinancialDataLoader
from portfolio_environment import PortfolioTradingEnv
from stable_baselines3 import PPO

def test_simple_phase2():
    """
    Simple test of Phase 2 functionality
    """
    print("🧪 Simple Phase 2 Test")
    print("=" * 40)
    
    # Load data
    print("📊 Loading data...")
    loader = FinancialDataLoader()
    
    # Load 3 stocks
    stock_list = ["RELIANCE", "TCS", "HDFCBANK"]
    df_dict = {}
    
    for stock in stock_list:
        try:
            df_dict[stock] = loader.load_stock_data(stock)
            print(f"✅ {stock}: {df_dict[stock].shape}")
        except Exception as e:
            print(f"❌ Error loading {stock}: {e}")
            return
    
    # Create environment
    print("\n🏗️ Creating environment...")
    try:
        env = PortfolioTradingEnv(
            df_dict=df_dict,
            stock_list=stock_list,
            state_space=150,
            initial_amount=1000000.0,
            transaction_cost_pct=0.001,
            reward_scaling=1e-4
        )
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return
    
    # Test environment
    print("\n🧪 Testing environment...")
    try:
        obs = env.reset()
        print(f"✅ Environment reset: obs shape = {obs.shape}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(f"   Step {i+1}: reward = {reward:.4f}, portfolio = {info['total_assets']:.2f}")
            
            if done:
                break
        
        print("✅ Environment test passed")
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return
    
    # Train a simple PPO agent
    print("\n🤖 Training simple PPO agent...")
    try:
        # Create PPO agent
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )
        
        # Train for a short time
        print("   Training for 5000 timesteps...")
        model.learn(total_timesteps=5000)
        print("✅ Training completed")
        
        # Test the trained agent
        print("\n📊 Testing trained agent...")
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        for i in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        
        print(f"✅ Agent test completed:")
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Steps: {steps}")
        print(f"   Final portfolio value: {info['total_assets']:.2f}")
        print(f"   Total return: {metrics.get('total_return', 0):.4f}")
        print(f"   Sharpe ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        
        # Check if we achieved positive returns
        total_return = metrics.get('total_return', 0)
        if total_return > 0:
            print(f"🎉 SUCCESS: Positive returns achieved! ({total_return*100:.2f}%)")
        else:
            print(f"⚠️ Negative returns: {total_return*100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_phase2()
    
    if success:
        print("\n🎉 Simple Phase 2 test completed successfully!")
        print("✅ Core functionality is working")
        print("✅ Ready for full Phase 2 implementation")
    else:
        print("\n❌ Simple Phase 2 test failed")
        print("🔧 Need to fix issues before running full Phase 2")
