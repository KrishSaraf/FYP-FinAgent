#!/usr/bin/env python3
"""
Debug script to show exactly what's happening with portfolio calculations
"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_portfolio_calculation():
    """Show the bug in portfolio calculation"""
    
    print("🐛 DEBUGGING PORTFOLIO CALCULATION")
    print("=" * 60)
    
    # Load some real data
    csv_file = Path("processed_data/RELIANCE_aligned.csv")
    if not csv_file.exists():
        print("❌ No data file found")
        return
        
    df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
    
    # Calculate returns if not present
    if 'returns_1d' not in df.columns:
        df['returns_1d'] = df['close'].pct_change()
    
    # Get recent data
    recent_data = df.tail(10)
    
    print("📊 RELIANCE Recent Data:")
    print(recent_data[['close', 'returns_1d']].round(4))
    
    print("\n🔍 CURRENT BUGGY CALCULATION:")
    print("-" * 40)
    
    # Simulate current buggy logic
    initial_capital = 1000000
    transaction_cost = 0.005  # 0.5%
    
    portfolio_value = initial_capital
    
    print(f"Initial Capital: ₹{portfolio_value:,.0f}")
    
    for i, (date, row) in enumerate(recent_data.tail(5).iterrows()):
        stock_return = row['returns_1d']
        
        # Simulate equal weight for 5 stocks (20% each)
        equal_weight = 0.20
        
        # All HOLD signals (50% position each)
        position_weight = equal_weight * 0.5  # 10% each stock
        
        # Calculate portfolio return from stock movements
        portfolio_return = 5 * position_weight * stock_return  # 5 stocks, same return
        
        # BUG: Always subtract transaction costs!
        portfolio_return -= transaction_cost
        
        portfolio_value = portfolio_value * (1 + portfolio_return)
        
        print(f"Day {i+1} ({date.strftime('%Y-%m-%d')}):")
        print(f"  Stock Return: {stock_return:.4f} ({stock_return:.2%})")
        print(f"  Portfolio Return (before costs): {5 * position_weight * stock_return:.4f}")
        print(f"  Transaction Cost Applied: -{transaction_cost:.3f} ({transaction_cost:.1%})")
        print(f"  Net Portfolio Return: {portfolio_return:.4f} ({portfolio_return:.2%})")
        print(f"  Portfolio Value: ₹{portfolio_value:,.0f}")
        print()
    
    print("🚨 THE PROBLEM:")
    print("- Transaction costs (0.5%) are applied EVERY DAY")
    print("- Even when no trades happen!")
    print("- This causes consistent -0.5% daily drag")
    print("- Portfolio bleeds money regardless of market performance")
    
    print("\n✅ CORRECT LOGIC SHOULD BE:")
    print("-" * 40)
    
    # Show correct calculation
    portfolio_value = initial_capital
    previous_positions = {}
    
    print(f"Initial Capital: ₹{portfolio_value:,.0f}")
    
    for i, (date, row) in enumerate(recent_data.tail(5).iterrows()):
        stock_return = row['returns_1d']
        
        # Current positions (same as before for this example)
        current_positions = {'RELIANCE': equal_weight * 0.5}  # 10%
        
        # Check if positions changed (transaction costs only if they did)
        positions_changed = (previous_positions != current_positions)
        
        # Portfolio return from market movements
        portfolio_return = current_positions['RELIANCE'] * stock_return
        
        # Apply transaction costs ONLY if positions changed
        if positions_changed and i > 0:  # Skip first day
            portfolio_return -= transaction_cost
            transaction_applied = True
        else:
            transaction_applied = False
        
        portfolio_value = portfolio_value * (1 + portfolio_return)
        previous_positions = current_positions.copy()
        
        print(f"Day {i+1} ({date.strftime('%Y-%m-%d')}):")
        print(f"  Stock Return: {stock_return:.4f} ({stock_return:.2%})")
        print(f"  Portfolio Return (from market): {current_positions['RELIANCE'] * stock_return:.4f}")
        print(f"  Transaction Cost Applied: {'Yes' if transaction_applied else 'No'}")
        print(f"  Net Portfolio Return: {portfolio_return:.4f} ({portfolio_return:.2%})")
        print(f"  Portfolio Value: ₹{portfolio_value:,.0f}")
        print()
    
    print("💡 FIXES NEEDED:")
    print("1. Only apply transaction costs when positions actually change")
    print("2. Track previous positions to detect changes")
    print("3. Consider that 0.5% transaction cost per trade is very high")
    print("4. Maybe use 0.05% (5 basis points) instead of 0.5%")

if __name__ == "__main__":
    debug_portfolio_calculation()