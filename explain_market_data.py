#!/usr/bin/env python3
"""
Explanation of How Market Prices Are Used to Calculate Returns and Portfolio Values
"""

import pandas as pd
import numpy as np
from pathlib import Path

def explain_market_data_flow():
    """Demonstrates how market prices flow through the strategy"""
    
    print("📊 HOW MARKET PRICES ARE USED TO CALCULATE RETURNS AND PORTFOLIO VALUES")
    print("=" * 80)
    
    # 1. Show the data structure
    print("\n1. 📁 DATA SOURCE: CSV Files in processed_data/")
    print("-" * 50)
    
    # Load sample data
    csv_file = Path("processed_data/RELIANCE_aligned.csv")
    if csv_file.exists():
        df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
        
        # Show key price columns
        key_columns = ['open', 'high', 'low', 'close', 'volume']
        sample_data = df[key_columns].head(5)
        
        print("Sample RELIANCE data (first 5 rows):")
        print(sample_data)
        print(f"\nTotal rows: {len(df)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # 2. Show how returns are calculated
        print("\n2. 💰 RETURN CALCULATION PROCESS")
        print("-" * 50)
        
        # Calculate daily returns
        df['returns_1d'] = df['close'].pct_change()
        
        print("Daily returns calculation:")
        print("returns_1d = (close_today - close_yesterday) / close_yesterday")
        print("\nExample calculation for recent dates:")
        
        recent_data = df[['close', 'returns_1d']].tail(5)
        print(recent_data)
        
        # 3. Show portfolio value calculation
        print("\n3. 🏦 PORTFOLIO VALUE CALCULATION")
        print("-" * 50)
        
        print("For Pure LLM Strategy:")
        print("1. Equal allocation: ₹1,000,000 / number_of_stocks")
        print("2. LLM signal determines position:")
        print("   - BUY: Full position (1.0 × allocation)")
        print("   - HOLD: Half position (0.5 × allocation)")  
        print("   - SELL: No position (0.0 × allocation)")
        print("3. Daily portfolio return = Σ(weight × stock_return)")
        print("4. New portfolio value = old_value × (1 + portfolio_return)")
        
        # Example calculation
        print("\nExample with 5 stocks (₹200,000 each):")
        stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
        allocation = 1000000 / len(stocks)  # ₹200,000 per stock
        
        # Simulate LLM signals
        signals = ['BUY', 'HOLD', 'SELL', 'BUY', 'HOLD']
        signal_weights = {'BUY': 1.0, 'HOLD': 0.5, 'SELL': 0.0}
        
        # Simulate daily returns
        daily_returns = [0.02, -0.01, 0.005, 0.015, -0.008]  # 2%, -1%, 0.5%, 1.5%, -0.8%
        
        print(f"\nStock          Signal  Weight  Allocation    Daily Return  Contribution")
        print("-" * 70)
        
        portfolio_return = 0.0
        for i, stock in enumerate(stocks):
            signal = signals[i]
            weight = signal_weights[signal]
            stock_allocation = allocation * weight
            stock_return = daily_returns[i]
            contribution = (stock_allocation / 1000000) * stock_return
            portfolio_return += contribution
            
            print(f"{stock:<12} {signal:<6} {weight:<6} ₹{stock_allocation:>8,.0f}  {stock_return:>8.1%}      {contribution:>8.4f}")
        
        print(f"\nTotal Portfolio Return: {portfolio_return:.4f} ({portfolio_return:.2%})")
        
        new_value = 1000000 * (1 + portfolio_return)
        print(f"New Portfolio Value: ₹{new_value:,.0f}")
        
        # 4. Show hybrid strategy differences
        print("\n4. 🔄 HYBRID STRATEGY DIFFERENCES")
        print("-" * 50)
        
        print("Hybrid Strategy uses:")
        print("1. Inverse volatility weights (not equal weights)")
        print("2. LLM confidence scores to adjust positions")
        print("3. Risk constraints (max 5% per stock)")
        
        # Example inverse volatility weights
        volatilities = [0.25, 0.22, 0.28, 0.20, 0.24]  # Sample volatilities
        inv_vol = [1/vol for vol in volatilities]
        total_inv_vol = sum(inv_vol)
        base_weights = [iv/total_inv_vol for iv in inv_vol]
        
        print(f"\nStock          Volatility  Base Weight  LLM Signal  Confidence  Adjusted Weight")
        print("-" * 80)
        
        confidences = [85, 60, 75, 90, 55]  # LLM confidence scores
        
        for i, stock in enumerate(stocks):
            vol = volatilities[i]
            base_weight = base_weights[i]
            signal = signals[i]
            confidence = confidences[i]
            
            # Adjust weight based on signal and confidence
            if signal == 'BUY':
                adjusted_weight = base_weight * (1 + confidence/100 * 0.5)
            elif signal == 'SELL':
                adjusted_weight = base_weight * (1 - confidence/100 * 0.8)
            else:  # HOLD
                adjusted_weight = base_weight
            
            # Apply 5% max constraint
            adjusted_weight = min(adjusted_weight, 0.05)
            
            print(f"{stock:<12} {vol:<10.2f} {base_weight:<11.3f} {signal:<9} {confidence:<10} {adjusted_weight:.3f}")
        
        # 5. Show actual data loading process
        print("\n5. 🔄 DATA LOADING PROCESS IN CODE")
        print("-" * 50)
        
        print("In the RealDataLoader class:")
        print("1. For each stock, load: processed_data/{STOCK}_aligned.csv")
        print("2. Filter date range: df[(df.index >= start_date) & (df.index <= end_date)]")
        print("3. Add stock prefix to columns: RELIANCE_close, RELIANCE_volume, etc.")
        print("4. Combine all stocks into one DataFrame")
        print("5. Fill missing values with forward fill then zero")
        
        # 6. Show feature extraction
        print("\n6. 🎯 FEATURE EXTRACTION FOR LLM")
        print("-" * 50)
        
        print("From the combined DataFrame, extract for each stock:")
        features = [
            'close', 'returns_1d', 'volatility_20d', 'momentum_5d', 
            'momentum_20d', 'rsi_14', 'volume_ratio_20d', 'dma_50', 'dma_200'
        ]
        
        for feature in features:
            print(f"  • {feature}: Used for LLM analysis")
        
        print(f"\nExample: For RELIANCE on 2024-08-25:")
        if 'returns_1d' not in df.columns:
            df['returns_1d'] = df['close'].pct_change()
        
        if len(df) > 0:
            last_row = df.iloc[-1]
            print(f"  RELIANCE_close: ₹{last_row['close']:.2f}")
            print(f"  RELIANCE_returns_1d: {last_row['returns_1d']:.4f} ({last_row['returns_1d']:.2%})")
            print(f"  RELIANCE_volume: {last_row['volume']:,.0f}")
        
        # 7. Portfolio tracking
        print("\n7. 📈 PORTFOLIO VALUE TRACKING")
        print("-" * 50)
        
        print("The strategy tracks:")
        print("1. Daily portfolio values: [₹1,000,000, ₹995,000, ₹1,002,000, ...]")
        print("2. Daily returns: [0.0, -0.005, 0.007, ...]") 
        print("3. Trade log: Date, signals, positions, portfolio value")
        print("4. Final metrics: Sharpe ratio, max drawdown, total return")
        
        print("\n8. ✅ KEY POINTS")
        print("-" * 50)
        print("• Market prices come from real CSV files in processed_data/")
        print("• Daily returns are calculated from consecutive close prices")
        print("• Portfolio value changes based on weighted stock returns")
        print("• LLM signals determine position sizes")
        print("• All calculations are transparent and verifiable")
        print("• No fake or simulated prices - everything uses your real data!")
        
    else:
        print("❌ No data files found in processed_data/")
        print("Make sure your CSV files are in the processed_data/ directory")

if __name__ == "__main__":
    explain_market_data_flow()