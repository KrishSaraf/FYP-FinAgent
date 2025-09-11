#!/usr/bin/env python3
"""
Simple command to run LLM Enhanced Portfolio Strategy

This script provides a working implementation that respects API rate limits
and provides clear results for both Pure LLM and Hybrid strategies.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Import our fixed strategy
from llm_enhanced_strategy_fixed import LLMEnhancedStrategyFixed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_strategy_test(test_mode: str = "demo"):
    """
    Run LLM Enhanced Strategy test
    
    Args:
        test_mode: "demo" for quick test, "full" for complete backtest
    """
    
    # Configuration
    MISTRAL_API_KEY = "5cqXuAMrvlEapMQjZMlJfChoH5npmMs8"
    
    # Load stocks
    stocks_file = Path("finagent/stocks.txt")
    if stocks_file.exists():
        with open(stocks_file, 'r') as f:
            all_stocks = [line.strip() for line in f.readlines() if line.strip()]
        
        # Use fewer stocks for demo to avoid rate limits
        if test_mode == "demo":
            stocks = all_stocks[:5]  # First 5 stocks
        else:
            stocks = all_stocks[:10]  # First 10 stocks
    else:
        stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    
    print(f"\n{'='*60}")
    print(f"LLM ENHANCED PORTFOLIO STRATEGY - {test_mode.upper()} MODE")
    print(f"{'='*60}")
    print(f"Testing with {len(stocks)} stocks: {stocks}")
    print(f"API: Mistral (Free tier - rate limited)")
    print(f"Data: Real market data from processed_data/")
    print(f"Strategies: Pure LLM vs LLM + Quant Hybrid")
    
    # Initialize strategy
    strategy = LLMEnhancedStrategyFixed(
        api_key=MISTRAL_API_KEY,
        stocks=stocks,
        data_root="processed_data/",
        initial_capital=1000000.0
    )
    
    # Test periods
    if test_mode == "demo":
        start_date = "2024-08-20"
        end_date = "2024-08-30"
    else:
        start_date = "2024-08-01"
        end_date = "2024-09-01"
    
    print(f"Test period: {start_date} to {end_date}")
    print("-" * 60)
    
    try:
        # Strategy 1: Pure LLM (Equal allocation + LLM signals)
        print("\n🤖 RUNNING PURE LLM STRATEGY...")
        print("   • Equal allocation across all stocks")
        print("   • LLM provides BUY/SELL/HOLD signals")
        print("   • BUY = full position, SELL = no position, HOLD = half position")
        
        pure_llm_results = strategy.run_pure_llm_strategy(start_date, end_date)
        
        if "error" not in pure_llm_results:
            print(f"   ✅ Pure LLM completed: {pure_llm_results['total_return']:.2%} return")
        else:
            print(f"   ❌ Pure LLM failed: {pure_llm_results['error']}")
            return
        
        # Add delay between strategies to respect rate limits
        print("   ⏳ Waiting 30 seconds for rate limit reset...")
        time.sleep(30)
        
        # Strategy 2: Hybrid (Inverse volatility + LLM confidence)
        print("\n🔬 RUNNING LLM + QUANT HYBRID STRATEGY...")
        print("   • Inverse volatility weighting for base allocation")
        print("   • LLM signals with confidence scores adjust positions")
        print("   • Higher confidence = larger position adjustments")
        
        hybrid_results = strategy.run_hybrid_strategy(start_date, end_date)
        
        if "error" not in hybrid_results:
            print(f"   ✅ Hybrid completed: {hybrid_results['total_return']:.2%} return")
        else:
            print(f"   ❌ Hybrid failed: {hybrid_results['error']}")
            return
        
        # Results Summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        # Pure LLM Results
        pure_metrics = pure_llm_results['metrics']
        print(f"\n📊 Pure LLM Strategy:")
        print(f"   Initial Capital:  ₹{pure_llm_results['initial_capital']:,.0f}")
        print(f"   Final Capital:    ₹{pure_llm_results['final_capital']:,.0f}")
        print(f"   Total Return:     {pure_llm_results['total_return']:.2%}")
        print(f"   Sharpe Ratio:     {pure_metrics.sharpe_ratio:.3f}")
        print(f"   Max Drawdown:     {pure_metrics.max_drawdown:.2%}")
        print(f"   Volatility:       {pure_metrics.volatility:.2%}")
        print(f"   Win/Loss Ratio:   {pure_metrics.win_loss_ratio:.2f}")
        print(f"   Number of Trades: {pure_llm_results['n_trades']}")
        
        # Hybrid Results
        hybrid_metrics = hybrid_results['metrics']
        print(f"\n🔄 LLM + Quant Hybrid Strategy:")
        print(f"   Initial Capital:  ₹{hybrid_results['initial_capital']:,.0f}")
        print(f"   Final Capital:    ₹{hybrid_results['final_capital']:,.0f}")
        print(f"   Total Return:     {hybrid_results['total_return']:.2%}")
        print(f"   Sharpe Ratio:     {hybrid_metrics.sharpe_ratio:.3f}")
        print(f"   Max Drawdown:     {hybrid_metrics.max_drawdown:.2%}")
        print(f"   Volatility:       {hybrid_metrics.volatility:.2%}")
        print(f"   Win/Loss Ratio:   {hybrid_metrics.win_loss_ratio:.2f}")
        print(f"   Number of Trades: {hybrid_results['n_trades']}")
        
        # Comparison
        return_diff = hybrid_results['total_return'] - pure_llm_results['total_return']
        sharpe_diff = hybrid_metrics.sharpe_ratio - pure_metrics.sharpe_ratio
        
        better_strategy = "🔄 Hybrid" if hybrid_results['total_return'] > pure_llm_results['total_return'] else "🤖 Pure LLM"
        
        print(f"\n🏆 WINNER: {better_strategy}")
        print(f"   Return Difference:  {return_diff:.2%}")
        print(f"   Sharpe Difference:  {sharpe_diff:.3f}")
        
        # Sample signals
        if pure_llm_results['trade_log']:
            print(f"\n📈 Sample LLM Signals (last day):")
            last_trade = pure_llm_results['trade_log'][-1]
            for stock, signal in last_trade['signals'].items():
                print(f"   {stock}: {signal}")
        
        print("\n" + "="*60)
        print("✅ STRATEGY TESTING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"llm_strategy_results_{timestamp}.json"
        
        import json
        results = {
            'test_mode': test_mode,
            'test_period': f"{start_date} to {end_date}",
            'stocks': stocks,
            'pure_llm': {
                'total_return': pure_llm_results['total_return'],
                'sharpe_ratio': pure_metrics.sharpe_ratio,
                'max_drawdown': pure_metrics.max_drawdown,
                'final_capital': pure_llm_results['final_capital']
            },
            'hybrid': {
                'total_return': hybrid_results['total_return'],
                'sharpe_ratio': hybrid_metrics.sharpe_ratio,
                'max_drawdown': hybrid_metrics.max_drawdown,
                'final_capital': hybrid_results['final_capital']
            },
            'winner': better_strategy,
            'return_difference': return_diff,
            'sharpe_difference': sharpe_diff
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Error running strategy: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function with command line arguments"""
    
    # Check for test mode argument
    if len(sys.argv) > 1:
        test_mode = sys.argv[1].lower()
        if test_mode not in ["demo", "full"]:
            print("Usage: python run_llm_strategy.py [demo|full]")
            print("  demo: Quick test with 5 stocks, 10 days")
            print("  full: Complete test with 10 stocks, 1 month")
            sys.exit(1)
    else:
        test_mode = "demo"
    
    run_strategy_test(test_mode)

if __name__ == "__main__":
    main()