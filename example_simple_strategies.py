"""
Example usage of Simple Portfolio Strategies

This script demonstrates how to use the simple portfolio strategies
and compare them with the existing RL models.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from simple_portfolio_strategies import SimplePortfolioStrategies

def compare_strategies():
    """Compare different portfolio strategies."""
    
    # Initialize strategies
    strategies = SimplePortfolioStrategies(
        data_root="processed_data/",
        start_date='2024-06-06',
        end_date='2025-06-06'
    )
    
    # Run all strategies
    print("Running all portfolio strategies...")
    results = strategies.run_all_strategies()
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Portfolio values over time
    plt.subplot(2, 2, 1)
    for strategy_name, result in results.items():
        plt.plot(result['dates'], result['portfolio_values'][1:], 
                label=strategy_name, linewidth=2)
    plt.title('Portfolio Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Returns distribution
    plt.subplot(2, 2, 2)
    for strategy_name, result in results.items():
        plt.hist(result['returns'], alpha=0.5, label=strategy_name, bins=30)
    plt.title('Returns Distribution')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Risk-Return scatter
    plt.subplot(2, 2, 3)
    returns_list = []
    volatilities_list = []
    strategy_names = []
    
    for strategy_name, result in results.items():
        returns_list.append(result['annualized_return'])
        volatilities_list.append(result['volatility'])
        strategy_names.append(strategy_name)
    
    plt.scatter(volatilities_list, returns_list, s=100, alpha=0.7)
    for i, name in enumerate(strategy_names):
        plt.annotate(name, (volatilities_list[i], returns_list[i]), 
                    xytext=(5, 5), textcoords='offset points')
    plt.title('Risk-Return Profile')
    plt.xlabel('Volatility (Annualized)')
    plt.ylabel('Return (Annualized)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Sharpe Ratio comparison
    plt.subplot(2, 2, 4)
    sharpe_ratios = [result['sharpe_ratio'] for result in results.values()]
    strategy_names = list(results.keys())
    
    bars = plt.bar(strategy_names, sharpe_ratios, alpha=0.7)
    plt.title('Sharpe Ratio Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Color bars by performance
    colors = ['green' if sr > 0 else 'red' for sr in sharpe_ratios]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/simple_strategy_results/strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def analyze_individual_strategy(strategy_name: str):
    """Analyze a specific strategy in detail."""
    
    strategies = SimplePortfolioStrategies(
        data_root="processed_data/",
        start_date='2024-06-06',
        end_date='2025-06-06'
    )
    
    # Get strategy function
    strategy_functions = {
        'equal_weight': strategies.equal_weight_strategy,
        'volatility_adjusted_equal_weight': strategies.volatility_adjusted_equal_weight_strategy,
        'ma_crossover': strategies.ma_crossover_strategy,
        'momentum': strategies.momentum_strategy,
        'black_litterman': strategies.black_litterman_strategy,
        'minimum_variance': strategies.minimum_variance_strategy,
        'technical_analysis': strategies.technical_analysis_strategy,
        'sentiment': strategies.sentiment_strategy
    }
    
    if strategy_name not in strategy_functions:
        print(f"Strategy {strategy_name} not found!")
        return
    
    # Run strategy
    result = strategies.backtest_strategy(strategy_functions[strategy_name], strategy_name)
    
    # Print detailed analysis
    print(f"\nDetailed Analysis for {strategy_name.upper()} Strategy:")
    print("="*60)
    print(f"Total Return: {result['total_return']:.4f} ({result['total_return']*100:.2f}%)")
    print(f"Annualized Return: {result['annualized_return']:.4f} ({result['annualized_return']*100:.2f}%)")
    print(f"Volatility: {result['volatility']:.4f} ({result['volatility']*100:.2f}%)")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {result['max_drawdown']:.4f} ({result['max_drawdown']*100:.2f}%)")
    
    # Analyze weight distribution
    if 'weights_history' in result and len(result['weights_history']) > 0:
        weights_array = np.array(result['weights_history'])
        avg_weights = np.mean(weights_array, axis=0)
        
        print(f"\nAverage Portfolio Weights:")
        for i, stock in enumerate(strategies.stocks):
            print(f"  {stock}: {avg_weights[i]:.4f} ({avg_weights[i]*100:.2f}%)")
    
    return result

def main():
    """Main function."""
    print("Simple Portfolio Strategies Analysis")
    print("="*50)
    
    # Compare all strategies
    results = compare_strategies()
    
    # Analyze best performing strategy
    best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
    print(f"\nBest performing strategy: {best_strategy}")
    
    # Detailed analysis of best strategy
    analyze_individual_strategy(best_strategy)
    
    # Save results
    strategies = SimplePortfolioStrategies()
    summary_df = strategies.save_results(results)
    
    print(f"\nResults saved to 'evaluation_results/simple_strategy_results/' directory")
    print("\nSummary:")
    print(summary_df.to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    main()
