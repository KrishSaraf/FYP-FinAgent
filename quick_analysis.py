"""
Quick Portfolio Analysis Script
Simple and fast analysis of backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def quick_analysis():
    """Quick analysis of portfolio performance"""
    
    # Load data
    print("Loading data...")
    portfolio_values = pd.read_csv('portfolio_values_daily.csv', index_col=0, parse_dates=True)
    pnl = pd.read_csv('pnl_daily.csv', index_col=0, parse_dates=True)
    turnover = pd.read_csv('turnover_daily.csv', index_col=0, parse_dates=True)
    
    # Calculate key metrics
    initial_value = portfolio_values.iloc[0,0]
    final_value = portfolio_values.iloc[-1,0]
    total_return = (final_value / initial_value - 1) * 100
    
    sharpe = (pnl.iloc[:,0].mean() / pnl.iloc[:,0].std()) * np.sqrt(252)
    max_dd = ((portfolio_values.iloc[:,0] / portfolio_values.iloc[:,0].expanding().max()) - 1).min() * 100
    volatility = pnl.iloc[:,0].std() * np.sqrt(252) * 100
    avg_turnover = turnover.iloc[:,0].mean() * 100
    
    # Print summary
    print("\n" + "="*50)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Initial Portfolio Value: ${initial_value:,.0f}")
    print(f"Final Portfolio Value:   ${final_value:,.0f}")
    print(f"Total Return:            {total_return:.1f}%")
    print(f"Sharpe Ratio:            {sharpe:.2f}")
    print(f"Maximum Drawdown:        {max_dd:.1f}%")
    print(f"Volatility (Annual):     {volatility:.1f}%")
    print(f"Average Turnover:        {avg_turnover:.1f}%")
    print(f"Trading Days:            {len(pnl)}")
    print("="*50)
    
    # Create simple plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio value
    ax1.plot(portfolio_values.index, portfolio_values.iloc[:,0], linewidth=2, color='blue')
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Daily returns
    colors = ['green' if x > 0 else 'red' for x in pnl.iloc[:,0]]
    ax2.bar(pnl.index, pnl.iloc[:,0] * 100, color=colors, alpha=0.7, width=1)
    ax2.set_title('Daily Returns', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Daily Return (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('quick_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nQuick analysis plot saved as 'quick_analysis.png'")

if __name__ == "__main__":
    quick_analysis()
