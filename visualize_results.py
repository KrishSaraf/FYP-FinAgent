"""
Portfolio Performance Visualization Script
Creates comprehensive graphs from the backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load all backtest results"""
    print("Loading backtest results...")
    
    # Load data
    weights_df = pd.read_csv('weights_daily.csv', index_col=0, parse_dates=True)
    pnl_df = pd.read_csv('pnl_daily.csv', index_col=0, parse_dates=True)
    turnover_df = pd.read_csv('turnover_daily.csv', index_col=0, parse_dates=True)
    portfolio_values_df = pd.read_csv('portfolio_values_daily.csv', index_col=0, parse_dates=True)
    
    print(f"Loaded data for {len(weights_df)} trading days")
    print(f"Portfolio value range: ${portfolio_values_df.iloc[0,0]:,.0f} to ${portfolio_values_df.iloc[-1,0]:,.0f}")
    
    return weights_df, pnl_df, turnover_df, portfolio_values_df

def plot_portfolio_performance(portfolio_values_df, pnl_df):
    """Plot portfolio value and cumulative returns"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Portfolio Value
    ax1.plot(portfolio_values_df.index, portfolio_values_df.iloc[:,0], 
             linewidth=2, color='darkblue', label='Portfolio Value')
    ax1.set_title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Daily Returns
    ax2.bar(pnl_df.index, pnl_df.iloc[:,0], 
            color=['green' if x > 0 else 'red' for x in pnl_df.iloc[:,0]], 
            alpha=0.7, width=1)
    ax2.set_title('Daily Returns', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Daily Return', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Cumulative Returns
    cumulative_returns = (1 + pnl_df.iloc[:,0]).cumprod() - 1
    ax3.plot(cumulative_returns.index, cumulative_returns * 100, 
             linewidth=2, color='purple', label='Cumulative Returns')
    ax3.set_title('Cumulative Returns', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('portfolio_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_risk_metrics(pnl_df):
    """Plot risk metrics and drawdown analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rolling Sharpe Ratio (30-day)
    rolling_returns = pnl_df.iloc[:,0].rolling(window=30)
    rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
    ax1.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='orange')
    ax1.set_title('30-Day Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1')
    ax1.legend()
    
    # Drawdown Analysis
    cumulative = (1 + pnl_df.iloc[:,0]).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    ax2.fill_between(drawdown.index, drawdown * 100, 0, 
                     color='red', alpha=0.3, label='Drawdown')
    ax2.plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
    ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Rolling Volatility (30-day)
    rolling_vol = pnl_df.iloc[:,0].rolling(window=30).std() * np.sqrt(252) * 100
    ax3.plot(rolling_vol.index, rolling_vol, linewidth=2, color='green')
    ax3.set_title('30-Day Rolling Volatility', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Volatility (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Return Distribution
    ax4.hist(pnl_df.iloc[:,0] * 100, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(pnl_df.iloc[:,0].mean() * 100, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {pnl_df.iloc[:,0].mean()*100:.2f}%')
    ax4.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Daily Return (%)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('risk_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_portfolio_weights(weights_df):
    """Plot portfolio weights over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top 10 holdings over time
    # Calculate average weights and get top 10
    avg_weights = weights_df.mean().sort_values(ascending=False)
    top_10_stocks = avg_weights.head(10).index
    
    # Plot top 10 holdings
    for stock in top_10_stocks:
        ax1.plot(weights_df.index, weights_df[stock] * 100, 
                linewidth=2, label=stock, alpha=0.8)
    
    ax1.set_title('Top 10 Holdings Over Time', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Weight (%)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Portfolio concentration (sum of top 5 weights)
    top_5_weights = weights_df[top_10_stocks[:5]].sum(axis=1) * 100
    ax2.plot(top_5_weights.index, top_5_weights, 
             linewidth=2, color='red', label='Top 5 Holdings')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% Line')
    ax2.set_title('Portfolio Concentration (Top 5 Holdings)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Concentration (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('portfolio_weights.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_turnover_analysis(turnover_df, pnl_df):
    """Plot turnover and its relationship with returns"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Daily Turnover
    ax1.plot(turnover_df.index, turnover_df.iloc[:,0] * 100, 
             linewidth=1, color='blue', alpha=0.7)
    ax1.set_title('Daily Portfolio Turnover', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Turnover (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Rolling Average Turnover
    rolling_turnover = turnover_df.iloc[:,0].rolling(window=20).mean() * 100
    ax2.plot(rolling_turnover.index, rolling_turnover, 
             linewidth=2, color='red')
    ax2.set_title('20-Day Rolling Average Turnover', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Turnover (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Turnover vs Returns Scatter
    ax3.scatter(turnover_df.iloc[:,0] * 100, pnl_df.iloc[:,0] * 100, 
               alpha=0.6, color='green')
    ax3.set_title('Turnover vs Daily Returns', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Turnover (%)', fontsize=12)
    ax3.set_ylabel('Daily Return (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(turnover_df.iloc[:,0], pnl_df.iloc[:,0])[0,1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=ax3.transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Turnover Distribution
    ax4.hist(turnover_df.iloc[:,0] * 100, bins=20, alpha=0.7, 
             color='orange', edgecolor='black')
    ax4.axvline(turnover_df.iloc[:,0].mean() * 100, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {turnover_df.iloc[:,0].mean()*100:.2f}%')
    ax4.set_title('Turnover Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Turnover (%)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('turnover_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_summary(portfolio_values_df, pnl_df, turnover_df):
    """Create a comprehensive performance summary"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Portfolio Value (top row, spans 3 columns)
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.plot(portfolio_values_df.index, portfolio_values_df.iloc[:,0], 
             linewidth=2, color='darkblue')
    ax1.set_title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Key Metrics (top right)
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.axis('off')
    
    # Calculate key metrics
    total_return = (portfolio_values_df.iloc[-1,0] / portfolio_values_df.iloc[0,0] - 1) * 100
    sharpe = (pnl_df.iloc[:,0].mean() / pnl_df.iloc[:,0].std()) * np.sqrt(252)
    max_dd = ((portfolio_values_df.iloc[:,0] / portfolio_values_df.iloc[:,0].expanding().max()) - 1).min() * 100
    avg_turnover = turnover_df.iloc[:,0].mean() * 100
    volatility = pnl_df.iloc[:,0].std() * np.sqrt(252) * 100
    
    metrics_text = f"""
    KEY METRICS
    
    Total Return: {total_return:.1f}%
    Sharpe Ratio: {sharpe:.2f}
    Max Drawdown: {max_dd:.1f}%
    Avg Turnover: {avg_turnover:.1f}%
    Volatility: {volatility:.1f}%
    
    Initial Value: ${portfolio_values_df.iloc[0,0]:,.0f}
    Final Value: ${portfolio_values_df.iloc[-1,0]:,.0f}
    Trading Days: {len(pnl_df)}
    """
    
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Daily Returns (middle left)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.bar(pnl_df.index, pnl_df.iloc[:,0] * 100, 
            color=['green' if x > 0 else 'red' for x in pnl_df.iloc[:,0]], 
            alpha=0.7, width=1)
    ax3.set_title('Daily Returns', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Daily Return (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Drawdown (middle right)
    ax4 = fig.add_subplot(gs[1, 2:])
    cumulative = (1 + pnl_df.iloc[:,0]).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    ax4.fill_between(drawdown.index, drawdown * 100, 0, 
                     color='red', alpha=0.3)
    ax4.plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
    ax4.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Drawdown (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Return Distribution (bottom left)
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.hist(pnl_df.iloc[:,0] * 100, bins=20, alpha=0.7, 
             color='skyblue', edgecolor='black')
    ax5.axvline(pnl_df.iloc[:,0].mean() * 100, color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {pnl_df.iloc[:,0].mean()*100:.2f}%')
    ax5.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Daily Return (%)', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Turnover (bottom right)
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.plot(turnover_df.index, turnover_df.iloc[:,0] * 100, 
             linewidth=1, color='blue', alpha=0.7)
    ax6.set_title('Daily Portfolio Turnover', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Turnover (%)', fontsize=12)
    ax6.set_xlabel('Date', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Portfolio Performance Summary', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to generate all visualizations"""
    print("=" * 60)
    print("PORTFOLIO PERFORMANCE VISUALIZATION")
    print("=" * 60)
    
    # Load data
    weights_df, pnl_df, turnover_df, portfolio_values_df = load_results()
    
    # Generate all plots
    print("\n1. Generating Portfolio Performance Plots...")
    plot_portfolio_performance(portfolio_values_df, pnl_df)
    
    print("2. Generating Risk Metrics Plots...")
    plot_risk_metrics(pnl_df)
    
    print("3. Generating Portfolio Weights Analysis...")
    plot_portfolio_weights(weights_df)
    
    print("4. Generating Turnover Analysis...")
    plot_turnover_analysis(turnover_df, pnl_df)
    
    print("5. Generating Performance Summary...")
    plot_performance_summary(portfolio_values_df, pnl_df, turnover_df)
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 60)
    print("Generated files:")
    print("- portfolio_performance.png")
    print("- risk_metrics.png") 
    print("- portfolio_weights.png")
    print("- turnover_analysis.png")
    print("- performance_summary.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
