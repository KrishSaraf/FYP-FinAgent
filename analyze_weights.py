"""
Portfolio Weights Analysis Script
Detailed analysis of portfolio allocation and stock selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_portfolio_weights():
    """Analyze portfolio weights and stock selection"""
    
    # Load weights data
    weights_df = pd.read_csv('weights_daily.csv', index_col=0, parse_dates=True)
    
    print("="*60)
    print("PORTFOLIO WEIGHTS ANALYSIS")
    print("="*60)
    
    # Calculate statistics
    avg_weights = weights_df.mean().sort_values(ascending=False)
    max_weights = weights_df.max().sort_values(ascending=False)
    min_weights = weights_df.min().sort_values(ascending=False)
    
    print(f"Total stocks in universe: {len(weights_df.columns)}")
    print(f"Average number of stocks with >0.1% weight: {(weights_df > 0.001).sum(axis=1).mean():.1f}")
    print(f"Average number of stocks with >1% weight: {(weights_df > 0.01).sum(axis=1).mean():.1f}")
    print(f"Average number of stocks with >5% weight: {(weights_df > 0.05).sum(axis=1).mean():.1f}")
    
    print("\n" + "="*40)
    print("TOP 15 HOLDINGS (Average Weight)")
    print("="*40)
    for i, (stock, weight) in enumerate(avg_weights.head(15).items(), 1):
        print(f"{i:2d}. {stock:15s}: {weight*100:6.2f}%")
    
    print("\n" + "="*40)
    print("STOCKS WITH HIGHEST MAXIMUM WEIGHT")
    print("="*40)
    for i, (stock, weight) in enumerate(max_weights.head(10).items(), 1):
        print(f"{i:2d}. {stock:15s}: {weight*100:6.2f}%")
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top 15 holdings bar chart
    top_15 = avg_weights.head(15)
    ax1.barh(range(len(top_15)), top_15.values * 100, color='skyblue')
    ax1.set_yticks(range(len(top_15)))
    ax1.set_yticklabels(top_15.index, fontsize=10)
    ax1.set_xlabel('Average Weight (%)')
    ax1.set_title('Top 15 Holdings (Average Weight)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Portfolio concentration over time
    top_5_concentration = weights_df[avg_weights.head(5).index].sum(axis=1) * 100
    top_10_concentration = weights_df[avg_weights.head(10).index].sum(axis=1) * 100
    
    ax2.plot(weights_df.index, top_5_concentration, label='Top 5', linewidth=2)
    ax2.plot(weights_df.index, top_10_concentration, label='Top 10', linewidth=2)
    ax2.set_title('Portfolio Concentration Over Time', fontweight='bold')
    ax2.set_ylabel('Concentration (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Number of holdings over time
    num_holdings = (weights_df > 0.001).sum(axis=1)
    ax3.plot(weights_df.index, num_holdings, linewidth=2, color='green')
    ax3.set_title('Number of Holdings Over Time (>0.1%)', fontweight='bold')
    ax3.set_ylabel('Number of Stocks')
    ax3.grid(True, alpha=0.3)
    
    # Weight distribution
    all_weights = weights_df.values.flatten()
    all_weights = all_weights[all_weights > 0.001]  # Only non-zero weights
    
    ax4.hist(all_weights * 100, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_title('Distribution of Portfolio Weights', fontweight='bold')
    ax4.set_xlabel('Weight (%)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weights_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create heatmap of top holdings over time
    top_10_stocks = avg_weights.head(10).index
    weights_subset = weights_df[top_10_stocks] * 100
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(weights_subset.T, cmap='YlOrRd', cbar_kws={'label': 'Weight (%)'})
    plt.title('Top 10 Holdings Over Time (Heatmap)', fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Stock')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('weights_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("- weights_analysis.png")
    print("- weights_heatmap.png")
    print("="*60)

if __name__ == "__main__":
    analyze_portfolio_weights()
