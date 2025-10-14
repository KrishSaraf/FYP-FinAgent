"""
Portfolio Weights Visualization for Ridge Model Results
Creates comprehensive visualizations of daily portfolio weights for 45 stocks
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

def load_and_prepare_data():
    """Load and prepare portfolio weights data"""
    print("Loading portfolio weights data...")
    
    # Load weights data
    weights_df = pd.read_csv('weights_daily.csv', index_col=0, parse_dates=True)
    
    # Load portfolio values for context
    portfolio_values = pd.read_csv('portfolio_values_daily.csv', index_col=0, parse_dates=True)
    
    print(f"Loaded {len(weights_df)} days of data for {len(weights_df.columns)} stocks")
    print(f"Date range: {weights_df.index[0]} to {weights_df.index[-1]}")
    
    return weights_df, portfolio_values

def create_weights_heatmap(weights_df, top_n=20):
    """Create heatmap of portfolio weights over time"""
    print("Creating weights heatmap...")
    
    # Calculate average weights and select top N stocks
    avg_weights = weights_df.mean().sort_values(ascending=False)
    top_stocks = avg_weights.head(top_n).index.tolist()
    
    # Create heatmap data
    heatmap_data = weights_df[top_stocks].T
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create heatmap
    im = ax.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto', 
                   vmin=0, vmax=heatmap_data.values.max())
    
    # Set labels
    ax.set_xticks(range(0, len(heatmap_data.columns), max(1, len(heatmap_data.columns)//10)))
    ax.set_xticklabels([heatmap_data.columns[i].strftime('%m/%d') 
                       for i in range(0, len(heatmap_data.columns), max(1, len(heatmap_data.columns)//10))])
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Portfolio Weight', rotation=270, labelpad=20)
    
    # Formatting
    ax.set_title(f'Portfolio Weights Heatmap - Top {top_n} Stocks\nRidge Model Results', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Ticker', fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('portfolio_weights_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return top_stocks

def create_weights_evolution(weights_df, top_stocks):
    """Create line plot showing evolution of top stock weights"""
    print("Creating weights evolution plot...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot top 10 stocks
    for i, stock in enumerate(top_stocks[:10]):
        ax.plot(weights_df.index, weights_df[stock], 
               label=stock, linewidth=2, alpha=0.8)
    
    # Formatting
    ax.set_title('Portfolio Weights Evolution - Top 10 Stocks\nRidge Model Results', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Weight', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('portfolio_weights_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_weights_distribution(weights_df):
    """Create distribution analysis of portfolio weights"""
    print("Creating weights distribution analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Portfolio Weights Distribution Analysis\nRidge Model Results', 
                fontsize=16, fontweight='bold')
    
    # 1. Average weights by stock (top 20)
    avg_weights = weights_df.mean().sort_values(ascending=False)
    top_20 = avg_weights.head(20)
    
    axes[0, 0].barh(range(len(top_20)), top_20.values, color='skyblue', alpha=0.7)
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20.index)
    axes[0, 0].set_xlabel('Average Portfolio Weight')
    axes[0, 0].set_title('Average Weights - Top 20 Stocks')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Weight concentration over time
    # Calculate Herfindahl index (concentration measure)
    concentration = (weights_df ** 2).sum(axis=1)
    axes[0, 1].plot(weights_df.index, concentration, color='red', linewidth=2)
    axes[0, 1].set_title('Portfolio Concentration Over Time\n(Herfindahl Index)')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Concentration Index')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Number of active positions over time
    # Count stocks with weight > 0.001 (0.1%)
    active_positions = (weights_df > 0.001).sum(axis=1)
    axes[1, 0].plot(weights_df.index, active_positions, color='green', linewidth=2)
    axes[1, 0].set_title('Number of Active Positions Over Time\n(Weight > 0.1%)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Number of Active Positions')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Weight distribution histogram
    all_weights = weights_df.values.flatten()
    # Filter out very small weights for better visualization
    significant_weights = all_weights[all_weights > 0.001]
    
    axes[1, 1].hist(significant_weights, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Distribution of Significant Weights\n(Weight > 0.1%)')
    axes[1, 1].set_xlabel('Portfolio Weight')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_weights_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_top_holdings_analysis(weights_df, portfolio_values):
    """Create detailed analysis of top holdings"""
    print("Creating top holdings analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Top Holdings Analysis\nRidge Model Results', 
                fontsize=16, fontweight='bold')
    
    # Calculate average weights
    avg_weights = weights_df.mean().sort_values(ascending=False)
    top_10 = avg_weights.head(10)
    
    # 1. Top 10 holdings pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_10)))
    wedges, texts, autotexts = axes[0, 0].pie(top_10.values, labels=top_10.index, 
                                             autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('Top 10 Holdings Distribution\n(Average Weights)')
    
    # 2. Top 5 holdings over time
    top_5 = top_10.head(5).index
    for stock in top_5:
        axes[0, 1].plot(weights_df.index, weights_df[stock], 
                       label=stock, linewidth=2, alpha=0.8)
    
    axes[0, 1].set_title('Top 5 Holdings Evolution')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Portfolio Weight')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Portfolio value vs top holdings
    portfolio_value_norm = portfolio_values.iloc[:, 0] / portfolio_values.iloc[0, 0]
    axes[1, 0].plot(portfolio_values.index, portfolio_value_norm, 
                   label='Portfolio Value', linewidth=3, color='black')
    
    # Add top 3 holdings
    for i, stock in enumerate(top_5[:3]):
        stock_value = weights_df[stock] * portfolio_values.iloc[:, 0]
        stock_value_norm = stock_value / portfolio_values.iloc[0, 0]
        axes[1, 0].plot(weights_df.index, stock_value_norm, 
                       label=f'{stock} Value', linewidth=2, alpha=0.7)
    
    axes[1, 0].set_title('Portfolio Value vs Top Holdings Value')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Normalized Value (Base = 1.0)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Weight volatility analysis
    weight_volatility = weights_df.std().sort_values(ascending=False)
    top_10_vol = weight_volatility.head(10)
    
    axes[1, 1].barh(range(len(top_10_vol)), top_10_vol.values, color='orange', alpha=0.7)
    axes[1, 1].set_yticks(range(len(top_10_vol)))
    axes[1, 1].set_yticklabels(top_10_vol.index)
    axes[1, 1].set_xlabel('Weight Volatility (Std Dev)')
    axes[1, 1].set_title('Top 10 Most Volatile Weights')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('top_holdings_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(weights_df):
    """Create summary statistics table"""
    print("Creating summary statistics...")
    
    # Calculate various statistics
    stats = {
        'Average Weight': weights_df.mean().sort_values(ascending=False),
        'Max Weight': weights_df.max().sort_values(ascending=False),
        'Min Weight': weights_df.min().sort_values(ascending=False),
        'Weight Volatility': weights_df.std().sort_values(ascending=False),
        'Days Active': (weights_df > 0.001).sum().sort_values(ascending=False)
    }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(stats)
    
    # Display top 15 stocks
    print("\n" + "="*80)
    print("PORTFOLIO WEIGHTS SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.head(15).round(4))
    
    # Save to CSV
    summary_df.to_csv('portfolio_weights_summary.csv')
    print(f"\nSummary statistics saved to 'portfolio_weights_summary.csv'")
    
    return summary_df

def main():
    """Main function to create all visualizations"""
    print("🚀 Creating Portfolio Weights Visualizations for Ridge Model")
    print("="*60)
    
    # Load data
    weights_df, portfolio_values = load_and_prepare_data()
    
    # Create visualizations
    top_stocks = create_weights_heatmap(weights_df, top_n=20)
    create_weights_evolution(weights_df, top_stocks)
    create_weights_distribution(weights_df)
    create_top_holdings_analysis(weights_df, portfolio_values)
    
    # Create summary statistics
    summary_df = create_summary_statistics(weights_df)
    
    print("\n✅ All visualizations created successfully!")
    print("\nGenerated files:")
    print("- portfolio_weights_heatmap.png")
    print("- portfolio_weights_evolution.png") 
    print("- portfolio_weights_distribution.png")
    print("- top_holdings_analysis.png")
    print("- portfolio_weights_summary.csv")

if __name__ == "__main__":
    main()
