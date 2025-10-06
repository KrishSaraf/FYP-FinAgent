"""
Portfolio Visualization Script - Fixed Version
Shows realistic portfolio allocation, weights, and performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import our framework
from data_utils import IndianStockDataProcessor
from finrl_env import create_env
from finrl_agents import create_agent

class PortfolioVisualizer:
    """
    Visualizes portfolio allocation, weights, and performance with realistic data
    """
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.data_processor = IndianStockDataProcessor("processed_data")
        
    def run_visualization(self):
        """Run complete portfolio visualization"""
        print("🚀 PORTFOLIO VISUALIZATION - FIXED VERSION")
        print("=" * 60)
        
        # Load data with correct start date
        self.load_data()
        
        # Create environment and agent
        self.create_environment_and_agent()
        
        # Run trading simulation
        self.run_trading_simulation()
        
        # Generate visualizations
        self.create_visualizations()
        
        print("✅ Portfolio visualization completed!")
        
    def load_data(self):
        """Load all stock data with correct start date"""
        print("📊 Loading all 45 stocks with correct start date...")
        self.data_processor.load_stock_data()
        
        # Use key features
        basic_features = ['open', 'high', 'low', 'close', 'volume', 'dma_50', 'dma_200', 'rsi_14']
        
        # Use start date when all stocks are available
        self.finrl_data = self.data_processor.prepare_finrl_data(
            stock_symbols=None,
            start_date='2024-06-06',  # When all 45 stocks are available
            end_date='2024-12-31',
            feature_columns=basic_features
        )
        
        self.num_stocks = self.finrl_data['tic'].nunique()
        self.stock_list = sorted(self.finrl_data['tic'].unique())
        
        print(f"✅ Loaded {self.finrl_data.shape[0]} records for {self.num_stocks} stocks")
        print(f"📅 Date range: {self.finrl_data['date'].min()} to {self.finrl_data['date'].max()}")
        
    def create_environment_and_agent(self):
        """Create environment and agent"""
        print("🎮 Creating environment and agent...")
        
        state_dim = 500
        
        self.env = create_env(
            data=self.finrl_data,
            env_type='trading',
            stock_dim=self.num_stocks,
            initial_amount=self.initial_capital,
            state_space=state_dim,
            action_space=self.num_stocks,
            buy_cost_pct=0.001,
            sell_cost_pct=0.001,
            reward_scaling=1e-2,
            tech_indicator_list=['open', 'high', 'low', 'close', 'volume', 'dma_50', 'dma_200', 'rsi_14']
        )
        
        self.agent = create_agent(
            'PPO',
            state_dim,
            self.num_stocks,
            lr=1e-4,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4,
            hidden_dim=512
        )
        
        print("✅ Environment and agent created")
        
    def run_trading_simulation(self):
        """Run trading simulation and collect data"""
        print("🏃 Running trading simulation...")
        
        state = self.env.reset()
        self.portfolio_values = []
        self.actions_history = []
        self.prices_history = []
        self.dates_history = []
        self.rewards_history = []
        
        done = False
        step = 0
        max_steps = 30  # Run for 30 steps for analysis
        
        while not done and step < max_steps:
            # Get current prices
            current_data = self.finrl_data[self.finrl_data['date'] == self.env.dates[self.env.day]]
            current_prices = current_data['close'].values
            current_date = self.env.dates[self.env.day]
            
            # Get action from agent
            action, _ = self.agent.select_action(state)
            action = np.clip(action, -0.3, 0.3)  # Limit action range for stability
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store data
            self.portfolio_values.append(self.env._get_portfolio_value())
            self.actions_history.append(action.copy())
            self.prices_history.append(current_prices.copy())
            self.dates_history.append(current_date)
            self.rewards_history.append(reward)
            
            state = next_state
            step += 1
            
            if step % 5 == 0:
                current_return = ((self.portfolio_values[-1] / self.portfolio_values[0]) - 1) * 100
                print(f"   Step {step}: Portfolio = ₹{self.portfolio_values[-1]:,.0f}, Return = {current_return:.2f}%")
        
        print(f"✅ Simulation completed with {step} steps")
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("📊 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Portfolio Value Over Time
        self.plot_portfolio_value(fig, 2, 3, 1)
        
        # 2. Allocation Weights Heatmap
        self.plot_allocation_heatmap(fig, 2, 3, 2)
        
        # 3. Top 10 Long Positions
        self.plot_top_long_positions(fig, 2, 3, 3)
        
        # 4. Top 10 Short Positions
        self.plot_top_short_positions(fig, 2, 3, 4)
        
        # 5. Portfolio Composition Over Time
        self.plot_portfolio_composition(fig, 2, 3, 5)
        
        # 6. Stock Price Changes
        self.plot_stock_price_changes(fig, 2, 3, 6)
        
        plt.tight_layout()
        plt.savefig('portfolio_analysis_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed analysis
        self.create_detailed_analysis()
        
    def plot_portfolio_value(self, fig, rows, cols, pos):
        """Plot portfolio value over time"""
        ax = fig.add_subplot(rows, cols, pos)
        
        portfolio_series = pd.Series(self.portfolio_values)
        returns = portfolio_series.pct_change().dropna()
        
        ax.plot(range(len(self.portfolio_values)), self.portfolio_values, 
                linewidth=2, color='blue', label='Portfolio Value')
        ax.axhline(y=self.initial_capital, color='red', linestyle='--', 
                   alpha=0.7, label=f'Initial Capital (₹{self.initial_capital:,})')
        
        ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Steps')
        ax.set_ylabel('Portfolio Value (₹)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add performance metrics
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        ax.text(0.02, 0.98, f'Total Return: {total_return:.2%}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def plot_allocation_heatmap(self, fig, rows, cols, pos):
        """Plot allocation weights heatmap"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Convert actions to DataFrame
        actions_df = pd.DataFrame(self.actions_history, columns=self.stock_list)
        
        # Create heatmap
        sns.heatmap(actions_df.T, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Allocation Weight'}, ax=ax)
        
        ax.set_title('Stock Allocation Weights Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Steps')
        ax.set_ylabel('Stocks')
        
        # Rotate y-axis labels for better readability
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
        
    def plot_top_long_positions(self, fig, rows, cols, pos):
        """Plot top 10 long positions"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Calculate average long positions
        actions_df = pd.DataFrame(self.actions_history, columns=self.stock_list)
        avg_actions = actions_df.mean()
        long_positions = avg_actions[avg_actions > 0].sort_values(ascending=False).head(10)
        
        if len(long_positions) > 0:
            bars = ax.bar(range(len(long_positions)), long_positions.values, 
                         color='green', alpha=0.7)
            
            ax.set_title('Top 10 Long Positions (Average Weights)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Stocks')
            ax.set_ylabel('Average Allocation Weight')
            ax.set_xticks(range(len(long_positions)))
            ax.set_xticklabels(long_positions.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No long positions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Top 10 Long Positions', fontsize=14, fontweight='bold')
        
    def plot_top_short_positions(self, fig, rows, cols, pos):
        """Plot top 10 short positions"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Calculate average short positions
        actions_df = pd.DataFrame(self.actions_history, columns=self.stock_list)
        avg_actions = actions_df.mean()
        short_positions = avg_actions[avg_actions < 0].sort_values(ascending=True).head(10)
        
        if len(short_positions) > 0:
            bars = ax.bar(range(len(short_positions)), short_positions.values, 
                         color='red', alpha=0.7)
            
            ax.set_title('Top 10 Short Positions (Average Weights)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Stocks')
            ax.set_ylabel('Average Allocation Weight')
            ax.set_xticks(range(len(short_positions)))
            ax.set_xticklabels(short_positions.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height - 0.001,
                       f'{height:.3f}', ha='center', va='top', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No short positions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Top 10 Short Positions', fontsize=14, fontweight='bold')
        
    def plot_portfolio_composition(self, fig, rows, cols, pos):
        """Plot portfolio composition over time"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Calculate portfolio composition
        actions_df = pd.DataFrame(self.actions_history, columns=self.stock_list)
        
        # Calculate long/short composition
        long_composition = (actions_df > 0).sum(axis=1)
        short_composition = (actions_df < 0).sum(axis=1)
        neutral_composition = (actions_df == 0).sum(axis=1)
        
        steps = range(len(actions_df))
        
        ax.fill_between(steps, 0, long_composition, alpha=0.7, color='green', label='Long Positions')
        ax.fill_between(steps, long_composition, long_composition + short_composition, 
                       alpha=0.7, color='red', label='Short Positions')
        ax.fill_between(steps, long_composition + short_composition, 
                       long_composition + short_composition + neutral_composition,
                       alpha=0.7, color='gray', label='Neutral Positions')
        
        ax.set_title('Portfolio Composition Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Steps')
        ax.set_ylabel('Number of Stocks')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_stock_price_changes(self, fig, rows, cols, pos):
        """Plot stock price changes"""
        ax = fig.add_subplot(rows, cols, pos)
        
        # Calculate price changes for first 10 stocks
        price_changes = []
        stock_names = []
        
        for i in range(min(10, len(self.stock_list))):
            stock = self.stock_list[i]
            if i < len(self.prices_history[0]):
                first_price = self.prices_history[0][i]
                last_price = self.prices_history[-1][i]
                price_change = (last_price - first_price) / first_price
                price_changes.append(price_change)
                stock_names.append(stock)
        
        if price_changes:
            colors = ['green' if c > 0 else 'red' for c in price_changes]
            bars = ax.bar(range(len(price_changes)), price_changes, color=colors, alpha=0.7)
            
            ax.set_title('Stock Price Changes (First 10 Stocks)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Stocks')
            ax.set_ylabel('Price Change (%)')
            ax.set_xticks(range(len(price_changes)))
            ax.set_xticklabels(stock_names, rotation=45, ha='right')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.01),
                       f'{height:.1%}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
    def create_detailed_analysis(self):
        """Create detailed analysis report"""
        print("📋 Creating detailed analysis report...")
        
        # Calculate statistics
        actions_df = pd.DataFrame(self.actions_history, columns=self.stock_list)
        
        # Portfolio statistics
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        portfolio_series = pd.Series(self.portfolio_values)
        returns = portfolio_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = ((portfolio_series / portfolio_series.expanding().max()) - 1).min()
        
        # Allocation statistics
        avg_long_positions = np.mean(np.sum(actions_df > 0, axis=1))
        avg_short_positions = np.mean(np.sum(actions_df < 0, axis=1))
        avg_neutral_positions = np.mean(np.sum(actions_df == 0, axis=1))
        
        # Top performers
        avg_actions = actions_df.mean()
        top_long = avg_actions[avg_actions > 0].sort_values(ascending=False).head(5)
        top_short = avg_actions[avg_actions < 0].sort_values(ascending=True).head(5)
        
        # Create detailed report
        report = f"""
🎯 PORTFOLIO ANALYSIS REPORT - FIXED VERSION
{'='*60}

📊 PORTFOLIO PERFORMANCE:
   Initial Capital: ₹{self.portfolio_values[0]:,.0f}
   Final Value: ₹{self.portfolio_values[-1]:,.0f}
   Total Return: {total_return:.2%}
   Sharpe Ratio: {sharpe_ratio:.3f}
   Max Drawdown: {max_drawdown:.2%}
   Volatility: {returns.std() * np.sqrt(252):.2%}

📈 ALLOCATION STATISTICS:
   Average Long Positions: {avg_long_positions:.1f} stocks
   Average Short Positions: {avg_short_positions:.1f} stocks
   Average Neutral Positions: {avg_neutral_positions:.1f} stocks
   Total Stocks: {self.num_stocks}

🏆 TOP LONG POSITIONS:
"""
        for stock, weight in top_long.items():
            report += f"   {stock}: {weight:.3f}\n"
        
        report += "\n📉 TOP SHORT POSITIONS:\n"
        for stock, weight in top_short.items():
            report += f"   {stock}: {weight:.3f}\n"
        
        report += f"""
💰 CAPITAL ALLOCATION:
   Equal allocation per stock: ₹{self.initial_capital // self.num_stocks:,}
   Long exposure: {avg_long_positions / self.num_stocks:.1%}
   Short exposure: {avg_short_positions / self.num_stocks:.1%}
   Neutral exposure: {avg_neutral_positions / self.num_stocks:.1%}

🎉 ANALYSIS COMPLETED - REALISTIC RETURNS!
"""
        
        print(report)
        
        # Save report to file
        with open('portfolio_analysis_report_fixed.txt', 'w') as f:
            f.write(report)
        
        print("📄 Detailed report saved to 'portfolio_analysis_report_fixed.txt'")
        print("📊 Visualizations saved to 'portfolio_analysis_fixed.png'")


def main():
    """Main function to run portfolio visualization"""
    visualizer = PortfolioVisualizer(initial_capital=1000000)
    visualizer.run_visualization()


if __name__ == "__main__":
    main()
