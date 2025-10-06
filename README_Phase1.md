# FinRL Phase 1: Single Stock Deep Dive - RELIANCE

## 🎯 Overview

This is Phase 1 of the FinRL implementation, focusing on single stock trading using RELIANCE data. The implementation leverages your rich dataset with 318 features to create a sophisticated trading agent.

## 📊 Dataset Features

### Core Features (9)
- **Price Data**: open, high, low, close, volume, vwap, value_traded, total_trades
- **Technical Indicators**: dma_50, dma_200, rsi_14, dma_cross, dma_distance

### Fundamental Metrics (222)
- **Valuation**: P/E ratios, P/B ratios, P/S ratios, PEG ratios
- **Profitability**: ROE, ROA, ROI, operating margins, gross margins
- **Financial Health**: Debt ratios, current ratios, quick ratios
- **Growth**: Revenue growth, EPS growth, free cash flow growth
- **Balance Sheet**: Cash, receivables, inventory, assets, liabilities

### Advanced Features (42)
- **Lag Features**: Historical price/volume patterns (1, 2, 3, 5, 10, 20 days)
- **Rolling Statistics**: Moving averages, standard deviations, momentum
- **Sentiment Data**: Reddit sentiment, news sentiment, social media signals
- **Corporate Actions**: Dividends, bonuses, corporate events

## 🏗️ Architecture

### 1. Data Loader (`data_loader.py`)
- Loads and preprocesses financial data
- Handles feature engineering and normalization
- Creates train/test splits
- Manages feature selection and importance ranking

### 2. Trading Environment (`trading_environment.py`)
- Custom FinRL environment for single stock trading
- Rich state space with 50+ features
- Realistic transaction costs and constraints
- Comprehensive reward function with risk adjustment

### 3. Training Pipeline (`training_pipeline.py`)
- PPO agent training with hyperparameter tuning
- Model evaluation and backtesting
- Performance metrics calculation
- Results visualization and saving

### 4. Main Execution (`run_phase1.py`)
- Orchestrates the entire training process
- Provides comprehensive logging and progress tracking
- Generates detailed results and comparisons

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Phase 1
```bash
python run_phase1.py
```

## 📈 Expected Output

### Training Process
1. **Data Loading**: Load RELIANCE data with 318 features
2. **Preprocessing**: Feature engineering and normalization
3. **Training**: PPO agent training (20,000 timesteps)
4. **Evaluation**: Model evaluation on test data
5. **Backtesting**: Strategy backtesting and performance analysis
6. **Results**: Plots and performance metrics

### Performance Metrics
- **Total Return**: Strategy performance vs. buy & hold
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum loss from peak
- **Win Rate**: Percentage of profitable trades
- **Volatility**: Return volatility

## 🔧 Customization

### Hyperparameters
```python
# In run_phase1.py
model = pipeline.train_ppo_agent(
    total_timesteps=20000,    # Training duration
    learning_rate=3e-4,       # Learning rate
    n_steps=2048,            # Steps per update
    batch_size=64,           # Batch size
    n_epochs=10,             # Epochs per update
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE parameter
    clip_range=0.2           # PPO clip range
)
```

### State Space
```python
# In trading_environment.py
state_space = 50  # Adjust based on feature selection
```

### Reward Function
```python
# In trading_environment.py
def _calculate_reward(self, previous_portfolio_value, current_price):
    # Customize reward calculation
    # Options: return-based, risk-adjusted, drawdown-aware
```

## 📊 Results Analysis

### Generated Files
- `trained_models/ppo_RELIANCE`: Trained model
- `results/RELIANCE_portfolio.csv`: Portfolio values over time
- `results/RELIANCE_metrics.csv`: Performance metrics
- `results/RELIANCE_evaluation.csv`: Evaluation results
- `results/RELIANCE_results.png`: Visualization plots

### Key Metrics to Monitor
1. **Strategy vs. Buy & Hold**: Outperformance analysis
2. **Risk-Adjusted Returns**: Sharpe ratio and volatility
3. **Drawdown Analysis**: Maximum loss and recovery
4. **Trading Frequency**: Number of trades and costs
5. **Feature Importance**: Which features drive decisions

## 🎯 Next Steps

### Phase 1 Improvements
1. **Hyperparameter Tuning**: Grid search for optimal parameters
2. **Feature Selection**: Identify most predictive features
3. **Algorithm Comparison**: Test A2C, DDPG, SAC
4. **Ensemble Methods**: Combine multiple models

### Phase 2 Preparation
1. **Multi-Stock Environment**: Portfolio management
2. **Cross-Asset Features**: Relative strength analysis
3. **Risk Management**: Portfolio-level constraints
4. **Sector Rotation**: Industry-specific strategies

## 🔍 Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch_size or n_steps
2. **Training Instability**: Lower learning_rate
3. **Poor Performance**: Increase total_timesteps
4. **Feature Errors**: Check data preprocessing

### Performance Tips
1. **Feature Engineering**: Add derived features
2. **Reward Tuning**: Adjust reward_scaling
3. **Action Space**: Modify hmax for position sizing
4. **Transaction Costs**: Realistic cost modeling

## 📚 References

- [FinRL Documentation](https://finrl.readthedocs.io/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm](https://openai.com/blog/openai-baselines-ppo/)
- [Financial Reinforcement Learning](https://arxiv.org/abs/2011.09607)

## 🤝 Contributing

Feel free to experiment with:
- Different algorithms (A2C, DDPG, SAC, TD3)
- Alternative reward functions
- Feature engineering techniques
- Hyperparameter optimization
- Ensemble methods

---

**Happy Trading! 🚀📈**
