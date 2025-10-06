# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Prerequisites

```bash
# Install required packages
pip install pandas numpy scikit-learn cvxpy scikit-covariance matplotlib seaborn
```

### 2. Run the Backtest

```bash
# Navigate to the project directory
cd /path/to/FYP-FinAgent

# Run the complete backtest
python portfolio_optimizer.py
```

**Expected Output:**
```
Loading data from processed_data directory...
Found 45 CSV files
...
==== Backtest Summary ====
Initial Capital: $1,000,000
Final Portfolio Value: $1,209,305
Total Return: 20.1%
Days: 105 | Daily Sharpe: 2.74 | MaxDD: -6.9% | Avg Turnover: 25.1%
Volatility: 17.2%
```

### 3. Generate Visualizations

```bash
# Quick analysis
python quick_analysis.py

# Comprehensive analysis
python visualize_results.py

# Portfolio weights analysis
python analyze_weights.py
```

### 4. View Results

Check the generated files:
- **CSV Files**: `weights_daily.csv`, `pnl_daily.csv`, `turnover_daily.csv`, `portfolio_values_daily.csv`
- **Charts**: `*.png` files with comprehensive visualizations

## 📊 Understanding the Results

### Key Metrics Explained

- **Total Return**: 20.1% - Portfolio grew from $1M to $1.21M
- **Sharpe Ratio**: 2.74 - Excellent risk-adjusted returns (anything >1 is good)
- **Max Drawdown**: -6.9% - Maximum loss from peak (lower is better)
- **Volatility**: 17.2% - Annualized standard deviation of returns
- **Turnover**: 25.1% - Average daily portfolio rebalancing

### Portfolio Characteristics

- **Concentrated**: Top 5 stocks average ~45% of portfolio
- **Active**: Only ~5 stocks with >1% weight on average
- **Diversified**: 45 stocks in universe, dynamic selection

## 🔧 Customization

### Modify Parameters

Edit `portfolio_optimizer.py` and change these parameters:

```python
w_df, pnl_s, turnover_s, portfolio_values_s = run_backtest(df,
    train_window=120,        # Training window (days)
    cov_lookback=90,         # Covariance lookback (days)
    topK=15,                 # Number of stocks to select
    risk_aversion=5e-3,      # Risk aversion (higher = more conservative)
    turnover_penalty=1e-3,   # Turnover penalty (higher = less trading)
    w_max=0.20,              # Maximum position size (20%)
    cost_bps_roundtrip=5.0,  # Transaction costs (5 basis points)
    min_history=150,         # Minimum history before trading
)
```

### Add New Features

1. **Add features to `non_static_columns.txt`**
2. **Ensure your CSV files contain the new columns**
3. **Run the backtest again**

### Change Universe

1. **Add/remove CSV files in `processed_data/` directory**
2. **Ensure consistent date ranges across files**
3. **Run the backtest**

## 📈 Interpreting the Charts

### Portfolio Performance
- **Blue line**: Portfolio value over time
- **Green/Red bars**: Daily returns (green = positive, red = negative)

### Risk Metrics
- **Orange line**: Rolling Sharpe ratio (higher is better)
- **Red area**: Drawdown periods (lower is better)
- **Green line**: Rolling volatility (stable is better)

### Portfolio Weights
- **Multiple lines**: Top holdings over time
- **Red line**: Portfolio concentration (top 5 stocks)

### Turnover Analysis
- **Blue line**: Daily turnover
- **Scatter plot**: Turnover vs returns correlation

## ⚠️ Common Issues

### 1. "No data loaded successfully"
- **Check**: CSV files exist in `processed_data/` directory
- **Check**: Files have correct naming format (`TICKER_aligned.csv`)

### 2. "Prediction failed"
- **Normal**: Some days may have missing data
- **Check**: Ensure sufficient historical data (150+ days)

### 3. "Optimization failed"
- **Normal**: Solver may fail on some days
- **Fallback**: System uses equal-weight portfolio

### 4. Import errors
- **Solution**: Install missing packages with `pip install package_name`

## 🎯 Next Steps

### For Researchers
1. **Read**: `README.md` for complete documentation
2. **Study**: `TECHNICAL_APPENDIX.md` for mathematical details
3. **Experiment**: Modify parameters and analyze results

### For Practitioners
1. **Validate**: Test on different time periods
2. **Optimize**: Tune parameters for your risk tolerance
3. **Monitor**: Set up real-time monitoring systems

### For Developers
1. **Extend**: Add new features or models
2. **Integrate**: Connect to live data feeds
3. **Deploy**: Set up production trading systems

## 📞 Getting Help

1. **Check logs**: Look for error messages in console output
2. **Verify data**: Ensure CSV files are properly formatted
3. **Review parameters**: Check if settings are reasonable
4. **Test incrementally**: Start with small datasets

## 🔗 Additional Resources

- **Main Documentation**: `README.md`
- **Technical Details**: `TECHNICAL_APPENDIX.md`
- **Source Code**: All `.py` files with inline documentation
- **Results**: CSV files and PNG charts

---

**Happy Trading!** 🚀📈
