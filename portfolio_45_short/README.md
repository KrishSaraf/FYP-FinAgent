# 45-Stock Portfolio with Shorting - FinRL Implementation

A sophisticated reinforcement learning portfolio management system for 45-stock daily trading with shorting capabilities, built using Stable-Baselines3 and Gymnasium.

## 🎯 Overview

This implementation provides a complete FinRL-style repository for managing a 45-stock portfolio with:
- **Shorting capabilities** with realistic borrowing costs
- **Comprehensive cost model** including commissions, slippage, and borrowing fees
- **Advanced constraint enforcement** for position limits and exposure controls
- **PPO-based training** with vectorized environments and normalization
- **Detailed backtesting** with comprehensive performance metrics

## 📊 Data Schema

The system expects a single pandas DataFrame with the following columns:

### Required Columns
```python
["date", "ticker", "open", "high", "low", "close", "volume", "vwap", 
 "value_traded", "total_trades", "dma_50", "dma_200"]
```

### Fundamental Metrics (Balance Sheet)
```python
["metric_Cash", "metric_CashEquivalents", "metric_ShortTermInvestments", 
 "metric_CashandShortTermInvestments", "metric_AccountsReceivable-TradeNet", 
 "metric_TotalReceivablesNet", "metric_TotalInventory", "metric_PrepaidExpenses", 
 "metric_OtherCurrentAssetsTotal", "metric_TotalCurrentAssets", 
 "metric_Property/Plant/EquipmentTotal-Gross", "metric_AccumulatedDepreciationTotal", 
 "metric_Property/Plant/EquipmentTotal-Net", "metric_GoodwillNet", 
 "metric_IntangiblesNet", "metric_LongTermInvestments", 
 "metric_NoteReceivable-LongTerm", "metric_OtherLongTermAssetsTotal", 
 "metric_TotalAssets", "metric_AccountsPayable", "metric_AccruedExpenses", 
 "metric_NotesPayable/ShortTermDebt", "metric_CurrentPortofLTDebt/CapitalLeases", 
 "metric_OtherCurrentliabilitiesTotal", "metric_TotalCurrentLiabilities", 
 "metric_LongTermDebt", "metric_CapitalLeaseObligations", 
 "metric_TotalLongTermDebt", "metric_TotalDebt", "metric_DeferredIncomeTax", 
 "metric_MinorityInterest", "metric_OtherLiabilitiesTotal", 
 "metric_TotalLiabilities", "metric_RetainedEarnings(AccumulatedDeficit)", 
 "metric_UnrealizedGain(Loss)", "metric_OtherEquityTotal", 
 "metric_TotalEquity", "metric_TotalLiabilitiesShareholders'Equity", 
 "metric_TangibleBookValueperShareCommonEq"]
```

### Income Statement Metrics
```python
["metric_periodLength", "metric_Revenue", "metric_TotalRevenue", 
 "metric_CostofRevenueTotal", "metric_GrossProfit", 
 "metric_Selling/General/AdminExpensesTotal", "metric_Depreciation/Amortization", 
 "metric_UnusualExpense(Income)", "metric_OtherOperatingExpensesTotal", 
 "metric_TotalOperatingExpense", "metric_OperatingIncome", 
 "metric_InterestInc(Exp)Net-Non-OpTotal", "metric_Gain(Loss)onSaleofAssets", 
 "metric_OtherNet", "metric_NetIncomeBeforeTaxes", 
 "metric_ProvisionforIncomeTaxes", "metric_NetIncomeAfterTaxes", 
 "metric_NetIncomeBeforeExtraItems", "metric_TotalExtraordinaryItems", 
 "metric_NetIncome", "metric_IncomeAvailabletoComExclExtraOrd", 
 "metric_IncomeAvailabletoComInclExtraOrd", "metric_DilutedNetIncome", 
 "metric_DilutedWeightedAverageShares", "metric_DilutedEPSExcludingExtraOrdItems", 
 "metric_DPS-CommonStockPrimaryIssue", "metric_DilutedNormalizedEPS"]
```

### Cash Flow Metrics
```python
["metric_NetIncome/StartingLine", "metric_Depreciation/Depletion", 
 "metric_Non-CashItems", "metric_ChangesinWorkingCapital", 
 "metric_CashfromOperatingActivities", "metric_CapitalExpenditures", 
 "metric_OtherInvestingCashFlowItemsTotal", "metric_CashfromInvestingActivities", 
 "metric_FinancingCashFlowItems", "metric_TotalCashDividendsPaid", 
 "metric_Issuance(Retirement)ofStockNet", "metric_Issuance(Retirement)ofDebtNet", 
 "metric_CashfromFinancingActivities", "metric_NetChangeinCash", 
 "metric_CashInterestPaid", "metric_CashTaxesPaid", "period_end_date"]
```

### Sentiment & Alternative Data
```python
["reddit_title_sentiments_mean", "reddit_title_sentiments_std", 
 "reddit_body_sentiments", "reddit_body_sentiments_std", 
 "reddit_score_mean", "reddit_score_sum", "reddit_posts_count", 
 "reddit_comments_sum", "dividend_amount", "dividend_type", 
 "news_sentiment_mean", "news_articles_count", "news_sentiment_std", 
 "news_sources"]
```

### Lag Features
```python
["open_lag_1", "open_lag_2", "open_lag_3", "open_lag_5", "open_lag_10", "open_lag_20",
 "high_lag_1", "high_lag_2", "high_lag_3", "high_lag_5", "high_lag_10", "high_lag_20",
 "low_lag_1", "low_lag_2", "low_lag_3", "low_lag_5", "low_lag_10", "low_lag_20",
 "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10", "close_lag_20",
 "volume_lag_1", "volume_lag_2", "volume_lag_3", "volume_lag_5", "volume_lag_10", "volume_lag_20",
 "dma_50_lag_1", "dma_50_lag_2", "dma_50_lag_3", "dma_50_lag_5", "dma_50_lag_10", "dma_50_lag_20",
 "dma_200_lag_1", "dma_200_lag_2", "dma_200_lag_3", "dma_200_lag_5", "dma_200_lag_10", "dma_200_lag_20"]
```

### Rolling Statistics
```python
["open_rolling_mean_5", "open_rolling_mean_20", "open_rolling_std_20", 
 "high_rolling_mean_5", "high_rolling_mean_20", "high_rolling_std_20", 
 "low_rolling_mean_5", "low_rolling_mean_20", "low_rolling_std_20", 
 "close_rolling_mean_5", "close_rolling_mean_20", "close_rolling_std_20", 
 "close_momentum_5", "close_momentum_20", "volume_rolling_mean_5", 
 "volume_rolling_mean_20", "volume_rolling_std_20", 
 "dma_50_rolling_mean_5", "dma_50_rolling_mean_20", "dma_50_rolling_std_20", 
 "dma_200_rolling_mean_5", "dma_200_rolling_mean_20", "dma_200_rolling_std_20", 
 "rsi_14", "volume_price_trend", "dma_cross", "dma_distance"]
```

## 🏗️ Architecture

### Environment: `Portfolio45ShortEnv`

The core trading environment with the following specifications:

#### Action Space
- **Type**: Continuous weights per stock
- **Shape**: `(45,)` - one weight per stock
- **Range**: `[-1, 1]` (raw weights, projected to constraints)

#### Observation Space
- **Type**: Feature panel for all 45 stocks
- **Shape**: `(45, n_features)` where n_features is the number of available features
- **Normalization**: Applied via VecNormalize

#### Constraints
- **Individual Position Limit**: `|w_i| ≤ w_max` (default: 0.10)
- **Gross Exposure Limit**: `Σ|w_i| ≤ gross_cap` (default: 1.5)
- **Net Exposure Target**: `Σw_i = target_net` (default: 1.0)

## 💰 Cost Model & PnL Equations

### Portfolio Value Calculation
```
V_t = Cash_t + Σ(Shares_i,t × Price_i,t)
```

### Daily Return
```
R_t = (V_t - V_{t-1}) / V_{t-1}
```

### Transaction Costs

#### Commission
```
Commission_i = |Quantity_i| × Price_i × commission_bps
```

#### Slippage
```
Slippage_i = |Trade_Value_i| × slippage_bps
```

#### Borrowing Costs (for short positions)
```
Borrow_Fee_i = |Short_Value_i| × borrow_rate_daily
```

#### Cash Rebate (for positive cash)
```
Cash_Rebate = Cash × rebate_rate_daily
```

### Total Costs
```
Total_Costs_t = Σ(Commission_i + Slippage_i + Borrow_Fee_i) - Cash_Rebate
```

### Net Portfolio Value
```
V_net,t = V_gross,t - Total_Costs_t
```

## 🎯 Weights Projection Algorithm

The `project_weights` function enforces portfolio constraints:

```python
def project_weights(w_raw, w_max=0.10, target_net=1.0, gross_cap=1.5):
    # Step 1: Clip individual positions
    w_clipped = np.clip(w_raw, -w_max, w_max)
    
    # Step 2: Check gross exposure constraint
    gross_exposure = np.sum(np.abs(w_clipped))
    
    if gross_exposure <= gross_cap:
        # Scale down if needed, then adjust for net target
        scale_factor = min(1.0, gross_cap / gross_exposure)
        w_scaled = w_clipped * scale_factor
        
        # Adjust for net target
        current_net = np.sum(w_scaled)
        net_adjustment = target_net - current_net
        w_final = w_scaled + net_adjustment * (w_scaled / gross_exposure)
    else:
        # Scale down and adjust
        w_final = w_clipped * (gross_cap / gross_exposure)
        # Additional net adjustment...
    
    return np.clip(w_final, -w_max, w_max)
```

## 🚀 Usage

### Installation
```bash
pip install -e .
```

### Training
```bash
python train_ppo.py
```

### Evaluation
```bash
python evaluate.py
```

### Custom Configuration
```python
from train_ppo import PortfolioTrainer

trainer = PortfolioTrainer(
    data=your_data,
    tickers=your_tickers,
    train_start_date='2023-01-01',
    train_end_date='2023-08-31',
    test_start_date='2023-09-01',
    test_end_date='2023-12-31',
    # Environment parameters
    initial_capital=1_000_000.0,
    commission_bps=1.0,
    slippage_bps=2.0,
    borrow_rate_annual=0.03,
    w_max=0.10,
    gross_cap=1.5,
    target_net=1.0,
)

trainer.train(total_timesteps=100_000)
```

## 📊 Performance Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio**: `(μ - r_f) / σ × √252`
- **Sortino Ratio**: `(μ - r_f) / σ_downside × √252`
- **Calmar Ratio**: `Annualized_Return / |Max_Drawdown|`

### Risk Metrics
- **Maximum Drawdown**: `min((V_t - Peak_t) / Peak_t)`
- **Volatility**: `σ × √252`
- **Value at Risk (VaR)**: 95th percentile of returns

### Trading Metrics
- **Turnover**: `Σ|w_{i,t} - w_{i,t-1}|`
- **Gross Exposure**: `Σ|w_i|`
- **Net Exposure**: `Σw_i`
- **Short Notional**: `Σ|w_i| × V_t` for w_i < 0

### Cost Analysis
- **Total Costs**: Commissions + Slippage + Borrowing Fees - Cash Rebate
- **Cost Ratio**: `Total_Costs / Initial_Capital`
- **Cost-Adjusted Returns**: `Raw_Returns - Daily_Costs / Portfolio_Value`

## 🔧 Key Features

### 1. Realistic Cost Modeling
- Commission costs (1 bps default)
- Market impact/slippage (2 bps default)
- Borrowing costs for short positions (3% annual default)
- Cash rebates for positive cash balances

### 2. Advanced Constraint Enforcement
- Individual position limits
- Gross exposure limits
- Net exposure targeting
- Smooth weight projections

### 3. Comprehensive Evaluation
- Multiple performance metrics
- Rolling analysis
- Trade-level analysis
- Cost attribution

### 4. Production-Ready Architecture
- Vectorized environments
- Observation normalization
- Deterministic evaluation
- Detailed logging

## 📈 Expected Performance

Based on the sophisticated cost model and constraint system:

- **Target**: Risk-adjusted returns with controlled drawdowns
- **Turnover**: Typically 20-50% daily (depending on strategy)
- **Gross Exposure**: 1.0-1.5 (leveraged strategies)
- **Net Exposure**: ~1.0 (market neutral to long bias)
- **Costs**: 2-5 bps daily (commissions + slippage + borrowing)

## 🎯 Optimization Tips

### For Better Performance
1. **Increase Training Time**: Use 100K+ timesteps
2. **Hyperparameter Tuning**: Optimize PPO parameters
3. **Feature Engineering**: Add more derived features
4. **Ensemble Methods**: Combine multiple models
5. **Risk Management**: Adjust constraint parameters

### For Lower Costs
1. **Reduce Turnover**: Increase transaction cost penalties
2. **Optimize Rebalancing**: Less frequent rebalancing
3. **Smarter Execution**: Implement TWAP/VWAP strategies
4. **Liquidity Focus**: Trade only liquid stocks

## 📚 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [FinRL Paper](https://arxiv.org/abs/2011.09607)
- [PPO Algorithm](https://openai.com/blog/openai-baselines-ppo/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for quantitative finance and reinforcement learning**
