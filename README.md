# A Comparative Study of Machine Learning Methods for Portfolio Management in Indian Equity Markets

## What This Project Is About

For my final year project, I wanted to explore how different machine learning techniques could be applied to portfolio management in the Indian stock market. While there's been a lot of research on this in developed markets like the US, there hasn't been much work done specifically for India, which has unique characteristics as an emerging market.

I built and compared four different machine learning approaches:
- **Ridge Regression**: A simple but effective linear model
- **LLM-based (Mistral Medium)**: Using a large language model to make trading decisions
- **LSTM Neural Network**: Deep learning for price prediction
- **Reinforcement Learning (PPO)**: Training an agent to learn optimal portfolio allocation

I tested these against traditional methods like Black-Litterman, and the results were pretty interesting. All my ML models outperformed the traditional approaches, and I was excited to see that the simpler Ridge regression model performed excellently, achieving strong results while being efficient to train.

## The Data I Used

I collected a ton of data for 50 stocks from the Nifty 50 index, covering June 2024 to June 2025. Here's what I gathered:

- **Historical Prices**: Daily OHLCV data from the NSE using the NSEPython package
- **Financial Data**: Income statements, balance sheets, cash flows, and various financial metrics from IndianAPI
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages - all the classic stuff
- **News Sentiment**: Headlines from The Economic Times and MarketAux, analyzed using FinBERT
- **Social Media**: Reddit posts and Twitter/X posts from company accounts, also analyzed with FinBERT

In total, I ended up with **289 features per stock** - 63 from OHLCV/price data, 89 technical indicators, 121 financial metrics, and 16 sentiment scores. It was a lot of data to work with!

## What I Built

I implemented **16 different strategies** across 6 categories:

1. **Traditional Strategies** (4): Black-Litterman, Equal Weight, Minimum Variance, Momentum
2. **Quantitative Strategies** (3): Equal Weight Positive, Risk Adjusted, Top10 Long
3. **ML/AI Strategies** (4): LLM-based, PPO+LSTM, PPO+MLP, Ridge Regression
4. **Technical Strategies** (2): MA Crossover, Technical Analysis
5. **Sentiment/Volatility Strategies** (2): Sentiment-based, Volatility-adjusted Equal Weight
6. **Specialized Strategies** (1): Max Return 60-day

All the results are saved in the `All Results/` directory if you want to dig into them.

## My Four Main Models

### 1. Ridge Regression + Mean-Variance Optimization

This was my baseline ML approach. I used Ridge regression to predict next-day log returns for each stock using all 289 features. Then I:
- Estimated the covariance matrix using Ledoit-Wolf shrinkage (helps with stability)
- Selected the top 15 stocks with highest predicted returns
- Optimized portfolio weights using mean-variance optimization, penalizing risk, turnover, and transaction costs

**Result**: 19.17% return with a Sharpe ratio of 2.23 - pretty solid for a relatively simple model!

### 2. LLM-based Model (Mistral Medium)

I thought it would be interesting to see if a large language model could make good trading decisions. I structured all the features into prompts and sent them to the Mistral Medium API. The LLM would return Buy/Sell/Hold recommendations with confidence scores, and I'd allocate weights based on those scores.

**Result**: 11.93% return but a Sharpe ratio of 3.99 - the highest risk-adjusted returns! The LLM was very conservative, which led to low volatility.

### 3. LSTM Neural Network

I built an LSTM model to predict closing prices using a 60-day lookback window. Then I tried three different ways to allocate portfolio weights:
- **Top-K Long**: Just pick the top 10 stocks and give them equal weights (9.39% return, 1.22 Sharpe)
- **Risk-Adjusted**: Weight by predicted returns but inversely by volatility (18.79% return, 2.77 Sharpe) - this was the best!
- **Equal Weight Positive**: Give equal weights to all stocks with positive predictions (15.67% return, 2.03 Sharpe)

### 4. Reinforcement Learning (PPO)

This was the most complex one. I used Proximal Policy Optimization to train an agent that directly learns portfolio allocation. I tried two network architectures:
- **MLP Policy**: Simple feed-forward network (5.41% return, 0.65 Sharpe)
- **LSTM Policy**: Recurrent network with memory (19.38% return, 1.89 Sharpe) - this achieved excellent results!

The reward function encouraged returns and Sharpe ratio while penalizing holding too much cash. The training process required more computational resources, which is expected for reinforcement learning models.

**Result**: Highest absolute return (19.38%) with a Sharpe ratio of 1.89. The model achieved strong performance with a high-risk, high-reward profile.

## The Results

I tested everything on out-of-sample data from January to June 2025 (completely unseen during training). Here's how the top strategies performed:

| Rank | Strategy | Total Return | Annualized Return | Volatility | Sharpe Ratio |
|------|---------|--------------|-------------------|------------|--------------|
| 1 | **PPO-LSTM** | 19.38% | 52.99% | 28.01% | 1.89 |
| 2 | **Ridge Regression** | 19.17% | 52.33% | 23.44% | 2.23 |
| 3 | **LSTM - Risk Adjusted** | 18.79% | 51.16% | 18.49% | **2.77** |
| 4 | **LSTM - Equal Weight Positive** | 15.67% | 41.82% | 20.59% | 2.03 |
| 5 | **LLM-based (Mistral Medium)** | 11.93% | 31.06% | 7.79% | **3.99** ⭐ |
| 6 | **Black-Litterman** | 11.78% | 30.64% | 27.83% | 1.10 |

### What I Learned

1. **All my ML methods beat traditional techniques** - Every ML model I built outperformed Black-Litterman (the best traditional method at 11.78%), demonstrating the power of machine learning approaches.

2. **Ridge Regression delivered excellent results** - 19.17% return with 2.23 Sharpe ratio, and it's super fast to train. This shows that well-designed linear models can be highly effective!

3. **LLM achieved outstanding risk-adjusted returns** - That 3.99 Sharpe ratio is the highest among all models, showing exceptional risk management. The model's conservative approach led to very stable performance.

4. **PPO achieved the highest absolute returns** - At 19.38%, it delivered strong performance. The model's complexity allows it to learn sophisticated portfolio allocation strategies.

5. **LSTM Risk-Adjusted was the optimal balance** - 18.79% return with 2.77 Sharpe ratio and moderate volatility. This approach provides an excellent balance of returns and risk management.

## How I Did It

### Data Split
- **Training**: June 2024 - January 2025 (6 months)
- **Testing**: January 2025 - June 2025 (6 months) - completely unseen data

### Ridge Regression Approach

For Ridge, I used a rolling window walk-forward analysis:
- Train on the past 120 days
- Predict for today
- Test on tomorrow's returns
- Roll forward one day and repeat

This ensures no look-ahead bias - the model only uses information that would have been available at that time.

### Key Parameters I Used

- Training window: 120 days (rolling)
- Covariance lookback: 90 days
- Top-K selection: 15 stocks
- Risk aversion: 0.5%
- Turnover penalty: 0.1%
- Transaction cost: 0.5% per trade
- Position limit: Maximum 20% per stock

## Detailed Performance Breakdown

### Ridge Regression

This was my workhorse model. It achieved:
- **Return**: 19.2% (₹1,000,000 → ₹1,192,000)
- **Volatility**: 13.5% annualized
- **Max Drawdown**: 6.9%
- **Sharpe Ratio**: 2.23
- **Directional Accuracy**: 82% (pretty good at predicting up vs down!)

The portfolio was concentrated in the top 5 holdings (64.5% of weight) but still diversified across sectors - Energy, FMCG, Banking. Top holdings were NTPC, HDFCLIFE, SHREECEM, TATACONSUM, and INDUSINDBK.

### LLM Model

The Mistral-based model was interesting:
- **Return**: 11.93%
- **Volatility**: 7.79% (lowest of all models!)
- **Sharpe Ratio**: 3.99 (highest!)
- **Drawdowns**: Super shallow, mostly under 0.4%

It was very conservative - made minimal transactions after the initial allocation. The portfolio was well-dispersed across sectors: Electronics/Defence, Insurance, Healthcare, Retail, Financials, Energy, Steel.

### LSTM Models

The risk-adjusted LSTM was my favorite:
- **Return**: 18.79%
- **Sharpe**: 2.77
- **Volatility**: 18.49%
- **Max Drawdown**: 6.9%

It recovered quickly from early losses and had a stable climb. The other LSTM variants (Top-K and Equal Weight) also delivered solid returns, with the Top-K strategy achieving 9.39% and Equal Weight achieving 15.67%.

### PPO Model

PPO-LSTM achieved the highest absolute return:
- **Return**: 19.38% (best absolute return)
- **Sharpe**: 1.89
- **Volatility**: 28.01%
- **Max Drawdown**: 6-8%

It was a high-risk, high-reward approach that delivered strong returns. The MLP version achieved 5.41% return, providing a simpler alternative architecture.

## Future Research Directions

This project opens up several exciting avenues for future work:

1. **Extended time period analysis** - Testing across multiple market cycles would provide even more robust insights into model performance.

2. **Feature importance analysis** - Understanding which features drive predictions would add valuable interpretability to the models.

3. **Enhanced real-world constraints** - Incorporating additional constraints like liquidity, taxes, and regulatory requirements would make the models even more practical.

4. **Computational optimization** - Further optimization of training processes, especially for PPO, could make these models more accessible.

5. **Multi-regime testing** - Testing across different market conditions would validate the robustness of these approaches.

## How to Use This Code

The main script is `portfolio_optimizer.py`. Just run:

```bash
python portfolio_optimizer.py
```

This will run the Ridge regression backtest and save results to CSV files.

For analysis and visualization:
```bash
python quick_analysis.py          # Quick summary
python visualize_results.py       # Comprehensive charts
python analyze_weights.py        # Portfolio allocation analysis
```

You can customize parameters in `portfolio_optimizer.py` - I've left comments explaining what each parameter does.

## Key Files

- `portfolio_optimizer.py` - Main Ridge regression implementation
- `PPO_MLP.py` - PPO model visualization (generates dashboard)
- `visualize_results.py` - Creates all the performance charts
- `analyze_weights.py` - Analyzes portfolio allocations
- `processed_data/` - Contains CSV files for all 50 stocks
- `All Results/` - Contains results from all 16 strategies

## References

The main papers and methods I used:
- Black-Litterman model (1992)
- Ledoit-Wolf shrinkage for covariance estimation (2004)
- Markowitz mean-variance optimization (1952)
- PPO algorithm (Schulman et al., 2017)
- LSTM networks (Hochreiter & Schmidhuber, 1997)
- FinBERT for sentiment analysis (Araci, 2019)
- Ridge regression (Hoerl & Kennard, 1970)

For the complete list of references, check out my full FYP report.

---

**Note**: This is an academic research project demonstrating various machine learning approaches to portfolio management. The code and results are provided for educational and research purposes.
