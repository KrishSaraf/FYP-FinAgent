# 🚀 Quick Start Guide - FYP-FinAgent

## One-Command Training & Evaluation

```bash
./run_full_pipeline.sh
```

This single command:
- ✅ Trains 12 models (3 architectures × 4 feature sets)
- ✅ Uses curriculum learning (3 stages per model)
- ✅ Logs everything to wandb
- ✅ Evaluates all models on test data
- ✅ Generates reports and visualizations

**Time:** 4-12 hours | **Output:** `evaluation_results/`

---

## Alternative: Step-by-Step

### 1️⃣ Train All Models (3-10 hours)
```bash
./train_all_models.sh
```

### 2️⃣ Evaluate All Models (30 min - 2 hours)
```bash
./evaluate_all_models.sh
```

---

## Individual Model Training

### PPO Feature Combinations (LSTM)
```bash
python train_ppo_feature_combinations.py \
    --feature_combination ohlcv+technical \
    --auto_curriculum \
    --use_wandb
```

### Plain RL LSTM
```bash
python train_plain_rl_lstm.py \
    --feature_combination ohlcv+technical \
    --auto_curriculum \
    --use_wandb
```

### PPO Transformer
```bash
python train_ppo_transformer.py \
    --feature_combination ohlcv+technical \
    --auto_curriculum \
    --use_wandb
```

---

## Feature Combinations

Choose from:
- `ohlcv` - Basic price data only
- `ohlcv+technical` - Price + technical indicators ⭐ Recommended
- `ohlcv+technical+sentiment` - Price + technical + sentiment
- `all` - All features

---

## What's New? ✨

### Longer Training (2.5-3x)
- Stage 1: 800-2500 updates (was 300-1000)
- Stage 2: 1000-2000 updates (was 400-800)
- Stage 3: 800-1500 updates (was 300-700)

### Fixed Bugs 🐛
- ✅ Sharpe ratio now uses raw rewards (was stuck at ~0)
- ✅ Early stopping actually works
- ✅ Feature control prevents unwanted expansion

### Better Tracking 📊
- Real-time wandb dashboard
- Raw vs normalized rewards logged
- Performance metrics from unbiased data

---

## Expected Results

After running, check:

📁 **Models:** `models/[architecture]/`
- Checkpoint files (.pkl)
- Curriculum stage saves

📁 **Results:** `evaluation_results/[architecture]/`
- JSON metrics
- Performance reports (.txt)
- Visualizations (.png)

📊 **Wandb:** https://wandb.ai
- Training curves
- Sharpe ratio evolution
- Comparative metrics

---

## Curriculum Stages

All models go through 3 stages automatically:

**Stage 1: Exploration**
- High exploration (std=0.3-1.0)
- Learning from scratch
- Patience: 300-500 updates

**Stage 2: Refinement**
- Moderate exploration (std=0.2-0.7)
- Improving performance
- Patience: 400-600 updates

**Stage 3: Optimization**
- Low exploration (std=0.1-0.5)
- Final polishing
- Patience: 500-700 updates

Early stopping triggers if no improvement after patience threshold.

---

## Key Metrics

**Primary:** Sharpe Ratio
- Risk-adjusted returns
- Higher is better
- >1.0 good, >2.0 excellent

**Secondary:**
- Mean Return (profitability)
- Max Drawdown (risk)
- Volatility (stability)
- Success Rate (consistency)

---

## Troubleshooting

**"Model not found"**
→ Training didn't complete. Check `training_logs/`

**"NaN detected"**
→ Automatic recovery kicks in. Check if it persists.

**"Out of memory"**
→ Reduce batch size or use CPU

**"Wandb not configured"**
→ Run `wandb login` first

---

## Next Steps

1. **During Training:** Monitor wandb dashboard
2. **After Training:** Review `training_results/training_summary.txt`
3. **After Evaluation:** Check `evaluation_results/comparative_analysis_*.txt`
4. **Compare Models:** Open evaluation JSON files
5. **Visual Analysis:** View PNG charts in results folders

---

## Need Help?

📖 Full documentation: `PIPELINE_GUIDE.md`
📁 Check logs: `training_logs/` or `evaluation_logs/`
🌐 Wandb issues: https://docs.wandb.ai

---

**Good luck with your FYP! 🎓📈**
