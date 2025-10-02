# FYP-FinAgent Training and Evaluation Pipeline Guide

This guide explains how to use the automated training and evaluation pipeline for the FYP-FinAgent project.

## 📋 Overview

The pipeline provides automated scripts to:
1. **Train** all model architectures with different feature combinations
2. **Evaluate** trained models on out-of-sample data
3. **Generate** comprehensive reports and visualizations
4. **Compare** performance across models and features

## 🏗️ Model Architectures

The pipeline trains and evaluates three different architectures:

1. **PPO with Feature Combinations (LSTM)**
   - PPO algorithm with LSTM architecture
   - Supports curriculum learning
   - File: `train_ppo_feature_combinations.py`

2. **Plain RL LSTM (REINFORCE)**
   - REINFORCE algorithm with LSTM
   - Simpler baseline model
   - File: `train_plain_rl_lstm.py`

3. **PPO Transformer**
   - PPO algorithm with Transformer architecture
   - Attention-based model
   - File: `train_ppo_transformer.py`

## 📊 Feature Combinations

Each model is trained with 4 feature combinations:

- `ohlcv` - Basic OHLCV price data only
- `ohlcv+technical` - Price data + technical indicators
- `ohlcv+technical+sentiment` - Price + technical + sentiment features
- `all` - All available features

## 🎓 Curriculum Learning

All models use **auto curriculum learning** with 3 stages:

### Stage 1: Exploration (Longer Duration)
- **PPO Feature Combinations**: 800 epochs (was 300)
- **Plain RL LSTM**: 2500 updates (was 1000)
- **PPO Transformer**: 800 epochs (was 300)
- High exploration (std=0.3-1.0)
- High entropy coefficient
- Higher learning rate
- Early stopping patience: 300-500

### Stage 2: Refinement (Longer Duration)
- **PPO Feature Combinations**: 1000 epochs (was 400)
- **Plain RL LSTM**: 2000 updates (was 800)
- **PPO Transformer**: 1200 epochs (was 500)
- Moderate exploration
- Balanced parameters
- Early stopping patience: 400-600

### Stage 3: Optimization (Longer Duration)
- **PPO Feature Combinations**: 800 epochs (was 300)
- **Plain RL LSTM**: 1500 updates (was 600)
- **PPO Transformer**: 1000 epochs (was 700)
- Low exploration
- Focused learning
- Lower learning rate
- Early stopping patience: 500-700

**Total Training Updates:** ~6000 updates per model (2.5-3x longer than before)

## 🚀 Quick Start

### Option 1: Run Full Pipeline (Training + Evaluation)

```bash
./run_full_pipeline.sh
```

This runs everything automatically:
- ✅ Trains all 12 models (3 architectures × 4 feature combinations)
- ✅ Auto curriculum learning enabled
- ✅ Wandb logging enabled
- ✅ Evaluates all trained models
- ✅ Generates reports and visualizations

**Estimated Time:** 4-12 hours (depending on hardware and early stopping)

### Option 2: Run Training Only

```bash
./train_all_models.sh
```

Trains all models without evaluation.

**Estimated Time:** 3-10 hours

### Option 3: Run Evaluation Only

```bash
./evaluate_all_models.sh
```

Evaluates all previously trained models.

**Estimated Time:** 30 minutes - 2 hours

### Option 4: Train Individual Models

Train a specific model with a specific feature combination:

```bash
# PPO Feature Combinations (LSTM)
python train_ppo_feature_combinations.py \
    --feature_combination ohlcv+technical \
    --auto_curriculum \
    --use_wandb

# Plain RL LSTM
python train_plain_rl_lstm.py \
    --feature_combination ohlcv+technical \
    --auto_curriculum \
    --use_wandb

# PPO Transformer
python train_ppo_transformer.py \
    --feature_combination ohlcv+technical \
    --auto_curriculum \
    --use_wandb
```

### Option 5: Evaluate Individual Models

```bash
# Plain RL LSTM
python eval_plain_rl_lstm.py \
    --model_path models/plain_rl_lstm/final_model.pkl \
    --feature_combination ohlcv+technical

# PPO Transformer
python eval_ppo_transformer.py \
    --model_path models/ppo_transformer/final_model.pkl \
    --feature_combination ohlcv+technical
```

## 📁 Output Structure

After running the pipeline, you'll find:

```
FYP-FinAgent/
├── models/                          # Trained model checkpoints
│   ├── ppo_feature_combinations/
│   │   ├── curriculum_stage_1_ohlcv.pkl
│   │   ├── curriculum_stage_2_ohlcv.pkl
│   │   ├── curriculum_stage_3_ohlcv.pkl
│   │   └── ...
│   ├── plain_rl_lstm/
│   └── ppo_transformer/
│
├── training_logs/                   # Training execution logs
│   └── 20241002_143000/
│       ├── ppo_feature_combinations_ohlcv.log
│       ├── plain_rl_lstm_ohlcv.log
│       └── ...
│
├── training_results/                # Training summaries
│   └── 20241002_143000/
│       └── training_summary.txt
│
├── evaluation_results/              # Evaluation outputs
│   ├── PPO_feature_combinations/
│   │   ├── ohlcv/
│   │   │   ├── evaluation_results_*.json
│   │   │   ├── performance_report_*.txt
│   │   │   └── ppo_evaluation_results_*.png
│   │   └── ...
│   ├── Plain_RL_LSTM/
│   ├── PPO_Transformer/
│   ├── comparative_analysis_*.txt
│   └── evaluation_summary_*.txt
│
├── evaluation_logs/                 # Evaluation execution logs
│   └── 20241002_183000/
│
└── wandb/                          # Weights & Biases logs
    └── run-*/
```

## 📊 Weights & Biases Integration

All training runs automatically log to wandb:

- **Training metrics**: Loss, rewards, Sharpe ratio, returns
- **Performance metrics**: Portfolio value, drawdowns, volatility
- **Hyperparameters**: Learning rate, action std, entropy coeff
- **Curriculum info**: Current stage, global step
- **Early stopping**: Best performance, patience counter

**View your runs:**
1. Visit https://wandb.ai
2. Check your project dashboard
3. Compare runs across models and features

## 🎯 What Gets Logged

### During Training (to wandb):
- `train/raw_reward` - Raw rewards before normalization
- `train/sharpe_ratio` - **Fixed!** Now uses raw rewards
- `train/mean_return` - Mean cumulative return
- `train/portfolio_value` - Portfolio value evolution
- `train/total_loss` - Combined loss
- `train/policy_loss` - Policy gradient loss
- `train/value_loss` - Value function loss
- `train/entropy_loss` - Entropy regularization
- `curriculum/stage` - Current curriculum stage
- `early_stopping/best_performance` - Best Sharpe ratio seen
- `early_stopping/patience_counter` - Steps without improvement

### During Evaluation (to files):
- **JSON**: Detailed metrics for programmatic analysis
- **TXT**: Human-readable performance reports
- **PNG**: Visualizations (6 subplots)
  - Portfolio value evolution
  - Return distribution
  - Sharpe ratio distribution
  - Max drawdown distribution
  - Feature category breakdown
  - Risk-return scatter plot

## 🔧 Configuration

### Training Configuration

Edit the script variables or use command-line arguments:

```bash
# Training dates (in-sample)
--train_start_date "2024-06-06"
--train_end_date "2025-03-06"

# Evaluation dates (out-of-sample)
--eval_start_date "2025-03-07"
--eval_end_date "2025-06-06"

# Model hyperparameters
--learning_rate 3e-4
--action_std 0.3
--entropy_coeff 0.02

# Curriculum learning
--auto_curriculum          # Enable curriculum
--start_stage 1           # Start from stage 1 (default)
```

### Evaluation Configuration

```bash
--num_episodes 20         # Number of evaluation episodes
--eval_seed 123          # Random seed for reproducibility
--results_dir ./evaluation_results
```

## 🐛 Troubleshooting

### Training Issues

**Issue: NaN values during training**
- Solution: Check data preprocessing, reduce learning rate
- The code has automatic NaN detection and recovery

**Issue: Early stopping triggers too quickly**
- Solution: Increase `early_stopping_patience` in curriculum config
- The updated configs already have higher patience values

**Issue: Out of memory**
- Solution: Reduce batch size or number of environments
- Use CPU instead of GPU for large models

### Evaluation Issues

**Issue: Model file not found**
- Solution: Check that training completed successfully
- Look in `models/[model_type]/` for saved checkpoints

**Issue: Feature mismatch**
- Solution: Ensure evaluation uses same feature combination as training
- The scripts handle this automatically if you use the shell scripts

## 📈 Interpreting Results

### Key Metrics

1. **Sharpe Ratio** (Primary metric)
   - Risk-adjusted returns
   - Higher is better
   - > 1.0 is good, > 2.0 is excellent

2. **Mean Return**
   - Average portfolio return across episodes
   - Positive means profitable

3. **Max Drawdown**
   - Largest peak-to-trough decline
   - Lower magnitude is better
   - Indicates risk management

4. **Volatility**
   - Return standard deviation
   - Lower is more stable

5. **Success Rate**
   - Percentage of profitable episodes
   - Higher indicates consistency

### Comparing Models

Look at the comparative analysis report:
- Best overall Sharpe ratio
- Most consistent performance (lowest std)
- Best risk-adjusted returns
- Feature combination effectiveness

## 🔄 Re-running Training

To retrain a model:

1. Delete or move old model files:
   ```bash
   rm -rf models/ppo_feature_combinations/
   ```

2. Run training again:
   ```bash
   ./train_all_models.sh
   # or
   python train_ppo_feature_combinations.py --feature_combination ohlcv --auto_curriculum --use_wandb
   ```

## 💡 Best Practices

1. **Always use auto curriculum** for better training
2. **Enable wandb logging** to track experiments
3. **Run full pipeline** for fair comparisons
4. **Save training logs** for debugging
5. **Compare multiple runs** to assess stability
6. **Check early stopping logs** to ensure proper convergence
7. **Monitor Sharpe ratio** as primary performance metric

## 🎓 Understanding the Updates

### What Changed from Original?

1. **Longer Training**:
   - Stages increased by 2.5-3x
   - More time for convergence
   - Better final performance

2. **Fixed Sharpe Ratio Bug**:
   - Now uses raw rewards instead of normalized
   - Actual performance tracking
   - Early stopping works correctly

3. **Improved Feature Control**:
   - CustomDataLoader prevents feature explosion
   - Exact feature specification
   - No unwanted features

4. **Better Logging**:
   - Raw vs normalized rewards tracked
   - Performance metrics from raw data
   - Comprehensive wandb integration

## 📞 Support

If you encounter issues:
1. Check the log files in `training_logs/` or `evaluation_logs/`
2. Verify all Python dependencies are installed
3. Ensure wandb is configured: `wandb login`
4. Check GPU/CPU compatibility settings

## 🎉 Success Indicators

You'll know everything worked when:
- ✅ All 12 models trained without errors
- ✅ Wandb shows meaningful Sharpe ratios (not stuck at 0)
- ✅ Early stopping triggered based on performance
- ✅ Evaluation reports show diverse results
- ✅ Visualizations display clear trends
- ✅ Models saved in correct directories

Happy training! 🚀
