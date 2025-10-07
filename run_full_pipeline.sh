#!/bin/bash

################################################################################
# Full Pipeline Script for FYP-FinAgent Project
#
# This script runs the complete pipeline:
# 1. Trains all models with auto curriculum and wandb logging
# 2. Evaluates all trained models on out-of-sample data
# 3. Generates comprehensive reports and visualizations
#
# Author: AI Assistant
# Date: 2024
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
clear
echo -e "${CYAN}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              FYP-FinAgent Complete Training & Evaluation Pipeline          ║
║                                                                            ║
║  This pipeline will:                                                       ║
║    1. Train all models (PPO LSTM, Plain RL LSTM, PPO Transformer, PPO MLP)║
║    2. Use auto curriculum learning for optimal training                   ║
║    3. Log all metrics to Weights & Biases (wandb)                         ║
║    4. Evaluate all trained models on out-of-sample data                   ║
║    5. Generate comprehensive performance reports                          ║
║    6. Create visualizations and comparative analysis                      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PIPELINE_LOG="pipeline_${TIMESTAMP}.log"

# Function to print section headers
print_section() {
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Function to print status messages
print_status() {
    echo -e "${GREEN}[$(date +"%H:%M:%S")]${NC} $1"
}

# Function to print error messages
print_error() {
    echo -e "${RED}[ERROR $(date +"%H:%M:%S")]${NC} $1"
}

# Function to print warnings
print_warning() {
    echo -e "${YELLOW}[WARNING $(date +"%H:%M:%S")]${NC} $1"
}

# Start pipeline log
exec > >(tee -a "${PIPELINE_LOG}")
exec 2>&1

print_status "Pipeline started at $(date)"
print_status "Pipeline log: ${PIPELINE_LOG}"
echo ""

# Check if required scripts exist
print_section "Pre-flight Checks"

if [ ! -f "train_all_models.sh" ]; then
    print_error "train_all_models.sh not found!"
    exit 1
fi

if [ ! -f "evaluate_all_models.sh" ]; then
    print_error "evaluate_all_models.sh not found!"
    exit 1
fi

print_status "✓ Training script found"
print_status "✓ Evaluation script found"

# Check if training scripts exist
required_scripts=(
    "train_ppo_feature_combinations.py"
    "train_plain_rl_lstm.py"
    "train_ppo_transformer.py"
    "eval_plain_rl_lstm.py"
    "eval_ppo_transformer.py"
)

for script in "${required_scripts[@]}"; do
    if [ ! -f "${script}" ]; then
        print_warning "${script} not found - some functionality may be limited"
    else
        print_status "✓ ${script} found"
    fi
done

echo ""

# Prompt user for confirmation
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}WARNING: This pipeline will take several hours to complete!${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "The pipeline will train 4 model architectures × 4 feature combinations = 16 models"
echo "Each model will go through 3 curriculum stages with auto early stopping"
echo ""
echo -e "Estimated time: ${CYAN}4-12 hours${NC} (depending on hardware and early stopping)"
echo ""

read -p "Do you want to proceed? (yes/no): " -r
echo ""
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    print_warning "Pipeline cancelled by user"
    exit 0
fi

# Record start time
pipeline_start=$(date +%s)

# ============================================================================
# PHASE 1: TRAINING
# ============================================================================

print_section "PHASE 1: Training All Models"
print_status "Starting model training with auto curriculum and wandb logging..."
echo ""

training_start=$(date +%s)

if ./train_all_models.sh; then
    training_end=$(date +%s)
    training_elapsed=$((training_end - training_start))
    training_hours=$((training_elapsed / 3600))
    training_minutes=$(((training_elapsed % 3600) / 60))
    training_seconds=$((training_elapsed % 60))

    print_status "✓ Training phase completed successfully!"
    print_status "Training time: ${training_hours}h ${training_minutes}m ${training_seconds}s"
else
    print_error "Training phase failed!"
    print_error "Check training logs for details"
    exit 1
fi

echo ""
sleep 2

# ============================================================================
# PHASE 2: EVALUATION
# ============================================================================

print_section "PHASE 2: Evaluating All Trained Models"
print_status "Starting model evaluation on out-of-sample data..."
echo ""

eval_start=$(date +%s)

if ./evaluate_all_models.sh; then
    eval_end=$(date +%s)
    eval_elapsed=$((eval_end - eval_start))
    eval_hours=$((eval_elapsed / 3600))
    eval_minutes=$(((eval_elapsed % 3600) / 60))
    eval_seconds=$((eval_elapsed % 60))

    print_status "✓ Evaluation phase completed successfully!"
    print_status "Evaluation time: ${eval_hours}h ${eval_minutes}m ${eval_seconds}s"
else
    print_warning "Evaluation phase completed with some failures"
    print_warning "Check evaluation logs for details"
fi

echo ""
sleep 2

# ============================================================================
# FINAL SUMMARY
# ============================================================================

pipeline_end=$(date +%s)
pipeline_elapsed=$((pipeline_end - pipeline_start))
pipeline_hours=$((pipeline_elapsed / 3600))
pipeline_minutes=$(((pipeline_elapsed % 3600) / 60))
pipeline_seconds=$((pipeline_elapsed % 60))

print_section "Pipeline Complete!"

echo -e "${GREEN}"
cat << "EOF"
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                          ✓ PIPELINE COMPLETED                              ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
echo -e "${GREEN}Summary:${NC}"
echo -e "  Total pipeline time: ${CYAN}${pipeline_hours}h ${pipeline_minutes}m ${pipeline_seconds}s${NC}"
echo -e "  Training time: ${CYAN}${training_hours}h ${training_minutes}m ${training_seconds}s${NC}"
echo -e "  Evaluation time: ${CYAN}${eval_hours}h ${eval_minutes}m ${eval_seconds}s${NC}"
echo ""

echo -e "${GREEN}Output Locations:${NC}"
echo -e "  📁 Trained models: ${CYAN}models/${NC}"
echo -e "  📁 Training logs: ${CYAN}training_logs/${NC}"
echo -e "  📁 Evaluation results: ${CYAN}evaluation_results/${NC}"
echo -e "  📁 Evaluation logs: ${CYAN}evaluation_logs/${NC}"
echo -e "  📄 Pipeline log: ${CYAN}${PIPELINE_LOG}${NC}"
echo ""

echo -e "${GREEN}Next Steps:${NC}"
echo "  1. Check wandb dashboard for training metrics: https://wandb.ai"
echo "  2. Review evaluation results in evaluation_results/"
echo "  3. Compare model performance across feature combinations"
echo "  4. Analyze visualizations and reports for insights"
echo ""

echo -e "${YELLOW}For detailed analysis:${NC}"
echo "  - Open evaluation JSON files for quantitative metrics"
echo "  - View PNG visualizations for performance trends"
echo "  - Read generated text reports for comprehensive summaries"
echo ""

print_status "Full pipeline log saved to: ${PIPELINE_LOG}"
print_status "Pipeline completed at $(date)"

echo ""
echo -e "${CYAN}Thank you for using FYP-FinAgent! 🚀${NC}"
echo ""

exit 0
