#!/bin/bash

################################################################################
# Comprehensive Training Script for FYP-FinAgent Project
#
# This script trains all three model architectures (PPO Feature Combinations,
# Plain RL LSTM, PPO Transformer) with different feature combinations,
# auto curriculum learning, and wandb logging.
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
NC='\033[0m' # No Color

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="training_logs/${TIMESTAMP}"
MODELS_DIR="models"
RESULTS_DIR="training_results/${TIMESTAMP}"

# Feature combinations to train
FEATURE_COMBINATIONS=(
    "ohlcv"
    "ohlcv+technical"
    "ohlcv+technical+sentiment"
    "all"
)

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Print banner
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}                    FYP-FinAgent Comprehensive Training Pipeline${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}Timestamp:${NC} ${TIMESTAMP}"
echo -e "${GREEN}Log Directory:${NC} ${LOG_DIR}"
echo -e "${GREEN}Models Directory:${NC} ${MODELS_DIR}"
echo -e "${GREEN}Results Directory:${NC} ${RESULTS_DIR}"
echo ""
echo -e "${YELLOW}Feature Combinations:${NC} ${FEATURE_COMBINATIONS[@]}"
echo ""
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
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

# Function to train PPO with feature combinations (LSTM architecture)
train_ppo_feature_combinations() {
    local feature_combo=$1
    local feature_combo_safe="${feature_combo//+/_}"
    local log_file="${LOG_DIR}/ppo_feature_combinations_${feature_combo_safe}.log"
    local model_subdir="${MODELS_DIR}/ppo_feature_combinations/${feature_combo_safe}"

    print_section "Training PPO Feature Combinations (LSTM): ${feature_combo}"
    print_status "Starting training with feature combination: ${feature_combo}"
    print_status "Log file: ${log_file}"
    print_status "Model directory: ${model_subdir}"

    python train_ppo_feature_combinations.py \
        --num_updates 2500 \
        --feature_combination "${feature_combo}" \
        --auto_curriculum \
        --use_wandb \
        --model_dir "${model_subdir}" \
        > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        print_status "✓ Successfully completed PPO Feature Combinations training for ${feature_combo}"
    else
        print_error "✗ Failed training PPO Feature Combinations for ${feature_combo}"
        print_error "Check log file: ${log_file}"
        return 1
    fi
}

# Function to train Plain RL LSTM
train_plain_rl_lstm() {
    local feature_combo=$1
    local feature_combo_safe="${feature_combo//+/_}"
    local log_file="${LOG_DIR}/plain_rl_lstm_${feature_combo_safe}.log"
    local model_subdir="${MODELS_DIR}/plain_rl_lstm/${feature_combo_safe}"

    print_section "Training Plain RL LSTM: ${feature_combo}"
    print_status "Starting training with feature combination: ${feature_combo}"
    print_status "Log file: ${log_file}"
    print_status "Model directory: ${model_subdir}"

    python train_plain_rl_lstm.py \
        --num_updates 2500 \
        --feature_combination "${feature_combo}" \
        --auto_curriculum \
        --use_wandb \
        --model_dir "${model_subdir}" \
        > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        print_status "✓ Successfully completed Plain RL LSTM training for ${feature_combo}"
    else
        print_error "✗ Failed training Plain RL LSTM for ${feature_combo}"
        print_error "Check log file: ${log_file}"
        return 1
    fi
}

# Function to train PPO Transformer
train_ppo_transformer() {
    local feature_combo=$1
    local feature_combo_safe="${feature_combo//+/_}"
    local log_file="${LOG_DIR}/ppo_transformer_${feature_combo_safe}.log"
    local model_subdir="${MODELS_DIR}/ppo_transformer/${feature_combo_safe}"

    print_section "Training PPO Transformer: ${feature_combo}"
    print_status "Starting training with feature combination: ${feature_combo}"
    print_status "Log file: ${log_file}"
    print_status "Model directory: ${model_subdir}"

    python train_ppo_transformer.py \
        --total_timesteps 2000000 \
        --feature_combination "${feature_combo}" \
        --auto_curriculum \
        --use_wandb \
        --n_envs 4 \
        --n_steps 64 \
        --model_dir "${model_subdir}" \
        > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        print_status "✓ Successfully completed PPO Transformer training for ${feature_combo}"
    else
        print_error "✗ Failed training PPO Transformer for ${feature_combo}"
        print_error "Check log file: ${log_file}"
        return 1
    fi
}

# Main training loop
main() {
    local start_time=$(date +%s)
    local total_models=$((${#FEATURE_COMBINATIONS[@]} * 3))
    local completed_models=0
    local failed_models=0

    print_section "Starting Training Pipeline"
    print_status "Total models to train: ${total_models}"

    # Array to track failed trainings
    declare -a failed_trainings

    # Train all models with all feature combinations
    for feature_combo in "${FEATURE_COMBINATIONS[@]}"; do

        # # 1. Train PPO Feature Combinations (LSTM)
        # print_status "[${completed_models}/${total_models}] Training PPO Feature Combinations - ${feature_combo}..."
        # if train_ppo_feature_combinations "${feature_combo}"; then
        #     ((completed_models++))
        # else
        #     ((failed_models++))
        #     failed_trainings+=("PPO_Feature_Combinations_${feature_combo}")
        # fi

        # # 2. Train Plain RL LSTM
        # print_status "[${completed_models}/${total_models}] Training Plain RL LSTM - ${feature_combo}..."
        # if train_plain_rl_lstm "${feature_combo}"; then
        #     ((completed_models++))
        # else
        #     ((failed_models++))
        #     failed_trainings+=("Plain_RL_LSTM_${feature_combo}")
        # fi

        # 3. Train PPO Transformer
        print_status "[${completed_models}/${total_models}] Training PPO Transformer - ${feature_combo}..."
        if train_ppo_transformer "${feature_combo}"; then
            ((completed_models++))
        else
            ((failed_models++))
            failed_trainings+=("PPO_Transformer_${feature_combo}")
        fi

        echo ""
    done

    # Calculate elapsed time
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))

    # Print summary
    print_section "Training Pipeline Completed"
    echo ""
    echo -e "${GREEN}Summary:${NC}"
    echo -e "  Total models: ${total_models}"
    echo -e "  Successfully trained: ${GREEN}${completed_models}${NC}"
    echo -e "  Failed: ${RED}${failed_models}${NC}"
    echo -e "  Elapsed time: ${hours}h ${minutes}m ${seconds}s"
    echo ""

    if [ ${failed_models} -gt 0 ]; then
        echo -e "${RED}Failed trainings:${NC}"
        for failed in "${failed_trainings[@]}"; do
            echo -e "  - ${failed}"
        done
        echo ""
    fi

    echo -e "${GREEN}Logs saved to:${NC} ${LOG_DIR}"
    echo -e "${GREEN}Models saved to:${NC} ${MODELS_DIR}"
    echo ""

    # Save summary to file
    local summary_file="${RESULTS_DIR}/training_summary.txt"
    {
        echo "FYP-FinAgent Training Summary"
        echo "=============================="
        echo ""
        echo "Timestamp: ${TIMESTAMP}"
        echo "Total models: ${total_models}"
        echo "Successfully trained: ${completed_models}"
        echo "Failed: ${failed_models}"
        echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s"
        echo ""
        echo "Feature Combinations:"
        for combo in "${FEATURE_COMBINATIONS[@]}"; do
            echo "  - ${combo}"
        done
        echo ""
        if [ ${failed_models} -gt 0 ]; then
            echo "Failed trainings:"
            for failed in "${failed_trainings[@]}"; do
                echo "  - ${failed}"
            done
        fi
    } > "${summary_file}"

    print_status "Summary saved to: ${summary_file}"

    # Return appropriate exit code
    if [ ${failed_models} -gt 0 ]; then
        print_warning "Training completed with ${failed_models} failures"
        return 1
    else
        print_status "All trainings completed successfully!"
        return 0
    fi
}

# Trap errors
trap 'print_error "Script interrupted"; exit 130' INT TERM

# Run main function
main

exit $?
