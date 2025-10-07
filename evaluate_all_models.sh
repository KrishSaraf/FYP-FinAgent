#!/bin/bash

################################################################################
# Comprehensive Evaluation Script for FYP-FinAgent Project
#
# This script evaluates all trained models (PPO Feature Combinations,
# Plain RL LSTM, PPO Transformer) on out-of-sample data and generates
# comprehensive performance reports and visualizations.
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
EVAL_LOG_DIR="evaluation_logs/${TIMESTAMP}"
EVAL_RESULTS_DIR="evaluation_results"
MODELS_DIR="models"

# Evaluation parameters
NUM_EVAL_EPISODES=20
EVAL_START_DATE="2025-03-07"
EVAL_END_DATE="2025-06-06"

# Feature combinations to evaluate (should match training)
FEATURE_COMBINATIONS=(
    "ohlcv"
    "ohlcv+technical"
    "ohlcv+technical+sentiment"
    "all"
)

# Model types
MODEL_TYPES=(
    "ppo_feature_combinations"
    "plain_rl_lstm"
    "ppo_transformer"
)

# Create directories
mkdir -p "${EVAL_LOG_DIR}"
mkdir -p "${EVAL_RESULTS_DIR}"

# Print banner
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}                    FYP-FinAgent Comprehensive Evaluation Pipeline${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}Timestamp:${NC} ${TIMESTAMP}"
echo -e "${GREEN}Evaluation Log Directory:${NC} ${EVAL_LOG_DIR}"
echo -e "${GREEN}Results Directory:${NC} ${EVAL_RESULTS_DIR}"
echo -e "${GREEN}Evaluation Period:${NC} ${EVAL_START_DATE} to ${EVAL_END_DATE}"
echo -e "${GREEN}Number of Episodes:${NC} ${NUM_EVAL_EPISODES}"
echo ""
echo -e "${YELLOW}Feature Combinations:${NC} ${FEATURE_COMBINATIONS[@]}"
echo -e "${YELLOW}Model Types:${NC} ${MODEL_TYPES[@]}"
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

# Function to find the latest model file for a given type and feature combination
find_latest_model() {
    local model_type=$1
    local feature_combo=$2
    local feature_combo_safe="${feature_combo//+/_}"

    # Path structure: models/{architecture}/{feature_combo}/
    local model_dir="${MODELS_DIR}/${model_type}/${feature_combo_safe}"

    # Look for curriculum stage models first, then final models
    # Different model types save with different naming conventions
    local model_patterns=()

    case "${model_type}" in
        "plain_rl_lstm")
            model_patterns=(
                "${model_dir}/curriculum_stage_3.pkl"
                "${model_dir}/curriculum_stage_2.pkl"
                "${model_dir}/curriculum_stage_1.pkl"
                "${model_dir}/final_model_plain_rl_lstm_${feature_combo_safe}.pkl"
                "${model_dir}/plain_rl_lstm_update_*.pkl"
            )
            ;;
        "ppo_transformer")
            model_patterns=(
                "${model_dir}/curriculum_stage_3.pkl"
                "${model_dir}/curriculum_stage_2.pkl"
                "${model_dir}/curriculum_stage_1.pkl"
                "${model_dir}/final_model_${feature_combo_safe}.pkl"
                "${model_dir}/final_model.pkl"
                "${model_dir}/checkpoint_*.pkl"
            )
            ;;
        "ppo_feature_combinations")
            model_patterns=(
                "${model_dir}/curriculum_stage_3.pkl"
                "${model_dir}/curriculum_stage_2.pkl"
                "${model_dir}/curriculum_stage_1.pkl"
                "${model_dir}/final_model_${feature_combo_safe}.pkl"
                "${model_dir}/final_model.pkl"
            )
            ;;
    esac

    for pattern in "${model_patterns[@]}"; do
        if ls ${pattern} 1> /dev/null 2>&1; then
            # Get the most recent file matching the pattern
            local latest=$(ls -t ${pattern} 2>/dev/null | head -1)
            if [ -n "${latest}" ]; then
                echo "${latest}"
                return 0
            fi
        fi
    done

    return 1
}

# Function to evaluate PPO Feature Combinations models
evaluate_ppo_feature_combinations() {
    local feature_combo=$1
    local model_path=$2
    local log_file="${EVAL_LOG_DIR}/eval_ppo_feature_combinations_${feature_combo//+/_}.log"
    local results_dir="${EVAL_RESULTS_DIR}/PPO_feature_combinations/${feature_combo//+/_}"

    print_section "Evaluating PPO Feature Combinations: ${feature_combo}"
    print_status "Model: ${model_path}"
    print_status "Results directory: ${results_dir}"
    print_status "Log file: ${log_file}"

    mkdir -p "${results_dir}"

    # Note: eval_ppo_feature_combinations.py doesn't exist yet, using the same structure
    # This assumes you'll create it or use the training script with eval mode
    python eval_ppo_feature_combinations.py \
        --model_path "${model_path}" \
        --feature_combination "${feature_combo}" \
        --num_episodes ${NUM_EVAL_EPISODES} \
        --eval_start_date "${EVAL_START_DATE}" \
        --eval_end_date "${EVAL_END_DATE}" \
        --results_dir "${results_dir}" \
        > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        print_status "✓ Successfully evaluated PPO Feature Combinations for ${feature_combo}"
        return 0
    else
        print_error "✗ Failed evaluating PPO Feature Combinations for ${feature_combo}"
        print_error "Check log file: ${log_file}"
        return 1
    fi
}

# Function to evaluate Plain RL LSTM models
evaluate_plain_rl_lstm() {
    local feature_combo=$1
    local model_path=$2
    local log_file="${EVAL_LOG_DIR}/eval_plain_rl_lstm_${feature_combo//+/_}.log"
    local results_dir="${EVAL_RESULTS_DIR}/Plain_RL_LSTM/${feature_combo//+/_}"

    print_section "Evaluating Plain RL LSTM: ${feature_combo}"
    print_status "Model: ${model_path}"
    print_status "Results directory: ${results_dir}"
    print_status "Log file: ${log_file}"

    mkdir -p "${results_dir}"

    python eval_plain_rl_lstm.py \
        --model_path "${model_path}" \
        --feature_combination "${feature_combo}" \
        --num_episodes ${NUM_EVAL_EPISODES} \
        --eval_start_date "${EVAL_START_DATE}" \
        --eval_end_date "${EVAL_END_DATE}" \
        --results_dir "${results_dir}" \
        > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        print_status "✓ Successfully evaluated Plain RL LSTM for ${feature_combo}"
        return 0
    else
        print_error "✗ Failed evaluating Plain RL LSTM for ${feature_combo}"
        print_error "Check log file: ${log_file}"
        return 1
    fi
}

# Function to evaluate PPO Transformer models
evaluate_ppo_transformer() {
    local feature_combo=$1
    local model_path=$2
    local log_file="${EVAL_LOG_DIR}/eval_ppo_transformer_${feature_combo//+/_}.log"
    local results_dir="${EVAL_RESULTS_DIR}/PPO_Transformer/${feature_combo//+/_}"

    print_section "Evaluating PPO Transformer: ${feature_combo}"
    print_status "Model: ${model_path}"
    print_status "Results directory: ${results_dir}"
    print_status "Log file: ${log_file}"

    mkdir -p "${results_dir}"

    python eval_ppo_transformer.py \
        --model_path "${model_path}" \
        --feature_combination "${feature_combo}" \
        --num_episodes ${NUM_EVAL_EPISODES} \
        --eval_start_date "${EVAL_START_DATE}" \
        --eval_end_date "${EVAL_END_DATE}" \
        --results_dir "${results_dir}" \
        > "${log_file}" 2>&1

    if [ $? -eq 0 ]; then
        print_status "✓ Successfully evaluated PPO Transformer for ${feature_combo}"
        return 0
    else
        print_error "✗ Failed evaluating PPO Transformer for ${feature_combo}"
        print_error "Check log file: ${log_file}"
        return 1
    fi
}

# Function to generate comparative analysis
generate_comparative_analysis() {
    print_section "Generating Comparative Analysis"

    local analysis_file="${EVAL_RESULTS_DIR}/comparative_analysis_${TIMESTAMP}.txt"
    local summary_file="${EVAL_RESULTS_DIR}/evaluation_summary_${TIMESTAMP}.txt"

    print_status "Creating comparative analysis report..."

    {
        echo "FYP-FinAgent Model Evaluation - Comparative Analysis"
        echo "===================================================="
        echo ""
        echo "Evaluation Date: $(date)"
        echo "Evaluation Period: ${EVAL_START_DATE} to ${EVAL_END_DATE}"
        echo "Number of Episodes per Model: ${NUM_EVAL_EPISODES}"
        echo ""
        echo "Model Architectures Evaluated:"
        echo "  1. PPO with Feature Combinations (LSTM)"
        echo "  2. Plain RL LSTM (REINFORCE)"
        echo "  3. PPO Transformer"
        echo ""
        echo "Feature Combinations Evaluated:"
        for combo in "${FEATURE_COMBINATIONS[@]}"; do
            echo "  - ${combo}"
        done
        echo ""
        echo "Results Location: ${EVAL_RESULTS_DIR}"
        echo "Logs Location: ${EVAL_LOG_DIR}"
        echo ""
        echo "Individual model results can be found in their respective directories."
        echo "Visualizations and detailed reports are saved within each model's result folder."
        echo ""
        echo "To compare models:"
        echo "  1. Check evaluation JSON files for quantitative metrics"
        echo "  2. Review visualization PNGs for performance trends"
        echo "  3. Compare Sharpe ratios, returns, and drawdowns across models"
        echo ""
    } > "${analysis_file}"

    print_status "Analysis report saved to: ${analysis_file}"

    # Create evaluation summary
    {
        echo "Evaluation Summary"
        echo "=================="
        echo ""
        echo "Timestamp: ${TIMESTAMP}"
        echo "Completed evaluations: ${completed_evals}"
        echo "Failed evaluations: ${failed_evals}"
        echo ""
        if [ ${#failed_evaluations[@]} -gt 0 ]; then
            echo "Failed Evaluations:"
            for failed in "${failed_evaluations[@]}"; do
                echo "  - ${failed}"
            done
        fi
    } > "${summary_file}"

    print_status "Summary saved to: ${summary_file}"
}

# Main evaluation loop
main() {
    local start_time=$(date +%s)
    local total_evals=$((${#FEATURE_COMBINATIONS[@]} * ${#MODEL_TYPES[@]}))
    completed_evals=0
    failed_evals=0

    # Array to track failed evaluations
    declare -a failed_evaluations

    print_section "Starting Evaluation Pipeline"
    print_status "Total evaluations to perform: ${total_evals}"

    # Disable exit on error for the entire evaluation loop
    set +e

    # Evaluate all models
    for feature_combo in "${FEATURE_COMBINATIONS[@]}"; do
        for model_type in "${MODEL_TYPES[@]}"; do

            # Find the model file
            model_path=$(find_latest_model "${model_type}" "${feature_combo}")

            if [ -z "${model_path}" ]; then
                print_warning "No model found for ${model_type} with feature combination ${feature_combo}"
                print_warning "Skipping evaluation..."
                ((failed_evals++))
                failed_evaluations+=("${model_type}_${feature_combo}_NO_MODEL_FOUND")
                continue
            fi

            # Evaluate based on model type
            case "${model_type}" in
                "ppo_feature_combinations")
                    evaluate_ppo_feature_combinations "${feature_combo}" "${model_path}"
                    if [ $? -eq 0 ]; then
                        ((completed_evals++))
                    else
                        ((failed_evals++))
                        failed_evaluations+=("PPO_Feature_Combinations_${feature_combo}")
                    fi
                    ;;
                "plain_rl_lstm")
                    evaluate_plain_rl_lstm "${feature_combo}" "${model_path}"
                    if [ $? -eq 0 ]; then
                        ((completed_evals++))
                    else
                        ((failed_evals++))
                        failed_evaluations+=("Plain_RL_LSTM_${feature_combo}")
                    fi
                    ;;
                "ppo_transformer")
                    evaluate_ppo_transformer "${feature_combo}" "${model_path}"
                    if [ $? -eq 0 ]; then
                        ((completed_evals++))
                    else
                        ((failed_evals++))
                        failed_evaluations+=("PPO_Transformer_${feature_combo}")
                    fi
                    ;;
            esac

            echo ""
        done
    done

    # Re-enable exit on error after evaluation loop
    set -e

    # Generate comparative analysis
    generate_comparative_analysis

    # Calculate elapsed time
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))

    # Print summary
    print_section "Evaluation Pipeline Completed"
    echo ""
    echo -e "${GREEN}Summary:${NC}"
    echo -e "  Total evaluations: ${total_evals}"
    echo -e "  Successfully completed: ${GREEN}${completed_evals}${NC}"
    echo -e "  Failed: ${RED}${failed_evals}${NC}"
    echo -e "  Elapsed time: ${hours}h ${minutes}m ${seconds}s"
    echo ""

    if [ ${failed_evals} -gt 0 ]; then
        echo -e "${RED}Failed evaluations:${NC}"
        for failed in "${failed_evaluations[@]}"; do
            echo -e "  - ${failed}"
        done
        echo ""
    fi

    echo -e "${GREEN}Evaluation logs saved to:${NC} ${EVAL_LOG_DIR}"
    echo -e "${GREEN}Results saved to:${NC} ${EVAL_RESULTS_DIR}"
    echo ""

    # Return appropriate exit code
    if [ ${failed_evals} -gt 0 ]; then
        print_warning "Evaluation completed with ${failed_evals} failures"
        return 1
    else
        print_status "All evaluations completed successfully!"
        return 0
    fi
}

# Trap errors
trap 'print_error "Script interrupted"; exit 130' INT TERM

# Run main function
main

exit $?
