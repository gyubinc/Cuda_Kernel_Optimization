#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Install Dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Define Tasks and Models
TASKS=("tpm" "matmul" "reduction")
MODELS=("random_forest" "svm_rbf" "xgboost" "mlp" "deep_dnn" "resnet_mlp" "tabnet" "ft_transformer")

# 2. Run All Combinations
echo "Starting comprehensive test..."

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "================================================================"
        echo " Running Task: $task | Model: $model"
        echo "================================================================"
        
        # Run the pipeline
        srun -t 01:00 --gres=gpu:1 python3 scripts/run_pipeline.py --task "$task" --model "$model"
        
        echo " [Pass] $task / $model"
        echo ""
    done
done

# 3. Run Baseline
echo "================================================================"
echo " Running Baseline Analysis (T_overhead)"
echo "================================================================"
python3 scripts/baseline.py

echo ""
echo "All tests passed successfully!"
