#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# 1. Install Dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 2. Run Simple Tests (1 Model per Task)
echo "================================================================"
echo " Running TPM (Model: SVM RBF)"
echo "================================================================"
srun -t 01:00 --gres=gpu:1 python3 scripts/run_pipeline.py --task tpm --model svm_rbf

echo "================================================================"
echo " Running MatMul (Model: Random Forest)"
echo "================================================================"
srun -t 01:00 --gres=gpu:1 python3 scripts/run_pipeline.py --task matmul --model random_forest

echo "================================================================"
echo " Running Reduction (Model: XGBoost)"
echo "================================================================"
srun -t 01:00 --gres=gpu:1 python3 scripts/run_pipeline.py --task reduction --model xgboost

# 3. Run Baseline
echo "================================================================"
echo " Running Baseline Analysis (T_overhead)"
echo "================================================================"
python3 scripts/baseline.py

echo ""
echo "Simple test completed successfully!"
