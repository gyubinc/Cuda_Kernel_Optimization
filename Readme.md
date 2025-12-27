# CUDA Kernel Optimization & Performance Prediction

This project analyzes the performance of various CUDA kernels (TPM, MatMul, Reduction) and predicts optimal configurations using Machine Learning and Deep Learning models.

## Quick Start

Follow these steps to set up the environment and run a sample experiment (Task: TPM, Model: SVM).

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Experiment (Model Training)
# The dataset is pre-included in the 'data/' folder, so you can train models immediately.

# Option A: TPM (Tridiagonal Matrix)
srun -t 01:00 --gres=gpu:1 python3 scripts/run_pipeline.py --task tpm --model svm_rbf

# Option B: MatMul (Matrix Multiplication)
srun -t 01:00 --gres=gpu:1 python3 scripts/run_pipeline.py --task matmul --model random_forest

# Option C: Reduction (Parallel Sum)
srun -t 01:00 --gres=gpu:1 python3 scripts/run_pipeline.py --task reduction --model xgboost 

```

## Available Options

You can customize the experiment by specifying the `--task` and `--model` arguments.

### Supported Tasks (`--task`)
| Task | Description |
| :--- | :--- |
| `tpm` | Tridiagonal Matrix Algorithm (Thomas Algorithm) Solver |
| `matmul` | Matrix Multiplication |
| `reduction` | Parallel Reduction (Sum) |

### Supported Models (`--model`)
You can specify a full name or a partial keyword (e.g., `mlp` runs all MLP-based models).

**Machine Learning Models:**
- `random_forest` : Random Forest Classifier
- `svm_rbf` : Support Vector Machine (RBF Kernel)
- `xgboost` : XGBoost

**Deep Learning Models:**
- `mlp` : Simple Multi-Layer Perceptron
- `deep_dnn` : Deep Dense Neural Network (BatchNorm + Dropout)
- `resnet_mlp` : ResNet-style MLP
- `tabnet` : TabNet (Attention-based network for tabular data)
- `ft_transformer` : FT-Transformer (Feature Tokenizer + Transformer)

## Project Structure

- **`src/`**: CUDA source codes (`tpm_solver.cu`, `matmul_solver.cu`, `reduction_solver.cu`).
- **`scripts/`**: Python scripts for data collection, preprocessing, and training.
  - `run_pipeline.py`: Main driver script.
  - `collect_data.py`: Runs CUDA binaries to collect execution time.
  - `preprocess.py`: Cleans data and splits into train/test sets.
  - `train_ml.py`: Trains ML models.
  - `train_dl.py`: Trains DL models.
- **`data/`**: Stores collected CSV data and training results (created automatically).
- **`bin/`**: Stores compiled CUDA executables (created automatically).