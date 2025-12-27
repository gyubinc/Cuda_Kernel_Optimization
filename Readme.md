# CUDA Kernel Optimization & Performance Prediction

This project analyzes the performance of various CUDA kernels (TPM, MatMul, Reduction) and predicts optimal configurations using Machine Learning and Deep Learning models.

## Quick Start

Follow these steps to set up the environment and run a sample experiment (Task: TPM, Model: SVM).


## Baseline 실험 (T_overhead)

The project includes an analytical baseline model (**T_overhead**) that predicts optimal stream configurations using theoretical overhead models. This serves as the fundamental comparison target for the ML/DL models.

```bash
# Run Baseline Analysis (T_overhead)
git clone https://github.com/gyubinc/Cuda_Kernel_Optimization.git
pip install -r requirements.txt
python3 scripts/baseline.py
```

## 모든 모델, task 실험

```bash
git clone https://github.com/gyubinc/Cuda_Kernel_Optimization.git
pip install -r requirements.txt

# 모든 조합 모두 실행
bash test_all.sh

# 모든 task를 1개 모델씩 실행
bash test_simple.sh

```

## 직접 task, model 지정

```bash
git clone https://github.com/gyubinc/Cuda_Kernel_Optimization.git

pip install -r requirements.txt

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
  - `baseline.py`: Implements the analytical T_overhead model and baseline comparison.
- **`data/`**: Stores collected CSV data and training results (created automatically).
- **`bin/`**: Stores compiled CUDA executables (created automatically).