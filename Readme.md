# CUDA Kernel Optimization & Performance Prediction

이 프로젝트는 다양한 CUDA 커널(TPM, MatMul, Reduction)의 성능을 분석하고, 머신러닝 및 딥러닝 모델을 활용하여 최적의 구성을 예측합니다.

# Quick Start

## Baseline 실험 (T_overhead)

base인 T_overhead 모델을 사용하여 최적의 스트림 구성을 예측하는 분석적 베이스라인 모델(**T_overhead**)을 포함합니다. 
이는 ML/DL 모델의 성능을 평가하기 위한 핵심적인 baseline 역할을 합니다.

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