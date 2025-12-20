# Project Goal: High-Precision ML/DL Optimization for Multi-Task CUDA Kernels

## 1. Project Overview
본 프로젝트는 "ML-Based Optimum Number of CUDA Streams" 논문의 Tridiagonal Partition Method(TPM) 최적화를 완벽하게 재현하고, 이를 확장하여 **Compute-bound(Matrix Mul)** 및 **Memory-bound(Reduction)** 커널에 대해서도 **5종의 ML 모델과 5종의 DL 모델**을 적용해 최적의 실행 파라미터(Stream Count, Block Size)를 예측하는 연구이다.

## 2. Dataset Construction & Specification
모든 데이터는 `srun`을 통해 실제 GPU 하드웨어에서 측정된 **Execution Time**을 기준으로 구축하며, 아래 3가지 Task에 대해 독립적인 데이터셋을 생성한다.

### 2.1. Task 1: TPM (Paper Reproduction - Solver Type)
* **Objective:** 논문의 실험 조건을 100% 재현 및 파라미터 확장.
* **Input Features (SLAE Size $N$):**
    * Range: $10^3 \sim 10^8$.
    * Detailed Points: $1.0, 2.5, 4.0, 5.0, 7.5, 8.0 \times 10^i$ (for $i=3,..,7$).
* **Target Labels (Output Class):**
    * **Stream Count:** 1, 2, 4, 8, 16, 32 (Powers of 2).
    * **Block Size:** 128, 256, 512 (논문은 256 고정, 본 연구는 가변 실험).
* **Dataset Size:** 약 1,500 Data Points (Size 조합 $\times$ Parameter 조합).

### 2.2. Task 2: Tiled Matrix Multiplication (Extension - Compute Bound)
* **Objective:** 연산 집약적 커널에서의 ML/DL 최적화 성능 검증.
* **Input Features (Matrix Dimension $N$):**
    * Range: $512 \sim 8192$.
    * Step: 256 단위로 dense하게 샘플링.
* **Target Labels:**
    * **Stream Count:** 1, 2, 4, 8, 16 (Tiled Execution).
    * **Tile/Block Size:** 8, 16, 32 (Shared Memory usage에 민감).
* **Dataset Size:** 약 1,200 Data Points.

### 2.3. Task 3: Large Vector Reduction (Extension - Memory Bound)
* **Objective:** 메모리 대역폭 한계 상황에서의 파라미터 튜닝.
* **Input Features (Vector Size $N$):**
    * Range: $10^5 \sim 10^8$ (Log scale sampling).
* **Target Labels:**
    * **Stream Count:** 1, 2, 4, 8, 16, 32.
    * **Block Size:** 64, 128, 256, 512, 1024 (Occupancy 최적화).
* **Dataset Size:** 약 1,800 Data Points.

### 2.4. Data Processing (Labeling)
* **Raw Data:** `[Input_Size, Stream, Block, Execution_Time]` 형태의 CSV.
* **Labeling Logic:** 각 Input Size 별로 **Execution Time이 가장 짧은(Min)** 행을 찾아 `Is_Optimal=1`, 나머지는 `0`으로 라벨링.
* **Split:** Training : Test = 8 : 2 (Random Shuffle).

## 3. Models for Experiments (Total 10 Models + Baseline)
각 Task 별로 아래 모델들을 모두 학습시키고 성능을 비교한다.

### 3.1. Baseline (Reference)
* **Paper's Heuristic:** TPM Task에 대해 논문의 수식 ($T_{overhead}$ vs $Sum$) 적용.
* **Polynomial Regression:** Task 2, 3에 대한 전통적 복잡도 모델링 ($O(N^2), O(N)$ 등).

### 3.2. Machine Learning Models (5 Types)
Scikit-learn 및 Boosting Library 활용.
1.  **Random Forest Classifier:** 비선형적 관계 및 Outlier에 강건함.
2.  **XGBoost:** Gradient Boosting의 표준, 높은 예측 성능.
3.  **LightGBM:** 대용량 데이터에서 빠른 학습 속도와 Leaf-wise growth.
4.  **CatBoost:** 범주형 변수 처리에 강하며 튜닝 없이도 높은 성능.
5.  **SVM (Support Vector Machine):** 고차원 공간에서의 최적 경계 탐색 (RBF Kernel).

### 3.3. Deep Learning Models (5 Types)
PyTorch 기반 구현 (CUDA 가속).
1.  **MLP (Multi-Layer Perceptron):** 기본 3-Layer DNN 구조 (ReLU).
2.  **Deep DNN:** 5-Layer 이상, Dropout & BatchNorm 적용.
3.  **ResNet-MLP:** Residual Connection(Skip connection)을 적용하여 깊은 망 학습 안정화.
4.  **TabNet:** Tabular 데이터에 특화된 Attention 기반 딥러닝 아키텍처.
5.  **FT-Transformer:** Numerical Embeddings와 Transformer Layer를 결합한 최신 아키텍처.

## 4. Final Deliverables (`report.md`)
보고서는 반드시 아래 구조와 포함 요소를 준수해야 한다.

1.  **Introduction**
    * 연구 배경 및 CUDA Stream Optimization의 중요성.
2.  **Existing Paper & Baseline**
    * 논문의 $T_{overhead}$ 모델링 방식 설명 및 재현 결과.
3.  **Experimental Setup**
    * Task 1, 2, 3의 데이터셋 구축 과정 상세 (Size, Parameters).
    * 사용된 5 ML + 5 DL 모델 구조 설명.
4.  **Results: Prediction Accuracy**
    * **[Table 1]** Task별 10개 모델의 최적 파라미터 예측 정확도(Top-1 Accuracy).
5.  **Results: Performance Speedup**
    * **[Table 2]** Default 설정 대비 각 모델이 추천한 설정의 실제 실행 시간 단축 비율(Speedup).
6.  **Conclusion**
    * Task 성격(Compute vs Memory bound)에 따른 Best Model 선정.

## 5. Execution Environment
* **System:** Linux Server (Slurm Workload Manager - `srun`).
* **Environment:** `conda activate gyubin`.
* **Language:** 한국어 (전문 용어는 영어).