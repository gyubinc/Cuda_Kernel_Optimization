#!/bin/bash
set -e  # 에러 발생 시 즉시 중단

# =============================================================================
# 프로젝트 실행 파이프라인 스크립트
# 사용법: srun -t 01:00 --gres=gpu:1 ./scripts/run_pipeline.sh
# 
# 기능:
# 1. CUDA 커널 컴파일 (TPM, MatMul, Reduction)
# 2. 데이터 수집 (1분 제한을 맞추기 위해 축소된 데이터셋 사용)
# 3. 데이터 전처리
# 4. ML/DL 모델 학습 (총 10개 모델)
# =============================================================================

# 현재 스크립트 위치 기준 프로젝트 루트 설정
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"
echo "[Info] Working Directory: $BASE_DIR"

# CUDA 컴파일러 경로 (환경에 따라 수정 필요)
NVCC="/usr/local/cuda-11.8/bin/nvcc"
if [ ! -f "$NVCC" ]; then
    NVCC="nvcc" # PATH에 있는 경우 fallback
fi

echo "=================================================="
echo " [1/5] Compiling CUDA Kernels"
echo "=================================================="
mkdir -p bin

echo "Compiling TPM Solver..."
$NVCC -o bin/tpm_solver src/tpm_solver.cu -O3 -arch=sm_70

echo "Compiling MatMul Solver..."
$NVCC -o bin/matmul_solver src/matmul_solver.cu -O3 -arch=sm_70

echo "Compiling Reduction Solver..."
$NVCC -o bin/reduction_solver src/reduction_solver.cu -O3 -arch=sm_70


echo "=================================================="
echo " [2/5] Collecting Data (Fast Mode)"
echo " Note: 1분 제한을 위해 데이터 크기를 제한하여 실행합니다."
echo "=================================================="

# --mode direct: 이미 srun으로 할당된 노드에서 실행하므로 내부에서 srun 호출 금지
# --n-max: 실행 시간을 단축하기 위해 문제 크기 제한

echo ">> Collecting TPM Data..."
python3 scripts/collect_data.py --task tpm --mode direct --n-max 20000

echo ">> Collecting MatMul Data..."
python3 scripts/collect_data.py --task matmul --mode direct --n-max 1024

echo ">> Collecting Reduction Data..."
# Reduction은 최소 사이즈가 10만이므로 10만~20만 사이로 제한
python3 scripts/collect_data.py --task reduction --mode direct --n-max 200000


echo "=================================================="
echo " [3/5] Preprocessing Data"
echo "=================================================="
python3 scripts/preprocess.py --task all


echo "=================================================="
echo " [4/5] Training ML Models (5 Models)"
echo "=================================================="
python3 scripts/train_ml.py


echo "=================================================="
echo " [5/5] Training DL Models (5 Models)"
echo "=================================================="
python3 scripts/train_dl.py


echo "=================================================="
echo " All Steps Completed Successfully!"
echo "=================================================="
