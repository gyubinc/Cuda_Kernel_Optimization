#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"

TASKS = ["tpm", "matmul", "reduction"]
ML_MODELS = ["random_forest", "svm_rbf", "xgboost", "lightgbm", "catboost"]
DL_MODELS = ["mlp", "deep_dnn", "resnet_mlp", "tabnet", "ft_transformer"]

def run_test(cmd, description):
    print(f"Testing: {description}...", end=" ", flush=True)
    start = time.time()
    try:
        # Run with capture_output to keep screen clean, print only if error
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=BASE_DIR)
        elapsed = time.time() - start
        print(f"\033[92mPASS\033[0m ({elapsed:.1f}s)")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\033[91mFAIL\033[0m ({elapsed:.1f}s)")
        print(f"  Error Output:\n{e.stderr}")
        return False, e.stderr

def main():
    print(f"=== Comprehensive System Verification ===")
    print(f"Tasks: {TASKS}")
    print(f"ML Models: {ML_MODELS}")
    print(f"DL Models: {DL_MODELS}")
    print(f"T_Overhead: Baseline script")
    print("="*40)

    failures = []
    
    # 1. Verify Pipeline (ML)
    for task in TASKS:
        for model in ML_MODELS:
            # We skip data collection/compilation if data exists (it should from previous steps)
            # relying on run_pipeline.py's auto-skip feature
            cmd = ["python3", str(SCRIPTS_DIR / "run_pipeline.py"), "--task", task, "--model", model]
            ok, _ = run_test(cmd, f"{task} + {model} (ML)")
            if not ok:
                failures.append(f"{task} + {model}")

    # 2. Verify Pipeline (DL)
    for task in TASKS:
        for model in DL_MODELS:
            cmd = ["python3", str(SCRIPTS_DIR / "run_pipeline.py"), "--task", task, "--model", model]
            ok, _ = run_test(cmd, f"{task} + {model} (DL)")
            if not ok:
                failures.append(f"{task} + {model}")

    # 3. Verify T_overhead (baseline.py)
    cmd = ["python3", str(SCRIPTS_DIR / "baseline.py")]
    ok, _ = run_test(cmd, "T_overhead (baseline.py)")
    if not ok:
        failures.append("T_overhead (baseline.py)")

    print("="*40)
    if not failures:
        print("\033[92mALL TESTS PASSED! System is fully operational.\033[0m")
        sys.exit(0)
    else:
        print(f"\033[91mFound {len(failures)} failures:\033[0m")
        for f in failures:
            print(f" - {f}")
        sys.exit(1)

if __name__ == "__main__":
    main()
