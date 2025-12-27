#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = BASE_DIR / "bin"
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_DIR = BASE_DIR / "data"

def run_cmd(cmd, cwd=BASE_DIR):
    # print(f"[CMD] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def compile_kernels(task="all"):
    BIN_DIR.mkdir(exist_ok=True)
    
    # NVCC path check
    nvcc = "/usr/local/cuda-11.8/bin/nvcc"
    if not os.path.exists(nvcc):
        nvcc = "nvcc"
        
    solvers = {
        "tpm": ("tpm_solver", "src/tpm_solver.cu"),
        "matmul": ("matmul_solver", "src/matmul_solver.cu"),
        "reduction": ("reduction_solver", "src/reduction_solver.cu")
    }
    
    tasks = [task] if task != "all" else solvers.keys()
    
    # print("=========================================")
    # print(f" [1] Compiling kernels for: {tasks}")
    # print("=========================================")
    
    for t in tasks:
        if t not in solvers: continue
        out_name, src_file = solvers[t]
        cmd = [nvcc, "-o", str(BIN_DIR / out_name), str(BASE_DIR / src_file), "-O3", "-arch=sm_70"]
        run_cmd(cmd)

def collect_data(task, n_max):
    # print("=========================================")
    # print(f" [2] Collecting Data for: {task}")
    # print("=========================================")
    
    # Map task to n_max default if not provided
    # However, user code previously had specific defaults. 
    # If user provided n_max, use it.
    
    cmd = ["python3", str(SCRIPTS_DIR / "collect_data.py"), "--task", task, "--mode", "direct"]
    if n_max:
        cmd.extend(["--n-max", str(n_max)])
    
    run_cmd(cmd)

def preprocess(task):
    # print("=========================================")
    # print(f" [3] Preprocessing for: {task}")
    # print("=========================================")
    run_cmd(["python3", str(SCRIPTS_DIR / "preprocess.py"), "--task", task])

def train(task, model):
    # print("=========================================")
    # print(f" [4] Training Models ({task}/{model})")
    # print("=========================================")
    
    # Try running ML training
    # Check if model string implies ML or DL or both
    # ML models: random_forest, svm_rbf, xgboost, lightgbm, catboost
    # DL models: mlp, deep_dnn, resnet_mlp, tabnet, ft_transformer
    
    ml_keywords = ["random", "svm", "xgboost", "lightgbm", "catboost"]
    dl_keywords = ["mlp", "dnn", "resnet", "tabnet", "transformer"]
    
    run_ml = False
    run_dl = False
    
    model_lower = model.lower()
    if model == "all":
        run_ml = True
        run_dl = True
    else:
        # Check partial matches
        models_list = [m.strip() for m in model_lower.split(",")]
        for m in models_list:
            if any(k in m for k in ml_keywords):
                run_ml = True
            if any(k in m for k in dl_keywords):
                run_dl = True
            # If no keyword matched, try running both just in case (e.g. unknown new model)
            if not any(k in m for k in ml_keywords + dl_keywords):
                run_ml = True
                run_dl = True

    if run_ml:
        # print(">> Running ML Training...")
        run_cmd(["python3", str(SCRIPTS_DIR / "train_ml.py"), "--task", task, "--model", model])
        
    if run_dl:
        # print(">> Running DL Training...")
        run_cmd(["python3", str(SCRIPTS_DIR / "train_dl.py"), "--task", task, "--model", model])

def main():
    parser = argparse.ArgumentParser(description="Pipeline Runner")
    parser.add_argument("--task", choices=["tpm", "matmul", "reduction", "all"], required=True, help="Task to run")
    parser.add_argument("--model", type=str, default="all", help="Model to train (e.g. 'svm', 'mlp', 'all')")
    parser.add_argument("--n-max", type=int, default=None, help="Max problem size for data collection (for speed)")
    args = parser.parse_args()
    
    # Check if data exists to decide whether to run steps 1-3
    data_exists = True
    processed_files = [
        DATA_DIR / "tpm_train.csv", DATA_DIR / "tpm_test.csv",
        DATA_DIR / "matmul_train.csv", DATA_DIR / "matmul_test.csv",
        DATA_DIR / "reduction_train.csv", DATA_DIR / "reduction_test.csv"
    ]
    # If specific task is requested, check only that task's files
    if args.task != "all":
        data_exists = (DATA_DIR / f"{args.task}_train.csv").exists() and (DATA_DIR / f"{args.task}_test.csv").exists()
    else:
        # For 'all', check if at least one complete set exists or all? 
        # Let's check if all exist to be safe, otherwise we might need to collecting missing ones.
        # But for simplicity, if ANY data is missing for 'all', we might want to collect?
        # Actually, let's just check if the requested task's data exists.
        data_exists = all(f.exists() for f in processed_files)

    if data_exists and args.n_max is None:
        print(" [Info] Found existing dataset. Skipping Compilation/Collection/Preprocessing.")
        print("        (Pass --n-max <val> to force new data collection)")
    else:
        # 1. Compile
        compile_kernels(args.task)
        
        # 2. Collect Data
        n_max = args.n_max
        if n_max is None:
            # Default defaults if forced to collect but no n_max provided
            # (Though user instructions now imply we use existing data mostly)
            if args.task == "tpm": n_max = 50000
            elif args.task == "matmul": n_max = 4096
            elif args.task == "reduction": n_max = 2000000
            elif args.task == "all": pass # handled in loop below

        tasks = [args.task] if args.task != "all" else ["tpm", "matmul", "reduction"]
        
        for t in tasks:
            current_n_max = n_max
            if args.n_max is None:
                 if t == "tpm": current_n_max = 50000
                 elif t == "matmul": current_n_max = 4096
                 elif t == "reduction": current_n_max = 2000000
            
            collect_data(t, current_n_max)
            preprocess(t)

    # 4. Train
    tasks = [args.task] if args.task != "all" else ["tpm", "matmul", "reduction"]
    for t in tasks:
        train(t, args.model)

if __name__ == "__main__":
    main()
