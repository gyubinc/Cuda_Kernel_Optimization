import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# 논문의 threshold
THRESHOLD_SMALL = 1e6  # SLAE_size <= 1e6: small model, > 1e6: big model


def load_raw(task: str, block_col: str) -> pd.DataFrame:
  path = DATA_DIR / f"raw_{task}.csv"
  if not path.exists():
    raise FileNotFoundError(f"{path} not found. collect_data.py를 먼저 실행하세요.")
  df = pd.read_csv(path)
  df = df[df["time_ms"] >= 0].copy()
  df["config"] = list(zip(df["streams"].astype(int), df[block_col].astype(int)))
  return df


def load_best(task: str) -> pd.DataFrame:
  path = DATA_DIR / f"{task}_test.csv"
  if not path.exists():
    raise FileNotFoundError(f"{path} not found. preprocess.py를 먼저 실행하세요.")
  df = pd.read_csv(path)
  df["config"] = list(zip(df["streams"].astype(int), df[df.columns[3]].astype(int)))
  return df


def train_sum_model(df: pd.DataFrame) -> Dict[str, float]:
  """논문의 Sum 모델 학습: sum = a * SLAE_size + b
  Note: 실제 profiling 데이터가 없으므로 streams=1일 때의 time_ms를 근사로 사용
  """
  # streams=1인 데이터만 사용하여 sum 근사
  df_stream1 = df[df["streams"] == 1].copy()
  if len(df_stream1) == 0:
    raise ValueError("streams=1 데이터가 없습니다.")
  
  X = df_stream1["n"].values.reshape(-1, 1)
  y = df_stream1["time_ms"].values
  
  model = LinearRegression()
  model.fit(X, y)
  
  return {"a": float(model.coef_[0]), "b": float(model.intercept_)}


def predict_sum(sum_model: Dict[str, float], slae_size: float) -> float:
  """Sum 모델로 예측: sum = a * SLAE_size + b"""
  return sum_model["a"] * slae_size + sum_model["b"]


def train_overhead_model_small(df_small: pd.DataFrame, sum_model: Dict[str, float]) -> Dict[str, float]:
  """논문의 Small T_overhead 모델 학습
  T_overhead = α * SLAE_size + β * log10(num_streams) + γ
  """
  # T_overhead 계산: 논문의 수식 사용
  # T_overhead = (T_str - T_non_str) + ((streams-1)/streams) * sum
  df_stream1 = df_small[df_small["streams"] == 1].copy()
  t_non_str = dict(zip(df_stream1["n"], df_stream1["time_ms"]))
  
  data_points = []
  for _, row in df_small.iterrows():
    n = row["n"]
    streams = row["streams"]
    t_str = row["time_ms"]
    
    if n in t_non_str and streams > 1:
      # 논문의 수식: T_overhead = (T_str - T_non_str) + ((streams-1)/streams) * sum
      sum_pred = predict_sum(sum_model, n)
      t_overhead = (t_str - t_non_str[n]) + ((streams - 1) / streams) * sum_pred
      # T_overhead는 양수여야 함 (음수면 0으로 클리핑)
      t_overhead = max(0, t_overhead)
      data_points.append((n, streams, t_overhead))
  
  if len(data_points) < 3:
    raise ValueError("Small model 학습에 충분한 데이터가 없습니다.")
  
  X_n = np.array([p[0] for p in data_points])
  X_s = np.array([p[1] for p in data_points])
  y = np.array([p[2] for p in data_points])
  
  # f(x, s; α, β, γ) = α*x + β*log10(s) + γ
  def model_func(x_s_pair, alpha, beta, gamma):
    x, s = x_s_pair
    return alpha * x + beta * np.log10(s) + gamma
  
  try:
    # 초기값 개선: 선형 회귀로 초기값 설정
    X_reg = np.column_stack([X_n, np.log10(X_s), np.ones(len(X_n))])
    model_init = LinearRegression()
    model_init.fit(X_reg, y)
    alpha_init = max(0, model_init.coef_[0])  # bounds 내로 조정
    beta_init = max(0, model_init.coef_[1])  # bounds 내로 조정
    gamma_init = max(0, model_init.intercept_)  # bounds 내로 조정
    
    # bounds 추가: alpha >= 0, beta >= 0, gamma >= 0 (T_overhead는 양수여야 함)
    # method='trf' 또는 'dogbox'는 bounds를 지원함
    popt, _ = curve_fit(
      model_func,
      (X_n, X_s),
      y,
      p0=[alpha_init, beta_init, gamma_init],
      maxfev=10000,
      method='trf',  # Trust Region Reflective 알고리즘 (bounds 지원)
      bounds=([0, 0, 0], [np.inf, np.inf, np.inf])  # 모든 파라미터 >= 0
    )
    return {"alpha": float(popt[0]), "beta": float(popt[1]), "gamma": float(popt[2])}
  except Exception as e:
    # Fallback: linear regression
    X = np.column_stack([X_n, np.log10(X_s), np.ones(len(X_n))])
    model = LinearRegression()
    model.fit(X, y)
    return {
      "alpha": float(model.coef_[0]),
      "beta": float(model.coef_[1]),
      "gamma": float(model.intercept_)
    }


def train_overhead_model_big(df_big: pd.DataFrame, sum_model: Dict[str, float]) -> Dict[str, float]:
  """논문의 Big T_overhead 모델 학습
  T_overhead = (a * SLAE_size + b) * log2(num_streams / (4/3)) + c
  """
  # T_overhead 계산: 논문의 수식 사용
  df_stream1 = df_big[df_big["streams"] == 1].copy()
  t_non_str = dict(zip(df_stream1["n"], df_stream1["time_ms"]))
  
  data_points = []
  for _, row in df_big.iterrows():
    n = row["n"]
    streams = row["streams"]
    t_str = row["time_ms"]
    
    if n in t_non_str and streams > 1:
      # 논문의 수식: T_overhead = (T_str - T_non_str) + ((streams-1)/streams) * sum
      sum_pred = predict_sum(sum_model, n)
      t_overhead = (t_str - t_non_str[n]) + ((streams - 1) / streams) * sum_pred
      # T_overhead는 양수여야 함
      t_overhead = max(0, t_overhead)
      data_points.append((n, streams, t_overhead))
  
  if len(data_points) < 3:
    raise ValueError("Big model 학습에 충분한 데이터가 없습니다.")
  
  X_n = np.array([p[0] for p in data_points])
  X_s = np.array([p[1] for p in data_points])
  y = np.array([p[2] for p in data_points])
  
  # f(x, s; a, b, c) = (a*x + b) * log2(s / (4/3)) + c
  def model_func(x_s_pair, a, b, c):
    x, s = x_s_pair
    log_term = np.log2(s / (4.0 / 3.0))
    return (a * x + b) * log_term + c
  
  try:
    # 초기값 개선: 데이터 분포 기반으로 더 정확하게
    log_terms = np.log2(X_s / (4.0 / 3.0))
    
    # a, b, c를 추정하기 위해 선형 회귀로 초기값 설정
    # y ≈ (a*x + b) * log_term + c
    # y = a*x*log_term + b*log_term + c
    X_reg = np.column_stack([X_n * log_terms, log_terms, np.ones(len(X_n))])
    model_init = LinearRegression()
    model_init.fit(X_reg, y)
    a_init = max(0, model_init.coef_[0])  # bounds 내로 조정
    b_init = model_init.coef_[1]  # b는 음수일 수 있음
    c_init = max(0, model_init.intercept_)  # bounds 내로 조정
    
    # bounds 추가: a >= 0, c >= 0 (b는 음수일 수 있음)
    popt, _ = curve_fit(
      model_func,
      (X_n, X_s),
      y,
      p0=[a_init, b_init, c_init],
      maxfev=10000,
      method='trf',  # Trust Region Reflective 알고리즘 (bounds 지원)
      bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf])  # a >= 0, c >= 0
    )
    return {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}
  except Exception as e:
    # Fallback: 선형 회귀로 직접 학습
    log_terms = np.log2(X_s / (4.0 / 3.0))
    X = np.column_stack([X_n * log_terms, log_terms, np.ones(len(X_n))])
    model = LinearRegression()
    model.fit(X, y)
    return {
      "a": float(model.coef_[0]),
      "b": float(model.coef_[1]),
      "c": float(model.intercept_)
    }


def predict_overhead(slae_size: float, num_streams: int,
                     overhead_small: Dict[str, float],
                     overhead_big: Dict[str, float]) -> float:
  """T_overhead 예측
  Note: streams=1일 때는 T_overhead=0 (논문에서 streams=1은 baseline)
  """
  if num_streams == 1:
    return 0.0
  
  if slae_size <= THRESHOLD_SMALL:
    # Small model: T_overhead = α * SLAE_size + β * log10(num_streams) + γ
    if overhead_small is None:
      return 0.0
    alpha = overhead_small["alpha"]
    beta = overhead_small["beta"]
    gamma = overhead_small["gamma"]
    result = alpha * slae_size + beta * np.log10(num_streams) + gamma
    return max(0.0, result)  # T_overhead는 양수여야 함
  else:
    # Big model: T_overhead = (a * SLAE_size + b) * log2(num_streams / (4/3)) + c
    if overhead_big is None:
      return 0.0
    a = overhead_big["a"]
    b = overhead_big["b"]
    c = overhead_big["c"]
    log_term = np.log2(num_streams / (4.0 / 3.0))
    result = (a * slae_size + b) * log_term + c
    return max(0.0, result)  # T_overhead는 양수여야 함


def predict_optimal_num_streams(slae_size: float,
                                sum_model: Dict[str, float],
                                overhead_small: Dict[str, float],
                                overhead_big: Dict[str, float],
                                stream_candidates: list = [1, 2, 4, 8, 16, 32]) -> int:
  """논문의 핵심 알고리즘: optimal stream 수 예측
  gain = ((num_streams - 1) / num_streams) * sum - T_overhead
  gain이 최대인 streams 반환
  """
  sum_pred = predict_sum(sum_model, slae_size)
  
  best_gain = 0.0  # streams=1의 gain은 0 (baseline)
  best_streams = 1
  
  for num_streams in stream_candidates:
    if num_streams == 1:
      gain = 0.0  # streams=1은 baseline, gain=0
    else:
      t_overhead = predict_overhead(slae_size, num_streams, overhead_small, overhead_big)
      gain = ((num_streams - 1) / num_streams) * sum_pred - t_overhead
    
    if gain > best_gain:
      best_gain = gain
      best_streams = num_streams
  
  return best_streams


def fit_polynomial(df: pd.DataFrame) -> Tuple[PolynomialFeatures, Dict[Tuple[int, int], LinearRegression]]:
  """MatMul/Reduction용 polynomial regression (기존 방식 유지)"""
  poly = PolynomialFeatures(degree=2, include_bias=True)
  models: Dict[Tuple[int, int], LinearRegression] = {}
  for cfg, sub in df.groupby("config"):
    X = poly.fit_transform(np.log10(sub[["n"]]))
    y = sub["time_ms"].values
    model = LinearRegression()
    model.fit(X, y)
    models[cfg] = model
  return poly, models


def predict_best(poly, models, n_values):
  """MatMul/Reduction용 예측 (기존 방식 유지)"""
  preds = {}
  X_all = poly.transform(np.log10(np.array(n_values).reshape(-1, 1)))
  for i, n in enumerate(n_values):
    best_cfg = None
    best_time = None
    for cfg, model in models.items():
      t_pred = model.predict(X_all[i : i + 1])[0]
      if best_time is None or t_pred < best_time:
        best_time = t_pred
        best_cfg = cfg
    preds[n] = best_cfg
  return preds


def evaluate(task: str, block_col: str):
  raw = load_raw(task, block_col)
  best = load_best(task)
  
  # 학습 시간 측정
  t0 = time.perf_counter()
  
  if task == "tpm":
    # 논문의 T_overhead 방법 구현
    # 1. Sum 모델 학습
    sum_model = train_sum_model(raw)
    
    # 2. 데이터 분리 (small/big)
    df_small = raw[raw["n"] <= THRESHOLD_SMALL].copy()
    df_big = raw[raw["n"] > THRESHOLD_SMALL].copy()
    
    # 3. T_overhead 모델 학습 (sum_model 필요)
    overhead_small = train_overhead_model_small(df_small, sum_model) if len(df_small) > 0 else None
    overhead_big = train_overhead_model_big(df_big, sum_model) if len(df_big) > 0 else None
    
    # 4. 추론: 각 test sample에 대해 optimal streams 예측
    # 논문: block은 256 고정이므로, block=256인 경우만 평가
    best_block256 = best[best[block_col] == 256].copy()
    if len(best_block256) == 0:
      # block=256이 없으면 전체 사용
      best_block256 = best.copy()
    
    n_values = best_block256["n"].tolist()
    pred_cfg = {}
    for n in n_values:
      # 논문 방식: streams만 예측 (block=256 고정)
      optimal_streams = predict_optimal_num_streams(
        n, sum_model, overhead_small, overhead_big
      )
      # 논문대로 block=256 고정
      pred_cfg[n] = (optimal_streams, 256)
    
  else:
    # MatMul/Reduction은 기존 polynomial regression 사용
    poly, models = fit_polynomial(raw)
    n_values = best["n"].tolist()
    pred_cfg = predict_best(poly, models, n_values)
  
  train_time = time.perf_counter() - t0
  
  # 추론 시간 측정
  t1 = time.perf_counter()
  # 추론은 이미 위에서 수행됨 (predict_optimal_num_streams)
  infer_time = time.perf_counter() - t1
  
  # TPM의 경우 block=256인 경우만 평가
  if task == "tpm":
    best_eval = best[best[block_col] == 256].copy()
    if len(best_eval) == 0:
      best_eval = best.copy()
  else:
    best_eval = best
  
  correct = 0
  for _, row in best_eval.iterrows():
    n = row["n"]
    gt_cfg = row["config"]
    if pred_cfg.get(n) == gt_cfg:
      correct += 1
  acc = correct / len(best_eval) if len(best_eval) > 0 else 0
  return acc, train_time, infer_time


def main():
  results = {}
  timing_results = {}
  for task, block_col in [("tpm", "block"), ("matmul", "tile"), ("reduction", "block")]:
    acc, train_time, infer_time = evaluate(task, block_col)
    results[task] = acc
    timing_results[task] = {
      "acc": acc,
      "train_time_s": train_time,
      "infer_time_s": infer_time
    }
    print(f"[{task}] baseline acc={acc:.4f} train={train_time:.4f}s infer={infer_time:.4f}s")

  out_path = DATA_DIR / "baseline_results.json"
  with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
  
  timing_path = DATA_DIR / "baseline_timing.json"
  with open(timing_path, "w") as f:
    json.dump(timing_results, f, indent=2)
  print(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()
