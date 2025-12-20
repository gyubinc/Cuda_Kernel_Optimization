import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def build_config_id(df: pd.DataFrame, block_col: str) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], int]]:
  df = df.copy()
  df["config"] = list(zip(df["streams"].astype(int), df[block_col].astype(int)))
  configs = sorted(df["config"].unique())
  config_to_id = {cfg: idx for idx, cfg in enumerate(configs)}
  df["config_id"] = df["config"].map(config_to_id)
  return df, config_to_id


def label_best(df: pd.DataFrame) -> pd.DataFrame:
  # 음수나 실패 측정값 제거
  df = df[df["time_ms"] >= 0].copy()
  best_indices = df.groupby("n")["time_ms"].idxmin()
  best = df.loc[best_indices].copy()
  best["optimal"] = 1
  return best


def process_task(task: str, block_col: str):
  raw_path = DATA_DIR / f"raw_{task}.csv"
  if not raw_path.exists():
    raise FileNotFoundError(f"{raw_path} not found. 먼저 collect_data.py를 실행하세요.")

  df = pd.read_csv(raw_path)
  df, config_to_id = build_config_id(df, block_col)
  best = label_best(df)
  best["log_n"] = np.log10(best["n"])

  counts = best["config_id"].value_counts()
  stratify_col = best["config_id"] if counts.min() >= 2 else None
  desired_test = max(0.2, min(0.5, 20 / len(best))) if len(best) > 0 else 0.2

  train, test = train_test_split(
      best,
      test_size=desired_test,
      random_state=42,
      stratify=stratify_col,
  )

  keep_cols = ["n", "log_n", "streams", block_col, "config_id", "time_ms"]
  train[keep_cols].to_csv(DATA_DIR / f"{task}_train.csv", index=False)
  test[keep_cols].to_csv(DATA_DIR / f"{task}_test.csv", index=False)

  map_path = DATA_DIR / f"{task}_config_map.csv"
  with open(map_path, "w") as f:
    f.write("config_id,streams," + block_col + "\n")
    for (streams, block), cid in config_to_id.items():
      f.write(f"{cid},{streams},{block}\n")

  print(f"[{task}] train={len(train)} test={len(test)} configs={len(config_to_id)}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--task",
      choices=["tpm", "matmul", "reduction", "all"],
      default="all",
      help="전처리할 데이터셋 선택",
  )
  args = parser.parse_args()

  if args.task in ("tpm", "all"):
    process_task("tpm", "block")
  if args.task in ("matmul", "all"):
    process_task("matmul", "tile")
  if args.task in ("reduction", "all"):
    process_task("reduction", "block")


if __name__ == "__main__":
  main()

