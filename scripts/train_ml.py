import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_dataset(task: str) -> pd.DataFrame:
  path = DATA_DIR / f"{task}_train.csv"
  test_path = DATA_DIR / f"{task}_test.csv"
  if not path.exists() or not test_path.exists():
    raise FileNotFoundError("preprocess.py 실행 후 train/test CSV가 필요합니다.")
  train_df = pd.read_csv(path)
  test_df = pd.read_csv(test_path)
  return train_df, test_df


def build_models(num_classes: int) -> Dict[str, object]:
  try:
    from xgboost import XGBClassifier
  except ImportError:
    XGBClassifier = None
  try:
    from lightgbm import LGBMClassifier
  except ImportError:
    LGBMClassifier = None
  try:
    from catboost import CatBoostClassifier
  except ImportError:
    CatBoostClassifier = None

  models: Dict[str, object] = {
      "random_forest": RandomForestClassifier(
          n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
      ),
      "svm_rbf": Pipeline(
          [
              ("scaler", StandardScaler()),
              ("clf", SVC(kernel="rbf", gamma="scale", C=10, random_state=42)),
          ]
      ),
  }

  if XGBClassifier:
    models["xgboost"] = XGBClassifier(
        max_depth=6,
        n_estimators=300,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=num_classes,
        random_state=42,
    )
  if LGBMClassifier:
    models["lightgbm"] = LGBMClassifier(
        num_leaves=31,
        n_estimators=400,
        learning_rate=0.05,
        random_state=42,
    )
  if CatBoostClassifier:
    models["catboost"] = CatBoostClassifier(
        depth=8,
        learning_rate=0.1,
        iterations=400,
        verbose=False,
        random_seed=42,
    )
  return models


def remap_labels(y_train: np.ndarray, y_test: np.ndarray):
  uniques = np.unique(y_train)
  mapping = {c: i for i, c in enumerate(sorted(uniques))}
  y_train_mapped = np.array([mapping[y] for y in y_train], dtype=int)
  y_test_mapped = np.array([mapping.get(y, -1) for y in y_test], dtype=int)
  return y_train_mapped, y_test_mapped, len(uniques)


def train_and_eval(task: str) -> Dict[str, Dict[str, float]]:
  train_df, test_df = load_dataset(task)
  feature_cols = ["n", "log_n"]
  X_train = train_df[feature_cols].values
  y_train = train_df["config_id"].values
  X_test = test_df[feature_cols].values
  y_test = test_df["config_id"].values

  y_train, y_test, num_classes = remap_labels(y_train, y_test)
  test_mask = y_test != -1
  X_test = X_test[test_mask]
  y_test = y_test[test_mask]
  models = build_models(num_classes)

  scores: Dict[str, Dict[str, float]] = {}
  for name, model in models.items():
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = model.predict(X_test)
    if preds.ndim > 1:
      preds = np.argmax(preds, axis=1)
    infer_time = time.perf_counter() - t1

    acc = accuracy_score(y_test, preds)
    scores[name] = {
        "acc": acc,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
    }
    print(
        f"[{task}] {name} acc={acc:.4f} train={train_time:.4f}s infer={infer_time:.4f}s "
        f"| n_test={len(y_test)} classes={num_classes}"
    )
  return scores


def main():
  results: Dict[str, Dict[str, float]] = {}
  for task in ["tpm", "matmul", "reduction"]:
    results[task] = train_and_eval(task)

  out_path = DATA_DIR / "ml_results.json"
  with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
  print(json.dumps(results, indent=2))


if __name__ == "__main__":
  main()

