import json
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = 42):
  torch.manual_seed(seed)
  np.random.seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class TabularDataset(Dataset):
  def __init__(self, df: pd.DataFrame, label_col: str = "config_id"):
    self.X = df[["n", "log_n"]].values.astype(np.float32)
    self.y = df[label_col].values.astype(np.int64)

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class MLP3(nn.Module):
  def __init__(self, in_dim: int, num_classes: int):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )

  def forward(self, x):
    return self.net(x)


class DeepDNN(nn.Module):
  def __init__(self, in_dim: int, num_classes: int):
    super().__init__()
    layers = []
    dims = [in_dim, 256, 256, 128, 128, 64]
    for i in range(len(dims) - 1):
      layers.append(nn.Linear(dims[i], dims[i + 1]))
      layers.append(nn.BatchNorm1d(dims[i + 1]))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(0.1))
    layers.append(nn.Linear(dims[-1], num_classes))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


class ResBlock(nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.block = nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
    )
    self.act = nn.ReLU()

  def forward(self, x):
    return self.act(x + self.block(x))


class ResNetMLP(nn.Module):
  def __init__(self, in_dim: int, num_classes: int, hidden: int = 128, depth: int = 3):
    super().__init__()
    self.input = nn.Linear(in_dim, hidden)
    self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
    self.head = nn.Linear(hidden, num_classes)

  def forward(self, x):
    x = torch.relu(self.input(x))
    for blk in self.blocks:
      x = blk(x)
    return self.head(x)


class SimpleTabNet(nn.Module):
  def __init__(self, in_dim: int, num_classes: int, hidden: int = 64, steps: int = 3):
    super().__init__()
    self.initial = nn.Linear(in_dim, hidden)
    self.attentions = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(steps)])
    self.transforms = nn.ModuleList(
        [nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(steps)]
    )
    self.bn = nn.BatchNorm1d(hidden)
    self.head = nn.Linear(hidden, num_classes)

  def forward(self, x):
    x = torch.relu(self.initial(x))
    agg = torch.zeros_like(x)
    for att, tr in zip(self.attentions, self.transforms):
      mask = torch.sigmoid(att(x))
      x = x * mask
      x = tr(x)
      agg = agg + x
    agg = self.bn(agg)
    return self.head(agg)


class FeatureTokenizer(nn.Module):
  def __init__(self, in_dim: int, emb_dim: int):
    super().__init__()
    self.proj = nn.Linear(in_dim, emb_dim)

  def forward(self, x):
    return self.proj(x).unsqueeze(1)  # shape: (B, 1, emb_dim)


class FTTransformer(nn.Module):
  def __init__(self, in_dim: int, num_classes: int, emb_dim: int = 64, num_heads: int = 4, depth: int = 2):
    super().__init__()
    self.tokenizer = FeatureTokenizer(in_dim, emb_dim)
    encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 2)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    self.head = nn.Linear(emb_dim, num_classes)

  def forward(self, x):
    tokens = self.tokenizer(x)  # (B, 1, E)
    tokens = tokens.transpose(0, 1)  # (S=1, B, E)
    encoded = self.encoder(tokens)  # (1, B, E)
    pooled = encoded.mean(dim=0)  # (B, E)
    return self.head(pooled)


def train_one_model(model: nn.Module, train_loader, val_loader, device, epochs=30, lr=1e-3):
  model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  best_acc = 0.0
  best_state = None
  t0 = time.perf_counter()

  for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
      xb = xb.to(device)
      yb = yb.to(device)
      optimizer.zero_grad()
      logits = model(xb)
      loss = criterion(logits, yb)
      loss.backward()
      optimizer.step()

    # 평가
    model.eval()
    correct = total = 0
    with torch.no_grad():
      for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    acc = correct / total if total > 0 else 0
    if acc > best_acc:
      best_acc = acc
      best_state = {k: v.cpu() for k, v in model.state_dict().items()}
  if best_state is not None:
    model.load_state_dict(best_state)
  train_time = time.perf_counter() - t0

  # inference time on validation set
  t1 = time.perf_counter()
  model.eval()
  with torch.no_grad():
    for xb, _ in val_loader:
      xb = xb.to(device)
      _ = model(xb)
  infer_time = time.perf_counter() - t1

  return best_acc, model, train_time, infer_time


def get_loaders(df_train: pd.DataFrame, df_test: pd.DataFrame, batch_size: int = 64, label_col: str = "config_id"):
  train_ds = TabularDataset(df_train, label_col)
  test_ds = TabularDataset(df_test, label_col)
  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
  return train_loader, test_loader


def train_task(task: str) -> Dict[str, float]:
  train_df = pd.read_csv(DATA_DIR / f"{task}_train.csv")
  test_df = pd.read_csv(DATA_DIR / f"{task}_test.csv")
  # 연속된 클래스 id로 remap
  uniques = sorted(train_df["config_id"].unique())
  mapping = {c: i for i, c in enumerate(uniques)}
  train_df = train_df.copy()
  test_df = test_df.copy()
  train_df["label"] = train_df["config_id"].map(mapping)
  test_df["label"] = test_df["config_id"].map(mapping).fillna(-1).astype(int)
  num_classes = len(mapping)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  train_loader, test_loader = get_loaders(train_df, test_df, label_col="label")

  models = {
      "mlp": MLP3(2, num_classes),
      "deep_dnn": DeepDNN(2, num_classes),
      "resnet_mlp": ResNetMLP(2, num_classes),
      "tabnet": SimpleTabNet(2, num_classes),
      "ft_transformer": FTTransformer(2, num_classes),
  }

  results: Dict[str, float] = {}
  for name, model in models.items():
    acc, trained_model, train_time, infer_time = train_one_model(model, train_loader, test_loader, device)
    results[name] = {
        "acc": acc,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
    }
    save_path = MODEL_DIR / f"{task}_{name}.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"[{task}] {name} acc={acc:.4f} train={train_time:.2f}s infer={infer_time:.4f}s saved={save_path.name}")
  return results


def main():
  set_seed(42)
  all_results = {}
  for task in ["tpm", "matmul", "reduction"]:
    all_results[task] = train_task(task)

  out_path = DATA_DIR / "dl_results.json"
  with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
  print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
  main()

