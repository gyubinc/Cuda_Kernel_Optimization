import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = BASE_DIR / "bin"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def run_kernel(binary: Path, args: List[str], mode: str) -> Tuple[bool, float, str]:
  if mode == "direct":
    cmd = [str(binary)] + args
  else:
    cmd = [
        "srun",
        "--ntasks=1",
        "--gres=gpu:1",
        str(binary),
    ] + args
  try:
    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True, cwd=BASE_DIR
    )
    stdout = result.stdout.strip()
    elapsed_ms = None
    for line in stdout.splitlines():
      if "Elapsed(ms)=" in line:
        try:
          val = line.split("Elapsed(ms)=")[-1].strip().split()[0]
          elapsed_ms = float(val)
          break
        except ValueError:
          continue
    if elapsed_ms is None:
      return False, -1.0, stdout
    return True, elapsed_ms, stdout
  except subprocess.CalledProcessError as e:
    return False, -1.0, f"Command failed: {e.stderr}"


def tpm_sizes(n_min: int = None, n_max: int = None) -> List[int]:
  bases = [1.0, 2.5, 4.0, 5.0, 7.5, 8.0]
  sizes = []
  for exp in range(3, 8):
    factor = 10 ** exp
    for b in bases:
      sizes.append(int(b * factor))
  sizes.append(int(1e8))
  if n_min is not None:
    sizes = [x for x in sizes if x >= n_min]
  if n_max is not None:
    sizes = [x for x in sizes if x <= n_max]
  return sizes


def matmul_sizes(n_min: int = None, n_max: int = None) -> List[int]:
  sizes = list(range(512, 8192 + 1, 256))
  if n_min is not None:
    sizes = [x for x in sizes if x >= n_min]
  if n_max is not None:
    sizes = [x for x in sizes if x <= n_max]
  return sizes


def reduction_sizes(n_min: int = None, n_max: int = None) -> List[int]:
  vals = []
  for i in range(20):
    n = int(round(10 ** (5 + 3 * i / 19)))
    vals.append(n)
  vals = sorted(list(set(vals)))
  if n_min is not None:
    vals = [x for x in vals if x >= n_min]
  if n_max is not None:
    vals = [x for x in vals if x <= n_max]
  return vals


def write_rows(path: Path, rows: List[tuple], append: bool):
  mode = "a" if append and path.exists() else "w"
  with open(path, mode, newline="") as f:
    writer = csv.writer(f)
    if mode == "w":
      writer.writerow(rows[0])
      writer.writerows(rows[1:])
    else:
      # skip header row
      writer.writerows(rows[1:])


def collect_tpm(mode: str, n_min: int, n_max: int, append: bool):
  binary = BIN_DIR / "tpm_solver"
  streams = [1, 2, 4, 8, 16, 32]
  # 논문 재현: block 256 고정
  blocks = [256]
  rows = [("task", "n", "streams", "block", "time_ms")]
  for n in tpm_sizes(n_min, n_max):
    for s in streams:
      for b in blocks:
        ok, elapsed, log = run_kernel(
            binary, ["-n", str(n), "-s", str(s), "-b", str(b)], mode
        )
        if not ok:
          print(f"[TPM] failed n={n} s={s} b={b}: {log}", file=sys.stderr)
        rows.append(("tpm", n, s, b, elapsed))
  write_rows(DATA_DIR / "raw_tpm.csv", rows, append)


def collect_matmul(mode: str, n_min: int, n_max: int, append: bool):
  binary = BIN_DIR / "matmul_solver"
  stream_list = list(range(1, 16 + 1))
  tiles = [8, 16, 32]
  rows = [("task", "n", "streams", "tile", "time_ms")]
  for n in matmul_sizes(n_min, n_max):
    for s in stream_list:
      for t in tiles:
        ok, elapsed, log = run_kernel(
            binary, ["-n", str(n), "-s", str(s), "-b", str(t)], mode
        )
        if not ok:
          print(f"[MatMul] failed n={n} s={s} tile={t}: {log}", file=sys.stderr)
        rows.append(("matmul", n, s, t, elapsed))
  write_rows(DATA_DIR / "raw_matmul.csv", rows, append)


def collect_reduction(mode: str, n_min: int, n_max: int, append: bool):
  binary = BIN_DIR / "reduction_solver"
  stream_list = list(range(1, 32 + 1))
  blocks = [64, 128, 256, 512, 1024]  # 파워오브투만 사용해 reduce 정확성 확보
  rows = [("task", "n", "streams", "block", "time_ms")]
  for n in reduction_sizes(n_min, n_max):
    for s in stream_list:
      for b in blocks:
        ok, elapsed, log = run_kernel(
            binary, ["-n", str(n), "-s", str(s), "-b", str(b)], mode
        )
        if not ok:
          print(f"[Reduction] failed n={n} s={s} b={b}: {log}", file=sys.stderr)
        rows.append(("reduction", n, s, b, elapsed))
  write_rows(DATA_DIR / "raw_reduction.csv", rows, append)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--task",
      choices=["tpm", "matmul", "reduction", "all"],
      default="all",
      help="어떤 작업을 수집할지 선택",
  )
  parser.add_argument(
      "--mode",
      choices=["srun", "direct"],
      default="srun",
      help="srun 호출 여부 (이미 srun 할당 안에서 실행 시 direct 권장)",
  )
  parser.add_argument("--n-min", type=int, default=None, help="사이즈 하한")
  parser.add_argument("--n-max", type=int, default=None, help="사이즈 상한")
  parser.add_argument("--append", action="store_true", help="기존 파일에 이어쓰기")
  args = parser.parse_args()

  if args.task in ("tpm", "all"):
    collect_tpm(args.mode, args.n_min, args.n_max, args.append)
  if args.task in ("matmul", "all"):
    collect_matmul(args.mode, args.n_min, args.n_max, args.append)
  if args.task in ("reduction", "all"):
    collect_reduction(args.mode, args.n_min, args.n_max, args.append)


if __name__ == "__main__":
  main()

