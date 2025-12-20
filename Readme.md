## Prerequisites
- NVIDIA GPU (Compute Capability 7.0 or higher recommended)
- CUDA Toolkit (Tested with CUDA 11.8)
- SLURM Workload Manager (for cluster execution via `srun`)

## Compilation
To compile the solver, use `nvcc`. Ensure you are using a CUDA version compatible with your driver.

```bash
# Example using CUDA 11.8
/usr/local/cuda-11.8/bin/nvcc -o tpm_solver src/tpm_solver.cu -O3 -arch=sm_70
```

## Execution
Run the solver using `srun` to allocate GPU resources.

### Usage
```bash
./tpm_solver -n <PROBLEM_SIZE> -b <BLOCK_SIZE>
```

- `-n`: Size of the tridiagonal system (e.g., 1024, 1048576).
- `-b`: CUDA block size (threads per block, default: 256).

### Example Command
 To run a test with $N=1,024,000$ and block size 256:

```bash
srun -t 00:10:00 --gres=gpu:1 ./tpm_solver -n 1048576 -b 256
```
