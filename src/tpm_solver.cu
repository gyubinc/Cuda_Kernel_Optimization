#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <algorithm>

#define CUDA_CHECK(ans)                                                       \
  {                                                                           \
    cudaError_t err = (ans);                                                  \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)                  \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  }

// Parallel Cyclic Reduction Kernel Step
// Solves: a[i]*x[i-offset] + b[i]*x[i] + c[i]*x[i+offset] = d[i]
// In each step p (offset = 2^p), we eliminate neighbors.
__global__ void pcr_step_kernel(const float* __restrict__ a_in, const float* __restrict__ b_in, 
                                const float* __restrict__ c_in, const float* __restrict__ d_in,
                                float* __restrict__ a_out, float* __restrict__ b_out, 
                                float* __restrict__ c_out, float* __restrict__ d_out,
                                int n, int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int left = i - offset;
  int right = i + offset;

  float a_L = 0.0f, b_L = 1.0f, c_L = 0.0f, d_L = 0.0f;
  float a_R = 0.0f, b_R = 1.0f, c_R = 0.0f, d_R = 0.0f;

  if (left >= 0) {
    a_L = a_in[left];
    b_L = b_in[left];
    c_L = c_in[left];
    d_L = d_in[left];
  }

  if (right < n) {
    a_R = a_in[right];
    b_R = b_in[right];
    c_R = c_in[right];
    d_R = d_in[right];
  }

  float cur_a = a_in[i];
  float cur_b = b_in[i];
  float cur_c = c_in[i];
  float cur_d = d_in[i];

  float inv_b_L = 1.0f / b_L;
  float inv_b_R = 1.0f / b_R;

  float alpha = -cur_a * inv_b_L;
  float gamma = -cur_c * inv_b_R;

  a_out[i] = (left >= 0) ? alpha * a_L : 0.0f;
  c_out[i] = (right < n) ? gamma * c_R : 0.0f;
  
  // New diagonal b'
  // b'_i = b_i + alpha * c_L + gamma * a_R
  b_out[i] = cur_b + ((left >= 0) ? alpha * c_L : 0.0f) + ((right < n) ? gamma * a_R : 0.0f);
  
  // New RHS d'
  // d'_i = d_i + alpha * d_L + gamma * d_R
  d_out[i] = cur_d + ((left >= 0) ? alpha * d_L : 0.0f) + ((right < n) ? gamma * d_R : 0.0f);
}

// Result extraction kernel
__global__ void solve_final_kernel(const float* __restrict__ b_in, const float* __restrict__ d_in, 
                                   float* __restrict__ x_out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x_out[i] = d_in[i] / b_in[i];
  }
}

// Check residual: r = ||Ax - d||^2
__global__ void residual_kernel(const float* a, const float* b, const float* c, const float* d, 
                                const float* x, float* r, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float val = b[i] * x[i];
    if (i > 0) val += a[i] * x[i-1];
    if (i < n - 1) val += c[i] * x[i+1];
    float diff = val - d[i];
    r[i] = diff * diff;
  }
}

void usage() {
  std::cout << "Usage: ./tpm_solver -n <problem_size> -b <block>\n"
            << "  -n : problem size (1e3 ~ 1e8)\n"
            << "  -b : block size (threads per block)\n";
}

struct Config {
  long long n = 0;
  int block = 256;
};

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
      cfg.n = std::stoll(argv[++i]);
    } else if (!std::strcmp(argv[i], "-b") && i + 1 < argc) {
      cfg.block = std::stoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-h")) {
      usage();
      std::exit(0);
    } 
  }
  if (cfg.n <= 0 || cfg.block <= 0) {
    usage();
    throw std::invalid_argument("Arguments must be positive");
  }
  return cfg;
}

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);
    const long long n = cfg.n;
    const int block = cfg.block;

    std::cout << "TPM (PCR) solver start | n=" << n << " block=" << block << std::endl;

    // Use vectors on host for initialization
    std::vector<float> h_a(n), h_b(n), h_c(n), h_d(n);

    // Initialize with a Diagonally Dominant system
    // a=-1, b=4, c=-1. Solution x=1 everywhere => d = -1+4-1 = 2 (edges separate)
    for (size_t i = 0; i < n; ++i) {
      h_a[i] = (i > 0) ? -1.0f : 0.0f;
      h_c[i] = (i < n - 1) ? -1.0f : 0.0f;
      h_b[i] = 4.0f; 
      
      // Expected x = 1.0f
      // d = A * x
      float val = h_b[i] * 1.0f;
      if (i > 0) val += h_a[i] * 1.0f;
      if (i < n - 1) val += h_c[i] * 1.0f;
      h_d[i] = val;
    }

    // Allocate Device memory (Double Buffering)
    float *d_a0, *d_b0, *d_c0, *d_d0;
    float *d_a1, *d_b1, *d_c1, *d_d1;
    float *d_x; 

    size_t bytes = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_a0, bytes));
    CUDA_CHECK(cudaMalloc(&d_b0, bytes));
    CUDA_CHECK(cudaMalloc(&d_c0, bytes));
    CUDA_CHECK(cudaMalloc(&d_d0, bytes));
    
    CUDA_CHECK(cudaMalloc(&d_a1, bytes));
    CUDA_CHECK(cudaMalloc(&d_b1, bytes));
    CUDA_CHECK(cudaMalloc(&d_c1, bytes));
    CUDA_CHECK(cudaMalloc(&d_d1, bytes));

    CUDA_CHECK(cudaMalloc(&d_x, bytes));

    // Copy to device (Buffer 0)
    CUDA_CHECK(cudaMemcpy(d_a0, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b0, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c0, h_c.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d0, h_d.data(), bytes, cudaMemcpyHostToDevice));

    // Pointers to current input/output buffers
    float *in_a = d_a0, *in_b = d_b0, *in_c = d_c0, *in_d = d_d0;
    float *out_a = d_a1, *out_b = d_b1, *out_c = d_c1, *out_d = d_d1;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // PCR Loop
    // Steps P = ceil(log2(n))
    int p_steps = 0;
    while ((1LL << p_steps) < n) p_steps++;

    int grid = (n + block - 1) / block;
    // Limit grid size if necessary for older GPUs, but modern ones handle large grids.
    // However, if n is huge (1e8), grid is ~400,000 which fits in dim3.x (limit 2^31-1 on CC 3.0+)
    
    for (int p = 0; p < p_steps; ++p) {
      int offset = 1 << p;
      pcr_step_kernel<<<grid, block>>>(in_a, in_b, in_c, in_d, 
                                       out_a, out_b, out_c, out_d, n, offset);
      // Swap pointers
      std::swap(in_a, out_a);
      std::swap(in_b, out_b);
      std::swap(in_c, out_c);
      std::swap(in_d, out_d);
    }

    // Final solve
    // Solution is in 'in_d / in_b'.
    solve_final_kernel<<<grid, block>>>(in_b, in_d, d_x, n);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "Elapsed(ms)=" << ms << std::endl;

    // Verify
    // Re-copy original system to in_* (which holds garbage now) to compute residual
    CUDA_CHECK(cudaMemcpy(in_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(in_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(in_c, h_c.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(in_d, h_d.data(), bytes, cudaMemcpyHostToDevice));
    
    float* d_res = out_a; // reuse memory
    residual_kernel<<<grid, block>>>(in_a, in_b, in_c, in_d, d_x, d_res, n);
    
    std::vector<float> h_res(n);
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, bytes, cudaMemcpyDeviceToHost));
    
    double total_res = 0.0;
    int non_zero_count = 0;
    for (int i = 0; i < n; ++i) {
        total_res += h_res[i];
        if (h_res[i] > 1e-20) {
             non_zero_count++;
             if (non_zero_count <= 5) {
                 std::cout << "Residual > 0 at [" << i << "]: " << h_res[i] << std::endl;
             }
        }
    }
    std::cout << "Total Residual SSD: " << total_res << " (Avg: " << total_res/n << ")" << std::endl;
    std::cout << "Non-zero residuals: " << non_zero_count << std::endl;

    // Debug: Print solution head/tail
    std::vector<float> h_sol(n);
    CUDA_CHECK(cudaMemcpy(h_sol.data(), d_x, bytes, cudaMemcpyDeviceToHost));
    
    std::cout << "Solution snippets (Expected 1.0):" << std::endl;
    for (int i = 0; i < std::min((long long)5, n); ++i) {
        std::cout << "x[" << i << "] = " << h_sol[i] << std::endl;
    }
    if (n > 10) std::cout << "..." << std::endl;
    for (int i = std::max((long long)0, n - 5); i < n; ++i) {
        std::cout << "x[" << i << "] = " << h_sol[i] << std::endl;
    }

    
    // Clean up
    CUDA_CHECK(cudaFree(d_a0)); CUDA_CHECK(cudaFree(d_b0)); CUDA_CHECK(cudaFree(d_c0)); CUDA_CHECK(cudaFree(d_d0));
    CUDA_CHECK(cudaFree(d_a1)); CUDA_CHECK(cudaFree(d_b1)); CUDA_CHECK(cudaFree(d_c1)); CUDA_CHECK(cudaFree(d_d1));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
