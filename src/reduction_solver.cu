#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>

#define CUDA_CHECK(ans)                                                       \
  {                                                                           \
    cudaError_t err = (ans);                                                  \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)                  \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  }

__global__ void init_data(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    data[i] = 1.0f;  // 단순 합산을 위해 상수로 초기화
  }
}

__global__ void reduce_kernel(const float* in, float* out, int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  float sum = 0.0f;
  if (idx < n) sum += in[idx];
  if (idx + blockDim.x < n) sum += in[idx + blockDim.x];
  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
  }
}

void usage() {
  std::cout << "Usage: ./reduction_solver -n <N> -s <streams> -b <block>\n"
            << "  -n : vector length\n"
            << "  -s : number of CUDA streams\n"
            << "  -b : block size (threads per block)\n";
}

struct Config {
  long long n = 0;
  int streams = 1;
  int block = 256;
};

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
      cfg.n = std::stoll(argv[++i]);
    } else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) {
      cfg.streams = std::stoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-b") && i + 1 < argc) {
      cfg.block = std::stoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-h")) {
      usage();
      std::exit(0);
    } else {
      usage();
      throw std::invalid_argument("Invalid argument");
    }
  }
  if (cfg.n <= 0 || cfg.streams <= 0 || cfg.block <= 0) {
    usage();
    throw std::invalid_argument("Arguments must be positive");
  }
  return cfg;
}

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);
    long long n = cfg.n;
    int streams = cfg.streams;
    int block = cfg.block;

    std::cout << "Reduction start | n=" << n << " streams=" << streams
              << " block=" << block << std::endl;

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));

    int init_grid = (n + block - 1) / block;
    init_grid = std::min(init_grid, 65535);
    init_data<<<init_grid, block>>>(d_in, n);
    CUDA_CHECK(cudaGetLastError());

    long long chunk = (n + streams - 1) / streams;
    long long max_chunk = std::min(chunk, n);
    int max_blocks = static_cast<int>((max_chunk + block * 2 - 1) / (block * 2));
    size_t workspace_elems = static_cast<size_t>(streams) * max_blocks * 2;

    float* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_elems * sizeof(float)));

    std::vector<float> h_results(streams, 0.0f);
    float* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_results, streams * sizeof(float)));

    std::vector<cudaStream_t> stream_vec(streams);
    for (int i = 0; i < streams; ++i) {
      CUDA_CHECK(cudaStreamCreate(&stream_vec[i]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int s = 0; s < streams; ++s) {
      long long offset = s * chunk;
      if (offset >= n) break;
      int len = static_cast<int>(std::min<long long>(chunk, n - offset));
      const float* curr_in = d_in + offset;

      int curr_blocks = (len + block * 2 - 1) / (block * 2);
      float* partialA = d_workspace + s * max_blocks;
      float* partialB = d_workspace + streams * max_blocks + s * max_blocks;
      float* curr_out = partialA;
      int curr_len = len;

      while (true) {
        reduce_kernel<<<curr_blocks, block, block * sizeof(float), stream_vec[s]>>>(
            curr_in, curr_out, curr_len);
        CUDA_CHECK(cudaGetLastError());
        if (curr_blocks <= 1) break;
        curr_len = curr_blocks;
        curr_in = curr_out;
        curr_blocks = (curr_len + block * 2 - 1) / (block * 2);
        curr_out = (curr_out == partialA) ? partialB : partialA;
      }
      CUDA_CHECK(cudaMemcpyAsync(d_results + s, curr_out, sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream_vec[s]));
    }

    for (int s = 0; s < streams; ++s) {
      CUDA_CHECK(cudaStreamSynchronize(stream_vec[s]));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, streams * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float total = 0.0f;
    for (float v : h_results) total += v;
    std::cout << "Elapsed(ms)=" << ms << " sum=" << total << std::endl;

    for (int s = 0; s < streams; ++s) {
      CUDA_CHECK(cudaStreamDestroy(stream_vec[s]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_results));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}


