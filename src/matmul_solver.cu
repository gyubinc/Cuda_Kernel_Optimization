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

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N,
                             int row_offset_tiles) {
  extern __shared__ float tile[];
  float* As = tile;
  float* Bs = tile + blockDim.y * blockDim.x;

  int global_row = (blockIdx.y + row_offset_tiles) * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float acc = 0.0f;
  int tileCount = (N + blockDim.x - 1) / blockDim.x;
  for (int t = 0; t < tileCount; ++t) {
    int a_col = t * blockDim.x + threadIdx.x;
    int b_row = t * blockDim.x + threadIdx.y;
    if (global_row < N && a_col < N)
      As[threadIdx.y * blockDim.x + threadIdx.x] = A[global_row * N + a_col];
    else
      As[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;

    if (b_row < N && col < N)
      Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[b_row * N + col];
    else
      Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
    __syncthreads();

    for (int k = 0; k < blockDim.x; ++k) {
      acc += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
    }
    __syncthreads();
  }

  if (global_row < N && col < N) {
    C[global_row * N + col] = acc;
  }
}

void usage() {
  std::cout << "Usage: ./matmul_solver -n <N> -s <streams> -b <tile>\n"
            << "  -n : matrix dimension (square)\n"
            << "  -s : number of CUDA streams\n"
            << "  -b : tile size (blockDim.x == blockDim.y)\n";
}

struct Config {
  int N = 0;
  int streams = 1;
  int tile = 16;
};

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
      cfg.N = std::stoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) {
      cfg.streams = std::stoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-b") && i + 1 < argc) {
      cfg.tile = std::stoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "-h")) {
      usage();
      std::exit(0);
    } else {
      usage();
      throw std::invalid_argument("Invalid argument");
    }
  }
  if (cfg.N <= 0 || cfg.streams <= 0 || cfg.tile <= 0) {
    usage();
    throw std::invalid_argument("Arguments must be positive");
  }
  return cfg;
}

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);
    int N = cfg.N;
    int streams = cfg.streams;
    int tile = cfg.tile;

    std::cout << "MatMul start | N=" << N << " streams=" << streams
              << " tile=" << tile << std::endl;

    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));

    // Initialize with small values
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));
    CUDA_CHECK(cudaMemset(C, 0, bytes));

    std::vector<cudaStream_t> stream_vec(streams);
    for (int i = 0; i < streams; ++i) {
      CUDA_CHECK(cudaStreamCreate(&stream_vec[i]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    int tilesPerDim = (N + tile - 1) / tile;
    int rowsPerStream = (tilesPerDim + streams - 1) / streams;
    for (int s = 0; s < streams; ++s) {
      int rowTileStart = s * rowsPerStream;
      int rowTileEnd = std::min(tilesPerDim, rowTileStart + rowsPerStream);
      if (rowTileStart >= rowTileEnd) break;
      dim3 block(tile, tile);
      dim3 grid(tilesPerDim, rowTileEnd - rowTileStart);
      size_t shmem = 2 * tile * tile * sizeof(float);
      matmul_tiled<<<grid, block, shmem, stream_vec[s]>>>(
          A, B, C, N, rowTileStart);
    }

    for (int s = 0; s < streams; ++s) {
      CUDA_CHECK(cudaStreamSynchronize(stream_vec[s]));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Elapsed(ms)=" << ms << std::endl;

    for (int s = 0; s < streams; ++s) {
      CUDA_CHECK(cudaStreamDestroy(stream_vec[s]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

