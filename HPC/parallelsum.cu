nvcc filename.cu -o filelname
./filename

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <limits>
#include <cstdlib>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// ---------------- Warp-level reduction for SUM ----------------
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ---------------- GPU SUM Kernel ----------------
__global__ void reduceSum(int* input, unsigned long long* output, int n) {
    __shared__ int shared[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < n) ? input[tid] : 0;

    val = warpReduceSum(val);

    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
        val = warpReduceSum(val);
    }

    if (threadIdx.x == 0) atomicAdd(output, (unsigned long long)val);
}

// ---------------- CPU SUM Function ----------------
long long sequentialSum(const std::vector<int>& data) {
    long long sum = 0;
    for (int val : data) sum += val;
    return sum;
}

// ---------------- MAIN ----------------
int main() {
    long long n = 10000000;  // Input size
    int maxVal = 1000;       // Maximum random value

    std::vector<int> data(n);
    for (long long j = 0; j < n; ++j)
        data[j] = rand() % maxVal;

    int* d_input;
    unsigned long long* d_sum;
    unsigned long long h_sum = 0;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_input, data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // CPU SUM
    auto cpuStart = std::chrono::high_resolution_clock::now();
    long long cpuSum = sequentialSum(data);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuTime = cpuEnd - cpuStart;

    // GPU SUM
    cudaEvent_t start, stop;
    float gpuTimeMs = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduceSum<<<numBlocks, BLOCK_SIZE>>>(d_input, d_sum, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);

    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // Output
    std::cout << "\nResults:\n";
    std::cout << "------------------------------\n";
    std::cout << "CPU Sum: " << cpuSum << "\n";
    std::cout << "GPU Sum: " << h_sum << "\n";
    std::cout << "CPU Time (s): " << std::fixed << std::setprecision(6) << cpuTime.count() << "\n";
    std::cout << "GPU Time (s): " << std::fixed << std::setprecision(6) << gpuTimeMs / 1000.0 << "\n";
    std::cout << "------------------------------\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_sum);

    return 0;
}