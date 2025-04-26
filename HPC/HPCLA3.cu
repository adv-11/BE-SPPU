#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <limits>
#include <cstdlib>

#define CUDA_CORES 768  // GTX 1050 Ti CUDA cores
#define BLOCK_SIZE 256  // Optimal block size
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

// ---------------- Warp-level reduction for MIN ----------------
__inline__ __device__ int warpReduceMin(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ---------------- Warp-level reduction for MAX ----------------
__inline__ __device__ int warpReduceMax(int val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
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

// ---------------- GPU MIN Kernel ----------------
__global__ void reduceMin(int* input, int* output, int n) {
    __shared__ int shared[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < n) ? input[tid] : INT_MAX;

    val = warpReduceMin(val);

    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : INT_MAX;
        val = warpReduceMin(val);
    }

    if (threadIdx.x == 0) atomicMin(output, val);
}

// ---------------- GPU MAX Kernel ----------------
__global__ void reduceMax(int* input, int* output, int n) {
    __shared__ int shared[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < n) ? input[tid] : INT_MIN;

    val = warpReduceMax(val);

    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : INT_MIN;
        val = warpReduceMax(val);
    }

    if (threadIdx.x == 0) atomicMax(output, val);
}

// ---------------- CPU Functions ----------------
long long sequentialSum(const std::vector<int>& data) {
    long long sum = 0;
    for (int val : data) sum += val;
    return sum;
}

int sequentialMin(const std::vector<int>& data) {
    int minVal = std::numeric_limits<int>::max();
    for (int val : data) minVal = std::min(minVal, val);
    return minVal;
}

int sequentialMax(const std::vector<int>& data) {
    int maxVal = std::numeric_limits<int>::min();
    for (int val : data) maxVal = std::max(maxVal, val);
    return maxVal;
}

double sequentialAverage(const std::vector<int>& data) {
    return static_cast<double>(sequentialSum(data)) / data.size();
}

// ---------------- MAIN ----------------
int main() {
    std::vector<long long> sizes = {10000000, 8000000, 5000000, 4000000};
    std::vector<int> maxValues = {1000, 2000, 3000, 4000};

 std::cout << "\n";
 std::cout << "Name: Gaaytri KUrulkar Roll no: 41039 CLass: BE A\n";
 std::cout << "\n";
 std::cout << "\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "| Input Size | Max Value | CPU Sum     | GPU Sum     | CPU Time (s) | GPU Time (s) | Speedup | Efficiency | CPU Min | GPU Min | CPU Max | GPU Max | CPU Avg | GPU Avg |\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";

    for (size_t i = 0; i < sizes.size(); i++) {
        long long n = sizes[i];
        int maxVal = maxValues[i];
        std::vector<int> data(n);

        for (long long j = 0; j < n; ++j)
            data[j] = rand() % maxVal;

        int* d_input;
        unsigned long long* d_sum;
        int* d_min;
        int* d_max;
        unsigned long long h_sum = 0;
        int h_min = INT_MAX;
        int h_max = INT_MIN;

        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_min, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_input, data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpy(d_min, &h_min, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_max, &h_max, sizeof(int), cudaMemcpyHostToDevice));

        int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // CPU
        auto cpuStart = std::chrono::high_resolution_clock::now();
        long long cpuSum = sequentialSum(data);
        int cpuMin = sequentialMin(data);
        int cpuMax = sequentialMax(data);
        double cpuAvg = sequentialAverage(data);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpuTime = cpuEnd - cpuStart;

        // GPU
        cudaEvent_t start, stop;
        float gpuTimeMs = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        reduceSum<<<numBlocks, BLOCK_SIZE>>>(d_input, d_sum, n);
        reduceMin<<<numBlocks, BLOCK_SIZE>>>(d_input, d_min, n);
        reduceMax<<<numBlocks, BLOCK_SIZE>>>(d_input, d_max, n);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTimeMs, start, stop);

        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost));

        double gpuAvg = static_cast<double>(h_sum) / n;
        double speedup = cpuTime.count() / (gpuTimeMs / 1000.0);
        double efficiency = speedup / CUDA_CORES;

        std::cout << "| " << std::setw(10) << n
                  << " | " << std::setw(9) << maxVal
                  << " | " << std::setw(11) << cpuSum
                  << " | " << std::setw(11) << h_sum
                  << " | " << std::setw(12) << std::fixed << std::setprecision(6) << cpuTime.count()
                  << " | " << std::setw(12) << gpuTimeMs / 1000.0
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << speedup
                  << " | " << std::setw(10) << std::fixed << std::setprecision(6) << efficiency
                  << " | " << std::setw(8) << cpuMin
                  << " | " << std::setw(8) << h_min
                  << " | " << std::setw(8) << cpuMax
                  << " | " << std::setw(8) << h_max
                  << " | " << std::setw(8) << std::fixed << std::setprecision(2) << cpuAvg
                  << " | " << std::setw(8) << std::fixed << std::setprecision(2) << gpuAvg
                  << " |\n";

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_sum);
        cudaFree(d_min);
        cudaFree(d_max);
    }

    std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";
    return 0;
}

