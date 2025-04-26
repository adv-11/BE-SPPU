#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <cstdlib>
#include <fstream>  


#define BLOCK_SIZE 16
#define CUDA_CORES 768  // GTX 1050 Ti CUDA cores

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// ------------------------ VECTOR ADDITION ------------------------
__global__ void vectorAddKernel(const int* A, const int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void cpuVectorAdd(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C) {
    for (size_t i = 0; i < A.size(); ++i)
        C[i] = A[i] + B[i];
}

// ------------------------ MATRIX MULTIPLICATION ------------------------
__global__ void matrixMulKernel(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void cpuMatrixMul(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// ------------------------ MAIN ------------------------
int main() {
	
    std::vector<int> vectorSizes = {100000,350000, 500000,800000};
    std::vector<int> matrixSizes = {128, 256,450,780};

    std::cout << "\n\nName: Gayatri Kurulkar  Roll No: 41039  Class: BE A\n\n";

    // ------------------------ VECTOR ADDITION TABLE ------------------------
    std::cout << "====================================================================================================================\n";
    std::cout << "                                     VECTOR ADDITION (CPU vs GPU)                                                  \n";
    std::cout << "====================================================================================================================\n";
    std::cout << "| Input Size | CPU Time (s) | GPU Time (s) | Speedup | Efficiency | Output Sample |\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    for (int size : vectorSizes) {
        std::vector<int> A(size), B(size), C_cpu(size), C_gpu(size);
        for (int i = 0; i < size; ++i) {
            A[i] = rand() % 100;
            B[i] = rand() % 100;
        }

        auto startCPU = std::chrono::high_resolution_clock::now();
        cpuVectorAdd(A, B, C_cpu);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpuTime = endCPU - startCPU;

        int *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_B, size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_C, size * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), size * sizeof(int), cudaMemcpyHostToDevice));

        float gpuTime = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vectorAddKernel<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, size);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);

        CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, size * sizeof(int), cudaMemcpyDeviceToHost));

        double speedup = cpuTime.count() / (gpuTime / 1000.0);
        double efficiency = speedup / CUDA_CORES;

        std::cout << "| " << std::setw(10) << size
                  << " | " << std::setw(12) << std::fixed << std::setprecision(6) << cpuTime.count()
                  << " | " << std::setw(12) << gpuTime / 1000.0
                  << " | " << std::setw(7) << speedup
                  << " | " << std::setw(10) << efficiency
                  << " | " << std::setw(13) << C_gpu[size / 2] << " |\n";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    std::cout << "------------------------------------------------------------------------------------\n\n";

    // ------------------------ MATRIX MULTIPLICATION TABLE ------------------------
    std::cout << "====================================================================================================================\n";
    std::cout << "                                  MATRIX MULTIPLICATION (CPU vs GPU)                                                \n";
    std::cout << "====================================================================================================================\n";
    std::cout << "| Matrix Size | CPU Time (s) | GPU Time (s) | Speedup | Efficiency | Output Sample |\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    for (int N : matrixSizes) {
        int size = N * N;
        std::vector<int> A(size), B(size), C_cpu(size), C_gpu(size);
        for (int i = 0; i < size; ++i) {
            A[i] = rand() % 100;
            B[i] = rand() % 100;
        }

        auto startCPU = std::chrono::high_resolution_clock::now();
        cpuMatrixMul(A, B, C_cpu, N);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpuTime = endCPU - startCPU;

        int *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_B, size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_C, size * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_A, A.data(), size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), size * sizeof(int), cudaMemcpyHostToDevice));

        float gpuTime = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);

        CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, size * sizeof(int), cudaMemcpyDeviceToHost));

        double speedup = cpuTime.count() / (gpuTime / 1000.0);
        double efficiency = speedup / CUDA_CORES;

        std::cout << "| " << std::setw(11) << (N * N)
                  << " | " << std::setw(12) << std::fixed << std::setprecision(6) << cpuTime.count()
                  << " | " << std::setw(12) << gpuTime / 1000.0
                  << " | " << std::setw(7) << speedup
                  << " | " << std::setw(10) << efficiency
                  << " | " << std::setw(13) << C_gpu[N * N / 2] << " |\n";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    std::cout << "------------------------------------------------------------------------------------\n";

    return 0;
}

