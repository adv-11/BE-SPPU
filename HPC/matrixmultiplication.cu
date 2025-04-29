nvcc filename.cu -o filelname
./filename

#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>

#define CUDA_CORES 768  // GTX 1050Ti CUDA cores

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

/********************** CUDA Kernel: Matrix Multiplication **********************/
__global__ void matrixMulCanon(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/********************** Sequential Matrix Multiplication **********************/
void sequentialMatrixMul(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/********************** Run Matrix Multiplication Test **********************/
void runMatrixMultiplication(int N) {
    std::vector<int> A(N * N), B(N * N), C(N * N);

    for (int i = 0; i < N * N; ++i) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(int)));

    cudaMemcpy(d_A, A.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    // CPU execution time measurement
    auto startSeq = std::chrono::high_resolution_clock::now();
    sequentialMatrixMul(A, B, C, N);
    auto endSeq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seqTime = endSeq - startSeq;

    // GPU execution time measurement
    float gpuTime;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);

    cudaEventRecord(start);
    matrixMulCanon<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpuTime, start, end);

    // Check for CUDA errors after kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    double speedup = seqTime.count() / (gpuTime / 1000.0);
    double efficiency = speedup / CUDA_CORES;

    std::cout << std::setw(15) << N 
              << std::setw(20) << seqTime.count() 
              << std::setw(20) << gpuTime 
              << std::setw(20) << speedup 
              << std::setw(20) << efficiency 
              << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/********************** Main Function **********************/
int main() {
std::cout << "\n";
   std::cout << "\n";
 std::cout << "\n";
 std::cout << "\n";
    std::cout << "\n==== Matrix Multiplication Tests ====\n";
    std::cout << std::setw(15) << "Matrix Size" 
              << std::setw(20) << "CPU Time (s)" 
              << std::setw(20) << "GPU Time (ms)" 
              << std::setw(20) << "Speedup" 
              << std::setw(20) << "Efficiency" 
              << std::endl;
    std::cout << std::string(95, '-') << "\n";

    int matrixTestCases[] = {200,500, 1024, 2000};  // Large matrix sizes
    for (int i = 0; i < 4; i++) {
        runMatrixMultiplication(matrixTestCases[i]);
    }

    return 0;
}