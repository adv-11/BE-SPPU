from numba import cuda
import numpy as np

# CUDA kernel for vector addition
@cuda.jit
def vector_add(a, b, result):
    idx = cuda.grid(1)
    if idx < a.size:
        result[idx] = a[idx] + b[idx]

# Host code
def main():
    n = 10
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)

    # Defensive check
    assert a.shape == b.shape, "Arrays must be of the same shape"

    result = np.zeros_like(a)

    # Copy to device
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_result = cuda.device_array_like(result)

    threads_per_block = 32
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    try:
        vector_add[blocks_per_grid, threads_per_block](d_a, d_b, d_result)
        d_result.copy_to_host(result)
        print("Input A:", a)
        print("Input B:", b)
        print("Result (A + B):", result)
    except cuda.CudaSupportError as e:
        print("CUDA error:", e)
    except AssertionError as ae:
        print("Assertion failed:", ae)
    except Exception as e:
        print("Unexpected error:", e)

if __name__ == '__main__':
    main()
