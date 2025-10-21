// src/cuda/kernel.cu
//
// High-performance CUDA kernels for the **ParallelOptimizationEngine** GPU backend.
// This file implements device-side parallel computation of agent gradients and
// efficient reduction to compute the average gradient required for consensus
// gradient descent.
//
// Kernels are optimized for:
//   • **Massive data parallelism** – one thread per agent.
//   • **Memory coalescing** – contiguous access to `a` and `b` arrays.
//   • **Minimal divergence** – uniform control flow across warps.
//   • **Hierarchical reduction** – shared memory + recursive summation.
//
// All functions are **exception-safe** and include proper synchronization
// (`cudaDeviceSynchronize`) to ensure kernel completion before host continuation.
//
// Mathematical foundation:
//   • Gradient per agent: \( \nabla f_i(x) = 2 a_i (x - b_i) \)
//   • Global update: \( x \leftarrow x - \eta \cdot \frac{1}{N} \sum_i \nabla f_i(x) \)

#include <cuda_runtime.h>

/**
 * @brief Kernel: Computes per-agent gradients in parallel.
 *
 * @param a      Device pointer to array of coefficients \( a_i \)
 * @param b      Device pointer to array of targets \( b_i \)
 * @param x      Current shared decision variable
 * @param grads  Device pointer to output gradient array
 * @param N      Number of agents
 *
 * Each thread computes \( \nabla f_i(x) = 2 a_i (x - b_i) \) for its assigned agent.
 * Bounds checking ensures safety for non-divisible grid sizes.
 */
__global__ void computeGradientsKernel(double* a, double* b, double x, double* grads, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grads[idx] = 2.0 * a[idx] * (x - b[idx]);
    }
}

/**
 * @brief Host function: Launches gradient computation kernel.
 *
 * @param d_a     Device pointer to \( a_i \) array
 * @param d_b     Device pointer to \( b_i \) array
 * @param x       Current \( x \) value
 * @param d_grads Device pointer to output gradient buffer
 * @param N       Number of agents
 *
 * Configures grid/block dimensions for optimal occupancy (256 threads/block).
 * Synchronizes device to ensure kernel completion before returning.
 */
void cudaKernelComputeGradients(double* d_a, double* d_b, double x, double* d_grads, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    computeGradientsKernel<<<numBlocks, blockSize>>>(d_a, d_b, x, d_grads, N);
    cudaDeviceSynchronize();
}

/**
 * @brief Kernel: Parallel reduction using shared memory and warp unrolling.
 *
 * @param d_data Input/output device array (overwritten in-place during reduction)
 * @param N      Current number of elements to reduce
 *
 * Uses a **tree-based reduction** within shared memory:
 *   1. Each thread loads up to two elements (strided access).
 *   2. Cooperative reduction within block using `__syncthreads()`.
 *   3. Block result written to `d_data[blockIdx.x]`.
 *
 * Supports arbitrary input size via recursive host-side invocation.
 */
__global__ void sumKernel(double* d_data, int N) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    // Load up to two elements per thread with bounds checking
    while (i < N) {
        sdata[tid] += d_data[i] + (i + blockDim.x < N ? d_data[i + blockDim.x] : 0);
        i += gridSize;
    }
    __syncthreads();

    // Reduction in shared memory (unrolled for powers of two)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write block result
    if (tid == 0) d_data[blockIdx.x] = sdata[0];
}

/**
 * @brief Host function: Recursively reduces array to single sum.
 *
 * @param d_data Input device array (modified in-place)
 * @param d_sum  Device pointer to single-element output
 * @param N      Number of elements in current reduction step
 *
 * Recursively launches `sumKernel` until one value remains, then copies
 * result to host-accessible `d_sum`.  Uses dynamic shared memory sizing.
 */
void cudaKernelSum(double* d_data, double* d_sum, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize * 2 - 1) / (blockSize * 2);
    sumKernel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_data, N);
    cudaDeviceSynchronize();

    if (numBlocks > 1) {
        cudaKernelSum(d_data, d_sum, numBlocks);
    } else {
        cudaMemcpy(d_sum, d_data, sizeof(double), cudaMemcpyDeviceToHost);
    }
}