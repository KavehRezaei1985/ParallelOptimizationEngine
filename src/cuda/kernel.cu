// src/cuda/kernel.cu
//
// CUDA kernels for the **ParallelOptimizationEngine** framework.
// Defines parallel computation kernels for gradient evaluation and reduction.
//
// This file includes:
// • computeGradientsKernel: Computes per-agent gradients.
// • sumKernel: Performs parallel reduction of gradients or local minima.
// • reduceBlocks: Reduces block-level sums to a final result.
//
// Kernels are designed for high performance with thread block synchronization
// and shared memory usage. All operations use double precision for accuracy.
//
// Modified: Simplified header comments; removed optimization details (e.g., memory coalescing,
// minimal divergence); added explicit kernel listing; removed mathematical foundation.
// Reason: To align with project's concise documentation style (e.g., util.hpp, CudaEngine.hpp),
// focus on core functionality, and reflect new kernel structure (sumKernel, reduceBlocks).
#include <cuda_runtime.h>

// Modified: Updated to use interleaved agents_d array and size_t N; renamed parameters.
// Reason: To support unified memory in CudaEngine.cu, improve scalability for large N,
// and align parameter names with host code for clarity.
/**
 * @brief Kernel: Computes per-agent gradients in parallel.
 *
 * @param agents_d Device pointer to interleaved agent coefficients [a0, b0, a1, b1, ...].
 * @param N Number of agents.
 * @param x Current decision variable.
 * @param grad_d Device pointer to output gradient array.
 *
 * Each thread computes \( \nabla f_i(x) = 2 a_i (x - b_i) \) for its assigned agent.
 * Uses interleaved data for unified memory access. Bounds checking ensures safety.
 */
__global__ void computeGradientsKernel(double* agents_d, size_t N, double x, double* grad_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Modified: Access a_i, b_i from interleaved array.
        // Reason: To support unified memory layout in CudaEngine.cu.
        double a = agents_d[2 * idx];
        double b = agents_d[2 * idx + 1];
        grad_d[idx] = 2.0 * a * (x - b); // Gradient: 2 * a * (x - b)
    }
}

// Modified: Added output parameter, is_gradient flag, and fixed shared memory size;
// simplified data loading; renamed parameters.
// Reason: To make kernel reusable for gradients and local minima, support unified memory,
// and simplify reduction logic for performance and clarity.
/**
 * @brief Kernel: Performs parallel reduction of gradients or local minima.
 *
 * @param input Device pointer to input data (gradients or interleaved b_i values).
 * @param N Number of elements to reduce.
 * @param output Device pointer to store block-level sums.
 * @param is_gradient True for gradient reduction, false for local minima (b_i).
 *
 * Uses shared memory for tree-based reduction within each block. Supports both
 * gradient sums (collaborative mode) and local minima sums (naive mode) via
 * is_gradient flag, accessing interleaved data when is_gradient=false.
 */
__global__ void sumKernel(double* input, size_t N, double* output, bool is_gradient) {
    // Modified: Fixed shared memory size to 256.
    // Reason: To simplify code, as block size is fixed at 256 in CudaEngine.cu.
    __shared__ double shared[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double sum = 0.0;
    // Modified: Conditional loading based on is_gradient for flexibility.
    // Reason: To support both gradient and b_i reductions, using interleaved data for naive mode.
    if (idx < N) {
        sum = is_gradient ? input[idx] : input[2 * idx];
    }
    shared[tid] = sum;
    __syncthreads();
    // Perform tree-based reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    // Write block-level sum to output
    if (tid == 0) {
        // Modified: Write to output array instead of overwriting input.
        // Reason: To avoid modifying input data, improving clarity and safety.
        output[blockIdx.x] = shared[0];
    }
}

// Added: New kernel to reduce block-level sums.
// Reason: To perform final GPU-side reduction, replacing recursive host-side
// reduction in old cudaKernelSum, improving performance by minimizing host-device
// communication (performance addendum).
/**
 * @brief Kernel: Reduces block-level sums to a final result.
 *
 * @param input Device pointer to block-level sums.
 * @param num_blocks Number of blocks to reduce.
 *
 * Reduces block-level sums from sumKernel into a single result stored in input[0].
 * Uses shared memory for tree-based reduction within a single block.
 */
__global__ void reduceBlocks(double* input, int num_blocks) {
    __shared__ double shared[256];
    int tid = threadIdx.x;
    double sum = 0.0;
    // Load block-level sum if within bounds
    if (tid < num_blocks) {
        sum = input[tid];
    }
    shared[tid] = sum;
    __syncthreads();
    // Perform tree-based reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    // Write final result to input[0]
    if (tid == 0) {
        input[0] = shared[0];
    }
}