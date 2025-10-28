// src/cuda/CudaEngine.cu
//
// Implements the **CudaOptimizationEngine** and its strategies for GPU-accelerated
// optimization in the **ParallelOptimizationEngine** framework.
//
// This file provides CUDA-based implementations for:
// • Naive strategy: Parallel reduction of local minima (\( x^* = \frac{1}{N} \sum b_i \)).
// • Collaborative strategy: Parallel gradient descent with diminishing step size.
//
// CUDA kernels are defined in `kernel.cu` and launched from here. Memory management
// uses unified memory for simplicity and performance, with pinned host memory to
// minimize PCIe transfer overhead. Error handling ensures robust execution.
//
// Mathematical foundation:
// • Naive: Same as CPU, but with O(log N) parallel reduction.
// • Collaborative: Gradient descent on \( F(x) = \sum a_i (x - b_i)^2 \),
// with diminishing step size \( \eta_k = \eta_0 / k \), convergence when
// \( \|x_{k+1} - x_k\| < 10^{-6} \).
#include "CudaEngine.hpp"
// Modified: Replaced external kernel declarations with direct inclusion of kernel.cu.
// Reason: To simplify the build process and ensure tight integration of kernel definitions,
// supporting the new compute_and_reduce_gradients wrapper for modularity.
#include "kernel.cu"
#include <cuda_runtime.h>
// Modified: Replaced <chrono>, <vector>, <iostream> with <stdexcept>, <string>, <stdio.h>.
// Reason: Removed <chrono> as timing is handled by OptimizationEngine::run; removed <vector>
// due to unified memory eliminating host-side vectors; added <stdexcept>, <string> for
// exception-based error handling, and <stdio.h> for debug printing of invalid gradients.
#include <stdexcept>
#include <string>
#include <stdio.h> // For debug printing

// Added: New kernel wrapper function to orchestrate gradient computation and reduction.
// Reason: To centralize kernel launches for CollaborativeCudaStrategy, improving modularity
// and enabling exception-based error handling, supporting robustness for correctness testing.
/**
 * @brief CUDA kernel wrapper for computing gradients and reducing to sum.
 *
 * @param agents_d Device pointer to agent coefficients (a_i, b_i) as interleaved array [a0, b0, a1, b1, ...].
 * @param N Number of agents.
 * @param x Current decision variable.
 * @param grad_d Device pointer to store per-agent gradients.
 * @param sum_d Device pointer to store sum of gradients.
 *
 * Launches computeGradientsKernel to compute per-agent gradients, followed by sumKernel
 * and reduceBlocks for parallel reduction. Synchronizes after each kernel and checks for
 * errors, throwing std::runtime_error on failure.
 */
void compute_and_reduce_gradients(double* agents_d, size_t N, double x, double* grad_d, double* sum_d) {
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeGradientsKernel<<<blocks, threadsPerBlock>>>(agents_d, N, x, grad_d);
    cudaDeviceSynchronize();
    sumKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(grad_d, N, sum_d, true);
    cudaDeviceSynchronize();
    // Reduce block results
    reduceBlocks<<<1, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(sum_d, blocks);
    cudaDeviceSynchronize();
    // Added: Exception-based error handling instead of CUDA_CHECK macro.
    // Reason: To improve robustness and debuggability for correctness testing, aligning with modern C++ practices.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in kernel execution: " + std::string(cudaGetErrorString(err)));
    }
}

// Added: Explicit constructor for clarity.
// Reason: To ensure proper initialization and consistency with CollaborativeCudaStrategy.
/**
 * @brief Constructor for NaiveCudaStrategy.
 */
NaiveCudaStrategy::NaiveCudaStrategy() {}

/**
 * @brief Executes naive optimization on GPU.
 *
 * @param agents Input agent ensemble (host).
 * @param iterations Set to 1.0 (single pass).
 * @param time_taken Not used (measured externally).
 * @return double Unweighted average of local minima.
 *
 * Copies agent coefficients to device, launches reduction kernel, and retrieves result.
 *
 * Modified: Switched to unified memory with cudaMallocManaged, used interleaved array for
 * a_i, b_i; added edge case check for N=0; replaced CUDA_CHECK with exception-based error
 * handling; used new sumKernel and reduceBlocks kernels.
 * Reason: Unified memory simplifies management and improves performance; edge case enhances
 * robustness; exceptions align with correctness testing; new kernels support interleaved data
 * and scalability for large N (performance addendum).
 */
double NaiveCudaStrategy::optimize(const std::vector<Agent>& agents, double& iterations, double& time_taken) {
    iterations = 1.0;
    size_t N = agents.size();
    // Added: Edge case check for empty input.
    // Reason: To prevent undefined behavior, improving robustness.
    if (N == 0) return 0.0;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    double* agents_d;
    double* sum_d;
    // Modified: Used cudaMallocManaged for unified memory.
    // Reason: To simplify memory management and reduce PCIe transfer overhead.
    cudaMallocManaged(&agents_d, 2 * N * sizeof(double));
    cudaMallocManaged(&sum_d, blocks * sizeof(double));
    cudaMemset(sum_d, 0, blocks * sizeof(double));
    // Modified: Store both a_i and b_i in interleaved array, though only b_i used for naive.
    // Reason: To support unified memory and potential future extensions.
    for (size_t i = 0; i < N; ++i) {
        agents_d[2 * i] = agents[i].a;
        agents_d[2 * i + 1] = agents[i].b;
    }
    // Modified: Use new sumKernel and reduceBlocks with interleaved data.
    // Reason: To support scalable parallel reduction for large N.
    sumKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(&agents_d[1], N, sum_d, false);
    cudaDeviceSynchronize();
    reduceBlocks<<<1, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(sum_d, blocks);
    cudaDeviceSynchronize();
    // Modified: Exception-based error handling.
    // Reason: For robust error reporting and correctness testing.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(agents_d);
        cudaFree(sum_d);
        throw std::runtime_error("CUDA error in naive strategy: " + std::string(cudaGetErrorString(err)));
    }
    double result = sum_d[0] / N;
    cudaFree(agents_d);
    cudaFree(sum_d);
    return result;
}

// Added: Explicit constructor for clarity.
// Reason: To ensure proper initialization and consistency with NaiveCudaStrategy.
/**
 * @brief Constructor for CollaborativeCudaStrategy.
 */
CollaborativeCudaStrategy::CollaborativeCudaStrategy() {}

/**
 * @brief Executes collaborative optimization on GPU.
 *
 * @param agents Input agent ensemble (host).
 * @param iterations Number of GD iterations performed.
 * @param time_taken Not used (measured externally).
 * @return double Converged value of \( x \).
 *
 * Implements diminishing-step gradient descent:
 * \( x_{k+1} = x_k - \eta_k \cdot \sum_{i=1}^N 2 a_i (x_k - b_i) \)
 * with \( \eta_k = \eta_0 / k \), \( \eta_0 = 0.01 \), stopping when
 * \( \|x_{k+1} - x_k\| < 10^{-9} \), max 10,000 iterations.
 *
 * Modified: Switched to unified memory; used compute_and_reduce_gradients wrapper;
 * implemented diminishing step size and delta-x convergence; added NaN/inf checks;
 * tightened tolerance to 1e-9, increased max_it to 100000; replaced CUDA_CHECK with
 * exceptions.
 * Reason: To align with convergence policy (diminishing step, delta-x stop), improve
 * performance with unified memory, enhance modularity with kernel wrapper, and ensure
 * numerical stability for correctness testing.
 */
double CollaborativeCudaStrategy::optimize(const std::vector<Agent>& agents, double& iterations, double& time_taken) {
    size_t N = agents.size();
    // Added: Edge case check for empty input.
    // Reason: To prevent undefined behavior, improving robustness.
    if (N == 0) return 0.0;
    double* agents_d;
    double* grad_d;
    double* sum_d;
    // Modified: Used cudaMallocManaged for unified memory.
    // Reason: To simplify memory management and reduce transfer overhead.
    cudaMallocManaged(&agents_d, 2 * N * sizeof(double));
    cudaMallocManaged(&grad_d, N * sizeof(double));
    cudaMallocManaged(&sum_d, sizeof(double));
    cudaMemset(sum_d, 0, sizeof(double));
    // Modified: Store a_i, b_i in interleaved array.
    // Reason: To support unified memory and consistent data layout.
    for (size_t i = 0; i < N; ++i) {
        agents_d[2 * i] = agents[i].a;
        agents_d[2 * i + 1] = agents[i].b;
    }
    double x = 0.0;
    const double eta_0 = 0.01;
    // Modified: Tightened tolerance to 1e-9, increased max_it to 100000.
    // Reason: To improve precision and allow more iterations for convergence.
    const double tol = 1e-9;
    const int max_it = 100000;
    iterations = 0;
    while (iterations < max_it) {
        // Modified: Used compute_and_reduce_gradients wrapper.
        // Reason: To centralize kernel launches for modularity.
        compute_and_reduce_gradients(agents_d, N, x, grad_d, sum_d);
        double total_grad = sum_d[0]; // Use raw sum without 1/N normalization
        // Added: NaN/inf checks for gradient.
        // Reason: To ensure numerical stability for correctness testing.
        if (std::isnan(total_grad) || std::isinf(total_grad)) {
            printf("Warning: Invalid gradient detected (total_grad = %f), breaking loop\n", total_grad);
            break;
        }
        // Modified: Implemented diminishing step size.
        // Reason: To align with convergence policy for consistent behavior.
        double eta = eta_0 / (iterations + 1);
        double x_new = x - eta * total_grad;
        // Modified: Used delta-x convergence criterion; added NaN/inf checks for x_new.
        // Reason: To match CPU strategies and ensure numerical stability.
        if (std::abs(x_new - x) < tol || std::isnan(x_new) || std::isinf(x_new)) break;
        x = x_new;
        ++iterations;
    }
    cudaFree(agents_d);
    cudaFree(grad_d);
    cudaFree(sum_d);
    return x;
}

/**
 * @brief Constructor for CudaOptimizationEngine.
 *
 * @param strategy Pointer to the strategy to be used (ownership transferred).
 *
 * Modified: Renamed parameter from strat to strategy.
 * Reason: To improve clarity and align with naming conventions.
 */
CudaOptimizationEngine::CudaOptimizationEngine(OptimizationStrategy* strategy) : OptimizationEngine(strategy) {
    // Strategy is initialized in the base class constructor
}

// Added: Explicit destructor for clarity.
// Reason: To ensure clear cleanup, though handled by base class.
/**
 * @brief Destructor ensures strategy cleanup.
 */
CudaOptimizationEngine::~CudaOptimizationEngine() {
    // Cleanup handled by base class destructor
}

// Added: Explicit run method.
// Reason: To clarify delegation to base class and ensure maintainability.
/**
 * @brief Executes the CUDA-based optimization.
 *
 * @param agents Input agent ensemble.
 * @param iterations Output: number of iterations.
 * @param time_taken Output: execution time in seconds.
 * @return double Final optimized \( x \).
 */
double CudaOptimizationEngine::run(const std::vector<Agent>& agents, double& iterations, double& time_taken) {
    return OptimizationEngine::run(agents, iterations, time_taken);
}