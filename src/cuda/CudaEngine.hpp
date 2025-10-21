// src/cuda/CudaEngine.hpp
//
// Header declaring **GPU-accelerated optimization strategies** and the 
// **CUDA-specific engine facade** for the **ParallelOptimizationEngine** framework.
//
// This file extends the core `OptimizationStrategy` interface with CUDA-optimized
// implementations that leverage high-throughput parallel kernels defined in 
// `kernel.cu`.  All GPU strategies inherit from the abstract base class 
// `OptimizationStrategy`, ensuring full compliance with the **Strategy Pattern** 
// and seamless integration into the polymorphic engine hierarchy.
//
// Key components:
//   • `CollaborativeCudaStrategy`: Consensus gradient descent using 
//     parallel gradient evaluation and hierarchical reduction.
//   • `NaiveCudaStrategy`: Single-pass parallel averaging of local minima.
//   • `CudaOptimizationEngine`: Facade subclass that owns GPU strategy instances 
//     and integrates with the Python binding layer.
//
// The design ensures:
//   • **Zero host-side overhead** beyond kernel launches and memory transfers.
//   • **Exception-safe resource management** via RAII in `.cu` implementations.
//   • **Hardware abstraction** — users interact via the unified `OptimizationEngine` 
//     interface without GPU-specific knowledge.
//
// Include this header only in translation units requiring GPU acceleration.
// The implementation resides in `CudaEngine.cu` and depends on `kernel.cu`.

#ifndef CUDA_ENGINE_HPP
#define CUDA_ENGINE_HPP

#include "../core/Engine.hpp"

/**
 * @class CollaborativeCudaStrategy
 * @brief GPU-accelerated collaborative optimization using consensus gradient descent.
 *
 * Implements fixed-step gradient descent on the global cost:
 *   \( F(x) = \sum_{i=1}^N a_i (x - b_i)^2 \)
 * using CUDA kernels for:
 *   - Per-agent gradient computation: \( 2 a_i (x - b_i) \)
 *   - Parallel reduction to compute average gradient
 *
 * Achieves near-linear scaling with \( N \) on NVIDIA GPUs.
 */
class CollaborativeCudaStrategy : public OptimizationStrategy {
public:
    /**
     * @brief Executes GPU-accelerated gradient descent.
     *
     * @param agents     Input ensemble of convex quadratic agents.
     * @param iterations Output: number of iterations until convergence.
     * @param time_taken Output: execution time (measured externally).
     * @return double    Converged shared variable \( x \).
     */
    double optimize(const std::vector<Agent>& agents, 
                    double& iterations, double& time_taken) override;
};

/**
 * @class NaiveCudaStrategy
 * @brief GPU-accelerated naive strategy: parallel unweighted averaging.
 *
 * Computes:
 *   \( x^* = \frac{1}{N} \sum_{i=1}^N b_i \)
 * using a single parallel reduction kernel over the \( b_i \) values.
 *
 * Ideal for baseline performance comparison and large-scale \( N \).
 */
class NaiveCudaStrategy : public OptimizationStrategy {
public:
    /**
     * @brief Executes parallel averaging on GPU.
     *
     * @param agents     Input agent ensemble.
     * @param iterations Set to 1.0 (single evaluation).
     * @param time_taken Not used (timing handled by facade).
     * @return double    Averaged local minima.
     */
    double optimize(const std::vector<Agent>& agents, 
                    double& iterations, double& time_taken) override;
};

/**
 * @class CudaOptimizationEngine
 * @brief Facade for GPU-accelerated optimization engines.
 *
 * Owns a pointer to a CUDA strategy and delegates execution to 
 * `OptimizationEngine::run()`.  Enables polymorphic use within the 
// Python binding layer while encapsulating GPU resource lifecycle.
 */
class CudaOptimizationEngine : public OptimizationEngine {
public:
    /**
     * @brief Constructs the CUDA engine with a GPU strategy.
     *
     * @param strat Heap-allocated CUDA strategy (ownership transferred).
     */
    CudaOptimizationEngine(OptimizationStrategy* strat);
};

#endif // CUDA_ENGINE_HPP