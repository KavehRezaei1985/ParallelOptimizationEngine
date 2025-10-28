// src/cuda/CudaEngine.hpp
//
// Header file declaring the **CudaOptimizationEngine** and its strategies
// for GPU-accelerated optimization in the **ParallelOptimizationEngine** framework.
//
// This file extends the **Strategy Design Pattern** to GPU backends, providing
// polymorphic strategies for naive and collaborative optimization. The
// **CudaOptimizationEngine** inherits from `OptimizationEngine` and manages
// CUDA-specific memory and kernel launches.
//
// Mathematical context:
// • Naive: \( x^* = \frac{1}{N} \sum b_i \) via parallel reduction.
// • Collaborative: Gradient descent on \( F(x) = \sum a_i (x - b_i)^2 \)
// with diminishing step size \( \eta_k = \eta_0 / k \).
#ifndef CUDA_ENGINE_HPP
#define CUDA_ENGINE_HPP
#include "../core/Engine.hpp" // Adjusted to relative path from src/cuda/ to src/core/ for clarity
// Added: Include <vector> to support std::vector<Agent> in method signatures.
// Reason: To ensure self-contained header usage, aligning with modern C++ practices and
// supporting correctness testing by providing necessary type definitions.
#include <vector>

// Modified: Simplified class documentation; added explicit constructor.
// Reason: To align with project's concise documentation style (e.g., util.hpp) and ensure
// proper initialization in CudaEngine.cu for consistency with implementation.
/**
 * @brief GPU-accelerated naive strategy for parallel unweighted averaging.
 *
 * Computes \( x^* = \frac{1}{N} \sum b_i \) using CUDA kernels for parallel reduction.
 */
class NaiveCudaStrategy : public OptimizationStrategy {
public:
    // Added: Explicit constructor declaration.
    // Reason: To match implementation in CudaEngine.cu, ensuring proper initialization.
    NaiveCudaStrategy();
    /**
     * @brief Executes parallel averaging on GPU.
     *
     * @param agents Input agent ensemble.
     * @param iterations Set to 1.0 (single evaluation).
     * @param time_taken Not used (timing handled by facade).
     * @return double Averaged local minima.
     */
    double optimize(const std::vector<Agent>& agents,
                    double& iterations, double& time_taken) override;
};

// Modified: Simplified class documentation; added explicit constructor.
// Reason: To align with project's concise style and reflect diminishing step size in
// CudaEngine.cu, ensuring consistency with updated convergence policy.
/**
 * @brief GPU-accelerated collaborative strategy using gradient descent.
 *
 * Performs gradient descent on \( F(x) = \sum a_i (x - b_i)^2 \) with diminishing
 * step size \( \eta_k = \eta_0 / k \).
 */
class CollaborativeCudaStrategy : public OptimizationStrategy {
public:
    // Added: Explicit constructor declaration.
    // Reason: To match implementation in CudaEngine.cu, ensuring proper initialization.
    CollaborativeCudaStrategy();
    /**
     * @brief Executes GPU-accelerated gradient descent.
     *
     * @param agents Input ensemble of convex quadratic agents.
     * @param iterations Output: number of iterations until convergence.
     * @param time_taken Output: execution time (measured externally).
     * @return double Converged shared variable \( x \).
     */
    double optimize(const std::vector<Agent>& agents,
                    double& iterations, double& time_taken) override;
};

// Modified: Simplified documentation; added destructor and run method; renamed parameter
// to strategy; added private strategy member.
// Reason: To align with concise style, match CudaEngine.cu implementation, and ensure
// explicit interface definition for clarity and maintainability.
/**
 * @brief Facade for GPU-accelerated optimization engines.
 *
 * Manages CUDA-specific strategies and delegates to OptimizationEngine::run().
 */
class CudaOptimizationEngine : public OptimizationEngine {
public:
    // Modified: Renamed parameter from strat to strategy.
    // Reason: To improve readability and align with naming conventions in CudaEngine.cu.
    /**
     * @brief Constructs the CUDA engine with a GPU strategy.
     *
     * @param strategy Heap-allocated CUDA strategy (ownership transferred).
     */
    CudaOptimizationEngine(OptimizationStrategy* strategy);
    // Added: Explicit destructor declaration.
    // Reason: To match CudaEngine.cu implementation, ensuring clear cleanup (handled by base class).
    ~CudaOptimizationEngine() override;
    // Added: Explicit run method declaration.
    // Reason: To match CudaEngine.cu implementation, clarifying delegation to base class for maintainability.
    /**
     * @brief Executes the CUDA-based optimization.
     *
     * @param agents Input agent ensemble.
     * @param iterations Output: number of iterations.
     * @param time_taken Output: execution time in seconds.
     * @return double Final optimized \( x \).
     */
    double run(const std::vector<Agent>& agents,
               double& iterations, double& time_taken) override;
private:
    // Added: Explicit strategy member declaration.
    // Reason: To clarify the role of the strategy pointer, though redundant with base class, for documentation purposes.
    OptimizationStrategy* strategy;
};
#endif // CUDA_ENGINE_HPP