// src/core/Engine.hpp
//
// Implements the **OptimizationEngine** facade and the full suite of
// optimization strategies for the **ParallelOptimizationEngine** framework.
//
// This file implements the **Strategy Design Pattern** to achieve:
// • **Polymorphic interchangeability** of optimization algorithms at runtime.
// • **Clean separation** between algorithm logic and execution orchestration.
// • **Extensibility** – new strategies (e.g., stochastic, distributed) can be
// added without modifying existing code.
//
// Additionally, `OptimizationEngine` serves as a **Facade**, abstracting
// hardware-specific details (CPU, GPU, ML) from the Python interface. The
// **Factory Pattern** (implemented in `python_bindings/binding.cpp`) uses this
// interface to instantiate the correct concrete strategy based on user-selected
// mode and detected hardware.
//
// All strategy classes inherit from `OptimizationStrategy` and implement the
// pure virtual `optimize()` method, which returns the final \( x \) value and
// populates iteration count and timing metrics by reference.
//
// Mathematical context:
// • Objective: Minimize \( F(x) = \sum_{i=1}^N a_i (x - b_i)^2 \)
// • Strategies range from closed-form averaging (naive) to iterative
// consensus gradient descent (collaborative).
#ifndef ENGINE_HPP
#define ENGINE_HPP
#include <vector>
#include "Agent.hpp"
/**
 * @brief Abstract base class defining the optimization strategy interface.
 *
 * All concrete strategies must implement `optimize()`, which executes the
 * algorithm and returns the converged value of the shared variable \( x \).
 *
 * @param agents Const reference to the ensemble of quadratic agents.
 * @param iterations Reference to store the number of optimization steps.
 * @param time_taken Reference to store execution time (populated externally).
 * @return double Final optimized value of \( x \).
 */
class OptimizationStrategy {
public:
    virtual ~OptimizationStrategy() = default;
    /**
     * @brief Executes the optimization algorithm.
     *
     * This pure virtual function is overridden by concrete strategies.
     * Implementations must be thread-safe when used with parallel backends.
     */
    virtual double optimize(const std::vector<Agent>& agents,
                           double& iterations, double& time_taken) = 0;
};
/**
 * @brief Naive sequential strategy: unweighted averaging of local minima.
 *
 * Computes \( x^* = \frac{1}{N} \sum b_i \) in a single pass.
 * Fastest baseline; ignores coefficient weights \( a_i \).
 */
class NaiveSequentialStrategy : public OptimizationStrategy {
public:
    double optimize(const std::vector<Agent>& agents,
                    double& iterations, double& time_taken) override;
};
/**
 * @brief Naive parallel CPU strategy using ThreadPool.
 *
 * Parallelizes local minimum summation across hardware threads.
 * Demonstrates data parallelism with zero overhead after pool initialization.
 */
class NaiveParallelCPUStrategy : public OptimizationStrategy {
public:
    double optimize(const std::vector<Agent>& agents,
                    double& iterations, double& time_taken) override;
};
// Added: New class declaration for NaiveOpenMPStrategy
// Reason: To define an interface for the OpenMP-based naive strategy, providing a parallel CPU variant using reductions, as per the requirement to implement OpenMP variants for performance comparisons and parallel gradient sums.
/**
 * @brief Naive parallel CPU strategy using OpenMP.
 *
 * Parallelizes local minimum summation using OpenMP reduction.
 * Provides a contrast to ThreadPool-based parallelism.
 */
class NaiveOpenMPStrategy : public OptimizationStrategy {
public:
    double optimize(const std::vector<Agent>& agents,
                    double& iterations, double& time_taken) override;
};
/**
 * @brief Placeholder for ML-accelerated naive strategy.
 *
 * Currently mirrors sequential behavior. Real ML prediction (neural approximation
 * of mean \( b_i \)) is implemented in Python and invoked via bindings.
 */
class NaiveMLStrategy : public OptimizationStrategy {
public:
    double optimize(const std::vector<Agent>& agents,
                    double& iterations, double& time_taken) override;
};
// Added: New class declaration for CollaborativeSequentialStrategy
// Reason: To provide a sequential baseline interface for collaborative optimization, enabling comparisons with parallel variants and supporting the convergence policy (diminishing step and delta-x stop) in a non-parallel context.
/**
 * @brief Collaborative sequential strategy using consensus gradient descent.
 *
 * Implements diminishing-step GD with sequential gradient evaluation.
 * Converges to the weighted global minimum \( x^* = \frac{\sum a_i b_i}{\sum a_i} \).
 */
class CollaborativeSequentialStrategy : public OptimizationStrategy {
public:
    double optimize(const std::vector<Agent>& agents,
                    double& iterations, double& time_taken) override;
};
/**
 * @brief Collaborative strategy using consensus gradient descent on CPU.
 *
 * Implements diminishing-step GD with parallel gradient evaluation via ThreadPool.
 * Converges to the weighted global minimum \( x^* = \frac{\sum a_i b_i}{\sum a_i} \).
 */
class CollaborativeParallelStrategy : public OptimizationStrategy {
public:
    double optimize(const std::vector<Agent>& agents,
                    double& iterations, double& time_taken) override;
};
// Added: New class declaration for CollaborativeOpenMPStrategy
// Reason: To define an interface for the OpenMP-based collaborative strategy, using reductions for efficient parallel summation, as per the requirement to add OpenMP variants and enable perf addendum comparisons.
/**
 * @brief Collaborative strategy using consensus gradient descent with OpenMP.
 *
 * Implements diminishing-step GD with parallel gradient evaluation via OpenMP reduction.
 * Converges to the weighted global minimum \( x^* = \frac{\sum a_i b_i}{\sum a_i} \).
 */
class CollaborativeOpenMPStrategy : public OptimizationStrategy {
public:
    double optimize(const std::vector<Agent>& agents,
                    double& iterations, double& time_taken) override;
};
// Added: New class declaration for CollaborativeMLStrategy
// Reason: To provide a placeholder interface for the ML-accelerated collaborative strategy, ensuring consistency with the naive ML variant and supporting the bonus AI integration without full C++ implementation (real logic in Python).
/**
 * @brief Placeholder for ML-accelerated collaborative strategy.
 *
 * Currently a minimal implementation. Intended to use neural approximation of
 * average gradients, implemented in Python and invoked via bindings.
 */
class CollaborativeMLStrategy : public OptimizationStrategy {
public:
    virtual double optimize(const std::vector<Agent>& agents,
                           double& iterations, double& time_taken) override;
};
/**
 * @brief Facade class orchestrating optimization with timing and strategy management.
 *
 * Encapsulates a pointer to a concrete `OptimizationStrategy` and provides
 * a unified `run()` interface used by Python bindings. Owns the strategy
 * lifetime via RAII (deletes in destructor).
 *
 * This class hides hardware and algorithmic complexity from the user,
 * enabling seamless switching between CPU, GPU, and ML backends.
 */
class OptimizationEngine {
protected:
    OptimizationStrategy* strategy; ///< Pointer to the active strategy.
public:
    /**
     * @brief Constructs the engine with a given strategy.
     *
     * @param s Raw pointer to a heap-allocated strategy (ownership transferred).
     */
    OptimizationEngine(OptimizationStrategy* s) : strategy(s) {}
    /**
     * @brief Destructor ensures proper cleanup of the strategy object.
     */
    virtual ~OptimizationEngine() { delete strategy; }
    /**
     * @brief Executes the optimization and measures wall-clock time.
     *
     * @param agents Input agent ensemble.
     * @param iterations Output: number of iterations performed.
     * @param time_taken Output: execution time in seconds.
     * @return double Final optimized \( x \).
     *
     * Uses `std::chrono::high_resolution_clock` for microsecond-accurate timing.
     */
    virtual double run(const std::vector<Agent>& agents,
                       double& iterations, double& time_taken);
};
#endif // ENGINE_HPP