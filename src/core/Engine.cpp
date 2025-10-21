// src/core/Engine.cpp
//
// Implements the **OptimizationEngine** facade and all CPU-based optimization
// strategies for the **ParallelOptimizationEngine** framework.
//
// This file provides:
//   • A unified runtime interface (`OptimizationEngine::run`) that measures
//     wall-clock execution time with high-resolution chronometry.
//   • Three **naive** strategies (sequential, parallel CPU, ML placeholder).
//   • One **collaborative** strategy using gradient consensus with a
//     high-performance `ThreadPool` for data-parallel gradient summation.
//
// All strategies adhere to the **Strategy Pattern**, enabling runtime
// polymorphism and seamless integration with GPU and ML backends via the
// Python facade.  Thread-safety is enforced via RAII (`std::mutex`, `std::atomic`),
// and the `ThreadPool` eliminates per-iteration thread creation overhead.
//
// Mathematical foundation:
//   • Naive: \( x^* = \frac{1}{N} \sum b_i \) (unweighted average of local minima)
//   • Collaborative: Gradient descent on \( F(x) = \sum a_i (x - b_i)^2 \)
//     with fixed step size \( \eta = 0.01 \), convergence when
//     \( \left| \frac{1}{N} \sum 2 a_i (x - b_i) \right| < 10^{-6} \).

#include "Engine.hpp"
#include "ThreadPool.hpp"
#include <chrono>
#include <numeric>
#include <mutex>
#include <atomic>

/**
 * @brief Executes the active optimization strategy and records performance metrics.
 *
 * @param agents     Const reference to the agent ensemble.
 * @param iterations Reference to store the number of optimization steps.
 * @param time_taken Reference to store wall-clock execution time in seconds.
 * @return double    Final optimized value of the shared variable \( x \).
 *
 * This method serves as the **Facade** entry point from Python bindings.
 * It encapsulates timing logic using `std::chrono::high_resolution_clock`
 * to ensure microsecond-accurate benchmarking across hardware platforms.
 */
double OptimizationEngine::run(const std::vector<Agent>& agents,
                               double& iterations, double& time_taken) {
    auto start = std::chrono::high_resolution_clock::now();
    double result = strategy->optimize(agents, iterations, time_taken);
    auto end = std::chrono::high_resolution_clock::now();
    time_taken = std::chrono::duration<double>(end - start).count();
    return result;
}

/* ============================== NAIVE STRATEGIES ============================== */

/**
 * @brief Naive sequential strategy: unweighted averaging of local minima.
 *
 * @param agents     Input agent ensemble.
 * @param iterations Set to 1.0 (single evaluation).
 * @param time_taken Not used (timing handled by OptimizationEngine).
 * @return double    \( x^* = \frac{1}{N} \sum b_i \)
 *
 * O(N) sequential loop.  Fastest baseline; ignores coefficient weights \( a_i \).
 */
double NaiveSequentialStrategy::optimize(const std::vector<Agent>& agents,
                                         double& iterations, double& time_taken) {
    iterations = 1.0;
    double sum = 0.0;
    for (const auto& ag : agents) sum += ag.getLocalMin();
    return sum / agents.size();
}

/**
 * @brief Naive parallel CPU strategy using ThreadPool for data parallelism.
 *
 * @param agents     Input agent ensemble.
 * @param iterations Set to 1.0.
 * @param time_taken Not used.
 * @return double    Parallel reduction of local minima.
 *
 * Each agent’s local minimum is computed in a separate task enqueued to a
 * reusable `ThreadPool`.  A mutex-protected accumulator ensures thread-safe
 * summation.  Demonstrates scalable parallelism for large \( N \).
 */
double NaiveParallelCPUStrategy::optimize(const std::vector<Agent>& agents,
                                          double& iterations, double& time_taken) {
    iterations = 1.0;
    double total = 0.0;
    std::mutex m;
    ThreadPool pool;  // Reusable thread pool with hardware-concurrency threads.

    for (const auto& ag : agents) {
        pool.enqueue([&total, &m, &ag] {
            double local = ag.getLocalMin();
            std::lock_guard<std::mutex> lk(m);
            total += local;
        });
    }
    // ThreadPool destructor implicitly joins all tasks.
    return total / agents.size();
}

/**
 * @brief Naive ML strategy placeholder.
 *
 * @param agents     Input agent ensemble.
 * @param iterations Set to 1.0.
 * @param time_taken Not used.
 * @return double    Identical to sequential naive (real ML in Python layer).
 *
 * This stub enables uniform interface testing.  Actual ML prediction
 * (e.g., neural approximation of mean \( b_i \)) is implemented in
 * `python/ml_agent.py` and invoked via Python bindings.
 */
double NaiveMLStrategy::optimize(const std::vector<Agent>& agents,
                                 double& iterations, double& time_taken) {
    iterations = 1.0;
    double sum = 0.0;
    for (const auto& ag : agents) sum += ag.getLocalMin();
    return sum / agents.size();  // Placeholder – real ML in Python
}

/* ======================= COLLABORATIVE CPU STRATEGY ======================= */

/**
 * @brief Collaborative parallel strategy using consensus gradient descent.
 *
 * @param agents     Input agent ensemble.
 * @param iterations Number of GD iterations performed.
 * @param time_taken Not used (measured externally).
 * @return double    Converged value of \( x \).
 *
 * Implements fixed-step gradient descent:
 *   \( x_{k+1} = x_k - \eta \cdot \frac{1}{N} \sum_{i=1}^N 2 a_i (x_k - b_i) \)
 * with \( \eta = 0.01 \), tolerance \( 10^{-6} \), max 10,000 iterations.
 *
 * Uses a **single reusable ThreadPool** across all iterations to minimize
 * overhead.  Gradient contributions are computed in parallel and aggregated
 * via a mutex-protected accumulator.  An atomic counter tracks pending tasks
 * to avoid busy-waiting.
 */
double CollaborativeParallelStrategy::optimize(const std::vector<Agent>& agents,
                                               double& iterations, double& time_taken) {
    const size_t N = agents.size();
    ThreadPool pool;  // ONE pool for all iterations — avoids reconstruction cost.
    double x = 0.0;
    const double lr = 0.01;
    const double tol = 1e-6;
    const int max_it = 10000;
    iterations = 0;

    while (iterations < max_it) {
        double total_grad = 0.0;
        std::mutex grad_mtx;
        std::atomic<size_t> pending(N);

        for (size_t i = 0; i < N; ++i) {
            pool.enqueue([&, i] {
                double g = agents[i].computeGradient(x);
                std::lock_guard<std::mutex> lk(grad_mtx);
                total_grad += g;
                --pending;
            });
        }

        // Wait for all gradient tasks to complete.
        while (pending > 0) std::this_thread::yield();

        double avg_grad = total_grad / N;
        if (std::abs(avg_grad) < tol) break;

        x -= lr * avg_grad;
        ++iterations;
    }
    return x;
}