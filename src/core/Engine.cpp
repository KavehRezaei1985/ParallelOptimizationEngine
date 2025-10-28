// src/core/Engine.cpp
//
// Implements the **OptimizationEngine** facade and all CPU-based optimization
// strategies for the **ParallelOptimizationEngine** framework.
//
// This file provides:
// • A unified runtime interface (`OptimizationEngine::run`) that measures
// wall-clock execution time with high-resolution chronometry.
// • Three **naive** strategies (sequential, parallel CPU, ML placeholder).
// • Three **collaborative** strategies using gradient consensus with
// high-performance `ThreadPool` or OpenMP for data-parallel gradient summation.
//
// All strategies adhere to the **Strategy Pattern**, enabling runtime
// polymorphism and seamless integration with GPU and ML backends via the
// Python facade. Thread-safety is enforced via RAII (`std::mutex`, `std::atomic`),
// and the `ThreadPool` eliminates per-iteration thread creation overhead.
//
// Mathematical foundation:
// • Naive: \( x^* = \frac{1}{N} \sum b_i \) (unweighted average of local minima)
// • Collaborative: Gradient descent on \( F(x) = \sum a_i (x - b_i)^2 \)
// with diminishing step size \( \eta = \eta_0 / k \), convergence when
// \( \|x_{k+1} - x_k\| < 10^{-9} \).
#include "Engine.hpp"
#include "ThreadPool.hpp"
#include <chrono>
#include <numeric>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <future>
#include <vector>
/**
 * @brief Executes the active optimization strategy and records performance metrics.
 *
 * @param agents Const reference to the agent ensemble.
 * @param iterations Reference to store the number of optimization steps.
 * @param time_taken Reference to store wall-clock execution time in seconds.
 * @return double Final optimized value of the shared variable \( x \).
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
 * @param agents Input agent ensemble.
 * @param iterations Set to 1.0 (single evaluation).
 * @param time_taken Not used (timing handled by OptimizationEngine).
 * @return double \( x^* = \frac{1}{N} \sum b_i \)
 *
 * O(N) sequential loop. Fastest baseline; ignores coefficient weights \( a_i \).
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
 * @param agents Input agent ensemble.
 * @param iterations Set to 1.0.
 * @param time_taken Not used.
 * @return double Parallel reduction of local minima.
 *
 * Each agent’s local minimum is computed in a separate task enqueued to a
 * reusable `ThreadPool`. Partial sums are collected via futures and aggregated
 * sequentially to ensure consistency.
 */
double NaiveParallelCPUStrategy::optimize(const std::vector<Agent>& agents,
double& iterations, double& time_taken) {
iterations = 1.0;
std::vector<std::future<double>> futures;
ThreadPool pool;
size_t N = agents.size();
size_t num_chunks = (N + pool.thread_count() - 1) / pool.thread_count();
size_t chunk_size = (N + num_chunks - 1) / num_chunks;
for (size_t i = 0; i < num_chunks && i * chunk_size < N; ++i) {
size_t start = i * chunk_size;
size_t end = std::min((i + 1) * chunk_size, N);
futures.push_back(pool.enqueue([start, end, &agents] () {
double local_sum = 0.0;
for (size_t j = start; j < end; ++j) {
local_sum += agents[j].getLocalMin();
            }
return local_sum;
        }));
    }
double total = 0.0;
for (auto& future : futures) {
total += future.get();
    }
return total / N;
}
// Added: New function for NaiveOpenMPStrategy::optimize
// Reason: To implement an OpenMP-based parallel variant for the naive strategy, using reduction for efficient summation without locks, as per the requirement to add OpenMP variants for parallel gradient sum and to provide a contrast to ThreadPool for performance comparisons in the perf addendum.
/**
 * @brief Naive parallel CPU strategy using OpenMP reduction.
 *
 * @param agents Input agent ensemble.
 * @param iterations Set to 1.0.
 * @param time_taken Not used.
 * @return double Parallel reduction of local minima.
 *
 * Uses OpenMP parallel for loop with reduction to sum local minima.
 * Provides a contrast to ThreadPool-based parallelism.
 */
double NaiveOpenMPStrategy::optimize(const std::vector<Agent>& agents,
double& iterations, double& time_taken) {
iterations = 1.0;
double total = 0.0;
#pragma omp parallel for reduction(+:total)
for (size_t i = 0; i < agents.size(); ++i) {
total += agents[i].getLocalMin();
    }
return total / agents.size();
}
/**
 * @brief Naive ML strategy placeholder.
 *
 * @param agents Input agent ensemble.
 * @param iterations Set to 1.0 (placeholder, real ML in Python).
 * @param time_taken Set to 0.0 (placeholder).
 * @return double Identical to sequential naive (real ML in Python layer).
 *
 * This stub enables uniform interface testing. Actual ML prediction
 * (e.g., neural approximation of mean \( b_i \)) is implemented in
 * `python/ml_agent.py` and invoked via Python bindings.
 */
double NaiveMLStrategy::optimize(const std::vector<Agent>& agents,
double& iterations, double& time_taken) {
iterations = 1.0;
double sum = 0.0;
for (const auto& ag : agents) sum += ag.getLocalMin();
return sum / agents.size(); // Placeholder – real ML in Python
}
/* ======================= COLLABORATIVE STRATEGIES ======================= */
// Added: New function for CollaborativeSequentialStrategy::optimize
// Reason: To provide a sequential baseline for collaborative optimization, enabling comparisons with parallel variants, and to incorporate the requested convergence policy (diminishing step size η_k = η_0 / k and stop on |x_{k+1} - x_k| < ε), preserving the overall gradient descent logic.
/**
 * @brief Collaborative sequential strategy using consensus gradient descent.
 *
 * @param agents Input agent ensemble.
 * @param iterations Number of GD iterations performed.
 * @param time_taken Not used (measured externally).
 * @return double Converged value of \( x \).
 *
 * Implements diminishing-step gradient descent:
 * \( x_{k+1} = x_k - \eta_k \cdot \sum_{i=1}^N 2 a_i (x_k - b_i) \)
 * with \( \eta_k = \eta_0 / k \), \( \eta_0 = 0.01 \), stopping when
 * \( \|x_{k+1} - x_k\| < 10^{-9} \), max 1,000,000 iterations.
 *
 * Uses a sequential loop for gradient summation without normalization.
 */
double CollaborativeSequentialStrategy::optimize(const std::vector<Agent>& agents,
double& iterations, double& time_taken) {
const size_t N = agents.size();
double x = 0.0;
const double eta_0 = 0.01;
const double tol = 1e-9;
const int max_it = 1000000;
iterations = 0;
while (iterations < max_it) {
double total_grad = 0.0;
for (const auto& ag : agents) {
total_grad += ag.computeGradient(x);
        }
double eta = eta_0 / (iterations + 1); // Diminishing step size
double x_new = x - eta * total_grad; // Use raw total_grad without 1/N
if (std::abs(x_new - x) < tol) break;
x = x_new;
++iterations;
    }
return x;
}
// Modified: Updated CollaborativeParallelStrategy::optimize from old single-task-per-agent with atomic pending/yield wait to chunk-based enqueuing, mutex-protected accumulation, and pool.wait().
// Reason: To improve scalability for large N (as per perf addendum), reduce mutex contention by fewer larger tasks, and integrate with updated ThreadPool (now supporting futures and wait()); also added diminishing η and delta-x tol to fulfill convergence policy request.
/**
 * @brief Collaborative parallel strategy using consensus gradient descent.
 *
 * @param agents Input agent ensemble.
 * @param iterations Number of GD iterations performed.
 * @param time_taken Not used (measured externally).
 * @return double Converged value of \( x \).
 *
 * Implements diminishing-step gradient descent:
 * \( x_{k+1} = x_k - \eta_k \cdot \sum_{i=1}^N 2 a_i (x_k - b_i) \)
 * with \( \eta_k = \eta_0 / k \), \( \eta_0 = 0.01 \), stopping when
 * \( \|x_{k+1} - x_k\| < 10^{-9} \), max 1,000,000 iterations.
 *
 * Uses a **single reusable ThreadPool** across all iterations to minimize
 * overhead. Gradient contributions are computed in parallel and aggregated
 * via a mutex-protected accumulator.
 */
double CollaborativeParallelStrategy::optimize(const std::vector<Agent>& agents,
double& iterations, double& time_taken) {
const size_t N = agents.size();
ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
double x = 0.0;
const double eta_0 = 0.01;
const double tol = 1e-9;
const int max_it = 1000000;
iterations = 0;
std::mutex m;
while (iterations < max_it) {
double total_grad = 0.0;
size_t chunk_size = (N + pool.thread_count() - 1) / pool.thread_count();
for (size_t i = 0; i < N; i += chunk_size) {
size_t start = i;
size_t end = std::min(i + chunk_size, N);
pool.enqueue([&total_grad, &m, start, end, x, &agents] () {
double local_grad = 0.0;
for (size_t j = start; j < end; ++j) {
local_grad += agents[j].computeGradient(x);
                }
std::lock_guard<std::mutex> lk(m);
total_grad += local_grad;
            });
        }
pool.wait();
double eta = eta_0 / (iterations + 1); // Diminishing step size
double x_new = x - eta * total_grad; // Use raw total_grad without 1/N
if (std::abs(x_new - x) < tol) break;
x = x_new;
++iterations;
    }
return x;
}
// Added: New function for CollaborativeOpenMPStrategy::optimize
// Reason: To implement an OpenMP-based parallel variant for the collaborative strategy, using reduction for efficient lock-free summation, as per the requirement to add OpenMP variants and provide performance contrasts in the perf addendum.
/**
 * @brief Collaborative parallel strategy using consensus gradient descent with OpenMP.
 *
 * @param agents Input agent ensemble.
 * @param iterations Number of GD iterations performed.
 * @param time_taken Not used (measured externally).
 * @return double Converged value of \( x \).
 *
 * Implements diminishing-step gradient descent:
 * \( x_{k+1} = x_k - \eta_k \cdot \sum_{i=1}^N 2 a_i (x_k - b_i) \)
 * with \( \eta_k = \eta_0 / k \), \( \eta_0 = 0.01 \), stopping when
 * \( \|x_{k+1} - x_k\| < 10^{-9} \), max 1,000,000 iterations.
 *
 * Uses OpenMP parallel for loop with reduction to sum gradients.
 */
double CollaborativeOpenMPStrategy::optimize(const std::vector<Agent>& agents,
double& iterations, double& time_taken) {
const size_t N = agents.size();
double x = 0.0;
const double eta_0 = 0.01;
const double tol = 1e-9;
const int max_it = 1000000;
iterations = 0;
while (iterations < max_it) {
double total_grad = 0.0;
#pragma omp parallel for reduction(+:total_grad)
for (size_t i = 0; i < N; ++i) {
total_grad += agents[i].computeGradient(x);
        }
double eta = eta_0 / (iterations + 1); // Diminishing step size
double x_new = x - eta * total_grad; // Use raw total_grad without 1/N
if (std::abs(x_new - x) < tol) break;
x = x_new;
++iterations;
    }
return x;
}
// Added: New function for CollaborativeMLStrategy::optimize
// Reason: To provide a placeholder for the ML variant in collaborative mode, ensuring uniform interface across strategies, with real implementation in Python (ml_agent.py) for PyTorch integration.
/**
 * @brief Collaborative ML strategy placeholder.
 *
 * @param agents Input agent ensemble.
 * @param iterations Set to 0.0 (placeholder, real ML in Python).
 * @param time_taken Set to 0.0 (placeholder).
 * @return double Placeholder value (real ML in Python).
 *
 * This stub enables uniform interface testing. Actual ML prediction
 * (e.g., neural approximation of average gradients) is implemented in
 * `python/ml_agent.py` and invoked via Python bindings.
 */
double CollaborativeMLStrategy::optimize(const std::vector<Agent>& agents,
double& iterations, double& time_taken) {
iterations = 0.0;
time_taken = 0.0;
return 0.0;
}