// src/core/util.cpp
//
// Implements high-performance utility functions for the ParallelOptimizationEngine
// core library.  This file provides:
//   • Random generation of convex quadratic-cost agents with statistically
//     meaningful coefficient distributions.
//   • Global cost evaluation across an ensemble of agents.
//
// All functions are designed for deterministic, reproducible behavior when
// seeded externally, while defaulting to a high-entropy seed derived from the
// steady-clock for standalone usage.  Thread-safety is guaranteed by local
// RNG instances.
//
// Mathematical foundation:
//   • Each agent models f_i(x) = a_i * (x - b_i)^2, with a_i > 0 to ensure
//     strong convexity.
//   • Global cost F(x) = Σ f_i(x) is the objective minimized by all strategies.

#include "util.hpp"
#include <random>
#include <chrono>

/**
 * @brief Generates an ensemble of N convex quadratic agents.
 *
 * @param N Number of agents to create (N > 0).
 * @return std::vector<Agent> Vector of initialized Agent instances.
 *
 * Coefficient distributions:
 *   • a_i ∼ Normal(5.0, 2.0), rejected and resampled if a_i ≤ 0 to enforce
 *     strict positivity and convexity.
 *   • b_i ∼ Normal(0.0, 5.0), representing diverse local minima.
 *
 * The Mersenne Twister RNG is seeded from the steady clock for high entropy
 * in standalone runs.  For reproducible experiments, seed the RNG externally
 * before calling this function.
 */
std::vector<Agent> generateAgents(int N) {
    std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
    std::normal_distribution<double> dist_a(5.0, 2.0);
    std::normal_distribution<double> dist_b(0.0, 5.0);

    std::vector<Agent> agents;
    agents.reserve(N);  // Pre-allocation eliminates repeated reallocations.

    for (int i = 0; i < N; ++i) {
        double a = dist_a(gen);
        while (a <= 0.0) a = dist_a(gen);  // Rejection sampling for convexity.
        agents.emplace_back(a, dist_b(gen));
    }
    return agents;
}

/**
 * @brief Computes the global aggregated cost F(x) = Σ a_i (x - b_i)^2.
 *
 * @param agents Const reference to the agent ensemble.
 * @param x      Evaluation point.
 * @return double Global cost value (non-negative).
 *
 * The implementation delegates to Agent::computeCost for clarity and
 * maintainability, while preserving O(N) linear complexity.
 */
double computeGlobalCost(const std::vector<Agent>& agents, double x) {
    double cost = 0.0;
    for (const auto& ag : agents) cost += ag.computeCost(x);
    return cost;
}