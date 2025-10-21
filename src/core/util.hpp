// src/core/util.hpp
//
// Header declaring utility functions for the **ParallelOptimizationEngine** core library.
// Provides high-level abstractions for:
//   • Random generation of convex quadratic agents with statistically controlled
//     coefficient distributions.
//   • Efficient evaluation of the global aggregated cost function across an
//     ensemble of agents.
//
// These utilities are designed for seamless integration with optimization
// strategies (naive, collaborative, GPU, ML) and support both standalone
// testing and large-scale HPC simulations.
//
// All functions are `constexpr`-compatible where possible and follow RAII
// principles. Thread-safety is ensured via local RNG instances in implementations.
//
// Mathematical context:
//   • Agent cost: \( f_i(x) = a_i (x - b_i)^2 \), \( a_i > 0 \)
//   • Global cost: \( F(x) = \sum_{i=1}^N f_i(x) \)
//
// Include this header in any translation unit requiring agent ensembles or
// performance benchmarking.

#ifndef UTIL_HPP
#define UTIL_HPP

#include <vector>
#include "Agent.hpp"

/**
 * @brief Generates a vector of N statistically distributed convex quadratic agents.
 *
 * @param N Number of agents to generate (must be positive).
 * @return std::vector<Agent> Ensemble of initialized agents.
 *
 * Distribution details (defined in util.cpp):
 *   • \( a_i \sim \mathcal{N}(5.0, 2.0) \), rejection-sampled to ensure \( a_i > 0 \)
 *   • \( b_i \sim \mathcal{N}(0.0, 5.0) \)
 *
 * The RNG is seeded from a high-resolution clock by default. For deterministic
 * runs, seed `std::mt19937` externally before invocation.
 */
std::vector<Agent> generateAgents(int N);

/**
 * @brief Evaluates the global cost \( F(x) = \sum_{i=1}^N a_i (x - b_i)^2 \).
 *
 * @param agents Constant reference to the agent ensemble.
 * @param x      Point at which to evaluate the global objective.
 * @return double Aggregated cost (≥ 0).
 *
 * Complexity: \( O(N) \). Uses Agent::computeCost() internally for modularity.
 */
double computeGlobalCost(const std::vector<Agent>& agents, double x);

#endif // UTIL_HPP