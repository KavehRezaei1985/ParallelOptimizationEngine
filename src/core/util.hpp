// src/core/util.hpp
//
// Utility header for the **ParallelOptimizationEngine** framework.
// Provides helper functions for agent generation, global cost computation,
// and closed-form solution calculation.
//
// Functions are designed for efficiency and numerical stability, using
// double-precision arithmetic and RAII for resource management.
// All utilities are thread-safe when used with const inputs.
#ifndef UTIL_HPP
#define UTIL_HPP
#include <vector>
#include "Agent.hpp"

/**
 * @brief Generates N agents with random quadratic coefficients.
 *
 * @param N Number of agents to generate (must be positive).
 * @return std::vector<Agent> Vector of randomly initialized agents.
 *
 * Coefficients: a_i ~ N(5, 2) (rejected if <= 0), b_i ~ N(0, 5).
 * Uses a random seed for reproducibility in testing.
 *
 * Modified: Simplified description from "statistically distributed convex quadratic agents" to "agents with random quadratic coefficients"; updated seeding note from "high-resolution clock" to "random seed" (though inaccurately stated as "static seed" in original).
 * Reason: To align with util.cpp's use of std::random_device for reproducibility in correctness tests and simplify documentation for clarity, removing LaTeX for consistency across the codebase.
 */
std::vector<Agent> generateAgents(int N);

/**
 * @brief Computes the global cost F(x) = sum(a_i (x - b_i)^2).
 *
 * @param agents Const reference to the agent ensemble.
 * @param x Decision variable to evaluate.
 * @return double Total cost across all agents (non-negative).
 *
 * Modified: Simplified description from LaTeX-based formula and complexity details to plain text; removed explicit complexity and modularity notes.
 * Reason: To streamline documentation for readability, align with util.cpp's concise style, and avoid LaTeX for consistent rendering across environments.
 */
double computeGlobalCost(const std::vector<Agent>& agents, double x);

/**
 * @brief Computes the closed-form global minimum x* = (sum a_i b_i) / (sum a_i).
 *
 * @param agents Const reference to the agent ensemble.
 * @return double The exact global minimum x*.
 *
 * Added: New function declaration for correctness testing.
 * Reason: To support the correctness harness requirement by providing the analytical solution for verifying solver outputs in test_correctness.cpp/py, ensuring scientific rigor across all modes.
 */
double computeClosedForm(const std::vector<Agent>& agents);

#endif // UTIL_HPP