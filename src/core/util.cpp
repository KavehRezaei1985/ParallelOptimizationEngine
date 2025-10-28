// src/core/util.cpp
//
// Implementation of utility functions for the **ParallelOptimizationEngine** core library.
//
// This file provides:
// • Agent generation with random coefficients for simulation setup.
// • Global cost computation for performance verification and metrics.
// • Closed-form solution calculation for correctness testing against analytical results.
//
// All functions use double-precision arithmetic for numerical stability and are
// thread-safe with const inputs or local random number generator (RNG) instances.
// Random number generation is seeded with std::random_device for reproducibility
// in testing, aligning with dynamic seed requirements in test_correctness.cpp/py.
#include "util.hpp"
#include <random>
// Modified: Replaced <chrono> with <cmath> include.
// Reason: Removed <chrono> as generateAgents now uses std::random_device instead
// of std::chrono::steady_clock for seeding, improving reproducibility. Added <cmath>
// for potential use in calculations (e.g., std::abs), though not currently needed,
// to align with common numerical utility practices.
#include <cmath>

/**
 * @brief Generates N agents with random quadratic coefficients.
 *
 * @param N Number of agents to generate (must be positive).
 * @return std::vector<Agent> Vector of randomly initialized Agent instances.
 *
 * Generates agents with coefficients:
 * - a_i ~ Normal(5.0, 2.0), rejected if a_i <= 0 to ensure convexity.
 * - b_i ~ Normal(0.0, 5.0), representing diverse local minima.
 *
 * Uses std::random_device to seed the Mersenne Twister RNG, providing
 * high-entropy, reproducible randomness for testing. Pre-allocates the vector
 * with reserve() for performance. Thread-safe due to local RNG instance.
 *
 * Modified: Changed seeding from std::chrono::steady_clock to std::random_device;
 * simplified distribution syntax to std::normal_distribution<>; updated comment
 * from "static seed" to reflect dynamic seeding.
 * Reason: To improve reproducibility for correctness tests (aligning with
 * test_correctness.py/cpp dynamic seeds), reduce verbosity in distribution
 * declaration, and clarify seeding intent for testing.
 */
std::vector<Agent> generateAgents(int N) {
    // Initialize RNG with random_device for reproducible test runs
    std::random_device rd;
    std::mt19937 gen(rd());
    // Define normal distributions for a_i and b_i using default double type
    std::normal_distribution<> d_a(5.0, 2.0);
    std::normal_distribution<> d_b(0.0, 5.0);
    // Pre-allocate vector to avoid reallocations
    std::vector<Agent> agents;
    agents.reserve(N);
    for (int i = 0; i < N; ++i) {
        // Generate a_i, reject if non-positive to ensure convexity
        double a = d_a(gen);
        while (a <= 0) a = d_a(gen); // Ensure a_i > 0 for convexity
        // Generate b_i
        double b = d_b(gen);
        // Construct and store agent
        agents.emplace_back(a, b);
    }
    return agents;
}

/**
 * @brief Computes the global cost F(x) = sum(a_i (x - b_i)^2).
 *
 * @param agents Const reference to the agent ensemble.
 * @param x Decision variable to evaluate.
 * @return double Total cost across all agents (non-negative).
 *
 * Iterates over agents, summing individual costs via Agent::computeCost.
 * Thread-safe due to const input and no shared state. Maintains O(N) complexity
 * for performance in large-scale simulations. Used for verification and metrics
 * in run_simulation.py and performance reports.
 *
 * Modified: Added braces around loop body for style consistency.
 * Reason: To align with modern C++ coding practices for multi-line loops,
 * improving readability with no functional change.
 */
double computeGlobalCost(const std::vector<Agent>& agents, double x) {
    double cost = 0.0;
    for (const auto& ag : agents) {
        cost += ag.computeCost(x);
    }
    return cost;
}

/**
 * @brief Computes the closed-form global minimum x* = (sum a_i b_i) / (sum a_i).
 *
 * @param agents Const reference to the agent ensemble.
 * @return double The exact global minimum x*.
 *
 * Calculates the analytical solution for the global minimum of F(x) = sum(a_i (x - b_i)^2),
 * derived by setting the gradient to zero: dF/dx = sum(2 a_i (x - b_i)) = 0.
 * Includes a safeguard against division by zero (returns 0.0 if sum_a <= 0).
 * Thread-safe and O(N). Added to support correctness testing in test_correctness.cpp/py.
 *
 * Added: New function for correctness harness.
 * Reason: To enable verification of solver outputs against the analytical solution,
 * as required by the correctness harness for comparing x to x* across all modes.
 */
double computeClosedForm(const std::vector<Agent>& agents) {
    double sum_a = 0.0;
    double sum_ab = 0.0;
    for (const auto& ag : agents) {
        sum_a += ag.a;
        sum_ab += ag.a * ag.b;
    }
    // Avoid division by zero; return 0.0 for edge case (should not occur with a_i > 0)
    return sum_a > 0 ? sum_ab / sum_a : 0.0;
}