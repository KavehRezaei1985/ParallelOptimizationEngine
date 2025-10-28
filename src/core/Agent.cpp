// src/core/Agent.cpp
//
// Implementation of the **Agent** class for the **ParallelOptimizationEngine**.
// This file defines the core mathematical operations of a single reasoning
// agent in the intelligence fabric simulation.
//
// Each agent encapsulates a **convex quadratic cost function**:
// \( f_i(x) = a_i (x - b_i)^2 \), where:
// • \( a_i > 0 \): Ensures strong convexity and unique global minimum.
// • \( b_i \): Location of the agent's local minimum.
//
// The class provides:
// • Cost evaluation: \( f_i(x) \)
// • Gradient computation: \( \nabla f_i(x) = 2 a_i (x - b_i) \)
// • Local minimum access: \( x^*_i = b_i \)
//
// All methods are `constexpr`-compatible and use double-precision arithmetic
// for numerical stability in large-scale optimization. The implementation
// follows RAII principles and is exception-safe.
#include "Agent.hpp"
/**
 * @brief Constructs an agent with given quadratic coefficients.
 *
 * @param a_val Positive scaling factor \( a_i > 0 \) (convexity enforced externally).
 * @param b_val Target point \( b_i \) defining the local minimum.
 *
 * The constructor performs direct member initialization for zero-overhead
 * setup. No validation is performed here; convexity is guaranteed by
 * `generateAgents()` in util.cpp.
 */
Agent::Agent(double a_val, double b_val) : a(a_val), b(b_val) {}
/**
 * @brief Evaluates the local quadratic cost at point \( x \).
 *
 * @param x Evaluation point in the shared decision space.
 * @return double Cost value \( f_i(x) = a_i (x - b_i)^2 \).
 *
 * The computation uses explicit multiplication to avoid temporary objects
 * and ensure optimal inlining by the compiler. The result is non-negative
 * due to the squared term and \( a_i > 0 \).
 */
double Agent::computeCost(double x) const {
return a * (x - b) * (x - b); // \( f_i(x) = a_i (x - b_i)^2 \)
}
/**
 * @brief Computes the gradient of the local cost with respect to \( x \).
 *
 * @param x Point at which to evaluate the derivative.
 * @return double Gradient \( \nabla f_i(x) = 2 a_i (x - b_i) \).
 *
 * The gradient is zero at the local minimum \( x = b_i \), positive when
 * \( x > b_i \), and negative when \( x < b_i \), enabling convergence in
 * consensus-based gradient descent.
 */
double Agent::computeGradient(double x) const {
return 2.0 * a * (x - b); // \( \frac{\partial f_i}{\partial x} = 2 a_i (x - b_i) \)
}