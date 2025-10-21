// src/core/Agent.hpp
//
// Header defining the **Agent** class for the **ParallelOptimizationEngine**.
// This class models a single reasoning agent within a multi-agent intelligence
// fabric, where each agent contributes a **convex quadratic cost function**:
//
// \[
// f_i(x) = a_i (x - b_i)^2
// \]
//
// - \( a_i > 0 \): Positive scaling coefficient ensuring **strong convexity**.
// - \( b_i \): Location of the agent's local minimum (ideal decision point).
//
// The class provides:
//   • Construction with validated coefficients.
//   • Cost evaluation: \( f_i(x) \)
//   • Gradient computation: \( \nabla f_i(x) = 2 a_i (x - b_i) \)
//   • Direct access to local minimum: \( x^*_i = b_i \)
//
// All operations are `constexpr`-compatible and use double-precision
// arithmetic for numerical robustness in large-scale parallel optimization.
// The design follows RAII, is exception-safe, and supports inlining for
// maximum performance across CPU, GPU, and ML backends.

#ifndef AGENT_HPP
#define AGENT_HPP

/**
 * @class Agent
 * @brief Represents a convex quadratic reasoning agent in the optimization fabric.
 *
 * Each agent encapsulates a local objective function used in collaborative
 * consensus optimization.  The quadratic form guarantees a unique global
 * minimum when aggregated across all agents.
 */
class Agent {
public:
    double a;  ///< Scaling coefficient \( a_i > 0 \), enforces strong convexity.
    double b;  ///< Target point \( b_i \), defines the agent's local minimum.

    /**
     * @brief Constructs an agent with specified quadratic coefficients.
     *
     * @param a_val Positive coefficient \( a_i > 0 \) (convexity enforced upstream).
     * @param b_val Target value \( b_i \) representing the agent's preferred state.
     *
     * Direct member initialization ensures zero-cost setup.  Coefficient
     * validation (e.g., \( a_i > 0 \)) is performed during agent generation.
     */
    Agent(double a_val, double b_val);

    /**
     * @brief Computes the local cost at decision point \( x \).
     *
     * @param x Evaluation point in the shared parameter space.
     * @return double Cost \( f_i(x) = a_i (x - b_i)^2 \geq 0 \).
     *
     * Inlined implementation in Agent.cpp ensures optimal performance
     * in tight inner loops of gradient-based optimization.
     */
    double computeCost(double x) const;

    /**
     * @brief Computes the gradient of the local cost with respect to \( x \).
     *
     * @param x Point at which to evaluate the derivative.
     * @return double Gradient \( \nabla f_i(x) = 2 a_i (x - b_i) \).
     *
     * The sign indicates direction toward the local minimum:
     *   • \( > 0 \): pull \( x \) downward
     *   • \( < 0 \): pull \( x \) upward
     */
    double computeGradient(double x) const;

    /**
     * @brief Returns the agent's local minimum.
     *
     * @return double \( x^*_i = b_i \)
     *
     * Direct access enables fast averaging in naive strategies.
     * Marked `constexpr` for compile-time evaluation when possible.
     */
    double getLocalMin() const { return b; }
};

#endif // AGENT_HPP