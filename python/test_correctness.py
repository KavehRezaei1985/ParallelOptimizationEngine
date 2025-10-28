# python/test_correctness.py
#
# Title: Correctness Test Suite for ParallelOptimizationEngine Solvers
#
# Description:
# This file implements a Pytest suite designed to verify the correctness of the solvers within the
# **ParallelOptimizationEngine** framework. It tests all solver modes (naive, naive_threadpool, naive_openmp,
# naive_gpu, naive_ml, collaborative_cpu, collaborative_threadpool, collaborative_openmp, collaborative_gpu,
# collaborative_ml) against the closed-form solution for various numbers of agents (N) and a dynamically
# generated list of random seeds. The suite ensures numerical accuracy and robustness.
# Purpose:
# Added to the repository to provide a rigorous unit testing framework that validates solver outputs
# against the analytical closed-form solution \( x^* = \frac{\sum a_i b_i}{\sum a_i} \). This ensures
# the optimization algorithms produce results within a specified tolerance (default 1e-6), enhancing
# scientific rigor and reliability across diverse hardware and configurations.
#
# Test Configuration:
# - **Test Cases**: Parametrized over N (100, 1000, 10000), random seeds (5 per run), and method/mode pairs.
# - **Tolerance**: Absolute error \( |x - x^*| \leq 1e-6 \).
# - **Random Seeds**: Generated dynamically at runtime to meet reproducibility and variability requirements.
#
# Dependencies:
# - pytest: For test framework and parameterization
# - numpy: For numerical operations and random seed generation
# - poe_bindings: For C++ integration (Agent, create_strategy, computeClosedForm)
# - ml_agent: For ML-enhanced optimization model
# - run_simulation: For agent generation and solver execution functions
#
import pytest
import numpy as np
import random
from poe_bindings import Agent, create_strategy, computeClosedForm
from ml_agent import MLGradientPredictor
from run_simulation import generate_agents, run_naive, run_collaborative

# Generate a list of random seeds at runtime for this test session
RANDOM_SEEDS = [random.randint(0, 2**32 - 1) for _ in range(5)]  # Define 5 random seeds per run for variability; added to meet Part 2, Task 3

@pytest.mark.parametrize("N", [100, 1000, 10000])  # Test across a range of agent counts for scalability
@pytest.mark.parametrize("seed", RANDOM_SEEDS)  # Test with dynamically generated seeds for reproducibility
@pytest.mark.parametrize("method,mode", [
    ("naive", "cpu"),
    ("naive", "threadpool"),
    ("naive", "openmp"),
    ("naive", "gpu"),
    ("naive", "ml"),
    ("collaborative", "cpu"),  # Maps to collaborative_sequential
    ("collaborative", "threadpool"),
    ("collaborative", "openmp"),
    ("collaborative", "gpu"),
    ("collaborative", "ml")
])  # Test all solver mode combinations
def test_solver_correctness(N, seed, method, mode):
    """Test solver correctness against closed-form solution with random seeds and agent data.
    
    Args:
        N (int): Number of agents to test
        seed (int): Random seed for agent generation
        method (str): Optimization method (naive or collaborative)
        mode (str): Execution mode (cpu, threadpool, openmp, gpu, ml)
    
    Notes:
    - Uses a tolerance of 1e-6 to ensure numerical accuracy.
    - Generates agents with the specified seed for deterministic testing.
    - Compares computed x with closed-form x* for error checking.
    """
    EPSILON = 1e-6  # Define tolerance for acceptable error
    args = type('Args', (), {'N': N, 'max_iter': 10000, 'tolerance': EPSILON})()  # Create mock args object
    agents = generate_agents(N, seed)  # Generate agents with fixed seed for reproducibility
    
    x_star = computeClosedForm(agents)  # Compute analytical closed-form solution for validation
    
    # Extract a_i and b_i for each agent to log test data
    a_values = [agent.a for agent in agents]
    b_values = [agent.b for agent in agents]
    
    # Print test case details for debugging and verification
    print(f"Seed={seed}, N={N}, method={method}, mode={mode}")
    print(f"Closed-form solution x*={x_star}")
    print("Agent data: a_i, b_i")
    for i in range(N):
        print(f"Agent {i}: a_i={a_values[i]}, b_i={b_values[i]}")
    
    x = None  # Initialize x to avoid UnboundLocalError, overwritten by solver
    if method == "naive":
        x, _, _ = run_naive(agents, mode, args)
    else:
        x, _, _ = run_collaborative(agents, mode, args)
    
    # Print computed result and error for analysis
    print(f"Computed x={x}")
    print(f"Comparison: x*={x_star}, x={x}, Difference={abs(x - x_star)}")
    error = abs(x - x_star)
    assert error <= EPSILON, f"Mode {mode} failed: |x - x*| = {error} > {EPSILON} (x={x}, x*={x_star}, seed={seed})"
    print(f"Error={error}")  # Optional: Print error for reference

# Optional: Print the generated seeds for reference at session start
def pytest_sessionstart(session):
    """Print generated random seeds at the start of the test session.
    
    Args:
        session (pytest.Session): Pytest session object
    
    Notes:
    - Added to provide transparency on seed values used for each test run.
    """
    print(f"Generated random seeds for this run: {RANDOM_SEEDS}")