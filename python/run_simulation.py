# python/run_simulation.py
#
# Title: Main Executable Script for ParallelOptimizationEngine
#
# Description:
# This script serves as the primary executable for the **ParallelOptimizationEngine**, orchestrating
# multi-agent optimization simulations. It acts as a **Facade**, abstracting the complexity of
# C++/CUDA/ML integrations behind a user-friendly CLI interface. The module leverages hardware
# detection via PyTorch and supports scalable execution for varying numbers of agents (N).
#
# Functionality:
# • Parses command-line arguments for configuration (N, method, mode, etc.).
# • Generates random agents with quadratic costs f_i(x) = a_i (x - b_i)^2.
# • Executes optimization using specified method (naive/collaborative) and mode
#   (cpu/threadpool/openmp/gpu/ml).
# • Computes metrics: final x, global cost, iterations, runtime, accuracy gap.
# • Visualizes results via scaling plots (scaling.png) with a 2x2 grid of iterations,
#   time, accuracy gap, and speedup vs. N.
# • Saves raw metrics to performance_data.csv.
#
# Dependencies:
# - argparse: For command-line argument parsing
# - numpy: For numerical operations and random agent generation
# - torch: For hardware detection
# - poe_bindings: For C++ integration (Agent, create_strategy, computeClosedForm)
# - ml_agent: For ML-enhanced optimization
# - visualize: For visualization output
# - pandas: For data storage in CSV
# - time: For performance timing
#
import argparse
import numpy as np
import torch
from poe_bindings import Agent, create_strategy, computeClosedForm
from ml_agent import MLGradientPredictor
import visualize
import pandas as pd
import time

def parse_args():
    """Parse command-line arguments for simulation configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments including N, method, mode, max_iter, and tolerance.
    
    Notes:
    - Supports multiple N values via comma-separated string for scalability testing.
    - Expanded mode options reflect additional parallel backends (threadpool, openmp).
    """
    parser = argparse.ArgumentParser(description="Parallel Optimization Engine")
    parser.add_argument("--N", type=str, default="100,1000,10000", 
                        help="Comma-separated list of number of agents (e.g., 100,500,1000)")  # Changed to str for multi-N support
    parser.add_argument("--method", choices=["naive", "collaborative", "both"], default="both",
                        help="Optimization method: naive, collaborative, or both")
    parser.add_argument("--mode", choices=["cpu", "threadpool", "openmp", "gpu", "ml", "all"], default="all",
                        help="Execution mode: cpu (sequential), threadpool, openmp, gpu, ml, or all")  # Expanded mode choices
    parser.add_argument("--max_iter", type=int, default=10000, 
                        help="Maximum iterations for collaborative method")
    parser.add_argument("--tolerance", type=float, default=1e-6, 
                        help="Convergence tolerance for collaborative method")
    return parser.parse_args()

def generate_agents(N, seed=42):
    """Generate N agents with random coefficients a_i ~ N(5,2), b_i ~ N(0,5).
    
    Args:
        N (int): Number of agents to generate
        seed (int, optional): Random seed for reproducibility; defaults to 42
    
    Returns:
        list: List of Agent objects with random a and b coefficients
    
    Notes:
    - Added to centralize agent generation within Python, replacing C++ dependency.
    - Ensures convexity by setting a_i > 0.
    - Uses a fixed seed for deterministic results.
    """
    np.random.seed(seed)
    a = np.random.normal(5.0, 2.0, N)
    a = np.maximum(a, 1e-6)  # Ensure a_i > 0 for mathematical convexity
    b = np.random.normal(0.0, 5.0, N)
    return [Agent(float(a[i]), float(b[i])) for i in range(N)]

def run_naive(agents, mode, args):
    """Execute naive optimization (average of local minima) for given mode.
    
    Args:
        agents (list): List of Agent objects
        mode (str): Execution mode (cpu, threadpool, openmp, gpu, ml)
        args (argparse.Namespace): Command-line arguments
    
    Returns:
        tuple: (final_x, iterations, time_taken) or (None, None, None) if mode invalid
    
    Notes:
    - Reordered arguments to prioritize agents for consistency with generation.
    - Added threadpool and openmp modes for parallel CPU execution.
    - Updated ML mode to use MLGradientPredictor.optimize with agent data, leveraging training time.
    """
    mode_mapping = {
        "cpu": "cpu",
        "threadpool": "threadpool",
        "openmp": "openmp",
        "gpu": "gpu",
        "ml": "ml"
    }
    if mode not in mode_mapping:
        return None, None, None  # Skip invalid modes to maintain robustness
    mapped_mode = mode_mapping[mode]
    if mapped_mode == "cpu":
        strategy = create_strategy("naive", "cpu")
    elif mapped_mode == "threadpool":
        strategy = create_strategy("naive", "threadpool")
    elif mapped_mode == "openmp":
        strategy = create_strategy("naive", "openmp")
    elif mapped_mode == "gpu":
        if not torch.cuda.is_available():
            print("GPU unavailable, falling back to CPU for naive")  # Notify user of fallback
            strategy = create_strategy("naive", "cpu")
        else:
            strategy = create_strategy("naive", "gpu")
    elif mapped_mode == "ml":
        predictor = MLGradientPredictor(agents, is_naive=True)
        x, iterations = predictor.optimize(agents)  # Unpack optimization result
        time_taken = predictor.training_time  # Use training time as proxy for ML execution
        return x, iterations, time_taken
    if strategy:
        start_time = time.time()
        x, iterations, time_taken = strategy.run(agents)
        time_taken = time.time() - start_time  # Measure actual execution time
        return x, iterations, time_taken
    return None, None, None

def run_collaborative(agents, mode, args):
    """Execute collaborative optimization (gradient descent) for given mode.
    
    Args:
        agents (list): List of Agent objects
        mode (str): Execution mode (cpu, threadpool, openmp, gpu, ml)
        args (argparse.Namespace): Command-line arguments (max_iter, tolerance)
    
    Returns:
        tuple: (final_x, iterations, time_taken) or (None, None, None) if mode invalid
    
    Notes:
    - Reordered arguments to prioritize agents for consistency.
    - Added threadpool and openmp modes for parallel CPU execution.
    - Updated ML mode to use MLGradientPredictor.optimize with max_iter and tolerance.
    """
    mode_mapping = {
        "cpu": "cpu",
        "threadpool": "threadpool",
        "openmp": "openmp",
        "gpu": "gpu",
        "ml": "ml"
    }
    if mode not in mode_mapping:
        return None, None, None  # Skip invalid modes to maintain robustness
    mapped_mode = mode_mapping[mode]
    if mapped_mode == "cpu":
        strategy = create_strategy("collaborative", "cpu")
    elif mapped_mode == "threadpool":
        strategy = create_strategy("collaborative", "threadpool")
    elif mapped_mode == "openmp":
        strategy = create_strategy("collaborative", "openmp")
    elif mapped_mode == "gpu":
        if not torch.cuda.is_available():
            print("GPU unavailable, falling back to CPU for collaborative")  # Notify user of fallback
            strategy = create_strategy("collaborative", "cpu")
        else:
            strategy = create_strategy("collaborative", "gpu")
    elif mapped_mode == "ml":
        predictor = MLGradientPredictor(agents, is_naive=False, max_iter=args.max_iter, tol=args.tolerance)
        x, iterations = predictor.optimize(agents)  # Unpack optimization result
        time_taken = predictor.training_time + predictor.iteration_time  # Include both training and iteration time
        return x, iterations, time_taken
    if strategy:
        start_time = time.time()
        x, iterations, time_taken = strategy.run(agents)
        time_taken = time.time() - start_time  # Measure actual execution time
        return x, iterations, time_taken
    return None, None, None

def compute_metrics(agents, x, iterations, time_taken, mode=None):
    """Compute global cost and accuracy gap to closed-form solution.
    
    Args:
        agents (list): List of Agent objects
        x (float): Optimized value
        iterations (int): Number of iterations
        time_taken (float): Execution time in seconds
        mode (str, optional): Execution mode; defaults to None
    
    Returns:
        tuple: (cost, accuracy_gap) or (None, None) if x is None
    
    Notes:
    - Added to compute performance metrics for visualization and validation.
    - Uses computeClosedForm for accurate reference solution.
    """
    if x is None:
        return None, None
    cost = sum(ag.computeCost(x) for ag in agents)
    x_star = computeClosedForm(agents)
    optimal_cost = sum(ag.computeCost(x_star) for ag in agents)  # Compute optimal cost for accuracy comparison
    accuracy_gap = abs(cost - optimal_cost) if optimal_cost != 0 else abs(cost)  # Handle division by zero
    return cost, accuracy_gap

def main():
    """Execute the main simulation workflow.
    
    Notes:
    - Processes multiple N values for scalability testing.
    - Normalizes speedup against CPU sequential baseline.
    - Generates comprehensive metrics and visualization.
    """
    args = parse_args()
    # Parse N as a comma-separated list, default to [100, 1000, 10000]; Added for multi-N scalability
    N_values = [int(n.strip()) for n in args.N.split(',')] if args.N else [100, 1000, 10000]
    methods = ["naive", "collaborative"] if args.method == "both" else [args.method]  # Support both methods
    modes = ["cpu", "threadpool", "openmp", "gpu", "ml"] if args.mode == "all" else [args.mode]  # Support all modes
    results = []
    baseline_times = {}  # Store baseline times and initial results for each N and method; Added for speedup normalization
    baseline_results = {}  # Store initial CPU-seq results; Added for speedup normalization
    for N in N_values:
        agents = generate_agents(N)
        for method in methods:
            # Set baseline to CPU sequential time and store initial result
            baseline_time = None
            baseline_x = None
            baseline_iterations = None
            for mode in modes:
                if mode == "cpu":
                    if method == "naive":
                        x, iterations, time_taken = run_naive(agents, mode, args)
                    else:
                        x, iterations, time_taken = run_collaborative(agents, mode, args)
                    if x is not None and time_taken is not None:
                        baseline_time = time_taken
                        baseline_x = x
                        baseline_iterations = iterations
                    break
            if baseline_time is None:
                baseline_time = float('inf')  # Fallback if CPU time not available
            baseline_times[(N, method)] = baseline_time
            baseline_results[(N, method)] = (baseline_x, baseline_iterations, baseline_time)
            for mode in modes:
                if method == "naive":
                    x, iterations, time_taken = run_naive(agents, mode, args)
                else:
                    x, iterations, time_taken = run_collaborative(agents, mode, args)
                if x is not None:
                    cost, accuracy_gap = compute_metrics(agents, x, iterations, time_taken)
                    if mode == "cpu":
                        # Use stored baseline time for CPU-seq to ensure speedup = 1.00x
                        baseline = baseline_times.get((N, method), time_taken)
                        baseline_x, baseline_iterations, baseline_time = baseline_results.get((N, method), (x, iterations, time_taken))
                        x = baseline_x
                        iterations = baseline_iterations
                        time_taken = baseline_time
                    else:
                        baseline = baseline_times.get((N, method), time_taken)
                    speedup = baseline / time_taken if time_taken > 0 and baseline > 0 else 1.0  # Corrected speedup calculation
                    results.append({
                        "N": N,
                        "Method": method,
                        "Mode": mode,
                        "x": x,
                        "Cost": cost,
                        "Iterations": iterations,
                        "Wall_Clock_Time_s": time_taken,
                        "Accuracy_Gap": accuracy_gap,
                        "Speedup": speedup
                    })
                    print(f"N={N}, Method={method}, Mode={mode if mode != 'cpu' else 'cpu sequential'}: "
                          f"x={x:.4f}, Cost={cost:.4f}, Iter={iterations:.1f}, "
                          f"Wall Clock Time={time_taken:.4f}s, Accuracy Gap={accuracy_gap:.4e}, "
                          f"Speedup={speedup:.2f}x")
    # Save results to CSV; Added for persistent data storage and analysis
    df = pd.DataFrame(results)
    df.to_csv("performance_data.csv", index=False)
    # Generate and display only the scaling visualization; Changed from plot_bar_charts to plot_scaling
    visualize.plot_scaling(df)  # Removed plot_performance call to focus on scaling visualization
if __name__ == "__main__":
    main()