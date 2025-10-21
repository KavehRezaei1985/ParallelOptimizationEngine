# python/visualize.py
#
# High-quality visualization module for **ParallelOptimizationEngine** performance
# benchmarking. Generates a three-panel bar chart comparing execution modes across
# key metrics:
#   • **Runtime** (wall-clock time in seconds)
#   • **Convergence Speed** (number of iterations)
#   • **Solution Accuracy** (final global cost)
#
# The plot is automatically saved as `performance.png` (300 DPI, publication-ready)
# and displayed interactively.  Designed for clarity, reproducibility, and integration
# with `run_simulation.py` output.
#
# Features:
#   • **Responsive layout**: Rotated x-labels, tight spacing, shared figure size
#   • **Professional styling**: Clear titles, labeled axes, consistent formatting
#   - **Export quality**: High-resolution PNG with preserved bounding box
#
# Usage:
#   >>> from visualize import plot_bar_charts
#   >>> plot_bar_charts(modes, times, iterations, costs)
#
# All inputs must be same-length iterables (lists, tuples, or NumPy arrays).

import matplotlib.pyplot as plt

def plot_bar_charts(modes, times, iters, costs):
    """
    Generate a comparative bar chart of optimization performance across modes.

    Parameters
    ----------
    modes : iterable of str
        Labels for each execution mode (e.g., ['cpu', 'gpu', 'ml']).
    times : iterable of float
        Execution time in seconds for each mode.
    iters : iterable of float
        Number of optimization iterations (lower = faster convergence).
    costs : iterable of float
        Final global cost F(x*) (lower = higher accuracy).

    Output
    ------
    Saves `performance.png` in the current directory and displays the figure.
    """
    # Create a 1×3 subplot grid with professional figure dimensions
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # ----------------------------------------------------------------------
    # Panel 1: Runtime (seconds)
    # ----------------------------------------------------------------------
    axs[0].bar(modes, times, color='tab:blue')
    axs[0].set_title('Runtime (s)', fontsize=14, pad=15)
    axs[0].set_ylabel('Time (seconds)', fontsize=12)
    axs[0].tick_params(axis='x', rotation=45, labelsize=10)
    axs[0].grid(axis='y', linestyle='--', alpha=0.3)

    # ----------------------------------------------------------------------
    # Panel 2: Convergence Speed (iterations)
    # ----------------------------------------------------------------------
    axs[1].bar(modes, iters, color='tab:orange')
    axs[1].set_title('Iterations (Convergence Speed)', fontsize=14, pad=15)
    axs[1].set_ylabel('Iterations', fontsize=12)
    axs[1].tick_params(axis='x', rotation=45, labelsize=10)
    axs[1].grid(axis='y', linestyle='--', alpha=0.3)

    # ----------------------------------------------------------------------
    # Panel 3: Solution Accuracy (final cost)
    # ----------------------------------------------------------------------
    axs[2].bar(modes, costs, color='tab:green')
    axs[2].set_title('Final Cost (Accuracy)', fontsize=14, pad=15)
    axs[2].set_ylabel('Cost', fontsize=12)
    axs[2].tick_params(axis='x', rotation=45, labelsize=10)
    axs[2].grid(axis='y', linestyle='--', alpha=0.3)

    # ----------------------------------------------------------------------
    # Layout and export
    # ----------------------------------------------------------------------
    plt.tight_layout(pad=3.0)                    # Prevent subplot overlap
    plt.subplots_adjust(bottom=0.25)             # Extra space for rotated labels
    plt.savefig('performance.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')       # Ensure white background on save
    plt.show()