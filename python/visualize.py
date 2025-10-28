# python/visualize.py
#
# Title: Visualization Utilities for ParallelOptimizationEngine Scaling Analysis
#
# Description:
# This module provides advanced visualization tools for the **ParallelOptimizationEngine**,
# supporting both interactive exploration and publication-quality static output.
# 
# Key Features:
# - **Interactive Plotly 2x2 Dashboard**: Real-time hover, zoom, and legend toggling.
#   Uses a single global legend due to Plotly's limitation on shared legends across subplots.
# - **Static Matplotlib Export**: High-resolution `scaling.png` with per-subplot legends
#   for clarity in reports and papers.
#
# Plots vs. Number of Agents (N):
#   (1) Iterations (linear y) — measures convergence speed
#   (2) Time (s) (log y) — highlights performance scaling
#   (3) Accuracy Gap (log y) — shows deviation from closed-form solution
#   (4) Speedup (×) (linear y) — normalized against CPU baseline
#
# All X-axes are log-scaled to accommodate wide N ranges (100 to 10,000+).
#
# Dependencies:
# - os: For file path handling
# - numpy: For numerical operations
# - pandas: For CSV reading and data manipulation
# - plotly: For interactive dashboard
# - matplotlib: For static export
# - seaborn: For enhanced Matplotlib styling
#
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # <-- for valid hex color palette; used to ensure consistent, visually distinct colors
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Column mapping (case-insensitive)
# ----------------------------
# Added to support flexible column names from run_simulation.py output.
# Allows backward compatibility and robustness if CSV headers change.
COLUMN_MAP = {
    'method': ['method', 'Method', 'optimization_method'],
    'mode': ['mode', 'Mode', 'execution_mode'],
    'N': ['N', 'n', 'num_agents'],
    'iterations': ['iterations', 'iter', 'Iter'],
    'time': ['time_taken', 'wall_time', 'Wall_Clock_Time_s', 'time', 'Time'],
    'accuracy_gap': ['accuracy_gap', 'error', 'Accuracy_Gap'],
    'speedup': ['speedup', 'speed_up', 'Speedup']
}

def _get_col(df: pd.DataFrame, key: str) -> str:
    """Return the actual column name in df for a logical key.
    
    Args:
        df (pd.DataFrame): Input performance data
        key (str): Logical column key (e.g., 'method')
    
    Returns:
        str: Actual column name in the DataFrame
    
    Raises:
        KeyError: If no matching column is found
    
    Notes:
    - Added to decouple visualization from exact CSV column names.
    - Enables scalability and robustness across different run_simulation.py versions.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in COLUMN_MAP[key]:
        cl = cand.lower()
        if cl in cols_lower:
            return cols_lower[cl]
    raise KeyError(f"Could not find a column for '{key}'. Available: {list(df.columns)}")

# ----------------------------
# Plotly (interactive dashboard)
# ----------------------------
def plotly_dashboard(df: pd.DataFrame,
                     method_col: str, mode_col: str, n_col: str,
                     it_col: str, tm_col: str, acc_col: str, spd_col: str):
    """
    Generate an interactive 2x2 scaling dashboard using Plotly.
    
    Args:
        df (pd.DataFrame): Performance data
        method_col, mode_col, n_col: Resolved column names
        it_col, tm_col, acc_col, spd_col: Metric column names
    
    Notes:
    - **Why Plotly?** Enables hover, zoom, and dynamic legend interaction.
    - **Single global legend**: Plotly does not support per-subplot shared legends,
      so we use `legendgroup` and `showlegend` to deduplicate entries.
    - **Color/dash/marker encoding**: Method = color/dash, Mode = marker for clarity.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Iterations vs. N", "Time (s) vs. N",
                        "Accuracy Gap vs. N", "Speedup (×) vs. N"),
        horizontal_spacing=0.12, vertical_spacing=0.12
    )
    
    # Define metric panels: (column, label, row, col, log_y, format, key)
    # Added to support scaling analysis with appropriate Y scaling
    metrics = [
        (it_col, "Iterations", 1, 1, False, "{:.0f}", "iterations"),
        (tm_col, "Time (s)", 1, 2, True, "{:.2e}", "time"),
        (acc_col, "Accuracy Gap", 2, 1, True, "{:.2e}", "accuracy"),
        (spd_col, "Speedup (×)", 2, 2, False, "{:.2f}", "speedup"),
    ]
    
    methods = df[method_col].unique()
    modes = df[mode_col].unique()
    
    # Color by method, dash by method, marker by mode
    # Using Plotly's qualitative palette for distinct, accessible colors
    palette = px.colors.qualitative.Plotly
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(methods)}
    dash_map = {m: ("solid" if i % 2 == 0 else "dash") for i, m in enumerate(methods)}
    marker_map = {
        "cpu": "circle", "threadpool": "square", "openmp": "x",
        "gpu": "star", "ml": "diamond"
    }
    
    # Track shown legend labels to avoid duplicates
    shown_labels = set()
    
    for metric_col, ylabel, r, c, ylog, fmt, metric_key in metrics:
        for method in methods:
            for mode in modes:
                sub = df[(df[method_col] == method) & (df[mode_col] == mode)]
                if sub.empty:
                    continue
                sub = sub.sort_values(n_col)
                legend_label = f"{method}_{mode}_{metric_key}"
                show_legend_here = legend_label not in shown_labels
                if show_legend_here:
                    shown_labels.add(legend_label)
                
                fig.add_trace(
                    go.Scatter(
                        x=sub[n_col], y=sub[metric_col], mode="lines+markers",
                        name=legend_label,
                        legendgroup=legend_label,
                        showlegend=show_legend_here,
                        line=dict(color=color_map[method], dash=dash_map[method]),
                        marker=dict(symbol=marker_map.get(mode, "circle"), size=8),
                        hovertemplate=(
                            f"<b>Method:</b> {method}<br>"
                            f"<b>Mode:</b> {mode}<br>"
                            f"<b>N:</b> %{{x}}<br>"
                            f"<b>{ylabel}:</b> %{{y:{fmt}}}"
                        ),
                    ),
                    row=r, col=c
                )
        
        # Apply log scaling to X and Y as needed
        fig.update_xaxes(type="log", title_text="Number of Agents (N)", row=r, col=c)
        if ylog:
            fig.update_yaxes(type="log", title_text=ylabel, row=r, col=c)
        else:
            fig.update_yaxes(title_text=ylabel, row=r, col=c)
    
    # Layout: increased size, global legend, unified hover
    fig.update_layout(
        height=900, width=1300,
        title_text="Scaling Plots for Parallel Optimization Engine",
        legend=dict(
            orientation="v",
            yanchor="top", y=0.98,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1
        ),
        margin=dict(l=60, r=220, t=60, b=60),
        hovermode="x unified"
    )
    fig.show()

# ----------------------------
# Matplotlib (static export with per-subplot legends)
# ----------------------------
def matplotlib_export(df: pd.DataFrame,
                      method_col: str, mode_col: str, n_col: str,
                      it_col: str, tm_col: str, acc_col: str, spd_col: str,
                      outfile: str = "scaling.png"):
    """
    Export a static 2x2 scaling plot with per-subplot legends using Matplotlib.
    
    Args:
        df (pd.DataFrame): Performance data
        method_col, mode_col, n_col: Resolved column names
        it_col, tm_col, acc_col, spd_col: Metric column names
        outfile (str): Output filename; defaults to "scaling.png"
    
    Notes:
    - **Why Matplotlib?** Required for publication-quality static images.
    - **Per-subplot legends**: Avoids clutter in static output vs. Plotly's global legend.
    - **Seaborn styling**: Enhances visual appeal with professional defaults.
    """
    sns.set_context("talk")
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_it, ax_tm = axs[0]
    ax_acc, ax_spd = axs[1]
    
    methods = df[method_col].unique()
    modes = df[mode_col].unique()
    
    # Consistent styling: color by method, linestyle by method, marker by mode
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3'])
    color_map = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(methods)}
    ls_map = {m: ('-' if i % 2 == 0 else '--') for i, m in enumerate(methods)}
    marker_map = {"cpu": "o", "threadpool": "P", "openmp": "X", "gpu": "*", "ml": "s"}
    
    # Helper function to plot one panel
    def plot_panel(ax, ycol, ylab, ylog=False):
        """Plot a single scaling panel with legend."""
        handles = []
        labels = []
        for method in methods:
            for mode in modes:
                sub = df[(df[method_col] == method) & (df[mode_col] == mode)]
                if sub.empty:
                    continue
                sub = sub.sort_values(n_col)
                (ln,) = ax.plot(
                    sub[n_col], sub[ycol],
                    label=f"{method} — {mode}",
                    color=color_map[method],
                    linestyle=ls_map[method],
                    marker=marker_map.get(mode, 'o'),
                    markersize=6,
                    linewidth=1.8
                )
                handles.append(ln); labels.append(f"{method} — {mode}")
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        ax.set_xlabel("Number of Agents (N)")
        ax.set_ylabel(ylab)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        # Per-subplot legend to the right
        leg = ax.legend(
            handles, labels, loc="upper left",
            bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
            fontsize=9, frameon=True
        )
        leg.set_title("Method — Mode", prop={'size': 9})
    
    # Configure each panel
    ax_it.set_title("Iterations vs. N")
    plot_panel(ax_it, it_col, "Iterations", ylog=False)
    ax_tm.set_title("Time (s) vs. N")
    plot_panel(ax_tm, tm_col, "Time (s)", ylog=True)
    ax_acc.set_title("Accuracy Gap vs. N")
    plot_panel(ax_acc, acc_col, "Accuracy Gap", ylog=True)
    ax_spd.set_title("Speedup (×) vs. N")
    plot_panel(ax_spd, spd_col, "Speedup (×)", ylog=False)
    
    # Adjust layout to accommodate legends
    fig.set_constrained_layout_pads(w_pad=2.5/72, h_pad=2.0/72, wspace=0.25, hspace=0.25)
    save_path = os.path.join(os.getcwd(), outfile)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved static figure: {save_path}")

# ----------------------------
# Orchestrator
# ----------------------------
def plot_scaling(df: pd.DataFrame):
    """
    Main entry point: Generate both static and interactive visualizations.
    
    Args:
        df (pd.DataFrame): Performance data from run_simulation.py
    
    Notes:
    - **Why two outputs?** Interactive for exploration, static for reports.
    - **Column resolution**: Uses _get_col() for robustness.
    - **Numeric conversion**: Ensures plotting compatibility.
    """
    # Resolve actual column names from CSV
    method_col = _get_col(df, 'method')
    mode_col = _get_col(df, 'mode')
    n_col = _get_col(df, 'N')
    it_col = _get_col(df, 'iterations')
    tm_col = _get_col(df, 'time')
    acc_col = _get_col(df, 'accuracy_gap')
    spd_col = _get_col(df, 'speedup')
    
    # Ensure numeric types for plotting
    for c in [n_col, it_col, tm_col, acc_col, spd_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Export static figure first (for reports)
    matplotlib_export(df, method_col, mode_col, n_col, it_col, tm_col, acc_col, spd_col, outfile="scaling.png")
    
    # Then show interactive dashboard
    plotly_dashboard(df, method_col, mode_col, n_col, it_col, tm_col, acc_col, spd_col)

# ----------------------------
# Main execution block
# ----------------------------
if __name__ == "__main__":
    """
    Allow standalone execution for quick visualization testing.
    
    Notes:
    - Loads performance_data.csv generated by run_simulation.py.
    - Exits gracefully with clear error if file is missing.
    """
    try:
        data = pd.read_csv("performance_data.csv")
    except FileNotFoundError:
        print("Error: performance_data.csv not found. Please run run_simulation.py first.")
        raise SystemExit(1)
    plot_scaling(data)