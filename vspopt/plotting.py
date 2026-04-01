"""
vspopt/plotting.py
------------------
All plotting functions for aerodynamic analysis and optimization results.

Every function:
  - Returns a matplotlib Figure (so it can be embedded in Jupyter).
  - Also accepts an optional ``export_dir`` argument to save the figure
    as a .png in the exports/ folder.
  - Uses a consistent house style (dark axes, clean grid).

Available plots
---------------
  plot_polar             — CL, CD, CM vs alpha on three sub-panels
  plot_drag_polar        — CL vs CD (drag polar) with L/D contours
  plot_ld_ratio          — L/D vs alpha, marks maximum
  plot_span_loading      — Spanwise lift distribution
  plot_performance_map   — L/D, CL, CD heatmap over alpha × Mach
  plot_optimization_history — Objective convergence over evaluations
  plot_variable_sensitivity — Sensitivity of objective to each design var
  plot_comparison_bar    — Bar chart comparing optimizer results
  plot_pareto_front      — 2-D Pareto front for multi-objective results
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------

EXPORT_DPI = 200
STYLE_PARAMS = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F8F8",
    "axes.edgecolor":    "#CCCCCC",
    "axes.linewidth":    0.8,
    "grid.color":        "#DDDDDD",
    "grid.linewidth":    0.6,
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "text.color":        "#222222",
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "legend.framealpha": 0.9,
}

# Colour palette (accessible, colour-blind friendly)
BLUE   = "#1F77B4"
ORANGE = "#FF7F0E"
GREEN  = "#2CA02C"
RED    = "#D62728"
PURPLE = "#9467BD"

def _apply_style():
    matplotlib.rcParams.update(STYLE_PARAMS)

def _export(fig: plt.Figure, name: str, export_dir: Optional[str | Path]):
    if export_dir is not None:
        p = Path(export_dir)
        p.mkdir(parents=True, exist_ok=True)
        filepath = p / f"{name}.png"
        fig.savefig(filepath, dpi=EXPORT_DPI, bbox_inches="tight")
        logger.info("Figure saved: %s", filepath)
    return fig


# ---------------------------------------------------------------------------
# 1. Aerodynamic polar (CL, CD, CM vs alpha)
# ---------------------------------------------------------------------------

def plot_polar(
    results,
    title: str = "Aerodynamic Polar",
    export_dir: Optional[str | Path] = None,
    label: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel plot: CL vs α, CD vs α, CM vs α.

    Parameters
    ----------
    results    : VSPAEROResults or list of VSPAEROResults
                 (multiple results are overlaid with different colours).
    title      : Figure title.
    export_dir : If given, saves as '{title}.png' in that directory.
    label      : Legend label when a single result is passed.
    """
    _apply_style()
    from vspopt.vspaero import VSPAEROResults

    if isinstance(results, VSPAEROResults):
        results_list = [results]
        labels = [label or f"M={results.mach:.2f}"]
    else:
        results_list = list(results)
        labels = [label or f"M={r.mach:.2f}" for r in results_list]

    colours = [BLUE, ORANGE, GREEN, RED, PURPLE]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(title, fontsize=13, y=1.01)

    panels = [
        ("CL", "CL [-]", "Lift coefficient"),
        ("CD", "CD [-]", "Total drag coefficient"),
        ("CM", "CM [-]",  "Pitching moment (nose-up +)"),
    ]

    for col, (key, ylabel, panel_title) in enumerate(panels):
        ax = axes[col]
        for i, (res, lbl) in enumerate(zip(results_list, labels)):
            arr = getattr(res, key)
            if len(arr) == 0:
                continue
            ax.plot(res.alpha, arr, "-o", color=colours[i % 5],
                    markersize=3, linewidth=1.5, label=lbl)

        # Mark α=0 and reference line for CM
        ax.axhline(0, color="#999999", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="#999999", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Angle of attack α [deg]")
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, linewidth=0.5)
        if len(results_list) > 1 or label:
            ax.legend()

    fig.tight_layout()
    return _export(fig, title.replace(" ", "_"), export_dir)


# ---------------------------------------------------------------------------
# 2. Drag polar (CL vs CD) with L/D contour lines
# ---------------------------------------------------------------------------

def plot_drag_polar(
    results,
    title: str = "Drag Polar",
    export_dir: Optional[str | Path] = None,
) -> plt.Figure:
    """
    CL vs CD drag polar with constant-L/D lines and the L/D_max point marked.
    """
    _apply_style()
    from vspopt.vspaero import VSPAEROResults

    if isinstance(results, VSPAEROResults):
        results_list = [results]
    else:
        results_list = list(results)

    colours = [BLUE, ORANGE, GREEN, RED, PURPLE]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(title, fontsize=13)

    for i, res in enumerate(results_list):
        if len(res.CL) == 0:
            continue
        c = colours[i % 5]
        lbl = f"M={res.mach:.2f}, Re={res.re_cref:.1e}"
        ax.plot(res.CD, res.CL, "-o", color=c, markersize=3,
                linewidth=1.5, label=lbl)

        # Mark L/D_max
        if np.isfinite(res.LD_max):
            ax.plot(res.CD_at_LD_max, res.CL_at_LD_max, "*",
                    color=c, markersize=12, zorder=5,
                    label=f"L/D_max={res.LD_max:.1f} @ α={res.alpha_at_LD_max:.1f}°")

    # Draw constant-L/D lines
    all_cd = np.concatenate([r.CD for r in results_list if len(r.CD) > 0])
    all_cl = np.concatenate([r.CL for r in results_list if len(r.CL) > 0])
    if len(all_cd) > 0:
        cd_range = np.linspace(all_cd.min() * 0.8, all_cd.max() * 1.1, 200)
        ld_max_all = max(r.LD_max for r in results_list if np.isfinite(r.LD_max))
        for ld_val in np.linspace(5, ld_max_all * 0.95, 5):
            ax.plot(cd_range, ld_val * cd_range, "--",
                    color="#BBBBBB", linewidth=0.7, zorder=0)
            # Label the line near the right edge
            x_label = cd_range[-1] * 0.9
            ax.text(x_label, ld_val * x_label, f"L/D={ld_val:.0f}",
                    fontsize=8, color="#999999", va="bottom")

    ax.set_xlabel("CD (total drag coefficient) [-]")
    ax.set_ylabel("CL (lift coefficient) [-]")
    ax.grid(True, linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return _export(fig, "drag_polar", export_dir)


# ---------------------------------------------------------------------------
# 3. L/D ratio vs alpha
# ---------------------------------------------------------------------------

def plot_ld_ratio(
    results,
    title: str = "Lift-to-Drag Ratio",
    export_dir: Optional[str | Path] = None,
) -> plt.Figure:
    """L/D vs angle of attack.  Marks maximum L/D and operating CL."""
    _apply_style()
    from vspopt.vspaero import VSPAEROResults

    if isinstance(results, VSPAEROResults):
        results_list = [results]
    else:
        results_list = list(results)

    colours = [BLUE, ORANGE, GREEN, RED, PURPLE]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title(title, fontsize=13)

    for i, res in enumerate(results_list):
        if len(res.LD) == 0:
            continue
        c = colours[i % 5]
        lbl = f"M={res.mach:.2f}"
        ax.plot(res.alpha, res.LD, "-", color=c, linewidth=2, label=lbl)

        # Mark maximum
        if np.isfinite(res.LD_max):
            ax.annotate(
                f" L/D_max = {res.LD_max:.2f}\n α = {res.alpha_at_LD_max:.1f}°",
                xy=(res.alpha_at_LD_max, res.LD_max),
                xytext=(res.alpha_at_LD_max + 1, res.LD_max * 0.97),
                fontsize=9, color=c,
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1),
            )

    ax.axhline(0, color="#999999", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Angle of attack α [deg]")
    ax.set_ylabel("L/D [-]")
    ax.grid(True, linewidth=0.5)
    if len(results_list) > 1:
        ax.legend()
    fig.tight_layout()
    return _export(fig, "ld_ratio", export_dir)


# ---------------------------------------------------------------------------
# 4. Performance map (L/D heatmap over alpha × Mach)
# ---------------------------------------------------------------------------

def plot_performance_map(
    results_matrix: list[list],
    alpha_values: list[float],
    mach_values:  list[float],
    metric:       str = "LD_max",
    title:        str = "Performance Map",
    export_dir:   Optional[str | Path] = None,
) -> plt.Figure:
    """
    Heatmap of a scalar metric over a 2-D grid of (Mach, alpha).

    Parameters
    ----------
    results_matrix : 2-D list[list[VSPAEROResults]], shape (n_mach, n_alpha)
    alpha_values   : list of alpha values along the x-axis
    mach_values    : list of Mach values along the y-axis
    metric         : Scalar metric to plot.  Options: "LD_max", "CL_alpha",
                     "CD0_estimate", "oswald_mean"
    """
    _apply_style()
    data = np.zeros((len(mach_values), len(alpha_values)))
    for i, row in enumerate(results_matrix):
        for j, res in enumerate(row):
            data[i, j] = getattr(res, metric, float("nan"))

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(
        alpha_values, mach_values, data,
        cmap="RdYlGn", shading="auto",
    )
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel("Angle of attack α [deg]")
    ax.set_ylabel("Mach number [-]")
    ax.set_title(f"{title} — {metric}", fontsize=13)
    fig.tight_layout()
    return _export(fig, f"performance_map_{metric}", export_dir)


# ---------------------------------------------------------------------------
# 5. Optimization history
# ---------------------------------------------------------------------------

def plot_optimization_history(
    opt_results,
    title: str = "Optimization Convergence",
    export_dir: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot objective value vs. number of function evaluations.
    Accepts a single OptimizationResult or a list (for comparison).
    """
    _apply_style()
    from vspopt.optimization import OptimizationResult

    if isinstance(opt_results, OptimizationResult):
        opt_list = [opt_results]
    else:
        opt_list = list(opt_results)

    colours = [BLUE, ORANGE, GREEN, RED, PURPLE]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title(title, fontsize=13)

    for i, opt in enumerate(opt_list):
        c = colours[i % 5]
        objs = np.array(opt.history_obj)
        # Running minimum (best-so-far curve)
        best_so_far = np.minimum.accumulate(objs)
        evals = np.arange(1, len(objs) + 1)
        ax.plot(evals, objs, ".", color=c, alpha=0.4, markersize=4)
        ax.plot(evals, best_so_far, "-", color=c, linewidth=2,
                label=f"{opt.method} (best={opt.best_objective:.4f})")

    ax.set_xlabel("Number of VSPAERO evaluations")
    ax.set_ylabel("Objective value")
    ax.grid(True, linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return _export(fig, "optimization_history", export_dir)


# ---------------------------------------------------------------------------
# 6. Design variable sensitivity
# ---------------------------------------------------------------------------

def plot_variable_sensitivity(
    opt_result: "OptimizationResult",
    title: str = "Design Variable Sensitivity",
    export_dir: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Bar chart showing how much the objective changed relative to
    the initial point for each design variable, estimated from the
    optimization history.  Helps identify which variables matter most.
    """
    _apply_style()
    dv_names = [dv.label for dv in opt_result.design_variables]
    n_vars   = len(dv_names)

    if len(opt_result.history_x) < 2:
        logger.warning("Insufficient history for sensitivity plot.")
        return plt.figure()

    x_arr  = np.array(opt_result.history_x)   # shape (n_evals, n_vars)
    obj_arr = np.array(opt_result.history_obj)

    # Pearson correlation between each variable column and the objective
    sensitivities = np.array([
        float(np.corrcoef(x_arr[:, j], obj_arr)[0, 1])
        for j in range(n_vars)
    ])
    order = np.argsort(np.abs(sensitivities))[::-1]

    fig, ax = plt.subplots(figsize=(max(6, n_vars * 0.8 + 2), 4.5))
    colours_ = [RED if s > 0 else BLUE for s in sensitivities[order]]
    ax.barh(
        [dv_names[o] for o in order],
        [sensitivities[o] for o in order],
        color=colours_, edgecolor="white", linewidth=0.5,
    )
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Correlation with objective (|r| → importance)")
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", linewidth=0.4)
    fig.tight_layout()
    return _export(fig, "variable_sensitivity", export_dir)


# ---------------------------------------------------------------------------
# 7. Optimizer comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    opt_results: list,
    metric: str = "best_objective",
    title: str = "Optimizer Comparison",
    export_dir: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Side-by-side bars comparing key metrics across optimizer runs.
    """
    _apply_style()
    import pandas as pd

    df = pd.DataFrame(
        [r.summary_series() for r in opt_results]
    ).set_index("method")

    if metric not in df.columns:
        logger.warning("Metric '%s' not found.  Available: %s", metric, list(df.columns))
        return plt.figure()

    fig, ax = plt.subplots(figsize=(max(6, len(opt_results) * 1.5 + 2), 4.5))
    colours_ = [BLUE, ORANGE, GREEN, RED, PURPLE]
    bars = ax.bar(
        df.index, df[metric].astype(float),
        color=colours_[:len(df)], edgecolor="white", linewidth=0.5,
    )
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=13)
    plt.xticks(rotation=15, ha="right")
    ax.grid(axis="y", linewidth=0.4)
    fig.tight_layout()
    return _export(fig, "comparison_bar", export_dir)


# ---------------------------------------------------------------------------
# 8. Multi-variable sweep grid (one line per sweep value)
# ---------------------------------------------------------------------------

def plot_sweep_grid(
    results_dict: dict[float, "VSPAEROResults"],
    x_key: str = "alpha",
    y_key: str = "CL",
    sweep_label: str = "parameter",
    title: str = "Parameter Sweep",
    export_dir: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot one y_key vs x_key curve per sweep value.

    Parameters
    ----------
    results_dict : {param_value: VSPAEROResults}
    x_key        : Key of the x-axis variable ('alpha', etc.)
    y_key        : Key of the y-axis variable ('CL', 'CD', 'LD', 'CM', ...)
    sweep_label  : Name of the swept parameter (for legend).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title, fontsize=13)

    cmap = matplotlib.colormaps["viridis"]
    vals = sorted(results_dict.keys())
    n    = len(vals)

    for i, val in enumerate(vals):
        res = results_dict[val]
        x_arr = getattr(res, x_key, res.alpha)
        y_arr = getattr(res, y_key)
        if len(y_arr) == 0:
            continue
        color = cmap(i / max(n - 1, 1))
        ax.plot(x_arr, y_arr, "-", color=color, linewidth=1.5,
                label=f"{sweep_label}={val:.3g}")

    ax.axhline(0, color="#AAAAAA", linewidth=0.6, linestyle="--")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(True, linewidth=0.5)
    ax.legend(fontsize=9, ncol=max(1, n // 8))
    fig.tight_layout()
    return _export(fig, f"sweep_{y_key}_vs_{x_key}", export_dir)


# ---------------------------------------------------------------------------
# Export all figures at once
# ---------------------------------------------------------------------------

def export_all(results, opt_results_list: list, export_dir: str | Path) -> None:
    """
    Convenience wrapper: run all plot functions and save to export_dir.

    Parameters
    ----------
    results          : VSPAEROResults from a single baseline sweep.
    opt_results_list : List of OptimizationResult objects.
    export_dir       : Output directory for .png files.
    """
    logger.info("Exporting all figures to: %s", export_dir)
    plot_polar(results, title="Aerodynamic Polar", export_dir=export_dir)
    plot_drag_polar(results, title="Drag Polar", export_dir=export_dir)
    plot_ld_ratio(results, title="L/D Ratio", export_dir=export_dir)

    if opt_results_list:
        plot_optimization_history(opt_results_list,
                                   title="Optimization Convergence",
                                   export_dir=export_dir)
        plot_comparison_bar(opt_results_list, export_dir=export_dir)
        for opt in opt_results_list:
            plot_variable_sensitivity(
                opt,
                title=f"Sensitivity — {opt.method}",
                export_dir=export_dir,
            )
    plt.close("all")
    logger.info("All figures exported.")
