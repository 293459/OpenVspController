"""
Helper functions for Jupyter notebooks: plot validation, data visualization, and diagnostics.

This module provides utilities for:
- Validating plot data before rendering
- Creating diagnostic summaries
- Handling empty or invalid aerodynamic results
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def validate_plot_data(
    x: np.ndarray,
    y: np.ndarray,
    name: str = "Plot data",
) -> tuple[bool, str]:
    """
    Validate that plot data is non-empty and contains finite values.

    Parameters
    ----------
    x, y : np.ndarray
        Data arrays to validate
    name : str
        Descriptive name for error messages

    Returns
    -------
    (is_valid, message) : (bool, str)
    """
    if len(x) == 0 or len(y) == 0:
        return False, f"{name}: Empty data array (len={len(x)}, len={len(y)})"

    if not np.any(np.isfinite(x)):
        return False, f"{name}: All X values are non-finite"

    if not np.any(np.isfinite(y)):
        return False, f"{name}: All Y values are non-finite"

    if len(x) != len(y):
        return False, f"{name}: Array length mismatch (x={len(x)}, y={len(y)})"

    return True, f"{name}: OK ({len(x)} points)"


def validate_vspaero_results(results) -> list[str]:
    """
    Validate VSPAERO results and return a list of diagnostic messages.

    Parameters
    ----------
    results : VSPAEROResults
        Results object from a VSPAERO sweep

    Returns
    -------
    messages : list[str]
        List of diagnostic messages (empty if all checks pass)
    """
    issues = []

    # Check array dimensions
    n_alpha = len(results.alpha)
    n_CL = len(results.CL)
    n_CD = len(results.CD)

    if n_alpha == 0:
        issues.append("No alpha points in results")
        return issues  # Can't proceed further

    if n_CL != n_alpha or n_CD != n_alpha:
        issues.append(f"Dimension mismatch: alpha={n_alpha}, CL={n_CL}, CD={n_CD}")
        return issues

    # Check for all-zero data (indicates failed/incomplete sweep)
    if np.allclose(results.CL, 0):
        issues.append("All CL values are zero (verify VSPAERO sweep completed successfully)")

    if np.allclose(results.CD, 0):
        issues.append("All CD values are zero (verify VSPAERO sweep completed successfully)")

    # Check for non-finite values
    non_finite_CL = (~np.isfinite(results.CL)).sum()
    non_finite_CD = (~np.isfinite(results.CD)).sum()

    if non_finite_CL > 0:
        issues.append(f"{non_finite_CL} non-finite CL values")

    if non_finite_CD > 0:
        issues.append(f"{non_finite_CD} non-finite CD values")

    # Check monotonicity in linear range
    if n_alpha >= 3:
        mask_linear = (results.alpha >= -5) & (results.alpha <= 10)
        if mask_linear.sum() >= 3:
            dCL = np.diff(results.CL[mask_linear])
            if np.any(dCL < -0.05):
                issues.append("CL is non-monotone in linear range (possible mesh/wake iteration issue)")

    return issues


def safe_plot_polar(
    results,
    title: str = "Aerodynamic Polar",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a polar plot (CL vs CD) with validation.

    Returns the figure and axes even if data is invalid, but shows diagnostic message.

    Parameters
    ----------
    results : VSPAEROResults
        Results from a VSPAERO sweep
    title : str
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to draw on. If None, creates new figure.

    Returns
    -------
    (fig, ax) : (plt.Figure, plt.Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    issues = validate_vspaero_results(results)

    if issues:
        # Plot is invalid; show diagnostic instead
        ax.text(
            0.5, 0.5, "\n".join(["Data validation issues:"] + issues),
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            transform=ax.transAxes
        )
        ax.set_title(f"{title} [INVALID DATA]", fontsize=12, color="red", weight="bold")
        ax.axis("off")
    else:
        # Data is valid; plot it
        valid_mask = np.isfinite(results.CL) & np.isfinite(results.CD)
        CD_valid = results.CD[valid_mask]
        CL_valid = results.CL[valid_mask]

        ax.plot(CD_valid, CL_valid, "o-", linewidth=2, markersize=6, label="Polar")
        ax.set_xlabel("Drag Coefficient (CD)", fontsize=11)
        ax.set_ylabel("Lift Coefficient (CL)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

    return fig, ax


def safe_plot_drag_polar(
    results,
    title: str = "Drag Polar",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a drag polar (CD vs CL^2) with validation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    issues = validate_vspaero_results(results)

    if issues:
        ax.text(
            0.5, 0.5, "\n".join(["Data validation issues:"] + issues),
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            transform=ax.transAxes
        )
        ax.set_title(f"{title} [INVALID DATA]", fontsize=12, color="red", weight="bold")
        ax.axis("off")
    else:
        valid_mask = np.isfinite(results.CL) & np.isfinite(results.CD)
        CL_squared = results.CL[valid_mask] ** 2
        CD_valid = results.CD[valid_mask]

        ax.plot(CL_squared, CD_valid, "s-", linewidth=2, markersize=6, label="Drag Polar")
        ax.set_xlabel("CL²", fontsize=11)
        ax.set_ylabel("Drag Coefficient (CD)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

    return fig, ax


def safe_plot_ld_ratio(
    results,
    title: str = "L/D vs Alpha",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create an L/D vs alpha plot with validation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    issues = validate_vspaero_results(results)

    if issues:
        ax.text(
            0.5, 0.5, "\n".join(["Data validation issues:"] + issues),
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            transform=ax.transAxes
        )
        ax.set_title(f"{title} [INVALID DATA]", fontsize=12, color="red", weight="bold")
        ax.axis("off")
    else:
        valid_mask = np.isfinite(results.LD)
        alpha_valid = results.alpha[valid_mask]
        LD_valid = results.LD[valid_mask]

        if len(LD_valid) > 0:
            ax.plot(alpha_valid, LD_valid, "d-", linewidth=2, markersize=6, label="L/D", color="darkgreen")
            ax.set_xlabel("Angle of Attack (deg)", fontsize=11)
            ax.set_ylabel("L/D Ratio", fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No valid L/D data",
                   ha="center", va="center", fontsize=10,
                   transform=ax.transAxes)
            ax.set_title(f"{title} [NO DATA]", fontsize=12, color="orange", weight="bold")
            ax.axis("off")

    return fig, ax


def print_results_diagnostic(results) -> None:
    """Print a comprehensive diagnostic of VSPAERO results to the console."""
    print("\n" + "="*70)
    print("VSPAERO RESULTS DIAGNOSTIC")
    print("="*70)

    # Sweep parameters
    print(f"\nSweep Parameters:")
    print(f"  Mach number       : {results.mach:.2f}")
    print(f"  Reynolds number   : {results.re_cref:.2e}")
    print(f"  Alpha range       : {results.alpha.min():.1f}° to {results.alpha.max():.1f}°")
    print(f"  Number of points  : {len(results.alpha)}")

    # Reference quantities
    print(f"\nReference Quantities:")
    print(f"  Sref (wing area)  : {results.Sref:.4f} m²")
    print(f"  bref (span)       : {results.bref:.4f} m")
    print(f"  cref (chord)      : {results.cref:.4f} m")

    # Data availability
    print(f"\nData Arrays:")
    has_CL = len(results.CL) > 0 and np.any(np.isfinite(results.CL))
    has_CD = len(results.CD) > 0 and np.any(np.isfinite(results.CD))
    has_LD = len(results.LD) > 0 and np.any(np.isfinite(results.LD))

    print(f"  CL values         : {'✓' if has_CL else '✗'} ({len(results.CL)} points, "
          f"{(~np.isfinite(results.CL)).sum()} non-finite)")
    print(f"  CD values         : {'✓' if has_CD else '✗'} ({len(results.CD)} points, "
          f"{(~np.isfinite(results.CD)).sum()} non-finite)")
    print(f"  L/D values        : {'✓' if has_LD else '✗'} ({len(results.LD)} points, "
          f"{(~np.isfinite(results.LD)).sum()} non-finite)")

    # Performance metrics
    if has_CL and has_CD:
        print(f"\nPerformance Metrics:")
        print(f"  CL range          : {np.nanmin(results.CL):.4f} to {np.nanmax(results.CL):.4f}")
        print(f"  CD range          : {np.nanmin(results.CD):.4f} to {np.nanmax(results.CD):.4f}")
        if has_LD:
            print(f"  L/D range         : {np.nanmin(results.LD):.4f} to {np.nanmax(results.LD):.4f}")
            print(f"  L/D max           : {results.LD_max:.4f}")

    # Issues
    issues = validate_vspaero_results(results)
    if issues:
        print(f"\n⚠ Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✓ All data validation checks passed.")

    print("="*70 + "\n")
