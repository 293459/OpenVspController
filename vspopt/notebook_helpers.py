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
import plotly.express as px
import plotly.graph_objects as go

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


# ============================================================================
# Model parameter visualization helpers for Jupyter notebooks
# ============================================================================

def create_complete_parameters_table(model_wrapper) -> pd.DataFrame:
    """
    Create a complete table of all aircraft model parameters.

    Parameters
    ----------
    model_wrapper : VSPWrapper
        The loaded VSP wrapper containing the aircraft model

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: Component, Group, Parameter, Value
    """
    all_params_data = []
    
    for component in model_wrapper.geom_names:
        try:
            all_obj_params = model_wrapper.get_all_params(component)
            for param_path, value in all_obj_params.items():
                if "/" in param_path:
                    group, param = param_path.split("/", 1)
                else:
                    group, param = param_path, ""
                
                all_params_data.append({
                    "Component": component,
                    "Group": group,
                    "Parameter": param,
                    "Value": value,
                })
        except Exception as e:
            logger.warning(f"Could not read parameters from {component}: {e}")
    
    return pd.DataFrame(all_params_data)


def create_hierarchical_treemap(model_wrapper):
    """
    Create an interactive hierarchical treemap of aircraft parameters.

    Parameters
    ----------
    model_wrapper : VSPWrapper
        The loaded VSP wrapper containing the aircraft model

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive treemap visualization
    """
    treemap_data = []
    root_name = "Aircraft"
    
    for component in model_wrapper.geom_names:
        try:
            all_obj_params = model_wrapper.get_all_params(component)
            
            # Add parameter nodes
            for param_path, value in all_obj_params.items():
                if "/" in param_path:
                    group, param = param_path.split("/", 1)
                else:
                    group, param = param_path, ""
                
                treemap_data.append({
                    "name": f"{param}",
                    "parent": group,
                    "value": abs(float(value)) if isinstance(value, (int, float)) else 1,
                    "component": component,
                    "group": group,
                    "param": param,
                    "param_value": value,
                })
            
            # Add group nodes
            groups = set()
            for param_path in all_obj_params.keys():
                if "/" in param_path:
                    groups.add(param_path.split("/", 1)[0])
            
            for group in groups:
                treemap_data.append({
                    "name": f"{group}",
                    "parent": component,
                    "value": 1,
                    "component": component,
                    "group": group,
                    "param": None,
                    "param_value": None,
                })
            
            # Add component node
            treemap_data.append({
                "name": component,
                "parent": root_name,
                "value": 1,
                "component": component,
                "group": None,
                "param": None,
                "param_value": None,
            })
        except Exception as e:
            logger.warning(f"Could not create treemap nodes for {component}: {e}")
    
    # Add root
    treemap_data.append({
        "name": root_name,
        "parent": "",
        "value": 1,
        "component": None,
        "group": None,
        "param": None,
        "param_value": None,
    })
    
    df_treemap = pd.DataFrame(treemap_data)
    
    fig_tree = px.treemap(
        df_treemap,
        names="name",
        parents="parent",
        values="value",
        color="component",
        hover_data={"component": False, "param_value": False},
        title="🌳 Hierarchical Parameter Tree: Aircraft → Component → Group → Parameter",
    )
    
    fig_tree.update_layout(
        height=700,
        font=dict(size=10),
    )
    
    return fig_tree


def create_conceptual_mindmap(model_wrapper):
    """
    Create an interactive sunburst visualization organizing parameters by domain.

    Parameters
    ----------
    model_wrapper : VSPWrapper
        The loaded VSP wrapper containing the aircraft model

    Returns
    -------
    (fig, df_mindmap_params) : (plotly.graph_objects.Figure, pd.DataFrame)
        Tuple of (sunburst visualization, parameters DataFrame with concept mapping)
    """
    # Define domain categories
    concept_map = {
        "Aerodynamics": {
            "keywords": ["cl", "cd", "lift", "drag", "aero", "pressure", "mach", "reynolds"],
            "color": "#FF6B6B",
        },
        "Geometry - Surfaces": {
            "keywords": ["span", "chord", "area", "sweep", "dihedral", "twist", "aspect", "taper", "wing", "tail", "fuselage"],
            "color": "#4ECDC4",
        },
        "Geometry - Airfoils": {
            "keywords": ["airfoil", "xsec", "thickness", "camber", "leading", "naca"],
            "color": "#45B7D1",
        },
        "Structural": {
            "keywords": ["mass", "weight", "inertia", "cg", "xcg", "ycg", "zcg", "moment", "bending"],
            "color": "#FFA07A",
        },
        "Flight Controls": {
            "keywords": ["flap", "aileron", "elevator", "rudder", "control", "deflect", "hinge"],
            "color": "#98D8C8",
        },
        "Performance": {
            "keywords": ["speed", "altitude", "range", "endurance", "payload", "fuel", "burn"],
            "color": "#F7DC6F",
        },
    }
    
    # Categorize parameters
    all_params_flat = []
    
    for component in model_wrapper.geom_names:
        try:
            all_obj_params = model_wrapper.get_all_params(component)
            for param_path, value in all_obj_params.items():
                if "/" in param_path:
                    group, param = param_path.split("/", 1)
                else:
                    group, param = param_path, ""
                
                param_name_lower = param.lower()
                
                # Find best matching concept
                assigned_concept = "Other"
                for concept, config in concept_map.items():
                    if any(keyword in param_name_lower for keyword in config["keywords"]):
                        assigned_concept = concept
                        break
                
                all_params_flat.append({
                    "param": param,
                    "component": component,
                    "group": group,
                    "value": value,
                    "concept": assigned_concept,
                })
        except Exception as e:
            logger.warning(f"Could not categorize parameters from {component}: {e}")
    
    df_mindmap_params = pd.DataFrame(all_params_flat)
    
    # Prepare sunburst data
    sunburst_data = []
    root_name = "Aircraft Systems"
    
    # Root node
    sunburst_data.append({
        "name": root_name,
        "parent": "",
        "value": 1,
        "concept": "root",
    })
    
    # Add concept nodes and parameters
    for concept in list(concept_map.keys()) + ["Other"]:
        concept_params = df_mindmap_params[df_mindmap_params["concept"] == concept]
        
        if len(concept_params) > 0:
            sunburst_data.append({
                "name": f"<b>{concept}</b><br>({len(concept_params)} params)",
                "parent": root_name,
                "value": len(concept_params),
                "concept": concept,
            })
            
            # Add parameters grouped by component
            for component in concept_params["component"].unique():
                comp_params = concept_params[concept_params["component"] == component]
                parent_concept = f"<b>{concept}</b><br>({len(concept_params)} params)"
                
                for _, row in comp_params.iterrows():
                    value_str = f"{row['value']:.4f}" if isinstance(row['value'], (int, float)) else str(row['value'])
                    sunburst_data.append({
                        "name": f"{row['param']}: {value_str}",
                        "parent": f"{component}",
                        "value": 1,
                        "concept": concept,
                    })
                
                # Add component node if not exists
                comp_check = [d for d in sunburst_data if d["name"] == component and d.get("parent") == parent_concept]
                if not comp_check:
                    sunburst_data.append({
                        "name": component,
                        "parent": parent_concept,
                        "value": len(comp_params),
                        "concept": concept,
                    })
    
    df_sunburst = pd.DataFrame(sunburst_data)
    
    # Create color map for concepts
    color_map = {concept: config["color"] for concept, config in concept_map.items()}
    color_map["root"] = "#FFFFFF"
    color_map["Other"] = "#D3D3D3"
    
    colors = [color_map.get(row["concept"], "#CCCCCC") for _, row in df_sunburst.iterrows()]
    
    fig_concept = go.Figure(go.Sunburst(
        labels=df_sunburst["name"],
        parents=df_sunburst["parent"],
        values=df_sunburst["value"],
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))
    
    fig_concept.update_layout(
        title={
            "text": "🧠 Conceptual Mind Map: Aircraft Parameters organized by Domain",
            "font": {"size": 18, "family": "Arial Black"},
        },
        height=800,
        font=dict(size=11),
        margin=dict(t=100, l=0, r=0, b=0),
    )
    
    return fig_concept, df_mindmap_params


def print_conceptual_summary(df_mindmap_params: pd.DataFrame, concept_map: dict) -> None:
    """
    Print a tabular summary of parameters organized by domain.

    Parameters
    ----------
    df_mindmap_params : pd.DataFrame
        DataFrame with concept-mapped parameters (from create_conceptual_mindmap)
    concept_map : dict
        Domain concept definitions (from create_conceptual_mindmap)
    """
    print("\n📋 CONCEPTUAL SUMMARY BY DOMAIN\n" + "="*100)
    
    for concept in list(concept_map.keys()) + ["Other"]:
        concept_params = df_mindmap_params[df_mindmap_params["concept"] == concept]
        
        if len(concept_params) > 0:
            print(f"\n🔹 {concept.upper()} ({len(concept_params)} parameters)")
            print("-" * 100)
            
            # Sort by component
            for component in concept_params["component"].unique():
                comp_data = concept_params[concept_params["component"] == component]
                print(f"\n   {component}:")
                
                summary_table = comp_data[[
                    "param", "group", "value"
                ]].drop_duplicates().sort_values("param")
                
                for _, row in summary_table.iterrows():
                    value_str = f"{row['value']:.6f}" if isinstance(row['value'], (int, float)) else str(row['value'])
                    print(f"     • {row['param']:30s} [{row['group']:20s}] = {value_str}")
    
    print("\n" + "="*100)


def print_conceptual_summary_sample(df_mindmap_params: pd.DataFrame, concept_map: dict, samples_per_concept: int = 3) -> None:
    """
    Print a sampled summary of parameters organized by domain (showing examples only).

    This shows only a few representative parameters per domain, useful for quick overview
    without overwhelming the output with all parameters.

    Parameters
    ----------
    df_mindmap_params : pd.DataFrame
        DataFrame with concept-mapped parameters (from create_conceptual_mindmap)
    concept_map : dict
        Domain concept definitions (from create_conceptual_mindmap)
    samples_per_concept : int, default 3
        Number of example parameters to show per domain
    """
    print("\n🎯 CONCEPTUAL OVERVIEW - SAMPLE PARAMETERS BY DOMAIN\n" + "="*100)
    
    for concept in list(concept_map.keys()) + ["Other"]:
        concept_params = df_mindmap_params[df_mindmap_params["concept"] == concept]
        
        if len(concept_params) > 0:
            total_count = len(concept_params)
            print(f"\n🔹 {concept.upper()} ({total_count} parameters total)")
            print("-" * 100)
            
            # Group by component and show samples
            components_shown = 0
            for component in concept_params["component"].unique():
                if components_shown >= samples_per_concept:
                    break
                    
                comp_data = concept_params[concept_params["component"] == component]
                print(f"\n   {component}:")
                
                summary_table = comp_data[[
                    "param", "group", "value"
                ]].drop_duplicates().sort_values("param").head(3)
                
                for _, row in summary_table.iterrows():
                    value_str = f"{row['value']:.6f}" if isinstance(row['value'], (int, float)) else str(row['value'])
                    print(f"     • {row['param']:30s} [{row['group']:20s}] = {value_str}")
                
                # Show "..." if there are more parameters in this component
                remaining_in_comp = len(comp_data) - len(summary_table)
                if remaining_in_comp > 0:
                    print(f"     • ... and {remaining_in_comp} more parameter(s)")
                
                components_shown += 1
            
            # Show summary line if more components not shown
            remaining_components = len(concept_params["component"].unique()) - components_shown
            if remaining_components > 0:
                print(f"\n   ... and parameters from {remaining_components} more component(s)")
    
    print("\n" + "="*100 + "\n📌 For complete details, see:\n  • CSV: exports/all_parameters.csv\n  • TXT: exports/all_parameters.txt")
