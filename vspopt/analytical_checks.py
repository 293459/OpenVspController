"""
First-order analytical checks for aerodynamic and stability outputs.

These functions are intentionally lightweight. They are meant to catch gross
OpenVSP setup issues and unexpected trends, not to replace VSPAERO or a full
DATCOM-style build-up.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0


@dataclass
class AnalyticalCheckResult:
    """Container for one analytical/OpenVSP comparison row."""

    parameter: str
    analytical: float
    openvsp: float
    delta_pct: float = float("nan")
    warning: str = ""


def lift_curve_slope_helmbold(
    aspect_ratio: float,
    *,
    sweep_quarter_chord_deg: float = 0.0,
    mach: float = 0.0,
    efficiency: float = 0.95,
    per_degree: bool = False,
) -> float:
    """
    Estimate finite-wing ``CL_alpha`` with a Helmbold/DATCOM-style formula.

    The default output is per radian, matching the derivatives in OpenVSP
    ``.stab`` files. Set ``per_degree=True`` only when comparing to alpha fits
    performed directly against degrees.
    """
    if aspect_ratio <= 0.0:
        return float("nan")

    beta = math.sqrt(max(1.0 - mach**2, 1e-6))
    sweep = math.radians(sweep_quarter_chord_deg)
    denominator = 2.0 + math.sqrt(
        4.0
        + (aspect_ratio * beta / max(efficiency, 1e-6)) ** 2
        * (1.0 + (math.tan(sweep) ** 2) / max(beta**2, 1e-6))
    )
    cl_alpha_rad = 2.0 * math.pi * aspect_ratio / denominator
    return float(cl_alpha_rad * DEG_TO_RAD if per_degree else cl_alpha_rad)


def downwash_gradient_datcom(
    wing_cl_alpha: float,
    wing_aspect_ratio: float,
    *,
    tail_efficiency: float = 1.0,
) -> float:
    """
    Estimate ``d epsilon / d alpha`` with a compact DATCOM-style relation.

    The formula is a far-field approximation, so the result is clipped to a
    physically useful range for notebook sanity checks.
    """
    if wing_aspect_ratio <= 0.0 or not np.isfinite(wing_cl_alpha):
        return float("nan")
    value = tail_efficiency * 2.0 * wing_cl_alpha / (math.pi * wing_aspect_ratio)
    return float(np.clip(value, 0.0, 0.9))


def total_lift_curve_slope(
    wing_cl_alpha: float,
    tail_cl_alpha: float,
    tail_area_ratio: float,
    downwash_gradient: float,
    *,
    tail_efficiency: float = 1.0,
) -> float:
    """Return a first-order aircraft ``CL_alpha`` in per-radian units."""
    tail_term = tail_efficiency * tail_cl_alpha * tail_area_ratio * (1.0 - downwash_gradient)
    return float(wing_cl_alpha + tail_term)


def neutral_point_location(
    wing_cl_alpha: float,
    wing_ac_x: float,
    tail_cl_alpha: float,
    tail_area_ratio: float,
    tail_ac_x: float,
    downwash_gradient: float,
    *,
    tail_efficiency: float = 1.0,
) -> float:
    """
    Estimate longitudinal neutral-point location for a wing-tail configuration.

    Positions are dimensional and must use the same reference frame.
    """
    tail_term = tail_efficiency * tail_cl_alpha * tail_area_ratio * (1.0 - downwash_gradient)
    denominator = wing_cl_alpha + tail_term
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    numerator = wing_cl_alpha * wing_ac_x + tail_term * tail_ac_x
    return float(numerator / denominator)


def static_margin(x_neutral_point: float, x_cg: float, mac: float) -> float:
    """Compute static margin ``Kn = (x_np - x_cg) / MAC``."""
    if not np.isfinite(mac) or mac <= 0.0:
        return float("nan")
    return float((x_neutral_point - x_cg) / mac)


def pitching_moment_slope(
    total_cl_alpha: float,
    x_cg: float,
    x_neutral_point: float,
    mac: float,
) -> float:
    """
    Estimate ``CM_alpha`` from neutral point and total lift-curve slope.

    With the usual stability convention, a stable aircraft has negative
    ``CM_alpha`` because ``x_np`` is aft of ``x_cg``.
    """
    if not np.isfinite(mac) or mac <= 0.0:
        return float("nan")
    return float(total_cl_alpha * (x_cg - x_neutral_point) / mac)


def run_basic_stability_checks(inputs: Mapping[str, float]) -> dict[str, float]:
    """Compute the analytical checks discussed in the notebook."""
    wing_cl_alpha = lift_curve_slope_helmbold(
        float(inputs.get("wing_aspect_ratio", float("nan"))),
        sweep_quarter_chord_deg=float(inputs.get("wing_sweep_quarter_chord_deg", 0.0)),
        mach=float(inputs.get("mach", 0.0)),
    )
    tail_cl_alpha = lift_curve_slope_helmbold(
        float(inputs.get("tail_aspect_ratio", float("nan"))),
        sweep_quarter_chord_deg=float(inputs.get("tail_sweep_quarter_chord_deg", 0.0)),
        mach=float(inputs.get("mach", 0.0)),
    )
    downwash = downwash_gradient_datcom(
        wing_cl_alpha,
        float(inputs.get("wing_aspect_ratio", float("nan"))),
        tail_efficiency=float(inputs.get("downwash_tail_efficiency", 1.0)),
    )
    tail_area_ratio = float(inputs.get("tail_area_ratio", float("nan")))
    tail_efficiency = float(inputs.get("tail_efficiency", 1.0))
    cl_alpha_total = total_lift_curve_slope(
        wing_cl_alpha,
        tail_cl_alpha,
        tail_area_ratio,
        downwash,
        tail_efficiency=tail_efficiency,
    )
    x_np = neutral_point_location(
        wing_cl_alpha,
        float(inputs.get("wing_ac_x", float("nan"))),
        tail_cl_alpha,
        tail_area_ratio,
        float(inputs.get("tail_ac_x", float("nan"))),
        downwash,
        tail_efficiency=tail_efficiency,
    )
    kn = static_margin(x_np, float(inputs.get("x_cg", float("nan"))), float(inputs.get("mac", float("nan"))))
    cm_alpha = pitching_moment_slope(
        cl_alpha_total,
        float(inputs.get("x_cg", float("nan"))),
        x_np,
        float(inputs.get("mac", float("nan"))),
    )
    return {
        "Static margin Kn [-]": kn,
        "Neutral point x_np [m]": x_np,
        "CM_alpha [1/rad]": cm_alpha,
        "CL_alpha [1/rad]": cl_alpha_total,
        "Downwash d_epsilon/d_alpha [-]": downwash,
    }


def compare_analytical_to_openvsp(
    analytical: Mapping[str, float],
    openvsp: Mapping[str, float],
    *,
    warning_threshold_pct: float = 15.0,
    emit_warnings: bool = True,
) -> pd.DataFrame:
    """
    Build ``| Parameter | Analytical | OpenVSP | Delta % |`` with soft warnings.

    Warnings are emitted and recorded in the table, but they never raise errors.
    """
    rows: list[AnalyticalCheckResult] = []
    for parameter, analytical_value in analytical.items():
        openvsp_value = float(openvsp.get(parameter, float("nan")))
        analytical_value = float(analytical_value)
        delta_pct = float("nan")
        warning = ""

        if not np.isfinite(openvsp_value):
            warning = "OpenVSP reference unavailable"
        elif not np.isfinite(analytical_value):
            warning = "Analytical value unavailable"
        elif abs(openvsp_value) > 1e-12:
            delta_pct = 100.0 * (analytical_value - openvsp_value) / abs(openvsp_value)
            if abs(delta_pct) > warning_threshold_pct:
                warning = f"|Delta| > {warning_threshold_pct:.0f}%"
        elif abs(analytical_value) > 1e-12:
            warning = "OpenVSP value is near zero"

        if warning and emit_warnings:
            warnings.warn(f"{parameter}: {warning}", RuntimeWarning, stacklevel=2)

        rows.append(
            AnalyticalCheckResult(
                parameter=parameter,
                analytical=analytical_value,
                openvsp=openvsp_value,
                delta_pct=delta_pct,
                warning=warning,
            )
        )

    return pd.DataFrame(
        [
            {
                "Parameter": row.parameter,
                "Analytical": row.analytical,
                "OpenVSP": row.openvsp,
                "Delta %": row.delta_pct,
                "Warning": row.warning,
            }
            for row in rows
        ]
    )
