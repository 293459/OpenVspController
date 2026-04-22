"""
Aerodynamic and stability-oriented optimization utilities.

The original optimization layer assumed every design variable was a plain
geometry parameter. That was too narrow for the current workflow, because the
analysis now needs to vary:
  - OpenVSP geometry parameters such as ``Vtail/XForm/X_Location``;
  - VSPAERO analysis inputs such as ``Xcg``;
  - control-surface group deflections when needed.

This module keeps one optimizer interface while letting each design variable
describe how it should be applied before a VSPAERO sweep.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ANALYSIS_INPUT_TO_SWEEP_KWARG = {
    # OpenVSP analysis-manager names use CamelCase, while run_vspaero_sweep
    # accepts explicit Python keyword arguments. Keep the mapping here so the
    # notebook can validate against the true OpenVSP input name and still call
    # the wrapper correctly.
    "Xcg": "xcg",
    "Ycg": "ycg",
    "Zcg": "zcg",
}


# ---------------------------------------------------------------------------
# Design-variable specification
# ---------------------------------------------------------------------------


@dataclass
class DesignVariable:
    """
    A single design variable fed to the optimizer.

    Parameters
    ----------
    label : str
        Human-readable name shown in tables and plots.
    geom_name / parm_name / group_name : str | None
        Target geometry parameter when ``kind='geometry'``.
    analysis_input : str | None
        Target VSPAERO analysis input when ``kind='analysis_input'``.
    control_surface_group : str | int | None
        Target control-surface group when ``kind='control_surface_group'``.
    lower / upper / initial : float
        Bounds and starting value.
    """

    label: str
    geom_name: str | None = None
    parm_name: str | None = None
    group_name: str | None = None
    lower: float = 0.0
    upper: float = 1.0
    initial: float = 0.0
    units: str = ""
    kind: str = "geometry"
    analysis_input: str | None = None
    control_surface_group: str | int | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not (self.lower <= self.initial <= self.upper):
            raise ValueError(
                f"DesignVariable '{self.label}': initial={self.initial} is outside bounds "
                f"[{self.lower}, {self.upper}]."
            )
        if self.lower >= self.upper:
            raise ValueError(
                f"DesignVariable '{self.label}': lower={self.lower} must be < upper={self.upper}."
            )

        supported_kinds = {"geometry", "analysis_input", "control_surface_group"}
        if self.kind not in supported_kinds:
            raise ValueError(
                f"DesignVariable '{self.label}': kind must be one of {sorted(supported_kinds)}."
            )

        if self.kind == "geometry":
            missing = [name for name, value in {
                "geom_name": self.geom_name,
                "parm_name": self.parm_name,
                "group_name": self.group_name,
            }.items() if not value]
            if missing:
                raise ValueError(
                    f"DesignVariable '{self.label}': geometry variables require "
                    + ", ".join(missing) + "."
                )

        if self.kind == "analysis_input" and not self.analysis_input:
            raise ValueError(
                f"DesignVariable '{self.label}': analysis_input variables require analysis_input."
            )

        if self.kind == "control_surface_group" and self.control_surface_group is None:
            raise ValueError(
                f"DesignVariable '{self.label}': control_surface_group variables require control_surface_group."
            )

    @classmethod
    def geometry(
        cls,
        *,
        label: str,
        geom_name: str,
        parm_name: str,
        group_name: str,
        lower: float,
        upper: float,
        initial: float,
        units: str = "",
        description: str = "",
    ) -> "DesignVariable":
        return cls(
            label=label,
            geom_name=geom_name,
            parm_name=parm_name,
            group_name=group_name,
            lower=lower,
            upper=upper,
            initial=initial,
            units=units,
            kind="geometry",
            description=description,
        )

    @classmethod
    def analysis_input_variable(
        cls,
        *,
        label: str,
        analysis_input: str,
        lower: float,
        upper: float,
        initial: float,
        units: str = "",
        description: str = "",
    ) -> "DesignVariable":
        return cls(
            label=label,
            lower=lower,
            upper=upper,
            initial=initial,
            units=units,
            kind="analysis_input",
            analysis_input=analysis_input,
            description=description,
        )

    @classmethod
    def control_surface_group_variable(
        cls,
        *,
        label: str,
        control_surface_group: str | int,
        lower: float,
        upper: float,
        initial: float,
        units: str = "deg",
        description: str = "",
    ) -> "DesignVariable":
        return cls(
            label=label,
            lower=lower,
            upper=upper,
            initial=initial,
            units=units,
            kind="control_surface_group",
            control_surface_group=control_surface_group,
            description=description,
        )

    @property
    def vsp_key(self) -> tuple[str, str, str] | None:
        """Return the geometry tuple used by :class:`vspopt.wrapper.VSPWrapper`."""
        if self.kind != "geometry":
            return None
        return (self.geom_name or "", self.parm_name or "", self.group_name or "")

    @property
    def target_description(self) -> str:
        """Return a readable description of what this variable modifies."""
        if self.kind == "geometry":
            return f"{self.geom_name}/{self.group_name}/{self.parm_name}"
        if self.kind == "analysis_input":
            return f"VSPAEROSweep/{self.analysis_input}"
        return f"ControlSurfaceGroup/{self.control_surface_group}"

    def apply(
        self,
        wrapper: "VSPWrapper",
        value: float,
        sweep_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply this variable and return the updated sweep keyword arguments.

        Geometry variables update the in-memory OpenVSP model directly.
        Analysis-input and control-surface variables update the sweep arguments.
        """
        updated_kwargs = dict(sweep_kwargs)
        scalar_value = float(value)

        if self.kind == "geometry":
            wrapper.set_param(
                self.geom_name or "",
                self.parm_name or "",
                self.group_name or "",
                scalar_value,
            )
        elif self.kind == "analysis_input":
            input_name = self.analysis_input or ""
            target_key = _ANALYSIS_INPUT_TO_SWEEP_KWARG.get(input_name, input_name)
            updated_kwargs[target_key] = scalar_value
        elif self.kind == "control_surface_group":
            deflections = dict(updated_kwargs.get("control_surface_deflections", {}))
            deflections[self.control_surface_group] = scalar_value
            updated_kwargs["control_surface_deflections"] = deflections
        return updated_kwargs

    def __repr__(self) -> str:
        return (
            f"DesignVariable(label='{self.label}', kind='{self.kind}', target='{self.target_description}', "
            f"bounds=[{self.lower}, {self.upper}], initial={self.initial})"
        )


def validate_design_variables(
    wrapper: "VSPWrapper",
    design_vars: list[DesignVariable],
) -> "pd.DataFrame":
    """
    Validate design-variable targets against the currently loaded model.

    The returned DataFrame is useful in the notebook because it makes the
    resolved names visible before an optimization starts.
    """
    import pandas as pd

    rows: list[dict[str, object]] = []
    available_inputs = wrapper.get_available_analysis_inputs("VSPAEROSweep")
    control_groups = wrapper.get_control_surface_groups()

    for design_var in design_vars:
        status = "ok"
        detail = design_var.target_description

        try:
            if design_var.kind == "geometry":
                current_value = wrapper.get_param(
                    design_var.geom_name or "",
                    design_var.parm_name or "",
                    design_var.group_name or "",
                )
                detail = f"{detail} = {current_value:.6g}"
            elif design_var.kind == "analysis_input":
                if (design_var.analysis_input or "") not in available_inputs:
                    raise ValueError("analysis input not available in VSPAEROSweep")
            else:
                wrapper._resolve_control_surface_group_index(design_var.control_surface_group)  # type: ignore[arg-type]
        except Exception as exc:
            status = "error"
            detail = str(exc)

        rows.append(
            {
                "label": design_var.label,
                "kind": design_var.kind,
                "target": design_var.target_description,
                "initial": design_var.initial,
                "lower": design_var.lower,
                "upper": design_var.upper,
                "units": design_var.units,
                "status": status,
                "detail": detail,
            }
        )

    if control_groups:
        rows.append(
            {
                "label": "available_control_groups",
                "kind": "info",
                "target": ", ".join(group["name"] for group in control_groups),
                "initial": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "units": "",
                "status": "info",
                "detail": "Detected control-surface groups in the loaded model.",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Objective-function builder
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveSpec:
    """
    Specification of the scalar objective function.

    The optimizer minimizes this objective, so negative weights correspond to
    maximization targets.
    """

    metrics: list[tuple[str, float]] = field(default_factory=lambda: [("LD_max", -1.0)])
    constraints: list[tuple[str, str, float]] = field(default_factory=list)
    cl_target: Optional[float] = None
    alpha_target: Optional[float] = None

    def compute(self, results: "VSPAEROResults") -> tuple[float, dict[str, float]]:
        from vspopt.vspaero import VSPAEROResults  # noqa: F401

        metrics_map: dict[str, float] = {
            "LD_max": results.LD_max,
            "CD_at_LD": results.CD_at_LD_max,
            "CL_at_LD": results.CL_at_LD_max,
            "CDmin": results.CD0_estimate,
            "CL_alpha": results.CL_alpha,
            "static_margin": results.static_margin,
            "StaticMargin": results.static_margin,
            "neutral_point_x": results.neutral_point_x,
            "NeutralPointX": results.neutral_point_x,
        }

        if self.cl_target is not None:
            try:
                interp = results.interpolate_at_CL(self.cl_target)
                metrics_map["CD_at_CL"] = interp["CD"]
                metrics_map["LD_at_CL"] = interp["LD"]
            except ValueError as exc:
                logger.warning("CL interpolation failed: %s", exc)

        if self.alpha_target is not None and len(results.alpha) > 0:
            idx = int(np.argmin(np.abs(results.alpha - self.alpha_target)))
            metrics_map["CL_at_alpha"] = float(results.CL[idx])
            metrics_map["CD_at_alpha"] = float(results.CD[idx])
            metrics_map["LD_at_alpha"] = float(results.LD[idx])

        objective = sum(weight * metrics_map.get(metric_name, 0.0) for metric_name, weight in self.metrics)

        penalty = 0.0
        for metric_name, relation, threshold in self.constraints:
            value = metrics_map.get(metric_name, float("nan"))
            if not np.isfinite(value):
                penalty += 1e6
                continue
            if relation == ">=" and value < threshold:
                penalty += 1000.0 * (threshold - value)
            elif relation == "<=" and value > threshold:
                penalty += 1000.0 * (value - threshold)

        return float(objective + penalty), metrics_map


# ---------------------------------------------------------------------------
# Optimization result container
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Stores the outcome of an optimization run."""

    method: str
    best_x: np.ndarray
    best_objective: float
    best_metrics: dict[str, float]
    history_x: list[np.ndarray] = field(default_factory=list)
    history_obj: list[float] = field(default_factory=list)
    n_evals: int = 0
    elapsed_sec: float = 0.0
    converged: bool = False
    message: str = ""
    design_variables: list[DesignVariable] = field(default_factory=list)

    def best_params_dict(self) -> dict[tuple[str, str, str], float]:
        """Return only the best geometry parameters as a VSP tuple dictionary."""
        params: dict[tuple[str, str, str], float] = {}
        for index, design_var in enumerate(self.design_variables):
            if design_var.kind == "geometry" and design_var.vsp_key is not None:
                params[design_var.vsp_key] = float(self.best_x[index])
        return params

    def best_variable_values(self) -> dict[str, float]:
        """Return best-found values keyed by the variable label."""
        return {design_var.label: float(self.best_x[index]) for index, design_var in enumerate(self.design_variables)}

    def apply_best(
        self,
        wrapper: "VSPWrapper",
        sweep_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Apply the best design to the wrapper and return the updated sweep kwargs.

        This is the easiest way for the notebook to re-run the final design while
        preserving any baseline analysis settings.
        """
        updated_kwargs = dict(sweep_kwargs or {})
        for index, design_var in enumerate(self.design_variables):
            updated_kwargs = design_var.apply(wrapper, float(self.best_x[index]), updated_kwargs)
        wrapper._vsp.Update()
        return updated_kwargs

    def summary_series(self) -> "pd.Series":
        import pandas as pd

        row = {
            "method": self.method,
            "n_evals": self.n_evals,
            "best_obj": round(self.best_objective, 6),
            "elapsed [s]": round(self.elapsed_sec, 1),
            "converged": self.converged,
        }
        row.update({key: round(value, 4) for key, value in self.best_metrics.items() if np.isfinite(value)})
        for design_var, value in zip(self.design_variables, self.best_x):
            row[f"x_{design_var.label}"] = round(float(value), 4)
        return pd.Series(row)

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(method='{self.method}', best_obj={self.best_objective:.4f}, "
            f"n_evals={self.n_evals}, elapsed={self.elapsed_sec:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Shared callback evaluator
# ---------------------------------------------------------------------------


class _Evaluator:
    """Wrap the wrapper and objective into a shared callable ``f(x) -> float``."""

    def __init__(
        self,
        wrapper: "VSPWrapper",
        design_vars: list[DesignVariable],
        objective: ObjectiveSpec,
        sweep_kwargs: dict[str, Any],
    ) -> None:
        self.wrapper = wrapper
        self.design_vars = design_vars
        self.objective = objective
        self.sweep_kwargs = dict(sweep_kwargs)
        self.history_x: list[np.ndarray] = []
        self.history_obj: list[float] = []
        self.n_evals = 0

    def evaluate_results(self, x: np.ndarray) -> "VSPAEROResults":
        run_kwargs = dict(self.sweep_kwargs)
        for design_var, x_value in zip(self.design_vars, x):
            run_kwargs = design_var.apply(self.wrapper, float(x_value), run_kwargs)
        self.wrapper._vsp.Update()
        return self.wrapper.run_vspaero_sweep(**run_kwargs)

    def __call__(self, x: np.ndarray) -> float:
        try:
            results = self.evaluate_results(x)
            objective_value, _ = self.objective.compute(results)
        except Exception as exc:
            logger.warning("Evaluation failed at x=%s: %s", x, exc)
            objective_value = 1e9

        if not np.isfinite(objective_value):
            objective_value = 1e9

        self.history_x.append(np.array(x, dtype=float).copy())
        self.history_obj.append(float(objective_value))
        self.n_evals += 1
        logger.debug("[eval #%d] obj=%.6f  x=%s", self.n_evals, objective_value, x)
        return float(objective_value)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(design_var.lower, design_var.upper) for design_var in self.design_vars]

    @property
    def x0(self) -> np.ndarray:
        return np.array([design_var.initial for design_var in self.design_vars], dtype=float)


# ---------------------------------------------------------------------------
# 1. Gradient-based optimizer (scipy SLSQP)
# ---------------------------------------------------------------------------


def run_gradient_optimization(
    wrapper: "VSPWrapper",
    design_vars: list[DesignVariable],
    objective: ObjectiveSpec,
    sweep_kwargs: dict[str, Any] | None = None,
    *,
    method: str = "SLSQP",
    options: dict[str, Any] | None = None,
    finite_diff_step: float = 1e-3,
) -> OptimizationResult:
    """Gradient-based optimization via :func:`scipy.optimize.minimize`."""
    from scipy.optimize import Bounds, minimize

    sweep_kwargs = dict(sweep_kwargs or {})
    options = options or {"ftol": 1e-6, "maxiter": 100, "disp": True}
    evaluator = _Evaluator(wrapper, design_vars, objective, sweep_kwargs)

    bounds = Bounds(
        lb=[design_var.lower for design_var in design_vars],
        ub=[design_var.upper for design_var in design_vars],
    )

    logger.info(
        "Starting gradient-based optimization (%s) with %d design variables.",
        method,
        len(design_vars),
    )
    start = time.perf_counter()

    result = minimize(
        fun=evaluator,
        x0=evaluator.x0,
        method=method,
        bounds=bounds,
        options={**options, "eps": finite_diff_step},
    )

    elapsed = time.perf_counter() - start
    best_results = evaluator.evaluate_results(np.array(result.x, dtype=float))
    _, metrics = objective.compute(best_results)

    opt_result = OptimizationResult(
        method=f"Gradient-based ({method})",
        best_x=np.array(result.x, dtype=float),
        best_objective=float(result.fun),
        best_metrics=metrics,
        history_x=evaluator.history_x,
        history_obj=evaluator.history_obj,
        n_evals=evaluator.n_evals,
        elapsed_sec=elapsed,
        converged=bool(result.success),
        message=str(result.message),
        design_variables=design_vars,
    )
    logger.info("Gradient optimization finished: %s", opt_result)
    return opt_result


# ---------------------------------------------------------------------------
# 2. Bayesian optimizer (Optuna TPE)
# ---------------------------------------------------------------------------


def run_bayesian_optimization(
    wrapper: "VSPWrapper",
    design_vars: list[DesignVariable],
    objective: ObjectiveSpec,
    sweep_kwargs: dict[str, Any] | None = None,
    *,
    n_trials: int = 60,
    n_startup_trials: int = 10,
    seed: int = 42,
    study_name: str = "vspopt_bayesian",
    show_progress: bool = True,
) -> OptimizationResult:
    """Bayesian optimization via Optuna's TPE sampler."""
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("optuna is required. Install it in the project environment first.") from exc

    sweep_kwargs = dict(sweep_kwargs or {})
    evaluator = _Evaluator(wrapper, design_vars, objective, sweep_kwargs)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
        seed=seed,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )

    def trial_objective(trial: "optuna.Trial") -> float:
        x = np.array(
            [trial.suggest_float(design_var.label, design_var.lower, design_var.upper) for design_var in design_vars],
            dtype=float,
        )
        return evaluator(x)

    logger.info(
        "Starting Bayesian optimization (%d trials, %d startup).",
        n_trials,
        n_startup_trials,
    )
    start = time.perf_counter()
    study.optimize(trial_objective, n_trials=n_trials, show_progress_bar=show_progress)
    elapsed = time.perf_counter() - start

    best_params = study.best_params
    best_x = np.array([best_params[design_var.label] for design_var in design_vars], dtype=float)
    best_results = evaluator.evaluate_results(best_x)
    _, metrics = objective.compute(best_results)

    opt_result = OptimizationResult(
        method="Bayesian (Optuna TPE)",
        best_x=best_x,
        best_objective=float(study.best_value),
        best_metrics=metrics,
        history_x=evaluator.history_x,
        history_obj=evaluator.history_obj,
        n_evals=evaluator.n_evals,
        elapsed_sec=elapsed,
        converged=True,
        message=f"Best found after {n_trials} trials.",
        design_variables=design_vars,
    )
    opt_result._study = study  # type: ignore[attr-defined]
    logger.info("Bayesian optimization finished: %s", opt_result)
    return opt_result


# ---------------------------------------------------------------------------
# 3. Two-phase optimizer (Bayesian warm-start -> SLSQP polish)
# ---------------------------------------------------------------------------


def run_two_phase_optimization(
    wrapper: "VSPWrapper",
    design_vars: list[DesignVariable],
    objective: ObjectiveSpec,
    sweep_kwargs: dict[str, Any] | None = None,
    *,
    n_bayesian_trials: int = 50,
    slsqp_options: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[OptimizationResult, OptimizationResult]:
    """Two-phase optimization: Bayesian exploration followed by SLSQP polish."""
    sweep_kwargs = dict(sweep_kwargs or {})

    logger.info("=== Two-phase optimization: Phase 1 (Bayesian) ===")
    bayesian_result = run_bayesian_optimization(
        wrapper,
        design_vars,
        objective,
        sweep_kwargs=sweep_kwargs,
        n_trials=n_bayesian_trials,
        seed=seed,
    )

    warm_start_vars = [
        DesignVariable(
            label=design_var.label,
            geom_name=design_var.geom_name,
            parm_name=design_var.parm_name,
            group_name=design_var.group_name,
            lower=design_var.lower,
            upper=design_var.upper,
            initial=float(bayesian_result.best_x[index]),
            units=design_var.units,
            kind=design_var.kind,
            analysis_input=design_var.analysis_input,
            control_surface_group=design_var.control_surface_group,
            description=design_var.description,
        )
        for index, design_var in enumerate(design_vars)
    ]

    logger.info("=== Two-phase optimization: Phase 2 (SLSQP polish) ===")
    slsqp_result = run_gradient_optimization(
        wrapper,
        warm_start_vars,
        objective,
        sweep_kwargs=sweep_kwargs,
        method="SLSQP",
        options=slsqp_options,
    )

    return bayesian_result, slsqp_result


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------


def compare_results(results: list[OptimizationResult]) -> "pd.DataFrame":
    """Build a comparison DataFrame from a list of :class:`OptimizationResult` objects."""
    import pandas as pd

    rows = [result.summary_series() for result in results]
    return pd.DataFrame(rows).set_index("method")
