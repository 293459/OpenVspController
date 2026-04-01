"""
vspopt/optimization.py
----------------------
Aerodynamic shape optimization using three complementary strategies:

  1. Gradient-based  (scipy SLSQP)       — fast, local, requires smooth landscape
  2. Bayesian        (Optuna / TPE)       — global, sample-efficient, noise-robust
  3. Two-phase       (Bayesian → SLSQP)  — recommended: global search + local polish

Mathematical background
-----------------------
The optimization problem is:

    min   f(x)        objective (e.g. –L/D or +CD)
    s.t.  g_i(x) ≤ 0  inequality constraints (e.g. CL ≥ CL_min)
          lb ≤ x ≤ ub  box bounds on design variables

where x ∈ ℝⁿ is the vector of design variables (span, sweep, etc.)
and f(x) is evaluated by running a full VSPAERO sweep.

Because f(x) is a black-box numerical function (not available in
closed form), finite-difference gradients are used for SLSQP.  The
Bayesian optimizer fits a Gaussian Process surrogate that models the
expensive f(x) with far fewer evaluations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Design-variable specification
# ---------------------------------------------------------------------------

@dataclass
class DesignVariable:
    """
    A single design variable fed to the optimizer.

    Parameters
    ----------
    label      : Human-readable name (e.g. "Wing span").
    geom_name  : OpenVSP geometry component name.
    parm_name  : OpenVSP parameter name.
    group_name : OpenVSP parameter group name.
    lower      : Lower bound.
    upper      : Upper bound.
    initial    : Starting value (used for gradient-based warm-start).
    units      : Physical unit string for display (e.g. "m", "deg").
    """
    label:      str
    geom_name:  str
    parm_name:  str
    group_name: str
    lower:      float
    upper:      float
    initial:    float
    units:      str = ""

    def __post_init__(self):
        if not (self.lower <= self.initial <= self.upper):
            raise ValueError(
                f"DesignVariable '{self.label}': initial={self.initial} is outside "
                f"bounds [{self.lower}, {self.upper}]."
            )
        if self.lower >= self.upper:
            raise ValueError(
                f"DesignVariable '{self.label}': lower={self.lower} must be < upper={self.upper}."
            )

    @property
    def vsp_key(self) -> tuple[str, str, str]:
        """Return the (geom, parm, group) tuple used by VSPWrapper.set_param()."""
        return (self.geom_name, self.parm_name, self.group_name)

    def __repr__(self) -> str:
        return (
            f"DesignVariable('{self.label}', [{self.lower}, {self.upper}] {self.units}, "
            f"x0={self.initial})"
        )


# ---------------------------------------------------------------------------
# Objective-function builder
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveSpec:
    """
    Specification of the scalar objective function.

    The optimizer minimizes ``objective``, so for maximization problems
    (e.g. L/D) use a negative weight:
        ObjectiveSpec([("LD_max", -1.0)])  → maximise L/D

    Parameters
    ----------
    metrics : list of (metric_name, weight) pairs.
        Supported metric names:
          "LD_max"    — maximum L/D in the alpha sweep
          "CD_at_LD"  — CD at maximum L/D
          "CL_at_LD"  — CL at maximum L/D
          "CDmin"     — minimum CD in the sweep
          "CL_alpha"  — lift-curve slope [1/deg]
    constraints : list of (metric, relation, threshold) triples.
        Example: [("CL_at_LD", ">=", 0.8), ("LD_max", ">=", 15.0)]
    """
    metrics:     list[tuple[str, float]] = field(
        default_factory=lambda: [("LD_max", -1.0)]
    )
    constraints: list[tuple[str, str, float]] = field(default_factory=list)
    cl_target:   Optional[float] = None  # Evaluate at specific CL if set
    alpha_target:Optional[float] = None  # Evaluate at specific alpha if set

    def compute(self, results: "VSPAEROResults") -> tuple[float, dict]:
        """
        Evaluate the scalar objective and constraint violations.

        Returns
        -------
        objective : float
        details   : dict of raw metric values (for logging)
        """
        from vspopt.vspaero import VSPAEROResults  # noqa: F401

        metrics_map = {
            "LD_max":    results.LD_max,
            "CD_at_LD":  results.CD_at_LD_max,
            "CL_at_LD":  results.CL_at_LD_max,
            "CDmin":     results.CD0_estimate,
            "CL_alpha":  results.CL_alpha,
        }

        if self.cl_target is not None:
            try:
                interp = results.interpolate_at_CL(self.cl_target)
                metrics_map["CD_at_CL"] = interp["CD"]
                metrics_map["LD_at_CL"] = interp["LD"]
            except ValueError as exc:
                logger.warning("CL interpolation failed: %s", exc)

        if self.alpha_target is not None:
            idx = np.argmin(np.abs(results.alpha - self.alpha_target))
            metrics_map["CL_at_alpha"] = float(results.CL[idx])
            metrics_map["CD_at_alpha"] = float(results.CD[idx])
            metrics_map["LD_at_alpha"] = float(results.LD[idx])

        obj = sum(
            w * metrics_map.get(m, 0.0) for m, w in self.metrics
        )

        # Constraint violations added as penalty
        penalty = 0.0
        for (m, rel, thresh) in self.constraints:
            val = metrics_map.get(m, float("nan"))
            if not np.isfinite(val):
                penalty += 1e6
                continue
            if rel == ">=" and val < thresh:
                penalty += 1000.0 * (thresh - val)
            elif rel == "<=" and val > thresh:
                penalty += 1000.0 * (val - thresh)

        return float(obj + penalty), metrics_map


# ---------------------------------------------------------------------------
# Optimization result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Stores the outcome of an optimization run."""
    method:          str
    best_x:          np.ndarray
    best_objective:  float
    best_metrics:    dict
    history_x:       list[np.ndarray] = field(default_factory=list)
    history_obj:     list[float]       = field(default_factory=list)
    n_evals:         int               = 0
    elapsed_sec:     float             = 0.0
    converged:       bool              = False
    message:         str               = ""
    design_variables: list[DesignVariable] = field(default_factory=list)

    def best_params_dict(self) -> dict[tuple, float]:
        """Return best-found parameters as a {(geom, parm, group): value} dict."""
        return {dv.vsp_key: float(self.best_x[i]) for i, dv in enumerate(self.design_variables)}

    def summary_series(self) -> "pd.Series":
        import pandas as pd
        row = {
            "method":      self.method,
            "n_evals":     self.n_evals,
            "best_obj":    round(self.best_objective, 6),
            "elapsed [s]": round(self.elapsed_sec, 1),
            "converged":   self.converged,
        }
        row.update({k: round(v, 4) for k, v in self.best_metrics.items()})
        for dv, xv in zip(self.design_variables, self.best_x):
            row[f"x_{dv.label}"] = round(float(xv), 4)
        return pd.Series(row)

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(method='{self.method}', "
            f"best_obj={self.best_objective:.4f}, "
            f"n_evals={self.n_evals}, "
            f"elapsed={self.elapsed_sec:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Shared callback evaluator
# ---------------------------------------------------------------------------

class _Evaluator:
    """
    Wraps the VSPWrapper + ObjectiveSpec into a single callable f(x) → float
    used by all optimizers.
    """

    def __init__(
        self,
        wrapper: "VSPWrapper",
        design_vars: list[DesignVariable],
        objective: ObjectiveSpec,
        sweep_kwargs: dict,
    ):
        self.wrapper      = wrapper
        self.design_vars  = design_vars
        self.objective    = objective
        self.sweep_kwargs = sweep_kwargs
        self.history_x:   list[np.ndarray] = []
        self.history_obj: list[float]       = []
        self.n_evals = 0

    def __call__(self, x: np.ndarray) -> float:
        """
        Set design variables x, run VSPAERO, return scalar objective.
        Infinite/NaN results are replaced with a large penalty.
        """
        param_dict = {
            dv.vsp_key: float(xi)
            for dv, xi in zip(self.design_vars, x)
        }
        try:
            results = self.wrapper.update_and_run(param_dict, self.sweep_kwargs)
            obj, details = self.objective.compute(results)
        except Exception as exc:
            logger.warning("Evaluation failed at x=%s: %s", x, exc)
            obj = 1e9
            details = {}

        if not np.isfinite(obj):
            obj = 1e9

        self.history_x.append(x.copy())
        self.history_obj.append(obj)
        self.n_evals += 1

        logger.debug("[eval #%d] obj=%.4f  x=%s", self.n_evals, obj, x)
        return float(obj)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(dv.lower, dv.upper) for dv in self.design_vars]

    @property
    def x0(self) -> np.ndarray:
        return np.array([dv.initial for dv in self.design_vars])


# ---------------------------------------------------------------------------
# 1. Gradient-based optimizer (scipy SLSQP)
# ---------------------------------------------------------------------------

def run_gradient_optimization(
    wrapper: "VSPWrapper",
    design_vars: list[DesignVariable],
    objective: ObjectiveSpec,
    sweep_kwargs: dict | None = None,
    *,
    method: str = "SLSQP",
    options: dict | None = None,
    finite_diff_step: float = 1e-3,
) -> OptimizationResult:
    """
    Gradient-based optimization via scipy.optimize.minimize.

    Gradients are approximated by forward finite differences:
        ∂f/∂xᵢ ≈ [f(x + ε·eᵢ) - f(x)] / ε

    This requires n+1 evaluations per gradient step (n = number of
    design variables), making it best suited for small design spaces
    (n ≤ 10) where the objective is relatively smooth.

    Parameters
    ----------
    method        : "SLSQP" (supports bounds + constraints, recommended)
                    or "L-BFGS-B" (bounds only, often faster).
    finite_diff_step : Step size ε for finite-difference gradients.
                       Too small → numerical noise; too large → inaccurate.
                       Typical range: 1e-4 to 1e-2 times the variable range.
    """
    from scipy.optimize import minimize, Bounds

    sweep_kwargs = sweep_kwargs or {}
    options = options or {"ftol": 1e-6, "maxiter": 100, "disp": True}

    evaluator = _Evaluator(wrapper, design_vars, objective, sweep_kwargs)
    bounds = Bounds(
        lb=[dv.lower for dv in design_vars],
        ub=[dv.upper for dv in design_vars],
    )

    logger.info(
        "Starting gradient-based optimization (%s), %d design variables.",
        method, len(design_vars),
    )
    t0 = time.perf_counter()

    result = minimize(
        fun=evaluator,
        x0=evaluator.x0,
        method=method,
        bounds=bounds,
        options={**options, "eps": finite_diff_step},
    )

    elapsed = time.perf_counter() - t0

    # Re-evaluate best point to get full metrics
    obj, metrics = objective.compute(
        wrapper.update_and_run(
            {dv.vsp_key: float(xi) for dv, xi in zip(design_vars, result.x)},
            sweep_kwargs,
        )
    )

    opt_result = OptimizationResult(
        method=f"Gradient-based ({method})",
        best_x=result.x,
        best_objective=float(result.fun),
        best_metrics=metrics,
        history_x=evaluator.history_x,
        history_obj=evaluator.history_obj,
        n_evals=evaluator.n_evals,
        elapsed_sec=elapsed,
        converged=bool(result.success),
        message=result.message,
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
    sweep_kwargs: dict | None = None,
    *,
    n_trials: int = 60,
    n_startup_trials: int = 10,
    seed: int = 42,
    study_name: str = "vspopt_bayesian",
    show_progress: bool = True,
) -> OptimizationResult:
    """
    Bayesian optimization via Optuna's TPE (Tree-structured Parzen Estimator).

    Why Bayesian?
    -------------
    VSPAERO runs are expensive (seconds each).  Bayesian optimization
    builds a probabilistic surrogate model of the objective function and
    uses an acquisition function (Expected Improvement) to decide WHERE
    to sample next, trading off exploitation (sampling near the current
    best) and exploration (sampling uncertain regions).

    TPE is a non-parametric Bayesian method that models P(x | good)
    and P(x | bad) separately and maximizes their ratio.  It is more
    robust to discontinuous or noisy objectives than GP-based methods
    and handles mixed (continuous + categorical) variables.

    Parameters
    ----------
    n_trials         : Total number of VSPAERO evaluations.
    n_startup_trials : Random samples before TPE takes over.
    seed             : Random seed for reproducibility.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("optuna is required.  Run: pip install optuna") from exc

    sweep_kwargs = sweep_kwargs or {}
    evaluator    = _Evaluator(wrapper, design_vars, objective, sweep_kwargs)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
        seed=seed,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )

    def _trial_objective(trial: "optuna.Trial") -> float:
        x = np.array([
            trial.suggest_float(dv.label, dv.lower, dv.upper)
            for dv in design_vars
        ])
        return evaluator(x)

    logger.info(
        "Starting Bayesian optimization (%d trials, %d startup).",
        n_trials, n_startup_trials,
    )
    t0 = time.perf_counter()

    study.optimize(
        _trial_objective,
        n_trials=n_trials,
        show_progress_bar=show_progress,
    )

    elapsed = time.perf_counter() - t0
    best_params = study.best_params
    best_x = np.array([best_params[dv.label] for dv in design_vars])

    # Re-evaluate best point to get full metrics
    obj, metrics = objective.compute(
        wrapper.update_and_run(
            {dv.vsp_key: float(xi) for dv, xi in zip(design_vars, best_x)},
            sweep_kwargs,
        )
    )

    opt_result = OptimizationResult(
        method="Bayesian (Optuna TPE)",
        best_x=best_x,
        best_objective=study.best_value,
        best_metrics=metrics,
        history_x=evaluator.history_x,
        history_obj=evaluator.history_obj,
        n_evals=evaluator.n_evals,
        elapsed_sec=elapsed,
        converged=True,  # Bayesian always "completes" its budget
        message=f"Best found after {n_trials} trials.",
        design_variables=design_vars,
    )
    opt_result._study = study  # Attach for further Optuna analysis
    logger.info("Bayesian optimization finished: %s", opt_result)
    return opt_result


# ---------------------------------------------------------------------------
# 3. Two-phase optimizer (Bayesian warm-start → SLSQP polish)
# ---------------------------------------------------------------------------

def run_two_phase_optimization(
    wrapper: "VSPWrapper",
    design_vars: list[DesignVariable],
    objective: ObjectiveSpec,
    sweep_kwargs: dict | None = None,
    *,
    n_bayesian_trials: int = 50,
    slsqp_options: dict | None = None,
    seed: int = 42,
) -> tuple[OptimizationResult, OptimizationResult]:
    """
    Two-phase optimization:

    Phase 1 — Bayesian (global exploration)
        Runs ``n_bayesian_trials`` VSPAERO evaluations using TPE to
        locate the region of the global optimum.

    Phase 2 — SLSQP (local refinement)
        Starting from the best point found by Bayesian, runs SLSQP
        with gradient information to precisely converge to the local
        optimum.

    Returns
    -------
    (bayesian_result, slsqp_result) : tuple of two OptimizationResult objects
    """
    sweep_kwargs = sweep_kwargs or {}

    logger.info("=== Two-phase optimization: Phase 1 (Bayesian) ===")
    bayesian_result = run_bayesian_optimization(
        wrapper, design_vars, objective, sweep_kwargs,
        n_trials=n_bayesian_trials, seed=seed,
    )

    # Warm-start Phase 2 from Bayesian best
    warm_start_vars = [
        DesignVariable(
            label=dv.label, geom_name=dv.geom_name,
            parm_name=dv.parm_name, group_name=dv.group_name,
            lower=dv.lower, upper=dv.upper,
            initial=float(bayesian_result.best_x[i]),
            units=dv.units,
        )
        for i, dv in enumerate(design_vars)
    ]

    logger.info("=== Two-phase optimization: Phase 2 (SLSQP polish) ===")
    slsqp_result = run_gradient_optimization(
        wrapper, warm_start_vars, objective, sweep_kwargs,
        method="SLSQP", options=slsqp_options,
    )

    return bayesian_result, slsqp_result


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_results(results: list[OptimizationResult]) -> "pd.DataFrame":
    """
    Build a comparison DataFrame from a list of OptimizationResult objects.
    One row per result, columns = method, n_evals, best_obj, elapsed, metrics.
    """
    import pandas as pd
    rows = [r.summary_series() for r in results]
    return pd.DataFrame(rows).set_index("method")
