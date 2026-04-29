"""
VSPAERO results parsing and the :class:`VSPAEROResults` dataclass.

The solver writes aerodynamic data into OpenVSP's Results Manager and, when
requested, also generates text artifacts such as ``.history`` and ``.stab``.
This module keeps the primary aerodynamic arrays, stability metadata, and
per-case artifact locations in one place for the notebook and optimization code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vspopt.postprocess import CD0Extraction, extract_cd0_from_arrays, first_finite_value

logger = logging.getLogger(__name__)


@dataclass
class VSPAEROResults:
    """Container for a single VSPAERO sweep run."""

    # Sweep conditions
    mach: float = 0.0
    re_cref: float = 0.0
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))

    # Aerodynamic coefficients
    CL: np.ndarray = field(default_factory=lambda: np.array([]))
    CD: np.ndarray = field(default_factory=lambda: np.array([]))
    CDi: np.ndarray = field(default_factory=lambda: np.array([]))
    CDo: np.ndarray = field(default_factory=lambda: np.array([]))
    CDsff: np.ndarray = field(default_factory=lambda: np.array([]))
    CM: np.ndarray = field(default_factory=lambda: np.array([]))
    CMx: np.ndarray = field(default_factory=lambda: np.array([]))
    CMz: np.ndarray = field(default_factory=lambda: np.array([]))
    CS: np.ndarray = field(default_factory=lambda: np.array([]))
    LD: np.ndarray = field(default_factory=lambda: np.array([]))
    E: np.ndarray = field(default_factory=lambda: np.array([]))

    # Force coefficients (body frame)
    CFx: np.ndarray = field(default_factory=lambda: np.array([]))
    CFy: np.ndarray = field(default_factory=lambda: np.array([]))
    CFz: np.ndarray = field(default_factory=lambda: np.array([]))

    # Reference quantities
    Sref: float = 0.0
    bref: float = 0.0
    cref: float = 0.0

    # Metadata
    n_points: int = 0
    case_name: str = ""
    working_dir: Optional[Path] = None
    model_path: Optional[Path] = None
    solver_log_path: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)
    polar_path: Optional[Path] = None
    history_path: Optional[Path] = None
    history_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    convergence: dict[str, object] = field(default_factory=dict)
    stab_path: Optional[Path] = None
    stability_records: list[dict[str, float]] = field(default_factory=list)
    stability_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    mass_properties: Optional["MassProperties"] = None
    cd0: float = float("nan")
    cd0_method: str = ""
    cd0_source: str = ""
    cd0_induced_factor: float = float("nan")
    cd0_fit_r_squared: float = float("nan")
    cd0_direct_at_alpha0: float = float("nan")
    cd0_cdo_at_alpha0: float = float("nan")
    cd0_n_points: int = 0

    @property
    def has_stability_data(self) -> bool:
        """Return ``True`` when a non-empty stability table is available."""
        return not self.stability_dataframe().empty

    @property
    def CL_alpha(self) -> float:
        """Lift-curve slope ``dCL/dalpha`` estimated over the near-linear range."""
        mask = np.abs(self.CL) < 1.2
        if mask.sum() < 3:
            return float("nan")
        coeffs = np.polyfit(self.alpha[mask], self.CL[mask], 1)
        return float(coeffs[0])

    @property
    def alpha_zero_lift(self) -> float:
        """Angle of zero lift from linear regression."""
        mask = np.abs(self.CL) < 1.2
        if mask.sum() < 3:
            return float("nan")
        coeffs = np.polyfit(self.alpha[mask], self.CL[mask], 1)
        return float(-coeffs[1] / coeffs[0])

    @property
    def LD_max(self) -> float:
        """Maximum lift-to-drag ratio in the sweep."""
        if len(self.LD) == 0 or not np.isfinite(self.LD).any():
            return float("nan")
        return float(np.nanmax(self.LD))

    def _ld_max_index(self) -> int | None:
        """Return the valid index of maximum L/D, or ``None`` when unavailable."""
        if len(self.LD) == 0:
            return None
        finite = np.isfinite(self.LD)
        if not finite.any():
            return None
        return int(np.nanargmax(np.where(finite, self.LD, -np.inf)))

    @property
    def alpha_at_LD_max(self) -> float:
        """Angle of attack at maximum lift-to-drag ratio."""
        idx = self._ld_max_index()
        if idx is None or idx >= len(self.alpha):
            return float("nan")
        return float(self.alpha[idx])

    @property
    def CL_at_LD_max(self) -> float:
        """Lift coefficient at maximum lift-to-drag ratio."""
        idx = self._ld_max_index()
        if idx is None or idx >= len(self.CL):
            return float("nan")
        return float(self.CL[idx])

    @property
    def CD_at_LD_max(self) -> float:
        """Drag coefficient at maximum lift-to-drag ratio."""
        idx = self._ld_max_index()
        if idx is None or idx >= len(self.CD):
            return float("nan")
        return float(self.CD[idx])

    @property
    def CD_min(self) -> float:
        """Minimum total drag coefficient observed in the sweep."""
        if len(self.CD) == 0:
            return float("nan")
        return float(np.nanmin(self.CD))

    @property
    def CD0(self) -> float:
        """Best available zero-lift drag coefficient."""
        return self.CD0_estimate

    @property
    def CD0_estimate(self) -> float:
        """Zero-lift drag from the polar fit, falling back to the current arrays."""
        if np.isfinite(self.cd0):
            return float(self.cd0)
        extraction = extract_cd0_from_arrays(self.alpha, self.CL, self.CD, self.CDo, source="VSPAEROResults arrays")
        return float(extraction.cd0)

    @property
    def oswald_mean(self) -> float:
        """Mean Oswald efficiency factor over the valid part of the sweep."""
        e_values = self.E[np.isfinite(self.E) & (self.E > 0)]
        return float(np.mean(e_values)) if len(e_values) > 0 else float("nan")

    @property
    def static_margin(self) -> float:
        """First finite static margin parsed from the ``.stab`` file."""
        return first_finite_value(self.stability_records, "SM")

    @property
    def neutral_point_x(self) -> float:
        """First finite longitudinal neutral point parsed from the ``.stab`` file."""
        return first_finite_value(self.stability_records, "X_np")

    @property
    def converged(self) -> bool:
        """Convergence flag parsed from the ``.history`` file."""
        return bool(self.convergence.get("converged", False))

    def set_cd0_extraction(self, extraction: CD0Extraction) -> None:
        """Attach detailed ``CD0`` extraction metadata to this result object."""
        self.cd0 = float(extraction.cd0)
        self.cd0_method = extraction.method
        self.cd0_source = extraction.source
        self.cd0_induced_factor = float(extraction.induced_factor)
        self.cd0_fit_r_squared = float(extraction.r_squared)
        self.cd0_direct_at_alpha0 = float(extraction.cd_at_alpha0)
        self.cd0_cdo_at_alpha0 = float(extraction.cdo_at_alpha0)
        self.cd0_n_points = int(extraction.n_points)
        self.warnings.extend(extraction.warnings)

    def interpolate_at_CL(self, cl_target: float) -> dict[str, float]:
        """Interpolate key coefficients at a given ``CL`` value."""
        if len(self.CL) == 0:
            raise ValueError("Cannot interpolate CL on an empty sweep.")
        if cl_target < self.CL.min() or cl_target > self.CL.max():
            raise ValueError(
                f"CL={cl_target:.3f} is outside the sweep range [{self.CL.min():.3f}, {self.CL.max():.3f}]."
            )

        out: dict[str, float] = {}
        for key in ("alpha", "CD", "CDi", "CDo", "CM", "LD"):
            arr = self.alpha if key == "alpha" else getattr(self, key)
            out[key] = float(np.interp(cl_target, self.CL, arr))
        out["CL"] = float(cl_target)
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Return the aerodynamic sweep as a tidy pandas DataFrame."""
        arrays = {
            "alpha [deg]": self.alpha,
            "CL [-]": self.CL,
            "CD [-]": self.CD,
            "CDi [-]": self.CDi,
            "CDo [-]": self.CDo,
            "CM [-]": self.CM,
            "CS [-]": self.CS,
            "L/D [-]": self.LD,
            "e (Oswald)": self.E,
        }
        data = {key: value for key, value in arrays.items() if len(value) == len(self.alpha)}
        df = pd.DataFrame(data)
        df["Mach"] = self.mach
        df["Re"] = self.re_cref
        df["Sref"] = self.Sref
        df["bref"] = self.bref
        df["cref"] = self.cref
        if self.case_name:
            df.insert(0, "case_name", self.case_name)
        return df

    def stability_dataframe(self) -> pd.DataFrame:
        """Return stability derivatives as a DataFrame."""
        if not self.stability_table.empty:
            return self.stability_table.copy()
        if self.stability_records:
            return pd.DataFrame(self.stability_records)
        return pd.DataFrame()

    def performance_summary(self) -> pd.Series:
        """Return a compact scalar summary for tables and exports."""
        summary = {
            "Case": self.case_name or "",
            "Mach": self.mach,
            "Re_cref": self.re_cref,
            "Sref [m^2]": self.Sref,
            "bref [m]": self.bref,
            "cref [m]": self.cref,
            "CL_alpha [1/deg]": self.CL_alpha,
            "alpha_0 [deg]": self.alpha_zero_lift,
            "L/D_max": self.LD_max,
            "alpha @ L/Dmax [deg]": self.alpha_at_LD_max,
            "CL @ L/Dmax": self.CL_at_LD_max,
            "CD @ L/Dmax": self.CD_at_LD_max,
            "CD_min": self.CD_min,
            "CD0": self.CD0_estimate,
            "CD0 method": self.cd0_method,
            "CD0 fit R^2": self.cd0_fit_r_squared,
            "Oswald e (mean)": self.oswald_mean,
            "alpha points": len(self.alpha),
            "stability rows": len(self.stability_records),
            "has .stab": bool(self.stab_path),
        }
        if np.isfinite(self.static_margin):
            summary["Static margin [-]"] = self.static_margin
        if np.isfinite(self.neutral_point_x):
            summary["X_np [m]"] = self.neutral_point_x
        if self.convergence:
            summary["Converged"] = self.converged
            summary["History iterations"] = self.convergence.get("n_iter", 0)
        return pd.Series(summary)

    def validate(self) -> list[str]:
        """
        Run lightweight integrity checks on the results.

        Returns a list of warning strings. An empty list means the arrays look
        consistent enough for downstream use.
        """
        issues: list[str] = []

        for name in ("CL", "CD", "CDi", "CDo", "CM", "LD"):
            arr = getattr(self, name)
            if len(arr) != len(self.alpha):
                issues.append(
                    f"Array length mismatch: {name} has {len(arr)} points but alpha has {len(self.alpha)}."
                )

        for name in ("CL", "CD", "LD"):
            arr = getattr(self, name)
            if len(arr) == 0:
                continue
            n_bad = int(np.sum(~np.isfinite(arr)))
            if n_bad > 0:
                issues.append(f"{n_bad} non-finite values found in {name}.")

        if len(self.CD) > 0 and np.any(self.CD < 0):
            issues.append("Negative CD values detected; check mesh and wake settings.")
        if len(self.CL) > 0 and np.max(np.abs(self.CL)) > 5.0:
            issues.append(f"Unrealistically large |CL| detected: {np.max(np.abs(self.CL)):.2f}.")
        if np.isfinite(self.LD_max) and self.LD_max > 80.0:
            issues.append(f"Suspiciously high L/D_max = {self.LD_max:.1f}. Check the reference area.")
        if 0 < self.n_points < 5:
            issues.append(
                f"Only {self.n_points} alpha points in sweep. Results may be coarse; consider increasing AlphaNpts."
            )

        if issues:
            logger.warning("VSPAEROResults validation issues:\n  %s", "\n  ".join(issues))
        return issues

    def __repr__(self) -> str:
        if len(self.alpha) == 0:
            return f"VSPAEROResults(M={self.mach:.2f}, Re={self.re_cref:.1e}, empty)"
        return (
            f"VSPAEROResults(case='{self.case_name or 'unnamed'}', M={self.mach:.2f}, Re={self.re_cref:.1e}, "
            f"alpha=[{self.alpha.min():.1f},{self.alpha.max():.1f}] x {self.n_points}, L/D_max={self.LD_max:.2f})"
        )


def _parse_polar_file_fallback(polar_path: str | Path, mach: float, re_cref: float) -> VSPAEROResults:
    """Read aerodynamic results directly from a physical ``.polar`` file."""
    polar_path = Path(polar_path)
    logger.info("Fallback active: parsing %s.", polar_path)

    if not polar_path.exists():
        logger.error("Fallback .polar file not found: %s", polar_path)
        return VSPAEROResults(mach=mach, re_cref=re_cref)

    try:
        with polar_path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()

        header_idx = -1
        for index, line in enumerate(lines):
            if "AoA" in line and "CLtot" in line:
                header_idx = index
                break

        if header_idx == -1:
            logger.error("Could not locate AoA/CLtot headers in %s.", polar_path)
            return VSPAEROResults(mach=mach, re_cref=re_cref)

        df = pd.read_csv(polar_path, sep=r"\s+", skiprows=header_idx)
        df.columns = df.columns.str.strip()
        n_pts = len(df)

        def get_col(*names: str) -> np.ndarray:
            for expected in names:
                for column in df.columns:
                    if column.lower() == expected.lower():
                        return df[column].to_numpy()
            return np.zeros(n_pts)

        cl_arr = get_col("CLtot", "CL")
        cd_arr = get_col("CDtot", "CD")
        ld_arr = get_col("L/D", "LD", "LoD")
        if len(ld_arr) == 0 or np.all(ld_arr == 0):
            raise RuntimeError("The .polar file did not provide a usable L/D column.")

        results = VSPAEROResults(
            mach=mach,
            re_cref=re_cref,
            alpha=get_col("AoA", "Alpha"),
            CL=cl_arr,
            CD=cd_arr,
            CDi=get_col("CDi"),
            CDo=get_col("CDo"),
            LD=ld_arr,
            E=get_col("E", "Ew"),
            CS=get_col("CStot", "CS"),
            CM=get_col("CMytot", "CMy", "CMoy", "CMm"),
            CMx=get_col("CMxtot", "CMx", "CMox"),
            CMz=get_col("CMztot", "CMz", "CMoz"),
            CFx=get_col("CFxtot", "CFx", "CFox"),
            CFy=get_col("CFytot", "CFy", "CFoy"),
            CFz=get_col("CFztot", "CFz", "CFoz"),
            n_points=n_pts,
        )
        results.warnings = results.validate()
        return results
    except Exception as exc:
        logger.error("Critical error during .polar parsing: %s", exc)
        return VSPAEROResults(mach=mach, re_cref=re_cref)


def results_from_stability_records(
    records: list[dict[str, float]],
    mach: float,
    re_cref: float,
) -> VSPAEROResults:
    """
    Rebuild the aerodynamic sweep from parsed ``.stab`` records.

    In stability mode, OpenVSP can omit the usual ``.polar`` file and expose
    incomplete data through the Results Manager. The ``.stab`` artifact still
    contains the base aerodynamic coefficients for each alpha point, so we use
    it as a reliable fallback whenever the primary sweep parser comes back
    empty.
    """
    if not records:
        return VSPAEROResults(mach=mach, re_cref=re_cref)

    def pick_array(ordered_records: list[dict[str, float]], *keys: str) -> np.ndarray:
        values: list[float] = []
        for record in ordered_records:
            value = float("nan")
            for key in keys:
                candidate = record.get(key)
                if candidate is None:
                    continue
                try:
                    candidate_value = float(candidate)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(candidate_value):
                    value = candidate_value
                    break
            values.append(value)
        return np.asarray(values, dtype=float)

    ordered = sorted(records, key=lambda entry: float(entry.get("AoA", float("nan"))))
    alpha = pick_array(ordered, "AoA")
    n_pts = len(alpha)

    cl = pick_array(ordered, "CL", "CFz")
    cd = pick_array(ordered, "CD", "CFx")
    cs = pick_array(ordered, "CS", "CFy")
    cm = pick_array(ordered, "CMy", "CMm")
    cmx = pick_array(ordered, "CMx", "CMl")
    cmz = pick_array(ordered, "CMz", "CMn")
    cfx = pick_array(ordered, "CFx")
    cfy = pick_array(ordered, "CFy")
    cfz = pick_array(ordered, "CFz")

    with np.errstate(divide="ignore", invalid="ignore"):
        ld = np.where(np.isfinite(cd) & (cd != 0.0), cl / cd, float("nan"))

    nan_array = np.full(n_pts, float("nan"))
    results = VSPAEROResults(
        mach=mach,
        re_cref=re_cref,
        alpha=alpha,
        CL=cl,
        CD=cd,
        CDi=nan_array.copy(),
        CDo=nan_array.copy(),
        CDsff=nan_array.copy(),
        CM=cm,
        CMx=cmx,
        CMz=cmz,
        CS=cs,
        LD=ld,
        E=nan_array.copy(),
        CFx=cfx,
        CFy=cfy,
        CFz=cfz,
        n_points=n_pts,
    )
    results.warnings = results.validate()
    return results


def _parse_results_manager(vsp, res_id: str, mach: float, re_cref: float, n_pts: int) -> VSPAEROResults:
    """
    Parse the OpenVSP Results Manager after a VSPAERO sweep run.

    OpenVSP sometimes stores sweep results as a vector of sub-results (one per
    alpha point) and sometimes stores them directly as arrays on the top-level
    result object. Both layouts are handled here.
    """

    data_names_cache: dict[str, set[str]] = {}

    def available_names(result_id: str) -> set[str]:
        if result_id not in data_names_cache:
            try:
                data_names_cache[result_id] = set(vsp.GetAllDataNames(result_id))
            except Exception:
                data_names_cache[result_id] = set()
        return data_names_cache[result_id]

    def safe_get_double(result_id: str, key: str) -> Optional[list[float]]:
        if key not in available_names(result_id):
            return None
        try:
            values = vsp.GetDoubleResults(result_id, key, 0)
            return list(values) if values else None
        except Exception:
            return None

    def safe_get_string(result_id: str, key: str) -> Optional[list[str]]:
        if key not in available_names(result_id):
            return None
        try:
            values = vsp.GetStringResults(result_id, key, 0)
            return list(values) if values else None
        except Exception:
            return None

    sub_ids = safe_get_string(res_id, "ResultsVec")

    if sub_ids:
        alphas, cls, cds, cdis, cdos, cdsffs, cms, cmxs, cmzs, css, lds, es = (
            [] for _ in range(12)
        )
        cfxs, cfys, cfzs = [], [], []

        debug_messages: list[str] = []
        seen_keys: set[str] = set()

        def get_scalar(sub_id: str, primary_key: str, fallback_key: str = "") -> float:
            values = safe_get_double(sub_id, primary_key)
            if values:
                if primary_key not in seen_keys:
                    debug_messages.append(f"{primary_key}: found directly")
                    seen_keys.add(primary_key)
                return float(values[0])

            if fallback_key:
                values = safe_get_double(sub_id, fallback_key)
                if values:
                    if primary_key not in seen_keys:
                        debug_messages.append(f"{primary_key}: resolved through fallback '{fallback_key}'")
                        seen_keys.add(primary_key)
                    return float(values[0])

            if primary_key not in seen_keys:
                debug_messages.append(
                    f"{primary_key}: missing from Results Manager"
                    + (f" (fallback '{fallback_key}' also missing)" if fallback_key else "")
                )
                seen_keys.add(primary_key)
            return float("nan")

        for sub_id in sub_ids:
            alphas.append(get_scalar(sub_id, "Alpha", "AoA"))
            cls.append(get_scalar(sub_id, "CL", "CLtot"))
            cds.append(get_scalar(sub_id, "CDtot", "CD"))
            cdis.append(get_scalar(sub_id, "CDi"))
            cdos.append(get_scalar(sub_id, "CDo"))
            cdsffs.append(get_scalar(sub_id, "CDsff"))
            cms.append(get_scalar(sub_id, "CMy", "CMytot"))
            cmxs.append(get_scalar(sub_id, "CMx", "CMxtot"))
            cmzs.append(get_scalar(sub_id, "CMz", "CMztot"))
            css.append(get_scalar(sub_id, "CS", "CStot"))
            lds.append(get_scalar(sub_id, "LD", "L/D"))
            es.append(get_scalar(sub_id, "E"))
            cfxs.append(get_scalar(sub_id, "CFx"))
            cfys.append(get_scalar(sub_id, "CFy"))
            cfzs.append(get_scalar(sub_id, "CFz"))

        if debug_messages:
            logger.debug("Results Manager mapping summary:\n%s", "\n".join(f"  - {msg}" for msg in debug_messages))

        alpha_arr = np.array(alphas)
        sort_idx = np.argsort(alpha_arr)

        def arr(values: list[float]) -> np.ndarray:
            return np.array(values)[sort_idx]

        results = VSPAEROResults(
            mach=mach,
            re_cref=re_cref,
            alpha=arr(alphas),
            CL=arr(cls),
            CD=arr(cds),
            CDi=arr(cdis),
            CDo=arr(cdos),
            CDsff=arr(cdsffs),
            CM=arr(cms),
            CMx=arr(cmxs),
            CMz=arr(cmzs),
            CS=arr(css),
            LD=arr(lds),
            E=arr(es),
            CFx=arr(cfxs),
            CFy=arr(cfys),
            CFz=arr(cfzs),
            n_points=len(alphas),
        )
    else:

        def arr(key: str) -> np.ndarray:
            values = safe_get_double(res_id, key)
            return np.array(values) if values else np.array([])

        alpha = arr("Alpha")
        cl = arr("CL")
        cdtot = arr("CDtot")
        cdi = arr("CDi")
        cdo = arr("CDo")
        cdsff = arr("CDsff")
        cm = arr("CMy")
        cmx = arr("CMx")
        cmz = arr("CMz")
        cs = arr("CS")
        ld = arr("LD")
        e_values = arr("E")
        cfx = arr("CFx")
        cfy = arr("CFy")
        cfz = arr("CFz")

        sort_idx = np.argsort(alpha) if len(alpha) > 0 else slice(None)
        results = VSPAEROResults(
            mach=mach,
            re_cref=re_cref,
            alpha=alpha[sort_idx],
            CL=cl[sort_idx] if len(cl) else cl,
            CD=cdtot[sort_idx] if len(cdtot) else cdtot,
            CDi=cdi[sort_idx] if len(cdi) else cdi,
            CDo=cdo[sort_idx] if len(cdo) else cdo,
            CDsff=cdsff[sort_idx] if len(cdsff) else cdsff,
            CM=cm[sort_idx] if len(cm) else cm,
            CMx=cmx[sort_idx] if len(cmx) else cmx,
            CMz=cmz[sort_idx] if len(cmz) else cmz,
            CS=cs[sort_idx] if len(cs) else cs,
            LD=ld[sort_idx] if len(ld) else ld,
            E=e_values[sort_idx] if len(e_values) else e_values,
            CFx=cfx[sort_idx] if len(cfx) else cfx,
            CFy=cfy[sort_idx] if len(cfy) else cfy,
            CFz=cfz[sort_idx] if len(cfz) else cfz,
            n_points=len(alpha),
        )

    if len(results.LD) == 0 or np.all(results.LD == 0):
        with np.errstate(divide="ignore", invalid="ignore"):
            results.LD = np.where(results.CD != 0, results.CL / results.CD, 0.0)

    if len(results.E) == 0 or np.all(results.E == 0):
        aspect_ratio = 10.0
        if results.Sref > 0 and results.bref > 0:
            aspect_ratio = results.bref**2 / results.Sref
        with np.errstate(divide="ignore", invalid="ignore"):
            results.E = np.where(
                results.CDi > 0,
                results.CL**2 / (np.pi * aspect_ratio * results.CDi),
                float("nan"),
            )

    results.warnings = results.validate()
    return results
