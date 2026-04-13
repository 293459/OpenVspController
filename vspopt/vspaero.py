"""
VSPAERO results parsing and the ``VSPAEROResults`` dataclass.

The VSPAERO solver writes its output into OpenVSP's Results Manager and, when
enabled by the solver, also into text files such as ``.history`` and ``.stab``.
This module keeps the main aerodynamic arrays in one place and exposes a few
extra diagnostics that were previously only available in the legacy notebook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from os import path
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from vspopt.postprocess import first_finite_value

logger = logging.getLogger(__name__)


@dataclass
class VSPAEROResults:
    """
    Container for a single VSPAERO sweep run.

    All 1-D arrays are aligned: result[i] corresponds to alpha[i].
    """

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
    warnings: list[str] = field(default_factory=list)
    history_path: Optional[Path] = None
    history_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    convergence: dict[str, object] = field(default_factory=dict)
    stab_path: Optional[Path] = None
    stability_records: list[dict[str, float]] = field(default_factory=list)
    stability_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    mass_properties: Optional["MassProperties"] = None

    @property
    def CL_alpha(self) -> float:
        """
        Lift-curve slope dCL/dalpha [1/deg], estimated by linear regression
        over the linear portion of the polar (typically |CL| < 1.2).
        """
        mask = np.abs(self.CL) < 1.2
        if mask.sum() < 3:
            return float("nan")
        coeffs = np.polyfit(self.alpha[mask], self.CL[mask], 1)
        return float(coeffs[0])

    @property
    def alpha_zero_lift(self) -> float:
        """Angle of zero lift alpha_0 [deg] from linear regression."""
        mask = np.abs(self.CL) < 1.2
        if mask.sum() < 3:
            return float("nan")
        coeffs = np.polyfit(self.alpha[mask], self.CL[mask], 1)
        return float(-coeffs[1] / coeffs[0])

    @property
    def LD_max(self) -> float:
        """Maximum L/D ratio in the sweep."""
        if len(self.LD) == 0:
            return float("nan")
        return float(np.nanmax(self.LD))

    @property
    def alpha_at_LD_max(self) -> float:
        """Angle of attack at maximum L/D [deg]."""
        if len(self.LD) == 0:
            return float("nan")
        return float(self.alpha[np.nanargmax(self.LD)])

    @property
    def CL_at_LD_max(self) -> float:
        """CL at maximum L/D."""
        if len(self.LD) == 0:
            return float("nan")
        return float(self.CL[np.nanargmax(self.LD)])

    @property
    def CD_at_LD_max(self) -> float:
        """CD at maximum L/D."""
        if len(self.CD) == 0:
            return float("nan")
        return float(self.CD[np.nanargmax(self.LD)])

    @property
    def CD0_estimate(self) -> float:
        """Estimate of the zero-lift drag coefficient from the minimum CD."""
        if len(self.CD) == 0:
            return float("nan")
        return float(np.nanmin(self.CD))

    @property
    def oswald_mean(self) -> float:
        """Mean Oswald efficiency factor over the sweep."""
        e_values = self.E[np.isfinite(self.E) & (self.E > 0)]
        return float(np.mean(e_values)) if len(e_values) > 0 else float("nan")

    @property
    def static_margin(self) -> float:
        """Return the first finite static margin parsed from the .stab file."""
        return first_finite_value(self.stability_records, "SM")

    @property
    def neutral_point_x(self) -> float:
        """Return the first finite longitudinal neutral point parsed from .stab."""
        return first_finite_value(self.stability_records, "X_np")

    @property
    def converged(self) -> bool:
        """Return the convergence flag parsed from the .history file."""
        return bool(self.convergence.get("converged", False))

    def interpolate_at_CL(self, cl_target: float) -> dict[str, float]:
        """
        Interpolate all coefficients at a given CL value.

        Parameters
        ----------
        cl_target : Target lift coefficient.

        Returns
        -------
        dict with keys: alpha, CL, CD, CDi, CDo, CM, LD
        """
        if cl_target < self.CL.min() or cl_target > self.CL.max():
            raise ValueError(
                f"CL={cl_target:.3f} is outside the sweep range "
                f"[{self.CL.min():.3f}, {self.CL.max():.3f}]."
            )
        out: dict[str, float] = {}
        for key in ("alpha", "CD", "CDi", "CDo", "CM", "LD"):
            arr = self.alpha if key == "alpha" else getattr(self, key)
            out[key] = float(np.interp(cl_target, self.CL, arr))
        out["CL"] = cl_target
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Return all sweep results as a tidy pandas DataFrame."""
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
        return df

    def stability_dataframe(self) -> pd.DataFrame:
        """Return the parsed stability derivatives as a DataFrame."""
        if not self.stability_table.empty:
            return self.stability_table.copy()
        if self.stability_records:
            return pd.DataFrame(self.stability_records)
        return pd.DataFrame()

    def performance_summary(self) -> pd.Series:
        """Return key scalar performance and diagnostics as a pandas Series."""
        summary = {
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
            "CD0 (min CD)": self.CD0_estimate,
            "Oswald e (mean)": self.oswald_mean,
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
        Run integrity checks on the results.

        Returns a list of warning strings. An empty list means the
        results look physically plausible.
        """
        issues: list[str] = []

        for name in ("CL", "CD", "CDi", "CDo", "CM", "LD"):
            arr = getattr(self, name)
            if len(arr) != len(self.alpha):
                issues.append(
                    f"Array length mismatch: {name} has {len(arr)} points but alpha has "
                    f"{len(self.alpha)}."
                )

        for name in ("CL", "CD", "LD"):
            arr = getattr(self, name)
            n_bad = int(np.sum(~np.isfinite(arr)))
            if n_bad > 0:
                issues.append(f"{n_bad} non-finite values found in {name}.")

        if len(self.CD) > 0 and np.any(self.CD < 0):
            issues.append("Negative CD values detected; check mesh and wake settings.")
        if len(self.CL) > 0 and np.max(np.abs(self.CL)) > 5.0:
            issues.append(f"Unrealistically large CL detected: {np.max(np.abs(self.CL)):.2f}.")
        if self.LD_max > 80:
            issues.append(f"Suspiciously high L/D_max = {self.LD_max:.1f}. Check reference area.")
        if self.n_points < 5:
            issues.append(
                f"Only {self.n_points} alpha points in sweep. Results may be coarse; "
                "consider increasing AlphaNpts."
            )

        if issues:
            logger.warning("VSPAEROResults validation issues:\n  %s", "\n  ".join(issues))
        return issues

    def __repr__(self) -> str:
        if len(self.alpha) == 0:
            return f"VSPAEROResults(M={self.mach:.2f}, Re={self.re_cref:.1e}, empty)"
        return (
            f"VSPAEROResults(M={self.mach:.2f}, Re={self.re_cref:.1e}, "
            f"alpha=[{self.alpha.min():.1f},{self.alpha.max():.1f}] x {self.n_points} pts, "
            f"L/D_max={self.LD_max:.2f})"
        )

def _parse_polar_file_fallback(polar_path: str | Path, mach: float, re_cref: float) -> VSPAEROResults:
    """
    Bulletproof fallback to read aerodynamic results directly from the .polar text file.
    Ensures all arrays have consistent lengths to prevent 'argmax' or validation crashes.
    """
    logger.info(f"Fallback active: Parsing physical file {polar_path}")
    path = Path(polar_path)
    
    if not path.exists():
        logger.error(f"File not found: {path}")
        return VSPAEROResults(mach=mach, re_cref=re_cref)

    try:
        # 1. Locate the data header row (skipping VSPAERO introductory text)
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            
        header_idx = -1
        for i, line in enumerate(lines):
            # Look for standard VSPAERO column headers
            if "AoA" in line and "CLtot" in line:
                header_idx = i
                break
                
        if header_idx == -1:
             logger.error("Could not find AoA/CLtot headers in the .polar file.")
             return VSPAEROResults(mach=mach, re_cref=re_cref)

        # 2. Read the table and sanitize column names
        df = pd.read_csv(path, sep=r"\s+", skiprows=header_idx)
        df.columns = df.columns.str.strip()  # Clean hidden whitespace/tabs
        n_pts = len(df)

        # 3. HELPER FUNCTION: Safely fetch columns or return zeros to prevent mismatches
        def get_col(*names):
            """
            Searches for a column by name (case-insensitive).
            Returns a zero-array of length n_pts if no match is found.
            """
            for name in names:
                for col in df.columns:
                    if col.lower() == name.lower():
                        return df[col].to_numpy()
            
            # Fallback to zeros to keep array lengths consistent (prevents crashes)
            return np.zeros(n_pts)

        cl_arr = get_col("CLtot", "CL")
        cd_arr = get_col("CDtot", "CD")

       
        ld_arr = get_col("L/D", "LD", "LoD")

        # If the solver returns zeros or fails to provide the column, we crash.
        if ld_arr is None or len(ld_arr) == 0 or np.all(ld_arr == 0):
            raise RuntimeError(
                "STRICT ERROR: L/D (Lift-to-Drag) column is missing or invalid. "
                "VSPAERO failed to provide performance ratios. Check for flow "
                "separation or solver divergence in the .history file."
            )
        # 5. Return the mapped data structure
        return VSPAEROResults(
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
            # Map pitching moments (Y-axis) and other moments using common VSPAERO naming conventions
            CM=get_col("CMytot", "CMy", "CMoy", "CMm"),
            CMx=get_col("CMxtot", "CMx", "CMox"),
            CMz=get_col("CMztot", "CMz", "CMoz"),
            CFx=get_col("CFxtot", "CFx", "CFox"),
            CFy=get_col("CFytot", "CFy", "CFoy"),
            CFz=get_col("CFztot", "CFz", "CFoz"),
            n_points=n_pts,
        )

    except Exception as e:
        logger.error(f"Critical error during .polar file parsing: {e}")
        return VSPAEROResults(mach=mach, re_cref=re_cref)
    

def _parse_results_manager(vsp, res_id: str, mach: float, re_cref: float, n_pts: int) -> VSPAEROResults:
    """
    Parse the OpenVSP Results Manager after a VSPAERO sweep run.

    OpenVSP stores sweep results either as a vector of sub-results (one per
    alpha point) or directly as arrays on the top-level result object.
    """

    def safe_get_double(rid: str, key: str) -> Optional[list[float]]:
        try:
            values = vsp.GetDoubleResults(rid, key, 0)
            return list(values) if values else None
        except Exception:
            return None

    def safe_get_string(rid: str, key: str) -> Optional[list[str]]:
        try:
            values = vsp.GetStringResults(rid, key, 0)
            return list(values) if values else None
        except Exception:
            return None

    sub_ids = safe_get_string(res_id, "ResultsVec")

    if sub_ids:
        alphas, cls, cds, cdis, cdos, cdsffs, cms, cmxs, cmzs, css, lds, es = (
            [] for _ in range(12)
        )
        cfxs, cfys, cfzs = [], [], []

        # Dictionary to track which keys were successfully mapped on the first pass
        # This prevents spamming the terminal with 26 identical debug messages.
        debug_tracked_keys = {}

        for iteration_index, sub_id in enumerate(sub_ids):

            def g(primary_key: str, fallback_key: str = "") -> float:
                # 1. Try to fetch using the primary key (e.g., "CL")
                values = safe_get_double(sub_id, primary_key)
                if values:
                    # Print debug info only on the very first iteration
                    if iteration_index == 0 and primary_key not in debug_tracked_keys:
                        print(f"[VSPAERO DEBUG] '{primary_key:<5}' -> Found directly.")
                        debug_tracked_keys[primary_key] = True
                    return float(values[0])
                
                # 2. Try to fetch using the fallback key (e.g., "CLtot")
                if fallback_key:
                    values = safe_get_double(sub_id, fallback_key)
                    if values:
                        if iteration_index == 0 and primary_key not in debug_tracked_keys:
                            print(f"[VSPAERO DEBUG] '{primary_key:<5}' -> Found using fallback: '{fallback_key}'")
                            debug_tracked_keys[primary_key] = True
                        return float(values[0])
                
                # 3. If both fail, return NaN and warn the user
                if iteration_index == 0 and primary_key not in debug_tracked_keys:
                    error_msg = f"'{primary_key}'" if not fallback_key else f"'{primary_key}' or '{fallback_key}'"
                    print(f"[VSPAERO DEBUG] WARNING: Could not find {error_msg} in Results Manager!")
                    debug_tracked_keys[primary_key] = True
                
                return float("nan")

            # --- DATA EXTRACTION WITH FALLBACKS ---
            alphas.append(g("Alpha", "AoA"))
            cls.append(g("CL", "CLtot"))
            cds.append(g("CDtot", "CD"))
            cdis.append(g("CDi"))
            cdos.append(g("CDo"))
            cdsffs.append(g("CDsff"))
            cms.append(g("CMy", "CMytot"))
            cmxs.append(g("CMx", "CMxtot"))
            cmzs.append(g("CMz", "CMztot"))
            css.append(g("CS", "CStot"))
            lds.append(g("LD", "L/D"))
            es.append(g("E"))
            cfxs.append(g("CFx"))
            cfys.append(g("CFy"))
            cfzs.append(g("CFz"))

        # Sort arrays based on Alpha to ensure the polar graph is plotted correctly
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
        aspect_ratio = (results.bref**2 / results.Sref) if (results.Sref > 0 and results.bref > 0) else 10.0
        with np.errstate(divide="ignore", invalid="ignore"):
            results.E = np.where(
                results.CDi > 0,
                results.CL**2 / (np.pi * aspect_ratio * results.CDi),
                float("nan"),
            )

    results.warnings = results.validate()
    return results
