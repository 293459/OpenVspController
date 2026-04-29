"""
Post-processing helpers inspired by the legacy OpenVSP notebook workflow.

These utilities keep the current project structure intact while importing a few
generic and reusable ideas from the colleague notebook:
  - parsing ``.stab`` stability-derivative files
  - checking solver convergence from ``.history`` files
  - collecting mass-property outputs
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

STAB_SCALAR_KEYS = {
    "Sref_": "Sref",
    "Cref_": "Cref",
    "Bref_": "Bref",
    "Xcg_": "Xcg",
    "Ycg_": "Ycg",
    "Zcg_": "Zcg",
    "Mach_": "Mach_cond",
    "AoA_": "AoA",
    "Beta_": "Beta_cond",
    "Rho_": "Rho",
    "Vinf_": "Vinf_cond",
    "SM": "SM",
    "X_np": "X_np",
}
STAB_SUFFIX_KEYS = {
    "Alpha": "_Alpha",
    "Beta": "_Beta",
    "p": "_p",
    "q": "_q",
    "r": "_r",
    "Mach": "_Mach",
    "U": "_U",
    "ConGrp1": "_ConGrp1",
    "ConGrp2": "_ConGrp2",
    "ConGrp3": "_ConGrp3",
}


@dataclass
class CD0Extraction:
    """Details of a zero-lift drag extraction from a VSPAERO polar."""

    cd0: float = float("nan")
    method: str = "unavailable"
    source: str = ""
    induced_factor: float = float("nan")
    r_squared: float = float("nan")
    cd_at_alpha0: float = float("nan")
    cdo_at_alpha0: float = float("nan")
    n_points: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_series(self) -> pd.Series:
        """Return a notebook-friendly one-row summary."""
        return pd.Series(
            {
                "CD0 [-]": self.cd0,
                "method": self.method,
                "source": self.source,
                "k [-]": self.induced_factor,
                "fit R^2 [-]": self.r_squared,
                "CD at AoA=0 [-]": self.cd_at_alpha0,
                "CDo at AoA=0 [-]": self.cdo_at_alpha0,
                "fit points": self.n_points,
                "warnings": "; ".join(self.warnings),
            }
        )


@dataclass
class MassProperties:
    """Structured subset of the OpenVSP ``MassProp`` output."""

    total_mass: float = float("nan")
    xcg: float = float("nan")
    ycg: float = float("nan")
    zcg: float = float("nan")
    ixx: float = float("nan")
    iyy: float = float("nan")
    izz: float = float("nan")
    ixy: float = float("nan")
    ixz: float = float("nan")
    iyz: float = float("nan")
    volume: float = float("nan")
    raw_results: dict[str, Any] = field(default_factory=dict)

    @property
    def cg_tuple(self) -> tuple[float, float, float]:
        """Return the center of gravity as ``(x, y, z)``."""
        return (self.xcg, self.ycg, self.zcg)

    @property
    def cg_is_finite(self) -> bool:
        """Return True when the center of gravity was successfully computed."""
        return all(np.isfinite(value) for value in self.cg_tuple)

    def to_series(self) -> pd.Series:
        """Return a compact pandas series for notebook and UI display."""
        return pd.Series(
            {
                "mass [kg]": self.total_mass,
                "xcg [m]": self.xcg,
                "ycg [m]": self.ycg,
                "zcg [m]": self.zcg,
                "Ixx [kg m^2]": self.ixx,
                "Iyy [kg m^2]": self.iyy,
                "Izz [kg m^2]": self.izz,
                "Ixy [kg m^2]": self.ixy,
                "Ixz [kg m^2]": self.ixz,
                "Iyz [kg m^2]": self.iyz,
                "volume [m^3]": self.volume,
                
                
            }
        )

    @classmethod
    def from_results(cls, raw_results: dict[str, Any]) -> "MassProperties":
        """Build a ``MassProperties`` object from scalar OpenVSP result values."""
        import math

        def pick(*names: str) -> float:
            for name in names:
                value = raw_results.get(name)
                if isinstance(value, list):
                    value = value[0] if value else float("nan")
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return float("nan")

        # --- NEW: Extract the Vector CG ---
        # Look for the exact key OpenVSP generated: 'Total_CG'
        cg_data = raw_results.get("Total_CG")
        
        # If it found the vector [(x, y, z)], unpack it
        if cg_data and isinstance(cg_data, list) and isinstance(cg_data[0], tuple):
            cg_x, cg_y, cg_z = cg_data[0]
        else:
            cg_x, cg_y, cg_z = float("nan"), float("nan"), float("nan")

        return cls(
            total_mass=pick("Total_Mass", "Mass"),
            # Use the unpacked vector coordinates, with a fallback just in case
            xcg=cg_x if not math.isnan(cg_x) else pick("Xcg", "X_cg"),
            ycg=cg_y if not math.isnan(cg_y) else pick("Ycg", "Y_cg"),
            zcg=cg_z if not math.isnan(cg_z) else pick("Zcg", "Z_cg"),
            ixx=pick("Total_Ixx", "Ixx"),
            iyy=pick("Total_Iyy", "Iyy"),
            izz=pick("Total_Izz", "Izz"),
            ixy=pick("Total_Ixy", "Ixy"),
            ixz=pick("Total_Ixz", "Ixz"),
            iyz=pick("Total_Iyz", "Iyz"),
            volume=pick("Total_Volume", "Volume"),
            raw_results=dict(raw_results),
        )


def _normalized_column_name(name: object) -> str:
    """Normalize OpenVSP column labels such as ``CDtot`` and ``CD [-]``."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _pick_numeric_column(df: pd.DataFrame, *names: str) -> np.ndarray | None:
    """Return the first numeric column whose normalized name matches ``names``."""
    lookup = {_normalized_column_name(column): column for column in df.columns}
    for name in names:
        column = lookup.get(_normalized_column_name(name))
        if column is None:
            continue
        return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    return None


def read_polar_file(path: str | Path) -> pd.DataFrame:
    """Read an OpenVSP ``.polar`` file into a numeric DataFrame."""
    polar_path = Path(path)
    if not polar_path.exists():
        raise FileNotFoundError(f"Polar file not found: {polar_path}")

    lines = polar_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = None
    for index, line in enumerate(lines):
        normalized = _normalized_column_name(line)
        has_alpha = "aoa" in normalized or "alpha" in normalized
        has_lift = "cltot" in normalized or re.search(r"\bcl\b", line, flags=re.IGNORECASE)
        has_drag = "cdtot" in normalized or re.search(r"\bcd\b", line, flags=re.IGNORECASE)
        if has_alpha and has_lift and has_drag:
            header_index = index
            break

    if header_index is None:
        raise ValueError(f"Could not locate AoA/CL/CD headers in {polar_path}")

    df = pd.read_csv(polar_path, sep=r"\s+", skiprows=header_index)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def _value_at_alpha_zero(alpha: np.ndarray | None, values: np.ndarray | None) -> float:
    """Return the value at AoA=0 when present, interpolating only if bracketed."""
    if alpha is None or values is None or len(alpha) == 0 or len(values) == 0:
        return float("nan")

    n = min(len(alpha), len(values))
    alpha = alpha[:n]
    values = values[:n]
    mask = np.isfinite(alpha) & np.isfinite(values)
    if mask.sum() == 0:
        return float("nan")

    alpha_valid = alpha[mask]
    values_valid = values[mask]
    exact = np.isclose(alpha_valid, 0.0, atol=1e-9)
    if exact.any():
        return float(values_valid[np.where(exact)[0][0]])

    order = np.argsort(alpha_valid)
    alpha_sorted = alpha_valid[order]
    values_sorted = values_valid[order]
    if alpha_sorted[0] <= 0.0 <= alpha_sorted[-1] and len(alpha_sorted) >= 2:
        return float(np.interp(0.0, alpha_sorted, values_sorted))
    return float("nan")


def extract_cd0_from_arrays(
    alpha: Iterable[float] | np.ndarray | None,
    cl: Iterable[float] | np.ndarray | None,
    cd: Iterable[float] | np.ndarray | None,
    cdo: Iterable[float] | np.ndarray | None = None,
    *,
    source: str = "arrays",
) -> CD0Extraction:
    """
    Extract ``CD0`` by fitting ``CD = CD0 + k CL^2``.

    The fit is the primary value because it uses the whole near-linear polar.
    Direct ``AoA=0`` values are kept as cross-checks and used only as fallback.
    """
    alpha_arr = None if alpha is None else np.asarray(list(alpha), dtype=float)
    cl_arr = None if cl is None else np.asarray(list(cl), dtype=float)
    cd_arr = None if cd is None else np.asarray(list(cd), dtype=float)
    cdo_arr = None if cdo is None else np.asarray(list(cdo), dtype=float)

    warnings: list[str] = []
    if cl_arr is None or cd_arr is None or len(cl_arr) == 0 or len(cd_arr) == 0:
        return CD0Extraction(source=source, warnings=["CL/CD arrays are empty."])

    n = min(len(cl_arr), len(cd_arr))
    cl_arr = cl_arr[:n]
    cd_arr = cd_arr[:n]
    if alpha_arr is not None:
        alpha_arr = alpha_arr[: min(len(alpha_arr), n)]
    if cdo_arr is not None:
        cdo_arr = cdo_arr[: min(len(cdo_arr), n)]

    cd_at_alpha0 = _value_at_alpha_zero(alpha_arr, cd_arr)
    cdo_at_alpha0 = _value_at_alpha_zero(alpha_arr, cdo_arr)

    mask = np.isfinite(cl_arr) & np.isfinite(cd_arr)
    cl_fit = cl_arr[mask]
    cd_fit = cd_arr[mask]
    extraction = CD0Extraction(
        source=source,
        cd_at_alpha0=cd_at_alpha0,
        cdo_at_alpha0=cdo_at_alpha0,
        n_points=int(mask.sum()),
    )

    if mask.sum() >= 2:
        x = cl_fit**2
        y = cd_fit
        if np.nanmax(x) - np.nanmin(x) > 1e-12:
            k, intercept = np.polyfit(x, y, 1)
            y_hat = k * x + intercept
            residual = float(np.sum((y - y_hat) ** 2))
            spread = float(np.sum((y - np.mean(y)) ** 2))
            r_squared = float(1.0 - residual / spread) if spread > 0 else float("nan")

            # A negative intercept is normally a bad polar or too-wide fit range,
            # so keep the diagnostic but fall back to a direct value below.
            if np.isfinite(intercept) and intercept >= 0.0:
                extraction.cd0 = float(intercept)
                extraction.method = "CD_vs_CL2_regression"
                extraction.induced_factor = float(k)
                extraction.r_squared = r_squared
                return extraction
            warnings.append(f"Regression intercept was non-physical ({intercept:.6g}).")
        else:
            warnings.append("CL^2 range is too small for a regression.")
    else:
        warnings.append("Fewer than two finite CL/CD points are available.")

    if np.isfinite(cdo_at_alpha0):
        extraction.cd0 = float(cdo_at_alpha0)
        extraction.method = "CDo_at_AoA0"
    elif np.isfinite(cd_at_alpha0):
        extraction.cd0 = float(cd_at_alpha0)
        extraction.method = "CD_at_AoA0"
    elif len(cd_fit) > 0:
        extraction.cd0 = float(np.nanmin(cd_fit))
        extraction.method = "minimum_CD_fallback"
        warnings.append("Using minimum CD because the fit and AoA=0 checks were unavailable.")
    else:
        extraction.method = "unavailable"

    extraction.warnings = warnings
    return extraction


def extract_cd0_details(polar_file: str | Path) -> CD0Extraction:
    """Parse a ``.polar`` file and return detailed ``CD0`` extraction metadata."""
    df = read_polar_file(polar_file)
    return extract_cd0_from_dataframe(df, source=str(Path(polar_file)))


def extract_cd0_from_dataframe(df: pd.DataFrame, *, source: str = "polar dataframe") -> CD0Extraction:
    """Extract ``CD0`` from a DataFrame containing OpenVSP polar columns."""
    alpha = _pick_numeric_column(df, "AoA", "Alpha")
    cl = _pick_numeric_column(df, "CLtot", "CL")
    cd = _pick_numeric_column(df, "CDtot", "CD")
    cdo = _pick_numeric_column(df, "CDo", "CD0")
    return extract_cd0_from_arrays(alpha, cl, cd, cdo, source=source)


def extract_cd0(polar_file: str | Path) -> float:
    """Return the zero-lift drag coefficient extracted from a ``.polar`` file."""
    return extract_cd0_details(polar_file).cd0


def _first_finite_number(mapping: dict[str, Any], *names: str, default: float = float("nan")) -> float:
    """Read the first finite numeric value from a flexible parameter mapping."""
    for name in names:
        value = mapping.get(name)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric
    return float(default)


def _skin_friction_coefficient(reynolds: float, mach: float, laminar_fraction: float = 0.0) -> float:
    """Raymer-style flat-plate skin-friction coefficient with optional laminar blend."""
    if not np.isfinite(reynolds) or reynolds <= 1.0:
        return float("nan")
    laminar_fraction = float(np.clip(laminar_fraction, 0.0, 1.0))
    cf_laminar = 1.328 / math.sqrt(reynolds)
    cf_turbulent = 0.455 / (
        (math.log10(reynolds) ** 2.58) * ((1.0 + 0.144 * mach**2) ** 0.65)
    )
    return float(laminar_fraction * cf_laminar + (1.0 - laminar_fraction) * cf_turbulent)


def _default_wetted_area(component: dict[str, Any], component_type: str) -> float:
    """Estimate wetted area when the caller does not provide one explicitly."""
    wetted_area = _first_finite_number(component, "S_wet", "Swet", "wetted_area")
    if np.isfinite(wetted_area):
        return wetted_area

    area = _first_finite_number(component, "area", "planform_area", "S")
    if component_type in {"wing", "tail", "htp", "vtp", "lifting_surface"} and np.isfinite(area):
        thickness_ratio = _first_finite_number(component, "t_c", "thickness_to_chord", default=0.12)
        return float(2.0 * area * (1.0 + 0.25 * thickness_ratio))

    length = _first_finite_number(component, "length", "l")
    diameter = _first_finite_number(component, "diameter", "max_diameter", "d")
    if component_type == "fuselage" and np.isfinite(length) and np.isfinite(diameter) and diameter > 0:
        fineness = length / diameter
        correction = (1.0 - 2.0 / max(fineness, 2.1)) ** (2.0 / 3.0) * (1.0 + 1.0 / fineness**2)
        return float(math.pi * diameter * length * correction)

    return float("nan")


def _default_form_factor(component: dict[str, Any], component_type: str) -> float:
    """Return a conservative Raymer/DATCOM-style form factor."""
    form_factor = _first_finite_number(component, "FF", "form_factor")
    if np.isfinite(form_factor):
        return form_factor

    if component_type in {"wing", "tail", "htp", "vtp", "lifting_surface"}:
        thickness_ratio = _first_finite_number(component, "t_c", "thickness_to_chord", default=0.12)
        return float(1.0 + 2.0 * thickness_ratio + 60.0 * thickness_ratio**4)

    if component_type == "fuselage":
        fineness = _first_finite_number(component, "fineness_ratio", "l_d")
        if not np.isfinite(fineness):
            length = _first_finite_number(component, "length", "l")
            diameter = _first_finite_number(component, "diameter", "max_diameter", "d")
            fineness = length / diameter if np.isfinite(length) and np.isfinite(diameter) and diameter > 0 else 6.0
        return float(1.0 + 60.0 / fineness**3 + fineness / 400.0)

    return 1.0


def estimate_cd0_breakdown(geometry_params: dict[str, Any]) -> dict[str, Any]:
    """
    Estimate ``CD0`` with the Raymer wetted-area method.

    ``geometry_params`` accepts ``S_ref`` and a ``components`` list. Each
    component may provide ``Cf``, ``FF``, ``Q`` and ``S_wet`` directly, or enough
    geometry/Reynolds data for the defaults to be estimated.
    """
    s_ref = _first_finite_number(geometry_params, "S_ref", "Sref", "s_ref")
    mach = _first_finite_number(geometry_params, "mach", "Mach", default=0.0)
    re_cref = _first_finite_number(geometry_params, "re_cref", "Re_cref", "ReCref")
    cref = _first_finite_number(geometry_params, "cref", "Cref", "c_ref")
    re_per_length = re_cref / cref if np.isfinite(re_cref) and np.isfinite(cref) and cref > 0 else float("nan")

    rows: list[dict[str, float | str]] = []
    warnings: list[str] = []
    components = geometry_params.get("components", [])

    if not np.isfinite(s_ref) or s_ref <= 0.0:
        return {
            "CD0": float("nan"),
            "S_ref": s_ref,
            "components": pd.DataFrame(),
            "warnings": ["S_ref is missing or non-positive."],
        }

    for index, raw_component in enumerate(components, start=1):
        component = dict(raw_component)
        component_type = str(component.get("type", "generic")).strip().lower()
        name = str(component.get("name", f"component_{index}"))

        characteristic_length = _first_finite_number(
            component,
            "characteristic_length",
            "length_ref",
            "mac",
            "chord",
            "length",
        )
        reynolds = _first_finite_number(component, "Re", "reynolds")
        if not np.isfinite(reynolds) and np.isfinite(re_per_length) and np.isfinite(characteristic_length):
            reynolds = re_per_length * characteristic_length

        cf = _first_finite_number(component, "Cf", "cf")
        if not np.isfinite(cf):
            cf = _skin_friction_coefficient(
                reynolds,
                mach,
                laminar_fraction=_first_finite_number(component, "laminar_fraction", default=0.0),
            )

        form_factor = _default_form_factor(component, component_type)
        interference = _first_finite_number(component, "Q", "interference_factor", default=1.0)
        wetted_area = _default_wetted_area(component, component_type)

        if not all(np.isfinite(value) for value in (cf, form_factor, interference, wetted_area)):
            warnings.append(f"{name}: insufficient data for a finite CD0 contribution.")
            contribution = float("nan")
        else:
            contribution = float(cf * form_factor * interference * wetted_area / s_ref)

        rows.append(
            {
                "component": name,
                "type": component_type,
                "Cf": cf,
                "FF": form_factor,
                "Q": interference,
                "S_wet [m^2]": wetted_area,
                "Re": reynolds,
                "CD0 contribution": contribution,
            }
        )

    component_table = pd.DataFrame(rows)
    cd0 = float(component_table["CD0 contribution"].sum(skipna=True)) if not component_table.empty else float("nan")
    if component_table.empty:
        warnings.append("No components were provided for the analytical CD0 estimate.")

    return {
        "CD0": cd0,
        "S_ref": s_ref,
        "components": component_table,
        "warnings": warnings,
    }


def estimate_cd0(geometry_params: dict[str, Any]) -> float:
    """Return the Raymer wetted-area estimate of ``CD0``."""
    return float(estimate_cd0_breakdown(geometry_params)["CD0"])


def cd0_design_driver_table() -> pd.DataFrame:
    """Return the design variables that most strongly influence ``CD0``."""
    rows = [
        {
            "rank": 1,
            "variable": "Wing thickness-to-chord ratio t/c",
            "effect": "Strong",
            "why": "Raises both pressure/form-factor drag and wetted-area drag.",
        },
        {
            "rank": 2,
            "variable": "Fuselage fineness ratio l/d",
            "effect": "Strong",
            "why": "Controls fuselage form factor and pressure recovery.",
        },
        {
            "rank": 3,
            "variable": "Surface wetted area",
            "effect": "Direct",
            "why": "CD0 scales almost linearly with S_wet / S_ref.",
        },
        {
            "rank": 4,
            "variable": "Transition location / laminar fraction",
            "effect": "Large",
            "why": "Changes the skin-friction coefficient Cf.",
        },
        {
            "rank": 5,
            "variable": "Leading-edge radius",
            "effect": "Moderate",
            "why": "Affects pressure drag and the usable attached-flow range.",
        },
    ]
    return pd.DataFrame(rows)



def read_history_file(path: str | Path) -> pd.DataFrame:
    """Parse an OpenVSP ``.history`` file into a pandas DataFrame."""
    history_path = Path(path)
    if not history_path.exists():
        return pd.DataFrame()

    header: list[str] | None = None
    rows: list[dict[str, float]] = []

    with history_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if header is None:
                header = tokens
                continue
            if len(tokens) < len(header):
                continue
            try:
                rows.append({header[i]: float(tokens[i]) for i in range(len(header))})
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(rows)


def check_history_convergence(
    path: str | Path,
    *,
    min_iter: int = 3,
    keys: Iterable[str] = ("CLtot", "CMytot"),
) -> dict[str, Any]:
    """Return a compact convergence summary parsed from a ``.history`` file."""
    history_df = read_history_file(path)
    if history_df.empty:
        return {
            "converged": False,
            "reason": "history file not found or empty",
            "last_vals": {},
            "rel_var": {},
            "n_iter": 0,
        }

    n_iter = int(len(history_df))
    if n_iter < min_iter:
        return {
            "converged": False,
            "reason": f"insufficient iterations ({n_iter} < {min_iter})",
            "last_vals": {},
            "rel_var": {},
            "n_iter": n_iter,
        }

    numeric = history_df.select_dtypes(include=[np.number])
    has_bad_values = not np.isfinite(numeric.to_numpy()).all()
    last_row = history_df.iloc[-1]
    last_vals = {
        key: float(last_row[key])
        for key in keys
        if key in history_df.columns and pd.notna(last_row[key])
    }

    return {
        "converged": not has_bad_values,
        "reason": "OK" if not has_bad_values else "NaN/Inf detected in history",
        "last_vals": last_vals,
        "rel_var": {},
        "n_iter": n_iter,
    }


def parse_stab_file(path: str | Path) -> list[dict[str, float]]:
    """Parse all alpha blocks from an OpenVSP ``.stab`` file."""
    stab_path = Path(path)
    if not stab_path.exists():
        return []

    raw_text = stab_path.read_text(encoding="utf-8", errors="replace")
    blocks = [block.strip() for block in re.split(r"\*{10,}", raw_text) if block.strip()]
    results: list[dict[str, float]] = []

    for block in blocks:
        record: dict[str, float] = {}
        lines = block.splitlines()

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if len(tokens) < 2:
                continue
            key = STAB_SCALAR_KEYS.get(tokens[0])
            if key is None:
                continue
            try:
                record[key] = float(tokens[1])
            except ValueError:
                continue

        header_index = None
        header_tokens: list[str] = []
        for idx, line in enumerate(lines):
            if "Coef" in line and "Alpha" in line and "Beta" in line:
                header_index = idx
                header_tokens = line.split()
                break

        if header_index is not None:
            for line in lines[header_index + 1:]:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("TITLE"):
                    continue
                tokens = stripped.split()
                if len(tokens) < 3:
                    break
                coef_name = tokens[0]
                try:
                    record[coef_name] = float(tokens[1])
                except ValueError:
                    continue

                for col_index, column_name in enumerate(header_tokens[2:], start=2):
                    if col_index >= len(tokens):
                        break
                    suffix = STAB_SUFFIX_KEYS.get(column_name)
                    if suffix is None:
                        continue
                    try:
                        record[f"{coef_name}{suffix}"] = float(tokens[col_index])
                    except ValueError:
                        continue

        record.setdefault("SM", float("nan"))
        record.setdefault("X_np", float("nan"))
        if record:
            results.append(record)

    return results


def stability_records_to_dataframe(records: list[dict[str, float]]) -> pd.DataFrame:
    """Convert parsed stability records to a pandas DataFrame."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def find_generated_artifact(
    search_dirs: Iterable[str | Path],
    model_stem: str,
    suffix: str,
) -> Path | None:
    """
    Locate an OpenVSP-generated artifact.

    The solver sometimes writes files in the model directory and sometimes in the
    current working directory, so we search both exact names and newest matches.
    """
    candidates: list[Path] = []
    for directory in search_dirs:
        directory_path = Path(directory)
        if not directory_path.exists():
            continue
        exact = directory_path / f"{model_stem}{suffix}"
        if exact.exists():
            return exact
        candidates.extend(directory_path.glob(f"*{suffix}"))

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def first_finite_value(records: list[dict[str, float]], key: str) -> float:
    """Return the first finite value stored under ``key`` across stability records."""
    for record in records:
        value = record.get(key, float("nan"))
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return float("nan")
