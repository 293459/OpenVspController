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
