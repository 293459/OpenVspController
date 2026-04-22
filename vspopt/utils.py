"""
vspopt/utils.py
---------------
Shared utilities: logging configuration, input validation,
file integrity checks, and general helpers.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from vspopt.openvsp_runtime import configure_embedded_openvsp


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[str | Path] = None) -> None:
    """
    Configure the root logger for the vspopt package.

    Parameters
    ----------
    level    : Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
    log_file : Optional path to a .log file.  If given, output is written
               to both the console and the file.
    """
    root_logger = logging.getLogger("vspopt")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if not root_logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root_logger.addHandler(ch)

    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)


# ---------------------------------------------------------------------------
# File integrity
# ---------------------------------------------------------------------------

def check_vsp3_integrity(path: str | Path) -> tuple[bool, str]:
    """
    Basic integrity check for a .vsp3 file.

    Returns
    -------
    (ok, message) : bool indicating success, and a human-readable message.
    """
    p = Path(path)

    if not p.exists():
        return False, f"File not found: '{p.resolve()}'"

    if p.suffix.lower() != ".vsp3":
        return False, f"Expected .vsp3 extension, got '{p.suffix}'"

    if p.stat().st_size == 0:
        return False, f"File is empty: '{p}'"

    # VSP3 files are XML-based; minimal check for the XML header
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.read(256)
        if "<" not in header:
            return False, "File does not appear to be a valid XML/VSP3 file."
        if "Vsp_Geometry" not in header and "vsp" not in header.lower():
            return (
                False,
                "File does not contain expected VSP geometry markers. "
                "It may not be a valid .vsp3 file.",
            )
    except OSError as exc:
        return False, f"Cannot read file: {exc}"

    return True, f"OK (size={p.stat().st_size / 1024:.1f} KB)"


def file_md5(path: str | Path) -> str:
    """Return MD5 hex digest of a file (for change detection)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Array / results validation helpers
# ---------------------------------------------------------------------------

def check_polar_sanity(alpha: np.ndarray, CL: np.ndarray, CD: np.ndarray) -> list[str]:
    """
    Run a series of physical-plausibility checks on aerodynamic polar data.

    Returns a list of warning strings.  Empty list = all checks passed.
    """
    issues = []

    if len(alpha) < 3:
        issues.append(f"Too few data points ({len(alpha)}).  Increase AlphaNpts.")

    if len(CL) != len(alpha) or len(CD) != len(alpha):
        issues.append("Array length mismatch between alpha, CL, CD.")
        return issues  # Can't do further checks

    if not np.all(np.isfinite(CL)):
        n = int(np.sum(~np.isfinite(CL)))
        issues.append(f"{n} non-finite CL values found.")

    if not np.all(np.isfinite(CD)):
        n = int(np.sum(~np.isfinite(CD)))
        issues.append(f"{n} non-finite CD values found.")

    if np.any(CD < 0):
        issues.append("Negative CD values detected - physically impossible.")

    if np.max(np.abs(CL)) > 4.0:
        issues.append(f"|CL| > 4 detected (max={np.max(np.abs(CL)):.2f}).  "
                      "Check reference area.")

    with np.errstate(divide="ignore", invalid="ignore"):
        LD = np.where(CD > 0, CL / CD, 0.0)
    if np.nanmax(np.abs(LD)) > 100:
        issues.append(f"L/D > 100 detected.  Check reference area Sref.")

    # Check that CL is approximately monotone over the linear range
    mask_linear = (alpha >= -5) & (alpha <= 10)
    if mask_linear.sum() >= 3:
        dCL = np.diff(CL[mask_linear])
        if np.any(dCL < -0.05):
            issues.append(
                "CL is non-monotone in the linear range (-5 deg to 10 deg).  "
                "May indicate a mesh or wake iteration issue."
            )

    return issues


# ---------------------------------------------------------------------------
# OpenVSP version check
# ---------------------------------------------------------------------------

def check_openvsp_version(min_version: tuple[int, int, int] = (3, 35, 0)) -> tuple[bool, str]:
    """
    Verify that the installed OpenVSP version meets the minimum requirement.

    Attempts to determine the version from:
      1. openvsp.GetVersionString() if available and valid
      2. The bundled OpenVSP-X.Y.Z-win64 folder name when GetVersionString() returns invalid/0.0.0
      3. Falls back to "0.0.0" only when all methods fail

    Returns
    -------
    (ok, message) : bool and diagnostic string
    """
    try:
        configure_embedded_openvsp()
        import openvsp as vsp
    except ImportError:
        return False, "openvsp is not importable."

    version_str = None
    version_source = None

    # 1. Try GetVersionString() first
    if hasattr(vsp, "GetVersionString"):
        try:
            api_version = vsp.GetVersionString()
            if api_version and api_version.strip() and api_version != "0.0.0":
                version_str = api_version.strip()
                version_source = "API (GetVersionString)"
        except Exception:
            pass

    # 2. If GetVersionString() didn't provide a valid version, extract from bundled folder name
    if not version_str:
        try:
            from vspopt.openvsp_runtime import get_default_openvsp_root
            root = get_default_openvsp_root()
            folder_name = root.name  # e.g., "OpenVSP-3.48.2-win64"
            if "OpenVSP-" in folder_name:
                # Extract version: "OpenVSP-3.48.2-win64" → "3.48.2"
                version_part = folder_name.split("-")[1]
                # Handle versions like "3.48.2" or "3.48.2-win64"
                version_str = version_part.split("-")[0]
                version_source = f"bundled folder name ({folder_name})"
        except Exception:
            pass

    # 3. Last resort: use placeholder version
    if not version_str:
        version_str = "0.0.0"
        version_source = "unknown (fallback)"

    try:
        parts = tuple(int(x) for x in version_str.split(".")[:3])
    except ValueError:
        return (
            True,
            f"OpenVSP version string '{version_str}' (from {version_source}) could not be parsed - proceeding anyway.",
        )

    min_version_str = ".".join(str(x) for x in min_version)
    if parts >= min_version:
        return (
            True,
            f"OpenVSP {version_str} (from {version_source}) meets requirement >={min_version_str}",
        )
    else:
        return (
            False,
            f"OpenVSP {version_str} (from {version_source}) is older than required {min_version_str}.",
        )


# ---------------------------------------------------------------------------
# Design variable helpers
# ---------------------------------------------------------------------------

def normalize(x: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    """
    Normalize design variables to [0, 1].

    x_norm = (x - lb) / (ub - lb)
    """
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    return (x - lb) / (ub - lb + 1e-12)


def denormalize(x_norm: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    """Inverse of ``normalize``."""
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    return lb + x_norm * (ub - lb)


# ---------------------------------------------------------------------------
# Formatting helpers for notebooks
# ---------------------------------------------------------------------------

def results_to_markdown_table(performance_series: "pd.Series") -> str:
    """
    Convert a VSPAEROResults.performance_summary() Series to a
    Markdown table string suitable for display in Jupyter.
    """
    rows = ["| Metric | Value |", "|--------|-------|"]
    for idx, val in performance_series.items():
        if isinstance(val, float):
            rows.append(f"| {idx} | {val:.4f} |")
        else:
            rows.append(f"| {idx} | {val} |")
    return "\n".join(rows)


def print_banner(title: str, width: int = 60) -> None:
    """Print a styled banner to stdout (used in notebook cells)."""
    print("\n" + "=" * width)
    pad = (width - len(title) - 2) // 2
    print(" " * pad + " " + title)
    print("=" * width + "\n")
