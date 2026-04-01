"""
Helpers for using the embedded OpenVSP distribution shipped with the project.

The repository already contains a Windows build of OpenVSP, so the user should
not need to download or install a separate copy.  This module wires the local
OpenVSP Python packages into ``sys.path`` and exposes a small amount of runtime
metadata, including the Python version required by the bundled binary module.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

OPENVSP_PACKAGE_DIRS = (
    "openvsp_config",
    "utilities",
    "degen_geom",
    "vsp_airfoils",
    "openvsp",
)
_PYTHON_DLL_RE = re.compile(rb"python(?P<major>\d)(?P<minor>\d{2})\.dll", re.IGNORECASE)
_DLL_HANDLES: list[object] = []
_CONFIGURED_ROOTS: set[Path] = set()


def get_repo_root() -> Path:
    """Return the repository root from the installed package location."""
    return Path(__file__).resolve().parents[1]


def get_default_openvsp_root() -> Path:
    """Return the bundled OpenVSP directory inside the repository."""
    return get_repo_root() / "OpenVSP-3.48.2-win64"


def resolve_openvsp_root(openvsp_root: str | Path | None = None) -> Path:
    """
    Resolve the OpenVSP root directory.

    Resolution order:
      1. Explicit ``openvsp_root`` argument
      2. ``OPENVSP_HOME`` environment variable
      3. The bundled repository copy
    """
    candidates: list[Path] = []
    if openvsp_root is not None:
        candidates.append(Path(openvsp_root))

    env_root = os.environ.get("OPENVSP_HOME")
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(get_default_openvsp_root())

    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved.exists():
            return resolved

    searched = "\n".join(f"  - {c.expanduser().resolve(strict=False)}" for c in candidates)
    raise FileNotFoundError(
        "OpenVSP directory not found. Checked:\n"
        f"{searched}"
    )


def get_openvsp_python_paths(openvsp_root: str | Path | None = None) -> list[Path]:
    """Return the Python package directories provided by the bundled OpenVSP copy."""
    root = resolve_openvsp_root(openvsp_root)
    python_root = root / "python"
    paths = [python_root / name for name in OPENVSP_PACKAGE_DIRS]
    return [path for path in paths if path.exists()]


def detect_supported_python_versions(openvsp_root: str | Path | None = None) -> list[tuple[int, int]]:
    """
    Inspect the bundled ``_vsp`` binaries and return the supported Python versions.

    The Windows module embeds a dependency on ``pythonXYZ.dll``; scanning for that
    string is a reliable way to discover the interpreter ABI required by the
    bundled OpenVSP build.
    """
    root = resolve_openvsp_root(openvsp_root)
    binary_dir = root / "python" / "openvsp" / "openvsp"
    versions: set[tuple[int, int]] = set()

    for pattern in ("_vsp*.pyd",):
        for binary_path in binary_dir.glob(pattern):
            data = binary_path.read_bytes()
            for match in _PYTHON_DLL_RE.finditer(data):
                major = int(match.group("major"))
                minor = int(match.group("minor"))
                versions.add((major, minor))

    return sorted(versions)


def format_supported_python_versions(versions: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> str:
    """Return a human-friendly string like ``Python 3.13`` or ``Python 3.11 / 3.13``."""
    if not versions:
        return "an unknown Python version"
    rendered = [f"{major}.{minor}" for major, minor in versions]
    if len(rendered) == 1:
        return f"Python {rendered[0]}"
    return "Python " + " / ".join(rendered)


def is_current_python_supported(openvsp_root: str | Path | None = None) -> bool:
    """Return True when the active interpreter matches the bundled OpenVSP build."""
    supported = detect_supported_python_versions(openvsp_root)
    return not supported or sys.version_info[:2] in supported


def configure_embedded_openvsp(openvsp_root: str | Path | None = None) -> Path:
    """
    Make the bundled OpenVSP importable in the current interpreter.

    This function is idempotent and safe to call more than once.
    """
    root = resolve_openvsp_root(openvsp_root)
    if root in _CONFIGURED_ROOTS:
        return root

    root_str = str(root)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    if root_str not in path_entries:
        os.environ["PATH"] = root_str + os.pathsep + current_path if current_path else root_str

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _DLL_HANDLES.append(os.add_dll_directory(root_str))

    for package_path in reversed(get_openvsp_python_paths(root)):
        package_str = str(package_path)
        if package_str not in sys.path:
            sys.path.insert(0, package_str)

    _CONFIGURED_ROOTS.add(root)
    return root
