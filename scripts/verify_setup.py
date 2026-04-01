"""
Run this script to confirm that the local project environment is ready.

Usage:
    python scripts/verify_setup.py

The check is intentionally self-contained:
  - it loads the bundled OpenVSP runtime directly from the repository
  - it verifies the Python ABI required by the embedded OpenVSP binary
  - it reports missing third-party packages needed by notebook and app flows
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = PROJECT_ROOT / "vspopt" / "openvsp_runtime.py"
OPENVSP_RELATIVE_PATH = "OpenVSP-3.48.2-win64"
LOCAL_LIBS_DIR = PROJECT_ROOT / "libs"
LOCAL_INTERPRETER = PROJECT_ROOT / "interpreter" / "python.exe"
REQUIRED_PACKAGES = {
    "numpy": "run_project.bat setup",
    "pandas": "run_project.bat setup",
    "matplotlib": "run_project.bat setup",
    "plotly": "run_project.bat setup",
    "scipy": "run_project.bat setup",
    "optuna": "run_project.bat setup",
    "scikit-optimize": "run_project.bat setup",
    "ipywidgets": "run_project.bat setup",
    "jupyterlab": "run_project.bat setup",
    "ipykernel": "run_project.bat setup",
    "streamlit": "run_project.bat setup",
    "rich": "run_project.bat setup",
}


def configure_repo_paths():
    """Prefer repository-local code and packages when they exist."""
    repo_entries = [PROJECT_ROOT, LOCAL_LIBS_DIR]
    for entry in reversed(repo_entries):
        if entry.exists():
            entry_str = str(entry)
            if entry_str not in sys.path:
                sys.path.insert(0, entry_str)


def load_runtime_module():
    """Load ``vspopt/openvsp_runtime.py`` without importing the whole package."""
    spec = importlib.util.spec_from_file_location("openvsp_runtime_standalone", RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runtime helpers from: {RUNTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


configure_repo_paths()
RUNTIME = load_runtime_module()
EMBEDDED_OPENVSP_ROOT = PROJECT_ROOT / OPENVSP_RELATIVE_PATH
try:
    SUPPORTED_PYTHON_VERSIONS = tuple(RUNTIME.detect_supported_python_versions())
except Exception:
    SUPPORTED_PYTHON_VERSIONS = tuple()


def check_python_version():
    """Validate the current interpreter against the bundled OpenVSP binary."""
    major, minor = sys.version_info[:2]

    if not SUPPORTED_PYTHON_VERSIONS:
        return (
            "WARN",
            f"Python {major}.{minor}.{sys.version_info[2]}",
            "Could not detect a required Python ABI from the embedded OpenVSP binary.",
        )

    ok = (major, minor) in SUPPORTED_PYTHON_VERSIONS
    status = "OK" if ok else "FAIL"
    supported = RUNTIME.format_supported_python_versions(list(SUPPORTED_PYTHON_VERSIONS))
    note = ""
    if not ok:
        note = (
            f"Embedded OpenVSP requires {supported}.\n"
            f"You are running Python {major}.{minor}.\n"
            "Create the local virtual environment with the matching interpreter first."
        )
    return status, f"Python {major}.{minor}.{sys.version_info[2]}", note


def check_embedded_openvsp():
    """Verify that the bundled OpenVSP directory exists in the repository."""
    if EMBEDDED_OPENVSP_ROOT.exists():
        return "OK", str(EMBEDDED_OPENVSP_ROOT), ""
    return (
        "FAIL",
        OPENVSP_RELATIVE_PATH,
        f"Bundled OpenVSP directory not found at:\n{EMBEDDED_OPENVSP_ROOT}",
    )


def check_local_interpreter():
    """Verify that the repository-local interpreter exists and is compatible."""
    if not LOCAL_INTERPRETER.exists():
        return (
            "WARN",
            str(LOCAL_INTERPRETER),
            "Repository-local interpreter not created yet. Run run_project.bat setup.",
        )

    try:
        completed = subprocess.run(
            [str(LOCAL_INTERPRETER), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"],
            capture_output=True,
            text=True,
            check=False,
        )
        version = completed.stdout.strip() or "unknown"
        if completed.returncode != 0:
            return "WARN", f"{LOCAL_INTERPRETER} ({version})", completed.stderr.strip() or "Unable to execute the local interpreter."
        return "OK", f"{LOCAL_INTERPRETER} ({version})", ""
    except OSError as exc:
        return "WARN", str(LOCAL_INTERPRETER), str(exc)


def check_local_libs():
    """Verify that the repository-local libs directory exists."""
    if not LOCAL_LIBS_DIR.exists():
        return "WARN", str(LOCAL_LIBS_DIR), "Repository-local libs folder not created yet. Run run_project.bat setup."

    has_packages = any(path.name != ".gitkeep" for path in LOCAL_LIBS_DIR.iterdir())
    if has_packages:
        return "OK", str(LOCAL_LIBS_DIR), ""
    return "WARN", str(LOCAL_LIBS_DIR), "The libs folder exists but does not contain installed packages yet."


def check_package(pkg_name, install_hint):
    """Import a package and report its detected version."""
    import_name = "skopt" if pkg_name == "scikit-optimize" else pkg_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown version")
        return "OK", f"{pkg_name} ({version})", ""
    except ImportError:
        return "FAIL", pkg_name, f"Install with: {install_hint}"


def check_openvsp_api():
    """
    Beyond importing openvsp, verify that VSPAERO analysis is registered.

    This catches cases where the module can be found but the runtime is still
    not usable because of an incompatible interpreter or missing DLLs.
    """
    supported = RUNTIME.format_supported_python_versions(list(SUPPORTED_PYTHON_VERSIONS))
    try:
        RUNTIME.configure_embedded_openvsp()
        import openvsp as vsp

        analyses = vsp.ListAnalysis()
        if "VSPAEROSweep" not in analyses:
            return (
                "WARN",
                "openvsp (VSPAEROSweep not found)",
                "The module imports, but VSPAEROSweep is not registered.",
            )
        version = vsp.GetVersionString() if hasattr(vsp, "GetVersionString") else "unknown"
        return "OK", f"openvsp API ({version})", ""
    except ImportError as exc:
        return (
            "FAIL",
            "openvsp",
            "Cannot import the embedded OpenVSP API.\n"
            f"Expected interpreter ABI: {supported}\n"
            f"Import error: {exc}",
        )
    except Exception as exc:
        return "WARN", "openvsp", str(exc)


def main():
    """Run the environment report."""
    has_failures = False

    try:
        from rich import box as rich_box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="OpenVSP Controller - Environment Check", box=rich_box.ROUNDED, show_lines=True)
        table.add_column("Status", style="bold", width=8)
        table.add_column("Component", width=44)
        table.add_column("Notes")

        def add_row(status, component, note):
            nonlocal has_failures
            if status == "FAIL":
                has_failures = True
            colour = {
                "OK": "green",
                "INFO": "cyan",
                "WARN": "yellow",
                "FAIL": "red",
            }.get(status, "white")
            table.add_row(f"[{colour}]{status}[/{colour}]", component, note.strip())

    except ImportError:
        console = None
        table = None

        def add_row(status, component, note):  # type: ignore[redefinition]
            nonlocal has_failures
            if status == "FAIL":
                has_failures = True
            print(f"[{status}] {component}")
            if note.strip():
                print(note)

        print("\n=== OpenVSP Controller - Environment Check ===\n")

    status, label, note = check_python_version()
    add_row(status, label, note)

    add_row("INFO", f"Platform: {platform.system()} {platform.machine()}", "")
    add_row("INFO", f"Project root: {PROJECT_ROOT}", "")

    status, label, note = check_local_interpreter()
    add_row(status, label, note)

    status, label, note = check_local_libs()
    add_row(status, label, note)

    status, label, note = check_embedded_openvsp()
    add_row(status, label, note)

    status, label, note = check_openvsp_api()
    add_row(status, label, note)

    for package_name, hint in REQUIRED_PACKAGES.items():
        status, label, note = check_package(package_name, hint)
        add_row(status, label, note)

    if console and table:
        console.print(table)
    else:
        print("\n=============================================\n")

    print(
        "\nIf any items show FAIL, fix them before running the notebook or app.\n"
        "Recommended entry point: run_project.bat\n"
    )

    raise SystemExit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
