"""
Bootstrap pip into the current interpreter using stdlib-bundled wheels.

This avoids a couple of Windows-specific failures seen with ``python -m
ensurepip`` on stripped-down or policy-managed machines, while still keeping
the bootstrap fully offline.
"""

from __future__ import annotations

import functools
import os
import runpy
import sys
from pathlib import Path


def ensure_windows_folder_environment() -> None:
    """Populate shell-folder environment variables if Windows is missing them."""
    if os.name != "nt":
        return

    user_profile = Path(os.environ.get("USERPROFILE") or Path.home())
    program_data = Path(os.environ.get("ALLUSERSPROFILE") or os.environ.get("PROGRAMDATA") or "C:/ProgramData")
    defaults = {
        "USERPROFILE": user_profile,
        "APPDATA": user_profile / "AppData" / "Roaming",
        "LOCALAPPDATA": user_profile / "AppData" / "Local",
        "ALLUSERSPROFILE": program_data,
        "PROGRAMDATA": program_data,
    }

    for name, value in defaults.items():
        os.environ.setdefault(name, str(value))

    for name in ("APPDATA", "LOCALAPPDATA", "ALLUSERSPROFILE", "PROGRAMDATA"):
        try:
            Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def find_bundled_pip_wheel() -> Path:
    """Locate the pip wheel shipped inside the current interpreter."""
    import ensurepip

    bundled_dir = Path(ensurepip.__file__).resolve().parent / "_bundled"
    wheels = sorted(bundled_dir.glob("pip-*.whl"))
    if not wheels:
        raise RuntimeError(f"No bundled pip wheel found in: {bundled_dir}")
    return wheels[-1]


def patch_platformdirs_windows() -> None:
    """Prefer environment-variable lookups over Windows registry lookups."""
    from pip._vendor.platformdirs import windows as platformdirs_windows

    platformdirs_windows.get_win_folder = functools.lru_cache(maxsize=None)(
        platformdirs_windows.get_win_folder_from_env_vars
    )


def main() -> int:
    ensure_windows_folder_environment()

    wheel_path = find_bundled_pip_wheel()
    sys.path.insert(0, str(wheel_path))
    patch_platformdirs_windows()

    sys.argv[1:] = [
        "install",
        "--no-cache-dir",
        "--no-index",
        "--find-links",
        str(wheel_path.parent),
        "--upgrade",
        "pip",
    ]
    runpy.run_module("pip", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
