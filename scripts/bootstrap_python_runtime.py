"""
Create a repository-local copy of a Windows Python runtime.

This script is used by ``scripts/run_project.ps1`` to seed an ``interpreter/``
folder inside the repository. The goal is not to invent a Python interpreter
from nothing, but to copy a compatible Python installation into the project so
that future runs depend less on machine-global state.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT_FILE_SUFFIXES = {
    ".dll",
    ".exe",
    ".pyd",
    ".txt",
    ".zip",
}
DIRECTORIES_TO_COPY = (
    "DLLs",
    "Include",
    "Lib",
    "Library",
    "libs",
    "tcl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-home", required=True, help="Source Python installation root")
    parser.add_argument("--target-home", required=True, help="Target repository-local runtime root")
    parser.add_argument("--source-python", default="", help="Optional source python executable")
    return parser.parse_args()


def copy_root_files(source_home: Path, target_home: Path) -> None:
    for item in source_home.iterdir():
        if item.is_dir():
            continue
        if item.suffix.lower() in ROOT_FILE_SUFFIXES:
            shutil.copy2(item, target_home / item.name)


def copy_directory(source_home: Path, target_home: Path, name: str) -> None:
    src_dir = source_home / name
    if not src_dir.exists():
        return

    dst_dir = target_home / name
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    if name == "Lib":
        ignore = shutil.ignore_patterns("site-packages", "__pycache__", "*.pyc", "*.pyo")

    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True, ignore=ignore)


def copy_runtime_dlls(source_home: Path, target_home: Path) -> None:
    """
    Copy Conda-style runtime DLLs next to ``python.exe``.

    Conda keeps many dependency DLLs in ``Library/bin`` instead of beside the
    interpreter executable. When we create a repository-local copy, placing
    those DLLs next to ``python.exe`` makes the copied runtime self-contained
    enough for stdlib extension modules such as ``pyexpat`` and ``_ssl``.
    """

    runtime_bin_dir = source_home / "Library" / "bin"
    if not runtime_bin_dir.exists():
        return

    for item in runtime_bin_dir.iterdir():
        if item.is_file() and item.suffix.lower() == ".dll":
            shutil.copy2(item, target_home / item.name)


def write_metadata(args: argparse.Namespace, target_home: Path) -> None:
    metadata = {
        "source_home": str(Path(args.source_home).resolve()),
        "source_python": args.source_python,
        "python_version": sys.version,
    }
    (target_home / "repo_runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_home = Path(args.source_home).expanduser().resolve()
    target_home = Path(args.target_home).expanduser().resolve()

    if not source_home.exists():
        raise SystemExit(f"Source Python home does not exist: {source_home}")
    if not (source_home / "python.exe").exists():
        raise SystemExit(f"Source Python home does not look valid: {source_home}")

    target_home.mkdir(parents=True, exist_ok=True)
    copy_root_files(source_home, target_home)
    copy_runtime_dlls(source_home, target_home)
    for directory_name in DIRECTORIES_TO_COPY:
        copy_directory(source_home, target_home, directory_name)

    site_packages_dir = target_home / "Lib" / "site-packages"
    site_packages_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args, target_home)

    print(f"Created repository-local Python runtime in: {target_home}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
