"""
Helpers for exporting baseline and sweep case collections.

The notebook now produces two kinds of outputs for every collection of cases:
  - machine-friendly ``.csv`` tables for later optimization and post-processing;
  - human-readable ``.txt`` snapshots for quick inspection outside Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


def _normalize_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    if metadata is None:
        return {}
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in metadata.items()
    }


def _frame_to_txt(df: pd.DataFrame) -> str:
    if df.empty:
        return "<empty table>\n"
    return df.to_string(index=False) + "\n"


def build_case_summary_row(
    case_name: str,
    results: "VSPAEROResults",
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a single summary row for one baseline/sweep/optimized case."""
    row = {
        "case_name": case_name,
        "alpha_points": len(results.alpha),
        "mach": results.mach,
        "re_cref": results.re_cref,
        "wake_iterations": results.wake_iterations,
        "wake_nodes": results.wake_nodes,
        "ld_max": results.LD_max,
        "cd_min": results.CD_min,
        "cd0": results.CD0_estimate,
        "cd0_estimate": results.CD0_estimate,
        "cd0_method": results.cd0_method,
        "cd0_fit_r_squared": results.cd0_fit_r_squared,
        "cl_alpha": results.CL_alpha,
        "static_margin": results.static_margin,
        "neutral_point_x": results.neutral_point_x,
        "converged": results.converged,
        "history_iterations": results.convergence.get("n_iter", 0) if results.convergence else 0,
        "has_history": bool(results.history_path),
        "has_stab": bool(results.stab_path),
        "stability_rows": len(results.stability_records),
        "history_path": str(results.history_path) if results.history_path else "",
        "polar_path": str(results.polar_path) if results.polar_path else "",
        "stab_path": str(results.stab_path) if results.stab_path else "",
        "model_path": str(results.model_path) if results.model_path else "",
        "solver_log_path": str(results.solver_log_path) if results.solver_log_path else "",
    }
    row.update(_normalize_metadata(metadata))
    return row


def collect_case_tables(
    cases: Mapping[str, "VSPAEROResults"],
    case_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert a case collection into summary, aerodynamic, and stability tables.

    Parameters
    ----------
    cases : mapping of case name to :class:`VSPAEROResults`
    case_metadata : optional mapping of case name to extra scalar metadata
    """
    summary_rows: list[dict[str, object]] = []
    aero_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []

    for case_name, results in cases.items():
        metadata = None if case_metadata is None else case_metadata.get(case_name)
        normalized_metadata = _normalize_metadata(metadata)

        summary_rows.append(build_case_summary_row(case_name, results, normalized_metadata))

        aero_df = results.to_dataframe().copy()
        if "case_name" not in aero_df.columns:
            aero_df.insert(0, "case_name", case_name)
        for key, value in normalized_metadata.items():
            aero_df[key] = value
        aero_frames.append(aero_df)

        stab_df = results.stability_dataframe().copy()
        if not stab_df.empty:
            stab_df.insert(0, "case_name", case_name)
            for key, value in normalized_metadata.items():
                stab_df[key] = value
            stability_frames.append(stab_df)

    summary_df = pd.DataFrame(summary_rows)
    aero_df = pd.concat(aero_frames, ignore_index=True) if aero_frames else pd.DataFrame()
    stability_df = (
        pd.concat(stability_frames, ignore_index=True) if stability_frames else pd.DataFrame()
    )
    return summary_df, aero_df, stability_df


def export_case_collection(
    cases: Mapping[str, "VSPAEROResults"],
    export_dir: str | Path,
    stem: str,
    case_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, Path]:
    """
    Export a case collection to paired ``.csv`` and ``.txt`` files.

    Three tables are written:
      - ``{stem}_summary`` for one row per case;
      - ``{stem}_aero`` for raw aerodynamic sweep points;
      - ``{stem}_stability`` for parsed ``.stab`` rows.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    summary_df, aero_df, stability_df = collect_case_tables(cases, case_metadata)
    tables = {
        f"{stem}_summary": summary_df,
        f"{stem}_aero": aero_df,
        f"{stem}_stability": stability_df,
    }

    written: dict[str, Path] = {}
    for table_name, df in tables.items():
        csv_path = export_dir / f"{table_name}.csv"
        txt_path = export_dir / f"{table_name}.txt"
        df.to_csv(csv_path, index=False)
        txt_path.write_text(_frame_to_txt(df), encoding="utf-8")
        written[f"{table_name}_csv"] = csv_path
        written[f"{table_name}_txt"] = txt_path

    return written


def print_case_collection_summary(
    cases: Mapping[str, "VSPAEROResults"],
    case_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Print and return the summary table for a case collection."""
    summary_df, _, _ = collect_case_tables(cases, case_metadata)
    if summary_df.empty:
        print("No cases to summarize.")
        return summary_df

    print(summary_df.to_string(index=False))
    return summary_df
