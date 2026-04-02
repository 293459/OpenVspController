"""
Low-level wrapper around the OpenVSP Python API (openvsp).

Design philosophy
-----------------
The raw OpenVSP API requires callers to:
  1. Look up a geometry ID by name
  2. Look up a parameter ID by (geom_id, parm_name, group_name)
  3. Call SetParmVal(parm_id, value)

This wrapper collapses those three steps into single, readable calls
and adds validation, logging, and error context at every boundary.

All public methods raise ``OpenVSPError`` (a subclass of RuntimeError)
on failure so callers can catch a single exception type.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from vspopt.openvsp_runtime import (
    configure_embedded_openvsp,
    detect_supported_python_versions,
    format_supported_python_versions,
)

logger = logging.getLogger(__name__)


def _import_vsp():
    """Import openvsp and return the module, with a helpful error on failure."""
    try:
        root = configure_embedded_openvsp()
        import openvsp as vsp

        return vsp
    except ImportError as exc:
        supported = detect_supported_python_versions()
        supported_msg = format_supported_python_versions(supported)
        current_msg = f"Python {os.sys.version_info[0]}.{os.sys.version_info[1]}"
        raise ImportError(
            "Cannot import 'openvsp'. Make sure:\n"
            f"  1. You are running a compatible interpreter ({supported_msg}).\n"
            "  2. The bundled OpenVSP copy is available at:\n"
            f"     {root}\n"
            "  3. The active interpreter is:\n"
            f"     {current_msg}\n"
            "  Run scripts/verify_setup.py for a detailed diagnosis."
        ) from exc


class OpenVSPError(RuntimeError):
    """Raised when an OpenVSP API call fails or returns unexpected data."""


class VSPWrapper:
    """
    Thin, validated wrapper around the OpenVSP Python API.

    Parameters
    ----------
    vsp3_path : str | Path
        Path to an existing .vsp3 model file.

    Examples
    --------
    >>> wrap = VSPWrapper("models/my_aircraft.vsp3")
    >>> print(wrap.geom_names)
    ['Wing', 'Fuselage', 'HTP', 'VTP']
    >>> wrap.set_param("Wing", "Span", "XSec_1", 14.0)
    >>> results = wrap.run_vspaero_sweep(alpha_start=-5, alpha_end=20, alpha_npts=26)
    """

    def __init__(self, vsp3_path: str | Path) -> None:
        self._vsp = _import_vsp()
        self._path = self._validate_vsp3_path(vsp3_path)
        self._loaded = False
        self._geom_id_cache: dict[str, str] = {}
        self._parm_id_cache: dict[tuple[str, str, str], str] = {}
        self.warnings: list[str] = []
        self._duplicate_names_found = False

    def load(self) -> "VSPWrapper":
        """
        Load the .vsp3 model into memory. Safe to call multiple times.

        Returns
        -------
        self : VSPWrapper
            Enables method chaining: ``wrap.load().run_vspaero_sweep(...)``
        """
        vsp = self._vsp
        logger.info("Loading VSP model: %s", self._path)

        vsp.ClearVSPModel()
        self._geom_id_cache.clear()
        self._parm_id_cache.clear()

        vsp.ReadVSPFile(str(self._path))
        self._loaded = True

        all_geoms = vsp.FindGeoms()
        for geom_id in all_geoms:
            name = vsp.GetGeomName(geom_id)
            if name in self._geom_id_cache:
                # Handle duplicate names by appending index
                suffix = 2
                new_name = f"{name}_{suffix}"
                while new_name in self._geom_id_cache:
                    suffix += 1
                    new_name = f"{name}_{suffix}"
                logger.warning(
                    "Duplicate geometry name '%s' detected. Renaming to '%s' in Python cache.",
                    name,
                    new_name,
                )
                self._geom_id_cache[new_name] = geom_id
                self.warnings.append(
                    f"Duplicate component name '{name}' found in model; cached as '{new_name}'.\n"
                    f"To avoid confusion, consider renaming the duplicate to a unique name in your VSP3 file."
                )
            else:
                self._geom_id_cache[name] = geom_id

        geom_count = len(self._geom_id_cache)
        actual_count = len(all_geoms)
        logger.info("Model loaded: %d geometry components found (%d in VSP, %d unique in cache).",
                    geom_count, actual_count, geom_count)

        if geom_count == 0:
            raise OpenVSPError(
                f"The model at '{self._path}' loaded but contains no geometry components. "
                "The file may be empty or corrupted."
            )

        if actual_count != geom_count:
            msg = (
                f"WARNING: Model has {actual_count} components in VSP but only {geom_count} unique names. "
                f"{actual_count - geom_count} duplicate name(s) detected and renamed."
            )
            logger.warning(msg)
            self._duplicate_names_found = True
        else:
            self._duplicate_names_found = False

        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise OpenVSPError("Model is not loaded. Call VSPWrapper.load() before analysis.")

    @property
    def geom_names(self) -> list[str]:
        """Return names of all geometry components in the model."""
        self._ensure_loaded()
        return list(self._geom_id_cache.keys())

    def get_geom_id(self, name: str) -> str:
        """
        Return the internal geometry ID for a component by name.

        Raises ``OpenVSPError`` with a helpful message listing available
        names if the requested name is not found.
        """
        self._ensure_loaded()
        if name not in self._geom_id_cache:
            available = ", ".join(f"'{candidate}'" for candidate in self._geom_id_cache)
            raise OpenVSPError(
                f"Geometry component '{name}' not found in model.\n"
                f"Available components: {available}"
            )
        return self._geom_id_cache[name]

    def get_geom_type(self, name: str) -> str:
        """Return the geometry type string (for example 'Wing' or 'Fuselage')."""
        geom_id = self.get_geom_id(name)
        return self._vsp.GetGeomTypeName(geom_id)

    def list_parm_groups(self, geom_name: str) -> list[str]:
        """Return all parameter group names for a given geometry component."""
        geom_id = self.get_geom_id(geom_name)
        return list(self._vsp.GetGeomParmIDs(geom_id))

    def get_all_params(self, geom_name: str) -> dict[str, float]:
        """
        Return a dictionary of {group_name/parm_name: current_value}.

        Useful for exploratory inspection of a new model.
        """
        geom_id = self.get_geom_id(geom_name)
        vsp = self._vsp
        result: dict[str, float] = {}

        for parm_id in vsp.GetGeomParmIDs(geom_id):
            parm_name = vsp.GetParmName(parm_id)
            group_name = vsp.GetParmGroupName(parm_id)
            try:
                value = vsp.GetParmVal(parm_id)
                result[f"{group_name}/{parm_name}"] = value
            except Exception:
                continue

        return result

    def get_param(self, geom_name: str, parm_name: str, group_name: str) -> float:
        """
        Get the current value of a named parameter.

        Parameters
        ----------
        geom_name  : Name of the geometry component (for example "Wing")
        parm_name  : OpenVSP parameter name (for example "Span")
        group_name : OpenVSP group name (for example "WingGeom" or "XSec_1")
        """
        parm_id = self._resolve_parm_id(geom_name, parm_name, group_name)
        return self._vsp.GetParmVal(parm_id)

    def set_param(
        self,
        geom_name: str,
        parm_name: str,
        group_name: str,
        value: float,
        *,
        clamp: bool = True,
    ) -> "VSPWrapper":
        """
        Set the value of a named parameter.

        Parameters
        ----------
        geom_name  : Name of the geometry component.
        parm_name  : OpenVSP parameter name.
        group_name : OpenVSP group name.
        value      : New value to assign.
        clamp      : If True, silently clamp to the parameter bounds.
        """
        parm_id = self._resolve_parm_id(geom_name, parm_name, group_name)
        vsp = self._vsp

        lower = vsp.GetParmLowerLimit(parm_id)
        upper = vsp.GetParmUpperLimit(parm_id)

        if not (lower <= value <= upper):
            if clamp:
                clamped = max(lower, min(upper, value))
                logger.warning(
                    "Parameter %s/%s/%s: requested value %.4f clamped to [%.4f, %.4f] -> %.4f",
                    geom_name,
                    group_name,
                    parm_name,
                    value,
                    lower,
                    upper,
                    clamped,
                )
                value = clamped
            else:
                raise OpenVSPError(
                    f"Parameter {geom_name}/{group_name}/{parm_name}: "
                    f"value {value:.4f} is outside bounds [{lower:.4f}, {upper:.4f}]."
                )

        vsp.SetParmVal(parm_id, value)
        logger.debug("Set %s/%s/%s = %.6f", geom_name, group_name, parm_name, value)
        return self

    def set_params(self, param_dict: dict[tuple[str, str, str], float], **kwargs) -> "VSPWrapper":
        """
        Set multiple parameters at once.

        Parameters
        ----------
        param_dict : dict mapping (geom_name, parm_name, group_name) -> value
        """
        for (geom_name, parm_name, group_name), value in param_dict.items():
            self.set_param(geom_name, parm_name, group_name, value, **kwargs)
        return self

    def _resolve_parm_id(self, geom_name: str, parm_name: str, group_name: str) -> str:
        """Resolve and cache parameter IDs."""
        self._ensure_loaded()
        cache_key = (geom_name, parm_name, group_name)
        if cache_key not in self._parm_id_cache:
            geom_id = self.get_geom_id(geom_name)
            parm_id = self._vsp.GetParm(geom_id, parm_name, group_name)
            if not parm_id:
                raise OpenVSPError(
                    f"Parameter not found: geom='{geom_name}', "
                    f"parm='{parm_name}', group='{group_name}'.\n"
                    f"Use get_all_params('{geom_name}') to list available parameters."
                )
            self._parm_id_cache[cache_key] = parm_id
        return self._parm_id_cache[cache_key]

    def get_reference_quantities(self) -> dict[str, float]:
        """
        Return the VSPAERO reference quantities currently set in the model:
        Sref (m^2), bref (m), cref (m).
        """
        vsp = self._vsp
        self._ensure_loaded()

        refs: dict[str, float] = {}
        vsp.SetAnalysisInputDefaults("VSPAEROSweep")
        refs["Sref"] = vsp.GetDoubleAnalysisInput("VSPAEROSweep", "Sref", 0)[0]
        refs["bref"] = vsp.GetDoubleAnalysisInput("VSPAEROSweep", "bref", 0)[0]
        refs["cref"] = vsp.GetDoubleAnalysisInput("VSPAEROSweep", "cref", 0)[0]
        return refs
    
    
    def run_mass_properties(
        self,
        *,
        num_slices: int = 100,
        working_dir: str | Path | None = None,
    ) -> "MassProperties":
        """
        Run the OpenVSP ``MassProp`` analysis and return structured mass data.
        """
        from vspopt.postprocess import MassProperties

        self._ensure_loaded()
        vsp = self._vsp

        if working_dir is None:
            working_dir = self._path.parent
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        # 1. Setup and Run Analysis (Modern Method)
        vsp.SetAnalysisInputDefaults("MassProp")
        vsp.SetIntAnalysisInput("MassProp", "NumMassSlice", [int(num_slices)])

        prev_dir = os.getcwd()
        try:
            os.chdir(str(working_dir))
            res_id = vsp.ExecAnalysis("MassProp")
        finally:
            os.chdir(prev_dir)

        if not res_id:
            raise OpenVSPError("MassProp returned an empty result ID.")

        # ... (setup code remains the same) ...

        # 2. Extract Data
        data_names = vsp.GetAllDataNames(res_id)
        raw_results = {}
        
        for name in data_names:
            rtype = vsp.GetResultsType(res_id, name)
            
            if rtype == vsp.DOUBLE_DATA or rtype == 2:
                raw_results[name] = list(vsp.GetDoubleResults(res_id, name))
                
            elif rtype == vsp.VEC3D_DATA or rtype == 4:
                vecs = vsp.GetVec3dResults(res_id, name)
                raw_results[name] = [(v.x(), v.y(), v.z()) for v in vecs]
                
            elif rtype == vsp.DOUBLE_MATRIX_DATA or rtype == 5:
                mat = vsp.GetDoubleMatResults(res_id, name)
                raw_results[name] = [list(row) for row in mat]
                
            # --- IMPROVEMENT: Better Fallback for Strings and Ints ---
            elif rtype == vsp.STRING_DATA or rtype == 3:
                raw_results[name] = list(vsp.GetStringResults(res_id, name))
                
            elif rtype == vsp.INT_DATA or rtype == 1:
                raw_results[name] = list(vsp.GetIntResults(res_id, name))
                
            else:
                raw_results[name] = None
                
        # --- IMPROVEMENT: Professional Logging ---
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"MassProp completed: extracted {len(raw_results)} result fields (including Total_CG).")

        return MassProperties.from_results(raw_results)
    
    def run_vspaero_sweep(
        self,
        *,
        alpha_start: float = -5.0,
        alpha_end: float = 20.0,
        alpha_npts: int = 26,
        mach: float = 0.2,
        re_cref: float = 1e6,
        wake_iter: int = 5,
        analysis_method: int = 0,  # 0 = VLM, 1 = Panel
        beta: float = 0.0,
        working_dir: str | Path | None = None,
        parse_history: bool = True,
        parse_stability: bool = True,
        min_convergence_iterations: int = 3,
        use_massprop_cg: bool = False,
        massprop_num_slices: int = 100,
    ) -> "VSPAEROResults":
        """
        
        Run a VSPAERO alpha sweep and return structured results.

        Parameters
        ----------
        alpha_start    : First angle of attack (degrees).
        alpha_end      : Last angle of attack (degrees).
        alpha_npts     : Number of evenly-spaced alpha points.
        mach           : Freestream Mach number.
        re_cref        : Reynolds number based on the reference chord.
        wake_iter      : Number of wake relaxation iterations.
        analysis_method: 0 for Vortex Lattice (fast), 1 for Panel (slow).
        beta           : Sideslip angle (degrees), default 0.
        working_dir    : Directory where VSPAERO writes output files.
        parse_history  : Parse the generated ``.history`` file when available.
        parse_stability: Parse the generated ``.stab`` file when available.
        min_convergence_iterations:
                         Minimum iterations required to mark history as usable.
        use_massprop_cg: Run ``MassProp`` first and feed the computed CG into
                         VSPAERO as Xcg/Ycg/Zcg.
        massprop_num_slices:
                         Number of slices used by the optional ``MassProp`` run.

        Returns
        -------
        VSPAEROResults
            Dataclass holding alpha, CL, CD, CM arrays and metadata.
        """
        from vspopt.postprocess import (
            check_history_convergence,
            find_generated_artifact,
            parse_stab_file,
            read_history_file,
            stability_records_to_dataframe,
        )
        from vspopt.vspaero import _parse_results_manager

        self._ensure_loaded()
        vsp = self._vsp

        if working_dir is None:
            working_dir = self._path.parent
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        reference_quantities = self.get_reference_quantities()
        mass_properties = None

        logger.info(
            "Running VSPAEROSweep: alpha=[%.1f deg, %.1f deg] x %d pts, M=%.2f, Re=%.2e",
            alpha_start,
            alpha_end,
            alpha_npts,
            mach,
            re_cref,
        )

        analysis = "VSPAEROSweep"
        vsp.SetAnalysisInputDefaults(analysis)

        vsp.SetIntAnalysisInput(analysis, "AnalysisMethod", [analysis_method])
        vsp.SetDoubleAnalysisInput(analysis, "AlphaStart", [float(alpha_start)])
        vsp.SetDoubleAnalysisInput(analysis, "AlphaEnd", [float(alpha_end)])
        vsp.SetIntAnalysisInput(analysis, "AlphaNpts", [int(alpha_npts)])

        vsp.SetDoubleAnalysisInput(analysis, "MachStart", [float(mach)])
        vsp.SetDoubleAnalysisInput(analysis, "MachEnd", [float(mach)])
        vsp.SetIntAnalysisInput(analysis, "MachNpts", [1])

        vsp.SetDoubleAnalysisInput(analysis, "BetaStart", [float(beta)])
        vsp.SetDoubleAnalysisInput(analysis, "BetaEnd", [float(beta)])
        vsp.SetIntAnalysisInput(analysis, "BetaNpts", [1])

        vsp.SetDoubleAnalysisInput(analysis, "ReCref", [float(re_cref)])
        vsp.SetIntAnalysisInput(analysis, "WakeNumIter", [int(wake_iter)])

        if use_massprop_cg:
            mass_properties = self.run_mass_properties(
                num_slices=massprop_num_slices,
                working_dir=working_dir,
                
            )
            if mass_properties.cg_is_finite:
                vsp.SetDoubleAnalysisInput(analysis, "Xcg", [float(mass_properties.xcg)])
                vsp.SetDoubleAnalysisInput(analysis, "Ycg", [float(mass_properties.ycg)])
                vsp.SetDoubleAnalysisInput(analysis, "Zcg", [float(mass_properties.zcg)])
            else:
                logger.warning(
                    "MassProp completed but did not produce a finite CG. "
                    "Proceeding without overriding Xcg/Ycg/Zcg."
                )

        prev_dir = os.getcwd()
        try:
            os.chdir(str(working_dir))
            res_id = vsp.ExecAnalysis(analysis)
        finally:
            os.chdir(prev_dir)

        if not res_id:
            raise OpenVSPError(
                "VSPAEROSweep returned an empty result ID. "
                "This usually means VSPAERO crashed. "
                "Check that the OpenVSP installation is complete and "
                "that vspaero.exe is present in the OpenVSP directory."
            )

        results = _parse_results_manager(vsp, res_id, mach, re_cref, alpha_npts)
        results.Sref = float(reference_quantities.get("Sref", 0.0))
        results.bref = float(reference_quantities.get("bref", 0.0))
        results.cref = float(reference_quantities.get("cref", 0.0))
        results.mass_properties = mass_properties

        search_dirs = [working_dir, self._path.parent]
        model_stem = self._path.stem

        if parse_history:
            history_path = find_generated_artifact(search_dirs, model_stem, ".history")
            if history_path is not None:
                results.history_path = history_path
                results.history_table = read_history_file(history_path)
                results.convergence = check_history_convergence(
                    history_path,
                    min_iter=min_convergence_iterations,
                )

        if parse_stability:
            stab_path = find_generated_artifact(search_dirs, model_stem, ".stab")
            if stab_path is not None:
                stability_records = parse_stab_file(stab_path)
                results.stab_path = stab_path
                results.stability_records = stability_records
                results.stability_table = stability_records_to_dataframe(stability_records)

        return results

    def update_and_run(
        self,
        param_dict: dict[tuple[str, str, str], float],
        sweep_kwargs: dict | None = None,
    ) -> "VSPAEROResults":
        """
        Convenience method: set a batch of parameters and immediately
        run a VSPAERO sweep. Used by the optimizer callback.
        """
        sweep_kwargs = sweep_kwargs or {}
        self.set_params(param_dict)
        return self.run_vspaero_sweep(**sweep_kwargs)

    def __enter__(self) -> "VSPWrapper":
        return self.load()

    def __exit__(self, *_) -> None:
        try:
            self._vsp.ClearVSPModel()
        except Exception:
            pass
        self._loaded = False

    @staticmethod
    def _validate_vsp3_path(path: str | Path) -> Path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"VSP3 model not found: '{p.resolve()}'\n"
                "Place your .vsp3 file in the models/ directory and update the path."
            )
        if p.suffix.lower() != ".vsp3":
            raise ValueError(
                f"Expected a .vsp3 file, got '{p.suffix}'. "
                "Make sure you are pointing to an OpenVSP model file."
            )
        if p.stat().st_size == 0:
            raise OpenVSPError(f"The model file '{p}' exists but is empty.")
        return p.resolve()

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "not loaded"
        return f"VSPWrapper(path='{self._path.name}', state={state})"
