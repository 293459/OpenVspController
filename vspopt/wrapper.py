"""
Low-level wrapper around the OpenVSP Python API (``openvsp``).

The raw API is powerful, but it is also easy to misuse:
  1. some parameters share the same raw group name across multiple sections;
  2. analysis input names differ across OpenVSP versions;
  3. VSPAERO artifacts are written using the active ``.vsp3`` filename, which
     makes baseline and sweep cases overwrite each other unless we isolate them.

This wrapper keeps those details in one place and exposes a small, validated
surface for the rest of the project.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vspopt.openvsp_runtime import (
    configure_embedded_openvsp,
    detect_supported_python_versions,
    format_supported_python_versions,
)

logger = logging.getLogger(__name__)


def _import_vsp():
    """Import ``openvsp`` and return the module, with a helpful error on failure."""
    try:
        root = configure_embedded_openvsp()
        import openvsp as vsp

        return vsp
    except ImportError as exc:
        supported = detect_supported_python_versions()
        supported_msg = format_supported_python_versions(supported)
        current_msg = f"Python {sys.version_info[0]}.{sys.version_info[1]}"
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
        Path to an existing ``.vsp3`` model file.
    """

    # OpenVSP reserves low set indices for built-in sets. User-defined sets
    # start at SET_FIRST_USER, which is 3 in the bundled 3.48.2 build.
    DEFAULT_THIN_SET = 3
    DEFAULT_THICK_SET = 4
    MIN_WAKE_ITERATIONS = 3
    MIN_WAKE_NODES = 3

    def __init__(self, vsp3_path: str | Path) -> None:
        self._vsp = _import_vsp()
        self._path = self._validate_vsp3_path(vsp3_path)
        self._loaded = False
        self._geom_id_cache: dict[str, str] = {}
        self._parm_id_cache: dict[tuple[str, str, str], str] = {}
        self._analysis_inputs_cache: dict[str, set[str]] = {}
        self.warnings: list[str] = []
        self._duplicate_names_found = False

    # ------------------------------------------------------------------
    # Model loading and diagnostics
    # ------------------------------------------------------------------

    def load(self) -> "VSPWrapper":
        """
        Load the ``.vsp3`` model into memory. Safe to call multiple times.

        Returns
        -------
        self : VSPWrapper
            Enables method chaining: ``wrap.load().run_vspaero_sweep(...)``.
        """
        vsp = self._vsp
        logger.info("Loading VSP model: %s", self._path)

        vsp.ClearVSPModel()
        self._geom_id_cache.clear()
        self._parm_id_cache.clear()
        self._analysis_inputs_cache.clear()
        self.warnings.clear()

        vsp.ReadVSPFile(str(self._path))
        self._loaded = True

        all_geoms = list(vsp.FindGeoms())
        for geom_id in all_geoms:
            name = vsp.GetGeomName(geom_id)
            cache_name = name
            suffix = 2
            while cache_name in self._geom_id_cache:
                cache_name = f"{name}_{suffix}"
                suffix += 1

            if cache_name != name:
                logger.warning(
                    "Duplicate geometry name '%s' detected. Renaming to '%s' in the Python cache.",
                    name,
                    cache_name,
                )
                self.warnings.append(
                    f"Duplicate component name '{name}' found in model; cached as '{cache_name}'."
                )

            self._geom_id_cache[cache_name] = geom_id

        geom_count = len(self._geom_id_cache)
        actual_count = len(all_geoms)
        logger.info(
            "Model loaded: %d geometry components found (%d in VSP, %d unique in cache).",
            geom_count,
            actual_count,
            geom_count,
        )

        if geom_count == 0:
            raise OpenVSPError(
                f"The model at '{self._path}' loaded but contains no geometry components. "
                "The file may be empty or corrupted."
            )

        self._duplicate_names_found = actual_count != geom_count
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise OpenVSPError("Model is not loaded. Call VSPWrapper.load() before analysis.")

    @property
    def geom_names(self) -> list[str]:
        """Return names of all geometry components in the model."""
        self._ensure_loaded()
        return list(self._geom_id_cache.keys())

    def get_component_diagnostics(self) -> dict[str, Any]:
        """Return a structured summary of the currently loaded geometry cache."""
        self._ensure_loaded()

        component_info = []
        for name in self.geom_names:
            geom_id = self._geom_id_cache[name]
            component_info.append(
                {
                    "name": name,
                    "geom_type": self._vsp.GetGeomTypeName(geom_id),
                    "geom_id": geom_id,
                    "is_renamed_duplicate": name != self._vsp.GetGeomName(geom_id),
                }
            )

        return {
            "total_components": len(self._vsp.FindGeoms()),
            "unique_names": len(self._geom_id_cache),
            "has_duplicates": self._duplicate_names_found,
            "warnings": list(self.warnings),
            "component_info": component_info,
        }

    def _drain_openvsp_errors(self) -> list[str]:
        """Collect and clear the OpenVSP error queue."""
        errors: list[str] = []
        try:
            while self._vsp.GetNumTotalErrors() > 0:
                err = self._vsp.PopLastError()
                if not err:
                    break
                errors.append(f"Error Code: {err.GetErrorCode()}, Desc: {err.GetErrorString()}")
        except Exception:
            return errors
        return errors

    # ------------------------------------------------------------------
    # Geometry sets
    # ------------------------------------------------------------------

    def setup_dual_aero_sets(
        self,
        thin_keywords: list[str] | tuple[str, ...],
        thick_keywords: list[str] | tuple[str, ...],
        *,
        thin_set_index: int = DEFAULT_THIN_SET,
        thick_set_index: int = DEFAULT_THICK_SET,
    ) -> dict[str, list[str]]:
        """
        Assign lifting surfaces to one set and bodies to another.

        OpenVSP 3.48.2 expects:
          - ``ThinGeomSet`` for lifting surfaces (wing, tail, control-surface carriers)
          - ``GeomSet`` for thick bodies (fuselage, pods, etc.)

        Returns a summary dictionary so notebooks can display what was assigned.
        """
        self._ensure_loaded()
        vsp = self._vsp
        all_geoms = list(vsp.FindGeoms())

        thin_hits: list[str] = []
        thick_hits: list[str] = []

        for geom_id in all_geoms:
            vsp.SetSetFlag(geom_id, thin_set_index, False)
            vsp.SetSetFlag(geom_id, thick_set_index, False)

        for geom_id in all_geoms:
            name = vsp.GetGeomName(geom_id)
            lowered = name.lower()

            if any(keyword.lower() in lowered for keyword in thin_keywords):
                vsp.SetSetFlag(geom_id, thin_set_index, True)
                thin_hits.append(name)
            elif any(keyword.lower() in lowered for keyword in thick_keywords):
                vsp.SetSetFlag(geom_id, thick_set_index, True)
                thick_hits.append(name)

        vsp.Update()

        logger.info(
            "VSPAERO set assignment complete: %d thin components in set %d, %d thick components in set %d.",
            len(thin_hits),
            thin_set_index,
            len(thick_hits),
            thick_set_index,
        )
        return {
            "thin_components": thin_hits,
            "thick_components": thick_hits,
        }

    def get_set_members(self, set_index: int) -> list[str]:
        """Return cached component names that currently belong to a given set."""
        self._ensure_loaded()
        members: list[str] = []
        for name, geom_id in self._geom_id_cache.items():
            try:
                if self._vsp.GetSetFlag(geom_id, int(set_index)):
                    members.append(name)
            except Exception:
                continue
        return members

    # ------------------------------------------------------------------
    # Geometry and parameter access
    # ------------------------------------------------------------------

    def get_geom_id(self, name: str) -> str:
        """Return the internal geometry ID for a component by name."""
        self._ensure_loaded()
        if name not in self._geom_id_cache:
            available = ", ".join(f"'{candidate}'" for candidate in self._geom_id_cache)
            raise OpenVSPError(
                f"Geometry component '{name}' not found in model.\n"
                f"Available components: {available}"
            )
        return self._geom_id_cache[name]

    def get_geom_type(self, name: str) -> str:
        """Return the geometry type string (for example ``Wing`` or ``Fuselage``)."""
        geom_id = self.get_geom_id(name)
        return self._vsp.GetGeomTypeName(geom_id)

    def _parm_group_name(self, parm_id: str) -> str:
        """Return the most informative parameter group name available."""
        vsp = self._vsp
        display_name = ""
        if hasattr(vsp, "GetParmDisplayGroupName"):
            try:
                display_name = vsp.GetParmDisplayGroupName(parm_id)
            except Exception:
                display_name = ""
        raw_name = vsp.GetParmGroupName(parm_id)
        return display_name or raw_name

    def list_parm_groups(self, geom_name: str) -> list[str]:
        """Return distinct parameter group names for a given geometry component."""
        geom_id = self.get_geom_id(geom_name)
        groups = {self._parm_group_name(parm_id) for parm_id in self._vsp.GetGeomParmIDs(geom_id)}
        return sorted(group for group in groups if group)

    def get_all_params(self, geom_name: str) -> dict[str, float]:
        """
        Return a dictionary of ``{display_group/parm_name: value}``.

        Using the display group name keeps section-specific parameters visible.
        For example, OpenVSP stores both ``XSec`` and ``XSec_1`` information; the
        latter is what users need for reliable automation.
        """
        geom_id = self.get_geom_id(geom_name)
        vsp = self._vsp
        result: dict[str, float] = {}
        duplicate_counts: dict[str, int] = {}

        for parm_id in vsp.GetGeomParmIDs(geom_id):
            parm_name = vsp.GetParmName(parm_id)
            group_name = self._parm_group_name(parm_id)

            try:
                value = float(vsp.GetParmVal(parm_id))
            except Exception:
                continue

            base_key = f"{group_name}/{parm_name}"
            count = duplicate_counts.get(base_key, 0) + 1
            duplicate_counts[base_key] = count
            key = base_key if count == 1 else f"{base_key}#{count}"
            result[key] = value

        return result

    def _resolve_parm_id(self, geom_name: str, parm_name: str, group_name: str) -> str:
        """
        Resolve and cache parameter IDs.

        The raw OpenVSP group name is not always unique across sections, so we
        first try the exact API lookup and then fall back to a display-group-aware
        search. Ambiguous lookups raise a helpful error instead of silently
        selecting the wrong parameter.
        """
        self._ensure_loaded()
        cache_key = (geom_name, parm_name, group_name)
        if cache_key in self._parm_id_cache:
            return self._parm_id_cache[cache_key]

        vsp = self._vsp
        geom_id = self.get_geom_id(geom_name)
        requested_group = group_name.strip()

        parm_id = vsp.GetParm(geom_id, parm_name, requested_group)
        if parm_id:
            self._parm_id_cache[cache_key] = parm_id
            return parm_id

        matches: list[tuple[str, str]] = []
        group_matches: list[tuple[str, str]] = []

        for candidate_id in vsp.GetGeomParmIDs(geom_id):
            candidate_name = vsp.GetParmName(candidate_id)
            if candidate_name.lower() != parm_name.lower():
                continue

            candidate_group = self._parm_group_name(candidate_id)
            matches.append((candidate_group, candidate_id))
            if candidate_group.lower() == requested_group.lower():
                group_matches.append((candidate_group, candidate_id))

        if len(group_matches) == 1:
            resolved_id = group_matches[0][1]
            logger.info(
                "Resolved parameter %s/%s/%s using display group name '%s'.",
                geom_name,
                requested_group,
                parm_name,
                group_matches[0][0],
            )
            self._parm_id_cache[cache_key] = resolved_id
            return resolved_id

        if len(group_matches) > 1:
            options = ", ".join(group for group, _ in group_matches)
            raise OpenVSPError(
                f"Parameter lookup for '{geom_name}/{requested_group}/{parm_name}' is ambiguous.\n"
                f"Matching groups: {options}\n"
                "Use an exact display group such as 'XSec_1'."
            )

        if len(matches) == 1:
            resolved_group, resolved_id = matches[0]
            logger.info(
                "Auto-resolved parameter %s/%s/%s to display group '%s'.",
                geom_name,
                requested_group,
                parm_name,
                resolved_group,
            )
            self._parm_id_cache[cache_key] = resolved_id
            return resolved_id

        if len(matches) > 1:
            options = ", ".join(sorted({group for group, _ in matches}))
            raise OpenVSPError(
                f"Parameter '{parm_name}' for component '{geom_name}' matches multiple groups and "
                f"cannot be resolved from '{requested_group}'.\n"
                f"Candidate groups: {options}\n"
                "Inspect wrapper.get_all_params(...) and use the exact display group name."
            )

        available_params = "\n".join(f" - {key}" for key in self.get_all_params(geom_name))
        raise OpenVSPError(
            f"Parameter '{parm_name}' not found for component '{geom_name}' in group '{requested_group}'.\n"
            f"Available parameters for this component:\n{available_params}"
        )

    def get_param(self, geom_name: str, parm_name: str, group_name: str) -> float:
        """Get the current value of a named parameter."""
        parm_id = self._resolve_parm_id(geom_name, parm_name, group_name)
        return float(self._vsp.GetParmVal(parm_id))

    def set_param(
        self,
        geom_name: str,
        parm_name: str,
        group_name: str,
        value: float,
        *,
        clamp: bool = True,
    ) -> "VSPWrapper":
        """Set the value of a named parameter."""
        parm_id = self._resolve_parm_id(geom_name, parm_name, group_name)
        vsp = self._vsp

        lower = float(vsp.GetParmLowerLimit(parm_id))
        upper = float(vsp.GetParmUpperLimit(parm_id))

        if not (lower <= value <= upper):
            if clamp:
                clamped = max(lower, min(upper, value))
                logger.warning(
                    "Parameter %s/%s/%s: requested %.6f clamped to [%.6f, %.6f] -> %.6f.",
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
                    f"value {value:.6f} is outside bounds [{lower:.6f}, {upper:.6f}]."
                )

        vsp.SetParmVal(parm_id, float(value))
        return self

    def set_params(
        self,
        param_dict: Mapping[tuple[str, str, str], float],
        **kwargs,
    ) -> "VSPWrapper":
        """Set multiple parameters at once."""
        for (geom_name, parm_name, group_name), value in param_dict.items():
            self.set_param(geom_name, parm_name, group_name, value, **kwargs)
        self._vsp.Update()
        return self

    # ------------------------------------------------------------------
    # Analysis input helpers
    # ------------------------------------------------------------------

    def get_available_analysis_inputs(self, analysis_name: str) -> set[str]:
        """Return the available input names for an OpenVSP analysis."""
        self._ensure_loaded()
        if analysis_name not in self._analysis_inputs_cache:
            self._analysis_inputs_cache[analysis_name] = set(
                self._vsp.GetAnalysisInputNames(analysis_name)
            )
        return self._analysis_inputs_cache[analysis_name]

    def _analysis_input_exists(self, analysis_name: str, input_name: str) -> bool:
        return input_name in self.get_available_analysis_inputs(analysis_name)

    def _set_int_analysis_input(self, analysis_name: str, input_name: str, values: list[int]) -> bool:
        if not self._analysis_input_exists(analysis_name, input_name):
            return False
        self._vsp.SetIntAnalysisInput(analysis_name, input_name, [int(v) for v in values])
        return True

    def _set_double_analysis_input(
        self,
        analysis_name: str,
        input_name: str,
        values: list[float],
    ) -> bool:
        if not self._analysis_input_exists(analysis_name, input_name):
            return False
        self._vsp.SetDoubleAnalysisInput(analysis_name, input_name, [float(v) for v in values])
        return True

    def _set_string_analysis_input(
        self,
        analysis_name: str,
        input_name: str,
        values: list[str],
    ) -> bool:
        if not self._analysis_input_exists(analysis_name, input_name):
            return False
        self._vsp.SetStringAnalysisInput(analysis_name, input_name, list(values))
        return True

    # ------------------------------------------------------------------
    # VSPAERO settings and control-surface helpers
    # ------------------------------------------------------------------

    def _get_vspaero_settings_container(self) -> str:
        container_id = self._vsp.FindContainer("VSPAEROSettings", 0)
        if not container_id:
            raise OpenVSPError("Could not locate the VSPAEROSettings container in the loaded model.")
        return container_id

    def get_vspaero_settings(self) -> dict[str, float]:
        """Return scalar parameters currently stored in the ``VSPAEROSettings`` container."""
        self._ensure_loaded()
        vsp = self._vsp
        container_id = self._get_vspaero_settings_container()
        result: dict[str, float] = {}
        duplicate_counts: dict[str, int] = {}

        for parm_id in vsp.FindContainerParmIDs(container_id):
            group_name = self._parm_group_name(parm_id)
            parm_name = vsp.GetParmName(parm_id)
            try:
                value = float(vsp.GetParmVal(parm_id))
            except Exception:
                continue

            base_key = f"{group_name}/{parm_name}"
            count = duplicate_counts.get(base_key, 0) + 1
            duplicate_counts[base_key] = count
            key = base_key if count == 1 else f"{base_key}#{count}"
            result[key] = value

        return result

    def get_vspaero_reference_cg(self) -> dict[str, float]:
        """Return the reference CG currently stored in the VSPAERO settings container."""
        settings = self.get_vspaero_settings()
        return {
            "Xcg": float(settings.get("VSPAERO/Xcg", 0.0)),
            "Ycg": float(settings.get("VSPAERO/Ycg", 0.0)),
            "Zcg": float(settings.get("VSPAERO/Zcg", 0.0)),
        }

    def set_vspaero_reference_cg(
        self,
        *,
        xcg: float | None = None,
        ycg: float | None = None,
        zcg: float | None = None,
    ) -> None:
        """Update the CG stored in the VSPAERO settings container."""
        vsp = self._vsp
        container_id = self._get_vspaero_settings_container()
        updates = {
            "Xcg": xcg,
            "Ycg": ycg,
            "Zcg": zcg,
        }

        for parm_name, value in updates.items():
            if value is None:
                continue
            parm_id = vsp.FindParm(container_id, parm_name, "VSPAERO")
            if parm_id:
                vsp.SetParmVal(parm_id, float(value))

        vsp.Update()

    def get_control_surface_groups(self) -> list[dict[str, Any]]:
        """
        Return a structured summary of VSPAERO control-surface groups.

        Group indices are zero-based in the OpenVSP Python API.
        """
        self._ensure_loaded()
        vsp = self._vsp
        container_id = self._get_vspaero_settings_container()
        groups: list[dict[str, Any]] = []

        for index in range(int(vsp.GetNumControlSurfaceGroups())):
            group_name = vsp.GetVSPAEROControlGroupName(index)
            group_key = f"ControlSurfaceGroup_{index}"

            deflection_id = vsp.FindParm(container_id, "DeflectionAngle", group_key)
            active_id = vsp.FindParm(container_id, "ActiveFlag", group_key)
            gains: dict[str, float] = {}

            for parm_id in vsp.FindContainerParmIDs(container_id):
                if self._parm_group_name(parm_id) != group_key:
                    continue
                parm_name = vsp.GetParmName(parm_id)
                if parm_name.startswith("Surf_") and parm_name.endswith("_Gain"):
                    gains[parm_name] = float(vsp.GetParmVal(parm_id))

            groups.append(
                {
                    "index": index,
                    "name": group_name,
                    "active_flag": bool(vsp.GetParmVal(active_id)) if active_id else False,
                    "deflection_angle": float(vsp.GetParmVal(deflection_id)) if deflection_id else 0.0,
                    "active_surfaces": list(vsp.GetActiveCSNameVec(index)),
                    "available_surfaces": list(vsp.GetAvailableCSNameVec(index)),
                    "gains": gains,
                }
            )

        return groups

    def _resolve_control_surface_group_index(self, group: str | int) -> int:
        """Resolve a control-surface group from either index or group name."""
        groups = self.get_control_surface_groups()

        if isinstance(group, int):
            if any(entry["index"] == group for entry in groups):
                return group
            raise OpenVSPError(f"Control-surface group index {group} does not exist.")

        lowered = group.strip().lower()
        for entry in groups:
            if entry["name"].strip().lower() == lowered:
                return int(entry["index"])

        available = ", ".join(entry["name"] for entry in groups)
        raise OpenVSPError(
            f"Control-surface group '{group}' not found. Available groups: {available or 'none'}."
        )

    def set_control_surface_deflections(self, deflections: Mapping[str | int, float]) -> None:
        """Set one or more VSPAERO control-surface group deflections in degrees."""
        if not deflections:
            return

        vsp = self._vsp
        container_id = self._get_vspaero_settings_container()

        for group, angle in deflections.items():
            group_index = self._resolve_control_surface_group_index(group)
            parm_id = vsp.FindParm(container_id, "DeflectionAngle", f"ControlSurfaceGroup_{group_index}")
            if not parm_id:
                raise OpenVSPError(
                    f"Could not resolve DeflectionAngle for control-surface group '{group}'."
                )
            vsp.SetParmVal(parm_id, float(angle))

        vsp.Update()

    # ------------------------------------------------------------------
    # Reference data
    # ------------------------------------------------------------------

    def get_reference_quantities(self) -> dict[str, float]:
        """Return the VSPAERO reference quantities currently stored in the model."""
        self._ensure_loaded()
        vsp = self._vsp

        refs: dict[str, float] = {}
        vsp.SetAnalysisInputDefaults("VSPAEROSweep")
        refs["Sref"] = float(vsp.GetDoubleAnalysisInput("VSPAEROSweep", "Sref", 0)[0])
        refs["bref"] = float(vsp.GetDoubleAnalysisInput("VSPAEROSweep", "bref", 0)[0])
        refs["cref"] = float(vsp.GetDoubleAnalysisInput("VSPAEROSweep", "cref", 0)[0])
        return refs

    # ------------------------------------------------------------------
    # Mass properties
    # ------------------------------------------------------------------

    def run_mass_properties(
        self,
        *,
        num_slices: int = 100,
        set_index: int = 0,
        working_dir: str | Path | None = None,
    ) -> "MassProperties":
        """Run the OpenVSP ``MassProp`` analysis and return structured mass data."""
        from vspopt.postprocess import MassProperties

        self._ensure_loaded()
        vsp = self._vsp

        if working_dir is None:
            working_dir = self._path.parent
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        analysis_name = "MassProp"
        vsp.SetAnalysisInputDefaults(analysis_name)
        if not self._set_int_analysis_input(analysis_name, "NumMassSlices", [int(num_slices)]):
            self._set_int_analysis_input(analysis_name, "NumMassSlice", [int(num_slices)])
        self._set_int_analysis_input(analysis_name, "Set", [int(set_index)])

        prev_dir = os.getcwd()
        try:
            os.chdir(str(working_dir))
            res_id = vsp.ExecAnalysis(analysis_name)
        finally:
            os.chdir(prev_dir)

        if not res_id:
            raise OpenVSPError("MassProp returned an empty result ID.")

        data_names = vsp.GetAllDataNames(res_id)
        raw_results: dict[str, Any] = {}
        for name in data_names:
            rtype = vsp.GetResultsType(res_id, name)
            if rtype == vsp.DOUBLE_DATA or rtype == 2:
                raw_results[name] = list(vsp.GetDoubleResults(res_id, name))
            elif rtype == vsp.VEC3D_DATA or rtype == 4:
                vecs = vsp.GetVec3dResults(res_id, name)
                raw_results[name] = [(vec.x(), vec.y(), vec.z()) for vec in vecs]
            elif rtype == vsp.DOUBLE_MATRIX_DATA or rtype == 5:
                raw_results[name] = [list(row) for row in vsp.GetDoubleMatResults(res_id, name)]
            elif rtype == vsp.STRING_DATA or rtype == 3:
                raw_results[name] = list(vsp.GetStringResults(res_id, name))
            elif rtype == vsp.INT_DATA or rtype == 1:
                raw_results[name] = list(vsp.GetIntResults(res_id, name))
            else:
                raw_results[name] = None

        logger.info("MassProp completed: extracted %d result fields.", len(raw_results))
        return MassProperties.from_results(raw_results)

    # ------------------------------------------------------------------
    # VSPAERO sweep execution
    # ------------------------------------------------------------------

    def _cleanup_run_artifacts(self, working_dir: Path, output_stem: str) -> None:
        """Remove stale files for a case so artifact discovery never picks an old run."""
        for path in working_dir.glob(f"{output_stem}*"):
            if path.is_file():
                path.unlink(missing_ok=True)

    def _write_case_model(self, working_dir: Path, output_stem: str) -> Path:
        """Save the current model state to a case-specific ``.vsp3`` file."""
        vsp = self._vsp
        case_model_path = working_dir / f"{output_stem}.vsp3"
        if hasattr(vsp, "SetVSP3FileName"):
            vsp.SetVSP3FileName(str(case_model_path))
        vsp.WriteVSPFile(str(case_model_path), getattr(vsp, "SET_ALL", 0))
        return case_model_path

    def run_vspaero_sweep(
        self,
        *,
        alpha_start: float = -5.0,
        alpha_end: float = 20.0,
        alpha_npts: int = 7,
        mach: float = 0.2,
        re_cref: float = 1e6,
        wake_iter: int | None = None,
        wake_iterations: int | None = None,
        wake_nodes: int | None = None,
        analysis_method: int = 0,
        beta: float = 0.0,
        xcg: float | None = None,
        ycg: float | None = None,
        zcg: float | None = None,
        control_surface_deflections: Mapping[str | int, float] | None = None,
        working_dir: str | Path | None = None,
        output_stem: str | None = None,
        parse_history: bool = True,
        parse_stability: bool = True,
        require_stability_file: bool = False,
        allow_incomplete_results: bool = False,
        min_convergence_iterations: int = 3,
        use_massprop_cg: bool = False,
        massprop_num_slices: int = 100,
        thin_geom_set: int = DEFAULT_THIN_SET,
        thick_geom_set: int = DEFAULT_THICK_SET,
        stability_mode: int | None = None,
        redirect_solver_output: bool = True,
    ) -> "VSPAEROResults":
        """
        Run a VSPAERO alpha sweep and return structured results.

        Parameters
        ----------
        The aerodynamic conditions stay sweep-based in alpha, while the design
        variables can now act on geometry, CG position, and control-surface
        deflections through the other wrapper helpers.
        """
        from vspopt.postprocess import (
            check_history_convergence,
            extract_cd0_details,
            extract_cd0_from_arrays,
            find_generated_artifact,
            parse_stab_file,
            read_history_file,
            stability_records_to_dataframe,
        )
        from vspopt.vspaero import (
            VSPAEROResults,
            _parse_polar_file_fallback,
            _parse_results_manager,
            results_from_stability_records,
        )

        self._ensure_loaded()
        vsp = self._vsp

        if working_dir is None:
            working_dir = self._path.parent
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)

        requested_wake_iterations = wake_iterations if wake_iterations is not None else wake_iter
        if wake_iterations is None:
            wake_iterations = 5 if wake_iter is None else int(wake_iter)
        elif wake_iter is not None and int(wake_iter) != int(wake_iterations):
            logger.warning(
                "Both wake_iter=%s and wake_iterations=%s were provided; using wake_iterations.",
                wake_iter,
                wake_iterations,
            )
        wake_iterations = int(wake_iterations)
        if wake_iterations < self.MIN_WAKE_ITERATIONS:
            logger.warning(
                "wake_iterations=%s is below the OpenVSP/VSPAERO practical minimum; using %s.",
                wake_iterations,
                self.MIN_WAKE_ITERATIONS,
            )
            wake_iterations = self.MIN_WAKE_ITERATIONS

        requested_wake_nodes = wake_nodes
        wake_nodes = int(wake_iterations if wake_nodes is None else wake_nodes)
        if wake_nodes < self.MIN_WAKE_NODES:
            logger.warning(
                "wake_nodes=%s would create a degenerate wake grid; using %s.",
                wake_nodes,
                self.MIN_WAKE_NODES,
            )
            wake_nodes = self.MIN_WAKE_NODES

        output_stem = output_stem or self._path.stem
        self._cleanup_run_artifacts(working_dir, output_stem)

        reference_quantities = self.get_reference_quantities()
        mass_properties = None

        if use_massprop_cg:
            mass_properties = self.run_mass_properties(
                num_slices=massprop_num_slices,
                working_dir=working_dir,
            )
            if mass_properties.cg_is_finite:
                xcg = mass_properties.xcg if xcg is None else xcg
                ycg = mass_properties.ycg if ycg is None else ycg
                zcg = mass_properties.zcg if zcg is None else zcg
            else:
                logger.warning(
                    "MassProp completed but did not produce a finite CG. Proceeding with the stored VSPAERO CG."
                )

        if xcg is not None or ycg is not None or zcg is not None:
            self.set_vspaero_reference_cg(xcg=xcg, ycg=ycg, zcg=zcg)

        if control_surface_deflections:
            self.set_control_surface_deflections(control_surface_deflections)

        case_model_path = self._write_case_model(working_dir, output_stem)

        logger.info(
            "Running VSPAEROSweep for '%s': alpha=[%.1f, %.1f] deg x %d, Mach=%.3f, Re=%.3e.",
            output_stem,
            alpha_start,
            alpha_end,
            alpha_npts,
            mach,
            re_cref,
        )

        geom_analysis = "VSPAEROComputeGeometry"
        vsp.SetAnalysisInputDefaults(geom_analysis)
        self._set_int_analysis_input(geom_analysis, "GeomSet", [int(thick_geom_set)])
        self._set_int_analysis_input(geom_analysis, "ThinGeomSet", [int(thin_geom_set)])

        analysis = "VSPAEROSweep"
        vsp.SetAnalysisInputDefaults(analysis)
        self._set_int_analysis_input(analysis, "GeomSet", [int(thick_geom_set)])
        self._set_int_analysis_input(analysis, "ThinGeomSet", [int(thin_geom_set)])

        self._set_double_analysis_input(analysis, "AlphaStart", [float(alpha_start)])
        self._set_double_analysis_input(analysis, "AlphaEnd", [float(alpha_end)])
        self._set_int_analysis_input(analysis, "AlphaNpts", [int(alpha_npts)])

        self._set_double_analysis_input(analysis, "MachStart", [float(mach)])
        self._set_double_analysis_input(analysis, "MachEnd", [float(mach)])
        self._set_int_analysis_input(analysis, "MachNpts", [1])

        self._set_double_analysis_input(analysis, "BetaStart", [float(beta)])
        self._set_double_analysis_input(analysis, "BetaEnd", [float(beta)])
        self._set_int_analysis_input(analysis, "BetaNpts", [1])

        self._set_double_analysis_input(analysis, "ReCref", [float(re_cref)])
        # OpenVSP 3.48 exposes both a solver wake-iteration count and a wake-node
        # count. The notebook-level ``wake_iterations`` parameter drives both by
        # default so convergence studies can vary wake discretisation in one place.
        self._set_int_analysis_input(analysis, "WakeNumIter", [int(wake_iterations)])
        self._set_int_analysis_input(analysis, "NumWakeNodes", [int(wake_nodes)])
        self._set_double_analysis_input(analysis, "Sref", [float(reference_quantities.get("Sref", 0.0))])
        self._set_double_analysis_input(analysis, "bref", [float(reference_quantities.get("bref", 0.0))])
        self._set_double_analysis_input(analysis, "cref", [float(reference_quantities.get("cref", 0.0))])

        active_cg = self.get_vspaero_reference_cg()
        self._set_double_analysis_input(analysis, "Xcg", [float(active_cg["Xcg"])])
        self._set_double_analysis_input(analysis, "Ycg", [float(active_cg["Ycg"])])
        self._set_double_analysis_input(analysis, "Zcg", [float(active_cg["Zcg"])])

        if redirect_solver_output:
            self._set_string_analysis_input(
                analysis,
                "RedirectFile",
                [str(working_dir / f"{output_stem}.vspaero.log")],
            )

        if parse_stability and self._analysis_input_exists(analysis, "UnsteadyType"):
            if stability_mode is None:
                stability_mode = getattr(vsp, "STABILITY_DEFAULT", 1)
            self._set_int_analysis_input(analysis, "UnsteadyType", [int(stability_mode)])
        elif self._analysis_input_exists(analysis, "UnsteadyType"):
            self._set_int_analysis_input(analysis, "UnsteadyType", [int(getattr(vsp, "STABILITY_OFF", 0))])

        if analysis_method != 0:
            logger.warning(
                "analysis_method=%s was requested, but OpenVSP 3.48.2 does not expose "
                "AnalysisMethod through the Analysis Manager in this environment. "
                "Proceeding with the model's stored VSPAERO method.",
                analysis_method,
            )

        vsp.Update()

        prev_dir = os.getcwd()
        try:
            os.chdir(str(working_dir))
            geom_res_id = vsp.ExecAnalysis(geom_analysis)
            sweep_res_id = vsp.ExecAnalysis(analysis)
        finally:
            os.chdir(prev_dir)

        errors = self._drain_openvsp_errors()
        if errors:
            logger.debug("OpenVSP reported the following messages:\n%s", "\n".join(errors))

        if not geom_res_id:
            raise OpenVSPError("VSPAEROComputeGeometry returned an empty result ID.")
        if not sweep_res_id:
            raise OpenVSPError("VSPAEROSweep returned an empty result ID.")

        results = _parse_results_manager(vsp, sweep_res_id, mach, re_cref, alpha_npts)
        api_failed = (
            len(results.CL) == 0
            or np.all(results.CL == 0)
            or np.all(np.isnan(results.CL))
        )

        if api_failed:
            logger.info("Results Manager returned unusable data. Falling back to the .polar file parser.")
            polar_path = working_dir / f"{output_stem}.polar"
            results = _parse_polar_file_fallback(polar_path, mach, re_cref)

        results.case_name = output_stem
        results.working_dir = working_dir
        results.model_path = case_model_path
        results.solver_log_path = working_dir / f"{output_stem}.vspaero.log"
        results.Sref = float(reference_quantities.get("Sref", 0.0))
        results.bref = float(reference_quantities.get("bref", 0.0))
        results.cref = float(reference_quantities.get("cref", 0.0))
        results.wake_iterations = int(wake_iterations)
        results.wake_nodes = int(wake_nodes)
        results.requested_wake_iterations = (
            int(requested_wake_iterations) if requested_wake_iterations is not None else int(wake_iterations)
        )
        results.requested_wake_nodes = (
            int(requested_wake_nodes) if requested_wake_nodes is not None else int(wake_nodes)
        )
        results.mass_properties = mass_properties

        search_dirs = [working_dir]
        if parse_history:
            history_path = find_generated_artifact(search_dirs, output_stem, ".history")
            if history_path is not None:
                results.history_path = history_path
                results.history_table = read_history_file(history_path)
                results.convergence = check_history_convergence(
                    history_path,
                    min_iter=min_convergence_iterations,
                )

        if parse_stability:
            stab_path = find_generated_artifact(search_dirs, output_stem, ".stab")
            if stab_path is not None:
                stability_records = parse_stab_file(stab_path)
                results.stab_path = stab_path
                results.stability_records = stability_records
                results.stability_table = stability_records_to_dataframe(stability_records)

                # In stability mode, the `.stab` file is the authoritative
                # source for the baseline aerodynamic totals at each alpha.
                # The Results Manager often mixes the perturbed derivative cases
                # into the output vectors, which makes the alpha/CL/CD arrays
                # unusable for sweep plots and optimization. We therefore
                # normalize the aerodynamic arrays from the parsed `.stab`
                # records whenever that file is available.
                if stability_records:
                    logger.info(
                        "Normalizing aerodynamic totals for case '%s' from the .stab file.",
                        output_stem,
                    )
                    stab_results = results_from_stability_records(stability_records, mach, re_cref)
                    if len(stab_results.alpha) > 0:
                        for field_name in (
                            "alpha",
                            "CL",
                            "CD",
                            "CDi",
                            "CDo",
                            "CDsff",
                            "CM",
                            "CMx",
                            "CMz",
                            "CS",
                            "LD",
                            "E",
                            "CFx",
                            "CFy",
                            "CFz",
                            "n_points",
                        ):
                            setattr(results, field_name, getattr(stab_results, field_name))

        results.warnings = results.validate()

        # CD0 is attached after the stability normalization so the fallback fit
        # always sees the same aerodynamic arrays that downstream users see.
        polar_path = find_generated_artifact(search_dirs, output_stem, ".polar")
        cd0_extraction = None
        if polar_path is not None:
            results.polar_path = polar_path
            try:
                cd0_extraction = extract_cd0_details(polar_path)
            except Exception as exc:
                logger.warning("Could not extract CD0 from %s: %s", polar_path, exc)

        if cd0_extraction is None or not np.isfinite(cd0_extraction.cd0):
            cd0_extraction = extract_cd0_from_arrays(
                results.alpha,
                results.CL,
                results.CD,
                results.CDo,
                source=f"{output_stem} aerodynamic arrays",
            )
        results.set_cd0_extraction(cd0_extraction)

        if len(results.alpha) == 0 and not allow_incomplete_results:
            artifact_names = sorted(path.name for path in working_dir.glob(f"{output_stem}*"))
            details = "\n".join(errors) if errors else "No OpenVSP API errors were reported."
            raise OpenVSPError(
                f"VSPAERO did not return valid aerodynamic data for case '{output_stem}'.\n"
                f"Artifacts found in {working_dir}:\n  " + "\n  ".join(artifact_names or ["<none>"]) + "\n"
                f"OpenVSP messages:\n{details}"
            )

        if require_stability_file and (results.stab_path is None or results.stability_table.empty):
            raise OpenVSPError(
                f"Case '{output_stem}' completed without a usable .stab file in {working_dir}. "
                "This project requires stability derivatives for downstream optimization."
            )

        return results

    def update_and_run(
        self,
        param_dict: Mapping[tuple[str, str, str], float],
        sweep_kwargs: dict[str, Any] | None = None,
    ) -> "VSPAEROResults":
        """
        Convenience method: set a batch of geometry parameters and immediately
        run a VSPAERO sweep.
        """
        sweep_kwargs = dict(sweep_kwargs or {})
        self.set_params(param_dict)
        return self.run_vspaero_sweep(**sweep_kwargs)

    # ------------------------------------------------------------------
    # Context manager and misc
    # ------------------------------------------------------------------

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
