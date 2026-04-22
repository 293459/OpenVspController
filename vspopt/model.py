"""
High-level aircraft model representation built on top of :mod:`vspopt.wrapper`.

The goal of this module is not to mirror every OpenVSP concept. Instead, it
extracts the subset that is most useful in the notebook:
  - lifting-surface geometry;
  - fuselage dimensions;
  - a lightweight component registry for everything else.

The parsing logic prefers explicit OpenVSP section names such as ``XSec_1`` and
``WingGeom`` so the values exposed here match the values that automation later
edits during sweeps and optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry component dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WingComponent:
    """Geometric parameters of a lifting surface."""

    name: str
    geom_id: str
    geom_type: str
    span: float = 0.0
    area: float = 0.0
    aspect_ratio: float = 0.0
    root_chord: float = 0.0
    tip_chord: float = 0.0
    taper_ratio: float = 0.0
    sweep_le: float = 0.0
    dihedral: float = 0.0
    twist_tip: float = 0.0
    raw_params: dict[str, float] = field(default_factory=dict)

    @property
    def mac(self) -> float:
        """Mean aerodynamic chord for a simple tapered wing."""
        if abs(self.taper_ratio - 1.0) < 1e-6:
            return self.root_chord
        lam = self.taper_ratio
        return (2.0 / 3.0) * self.root_chord * (1.0 + lam + lam**2) / (1.0 + lam)

    @property
    def area_half(self) -> float:
        """Semi-span area."""
        return self.area / 2.0

    def to_series(self) -> "pd.Series":
        return pd.Series(
            {
                "span [m]": self.span,
                "area [m^2]": self.area,
                "aspect_ratio [-]": self.aspect_ratio,
                "root_chord [m]": self.root_chord,
                "tip_chord [m]": self.tip_chord,
                "taper_ratio [-]": self.taper_ratio,
                "mac [m]": self.mac,
                "sweep_le [deg]": self.sweep_le,
                "dihedral [deg]": self.dihedral,
                "twist_tip [deg]": self.twist_tip,
            },
            name=self.name,
        )


@dataclass
class FuselageComponent:
    """Geometric parameters of the fuselage."""

    name: str
    geom_id: str
    length: float = 0.0
    max_diameter: float = 0.0
    fineness_ratio: float = 0.0
    raw_params: dict[str, float] = field(default_factory=dict)

    def to_series(self) -> "pd.Series":
        return pd.Series(
            {
                "length [m]": self.length,
                "max_diameter [m]": self.max_diameter,
                "fineness_ratio [-]": self.fineness_ratio,
            },
            name=self.name,
        )


@dataclass
class PropellerComponent:
    """Geometric parameters of a propeller or rotor."""

    name: str
    geom_id: str
    diameter: float = 0.0
    num_blades: int = 0
    design_rpm: float = 0.0
    raw_params: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aircraft model class
# ---------------------------------------------------------------------------


class AircraftModel:
    """
    High-level representation of an OpenVSP aircraft model.

    After loading a ``.vsp3`` file, the class classifies components by name and
    geometry type, then exposes the most useful ones as typed attributes.
    """

    _WING_KEYWORDS = ("wing", "main", "main_wing", "mainwing")
    _HTP_KEYWORDS = ("htp", "horiz", "horizontal", "stabilizer", "htail", "h_tail")
    _VTP_KEYWORDS = ("vtp", "vert", "vertical", "fin", "vtail", "v_tail")
    _FUSE_KEYWORDS = ("fuse", "fuselage", "body", "hull")
    _PROP_KEYWORDS = ("prop", "rotor", "propeller", "fan")

    def __init__(self, vsp3_path: str | Path) -> None:
        from vspopt.wrapper import VSPWrapper

        self._wrapper = VSPWrapper(vsp3_path)
        self.path = Path(vsp3_path)

        self.wing: Optional[WingComponent] = None
        self.htp: Optional[WingComponent] = None
        self.vtp: Optional[WingComponent] = None
        self.fuselage: Optional[FuselageComponent] = None
        self.propellers: list[PropellerComponent] = []
        self.components: dict[str, object] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "AircraftModel":
        """Load the ``.vsp3`` file and populate all component attributes."""
        self._wrapper.load()
        self._parse_components()
        self._loaded = True
        logger.info(
            "AircraftModel loaded: %d components (%s).",
            len(self.components),
            ", ".join(self.components.keys()),
        )
        return self

    def setup_aero_set(
        self,
        thin_keywords: list[str] | None = None,
        thick_keywords: list[str] | None = None,
    ) -> "AircraftModel":
        """Organize the model into thin and thick VSPAERO sets."""
        if thin_keywords is None:
            thin_keywords = ["Wing", "tail", "Fin", "Stabilizer", "vtail"]
        if thick_keywords is None:
            thick_keywords = ["Fuselage", "Body", "hull", "Fuse"]

        self._wrapper.setup_dual_aero_sets(thin_keywords, thick_keywords)
        return self

    def _parse_components(self) -> None:
        """Iterate over all geometry components and classify them."""
        self.components.clear()
        self.wing = None
        self.htp = None
        self.vtp = None
        self.fuselage = None
        self.propellers = []

        vsp = self._wrapper._vsp
        for geom_name, geom_id in self._wrapper._geom_id_cache.items():
            geom_type = vsp.GetGeomTypeName(geom_id)
            raw = self._wrapper.get_all_params(geom_name)
            role = self._classify_role(geom_name, geom_type)

            if geom_type == "Wing" or role in {"wing", "htp", "vtp"}:
                comp = self._parse_wing(geom_name, geom_id, geom_type, raw)
                self.components[geom_name] = comp
                if role == "wing" and self.wing is None:
                    self.wing = comp
                elif role == "htp" and self.htp is None:
                    self.htp = comp
                elif role == "vtp" and self.vtp is None:
                    self.vtp = comp

            elif geom_type in {"Fuselage", "BodyOfRevolution"} or role == "fuselage":
                comp = self._parse_fuselage(geom_name, geom_id, raw)
                self.components[geom_name] = comp
                if self.fuselage is None:
                    self.fuselage = comp

            elif geom_type == "Prop" or role == "propeller":
                comp = self._parse_propeller(geom_name, geom_id, raw)
                self.components[geom_name] = comp
                self.propellers.append(comp)

            else:
                self.components[geom_name] = {"type": geom_type, "params": raw}

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _classify_role(self, name: str, geom_type: str) -> str:
        n = name.lower().replace("-", "_").replace(" ", "_")
        if any(keyword in n for keyword in self._WING_KEYWORDS):
            return "wing"
        if any(keyword in n for keyword in self._HTP_KEYWORDS):
            return "htp"
        if any(keyword in n for keyword in self._VTP_KEYWORDS):
            return "vtp"
        if any(keyword in n for keyword in self._FUSE_KEYWORDS):
            return "fuselage"
        if any(keyword in n for keyword in self._PROP_KEYWORDS):
            return "propeller"
        return geom_type.lower()

    def _get_param_with_fallback(
        self,
        geom_name: str,
        raw: dict[str, float],
        *,
        param_name: str,
        groups: tuple[str, ...],
        fallback_keys: tuple[str, ...] = (),
        default: float = 0.0,
    ) -> float:
        """
        Read a parameter using exact wrapper access first, then the raw cache.

        The wrapper path is preferred because it understands display group names
        such as ``XSec_1``. The raw cache remains a useful fallback for global
        quantities that are already stored with stable names.
        """
        for group_name in groups:
            try:
                return float(self._wrapper.get_param(geom_name, param_name, group_name))
            except Exception:
                continue

        keys_to_try = [f"{group}/{param_name}" for group in groups]
        keys_to_try.extend(fallback_keys)
        for key in keys_to_try:
            if key in raw:
                return float(raw[key])

        for key, value in raw.items():
            if key.split("/")[-1] == param_name:
                return float(value)

        return float(default)

    def _parse_wing(
        self,
        name: str,
        geom_id: str,
        geom_type: str,
        raw: dict[str, float],
    ) -> WingComponent:
        """
        Extract wing geometry from the OpenVSP parameter tree.

        ``WingGeom`` stores the global quantities we care about for notebook
        summaries, while ``XSec_1`` stores the editable section values we use in
        sweeps and optimization.
        """
        span = self._get_param_with_fallback(
            name,
            raw,
            param_name="TotalSpan",
            groups=("WingGeom",),
            fallback_keys=("WingGeom/TotalSpan", "WingGeom/TotalProjectedSpan", "XSec_1/Span"),
        )
        area = self._get_param_with_fallback(
            name,
            raw,
            param_name="TotalArea",
            groups=("WingGeom",),
            fallback_keys=("WingGeom/TotalArea", "XSec_1/Area"),
        )
        root_chord = self._get_param_with_fallback(
            name,
            raw,
            param_name="Root_Chord",
            groups=("XSec_1",),
            fallback_keys=("XSec_1/Root_Chord", "XSec/Root_Chord"),
        )
        tip_chord = self._get_param_with_fallback(
            name,
            raw,
            param_name="Tip_Chord",
            groups=("XSec_1",),
            fallback_keys=("XSec_1/Tip_Chord", "XSec/Tip_Chord"),
            default=root_chord,
        )
        sweep = self._get_param_with_fallback(
            name,
            raw,
            param_name="Sweep",
            groups=("XSec_1",),
            fallback_keys=("XSec_1/Sweep", "XSec/Sweep"),
        )
        dihedral = self._get_param_with_fallback(
            name,
            raw,
            param_name="Dihedral",
            groups=("XSec_1",),
            fallback_keys=("XSec_1/Dihedral", "XSec/Dihedral"),
        )
        twist = self._get_param_with_fallback(
            name,
            raw,
            param_name="Twist",
            groups=("XSec_1",),
            fallback_keys=("XSec_1/Twist", "XSec/Twist"),
        )

        taper = (tip_chord / root_chord) if root_chord > 0 else 1.0
        aspect_ratio = span**2 / area if span > 0 and area > 0 else 0.0

        return WingComponent(
            name=name,
            geom_id=geom_id,
            geom_type=geom_type,
            span=span,
            area=area,
            aspect_ratio=aspect_ratio,
            root_chord=root_chord,
            tip_chord=tip_chord,
            taper_ratio=taper,
            sweep_le=sweep,
            dihedral=dihedral,
            twist_tip=twist,
            raw_params=raw,
        )

    def _parse_fuselage(
        self,
        name: str,
        geom_id: str,
        raw: dict[str, float],
    ) -> FuselageComponent:
        """
        Extract fuselage dimensions.

        Many OpenVSP fuselage files do not expose a single global ``Max_Diameter``
        parameter. The bounding-box extents are a more reliable fallback for
        notebook-level summaries, and they stay consistent with the saved model.
        """
        length = 0.0
        for key in ("Design/Length", "BBox/X_Len"):
            if key in raw:
                length = float(raw[key])
                break

        max_diameter = max(
            float(raw.get("BBox/Y_Len", 0.0)),
            float(raw.get("BBox/Z_Len", 0.0)),
            float(raw.get("Design/Max_Diameter", 0.0)),
            float(raw.get("Design/Diameter", 0.0)),
        )

        fineness_ratio = (length / max_diameter) if max_diameter > 0 else 0.0
        return FuselageComponent(
            name=name,
            geom_id=geom_id,
            length=length,
            max_diameter=max_diameter,
            fineness_ratio=fineness_ratio,
            raw_params=raw,
        )

    def _parse_propeller(
        self,
        name: str,
        geom_id: str,
        raw: dict[str, float],
    ) -> PropellerComponent:
        diameter = float(raw.get("Design/Diameter", raw.get("XSecCurve/Diameter", 0.0)))
        num_blades = int(raw.get("Design/NumBlade", 0.0))
        design_rpm = float(raw.get("Rotor/RotorRPM", 0.0))
        return PropellerComponent(
            name=name,
            geom_id=geom_id,
            diameter=diameter,
            num_blades=num_blades,
            design_rpm=design_rpm,
            raw_params=raw,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def geometry_table(self) -> "pd.DataFrame":
        """Return a DataFrame summarizing the main recognized geometry."""
        rows = []
        for name, comp in self.components.items():
            if isinstance(comp, WingComponent):
                row = comp.to_series()
                row["role"] = self._classify_role(name, comp.geom_type)
                rows.append(row)
            elif isinstance(comp, FuselageComponent):
                row = comp.to_series()
                row["role"] = "fuselage"
                rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).fillna("-")

    def summary(self) -> str:
        """Return a human-readable summary of the aircraft model."""
        lines = [f"\n{'=' * 60}", f"  Aircraft Model: {self.path.name}", f"{'=' * 60}"]
        if self.wing:
            wing = self.wing
            lines.extend(
                [
                    "",
                    "  Main Wing:",
                    f"    Span:         {wing.span:.3f} m",
                    f"    Area:         {wing.area:.3f} m^2",
                    f"    Aspect ratio: {wing.aspect_ratio:.2f}",
                    f"    MAC:          {wing.mac:.3f} m",
                    f"    Sweep LE:     {wing.sweep_le:.1f} deg",
                    f"    Taper ratio:  {wing.taper_ratio:.3f}",
                ]
            )
        if self.htp:
            htp = self.htp
            lines.extend(
                [
                    "",
                    "  Horizontal Tail:",
                    f"    Span:         {htp.span:.3f} m",
                    f"    Area:         {htp.area:.3f} m^2",
                ]
            )
        if self.vtp:
            vtp = self.vtp
            lines.extend(
                [
                    "",
                    "  Vertical Tail:",
                    f"    Span:         {vtp.span:.3f} m",
                    f"    Area:         {vtp.area:.3f} m^2",
                ]
            )
        if self.fuselage:
            fuselage = self.fuselage
            lines.extend(
                [
                    "",
                    "  Fuselage:",
                    f"    Length:       {fuselage.length:.3f} m",
                    f"    Max diameter: {fuselage.max_diameter:.3f} m",
                    f"    Fineness:     {fuselage.fineness_ratio:.2f}",
                ]
            )
        if self.propellers:
            lines.append(f"\n  Propellers: {len(self.propellers)} unit(s)")
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Delegates to wrapper for analysis
    # ------------------------------------------------------------------

    def reference_quantities(self) -> dict[str, float]:
        """Return the VSPAERO reference quantities currently stored in the model."""
        return self._wrapper.get_reference_quantities()

    def mass_properties(
        self,
        *,
        num_slices: int = 100,
        working_dir: str | Path | None = None,
    ) -> "MassProperties":
        """Run MassProp through the wrapper and return structured mass data."""
        return self._wrapper.run_mass_properties(
            num_slices=num_slices,
            working_dir=working_dir,
        )

    @property
    def wrapper(self) -> "VSPWrapper":
        """Direct access to the underlying wrapper for advanced usage."""
        return self._wrapper

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "not loaded"
        return f"AircraftModel('{self.path.name}', {state})"
