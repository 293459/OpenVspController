"""
vspopt/model.py
---------------
High-level aircraft model representation.

After loading a .vsp3 file, this module converts the OpenVSP geometry
into a hierarchy of Python objects with named attributes.

Usage
-----
>>> from vspopt.model import AircraftModel
>>> model = AircraftModel("models/my_aircraft.vsp3")
>>> model.load()
>>> print(model.wing.span)
12.5
>>> print(model.summary())
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
    """Geometric parameters of a lifting surface (wing, HTP, VTP)."""
    name: str
    geom_id: str
    geom_type: str

    # Primary wing geometry (read from OpenVSP)
    span: float = 0.0           # [m] Total span
    area: float = 0.0           # [m²] Reference area
    aspect_ratio: float = 0.0   # [-] AR = b² / S
    root_chord: float = 0.0     # [m]
    tip_chord: float = 0.0      # [m]
    taper_ratio: float = 0.0    # [-] λ = c_tip / c_root
    sweep_le: float = 0.0       # [deg] Leading-edge sweep
    dihedral: float = 0.0       # [deg]
    twist_tip: float = 0.0      # [deg] Washout at tip

    # All raw parameters (group/param: value)
    raw_params: dict = field(default_factory=dict)

    @property
    def mac(self) -> float:
        """Mean aerodynamic chord [m] for a simple tapered wing."""
        if abs(self.taper_ratio - 1.0) < 1e-6:
            return self.root_chord
        lam = self.taper_ratio
        return (2 / 3) * self.root_chord * (1 + lam + lam**2) / (1 + lam)

    @property
    def area_half(self) -> float:
        """Semi-span area [m²]."""
        return self.area / 2.0

    def to_series(self) -> "pd.Series":
        """Return a pandas Series of the main geometric parameters."""
        return pd.Series({
            "span [m]":          self.span,
            "area [m²]":         self.area,
            "aspect_ratio [-]":  self.aspect_ratio,
            "root_chord [m]":    self.root_chord,
            "tip_chord [m]":     self.tip_chord,
            "taper_ratio [-]":   self.taper_ratio,
            "mac [m]":           self.mac,
            "sweep_le [deg]":    self.sweep_le,
            "dihedral [deg]":    self.dihedral,
            "twist_tip [deg]":   self.twist_tip,
        }, name=self.name)


@dataclass
class FuselageComponent:
    """Geometric parameters of the fuselage."""
    name: str
    geom_id: str

    length: float = 0.0         # [m]
    max_diameter: float = 0.0   # [m]
    fineness_ratio: float = 0.0 # [-] L / D_max

    raw_params: dict = field(default_factory=dict)

    def to_series(self) -> "pd.Series":
        return pd.Series({
            "length [m]":         self.length,
            "max_diameter [m]":   self.max_diameter,
            "fineness_ratio [-]": self.fineness_ratio,
        }, name=self.name)


@dataclass
class PropellerComponent:
    """Geometric parameters of a propeller or rotor."""
    name: str
    geom_id: str

    diameter: float = 0.0       # [m]
    num_blades: int = 0         # [-]
    design_rpm: float = 0.0     # [rpm]

    raw_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aircraft model class
# ---------------------------------------------------------------------------

class AircraftModel:
    """
    High-level representation of an OpenVSP aircraft model.

    Reads geometry from a .vsp3 file and exposes named components
    (wing, fuselage, htp, vtp, …) as Python objects with typed attributes.

    Parameters
    ----------
    vsp3_path : str | Path
        Path to the .vsp3 model file.

    Examples
    --------
    >>> m = AircraftModel("models/aircraft.vsp3")
    >>> m.load()
    >>> print(m.wing.span)       # 12.5
    >>> print(m.htp.sweep_le)    # 25.0
    >>> df = m.geometry_table()  # pandas DataFrame of all components
    """

    # Heuristic keywords used to auto-detect component roles from their names.
    # The user can always override by accessing components by name.
    _WING_KEYWORDS   = ("wing", "main", "main_wing", "mainwing")
    _HTP_KEYWORDS    = ("htp", "horiz", "horizontal", "stabilizer", "htail", "h_tail")
    _VTP_KEYWORDS    = ("vtp", "vert", "vertical", "fin", "vtail", "v_tail")
    _FUSE_KEYWORDS   = ("fuse", "fuselage", "body", "hull")
    _PROP_KEYWORDS   = ("prop", "rotor", "propeller", "fan")

    def __init__(self, vsp3_path: str | Path) -> None:
        from vspopt.wrapper import VSPWrapper
        self._wrapper = VSPWrapper(vsp3_path)
        self.path = Path(vsp3_path)

        # Named components, populated after load()
        self.wing:       Optional[WingComponent]      = None
        self.htp:        Optional[WingComponent]      = None
        self.vtp:        Optional[WingComponent]      = None
        self.fuselage:   Optional[FuselageComponent]  = None
        self.propellers: list[PropellerComponent]     = []

        # All components by name
        self.components: dict[str, object] = {}

        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "AircraftModel":
        """
        Load the .vsp3 file and populate all component attributes.

        Returns self for method chaining.
        """
        self._wrapper.load()
        self._parse_components()
        self._loaded = True
        logger.info(
            "AircraftModel loaded: %d components (%s)",
            len(self.components),
            ", ".join(self.components.keys()),
        )
        return self

    def setup_aero_set(self, thin_keywords=None, thick_keywords=None) -> "AircraftModel":
        """
        Organizes the aircraft components into Thin (Set 1) and Thick (Set 2) groups.
        Returns self for method chaining.
        """
        if thin_keywords is None:
            thin_keywords = ["Wing", "tail", "Fin", "Stabilizer", "vtail"]
        if thick_keywords is None:
            thick_keywords = ["Fuselage", "Body", "hull", "Fuse"]

        self._wrapper.setup_dual_aero_sets(thin_keywords, thick_keywords)
        return self
    
    
    
    
    def _parse_components(self):
        """Iterate over all geometry components and classify them."""
        vsp = self._wrapper._vsp

        for geom_name, geom_id in self._wrapper._geom_id_cache.items():
            geom_type = vsp.GetGeomTypeName(geom_id)
            raw = self._wrapper.get_all_params(geom_name)
            role = self._classify_role(geom_name, geom_type)

            if geom_type in ("Wing",) or role in ("wing", "htp", "vtp"):
                comp = self._parse_wing(geom_name, geom_id, geom_type, raw)
                self.components[geom_name] = comp
                if role == "wing" and self.wing is None:
                    self.wing = comp
                elif role == "htp" and self.htp is None:
                    self.htp = comp
                elif role == "vtp" and self.vtp is None:
                    self.vtp = comp

            elif geom_type in ("Fuselage", "BodyOfRevolution") or role == "fuselage":
                comp = self._parse_fuselage(geom_name, geom_id, raw)
                self.components[geom_name] = comp
                if self.fuselage is None:
                    self.fuselage = comp

            elif geom_type == "Prop" or role == "propeller":
                comp = self._parse_propeller(geom_name, geom_id, raw)
                self.components[geom_name] = comp
                self.propellers.append(comp)

            else:
                # Store raw data for unknown types
                self.components[geom_name] = {"type": geom_type, "params": raw}

    def _classify_role(self, name: str, geom_type: str) -> str:
        """Heuristic role classification based on component name."""
        n = name.lower().replace("-", "_").replace(" ", "_")
        if any(kw in n for kw in self._WING_KEYWORDS):
            return "wing"
        if any(kw in n for kw in self._HTP_KEYWORDS):
            return "htp"
        if any(kw in n for kw in self._VTP_KEYWORDS):
            return "vtp"
        if any(kw in n for kw in self._FUSE_KEYWORDS):
            return "fuselage"
        if any(kw in n for kw in self._PROP_KEYWORDS):
            return "propeller"
        return geom_type.lower()

    def _parse_wing(
        self, name: str, geom_id: str, geom_type: str, raw: dict
    ) -> WingComponent:
        """Extract wing geometry from raw parameters."""

        def _get(key: str, default: float = 0.0) -> float:
            # Try full key first, then just the parm name part
            if key in raw:
                return raw[key]
            for k, v in raw.items():
                if k.split("/")[-1] == key:
                    return v
            return default

        span   = _get("Span")
        area   = _get("TotalArea")
        ar     = _get("Aspect")
        rc     = _get("Root_Chord") or _get("Chord")
        tc     = _get("Tip_Chord")  or rc * _get("Taper", 1.0)
        sweep  = _get("Sweep")
        dihed  = _get("Dihedral")
        twist  = _get("Twist")

        taper = (tc / rc) if rc > 0 else 1.0
        if ar <= 0 and span > 0 and area > 0:
            ar = span**2 / area

        return WingComponent(
            name=name, geom_id=geom_id, geom_type=geom_type,
            span=span, area=area, aspect_ratio=ar,
            root_chord=rc, tip_chord=tc, taper_ratio=taper,
            sweep_le=sweep, dihedral=dihed, twist_tip=twist,
            raw_params=raw,
        )

    def _parse_fuselage(self, name: str, geom_id: str, raw: dict) -> FuselageComponent:
        # We change the default by putting an absurd number like 99.0
        def _get(key, default=99.0):
            if key in raw:
                return raw[key]
            
            for k, v in raw.items():
                if k.split("/")[-1] == key:
                    return v
            
            # TEST SECTION: If the code gets here, it means the search failed
            print(f"TEST FAILED: Could not find parameter '{key}'. Using default: {default}")
            return default

        # Search for length
        length = _get("Length")
        
        # Search for diameter. If it fails, 'diam' will become 99.0
        diam = _get("Max_Diameter", default=99.0)
        
        # TEST SECTION: Verify if it took the absurd default
        if diam == 99.0:
            print("TEST RESULT: Theory confirmed! OpenVSP does not give us the global diameter.")
            # We put the patch back to 0.16 to avoid crashing the aerodynamics right after
            print("Restoring the forced diameter to 0.16 to continue the calculation.")
            diam = 0.30
            
        fr = (length / diam) if diam > 0 else 0.0

        return FuselageComponent(
            name=name, geom_id=geom_id,
            length=length, max_diameter=diam, fineness_ratio=fr,
            raw_params=raw,
        )
        
        
    def _parse_propeller( 
        self, name: str, geom_id: str, raw: dict
    ) -> PropellerComponent:
        def _get(key, default=0.0):
            if key in raw:
                return raw[key]
            for k, v in raw.items():
                if k.split("/")[-1] == key:
                    return v
            return default

        diameter = _get("Diameter")
        return PropellerComponent(
            name=name, geom_id=geom_id,
            diameter=diameter,
            raw_params=raw,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def geometry_table(self) -> "pd.DataFrame":
        """
        Return a DataFrame summarising the main geometry of all
        recognised lifting surfaces and the fuselage.
        """
        rows = []
        for name, comp in self.components.items():
            if isinstance(comp, WingComponent):
                s = comp.to_series()
                s["role"] = self._classify_role(name, comp.geom_type)
                rows.append(s)
            elif isinstance(comp, FuselageComponent):
                s = comp.to_series()
                s["role"] = "fuselage"
                rows.append(s)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).fillna("—")

    def summary(self) -> str:
        """Return a human-readable summary of the aircraft model."""
        lines = [f"\n{'='*60}", f"  Aircraft Model: {self.path.name}", f"{'='*60}"]
        if self.wing:
            w = self.wing
            lines += [
                f"\n  Main Wing:",
                f"    Span:         {w.span:.3f} m",
                f"    Area:         {w.area:.3f} m²",
                f"    Aspect ratio: {w.aspect_ratio:.2f}",
                f"    MAC:          {w.mac:.3f} m",
                f"    Sweep LE:     {w.sweep_le:.1f}°",
                f"    Taper ratio:  {w.taper_ratio:.3f}",
            ]
        if self.htp:
            h = self.htp
            lines += [
                f"\n  Horizontal Tail (HTP):",
                f"    Span:         {h.span:.3f} m",
                f"    Area:         {h.area:.3f} m²",
            ]
        if self.fuselage:
            f_ = self.fuselage
            lines += [
                f"\n  Fuselage:",
                f"    Length:       {f_.length:.3f} m",
                f"    Max diameter: {f_.max_diameter:.3f} m",
                f"    Fineness:     {f_.fineness_ratio:.2f}",
            ]
        if self.propellers:
            lines.append(f"\n  Propellers: {len(self.propellers)} unit(s)")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Delegate to wrapper for analysis
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
        """Direct access to the underlying VSPWrapper for advanced usage."""
        return self._wrapper

    def __repr__(self) -> str:
        state = "loaded" if self._loaded else "not loaded"
        return f"AircraftModel('{self.path.name}', {state})"
