"""
app/streamlit_app.py
--------------------
Streamlit web interface for the openvsp-aero-optimizer project.

Run with:
    streamlit run app/streamlit_app.py

This app provides:
  - Model file upload / path input
  - Analysis parameter controls (sliders)
  - Live polar sweep execution
  - Interactive Plotly charts
  - Design variable setup for optimisation
  - Optimisation run with live progress
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from vspopt import (
    AircraftModel, DesignVariable, ObjectiveSpec,
    run_gradient_optimization, run_bayesian_optimization,
    run_two_phase_optimization, check_vsp3_integrity,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OpenVSP Aero Optimizer",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✈ OpenVSP Aerodynamic Optimizer")
st.markdown(
    "_Automated aerodynamic analysis and shape optimisation via OpenVSP 3.48.2 Python API_"
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    # Model path
    model_path_str = st.text_input(
        "Model path (.vsp3)",
        value="models/your_aircraft.vsp3",
        help="Path relative to the project root.",
    )
    model_path = Path(model_path_str)

    st.divider()
    st.subheader("Sweep parameters")
    alpha_start = st.slider("Alpha start [deg]", -15.0, 0.0, -5.0, 0.5)
    alpha_end   = st.slider("Alpha end [deg]", 10.0, 30.0, 20.0, 0.5)
    alpha_npts  = st.slider("Alpha points", 5, 50, 26, 1)
    mach        = st.slider("Mach number", 0.05, 0.85, 0.20, 0.01)
    re_cref     = st.number_input("Reynolds (Re_cref)", 1e4, 1e8, 1e6, format="%.0e")
    wake_iter   = st.slider("Wake iterations", 1, 10, 5)
    use_massprop_cg = st.checkbox(
        "Use MassProp CG",
        value=True,
        help="Run OpenVSP MassProp first and feed the computed CG into VSPAERO.",
    )

    st.divider()
    st.subheader("Optimisation")
    n_trials    = st.slider("Bayesian trials", 10, 200, 60, 5)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_model, tab_polar, tab_sweep, tab_optim = st.tabs(
    ["Model", "Polar Analysis", "Parametric Sweep", "Optimisation"]
)

# ── Tab 1: Model ─────────────────────────────────────────────────────────

with tab_model:
    st.header("Aircraft Model")

    ok, msg = check_vsp3_integrity(model_path)
    if ok:
        st.success(f"Model file OK: {msg}")
    else:
        st.error(f"Model file issue: {msg}")
        st.stop()

    if st.button("Load model"):
        with st.spinner("Loading..."):
            try:
                model = AircraftModel(model_path).load()
                st.session_state["model"] = model
                st.success("Model loaded.")
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")

    if "model" in st.session_state:
        m = st.session_state["model"]
        st.text(m.summary())
        st.subheader("Geometry table")
        st.dataframe(m.geometry_table(), use_container_width=True)

        st.subheader("Reference quantities")
        st.json(m.reference_quantities())

        if st.button("Run mass properties"):
            try:
                mass_props = m.mass_properties()
                st.dataframe(
                    mass_props.to_series().rename("value").to_frame(),
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"MassProp failed: {exc}")

        if m.wing:
            with st.expander(f"All parameters: {m.wing.name}"):
                params = m.wrapper.get_all_params(m.wing.name)
                st.dataframe(
                    pd.Series(params, name="value").to_frame(),
                    use_container_width=True,
                )

# ── Tab 2: Polar Analysis ─────────────────────────────────────────────────

with tab_polar:
    st.header("Aerodynamic Polar")

    if "model" not in st.session_state:
        st.info("Load a model in the Model tab first.")
    else:
        if st.button("Run VSPAERO sweep"):
            model = st.session_state["model"]
            with st.spinner("Running VSPAERO..."):
                try:
                    results = model.wrapper.run_vspaero_sweep(
                        alpha_start=alpha_start, alpha_end=alpha_end,
                        alpha_npts=alpha_npts, mach=mach,
                        re_cref=re_cref, wake_iter=wake_iter,
                        use_massprop_cg=use_massprop_cg,
                    )
                    st.session_state["baseline"] = results
                except Exception as exc:
                    st.error(str(exc))

        if "baseline" in st.session_state:
            res = st.session_state["baseline"]

            # Performance summary
            st.subheader("Performance metrics")
            cols = st.columns(4)
            cols[0].metric("L/D max", f"{res.LD_max:.2f}")
            cols[1].metric("α @ L/D max", f"{res.alpha_at_LD_max:.1f}°")
            cols[2].metric("CL_α [1/deg]", f"{res.CL_alpha:.4f}")
            cols[3].metric("Oswald e", f"{res.oswald_mean:.3f}")

            st.subheader("Reference and diagnostics")
            diag_cols = st.columns(3)
            diag_cols[0].metric("Sref [m^2]", f"{res.Sref:.4f}")
            diag_cols[1].metric("bref [m]", f"{res.bref:.4f}")
            diag_cols[2].metric("cref [m]", f"{res.cref:.4f}")

            if res.convergence:
                status = "OK" if res.converged else "Check"
                st.write(
                    f"Convergence: {status} | iterations = {res.convergence.get('n_iter', 0)} | "
                    f"reason = {res.convergence.get('reason', 'n/a')}"
                )

            if np.isfinite(res.static_margin):
                st.write(
                    f"Static margin = {res.static_margin:.4f} | "
                    f"X_np = {res.neutral_point_x:.4f} m"
                )

            df = res.to_dataframe()
            alpha_col = "alpha [deg]"

            # CL, CD, CM plots
            fig_cl = px.line(df, x=alpha_col, y="CL [-]", title="CL vs α", markers=True)
            fig_cd = px.line(df, x=alpha_col, y="CD [-]", title="CD vs α", markers=True)
            fig_ld = px.line(df, x=alpha_col, y="L/D [-]", title="L/D vs α", markers=True)
            fig_dp = px.line(df, x="CD [-]",  y="CL [-]", title="Drag polar (CL vs CD)", markers=True)

            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_cl, use_container_width=True)
            c2.plotly_chart(fig_cd, use_container_width=True)
            c1.plotly_chart(fig_ld, use_container_width=True)
            c2.plotly_chart(fig_dp, use_container_width=True)

            st.subheader("Raw data")
            st.dataframe(df.round(5), use_container_width=True)

            stab_df = res.stability_dataframe()
            if not stab_df.empty:
                st.subheader("Stability derivatives")
                st.dataframe(stab_df.round(5), use_container_width=True)

# ── Tab 3: Parametric Sweep ───────────────────────────────────────────────

with tab_sweep:
    st.header("Parametric Sweep (OAT sensitivity)")

    if "model" not in st.session_state:
        st.info("Load a model in the Model tab first.")
    else:
        model = st.session_state["model"]
        comp_names = model.wrapper.geom_names

        col1, col2, col3 = st.columns(3)
        sweep_comp  = col1.selectbox("Component", comp_names)
        sweep_group = col2.text_input("Group name", value="WingGeom")
        sweep_param = col3.text_input("Param name", value="Span")

        val_min = st.number_input("Min value", value=8.0)
        val_max = st.number_input("Max value", value=18.0)
        n_sweep = st.slider("Number of steps", 3, 15, 6)
        y_key   = st.selectbox("Plot y-axis", ["CL", "CD", "LD", "CM", "CDi"])

        if st.button("Run parametric sweep"):
            sweep_vals = np.linspace(val_min, val_max, n_sweep)
            try:
                orig = model.wrapper.get_param(sweep_comp, sweep_param, sweep_group)
            except Exception as e:
                st.error(f"Cannot read parameter: {e}")
                st.stop()

            sweep_rows = []
            prog = st.progress(0)
            for i, v in enumerate(sweep_vals):
                model.wrapper.set_param(sweep_comp, sweep_param, sweep_group, v)
                r = model.wrapper.run_vspaero_sweep(
                    alpha_start=alpha_start, alpha_end=alpha_end,
                    alpha_npts=alpha_npts, mach=mach,
                    re_cref=re_cref, wake_iter=wake_iter,
                    use_massprop_cg=use_massprop_cg,
                )
                for a, y in zip(r.alpha, getattr(r, y_key)):
                    sweep_rows.append({"alpha [deg]": a, "value": y, sweep_param: round(v, 3)})
                prog.progress((i + 1) / n_sweep)

            model.wrapper.set_param(sweep_comp, sweep_param, sweep_group, orig)
            df_sw = pd.DataFrame(sweep_rows)
            fig = px.line(
                df_sw, x="alpha [deg]", y="value", color=sweep_param,
                title=f"{y_key} vs alpha — {sweep_param} sweep",
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Optimisation ───────────────────────────────────────────────────

with tab_optim:
    st.header("Aerodynamic Shape Optimisation")

    if "model" not in st.session_state:
        st.info("Load a model in the Model tab first.")
    else:
        model = st.session_state["model"]
        st.subheader("Design variables")
        st.markdown(
            "Edit the table below.  `geom_name`, `parm_name`, `group_name` must match "
            "the exact names shown in the Model tab."
        )

        dv_df = pd.DataFrame([
            {"label": "Wing span",  "geom": "Wing", "parm": "Span",  "group": "WingGeom",
             "lower": 8.0, "upper": 18.0, "initial": getattr(model.wing, "span", 12.0)},
            {"label": "LE sweep",   "geom": "Wing", "parm": "Sweep", "group": "XSec_1",
             "lower": 0.0, "upper": 45.0, "initial": getattr(model.wing, "sweep_le", 15.0)},
        ])
        edited_dv = st.data_editor(dv_df, num_rows="dynamic", use_container_width=True)

        method = st.selectbox("Optimisation method",
                               ["Two-phase (Bayesian → SLSQP)", "Bayesian only", "Gradient only"])

        if st.button("Run optimisation"):
            dvars = []
            for _, row in edited_dv.iterrows():
                try:
                    dvars.append(DesignVariable(
                        label=row["label"],
                        geom_name=row["geom"], parm_name=row["parm"], group_name=row["group"],
                        lower=float(row["lower"]), upper=float(row["upper"]),
                        initial=float(row["initial"]),
                    ))
                except Exception as e:
                    st.error(f"Invalid design variable '{row['label']}': {e}")
                    st.stop()

            objective = ObjectiveSpec(metrics=[("LD_max", -1.0)])
            sw_kw = dict(alpha_start=alpha_start, alpha_end=alpha_end,
                         alpha_npts=alpha_npts, mach=mach, re_cref=re_cref,
                         wake_iter=wake_iter, use_massprop_cg=use_massprop_cg)

            with st.spinner("Optimising — this may take several minutes..."):
                try:
                    if method.startswith("Two"):
                        r1, r2 = run_two_phase_optimization(
                            model.wrapper, dvars, objective, sw_kw,
                            n_bayesian_trials=n_trials // 2,
                        )
                        opt = r2
                    elif method.startswith("Bayesian"):
                        opt = run_bayesian_optimization(
                            model.wrapper, dvars, objective, sw_kw, n_trials=n_trials,
                        )
                    else:
                        opt = run_gradient_optimization(model.wrapper, dvars, objective, sw_kw)

                    st.success(f"Done — best objective = {opt.best_objective:.4f}")
                    st.json({dv.label: round(float(xv), 4)
                             for dv, xv in zip(dvars, opt.best_x)})

                    # Convergence chart
                    df_hist = pd.DataFrame({
                        "Evaluation": range(1, len(opt.history_obj) + 1),
                        "Objective":  opt.history_obj,
                        "Best so far": np.minimum.accumulate(opt.history_obj),
                    })
                    fig_conv = px.line(df_hist, x="Evaluation", y=["Objective", "Best so far"],
                                       title="Optimisation convergence")
                    st.plotly_chart(fig_conv, use_container_width=True)

                except Exception as exc:
                    st.error(f"Optimisation failed: {exc}")
