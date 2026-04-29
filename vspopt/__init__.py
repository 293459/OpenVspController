"""
vspopt — OpenVSP aerodynamic analysis and optimization toolkit
==============================================================

Quick start
-----------
>>> from vspopt import AircraftModel, DesignVariable, ObjectiveSpec
>>> from vspopt import run_bayesian_optimization, run_gradient_optimization
>>> from vspopt import plot_polar, plot_drag_polar, plot_optimization_history

>>> model = AircraftModel("models/my_aircraft.vsp3").load()
>>> print(model.summary())

>>> results = model.wrapper.run_vspaero_sweep(alpha_start=-5, alpha_end=20, alpha_npts=26)
>>> print(results.LD_max)

>>> dvars = [DesignVariable("Span", "Wing", "Span", "WingGeom", 8.0, 18.0, 12.0, "m")]
>>> obj   = ObjectiveSpec([("LD_max", -1.0)])
>>> opt   = run_bayesian_optimization(model.wrapper, dvars, obj, n_trials=60)
"""

from vspopt.model          import AircraftModel, WingComponent, FuselageComponent
from vspopt.wrapper        import VSPWrapper, OpenVSPError
from vspopt.vspaero        import VSPAEROResults
from vspopt.optimization   import (
    DesignVariable, ObjectiveSpec, OptimizationResult,
    run_gradient_optimization, run_bayesian_optimization,
    run_two_phase_optimization, compare_results, validate_design_variables,
)
from vspopt.postprocess    import (
    CD0Extraction, cd0_design_driver_table, estimate_cd0,
    estimate_cd0_breakdown, extract_cd0, extract_cd0_details,
    MassProperties, check_history_convergence, parse_stab_file,
    read_history_file, stability_records_to_dataframe,
)
from vspopt.plotting       import (
    plot_polar, plot_drag_polar, plot_ld_ratio,
    plot_optimization_history, plot_variable_sensitivity,
    plot_comparison_bar, plot_sweep_grid, plot_wake_convergence, export_all,
)
from vspopt.openvsp_runtime import (
    configure_embedded_openvsp, detect_supported_python_versions,
    format_supported_python_versions, get_default_openvsp_root,
)
from vspopt.reporting      import (
    build_case_summary_row, collect_case_tables, export_case_collection,
    print_case_collection_summary,
)
from vspopt.analytical_checks import (
    compare_analytical_to_openvsp, downwash_gradient_datcom,
    lift_curve_slope_helmbold, neutral_point_location,
    pitching_moment_slope, run_basic_stability_checks,
    static_margin, total_lift_curve_slope,
)
from vspopt.utils          import (
    setup_logging, check_vsp3_integrity, check_openvsp_version,
    check_polar_sanity, results_to_markdown_table, print_banner,
)
from vspopt.notebook_helpers import (
    validate_plot_data, validate_vspaero_results,
    safe_plot_polar, safe_plot_drag_polar, safe_plot_ld_ratio,
    print_results_diagnostic,
    create_complete_parameters_table,
    create_hierarchical_treemap, create_conceptual_mindmap,
    print_conceptual_summary, print_conceptual_summary_sample,
)

__version__ = "0.1.0"
__all__ = [
    "AircraftModel", "WingComponent", "FuselageComponent",
    "VSPWrapper", "OpenVSPError",
    "VSPAEROResults",
    "CD0Extraction", "MassProperties",
    "DesignVariable", "ObjectiveSpec", "OptimizationResult",
    "run_gradient_optimization", "run_bayesian_optimization",
    "run_two_phase_optimization", "compare_results", "validate_design_variables",
    "cd0_design_driver_table", "estimate_cd0", "estimate_cd0_breakdown",
    "extract_cd0", "extract_cd0_details",
    "check_history_convergence", "parse_stab_file",
    "read_history_file", "stability_records_to_dataframe",
    "plot_polar", "plot_drag_polar", "plot_ld_ratio",
    "plot_optimization_history", "plot_variable_sensitivity",
    "plot_comparison_bar", "plot_sweep_grid", "plot_wake_convergence", "export_all",
    "configure_embedded_openvsp", "detect_supported_python_versions",
    "format_supported_python_versions", "get_default_openvsp_root",
    "build_case_summary_row", "collect_case_tables", "export_case_collection",
    "print_case_collection_summary",
    "compare_analytical_to_openvsp", "downwash_gradient_datcom",
    "lift_curve_slope_helmbold", "neutral_point_location",
    "pitching_moment_slope", "run_basic_stability_checks",
    "static_margin", "total_lift_curve_slope",
    "setup_logging", "check_vsp3_integrity", "check_openvsp_version",
    "check_polar_sanity", "results_to_markdown_table", "print_banner",
    # notebook helpers
    "validate_plot_data", "validate_vspaero_results",
    "safe_plot_polar", "safe_plot_drag_polar", "safe_plot_ld_ratio",
    "print_results_diagnostic",
    "create_complete_parameters_table",
    "create_hierarchical_treemap", "create_conceptual_mindmap",
    "print_conceptual_summary", "print_conceptual_summary_sample",
]
