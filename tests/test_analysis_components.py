# tests/test_analysis_components.py

"""Tests for OpenMDAO FAST analysis helper components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

import fast_python.analysis as fast_analysis
from fast_openmdao import (
    AnalysisWeightUpdate,
    ConvergenceError,
    SourceWeightVector,
    WeightSum,
    WingAreaFromLoading,
)
from fast_python.analysis import convergence_error, initial_source_weight, sum_weight


def test_convergence_error_matches_fast_python():
    """Check scalar convergence error parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem("convergence", ConvergenceError(), promotes=["*"])
    problem.setup()
    problem.set_val("delta", -2.0)
    problem.set_val("baseline", 10.0)
    problem.run_model()

    expected = convergence_error(np.asarray([-2.0]), np.asarray([10.0]))[0]

    assert np.isclose(problem.get_val("convergence_error")[0], expected)


def test_convergence_error_declares_analytic_partials():
    """Check convergence error analytical partials against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem("convergence", ConvergenceError(), promotes=["*"])
    problem.setup()
    problem.set_val("delta", -2.0)
    problem.set_val("baseline", 10.0)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["convergence"].values():
        assert partial_data["abs error"].forward < 1.0e-8


def test_weight_sum_matches_fast_python():
    """Check source-weight summation parity with FAST-Python."""

    values = np.asarray([12.0, 4.0, 3.5])
    problem = om.Problem()
    problem.model.add_subsystem(
        "weight",
        WeightSum(vec_size=values.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("weight_values", values, units="kg")
    problem.run_model()

    assert np.isclose(problem.get_val("weight_sum", units="kg")[0], sum_weight(values))


def test_weight_sum_declares_analytic_partials():
    """Check source-weight summation derivatives against finite difference."""

    values = np.asarray([12.0, 4.0, 3.5])
    problem = om.Problem()
    problem.model.add_subsystem(
        "weight",
        WeightSum(vec_size=values.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("weight_values", values, units="kg")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["weight"].values():
        assert partial_data["abs error"].forward < 1.0e-8


def test_wing_area_from_loading_matches_fast_python_eap_setup():
    """Check FAST analysis setup wing-area parity."""

    aircraft = make_analysis_weight_update_aircraft()
    expected = run_fast_python_single_iteration(aircraft)
    problem = om.Problem()
    problem.model.add_subsystem("wing", WingAreaFromLoading(), promotes=["*"])
    problem.setup()
    problem.set_val("mtow", 1200.0, units="kg")
    problem.set_val("wing_loading", 120.0, units="kg/m**2")
    problem.run_model()

    assert np.isclose(
        problem.get_val("wing_area", units="m**2")[0],
        expected["Specs"]["Aero"]["S"],
    )


def test_wing_area_from_loading_declares_analytic_partials():
    """Check wing area setup derivatives against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem("wing", WingAreaFromLoading(), promotes=["*"])
    problem.setup()
    problem.set_val("mtow", 1200.0, units="kg")
    problem.set_val("wing_loading", 120.0, units="kg/m**2")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["wing"].values():
        assert partial_data["abs error"].forward < 1.0e-8


def test_source_weight_vector_matches_fast_python():
    """Check FAST source-weight vector expansion parity."""

    scalar_problem = om.Problem()
    scalar_problem.model.add_subsystem(
        "source",
        SourceWeightVector(source_count=3, input_size=1),
        promotes=["*"],
    )
    scalar_problem.setup()
    scalar_problem.set_val("source_weight", [11.0], units="kg")
    scalar_problem.run_model()

    expected_scalar = initial_source_weight(11.0, np.asarray([True, True, True]))

    assert np.allclose(
        scalar_problem.get_val("source_weight_vector", units="kg"),
        expected_scalar,
    )

    vector = np.asarray([11.0, 3.0, 4.0])
    vector_problem = om.Problem()
    vector_problem.model.add_subsystem(
        "source",
        SourceWeightVector(source_count=3, input_size=3),
        promotes=["*"],
    )
    vector_problem.setup()
    vector_problem.set_val("source_weight", vector, units="kg")
    vector_problem.run_model()

    assert np.allclose(
        vector_problem.get_val("source_weight_vector", units="kg"),
        vector,
    )


def test_source_weight_vector_declares_analytic_partials():
    """Check source-weight vector derivatives against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "source",
        SourceWeightVector(source_count=3, input_size=1),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("source_weight", [11.0], units="kg")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["source"].values():
        assert partial_data["abs error"].forward < 1.0e-8


def test_analysis_weight_update_matches_fast_python_eap_iteration():
    """Check FAST analysis-loop weight update parity."""

    aircraft = make_analysis_weight_update_aircraft()
    expected = run_fast_python_single_iteration(aircraft)
    problem = om.Problem()
    problem.model.add_subsystem(
        "update",
        AnalysisWeightUpdate(
            num_fuel_sources=1,
            num_battery_sources=1,
            update_oew=True,
        ),
        promotes=["*"],
    )
    problem.setup()
    set_analysis_weight_update_parity_values(problem)
    problem.run_model()

    assert np.isclose(
        problem.get_val("mtow_new", units="kg")[0],
        expected["Specs"]["Weight"]["MTOW"],
    )
    assert np.allclose(
        problem.get_val("fuel_weight_new", units="kg"),
        [expected["Specs"]["Weight"]["Fuel"]],
    )
    assert np.allclose(
        problem.get_val("battery_weight_new", units="kg"),
        [expected["Specs"]["Weight"]["Batt"]],
    )
    assert np.isclose(
        problem.get_val("oew_new", units="kg")[0],
        expected["Specs"]["Weight"]["OEW"],
    )


def test_analysis_weight_update_declares_analytic_partials():
    """Check analysis-loop weight update derivatives."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "update",
        AnalysisWeightUpdate(
            num_fuel_sources=2,
            num_battery_sources=2,
            update_oew=True,
        ),
        promotes=["*"],
    )
    problem.setup()
    set_analysis_weight_update_values(problem)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["update"].values():
        assert partial_data["abs error"].forward < 1.0e-6


def set_analysis_weight_update_values(problem):
    """Set shared analysis weight-update test values."""

    problem.set_val("mtow", 1200.0, units="kg")
    problem.set_val("fuel_weight", [100.0, 80.0], units="kg")
    problem.set_val("battery_weight", [50.0, 40.0], units="kg")
    problem.set_val("fuel_burn", [88.0, 70.0], units="kg")
    problem.set_val("resized_battery_weight", [54.0, 45.0], units="kg")
    problem.set_val("payload_weight", 220.0, units="kg")
    problem.set_val("crew_weight", 90.0, units="kg")
    problem.set_val("oew", 710.0, units="kg")


def set_analysis_weight_update_parity_values(problem):
    """Set scalar-source values for FAST-Python eap_analysis parity."""

    problem.set_val("mtow", 1200.0, units="kg")
    problem.set_val("fuel_weight", [100.0], units="kg")
    problem.set_val("battery_weight", [50.0], units="kg")
    problem.set_val("fuel_burn", [88.0], units="kg")
    problem.set_val("resized_battery_weight", [54.0], units="kg")
    problem.set_val("payload_weight", 220.0, units="kg")
    problem.set_val("crew_weight", 90.0, units="kg")
    problem.set_val("oew", 740.0, units="kg")


def make_analysis_weight_update_aircraft():
    """Return a minimal aircraft for one controlled FAST analysis iteration."""

    return {
        "Settings": {
            "Analysis": {
                "Type": 0,
                "MaxIter": 1,
            },
        },
        "Specs": {
            "Weight": {
                "MTOW": 1200.0,
                "OEW": 710.0,
                "Crew": 90.0,
                "Payload": 220.0,
                "Fuel": 100.0,
                "Batt": 50.0,
            },
            "Aero": {
                "W_S": {
                    "SLS": 120.0,
                },
            },
            "Power": {
                "Battery": {
                    "SerCells": np.nan,
                    "ParCells": np.nan,
                },
            },
            "Propulsion": {
                "PropArch": {
                    "SrcType": [1.0, 0.0],
                },
            },
        },
        "Mission": {
            "History": {
                "SI": {
                    "Weight": {
                        "Fburn": [[88.0]],
                    },
                },
            },
        },
    }


def run_fast_python_single_iteration(aircraft):
    """Run FAST-Python eap_analysis with deterministic dependency stubs."""

    original_init = fast_analysis.init_mission_history
    original_propulsion = fast_analysis.propulsion_sizing
    original_fly = fast_analysis.fly_mission
    original_resize = fast_analysis.resize_battery

    def init_stub(value):
        return value

    def propulsion_stub(value):
        return value

    def fly_stub(value):
        return value

    def resize_stub(value):
        value["Specs"]["Weight"]["Batt"] = 54.0
        return value

    try:
        fast_analysis.init_mission_history = init_stub
        fast_analysis.propulsion_sizing = propulsion_stub
        fast_analysis.fly_mission = fly_stub
        fast_analysis.resize_battery = resize_stub
        return fast_analysis.eap_analysis(aircraft)
    finally:
        fast_analysis.init_mission_history = original_init
        fast_analysis.propulsion_sizing = original_propulsion
        fast_analysis.fly_mission = original_fly
        fast_analysis.resize_battery = original_resize
