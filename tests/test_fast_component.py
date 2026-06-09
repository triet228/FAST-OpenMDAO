# tests/test_fast_component.py

"""Tests for FAST-Python OpenMDAO components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    FastPythonComponent,
    make_fast_optimization_problem,
    make_fast_problem,
)
from tests.fixtures import make_compact_aircraft, make_compact_mission


def test_component_maps_openmdao_inputs_to_fast_paths():
    """Check scalar OpenMDAO inputs mutate aircraft and mission copies."""

    problem = make_fast_problem(
        aircraft=make_fake_aircraft(),
        mission=make_fake_mission(),
        input_specs=fake_input_specs(),
        output_specs=fake_output_specs(),
        runner=fake_runner,
    )
    problem.setup()
    problem.set_val("mission_range", 40000.0, units="m")
    problem.set_val("cruise_lift_to_drag", 20.0)
    problem.run_model()

    assert abs(problem.get_val("mtow", units="kg")[0] - 3000.0) < 1.0e-9
    assert abs(problem.get_val("output_mtow", units="kg")[0] - 3000.0) < 1.0e-9


def test_component_supports_openmdao_total_derivatives():
    """Check finite-difference partials make totals available to drivers."""

    problem = make_fast_problem(
        aircraft=make_fake_aircraft(),
        mission=make_fake_mission(),
        input_specs=fake_input_specs(),
        output_specs=fake_output_specs(),
        runner=fake_runner,
    )
    problem.model.add_design_var("mission_range")
    problem.model.add_design_var("cruise_lift_to_drag")
    problem.model.add_objective("mtow")
    problem.setup()
    problem.set_val("mission_range", 40000.0, units="m")
    problem.set_val("cruise_lift_to_drag", 20.0)
    problem.run_model()

    totals = problem.compute_totals(
        of=["mtow"],
        wrt=["mission_range", "cruise_lift_to_drag"],
    )

    assert np.allclose(totals[("mtow", "mission_range")], [[0.05]], rtol=1.0e-5)
    assert np.allclose(
        totals[("mtow", "cruise_lift_to_drag")],
        [[-100.0]],
        rtol=1.0e-5,
    )


def test_default_runner_executes_fast_python_native_smoke_case():
    """Check the component can run the installed FAST-Python backend."""

    aircraft = make_compact_aircraft()
    mission = aircraft.pop("Mission")["Profile"]
    problem = make_fast_problem(
        aircraft=aircraft,
        mission=mission,
        input_specs=[
            {
                "name": "mission_range",
                "target": "mission",
                "path": ("Target", "Valu", 0),
                "val": 20000.0,
                "units": "m",
            },
        ],
    )
    problem.setup()
    problem.set_val("mission_range", 20000.0, units="m")
    problem.run_model()

    assert abs(problem.get_val("mtow", units="kg")[0] - 1000.0) < 1.0e-9


def test_component_can_be_added_directly_to_openmdao_problem():
    """Check users can instantiate the component without the builder."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "fast",
        FastPythonComponent(
            aircraft=make_fake_aircraft(),
            mission=make_fake_mission(),
            input_specs=fake_input_specs(),
            output_specs=fake_output_specs(),
            runner=fake_runner,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.run_model()

    assert abs(problem.get_val("mtow", units="kg")[0] - 3000.0) < 1.0e-9


def test_optimization_builder_runs_slsqp_driver():
    """Check the problem builder can drive a scalar FAST design variable."""

    input_specs = fake_input_specs()
    input_specs[0]["val"] = 40000.0
    problem = make_fast_optimization_problem(
        aircraft=make_fake_aircraft(),
        mission=make_fake_mission(),
        input_specs=input_specs,
        output_specs=fake_output_specs(),
        runner=fake_runner,
        design_vars=[
            {
                "name": "cruise_lift_to_drag",
                "lower": 5.0,
                "upper": 20.0,
            },
        ],
        objective={
            "name": "mtow",
        },
        driver_options={
            "maxiter": 20,
            "tol": 1.0e-9,
        },
    )
    problem.setup()
    result = problem.run_driver()

    assert result.success
    assert problem.get_val("cruise_lift_to_drag")[0] > 19.99
    assert abs(problem.get_val("mtow", units="kg")[0] - 3000.0) < 1.0e-5


def fake_input_specs():
    """Return scalar input specs for the fake differentiable runner."""

    return [
        {
            "name": "mission_range",
            "target": "mission",
            "path": ("Target", "Valu", 0),
            "val": 20000.0,
            "units": "m",
            "desc": "Mission range target.",
        },
        {
            "name": "cruise_lift_to_drag",
            "target": "aircraft",
            "path": ("Specs", "Aero", "L_D", "Crs"),
            "val": 10.0,
            "desc": "Cruise lift-to-drag ratio.",
        },
    ]


def fake_output_specs():
    """Return scalar output specs for the fake differentiable runner."""

    return [
        {
            "name": "mtow",
            "path": ("mtow",),
            "units": "kg",
        },
        {
            "name": "output_mtow",
            "path": ("aircraft", "Specs", "Weight", "MTOW"),
            "units": "kg",
        },
    ]


def make_fake_aircraft():
    """Return a tiny FAST-shaped aircraft for deterministic component tests."""

    return {
        "Specs": {
            "Aero": {
                "L_D": {
                    "Crs": 10.0,
                },
            },
            "Weight": {
                "MTOW": 1000.0,
            },
        },
    }


def make_fake_mission():
    """Return a tiny FAST-shaped mission for deterministic component tests."""

    return {
        "Target": {
            "Valu": [20000.0],
        },
    }


def fake_runner(aircraft, mission):
    """Return a differentiable FAST-shaped result for OpenMDAO plumbing tests."""

    base_mtow = aircraft["Specs"]["Weight"]["MTOW"]
    cruise_lift_to_drag = aircraft["Specs"]["Aero"]["L_D"]["Crs"]
    mission_range = mission["Target"]["Valu"][0]
    mtow = base_mtow + mission_range / cruise_lift_to_drag

    return {
        "status": "success",
        "backend": "fake",
        "mtow": mtow,
        "aircraft": {
            "Specs": {
                "Weight": {
                    "MTOW": mtow,
                },
            },
        },
    }
