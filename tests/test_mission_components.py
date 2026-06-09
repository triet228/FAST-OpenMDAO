# tests/test_mission_components.py

"""Tests for OpenMDAO FAST mission primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    CruiseBreguetEfficiencyTriplet,
    CruiseTimeTargetDistance,
    FlightConditions,
)
from fast_python.mission import (
    compute_flight_conditions,
    cruise_breguet_efficiency_triplet,
    cruise_time_target_to_distance,
)


def test_flight_conditions_match_fast_python_for_velocity_types():
    """Check flight-condition parity for TAS, EAS, and Mach inputs."""

    cases = [
        ("TAS", 100.0),
        ("EAS", 95.0),
        ("Mach", 0.35),
    ]

    for velocity_type, velocity in cases:
        problem = om.Problem()
        problem.model.add_subsystem(
            "conditions",
            FlightConditions(velocity_type=velocity_type),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("altitude", 1000.0, units="m")
        problem.set_val("disa", 5.0, units="K")
        problem.set_val("velocity", velocity)
        problem.run_model()

        expected = compute_flight_conditions(1000.0, 5.0, velocity_type, velocity)

        for name, expected_value in zip(output_names(), expected):
            assert np.isclose(problem.get_val(name)[0], expected_value)


def test_flight_conditions_declares_analytic_partials():
    """Check flight-condition analytical partials against finite difference."""

    for velocity_type, velocity in [("TAS", 100.0), ("EAS", 95.0), ("Mach", 0.35)]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "conditions",
            FlightConditions(velocity_type=velocity_type),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("altitude", 1000.0, units="m")
        problem.set_val("disa", 5.0, units="K")
        problem.set_val("velocity", velocity)
        problem.run_model()

        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-4,
        )

        for partial_data in partials["conditions"].values():
            assert partial_data["abs error"].forward < 1.0e-5


def test_cruise_time_target_distance_matches_fast_python():
    """Check cruise time target conversion parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "target",
        CruiseTimeTargetDistance(type_begin="TAS", type_end="EAS"),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("altitude_begin", 1000.0, units="m")
    problem.set_val("altitude_end", 2000.0, units="m")
    problem.set_val("velocity_begin", 100.0)
    problem.set_val("velocity_end", 90.0)
    problem.set_val("target_minutes", 10.0, units="min")
    problem.run_model()

    mission = make_cruise_target_mission()

    assert np.isclose(
        problem.get_val("distance", units="m")[0],
        cruise_time_target_to_distance(mission, 0, 10.0),
    )


def test_cruise_time_target_distance_declares_analytic_partials():
    """Check cruise time target conversion partials against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "target",
        CruiseTimeTargetDistance(type_begin="TAS", type_end="EAS"),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("altitude_begin", 1000.0, units="m")
    problem.set_val("altitude_end", 2000.0, units="m")
    problem.set_val("velocity_begin", 100.0)
    problem.set_val("velocity_end", 90.0)
    problem.set_val("target_minutes", 10.0, units="min")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-5,
    )

    for partial_data in partials["target"].values():
        assert partial_data["abs error"].forward < 1.0e-5


def test_cruise_breguet_efficiency_triplet_matches_fast_python():
    """Check CruiseBRE efficiency triplet parity for each architecture."""

    eta_prop = 0.84
    eta_em = 0.95
    eta_eg = 0.91
    eta_gt = 0.36

    for architecture in ("AC", "PHE", "SHE", "TE"):
        problem = om.Problem()
        problem.model.add_subsystem(
            "efficiency",
            CruiseBreguetEfficiencyTriplet(architecture=architecture),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("propulsive_efficiency", eta_prop)
        problem.set_val("electric_motor_efficiency", eta_em)
        problem.set_val("electric_generator_efficiency", eta_eg)
        problem.set_val("gas_turbine_efficiency", eta_gt)
        problem.run_model()

        expected = cruise_breguet_efficiency_triplet(
            architecture,
            eta_prop,
            eta_em,
            eta_eg,
            eta_gt,
        )

        for name, expected_value in zip(("eta1", "eta2", "eta3"), expected):
            assert np.isclose(problem.get_val(name)[0], expected_value)


def test_cruise_breguet_efficiency_triplet_declares_analytic_partials():
    """Check CruiseBRE efficiency triplet derivatives by architecture."""

    for architecture in ("AC", "PHE", "SHE", "TE"):
        problem = om.Problem()
        problem.model.add_subsystem(
            "efficiency",
            CruiseBreguetEfficiencyTriplet(architecture=architecture),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("propulsive_efficiency", 0.84)
        problem.set_val("electric_motor_efficiency", 0.95)
        problem.set_val("electric_generator_efficiency", 0.91)
        problem.set_val("gas_turbine_efficiency", 0.36)
        problem.run_model()

        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-6,
        )

        for partial_data in partials["efficiency"].values():
            assert partial_data["abs error"].forward < 1.0e-8


def output_names():
    """Return output names in FAST-Python flight-condition order."""

    return (
        "eas",
        "tas",
        "mach",
        "temperature",
        "pressure",
        "density",
        "viscosity",
    )


def make_cruise_target_mission():
    """Return minimal mission profile for cruise time-target conversion."""

    return {
        "AltBeg": [1000.0],
        "AltEnd": [2000.0],
        "VelBeg": [100.0],
        "VelEnd": [90.0],
        "TypeBeg": ["TAS"],
        "TypeEnd": ["EAS"],
    }
