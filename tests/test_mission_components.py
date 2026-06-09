# tests/test_mission_components.py

"""Tests for OpenMDAO FAST mission primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    CruiseBreguetEfficiencyTriplet,
    CruiseBreguetSourceEnergy,
    CruiseTimeTargetDistance,
    FlightConditions,
)
from fast_python.mission import (
    compute_flight_conditions,
    cruise_breguet_efficiency_triplet,
    cruise_breguet_source_energy,
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


def test_cruise_breguet_source_energy_matches_fast_python():
    """Check CruiseBRE source-energy allocation parity with FAST-Python."""

    values = make_breguet_source_energy_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "source",
        CruiseBreguetSourceEnergy(src_type=values["src_type"], npoint=4),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("initial_source_energy", values["initial_energy"], units="J")
    problem.set_val(
        "initial_source_energy_left",
        values["initial_energy_left"],
        units="J",
    )
    problem.set_val("fuel_energy", values["fuel_energy"], units="J")
    problem.set_val("battery_energy", values["battery_energy"], units="J")
    problem.run_model()

    expected_energy, expected_left = cruise_breguet_source_energy(
        values["specs"],
        values["history"],
        1,
        4,
        values["fuel_energy"],
        values["battery_energy"],
    )

    assert np.allclose(problem.get_val("source_energy", units="J"), expected_energy)
    assert np.allclose(problem.get_val("source_energy_left", units="J"), expected_left)


def test_cruise_breguet_source_energy_declares_analytic_partials():
    """Check CruiseBRE source-energy allocation derivatives."""

    values = make_breguet_source_energy_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "source",
        CruiseBreguetSourceEnergy(src_type=values["src_type"], npoint=4),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("initial_source_energy", values["initial_energy"], units="J")
    problem.set_val(
        "initial_source_energy_left",
        values["initial_energy_left"],
        units="J",
    )
    problem.set_val("fuel_energy", values["fuel_energy"], units="J")
    problem.set_val("battery_energy", values["battery_energy"], units="J")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-3,
    )

    for partial_data in partials["source"].values():
        assert partial_data["abs error"].forward < 1.0e-5


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


def make_breguet_source_energy_values():
    """Return source-energy allocation values and FAST-shaped inputs."""

    src_type = np.asarray([1.0, 0.0, 0.0])
    initial_energy = np.asarray([100.0, 20.0, 40.0])
    initial_energy_left = np.asarray([900.0, 500.0, 700.0])
    fuel_energy = np.asarray([10.0, 70.0, 130.0, 190.0])
    battery_energy = np.asarray([30.0, 90.0, 150.0, 210.0])

    return {
        "src_type": src_type,
        "initial_energy": initial_energy,
        "initial_energy_left": initial_energy_left,
        "fuel_energy": fuel_energy,
        "battery_energy": battery_energy,
        "specs": {
            "Propulsion": {
                "PropArch": {
                    "SrcType": src_type,
                },
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 1000.0,
                    "Batt": 1000.0,
                },
            },
            "Weight": {
                "Fuel": 1.0,
                "Batt": [1.0, 1.0],
            },
        },
        "history": {
            "Energy": {
                "E_ES": [
                    [0.0, 0.0, 0.0],
                    initial_energy,
                ],
                "Eleft_ES": [
                    [0.0, 0.0, 0.0],
                    initial_energy_left,
                ],
            },
        },
    }
