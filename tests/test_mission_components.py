# tests/test_mission_components.py

"""Tests for OpenMDAO FAST mission primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import FlightConditions
from fast_python.mission import compute_flight_conditions


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
