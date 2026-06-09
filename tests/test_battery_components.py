# tests/test_battery_components.py

"""Tests for OpenMDAO FAST battery primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import AvailableCellCapacity, BatteryCurrent
from fast_python.battery import available_cell_capacity, solve_battery_current


def test_available_cell_capacity_matches_fast_python():
    """Check effective cell capacity parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "capacity",
        AvailableCellCapacity(analysis_type=-2, degradation=1),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("cap_cell", 2.4)
    problem.set_val("state_of_health", 90.0)
    problem.run_model()

    assert np.isclose(
        problem.get_val("available_cell_capacity")[0],
        available_cell_capacity(make_battery_aircraft(2.4, 90.0)),
    )


def test_battery_current_matches_fast_python_real_roots():
    """Check battery current root parity with FAST-Python."""

    for is_discharge, requested_power in [(True, 10.0), (False, -5.0)]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "current",
            BatteryCurrent(is_discharge=is_discharge),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("hot_voltage", 0.01)
        problem.set_val("cold_voltage", 4.0)
        problem.set_val("requested_cell_power", requested_power, units="W")
        problem.run_model()

        assert np.isclose(
            problem.get_val("cell_current", units="A")[0],
            solve_battery_current(0.01, 4.0, requested_power, is_discharge),
        )


def test_battery_primitives_declare_analytic_partials():
    """Check battery primitive derivatives against finite difference."""

    cases = [
        (
            "capacity",
            AvailableCellCapacity(analysis_type=-2, degradation=1),
            {"cap_cell": 2.4, "state_of_health": 90.0},
        ),
        (
            "current",
            BatteryCurrent(is_discharge=True),
            {
                "hot_voltage": 0.01,
                "cold_voltage": 4.0,
                "requested_cell_power": 10.0,
            },
        ),
        (
            "charge_current",
            BatteryCurrent(is_discharge=False),
            {
                "hot_voltage": 0.01,
                "cold_voltage": 4.0,
                "requested_cell_power": -5.0,
            },
        ),
    ]

    for name, component, values in cases:
        problem = om.Problem()
        problem.model.add_subsystem(name, component, promotes=["*"])
        problem.setup()

        for variable, value in values.items():
            problem.set_val(variable, value)

        problem.run_model()
        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-6,
        )

        for partial_data in partials[name].values():
            assert partial_data["abs error"].forward < 1.0e-6


def make_battery_aircraft(cap_cell, state_of_health):
    """Return minimal aircraft dictionary for FAST-Python battery helpers."""

    return {
        "Settings": {
            "Analysis": {
                "Type": -2,
            },
        },
        "Specs": {
            "Battery": {
                "CapCell": cap_cell,
                "Degradation": 1,
                "SOH": [100.0, state_of_health],
            },
        },
    }
