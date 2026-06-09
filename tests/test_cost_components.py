# tests/test_cost_components.py

"""Tests for OpenMDAO FAST cost components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    BMSCostFraction,
    BatteryCapacityCost,
    BatteryReplacementCost,
)
from fast_python.cost import (
    battery_capacity_cost,
    battery_replacement_cost,
    bms_cost_fraction,
)


YEARS = np.asarray([2023.0, 2026.0, 2030.0, 2035.0])


def test_battery_replacement_cost_matches_fast_python():
    """Check battery replacement cost parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "cost",
        BatteryReplacementCost(chemistry=1, bms=1, lifespan=5),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("year", 2026.0)
    problem.set_val("battery_specific_energy", 720000.0, units="J/kg")
    problem.set_val("battery_weight", 1000.0, units="kg")
    problem.run_model()

    expected = battery_replacement_cost(make_cost_aircraft(1), 2026.0, 1, 5)

    assert np.isclose(problem.get_val("battery_replacement_cost")[0], expected)


def test_battery_replacement_cost_declares_analytic_partials():
    """Check battery replacement cost derivatives against finite difference."""

    for chemistry in [1, 2]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "cost",
            BatteryReplacementCost(chemistry=chemistry, bms=1, lifespan=4),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("year", 2028.0)
        problem.set_val("battery_specific_energy", 720000.0, units="J/kg")
        problem.set_val("battery_weight", 1000.0, units="kg")
        problem.run_model()

        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-3,
        )

        for key, partial_data in partials["cost"].items():
            assert partial_data["abs error"].forward < 1.0e-2, (
                key,
                partial_data["J_fwd"],
                partial_data["J_fd"],
                partial_data["abs error"].forward,
            )


def test_bms_cost_fraction_matches_fast_python():
    """Check BMS cost percentage curve parity with FAST-Python."""

    for chemistry, bms in [(1, 1), (2, 1), (1, 0)]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "curve",
            BMSCostFraction(chemistry=chemistry, bms=bms),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("year", 2028.5)
        problem.run_model()

        expected = bms_cost_fraction(YEARS, chemistry, bms, 2028.5)

        assert np.isclose(problem.get_val("bms_cost_fraction")[0], expected)


def test_bms_cost_fraction_declares_analytic_partials():
    """Check BMS cost percentage curve derivatives."""

    for chemistry, bms in [(1, 1), (2, 1), (1, 0)]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "curve",
            BMSCostFraction(chemistry=chemistry, bms=bms),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("year", 2028.5)
        problem.run_model()

        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-3,
        )

        for key, partial_data in partials["curve"].items():
            assert partial_data["abs error"].forward < 1.0e-6, (
                key,
                partial_data["J_fwd"],
                partial_data["J_fd"],
                partial_data["abs error"].forward,
            )


def test_battery_capacity_cost_matches_fast_python():
    """Check capacity-cost curve parity with FAST-Python."""

    for chemistry in [1, 2]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "curve",
            BatteryCapacityCost(chemistry=chemistry),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("year", 2028.5)
        problem.run_model()

        expected = battery_capacity_cost(YEARS, chemistry, 2028.5)

        assert np.isclose(problem.get_val("battery_capacity_cost")[0], expected)


def test_battery_capacity_cost_declares_analytic_partials():
    """Check capacity-cost curve derivatives."""

    for chemistry in [1, 2]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "curve",
            BatteryCapacityCost(chemistry=chemistry),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("year", 2028.5)
        problem.run_model()

        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-3,
        )

        for key, partial_data in partials["curve"].items():
            assert partial_data["abs error"].forward < 1.0e-2, (
                key,
                partial_data["J_fwd"],
                partial_data["J_fd"],
                partial_data["abs error"].forward,
            )


def make_cost_aircraft(chemistry):
    """Return a minimal aircraft dictionary for FAST-Python cost parity."""

    return {
        "Specs": {
            "Battery": {
                "Chem": chemistry,
            },
            "Power": {
                "SpecEnergy": {
                    "Batt": 720000,
                },
            },
            "Weight": {
                "Batt": 1000,
            },
        }
    }
