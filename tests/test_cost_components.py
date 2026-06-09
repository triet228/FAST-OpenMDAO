# tests/test_cost_components.py

"""Tests for OpenMDAO FAST cost components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import BatteryReplacementCost
from fast_python.cost import battery_replacement_cost


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
