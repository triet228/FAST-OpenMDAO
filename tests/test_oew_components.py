# tests/test_oew_components.py

"""Tests for OpenMDAO FAST OEW components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import NumericSum, TurbopropAirframeWeight
from fast_python.oew import numeric_sum, turboprop_airframe_fit


def test_turboprop_airframe_weight_matches_fast_python_fit():
    """Check turboprop airframe-weight parity with FAST-Python fit."""

    slope, intercept = turboprop_airframe_fit(make_aircraft_database())
    problem = om.Problem()
    problem.model.add_subsystem(
        "airframe",
        TurbopropAirframeWeight(slope=slope, intercept=intercept),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("mtow", 250.0, units="kg")
    problem.run_model()

    assert np.isclose(
        problem.get_val("airframe_weight", units="kg")[0],
        np.polyval([slope, intercept], 250.0),
    )


def test_turboprop_airframe_weight_declares_analytic_partials():
    """Check airframe-weight analytical partials against finite difference."""

    slope, intercept = turboprop_airframe_fit(make_aircraft_database())
    problem = om.Problem()
    problem.model.add_subsystem(
        "airframe",
        TurbopropAirframeWeight(slope=slope, intercept=intercept),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("mtow", 250.0, units="kg")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["airframe"].values():
        assert partial_data["abs error"].forward < 1.0e-8


def test_numeric_sum_matches_fast_python():
    """Check OEW numeric summation parity with FAST-Python."""

    values = np.asarray([12.0, 4.0, 3.5])
    problem = om.Problem()
    problem.model.add_subsystem(
        "sum",
        NumericSum(vec_size=values.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("values", values)
    problem.run_model()

    assert np.isclose(problem.get_val("numeric_sum")[0], numeric_sum(values))


def test_numeric_sum_declares_analytic_partials():
    """Check OEW numeric summation derivatives against finite difference."""

    values = np.asarray([12.0, 4.0, 3.5])
    problem = om.Problem()
    problem.model.add_subsystem(
        "sum",
        NumericSum(vec_size=values.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("values", values)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["sum"].values():
        assert partial_data["abs error"].forward < 1.0e-8


def make_aircraft_database():
    """Return synthetic historical aircraft data for FAST-Python OEW fit."""

    return {
        "A": {
            "Specs": {
                "Weight": {
                    "MTOW": 100,
                    "Airframe": 50,
                }
            }
        },
        "B": {
            "Specs": {
                "Weight": {
                    "MTOW": 200,
                    "Airframe": 100,
                }
            }
        },
        "C": {
            "Specs": {
                "Weight": {
                    "MTOW": 300,
                    "Airframe": 150,
                }
            }
        },
    }
