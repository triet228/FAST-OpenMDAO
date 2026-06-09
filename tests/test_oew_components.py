# tests/test_oew_components.py

"""Tests for OpenMDAO FAST OEW components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import NumericSum, TurbofanAirframeWeight, TurbopropAirframeWeight
from fast_python.oew import (
    numeric_sum,
    regression_airframe_weight,
    turboprop_airframe_fit,
)
from fast_python.regression import square_exp_kernel


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


def test_turbofan_airframe_weight_matches_fast_python_regression():
    """Check turbofan airframe-weight GPR parity with FAST-Python."""

    data_matrix, hyperparams, inverse_term = make_turbofan_regression_data()
    target = np.asarray([120.0, 150000.0, 2035.0, 90000.0])
    frame_factor = 1.08
    problem = om.Problem()
    problem.model.add_subsystem(
        "airframe",
        TurbofanAirframeWeight(
            data_matrix=data_matrix,
            hyperparams=hyperparams,
            inverse_term=inverse_term,
            prior=np.mean(data_matrix[:, -1]),
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("wing_area", target[0], units="m**2")
    problem.set_val("thrust", target[1], units="N")
    problem.set_val("eis", target[2])
    problem.set_val("mtow", target[3], units="kg")
    problem.set_val("frame_factor", frame_factor)
    problem.run_model()

    expected = regression_airframe_weight(
        make_turbofan_aircraft_database(data_matrix),
        target,
        {
            "DataMatrix": data_matrix,
            "HyperParams": hyperparams,
            "InverseTerm": inverse_term,
        },
    )

    assert np.isclose(
        problem.get_val("airframe_weight", units="kg")[0],
        expected * frame_factor,
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


def test_turbofan_airframe_weight_declares_analytic_partials():
    """Check turbofan airframe-weight derivatives against finite difference."""

    data_matrix, hyperparams, inverse_term = make_turbofan_regression_data()
    problem = om.Problem()
    problem.model.add_subsystem(
        "airframe",
        TurbofanAirframeWeight(
            data_matrix=data_matrix,
            hyperparams=hyperparams,
            inverse_term=inverse_term,
            prior=np.mean(data_matrix[:, -1]),
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("wing_area", 120.0, units="m**2")
    problem.set_val("thrust", 150000.0, units="N")
    problem.set_val("eis", 2035.0)
    problem.set_val("mtow", 90000.0, units="kg")
    problem.set_val("frame_factor", 1.08)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-5,
    )

    for partial_data in partials["airframe"].values():
        assert partial_data["abs error"].forward < 1.0e-5


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


def make_turbofan_regression_data():
    """Return fixed turbofan OEW GPR preprocessing matrices."""

    data_matrix = np.asarray(
        [
            [100.0, 120000.0, 2030.0, 80000.0, 32000.0],
            [130.0, 160000.0, 2035.0, 95000.0, 39000.0],
            [160.0, 210000.0, 2040.0, 120000.0, 51000.0],
        ]
    )
    hyperparams = np.asarray([900.0, 2.5e9, 25.0, 2.0e8, 1.0e8])
    kbarbar = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))

    for irow in range(data_matrix.shape[0]):
        for jrow in range(data_matrix.shape[0]):
            kbarbar[irow, jrow] = square_exp_kernel(
                data_matrix[irow, :-1],
                data_matrix[jrow, :-1],
                hyperparams,
            )[0]

    inverse_term = np.linalg.inv(kbarbar + 1.0e6 * np.eye(data_matrix.shape[0]))
    return data_matrix, hyperparams, inverse_term


def make_turbofan_aircraft_database(data_matrix):
    """Return aircraft database rows for FAST-Python prior calculation."""

    database = {}

    for index, row in enumerate(data_matrix):
        database[f"AC{index}"] = {
            "Specs": {
                "Aero": {
                    "S": row[0],
                },
                "Propulsion": {
                    "Thrust": {
                        "SLS": row[1],
                    },
                },
                "TLAR": {
                    "EIS": row[2],
                },
                "Weight": {
                    "MTOW": row[3],
                    "Airframe": row[4],
                },
            }
        }

    return database
