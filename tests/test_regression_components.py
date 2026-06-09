# tests/test_regression_components.py

"""Tests for OpenMDAO FAST regression components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import GaussianProcessPrediction, SquaredExponentialKernel
from fast_python.regression import nlgpr, square_exp_kernel


def test_squared_exponential_kernel_matches_fast_python():
    """Check squared-exponential kernel parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "kernel",
        SquaredExponentialKernel(size=2),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("x", [1.0, 2.0])
    problem.set_val("y", [2.0, 4.0])
    problem.set_val("length_scales", [4.0, 9.0])
    problem.set_val("signal_variance", 16.0)
    problem.run_model()

    expected = square_exp_kernel([[1.0, 2.0]], [[2.0, 4.0]], [[4.0, 9.0, 16.0]])

    assert np.isclose(problem.get_val("covariance")[0], expected[0])


def test_gaussian_process_prediction_matches_fast_python():
    """Check posterior GP prediction parity with FAST-Python."""

    data_matrix, hyperparams, inverse_term = make_gp_prediction_data()
    target = np.asarray([1.25, 2.25])
    prior = 11.0
    problem = om.Problem()
    problem.model.add_subsystem(
        "prediction",
        GaussianProcessPrediction(
            data_matrix=data_matrix,
            hyperparams=hyperparams,
            inverse_term=inverse_term,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("target", target)
    problem.set_val("prior", prior)
    problem.run_model()

    expected_mean, expected_variance = nlgpr(
        {},
        [[], [], []],
        target,
        prior=prior,
        preprocessing={
            "DataMatrix": data_matrix,
            "HyperParams": hyperparams,
            "InverseTerm": inverse_term,
        },
    )

    assert np.isclose(problem.get_val("posterior_mean")[0], expected_mean[0])
    assert np.isclose(
        problem.get_val("posterior_variance")[0],
        expected_variance[0],
    )


def test_squared_exponential_kernel_declares_analytic_partials():
    """Check kernel analytical partials against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "kernel",
        SquaredExponentialKernel(size=2),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("x", [1.0, 2.0])
    problem.set_val("y", [2.0, 4.0])
    problem.set_val("length_scales", [4.0, 9.0])
    problem.set_val("signal_variance", 16.0)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["kernel"].values():
        assert partial_data["abs error"].forward < 1.0e-6


def test_gaussian_process_prediction_declares_analytic_partials():
    """Check posterior GP prediction derivatives against finite difference."""

    data_matrix, hyperparams, inverse_term = make_gp_prediction_data()
    problem = om.Problem()
    problem.model.add_subsystem(
        "prediction",
        GaussianProcessPrediction(
            data_matrix=data_matrix,
            hyperparams=hyperparams,
            inverse_term=inverse_term,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("target", [1.25, 2.25])
    problem.set_val("prior", 11.0)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["prediction"].values():
        assert partial_data["abs error"].forward < 1.0e-6


def make_gp_prediction_data():
    """Return fixed FAST NLGPR preprocessing matrices for component tests."""

    data_matrix = np.asarray(
        [
            [1.0, 2.0, 10.0],
            [2.0, 3.0, 12.0],
            [3.0, 4.0, 15.0],
        ]
    )
    hyperparams = np.asarray([2.0, 3.0, 4.0])
    kbarbar = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))

    for irow in range(data_matrix.shape[0]):
        for jrow in range(data_matrix.shape[0]):
            kbarbar[irow, jrow] = square_exp_kernel(
                data_matrix[irow, :-1],
                data_matrix[jrow, :-1],
                hyperparams,
            )[0]

    inverse_term = np.linalg.inv(kbarbar + 0.25 * np.eye(data_matrix.shape[0]))
    return data_matrix, hyperparams, inverse_term
