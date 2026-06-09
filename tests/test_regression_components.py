# tests/test_regression_components.py

"""Tests for OpenMDAO FAST regression components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import SquaredExponentialKernel
from fast_python.regression import square_exp_kernel


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
