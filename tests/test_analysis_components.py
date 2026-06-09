# tests/test_analysis_components.py

"""Tests for OpenMDAO FAST analysis helper components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import ConvergenceError, SourceWeightVector, WeightSum
from fast_python.analysis import convergence_error, initial_source_weight, sum_weight


def test_convergence_error_matches_fast_python():
    """Check scalar convergence error parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem("convergence", ConvergenceError(), promotes=["*"])
    problem.setup()
    problem.set_val("delta", -2.0)
    problem.set_val("baseline", 10.0)
    problem.run_model()

    expected = convergence_error(np.asarray([-2.0]), np.asarray([10.0]))[0]

    assert np.isclose(problem.get_val("convergence_error")[0], expected)


def test_convergence_error_declares_analytic_partials():
    """Check convergence error analytical partials against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem("convergence", ConvergenceError(), promotes=["*"])
    problem.setup()
    problem.set_val("delta", -2.0)
    problem.set_val("baseline", 10.0)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["convergence"].values():
        assert partial_data["abs error"].forward < 1.0e-8


def test_weight_sum_matches_fast_python():
    """Check source-weight summation parity with FAST-Python."""

    values = np.asarray([12.0, 4.0, 3.5])
    problem = om.Problem()
    problem.model.add_subsystem(
        "weight",
        WeightSum(vec_size=values.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("weight_values", values, units="kg")
    problem.run_model()

    assert np.isclose(problem.get_val("weight_sum", units="kg")[0], sum_weight(values))


def test_weight_sum_declares_analytic_partials():
    """Check source-weight summation derivatives against finite difference."""

    values = np.asarray([12.0, 4.0, 3.5])
    problem = om.Problem()
    problem.model.add_subsystem(
        "weight",
        WeightSum(vec_size=values.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("weight_values", values, units="kg")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["weight"].values():
        assert partial_data["abs error"].forward < 1.0e-8


def test_source_weight_vector_matches_fast_python():
    """Check FAST source-weight vector expansion parity."""

    scalar_problem = om.Problem()
    scalar_problem.model.add_subsystem(
        "source",
        SourceWeightVector(source_count=3, input_size=1),
        promotes=["*"],
    )
    scalar_problem.setup()
    scalar_problem.set_val("source_weight", [11.0], units="kg")
    scalar_problem.run_model()

    expected_scalar = initial_source_weight(11.0, np.asarray([True, True, True]))

    assert np.allclose(
        scalar_problem.get_val("source_weight_vector", units="kg"),
        expected_scalar,
    )

    vector = np.asarray([11.0, 3.0, 4.0])
    vector_problem = om.Problem()
    vector_problem.model.add_subsystem(
        "source",
        SourceWeightVector(source_count=3, input_size=3),
        promotes=["*"],
    )
    vector_problem.setup()
    vector_problem.set_val("source_weight", vector, units="kg")
    vector_problem.run_model()

    assert np.allclose(
        vector_problem.get_val("source_weight_vector", units="kg"),
        vector,
    )


def test_source_weight_vector_declares_analytic_partials():
    """Check source-weight vector derivatives against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "source",
        SourceWeightVector(source_count=3, input_size=1),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("source_weight", [11.0], units="kg")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["source"].values():
        assert partial_data["abs error"].forward < 1.0e-8
