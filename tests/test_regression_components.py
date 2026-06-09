# tests/test_regression_components.py

"""Tests for OpenMDAO FAST regression components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    GaussianProcessPrediction,
    RegressionInverseTerm,
    RegressionNumericColumn,
    RegressionNumericScalar,
    RegressionPriorMean,
    RegressionSampleVariance,
    RegressionTargetMatrix,
    RegressionTwoDimensionalArray,
    RegressionVector,
    RegressionWeightedHyperparameters,
    SquaredExponentialKernel,
)
from fast_python.regression import (
    as_2d,
    as_vector,
    build_data,
    nlgpr,
    numeric_column,
    numeric_scalar,
    prior_calculation,
    reg_processing,
    sample_variance,
    square_exp_kernel,
    target_matrix,
)


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


def test_regression_inverse_term_matches_fast_python():
    """Check regression inverse covariance preprocessing against FAST-Python."""

    database = make_weighted_hyperparameter_database()
    io_space = make_weighted_hyperparameter_io_space()
    weights = np.asarray([1.0, 2.0])
    prior = np.asarray([140.0, 160.0])
    data_matrix, hyperparams, expected_inverse = reg_processing(
        database,
        io_space,
        prior,
        weights,
    )
    problem = om.Problem()
    problem.model.add_subsystem(
        "inverse",
        RegressionInverseTerm(data_matrix=data_matrix, prior_size=prior.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("hyperparams", hyperparams)
    problem.set_val("prior", prior)
    problem.run_model()

    assert np.allclose(problem.get_val("inverse_term"), expected_inverse)


def test_regression_shape_and_variance_helpers_match_fast_python():
    """Check regression normalizers and sample variance against FAST-Python."""

    vector_values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    vector_problem = om.Problem()
    vector_problem.model.add_subsystem(
        "vector",
        RegressionVector(input_shape=vector_values.shape),
        promotes=["*"],
    )
    vector_problem.setup()
    vector_problem.set_val("values", vector_values)
    vector_problem.run_model()

    assert np.allclose(vector_problem.get_val("vector"), as_vector(vector_values))

    two_dimensional_values = np.asarray([5.0, 6.0, 7.0])
    array_problem = om.Problem()
    array_problem.model.add_subsystem(
        "array",
        RegressionTwoDimensionalArray(input_shape=two_dimensional_values.shape),
        promotes=["*"],
    )
    array_problem.setup()
    array_problem.set_val("values", two_dimensional_values)
    array_problem.run_model()

    assert np.allclose(
        array_problem.get_val("two_dimensional_values"),
        as_2d(two_dimensional_values),
    )

    target_values = np.asarray([8.0, 9.0])
    target_problem = om.Problem()
    target_problem.model.add_subsystem(
        "target",
        RegressionTargetMatrix(input_shape=target_values.shape),
        promotes=["*"],
    )
    target_problem.setup()
    target_problem.set_val("target_values", target_values)
    target_problem.run_model()

    assert np.allclose(
        target_problem.get_val("target_matrix"),
        target_matrix(target_values),
    )

    variance_values = np.asarray([2.0, 4.0, 7.0, 11.0])
    variance_problem = om.Problem()
    variance_problem.model.add_subsystem(
        "variance",
        RegressionSampleVariance(vec_size=variance_values.size),
        promotes=["*"],
    )
    variance_problem.setup()
    variance_problem.set_val("values", variance_values)
    variance_problem.run_model()

    assert np.isclose(
        variance_problem.get_val("sample_variance")[0],
        sample_variance(variance_values),
    )

    weights = np.asarray([1.0, 2.0])
    hyper_database = make_weighted_hyperparameter_database()
    hyper_io_space = make_weighted_hyperparameter_io_space()
    _, expected_hyperparams = build_data(hyper_database, hyper_io_space, weights)
    hyper_variances = np.asarray(
        [
            sample_variance([1.0, 2.0, 4.0]),
            sample_variance([10.0, 20.0, 40.0]),
            sample_variance([100.0, 120.0, 160.0]),
        ]
    )
    hyper_problem = om.Problem()
    hyper_problem.model.add_subsystem(
        "hyperparams",
        RegressionWeightedHyperparameters(num_inputs=2),
        promotes=["*"],
    )
    hyper_problem.setup()
    hyper_problem.set_val("variances", hyper_variances)
    hyper_problem.set_val("weights", weights)
    hyper_problem.run_model()

    assert np.allclose(hyper_problem.get_val("hyperparams"), expected_hyperparams)

    prior_values = np.asarray([10.0, 20.0, np.nan])
    prior_problem = om.Problem()
    prior_problem.model.add_subsystem(
        "prior",
        RegressionPriorMean(vec_size=prior_values.size),
        promotes=["*"],
    )
    prior_problem.setup()
    prior_problem.set_val("values", prior_values)
    prior_problem.run_model()

    assert np.isclose(
        prior_problem.get_val("prior_mean")[0],
        prior_calculation(make_regression_database(), make_regression_io_space()),
    )

    scalar_values = np.asarray([[12.0, 13.0], [14.0, 15.0]])
    scalar_problem = om.Problem()
    scalar_problem.model.add_subsystem(
        "scalar",
        RegressionNumericScalar(input_shape=scalar_values.shape),
        promotes=["*"],
    )
    scalar_problem.setup()
    scalar_problem.set_val("values", scalar_values)
    scalar_problem.run_model()

    assert np.isclose(
        scalar_problem.get_val("numeric_scalar")[0],
        numeric_scalar(scalar_values),
    )

    column_values = np.asarray([[16.0, 17.0], [18.0, 19.0], [20.0, 21.0]])
    column_problem = om.Problem()
    column_problem.model.add_subsystem(
        "column",
        RegressionNumericColumn(input_shape=column_values.shape),
        promotes=["*"],
    )
    column_problem.setup()
    column_problem.set_val("values", column_values)
    column_problem.run_model()

    assert np.allclose(
        column_problem.get_val("numeric_column"),
        numeric_column(column_values),
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


def test_regression_inverse_term_declares_analytic_partials():
    """Check inverse covariance preprocessing derivatives."""

    data_matrix = np.asarray(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 120.0],
            [4.0, 40.0, 160.0],
        ]
    )
    problem = om.Problem()
    problem.model.add_subsystem(
        "inverse",
        RegressionInverseTerm(data_matrix=data_matrix, prior_size=2),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("hyperparams", [3.5, 350.0, 1200.0])
    problem.set_val("prior", [140.0, 160.0])
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["inverse"].values():
        assert partial_data["abs error"].forward < 1.0e-6


def test_regression_shape_and_variance_helpers_declare_analytic_partials():
    """Check regression helper derivatives against finite difference."""

    cases = [
        (
            "vector",
            RegressionVector(input_shape=(2, 2)),
            {"values": np.asarray([[1.0, 2.0], [3.0, 4.0]])},
        ),
        (
            "array",
            RegressionTwoDimensionalArray(input_shape=(3,)),
            {"values": np.asarray([5.0, 6.0, 7.0])},
        ),
        (
            "target",
            RegressionTargetMatrix(input_shape=(2,)),
            {"target_values": np.asarray([8.0, 9.0])},
        ),
        (
            "variance",
            RegressionSampleVariance(vec_size=4),
            {"values": np.asarray([2.0, 4.0, 7.0, 11.0])},
        ),
        (
            "hyperparams",
            RegressionWeightedHyperparameters(num_inputs=2),
            {
                "variances": np.asarray([3.0, 30.0, 300.0]),
                "weights": np.asarray([1.0, 2.0]),
            },
        ),
        (
            "prior",
            RegressionPriorMean(vec_size=3),
            {"values": np.asarray([10.0, 20.0, 30.0])},
        ),
        (
            "scalar",
            RegressionNumericScalar(input_shape=(2, 2)),
            {"values": np.asarray([[12.0, 13.0], [14.0, 15.0]])},
        ),
        (
            "column",
            RegressionNumericColumn(input_shape=(3, 2)),
            {"values": np.asarray([[16.0, 17.0], [18.0, 19.0], [20.0, 21.0]])},
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


def make_regression_database():
    """Return small FAST-style regression database with one missing output."""

    return {
        "AC1": {"Specs": {"Weight": {"Fuel": 10.0}}},
        "AC2": {"Specs": {"Weight": {"Fuel": 20.0}}},
        "AC3": {"Specs": {"Weight": {"Fuel": np.nan}}},
    }


def make_regression_io_space():
    """Return FAST regression input/output path list for prior tests."""

    return [["Specs", "Weight", "Fuel"]]


def make_weighted_hyperparameter_database():
    """Return small FAST-style database for build_data hyperparameter tests."""

    return {
        "AC1": {"Specs": {"A": {"x": 1.0, "y": 10.0, "z": 100.0}}},
        "AC2": {"Specs": {"A": {"x": 2.0, "y": 20.0, "z": 120.0}}},
        "AC3": {"Specs": {"A": {"x": 4.0, "y": 40.0, "z": 160.0}}},
    }


def make_weighted_hyperparameter_io_space():
    """Return FAST regression paths for weighted hyperparameter tests."""

    return [
        ["Specs", "A", "x"],
        ["Specs", "A", "y"],
        ["Specs", "A", "z"],
    ]
