# tests/test_safety_components.py

"""Tests for OpenMDAO FAST safety components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import FailureModel
from fast_python.safety import failure_model


def test_failure_model_matches_fast_python_scalar_exposure():
    """Check FAST failure probability parity for shared exposure time."""

    base_rate = np.asarray([0.1, 0.2])
    exposure = 2.0
    problem = om.Problem()
    problem.model.add_subsystem(
        "failure",
        FailureModel(vec_size=base_rate.size, exposure_size=1),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("base_rate", base_rate)
    problem.set_val("exposure", exposure)
    problem.run_model()

    assert np.allclose(
        problem.get_val("failure_probability"),
        failure_model(base_rate, exposure),
    )


def test_failure_model_matches_fast_python_vector_exposure():
    """Check FAST failure probability parity for per-component exposure."""

    base_rate = np.asarray([0.1, 0.2, 0.3])
    exposure = np.asarray([1.0, 2.0, 3.0])
    problem = om.Problem()
    problem.model.add_subsystem(
        "failure",
        FailureModel(vec_size=base_rate.size, exposure_size=exposure.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("base_rate", base_rate)
    problem.set_val("exposure", exposure)
    problem.run_model()

    assert np.allclose(
        problem.get_val("failure_probability"),
        failure_model(base_rate, exposure),
    )


def test_failure_model_declares_analytic_partials():
    """Check failure probability derivatives against finite difference."""

    cases = [
        (
            FailureModel(vec_size=2, exposure_size=1),
            np.asarray([0.1, 0.2]),
            np.asarray([2.0]),
        ),
        (
            FailureModel(vec_size=3, exposure_size=3),
            np.asarray([0.1, 0.2, 0.3]),
            np.asarray([1.0, 2.0, 3.0]),
        ),
    ]

    for component, base_rate, exposure in cases:
        problem = om.Problem()
        problem.model.add_subsystem("failure", component, promotes=["*"])
        problem.setup()
        problem.set_val("base_rate", base_rate)
        problem.set_val("exposure", exposure)
        problem.run_model()

        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-6,
        )

        for partial_data in partials["failure"].values():
            assert partial_data["abs error"].forward < 1.0e-6
