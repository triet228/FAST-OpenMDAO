# tests/test_database_components.py

"""Tests for OpenMDAO FAST database-derived equation components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import MacLiftDragEstimate
from fast_python.database import mac_ld


def test_mac_lift_drag_estimate_matches_fast_python():
    """Check FAST MAC L/D estimate parity."""

    aspect_ratio = 9.5
    reynolds = 1.25e7
    problem = om.Problem()
    problem.model.add_subsystem("mac", MacLiftDragEstimate(), promotes=["*"])
    problem.setup()
    problem.set_val("aspect_ratio", aspect_ratio)
    problem.set_val("reynolds", reynolds)
    problem.run_model()

    assert np.isclose(
        problem.get_val("lift_drag")[0],
        mac_ld(aspect_ratio, reynolds),
    )


def test_mac_lift_drag_estimate_declares_analytic_partials():
    """Check MAC L/D analytical derivatives against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem("mac", MacLiftDragEstimate(), promotes=["*"])
    problem.setup()
    problem.set_val("aspect_ratio", 9.5)
    problem.set_val("reynolds", 1.25e7)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-4,
    )

    for partial_data in partials["mac"].values():
        assert (
            partial_data["abs error"].forward < 1.0e-6
            or partial_data["rel error"].forward < 1.0e-6
        )
