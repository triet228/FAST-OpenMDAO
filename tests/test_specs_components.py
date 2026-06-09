# tests/test_specs_components.py

"""Tests for OpenMDAO FAST aircraft specification primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import LM100JHybridOperationMatrices
from fast_python.specs import lm100j_hybrid_oper_dwn, lm100j_hybrid_oper_ups


def test_lm100j_hybrid_operation_matrices_match_fast_python():
    """Check LM100J_Hybrid split matrix parity with FAST-Python."""

    power_split = 0.31
    problem = om.Problem()
    problem.model.add_subsystem(
        "operation",
        LM100JHybridOperationMatrices(),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("power_split", power_split)
    problem.run_model()

    assert np.allclose(
        problem.get_val("upstream_split"),
        np.asarray(lm100j_hybrid_oper_ups(power_split), dtype=float),
    )
    assert np.allclose(
        problem.get_val("downstream_split"),
        np.asarray(lm100j_hybrid_oper_dwn(power_split), dtype=float),
    )


def test_lm100j_hybrid_operation_matrices_partials():
    """Check exact LM100J_Hybrid split matrix derivatives."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "operation",
        LM100JHybridOperationMatrices(),
        promotes=["*"],
    )
    problem.setup(force_alloc_complex=True)
    problem.set_val("power_split", 0.31)
    problem.run_model()

    partials = problem.check_partials(method="cs", out_stream=None)
    for component_partials in partials.values():
        for partial in component_partials.values():
            assert partial["abs error"].forward < 1.0e-12
