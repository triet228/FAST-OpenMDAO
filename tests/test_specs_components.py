# tests/test_specs_components.py

"""Tests for OpenMDAO FAST aircraft specification primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    AEACustomArchitecture,
    LM100JHybridArchitecture,
    LM100JHybridOperationMatrices,
    SplitScalar,
    ZeroSegmentSplits,
)
from fast_python.specs import (
    aea_architecture_matrices,
    lm100j_hybrid_architecture,
    lm100j_hybrid_oper_dwn,
    lm100j_hybrid_oper_ups,
    split_scalar,
    zero_segment_splits,
)


def test_aea_custom_architecture_matches_fast_python():
    """Check AEA custom architecture parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "architecture",
        AEACustomArchitecture(),
        promotes=["*"],
    )
    problem.setup()
    problem.run_model()

    expected = aea_architecture_matrices()
    assert np.allclose(problem.get_val("architecture"), expected["Arch"])
    assert np.allclose(problem.get_val("upstream_split"), expected["OperUps"])
    assert np.allclose(problem.get_val("downstream_split"), expected["OperDwn"])
    assert np.allclose(problem.get_val("upstream_efficiency"), expected["EtaUps"])
    assert np.allclose(problem.get_val("downstream_efficiency"), expected["EtaDwn"])


def test_lm100j_hybrid_architecture_matches_fast_python():
    """Check full LM100J_Hybrid architecture parity with FAST-Python."""

    power_split = 0.31
    problem = om.Problem()
    problem.model.add_subsystem(
        "architecture",
        LM100JHybridArchitecture(),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("power_split", power_split)
    problem.run_model()

    expected = lm100j_hybrid_architecture()
    assert np.allclose(problem.get_val("architecture"), expected["Arch"])
    assert np.allclose(
        problem.get_val("upstream_split"),
        np.asarray(expected["OperUps"](power_split), dtype=float),
    )
    assert np.allclose(
        problem.get_val("downstream_split"),
        np.asarray(expected["OperDwn"](power_split), dtype=float),
    )
    assert np.allclose(problem.get_val("upstream_efficiency"), expected["EtaUps"])
    assert np.allclose(problem.get_val("downstream_efficiency"), expected["EtaDwn"])
    assert np.allclose(problem.get_val("source_type"), expected["SrcType"])
    assert np.allclose(problem.get_val("transmitter_type"), expected["TrnType"])


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


def test_split_scalar_matches_fast_python():
    """Check fixed-shape split scalar flattening parity with FAST-Python."""

    split_values = np.asarray([[0.42, 0.31], [0.27, 0.15]])
    problem = om.Problem()
    problem.model.add_subsystem(
        "split",
        SplitScalar(input_shape=split_values.shape),
        promotes=["*"],
    )
    problem.setup(force_alloc_complex=True)
    problem.set_val("split_values", split_values)
    problem.run_model()

    assert np.allclose(problem.get_val("split_scalar"), split_scalar(split_values))

    partials = problem.check_partials(method="cs", out_stream=None)
    for component_partials in partials.values():
        for partial in component_partials.values():
            assert partial["abs error"].forward < 1.0e-12


def test_zero_segment_splits_match_fast_python():
    """Check zero segment split parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "splits",
        ZeroSegmentSplits(),
        promotes=["*"],
    )
    problem.setup()
    problem.run_model()

    expected = zero_segment_splits()
    assert np.allclose(problem.get_val("sls"), expected["SLS"])
    assert np.allclose(problem.get_val("takeoff"), expected["Tko"])
    assert np.allclose(problem.get_val("climb"), expected["Clb"])
    assert np.allclose(problem.get_val("cruise"), expected["Crs"])
    assert np.allclose(problem.get_val("descent"), expected["Des"])
    assert np.allclose(problem.get_val("landing"), expected["Lnd"])


def test_lm100j_hybrid_operation_matrices_partials():
    """Check exact LM100J_Hybrid matrix derivatives."""

    for component in [LM100JHybridArchitecture(), LM100JHybridOperationMatrices()]:
        problem = om.Problem()
        problem.model.add_subsystem("component", component, promotes=["*"])
        problem.setup(force_alloc_complex=True)
        problem.set_val("power_split", 0.31)
        problem.run_model()

        partials = problem.check_partials(method="cs", out_stream=None)
        for component_partials in partials.values():
            for partial in component_partials.values():
                assert partial["abs error"].forward < 1.0e-12
