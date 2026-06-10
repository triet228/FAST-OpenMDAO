# tests/test_split_optimization.py

"""Tests for arbitrary propulsion split optimization helpers."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    SplitGroupSums,
    infer_propulsion_split_specs,
    initialize_missing_propulsion_split_matrices,
    make_fast_auto_split_optimization_problem,
    make_fast_mission_split_schedule_optimization_problem,
    make_fast_split_optimization_problem,
    propulsion_split_diagnostics,
    split_matrix_design_specs,
    split_schedule_design_specs,
)


def test_split_matrix_design_specs_discovers_branch_entries():
    """Check split matrix specs expose only active branching entries."""

    generated = split_matrix_design_specs(
        make_split_aircraft(),
        split_specs=[make_downstream_split_spec()],
    )

    assert generated["input_names"] == (
        "downstream_0_1",
        "downstream_0_2",
        "downstream_1_2",
        "downstream_1_3",
    )
    assert generated["groups"] == (
        ("downstream_0_1", "downstream_0_2"),
        ("downstream_1_2", "downstream_1_3"),
    )
    assert generated["input_specs"][0]["path"] == (
        "Specs",
        "Propulsion",
        "PropArch",
        "OperDwn",
        0,
        1,
    )
    assert generated["design_vars"][0]["lower"] == 0.0
    assert generated["design_vars"][0]["upper"] == 1.0


def test_split_group_sums_has_linear_partials():
    """Check split-sum constraints and analytical partials."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "sums",
        SplitGroupSums(
            groups=(
                ("split_0", "split_1"),
                ("split_2", "split_3"),
            ),
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("split_0", 0.25)
    problem.set_val("split_1", 0.75)
    problem.set_val("split_2", 0.4)
    problem.set_val("split_3", 0.6)
    problem.run_model()

    assert np.allclose(problem.get_val("split_sums"), [1.0, 1.0])

    partials = problem.check_partials(out_stream=None, method="fd")

    for partial_data in partials["sums"].values():
        assert partial_data["abs error"].forward < 1.0e-9


def test_fast_split_optimization_problem_optimizes_matrix_entries():
    """Check generated split variables drive a constrained FAST optimization."""

    problem = make_fast_split_optimization_problem(
        aircraft=make_split_aircraft(),
        split_specs=[make_downstream_split_spec()],
        output_specs=[
            {
                "name": "split_objective",
                "path": ("split_objective",),
            },
        ],
        runner=split_objective_runner,
        objective={
            "name": "split_objective",
        },
        driver_options={
            "maxiter": 40,
            "tol": 1.0e-10,
        },
    )
    problem.setup()
    result = problem.run_driver()

    assert result.success
    assert np.allclose(problem.get_val("split_sum_constraints"), [1.0, 1.0])
    assert abs(problem.get_val("downstream_0_1")[0] - 0.2) < 1.0e-5
    assert abs(problem.get_val("downstream_0_2")[0] - 0.8) < 1.0e-5
    assert abs(problem.get_val("downstream_1_2")[0] - 0.65) < 1.0e-5
    assert abs(problem.get_val("downstream_1_3")[0] - 0.35) < 1.0e-5
    assert problem.get_val("split_objective")[0] < 1.0e-10


def test_infer_propulsion_split_specs_selects_oper_dwn_rows():
    """Check auto inference chooses the editable downstream row split."""

    specs = infer_propulsion_split_specs(make_split_aircraft())

    assert len(specs) == 1
    assert specs[0]["matrix_path"] == (
        "Specs",
        "Propulsion",
        "PropArch",
        "OperDwn",
    )
    assert specs[0]["axis"] == "row"


def test_auto_split_optimization_problem_uses_inferred_spec():
    """Check no-spec auto mode optimizes the discovered split matrix."""

    problem = make_fast_auto_split_optimization_problem(
        aircraft=make_split_aircraft(),
        output_specs=[
            {
                "name": "split_objective",
                "path": ("split_objective",),
            },
        ],
        runner=split_objective_runner,
        objective={
            "name": "split_objective",
        },
        driver_options={
            "maxiter": 40,
            "tol": 1.0e-10,
        },
    )
    problem.setup()
    result = problem.run_driver()

    assert result.success
    assert problem.fast_auto_split_specs[0]["axis"] == "row"
    assert abs(problem.get_val("operdwn_0_1")[0] - 0.2) < 1.0e-5
    assert abs(problem.get_val("operdwn_0_2")[0] - 0.8) < 1.0e-5


def test_auto_split_initializes_missing_upstream_matrix_from_architecture():
    """Check Arch-only merge topology creates an upstream split matrix."""

    aircraft = make_merge_aircraft()
    problem = make_fast_auto_split_optimization_problem(
        aircraft=aircraft,
        output_specs=[
            {
                "name": "split_objective",
                "path": ("split_objective",),
            },
        ],
        runner=upstream_split_objective_runner,
        objective={
            "name": "split_objective",
        },
        driver_options={
            "maxiter": 40,
            "tol": 1.0e-10,
        },
    )
    problem.setup()
    result = problem.run_driver()

    assert result.success
    assert "OperUps" not in aircraft["Specs"]["Propulsion"]["PropArch"]
    assert (
        problem.fast_auto_split_initialization["initialized"][0]["matrix_name"]
        == "OperUps"
    )
    assert problem.fast_auto_split_specs[0]["axis"] == "column"
    assert abs(problem.get_val("operups_0_2")[0] - 0.25) < 1.0e-5
    assert abs(problem.get_val("operups_1_2")[0] - 0.75) < 1.0e-5


def test_auto_split_initializes_missing_downstream_matrix_from_architecture():
    """Check Arch-only branch topology creates a downstream split matrix."""

    problem = make_fast_auto_split_optimization_problem(
        aircraft=make_branch_aircraft(),
        output_specs=[
            {
                "name": "split_objective",
                "path": ("split_objective",),
            },
        ],
        runner=downstream_split_objective_runner,
        objective={
            "name": "split_objective",
        },
        driver_options={
            "maxiter": 40,
            "tol": 1.0e-10,
        },
    )
    problem.setup()
    result = problem.run_driver()

    assert result.success
    assert (
        problem.fast_auto_split_initialization["initialized"][0]["matrix_name"]
        == "OperDwn"
    )
    assert problem.fast_auto_split_specs[0]["axis"] == "row"
    assert abs(problem.get_val("operdwn_0_1")[0] - 0.4) < 1.0e-5
    assert abs(problem.get_val("operdwn_0_2")[0] - 0.6) < 1.0e-5


def test_missing_split_initialization_requires_choice_for_merge_branch_topology():
    """Check merge-plus-branch topology is not over-controlled by default."""

    try:
        initialize_missing_propulsion_split_matrices(make_merge_branch_aircraft())
    except ValueError as error:
        assert "several missing split matrix conventions" in str(error)
    else:
        raise AssertionError("Expected merge-plus-branch initialization to fail.")


def test_auto_split_initializes_all_missing_matrices_when_not_strict():
    """Check strict=False exposes both upstream and downstream split controls."""

    problem = make_fast_auto_split_optimization_problem(
        aircraft=make_merge_branch_aircraft(),
        output_specs=[
            {
                "name": "split_objective",
                "path": ("split_objective",),
            },
        ],
        runner=merge_branch_split_objective_runner,
        objective={
            "name": "split_objective",
        },
        strict=False,
        driver_options={
            "maxiter": 60,
            "tol": 1.0e-10,
        },
    )
    problem.setup()
    result = problem.run_driver()

    initialized = problem.fast_auto_split_initialization["initialized"]
    matrix_names = tuple(item["matrix_name"] for item in initialized)

    assert result.success
    assert matrix_names == ("OperDwn", "OperUps")
    assert len(problem.fast_auto_split_specs) == 2
    assert abs(problem.get_val("operdwn_2_3")[0] - 0.3) < 1.0e-5
    assert abs(problem.get_val("operdwn_2_4")[0] - 0.7) < 1.0e-5
    assert abs(problem.get_val("operups_0_2")[0] - 0.8) < 1.0e-5
    assert abs(problem.get_val("operups_1_2")[0] - 0.2) < 1.0e-5


def test_auto_split_inference_requires_preferred_matrix_when_ambiguous():
    """Check auto mode refuses to guess between valid split matrices."""

    aircraft = make_split_aircraft()
    aircraft["Specs"]["Propulsion"]["PropArch"]["OperUps"] = [
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    try:
        infer_propulsion_split_specs(aircraft)
    except ValueError as error:
        assert "Several editable split matrices" in str(error)
    else:
        raise AssertionError("Expected ambiguous split inference to fail.")

    specs = infer_propulsion_split_specs(aircraft, preferred_matrix="OperDwn")
    assert specs[0]["matrix_path"][-1] == "OperDwn"


def test_auto_split_diagnostics_reports_callable_generators():
    """Check callable split generators are reported instead of guessed."""

    aircraft = make_split_aircraft()
    aircraft["Specs"]["Propulsion"]["PropArch"]["OperDwn"] = lambda: []
    diagnostics = propulsion_split_diagnostics(aircraft)

    assert diagnostics[0]["matrix_name"] == "OperDwn"
    assert not diagnostics[0]["usable"]
    assert "callable" in diagnostics[0]["reason"]


def test_split_schedule_design_specs_exposes_mission_points():
    """Check schedule specs expose independent mission-point split entries."""

    generated = split_schedule_design_specs(
        make_schedule_aircraft(),
        schedule_specs=[make_climb_schedule_spec()],
    )

    assert generated["input_names"] == (
        "climb_split_0",
        "climb_split_1",
        "climb_split_2",
    )
    assert generated["input_specs"][1]["path"] == (
        "Specs",
        "Power",
        "LamDwn",
        "Clb",
        1,
    )
    assert generated["entries"][2]["point"] == 2
    assert generated["design_vars"][0]["lower"] == 0.0
    assert generated["design_vars"][0]["upper"] == 1.0


def test_mission_split_schedule_problem_optimizes_multiple_points():
    """Check generated schedule variables optimize several mission points."""

    problem = make_fast_mission_split_schedule_optimization_problem(
        aircraft=make_schedule_aircraft(),
        schedule_specs=[make_climb_schedule_spec()],
        output_specs=[
            {
                "name": "schedule_objective",
                "path": ("schedule_objective",),
            },
        ],
        runner=schedule_objective_runner,
        objective={
            "name": "schedule_objective",
        },
        driver_options={
            "maxiter": 60,
            "tol": 1.0e-10,
        },
    )
    problem.setup()
    result = problem.run_driver()

    assert result.success
    assert problem.fast_split_schedule_metadata["input_names"] == (
        "climb_split_0",
        "climb_split_1",
        "climb_split_2",
    )
    assert abs(problem.get_val("climb_split_0")[0] - 0.15) < 1.0e-5
    assert abs(problem.get_val("climb_split_1")[0] - 0.72) < 1.0e-5
    assert abs(problem.get_val("climb_split_2")[0] - 0.35) < 1.0e-5
    assert problem.get_val("schedule_objective")[0] < 1.0e-10


def make_downstream_split_spec():
    """Return a split spec for the fake downstream operation matrix."""

    return {
        "label": "downstream",
        "prefix": "downstream",
        "target": "aircraft",
        "matrix_path": ("Specs", "Propulsion", "PropArch", "OperDwn"),
        "architecture_path": ("Specs", "Propulsion", "PropArch", "Arch"),
        "axis": "row",
        "lower": 0.0,
        "upper": 1.0,
    }


def make_climb_schedule_spec():
    """Return a split schedule spec for all climb mission points."""

    return {
        "label": "climb split",
        "prefix": "climb_split",
        "target": "aircraft",
        "path": ("Specs", "Power", "LamDwn", "Clb"),
        "points": (0, 1, 2),
        "lower": 0.0,
        "upper": 1.0,
    }


def make_split_aircraft():
    """Return a tiny FAST-shaped aircraft with two branching split rows."""

    return {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "Arch": [
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                    "OperDwn": [
                        [0.0, 0.5, 0.5, 0.0],
                        [0.0, 0.0, 0.5, 0.5],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                },
            },
        },
    }


def make_merge_aircraft():
    """Return a tiny Arch-only aircraft where two sources merge at one node."""

    return {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "Arch": [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                },
            },
        },
    }


def make_branch_aircraft():
    """Return a tiny Arch-only aircraft where one source branches downstream."""

    return {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "Arch": [
                        [0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                },
            },
        },
    }


def make_merge_branch_aircraft():
    """Return an Arch-only aircraft with one merge and one downstream branch."""

    return {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "Arch": [
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                },
            },
        },
    }


def make_schedule_aircraft():
    """Return a tiny FAST-shaped aircraft with a climb split schedule."""

    return {
        "Specs": {
            "Power": {
                "LamDwn": {
                    "Clb": [0.3, 0.3, 0.3],
                },
            },
        },
    }


def split_objective_runner(aircraft, _mission):
    """Return a quadratic objective with a known feasible split optimum."""

    split = np.asarray(
        aircraft["Specs"]["Propulsion"]["PropArch"]["OperDwn"],
        dtype=float,
    )
    objective = (
        (split[0, 1] - 0.2) ** 2
        + (split[0, 2] - 0.8) ** 2
        + (split[1, 2] - 0.65) ** 2
        + (split[1, 3] - 0.35) ** 2
    )
    return {
        "split_objective": objective,
    }


def upstream_split_objective_runner(aircraft, _mission):
    """Return a quadratic objective for one inferred upstream split group."""

    split = np.asarray(
        aircraft["Specs"]["Propulsion"]["PropArch"]["OperUps"],
        dtype=float,
    )
    objective = (split[0, 2] - 0.25) ** 2 + (split[1, 2] - 0.75) ** 2
    return {
        "split_objective": objective,
    }


def downstream_split_objective_runner(aircraft, _mission):
    """Return a quadratic objective for one inferred downstream split group."""

    split = np.asarray(
        aircraft["Specs"]["Propulsion"]["PropArch"]["OperDwn"],
        dtype=float,
    )
    objective = (split[0, 1] - 0.4) ** 2 + (split[0, 2] - 0.6) ** 2
    return {
        "split_objective": objective,
    }


def merge_branch_split_objective_runner(aircraft, _mission):
    """Return a quadratic objective for inferred merge and branch splits."""

    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]
    downstream = np.asarray(prop_arch["OperDwn"], dtype=float)
    upstream = np.asarray(prop_arch["OperUps"], dtype=float)
    objective = (
        (downstream[2, 3] - 0.3) ** 2
        + (downstream[2, 4] - 0.7) ** 2
        + (upstream[0, 2] - 0.8) ** 2
        + (upstream[1, 2] - 0.2) ** 2
    )
    return {
        "split_objective": objective,
    }


def schedule_objective_runner(aircraft, _mission):
    """Return a quadratic objective with distinct per-point optima."""

    split = np.asarray(aircraft["Specs"]["Power"]["LamDwn"]["Clb"], dtype=float)
    target = np.asarray([0.15, 0.72, 0.35])
    return {
        "schedule_objective": np.sum((split - target) ** 2),
    }
