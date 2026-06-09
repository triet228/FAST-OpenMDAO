# tests/test_optimization_components.py

"""Tests for OpenMDAO FAST optimization helper components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    BatteryEnergyAvailable,
    CruisePowerAvailableConstraint,
    DesignSplitBounds,
    ElectricMotorPowerAvailable,
    FeasibleStep,
    GaussianEliminationPivot,
    HessianUpdate,
    MeritFunction,
    OneBasedHistoryValues,
    OperationalObjective,
    OperationalSplitConstraints,
    PowerManagementObjective,
    PowerLimitConstraints,
)
from fast_python.optimization import (
    battery_energy_available,
    con_size_opt,
    electric_motor_power_available,
    feas_step,
    gauss_elim,
    hess_upd,
    merit_function,
    one_based_history_values,
    operational_objective_value,
    operational_split_constraint_blocks,
    power_management_objective,
    power_limit_constraints,
)


def test_available_power_and_energy_match_fast_python():
    """Check available power and energy parity with FAST-Python."""

    motor = om.Problem()
    motor.model.add_subsystem(
        "motor",
        ElectricMotorPowerAvailable(),
        promotes=["*"],
    )
    motor.setup()
    motor.set_val("electric_motor_specific_power", 6200.0, units="W/kg")
    motor.set_val("electric_motor_weight", 135.0, units="kg")
    motor.run_model()

    assert np.isclose(
        motor.get_val("electric_motor_power_available", units="W")[0],
        electric_motor_power_available(make_optimization_aircraft()),
    )

    battery = om.Problem()
    battery.model.add_subsystem(
        "battery",
        BatteryEnergyAvailable(),
        promotes=["*"],
    )
    battery.setup()
    battery.set_val("battery_specific_energy", 745000.0, units="J/kg")
    battery.set_val("battery_weight", 860.0, units="kg")
    battery.run_model()

    assert np.isclose(
        battery.get_val("battery_energy_available", units="J")[0],
        battery_energy_available(make_optimization_aircraft()),
    )


def test_power_limit_constraints_match_fast_python():
    """Check paired power-limit residual parity with FAST-Python."""

    used = np.asarray([20.0, 50.0, 75.0])
    available = np.asarray([100.0, 80.0, 75.0])
    problem = om.Problem()
    problem.model.add_subsystem(
        "limits",
        PowerLimitConstraints(vec_size=used.size, eps=1.0e-6),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("used", used)
    problem.set_val("available", available)
    problem.run_model()

    expected_lower, expected_upper = power_limit_constraints(
        True,
        {
            "used": used,
            "available": available,
        },
        "used",
        "available",
        1.0e-6,
    )

    assert np.allclose(problem.get_val("lower_limit"), expected_lower)
    assert np.allclose(problem.get_val("upper_limit"), expected_upper)


def test_feasible_step_matches_fast_python():
    """Check interior-point feasible step parity with FAST-Python."""

    slack = np.asarray([0.5, 0.8, 0.3])
    slack_direction = np.asarray([-1.0, -3.0, 0.4])
    problem = om.Problem()
    problem.model.add_subsystem(
        "feasible",
        FeasibleStep(num_constraints=slack.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("slack", slack)
    problem.set_val("slack_direction", slack_direction)
    problem.run_model()

    assert np.isclose(
        problem.get_val("feasible_step")[0],
        feas_step(slack.size, slack, slack_direction),
    )


def test_gaussian_elimination_pivot_matches_fast_python():
    """Check one FAST Gaussian-elimination pivot against FAST-Python."""

    matrix = np.asarray(
        [
            [2.0, -1.0, 0.5],
            [4.0, 3.0, -2.0],
            [1.5, 2.0, 5.0],
        ]
    )
    problem = om.Problem()
    problem.model.add_subsystem(
        "pivot",
        GaussianEliminationPivot(num_rows=3, num_cols=3, pivot_row=2, pivot_col=1),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("matrix", matrix)
    problem.run_model()

    assert np.allclose(
        problem.get_val("eliminated_matrix"),
        gauss_elim(matrix, 2, 1),
    )


def test_hessian_update_matches_fast_python():
    """Check FAST damped BFGS Hessian update parity with FAST-Python."""

    hessian, step, gradient_delta = make_hessian_update_case()
    problem = om.Problem()
    problem.model.add_subsystem(
        "hessian",
        HessianUpdate(size=step.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("hessian", hessian)
    problem.set_val("step", step)
    problem.set_val("gradient_delta", gradient_delta)
    problem.run_model()

    assert np.allclose(
        problem.get_val("updated_hessian"),
        hess_upd(hessian, step, gradient_delta),
    )


def test_one_based_history_values_match_fast_python():
    """Check fixed-index mission-history extraction parity with FAST-Python."""

    history_values = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0])
    indices = np.asarray([1, 3, 5])
    problem = om.Problem()
    problem.model.add_subsystem(
        "history",
        OneBasedHistoryValues(history_size=history_values.size, indices=indices),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("history_values", history_values)
    problem.run_model()

    assert np.allclose(
        problem.get_val("selected_values"),
        one_based_history_values(history_values, indices),
    )


def test_merit_function_matches_fast_python():
    """Check FAST line-search merit value parity."""

    objective = 12.5
    g = np.asarray([-0.2, 0.1])
    h = np.asarray([0.05])
    slack = np.asarray([0.4, 0.7])
    mu = 0.3
    problem = om.Problem()
    problem.model.add_subsystem(
        "merit",
        MeritFunction(num_inequality=2, num_equality=1, use_slack=True),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("objective", objective)
    problem.set_val("inequality_constraints", g)
    problem.set_val("equality_constraints", h)
    problem.set_val("slack", slack)
    problem.set_val("barrier_parameter", mu)
    problem.run_model()

    assert np.isclose(
        problem.get_val("merit")[0],
        merit_function(
            make_merit_objective(objective),
            np.asarray([1.0]),
            make_merit_constraints(g, h),
            slack,
            mu,
        ),
    )

    no_slack = om.Problem()
    no_slack.model.add_subsystem(
        "merit",
        MeritFunction(num_inequality=2, num_equality=1, use_slack=False),
        promotes=["*"],
    )
    no_slack.setup()
    no_slack.set_val("objective", objective)
    no_slack.set_val("inequality_constraints", g)
    no_slack.set_val("equality_constraints", h)
    no_slack.run_model()

    assert np.isclose(
        no_slack.get_val("merit")[0],
        merit_function(
            make_merit_objective(objective),
            np.asarray([1.0]),
            make_merit_constraints(g, h),
        ),
    )


def test_operational_split_constraints_match_fast_python():
    """Check operational split residual parity with FAST-Python."""

    operational_splits = np.asarray([0.2, 0.3, 0.4, 0.1, 0.2, 0.3])
    design_splits = np.asarray([0.5, 0.4])
    x = np.concatenate([operational_splits, design_splits])
    aircraft = make_operational_split_aircraft(
        npoint=3,
        nopers=6,
        ndvars=8,
        narg=2,
        lam_max=0.8,
    )
    expected = operational_split_constraint_blocks(
        x,
        aircraft,
        [("TS", 1, 1)],
        1.0e-6,
    )["TS"]

    problem = om.Problem()
    problem.model.add_subsystem(
        "splits",
        OperationalSplitConstraints(
            npoint=3,
            nsplit=2,
            design_active=True,
            eps=1.0e-6,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("operational_splits", operational_splits)
    problem.set_val("design_splits", design_splits)
    problem.run_model()

    assert np.allclose(problem.get_val("split_constraints"), expected)

    fixed_aircraft = make_operational_split_aircraft(
        npoint=3,
        nopers=6,
        ndvars=6,
        narg=2,
        lam_max=0.8,
    )
    fixed_expected = operational_split_constraint_blocks(
        operational_splits,
        fixed_aircraft,
        [("TS", 1, 0)],
        1.0e-6,
    )["TS"]
    fixed_problem = om.Problem()
    fixed_problem.model.add_subsystem(
        "splits",
        OperationalSplitConstraints(
            npoint=3,
            nsplit=2,
            design_active=False,
            lam_max=0.8,
            eps=1.0e-6,
        ),
        promotes=["*"],
    )
    fixed_problem.setup()
    fixed_problem.set_val("operational_splits", operational_splits)
    fixed_problem.run_model()

    assert np.allclose(fixed_problem.get_val("split_constraints"), fixed_expected)


def test_design_split_bounds_match_fast_python_con_size_opt():
    """Check FAST ConSizeOpt design split bound residuals."""

    design_splits = np.asarray([0.2, 0.5, 0.8])
    problem = om.Problem()
    problem.model.add_subsystem(
        "bounds",
        DesignSplitBounds(vec_size=design_splits.size),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("design_splits", design_splits)
    problem.run_model()

    expected, _, _, _ = con_size_opt(
        design_splits,
        0,
        make_design_split_bound_aircraft(design_splits.size),
    )

    assert np.allclose(problem.get_val("lower_bounds"), expected[:design_splits.size])
    assert np.allclose(problem.get_val("upper_bounds"), expected[design_splits.size:])


def test_cruise_power_available_constraint_matches_fast_python_con_size_opt():
    """Check FAST ConSizeOpt cruise power availability residuals."""

    cruise_power = np.asarray([8.0e5, 9.5e5])
    available_power = np.asarray([1.0e6, 1.0e6])
    problem = om.Problem()
    problem.model.add_subsystem(
        "cruise",
        CruisePowerAvailableConstraint(vec_size=cruise_power.size, eps=1.0e-6),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("cruise_power", cruise_power, units="W")
    problem.set_val("gas_turbine_power_available", available_power, units="W")
    problem.run_model()

    expected, _, _, _ = con_size_opt(
        np.asarray([]),
        0,
        make_cruise_power_constraint_aircraft(cruise_power, available_power),
    )

    assert np.allclose(problem.get_val("cruise_power_constraint"), expected)


def test_optimization_objectives_match_fast_python():
    """Check operational and power-management objective parity."""

    for objective_type in ("DOC", "FuelBurn", "Energy"):
        operational = om.Problem()
        operational.model.add_subsystem(
            "objective",
            OperationalObjective(objective_type=objective_type),
            promotes=["*"],
        )
        operational.setup()
        set_objective_inputs(operational)
        operational.run_model()

        assert np.isclose(
            operational.get_val("operational_objective")[0],
            operational_objective_value(make_objective_aircraft(objective_type)),
        )

        management = om.Problem()
        management.model.add_subsystem(
            "objective",
            PowerManagementObjective(objective_type=objective_type),
            promotes=["*"],
        )
        management.setup()
        set_objective_inputs(management)
        management.run_model()

        assert np.isclose(
            management.get_val("power_management_objective")[0],
            power_management_objective(
                make_objective_aircraft(objective_type),
                objective_type,
            ),
        )


def test_optimization_helpers_declare_analytic_partials():
    """Check optimization helper derivatives against finite difference."""

    cases = [
        (
            "motor",
            ElectricMotorPowerAvailable(),
            {
                "electric_motor_specific_power": 6200.0,
                "electric_motor_weight": 135.0,
            },
        ),
        (
            "battery",
            BatteryEnergyAvailable(),
            {
                "battery_specific_energy": 745000.0,
                "battery_weight": 860.0,
            },
        ),
        (
            "limits",
            PowerLimitConstraints(vec_size=3, eps=1.0e-6),
            {
                "used": np.asarray([20.0, 50.0, 75.0]),
                "available": np.asarray([100.0, 80.0, 75.0]),
            },
        ),
        (
            "feasible",
            FeasibleStep(num_constraints=3),
            {
                "slack": np.asarray([0.5, 0.8, 0.3]),
                "slack_direction": np.asarray([-1.0, -3.0, 0.4]),
            },
        ),
        (
            "pivot",
            GaussianEliminationPivot(num_rows=3, num_cols=3, pivot_row=2, pivot_col=1),
            {
                "matrix": np.asarray(
                    [
                        [2.0, -1.0, 0.5],
                        [4.0, 3.0, -2.0],
                        [1.5, 2.0, 5.0],
                    ]
                ),
            },
        ),
        (
            "hessian",
            HessianUpdate(size=3),
            {
                "hessian": make_hessian_update_case()[0],
                "step": make_hessian_update_case()[1],
                "gradient_delta": make_hessian_update_case()[2],
            },
        ),
        (
            "history",
            OneBasedHistoryValues(history_size=5, indices=np.asarray([1, 3, 5])),
            {
                "history_values": np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]),
            },
        ),
        (
            "merit",
            MeritFunction(num_inequality=2, num_equality=1, use_slack=True),
            {
                "objective": 12.5,
                "inequality_constraints": np.asarray([-0.2, 0.1]),
                "equality_constraints": np.asarray([0.05]),
                "slack": np.asarray([0.4, 0.7]),
                "barrier_parameter": 0.3,
            },
        ),
        (
            "merit_no_slack",
            MeritFunction(num_inequality=2, num_equality=1, use_slack=False),
            {
                "objective": 12.5,
                "inequality_constraints": np.asarray([-0.2, 0.1]),
                "equality_constraints": np.asarray([0.05]),
            },
        ),
        (
            "splits",
            OperationalSplitConstraints(
                npoint=3,
                nsplit=2,
                design_active=True,
                eps=1.0e-6,
            ),
            {
                "operational_splits": np.asarray([0.2, 0.3, 0.4, 0.1, 0.2, 0.3]),
                "design_splits": np.asarray([0.5, 0.4]),
            },
        ),
        (
            "fixed_splits",
            OperationalSplitConstraints(
                npoint=3,
                nsplit=2,
                design_active=False,
                lam_max=0.8,
                eps=1.0e-6,
            ),
            {
                "operational_splits": np.asarray([0.2, 0.3, 0.4, 0.1, 0.2, 0.3]),
            },
        ),
        (
            "design_bounds",
            DesignSplitBounds(vec_size=3),
            {
                "design_splits": np.asarray([0.2, 0.5, 0.8]),
            },
        ),
        (
            "cruise_power",
            CruisePowerAvailableConstraint(vec_size=2, eps=1.0e-6),
            {
                "cruise_power": np.asarray([8.0e5, 9.5e5]),
                "gas_turbine_power_available": np.asarray([1.0e6, 1.0e6]),
            },
        ),
        (
            "operational_objective",
            OperationalObjective(objective_type="Energy"),
            {
                "fuel_burn": 42.0,
                "fuel_energy": 1.5e9,
                "battery_energy": 2.0e8,
            },
        ),
        (
            "management_objective",
            PowerManagementObjective(objective_type="FuelBurn"),
            {
                "fuel_burn": 42.0,
                "fuel_energy": 1.5e9,
                "battery_energy": 2.0e8,
            },
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
            step=1.0e-5,
        )

        for partial_data in partials[name].values():
            assert (
                partial_data["abs error"].forward < 2.0e-3
                or partial_data["rel error"].forward < 1.0e-8
            )


def make_optimization_aircraft():
    """Return minimal FAST aircraft dictionary for optimization helpers."""

    return {
        "Specs": {
            "Power": {
                "P_W": {
                    "EM": 6200.0,
                },
                "SpecEnergy": {
                    "Batt": 745000.0,
                },
            },
            "Weight": {
                "EM": 135.0,
                "Batt": 860.0,
            },
        },
    }


def make_operational_split_aircraft(npoint, nopers, ndvars, narg, lam_max):
    """Return minimal aircraft dictionary for split constraint helpers."""

    return {
        "Settings": {
            "nargTS": narg,
        },
        "Specs": {
            "Power": {
                "LamTS": {
                    "SLS": lam_max,
                },
            },
        },
        "PowerOpt": {
            "npoint": npoint,
            "nopers": nopers,
            "ndvars": ndvars,
        },
    }


def make_merit_objective(value):
    """Return FAST objective callable with a fixed value."""

    def objective(_x, _need_grad):
        return value, np.asarray([0.0]), None

    return objective


def make_merit_constraints(g, h):
    """Return FAST constraint callable with fixed residuals."""

    def constraints(_x, _need_grad, _info=None):
        return g, h, np.zeros((len(g), 1)), np.zeros((len(h), 1))

    return constraints


def make_hessian_update_case():
    """Return nonsingular values for FAST damped BFGS update checks."""

    return (
        np.asarray(
            [
                [3.0, 0.4, 0.2],
                [0.4, 2.5, 0.3],
                [0.2, 0.3, 1.8],
            ]
        ),
        np.asarray([0.3, -0.2, 0.4]),
        np.asarray([0.7, -0.1, 0.5]),
    )


def make_design_split_bound_aircraft(num_design_splits):
    """Return minimal aircraft with active design split constraints only."""

    return {
        "Settings": {
            "Analysis": {
                "Type": 0,
            },
        },
        "Specs": {
            "Propulsion": {
                "PropArch": {},
            },
        },
        "PowerOpt": {
            "Settings": {
                "DesnTS": 1,
            },
            "Constraints": {},
            "nopers": 0,
            "ndesns": num_design_splits,
            "ndvars": num_design_splits,
        },
    }


def make_cruise_power_constraint_aircraft(cruise_power, available_power):
    """Return minimal aircraft with active cruise power constraint only."""

    return {
        "Settings": {
            "Analysis": {
                "Type": 1,
            },
        },
        "Specs": {
            "Propulsion": {
                "PropArch": {},
            },
        },
        "PowerOpt": {
            "Settings": {},
            "Constraints": {
                "DesCrsPow": cruise_power,
                "DesPavGT": available_power,
            },
            "nopers": 0,
            "ndesns": 0,
            "ndvars": 0,
        },
    }


def set_objective_inputs(problem):
    """Set common objective component inputs."""

    problem.set_val("fuel_burn", 42.0, units="kg")
    problem.set_val("fuel_energy", 1.5e9, units="J")
    problem.set_val("battery_energy", 2.0e8, units="J")


def make_objective_aircraft(objective_type):
    """Return FAST-shaped objective fixture."""

    return {
        "PowerOpt": {
            "ObjFun": objective_type,
        },
        "Mission": {
            "History": {
                "SI": {
                    "Weight": {
                        "Fburn": [0.0, 42.0],
                    },
                    "Energy": {
                        "Fuel": [0.0, 1.5e9],
                        "Batt": [0.0, 2.0e8],
                    },
                },
            },
        },
    }
