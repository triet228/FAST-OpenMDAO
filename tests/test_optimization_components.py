# tests/test_optimization_components.py

"""Tests for OpenMDAO FAST optimization helper components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    BatteryEnergyAvailable,
    ElectricMotorPowerAvailable,
    OperationalObjective,
    OperationalSplitConstraints,
    PowerManagementObjective,
    PowerLimitConstraints,
)
from fast_python.optimization import (
    battery_energy_available,
    electric_motor_power_available,
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
