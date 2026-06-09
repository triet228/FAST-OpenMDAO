# tests/test_optimization_components.py

"""Tests for OpenMDAO FAST optimization helper components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    BatteryEnergyAvailable,
    ElectricMotorPowerAvailable,
    PowerLimitConstraints,
)
from fast_python.optimization import (
    battery_energy_available,
    electric_motor_power_available,
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
                partial_data["abs error"].forward < 1.0e-3
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
