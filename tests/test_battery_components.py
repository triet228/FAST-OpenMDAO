# tests/test_battery_components.py

"""Tests for OpenMDAO FAST battery primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    AvailableCellCapacity,
    BatteryCurrent,
    BatteryPowerStep,
    BatteryWeightFromEnergy,
)
from fast_python.battery import (
    available_cell_capacity,
    charging,
    discharging,
    resize_battery,
    solve_battery_current,
)


def test_available_cell_capacity_matches_fast_python():
    """Check effective cell capacity parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "capacity",
        AvailableCellCapacity(analysis_type=-2, degradation=1),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("cap_cell", 2.4)
    problem.set_val("state_of_health", 90.0)
    problem.run_model()

    assert np.isclose(
        problem.get_val("available_cell_capacity")[0],
        available_cell_capacity(make_battery_aircraft(2.4, 90.0)),
    )


def test_battery_current_matches_fast_python_real_roots():
    """Check battery current root parity with FAST-Python."""

    for is_discharge, requested_power in [(True, 10.0), (False, -5.0)]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "current",
            BatteryCurrent(is_discharge=is_discharge),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("hot_voltage", 0.01)
        problem.set_val("cold_voltage", 4.0)
        problem.set_val("requested_cell_power", requested_power, units="W")
        problem.run_model()

        assert np.isclose(
            problem.get_val("cell_current", units="A")[0],
            solve_battery_current(0.01, 4.0, requested_power, is_discharge),
        )


def test_battery_weight_from_energy_matches_fast_python_resize_battery():
    """Check simple ResizeBattery energy sizing parity with FAST-Python."""

    src_type = np.asarray([1.0, 0.0, 0.0])
    energy = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [10.0, 100.0, 120.0],
        ]
    )
    specific_energy = 0.4 * 3.6e6
    problem = om.Problem()
    problem.model.add_subsystem(
        "weight",
        BatteryWeightFromEnergy(src_type=src_type, npoint=2),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("battery_source_energy", energy, units="J")
    problem.set_val("battery_specific_energy", specific_energy, units="J/kg")
    problem.run_model()

    result = resize_battery(make_resize_battery_aircraft(src_type, energy, specific_energy))

    assert np.allclose(
        problem.get_val("battery_weight", units="kg"),
        np.asarray(result["Specs"]["Weight"]["Batt"]),
    )


def test_battery_power_step_matches_fast_python_discharge_and_charge():
    """Check one-step battery equivalent-circuit parity."""

    aircraft = make_power_step_aircraft()

    for is_discharge, requested_power, fast_function in [
        (True, 1000.0, discharging),
        (False, -500.0, charging),
    ]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "step",
            BatteryPowerStep(is_discharge=is_discharge),
            promotes=["*"],
        )
        problem.setup()
        set_power_step_values(problem, requested_power)
        problem.run_model()
        expected = fast_function(aircraft, requested_power, 60.0, 80.0, 10, 100)

        for output, expected_index in [
            ("voltage", 0),
            ("current", 1),
            ("output_power", 2),
            ("capacity", 3),
            ("soc_end", 4),
            ("c_rate", 5),
        ]:
            assert np.isclose(problem.get_val(output)[0], expected[expected_index][-1])


def test_battery_primitives_declare_analytic_partials():
    """Check battery primitive derivatives against finite difference."""

    cases = [
        (
            "capacity",
            AvailableCellCapacity(analysis_type=-2, degradation=1),
            {"cap_cell": 2.4, "state_of_health": 90.0},
        ),
        (
            "weight",
            BatteryWeightFromEnergy(src_type=np.asarray([1.0, 0.0, 0.0]), npoint=2),
            {
                "battery_source_energy": np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [10.0, 100.0, 120.0],
                    ]
                ),
                "battery_specific_energy": 0.4 * 3.6e6,
            },
        ),
        (
            "current",
            BatteryCurrent(is_discharge=True),
            {
                "hot_voltage": 0.01,
                "cold_voltage": 4.0,
                "requested_cell_power": 10.0,
            },
        ),
        (
            "charge_current",
            BatteryCurrent(is_discharge=False),
            {
                "hot_voltage": 0.01,
                "cold_voltage": 4.0,
                "requested_cell_power": -5.0,
            },
        ),
        (
            "power_step",
            BatteryPowerStep(is_discharge=True),
            power_step_values(1000.0),
        ),
        (
            "charge_step",
            BatteryPowerStep(is_discharge=False),
            power_step_values(-500.0),
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
            tolerance = 1.0e-4 if "step" in name else 1.0e-6
            assert partial_data["abs error"].forward < tolerance


def make_battery_aircraft(cap_cell, state_of_health):
    """Return minimal aircraft dictionary for FAST-Python battery helpers."""

    return {
        "Settings": {
            "Analysis": {
                "Type": -2,
            },
        },
        "Specs": {
            "Battery": {
                "CapCell": cap_cell,
                "Degradation": 1,
                "SOH": [100.0, state_of_health],
            },
        },
    }


def make_resize_battery_aircraft(src_type, source_energy, specific_energy):
    """Return minimal aircraft dictionary for FAST-Python ResizeBattery."""

    return {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "SrcType": src_type.tolist(),
                    "Arch": 1,
                }
            },
            "Power": {
                "SpecEnergy": {
                    "Batt": specific_energy,
                }
            },
        },
        "Mission": {
            "History": {
                "SI": {
                    "Energy": {
                        "E_ES": source_energy.tolist(),
                        "Eleft_ES": np.zeros_like(source_energy).tolist(),
                    }
                }
            }
        },
        "Settings": {
            "DetailedBatt": 0,
        },
    }


def make_power_step_aircraft():
    """Return minimal aircraft dictionary for FAST-Python battery dynamics."""

    return {
        "Specs": {
            "Battery": {
                "MaxExtVolCell": 4.2,
                "IntResist": 0.01,
                "ExpVol": 0.1,
                "ExpCap": 1.0,
                "CapCell": 2.4,
            },
        },
    }


def power_step_values(requested_power):
    """Return OpenMDAO inputs for one battery power step."""

    return {
        "requested_power": requested_power,
        "time": 60.0,
        "soc_begin": 80.0,
        "parallel_cells": 10.0,
        "series_cells": 100.0,
        "max_cell_voltage": 4.2,
        "internal_resistance": 0.01,
        "exponential_voltage": 0.1,
        "exponential_capacity": 1.0,
        "cap_cell": 2.4,
        "state_of_health": 100.0,
    }


def set_power_step_values(problem, requested_power):
    """Set scalar OpenMDAO battery power-step inputs."""

    for variable, value in power_step_values(requested_power).items():
        problem.set_val(variable, value)
