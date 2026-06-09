# tests/test_battery_components.py

"""Tests for OpenMDAO FAST battery primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    AvailableCellCapacity,
    BatteryCyclingAging,
    BatteryCurrent,
    DetailedBatterySizing,
    BatteryPowerHistory,
    BatteryPowerStep,
    BatteryWeightFromEnergy,
)
from fast_python.battery import (
    available_cell_capacity,
    charging,
    cycling_aging_parameters,
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


def test_detailed_battery_sizing_matches_fast_python_resize_battery():
    """Check detailed ResizeBattery cell sizing parity with FAST-Python."""

    values = detailed_battery_sizing_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "sizing",
        DetailedBatterySizing(
            num_points=values["soc"].shape[0],
            num_batteries=values["soc"].shape[1],
        ),
        promotes=["*"],
    )
    problem.setup()
    set_detailed_battery_sizing_values(problem, values)
    problem.run_model()

    aircraft = make_detailed_resize_battery_aircraft(values)
    result = resize_battery(aircraft)
    expected_weight = np.asarray(result["Specs"]["Weight"]["Batt"])
    expected_parallel = np.asarray(result["Specs"]["Power"]["Battery"]["ParCells"])
    expected_c_rate = np.asarray(
        result["Mission"]["History"]["SI"]["Power"]["C_rate"]
    )

    assert np.allclose(problem.get_val("parallel_cells"), expected_parallel)
    assert np.allclose(problem.get_val("battery_weight", units="kg"), expected_weight)
    assert np.allclose(problem.get_val("c_rate"), expected_c_rate)


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


def test_battery_power_history_matches_fast_python_discharge_and_charge():
    """Check fixed-length battery history parity with FAST-Python."""

    aircraft = make_power_step_aircraft()
    powers = np.asarray([1000.0, 850.0, 620.0])
    times = np.asarray([60.0, 75.0, 45.0])

    for is_discharge, drop_initial_soc, requested_power, fast_function in [
        (True, False, powers, discharging),
        (False, True, -0.5 * powers, charging),
    ]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "history",
            BatteryPowerHistory(
                num_steps=len(powers),
                is_discharge=is_discharge,
                drop_initial_soc=drop_initial_soc,
            ),
            promotes=["*"],
        )
        problem.setup()
        set_power_history_values(problem, requested_power, times)
        problem.run_model()
        expected = fast_function(aircraft, requested_power, times, 80.0, 10, 100)

        for output, expected_index in [
            ("voltage", 0),
            ("current", 1),
            ("output_power", 2),
            ("capacity", 3),
            ("soc", 4),
            ("c_rate", 5),
        ]:
            assert np.allclose(problem.get_val(output), expected[expected_index])


def test_battery_cycling_aging_matches_fast_python_empirical_formula():
    """Check cycling-aging SOH parity with FAST-Python chemistry constants."""

    for chemistry in [1, 2]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "aging",
            BatteryCyclingAging(chemistry=chemistry),
            promotes=["*"],
        )
        problem.setup()
        set_cycling_aging_values(problem)
        problem.run_model()

        params = cycling_aging_parameters(chemistry)
        values = cycling_aging_values()
        fec = (
            values["discharge_capacity_delta"] + values["charge_capacity_delta"]
        ) / (2.0 * values["cap_cell"] * values["parallel_cells"])
        fec += values["cumulative_fecs"]
        temp_actual = values["operating_temperature"] + 273.15
        theta_temp = params["coeff_T"] * (
            (temp_actual - params["temp_ref"]) / temp_actual
        )
        theta_dod = params["coeff_DOD"] * values["depth_of_discharge"]
        theta_c = params["coeff_Cch"] * values["charge_c_rate"]
        theta_c += params["coeff_Cdch"] * values["discharge_c_rate"]
        soc_shape = 1.0 + params["coeff_mSOC"] * values["mean_soc"] * (
            1.0 - values["mean_soc"] / (2.0 * params["mSOC_ref"])
        )
        degradation = params["beta"] * np.exp(theta_temp + theta_dod + theta_c)
        degradation *= soc_shape * fec ** params["alpha"]

        assert np.isclose(problem.get_val("full_equivalent_cycles")[0], fec)
        assert np.isclose(problem.get_val("state_of_health")[0], 100.0 - degradation)


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
            "detailed_sizing",
            DetailedBatterySizing(num_points=3, num_batteries=2),
            detailed_battery_sizing_derivative_values(),
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
        (
            "power_history",
            BatteryPowerHistory(num_steps=3, is_discharge=True),
            power_history_values(np.asarray([1000.0, 850.0, 620.0])),
        ),
        (
            "charge_history",
            BatteryPowerHistory(
                num_steps=3,
                is_discharge=False,
                drop_initial_soc=True,
            ),
            power_history_values(np.asarray([-500.0, -425.0, -310.0])),
        ),
        (
            "aging",
            BatteryCyclingAging(chemistry=1),
            cycling_aging_values(),
        ),
        (
            "aging_lfp",
            BatteryCyclingAging(chemistry=2),
            cycling_aging_values(),
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
            tolerance = 5.0e-4 if "history" in name else 1.0e-4 if "step" in name else 1.0e-6
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


def make_detailed_resize_battery_aircraft(values):
    """Return aircraft data for FAST-Python detailed battery resizing."""

    num_batteries = values["soc"].shape[1]
    src_type = np.zeros(num_batteries)
    energy = np.ones_like(values["soc"])
    return {
        "Specs": {
            "Propulsion": {
                "PropArch": {
                    "SrcType": src_type.tolist(),
                    "Arch": np.eye(num_batteries).tolist(),
                },
            },
            "Power": {
                "SpecEnergy": {
                    "Batt": values["battery_specific_energy"],
                },
                "Battery": {
                    "SerCells": values["series_cells"],
                    "ParCells": values["initial_parallel_cells"],
                },
            },
            "Battery": {
                "NomVolCell": values["nominal_cell_voltage"],
                "CapCell": values["cap_cell"],
                "MinSOC": values["min_soc"],
                "MaxAllowCRate": values["max_c_rate"],
            },
        },
        "Mission": {
            "History": {
                "SI": {
                    "Energy": {
                        "E_ES": energy.tolist(),
                        "Eleft_ES": np.zeros_like(energy).tolist(),
                    },
                    "Power": {
                        "SOC": values["soc"].tolist(),
                        "Current": values["current"].tolist(),
                    },
                },
            },
        },
        "Settings": {
            "DetailedBatt": 1,
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


def detailed_battery_sizing_values():
    """Return inputs that trigger FAST detailed battery resizing branches."""

    return {
        "soc": np.asarray(
            [
                [95.0, 88.0],
                [30.0, 55.0],
                [18.0, 33.0],
            ]
        ),
        "current": np.asarray(
            [
                [16.0, 10.0],
                [24.0, 18.0],
                [32.0, 12.0],
            ]
        ),
        "cap_cell": 2.5,
        "nominal_cell_voltage": 3.7,
        "min_soc": 20.0,
        "max_c_rate": 1.0,
        "initial_parallel_cells": np.asarray([10.0, 12.0]),
        "series_cells": np.asarray([90.0, 96.0]),
        "battery_specific_energy": 0.45 * 3.6e6,
    }


def detailed_battery_sizing_derivative_values():
    """Return detailed sizing inputs away from ceil and max thresholds."""

    values = detailed_battery_sizing_values()
    values["soc"] = np.asarray(
        [
            [96.0, 92.0],
            [45.0, 55.0],
            [35.0, 40.0],
        ]
    )
    values["current"] = np.asarray(
        [
            [1.6, 1.0],
            [2.4, 1.8],
            [3.2, 1.2],
        ]
    )
    values["max_c_rate"] = 5.0
    return values


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


def power_history_values(requested_power):
    """Return OpenMDAO inputs for a battery power history."""

    values = power_step_values(float(np.asarray(requested_power).reshape(-1)[0]))
    values["requested_power"] = requested_power
    values["time"] = np.asarray([60.0, 75.0, 45.0])
    return values


def cycling_aging_values():
    """Return OpenMDAO inputs for FAST empirical battery cycling aging."""

    return {
        "depth_of_discharge": 55.0,
        "discharge_c_rate": 0.8,
        "charge_c_rate": 0.6,
        "mean_soc": 62.0,
        "discharge_capacity_delta": 10.0,
        "charge_capacity_delta": 8.0,
        "cap_cell": 2.4,
        "parallel_cells": 12.0,
        "cumulative_fecs": 120.0,
        "operating_temperature": 30.0,
    }


def set_power_step_values(problem, requested_power):
    """Set scalar OpenMDAO battery power-step inputs."""

    for variable, value in power_step_values(requested_power).items():
        problem.set_val(variable, value)


def set_power_history_values(problem, requested_power, time):
    """Set OpenMDAO battery power-history inputs."""

    for variable, value in power_history_values(requested_power).items():
        if variable == "time":
            value = time

        problem.set_val(variable, value)


def set_detailed_battery_sizing_values(problem, values):
    """Set OpenMDAO detailed battery sizing inputs."""

    problem.set_val("soc", values["soc"])
    problem.set_val("current", values["current"], units="A")
    problem.set_val("cap_cell", values["cap_cell"])
    problem.set_val(
        "nominal_cell_voltage",
        values["nominal_cell_voltage"],
        units="V",
    )
    problem.set_val("min_soc", values["min_soc"])
    problem.set_val("max_c_rate", values["max_c_rate"])
    problem.set_val("initial_parallel_cells", values["initial_parallel_cells"])
    problem.set_val("series_cells", values["series_cells"])
    problem.set_val(
        "battery_specific_energy",
        values["battery_specific_energy"],
        units="J/kg",
    )


def set_cycling_aging_values(problem):
    """Set scalar OpenMDAO battery cycling-aging inputs."""

    for variable, value in cycling_aging_values().items():
        problem.set_val(variable, value)
