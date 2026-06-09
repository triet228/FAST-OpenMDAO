# tests/test_propulsion_components.py

"""Tests for OpenMDAO FAST propulsion primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    CableWeightForSizing,
    EngineLapse,
    EngineThrustRequirement,
    PowerAvailable,
    PowerFlow,
    PowerSupplementCheck,
    SafeComponentWeight,
    ThrustSinkEfficiency,
    TransmitterFanEfficiency,
    TurbopropEngineWeightForSizing,
)
from fast_python.propulsion import (
    cable_weight_for_sizing,
    engine_lapse,
    engine_thrust_requirement,
    engine_weights_for_sizing,
    get_thrust_sink_efficiency,
    power_available,
    power_flow,
    power_supplement_check,
    safe_component_weight,
    transmitter_fan_efficiency,
)


def test_engine_lapse_matches_fast_python():
    """Check engine lapse parity for turbofan and turboprop classes."""

    for aircraft_class in ["Turbofan", "Turboprop"]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "lapse",
            EngineLapse(aircraft_class=aircraft_class),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("sea_level_static", 100000.0)
        problem.set_val("density", 0.9, units="kg/m**3")
        problem.run_model()

        assert np.isclose(
            problem.get_val("lapsed_output")[0],
            engine_lapse(100000.0, aircraft_class, 0.9),
        )


def test_safe_component_weight_matches_fast_python():
    """Check component weight parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem("weight", SafeComponentWeight(), promotes=["*"])
    problem.setup()
    problem.set_val("power", 500000.0, units="W")
    problem.set_val("power_to_weight", 2500.0)
    problem.run_model()

    assert np.isclose(
        problem.get_val("component_weight", units="kg")[0],
        safe_component_weight(500000.0, 2500.0),
    )


def test_efficiency_selectors_match_fast_python():
    """Check propulsion efficiency selector parity with FAST-Python."""

    specs = make_efficiency_specs()
    thrust_sink = om.Problem()
    thrust_sink.model.add_subsystem(
        "efficiency",
        ThrustSinkEfficiency(aircraft_class="Turboprop"),
        promotes=["*"],
    )
    thrust_sink.setup()
    thrust_sink.set_val("fan_efficiency", 0.91)
    thrust_sink.set_val("propeller_efficiency", 0.83)
    thrust_sink.run_model()

    assert np.isclose(
        thrust_sink.get_val("thrust_sink_efficiency")[0],
        get_thrust_sink_efficiency(specs, "Turboprop"),
    )

    transmitter = om.Problem()
    transmitter.model.add_subsystem(
        "efficiency",
        TransmitterFanEfficiency(aircraft_class="Turbofan"),
        promotes=["*"],
    )
    transmitter.setup()
    transmitter.set_val("fan_efficiency", 0.91)
    transmitter.run_model()

    assert np.isclose(
        transmitter.get_val("transmitter_fan_efficiency")[0],
        transmitter_fan_efficiency(specs, "Turbofan"),
    )


def test_power_supplement_check_matches_fast_python():
    """Check supplemental transmitter power parity with FAST-Python."""

    architecture, transmitter_type, required_power, split, efficiency = (
        make_power_supplement_case()
    )
    problem = om.Problem()
    problem.model.add_subsystem(
        "supplement",
        PowerSupplementCheck(
            num_points=required_power.shape[0],
            architecture=architecture,
            transmitter_type=transmitter_type,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("required_power", required_power, units="W")
    problem.set_val("split", split)
    problem.set_val("efficiency", efficiency)
    problem.set_val("fan_efficiency", 0.88)
    problem.run_model()

    expected = power_supplement_check(
        required_power,
        architecture,
        split,
        efficiency,
        transmitter_type,
        0.88,
    )

    assert np.allclose(problem.get_val("supplemental_power", units="W"), expected)


def test_power_flow_matches_fast_python_upstream_and_downstream():
    """Check FAST power-flow iteration parity in both directions."""

    cases = make_power_flow_cases()

    for name, direction, initial_power, architecture, split, efficiency in cases:
        problem = om.Problem()
        problem.model.add_subsystem(
            name,
            PowerFlow(architecture=architecture, direction=direction),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("initial_power", initial_power, units="W")
        problem.set_val("split", split)
        problem.set_val("efficiency", efficiency)
        problem.run_model()

        expected = power_flow(
            initial_power,
            architecture,
            split,
            efficiency,
            direction,
        )

        assert np.allclose(problem.get_val("propagated_power", units="W"), expected)


def test_power_available_matches_fast_python_core_outputs():
    """Check available power propagation parity with FAST-Python."""

    aircraft = make_power_available_aircraft()
    prop_arch = aircraft["Specs"]["Propulsion"]["PropArch"]
    history = aircraft["Mission"]["History"]["SI"]
    split = np.asarray([prop_arch["OperUps"]() for _ in range(2)])
    problem = om.Problem()
    problem.model.add_subsystem(
        "available",
        PowerAvailable(
            num_points=2,
            architecture=prop_arch["Arch"],
            efficiency=prop_arch["EtaUps"],
            transmitter_type=prop_arch["TrnType"],
            num_sources=len(prop_arch["SrcType"]),
            aircraft_class=aircraft["Specs"]["TLAR"]["Class"],
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("true_airspeed", history["Performance"]["TAS"], units="m/s")
    problem.set_val("density", history["Performance"]["Rho"], units="kg/m**3")
    problem.set_val(
        "sea_level_static_thrust",
        aircraft["Specs"]["Propulsion"]["SLSThrust"],
        units="N",
    )
    problem.set_val(
        "sea_level_static_power",
        aircraft["Specs"]["Propulsion"]["SLSPower"],
        units="W",
    )
    problem.set_val("split", split)
    problem.run_model()

    expected = power_available(aircraft)["Mission"]["History"]["SI"]["Power"]
    assert np.allclose(problem.get_val("available_power", units="W"), expected["Pav"])
    assert np.allclose(problem.get_val("available_thrust", units="N"), expected["Tav"])
    assert np.allclose(problem.get_val("thrust_velocity", units="W"), expected["TV"])


def test_engine_thrust_requirement_matches_fast_python():
    """Check connected thrust sink selector parity with FAST-Python."""

    architecture, transmitter_type, thrust_output = make_engine_thrust_case()
    num_sources = 1
    component = 1
    problem = om.Problem()
    problem.model.add_subsystem(
        "thrust",
        EngineThrustRequirement(
            architecture=architecture,
            transmitter_type=transmitter_type,
            num_sources=num_sources,
            component=component,
            num_points=thrust_output.shape[0],
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("thrust_output", thrust_output, units="N")
    problem.run_model()

    expected = [
        engine_thrust_requirement(
            architecture,
            transmitter_type,
            num_sources,
            thrust_output,
            component,
            point,
        )
        for point in range(thrust_output.shape[0])
    ]

    assert np.allclose(problem.get_val("required_thrust", units="N"), expected)


def test_turboprop_engine_weight_for_sizing_matches_fast_python():
    """Check turboprop/piston engine sizing weight parity."""

    aircraft, engines, downstream_power = make_turboprop_engine_weight_case()
    power_sls = [100.0, 200.0, 300.0]
    dry_weight = [10.0, 20.0, 30.0]
    problem = om.Problem()
    problem.model.add_subsystem(
        "engine_weight",
        TurbopropEngineWeightForSizing(
            engines=engines,
            power_sls=power_sls,
            dry_weight=dry_weight,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("downstream_power", downstream_power, units="W")
    problem.run_model()

    expected = engine_weights_for_sizing(
        aircraft,
        "Turboprop",
        engines,
        downstream_power,
        np.zeros_like(downstream_power),
        np.zeros_like(downstream_power),
        np.zeros_like(downstream_power),
    )

    assert np.allclose(problem.get_val("engine_weight", units="kg"), expected)


def test_cable_weight_for_sizing_matches_fast_python():
    """Check cable sizing weight parity with FAST-Python."""

    aircraft, cables, downstream_power = make_cable_weight_case()
    cable_connections = aircraft["Specs"]["Propulsion"]["PropArch"]["CableConns"]
    cable_lengths = aircraft["Specs"]["Propulsion"]["PropArch"]["CableLengths"]
    cable_power_to_weight = aircraft["Specs"]["Power"]["P_W"]["Cables"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "cables",
        CableWeightForSizing(
            cables=cables,
            cable_connections=cable_connections,
            cable_lengths=cable_lengths,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("downstream_power", downstream_power, units="W")
    problem.set_val("cable_power_to_weight", cable_power_to_weight)
    problem.run_model()

    expected = cable_weight_for_sizing(aircraft, cables, downstream_power)

    assert np.isclose(problem.get_val("cable_weight", units="kg")[0], expected)


def test_propulsion_primitives_declare_analytic_partials():
    """Check propulsion primitive derivatives against finite difference."""

    architecture, transmitter_type, required_power, split, efficiency = (
        make_power_supplement_case()
    )
    flow_up, flow_down = make_power_flow_cases()
    thrust_architecture, thrust_type, thrust_output = make_engine_thrust_case()
    cable_aircraft, cables, downstream_power = make_cable_weight_case()
    _, engines, engine_downstream_power = make_turboprop_engine_weight_case()
    cable_architecture = cable_aircraft["Specs"]["Propulsion"]["PropArch"]
    available_aircraft = make_power_available_aircraft()
    available_arch = available_aircraft["Specs"]["Propulsion"]["PropArch"]
    available_history = available_aircraft["Mission"]["History"]["SI"]
    available_split = np.asarray([available_arch["OperUps"]() for _ in range(2)])
    available_split_check = available_split.copy()
    available_split_check[available_split_check == 0.0] = -1.0e-3
    cases = [
        (
            "engine_weight",
            TurbopropEngineWeightForSizing(
                engines=engines,
                power_sls=[100.0, 200.0, 300.0],
                dry_weight=[10.0, 20.0, 30.0],
            ),
            {"downstream_power": engine_downstream_power},
        ),
        (
            "cables",
            CableWeightForSizing(
                cables=cables,
                cable_connections=cable_architecture["CableConns"],
                cable_lengths=cable_architecture["CableLengths"],
            ),
            {
                "downstream_power": downstream_power,
                "cable_power_to_weight": cable_aircraft["Specs"]["Power"]["P_W"][
                    "Cables"
                ],
            },
        ),
        (
            "thrust",
            EngineThrustRequirement(
                architecture=thrust_architecture,
                transmitter_type=thrust_type,
                num_sources=1,
                component=1,
                num_points=thrust_output.shape[0],
            ),
            {"thrust_output": thrust_output},
        ),
        (
            "flow_up",
            PowerFlow(
                architecture=flow_up[3],
                direction=flow_up[1],
            ),
            {
                "initial_power": flow_up[2],
                "split": flow_up[4],
                "efficiency": flow_up[5],
            },
        ),
        (
            "flow_down",
            PowerFlow(
                architecture=flow_down[3],
                direction=flow_down[1],
            ),
            {
                "initial_power": flow_down[2],
                "split": flow_down[4],
                "efficiency": flow_down[5],
            },
        ),
        (
            "available",
            PowerAvailable(
                num_points=2,
                architecture=available_arch["Arch"],
                efficiency=available_arch["EtaUps"],
                transmitter_type=available_arch["TrnType"],
                num_sources=len(available_arch["SrcType"]),
                aircraft_class=available_aircraft["Specs"]["TLAR"]["Class"],
            ),
            {
                "true_airspeed": np.asarray(
                    available_history["Performance"]["TAS"],
                ),
                "density": np.asarray(available_history["Performance"]["Rho"]),
                "sea_level_static_thrust": np.asarray(
                    available_aircraft["Specs"]["Propulsion"]["SLSThrust"],
                ),
                "sea_level_static_power": np.asarray(
                    available_aircraft["Specs"]["Propulsion"]["SLSPower"],
                ),
                "split": available_split_check,
            },
        ),
        (
            "lapse",
            EngineLapse(aircraft_class="Turbofan"),
            {"sea_level_static": 100000.0, "density": 0.9},
        ),
        (
            "weight",
            SafeComponentWeight(),
            {"power": 500000.0, "power_to_weight": 2500.0},
        ),
        (
            "sink",
            ThrustSinkEfficiency(aircraft_class="Turbofan"),
            {"fan_efficiency": 0.91, "propeller_efficiency": 0.83},
        ),
        (
            "transmitter",
            TransmitterFanEfficiency(aircraft_class="Turbofan"),
            {"fan_efficiency": 0.91},
        ),
        (
            "supplement",
            PowerSupplementCheck(
                num_points=required_power.shape[0],
                architecture=architecture,
                transmitter_type=transmitter_type,
            ),
            {
                "required_power": required_power,
                "split": split,
                "efficiency": efficiency,
                "fan_efficiency": 0.88,
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
            tolerance = 1.0e-3 if name == "available" else 1.0e-6
            assert partial_data["abs error"].forward < tolerance


def make_efficiency_specs():
    """Return minimal propulsion specs for FAST-Python efficiency helpers."""

    return {
        "Propulsion": {
            "Engine": {
                "EtaPoly": {
                    "Fan": 0.91,
                },
            },
        },
        "Power": {
            "Eta": {
                "Propeller": 0.83,
            },
        },
    }


def make_power_supplement_case():
    """Return a fixed hybrid transmitter topology for supplemental power checks."""

    architecture = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    transmitter_type = np.asarray([1.0, 0.0, 0.0])
    required_power = np.asarray(
        [
            [120.0, 25.0, 10.0],
            [135.0, 30.0, 12.0],
        ]
    )
    split = np.asarray(
        [
            [1.0, 0.4, 0.2],
            [0.1, 1.0, 0.5],
            [0.3, 0.6, 1.0],
        ]
    )
    efficiency = np.asarray(
        [
            [0.92, 0.90, 0.89],
            [0.88, 0.93, 0.91],
            [0.87, 0.86, 0.94],
        ]
    )
    return architecture, transmitter_type, required_power, split, efficiency


def make_power_flow_cases():
    """Return small acyclic FAST power-flow cases for both directions."""

    architecture = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    upstream_split = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    upstream_efficiency = np.asarray(
        [
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 0.25],
            [1.0, 1.0, 1.0],
        ]
    )
    downstream_split = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    downstream_efficiency = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [0.5, 1.0, 1.0],
            [1.0, 0.25, 1.0],
        ]
    )
    return [
        (
            "flow_up",
            1,
            np.asarray([100.0, 0.0, 0.0]),
            architecture,
            upstream_split,
            upstream_efficiency,
        ),
        (
            "flow_down",
            -1,
            np.asarray([0.0, 0.0, 100.0]),
            architecture.T,
            downstream_split,
            downstream_efficiency,
        ),
    ]


def make_engine_thrust_case():
    """Return a fixed architecture with one engine-to-sink thrust path."""

    architecture = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    transmitter_type = np.asarray([1.0, 2.0, 2.0])
    thrust_output = np.asarray(
        [
            [0.0, 100.0, 200.0, 0.0],
            [0.0, 110.0, 210.0, 0.0],
        ]
    )
    return architecture, transmitter_type, thrust_output


def make_turboprop_engine_weight_case():
    """Return FAST-Python aircraft data for turboprop engine-weight sizing."""

    aircraft = {
        "Specs": {
            "Propulsion": {},
        },
        "HistData": {
            "Eng": {
                "E1": {
                    "Power_SLS": 100.0,
                    "DryWeight": 10.0,
                },
                "E2": {
                    "Power_SLS": 200.0,
                    "DryWeight": 20.0,
                },
                "E3": {
                    "Power_SLS": 300.0,
                    "DryWeight": 30.0,
                },
            }
        },
    }
    engines = np.asarray([True, True, False])
    downstream_power = np.asarray([100000.0, 200000.0, 0.0, 0.0])
    return aircraft, engines, downstream_power


def make_cable_weight_case():
    """Return fixed cable sizing data with two cable transmitters."""

    aircraft = {
        "Specs": {
            "Power": {
                "P_W": {
                    "Cables": 4.0,
                },
            },
            "Propulsion": {
                "PropArch": {
                    "CableConns": [
                        [1.0, 0.5],
                        [0.0, 1.0],
                    ],
                    "CableLengths": [
                        [12.0, 8.0],
                        [9.0, 10.0],
                    ],
                },
            },
        },
    }
    cables = np.asarray([False, True, False, True])
    downstream_power = np.asarray([0.0, 400000.0, 0.0, 250000.0, 0.0])
    return aircraft, cables, downstream_power


def make_power_available_aircraft():
    """Return a fixed conventional turbofan power-available aircraft."""

    return {
        "Specs": {
            "TLAR": {
                "Class": "Turbofan",
            },
            "Propulsion": {
                "SLSThrust": [120000.0, 110000.0, 0.0, 0.0],
                "SLSPower": [1.0e8, 1.0e8, 0.0, 0.0],
                "PropArch": {
                    "Arch": [
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    "EtaUps": [
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 0.9, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 0.88, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                    "OperUps": lambda: [
                        [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    "SrcType": [1.0],
                    "TrnType": [1.0, 1.0, 2.0, 2.0],
                },
            },
        },
        "Mission": {
            "Profile": {
                "SegsID": 1,
                "SegBeg": [1],
                "SegEnd": [2],
            },
            "History": {
                "SI": {
                    "Performance": {
                        "TAS": [210.0, 230.0],
                        "Rho": [1.1, 0.95],
                    },
                    "Power": {
                        "LamUps": [[0.0], [0.0]],
                    },
                },
            },
        },
    }
