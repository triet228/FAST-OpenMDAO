# tests/test_propulsion_components.py

"""Tests for OpenMDAO FAST propulsion primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    BatteryEnergyCutoff,
    BatteryEnergyHistory,
    CableWeightForSizing,
    EngineLapse,
    EngineThrustRequirement,
    FuelUseHistory,
    ParallelHybridArchitecture,
    PartialTurboelectricArchitecture,
    PowerAvailable,
    PowerFlow,
    PowerSupplementCheck,
    SafeComponentWeight,
    SeriesHybridArchitecture,
    SimpleSourceTransmitterArchitecture,
    ThrustSinkEfficiency,
    TransmitterFanEfficiency,
    TurboelectricArchitecture,
    TurbofanEngineWeightForSizing,
    TurbopropEngineWeightForSizing,
)
from fast_python.propulsion import (
    cable_weight_for_sizing,
    create_prop_arch,
    engine_lapse,
    engine_thrust_requirement,
    engine_weights_for_sizing,
    estimate_fuel_use,
    get_thrust_sink_efficiency,
    parallel_hybrid_architecture,
    partial_turboelectric_architecture,
    power_available,
    power_flow,
    power_supplement_check,
    safe_component_weight,
    series_hybrid_architecture,
    transmitter_fan_efficiency,
    turboelectric_architecture,
    update_battery_energy,
)
from fast_python.regression import prior_calculation, reg_processing


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


def test_simple_source_transmitter_architecture_matches_fast_python():
    """Check C/E architecture matrix builders against FAST-Python."""

    for architecture_type in ("C", "E"):
        values = make_simple_source_architecture_values(architecture_type)
        problem = om.Problem()
        problem.model.add_subsystem(
            "architecture",
            SimpleSourceTransmitterArchitecture(
                num_engines=values["num_engines"],
                architecture_type=architecture_type,
            ),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val(
            "electric_motor_efficiency",
            values["electric_motor_efficiency"],
        )
        problem.set_val("thrust_sink_efficiency", values["thrust_sink_efficiency"])
        problem.run_model()

        expected_aircraft = create_prop_arch(
            make_arch_aircraft(architecture_type, values["num_engines"])
        )
        expected = expected_aircraft["Specs"]["Propulsion"]["PropArch"]
        assert np.allclose(problem.get_val("architecture"), expected["Arch"])
        assert np.allclose(problem.get_val("upstream_split"), expected["OperUps"]())
        assert np.allclose(problem.get_val("downstream_split"), expected["OperDwn"]())
        assert np.allclose(problem.get_val("upstream_efficiency"), expected["EtaUps"])
        assert np.allclose(
            problem.get_val("downstream_efficiency"),
            expected["EtaDwn"],
        )
        assert np.allclose(problem.get_val("source_type"), expected["SrcType"])
        assert np.allclose(problem.get_val("transmitter_type"), expected["TrnType"])


def test_parallel_hybrid_architecture_matches_fast_python():
    """Check PHE architecture matrix builder parity with FAST-Python."""

    values = make_parallel_hybrid_architecture_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "architecture",
        ParallelHybridArchitecture(num_engines=values["num_engines"]),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("power_split", values["power_split"])
    problem.set_val("electric_motor_efficiency", values["electric_motor_efficiency"])
    problem.set_val("thrust_sink_efficiency", values["thrust_sink_efficiency"])
    problem.run_model()

    expected = parallel_hybrid_architecture(
        values["num_engines"],
        values["electric_motor_efficiency"],
        values["thrust_sink_efficiency"],
    )
    assert np.allclose(problem.get_val("architecture"), expected[0])
    assert np.allclose(
        problem.get_val("upstream_split"),
        expected[1](values["power_split"]),
    )
    assert np.allclose(
        problem.get_val("downstream_split"),
        expected[2](values["power_split"]),
    )
    assert np.allclose(problem.get_val("upstream_efficiency"), expected[3])
    assert np.allclose(problem.get_val("downstream_efficiency"), expected[4])
    assert np.allclose(problem.get_val("source_type"), expected[5])
    assert np.allclose(problem.get_val("transmitter_type"), expected[6])


def test_series_hybrid_architecture_matches_fast_python():
    """Check SHE architecture matrix builder parity with FAST-Python."""

    values = make_series_hybrid_architecture_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "architecture",
        SeriesHybridArchitecture(num_engines=values["num_engines"]),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("power_split", values["power_split"])
    problem.set_val("electric_motor_efficiency", values["electric_motor_efficiency"])
    problem.set_val(
        "electric_generator_efficiency",
        values["electric_generator_efficiency"],
    )
    problem.set_val("thrust_sink_efficiency", values["thrust_sink_efficiency"])
    problem.run_model()

    expected = series_hybrid_architecture(
        values["num_engines"],
        values["electric_motor_efficiency"],
        values["electric_generator_efficiency"],
        values["thrust_sink_efficiency"],
    )
    assert np.allclose(problem.get_val("architecture"), expected[0])
    assert np.allclose(
        problem.get_val("upstream_split"),
        expected[1](values["power_split"]),
    )
    assert np.allclose(
        problem.get_val("downstream_split"),
        expected[2](values["power_split"]),
    )
    assert np.allclose(problem.get_val("upstream_efficiency"), expected[3])
    assert np.allclose(problem.get_val("downstream_efficiency"), expected[4])
    assert np.allclose(problem.get_val("source_type"), expected[5])
    assert np.allclose(problem.get_val("transmitter_type"), expected[6])


def test_turboelectric_architecture_matches_fast_python():
    """Check TE architecture matrix builder parity with FAST-Python."""

    values = make_turboelectric_architecture_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "architecture",
        TurboelectricArchitecture(num_engines=values["num_engines"]),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("electric_motor_efficiency", values["electric_motor_efficiency"])
    problem.set_val(
        "electric_generator_efficiency",
        values["electric_generator_efficiency"],
    )
    problem.set_val("thrust_sink_efficiency", values["thrust_sink_efficiency"])
    problem.run_model()

    expected = turboelectric_architecture(
        values["num_engines"],
        values["electric_motor_efficiency"],
        values["electric_generator_efficiency"],
        values["thrust_sink_efficiency"],
    )
    assert np.allclose(problem.get_val("architecture"), expected[0])
    assert np.allclose(problem.get_val("upstream_split"), expected[1])
    assert np.allclose(problem.get_val("downstream_split"), expected[2])
    assert np.allclose(problem.get_val("upstream_efficiency"), expected[3])
    assert np.allclose(problem.get_val("downstream_efficiency"), expected[4])
    assert np.allclose(problem.get_val("source_type"), expected[5])
    assert np.allclose(problem.get_val("transmitter_type"), expected[6])


def test_partial_turboelectric_architecture_matches_fast_python():
    """Check PE architecture matrix builder parity with FAST-Python."""

    values = make_partial_turboelectric_architecture_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "architecture",
        PartialTurboelectricArchitecture(num_engines=values["num_engines"]),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("power_split", values["power_split"])
    problem.set_val("electric_motor_efficiency", values["electric_motor_efficiency"])
    problem.set_val(
        "electric_generator_efficiency",
        values["electric_generator_efficiency"],
    )
    problem.set_val("thrust_sink_efficiency", values["thrust_sink_efficiency"])
    problem.run_model()

    expected = partial_turboelectric_architecture(
        values["num_engines"],
        values["electric_motor_efficiency"],
        values["electric_generator_efficiency"],
        values["thrust_sink_efficiency"],
    )
    assert np.allclose(problem.get_val("architecture"), expected[0])
    assert np.allclose(
        problem.get_val("upstream_split"),
        expected[1](values["power_split"]),
    )
    assert np.allclose(
        problem.get_val("downstream_split"),
        expected[2](values["power_split"]),
    )
    assert np.allclose(problem.get_val("upstream_efficiency"), expected[3])
    assert np.allclose(problem.get_val("downstream_efficiency"), expected[4])
    assert np.allclose(problem.get_val("source_type"), expected[5])
    assert np.allclose(problem.get_val("transmitter_type"), expected[6])


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


def test_turbofan_engine_weight_for_sizing_matches_fast_python():
    """Check turbofan engine sizing weight GPR parity."""

    aircraft, engines, downstream_thrust = make_turbofan_engine_weight_case()
    io_space = [["Thrust_Max"], ["DryWeight"]]
    prior = prior_calculation(aircraft["HistData"]["Eng"], io_space)
    data_matrix, hyperparams, inverse_term = reg_processing(
        aircraft["HistData"]["Eng"],
        io_space,
        [prior],
        [1],
    )
    problem = om.Problem()
    problem.model.add_subsystem(
        "engine_weight",
        TurbofanEngineWeightForSizing(
            engines=engines,
            data_matrix=data_matrix,
            hyperparams=hyperparams,
            inverse_term=inverse_term,
            prior=prior,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("downstream_thrust", downstream_thrust, units="N")
    problem.run_model()

    expected = engine_weights_for_sizing(
        aircraft,
        "Turbofan",
        engines,
        np.zeros_like(downstream_thrust),
        downstream_thrust,
        np.zeros_like(downstream_thrust),
        np.zeros_like(downstream_thrust),
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


def test_battery_energy_history_matches_fast_python_update():
    """Check smooth battery energy accumulation parity with FAST-Python."""

    case = make_battery_energy_history_case()
    problem = om.Problem()
    problem.model.add_subsystem(
        "battery_energy",
        BatteryEnergyHistory(
            num_points=case["source_power"].shape[0],
            battery_sources=case["battery_sources"],
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("source_power", case["source_power"], units="W")
    problem.set_val("time_step", case["time_step"], units="s")
    problem.set_val("initial_source_energy", case["initial_source_energy"], units="J")
    problem.set_val(
        "initial_source_energy_left",
        case["initial_source_energy_left"],
        units="J",
    )
    problem.run_model()

    expected = evaluate_fast_python_battery_energy_case(case)

    assert np.allclose(problem.get_val("source_energy", units="J"), expected["energy"])
    assert np.allclose(
        problem.get_val("source_energy_left", units="J"),
        expected["energy_left"],
    )


def test_battery_energy_cutoff_matches_fast_python_update():
    """Check fixed-branch battery cutoff parity with FAST-Python."""

    case = make_battery_energy_cutoff_case()
    problem = om.Problem()
    problem.model.add_subsystem(
        "cutoff",
        BatteryEnergyCutoff(
            num_points=case["component_power"].shape[0],
            num_components=case["component_power"].shape[1],
            num_sources=case["initial_source_energy"].size,
            num_lam_down=case["lam_down"].shape[1],
            battery_source=case["battery_source"],
            engine_components=case["engine_components"],
            cutoff_index=case["cutoff_index"],
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("component_power", case["component_power"], units="W")
    problem.set_val("time_step", case["time_step"], units="s")
    problem.set_val("initial_source_energy", case["initial_source_energy"], units="J")
    problem.set_val(
        "initial_source_energy_left",
        case["initial_source_energy_left"],
        units="J",
    )
    problem.set_val("lam_down", case["lam_down"])
    problem.set_val("soc", case["soc"])
    problem.run_model()

    expected = evaluate_fast_python_battery_cutoff_case(case)
    assert np.allclose(
        problem.get_val("adjusted_component_power", units="W"),
        expected["component_power"],
    )
    assert np.allclose(problem.get_val("source_energy", units="J"), expected["energy"])
    assert np.allclose(
        problem.get_val("source_energy_left", units="J"),
        expected["energy_left"],
    )
    assert np.allclose(problem.get_val("adjusted_lam_down"), expected["lam_down"])
    assert np.allclose(problem.get_val("adjusted_soc"), expected["soc"])


def test_fuel_use_history_matches_fast_python_custom_model():
    """Check fuel-use accumulation parity with FAST-Python."""

    case = make_fuel_use_case()
    problem = om.Problem()
    problem.model.add_subsystem(
        "fuel",
        FuelUseHistory(
            num_steps=len(case["time_step"]),
            num_engines=case["fuel_mass_flow"].shape[1],
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("fuel_mass_flow", case["fuel_mass_flow"], units="kg/s")
    problem.set_val("specific_fuel_consumption", case["sfc"])
    problem.set_val("time_step", case["time_step"], units="s")
    problem.set_val("fuel_specific_energy", case["fuel_specific_energy"], units="J/kg")
    problem.set_val("initial_fuel_burn", case["initial_fuel_burn"], units="kg")
    problem.set_val("initial_mass", case["initial_mass"], units="kg")
    problem.set_val("initial_fuel_energy", case["initial_fuel_energy"], units="J")
    problem.set_val(
        "initial_fuel_energy_left",
        case["initial_fuel_energy_left"],
        units="J",
    )
    problem.set_val("mass_flow_correction", case["mass_flow_correction"])
    problem.run_model()

    expected = evaluate_fast_python_fuel_use_case(case)

    assert np.allclose(
        problem.get_val("corrected_fuel_mass_flow", units="kg/s"),
        expected["mdot_fuel"],
    )
    assert np.allclose(
        problem.get_val("corrected_specific_fuel_consumption"),
        expected["sfc"],
    )
    assert np.allclose(
        problem.get_val("source_mass_flow", units="kg/s"),
        expected["dmdt"][:, 0],
    )
    assert np.allclose(problem.get_val("fuel_burn", units="kg"), expected["fburn"])
    assert np.allclose(problem.get_val("mass", units="kg"), expected["mass"])
    assert np.allclose(
        problem.get_val("fuel_energy", units="J"),
        expected["fuel_energy"],
    )
    assert np.allclose(
        problem.get_val("fuel_energy_left", units="J"),
        expected["fuel_energy_left"],
    )


def test_propulsion_primitives_declare_analytic_partials():
    """Check propulsion primitive derivatives against finite difference."""

    architecture, transmitter_type, required_power, split, efficiency = (
        make_power_supplement_case()
    )
    flow_up, flow_down = make_power_flow_cases()
    thrust_architecture, thrust_type, thrust_output = make_engine_thrust_case()
    cable_aircraft, cables, downstream_power = make_cable_weight_case()
    _, engines, engine_downstream_power = make_turboprop_engine_weight_case()
    turbofan_aircraft, turbofan_engines, turbofan_downstream_thrust = (
        make_turbofan_engine_weight_case()
    )
    turbofan_io_space = [["Thrust_Max"], ["DryWeight"]]
    turbofan_prior = prior_calculation(
        turbofan_aircraft["HistData"]["Eng"],
        turbofan_io_space,
    )
    turbofan_data_matrix, turbofan_hyperparams, turbofan_inverse_term = (
        reg_processing(
            turbofan_aircraft["HistData"]["Eng"],
            turbofan_io_space,
            [turbofan_prior],
            [1],
        )
    )
    cable_architecture = cable_aircraft["Specs"]["Propulsion"]["PropArch"]
    available_aircraft = make_power_available_aircraft()
    available_arch = available_aircraft["Specs"]["Propulsion"]["PropArch"]
    available_history = available_aircraft["Mission"]["History"]["SI"]
    available_split = np.asarray([available_arch["OperUps"]() for _ in range(2)])
    available_split_check = available_split.copy()
    available_split_check[available_split_check == 0.0] = -1.0e-3
    fuel_case = make_fuel_use_case()
    fuel_derivative_case = dict(fuel_case)
    fuel_derivative_case["fuel_specific_energy"] = 1000.0
    fuel_derivative_case["initial_fuel_energy"] = 5.0
    fuel_derivative_case["initial_fuel_energy_left"] = 1000.0
    battery_energy_case = make_battery_energy_history_case()
    battery_energy_derivative_case = dict(battery_energy_case)
    battery_energy_derivative_case["initial_source_energy_left"] = np.asarray(
        [1000.0, 2000.0, 3000.0]
    )
    battery_cutoff_case = make_battery_energy_cutoff_case()
    simple_conventional_values = make_simple_source_architecture_values("C")
    simple_electric_values = make_simple_source_architecture_values("E")
    architecture_values = make_parallel_hybrid_architecture_values()
    series_architecture_values = make_series_hybrid_architecture_values()
    turboelectric_architecture_values = make_turboelectric_architecture_values()
    partial_turboelectric_values = make_partial_turboelectric_architecture_values()
    cases = [
        (
            "simple_conventional",
            SimpleSourceTransmitterArchitecture(
                num_engines=simple_conventional_values["num_engines"],
                architecture_type="C",
            ),
            {
                "electric_motor_efficiency": simple_conventional_values[
                    "electric_motor_efficiency"
                ],
                "thrust_sink_efficiency": simple_conventional_values[
                    "thrust_sink_efficiency"
                ],
            },
        ),
        (
            "simple_electric",
            SimpleSourceTransmitterArchitecture(
                num_engines=simple_electric_values["num_engines"],
                architecture_type="E",
            ),
            {
                "electric_motor_efficiency": simple_electric_values[
                    "electric_motor_efficiency"
                ],
                "thrust_sink_efficiency": simple_electric_values[
                    "thrust_sink_efficiency"
                ],
            },
        ),
        (
            "parallel_hybrid",
            ParallelHybridArchitecture(
                num_engines=architecture_values["num_engines"],
            ),
            {
                "power_split": architecture_values["power_split"],
                "electric_motor_efficiency": architecture_values[
                    "electric_motor_efficiency"
                ],
                "thrust_sink_efficiency": architecture_values[
                    "thrust_sink_efficiency"
                ],
            },
        ),
        (
            "series_hybrid",
            SeriesHybridArchitecture(
                num_engines=series_architecture_values["num_engines"],
            ),
            {
                "power_split": series_architecture_values["power_split"],
                "electric_motor_efficiency": series_architecture_values[
                    "electric_motor_efficiency"
                ],
                "electric_generator_efficiency": series_architecture_values[
                    "electric_generator_efficiency"
                ],
                "thrust_sink_efficiency": series_architecture_values[
                    "thrust_sink_efficiency"
                ],
            },
        ),
        (
            "turboelectric",
            TurboelectricArchitecture(
                num_engines=turboelectric_architecture_values["num_engines"],
            ),
            {
                "electric_motor_efficiency": turboelectric_architecture_values[
                    "electric_motor_efficiency"
                ],
                "electric_generator_efficiency": turboelectric_architecture_values[
                    "electric_generator_efficiency"
                ],
                "thrust_sink_efficiency": turboelectric_architecture_values[
                    "thrust_sink_efficiency"
                ],
            },
        ),
        (
            "partial_turboelectric",
            PartialTurboelectricArchitecture(
                num_engines=partial_turboelectric_values["num_engines"],
            ),
            {
                "power_split": partial_turboelectric_values["power_split"],
                "electric_motor_efficiency": partial_turboelectric_values[
                    "electric_motor_efficiency"
                ],
                "electric_generator_efficiency": partial_turboelectric_values[
                    "electric_generator_efficiency"
                ],
                "thrust_sink_efficiency": partial_turboelectric_values[
                    "thrust_sink_efficiency"
                ],
            },
        ),
        (
            "fuel",
            FuelUseHistory(
                num_steps=len(fuel_derivative_case["time_step"]),
                num_engines=fuel_derivative_case["fuel_mass_flow"].shape[1],
            ),
            {
                "fuel_mass_flow": fuel_derivative_case["fuel_mass_flow"],
                "specific_fuel_consumption": fuel_derivative_case["sfc"],
                "time_step": fuel_derivative_case["time_step"],
                "fuel_specific_energy": fuel_derivative_case[
                    "fuel_specific_energy"
                ],
                "initial_fuel_burn": fuel_derivative_case["initial_fuel_burn"],
                "initial_mass": fuel_derivative_case["initial_mass"],
                "initial_fuel_energy": fuel_derivative_case[
                    "initial_fuel_energy"
                ],
                "initial_fuel_energy_left": fuel_derivative_case[
                    "initial_fuel_energy_left"
                ],
                "mass_flow_correction": fuel_derivative_case[
                    "mass_flow_correction"
                ],
            },
        ),
        (
            "battery_energy",
            BatteryEnergyHistory(
                num_points=battery_energy_derivative_case["source_power"].shape[0],
                battery_sources=battery_energy_derivative_case["battery_sources"],
            ),
            {
                "source_power": battery_energy_derivative_case["source_power"],
                "time_step": battery_energy_derivative_case["time_step"],
                "initial_source_energy": battery_energy_derivative_case[
                    "initial_source_energy"
                ],
                "initial_source_energy_left": battery_energy_derivative_case[
                    "initial_source_energy_left"
                ],
            },
        ),
        (
            "battery_cutoff",
            BatteryEnergyCutoff(
                num_points=battery_cutoff_case["component_power"].shape[0],
                num_components=battery_cutoff_case["component_power"].shape[1],
                num_sources=battery_cutoff_case["initial_source_energy"].size,
                num_lam_down=battery_cutoff_case["lam_down"].shape[1],
                battery_source=battery_cutoff_case["battery_source"],
                engine_components=battery_cutoff_case["engine_components"],
                cutoff_index=battery_cutoff_case["cutoff_index"],
            ),
            {
                "component_power": battery_cutoff_case["component_power"],
                "time_step": battery_cutoff_case["time_step"],
                "initial_source_energy": battery_cutoff_case[
                    "initial_source_energy"
                ],
                "initial_source_energy_left": battery_cutoff_case[
                    "initial_source_energy_left"
                ],
                "lam_down": battery_cutoff_case["lam_down"],
                "soc": battery_cutoff_case["soc"],
            },
        ),
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
            "turbofan_engine_weight",
            TurbofanEngineWeightForSizing(
                engines=turbofan_engines,
                data_matrix=turbofan_data_matrix,
                hyperparams=turbofan_hyperparams,
                inverse_term=turbofan_inverse_term,
                prior=turbofan_prior,
            ),
            {"downstream_thrust": turbofan_downstream_thrust},
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


def make_simple_source_architecture_values(architecture_type):
    """Return scalar inputs for FAST C/E architecture construction."""

    values = {
        "num_engines": 2,
        "electric_motor_efficiency": 0.96,
        "thrust_sink_efficiency": 0.8,
    }

    if architecture_type == "C":
        values["electric_motor_efficiency"] = 0.91

    return values


def make_parallel_hybrid_architecture_values():
    """Return scalar inputs for FAST parallel-hybrid architecture construction."""

    return {
        "num_engines": 2,
        "power_split": 0.37,
        "electric_motor_efficiency": 0.94,
        "thrust_sink_efficiency": 0.86,
    }


def make_series_hybrid_architecture_values():
    """Return scalar inputs for FAST series-hybrid architecture construction."""

    return {
        "num_engines": 2,
        "power_split": 0.42,
        "electric_motor_efficiency": 0.93,
        "electric_generator_efficiency": 0.91,
        "thrust_sink_efficiency": 0.84,
    }


def make_turboelectric_architecture_values():
    """Return scalar inputs for FAST turboelectric architecture construction."""

    return {
        "num_engines": 2,
        "electric_motor_efficiency": 0.92,
        "electric_generator_efficiency": 0.9,
        "thrust_sink_efficiency": 0.83,
    }


def make_partial_turboelectric_architecture_values():
    """Return scalar inputs for FAST partial-turboelectric construction."""

    return {
        "num_engines": 2,
        "power_split": 0.28,
        "electric_motor_efficiency": 0.925,
        "electric_generator_efficiency": 0.905,
        "thrust_sink_efficiency": 0.835,
    }


def make_arch_aircraft(architecture_type, num_engines):
    """Return a minimal aircraft dictionary for FAST-Python architecture setup."""

    return {
        "Settings": {},
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
            },
            "Power": {
                "Eta": {
                    "EM": 0.96,
                    "EG": 0.96,
                    "Propeller": 0.8,
                },
            },
            "Propulsion": {
                "NumEngines": num_engines,
                "Engine": {
                    "EtaPoly": {
                        "Fan": 0.99,
                    },
                },
                "PropArch": {
                    "Type": architecture_type,
                },
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


def make_turbofan_engine_weight_case():
    """Return FAST-Python aircraft data for turbofan engine-weight sizing."""

    aircraft = {
        "Specs": {
            "Propulsion": {
                "Engine": {
                    "Mach": 0.05,
                    "Alt": 0.0,
                    "OPR": 30.0,
                    "BPR": 5.0,
                    "FPR": 1.5,
                    "Tt4Max": 1600.0,
                    "DesignThrust": 120000.0,
                    "NoSpools": 2,
                    "RPMs": [7400.0, 17820.0],
                    "FanGearRatio": np.nan,
                    "FanBoosters": False,
                    "MaxIter": 300,
                    "CoreFlow": {
                        "PaxBleed": 0.03,
                        "Leakage": 0.01,
                        "Cooling": 0.0,
                    },
                    "EtaPoly": {
                        "Inlet": 0.99,
                        "Diffusers": 0.99,
                        "Fan": 0.92,
                        "Compressors": 0.9,
                        "BypassNozzle": 0.98,
                        "Combustor": 0.99,
                        "Turbines": 0.9,
                        "CoreNozzle": 0.98,
                        "Nozzles": 0.99,
                        "Mixing": 0.0,
                    },
                },
            },
        },
        "HistData": {
            "Eng": {
                "E1": {
                    "Thrust_Max": 80000.0,
                    "DryWeight": 1000.0,
                },
                "E2": {
                    "Thrust_Max": 120000.0,
                    "DryWeight": 1500.0,
                },
                "E3": {
                    "Thrust_Max": 160000.0,
                    "DryWeight": 2200.0,
                },
            }
        },
    }
    engines = np.asarray([True, False, False])
    downstream_thrust = np.asarray([120000.0, 0.0, 0.0, 0.0])
    return aircraft, engines, downstream_thrust


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


def make_battery_energy_history_case():
    """Return smooth battery source-power history for energy checks."""

    return {
        "battery_sources": np.asarray([False, True, True]),
        "engine_transmitters": np.asarray([True, False]),
        "source_power": np.asarray(
            [
                [0.0, 900.0, 600.0],
                [0.0, 800.0, 550.0],
                [0.0, 700.0, 500.0],
                [0.0, 650.0, 450.0],
            ]
        ),
        "time_step": np.asarray([20.0, 30.0, 25.0]),
        "initial_source_energy": np.asarray([1000.0, 2000.0, 3000.0]),
        "initial_source_energy_left": np.asarray([1.0e9, 8.0e8, 7.0e8]),
    }


def evaluate_fast_python_battery_energy_case(case):
    """Run FAST-Python update_battery_energy on the smooth active branch."""

    num_points, num_sources = case["source_power"].shape
    num_engines = case["engine_transmitters"].size
    num_components = num_sources + num_engines
    source_power = case["source_power"]
    pout = np.zeros((num_points, num_components))
    pout[:, :num_sources] = source_power
    source_energy = np.tile(case["initial_source_energy"], (num_points, 1))
    source_energy_left = np.tile(
        case["initial_source_energy_left"],
        (num_points, 1),
    )
    soc = np.ones((num_points, num_sources)) * 100.0
    voltage = np.zeros((num_points, num_sources))
    current = np.zeros((num_points, num_sources))
    capacity = np.zeros((num_points, num_sources))
    c_rate = np.zeros((num_points, num_sources))
    lam_dwn = np.ones((num_points, num_components, num_components))
    mass = np.ones(num_points) * 1000.0
    aircraft = {
        "Settings": {
            "Analysis": {
                "Type": 1,
            },
        },
        "Specs": {
            "Power": {
                "Battery": {
                    "SerCells": float("nan"),
                    "ParCells": float("nan"),
                },
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "PHE",
                },
            },
        },
        "Mission": {
            "Profile": {
                "MissID": 1,
            },
            "History": {
                "Flags": {
                    "SOCOff": [0],
                },
            },
        },
    }

    update_battery_energy(
        aircraft,
        pout,
        case["time_step"],
        case["battery_sources"],
        case["engine_transmitters"],
        num_sources,
        source_energy,
        source_energy_left,
        soc,
        voltage,
        current,
        capacity,
        c_rate,
        lam_dwn,
        mass,
        0,
    )

    return {
        "energy": source_energy,
        "energy_left": source_energy_left,
    }


def make_battery_energy_cutoff_case():
    """Return a non-detailed battery case that triggers FAST cutoff handling."""

    return {
        "battery_source": 0,
        "battery_sources": np.asarray([True]),
        "engine_transmitters": np.asarray([True, True]),
        "engine_components": np.asarray([1, 2]),
        "cutoff_index": 1,
        "component_power": np.asarray(
            [
                [200.0, 20.0, 30.0],
                [300.0, 25.0, 35.0],
                [400.0, 30.0, 40.0],
                [500.0, 35.0, 45.0],
            ]
        ),
        "time_step": np.asarray([1.0, 1.0, 1.0]),
        "initial_source_energy": np.asarray([0.0]),
        "initial_source_energy_left": np.asarray([250.0]),
        "lam_down": np.asarray(
            [
                [0.3, 0.7],
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
            ]
        ),
        "soc": np.asarray(
            [
                [80.0],
                [75.0],
                [70.0],
                [65.0],
            ]
        ),
    }


def evaluate_fast_python_battery_cutoff_case(case):
    """Run FAST-Python update_battery_energy on the cutoff active branch."""

    component_power = np.array(case["component_power"], dtype=float, copy=True)
    num_points, num_components = component_power.shape
    num_sources = case["initial_source_energy"].size
    source_energy = np.tile(case["initial_source_energy"], (num_points, 1))
    source_energy_left = np.tile(
        case["initial_source_energy_left"],
        (num_points, 1),
    )
    soc = np.array(case["soc"], dtype=float, copy=True)
    voltage = np.zeros((num_points, num_sources))
    current = np.zeros((num_points, num_sources))
    capacity = np.zeros((num_points, num_sources))
    c_rate = np.zeros((num_points, num_sources))
    lam_down = np.array(case["lam_down"], dtype=float, copy=True)
    mass = np.ones(num_points) * 1000.0
    aircraft = {
        "Settings": {
            "Analysis": {
                "Type": -1,
            },
        },
        "Specs": {
            "Power": {
                "Battery": {
                    "SerCells": float("nan"),
                    "ParCells": float("nan"),
                },
            },
            "Propulsion": {
                "PropArch": {
                    "Type": "PHE",
                },
            },
        },
        "Mission": {
            "Profile": {
                "MissID": 1,
            },
            "History": {
                "Flags": {
                    "SOCOff": [0],
                },
            },
        },
    }

    update_battery_energy(
        aircraft,
        component_power,
        case["time_step"],
        case["battery_sources"],
        case["engine_transmitters"],
        num_sources,
        source_energy,
        source_energy_left,
        soc,
        voltage,
        current,
        capacity,
        c_rate,
        lam_down,
        mass,
        0,
    )

    return {
        "component_power": component_power,
        "energy": source_energy,
        "energy_left": source_energy_left,
        "lam_down": lam_down,
        "soc": soc,
    }


def make_fuel_use_case():
    """Return deterministic engine fuel-flow histories for fuel-use checks."""

    return {
        "fuel_mass_flow": np.asarray(
            [
                [0.12, 0.08],
                [0.10, 0.07],
                [0.09, 0.05],
            ]
        ),
        "sfc": np.asarray(
            [
                [1.4, 1.2],
                [1.3, 1.1],
                [1.25, 1.05],
            ]
        ),
        "time_step": np.asarray([60.0, 55.0, 50.0]),
        "fuel_specific_energy": 43.0e6,
        "initial_fuel_burn": 4.0,
        "initial_mass": 1000.0,
        "initial_fuel_energy": 10.0,
        "initial_fuel_energy_left": 1.0e8,
        "mass_flow_correction": 1.07,
    }


def evaluate_fast_python_fuel_use_case(case):
    """Run FAST-Python fuel-use mutation path for one deterministic case."""

    fuel_flow = case["fuel_mass_flow"]
    sfc_input = case["sfc"]
    time_step = case["time_step"]
    num_steps, num_engines = fuel_flow.shape
    num_sources = 1
    num_transmitters = num_engines
    num_components = num_sources + num_transmitters

    def fuel_model(aircraft, ipnt, component, pout_value, psupp_value):
        engine = component - num_sources
        return fuel_flow[ipnt, engine], sfc_input[ipnt, engine]

    aircraft = {
        "Specs": {
            "TLAR": {
                "Class": "Turbofan",
            },
            "Propulsion": {
                "FuelFlowModel": fuel_model,
                "MDotCF": case["mass_flow_correction"],
                "PropArch": {
                    "Arch": np.eye(num_components),
                    "TrnType": np.ones(num_transmitters),
                },
            },
        },
    }
    pout = np.zeros((num_steps, num_components))
    psupp = np.zeros((num_steps, num_components))
    tout = np.zeros((num_steps, num_components))
    fburn = np.zeros(num_steps + 1)
    mass = np.zeros(num_steps + 1)
    fuel_energy = np.zeros((num_steps + 1, num_sources))
    fuel_energy_left = np.zeros((num_steps + 1, num_sources))
    sfc = np.zeros((num_steps, num_engines))
    mdot_fuel = np.zeros((num_steps, num_engines))
    dmdt = np.zeros((num_steps, num_sources))
    fburn[0] = case["initial_fuel_burn"]
    mass[0] = case["initial_mass"]
    fuel_energy[:, 0] = case["initial_fuel_energy"]
    fuel_energy_left[0, 0] = case["initial_fuel_energy_left"]

    estimate_fuel_use(
        aircraft,
        pout,
        psupp,
        time_step,
        np.asarray([True]),
        np.asarray([True] * num_engines),
        num_sources,
        num_transmitters,
        case["fuel_specific_energy"],
        np.zeros(num_steps),
        np.zeros(num_steps),
        tout,
        fburn,
        mass,
        fuel_energy,
        fuel_energy_left,
        sfc,
        mdot_fuel,
        dmdt,
    )

    return {
        "mdot_fuel": mdot_fuel,
        "sfc": sfc,
        "dmdt": dmdt,
        "fburn": fburn,
        "mass": mass,
        "fuel_energy": fuel_energy[:, 0],
        "fuel_energy_left": fuel_energy_left[:, 0],
    }


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
