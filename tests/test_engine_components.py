# tests/test_engine_components.py

"""Tests for OpenMDAO FAST engine primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    AirIntegratedHeat,
    AirSpecificHeat,
    AirSpecificHeatVolume,
    AirTemperatureFromHeatAdded,
    AirTemperatureFromHeatRemoved,
    BurnerFlow,
    ChokedArea,
    CompressorStageFlow,
    FlowArea,
    JetAIntegratedHeat,
    LocalEfficiency,
    LocalReynolds,
    MassFlowParameter,
    OffDesignNozzleMach,
    SimpleOffDesignTurbofan,
    StaticDensity,
    StaticPressure,
    StaticTemperature,
    ThermalPerfectGamma,
    TotalPressure,
    TotalTemperature,
    TurbineStageFlow,
)
from fast_python.engine import (
    a_astar,
    astar_a,
    burner,
    comp_stage,
    cp_air,
    cp_jeta,
    cv_air,
    local_efficiency,
    local_reynolds,
    mass_flow_parameter,
    new_gamma,
    newton_raphson_tt1,
    newton_raphson_tt3,
    off_design_nozzle,
    ps_pt,
    pt_ps,
    rhos_rhot,
    simple_off_design,
    ts_tt,
    tt_ts,
    turb_stage,
)


def test_engine_pressure_temperature_primitives_match_fast_python():
    """Check isentropic pressure and temperature primitive parity."""

    pressure = om.Problem()
    pressure.model.add_subsystem("total", TotalPressure(), promotes=["*"])
    pressure.setup()
    pressure.set_val("static_pressure", 90000.0, units="Pa")
    pressure.set_val("mach", 0.55)
    pressure.set_val("gamma", 1.37)
    pressure.run_model()

    assert np.isclose(
        pressure.get_val("total_pressure", units="Pa")[0],
        pt_ps(90000.0, 0.55, 1.37),
    )

    static_pressure = om.Problem()
    static_pressure.model.add_subsystem("static", StaticPressure(), promotes=["*"])
    static_pressure.setup()
    static_pressure.set_val("total_pressure", 110000.0, units="Pa")
    static_pressure.set_val("mach", 0.55)
    static_pressure.set_val("gamma", 1.37)
    static_pressure.run_model()

    assert np.isclose(
        static_pressure.get_val("static_pressure", units="Pa")[0],
        ps_pt(110000.0, 0.55, 1.37),
    )

    total_temperature = om.Problem()
    total_temperature.model.add_subsystem("total", TotalTemperature(), promotes=["*"])
    total_temperature.setup()
    total_temperature.set_val("static_temperature", 260.0, units="K")
    total_temperature.set_val("mach", 0.55)
    total_temperature.set_val("gamma", 1.37)
    total_temperature.run_model()

    assert np.isclose(
        total_temperature.get_val("total_temperature", units="K")[0],
        tt_ts(260.0, 0.55, 1.37),
    )

    static_temperature = om.Problem()
    static_temperature.model.add_subsystem("static", StaticTemperature(), promotes=["*"])
    static_temperature.setup()
    static_temperature.set_val("total_temperature", 285.0, units="K")
    static_temperature.set_val("mach", 0.55)
    static_temperature.set_val("gamma", 1.37)
    static_temperature.run_model()

    assert np.isclose(
        static_temperature.get_val("static_temperature", units="K")[0],
        ts_tt(285.0, 0.55, 1.37),
    )


def test_engine_area_massflow_density_primitives_match_fast_python():
    """Check area, mass-flow, and density primitive parity."""

    choked = om.Problem()
    choked.model.add_subsystem("choked", ChokedArea(), promotes=["*"])
    choked.setup()
    choked.set_val("area", 1.4, units="m**2")
    choked.set_val("mach", 0.65)
    choked.set_val("gamma", 1.4)
    choked.run_model()

    assert np.isclose(choked.get_val("area_star", units="m**2")[0], astar_a(1.4, 0.65, 1.4))

    area = om.Problem()
    area.model.add_subsystem("area", FlowArea(), promotes=["*"])
    area.setup()
    area.set_val("area_star", 0.9, units="m**2")
    area.set_val("mach", 0.65)
    area.set_val("gamma", 1.4)
    area.run_model()

    assert np.isclose(area.get_val("area", units="m**2")[0], a_astar(0.9, 0.65, 1.4))

    mass_flow = om.Problem()
    mass_flow.model.add_subsystem("mass_flow", MassFlowParameter(), promotes=["*"])
    mass_flow.setup()
    mass_flow.set_val("mach", 0.65)
    mass_flow.set_val("gamma", 1.4)
    mass_flow.run_model()

    assert np.isclose(
        mass_flow.get_val("mass_flow_parameter")[0],
        mass_flow_parameter(0.65, 1.4),
    )

    nozzle = om.Problem()
    nozzle.model.add_subsystem("nozzle", OffDesignNozzleMach(), promotes=["*"])
    nozzle.setup()
    nozzle.set_val("area_1", 1.4, units="m**2")
    nozzle.set_val("area_2", 1.8, units="m**2")
    nozzle.set_val("mach_1", 0.65)
    nozzle.set_val("gamma", 1.4)
    nozzle.run_model()

    assert np.isclose(
        nozzle.get_val("mach_2")[0],
        off_design_nozzle(1.4, 1.8, 0.65, 1.4),
    )

    density = om.Problem()
    density.model.add_subsystem("density", StaticDensity(), promotes=["*"])
    density.setup()
    density.set_val("total_density", 1.3, units="kg/m**3")
    density.set_val("mach", 0.65)
    density.set_val("gamma", 1.4)
    density.run_model()

    assert np.isclose(
        density.get_val("static_density", units="kg/m**3")[0],
        rhos_rhot(1.3, 0.65, 1.4),
    )


def test_engine_specific_heat_components_match_fast_python():
    """Check FAST fitted specific heat component parity."""

    cp_problem = om.Problem()
    cp_problem.model.add_subsystem("cp", AirSpecificHeat(), promotes=["*"])
    cp_problem.setup()
    cp_problem.set_val("temperature", 300.0, units="K")
    cp_problem.run_model()

    assert np.isclose(cp_problem.get_val("cp_air")[0], cp_air(300.0))

    cv_problem = om.Problem()
    cv_problem.model.add_subsystem("cv", AirSpecificHeatVolume(), promotes=["*"])
    cv_problem.setup()
    cv_problem.set_val("temperature", 300.0, units="K")
    cv_problem.run_model()

    assert np.isclose(cv_problem.get_val("cv_air")[0], cv_air(300.0))

    gamma_problem = om.Problem()
    gamma_problem.model.add_subsystem("gamma", ThermalPerfectGamma(), promotes=["*"])
    gamma_problem.setup()
    gamma_problem.set_val("total_temperature", 800.0, units="K")
    gamma_problem.set_val("mach", 0.55)
    gamma_problem.set_val("gamma", 1.37)
    gamma_problem.run_model()

    ts, cp, cv, gamma = new_gamma(800.0, 0.55, 1.37)

    assert np.isclose(gamma_problem.get_val("static_temperature", units="K")[0], ts)
    assert np.isclose(gamma_problem.get_val("cp_air")[0], cp)
    assert np.isclose(gamma_problem.get_val("cv_air")[0], cv)
    assert np.isclose(gamma_problem.get_val("updated_gamma")[0], gamma)

    air_heat = om.Problem()
    air_heat.model.add_subsystem("heat", AirIntegratedHeat(), promotes=["*"])
    air_heat.setup()
    air_heat.set_val("temperature_low", 300.0, units="K")
    air_heat.set_val("temperature_high", 1200.0, units="K")
    air_heat.run_model()

    assert np.isclose(air_heat.get_val("integrated_cp_air")[0], cp_air(300.0, 1200.0))

    jeta_heat = om.Problem()
    jeta_heat.model.add_subsystem("heat", JetAIntegratedHeat(), promotes=["*"])
    jeta_heat.setup()
    jeta_heat.set_val("temperature_low", 300.0, units="K")
    jeta_heat.set_val("temperature_high", 1200.0, units="K")
    jeta_heat.run_model()

    assert np.isclose(jeta_heat.get_val("integrated_cp_jeta")[0], cp_jeta(300.0, 1200.0))

    heat_added = om.Problem()
    heat_added.model.add_subsystem("inverse", AirTemperatureFromHeatAdded(), promotes=["*"])
    heat_added.setup()
    heat_added.set_val("temperature_start", 300.0, units="K")
    heat_added.set_val("heat", 100000.0)
    heat_added.run_model()

    assert np.isclose(
        heat_added.get_val("temperature_end", units="K")[0],
        newton_raphson_tt1(300.0, 100000.0),
    )

    heat_removed = om.Problem()
    heat_removed.model.add_subsystem("inverse", AirTemperatureFromHeatRemoved(), promotes=["*"])
    heat_removed.setup()
    heat_removed.set_val("temperature_start", 1200.0, units="K")
    heat_removed.set_val("heat", 100000.0)
    heat_removed.run_model()

    assert np.isclose(
        heat_removed.get_val("temperature_end", units="K")[0],
        newton_raphson_tt3(1200.0, 100000.0),
    )


def test_engine_local_efficiency_and_reynolds_match_fast_python():
    """Check local efficiency and Reynolds primitive parity."""

    efficiency = om.Problem()
    efficiency.model.add_subsystem("efficiency", LocalEfficiency(), promotes=["*"])
    efficiency.setup()
    efficiency.set_val("reynolds", 2.5e7)
    efficiency.run_model()

    assert np.isclose(
        efficiency.get_val("local_efficiency")[0],
        local_efficiency(2.5e7),
    )

    flow_state = {
        "Ps": 85000.0,
        "Ts": 260.0,
        "Ro": 0.9,
        "Ri": 0.35,
        "Mach": 0.45,
        "Gam": 1.36,
    }
    reynolds = om.Problem()
    reynolds.model.add_subsystem("reynolds", LocalReynolds(), promotes=["*"])
    reynolds.setup()
    reynolds.set_val("static_pressure", flow_state["Ps"], units="Pa")
    reynolds.set_val("static_temperature", flow_state["Ts"], units="K")
    reynolds.set_val("outer_radius", flow_state["Ro"], units="m")
    reynolds.set_val("inner_radius", flow_state["Ri"], units="m")
    reynolds.set_val("mach", flow_state["Mach"])
    reynolds.set_val("gamma", flow_state["Gam"])
    reynolds.run_model()

    assert np.isclose(
        reynolds.get_val("local_reynolds")[0],
        local_reynolds(flow_state),
    )


def test_burner_flow_matches_fast_python():
    """Check FAST on-design burner parity for scalar flow-state outputs."""

    state31 = make_burner_state()
    eta_poly = {"Combustor": 0.99}
    expected_state, expected_fuel, expected_ratio = burner(
        state31,
        1400.0,
        43.0e6,
        eta_poly,
    )
    problem = om.Problem()
    problem.model.add_subsystem("burner", BurnerFlow(), promotes=["*"])
    problem.setup()
    set_burner_values(problem, state31, 1400.0, 43.0e6, eta_poly["Combustor"])
    problem.run_model()

    assert np.isclose(problem.get_val("diffuser_mach_32")[0], expected_state["Mach"])
    assert np.isclose(problem.get_val("diffuser_area_32", units="m**2")[0], expected_state["Area"])
    assert np.isclose(problem.get_val("fuel_flow", units="kg/s")[0], expected_fuel)
    assert np.isclose(problem.get_val("fuel_air_ratio")[0], expected_ratio)
    assert np.isclose(problem.get_val("mass_flow_39", units="kg/s")[0], expected_state["MDot"])
    assert np.isclose(problem.get_val("total_pressure_39", units="Pa")[0], expected_state["Pt"])
    assert np.isclose(problem.get_val("total_temperature_39", units="K")[0], expected_state["Tt"])
    assert np.isclose(problem.get_val("static_temperature_39", units="K")[0], expected_state["Ts"])
    assert np.isclose(problem.get_val("cp_air_39")[0], expected_state["Cp"])
    assert np.isclose(problem.get_val("cv_air_39")[0], expected_state["Cv"])
    assert np.isclose(problem.get_val("gamma_39")[0], expected_state["Gam"])
    assert np.isclose(problem.get_val("static_pressure_39", units="Pa")[0], expected_state["Ps"])
    assert np.isclose(problem.get_val("inner_radius_39", units="m")[0], expected_state["Ri"])


def test_compressor_stage_flow_matches_fast_python():
    """Check FAST one-stage compressor flow parity."""

    state1 = make_compressor_stage_state()
    eta_poly = {"Compressors": 0.9, "Fan": 0.88}
    expected_state, expected_work, expected_tau = comp_stage(
        state1,
        eta_poly,
        1.18,
        6200.0,
    )
    problem = om.Problem()
    problem.model.add_subsystem("stage", CompressorStageFlow(), promotes=["*"])
    problem.setup()
    set_compressor_stage_values(problem, state1, 1.18, 6200.0, eta_poly["Compressors"])
    problem.run_model()

    assert np.isclose(problem.get_val("mass_flow_3", units="kg/s")[0], expected_state["MDot"])
    assert np.isclose(problem.get_val("total_pressure_3", units="Pa")[0], expected_state["Pt"])
    assert np.isclose(problem.get_val("total_temperature_3", units="K")[0], expected_state["Tt"])
    assert np.isclose(problem.get_val("static_temperature_3", units="K")[0], expected_state["Ts"])
    assert np.isclose(problem.get_val("mach_3")[0], expected_state["Mach"])
    assert np.isclose(problem.get_val("cp_air_3")[0], expected_state["Cp"])
    assert np.isclose(problem.get_val("cv_air_3")[0], expected_state["Cv"])
    assert np.isclose(problem.get_val("gamma_3")[0], expected_state["Gam"])
    assert np.isclose(problem.get_val("static_pressure_3", units="Pa")[0], expected_state["Ps"])
    assert np.isclose(problem.get_val("area_3", units="m**2")[0], expected_state["Area"])
    assert np.isclose(problem.get_val("outer_radius_3", units="m")[0], expected_state["Ro"])
    assert np.isclose(problem.get_val("inner_radius_3", units="m")[0], expected_state["Ri"])
    assert np.isclose(problem.get_val("pitch_radius_3", units="m")[0], expected_state["Rp"])
    assert np.isclose(problem.get_val("work", units="W")[0], expected_work)
    assert np.isclose(problem.get_val("temperature_ratio")[0], expected_tau)
    assert np.isclose(problem.get_val("stage_eta")[0], expected_state["Eta"])
    assert np.isclose(problem.get_val("stage_psi")[0], expected_state["Psi"])
    assert np.isclose(problem.get_val("corrected_mass_flow")[0], expected_state["MNorm"])
    assert np.isclose(problem.get_val("corrected_speed")[0], expected_state["NNorm"])
    assert np.isclose(problem.get_val("stage_phi")[0], expected_state["Phi"])
    assert np.isclose(problem.get_val("stage_zeta")[0], expected_state["Zeta"])


def test_turbine_stage_flow_matches_fast_python():
    """Check FAST one-stage turbine flow parity."""

    state1 = make_turbine_stage_state()
    eta_poly = {"Turbines": 0.91}
    expected_state, expected_pi, expected_tau = turb_stage(
        state1,
        1280.0,
        6200.0,
        True,
        eta_poly,
    )
    problem = om.Problem()
    problem.model.add_subsystem("stage", TurbineStageFlow(), promotes=["*"])
    problem.setup()
    set_turbine_stage_values(problem, state1, 1280.0, 1.1, 6200.0, eta_poly["Turbines"])
    problem.run_model()

    assert np.isclose(problem.get_val("total_temperature_3", units="K")[0], expected_state["Tt"])
    assert np.isclose(problem.get_val("total_pressure_3", units="Pa")[0], expected_state["Pt"])
    assert np.isclose(problem.get_val("mach_3")[0], expected_state["Mach"])
    assert np.isclose(problem.get_val("static_temperature_3", units="K")[0], expected_state["Ts"])
    assert np.isclose(problem.get_val("cp_air_3")[0], expected_state["Cp"])
    assert np.isclose(problem.get_val("cv_air_3")[0], expected_state["Cv"])
    assert np.isclose(problem.get_val("gamma_3")[0], expected_state["Gam"])
    assert np.isclose(problem.get_val("static_pressure_3", units="Pa")[0], expected_state["Ps"])
    assert np.isclose(problem.get_val("area_3", units="m**2")[0], expected_state["Area"])
    assert np.isclose(problem.get_val("inner_radius_3", units="m")[0], expected_state["Ri"])
    assert np.isclose(problem.get_val("outer_radius_3", units="m")[0], expected_state["Ro"])
    assert np.isclose(problem.get_val("pressure_ratio")[0], expected_pi)
    assert np.isclose(problem.get_val("temperature_ratio")[0], expected_tau)


def test_simple_off_design_turbofan_matches_fast_python():
    """Check BADA-style simple off-design turbofan parity."""

    aircraft = make_simple_off_design_aircraft()
    off_params = {
        "FlightCon": {
            "Alt": 1000.0,
            "Mach": 0.2,
        },
        "Thrust": 10000.0,
    }
    expected = simple_off_design(aircraft, off_params, 1000.0, 1, 0)
    problem = om.Problem()
    problem.model.add_subsystem("off_design", SimpleOffDesignTurbofan(), promotes=["*"])
    problem.setup()
    problem.set_val("altitude", 1000.0, units="m")
    problem.set_val("mach", 0.2)
    problem.set_val("required_thrust", 10000.0, units="N")
    problem.set_val("electric_load", 1000.0, units="W")
    problem.set_val("thrust_available", 20000.0, units="N")
    problem.set_val("sea_level_static_thrust", 20000.0, units="N")
    problem.set_val("thrust_supplement", 0.0, units="N")
    problem.set_val("fuel_coeff_3", 0.0)
    problem.set_val("fuel_coeff_2", 0.0)
    problem.set_val("fuel_coeff_1", 2.0)
    problem.set_val("fuel_coeff_altitude", 0.0)
    problem.set_val("he_coefficient", 1.0)
    problem.run_model()

    assert np.isclose(problem.get_val("fuel_flow")[0], expected["Fuel"])
    assert np.isclose(problem.get_val("thrust", units="N")[0], expected["Thrust"])
    assert np.isclose(problem.get_val("tsfc")[0], expected["TSFC"])
    assert np.isclose(problem.get_val("tsfc_imperial")[0], expected["TSFC_Imperial"])
    assert np.isclose(problem.get_val("he_coeff")[0], expected["C"])


def test_engine_primitives_declare_analytic_partials():
    """Check engine primitive derivatives against finite difference."""

    cases = [
        ("total_pressure", TotalPressure(), {"static_pressure": 90000.0, "mach": 0.55, "gamma": 1.37}),
        ("static_pressure", StaticPressure(), {"total_pressure": 110000.0, "mach": 0.55, "gamma": 1.37}),
        ("total_temperature", TotalTemperature(), {"static_temperature": 260.0, "mach": 0.55, "gamma": 1.37}),
        ("static_temperature", StaticTemperature(), {"total_temperature": 285.0, "mach": 0.55, "gamma": 1.37}),
        ("choked_area", ChokedArea(), {"area": 1.4, "mach": 0.65, "gamma": 1.4}),
        ("flow_area", FlowArea(), {"area_star": 0.9, "mach": 0.65, "gamma": 1.4}),
        ("mass_flow", MassFlowParameter(), {"mach": 0.65, "gamma": 1.4}),
        ("nozzle", OffDesignNozzleMach(), {"area_1": 1.4, "area_2": 1.8, "mach_1": 0.65, "gamma": 1.4}),
        ("static_density", StaticDensity(), {"total_density": 1.3, "mach": 0.65, "gamma": 1.4}),
        ("cp", AirSpecificHeat(), {"temperature": 300.0}),
        ("cv", AirSpecificHeatVolume(), {"temperature": 300.0}),
        (
            "gamma",
            ThermalPerfectGamma(),
            {"total_temperature": 800.0, "mach": 0.55, "gamma": 1.37},
        ),
        ("air_heat", AirIntegratedHeat(), {"temperature_low": 300.0, "temperature_high": 1200.0}),
        ("jeta_heat", JetAIntegratedHeat(), {"temperature_low": 300.0, "temperature_high": 1200.0}),
        ("heat_added", AirTemperatureFromHeatAdded(), {"temperature_start": 300.0, "heat": 100000.0}),
        ("heat_removed", AirTemperatureFromHeatRemoved(), {"temperature_start": 1200.0, "heat": 100000.0}),
        ("efficiency", LocalEfficiency(), {"reynolds": 2.5e7}),
        (
            "reynolds",
            LocalReynolds(),
            {
                "static_pressure": 85000.0,
                "static_temperature": 260.0,
                "outer_radius": 0.9,
                "inner_radius": 0.35,
                "mach": 0.45,
                "gamma": 1.36,
            },
        ),
        (
            "simple_off_design",
            SimpleOffDesignTurbofan(),
            {
                "altitude": 1000.0,
                "mach": 0.2,
                "required_thrust": 10000.0,
                "electric_load": 1000.0,
                "thrust_available": 20000.0,
                "sea_level_static_thrust": 20000.0,
                "thrust_supplement": 0.0,
                "fuel_coeff_3": 0.01,
                "fuel_coeff_2": 0.1,
                "fuel_coeff_1": 2.0,
                "fuel_coeff_altitude": 1.0e-7,
                "he_coefficient": 1.0,
            },
        ),
        (
            "burner",
            BurnerFlow(),
            {
                "mass_flow_31": 50.0,
                "area_31": 0.45,
                "total_pressure_31": 800000.0,
                "total_temperature_31": 750.0,
                "mach_31": 0.25,
                "gamma_31": 1.35,
                "outer_radius_31": 0.8,
                "total_temperature_4": 1400.0,
                "fuel_lhv": 43.0e6,
                "combustor_efficiency": 0.99,
            },
        ),
        (
            "stage",
            CompressorStageFlow(),
            {
                "mass_flow_1": 45.0,
                "area_1": 0.55,
                "total_pressure_1": 250000.0,
                "total_temperature_1": 360.0,
                "static_pressure_1": ps_pt(250000.0, 0.32, 1.38),
                "static_temperature_1": ts_tt(360.0, 0.32, 1.38),
                "mach_1": 0.32,
                "gamma_1": 1.38,
                "outer_radius_1": 0.8,
                "stage_pressure_ratio": 1.18,
                "rpm": 6200.0,
                "stage_efficiency": 0.9,
            },
        ),
        (
            "turbine_stage",
            TurbineStageFlow(),
            {
                "mass_flow_1": 48.0,
                "total_pressure_1": 750000.0,
                "total_temperature_1": 1400.0,
                "mach_1": 0.42,
                "gamma_1": 1.32,
                "outer_radius_1": 0.82,
                "inner_radius_1": 0.45,
                "target_total_temperature_3": 1280.0,
                "stage_mach_2": 1.1,
                "rpm": 6200.0,
                "turbine_efficiency": 0.91,
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
            tolerance = 1.0e-3 if name in (
                "nozzle",
                "burner",
                "stage",
                "turbine_stage",
            ) else 1.0e-4
            assert partial_data["abs error"].forward < tolerance


def make_burner_state():
    """Return a compact FAST-Python flow state for burner tests."""

    return {
        "MDot": 50.0,
        "Area": 0.45,
        "Pt": 800000.0,
        "Tt": 750.0,
        "Mach": 0.25,
        "Gam": 1.35,
        "Ro": 0.8,
        "Ri": 0.48,
        "Ts": ts_tt(750.0, 0.25, 1.35),
        "Cp": cp_air(ts_tt(750.0, 0.25, 1.35)),
        "Cv": cv_air(ts_tt(750.0, 0.25, 1.35)),
        "Ps": ps_pt(800000.0, 0.25, 1.35),
    }


def set_burner_values(problem, state, total_temperature_4, fuel_lhv, combustor_efficiency):
    """Set OpenMDAO burner inputs from a FAST-Python flow-state dictionary."""

    problem.set_val("mass_flow_31", state["MDot"], units="kg/s")
    problem.set_val("area_31", state["Area"], units="m**2")
    problem.set_val("total_pressure_31", state["Pt"], units="Pa")
    problem.set_val("total_temperature_31", state["Tt"], units="K")
    problem.set_val("mach_31", state["Mach"])
    problem.set_val("gamma_31", state["Gam"])
    problem.set_val("outer_radius_31", state["Ro"], units="m")
    problem.set_val("total_temperature_4", total_temperature_4, units="K")
    problem.set_val("fuel_lhv", fuel_lhv)
    problem.set_val("combustor_efficiency", combustor_efficiency)


def make_compressor_stage_state():
    """Return a compact FAST-Python flow state for compressor-stage tests."""

    return {
        "MDot": 45.0,
        "Area": 0.55,
        "Pt": 250000.0,
        "Tt": 360.0,
        "Mach": 0.32,
        "Gam": 1.38,
        "Ro": 0.8,
        "Ri": 0.45,
        "Ts": ts_tt(360.0, 0.32, 1.38),
        "Cp": cp_air(ts_tt(360.0, 0.32, 1.38)),
        "Cv": cv_air(ts_tt(360.0, 0.32, 1.38)),
        "Ps": ps_pt(250000.0, 0.32, 1.38),
    }


def set_compressor_stage_values(problem, state, stage_pressure_ratio, rpm, efficiency):
    """Set OpenMDAO compressor-stage inputs from a FAST-Python flow state."""

    problem.set_val("mass_flow_1", state["MDot"], units="kg/s")
    problem.set_val("area_1", state["Area"], units="m**2")
    problem.set_val("total_pressure_1", state["Pt"], units="Pa")
    problem.set_val("total_temperature_1", state["Tt"], units="K")
    problem.set_val("static_pressure_1", state["Ps"], units="Pa")
    problem.set_val("static_temperature_1", state["Ts"], units="K")
    problem.set_val("mach_1", state["Mach"])
    problem.set_val("gamma_1", state["Gam"])
    problem.set_val("outer_radius_1", state["Ro"], units="m")
    problem.set_val("stage_pressure_ratio", stage_pressure_ratio)
    problem.set_val("rpm", rpm, units="rpm")
    problem.set_val("stage_efficiency", efficiency)


def make_turbine_stage_state():
    """Return a compact FAST-Python flow state for turbine-stage tests."""

    return {
        "MDot": 48.0,
        "Area": 0.6,
        "Pt": 750000.0,
        "Tt": 1400.0,
        "Mach": 0.42,
        "Gam": 1.32,
        "Ro": 0.82,
        "Ri": 0.45,
        "Ts": ts_tt(1400.0, 0.42, 1.32),
        "Cp": cp_air(ts_tt(1400.0, 0.42, 1.32)),
        "Cv": cv_air(ts_tt(1400.0, 0.42, 1.32)),
        "Ps": ps_pt(750000.0, 0.42, 1.32),
    }


def set_turbine_stage_values(problem, state, total_temperature_3, mach_2, rpm, efficiency):
    """Set OpenMDAO turbine-stage inputs from a FAST-Python flow state."""

    problem.set_val("mass_flow_1", state["MDot"], units="kg/s")
    problem.set_val("total_pressure_1", state["Pt"], units="Pa")
    problem.set_val("total_temperature_1", state["Tt"], units="K")
    problem.set_val("mach_1", state["Mach"])
    problem.set_val("gamma_1", state["Gam"])
    problem.set_val("outer_radius_1", state["Ro"], units="m")
    problem.set_val("inner_radius_1", state["Ri"], units="m")
    problem.set_val("target_total_temperature_3", total_temperature_3, units="K")
    problem.set_val("stage_mach_2", mach_2)
    problem.set_val("rpm", rpm, units="rpm")
    problem.set_val("turbine_efficiency", efficiency)


def make_simple_off_design_aircraft():
    """Return a compact FAST-Python aircraft for simple off-design tests."""

    return {
        "Specs": {
            "Propulsion": {
                "SLSThrust": [20000.0],
                "ThrustSupp": [0.0],
                "PropArch": {
                    "SrcType": [1],
                },
                "Engine": {
                    "Cff3": 0.0,
                    "Cff2": 0.0,
                    "Cff1": 2.0,
                    "Cffch": 0.0,
                    "HEcoeff": 1.0,
                },
            }
        },
        "Mission": {
            "History": {
                "SI": {
                    "Power": {
                        "Tav": [
                            [0.0, 20000.0],
                        ],
                    }
                }
            }
        },
    }
