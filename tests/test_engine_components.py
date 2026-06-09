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
    ChokedArea,
    FlowArea,
    JetAIntegratedHeat,
    LocalEfficiency,
    LocalReynolds,
    MassFlowParameter,
    OffDesignNozzleMach,
    StaticDensity,
    StaticPressure,
    StaticTemperature,
    ThermalPerfectGamma,
    TotalPressure,
    TotalTemperature,
)
from fast_python.engine import (
    a_astar,
    astar_a,
    cp_air,
    cp_jeta,
    cv_air,
    local_efficiency,
    local_reynolds,
    mass_flow_parameter,
    new_gamma,
    off_design_nozzle,
    ps_pt,
    pt_ps,
    rhos_rhot,
    ts_tt,
    tt_ts,
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
            tolerance = 1.0e-3 if name == "nozzle" else 1.0e-4
            assert partial_data["abs error"].forward < tolerance
