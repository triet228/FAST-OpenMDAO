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
    MassFlowParameter,
    StaticDensity,
    StaticPressure,
    StaticTemperature,
    TotalPressure,
    TotalTemperature,
)
from fast_python.engine import (
    a_astar,
    astar_a,
    cp_air,
    cp_jeta,
    cv_air,
    mass_flow_parameter,
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
        ("static_density", StaticDensity(), {"total_density": 1.3, "mach": 0.65, "gamma": 1.4}),
        ("cp", AirSpecificHeat(), {"temperature": 300.0}),
        ("cv", AirSpecificHeatVolume(), {"temperature": 300.0}),
        ("air_heat", AirIntegratedHeat(), {"temperature_low": 300.0, "temperature_high": 1200.0}),
        ("jeta_heat", JetAIntegratedHeat(), {"temperature_low": 300.0, "temperature_high": 1200.0}),
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
            assert partial_data["abs error"].forward < 1.0e-4
