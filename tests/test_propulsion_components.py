# tests/test_propulsion_components.py

"""Tests for OpenMDAO FAST propulsion primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    EngineLapse,
    SafeComponentWeight,
    ThrustSinkEfficiency,
    TransmitterFanEfficiency,
)
from fast_python.propulsion import (
    engine_lapse,
    get_thrust_sink_efficiency,
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


def test_propulsion_primitives_declare_analytic_partials():
    """Check propulsion primitive derivatives against finite difference."""

    cases = [
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
            assert partial_data["abs error"].forward < 1.0e-6


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
