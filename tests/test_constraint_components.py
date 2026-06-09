# tests/test_constraint_components.py

"""Tests for OpenMDAO FAST constraint primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    CruiseDynamicPressure,
    FAR25EngineGradient,
    OEIMultiplier,
    PsLossSigmoid,
)
from fast_python.constraint import (
    cruise_dynamic_pressure,
    far25_engine_gradient,
    oei_multiplier,
    sigmoid,
)


def test_ps_loss_sigmoid_matches_fast_python():
    """Check PsLoss sigmoid parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "sigmoid",
        PsLossSigmoid(a=10.0, b=2.0, c=0.2, d=1.0),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("ps_loss", 0.3)
    problem.run_model()

    aircraft = make_constraint_aircraft(constraint_type=1, ps_loss=0.3)

    assert np.isclose(
        problem.get_val("sigmoid")[0],
        sigmoid(aircraft, 10.0, 2.0, 0.2, 1.0),
    )


def test_oei_multiplier_matches_fast_python():
    """Check OEI multiplier parity for both FAST modes."""

    engine_problem = om.Problem()
    engine_problem.model.add_subsystem(
        "oei",
        OEIMultiplier(constraint_type=0),
        promotes=["*"],
    )
    engine_problem.setup()
    engine_problem.set_val("num_engines", 4.0)
    engine_problem.run_model()

    assert np.isclose(
        engine_problem.get_val("oei_multiplier")[0],
        oei_multiplier(make_constraint_aircraft(constraint_type=0, num_engines=4)),
    )

    ps_problem = om.Problem()
    ps_problem.model.add_subsystem(
        "oei",
        OEIMultiplier(constraint_type=1),
        promotes=["*"],
    )
    ps_problem.setup()
    ps_problem.set_val("ps_loss", 0.3)
    ps_problem.run_model()

    assert np.isclose(
        ps_problem.get_val("oei_multiplier")[0],
        oei_multiplier(make_constraint_aircraft(constraint_type=1, ps_loss=0.3)),
    )


def test_far25_engine_gradient_matches_fast_python():
    """Check FAR 25 engine-gradient selector parity."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "gradient",
        FAR25EngineGradient(num_engines=3, constraint_type=0),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("two_engine", 0.024)
    problem.set_val("three_engine", 0.027)
    problem.set_val("four_engine", 0.030)
    problem.run_model()

    aircraft = make_constraint_aircraft(constraint_type=0, num_engines=3)

    assert np.isclose(
        problem.get_val("engine_gradient")[0],
        far25_engine_gradient(aircraft, 0.024, 0.027, 0.030),
    )


def test_cruise_dynamic_pressure_matches_fast_python():
    """Check cruise dynamic pressure parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem("cruise", CruiseDynamicPressure(), promotes=["*"])
    problem.setup()
    problem.set_val("altitude", 9000.0, units="m")
    problem.set_val("mach", 0.58)
    problem.run_model()

    expected = cruise_dynamic_pressure(9000.0, 0.58)

    assert np.isclose(problem.get_val("dynamic_pressure")[0], expected[0])
    assert np.isclose(problem.get_val("density_ratio")[0], expected[1])
    assert np.isclose(problem.get_val("velocity", units="ft/s")[0], expected[2])


def test_constraint_primitives_declare_analytic_partials():
    """Check constraint primitive derivatives against finite difference."""

    cases = [
        ("sigmoid", PsLossSigmoid(), {"ps_loss": 0.3}),
        ("oei", OEIMultiplier(constraint_type=1), {"num_engines": 2.0, "ps_loss": 0.3}),
        ("gradient", FAR25EngineGradient(num_engines=2), {"two_engine": 0.024, "three_engine": 0.027, "four_engine": 0.030}),
        ("cruise", CruiseDynamicPressure(), {"altitude": 9000.0, "mach": 0.58}),
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
            assert partial_data["abs error"].forward < 1.0e-5


def make_constraint_aircraft(constraint_type, ps_loss=0.2, num_engines=2):
    """Return minimal aircraft dictionary for FAST-Python constraint helpers."""

    return {
        "Settings": {
            "ConstraintType": constraint_type,
        },
        "Specs": {
            "Performance": {
                "PsLoss": ps_loss,
            },
            "Propulsion": {
                "NumEngines": num_engines,
            },
        },
    }
