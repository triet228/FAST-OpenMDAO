# tests/test_unit_components.py

"""Tests for OpenMDAO FAST unit conversion components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import UnitArrayConversion, UnitConversion
from fast_python.units import convert_length, convert_temperature, convert_tsfc


def test_unit_conversion_matches_fast_python_multiplicative_tables():
    """Check scalar unit conversion parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "length",
        UnitConversion(quantity="length", oldunit="m", newunit="ft"),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("value", 10.0)
    problem.run_model()

    assert np.isclose(problem.get_val("converted_value")[0], convert_length(10.0, "m", "ft"))


def test_unit_conversion_matches_fast_python_temperature():
    """Check affine temperature conversion parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "temperature",
        UnitConversion(quantity="temperature", oldunit="C", newunit="F"),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("value", 25.0)
    problem.run_model()

    assert np.isclose(
        problem.get_val("converted_value")[0],
        convert_temperature(25.0, "C", "F"),
    )


def test_unit_conversion_declares_analytic_partials():
    """Check unit conversion derivatives against finite difference."""

    cases = [
        UnitConversion(quantity="length", oldunit="m", newunit="ft"),
        UnitConversion(quantity="tsfc", oldunit="SI", newunit="Imp"),
        UnitConversion(quantity="temperature", oldunit="F", newunit="K"),
    ]

    for component in cases:
        problem = om.Problem()
        problem.model.add_subsystem("conversion", component, promotes=["*"])
        problem.setup()
        problem.set_val("value", 10.0)
        problem.run_model()
        partials = problem.check_partials(out_stream=None, method="fd")

        for partial_data in partials["conversion"].values():
            assert partial_data["abs error"].forward < 1.0e-5


def test_unit_conversion_supports_custom_variable_names():
    """Check conversion components can avoid promoted-name collisions."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "tsfc",
        UnitConversion(
            quantity="tsfc",
            oldunit="SI",
            newunit="Imp",
            input_name="tsfc_si",
            output_name="tsfc_imp",
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("tsfc_si", 1.826)
    problem.run_model()

    assert np.isclose(problem.get_val("tsfc_imp")[0], convert_tsfc(1.826, "SI", "Imp"))


def test_unit_array_conversion_matches_fast_python_shape_preserving_values():
    """Check fixed-shape array conversion parity with FAST-Python."""

    values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    problem = om.Problem()
    problem.model.add_subsystem(
        "length",
        UnitArrayConversion(
            quantity="length",
            oldunit="m",
            newunit="ft",
            input_shape=values.shape,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("values", values)
    problem.run_model()

    assert np.allclose(
        problem.get_val("converted_values"),
        np.asarray(convert_length(values.tolist(), "m", "ft")),
    )


def test_unit_array_conversion_matches_fast_python_temperature():
    """Check fixed-shape affine temperature array conversion parity."""

    values = np.asarray([0.0, 25.0, 100.0])
    problem = om.Problem()
    problem.model.add_subsystem(
        "temperature",
        UnitArrayConversion(
            quantity="temperature",
            oldunit="C",
            newunit="F",
            input_shape=values.shape,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("values", values)
    problem.run_model()

    assert np.allclose(
        problem.get_val("converted_values"),
        np.asarray(convert_temperature(values.tolist(), "C", "F")),
    )


def test_unit_array_conversion_declares_analytic_partials():
    """Check fixed-shape unit array conversion derivatives."""

    cases = [
        (
            UnitArrayConversion(
                quantity="length",
                oldunit="m",
                newunit="ft",
                input_shape=(2, 2),
            ),
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        ),
        (
            UnitArrayConversion(
                quantity="temperature",
                oldunit="C",
                newunit="F",
                input_shape=(3,),
            ),
            np.asarray([0.0, 25.0, 100.0]),
        ),
    ]

    for component, values in cases:
        problem = om.Problem()
        problem.model.add_subsystem("conversion", component, promotes=["*"])
        problem.setup()
        problem.set_val("values", values)
        problem.run_model()
        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-6,
        )

        for partial_data in partials["conversion"].values():
            assert partial_data["abs error"].forward < 1.0e-8
