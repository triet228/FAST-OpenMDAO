# tests/test_data_struct_components.py

"""Tests for OpenMDAO FAST data-structure preprocessing components."""

import numpy as np
import openmdao.api as om

from fast_openmdao import SpecPowerUnitConversion
from fast_python.data_struct import convert_spec_units


def test_spec_power_unit_conversion_matches_fast_python():
    """Check SpecProcessing power unit conversion parity."""

    case = make_spec_power_case()
    expected = evaluate_fast_python_spec_power_case(case, analysis_type=1)
    problem = make_spec_power_problem(analysis_type=1)
    set_spec_power_values(problem, case)
    problem.run_model()

    assert np.isclose(problem.get_val("power_to_weight_sls_si")[0], expected["P_W"]["SLS"])
    assert np.isclose(
        problem.get_val("power_to_weight_generator_si")[0],
        expected["P_W"]["EG"],
    )
    assert np.isclose(
        problem.get_val("power_to_weight_motor_si")[0],
        expected["P_W"]["EM"],
    )
    assert np.isclose(
        problem.get_val("fuel_specific_energy_si")[0],
        expected["SpecEnergy"]["Fuel"],
    )
    assert np.isclose(
        problem.get_val("battery_specific_energy_si")[0],
        expected["SpecEnergy"]["Batt"],
    )


def test_spec_power_unit_conversion_skip_branch_matches_fast_python():
    """Check FAST's analysis type -2 no-conversion branch."""

    case = make_spec_power_case()
    expected = evaluate_fast_python_spec_power_case(case, analysis_type=-2)
    problem = make_spec_power_problem(analysis_type=-2)
    set_spec_power_values(problem, case)
    problem.run_model()

    assert np.isclose(problem.get_val("power_to_weight_sls_si")[0], expected["P_W"]["SLS"])
    assert np.isclose(
        problem.get_val("power_to_weight_generator_si")[0],
        expected["P_W"]["EG"],
    )
    assert np.isclose(
        problem.get_val("power_to_weight_motor_si")[0],
        expected["P_W"]["EM"],
    )
    assert np.isclose(
        problem.get_val("fuel_specific_energy_si")[0],
        expected["SpecEnergy"]["Fuel"],
    )
    assert np.isclose(
        problem.get_val("battery_specific_energy_si")[0],
        expected["SpecEnergy"]["Batt"],
    )


def test_data_struct_primitives_declare_analytic_partials():
    """Check data-structure preprocessing derivatives."""

    problem = make_spec_power_problem(analysis_type=1)
    set_spec_power_values(problem, make_spec_power_case())
    problem.run_model()
    partials = problem.check_partials(out_stream=None, method="fd")

    for component in partials.values():
        for metadata in component.values():
            assert np.max(metadata["rel error"].forward) < 1.0e-8


def make_spec_power_problem(analysis_type):
    """Return an OpenMDAO problem for SpecPowerUnitConversion."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "conversion",
        SpecPowerUnitConversion(analysis_type=analysis_type),
        promotes=["*"],
    )
    problem.setup()
    return problem


def set_spec_power_values(problem, case):
    """Set OpenMDAO values for the SpecProcessing power conversion case."""

    problem.set_val("power_to_weight_sls", case["P_W"]["SLS"])
    problem.set_val("power_to_weight_generator", case["P_W"]["EG"])
    problem.set_val("power_to_weight_motor", case["P_W"]["EM"])
    problem.set_val("fuel_specific_energy", case["SpecEnergy"]["Fuel"])
    problem.set_val("battery_specific_energy", case["SpecEnergy"]["Batt"])


def evaluate_fast_python_spec_power_case(case, analysis_type):
    """Return FAST-Python SpecProcessing power conversion output."""

    settings = {
        "Analysis": {
            "Type": analysis_type,
        },
    }
    power = {
        "P_W": dict(case["P_W"]),
        "SpecEnergy": dict(case["SpecEnergy"]),
    }
    convert_spec_units(settings, power)
    return power


def make_spec_power_case():
    """Return a compact FAST power dictionary for conversion tests."""

    return {
        "P_W": {
            "SLS": 0.12,
            "EG": 0.34,
            "EM": 0.56,
        },
        "SpecEnergy": {
            "Fuel": 11.8,
            "Batt": 0.42,
        },
    }
