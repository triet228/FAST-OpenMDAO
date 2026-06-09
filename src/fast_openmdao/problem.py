# src/fast_openmdao/problem.py

"""Problem builders for FAST OpenMDAO models."""

import openmdao.api as om

from fast_openmdao.components import FastPythonComponent, default_mtow_output


def make_fast_problem(
    aircraft,
    mission=None,
    input_specs=(),
    output_specs=None,
    runner=None,
    component_name="fast",
    promotes=None,
):
    """Create an OpenMDAO Problem containing one FAST-Python component.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        mission: Optional baseline FAST mission/profile dictionary.
        input_specs: Scalar OpenMDAO inputs mapped to FAST paths.
        output_specs: Scalar outputs extracted from the FAST result dictionary.
        runner: Optional callable used instead of the default FAST-Python run.
        component_name: Subsystem name used in the OpenMDAO model.
        promotes: Optional OpenMDAO promotions list.

    Outputs:
        Unsetup OpenMDAO Problem so callers can add design variables,
        objectives, constraints, drivers, and recorders before setup().
    """

    if promotes is None:
        promotes = ["*"]

    problem = om.Problem()
    problem.model.add_subsystem(
        component_name,
        FastPythonComponent(
            aircraft=aircraft,
            mission=mission,
            input_specs=input_specs,
            output_specs=default_outputs(output_specs),
            runner=runner,
        ),
        promotes=promotes,
    )
    return problem


def default_outputs(output_specs):
    """Return default output specs when callers omit explicit outputs."""

    if output_specs is None:
        return (default_mtow_output(),)

    return output_specs
