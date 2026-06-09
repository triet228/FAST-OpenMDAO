# src/fast_openmdao/problem.py

"""Problem builders for FAST OpenMDAO models."""

import openmdao.api as om

from fast_openmdao.components import FastPythonComponent, default_mtow_output


DESIGN_VAR_KEYS = (
    "lower",
    "upper",
    "ref",
    "ref0",
    "indices",
    "adder",
    "scaler",
    "units",
    "parallel_deriv_color",
    "cache_linear_solution",
    "flat_indices",
)
OBJECTIVE_KEYS = (
    "ref",
    "ref0",
    "index",
    "units",
    "adder",
    "scaler",
    "parallel_deriv_color",
    "cache_linear_solution",
    "flat_indices",
    "alias",
)
CONSTRAINT_KEYS = (
    "lower",
    "upper",
    "equals",
    "ref",
    "ref0",
    "adder",
    "scaler",
    "units",
    "indices",
    "linear",
    "parallel_deriv_color",
    "cache_linear_solution",
    "flat_indices",
    "alias",
)


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


def make_fast_optimization_problem(
    aircraft,
    mission=None,
    input_specs=(),
    output_specs=None,
    runner=None,
    component_name="fast",
    promotes=None,
    design_vars=(),
    objective=None,
    constraints=(),
    driver=None,
    driver_options=None,
):
    """Create a driver-ready OpenMDAO optimization problem for FAST.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        mission: Optional baseline FAST mission/profile dictionary.
        input_specs: Scalar inputs mapped to FAST paths.
        output_specs: Scalar outputs extracted from the FAST result.
        runner: Optional callable used instead of the default FAST-Python run.
        component_name: Subsystem name used in the OpenMDAO model.
        promotes: Optional OpenMDAO promotions list.
        design_vars: Design variable specs with at least a name field.
        objective: Objective spec. Defaults to {"name": "mtow"}.
        constraints: Constraint specs with at least a name field.
        driver: Optional OpenMDAO driver. Defaults to ScipyOptimizeDriver.
        driver_options: Driver option values such as optimizer and maxiter.

    Outputs:
        Unsetup OpenMDAO Problem configured with driver, design variables,
        objective, and constraints.
    """

    problem = make_fast_problem(
        aircraft=aircraft,
        mission=mission,
        input_specs=input_specs,
        output_specs=output_specs,
        runner=runner,
        component_name=component_name,
        promotes=promotes,
    )
    configure_driver(problem, driver, driver_options)
    add_design_variables(problem.model, design_vars)
    add_objective(problem.model, objective)
    add_constraints(problem.model, constraints)
    return problem


def configure_driver(problem, driver=None, driver_options=None):
    """Attach a driver and apply driver options."""

    if driver is None:
        driver = om.ScipyOptimizeDriver()
        driver.options["optimizer"] = "SLSQP"
        driver.options["disp"] = False

    problem.driver = driver

    for key, value in (driver_options or {}).items():
        if key in problem.driver.options:
            problem.driver.options[key] = value
        else:
            problem.driver.opt_settings[key] = value


def add_design_variables(model, design_vars):
    """Add design variable specs to an OpenMDAO model."""

    for spec in design_vars:
        name, kwargs = split_spec(spec, DESIGN_VAR_KEYS)
        model.add_design_var(name, **kwargs)


def add_objective(model, objective=None):
    """Add the objective spec to an OpenMDAO model."""

    if objective is None:
        objective = {
            "name": "mtow",
        }

    name, kwargs = split_spec(objective, OBJECTIVE_KEYS)
    model.add_objective(name, **kwargs)


def add_constraints(model, constraints):
    """Add constraint specs to an OpenMDAO model."""

    for spec in constraints:
        name, kwargs = split_spec(spec, CONSTRAINT_KEYS)
        model.add_constraint(name, **kwargs)


def split_spec(spec, allowed_keys):
    """Return a spec name and filtered OpenMDAO keyword arguments."""

    if isinstance(spec, str):
        return spec, {}

    name = spec["name"]
    kwargs = {
        key: value
        for key, value in spec.items()
        if key in allowed_keys
    }
    return name, kwargs


def default_outputs(output_specs):
    """Return default output specs when callers omit explicit outputs."""

    if output_specs is None:
        return (default_mtow_output(),)

    return output_specs
