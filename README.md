# FAST OpenMDAO

FAST OpenMDAO is the OpenMDAO integration layer for **Future Aircraft Sizing
Tool (FAST)** by The IDEAS Lab in the Aerospace Engineering Department at the
University of Michigan.

The goal is to turn the native `FAST-Python` workflow into a gradient-based
aircraft sizing framework. This repository will hold OpenMDAO components,
groups, derivative checks, and optimization examples while keeping
`FAST-Python` as the source of the FAST equations and data model.

## Current Status

This repository now has its first OpenMDAO bridge:

- `FastPythonComponent`: an `om.ExplicitComponent` that wraps a `FAST-Python`
  run.
- `make_fast_optimization_problem`: a driver-ready builder that attaches
  design variables, objective, constraints, and an SLSQP driver by default.
- Path-based scalar input specs that write OpenMDAO values into nested FAST
  aircraft or mission dictionaries.
- Path-based scalar output specs that extract values from the FAST result.
- Analytic scalar partial derivatives when supplied through
  `partial_derivatives`, with OpenMDAO finite-difference fallback for
  black-box FAST-Python quantities.
- `fast-openmdao-compact`: a small command-line optimization example using the
  real FAST-Python backend.

## Repository Layout

```text
FAST-OpenMDAO/
  src/fast_openmdao/     Python package for OpenMDAO integration
  tests/                 Unit and smoke tests
  README.md              Project overview and setup notes
  pyproject.toml         Package metadata and test configuration
```

## Local Setup

Use Python 3.11 so the environment matches `FAST-Python`.

```powershell
conda create -n FAST-OpenMDAO python=3.11 -y
conda activate FAST-OpenMDAO
python -m pip install -e C:\Users\homin\Projects\FAST-Python
python -m pip install -e .[dev]
```

If Conda resolves that name to an unwritable system prefix, create and activate
the user-local path explicitly:

```powershell
conda create -p C:\Users\homin\.conda\envs\FAST-OpenMDAO python=3.11 -y
conda activate C:\Users\homin\.conda\envs\FAST-OpenMDAO
python -m pip install -e C:\Users\homin\Projects\FAST-Python
python -m pip install -e .[dev]
```

If the environment already exists, activate it and run the two install commands
again to refresh editable installs.

## Validate

```powershell
python -m pytest -q
```

## Minimal OpenMDAO Usage

```python
from fast_openmdao import make_fast_problem
from fast_python import native_case

aircraft, mission = native_case("ATR42")

problem = make_fast_problem(
    aircraft=aircraft,
    mission=mission,
    input_specs=[
        {
            "name": "mission_range",
            "target": "mission",
            "path": ("Target", "Valu", 0),
            "val": 926000.0,
            "units": "m",
        },
    ],
)

problem.model.add_design_var("mission_range", lower=500000.0, upper=1200000.0)
problem.model.add_objective("mtow")
problem.setup()
problem.run_model()
```

When a FAST quantity has an analytical derivative, pass it through
`partial_derivatives`:

```python
def d_mtow_d_range(_inputs, _result, aircraft, _mission):
    return 1.0 / aircraft["Specs"]["Aero"]["L_D"]["Crs"]


problem = make_fast_problem(
    aircraft=aircraft,
    mission=mission,
    input_specs=[range_spec],
    partial_derivatives={
        ("mtow", "mission_range"): d_mtow_d_range,
    },
)
```

The component declares supplied partials as exact OpenMDAO derivatives. Missing
partials still use finite difference, which keeps black-box FAST-Python runs
driver-ready while internals are converted toward derivative-native equations.

## Minimal Optimization Setup

```python
from fast_openmdao import make_fast_optimization_problem

problem = make_fast_optimization_problem(
    aircraft=aircraft,
    mission=mission,
    input_specs=[
        {
            "name": "cruise_lift_to_drag",
            "target": "aircraft",
            "path": ("Specs", "Aero", "L_D", "Crs"),
            "val": 10.0,
        },
    ],
    design_vars=[
        {
            "name": "cruise_lift_to_drag",
            "lower": 5.0,
            "upper": 25.0,
        },
    ],
    objective={
        "name": "mtow",
    },
)

problem.setup()
problem.run_driver()
```

## Compact Example

Run the real FAST-Python compact electric optimization smoke example:

```powershell
fast-openmdao-compact --range-initial 20000 --range-lower 10000 --range-upper 40000
```

## Development Roadmap

1. Add a minimal OpenMDAO component that wraps a `FAST-Python` case. Done.
2. Expose scalar FAST inputs as OpenMDAO design variables. Done.
3. Add finite-difference checks around the component bridge. Done.
4. Add a driver-ready optimization problem builder. Done.
5. Add optimization examples that run from the command line. Done.
6. Add analytic partial hooks for components when derivatives are available.
   Done.
7. Add complex-step derivative checks where supported by FAST-Python internals.
8. Build reusable groups for mission, propulsion, weights, and objective
   functions.
