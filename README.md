# FAST OpenMDAO

FAST OpenMDAO is the OpenMDAO integration layer for **Future Aircraft Sizing
Tool (FAST)** by The IDEAS Lab in the Aerospace Engineering Department at the
University of Michigan.

The package wraps `FAST-Python` cases and provides derivative-native OpenMDAO
components for converted FAST equation kernels. `FAST-Python` remains the
source of FAST equations and data structures; this repository focuses on
OpenMDAO components, derivative checks, optimization setup, and examples.

## Status

- `FastPythonComponent` wraps a `FAST-Python` run as an `om.ExplicitComponent`.
- `make_fast_problem` builds path-based FAST input/output bridges.
- `make_fast_optimization_problem` adds design variables, objective,
  constraints, and an SLSQP driver by default.
- Native components cover atmosphere, mission, propulsion, engine, battery,
  constraints, cost, database, analysis, OEW, regression, safety, optimization,
  units, specs, and data-structure helpers.
- Converted fixed-shape equation kernels include analytical partial derivatives
  and FAST-Python parity tests where practical.
- Remaining FAST-Python behavior is treated as support, orchestration,
  adaptive-loop logic, or driver setup rather than forced into standalone
  differentiable components.

See `docs/conversion_inventory.md` for the module-by-module conversion status.

## Layout

```text
src/fast_openmdao/     Python package
tests/                 parity, derivative, regression, and smoke tests
docs/                  conversion inventory and audit notes
tools/                 conversion audit utilities
```

## Documentation

- `docs/index.md`: documentation map and project workflow.
- `docs/setup.md`: environment setup, validation, and troubleshooting.
- `docs/component_authoring.md`: conventions for adding native OpenMDAO
  components.
- `docs/conversion_inventory.md`: FAST-Python conversion status by module.

## Setup

Use the `FAST-OpenMDAO` conda environment.

```powershell
conda activate FAST-OpenMDAO
pip install -e .[dev]
```

`FAST-Python` must be importable. This project expects it at:

```text
C:\Users\homin\Projects\FAST-Python
```

If needed, add it to `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "C:\Users\homin\Projects\FAST-Python;$env:PYTHONPATH"
```

## Validate

Run the test suite:

```powershell
pytest
```

Run the conversion audit:

```powershell
python tools/conversion_audit.py
```

Show remaining FAST-Python functions classified as non-component scope:

```powershell
python tools/conversion_audit.py --show-remaining
```

## Basic Usage

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

Analytical derivatives can be supplied through `partial_derivatives`. Missing
partials use OpenMDAO finite difference, keeping black-box FAST-Python runs
driver-ready while native components are added.

## Optimization

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
    objective={"name": "mtow"},
)

problem.setup()
problem.run_driver()
```

## Split Matrix Optimization

Use `make_fast_auto_split_optimization_problem` when a custom propulsion
architecture stores operational split matrices directly in the FAST aircraft
dictionary. The helper inspects `Specs.Propulsion.PropArch`, finds editable
`OperDwn` or `OperUps` split matrices, infers row or column normalization, and
then exposes each active branching entry as a scalar OpenMDAO design variable.
It fails closed if the split matrix choice or normalization direction is
ambiguous.

```python
from fast_openmdao import make_fast_auto_split_optimization_problem

problem = make_fast_auto_split_optimization_problem(
    aircraft=aircraft,
    mission=mission,
    preferred_matrix="OperDwn",
    output_specs=[
        {
            "name": "fuel_burn",
            "path": ("aircraft", "Mission", "History", "SI", "Weight", "Fburn", -1),
            "units": "kg",
        },
    ],
    objective={"name": "fuel_burn"},
)

problem.setup()
problem.run_driver()
```

When both `OperDwn` and `OperUps` are editable and valid, pass
`preferred_matrix` so the optimizer does not guess. Use
`make_fast_split_optimization_problem` for explicit matrix control:

```python
from fast_openmdao import make_fast_split_optimization_problem

problem = make_fast_split_optimization_problem(
    aircraft=aircraft,
    mission=mission,
    split_specs=[
        {
            "label": "downstream",
            "prefix": "downstream",
            "target": "aircraft",
            "matrix_path": ("Specs", "Propulsion", "PropArch", "OperDwn"),
            "architecture_path": ("Specs", "Propulsion", "PropArch", "Arch"),
            "axis": "row",
        },
    ],
    output_specs=[
        {
            "name": "fuel_burn",
            "path": ("aircraft", "Mission", "History", "SI", "Weight", "Fburn", -1),
            "units": "kg",
        },
    ],
    objective={"name": "fuel_burn"},
)

problem.setup()
problem.run_driver()
```

Set `axis` to `"row"` for FAST-style split rows that distribute one upstream
component to multiple downstream components, or `"column"` for architectures
that normalize incoming split columns. Single-connection rows or columns remain
fixed by default; set `include_singletons=True` only when those entries should
also become design variables.

## Compact Example

Run the compact electric optimization smoke example:

```powershell
fast-openmdao-compact --range-initial 20000 --range-lower 10000 --range-upper 40000
```

Validate the OpenMDAO optimum against repeated FAST-Python point evaluations:

```powershell
fast-openmdao-compact --range-initial 20000 --range-lower 10000 --range-upper 40000 --validate-samples 7
```

## Roadmap

- Add complex-step derivative checks where supported by FAST-Python internals.
- Build reusable OpenMDAO groups for mission, propulsion, weights, and
  objective functions.
