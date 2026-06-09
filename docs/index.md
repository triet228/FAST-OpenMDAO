# FAST OpenMDAO Documentation

This directory contains the working documentation for FAST OpenMDAO. The
README stays short; use these files for setup details, contribution workflow,
and conversion status.

## Start Here

1. Read `setup.md` to install the editable package, make `FAST-Python`
   importable, and run validation commands.
2. Read `component_authoring.md` before adding or changing derivative-native
   OpenMDAO components.
3. Read `conversion_inventory.md` when deciding whether a FAST-Python function
   should become a native component, remain bridge-only behavior, or be treated
   as orchestration/support.

## Project Model

FAST OpenMDAO has two layers:

- Bridge layer: `FastPythonComponent` runs a FAST-Python case from OpenMDAO
  scalar inputs and extracts scalar outputs from FAST-shaped result
  dictionaries.
- Native component layer: fixed-shape FAST equation kernels are implemented as
  OpenMDAO explicit components with analytical partial derivatives and parity
  tests against FAST-Python.

The bridge keeps full FAST workflows usable while native components are added
for equations that are stable, differentiable, and useful to optimization
drivers.

## Common Tasks

| Task | Where to look |
| --- | --- |
| Install and validate locally | `setup.md` |
| Map an OpenMDAO variable to a FAST dictionary path | `setup.md` |
| Add a new native component | `component_authoring.md` |
| Decide if FAST-Python code is component scope | `conversion_inventory.md` |
| Re-check conversion coverage | `python tools/conversion_audit.py` |
| Run the compact optimization example | `fast-openmdao-compact --help` |

## Documentation Rules

- Keep README concise and link to focused docs instead of restoring long
  inventories there.
- Keep examples runnable in the `FAST-OpenMDAO` conda environment.
- State units and fixed-shape assumptions whenever an example crosses FAST and
  OpenMDAO boundaries.
- Update `conversion_inventory.md` when a FAST-Python module changes status or
  a new equation kernel is converted.
