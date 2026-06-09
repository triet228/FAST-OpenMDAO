# Component Authoring Guide

Use this guide when converting a FAST-Python equation kernel into a native
OpenMDAO component.

## Convert Only Stable Equation Kernels

Good component candidates are deterministic, fixed-shape equations with clear
inputs, outputs, units, and differentiable branches. Keep the following as
support or group/driver logic unless there is a specific downstream need:

- File I/O, plotting, fixtures, case factories, and data loading.
- Dictionary assembly or mutation without a standalone equation.
- Adaptive loops whose iteration count changes with input values.
- Custom optimizer logic that belongs in an OpenMDAO driver.
- Discrete fault-tree or branch-selection behavior that is not differentiable.

Record the decision in `conversion_inventory.md`.

## Implementation Checklist

1. Preserve FAST-Python equations and units.
2. Add an `om.ExplicitComponent` in the closest existing module under
   `src/fast_openmdao`.
3. Declare every input and output with units where OpenMDAO supports them.
4. Implement `compute` for the exact fixed-shape behavior.
5. Implement analytical `compute_partials`.
6. Export the component from `src/fast_openmdao/__init__.py`.
7. Add parity tests against FAST-Python values.
8. Add derivative checks with `check_partials`.
9. Update `docs/conversion_inventory.md`.

## Shape and Branch Assumptions

Native components should make shape and branch assumptions explicit. Prefer
constructor options for fixed dimensions or active branches when the original
FAST-Python function accepts flexible lists, nested dictionaries, or mode flags.

Do not silently resize arrays inside `compute`. If a component requires a fixed
number of sources, segments, split entries, history rows, or architecture
columns, document that in the class docstring and tests.

## Derivative Expectations

Analytical partial derivatives are required for converted equation kernels.
Use finite difference only for bridge-level black-box FAST-Python execution or
for an explicitly documented unsupported derivative path.

Derivative tests should check both value parity and partials. Keep tolerances
tight enough to catch unit mistakes, but avoid brittle tolerances when the
FAST-Python reference uses iterative solvers or fitted data.

## Test Pattern

Most component tests should follow this shape:

```python
import openmdao.api as om


def test_component_matches_fast_python_reference():
    problem = om.Problem()
    problem.model.add_subsystem("component", MyComponent(), promotes=["*"])
    problem.setup()
    problem.set_val("input_name", 1.0, units="m")
    problem.run_model()

    assert abs(problem.get_val("output_name", units="kg")[0] - expected) < 1.0e-9


def test_component_partials():
    problem = om.Problem()
    problem.model.add_subsystem("component", MyComponent(), promotes=["*"])
    problem.setup()
    problem.run_model()
    partials = problem.check_partials(out_stream=None, method="fd")

    for component_partials in partials.values():
        for partial_data in component_partials.values():
            assert partial_data["abs error"].forward < 1.0e-6
```

For components with branch-dependent derivatives, add one test per supported
active branch.

## Conversion Audit

`tools/conversion_audit.py` compares FAST-Python functions with known
OpenMDAO conversion evidence and the statuses in `conversion_inventory.md`.
After adding a component, update audit aliases if the FAST-Python function name
does not map directly to the new component class name.

The audit is a review aid. Passing the audit does not replace parity tests,
derivative checks, or engineering review of units and branch assumptions.
