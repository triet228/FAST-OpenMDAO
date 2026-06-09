# Setup and Validation

Use the `FAST-OpenMDAO` conda environment for local work.

```powershell
conda activate FAST-OpenMDAO
pip install -e .[dev]
```

`FAST-Python` must also be importable. The expected local checkout is:

```text
C:\Users\homin\Projects\FAST-Python
```

If it is not already installed into the environment, install it in editable
mode or add it to `PYTHONPATH`:

```powershell
pip install -e C:\Users\homin\Projects\FAST-Python
```

```powershell
$env:PYTHONPATH = "C:\Users\homin\Projects\FAST-Python;$env:PYTHONPATH"
```

## Validate the Checkout

Run the tests:

```powershell
pytest
```

Run the conversion audit:

```powershell
python tools/conversion_audit.py
```

Show functions intentionally kept outside native component scope:

```powershell
python tools/conversion_audit.py --show-remaining
```

Run the compact electric example:

```powershell
fast-openmdao-compact --range-initial 20000 --range-lower 10000 --range-upper 40000
```

Validate the compact optimum against repeated FAST-Python evaluations:

```powershell
fast-openmdao-compact --range-initial 20000 --range-lower 10000 --range-upper 40000 --validate-samples 7
```

## FAST Path Specs

`make_fast_problem` and `FastPythonComponent` use path specs to move scalar
OpenMDAO values into copied FAST dictionaries before each run.

Input specs write OpenMDAO values into either `aircraft` or `mission`:

```python
{
    "name": "mission_range",
    "target": "mission",
    "path": ("Target", "Valu", 0),
    "val": 926000.0,
    "units": "m",
}
```

Output specs extract scalar values from the FAST result dictionary:

```python
{
    "name": "mtow",
    "path": ("mtow",),
    "units": "kg",
}
```

If `output_specs` is omitted, the default output is `mtow` at path
`("mtow",)` with units of `kg`.

## Analytical Partials

Bridge-level analytical partials are optional. Provide them as a mapping keyed
by `(output_name, input_name)`. Values may be constants or callables accepting
`inputs`, `result`, `aircraft`, and `mission`.

```python
def d_mtow_d_range(_inputs, _result, aircraft, _mission):
    return 1.0 / aircraft["Specs"]["Aero"]["L_D"]["Crs"]


partial_derivatives = {
    ("mtow", "mission_range"): d_mtow_d_range,
}
```

Configured partials are declared as exact OpenMDAO derivatives. Missing
partials fall back to finite difference so black-box FAST-Python workflows
remain driver-ready.

## Troubleshooting

If `fast_python` cannot be imported, confirm the conda environment is active
and either install `FAST-Python` in editable mode or set `PYTHONPATH`.

If OpenMDAO report files clutter the working tree during tests, set:

```powershell
$env:OPENMDAO_REPORTS = "0"
```

If the conversion audit reports a new unclassified FAST-Python function, update
`docs/conversion_inventory.md` with the intended status and add conversion
evidence or rationale.
