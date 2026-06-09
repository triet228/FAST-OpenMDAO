# FAST-Python to FAST-OpenMDAO Conversion Inventory

This inventory tracks the FAST-Python modules that must be reviewed before the
OpenMDAO port can be considered equation-complete. The current repository
started as a wrapper around FAST-Python; derivative-native components should
replace wrapper-only behavior module by module where equations are stable and
closed-form derivatives are practical.

## Status Key

- `converted`: Native OpenMDAO component exists with analytical partials.
- `bridge`: OpenMDAO wrapper can execute FAST-Python, but equations are not yet
  represented as OpenMDAO components.
- `candidate`: Contains numerical equations worth converting.
- `support`: Data, I/O, fixtures, plotting, or orchestration; usually not an
  OpenMDAO component.

## Module Inventory

| FAST-Python module | Status | OpenMDAO target |
| --- | --- | --- |
| `aircraft.py` | support | Keep as FAST data preparation until the data contract stabilizes. |
| `analysis.py` | candidate | Scalar convergence-error and weight-sum helpers converted; top-level analysis group coordinating weights, mission, propulsion, and convergence remains. |
| `atmosphere.py` | converted | `fast_openmdao.atmosphere.Gravity`, `fast_openmdao.atmosphere.StandardAtmosphere`. |
| `battery.py` | candidate | Effective cell capacity and selected current root converted; discharge, charge, aging, and sizing components remain. |
| `cases.py` | support | Case factories, not derivative components. |
| `compare.py` | support | Regression comparison utilities. |
| `constraint.py` | candidate | PsLoss sigmoid, OEI multiplier, FAR 25 engine-gradient selector, cruise dynamic pressure, JetApp, JetTOFL, JetLFL, JetCrs, JetDiv, and shared FAR 25 climb residual converted; full constraint diagram and remaining FAR/jet residual wrappers remain. |
| `core.py` | bridge | Existing `FastPythonComponent` wrapper preserves end-to-end execution. |
| `cost.py` | converted | `fast_openmdao.cost.BatteryReplacementCost` for replacement cost; add more cost components if FAST-Python grows. |
| `data_struct.py` | support | Defaults and schema preparation; may later feed OpenMDAO options. |
| `database.py` | support | Database loading and regression source processing. |
| `engine.py` | candidate | Isentropic pressure, temperature, area-Mach, mass-flow, density, fitted specific-heat, local Reynolds, and local efficiency primitives converted; nozzle iteration, thermodynamic cycle, fan, compressor, turbine, and sizing components remain. |
| `history.py` | support | Mission history table formatting. |
| `io.py` | support | JSON input/output validation. |
| `main.py` | support | CLI entry point. |
| `markers.py` | support | MATLAB marker compatibility. |
| `mission.py` | candidate | `fast_openmdao.mission.FlightConditions`, cruise time-target distance conversion, Breguet cruise efficiency triplet, and Breguet source-energy allocation converted; segment, remaining Breguet cruise, and mission groups remain. |
| `oew.py` | candidate | Turboprop airframe-weight linear fit and numeric-sum helper converted; OEW iteration and turbofan airframe-weight regression remain. |
| `optimization.py` | candidate | Available electric motor power, battery energy, active power-limit residual, and operational split bound helpers converted; recast remaining useful objective and constraint functions as OpenMDAO driver setup, and avoid porting custom optimizers unless needed. |
| `plotting.py` | support | Plot preparation and rendering utilities. |
| `profiles.py` | support | Mission profile factories. |
| `projection.py` | converted | `fast_openmdao.projection.KPPProjection` plus battery and electric motor projection aliases. |
| `propulsion.py` | candidate | Engine lapse, safe component weight, and efficiency selectors converted; power-flow, architecture, sizing, fuel-use, and history components remain. |
| `reference.py` | support | Reference case store and archive access. |
| `regression.py` | candidate | Squared-exponential kernel converted; GP predictor components remain. |
| `retrofit.py` | support | Retrofit option factory placeholder. |
| `safety.py` | support | Discrete fault-tree logic; not a continuous derivative component. |
| `specs.py` | support | Aircraft and engine spec factories. |
| `units.py` | converted | `fast_openmdao.units.UnitConversion` for scalar OpenMDAO variables. |

## Conversion Rules

1. Preserve FAST-Python equations and units first; only refactor once parity
   tests pass.
2. Add analytical `compute_partials` for every converted component.
3. Add parity tests against FAST-Python values and OpenMDAO derivative checks.
4. Keep wrapper support for workflows that have not been decomposed yet.
5. Do not convert plotting, JSON I/O, case factories, or discrete fault-tree
   logic into differentiable OpenMDAO components unless a downstream model
   explicitly needs them.
