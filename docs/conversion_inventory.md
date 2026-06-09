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
| `analysis.py` | candidate | Scalar convergence-error, source-weight vectorization, and weight-sum helpers converted; top-level analysis group coordinating weights, mission, propulsion, and convergence remains. |
| `atmosphere.py` | converted | `fast_openmdao.atmosphere.Gravity`, `fast_openmdao.atmosphere.StandardAtmosphere`. |
| `battery.py` | candidate | Effective cell capacity, selected current root, ground-charge OCV estimation, empirical cycling-aging SOH, detailed cell sizing, one-step and fixed-length equivalent-circuit discharge/charge histories, and simple energy-based battery source weight converted; ground-charge loop remains. |
| `cases.py` | support | Case factories, not derivative components. |
| `compare.py` | support | Regression comparison utilities. |
| `constraint.py` | candidate | PsLoss sigmoid, OEI multiplier, FAR 25 engine-gradient selector, cruise dynamic pressure, JetApp, JetTOFL, JetLFL, JetCrs, JetDiv, JetCeil, JetAEOClimb, shared FAR 25 climb residual, and named Jet25 climb wrappers converted; full constraint diagram orchestration remains. |
| `core.py` | bridge | Existing `FastPythonComponent` wrapper preserves end-to-end execution. |
| `cost.py` | converted | `fast_openmdao.cost.BatteryReplacementCost` for replacement cost; add more cost components if FAST-Python grows. |
| `data_struct.py` | support | Defaults and schema preparation; may later feed OpenMDAO options. |
| `database.py` | candidate | MAC lift-to-drag estimate converted; database loading, field mutation, keyword classification, and regression source processing remain support. |
| `engine.py` | candidate | Isentropic pressure, temperature, area-Mach, mass-flow, density, off-design nozzle Mach, perfect-expansion nozzle flow, fitted specific-heat, inverse air-heat Newton solvers, thermally perfect gamma update, diffuser flow, burner fuel/exit-state flow, compressor/fan stage flow, turbine stage flow, local Reynolds, local efficiency, simple off-design turbofan fuel-flow primitives, and low-fidelity turboprop/turbofan linear sizing converted; thermodynamic cycle, multi-stage compressor/fan/turbine wrappers, and nonlinear sizing components remain. |
| `history.py` | support | Mission history table formatting. |
| `io.py` | support | JSON input/output validation. |
| `main.py` | support | CLI entry point. |
| `markers.py` | support | MATLAB marker compatibility. |
| `mission.py` | candidate | `fast_openmdao.mission.FlightConditions`, cruise time-target distance conversion, Breguet cruise efficiency triplet, Breguet propulsive-efficiency and power-split selectors, non-detailed-battery Breguet mass/power/energy history, detailed-battery Breguet discharge behavior, Breguet source-energy allocation, initial source-energy remaining, simple and detailed EvalTakeoff kinematics, EvalLanding kinematics/reverse-power demand, smooth EvalCruise kinematics/required-power kernel, and prescribed-rate EvalClimb/EvalDescent kinematics/required-power kernel converted; remaining segment mutation/orchestration and mission groups remain. |
| `oew.py` | candidate | Turboprop airframe-weight linear fit, turbofan airframe-weight regression, and numeric-sum helper converted; OEW iteration loops remain. |
| `optimization.py` | candidate | Available electric motor power, battery energy, feasible slack-step limit, line-search merit value, objective selectors, active power-limit residual, design split bound residual, cruise power availability residual, and operational split bound helpers converted; recast remaining useful objective and constraint functions as OpenMDAO driver setup, and avoid porting custom optimizers unless needed. |
| `plotting.py` | support | Plot preparation and rendering utilities. |
| `profiles.py` | support | Mission profile factories. |
| `projection.py` | converted | `fast_openmdao.projection.KPPProjection` plus battery and electric motor projection aliases. |
| `propulsion.py` | candidate | Cable sizing weight, engine lapse, engine thrust requirement, turboprop/piston linear and turbofan GPR engine sizing weight, conventional/electric, parallel-hybrid, series-hybrid, turboelectric, and partial-turboelectric architecture matrix construction, power-flow propagation, power-available propagation, fuel-use accumulation history, smooth battery-source energy accumulation history, safe component weight, efficiency selectors, and supplemental transmitter power converted; architecture orchestration, battery cutoff branches, turbofan nonlinear sizing side effects, and full propulsion-history groups remain. |
| `reference.py` | support | Reference case store and archive access. |
| `regression.py` | candidate | Squared-exponential kernel and fixed-preprocessing GP posterior prediction converted; database search, data-build preprocessing, and variable input classification remain support/orchestration. |
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
