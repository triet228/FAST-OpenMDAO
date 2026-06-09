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
- `complete-equation-scope`: Reviewed numerical equations have native
  derivative components; remaining behavior is orchestration, mutation,
  adaptive loops, or custom solver setup rather than a standalone differentiable
  equation kernel.
- `candidate`: Contains numerical equations still worth converting.
- `support`: Data, I/O, fixtures, plotting, or orchestration; usually not an
  OpenMDAO component.

## Module Inventory

| FAST-Python module | Status | OpenMDAO target |
| --- | --- | --- |
| `aircraft.py` | support | Keep as FAST data preparation until the data contract stabilizes. |
| `analysis.py` | complete-equation-scope | Scalar convergence-error, source-weight vectorization/restoration, source-weight summation, analysis setup wing-area, detailed-battery flag setup, and analysis-loop MTOW/OEW/source-weight update helpers converted; remaining top-level analysis behavior coordinates weights, mission, propulsion, and convergence as group/orchestration logic. |
| `atmosphere.py` | converted | `fast_openmdao.atmosphere.Gravity`, `fast_openmdao.atmosphere.StandardAtmosphere`. |
| `battery.py` | complete-equation-scope | Effective cell capacity, selected current root, ground-charge OCV estimation, nonzero history averaging, requested-power/time broadcast, initial-SOC normalization, fixed-shape vector/history-matrix normalization, fixed-shape scalar/list restoration, empirical cycling-aging SOH, detailed cell sizing, one-step and fixed-length equivalent-circuit discharge/charge histories, and simple energy-based battery source weight converted; remaining ground-charge behavior is loop orchestration around converted kernels. |
| `cases.py` | support | Case factories, not derivative components. |
| `compare.py` | support | Regression comparison utilities. |
| `constraint.py` | complete-equation-scope | PsLoss sigmoid, OEI multiplier, FAR 25 engine-gradient selector, cruise dynamic pressure, JetApp, JetTOFL, JetLFL, JetCrs, JetDiv, JetCeil, JetAEOClimb, shared FAR 25 climb residual, and named Jet25 climb wrappers converted; full constraint diagram behavior is optimization/group orchestration. |
| `core.py` | bridge | Existing `FastPythonComponent` wrapper preserves end-to-end execution. |
| `cost.py` | converted | `fast_openmdao.cost.BMSCostFraction`, `BatteryCapacityCost`, and `BatteryReplacementCost` for FAST battery cost curves and replacement cost. |
| `data_struct.py` | complete-equation-scope | SpecProcessing power-to-weight and specific-energy unit conversion helper plus class-dependent default fuel specific energy converted; remaining schema preparation, validation, regression default filling, and nested dictionary mutation are support/orchestration. |
| `database.py` | complete-equation-scope | MAC, turbofan cruise, turbofan MAC/Reynolds, and turboprop cruise lift-to-drag estimates plus shared airframe weight, weight fraction, wing-loading, geometry, payload, burden, fixed-branch airplane design-group percent margins, fan thrust-normalization, prop power-normalization, and turboprop default thrust-loading preprocessing equations converted; remaining database loading, field mutation, keyword classification, and regression source processing are support. |
| `engine.py` | complete-equation-scope | Isentropic pressure, temperature, area-Mach, mass-flow, density, off-design nozzle Mach, perfect-expansion nozzle flow, fitted specific-heat, inverse air-heat Newton solvers, thermally perfect gamma update, diffuser flow, burner fuel/exit-state flow, compressor/fan stage flow, fan-exit core/bypass split, turbine stage flow, local Reynolds, local efficiency, fixed-shape vector normalization, fixed-shape scalar/list restoration, simple off-design turbofan fuel-flow primitives, and low-fidelity turboprop/turbofan linear sizing converted; remaining thermodynamic cycles, multi-stage compressor/fan/turbine wrappers, and nonlinear sizing routines choose branch counts or mutate nested engine objects and should be represented as OpenMDAO groups or solver setup, not single explicit components. |
| `history.py` | support | Mission history table formatting. |
| `io.py` | support | JSON input/output validation. |
| `main.py` | support | CLI entry point. |
| `markers.py` | support | MATLAB marker compatibility. |
| `mission.py` | complete-equation-scope | `fast_openmdao.mission.FlightConditions`, cruise time-target distance conversion, Breguet cruise efficiency triplet, Breguet propulsive-efficiency and power-split selectors, non-detailed-battery Breguet mass/power/energy history, detailed-battery Breguet discharge behavior, Breguet source-energy allocation, aggregate source-delta application, initial source-energy remaining, fixed-shape row-matrix history expansion, fixed-shape segment split-history expansion, fixed-shape scalar/list restoration, fixed-slice history vector/matrix assignment, simple and detailed EvalTakeoff kinematics, EvalLanding kinematics/reverse-power demand, smooth EvalCruise kinematics/required-power kernel, and prescribed-rate EvalClimb/EvalDescent kinematics/required-power kernel converted; remaining segment mutation and mission assembly belong in OpenMDAO groups. |
| `oew.py` | complete-equation-scope | Turboprop airframe-weight linear fit, one-step turboprop and turbofan OEW fixed-point balances, turbofan airframe-weight regression, and numeric-sum helper converted; remaining OEW behavior is fixed-point loop orchestration. |
| `optimization.py` | complete-equation-scope | Available electric motor power, battery energy, feasible slack-step limit, Gaussian-elimination pivot, damped BFGS Hessian update, fixed-index mission-history selection helpers, fixed-index split-schedule filling, aggregate optimized split schedule assembly, fixed-index optimized split write-back, operational simplex tableau setup, fixed-shape gradient-block formatting, gradient-matrix reshaping, fixed-piece vector/matrix concatenation, split-array dimensional normalization, NaN/Inf sanitizers, empty-output normalization, line-search merit value, objective selectors, active power-limit residual, design split bound residual, cruise power availability residual, operational split bound helpers, and aggregate design/operational sizing constraint assembly converted; remaining optimization routines are custom optimizer/driver logic and should remain OpenMDAO driver setup unless a fixed differentiable residual is needed. |
| `plotting.py` | support | Plot preparation and rendering utilities. |
| `profiles.py` | support | Mission profile factories. |
| `projection.py` | converted | `fast_openmdao.projection.KPPProjection` plus battery and electric motor projection aliases. |
| `propulsion.py` | complete-equation-scope | Cable sizing weight, engine lapse, engine thrust requirement, turboprop/piston linear and turbofan GPR engine sizing weight, conventional/electric, parallel-hybrid, series-hybrid, turboelectric, and partial-turboelectric architecture matrix construction, power-flow propagation, power-available propagation, fuel-use accumulation history, smooth battery-source energy accumulation history, fixed-branch non-detailed battery cutoff handling, fixed-branch detailed-battery SOC cutoff handling, fixed-shape split/vector/2D/history-matrix normalization, fixed-shape scalar/list restoration, fixed-slice history vector/matrix assignment, safe component weight, efficiency selectors, and supplemental transmitter power converted; remaining architecture creation, split-callable inspection, turbofan nonlinear sizing side effects, and propulsion-history assembly are orchestration/support. |
| `reference.py` | support | Reference case store and archive access. |
| `regression.py` | complete-equation-scope | Squared-exponential kernel, fixed-preprocessing GP posterior prediction, fixed-shape vector/2D/target normalization, numeric-scalar/column preprocessing, sample-variance, weighted-hyperparameter scaling, prior-mean, and inverse-covariance preprocessing helpers converted; remaining database search, data-build row filtering, and variable input classification are support/orchestration. |
| `retrofit.py` | support | Retrofit option factory placeholder. |
| `safety.py` | support | Continuous failure-probability helper converted; discrete fault-tree logic remains support rather than differentiable OpenMDAO components. |
| `specs.py` | support | AEA custom architecture, FAST split-scalar and zero segment split helpers, plus LM100J_Hybrid custom architecture and operation split matrices converted; remaining aircraft and engine spec factories are dictionary assembly and preset data. |
| `units.py` | converted | `fast_openmdao.units.UnitConversion` for scalar OpenMDAO variables and `UnitArrayConversion` for fixed-shape unit arrays. |

## Conversion Rules

1. Preserve FAST-Python equations and units first; only refactor once parity
   tests pass.
2. Add analytical `compute_partials` for every converted component.
3. Add parity tests against FAST-Python values and OpenMDAO derivative checks.
4. Keep wrapper support for workflows that have not been decomposed yet.
5. Do not convert plotting, JSON I/O, case factories, or discrete fault-tree
   logic into differentiable OpenMDAO components unless a downstream model
   explicitly needs them.
