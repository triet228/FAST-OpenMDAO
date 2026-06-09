# FAST OpenMDAO

FAST OpenMDAO is the OpenMDAO integration layer for **Future Aircraft Sizing
Tool (FAST)** by The IDEAS Lab in the Aerospace Engineering Department at the
University of Michigan.

The goal is to turn the native `FAST-Python` workflow into a gradient-based
aircraft sizing framework. This repository will hold OpenMDAO components,
groups, derivative checks, and optimization examples while keeping
`FAST-Python` as the source of the FAST equations and data model.

## Current Status

This repository has a working OpenMDAO bridge plus a growing set of
derivative-native FAST equation components:

- `FastPythonComponent`: an `om.ExplicitComponent` that wraps a `FAST-Python`
  run.
- `Gravity` and `StandardAtmosphere`: native OpenMDAO components matching
  `fast_python.atmosphere` with analytical partial derivatives.
- `KPPProjection`, `BatteryReplacementCost`, and `UnitConversion`: native
  OpenMDAO utility components for projection, cost, and scalar conversion
  equations with analytical partial derivatives.
- `FlightConditions`: native mission primitive matching
  `fast_python.mission.compute_flight_conditions` with analytical partial
  derivatives.
- Mission Breguet cruise efficiency triplet helper with analytical partial
  derivatives for fixed architectures.
- Mission Breguet propulsive-efficiency and power-split selectors with
  analytical partial derivatives for fixed storage paths.
- Mission Breguet mass, power, and energy history component with analytical
  partial derivatives for fixed non-detailed-battery architectures.
- Mission Breguet detailed-battery discharge and SOC cutoff helper with
  analytical partial derivatives for fixed active branches.
- Mission Breguet source-energy allocation helper with analytical partial
  derivatives for fixed source layouts.
- Mission Breguet aggregate source-delta application helper with analytical
  partial derivatives for fixed source columns.
- Mission initial source-energy remaining helper with analytical partial
  derivatives for fixed source layouts.
- Mission fixed-shape row-matrix history expansion helper with analytical
  partial derivatives.
- Mission simple and detailed EvalTakeoff, EvalLanding, smooth EvalCruise, and
  prescribed-rate EvalClimb/EvalDescent kinematics and required-power kernels
  with analytical partial derivatives.
- Engine primitives for isentropic pressure, temperature, area-Mach,
  mass-flow parameter, off-design nozzle Mach, perfect-expansion nozzle flow,
  and static-density equations with analytical partial derivatives.
- Engine specific-heat components for air Cp, air Cv, thermally perfect gamma
  update, integrated air heat, integrated Jet-A heat, and inverse air heat
  solvers with analytical partial derivatives.
- Engine local Reynolds and local efficiency components with analytical
  partial derivatives.
- Engine on-design diffuser flow-state component with analytical partial
  derivatives.
- Engine on-design burner flow, fuel addition, combustor pressure loss, and
  exit-state component with analytical partial derivatives.
- Engine one-stage compressor/fan flow component with corrected map scalars
  and analytical partial derivatives.
- Engine fan-exit core/bypass flow split and annulus geometry component with
  analytical partial derivatives.
- Engine one-stage turbine flow component with analytical partial derivatives.
- Engine BADA-style simple off-design turbofan fuel-flow component with
  analytical partial derivatives.
- Engine low-fidelity turboprop linear sizing component with analytical
  partial derivatives.
- Engine low-fidelity turbofan linear sizing component with analytical partial
  derivatives.
- Engine fixed-shape vector normalization helper with analytical partial
  derivatives.
- Propulsion primitives for cable sizing weight, engine lapse, engine thrust
  requirement, turboprop/piston engine sizing weight, power-flow propagation,
  power-available propagation, safe component weight, efficiency selection, and
  supplemental transmitter power with analytical partial derivatives.
- Propulsion fuel-use, battery-source energy history accumulation, and
  fixed-branch battery depletion cutoff with analytical partial derivatives.
- Propulsion fixed-shape split, vector, two-dimensional array, and history
  matrix normalization helpers with analytical partial derivatives.
- Propulsion conventional/electric, parallel-hybrid, series-hybrid,
  turboelectric, and partial-turboelectric architecture matrix builders for
  architecture, split, efficiency, source-type, and transmitter-type arrays
  with analytical partial derivatives.
- Constraint scalar primitives for PsLoss sigmoid, OEI multiplier, FAR 25
  engine-gradient selection, and cruise dynamic pressure with analytical
  partial derivatives.
- Constraint residual components for approach speed, takeoff field length,
  landing field length, all-engines-operative climb, service ceiling, and
  cruise/diversion with analytical partial derivatives and optimization
  validation against FAST-Python residual roots.
- Generic and named FAR 25 climb residual components with analytical partial
  derivatives.
- Battery scalar primitives for effective cell capacity, selected current root,
  ground-charge OCV estimation, nonzero history averaging, empirical
  cycling-aging SOH, detailed parallel-cell resizing, one-step and fixed-history
  equivalent-circuit power dynamics, and simple energy-based battery source
  weight with analytical partial derivatives.
- Battery fixed-shape vector and history-matrix normalization helpers with
  analytical partial derivatives.
- Battery requested-power/time broadcast and initial-SOC normalization helpers
  with analytical partial derivatives for fixed shapes.
- Analysis source-weight vectorization plus analysis and OEW summation helpers
  for scalar/vector weight values with analytical partial derivatives.
- Optimization helper primitives for available electric motor power and
  battery energy plus interior-point feasible slack-step limits,
  Gaussian-elimination pivots, damped BFGS Hessian updates, fixed-shape
  gradient-block formatting, gradient-matrix reshaping, fixed-piece vector and
  matrix concatenation, split-array dimensional normalization, and merit values
  with analytical partial derivatives.
- Optimization power/energy, design split bound, and cruise power availability
  residual helpers with analytical partial derivatives for active constraints.
- Optimization operational split bound residual helper with analytical partial
  derivatives.
- Optimization fixed-index mission-history selectors with analytical partial
  derivatives.
- Optimization fixed-index split-schedule filling helper with analytical
  partial derivatives.
- Optimization objective selector helpers for operational and power-management
  objective values with analytical partial derivatives.
- Regression squared-exponential kernel, fixed-preprocessing Gaussian process
  posterior prediction, fixed-shape target/vector/2D normalization, and
  numeric-scalar, numeric-column, and sample-variance helpers with analytical
  partial derivatives.
- OEW turboprop linear fit, one-step turboprop and turbofan fixed-point
  balances, turbofan airframe GPR weight, and numeric-sum helpers with
  analytical partial derivatives.
- Propulsion turboprop/piston linear and turbofan GPR engine sizing-weight
  helpers with analytical partial derivatives.
- Database-derived MAC and turboprop cruise lift-to-drag estimates with
  analytical partial derivatives.
- Database-derived airframe weight, weight fractions, and wing loading with
  analytical partial derivatives.
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

## Converted Component Index

The public `fast_openmdao` package currently exports:

- Atmosphere: `Gravity`, `StandardAtmosphere`
- Analysis: `AnalysisWeightUpdate`, `ConvergenceError`, `SourceWeightVector`,
  `WeightSum`, `WingAreaFromLoading`
- Battery: `AvailableCellCapacity`, `BatteryChargeOCV`, `BatteryCyclingAging`,
  `BatteryCurrent`, `BatteryHistoryColumnMatrix`, `BatteryHistoryMatrix`,
  `BatteryInitialSOC`, `BatteryNonzeroMean`, `BatteryPowerHistory`,
  `BatteryPowerStep`, `BatteryPowerTimeBroadcast`, `BatteryVector`,
  `DetailedBatterySizing`, `BatteryWeightFromEnergy`
- Constraint analysis: `CruiseDynamicPressure`, `FAR25ClimbConstraint`,
  `FAR25EngineGradient`, `JetAEOClimbConstraint`, `JetApproachConstraint`,
  `JetCeilingConstraint`, `JetCruiseConstraint`, `JetFAR25NamedClimbConstraint`,
  `JetLandingFieldLengthConstraint`, `JetTakeoffFieldLengthConstraint`,
  `OEIMultiplier`, `PsLossSigmoid`
- Cost: `BatteryReplacementCost`
- Database: `DatabaseFanThrustNormalization`, `DatabaseGeometryLoads`,
  `DatabasePropPowerNormalization`, `DatabaseWeightFractions`,
  `MacLiftDragEstimate`, `TurbofanCruiseLiftDragEstimate`,
  `TurbofanMacLiftDragEstimate`, `TurbopropCruiseLiftDragEstimate`
- Engine: `AirIntegratedHeat`, `AirSpecificHeat`, `AirSpecificHeatVolume`,
  `AirTemperatureFromHeatAdded`, `AirTemperatureFromHeatRemoved`, `BurnerFlow`,
  `ChokedArea`, `CompressorStageFlow`, `DiffuserFlow`, `FanFlowSplit`,
  `FlowArea`,
  `EngineVector`, `JetAIntegratedHeat`, `LocalEfficiency`, `LocalReynolds`,
  `MassFlowParameter`, `OffDesignNozzleMach`, `PerfectExpansionNozzleFlow`,
  `SimpleOffDesignTurbofan`, `StaticDensity`, `StaticPressure`,
  `StaticTemperature`, `ThermalPerfectGamma`, `TotalPressure`,
  `TotalTemperature`, `TurbineStageFlow`, `TurbofanLinearSizing`,
  `TurbopropLinearSizing`
- Mission: `CruiseBreguetDetailedBattery`, `CruiseBreguetEfficiencyTriplet`,
  `CruiseBreguetPowerHistory`, `CruiseBreguetPowerSplit`,
  `CruiseBreguetPropulsiveEfficiency`, `CruiseBreguetSourceDelta`,
  `CruiseBreguetSourceEnergy`,
  `CruiseSegmentKinematicsPower`, `CruiseTimeTargetDistance`,
  `DetailedTakeoffSegmentKinematicsPower`, `FlightConditions`, `InitialEnergyRemaining`,
  `LandingSegmentKinematicsPower`, `PrescribedRateSegmentKinematicsPower`,
  `RowMatrix`, `TakeoffSegmentKinematics`
- OEW: `NumericSum`, `TurbofanAirframeWeight`, `TurbofanOEWIterationStep`,
  `TurbopropAirframeWeight`, `TurbopropOEWIterationStep`
- Optimization: `BatteryEnergyAvailable`, `ElectricMotorPowerAvailable`,
  `CruisePowerAvailableConstraint`, `DesignSplitBounds`, `FeasibleStep`,
  `GaussianEliminationPivot`, `GradientBlock`, `GradientMatrix`,
  `ConcatenateMatrices`, `ConcatenateVectors`, `HessianUpdate`, `HistoryArray`,
  `MeritFunction`, `OneBasedHistoryValues`, `OperationalObjective`,
  `OperationalSplitConstraints`, `PowerManagementObjective`,
  `PowerLimitConstraints`, `SplitScheduleFill`, `TwoDimensionalArray`
- Projection: `BatterySpecificEnergyProjection`,
  `ElectricMotorSpecificPowerProjection`, `KPPProjection`
- Propulsion: `BatteryEnergyCutoff`, `BatteryEnergyHistory`,
  `CableWeightForSizing`, `EngineLapse`, `EngineThrustRequirement`,
  `FuelUseHistory`, `ParallelHybridArchitecture`,
  `PartialTurboelectricArchitecture`, `PowerAvailable`, `PowerFlow`,
  `PowerSupplementCheck`, `PropulsionHistoryMatrix`,
  `PropulsionTwoDimensionalArray`, `PropulsionVector`, `SafeComponentWeight`,
  `SeriesHybridArchitecture`, `SimpleSourceTransmitterArchitecture`,
  `SplitValuesVector`, `ThrustSinkEfficiency`, `TransmitterFanEfficiency`,
  `TurboelectricArchitecture`, `TurbofanEngineWeightForSizing`,
  `TurbopropEngineWeightForSizing`
- Regression: `GaussianProcessPrediction`, `RegressionSampleVariance`,
  `RegressionNumericColumn`, `RegressionNumericScalar`,
  `RegressionTargetMatrix`, `RegressionTwoDimensionalArray`, `RegressionVector`,
  `SquaredExponentialKernel`
- Units: `UnitConversion`
- Bridge/builders: `FastPythonComponent`, `make_fast_problem`,
  `make_fast_optimization_problem`

## Repository Layout

```text
FAST-OpenMDAO/
  docs/                  Conversion inventory and design notes
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
conda run -n FAST-OpenMDAO python -m pytest -q
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

Validate the OpenMDAO optimum against repeated FAST-Python point evaluations:

```powershell
fast-openmdao-compact --range-initial 20000 --range-lower 10000 --range-upper 40000 --validate-samples 7
```

The test suite also includes an independent compact cruise-L/D system-level
optimization checked against a FAST-Python L/D sweep, compact hybrid
power-split multi-start validation checked against a shared FAST-Python split
sweep, plus constraint-residual optimization checks that solve OpenMDAO
zero-residual points and verify the optimized values against FAST-Python
residual functions.

## Development Roadmap

1. Add a minimal OpenMDAO component that wraps a `FAST-Python` case. Done.
2. Expose scalar FAST inputs as OpenMDAO design variables. Done.
3. Add finite-difference checks around the component bridge. Done.
4. Add a driver-ready optimization problem builder. Done.
5. Add optimization examples that run from the command line. Done.
6. Add analytic partial hooks for components when derivatives are available.
   Done.
7. Inventory every FAST-Python module for OpenMDAO conversion scope. Done.
8. Convert atmosphere utilities into native OpenMDAO components with analytical
   partial derivatives. Done.
9. Convert projection, cost, and scalar unit-conversion utilities with
   analytical partial derivatives. Done.
10. Convert scalar mission flight-condition primitive with analytical partial
   derivatives. Done.
11. Convert scalar engine primitive equations and fitted specific-heat
   equations with analytical partial derivatives. Done.
12. Convert scalar propulsion primitive equations with analytical partial
   derivatives. Done.
13. Convert scalar constraint primitive equations with analytical partial
   derivatives. Done.
14. Convert scalar battery primitive equations with analytical partial
   derivatives. Done.
15. Convert approach, takeoff field length, landing field length, and
   cruise/diversion constraint residuals with analytical partial derivatives.
   Done.
16. Convert shared FAR 25 climb residual with analytical partial derivatives.
   Done.
17. Convert squared-exponential regression kernel with analytical partial
   derivatives. Done.
18. Convert turboprop airframe-weight linear fit with analytical partial
   derivatives. Done.
19. Convert cruise time-target distance helper with analytical partial
   derivatives. Done.
20. Convert scalar convergence-error helper with analytical partial
   derivatives. Done.
21. Convert optimization available-power and available-energy helpers with
   analytical partial derivatives. Done.
22. Convert scalar/vector analysis and OEW summation helpers with analytical
   partial derivatives. Done.
23. Convert Breguet cruise efficiency triplet helper with analytical partial
   derivatives. Done.
24. Convert Breguet source-energy allocation helper with analytical partial
   derivatives. Done.
25. Convert initial source-energy remaining helper with analytical partial
   derivatives. Done.
26. Convert simple energy-based battery source weight helper with analytical
   partial derivatives. Done.
27. Convert fixed-preprocessing Gaussian process posterior prediction with
   analytical partial derivatives. Done.
28. Convert turbofan airframe-weight regression helper with analytical partial
   derivatives. Done.
29. Convert off-design nozzle Mach helper with analytical partial derivatives.
   Done.
30. Convert inverse air heat Newton helpers with analytical partial
   derivatives. Done.
31. Convert simple off-design turbofan fuel-flow helper with analytical
   partial derivatives. Done.
32. Convert propulsion fuel-use history accumulation with analytical partial
   derivatives. Done.
33. Convert one-step battery equivalent-circuit dynamics with analytical
   partial derivatives. Done.
34. Convert Breguet cruise selector helpers with analytical partial
   derivatives. Done.
35. Convert named FAR 25 climb wrappers with analytical partial derivatives.
   Done.
36. Convert all-engines-operative climb residual with analytical partial
   derivatives. Done.
37. Convert service-ceiling residual with analytical partial derivatives.
   Done.
38. Convert turboprop engine-weight sizing helper with analytical partial
   derivatives. Done.
39. Convert optimization power-limit residual helper with analytical partial
   derivatives. Done.
40. Convert optimization operational split bound helper with analytical
   partial derivatives. Done.
41. Convert optimization objective selector helpers with analytical partial
   derivatives. Done.
42. Convert analysis source-weight vectorization helper with analytical partial
   derivatives. Done.
43. Convert thermally perfect gamma update helper with analytical partial
   derivatives. Done.
44. Convert supplemental transmitter power helper with analytical partial
   derivatives. Done.
45. Convert propulsion power-flow propagation with analytical partial
   derivatives. Done.
46. Convert engine thrust requirement selector with analytical partial
   derivatives. Done.
47. Convert cable sizing weight helper with analytical partial derivatives.
   Done.
48. Convert empirical battery cycling-aging SOH helper with analytical partial
   derivatives. Done.
49. Convert propulsion power-available propagation helper with analytical
   partial derivatives. Done.
50. Convert one-step turboprop OEW fixed-point balance with analytical partial
   derivatives. Done.
51. Convert one-step turbofan OEW fixed-point balance with analytical partial
   derivatives. Done.
52. Convert fan-exit core/bypass flow split with analytical partial
   derivatives. Done.
53. Add complex-step derivative checks where supported by FAST-Python internals.
54. Build reusable groups for mission, propulsion, weights, and objective
   functions.

See `docs/conversion_inventory.md` for module-by-module conversion status.
