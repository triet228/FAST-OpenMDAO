# src/fast_openmdao/__init__.py

"""OpenMDAO integration layer for FAST-Python."""

from fast_openmdao.analysis import ConvergenceError, WeightSum
from fast_openmdao.atmosphere import Gravity, StandardAtmosphere
from fast_openmdao.battery import AvailableCellCapacity, BatteryCurrent
from fast_openmdao.components import FastPythonComponent
from fast_openmdao.constraint import (
    CruiseDynamicPressure,
    FAR25ClimbConstraint,
    FAR25EngineGradient,
    JetApproachConstraint,
    JetCruiseConstraint,
    JetLandingFieldLengthConstraint,
    JetTakeoffFieldLengthConstraint,
    OEIMultiplier,
    PsLossSigmoid,
)
from fast_openmdao.cost import BatteryReplacementCost
from fast_openmdao.engine import (
    AirIntegratedHeat,
    AirSpecificHeat,
    AirSpecificHeatVolume,
    ChokedArea,
    FlowArea,
    JetAIntegratedHeat,
    LocalEfficiency,
    LocalReynolds,
    MassFlowParameter,
    StaticDensity,
    StaticPressure,
    StaticTemperature,
    TotalPressure,
    TotalTemperature,
)
from fast_openmdao.mission import (
    CruiseBreguetEfficiencyTriplet,
    CruiseBreguetSourceEnergy,
    CruiseTimeTargetDistance,
    FlightConditions,
)
from fast_openmdao.oew import NumericSum, TurbopropAirframeWeight
from fast_openmdao.optimization import (
    BatteryEnergyAvailable,
    ElectricMotorPowerAvailable,
    PowerLimitConstraints,
)
from fast_openmdao.problem import make_fast_optimization_problem, make_fast_problem
from fast_openmdao.projection import (
    BatterySpecificEnergyProjection,
    ElectricMotorSpecificPowerProjection,
    KPPProjection,
)
from fast_openmdao.regression import SquaredExponentialKernel
from fast_openmdao.propulsion import (
    EngineLapse,
    SafeComponentWeight,
    ThrustSinkEfficiency,
    TransmitterFanEfficiency,
)
from fast_openmdao.units import UnitConversion

__version__ = "0.1.0"

__all__ = [
    "AirIntegratedHeat",
    "AirSpecificHeat",
    "AirSpecificHeatVolume",
    "BatteryEnergyAvailable",
    "BatteryReplacementCost",
    "BatterySpecificEnergyProjection",
    "AvailableCellCapacity",
    "BatteryCurrent",
    "ChokedArea",
    "ConvergenceError",
    "CruiseDynamicPressure",
    "CruiseBreguetEfficiencyTriplet",
    "CruiseBreguetSourceEnergy",
    "CruiseTimeTargetDistance",
    "ElectricMotorSpecificPowerProjection",
    "ElectricMotorPowerAvailable",
    "EngineLapse",
    "FAR25EngineGradient",
    "FAR25ClimbConstraint",
    "FastPythonComponent",
    "FlightConditions",
    "FlowArea",
    "Gravity",
    "JetApproachConstraint",
    "JetAIntegratedHeat",
    "JetCruiseConstraint",
    "JetLandingFieldLengthConstraint",
    "JetTakeoffFieldLengthConstraint",
    "KPPProjection",
    "LocalEfficiency",
    "LocalReynolds",
    "MassFlowParameter",
    "NumericSum",
    "OEIMultiplier",
    "PowerLimitConstraints",
    "PsLossSigmoid",
    "SafeComponentWeight",
    "StaticDensity",
    "StaticPressure",
    "StaticTemperature",
    "StandardAtmosphere",
    "SquaredExponentialKernel",
    "TotalPressure",
    "TotalTemperature",
    "ThrustSinkEfficiency",
    "TransmitterFanEfficiency",
    "TurbopropAirframeWeight",
    "UnitConversion",
    "WeightSum",
    "__version__",
    "make_fast_optimization_problem",
    "make_fast_problem",
]
