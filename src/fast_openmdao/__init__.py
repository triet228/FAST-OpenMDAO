# src/fast_openmdao/__init__.py

"""OpenMDAO integration layer for FAST-Python."""

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
    ChokedArea,
    FlowArea,
    MassFlowParameter,
    StaticDensity,
    StaticPressure,
    StaticTemperature,
    TotalPressure,
    TotalTemperature,
)
from fast_openmdao.mission import FlightConditions
from fast_openmdao.problem import make_fast_optimization_problem, make_fast_problem
from fast_openmdao.projection import (
    BatterySpecificEnergyProjection,
    ElectricMotorSpecificPowerProjection,
    KPPProjection,
)
from fast_openmdao.propulsion import (
    EngineLapse,
    SafeComponentWeight,
    ThrustSinkEfficiency,
    TransmitterFanEfficiency,
)
from fast_openmdao.units import UnitConversion

__version__ = "0.1.0"

__all__ = [
    "BatteryReplacementCost",
    "BatterySpecificEnergyProjection",
    "AvailableCellCapacity",
    "BatteryCurrent",
    "ChokedArea",
    "CruiseDynamicPressure",
    "ElectricMotorSpecificPowerProjection",
    "EngineLapse",
    "FAR25EngineGradient",
    "FAR25ClimbConstraint",
    "FastPythonComponent",
    "FlightConditions",
    "FlowArea",
    "Gravity",
    "JetApproachConstraint",
    "JetCruiseConstraint",
    "JetLandingFieldLengthConstraint",
    "JetTakeoffFieldLengthConstraint",
    "KPPProjection",
    "MassFlowParameter",
    "OEIMultiplier",
    "PsLossSigmoid",
    "SafeComponentWeight",
    "StaticDensity",
    "StaticPressure",
    "StaticTemperature",
    "StandardAtmosphere",
    "TotalPressure",
    "TotalTemperature",
    "ThrustSinkEfficiency",
    "TransmitterFanEfficiency",
    "UnitConversion",
    "__version__",
    "make_fast_optimization_problem",
    "make_fast_problem",
]
