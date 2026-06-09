# src/fast_openmdao/__init__.py

"""OpenMDAO integration layer for FAST-Python."""

from fast_openmdao.atmosphere import Gravity, StandardAtmosphere
from fast_openmdao.components import FastPythonComponent
from fast_openmdao.cost import BatteryReplacementCost
from fast_openmdao.mission import FlightConditions
from fast_openmdao.problem import make_fast_optimization_problem, make_fast_problem
from fast_openmdao.projection import (
    BatterySpecificEnergyProjection,
    ElectricMotorSpecificPowerProjection,
    KPPProjection,
)
from fast_openmdao.units import UnitConversion

__version__ = "0.1.0"

__all__ = [
    "BatteryReplacementCost",
    "BatterySpecificEnergyProjection",
    "ElectricMotorSpecificPowerProjection",
    "FastPythonComponent",
    "FlightConditions",
    "Gravity",
    "KPPProjection",
    "StandardAtmosphere",
    "UnitConversion",
    "__version__",
    "make_fast_optimization_problem",
    "make_fast_problem",
]
