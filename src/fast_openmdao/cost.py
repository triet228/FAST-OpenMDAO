# src/fast_openmdao/cost.py

"""OpenMDAO components for FAST cost equations."""

import numpy as np
import openmdao.api as om

from fast_python.cost import battery_replacement_cost


YEARS = np.asarray([2023.0, 2026.0, 2030.0, 2035.0])
NMC_BMS = np.asarray([2.0, 2.2, 3.0, 3.5])
LFP_BMS = np.asarray([2.2, 3.1, 3.5, 2.6])
NMC_CAPACITY_COST = np.asarray([127.61, 112.42, 96.52, 76.35])
LFP_CAPACITY_COST = np.asarray([120.91, 97.56, 82.73, 76.62])
DISCOUNT_RATE = 0.07


class BatteryReplacementCost(om.ExplicitComponent):
    """Compute FAST battery replacement cost with analytical partials.

    Inputs:
        year: Calendar year for battery cost curves.
        battery_specific_energy: Battery specific energy in J/kg.
        battery_weight: Installed battery weight in kg.

    Outputs:
        battery_replacement_cost: Discounted replacement cost in dollars.

    Assumptions:
        Battery chemistry, BMS inclusion, and lifespan are discrete design
        choices and are therefore OpenMDAO options rather than differentiable
        inputs.
    """

    def initialize(self):
        self.options.declare("chemistry", default=1)
        self.options.declare("bms", default=1)
        self.options.declare("lifespan", default=5.0)

    def setup(self):
        self.add_input("year", val=2026.0)
        self.add_input("battery_specific_energy", val=720000.0, units="J/kg")
        self.add_input("battery_weight", val=1000.0, units="kg")
        self.add_output("battery_replacement_cost", val=1.0)
        self.declare_partials(of="battery_replacement_cost", wrt="year")
        self.declare_partials(
            of="battery_replacement_cost",
            wrt="battery_specific_energy",
        )
        self.declare_partials(of="battery_replacement_cost", wrt="battery_weight")

    def compute(self, inputs, outputs):
        values = battery_cost_values(
            self.options["chemistry"],
            self.options["bms"],
            self.options["lifespan"],
            inputs["year"][0],
            inputs["battery_specific_energy"][0],
            inputs["battery_weight"][0],
        )
        outputs["battery_replacement_cost"] = values["cost"]

    def compute_partials(self, inputs, partials):
        values = battery_cost_values(
            self.options["chemistry"],
            self.options["bms"],
            self.options["lifespan"],
            inputs["year"][0],
            inputs["battery_specific_energy"][0],
            inputs["battery_weight"][0],
        )
        partials["battery_replacement_cost", "year"] = values["dcost_dyear"]
        partials["battery_replacement_cost", "battery_specific_energy"] = values[
            "dcost_dbattery_specific_energy"
        ]
        partials["battery_replacement_cost", "battery_weight"] = values[
            "dcost_dbattery_weight"
        ]


def battery_cost_aircraft(chemistry, battery_specific_energy, battery_weight):
    """Return a minimal FAST aircraft dictionary for cost parity calls."""

    return {
        "Specs": {
            "Battery": {
                "Chem": chemistry,
            },
            "Power": {
                "SpecEnergy": {
                    "Batt": battery_specific_energy,
                },
            },
            "Weight": {
                "Batt": battery_weight,
            },
        }
    }


def battery_cost_values(
    chemistry,
    bms,
    lifespan,
    year,
    battery_specific_energy,
    battery_weight,
):
    """Return battery replacement cost intermediates and derivatives."""

    bms_fraction, dbms_fraction_dyear = bms_cost_fraction_with_derivative(
        chemistry,
        bms,
        year,
    )
    capacity_cost, dcapacity_cost_dyear = capacity_cost_with_derivative(
        chemistry,
        year,
    )
    discount = (1.0 + DISCOUNT_RATE) ** lifespan
    rated_capacity = battery_specific_energy / 3600.0 / 1000.0 * battery_weight
    bms_multiplier = 1.0 + bms_fraction / 100.0
    cost = bms_multiplier * capacity_cost * rated_capacity / discount
    dcost_dyear = rated_capacity / discount * (
        dbms_fraction_dyear / 100.0 * capacity_cost
        + bms_multiplier * dcapacity_cost_dyear
    )
    dcost_drated_capacity = bms_multiplier * capacity_cost / discount

    return {
        "cost": cost,
        "dcost_dyear": dcost_dyear,
        "dcost_dbattery_specific_energy": (
            dcost_drated_capacity * battery_weight / 3600.0 / 1000.0
        ),
        "dcost_dbattery_weight": (
            dcost_drated_capacity * battery_specific_energy / 3600.0 / 1000.0
        ),
    }


def bms_cost_fraction_with_derivative(chemistry, bms, year):
    """Return BMS cost percentage and derivative with respect to year."""

    if bms == 0:
        return 0.0, 0.0

    if bms != 1:
        battery_replacement_cost(
            battery_cost_aircraft(chemistry, 720000.0, 1000.0),
            year,
            bms,
            5.0,
        )

    if chemistry == 1:
        return polynomial_value_and_derivative(YEARS, NMC_BMS, 1, year)

    if chemistry == 2:
        return polynomial_value_and_derivative(YEARS, LFP_BMS, 2, year)

    battery_replacement_cost(
        battery_cost_aircraft(chemistry, 720000.0, 1000.0),
        year,
        bms,
        5.0,
    )
    return 0.0, 0.0


def capacity_cost_with_derivative(chemistry, year):
    """Return capacity cost and derivative with respect to year."""

    if chemistry == 1:
        return polynomial_value_and_derivative(YEARS, NMC_CAPACITY_COST, 2, year)

    if chemistry == 2:
        return polynomial_value_and_derivative(YEARS, LFP_CAPACITY_COST, 3, year)

    battery_replacement_cost(
        battery_cost_aircraft(chemistry, 720000.0, 1000.0),
        year,
        1,
        5.0,
    )
    return 0.0, 0.0


def polynomial_value_and_derivative(x_values, y_values, degree, x_value):
    """Return polynomial fit value and exact derivative at one point."""

    coefficients = np.polyfit(x_values, y_values, degree)
    derivative_coefficients = np.polyder(coefficients)
    return (
        float(np.polyval(coefficients, x_value)),
        float(np.polyval(derivative_coefficients, x_value)),
    )
