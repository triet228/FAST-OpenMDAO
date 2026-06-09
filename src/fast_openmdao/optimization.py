# src/fast_openmdao/optimization.py

"""OpenMDAO components for FAST optimization helper equations."""

import numpy as np
import openmdao.api as om


class ElectricMotorPowerAvailable(om.ExplicitComponent):
    """Compute FAST available electric motor power for sizing constraints."""

    def setup(self):
        self.add_input("electric_motor_specific_power", val=5000.0, units="W/kg")
        self.add_input("electric_motor_weight", val=100.0, units="kg")
        self.add_output("electric_motor_power_available", val=5.0e5, units="W")
        self.declare_partials(of="electric_motor_power_available", wrt="*")

    def compute(self, inputs, outputs):
        outputs["electric_motor_power_available"] = available_product_value(
            inputs["electric_motor_specific_power"][0],
            inputs["electric_motor_weight"][0],
        )

    def compute_partials(self, inputs, partials):
        partials[
            "electric_motor_power_available",
            "electric_motor_specific_power",
        ] = inputs["electric_motor_weight"][0]
        partials[
            "electric_motor_power_available",
            "electric_motor_weight",
        ] = inputs["electric_motor_specific_power"][0]


class BatteryEnergyAvailable(om.ExplicitComponent):
    """Compute FAST available battery energy for sizing constraints."""

    def setup(self):
        self.add_input("battery_specific_energy", val=720000.0, units="J/kg")
        self.add_input("battery_weight", val=1000.0, units="kg")
        self.add_output("battery_energy_available", val=7.2e8, units="J")
        self.declare_partials(of="battery_energy_available", wrt="*")

    def compute(self, inputs, outputs):
        outputs["battery_energy_available"] = available_product_value(
            inputs["battery_specific_energy"][0],
            inputs["battery_weight"][0],
        )

    def compute_partials(self, inputs, partials):
        partials["battery_energy_available", "battery_specific_energy"] = inputs[
            "battery_weight"
        ][0]
        partials["battery_energy_available", "battery_weight"] = inputs[
            "battery_specific_energy"
        ][0]


class PowerLimitConstraints(om.ExplicitComponent):
    """Compute FAST paired lower and upper power/energy limit constraints.

    Inputs:
        used: Used power or energy values.
        available: Available power or energy values.

    Outputs:
        lower_limit: FAST lower residual ``-used / available``.
        upper_limit: FAST upper residual ``used / available - 1``.

    Assumptions:
        This component represents the active, finite-input branch of
        ``fast_python.optimization.power_limit_constraints``. FAST's NaN/Inf
        sanitization is preserved in compute; analytical derivatives are
        meaningful for finite, nonzero available values.
    """

    def initialize(self):
        self.options.declare("vec_size", default=1)
        self.options.declare("eps", default=1.0e-6)

    def setup(self):
        vec_size = self.options["vec_size"]
        rows = np.arange(vec_size)
        self.add_input("used", val=np.ones(vec_size))
        self.add_input("available", val=np.ones(vec_size))
        self.add_output("lower_limit", val=np.zeros(vec_size))
        self.add_output("upper_limit", val=np.zeros(vec_size))
        self.declare_partials(
            of=["lower_limit", "upper_limit"],
            wrt=["used", "available"],
            rows=rows,
            cols=rows,
        )

    def compute(self, inputs, outputs):
        values = power_limit_constraint_values(
            inputs["used"],
            inputs["available"],
            self.options["eps"],
        )
        outputs["lower_limit"] = values["lower"]
        outputs["upper_limit"] = values["upper"]

    def compute_partials(self, inputs, partials):
        values = power_limit_constraint_values(
            inputs["used"],
            inputs["available"],
            self.options["eps"],
        )
        partials["lower_limit", "used"] = values["dlower_dused"]
        partials["lower_limit", "available"] = values["dlower_davailable"]
        partials["upper_limit", "used"] = values["dupper_dused"]
        partials["upper_limit", "available"] = values["dupper_davailable"]


def available_product_value(specific_capacity, installed_weight):
    """Return FAST available power or energy product."""

    return specific_capacity * installed_weight


def power_limit_constraint_values(used, available, eps):
    """Return FAST power-limit residuals and analytical derivatives."""

    used = np.asarray(used, dtype=float).reshape(-1)
    available = np.asarray(available, dtype=float).reshape(-1)
    lower_raw = -used / available
    upper_raw = used / available - 1.0
    finite = (
        np.isfinite(lower_raw)
        & np.isfinite(upper_raw)
        & (available != 0.0)
    )
    lower = sanitize_values(lower_raw, eps)
    upper = sanitize_values(upper_raw, eps)
    dlower_dused = np.zeros_like(used)
    dlower_davailable = np.zeros_like(used)
    dupper_dused = np.zeros_like(used)
    dupper_davailable = np.zeros_like(used)
    dlower_dused[finite] = -1.0 / available[finite]
    dlower_davailable[finite] = used[finite] / available[finite] ** 2
    dupper_dused[finite] = 1.0 / available[finite]
    dupper_davailable[finite] = -used[finite] / available[finite] ** 2
    return {
        "lower": lower,
        "upper": upper,
        "dlower_dused": dlower_dused,
        "dlower_davailable": dlower_davailable,
        "dupper_dused": dupper_dused,
        "dupper_davailable": dupper_davailable,
    }


def sanitize_values(values, eps):
    """Replace NaN and Inf values the way FAST optimization constraints do."""

    array = np.asarray(values, dtype=float).reshape(-1)
    array[np.isnan(array)] = 0.0
    array[np.isinf(array)] = eps
    return array
