# src/fast_openmdao/optimization.py

"""OpenMDAO components for FAST optimization helper equations."""

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


def available_product_value(specific_capacity, installed_weight):
    """Return FAST available power or energy product."""

    return specific_capacity * installed_weight
