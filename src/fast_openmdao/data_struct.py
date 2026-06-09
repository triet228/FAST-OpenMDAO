# src/fast_openmdao/data_struct.py

"""OpenMDAO components for FAST data-structure preprocessing equations."""

import openmdao.api as om


class DefaultFuelSpecificEnergy(om.ExplicitComponent):
    """Return FAST's class-dependent default fuel specific energy.

    Outputs:
        fuel_specific_energy: Default fuel specific energy in kWh/kg.

    Assumptions:
        FAST chooses this value from the discrete aircraft class during
        SpecProcessing. There is no continuous derivative surface because the
        output is a constant for each aircraft-class branch.
    """

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")

    def setup(self):
        self.add_output("fuel_specific_energy", val=12.0)

    def compute(self, inputs, outputs):
        outputs["fuel_specific_energy"] = default_fuel_specific_energy_value(
            self.options["aircraft_class"],
        )


class SpecPowerUnitConversion(om.ExplicitComponent):
    """Convert FAST SpecProcessing power fields to SI units.

    Inputs:
        power_to_weight_sls: Sea-level power loading in preprocessed units.
        power_to_weight_generator: Electric-generator power loading.
        power_to_weight_motor: Electric-motor power loading.
        fuel_specific_energy: Fuel specific energy before SI conversion.
        battery_specific_energy: Battery specific energy before SI conversion.

    Outputs:
        Converted values matching ``fast_python.data_struct.convert_spec_units``.

    Assumptions:
        FAST skips this conversion for analysis type -2. The analysis type is a
        discrete setup option because it changes the derivative constants for
        the whole component.
    """

    def initialize(self):
        self.options.declare("analysis_type", default=1)

    def setup(self):
        self.add_input("power_to_weight_sls", val=1.0)
        self.add_input("power_to_weight_generator", val=1.0)
        self.add_input("power_to_weight_motor", val=1.0)
        self.add_input("fuel_specific_energy", val=1.0)
        self.add_input("battery_specific_energy", val=1.0)
        self.add_output("power_to_weight_sls_si", val=1000.0)
        self.add_output("power_to_weight_generator_si", val=1000.0)
        self.add_output("power_to_weight_motor_si", val=1000.0)
        self.add_output("fuel_specific_energy_si", val=3.6e6)
        self.add_output("battery_specific_energy_si", val=3.6e6)

        scale = spec_power_unit_conversion_scale(self.options["analysis_type"])
        self.declare_partials(
            of="power_to_weight_sls_si",
            wrt="power_to_weight_sls",
            val=scale["power_to_weight"],
        )
        self.declare_partials(
            of="power_to_weight_generator_si",
            wrt="power_to_weight_generator",
            val=scale["power_to_weight"],
        )
        self.declare_partials(
            of="power_to_weight_motor_si",
            wrt="power_to_weight_motor",
            val=scale["power_to_weight"],
        )
        self.declare_partials(
            of="fuel_specific_energy_si",
            wrt="fuel_specific_energy",
            val=scale["specific_energy"],
        )
        self.declare_partials(
            of="battery_specific_energy_si",
            wrt="battery_specific_energy",
            val=scale["specific_energy"],
        )

    def compute(self, inputs, outputs):
        values = spec_power_unit_conversion_values(
            inputs["power_to_weight_sls"][0],
            inputs["power_to_weight_generator"][0],
            inputs["power_to_weight_motor"][0],
            inputs["fuel_specific_energy"][0],
            inputs["battery_specific_energy"][0],
            self.options["analysis_type"],
        )

        for name, value in values.items():
            outputs[name] = value


def spec_power_unit_conversion_scale(analysis_type):
    """Return FAST SpecProcessing conversion factors for one analysis mode."""

    if int(analysis_type) == -2:
        return {
            "power_to_weight": 1.0,
            "specific_energy": 1.0,
        }

    return {
        "power_to_weight": 1.0e3,
        "specific_energy": 3.6e6,
    }


def default_fuel_specific_energy_value(aircraft_class):
    """Return FAST SpecProcessing default fuel specific energy in kWh/kg."""

    if aircraft_class == "Piston":
        return 4.465e7 / 3.6e6

    return 4.32e7 / 3.6e6


def spec_power_unit_conversion_values(
    power_to_weight_sls,
    power_to_weight_generator,
    power_to_weight_motor,
    fuel_specific_energy,
    battery_specific_energy,
    analysis_type,
):
    """Return converted FAST power fields for SpecPowerUnitConversion."""

    scale = spec_power_unit_conversion_scale(analysis_type)
    power_scale = scale["power_to_weight"]
    energy_scale = scale["specific_energy"]
    return {
        "power_to_weight_sls_si": float(power_to_weight_sls) * power_scale,
        "power_to_weight_generator_si": (
            float(power_to_weight_generator) * power_scale
        ),
        "power_to_weight_motor_si": float(power_to_weight_motor) * power_scale,
        "fuel_specific_energy_si": float(fuel_specific_energy) * energy_scale,
        "battery_specific_energy_si": float(battery_specific_energy) * energy_scale,
    }
