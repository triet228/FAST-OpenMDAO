# src/fast_openmdao/propulsion.py

"""OpenMDAO components for FAST propulsion primitive equations."""

import openmdao.api as om

from fast_python.atmosphere import standard_atmosphere


RHO_SL_STD = standard_atmosphere(0.0)[2]


class EngineLapse(om.ExplicitComponent):
    """Estimate lapsed thrust or power from sea-level-static output.

    Inputs:
        sea_level_static: Sea-level-static thrust or power.
        density: Air density in kg/m**3.

    Outputs:
        lapsed_output: Lapsed thrust or power in the same units as input.

    Assumptions:
        Aircraft class is a discrete option. Turbofan output scales linearly
        with density ratio; turboprop and piston output use FAST's zero
        exponent and therefore remain equal to sea-level-static output.
    """

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")

    def setup(self):
        self.add_input("sea_level_static", val=1.0)
        self.add_input("density", val=RHO_SL_STD, units="kg/m**3")
        self.add_output("lapsed_output", val=1.0)
        self.declare_partials(of="lapsed_output", wrt="*")

    def compute(self, inputs, outputs):
        outputs["lapsed_output"] = engine_lapse_value(
            inputs["sea_level_static"][0],
            self.options["aircraft_class"],
            inputs["density"][0],
        )

    def compute_partials(self, inputs, partials):
        derivatives = engine_lapse_derivatives(
            inputs["sea_level_static"][0],
            self.options["aircraft_class"],
            inputs["density"][0],
        )
        partials["lapsed_output", "sea_level_static"] = derivatives[
            "doutput_dsea_level_static"
        ]
        partials["lapsed_output", "density"] = derivatives["doutput_ddensity"]


class SafeComponentWeight(om.ExplicitComponent):
    """Compute component weight from power and power-to-weight ratio."""

    def setup(self):
        self.add_input("power", val=1.0, units="W")
        self.add_input("power_to_weight", val=1.0)
        self.add_output("component_weight", val=1.0, units="kg")
        self.declare_partials(of="component_weight", wrt="*")

    def compute(self, inputs, outputs):
        outputs["component_weight"] = safe_component_weight_value(
            inputs["power"][0],
            inputs["power_to_weight"][0],
        )

    def compute_partials(self, inputs, partials):
        power = inputs["power"][0]
        power_to_weight = inputs["power_to_weight"][0]

        if abs(power) < 1.0e-12:
            partials["component_weight", "power"] = 0.0
            partials["component_weight", "power_to_weight"] = 0.0
            return

        partials["component_weight", "power"] = 1.0 / power_to_weight
        partials["component_weight", "power_to_weight"] = (
            -power / power_to_weight ** 2
        )


class ThrustSinkEfficiency(om.ExplicitComponent):
    """Select FAST thrust-sink fan or propeller efficiency."""

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")

    def setup(self):
        self.add_input("fan_efficiency", val=0.9)
        self.add_input("propeller_efficiency", val=0.85)
        self.add_output("thrust_sink_efficiency", val=0.9)
        self.declare_partials(of="thrust_sink_efficiency", wrt="*")

    def compute(self, inputs, outputs):
        aircraft_class = self.options["aircraft_class"].lower()

        if aircraft_class == "turbofan":
            outputs["thrust_sink_efficiency"] = inputs["fan_efficiency"][0]
        elif aircraft_class in ("turboprop", "piston"):
            outputs["thrust_sink_efficiency"] = inputs["propeller_efficiency"][0]
        else:
            raise ValueError(f"Invalid aircraft class: {self.options['aircraft_class']}")

    def compute_partials(self, inputs, partials):
        aircraft_class = self.options["aircraft_class"].lower()
        partials["thrust_sink_efficiency", "fan_efficiency"] = 0.0
        partials["thrust_sink_efficiency", "propeller_efficiency"] = 0.0

        if aircraft_class == "turbofan":
            partials["thrust_sink_efficiency", "fan_efficiency"] = 1.0
        elif aircraft_class in ("turboprop", "piston"):
            partials["thrust_sink_efficiency", "propeller_efficiency"] = 1.0
        else:
            raise ValueError(f"Invalid aircraft class: {self.options['aircraft_class']}")


class TransmitterFanEfficiency(om.ExplicitComponent):
    """Select FAST transmitter fan efficiency for supplemental power bookkeeping."""

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")

    def setup(self):
        self.add_input("fan_efficiency", val=0.9)
        self.add_output("transmitter_fan_efficiency", val=0.9)
        self.declare_partials(of="transmitter_fan_efficiency", wrt="fan_efficiency")

    def compute(self, inputs, outputs):
        aircraft_class = self.options["aircraft_class"].lower()

        if aircraft_class == "turbofan":
            outputs["transmitter_fan_efficiency"] = inputs["fan_efficiency"][0]
        elif aircraft_class in ("turboprop", "piston"):
            outputs["transmitter_fan_efficiency"] = 1.0
        else:
            raise ValueError(f"Invalid aircraft class: {self.options['aircraft_class']}")

    def compute_partials(self, inputs, partials):
        aircraft_class = self.options["aircraft_class"].lower()

        if aircraft_class == "turbofan":
            partials["transmitter_fan_efficiency", "fan_efficiency"] = 1.0
        elif aircraft_class in ("turboprop", "piston"):
            partials["transmitter_fan_efficiency", "fan_efficiency"] = 0.0
        else:
            raise ValueError(f"Invalid aircraft class: {self.options['aircraft_class']}")


def engine_lapse_value(sea_level_static, aircraft_class, density):
    """Return scalar FAST engine-lapse value."""

    aircraft_class = aircraft_class.lower()

    if aircraft_class == "turbofan":
        return sea_level_static * density / RHO_SL_STD

    if aircraft_class in ("turboprop", "piston"):
        return sea_level_static

    raise ValueError(f"Invalid aircraft class: {aircraft_class}")


def engine_lapse_derivatives(sea_level_static, aircraft_class, density):
    """Return scalar FAST engine-lapse derivatives."""

    aircraft_class = aircraft_class.lower()

    if aircraft_class == "turbofan":
        return {
            "doutput_dsea_level_static": density / RHO_SL_STD,
            "doutput_ddensity": sea_level_static / RHO_SL_STD,
        }

    if aircraft_class in ("turboprop", "piston"):
        return {
            "doutput_dsea_level_static": 1.0,
            "doutput_ddensity": 0.0,
        }

    raise ValueError(f"Invalid aircraft class: {aircraft_class}")


def safe_component_weight_value(power, power_to_weight):
    """Return FAST safe component weight scalar value."""

    if abs(power) < 1.0e-12:
        return 0.0

    return power / power_to_weight
