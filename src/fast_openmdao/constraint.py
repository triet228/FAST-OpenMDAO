# src/fast_openmdao/constraint.py

"""OpenMDAO components for FAST constraint primitive equations."""

import math

import openmdao.api as om

from fast_openmdao.atmosphere import atmosphere_layer
from fast_python.atmosphere import standard_atmosphere
from fast_python.units import convert_length, convert_mass


RHO_SL_STD = standard_atmosphere(0.0)[2]
KG_TO_SLUG = convert_mass(1.0, "kg", "slug")
M_TO_FT = convert_length(1.0, "m", "ft")
RHO_SI_TO_ENGLISH = KG_TO_SLUG / M_TO_FT ** 3


class PsLossSigmoid(om.ExplicitComponent):
    """Evaluate FAST PsLoss sigmoid climb-gradient helper."""

    def initialize(self):
        self.options.declare("a", default=10.0)
        self.options.declare("b", default=2.0)
        self.options.declare("c", default=0.2)
        self.options.declare("d", default=1.0)

    def setup(self):
        self.add_input("ps_loss", val=0.1)
        self.add_output("sigmoid", val=0.01)
        self.declare_partials(of="sigmoid", wrt="ps_loss")

    def compute(self, inputs, outputs):
        outputs["sigmoid"] = ps_loss_sigmoid_value(
            inputs["ps_loss"][0],
            self.options["a"],
            self.options["b"],
            self.options["c"],
            self.options["d"],
        )

    def compute_partials(self, inputs, partials):
        partials["sigmoid", "ps_loss"] = ps_loss_sigmoid_derivative(
            inputs["ps_loss"][0],
            self.options["a"],
            self.options["b"],
            self.options["c"],
        )


class OEIMultiplier(om.ExplicitComponent):
    """Compute FAST engine-inoperative multiplier."""

    def initialize(self):
        self.options.declare("constraint_type", default=0)

    def setup(self):
        self.add_input("num_engines", val=2.0)
        self.add_input("ps_loss", val=0.2)
        self.add_output("oei_multiplier", val=2.0)
        self.declare_partials(of="oei_multiplier", wrt="*")

    def compute(self, inputs, outputs):
        constraint_type = self.options["constraint_type"]

        if constraint_type == 0:
            num_engines = inputs["num_engines"][0]
            outputs["oei_multiplier"] = num_engines / (num_engines - 1.0)
        elif constraint_type == 1:
            ps_loss = inputs["ps_loss"][0]
            outputs["oei_multiplier"] = 1.045 * ps_loss ** 2 + 1.0
        else:
            raise ValueError("OEIMultiplier Type must be 0 or 1.")

    def compute_partials(self, inputs, partials):
        constraint_type = self.options["constraint_type"]
        partials["oei_multiplier", "num_engines"] = 0.0
        partials["oei_multiplier", "ps_loss"] = 0.0

        if constraint_type == 0:
            num_engines = inputs["num_engines"][0]
            partials["oei_multiplier", "num_engines"] = (
                -1.0 / (num_engines - 1.0) ** 2
            )
        elif constraint_type == 1:
            partials["oei_multiplier", "ps_loss"] = 2.09 * inputs["ps_loss"][0]
        else:
            raise ValueError("OEIMultiplier Type must be 0 or 1.")


class FAR25EngineGradient(om.ExplicitComponent):
    """Select FAR 25 climb gradient by discrete engine count."""

    def initialize(self):
        self.options.declare("num_engines", default=2)
        self.options.declare("constraint_type", default=0)

    def setup(self):
        self.add_input("two_engine", val=0.024)
        self.add_input("three_engine", val=0.027)
        self.add_input("four_engine", val=0.030)
        self.add_output("engine_gradient", val=0.024)
        self.declare_partials(of="engine_gradient", wrt="*")

    def compute(self, inputs, outputs):
        outputs["engine_gradient"] = selected_engine_gradient(
            self.options["constraint_type"],
            self.options["num_engines"],
            inputs["two_engine"][0],
            inputs["three_engine"][0],
            inputs["four_engine"][0],
        )

    def compute_partials(self, inputs, partials):
        num_engines = self.options["num_engines"]
        constraint_type = self.options["constraint_type"]

        if constraint_type != 0:
            raise ValueError("FAR 25 constraint Type must be 0 for gradients.")

        partials["engine_gradient", "two_engine"] = 1.0 if num_engines == 2 else 0.0
        partials["engine_gradient", "three_engine"] = 1.0 if num_engines == 3 else 0.0
        partials["engine_gradient", "four_engine"] = 0.0 if num_engines in (2, 3) else 1.0


class CruiseDynamicPressure(om.ExplicitComponent):
    """Compute FAST English-unit cruise dynamic pressure quantities."""

    def setup(self):
        self.add_input("altitude", val=10000.0, units="m")
        self.add_input("mach", val=0.5)
        self.add_output("dynamic_pressure", val=100.0)
        self.add_output("density_ratio", val=0.5)
        self.add_output("velocity", val=500.0, units="ft/s")
        self.declare_partials(of="*", wrt=["altitude", "mach"])

    def compute(self, inputs, outputs):
        values = cruise_dynamic_pressure_values(inputs["altitude"][0], inputs["mach"][0])
        outputs["dynamic_pressure"] = values["dynamic_pressure"]
        outputs["density_ratio"] = values["density_ratio"]
        outputs["velocity"] = values["velocity"]

    def compute_partials(self, inputs, partials):
        values = cruise_dynamic_pressure_values(inputs["altitude"][0], inputs["mach"][0])

        for output_name in ("dynamic_pressure", "density_ratio", "velocity"):
            for input_name in ("altitude", "mach"):
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


def ps_loss_sigmoid_value(ps_loss, a, b, c, d):
    """Return FAST PsLoss sigmoid value."""

    return a / (1.0 + math.exp(-b * (ps_loss - c))) / 100.0 + d / 100.0


def ps_loss_sigmoid_derivative(ps_loss, a, b, c):
    """Return derivative of FAST PsLoss sigmoid with respect to PsLoss."""

    exponential = math.exp(-b * (ps_loss - c))
    return a * b * exponential / (1.0 + exponential) ** 2 / 100.0


def selected_engine_gradient(constraint_type, num_engines, two_engine, three_engine, four_engine):
    """Return FAR 25 climb gradient for a discrete engine count."""

    if constraint_type != 0:
        raise ValueError("FAR 25 constraint Type must be 0 for gradients.")

    if num_engines == 2:
        return two_engine

    if num_engines == 3:
        return three_engine

    return four_engine


def cruise_dynamic_pressure_values(altitude, mach):
    """Return cruise dynamic-pressure values and derivatives."""

    atmosphere = atmosphere_layer(altitude)
    temperature_rankine = 1.8 * atmosphere["temperature"]
    density = atmosphere["density"]
    density_slug = density * RHO_SI_TO_ENGLISH
    sound_speed = (1.4 * 1716.0 * temperature_rankine) ** 0.5
    velocity = sound_speed * mach
    dynamic_pressure = 0.5 * density_slug * velocity ** 2
    density_ratio = density / RHO_SL_STD
    dtemperature_rankine_daltitude = 1.8 * atmosphere["dtemperature_daltitude"]
    ddensity_daltitude = atmosphere["ddensity_daltitude"]
    ddensity_slug_daltitude = ddensity_daltitude * RHO_SI_TO_ENGLISH
    dsound_speed_daltitude = (
        0.5 * sound_speed / temperature_rankine * dtemperature_rankine_daltitude
    )
    dvelocity_daltitude = mach * dsound_speed_daltitude
    dvelocity_dmach = sound_speed
    ddynamic_pressure_daltitude = 0.5 * (
        ddensity_slug_daltitude * velocity ** 2
        + 2.0 * density_slug * velocity * dvelocity_daltitude
    )
    ddynamic_pressure_dmach = density_slug * velocity * dvelocity_dmach

    return {
        "dynamic_pressure": dynamic_pressure,
        "density_ratio": density_ratio,
        "velocity": velocity,
        "ddynamic_pressure_daltitude": ddynamic_pressure_daltitude,
        "ddynamic_pressure_dmach": ddynamic_pressure_dmach,
        "ddensity_ratio_daltitude": ddensity_daltitude / RHO_SL_STD,
        "ddensity_ratio_dmach": 0.0,
        "dvelocity_daltitude": dvelocity_daltitude,
        "dvelocity_dmach": dvelocity_dmach,
    }
