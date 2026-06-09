# src/fast_openmdao/projection.py

"""OpenMDAO components for FAST technology projection equations."""

import math

import openmdao.api as om

from fast_python.projection import kpp_projection


class KPPProjection(om.ExplicitComponent):
    """Project one FAST key performance parameter as an OpenMDAO component.

    Inputs:
        year: Calendar year for the projection curve.

    Outputs:
        value: Projected FAST key performance parameter.

    Assumptions:
        Aircraft class and KPP name are discrete model options. The component
        differentiates only the continuous year input.
    """

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")
        self.options.declare("kpp", default="Battery Specific Energy")
        self.options.declare("output_name", default="value")
        self.options.declare("output_units", default=None)

    def setup(self):
        output_name = self.options["output_name"]
        self.add_input("year", val=2030.0)
        self.add_output(output_name, val=1.0, units=self.options["output_units"])
        self.declare_partials(of=output_name, wrt="year")

    def compute(self, inputs, outputs):
        output_name = self.options["output_name"]
        outputs[output_name] = kpp_projection(
            self.options["aircraft_class"],
            inputs["year"][0],
            self.options["kpp"],
        )

    def compute_partials(self, inputs, partials):
        output_name = self.options["output_name"]
        partials[output_name, "year"] = kpp_projection_derivative(
            self.options["aircraft_class"],
            inputs["year"][0],
            self.options["kpp"],
        )


class BatterySpecificEnergyProjection(KPPProjection):
    """Project FAST battery specific energy in Wh/kg."""

    def initialize(self):
        super().initialize()
        self.options["kpp"] = "Battery Specific Energy"
        self.options["output_name"] = "battery_specific_energy"
        self.options["output_units"] = "W*h/kg"


class ElectricMotorSpecificPowerProjection(KPPProjection):
    """Project FAST electric motor specific power in kW/kg."""

    def initialize(self):
        super().initialize()
        self.options["kpp"] = "Electric Motor Specific Power"
        self.options["output_name"] = "electric_motor_specific_power"
        self.options["output_units"] = "kW/kg"


def kpp_projection_derivative(aircraft_class, year, kpp):
    """Return derivative of a FAST projection with respect to year."""

    aircraft_class = str(aircraft_class)
    kpp = str(kpp)
    year = float(year)

    if kpp == "Battery Specific Energy":
        return sigmoid_derivative(year, 0.0, 801.8, 0.0607, 2030.8)

    if kpp == "Electric Motor Specific Power":
        return sigmoid_derivative(year, 0.0, 37.8, 0.1213, 2030.0)

    if aircraft_class == "Turbofan":
        if kpp == "Cruise SFC":
            return sigmoid_derivative(year, 0.3335, 0.851, -0.0727, 1992.0)

        if kpp == "Total Takeoff T/ MTOW":
            return sigmoid_derivative(year, 0.15, 0.3465, 0.1132, 1980.0)

        if kpp in ("OEW/MTOW", "M(L/D)"):
            return 0.0

    if aircraft_class == "Turboprop":
        if kpp == "Cruise SFC":
            return sigmoid_derivative(year, 0.3335, 0.6, -0.1012, 1993.0)

        if kpp == "Total Takeoff T/ MTOW":
            return sigmoid_derivative(year, 0.3, 0.9, 0.5334, 2011.0)

        if kpp == "M(L/D)":
            return sigmoid_derivative(year, 1.0, 10.0, 0.1683, 2010.0)

        if kpp == "OEW/MTOW":
            return 0.0

    kpp_projection(aircraft_class, year, kpp)
    return 0.0


def sigmoid_derivative(year, low, high, growth_rate, inflection):
    """Return FAST logistic projection derivative with respect to year."""

    exponential = math.exp((-growth_rate) * (year - inflection))
    return (high - low) * growth_rate * exponential / (1.0 + exponential) ** 2
