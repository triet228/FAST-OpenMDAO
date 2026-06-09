# src/fast_openmdao/engine.py

"""OpenMDAO components for FAST engine primitive equations."""

import math

import openmdao.api as om


GAS_CONSTANT_AIR = 287.0


class TotalPressure(om.ExplicitComponent):
    """Compute stagnation pressure from static pressure and Mach number."""

    def setup(self):
        self.add_input("static_pressure", val=101300.0, units="Pa")
        self.add_input("mach", val=0.3)
        self.add_input("gamma", val=1.4)
        self.add_output("total_pressure", val=107853.0, units="Pa")
        self.declare_partials(of="total_pressure", wrt="*")

    def compute(self, inputs, outputs):
        values = pressure_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        outputs["total_pressure"] = inputs["static_pressure"][0] * values["ratio"]

    def compute_partials(self, inputs, partials):
        static_pressure = inputs["static_pressure"][0]
        values = pressure_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        partials["total_pressure", "static_pressure"] = values["ratio"]
        partials["total_pressure", "mach"] = static_pressure * values["dratio_dmach"]
        partials["total_pressure", "gamma"] = static_pressure * values["dratio_dgamma"]


class StaticPressure(om.ExplicitComponent):
    """Compute static pressure from stagnation pressure and Mach number."""

    def setup(self):
        self.add_input("total_pressure", val=107853.0, units="Pa")
        self.add_input("mach", val=0.3)
        self.add_input("gamma", val=1.4)
        self.add_output("static_pressure", val=101300.0, units="Pa")
        self.declare_partials(of="static_pressure", wrt="*")

    def compute(self, inputs, outputs):
        values = pressure_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        outputs["static_pressure"] = inputs["total_pressure"][0] / values["ratio"]

    def compute_partials(self, inputs, partials):
        total_pressure = inputs["total_pressure"][0]
        values = pressure_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        inverse_ratio = 1.0 / values["ratio"]
        partials["static_pressure", "total_pressure"] = inverse_ratio
        partials["static_pressure", "mach"] = (
            -total_pressure * values["dratio_dmach"] * inverse_ratio ** 2
        )
        partials["static_pressure", "gamma"] = (
            -total_pressure * values["dratio_dgamma"] * inverse_ratio ** 2
        )


class TotalTemperature(om.ExplicitComponent):
    """Compute stagnation temperature from static temperature and Mach number."""

    def setup(self):
        self.add_input("static_temperature", val=288.15, units="K")
        self.add_input("mach", val=0.3)
        self.add_input("gamma", val=1.4)
        self.add_output("total_temperature", val=293.34, units="K")
        self.declare_partials(of="total_temperature", wrt="*")

    def compute(self, inputs, outputs):
        q = isentropic_q(inputs["mach"][0], inputs["gamma"][0])
        outputs["total_temperature"] = inputs["static_temperature"][0] * q

    def compute_partials(self, inputs, partials):
        static_temperature = inputs["static_temperature"][0]
        mach = inputs["mach"][0]
        gamma = inputs["gamma"][0]
        partials["total_temperature", "static_temperature"] = isentropic_q(
            mach,
            gamma,
        )
        partials["total_temperature", "mach"] = static_temperature * (
            gamma - 1.0
        ) * mach
        partials["total_temperature", "gamma"] = (
            0.5 * static_temperature * mach ** 2
        )


class StaticTemperature(om.ExplicitComponent):
    """Compute static temperature from stagnation temperature and Mach number."""

    def setup(self):
        self.add_input("total_temperature", val=293.34, units="K")
        self.add_input("mach", val=0.3)
        self.add_input("gamma", val=1.4)
        self.add_output("static_temperature", val=288.15, units="K")
        self.declare_partials(of="static_temperature", wrt="*")

    def compute(self, inputs, outputs):
        q = isentropic_q(inputs["mach"][0], inputs["gamma"][0])
        outputs["static_temperature"] = inputs["total_temperature"][0] / q

    def compute_partials(self, inputs, partials):
        total_temperature = inputs["total_temperature"][0]
        mach = inputs["mach"][0]
        gamma = inputs["gamma"][0]
        q = isentropic_q(mach, gamma)
        partials["static_temperature", "total_temperature"] = 1.0 / q
        partials["static_temperature", "mach"] = (
            -total_temperature * (gamma - 1.0) * mach / q ** 2
        )
        partials["static_temperature", "gamma"] = (
            -0.5 * total_temperature * mach ** 2 / q ** 2
        )


class ChokedArea(om.ExplicitComponent):
    """Compute choked area from flow area, Mach number, and gamma."""

    def setup(self):
        self.add_input("area", val=1.0, units="m**2")
        self.add_input("mach", val=0.5)
        self.add_input("gamma", val=1.4)
        self.add_output("area_star", val=0.75, units="m**2")
        self.declare_partials(of="area_star", wrt="*")

    def compute(self, inputs, outputs):
        values = area_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        outputs["area_star"] = inputs["area"][0] / values["ratio"]

    def compute_partials(self, inputs, partials):
        area = inputs["area"][0]
        values = area_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        inverse_ratio = 1.0 / values["ratio"]
        partials["area_star", "area"] = inverse_ratio
        partials["area_star", "mach"] = (
            -area * values["dratio_dmach"] * inverse_ratio ** 2
        )
        partials["area_star", "gamma"] = (
            -area * values["dratio_dgamma"] * inverse_ratio ** 2
        )


class FlowArea(om.ExplicitComponent):
    """Compute flow area from choked area, Mach number, and gamma."""

    def setup(self):
        self.add_input("area_star", val=0.75, units="m**2")
        self.add_input("mach", val=0.5)
        self.add_input("gamma", val=1.4)
        self.add_output("area", val=1.0, units="m**2")
        self.declare_partials(of="area", wrt="*")

    def compute(self, inputs, outputs):
        values = area_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        outputs["area"] = inputs["area_star"][0] * values["ratio"]

    def compute_partials(self, inputs, partials):
        area_star = inputs["area_star"][0]
        values = area_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        partials["area", "area_star"] = values["ratio"]
        partials["area", "mach"] = area_star * values["dratio_dmach"]
        partials["area", "gamma"] = area_star * values["dratio_dgamma"]


class MassFlowParameter(om.ExplicitComponent):
    """Compute FAST nondimensional mass-flow parameter."""

    def setup(self):
        self.add_input("mach", val=0.5)
        self.add_input("gamma", val=1.4)
        self.add_output("mass_flow_parameter", val=0.03)
        self.declare_partials(of="mass_flow_parameter", wrt="*")

    def compute(self, inputs, outputs):
        values = mass_flow_parameter_values(inputs["mach"][0], inputs["gamma"][0])
        outputs["mass_flow_parameter"] = values["value"]

    def compute_partials(self, inputs, partials):
        values = mass_flow_parameter_values(inputs["mach"][0], inputs["gamma"][0])
        partials["mass_flow_parameter", "mach"] = values["dvalue_dmach"]
        partials["mass_flow_parameter", "gamma"] = values["dvalue_dgamma"]


class StaticDensity(om.ExplicitComponent):
    """Compute static density from stagnation density and Mach number."""

    def setup(self):
        self.add_input("total_density", val=1.2, units="kg/m**3")
        self.add_input("mach", val=0.3)
        self.add_input("gamma", val=1.4)
        self.add_output("static_density", val=1.1, units="kg/m**3")
        self.declare_partials(of="static_density", wrt="*")

    def compute(self, inputs, outputs):
        values = density_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        outputs["static_density"] = inputs["total_density"][0] * values["ratio"]

    def compute_partials(self, inputs, partials):
        total_density = inputs["total_density"][0]
        values = density_ratio_values(inputs["mach"][0], inputs["gamma"][0])
        partials["static_density", "total_density"] = values["ratio"]
        partials["static_density", "mach"] = total_density * values["dratio_dmach"]
        partials["static_density", "gamma"] = total_density * values["dratio_dgamma"]


def isentropic_q(mach, gamma):
    """Return FAST isentropic ``1 + (gamma - 1) / 2 * mach ** 2`` term."""

    return 1.0 + 0.5 * (gamma - 1.0) * mach ** 2


def pressure_ratio_values(mach, gamma):
    """Return pressure ratio and derivatives for isentropic pressure equations."""

    q = isentropic_q(mach, gamma)
    exponent = gamma / (gamma - 1.0)
    ratio = q ** exponent
    dexponent_dgamma = -1.0 / (gamma - 1.0) ** 2
    dlogratio_dmach = exponent * (gamma - 1.0) * mach / q
    dlogratio_dgamma = (
        dexponent_dgamma * math.log(q)
        + exponent * 0.5 * mach ** 2 / q
    )
    return {
        "ratio": ratio,
        "dratio_dmach": ratio * dlogratio_dmach,
        "dratio_dgamma": ratio * dlogratio_dgamma,
    }


def area_ratio_values(mach, gamma):
    """Return area ratio and derivatives for FAST area-Mach equations."""

    q = isentropic_q(mach, gamma)
    base = 0.5 * (gamma + 1.0)
    exponent = 0.5 * (gamma + 1.0) / (gamma - 1.0)
    ratio = base ** (-exponent) * q ** exponent / mach
    dexponent_dgamma = -1.0 / (gamma - 1.0) ** 2
    dlogratio_dmach = exponent * (gamma - 1.0) * mach / q - 1.0 / mach
    dlogratio_dgamma = (
        dexponent_dgamma * (math.log(q) - math.log(base))
        + exponent * (0.5 * mach ** 2 / q - 0.5 / base)
    )
    return {
        "ratio": ratio,
        "dratio_dmach": ratio * dlogratio_dmach,
        "dratio_dgamma": ratio * dlogratio_dgamma,
    }


def mass_flow_parameter_values(mach, gamma):
    """Return FAST mass-flow parameter and derivatives."""

    q = isentropic_q(mach, gamma)
    exponent = 0.5 * (gamma + 1.0) / (gamma - 1.0)
    value = (gamma / GAS_CONSTANT_AIR) ** 0.5 * mach * q ** exponent
    dexponent_dgamma = -1.0 / (gamma - 1.0) ** 2
    dlogvalue_dmach = 1.0 / mach + exponent * (gamma - 1.0) * mach / q
    dlogvalue_dgamma = (
        0.5 / gamma
        + dexponent_dgamma * math.log(q)
        + exponent * 0.5 * mach ** 2 / q
    )
    return {
        "value": value,
        "dvalue_dmach": value * dlogvalue_dmach,
        "dvalue_dgamma": value * dlogvalue_dgamma,
    }


def density_ratio_values(mach, gamma):
    """Return static-to-total density ratio and derivatives."""

    q = isentropic_q(mach, gamma)
    exponent = -1.0 / (gamma - 1.0)
    ratio = q ** exponent
    dexponent_dgamma = 1.0 / (gamma - 1.0) ** 2
    dlogratio_dmach = exponent * (gamma - 1.0) * mach / q
    dlogratio_dgamma = (
        dexponent_dgamma * math.log(q)
        + exponent * 0.5 * mach ** 2 / q
    )
    return {
        "ratio": ratio,
        "dratio_dmach": ratio * dlogratio_dmach,
        "dratio_dgamma": ratio * dlogratio_dgamma,
    }
