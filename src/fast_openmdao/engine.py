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


class AirSpecificHeat(om.ExplicitComponent):
    """Compute FAST fitted air Cp at one temperature."""

    def setup(self):
        self.add_input("temperature", val=300.0, units="K")
        self.add_output("cp_air", val=1007.0)
        self.declare_partials(of="cp_air", wrt="temperature")

    def compute(self, inputs, outputs):
        outputs["cp_air"] = sigmoid_heat_value(
            inputs["temperature"][0],
            233.0,
            1.0 / 210.0,
            875.0,
            993.0,
        )

    def compute_partials(self, inputs, partials):
        partials["cp_air", "temperature"] = sigmoid_heat_derivative(
            inputs["temperature"][0],
            233.0,
            1.0 / 210.0,
            875.0,
        )


class AirSpecificHeatVolume(om.ExplicitComponent):
    """Compute FAST fitted air Cv at one temperature."""

    def setup(self):
        self.add_input("temperature", val=300.0, units="K")
        self.add_output("cv_air", val=720.0)
        self.declare_partials(of="cv_air", wrt="temperature")

    def compute(self, inputs, outputs):
        outputs["cv_air"] = sigmoid_heat_value(
            inputs["temperature"][0],
            233.0,
            1.0 / 210.0,
            875.0,
            993.0 - 287.0,
        )

    def compute_partials(self, inputs, partials):
        partials["cv_air", "temperature"] = sigmoid_heat_derivative(
            inputs["temperature"][0],
            233.0,
            1.0 / 210.0,
            875.0,
        )


class ThermalPerfectGamma(om.ExplicitComponent):
    """Iterate FAST thermally perfect static temperature and gamma update."""

    def setup(self):
        self.add_input("total_temperature", val=800.0, units="K")
        self.add_input("mach", val=0.5)
        self.add_input("gamma", val=1.4)
        self.add_output("static_temperature", val=760.0, units="K")
        self.add_output("cp_air", val=1100.0)
        self.add_output("cv_air", val=813.0)
        self.add_output("updated_gamma", val=1.35)
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = thermal_perfect_gamma_values(
            inputs["total_temperature"][0],
            inputs["mach"][0],
            inputs["gamma"][0],
        )
        outputs["static_temperature"] = values["static_temperature"]
        outputs["cp_air"] = values["cp_air"]
        outputs["cv_air"] = values["cv_air"]
        outputs["updated_gamma"] = values["updated_gamma"]

    def compute_partials(self, inputs, partials):
        values = thermal_perfect_gamma_values(
            inputs["total_temperature"][0],
            inputs["mach"][0],
            inputs["gamma"][0],
        )

        for output in ("static_temperature", "cp_air", "cv_air", "updated_gamma"):
            for variable in ("total_temperature", "mach", "gamma"):
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


class AirIntegratedHeat(om.ExplicitComponent):
    """Compute FAST integrated air specific heat between two temperatures."""

    def setup(self):
        self.add_input("temperature_low", val=300.0, units="K")
        self.add_input("temperature_high", val=1200.0, units="K")
        self.add_output("integrated_cp_air", val=1.0e6)
        self.declare_partials(of="integrated_cp_air", wrt="*")

    def compute(self, inputs, outputs):
        outputs["integrated_cp_air"] = integrated_heat_value(
            inputs["temperature_low"][0],
            inputs["temperature_high"][0],
            233.0,
            1.0 / 210.0,
            875.0,
            993.0,
        )

    def compute_partials(self, inputs, partials):
        partials["integrated_cp_air", "temperature_low"] = -sigmoid_heat_value(
            inputs["temperature_low"][0],
            233.0,
            1.0 / 210.0,
            875.0,
            993.0,
        )
        partials["integrated_cp_air", "temperature_high"] = sigmoid_heat_value(
            inputs["temperature_high"][0],
            233.0,
            1.0 / 210.0,
            875.0,
            993.0,
        )


class JetAIntegratedHeat(om.ExplicitComponent):
    """Compute FAST integrated Jet-A specific heat between two temperatures."""

    def setup(self):
        self.add_input("temperature_low", val=300.0, units="K")
        self.add_input("temperature_high", val=1200.0, units="K")
        self.add_output("integrated_cp_jeta", val=2.7e6)
        self.declare_partials(of="integrated_cp_jeta", wrt="*")

    def compute(self, inputs, outputs):
        outputs["integrated_cp_jeta"] = integrated_heat_value(
            inputs["temperature_low"][0],
            inputs["temperature_high"][0],
            4600.0,
            1.0 / 410.0,
            500.0,
            100.0,
        )

    def compute_partials(self, inputs, partials):
        partials["integrated_cp_jeta", "temperature_low"] = -sigmoid_heat_value(
            inputs["temperature_low"][0],
            4600.0,
            1.0 / 410.0,
            500.0,
            100.0,
        )
        partials["integrated_cp_jeta", "temperature_high"] = sigmoid_heat_value(
            inputs["temperature_high"][0],
            4600.0,
            1.0 / 410.0,
            500.0,
            100.0,
        )


class LocalEfficiency(om.ExplicitComponent):
    """Compute FAST's local engine efficiency fit from Reynolds number."""

    def setup(self):
        self.add_input("reynolds", val=1.0e7)
        self.add_output("local_efficiency", val=0.875)
        self.declare_partials(of="local_efficiency", wrt="reynolds")

    def compute(self, inputs, outputs):
        outputs["local_efficiency"] = local_efficiency_value(inputs["reynolds"][0])

    def compute_partials(self, inputs, partials):
        partials["local_efficiency", "reynolds"] = local_efficiency_derivative(
            inputs["reynolds"][0],
        )


class LocalReynolds(om.ExplicitComponent):
    """Compute FAST's local engine Reynolds number from flow-state fields."""

    def setup(self):
        self.add_input("static_pressure", val=101300.0, units="Pa")
        self.add_input("static_temperature", val=288.15, units="K")
        self.add_input("outer_radius", val=1.0, units="m")
        self.add_input("inner_radius", val=0.5, units="m")
        self.add_input("mach", val=0.3)
        self.add_input("gamma", val=1.4)
        self.add_output("local_reynolds", val=1.0e7)
        self.declare_partials(of="local_reynolds", wrt="*")

    def compute(self, inputs, outputs):
        outputs["local_reynolds"] = local_reynolds_value(
            inputs["static_pressure"][0],
            inputs["static_temperature"][0],
            inputs["outer_radius"][0],
            inputs["inner_radius"][0],
            inputs["mach"][0],
            inputs["gamma"][0],
        )

    def compute_partials(self, inputs, partials):
        values = local_reynolds_values(
            inputs["static_pressure"][0],
            inputs["static_temperature"][0],
            inputs["outer_radius"][0],
            inputs["inner_radius"][0],
            inputs["mach"][0],
            inputs["gamma"][0],
        )
        partials["local_reynolds", "static_pressure"] = values[
            "dreynolds_dstatic_pressure"
        ]
        partials["local_reynolds", "static_temperature"] = values[
            "dreynolds_dstatic_temperature"
        ]
        partials["local_reynolds", "outer_radius"] = values[
            "dreynolds_douter_radius"
        ]
        partials["local_reynolds", "inner_radius"] = values[
            "dreynolds_dinner_radius"
        ]
        partials["local_reynolds", "mach"] = values["dreynolds_dmach"]
        partials["local_reynolds", "gamma"] = values["dreynolds_dgamma"]


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


def thermal_perfect_gamma_values(total_temperature, mach, gamma):
    """Return FAST new_gamma outputs and loop-propagated derivatives."""

    gamma_new = gamma
    gamma_derivatives = {
        "total_temperature": 0.0,
        "mach": 0.0,
        "gamma": 1.0,
    }
    delta_gamma = 1.0
    iteration = 0
    static_temperature = total_temperature / isentropic_q(mach, gamma_new)
    cp = None
    cv = None
    output_derivatives = None

    while delta_gamma > 1.0e-3 and iteration < 10:
        q = isentropic_q(mach, gamma_new)
        dq_dmach = (gamma_new - 1.0) * mach
        dq_dgamma = 0.5 * mach ** 2
        static_temperature = total_temperature / q
        static_derivatives = {}

        for variable in ("total_temperature", "mach", "gamma"):
            dtotal = 1.0 if variable == "total_temperature" else 0.0
            dmach = 1.0 if variable == "mach" else 0.0
            dgamma = gamma_derivatives[variable]
            dq = dq_dmach * dmach + dq_dgamma * dgamma
            static_derivatives[variable] = dtotal / q - total_temperature * dq / q ** 2

        cp = sigmoid_heat_value(static_temperature, 233.0, 1.0 / 210.0, 875.0, 993.0)
        cv = sigmoid_heat_value(
            static_temperature,
            233.0,
            1.0 / 210.0,
            875.0,
            993.0 - GAS_CONSTANT_AIR,
        )
        dcp_dtemperature = sigmoid_heat_derivative(
            static_temperature,
            233.0,
            1.0 / 210.0,
            875.0,
        )
        dcv_dtemperature = dcp_dtemperature
        gamma_next = cp / cv
        next_gamma_derivatives = {}

        for variable in ("total_temperature", "mach", "gamma"):
            dstatic = static_derivatives[variable]
            dcp = dcp_dtemperature * dstatic
            dcv = dcv_dtemperature * dstatic
            next_gamma_derivatives[variable] = (dcp * cv - cp * dcv) / cv ** 2

        delta_gamma = abs(gamma_next - gamma_new) / gamma_new
        gamma_new = gamma_next
        gamma_derivatives = next_gamma_derivatives
        output_derivatives = {
            "static_temperature": static_derivatives,
            "cp_air": {
                variable: dcp_dtemperature * static_derivatives[variable]
                for variable in ("total_temperature", "mach", "gamma")
            },
            "cv_air": {
                variable: dcv_dtemperature * static_derivatives[variable]
                for variable in ("total_temperature", "mach", "gamma")
            },
            "updated_gamma": gamma_derivatives,
        }
        iteration += 1

    values = {
        "static_temperature": static_temperature,
        "cp_air": cp,
        "cv_air": cv,
        "updated_gamma": gamma_new,
    }

    for output, output_partials in output_derivatives.items():
        for variable, derivative in output_partials.items():
            values["d%s_d%s" % (output, variable)] = derivative

    return values


def sigmoid_heat_value(temperature, length, rate, midpoint, offset):
    """Return FAST fitted S-curve heat capacity."""

    return length / (1.0 + math.exp(-rate * (temperature - midpoint))) + offset


def sigmoid_heat_derivative(temperature, length, rate, midpoint):
    """Return temperature derivative of FAST fitted heat capacity."""

    exponential = math.exp(-rate * (temperature - midpoint))
    return length * rate * exponential / (1.0 + exponential) ** 2


def integrated_heat_value(t_low, t_high, length, rate, midpoint, offset):
    """Return FAST integrated heat from the fitted antiderivative."""

    return (
        heat_antiderivative(t_high, length, rate, midpoint, offset)
        - heat_antiderivative(t_low, length, rate, midpoint, offset)
    )


def heat_antiderivative(temperature, length, rate, midpoint, offset):
    """Return FAST fitted heat-capacity antiderivative."""

    return temperature * (offset + length) + length * math.log1p(
        math.exp(rate * (midpoint - temperature))
    ) / rate


def local_efficiency_value(reynolds):
    """Return FAST's local efficiency fit."""

    high = 1.0
    low = 0.75
    growth_rate = 0.5
    inflection = 7.0
    return low + (high - low) / (
        1.0 + math.exp(-growth_rate * (math.log10(reynolds) - inflection))
    )


def local_efficiency_derivative(reynolds):
    """Return derivative of FAST's local efficiency fit."""

    high = 1.0
    low = 0.75
    growth_rate = 0.5
    inflection = 7.0
    exponential = math.exp(-growth_rate * (math.log10(reynolds) - inflection))
    return (
        (high - low)
        * growth_rate
        * exponential
        / ((1.0 + exponential) ** 2 * reynolds * math.log(10.0))
    )


def local_reynolds_value(
    static_pressure,
    static_temperature,
    outer_radius,
    inner_radius,
    mach,
    gamma,
):
    """Return FAST local Reynolds number from scalar flow-state fields."""

    return local_reynolds_values(
        static_pressure,
        static_temperature,
        outer_radius,
        inner_radius,
        mach,
        gamma,
    )["reynolds"]


def local_reynolds_values(
    static_pressure,
    static_temperature,
    outer_radius,
    inner_radius,
    mach,
    gamma,
):
    """Return FAST local Reynolds number and analytical derivatives."""

    length = outer_radius - inner_radius
    coefficient = local_reynolds_coefficient()
    reynolds = (
        static_pressure
        * mach
        * length
        * math.sqrt(gamma * GAS_CONSTANT_AIR)
        / (GAS_CONSTANT_AIR * coefficient * static_temperature)
    )

    return {
        "reynolds": reynolds,
        "dreynolds_dstatic_pressure": reynolds / static_pressure,
        "dreynolds_dstatic_temperature": -reynolds / static_temperature,
        "dreynolds_douter_radius": reynolds / length,
        "dreynolds_dinner_radius": -reynolds / length,
        "dreynolds_dmach": reynolds / mach,
        "dreynolds_dgamma": 0.5 * reynolds / gamma,
    }


def local_reynolds_coefficient():
    """Return the temperature-root viscosity coefficient from FAST."""

    sigma = 350e-12
    boltzmann = 1.380639e-23
    molecular_mass = 0.02869 / 6.022e23
    return (
        1.106
        * 5.0
        / 16.0
        / sigma ** 2
        * math.sqrt(boltzmann * molecular_mass / math.pi)
    )
