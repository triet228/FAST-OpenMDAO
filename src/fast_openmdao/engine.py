# src/fast_openmdao/engine.py

"""OpenMDAO components for FAST engine primitive equations."""

import math

import openmdao.api as om

from fast_openmdao.atmosphere import atmosphere_layer


GAS_CONSTANT_AIR = 287.0
TSFC_SI_TO_IMPERIAL = 3600.0 / 0.224808943099711 * 2.204622621848776


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


class OffDesignNozzleMach(om.ExplicitComponent):
    """Solve FAST off-design nozzle exit Mach from inlet and exit areas."""

    def setup(self):
        self.add_input("area_1", val=1.0, units="m**2")
        self.add_input("area_2", val=1.1, units="m**2")
        self.add_input("mach_1", val=0.5)
        self.add_input("gamma", val=1.4)
        self.add_output("mach_2", val=0.4)
        self.declare_partials(of="mach_2", wrt="*")

    def compute(self, inputs, outputs):
        values = off_design_nozzle_values(
            inputs["area_1"][0],
            inputs["area_2"][0],
            inputs["mach_1"][0],
            inputs["gamma"][0],
        )
        outputs["mach_2"] = values["mach_2"]

    def compute_partials(self, inputs, partials):
        values = off_design_nozzle_values(
            inputs["area_1"][0],
            inputs["area_2"][0],
            inputs["mach_1"][0],
            inputs["gamma"][0],
        )
        partials["mach_2", "area_1"] = values["dmach_2_darea_1"]
        partials["mach_2", "area_2"] = values["dmach_2_darea_2"]
        partials["mach_2", "mach_1"] = values["dmach_2_dmach_1"]
        partials["mach_2", "gamma"] = values["dmach_2_dgamma"]


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


class BurnerFlow(om.ExplicitComponent):
    """Compute FAST on-design burner flow, fuel addition, and exit state."""

    def setup(self):
        self.add_input("mass_flow_31", val=50.0, units="kg/s")
        self.add_input("area_31", val=0.5, units="m**2")
        self.add_input("total_pressure_31", val=800000.0, units="Pa")
        self.add_input("total_temperature_31", val=750.0, units="K")
        self.add_input("mach_31", val=0.25)
        self.add_input("gamma_31", val=1.35)
        self.add_input("outer_radius_31", val=0.5, units="m")
        self.add_input("total_temperature_4", val=1400.0, units="K")
        self.add_input("fuel_lhv", val=43.0e6)
        self.add_input("combustor_efficiency", val=0.99)
        self.add_output("diffuser_mach_32", val=0.08)
        self.add_output("diffuser_area_32", val=0.6, units="m**2")
        self.add_output("diffuser_pressure_ratio", val=0.98)
        self.add_output("fuel_flow", val=1.0, units="kg/s")
        self.add_output("fuel_air_ratio", val=0.02)
        self.add_output("mass_flow_39", val=51.0, units="kg/s")
        self.add_output("total_pressure_39", val=740000.0, units="Pa")
        self.add_output("total_temperature_39", val=1400.0, units="K")
        self.add_output("static_temperature_39", val=1300.0, units="K")
        self.add_output("cp_air_39", val=1200.0)
        self.add_output("cv_air_39", val=913.0)
        self.add_output("gamma_39", val=1.31)
        self.add_output("static_pressure_39", val=730000.0, units="Pa")
        self.add_output("inner_radius_39", val=0.4, units="m")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = burner_flow_values(
            inputs["mass_flow_31"][0],
            inputs["area_31"][0],
            inputs["total_pressure_31"][0],
            inputs["total_temperature_31"][0],
            inputs["mach_31"][0],
            inputs["gamma_31"][0],
            inputs["outer_radius_31"][0],
            inputs["total_temperature_4"][0],
            inputs["fuel_lhv"][0],
            inputs["combustor_efficiency"][0],
        )

        for output in burner_flow_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = burner_flow_values(
            inputs["mass_flow_31"][0],
            inputs["area_31"][0],
            inputs["total_pressure_31"][0],
            inputs["total_temperature_31"][0],
            inputs["mach_31"][0],
            inputs["gamma_31"][0],
            inputs["outer_radius_31"][0],
            inputs["total_temperature_4"][0],
            inputs["fuel_lhv"][0],
            inputs["combustor_efficiency"][0],
        )

        for output in burner_flow_output_names():
            for variable in burner_flow_input_names():
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


class CompressorStageFlow(om.ExplicitComponent):
    """Compute one FAST on-design compressor or fan stage."""

    def setup(self):
        self.add_input("mass_flow_1", val=50.0, units="kg/s")
        self.add_input("area_1", val=0.5, units="m**2")
        self.add_input("total_pressure_1", val=200000.0, units="Pa")
        self.add_input("total_temperature_1", val=350.0, units="K")
        self.add_input("static_pressure_1", val=190000.0, units="Pa")
        self.add_input("static_temperature_1", val=345.0, units="K")
        self.add_input("mach_1", val=0.35)
        self.add_input("gamma_1", val=1.38)
        self.add_input("outer_radius_1", val=0.8, units="m")
        self.add_input("stage_pressure_ratio", val=1.2)
        self.add_input("rpm", val=6000.0, units="rpm")
        self.add_input("stage_efficiency", val=0.9)
        self.add_output("mass_flow_3", val=50.0, units="kg/s")
        self.add_output("total_pressure_3", val=240000.0, units="Pa")
        self.add_output("total_temperature_3", val=370.0, units="K")
        self.add_output("static_temperature_3", val=365.0, units="K")
        self.add_output("mach_3", val=0.3)
        self.add_output("cp_air_3", val=1010.0)
        self.add_output("cv_air_3", val=723.0)
        self.add_output("gamma_3", val=1.38)
        self.add_output("static_pressure_3", val=230000.0, units="Pa")
        self.add_output("area_3", val=0.48, units="m**2")
        self.add_output("outer_radius_3", val=0.8, units="m")
        self.add_output("inner_radius_3", val=0.65, units="m")
        self.add_output("pitch_radius_3", val=0.72, units="m")
        self.add_output("work", val=1.0e6, units="W")
        self.add_output("temperature_ratio", val=1.05)
        self.add_output("stage_eta", val=0.9)
        self.add_output("stage_psi", val=0.2)
        self.add_output("corrected_mass_flow", val=50.0)
        self.add_output("corrected_speed", val=6000.0)
        self.add_output("stage_phi", val=0.5)
        self.add_output("stage_zeta", val=0.3)
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = compressor_stage_flow_values(
            inputs["mass_flow_1"][0],
            inputs["area_1"][0],
            inputs["total_pressure_1"][0],
            inputs["total_temperature_1"][0],
            inputs["static_pressure_1"][0],
            inputs["static_temperature_1"][0],
            inputs["mach_1"][0],
            inputs["gamma_1"][0],
            inputs["outer_radius_1"][0],
            inputs["stage_pressure_ratio"][0],
            inputs["rpm"][0],
            inputs["stage_efficiency"][0],
        )

        for output in compressor_stage_flow_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = compressor_stage_flow_values(
            inputs["mass_flow_1"][0],
            inputs["area_1"][0],
            inputs["total_pressure_1"][0],
            inputs["total_temperature_1"][0],
            inputs["static_pressure_1"][0],
            inputs["static_temperature_1"][0],
            inputs["mach_1"][0],
            inputs["gamma_1"][0],
            inputs["outer_radius_1"][0],
            inputs["stage_pressure_ratio"][0],
            inputs["rpm"][0],
            inputs["stage_efficiency"][0],
        )

        for output in compressor_stage_flow_output_names():
            for variable in compressor_stage_flow_input_names():
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


class TurbineStageFlow(om.ExplicitComponent):
    """Compute one FAST on-design turbine stage."""

    def setup(self):
        self.add_input("mass_flow_1", val=50.0, units="kg/s")
        self.add_input("total_pressure_1", val=800000.0, units="Pa")
        self.add_input("total_temperature_1", val=1400.0, units="K")
        self.add_input("mach_1", val=0.35)
        self.add_input("gamma_1", val=1.32)
        self.add_input("outer_radius_1", val=0.8, units="m")
        self.add_input("inner_radius_1", val=0.45, units="m")
        self.add_input("target_total_temperature_3", val=1300.0, units="K")
        self.add_input("stage_mach_2", val=1.1)
        self.add_input("rpm", val=6200.0, units="rpm")
        self.add_input("turbine_efficiency", val=0.9)
        self.add_output("total_temperature_3", val=1300.0, units="K")
        self.add_output("total_pressure_3", val=600000.0, units="Pa")
        self.add_output("mach_3", val=0.8)
        self.add_output("static_temperature_3", val=1250.0, units="K")
        self.add_output("cp_air_3", val=1200.0)
        self.add_output("cv_air_3", val=913.0)
        self.add_output("gamma_3", val=1.31)
        self.add_output("static_pressure_3", val=500000.0, units="Pa")
        self.add_output("area_3", val=0.5, units="m**2")
        self.add_output("inner_radius_3", val=0.45, units="m")
        self.add_output("outer_radius_3", val=0.8, units="m")
        self.add_output("pressure_ratio", val=0.8)
        self.add_output("temperature_ratio", val=0.9)
        self.add_output("stage_psi", val=0.5)
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = turbine_stage_flow_values(
            inputs["mass_flow_1"][0],
            inputs["total_pressure_1"][0],
            inputs["total_temperature_1"][0],
            inputs["mach_1"][0],
            inputs["gamma_1"][0],
            inputs["outer_radius_1"][0],
            inputs["inner_radius_1"][0],
            inputs["target_total_temperature_3"][0],
            inputs["stage_mach_2"][0],
            inputs["rpm"][0],
            inputs["turbine_efficiency"][0],
        )

        for output in turbine_stage_flow_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = turbine_stage_flow_values(
            inputs["mass_flow_1"][0],
            inputs["total_pressure_1"][0],
            inputs["total_temperature_1"][0],
            inputs["mach_1"][0],
            inputs["gamma_1"][0],
            inputs["outer_radius_1"][0],
            inputs["inner_radius_1"][0],
            inputs["target_total_temperature_3"][0],
            inputs["stage_mach_2"][0],
            inputs["rpm"][0],
            inputs["turbine_efficiency"][0],
        )

        for output in turbine_stage_flow_output_names():
            for variable in turbine_stage_flow_input_names():
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


class PerfectExpansionNozzleFlow(om.ExplicitComponent):
    """Compute FAST perfect-expansion nozzle flow and thrust."""

    def setup(self):
        self.add_input("mass_flow_5", val=50.0, units="kg/s")
        self.add_input("total_pressure_5", val=300000.0, units="Pa")
        self.add_input("total_temperature_5", val=900.0, units="K")
        self.add_input("cp_air_5", val=1100.0)
        self.add_input("gamma_5", val=1.33)
        self.add_input("inner_radius_5", val=0.2, units="m")
        self.add_input("ambient_total_pressure", val=101325.0, units="Pa")
        self.add_input("ambient_mach", val=0.0)
        self.add_input("ambient_gamma", val=1.4)
        self.add_input("nozzle_pressure_ratio", val=1.3)
        self.add_input("nozzle_efficiency", val=0.95)
        self.add_output("total_temperature_9", val=900.0, units="K")
        self.add_output("static_temperature_9", val=800.0, units="K")
        self.add_output("total_pressure_9", val=150000.0, units="Pa")
        self.add_output("static_pressure_9", val=130000.0, units="Pa")
        self.add_output("mach_9", val=0.8)
        self.add_output("cp_air_9", val=1100.0)
        self.add_output("cv_air_9", val=813.0)
        self.add_output("gamma_9", val=1.35)
        self.add_output("area_9", val=0.4, units="m**2")
        self.add_output("core_outer_radius_9", val=0.35, units="m")
        self.add_output("bypass_outer_radius_9", val=0.3, units="m")
        self.add_output("exit_velocity", val=400.0, units="m/s")
        self.add_output("thrust", val=20000.0, units="N")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = perfect_expansion_nozzle_flow_values(
            inputs["mass_flow_5"][0],
            inputs["total_pressure_5"][0],
            inputs["total_temperature_5"][0],
            inputs["cp_air_5"][0],
            inputs["gamma_5"][0],
            inputs["inner_radius_5"][0],
            inputs["ambient_total_pressure"][0],
            inputs["ambient_mach"][0],
            inputs["ambient_gamma"][0],
            inputs["nozzle_pressure_ratio"][0],
            inputs["nozzle_efficiency"][0],
        )

        for output in perfect_expansion_nozzle_flow_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = perfect_expansion_nozzle_flow_values(
            inputs["mass_flow_5"][0],
            inputs["total_pressure_5"][0],
            inputs["total_temperature_5"][0],
            inputs["cp_air_5"][0],
            inputs["gamma_5"][0],
            inputs["inner_radius_5"][0],
            inputs["ambient_total_pressure"][0],
            inputs["ambient_mach"][0],
            inputs["ambient_gamma"][0],
            inputs["nozzle_pressure_ratio"][0],
            inputs["nozzle_efficiency"][0],
        )

        for output in perfect_expansion_nozzle_flow_output_names():
            for variable in perfect_expansion_nozzle_flow_input_names():
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


class AirTemperatureFromHeatAdded(om.ExplicitComponent):
    """Invert FAST integrated air Cp for heat added from a start temperature."""

    def setup(self):
        self.add_input("temperature_start", val=300.0, units="K")
        self.add_input("heat", val=100000.0)
        self.add_output("temperature_end", val=383.0, units="K")
        self.declare_partials(of="temperature_end", wrt="*")

    def compute(self, inputs, outputs):
        values = air_temperature_from_heat_added_values(
            inputs["temperature_start"][0],
            inputs["heat"][0],
        )
        outputs["temperature_end"] = values["temperature_end"]

    def compute_partials(self, inputs, partials):
        values = air_temperature_from_heat_added_values(
            inputs["temperature_start"][0],
            inputs["heat"][0],
        )
        partials["temperature_end", "temperature_start"] = values[
            "dtemperature_end_dtemperature_start"
        ]
        partials["temperature_end", "heat"] = values["dtemperature_end_dheat"]


class AirTemperatureFromHeatRemoved(om.ExplicitComponent):
    """Invert FAST integrated air Cp for heat removed from a start temperature."""

    def setup(self):
        self.add_input("temperature_start", val=1200.0, units="K")
        self.add_input("heat", val=100000.0)
        self.add_output("temperature_end", val=1115.0, units="K")
        self.declare_partials(of="temperature_end", wrt="*")

    def compute(self, inputs, outputs):
        values = air_temperature_from_heat_removed_values(
            inputs["temperature_start"][0],
            inputs["heat"][0],
        )
        outputs["temperature_end"] = values["temperature_end"]

    def compute_partials(self, inputs, partials):
        values = air_temperature_from_heat_removed_values(
            inputs["temperature_start"][0],
            inputs["heat"][0],
        )
        partials["temperature_end", "temperature_start"] = values[
            "dtemperature_end_dtemperature_start"
        ]
        partials["temperature_end", "heat"] = values["dtemperature_end_dheat"]


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


class SimpleOffDesignTurbofan(om.ExplicitComponent):
    """Evaluate FAST's BADA-style simple off-design turbofan fuel model."""

    def setup(self):
        self.add_input("altitude", val=1000.0, units="m")
        self.add_input("mach", val=0.2)
        self.add_input("required_thrust", val=10000.0, units="N")
        self.add_input("electric_load", val=1000.0, units="W")
        self.add_input("thrust_available", val=20000.0, units="N")
        self.add_input("sea_level_static_thrust", val=20000.0, units="N")
        self.add_input("thrust_supplement", val=0.0, units="N")
        self.add_input("fuel_coeff_3", val=0.0)
        self.add_input("fuel_coeff_2", val=0.0)
        self.add_input("fuel_coeff_1", val=2.0)
        self.add_input("fuel_coeff_altitude", val=0.0)
        self.add_input("he_coefficient", val=1.0)
        self.add_output("fuel_flow", val=1.0)
        self.add_output("thrust", val=10000.0, units="N")
        self.add_output("tsfc", val=1.0e-5)
        self.add_output("tsfc_imperial", val=0.1)
        self.add_output("he_coeff", val=1.0)
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = simple_off_design_turbofan_values(
            inputs["altitude"][0],
            inputs["mach"][0],
            inputs["required_thrust"][0],
            inputs["electric_load"][0],
            inputs["thrust_available"][0],
            inputs["sea_level_static_thrust"][0],
            inputs["thrust_supplement"][0],
            inputs["fuel_coeff_3"][0],
            inputs["fuel_coeff_2"][0],
            inputs["fuel_coeff_1"][0],
            inputs["fuel_coeff_altitude"][0],
            inputs["he_coefficient"][0],
        )
        outputs["fuel_flow"] = values["fuel_flow"]
        outputs["thrust"] = values["thrust"]
        outputs["tsfc"] = values["tsfc"]
        outputs["tsfc_imperial"] = values["tsfc_imperial"]
        outputs["he_coeff"] = values["he_coeff"]

    def compute_partials(self, inputs, partials):
        values = simple_off_design_turbofan_values(
            inputs["altitude"][0],
            inputs["mach"][0],
            inputs["required_thrust"][0],
            inputs["electric_load"][0],
            inputs["thrust_available"][0],
            inputs["sea_level_static_thrust"][0],
            inputs["thrust_supplement"][0],
            inputs["fuel_coeff_3"][0],
            inputs["fuel_coeff_2"][0],
            inputs["fuel_coeff_1"][0],
            inputs["fuel_coeff_altitude"][0],
            inputs["he_coefficient"][0],
        )

        for output in (
            "fuel_flow",
            "thrust",
            "tsfc",
            "tsfc_imperial",
            "he_coeff",
        ):
            for variable in simple_off_design_input_names():
                partials[output, variable] = values[
                    "d%s_d%s" % (output, variable)
                ]


def simple_off_design_input_names():
    """Return scalar input names for SimpleOffDesignTurbofan derivatives."""

    return (
        "altitude",
        "mach",
        "required_thrust",
        "electric_load",
        "thrust_available",
        "sea_level_static_thrust",
        "thrust_supplement",
        "fuel_coeff_3",
        "fuel_coeff_2",
        "fuel_coeff_1",
        "fuel_coeff_altitude",
        "he_coefficient",
    )


def burner_flow_input_names():
    """Return scalar inputs for BurnerFlow derivative bookkeeping."""

    return (
        "mass_flow_31",
        "area_31",
        "total_pressure_31",
        "total_temperature_31",
        "mach_31",
        "gamma_31",
        "outer_radius_31",
        "total_temperature_4",
        "fuel_lhv",
        "combustor_efficiency",
    )


def burner_flow_output_names():
    """Return scalar BurnerFlow outputs."""

    return (
        "diffuser_mach_32",
        "diffuser_area_32",
        "diffuser_pressure_ratio",
        "fuel_flow",
        "fuel_air_ratio",
        "mass_flow_39",
        "total_pressure_39",
        "total_temperature_39",
        "static_temperature_39",
        "cp_air_39",
        "cv_air_39",
        "gamma_39",
        "static_pressure_39",
        "inner_radius_39",
    )


def burner_flow_values(
    mass_flow_31,
    area_31,
    total_pressure_31,
    total_temperature_31,
    mach_31,
    gamma_31,
    outer_radius_31,
    total_temperature_4,
    fuel_lhv,
    combustor_efficiency,
):
    """Return FAST burner outputs with forward chain-rule derivatives."""

    raw_inputs = {
        "mass_flow_31": mass_flow_31,
        "area_31": area_31,
        "total_pressure_31": total_pressure_31,
        "total_temperature_31": total_temperature_31,
        "mach_31": mach_31,
        "gamma_31": gamma_31,
        "outer_radius_31": outer_radius_31,
        "total_temperature_4": total_temperature_4,
        "fuel_lhv": fuel_lhv,
        "combustor_efficiency": combustor_efficiency,
    }
    values = {
        name: _Ad.variable(raw_inputs[name], name)
        for name in burner_flow_input_names()
    }

    mass31 = values["mass_flow_31"]
    area31 = values["area_31"]
    pt31 = values["total_pressure_31"]
    tt31 = values["total_temperature_31"]
    mach31 = values["mach_31"]
    gamma31 = values["gamma_31"]
    outer_radius = values["outer_radius_31"]
    tt4 = values["total_temperature_4"]
    lhv = values["fuel_lhv"]
    eta_combustor = values["combustor_efficiency"]

    diffuser_speed = 40.0
    blockage = 0.05
    gamma_loop = gamma31
    ts31 = tt31 / _ad_isentropic_q(mach31, gamma_loop)

    for _ in range(10):
        mach32 = diffuser_speed / _ad_sqrt(gamma_loop * GAS_CONSTANT_AIR * ts31)
        ts31 = tt31 / _ad_isentropic_q(mach32, gamma_loop)
        cp32 = _ad_sigmoid_heat_value(ts31, 233.0, 1.0 / 210.0, 875.0, 993.0)
        cv32 = _ad_sigmoid_heat_value(
            ts31,
            233.0,
            1.0 / 210.0,
            875.0,
            993.0 - GAS_CONSTANT_AIR,
        )
        gamma32 = cp32 / cv32
        gamma_loop = gamma32

    area_star = area31 / _ad_area_ratio(mach31, gamma_loop)
    area32 = area_star * _ad_area_ratio(mach32, gamma32)
    area_ratio = area32 / area31
    eta_dm = 0.965 - 2.72 * blockage
    eta_dopt = (
        1.0 - 2.0 * eta_dm + (eta_dm * area_ratio) ** 2.0
    ) / (eta_dm * area_ratio ** 2.0 - eta_dm)
    pi_d = 1.0 - (1.0 - 1.0 / area_ratio ** 2.0) * (1.0 - eta_dopt) / (
        1.0 + 2.0 / gamma_loop * mach31 ** 2.0
    )
    cp_air_combustion = _ad_integrated_heat_value(
        tt31,
        tt4,
        233.0,
        1.0 / 210.0,
        875.0,
        993.0,
    )
    cp_jeta_combustion = _ad_integrated_heat_value(
        tt31,
        tt4,
        4600.0,
        1.0 / 410.0,
        500.0,
        100.0,
    )
    fuel_flow = mass31 * cp_air_combustion / (
        eta_combustor * lhv - cp_jeta_combustion
    )
    mass39 = mass31 + fuel_flow
    fuel_air_ratio = fuel_flow / mass31
    pt39 = pt31 * pi_d * 0.95
    thermals = _ad_thermal_perfect_gamma(tt4, mach32, gamma32)
    ps39 = pt39 / _ad_pressure_ratio(mach32, thermals["updated_gamma"])
    inner_radius = _ad_sqrt(outer_radius ** 2.0 - area32 / math.pi)

    ad_outputs = {
        "diffuser_mach_32": mach32,
        "diffuser_area_32": area32,
        "diffuser_pressure_ratio": pi_d,
        "fuel_flow": fuel_flow,
        "fuel_air_ratio": fuel_air_ratio,
        "mass_flow_39": mass39,
        "total_pressure_39": pt39,
        "total_temperature_39": tt4,
        "static_temperature_39": thermals["static_temperature"],
        "cp_air_39": thermals["cp_air"],
        "cv_air_39": thermals["cv_air"],
        "gamma_39": thermals["updated_gamma"],
        "static_pressure_39": ps39,
        "inner_radius_39": inner_radius,
    }
    result = {}

    for output_name, ad_value in ad_outputs.items():
        result[output_name] = ad_value.value

        for input_name in burner_flow_input_names():
            result["d%s_d%s" % (output_name, input_name)] = ad_value.derivatives.get(
                input_name,
                0.0,
            )

    return result


def compressor_stage_flow_input_names():
    """Return scalar inputs for CompressorStageFlow derivative bookkeeping."""

    return (
        "mass_flow_1",
        "area_1",
        "total_pressure_1",
        "total_temperature_1",
        "static_pressure_1",
        "static_temperature_1",
        "mach_1",
        "gamma_1",
        "outer_radius_1",
        "stage_pressure_ratio",
        "rpm",
        "stage_efficiency",
    )


def compressor_stage_flow_output_names():
    """Return scalar CompressorStageFlow outputs."""

    return (
        "mass_flow_3",
        "total_pressure_3",
        "total_temperature_3",
        "static_temperature_3",
        "mach_3",
        "cp_air_3",
        "cv_air_3",
        "gamma_3",
        "static_pressure_3",
        "area_3",
        "outer_radius_3",
        "inner_radius_3",
        "pitch_radius_3",
        "work",
        "temperature_ratio",
        "stage_eta",
        "stage_psi",
        "corrected_mass_flow",
        "corrected_speed",
        "stage_phi",
        "stage_zeta",
    )


def compressor_stage_flow_values(
    mass_flow_1,
    area_1,
    total_pressure_1,
    total_temperature_1,
    static_pressure_1,
    static_temperature_1,
    mach_1,
    gamma_1,
    outer_radius_1,
    stage_pressure_ratio,
    rpm,
    stage_efficiency,
):
    """Return FAST compressor-stage outputs with forward derivatives."""

    raw_inputs = {
        "mass_flow_1": mass_flow_1,
        "area_1": area_1,
        "total_pressure_1": total_pressure_1,
        "total_temperature_1": total_temperature_1,
        "static_pressure_1": static_pressure_1,
        "static_temperature_1": static_temperature_1,
        "mach_1": mach_1,
        "gamma_1": gamma_1,
        "outer_radius_1": outer_radius_1,
        "stage_pressure_ratio": stage_pressure_ratio,
        "rpm": rpm,
        "stage_efficiency": stage_efficiency,
    }
    values = {
        name: _Ad.variable(raw_inputs[name], name)
        for name in compressor_stage_flow_input_names()
    }
    mass1 = values["mass_flow_1"]
    area1 = values["area_1"]
    pt1 = values["total_pressure_1"]
    tt1 = values["total_temperature_1"]
    ps1 = values["static_pressure_1"]
    ts1 = values["static_temperature_1"]
    mach1 = values["mach_1"]
    gamma1 = values["gamma_1"]
    ro1 = values["outer_radius_1"]
    stage_pr = values["stage_pressure_ratio"]
    rpm_ad = values["rpm"]
    efficiency = values["stage_efficiency"]
    pressure_std = 101325.353
    temperature_std = 288.15
    rho1 = ps1 / ts1 / GAS_CONSTANT_AIR
    tau = stage_pr ** ((gamma1 - 1.0) / gamma1)
    mach3 = mach1 * _ad_sqrt(
        1.0
        / (
            tau * (1.0 + (gamma1 - 1.0) * mach1 ** 2.0 / 2.0)
            - (gamma1 - 1.0) * mach1 ** 2.0 / 2.0
        )
    )
    tt3 = tau * tt1
    thermals = _ad_thermal_perfect_gamma(tt3, mach3, gamma1)
    gamma3 = thermals["updated_gamma"]
    pt3 = stage_pr * pt1
    ps3 = pt3 / _ad_pressure_ratio(mach3, gamma3)
    rho3 = ps3 / GAS_CONSTANT_AIR / thermals["static_temperature"]
    area3 = area1 * rho1 / rho3
    ri3 = _ad_sqrt(ro1 ** 2.0 - area3 / math.pi)
    rp3 = (ro1 + ri3) / 2.0
    heat_added = _ad_integrated_heat_value(
        tt1,
        tt3,
        233.0,
        1.0 / 210.0,
        875.0,
        993.0,
    )
    work = heat_added * mass1 / efficiency
    omega = rpm_ad / 60.0 * 2.0 * math.pi
    blade_speed = omega * rp3
    ts1_cp = _ad_sigmoid_heat_value(ts1, 233.0, 1.0 / 210.0, 875.0, 993.0)
    stage_eta = (
        ts1 * stage_pr ** ((gamma3 - 1.0) / gamma3) * ts1_cp
        - ts1 * ts1_cp
    ) / work * mass1
    stage_psi = heat_added / blade_speed ** 2.0
    corrected_mass_flow = (
        mass1 * _ad_sqrt(thermals["static_temperature"] / temperature_std)
        / (ps3 / pressure_std)
    )
    corrected_speed = rpm_ad / _ad_sqrt(
        thermals["static_temperature"] / temperature_std
    )
    stage_phi = (
        GAS_CONSTANT_AIR
        * 60.0
        / (2.0 * math.pi * area3 * rp3)
        * (corrected_mass_flow / pressure_std * math.sqrt(temperature_std))
        / (corrected_speed / math.sqrt(temperature_std))
    )
    stage_zeta = stage_psi / stage_eta
    ad_outputs = {
        "mass_flow_3": mass1,
        "total_pressure_3": pt3,
        "total_temperature_3": tt3,
        "static_temperature_3": thermals["static_temperature"],
        "mach_3": mach3,
        "cp_air_3": thermals["cp_air"],
        "cv_air_3": thermals["cv_air"],
        "gamma_3": gamma3,
        "static_pressure_3": ps3,
        "area_3": area3,
        "outer_radius_3": ro1,
        "inner_radius_3": ri3,
        "pitch_radius_3": rp3,
        "work": work,
        "temperature_ratio": tau,
        "stage_eta": stage_eta,
        "stage_psi": stage_psi,
        "corrected_mass_flow": corrected_mass_flow,
        "corrected_speed": corrected_speed,
        "stage_phi": stage_phi,
        "stage_zeta": stage_zeta,
    }
    result = {}

    for output_name, ad_value in ad_outputs.items():
        result[output_name] = ad_value.value

        for input_name in compressor_stage_flow_input_names():
            result["d%s_d%s" % (output_name, input_name)] = (
                ad_value.derivatives.get(input_name, 0.0)
            )

    return result


def turbine_stage_flow_input_names():
    """Return scalar inputs for TurbineStageFlow derivative bookkeeping."""

    return (
        "mass_flow_1",
        "total_pressure_1",
        "total_temperature_1",
        "mach_1",
        "gamma_1",
        "outer_radius_1",
        "inner_radius_1",
        "target_total_temperature_3",
        "stage_mach_2",
        "rpm",
        "turbine_efficiency",
    )


def turbine_stage_flow_output_names():
    """Return scalar TurbineStageFlow outputs."""

    return (
        "total_temperature_3",
        "total_pressure_3",
        "mach_3",
        "static_temperature_3",
        "cp_air_3",
        "cv_air_3",
        "gamma_3",
        "static_pressure_3",
        "area_3",
        "inner_radius_3",
        "outer_radius_3",
        "pressure_ratio",
        "temperature_ratio",
        "stage_psi",
    )


def turbine_stage_flow_values(
    mass_flow_1,
    total_pressure_1,
    total_temperature_1,
    mach_1,
    gamma_1,
    outer_radius_1,
    inner_radius_1,
    target_total_temperature_3,
    stage_mach_2,
    rpm,
    turbine_efficiency,
):
    """Return FAST turbine-stage outputs with forward derivatives."""

    raw_inputs = {
        "mass_flow_1": mass_flow_1,
        "total_pressure_1": total_pressure_1,
        "total_temperature_1": total_temperature_1,
        "mach_1": mach_1,
        "gamma_1": gamma_1,
        "outer_radius_1": outer_radius_1,
        "inner_radius_1": inner_radius_1,
        "target_total_temperature_3": target_total_temperature_3,
        "stage_mach_2": stage_mach_2,
        "rpm": rpm,
        "turbine_efficiency": turbine_efficiency,
    }
    values = {
        name: _Ad.variable(raw_inputs[name], name)
        for name in turbine_stage_flow_input_names()
    }
    mass1 = values["mass_flow_1"]
    pt1 = values["total_pressure_1"]
    tt1 = values["total_temperature_1"]
    mach1 = values["mach_1"]
    gamma1 = values["gamma_1"]
    ro1 = values["outer_radius_1"]
    ri1 = values["inner_radius_1"]
    tt3 = values["target_total_temperature_3"]
    mach2 = values["stage_mach_2"]
    rpm_ad = values["rpm"]
    eta_turbine = values["turbine_efficiency"]
    ts1 = tt1 / _ad_isentropic_q(mach1, gamma1)
    u1 = mach1 * _ad_sqrt(gamma1 * GAS_CONSTANT_AIR * ts1)
    omega = rpm_ad / 60.0 * 2.0 * math.pi
    radius_pitch = 0.5 * (ro1 + ri1)
    heat_removed = _ad_integrated_heat_value(
        tt3,
        tt1,
        233.0,
        1.0 / 210.0,
        875.0,
        993.0,
    )
    psi = heat_removed / (omega * radius_pitch) ** 2.0
    tau = tt3 / tt1
    pressure_ratio = tau ** (gamma1 / (gamma1 - 1.0) * eta_turbine)
    cp_total = _ad_sigmoid_heat_value(tt1, 233.0, 1.0 / 210.0, 875.0, 993.0)
    velocity_prime = _ad_sqrt(cp_total * tt1)
    velocity2 = velocity_prime * _ad_sqrt(
        (gamma1 - 1.0) * mach2 ** 2.0
        / (1.0 + (gamma1 - 1.0) / 2.0 * mach2 ** 2.0)
    )
    mach3 = mach2 * (u1 / velocity2) / _ad_sqrt(
        1.0
        - (1.0 - tau)
        * (1.0 - psi / 2.0)
        * (1.0 + (gamma1 - 1.0) / 2.0 * mach2 ** 2.0)
    )
    pt3 = pressure_ratio * pt1
    thermals = _ad_thermal_perfect_gamma(tt3, mach3, gamma1)
    gamma3 = thermals["updated_gamma"]
    ps3 = pt3 / _ad_pressure_ratio(mach3, gamma3)
    u3 = mach3 * _ad_sqrt(thermals["static_temperature"] * gamma3 * GAS_CONSTANT_AIR)
    rho3 = ps3 / thermals["static_temperature"] / GAS_CONSTANT_AIR
    area3 = mass1 / u3 / rho3
    ro3 = _ad_sqrt(ri1 ** 2.0 + area3 / math.pi)
    ad_outputs = {
        "total_temperature_3": tt3,
        "total_pressure_3": pt3,
        "mach_3": mach3,
        "static_temperature_3": thermals["static_temperature"],
        "cp_air_3": thermals["cp_air"],
        "cv_air_3": thermals["cv_air"],
        "gamma_3": gamma3,
        "static_pressure_3": ps3,
        "area_3": area3,
        "inner_radius_3": ri1,
        "outer_radius_3": ro3,
        "pressure_ratio": pressure_ratio,
        "temperature_ratio": tau,
        "stage_psi": psi,
    }
    result = {}

    for output_name, ad_value in ad_outputs.items():
        result[output_name] = ad_value.value

        for input_name in turbine_stage_flow_input_names():
            result["d%s_d%s" % (output_name, input_name)] = (
                ad_value.derivatives.get(input_name, 0.0)
            )

    return result


def perfect_expansion_nozzle_flow_input_names():
    """Return scalar inputs for PerfectExpansionNozzleFlow derivatives."""

    return (
        "mass_flow_5",
        "total_pressure_5",
        "total_temperature_5",
        "cp_air_5",
        "gamma_5",
        "inner_radius_5",
        "ambient_total_pressure",
        "ambient_mach",
        "ambient_gamma",
        "nozzle_pressure_ratio",
        "nozzle_efficiency",
    )


def perfect_expansion_nozzle_flow_output_names():
    """Return scalar PerfectExpansionNozzleFlow outputs."""

    return (
        "total_temperature_9",
        "static_temperature_9",
        "total_pressure_9",
        "static_pressure_9",
        "mach_9",
        "cp_air_9",
        "cv_air_9",
        "gamma_9",
        "area_9",
        "core_outer_radius_9",
        "bypass_outer_radius_9",
        "exit_velocity",
        "thrust",
    )


def perfect_expansion_nozzle_flow_values(
    mass_flow_5,
    total_pressure_5,
    total_temperature_5,
    cp_air_5,
    gamma_5,
    inner_radius_5,
    ambient_total_pressure,
    ambient_mach,
    ambient_gamma,
    nozzle_pressure_ratio,
    nozzle_efficiency,
):
    """Return FAST perfect-expansion nozzle outputs and derivatives."""

    raw_inputs = {
        "mass_flow_5": mass_flow_5,
        "total_pressure_5": total_pressure_5,
        "total_temperature_5": total_temperature_5,
        "cp_air_5": cp_air_5,
        "gamma_5": gamma_5,
        "inner_radius_5": inner_radius_5,
        "ambient_total_pressure": ambient_total_pressure,
        "ambient_mach": ambient_mach,
        "ambient_gamma": ambient_gamma,
        "nozzle_pressure_ratio": nozzle_pressure_ratio,
        "nozzle_efficiency": nozzle_efficiency,
    }
    values = {
        name: _Ad.variable(raw_inputs[name], name)
        for name in perfect_expansion_nozzle_flow_input_names()
    }
    mass5 = values["mass_flow_5"]
    pt5 = values["total_pressure_5"]
    tt5 = values["total_temperature_5"]
    cp5 = values["cp_air_5"]
    gamma5 = values["gamma_5"]
    inner_radius = values["inner_radius_5"]
    ambient_pt = values["ambient_total_pressure"]
    ambient_mach_ad = values["ambient_mach"]
    ambient_gamma_ad = values["ambient_gamma"]
    npr = values["nozzle_pressure_ratio"]
    efficiency = values["nozzle_efficiency"]
    ambient_ps = ambient_pt / _ad_pressure_ratio(ambient_mach_ad, ambient_gamma_ad)
    ps9 = ambient_ps * npr
    ts9_ideal = tt5 * (ps9 / pt5) ** ((gamma5 - 1.0) / gamma5)
    u9_ideal = _ad_sqrt((tt5 - ts9_ideal) * 2.0 * cp5)
    u9 = u9_ideal * efficiency
    ts9 = tt5 - u9 ** 2.0 / 2.0 / cp5
    pt9 = ps9 * (tt5 / ts9) ** (gamma5 / (gamma5 - 1.0))
    mach9 = u9 / _ad_sqrt(gamma5 * GAS_CONSTANT_AIR * ts9)

    if mach9.value > 1.0:
        mach9 = _Ad(1.0)
        u9 = mach9 * _ad_sqrt(gamma5 * GAS_CONSTANT_AIR * ts9)
        ts9 = tt5 - u9 ** 2.0 / 2.0 / cp5
        ps9 = pt9 / (tt5 / ts9) ** (gamma5 / (gamma5 - 1.0))

    cp9 = _ad_sigmoid_heat_value(ts9, 233.0, 1.0 / 210.0, 875.0, 993.0)
    cv9 = _ad_sigmoid_heat_value(
        ts9,
        233.0,
        1.0 / 210.0,
        875.0,
        993.0 - GAS_CONSTANT_AIR,
    )
    gamma9 = cp9 / cv9
    rho9 = ps9 / ts9 / GAS_CONSTANT_AIR
    area9 = mass5 / u9 / rho9
    core_outer_radius = _ad_sqrt(area9 / math.pi)
    bypass_radius_argument = area9 / math.pi - inner_radius ** 2.0

    if bypass_radius_argument.value > 0.0:
        bypass_outer_radius = _ad_sqrt(bypass_radius_argument)
    else:
        bypass_outer_radius = _Ad(0.0)

    thrust = mass5 * u9 + (ps9 - ambient_ps) * area9
    ad_outputs = {
        "total_temperature_9": tt5,
        "static_temperature_9": ts9,
        "total_pressure_9": pt9,
        "static_pressure_9": ps9,
        "mach_9": mach9,
        "cp_air_9": cp9,
        "cv_air_9": cv9,
        "gamma_9": gamma9,
        "area_9": area9,
        "core_outer_radius_9": core_outer_radius,
        "bypass_outer_radius_9": bypass_outer_radius,
        "exit_velocity": u9,
        "thrust": thrust,
    }
    result = {}

    for output_name, ad_value in ad_outputs.items():
        result[output_name] = ad_value.value

        for input_name in perfect_expansion_nozzle_flow_input_names():
            result["d%s_d%s" % (output_name, input_name)] = (
                ad_value.derivatives.get(input_name, 0.0)
            )

    return result


def simple_off_design_turbofan_values(
    altitude,
    mach,
    required_thrust,
    electric_load,
    thrust_available,
    sea_level_static_thrust,
    thrust_supplement,
    fuel_coeff_3,
    fuel_coeff_2,
    fuel_coeff_1,
    fuel_coeff_altitude,
    he_coefficient,
):
    """Return FAST simple off-design turbofan outputs and derivatives."""

    derivatives = {}

    for output in (
        "fuel_flow",
        "thrust",
        "tsfc",
        "tsfc_imperial",
        "he_coeff",
    ):
        for variable in simple_off_design_input_names():
            derivatives["d%s_d%s" % (output, variable)] = 0.0

    atmosphere = atmosphere_layer(altitude)
    temperature = atmosphere["temperature"]
    dtemperature_daltitude = atmosphere["dtemperature_daltitude"]
    speed_factor = math.sqrt(1.4 * GAS_CONSTANT_AIR * temperature)
    tas = mach * speed_factor

    if math.isfinite(tas) and tas != 0.0:
        thrust_preclip = required_thrust - electric_load / tas
        dtas_dmach = speed_factor
        dtas_daltitude = (
            mach
            * 0.5
            * math.sqrt(1.4 * GAS_CONSTANT_AIR)
            / math.sqrt(temperature)
            * dtemperature_daltitude
        )
        dthrust_daltitude = electric_load * dtas_daltitude / tas ** 2
        dthrust_dmach = electric_load * dtas_dmach / tas ** 2
        dthrust_drequired = 1.0
        dthrust_delectric = -1.0 / tas
    else:
        thrust_preclip = required_thrust
        dthrust_daltitude = 0.0
        dthrust_dmach = 0.0
        dthrust_drequired = 1.0
        dthrust_delectric = 0.0

    if thrust_preclip < -1.0e-6:
        thrust = 0.0
        thrust_derivatives = {}
    elif thrust_preclip > thrust_available:
        thrust = thrust_available
        thrust_derivatives = {"thrust_available": 1.0}
    else:
        thrust = thrust_preclip
        thrust_derivatives = {
            "altitude": dthrust_daltitude,
            "mach": dthrust_dmach,
            "required_thrust": dthrust_drequired,
            "electric_load": dthrust_delectric,
        }

    supplement = max(thrust_supplement, 0.0)
    sls_thrust_conv = (sea_level_static_thrust + supplement) / 1000.0
    thrust_kn = thrust / 1000.0
    denominator = he_coefficient * sls_thrust_conv
    thrust_frac = thrust_kn / denominator
    fuel_flow = (
        fuel_coeff_3 * thrust_frac ** 3
        + fuel_coeff_2 * thrust_frac ** 2
        + fuel_coeff_1 * thrust_frac
        + fuel_coeff_altitude * thrust_kn * altitude
    )

    if thrust_kn <= 0.0:
        tsfc = 0.0
    else:
        tsfc = fuel_flow / thrust

    derivatives["dhe_coeff_dhe_coefficient"] = 1.0
    dfuel_dfrac = (
        3.0 * fuel_coeff_3 * thrust_frac ** 2
        + 2.0 * fuel_coeff_2 * thrust_frac
        + fuel_coeff_1
    )
    dfuel_dthrust_kn = dfuel_dfrac / denominator + fuel_coeff_altitude * altitude
    dfuel_ddenominator = -dfuel_dfrac * thrust_frac / denominator

    for variable, derivative in thrust_derivatives.items():
        derivatives["dthrust_d%s" % variable] = derivative
        derivatives["dfuel_flow_d%s" % variable] += (
            dfuel_dthrust_kn * derivative / 1000.0
        )

    derivatives["dfuel_flow_daltitude"] += fuel_coeff_altitude * thrust_kn
    derivatives["dfuel_flow_dfuel_coeff_3"] = thrust_frac ** 3
    derivatives["dfuel_flow_dfuel_coeff_2"] = thrust_frac ** 2
    derivatives["dfuel_flow_dfuel_coeff_1"] = thrust_frac
    derivatives["dfuel_flow_dfuel_coeff_altitude"] = thrust_kn * altitude
    derivatives["dfuel_flow_dhe_coefficient"] = (
        dfuel_ddenominator * sls_thrust_conv
    )
    derivatives["dfuel_flow_dsea_level_static_thrust"] = (
        dfuel_ddenominator * he_coefficient / 1000.0
    )

    if thrust_supplement >= 0.0:
        derivatives["dfuel_flow_dthrust_supplement"] = (
            dfuel_ddenominator * he_coefficient / 1000.0
        )

    if thrust_kn > 0.0:
        for variable in simple_off_design_input_names():
            dfuel = derivatives["dfuel_flow_d%s" % variable]
            dthrust = derivatives["dthrust_d%s" % variable]
            derivatives["dtsfc_d%s" % variable] = (
                dfuel * thrust - fuel_flow * dthrust
            ) / thrust ** 2
            derivatives["dtsfc_imperial_d%s" % variable] = (
                TSFC_SI_TO_IMPERIAL * derivatives["dtsfc_d%s" % variable]
            )

    derivatives.update(
        {
            "fuel_flow": fuel_flow,
            "thrust": thrust,
            "tsfc": tsfc,
            "tsfc_imperial": tsfc * TSFC_SI_TO_IMPERIAL,
            "he_coeff": he_coefficient,
        }
    )

    return derivatives


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


def off_design_nozzle_values(area_1, area_2, mach_1, gamma):
    """Return FAST off-design nozzle Mach and smooth implicit derivatives."""

    ratio_1 = area_ratio_values(mach_1, gamma)
    area_star = area_1 / ratio_1["ratio"]
    mach_2 = solve_off_design_nozzle_mach(area_star, area_2, mach_1, gamma)

    if mach_2 >= 1.0:
        return {
            "mach_2": 1.0,
            "dmach_2_darea_1": 0.0,
            "dmach_2_darea_2": 0.0,
            "dmach_2_dmach_1": 0.0,
            "dmach_2_dgamma": 0.0,
        }

    if mach_2 < 0.0:
        return {
            "mach_2": mach_1,
            "dmach_2_darea_1": 0.0,
            "dmach_2_darea_2": 0.0,
            "dmach_2_dmach_1": 1.0,
            "dmach_2_dgamma": 0.0,
        }

    ratio_2 = area_ratio_values(mach_2, gamma)
    darea_star_darea_1 = 1.0 / ratio_1["ratio"]
    darea_star_dmach_1 = (
        -area_1 * ratio_1["dratio_dmach"] / ratio_1["ratio"] ** 2
    )
    darea_star_dgamma = (
        -area_1 * ratio_1["dratio_dgamma"] / ratio_1["ratio"] ** 2
    )
    dresidual_dmach_2 = area_star * ratio_2["dratio_dmach"]
    dresidual_darea_1 = darea_star_darea_1 * ratio_2["ratio"]
    dresidual_darea_2 = -1.0
    dresidual_dmach_1 = darea_star_dmach_1 * ratio_2["ratio"]
    dresidual_dgamma = (
        darea_star_dgamma * ratio_2["ratio"]
        + area_star * ratio_2["dratio_dgamma"]
    )

    return {
        "mach_2": mach_2,
        "dmach_2_darea_1": -dresidual_darea_1 / dresidual_dmach_2,
        "dmach_2_darea_2": -dresidual_darea_2 / dresidual_dmach_2,
        "dmach_2_dmach_1": -dresidual_dmach_1 / dresidual_dmach_2,
        "dmach_2_dgamma": -dresidual_dgamma / dresidual_dmach_2,
    }


def solve_off_design_nozzle_mach(area_star, area_2, mach_1, gamma):
    """Return the same Newton exit-Mach estimate as FAST-Python."""

    mach_2 = mach_1
    prime = 1.0
    iteration = 0

    while abs(prime) > 1.0e-5 and iteration < 10:
        ratio = area_ratio_values(mach_2, gamma)
        area_guess = area_star * ratio["ratio"]
        residual = (area_guess - area_2) ** 2
        prime = -2.0 * area_star * ratio["dratio_dmach"] * (
            area_2 - area_guess
        )
        mach_2 = mach_2 - residual / prime
        iteration += 1

    if mach_2 > 1.0:
        mach_2 = 1.0

    if mach_2 < 0.0:
        mach_2 = mach_1

    return mach_2


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


def air_temperature_from_heat_added_values(temperature_start, heat):
    """Return FAST heat-added inverse Cp temperature and loop derivatives."""

    return solve_air_temperature_from_heat_added(
        temperature_start,
        heat,
        return_derivatives=True,
    )


def air_temperature_from_heat_removed_values(temperature_start, heat):
    """Return FAST heat-removed inverse Cp temperature and loop derivatives."""

    return solve_air_temperature_from_heat_removed(
        temperature_start,
        heat,
        return_derivatives=True,
    )


def solve_air_temperature_from_heat_added(
    temperature_start,
    heat,
    return_derivatives=False,
):
    """Return the same added-heat Newton estimate as FAST-Python."""

    temperature_end = temperature_start * 1.05
    dtemperature_dstart = 1.05
    dtemperature_dheat = 0.0
    iteration = 0

    while (
        abs(
            integrated_heat_value(
                temperature_start,
                temperature_end,
                233.0,
                1.0 / 210.0,
                875.0,
                993.0,
            )
            - heat
        )
        / heat
        > 1.0e-3
        and iteration < 10
    ):
        numerator = (
            integrated_heat_value(
                temperature_end,
                temperature_start,
                233.0,
                1.0 / 210.0,
                875.0,
                993.0,
            )
            + heat
        )
        denominator = cp_air_inverse_prime(temperature_end)
        denominator_prime = cp_air_inverse_prime_derivative(temperature_end)
        cp_start = sigmoid_heat_value(
            temperature_start,
            233.0,
            1.0 / 210.0,
            875.0,
            993.0,
        )
        cp_end = sigmoid_heat_value(
            temperature_end,
            233.0,
            1.0 / 210.0,
            875.0,
            993.0,
        )
        dnumerator_dstart = cp_start - cp_end * dtemperature_dstart
        dnumerator_dheat = 1.0 - cp_end * dtemperature_dheat
        ddenominator_dstart = denominator_prime * dtemperature_dstart
        ddenominator_dheat = denominator_prime * dtemperature_dheat
        dtemperature_dstart = dtemperature_dstart - (
            dnumerator_dstart * denominator - numerator * ddenominator_dstart
        ) / denominator ** 2
        dtemperature_dheat = dtemperature_dheat - (
            dnumerator_dheat * denominator - numerator * ddenominator_dheat
        ) / denominator ** 2
        temperature_end = temperature_end - numerator / denominator
        iteration += 1

    if return_derivatives:
        return {
            "temperature_end": temperature_end,
            "dtemperature_end_dtemperature_start": dtemperature_dstart,
            "dtemperature_end_dheat": dtemperature_dheat,
        }

    return temperature_end


def solve_air_temperature_from_heat_removed(
    temperature_start,
    heat,
    return_derivatives=False,
):
    """Return the same removed-heat Newton estimate as FAST-Python."""

    temperature_end = temperature_start * 0.95
    dtemperature_dstart = 0.95
    dtemperature_dheat = 0.0
    iteration = 0

    while (
        abs(
            integrated_heat_value(
                temperature_end,
                temperature_start,
                233.0,
                1.0 / 210.0,
                875.0,
                993.0,
            )
            - heat
        )
        / heat
        > 1.0e-3
        and iteration < 10
    ):
        numerator = (
            integrated_heat_value(
                temperature_end,
                temperature_start,
                233.0,
                1.0 / 210.0,
                875.0,
                993.0,
            )
            - heat
        )
        denominator = cp_air_inverse_prime(temperature_end)
        denominator_prime = cp_air_inverse_prime_derivative(temperature_end)
        cp_start = sigmoid_heat_value(
            temperature_start,
            233.0,
            1.0 / 210.0,
            875.0,
            993.0,
        )
        cp_end = sigmoid_heat_value(
            temperature_end,
            233.0,
            1.0 / 210.0,
            875.0,
            993.0,
        )
        dnumerator_dstart = cp_start - cp_end * dtemperature_dstart
        dnumerator_dheat = -1.0 - cp_end * dtemperature_dheat
        ddenominator_dstart = denominator_prime * dtemperature_dstart
        ddenominator_dheat = denominator_prime * dtemperature_dheat
        dtemperature_dstart = dtemperature_dstart - (
            dnumerator_dstart * denominator - numerator * ddenominator_dstart
        ) / denominator ** 2
        dtemperature_dheat = dtemperature_dheat - (
            dnumerator_dheat * denominator - numerator * ddenominator_dheat
        ) / denominator ** 2
        temperature_end = temperature_end - numerator / denominator
        iteration += 1

    if return_derivatives:
        return {
            "temperature_end": temperature_end,
            "dtemperature_end_dtemperature_start": dtemperature_dstart,
            "dtemperature_end_dheat": dtemperature_dheat,
        }

    return temperature_end


def cp_air_inverse_prime(temperature):
    """Return FAST's Newton derivative expression for inverse Cp solvers."""

    return -(
        993.0
        + 233.0 / (math.exp((1.0 / 210.0) * (875.0 - temperature)) + 1.0)
        - 2.0 * 233.0
    )


def cp_air_inverse_prime_derivative(temperature):
    """Return derivative of FAST's inverse-Cp Newton denominator."""

    return -sigmoid_heat_derivative(
        temperature,
        233.0,
        1.0 / 210.0,
        875.0,
    )


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


class _Ad:
    """Small scalar forward derivative value used for compact engine chains."""

    def __init__(self, value, derivatives=None):
        self.value = float(value)
        self.derivatives = dict(derivatives or {})

    @classmethod
    def variable(cls, value, name):
        return cls(value, {name: 1.0})

    def __add__(self, other):
        other = _ad_value(other)
        derivatives = self.derivatives.copy()

        for name, derivative in other.derivatives.items():
            derivatives[name] = derivatives.get(name, 0.0) + derivative

        return _Ad(self.value + other.value, derivatives)

    __radd__ = __add__

    def __sub__(self, other):
        other = _ad_value(other)
        derivatives = self.derivatives.copy()

        for name, derivative in other.derivatives.items():
            derivatives[name] = derivatives.get(name, 0.0) - derivative

        return _Ad(self.value - other.value, derivatives)

    def __rsub__(self, other):
        return _ad_value(other).__sub__(self)

    def __mul__(self, other):
        other = _ad_value(other)
        derivatives = {}

        for name in set(self.derivatives) | set(other.derivatives):
            derivatives[name] = (
                self.derivatives.get(name, 0.0) * other.value
                + other.derivatives.get(name, 0.0) * self.value
            )

        return _Ad(self.value * other.value, derivatives)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _ad_value(other)
        derivatives = {}

        for name in set(self.derivatives) | set(other.derivatives):
            derivatives[name] = (
                self.derivatives.get(name, 0.0) * other.value
                - self.value * other.derivatives.get(name, 0.0)
            ) / other.value ** 2

        return _Ad(self.value / other.value, derivatives)

    def __rtruediv__(self, other):
        return _ad_value(other).__truediv__(self)

    def __pow__(self, other):
        other = _ad_value(other)
        value = self.value ** other.value

        if not other.derivatives or all(derivative == 0.0 for derivative in other.derivatives.values()):
            derivatives = {}
            if self.derivatives:
                scale = other.value * self.value ** (other.value - 1.0)
                for name, derivative in self.derivatives.items():
                    derivatives[name] = scale * derivative
            return _Ad(value, derivatives)

        derivatives = {}

        for name in set(self.derivatives) | set(other.derivatives):
            derivatives[name] = value * (
                other.derivatives.get(name, 0.0) * math.log(self.value)
                + other.value * self.derivatives.get(name, 0.0) / self.value
            )

        return _Ad(value, derivatives)

    def __rpow__(self, other):
        return _ad_value(other).__pow__(self)

    def __neg__(self):
        return _Ad(
            -self.value,
            {name: -derivative for name, derivative in self.derivatives.items()},
        )


def _ad_value(value):
    """Return ``value`` as an automatic derivative scalar."""

    if isinstance(value, _Ad):
        return value

    return _Ad(value)


def _ad_exp(value):
    """Return exponential and derivatives."""

    value = _ad_value(value)
    exponential = math.exp(value.value)
    return _Ad(
        exponential,
        {
            name: exponential * derivative
            for name, derivative in value.derivatives.items()
        },
    )


def _ad_log(value):
    """Return natural log and derivatives."""

    value = _ad_value(value)
    return _Ad(
        math.log(value.value),
        {
            name: derivative / value.value
            for name, derivative in value.derivatives.items()
        },
    )


def _ad_sqrt(value):
    """Return square root and derivatives."""

    return _ad_value(value) ** 0.5


def _ad_isentropic_q(mach, gamma):
    """Return FAST isentropic temperature-ratio term with derivatives."""

    return 1.0 + 0.5 * (gamma - 1.0) * mach ** 2.0


def _ad_pressure_ratio(mach, gamma):
    """Return FAST isentropic total-to-static pressure ratio with derivatives."""

    q = _ad_isentropic_q(mach, gamma)
    return q ** (gamma / (gamma - 1.0))


def _ad_area_ratio(mach, gamma):
    """Return FAST area-to-choked-area ratio with derivatives."""

    q = _ad_isentropic_q(mach, gamma)
    base = 0.5 * (gamma + 1.0)
    exponent = 0.5 * (gamma + 1.0) / (gamma - 1.0)
    return base ** (-exponent) * q ** exponent / mach


def _ad_sigmoid_heat_value(temperature, length, rate, midpoint, offset):
    """Return FAST fitted heat capacity and derivatives."""

    return length / (1.0 + _ad_exp(-rate * (temperature - midpoint))) + offset


def _ad_heat_antiderivative(temperature, length, rate, midpoint, offset):
    """Return fitted heat-capacity antiderivative and derivatives."""

    exponential = _ad_exp(rate * (midpoint - temperature))
    return temperature * (offset + length) + length * _ad_log(1.0 + exponential) / rate


def _ad_integrated_heat_value(t_low, t_high, length, rate, midpoint, offset):
    """Return integrated heat and derivatives."""

    return _ad_heat_antiderivative(
        t_high,
        length,
        rate,
        midpoint,
        offset,
    ) - _ad_heat_antiderivative(
        t_low,
        length,
        rate,
        midpoint,
        offset,
    )


def _ad_thermal_perfect_gamma(total_temperature, mach, gamma):
    """Return FAST thermally perfect gamma loop with derivative scalars."""

    gamma_new = gamma
    delta_gamma = 1.0
    iteration = 0
    static_temperature = total_temperature / _ad_isentropic_q(mach, gamma_new)
    cp = None
    cv = None

    while delta_gamma > 1.0e-3 and iteration < 10:
        static_temperature = total_temperature / _ad_isentropic_q(mach, gamma_new)
        cp = _ad_sigmoid_heat_value(
            static_temperature,
            233.0,
            1.0 / 210.0,
            875.0,
            993.0,
        )
        cv = _ad_sigmoid_heat_value(
            static_temperature,
            233.0,
            1.0 / 210.0,
            875.0,
            993.0 - GAS_CONSTANT_AIR,
        )
        gamma_next = cp / cv
        delta_gamma = abs(gamma_next.value - gamma_new.value) / gamma_new.value
        gamma_new = gamma_next
        iteration += 1

    return {
        "static_temperature": static_temperature,
        "cp_air": cp,
        "cv_air": cv,
        "updated_gamma": gamma_new,
    }


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
