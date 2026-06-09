# src/fast_openmdao/constraint.py

"""OpenMDAO components for FAST constraint primitive equations."""

import math

import openmdao.api as om

from fast_openmdao.atmosphere import atmosphere_layer
from fast_python.atmosphere import standard_atmosphere
from fast_python.units import convert_force, convert_length, convert_mass, convert_velocity


RHO_SL_STD = standard_atmosphere(0.0)[2]
KG_TO_SLUG = convert_mass(1.0, "kg", "slug")
M_TO_FT = convert_length(1.0, "m", "ft")
KG_M2_TO_LBM_FT2 = convert_mass(1.0, "kg", "lbm") / M_TO_FT ** 2
N_M2_TO_LBF_FT2 = 9.81 * convert_force(1.0, "N", "lbf") / M_TO_FT ** 2
RHO_SI_TO_ENGLISH = KG_TO_SLUG / M_TO_FT ** 3


class JetApproachConstraint(om.ExplicitComponent):
    """Compute FAST approach-speed constraint residual."""

    def initialize(self):
        self.options.declare("req_type", default=1)
        self.options.declare("cl_landing", default=2.5)
        self.options.declare("wland_mtow", default=0.85)
        self.options.declare("approach_velocity", default=70.0)

    def setup(self):
        self.add_input("wing_loading", val=400.0, units="kg/m**2")
        self.add_input("thrust_loading", val=0.3)
        self.add_output("approach_residual", val=0.0)
        self.declare_partials(of="approach_residual", wrt="*")

    def compute(self, inputs, outputs):
        outputs["approach_residual"] = jet_approach_residual(
            inputs["wing_loading"][0],
            self.options["req_type"],
            self.options["cl_landing"],
            self.options["wland_mtow"],
            self.options["approach_velocity"],
        )

    def compute_partials(self, inputs, partials):
        if self.options["req_type"] == 0:
            partials["approach_residual", "wing_loading"] = 0.0
        else:
            partials["approach_residual", "wing_loading"] = KG_M2_TO_LBM_FT2

        partials["approach_residual", "thrust_loading"] = 0.0


class JetTakeoffFieldLengthConstraint(om.ExplicitComponent):
    """Compute FAST takeoff-field-length constraint residual."""

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")
        self.options.declare("cl_takeoff", default=2.0)
        self.options.declare("balanced_field_length", default=1500.0)
        self.options.declare("stall_velocity", default=60.0)

    def setup(self):
        self.add_input("wing_loading", val=400.0, units="kg/m**2")
        self.add_input("thrust_loading", val=0.3)
        self.add_output("takeoff_field_length_residual", val=0.0)
        self.declare_partials(of="takeoff_field_length_residual", wrt="*")

    def compute(self, inputs, outputs):
        outputs["takeoff_field_length_residual"] = jet_takeoff_field_length_residual(
            inputs["wing_loading"][0],
            inputs["thrust_loading"][0],
            self.options["aircraft_class"],
            self.options["cl_takeoff"],
            self.options["balanced_field_length"],
            self.options["stall_velocity"],
        )

    def compute_partials(self, inputs, partials):
        derivatives = jet_takeoff_field_length_derivatives(
            inputs["thrust_loading"][0],
            self.options["aircraft_class"],
            self.options["cl_takeoff"],
            self.options["balanced_field_length"],
            self.options["stall_velocity"],
        )
        partials["takeoff_field_length_residual", "wing_loading"] = derivatives[
            "dresidual_dwing_loading"
        ]
        partials["takeoff_field_length_residual", "thrust_loading"] = derivatives[
            "dresidual_dthrust_loading"
        ]


class JetLandingFieldLengthConstraint(om.ExplicitComponent):
    """Compute FAST landing-field-length constraint residual."""

    def initialize(self):
        self.options.declare("req_type", default=1)
        self.options.declare("cl_landing", default=2.5)
        self.options.declare("landing_field_length", default=1200.0)
        self.options.declare("obstacle_length", default=15.0)
        self.options.declare("wland_mtow", default=0.85)

    def setup(self):
        self.add_input("wing_loading", val=400.0, units="kg/m**2")
        self.add_input("thrust_loading", val=0.3)
        self.add_output("landing_field_length_residual", val=0.0)
        self.declare_partials(of="landing_field_length_residual", wrt="*")

    def compute(self, inputs, outputs):
        outputs["landing_field_length_residual"] = jet_landing_field_length_residual(
            inputs["wing_loading"][0],
            self.options["req_type"],
            self.options["cl_landing"],
            self.options["landing_field_length"],
            self.options["obstacle_length"],
            self.options["wland_mtow"],
        )

    def compute_partials(self, inputs, partials):
        partials["landing_field_length_residual", "wing_loading"] = N_M2_TO_LBF_FT2
        partials["landing_field_length_residual", "thrust_loading"] = 0.0


class JetCruiseConstraint(om.ExplicitComponent):
    """Compute FAST cruise or diversion performance constraint residual."""

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")
        self.options.declare("req_type", default=1)
        self.options.declare("cd0", default=0.02)
        self.options.declare("aspect_ratio", default=9.0)
        self.options.declare("oswald", default=0.8)
        self.options.declare("altitude", default=10000.0)
        self.options.declare("mach", default=0.78)
        self.options.declare("lapse_exp", default=0.6)
        self.options.declare("devries_exp", default=0.1)

    def setup(self):
        self.add_input("wing_loading", val=400.0, units="kg/m**2")
        self.add_input("thrust_loading", val=0.3)
        self.add_output("cruise_residual", val=0.0)
        self.declare_partials(of="cruise_residual", wrt="*")

    def compute(self, inputs, outputs):
        outputs["cruise_residual"] = jet_cruise_residual(
            inputs["wing_loading"][0],
            inputs["thrust_loading"][0],
            self.options["aircraft_class"],
            self.options["req_type"],
            self.options["cd0"],
            self.options["aspect_ratio"],
            self.options["oswald"],
            self.options["altitude"],
            self.options["mach"],
            self.options["lapse_exp"],
            self.options["devries_exp"],
        )

    def compute_partials(self, inputs, partials):
        derivatives = jet_cruise_derivatives(
            inputs["wing_loading"][0],
            inputs["thrust_loading"][0],
            self.options["aircraft_class"],
            self.options["req_type"],
            self.options["cd0"],
            self.options["aspect_ratio"],
            self.options["oswald"],
            self.options["altitude"],
            self.options["mach"],
            self.options["lapse_exp"],
            self.options["devries_exp"],
        )
        partials["cruise_residual", "wing_loading"] = derivatives[
            "dresidual_dwing_loading"
        ]
        partials["cruise_residual", "thrust_loading"] = derivatives[
            "dresidual_dthrust_loading"
        ]


class FAR25ClimbConstraint(om.ExplicitComponent):
    """Compute FAST shared FAR 25 climb residual."""

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")
        self.options.declare("req_type", default=1)
        self.options.declare("cl", default=2.0)
        self.options.declare("cd0", default=0.025)
        self.options.declare("aspect_ratio", default=9.0)
        self.options.declare("oswald", default=0.75)
        self.options.declare("correction", default=1.0)
        self.options.declare("gradient", default=0.012)
        self.options.declare("ks", default=1.2)
        self.options.declare("stall_velocity", default=60.0)

    def setup(self):
        self.add_input("wing_loading", val=400.0, units="kg/m**2")
        self.add_input("thrust_loading", val=0.3)
        self.add_output("far25_climb_residual", val=0.0)
        self.declare_partials(of="far25_climb_residual", wrt="*")

    def compute(self, inputs, outputs):
        outputs["far25_climb_residual"] = far25_climb_residual(
            inputs["wing_loading"][0],
            inputs["thrust_loading"][0],
            self.options["aircraft_class"],
            self.options["req_type"],
            self.options["cl"],
            self.options["cd0"],
            self.options["aspect_ratio"],
            self.options["oswald"],
            self.options["correction"],
            self.options["gradient"],
            self.options["ks"],
            self.options["stall_velocity"],
        )

    def compute_partials(self, inputs, partials):
        derivatives = far25_climb_derivatives(
            inputs["wing_loading"][0],
            inputs["thrust_loading"][0],
            self.options["aircraft_class"],
            self.options["req_type"],
            self.options["cl"],
            self.options["cd0"],
            self.options["aspect_ratio"],
            self.options["oswald"],
            self.options["correction"],
            self.options["ks"],
            self.options["stall_velocity"],
        )
        partials["far25_climb_residual", "wing_loading"] = derivatives[
            "dresidual_dwing_loading"
        ]
        partials["far25_climb_residual", "thrust_loading"] = derivatives[
            "dresidual_dthrust_loading"
        ]


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


def jet_approach_residual(
    wing_loading,
    req_type,
    cl_landing,
    wland_mtow,
    approach_velocity,
):
    """Return FAST JetApp residual for one scalar wing loading."""

    if req_type == 0:
        return 0.0

    vapp_ft_s = convert_velocity(approach_velocity, "m/s", "ft/s")
    vstall = vapp_ft_s / 1.3
    required = 0.5 * 0.002377 * vstall ** 2 * cl_landing / wland_mtow
    return wing_loading * KG_M2_TO_LBM_FT2 - required


def jet_takeoff_field_length_residual(
    wing_loading,
    thrust_loading,
    aircraft_class,
    cl_takeoff,
    balanced_field_length,
    stall_velocity,
):
    """Return FAST JetTOFL residual for one scalar design point."""

    top25 = balanced_field_length * M_TO_FT / 37.5
    converted_wing_loading = wing_loading * N_M2_TO_LBF_FT2
    effective_thrust_loading = thrust_loading

    if aircraft_class.lower() in ("turboprop", "piston"):
        effective_thrust_loading = 1.0 / (1.1 * stall_velocity * thrust_loading)

    return converted_wing_loading / (cl_takeoff * top25) - effective_thrust_loading


def jet_takeoff_field_length_derivatives(
    thrust_loading,
    aircraft_class,
    cl_takeoff,
    balanced_field_length,
    stall_velocity,
):
    """Return FAST JetTOFL scalar derivatives."""

    top25 = balanced_field_length * M_TO_FT / 37.5
    dresidual_dwing_loading = N_M2_TO_LBF_FT2 / (cl_takeoff * top25)

    if aircraft_class.lower() in ("turboprop", "piston"):
        dresidual_dthrust_loading = 1.0 / (
            1.1 * stall_velocity * thrust_loading ** 2
        )
    else:
        dresidual_dthrust_loading = -1.0

    return {
        "dresidual_dwing_loading": dresidual_dwing_loading,
        "dresidual_dthrust_loading": dresidual_dthrust_loading,
    }


def jet_landing_field_length_residual(
    wing_loading,
    req_type,
    cl_landing,
    landing_field_length,
    obstacle_length,
    wland_mtow,
):
    """Return FAST JetLFL residual for one scalar wing loading."""

    converted_wing_loading = wing_loading * N_M2_TO_LBF_FT2
    landing_ft = landing_field_length * M_TO_FT
    obstacle_ft = obstacle_length * M_TO_FT

    if req_type == 0:
        distance = 0.6 * landing_ft - obstacle_ft
        required = 0.95 * cl_landing * distance / 80.0 / wland_mtow
        return converted_wing_loading - required

    vapp = (landing_ft / 0.3) ** 0.5
    vstall = vapp / 1.3 * convert_velocity(1.0, "kts", "ft/s")
    required = 0.5 * 0.002377 * vstall ** 2 * cl_landing / wland_mtow
    return converted_wing_loading - required


def jet_cruise_residual(
    wing_loading,
    thrust_loading,
    aircraft_class,
    req_type,
    cd0,
    aspect_ratio,
    oswald,
    altitude,
    mach,
    lapse_exp,
    devries_exp,
):
    """Return FAST JetCrs/JetDiv residual for one scalar design point."""

    q, rho_ratio, velocity = cruise_dynamic_pressure_base(altitude, mach)
    converted_wing_loading = wing_loading * KG_M2_TO_LBM_FT2
    base = cruise_drag_residual_base(converted_wing_loading, q, cd0, aspect_ratio, oswald)

    if aircraft_class.lower() in ("turboprop", "piston"):
        velocity_mps = convert_velocity(velocity, "ft/s", "m/s")
        effective_thrust_loading = 1.0 / (velocity_mps * thrust_loading)
        return base - effective_thrust_loading

    if req_type in (0, 1):
        return base / rho_ratio ** lapse_exp - thrust_loading

    if req_type == 2:
        return base / rho_ratio ** devries_exp - thrust_loading

    raise ValueError("Jet cruise constraints require ReqType 0, 1, or 2.")


def jet_cruise_derivatives(
    wing_loading,
    thrust_loading,
    aircraft_class,
    req_type,
    cd0,
    aspect_ratio,
    oswald,
    altitude,
    mach,
    lapse_exp,
    devries_exp,
):
    """Return scalar FAST JetCrs/JetDiv derivatives."""

    q, rho_ratio, velocity = cruise_dynamic_pressure_base(altitude, mach)
    converted_wing_loading = wing_loading * KG_M2_TO_LBM_FT2
    dbase_dwing_loading = (
        cruise_drag_residual_base_derivative(
            converted_wing_loading,
            q,
            cd0,
            aspect_ratio,
            oswald,
        )
        * KG_M2_TO_LBM_FT2
    )

    if aircraft_class.lower() in ("turboprop", "piston"):
        velocity_mps = convert_velocity(velocity, "ft/s", "m/s")
        return {
            "dresidual_dwing_loading": dbase_dwing_loading,
            "dresidual_dthrust_loading": 1.0 / (
                velocity_mps * thrust_loading ** 2
            ),
        }

    if req_type in (0, 1):
        scale = rho_ratio ** lapse_exp
    elif req_type == 2:
        scale = rho_ratio ** devries_exp
    else:
        raise ValueError("Jet cruise constraints require ReqType 0, 1, or 2.")

    return {
        "dresidual_dwing_loading": dbase_dwing_loading / scale,
        "dresidual_dthrust_loading": -1.0,
    }


def cruise_dynamic_pressure_base(altitude, mach):
    """Return q, density ratio, and velocity for cruise residual equations."""

    values = cruise_dynamic_pressure_values(altitude, mach)
    return (
        values["dynamic_pressure"],
        values["density_ratio"],
        values["velocity"],
    )


def cruise_drag_residual_base(wing_loading, dynamic_pressure, cd0, aspect_ratio, oswald):
    """Return shared cruise drag residual before lapse/thrust terms."""

    induced_factor = math.pi * aspect_ratio * oswald
    return (
        dynamic_pressure * cd0 / wing_loading
        + wing_loading / (dynamic_pressure * induced_factor)
    )


def cruise_drag_residual_base_derivative(
    wing_loading,
    dynamic_pressure,
    cd0,
    aspect_ratio,
    oswald,
):
    """Return derivative of shared cruise drag residual with wing loading."""

    induced_factor = math.pi * aspect_ratio * oswald
    return (
        -dynamic_pressure * cd0 / wing_loading ** 2
        + 1.0 / (dynamic_pressure * induced_factor)
    )


def far25_climb_residual(
    wing_loading,
    thrust_loading,
    aircraft_class,
    req_type,
    cl,
    cd0,
    aspect_ratio,
    oswald,
    correction,
    gradient,
    ks,
    stall_velocity,
):
    """Return FAST shared FAR 25 climb residual for one scalar design point."""

    aircraft_class = aircraft_class.lower()

    if req_type == 0:
        effective_thrust_loading = thrust_loading

        if aircraft_class in ("turboprop", "piston"):
            effective_thrust_loading = 1.0 / (ks * stall_velocity * thrust_loading)

        base = (
            ks ** 2 * cd0 / cl
            + cl / ks ** 2 / math.pi / aspect_ratio / oswald
        )
        return correction * (base + gradient) - effective_thrust_loading

    converted_wing_loading = wing_loading * N_M2_TO_LBF_FT2

    if req_type == 1:
        vstall_ft_s = stall_velocity * convert_velocity(1.0, "m/s", "ft/s")
        q = 0.5 * RHO_SL_STD * RHO_SI_TO_ENGLISH * (vstall_ft_s * ks) ** 2
        base = cruise_drag_residual_base(
            converted_wing_loading,
            q,
            cd0,
            aspect_ratio,
            oswald,
        )
        effective_thrust_loading = thrust_loading

        if aircraft_class in ("turboprop", "piston"):
            velocity = convert_velocity(vstall_ft_s * ks, "ft/s", "m/s")
            effective_thrust_loading = 1.0 / (velocity * thrust_loading)

        return correction * (base + gradient) - effective_thrust_loading

    if req_type == 2:
        cl_eff = cl / ks ** 2
        qinf = converted_wing_loading / cl_eff
        vinf = (2.0 * qinf / (RHO_SL_STD * RHO_SI_TO_ENGLISH)) ** 0.5
        effective_thrust_loading = thrust_loading

        if aircraft_class in ("turboprop", "piston"):
            velocity = convert_velocity(vinf, "ft/s", "m/s")
            effective_thrust_loading = 1.0 / (velocity * thrust_loading)

        base = qinf / converted_wing_loading * (
            cd0 + cl_eff ** 2 / (math.pi * aspect_ratio * oswald)
        )
        return correction * (base + gradient) - effective_thrust_loading

    raise ValueError("FAR 25 climb ReqType must be 0, 1, or 2.")


def far25_climb_derivatives(
    wing_loading,
    thrust_loading,
    aircraft_class,
    req_type,
    cl,
    cd0,
    aspect_ratio,
    oswald,
    correction,
    ks,
    stall_velocity,
):
    """Return scalar derivatives for FAST shared FAR 25 climb residual."""

    aircraft_class = aircraft_class.lower()

    if req_type == 0:
        if aircraft_class in ("turboprop", "piston"):
            return {
                "dresidual_dwing_loading": 0.0,
                "dresidual_dthrust_loading": 1.0 / (
                    ks * stall_velocity * thrust_loading ** 2
                ),
            }

        return {
            "dresidual_dwing_loading": 0.0,
            "dresidual_dthrust_loading": -1.0,
        }

    converted_wing_loading = wing_loading * N_M2_TO_LBF_FT2

    if req_type == 1:
        vstall_ft_s = stall_velocity * convert_velocity(1.0, "m/s", "ft/s")
        q = 0.5 * RHO_SL_STD * RHO_SI_TO_ENGLISH * (vstall_ft_s * ks) ** 2
        dbase_dwing_loading = (
            cruise_drag_residual_base_derivative(
                converted_wing_loading,
                q,
                cd0,
                aspect_ratio,
                oswald,
            )
            * N_M2_TO_LBF_FT2
        )

        if aircraft_class in ("turboprop", "piston"):
            velocity = convert_velocity(vstall_ft_s * ks, "ft/s", "m/s")
            return {
                "dresidual_dwing_loading": correction * dbase_dwing_loading,
                "dresidual_dthrust_loading": 1.0 / (
                    velocity * thrust_loading ** 2
                ),
            }

        return {
            "dresidual_dwing_loading": correction * dbase_dwing_loading,
            "dresidual_dthrust_loading": -1.0,
        }

    if req_type == 2:
        if aircraft_class in ("turboprop", "piston"):
            cl_eff = cl / ks ** 2
            qinf = converted_wing_loading / cl_eff
            vinf_ft_s = (2.0 * qinf / (RHO_SL_STD * RHO_SI_TO_ENGLISH)) ** 0.5
            vinf_m_s = convert_velocity(vinf_ft_s, "ft/s", "m/s")
            dvinf_ft_s_dwing_loading = (
                0.5 * vinf_ft_s / converted_wing_loading * N_M2_TO_LBF_FT2
            )
            dvinf_m_s_dwing_loading = convert_velocity(
                dvinf_ft_s_dwing_loading,
                "ft/s",
                "m/s",
            )
            return {
                "dresidual_dwing_loading": (
                    dvinf_m_s_dwing_loading / (vinf_m_s ** 2 * thrust_loading)
                ),
                "dresidual_dthrust_loading": 1.0 / (
                    vinf_m_s * thrust_loading ** 2
                ),
            }

        return {
            "dresidual_dwing_loading": 0.0,
            "dresidual_dthrust_loading": -1.0,
        }

    raise ValueError("FAR 25 climb ReqType must be 0, 1, or 2.")


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
