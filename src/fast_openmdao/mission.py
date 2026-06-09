# src/fast_openmdao/mission.py

"""OpenMDAO components for FAST mission primitive equations."""

import openmdao.api as om

from fast_openmdao.atmosphere import GAS_CONSTANT_AIR, atmosphere_layer
from fast_python.atmosphere import standard_atmosphere


GAMMA_AIR = 1.4
RHO_SL_STD = standard_atmosphere(0.0)[2]


class FlightConditions(om.ExplicitComponent):
    """Compute FAST flight conditions from altitude and speed.

    Inputs:
        altitude: Geometric altitude in meters.
        disa: Standard-atmosphere temperature deviation in K.
        velocity: TAS/EAS in m/s, or Mach number when velocity_type is Mach.

    Outputs:
        eas: Equivalent airspeed in m/s.
        tas: True airspeed in m/s.
        mach: Mach number.
        temperature: Static temperature including DISA in K.
        pressure: Static pressure in Pa.
        density: Air density in kg/m**3.
        viscosity: FAST polynomial air viscosity.

    Assumptions:
        ``velocity_type`` is a discrete option matching FAST-Python:
        TAS, EAS, or Mach.
    """

    def initialize(self):
        self.options.declare("velocity_type", default="TAS")

    def setup(self):
        self.add_input("altitude", val=0.0, units="m")
        self.add_input("disa", val=0.0, units="K")
        self.add_input("velocity", val=100.0)
        self.add_output("eas", val=100.0, units="m/s")
        self.add_output("tas", val=100.0, units="m/s")
        self.add_output("mach", val=0.3)
        self.add_output("temperature", val=288.15, units="K")
        self.add_output("pressure", val=101300.0, units="Pa")
        self.add_output("density", val=RHO_SL_STD, units="kg/m**3")
        self.add_output("viscosity", val=1.81e-5)
        self.declare_partials(of="*", wrt=["altitude", "disa", "velocity"])

    def compute(self, inputs, outputs):
        values = flight_condition_values(
            inputs["altitude"][0],
            inputs["disa"][0],
            self.options["velocity_type"],
            inputs["velocity"][0],
        )

        for name in flight_condition_output_names():
            outputs[name] = values[name]

    def compute_partials(self, inputs, partials):
        values = flight_condition_values(
            inputs["altitude"][0],
            inputs["disa"][0],
            self.options["velocity_type"],
            inputs["velocity"][0],
        )

        inputs_names = ("altitude", "disa", "velocity")

        for output_name in flight_condition_output_names():
            for input_name in inputs_names:
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


def flight_condition_values(altitude, disa, velocity_type, velocity):
    """Return FAST flight-condition values and analytical derivatives."""

    atmosphere = atmosphere_layer(altitude)
    temperature = atmosphere["temperature"] + disa
    pressure = atmosphere["pressure"]
    density = pressure / (GAS_CONSTANT_AIR * temperature)
    sound_speed = (GAMMA_AIR * GAS_CONSTANT_AIR * temperature) ** 0.5
    viscosity = viscosity_polynomial(altitude)

    dtemperature_daltitude = atmosphere["dtemperature_daltitude"]
    dpressure_daltitude = atmosphere["dpressure_daltitude"]
    ddensity_daltitude = density * (
        dpressure_daltitude / pressure
        - dtemperature_daltitude / temperature
    )
    ddensity_ddisa = -density / temperature
    dsound_speed_daltitude = 0.5 * sound_speed / temperature * dtemperature_daltitude
    dsound_speed_ddisa = 0.5 * sound_speed / temperature

    values = {
        "temperature": temperature,
        "pressure": pressure,
        "density": density,
        "viscosity": viscosity,
        "dtemperature_daltitude": dtemperature_daltitude,
        "dtemperature_ddisa": 1.0,
        "dtemperature_dvelocity": 0.0,
        "dpressure_daltitude": dpressure_daltitude,
        "dpressure_ddisa": 0.0,
        "dpressure_dvelocity": 0.0,
        "ddensity_daltitude": ddensity_daltitude,
        "ddensity_ddisa": ddensity_ddisa,
        "ddensity_dvelocity": 0.0,
        "dviscosity_daltitude": viscosity_derivative(altitude),
        "dviscosity_ddisa": 0.0,
        "dviscosity_dvelocity": 0.0,
    }

    add_velocity_outputs(
        values,
        velocity_type,
        velocity,
        density,
        sound_speed,
        ddensity_daltitude,
        ddensity_ddisa,
        dsound_speed_daltitude,
        dsound_speed_ddisa,
    )
    return values


def flight_condition_output_names():
    """Return output variable names for the flight-condition component."""

    return (
        "eas",
        "tas",
        "mach",
        "temperature",
        "pressure",
        "density",
        "viscosity",
    )


def add_velocity_outputs(
    values,
    velocity_type,
    velocity,
    density,
    sound_speed,
    ddensity_daltitude,
    ddensity_ddisa,
    dsound_speed_daltitude,
    dsound_speed_ddisa,
):
    """Add speed outputs and derivatives to a flight-condition value mapping."""

    vel_type = velocity_type.lower()

    if vel_type == "tas":
        add_tas_outputs(
            values,
            velocity,
            density,
            sound_speed,
            ddensity_daltitude,
            ddensity_ddisa,
            dsound_speed_daltitude,
            dsound_speed_ddisa,
        )
        return

    if vel_type == "eas":
        add_eas_outputs(
            values,
            velocity,
            density,
            sound_speed,
            ddensity_daltitude,
            ddensity_ddisa,
            dsound_speed_daltitude,
            dsound_speed_ddisa,
        )
        return

    if vel_type == "mach":
        add_mach_outputs(
            values,
            velocity,
            density,
            sound_speed,
            ddensity_daltitude,
            ddensity_ddisa,
            dsound_speed_daltitude,
            dsound_speed_ddisa,
        )
        return

    raise ValueError("velocity_type must be TAS, EAS, or Mach.")


def add_tas_outputs(
    values,
    tas,
    density,
    sound_speed,
    ddensity_daltitude,
    ddensity_ddisa,
    dsound_speed_daltitude,
    dsound_speed_ddisa,
):
    """Add outputs for a TAS input."""

    density_ratio_root = (density / RHO_SL_STD) ** 0.5
    eas = tas * density_ratio_root
    mach = tas / sound_speed
    values.update(
        speed_derivatives(
            eas,
            tas,
            mach,
            density,
            sound_speed,
            ddensity_daltitude,
            ddensity_ddisa,
            dsound_speed_daltitude,
            dsound_speed_ddisa,
            density_ratio_root,
            1.0,
            1.0 / sound_speed,
        )
    )


def add_eas_outputs(
    values,
    eas,
    density,
    sound_speed,
    ddensity_daltitude,
    ddensity_ddisa,
    dsound_speed_daltitude,
    dsound_speed_ddisa,
):
    """Add outputs for an EAS input."""

    inverse_density_ratio_root = (RHO_SL_STD / density) ** 0.5
    tas = eas * inverse_density_ratio_root
    mach = tas / sound_speed
    dtas_dvelocity = inverse_density_ratio_root
    dmach_dvelocity = dtas_dvelocity / sound_speed
    values.update(
        speed_derivatives(
            eas,
            tas,
            mach,
            density,
            sound_speed,
            ddensity_daltitude,
            ddensity_ddisa,
            dsound_speed_daltitude,
            dsound_speed_ddisa,
            1.0,
            dtas_dvelocity,
            dmach_dvelocity,
        )
    )
    values["deas_daltitude"] = 0.0
    values["deas_ddisa"] = 0.0


def add_mach_outputs(
    values,
    mach,
    density,
    sound_speed,
    ddensity_daltitude,
    ddensity_ddisa,
    dsound_speed_daltitude,
    dsound_speed_ddisa,
):
    """Add outputs for a Mach input."""

    tas = mach * sound_speed
    density_ratio_root = (density / RHO_SL_STD) ** 0.5
    eas = tas * density_ratio_root
    values.update(
        speed_derivatives(
            eas,
            tas,
            mach,
            density,
            sound_speed,
            ddensity_daltitude,
            ddensity_ddisa,
            dsound_speed_daltitude,
            dsound_speed_ddisa,
            sound_speed * density_ratio_root,
            sound_speed,
            1.0,
        )
    )


def speed_derivatives(
    eas,
    tas,
    mach,
    density,
    sound_speed,
    ddensity_daltitude,
    ddensity_ddisa,
    dsound_speed_daltitude,
    dsound_speed_ddisa,
    deas_dvelocity,
    dtas_dvelocity,
    dmach_dvelocity,
):
    """Return shared speed values and derivative entries."""

    dtas_daltitude = 0.0
    dtas_ddisa = 0.0

    if dtas_dvelocity != 1.0:
        dtas_daltitude = -0.5 * tas * ddensity_daltitude / density
        dtas_ddisa = -0.5 * tas * ddensity_ddisa / density

    if dmach_dvelocity == 1.0:
        dtas_daltitude = mach * dsound_speed_daltitude
        dtas_ddisa = mach * dsound_speed_ddisa
        dmach_daltitude = 0.0
        dmach_ddisa = 0.0
    else:
        dmach_daltitude = (
            dtas_daltitude / sound_speed
            - mach * dsound_speed_daltitude / sound_speed
        )
        dmach_ddisa = (
            dtas_ddisa / sound_speed
            - mach * dsound_speed_ddisa / sound_speed
        )

    deas_daltitude = eas * (
        dtas_daltitude / tas
        + 0.5 * ddensity_daltitude / density
    )
    deas_ddisa = eas * (
        dtas_ddisa / tas
        + 0.5 * ddensity_ddisa / density
    )

    return {
        "eas": eas,
        "tas": tas,
        "mach": mach,
        "deas_daltitude": deas_daltitude,
        "deas_ddisa": deas_ddisa,
        "deas_dvelocity": deas_dvelocity,
        "dtas_daltitude": dtas_daltitude,
        "dtas_ddisa": dtas_ddisa,
        "dtas_dvelocity": dtas_dvelocity,
        "dmach_daltitude": dmach_daltitude,
        "dmach_ddisa": dmach_ddisa,
        "dmach_dvelocity": dmach_dvelocity,
    }


def viscosity_polynomial(altitude):
    """Return FAST air-viscosity polynomial value."""

    return (
        -1.51e-19 * altitude ** 3
        + 1.64e-14 * altitude ** 2
        - 4.67e-10 * altitude
        + 1.81e-5
    )


def viscosity_derivative(altitude):
    """Return derivative of FAST air-viscosity polynomial with altitude."""

    return (
        -3.0 * 1.51e-19 * altitude ** 2
        + 2.0 * 1.64e-14 * altitude
        - 4.67e-10
    )
