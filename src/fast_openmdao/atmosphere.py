# src/fast_openmdao/atmosphere.py

"""OpenMDAO components for FAST atmosphere equations."""

import math

import openmdao.api as om


GRAVITY_SL = 9.80665
EARTH_RADIUS = 6.371009e6
STD_GRAVITY = 9.81
GAS_CONSTANT_AIR = 287.0


class Gravity(om.ExplicitComponent):
    """Compute FAST local gravitational acceleration from geometric altitude.

    Inputs:
        altitude: Geometric altitude in meters.

    Outputs:
        gravity: Local gravitational acceleration in m/s**2.

    Assumptions:
        Matches FAST-Python's spherical-Earth correction, which is used as a
        lightweight mission-energy gravity model.
    """

    def setup(self):
        self.add_input("altitude", val=0.0, units="m")
        self.add_output("gravity", val=GRAVITY_SL, units="m/s**2")
        self.declare_partials(of="gravity", wrt="altitude")

    def compute(self, inputs, outputs):
        altitude = inputs["altitude"][0]
        ratio = EARTH_RADIUS / (EARTH_RADIUS + altitude)
        outputs["gravity"] = GRAVITY_SL * ratio ** 2

    def compute_partials(self, inputs, partials):
        altitude = inputs["altitude"][0]
        denominator = EARTH_RADIUS + altitude
        partials["gravity", "altitude"] = (
            -2.0 * GRAVITY_SL * EARTH_RADIUS ** 2 / denominator ** 3
        )


class StandardAtmosphere(om.ExplicitComponent):
    """Compute FAST standard-atmosphere temperature, pressure, and density.

    Inputs:
        altitude: Geometric altitude in meters, valid from 0 to 100000 m.

    Outputs:
        temperature: Static temperature in K.
        pressure: Static pressure in Pa.
        density: Air density in kg/m**3.

    Assumptions:
        Constants, layer breakpoints, and base values match
        ``fast_python.atmosphere.standard_atmosphere``.
    """

    def setup(self):
        self.add_input("altitude", val=0.0, units="m")
        self.add_output("temperature", val=288.15, units="K")
        self.add_output("pressure", val=101300.0, units="Pa")
        self.add_output("density", val=1.225, units="kg/m**3")
        self.declare_partials(of="temperature", wrt="altitude")
        self.declare_partials(of="pressure", wrt="altitude")
        self.declare_partials(of="density", wrt="altitude")

    def compute(self, inputs, outputs):
        altitude = inputs["altitude"][0]
        layer = atmosphere_layer(altitude)

        outputs["temperature"] = layer["temperature"]
        outputs["pressure"] = layer["pressure"]
        outputs["density"] = layer["density"]

    def compute_partials(self, inputs, partials):
        altitude = inputs["altitude"][0]
        layer = atmosphere_layer(altitude)

        partials["temperature", "altitude"] = layer["dtemperature_daltitude"]
        partials["pressure", "altitude"] = layer["dpressure_daltitude"]
        partials["density", "altitude"] = layer["ddensity_daltitude"]


def atmosphere_layer(altitude):
    """Return atmosphere values and altitude derivatives for one scalar altitude."""

    if altitude < 0.0 or altitude > 100000.0:
        raise ValueError("Altitude must be between 0 and 100000 m.")

    if altitude < 11000.0:
        return gradient_layer(altitude, 0.0, 288.15, 101300.0, -0.0065)

    if altitude < 20000.0:
        return isothermal_layer(altitude, 11000.0, 216.65, 2.2609e4)

    if altitude < 32000.0:
        return gradient_layer(altitude, 20000.0, 216.65, 5.4731e3, 0.0010)

    if altitude < 47000.0:
        return gradient_layer(altitude, 32000.0, 228.65, 866.8940, 0.0028)

    if altitude < 51000.0:
        return isothermal_layer(altitude, 47000.0, 270.65, 110.6427)

    if altitude < 71000.0:
        return gradient_layer(altitude, 51000.0, 270.65, 66.7260, -0.0028)

    if altitude < 85000.0:
        return gradient_layer(altitude, 71000.0, 214.65, 3.9401, -0.0020)

    return isothermal_layer(altitude, 85000.0, 186.65, 0.3615)


def gradient_layer(altitude, base_altitude, base_temperature, base_pressure, lapse):
    """Return atmosphere values and derivatives for a non-isothermal layer."""

    temperature = base_temperature + lapse * (altitude - base_altitude)
    exponent = -STD_GRAVITY / (GAS_CONSTANT_AIR * lapse)
    pressure = base_pressure * (temperature / base_temperature) ** exponent
    return atmosphere_values(temperature, pressure, lapse)


def isothermal_layer(altitude, base_altitude, temperature, base_pressure):
    """Return atmosphere values and derivatives for an isothermal layer."""

    pressure = base_pressure * math.exp(
        -STD_GRAVITY * (altitude - base_altitude)
        / (GAS_CONSTANT_AIR * temperature)
    )
    return atmosphere_values(temperature, pressure, 0.0)


def atmosphere_values(temperature, pressure, dtemperature_daltitude):
    """Return atmosphere values with closed-form pressure and density partials."""

    density = pressure / (GAS_CONSTANT_AIR * temperature)
    dpressure_daltitude = -STD_GRAVITY * pressure / (GAS_CONSTANT_AIR * temperature)
    ddensity_daltitude = density * (
        dpressure_daltitude / pressure
        - dtemperature_daltitude / temperature
    )

    return {
        "temperature": temperature,
        "pressure": pressure,
        "density": density,
        "dtemperature_daltitude": dtemperature_daltitude,
        "dpressure_daltitude": dpressure_daltitude,
        "ddensity_daltitude": ddensity_daltitude,
    }
