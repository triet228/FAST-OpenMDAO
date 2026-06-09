# src/fast_openmdao/mission.py

"""OpenMDAO components for FAST mission primitive equations."""

import numpy as np
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


class CruiseTimeTargetDistance(om.ExplicitComponent):
    """Convert FAST cruise time target in minutes to distance in meters.

    Inputs:
        altitude_begin: Beginning altitude in meters.
        altitude_end: Ending altitude in meters.
        velocity_begin: Beginning TAS/EAS speed in m/s or Mach number.
        velocity_end: Ending TAS/EAS speed in m/s or Mach number.
        target_minutes: Cruise time target in minutes.

    Outputs:
        distance: Cruise distance in meters.

    Assumptions:
        Beginning and ending velocity types are discrete options matching
        FAST-Python's TAS, EAS, or Mach labels.
    """

    def initialize(self):
        self.options.declare("type_begin", default="TAS")
        self.options.declare("type_end", default="TAS")

    def setup(self):
        self.add_input("altitude_begin", val=1000.0, units="m")
        self.add_input("altitude_end", val=1000.0, units="m")
        self.add_input("velocity_begin", val=100.0)
        self.add_input("velocity_end", val=100.0)
        self.add_input("target_minutes", val=10.0, units="min")
        self.add_output("distance", val=60000.0, units="m")
        self.declare_partials(
            of="distance",
            wrt=[
                "altitude_begin",
                "altitude_end",
                "velocity_begin",
                "velocity_end",
                "target_minutes",
            ],
        )

    def compute(self, inputs, outputs):
        values = cruise_time_target_distance_values(
            inputs["altitude_begin"][0],
            inputs["altitude_end"][0],
            inputs["velocity_begin"][0],
            inputs["velocity_end"][0],
            self.options["type_begin"],
            self.options["type_end"],
            inputs["target_minutes"][0],
        )
        outputs["distance"] = values["distance"]

    def compute_partials(self, inputs, partials):
        values = cruise_time_target_distance_values(
            inputs["altitude_begin"][0],
            inputs["altitude_end"][0],
            inputs["velocity_begin"][0],
            inputs["velocity_end"][0],
            self.options["type_begin"],
            self.options["type_end"],
            inputs["target_minutes"][0],
        )

        for name in (
            "altitude_begin",
            "altitude_end",
            "velocity_begin",
            "velocity_end",
            "target_minutes",
        ):
            partials["distance", name] = values[f"ddistance_d{name}"]


class CruiseBreguetEfficiencyTriplet(om.ExplicitComponent):
    """Compute FAST CruiseBRE eta1, eta2, and eta3 coefficients.

    Inputs:
        propulsive_efficiency: Propulsive efficiency.
        electric_motor_efficiency: Electric motor efficiency.
        electric_generator_efficiency: Electric generator efficiency.
        gas_turbine_efficiency: Gas turbine efficiency.

    Outputs:
        eta1, eta2, eta3: FAST Breguet cruise coefficients for the selected
            architecture.

    Assumptions:
        ``architecture`` is a discrete FAST CruiseBRE architecture label:
        AC, PHE, SHE, or TE. The component is differentiable within a fixed
        architecture; switching architectures is a discrete model change.
    """

    def initialize(self):
        self.options.declare("architecture", default="AC")

    def setup(self):
        self.add_input("propulsive_efficiency", val=0.85)
        self.add_input("electric_motor_efficiency", val=0.95)
        self.add_input("electric_generator_efficiency", val=0.92)
        self.add_input("gas_turbine_efficiency", val=0.35)
        self.add_output("eta1", val=0.35)
        self.add_output("eta2", val=0.0)
        self.add_output("eta3", val=0.85)
        self.declare_partials(of=["eta1", "eta2", "eta3"], wrt="*")

    def compute(self, inputs, outputs):
        values = cruise_breguet_efficiency_values(
            self.options["architecture"],
            inputs["propulsive_efficiency"][0],
            inputs["electric_motor_efficiency"][0],
            inputs["electric_generator_efficiency"][0],
            inputs["gas_turbine_efficiency"][0],
        )
        outputs["eta1"] = values["eta1"]
        outputs["eta2"] = values["eta2"]
        outputs["eta3"] = values["eta3"]

    def compute_partials(self, inputs, partials):
        values = cruise_breguet_efficiency_values(
            self.options["architecture"],
            inputs["propulsive_efficiency"][0],
            inputs["electric_motor_efficiency"][0],
            inputs["electric_generator_efficiency"][0],
            inputs["gas_turbine_efficiency"][0],
        )

        for output_name in ("eta1", "eta2", "eta3"):
            for input_name in breguet_efficiency_input_names():
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


class CruiseBreguetSourceEnergy(om.ExplicitComponent):
    """Map aggregate Breguet fuel and battery energy onto source columns.

    Inputs:
        initial_source_energy: Source energy at the beginning row.
        initial_source_energy_left: Remaining source energy at the beginning
            row.
        fuel_energy: Aggregate fuel energy history.
        battery_energy: Aggregate battery energy history.

    Outputs:
        source_energy: Per-source used energy matrix.
        source_energy_left: Per-source remaining energy matrix.

    Assumptions:
        ``src_type`` is a fixed FAST source-type vector where 1 marks fuel
        sources and 0 marks battery sources. Aggregate energy deltas are shared
        evenly across matching source columns, matching FAST-Python
        ``cruise_breguet_source_energy``.
    """

    def initialize(self):
        self.options.declare("src_type", default=(1.0, 0.0))
        self.options.declare("npoint", default=3)

    def setup(self):
        src_type = np.asarray(self.options["src_type"], dtype=float).reshape(-1)
        nsrc = src_type.size
        npoint = self.options["npoint"]
        self.add_input("initial_source_energy", val=np.zeros(nsrc), units="J")
        self.add_input("initial_source_energy_left", val=np.ones(nsrc), units="J")
        self.add_input("fuel_energy", val=np.zeros(npoint), units="J")
        self.add_input("battery_energy", val=np.zeros(npoint), units="J")
        self.add_output(
            "source_energy",
            val=np.zeros((npoint, nsrc)),
            units="J",
        )
        self.add_output(
            "source_energy_left",
            val=np.zeros((npoint, nsrc)),
            units="J",
        )
        self.declare_partials(of="source_energy", wrt="*")
        self.declare_partials(of="source_energy_left", wrt="*")

    def compute(self, inputs, outputs):
        values = cruise_breguet_source_energy_values(
            self.options["src_type"],
            self.options["npoint"],
            inputs["initial_source_energy"],
            inputs["initial_source_energy_left"],
            inputs["fuel_energy"],
            inputs["battery_energy"],
        )
        outputs["source_energy"] = values["source_energy"]
        outputs["source_energy_left"] = values["source_energy_left"]

    def compute_partials(self, inputs, partials):
        values = cruise_breguet_source_energy_values(
            self.options["src_type"],
            self.options["npoint"],
            inputs["initial_source_energy"],
            inputs["initial_source_energy_left"],
            inputs["fuel_energy"],
            inputs["battery_energy"],
        )
        for output_name in ("source_energy", "source_energy_left"):
            for input_name in (
                "initial_source_energy",
                "initial_source_energy_left",
                "fuel_energy",
                "battery_energy",
            ):
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


class CruiseBreguetPropulsiveEfficiency(om.ExplicitComponent):
    """Select FAST CruiseBRE propulsive efficiency from fixed storage path."""

    def initialize(self):
        self.options.declare("source", default="propulsion")

    def setup(self):
        self.add_input("propulsion_propulsive_efficiency", val=0.84)
        self.add_input("power_propeller_efficiency", val=0.82)
        self.add_output("propulsive_efficiency", val=0.84)
        if self.options["source"] == "propulsion":
            self.declare_partials(
                of="propulsive_efficiency",
                wrt="propulsion_propulsive_efficiency",
            )
        else:
            self.declare_partials(
                of="propulsive_efficiency",
                wrt="power_propeller_efficiency",
            )

    def compute(self, inputs, outputs):
        values = cruise_breguet_propulsive_efficiency_values(
            self.options["source"],
            inputs["propulsion_propulsive_efficiency"][0],
            inputs["power_propeller_efficiency"][0],
        )
        outputs["propulsive_efficiency"] = values["propulsive_efficiency"]

    def compute_partials(self, inputs, partials):
        values = cruise_breguet_propulsive_efficiency_values(
            self.options["source"],
            inputs["propulsion_propulsive_efficiency"][0],
            inputs["power_propeller_efficiency"][0],
        )
        if self.options["source"] == "propulsion":
            partials[
                "propulsive_efficiency",
                "propulsion_propulsive_efficiency",
            ] = values["dpropulsive_efficiency_dpropulsion_propulsive_efficiency"]
        else:
            partials[
                "propulsive_efficiency",
                "power_propeller_efficiency",
            ] = values["dpropulsive_efficiency_dpower_propeller_efficiency"]


class CruiseBreguetPowerSplit(om.ExplicitComponent):
    """Select FAST CruiseBRE power split from fixed storage path."""

    def initialize(self):
        self.options.declare("source", default="phi")

    def setup(self):
        self.add_input("phi_cruise", val=0.3)
        self.add_input("lambda_down_cruise", val=0.2)
        self.add_output("power_split", val=0.3)
        if self.options["source"] == "phi":
            self.declare_partials(of="power_split", wrt="phi_cruise")
        elif self.options["source"] == "lambda_down":
            self.declare_partials(of="power_split", wrt="lambda_down_cruise")

    def compute(self, inputs, outputs):
        values = cruise_breguet_power_split_values(
            self.options["source"],
            inputs["phi_cruise"][0],
            inputs["lambda_down_cruise"][0],
        )
        outputs["power_split"] = values["power_split"]

    def compute_partials(self, inputs, partials):
        values = cruise_breguet_power_split_values(
            self.options["source"],
            inputs["phi_cruise"][0],
            inputs["lambda_down_cruise"][0],
        )
        if self.options["source"] == "phi":
            partials["power_split", "phi_cruise"] = values[
                "dpower_split_dphi_cruise"
            ]
        elif self.options["source"] == "lambda_down":
            partials["power_split", "lambda_down_cruise"] = values[
                "dpower_split_dlambda_down_cruise"
            ]


class InitialEnergyRemaining(om.ExplicitComponent):
    """Initialize FAST mission source-energy remaining history.

    Inputs:
        fuel_specific_energy: Fuel specific energy in J/kg.
        battery_specific_energy: Battery specific energy in J/kg.
        fuel_weight: Fuel source weight in kg.
        battery_weight: Battery source weights in kg.

    Outputs:
        source_energy_left: Initial remaining source energy matrix in J.

    Assumptions:
        Source types are fixed architecture data where 1 marks fuel and 0
        marks battery. The initialized value is repeated for every mission
        point, matching FAST-Python's first-segment history setup.
    """

    def initialize(self):
        self.options.declare("src_type", default=(1.0, 0.0))
        self.options.declare("npoint", default=1)

    def setup(self):
        src_type = np.asarray(self.options["src_type"], dtype=float).reshape(-1)
        npoint = self.options["npoint"]
        nsrc = src_type.size
        nfuel = max(1, np.count_nonzero(src_type == 1))
        nbattery = max(1, np.count_nonzero(src_type == 0))
        self.add_input("fuel_specific_energy", val=43200000.0, units="J/kg")
        self.add_input("battery_specific_energy", val=1000.0, units="J/kg")
        self.add_input("fuel_weight", val=np.ones(nfuel), units="kg")
        self.add_input("battery_weight", val=np.ones(nbattery), units="kg")
        self.add_output(
            "source_energy_left",
            val=np.zeros((npoint, nsrc)),
            units="J",
        )
        self.declare_partials(of="source_energy_left", wrt="*")

    def compute(self, inputs, outputs):
        values = initial_energy_remaining_values(
            self.options["src_type"],
            self.options["npoint"],
            inputs["fuel_specific_energy"][0],
            inputs["battery_specific_energy"][0],
            inputs["fuel_weight"],
            inputs["battery_weight"],
        )
        outputs["source_energy_left"] = values["source_energy_left"]

    def compute_partials(self, inputs, partials):
        values = initial_energy_remaining_values(
            self.options["src_type"],
            self.options["npoint"],
            inputs["fuel_specific_energy"][0],
            inputs["battery_specific_energy"][0],
            inputs["fuel_weight"],
            inputs["battery_weight"],
        )
        for input_name in (
            "fuel_specific_energy",
            "battery_specific_energy",
            "fuel_weight",
            "battery_weight",
        ):
            partials["source_energy_left", input_name] = values[
                f"dsource_energy_left_d{input_name}"
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


def initial_energy_remaining_values(
    src_type,
    npoint,
    fuel_specific_energy,
    battery_specific_energy,
    fuel_weight,
    battery_weight,
):
    """Return initial source-energy-left matrix and derivative blocks."""

    src_type = np.asarray(src_type, dtype=float).reshape(-1)
    fuel_weight = np.asarray(fuel_weight, dtype=float).reshape(-1)
    battery_weight = np.asarray(battery_weight, dtype=float).reshape(-1)
    nsrc = src_type.size
    output_size = npoint * nsrc
    source_energy_left = np.zeros((npoint, nsrc))
    dfuel_specific = np.zeros((output_size, 1))
    dbattery_specific = np.zeros((output_size, 1))
    dfuel_weight = np.zeros((output_size, fuel_weight.size))
    dbattery_weight = np.zeros((output_size, battery_weight.size))
    fuel_columns = np.where(src_type == 1)[0]
    battery_columns = np.where(src_type == 0)[0]

    for local_index, column in enumerate(fuel_columns):
        weight_index = min(local_index, fuel_weight.size - 1)
        value = fuel_specific_energy * fuel_weight[weight_index]
        source_energy_left[:, column] = value

        for point in range(npoint):
            output_index = point * nsrc + column
            dfuel_specific[output_index, 0] = fuel_weight[weight_index]
            dfuel_weight[output_index, weight_index] = fuel_specific_energy

    for local_index, column in enumerate(battery_columns):
        weight_index = min(local_index, battery_weight.size - 1)
        value = battery_specific_energy * battery_weight[weight_index]
        source_energy_left[:, column] = value

        for point in range(npoint):
            output_index = point * nsrc + column
            dbattery_specific[output_index, 0] = battery_weight[weight_index]
            dbattery_weight[output_index, weight_index] = battery_specific_energy

    return {
        "source_energy_left": source_energy_left,
        "dsource_energy_left_dfuel_specific_energy": dfuel_specific,
        "dsource_energy_left_dbattery_specific_energy": dbattery_specific,
        "dsource_energy_left_dfuel_weight": dfuel_weight,
        "dsource_energy_left_dbattery_weight": dbattery_weight,
    }


def cruise_breguet_source_energy_values(
    src_type,
    npoint,
    initial_source_energy,
    initial_source_energy_left,
    fuel_energy,
    battery_energy,
):
    """Return Breguet source-energy matrices and exact derivative blocks."""

    src_type = np.asarray(src_type, dtype=float).reshape(-1)
    initial_source_energy = np.asarray(initial_source_energy, dtype=float).reshape(-1)
    initial_source_energy_left = np.asarray(
        initial_source_energy_left,
        dtype=float,
    ).reshape(-1)
    fuel_energy = np.asarray(fuel_energy, dtype=float).reshape(-1)
    battery_energy = np.asarray(battery_energy, dtype=float).reshape(-1)
    nsrc = src_type.size
    source_energy = np.tile(initial_source_energy, (npoint, 1))
    source_energy_left = np.tile(initial_source_energy_left, (npoint, 1))

    apply_breguet_source_delta(
        source_energy,
        source_energy_left,
        np.where(src_type == 1)[0],
        fuel_energy - fuel_energy[0],
    )
    apply_breguet_source_delta(
        source_energy,
        source_energy_left,
        np.where(src_type == 0)[0],
        battery_energy - battery_energy[0],
    )

    values = {
        "source_energy": source_energy,
        "source_energy_left": source_energy_left,
    }
    values.update(
        cruise_breguet_source_energy_partials(
            src_type,
            npoint,
            nsrc,
        )
    )
    return values


def apply_breguet_source_delta(source_energy, source_energy_left, columns, delta):
    """Apply one aggregate energy delta to selected source columns."""

    if len(columns) == 0:
        return

    share = delta / len(columns)

    for column in columns:
        source_energy[:, column] = source_energy[0, column] + share
        source_energy_left[:, column] = source_energy_left[0, column] - share


def cruise_breguet_source_energy_partials(src_type, npoint, nsrc):
    """Return derivative matrices for Breguet source-energy allocation."""

    source_size = npoint * nsrc
    energy_initial = np.zeros((source_size, nsrc))
    energy_left_initial = np.zeros((source_size, nsrc))
    left_energy_initial = np.zeros((source_size, nsrc))
    left_energy_left_initial = np.zeros((source_size, nsrc))
    energy_fuel = np.zeros((source_size, npoint))
    energy_battery = np.zeros((source_size, npoint))
    left_fuel = np.zeros((source_size, npoint))
    left_battery = np.zeros((source_size, npoint))

    for row in range(npoint):
        for column in range(nsrc):
            output_index = row * nsrc + column
            energy_initial[output_index, column] = 1.0
            left_energy_left_initial[output_index, column] = 1.0

    add_breguet_source_delta_partials(
        energy_fuel,
        left_fuel,
        np.where(src_type == 1)[0],
        npoint,
        nsrc,
    )
    add_breguet_source_delta_partials(
        energy_battery,
        left_battery,
        np.where(src_type == 0)[0],
        npoint,
        nsrc,
    )

    return {
        "dsource_energy_dinitial_source_energy": energy_initial,
        "dsource_energy_dinitial_source_energy_left": energy_left_initial,
        "dsource_energy_dfuel_energy": energy_fuel,
        "dsource_energy_dbattery_energy": energy_battery,
        "dsource_energy_left_dinitial_source_energy": left_energy_initial,
        "dsource_energy_left_dinitial_source_energy_left": left_energy_left_initial,
        "dsource_energy_left_dfuel_energy": left_fuel,
        "dsource_energy_left_dbattery_energy": left_battery,
    }


def add_breguet_source_delta_partials(energy_partials, left_partials, columns, npoint, nsrc):
    """Add aggregate-delta derivative entries for selected source columns."""

    if len(columns) == 0:
        return

    scale = 1.0 / len(columns)

    for row in range(npoint):
        for column in columns:
            output_index = row * nsrc + column
            energy_partials[output_index, row] += scale
            energy_partials[output_index, 0] -= scale
            left_partials[output_index, row] -= scale
            left_partials[output_index, 0] += scale


def cruise_breguet_efficiency_values(
    architecture,
    propulsive_efficiency,
    electric_motor_efficiency,
    electric_generator_efficiency,
    gas_turbine_efficiency,
):
    """Return FAST CruiseBRE efficiency coefficients and derivatives."""

    arch = architecture.upper()
    values = zero_breguet_efficiency_derivatives()

    if arch == "AC":
        values.update(
            {
                "eta1": gas_turbine_efficiency,
                "eta2": 0.0,
                "eta3": propulsive_efficiency,
                "deta1_dgas_turbine_efficiency": 1.0,
                "deta3_dpropulsive_efficiency": 1.0,
            }
        )
        return values

    if arch == "PHE":
        values.update(
            {
                "eta1": gas_turbine_efficiency,
                "eta2": electric_motor_efficiency,
                "eta3": propulsive_efficiency,
                "deta1_dgas_turbine_efficiency": 1.0,
                "deta2_delectric_motor_efficiency": 1.0,
                "deta3_dpropulsive_efficiency": 1.0,
            }
        )
        return values

    if arch == "SHE":
        values.update(
            {
                "eta1": gas_turbine_efficiency * electric_generator_efficiency,
                "eta2": 1.0,
                "eta3": electric_motor_efficiency * propulsive_efficiency,
                "deta1_dgas_turbine_efficiency": electric_generator_efficiency,
                "deta1_delectric_generator_efficiency": gas_turbine_efficiency,
                "deta3_delectric_motor_efficiency": propulsive_efficiency,
                "deta3_dpropulsive_efficiency": electric_motor_efficiency,
            }
        )
        return values

    if arch == "TE":
        values.update(
            {
                "eta1": gas_turbine_efficiency * electric_generator_efficiency,
                "eta2": 0.0,
                "eta3": electric_motor_efficiency * propulsive_efficiency,
                "deta1_dgas_turbine_efficiency": electric_generator_efficiency,
                "deta1_delectric_generator_efficiency": gas_turbine_efficiency,
                "deta3_delectric_motor_efficiency": propulsive_efficiency,
                "deta3_dpropulsive_efficiency": electric_motor_efficiency,
            }
        )
        return values

    raise ValueError("architecture must be AC, PHE, SHE, or TE.")


def cruise_breguet_propulsive_efficiency_values(
    source,
    propulsion_propulsive_efficiency,
    power_propeller_efficiency,
):
    """Return fixed-path CruiseBRE propulsive efficiency selection."""

    values = {
        "dpropulsive_efficiency_dpropulsion_propulsive_efficiency": 0.0,
        "dpropulsive_efficiency_dpower_propeller_efficiency": 0.0,
    }

    if source == "propulsion":
        values["propulsive_efficiency"] = propulsion_propulsive_efficiency
        values["dpropulsive_efficiency_dpropulsion_propulsive_efficiency"] = 1.0
        return values

    if source == "power":
        values["propulsive_efficiency"] = power_propeller_efficiency
        values["dpropulsive_efficiency_dpower_propeller_efficiency"] = 1.0
        return values

    raise ValueError(
        "CruiseBreguetPropulsiveEfficiency source must be propulsion or power."
    )


def cruise_breguet_power_split_values(source, phi_cruise, lambda_down_cruise):
    """Return fixed-path CruiseBRE power split selection."""

    values = {
        "dpower_split_dphi_cruise": 0.0,
        "dpower_split_dlambda_down_cruise": 0.0,
    }

    if source == "phi":
        values["power_split"] = phi_cruise
        values["dpower_split_dphi_cruise"] = 1.0
        return values

    if source == "lambda_down":
        values["power_split"] = lambda_down_cruise
        values["dpower_split_dlambda_down_cruise"] = 1.0
        return values

    if source == "zero":
        values["power_split"] = 0.0
        return values

    raise ValueError("CruiseBreguetPowerSplit source must be phi, lambda_down, or zero.")


def breguet_efficiency_input_names():
    """Return CruiseBRE efficiency input names."""

    return (
        "propulsive_efficiency",
        "electric_motor_efficiency",
        "electric_generator_efficiency",
        "gas_turbine_efficiency",
    )


def zero_breguet_efficiency_derivatives():
    """Return zero-initialized CruiseBRE efficiency derivative mapping."""

    values = {
        "eta1": 0.0,
        "eta2": 0.0,
        "eta3": 0.0,
    }

    for output_name in ("eta1", "eta2", "eta3"):
        for input_name in breguet_efficiency_input_names():
            values[f"d{output_name}_d{input_name}"] = 0.0

    return values


def cruise_time_target_distance_values(
    altitude_begin,
    altitude_end,
    velocity_begin,
    velocity_end,
    type_begin,
    type_end,
    target_minutes,
):
    """Return cruise time-target distance and analytical derivatives."""

    begin = flight_condition_values(
        altitude_begin,
        0.0,
        type_begin,
        velocity_begin,
    )
    end = flight_condition_values(
        altitude_end,
        0.0,
        type_end,
        velocity_end,
    )
    tas_begin = begin["tas"]
    tas_end = end["tas"]
    dtime = 60.0 * target_minutes
    distance = 0.5 * (tas_begin + tas_end) * dtime

    return {
        "distance": distance,
        "ddistance_daltitude_begin": 0.5 * dtime * begin["dtas_daltitude"],
        "ddistance_daltitude_end": 0.5 * dtime * end["dtas_daltitude"],
        "ddistance_dvelocity_begin": 0.5 * dtime * begin["dtas_dvelocity"],
        "ddistance_dvelocity_end": 0.5 * dtime * end["dtas_dvelocity"],
        "ddistance_dtarget_minutes": 30.0 * (tas_begin + tas_end),
    }


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
