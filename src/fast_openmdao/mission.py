# src/fast_openmdao/mission.py

"""OpenMDAO components for FAST mission primitive equations."""

import numpy as np
import openmdao.api as om

from fast_openmdao.atmosphere import GAS_CONSTANT_AIR, atmosphere_layer
from fast_openmdao.battery import battery_power_history_values
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


class CruiseBreguetPowerHistory(om.ExplicitComponent):
    """Compute FAST CruiseBRE mass, power, and energy histories.

    Inputs:
        initial_mass: Segment starting mass in kg.
        distance_step: Cruise distance increments in m.
        time_step: Cruise time increments in s.
        fuel_specific_energy: Fuel specific energy in J/kg.
        battery_specific_energy: Battery specific energy in J/kg.
        lift_drag: Segment lift-to-drag ratio.
        propulsive_efficiency, electric_motor_efficiency,
            electric_generator_efficiency, gas_turbine_efficiency: CruiseBRE
            branch efficiencies.
        power_split: FAST CruiseBRE electric/fuel split phi.

    Outputs:
        mass: Segment mass history in kg.
        fuel_power, battery_power, propulsor_power, motor_power,
            generator_power, required_power: FAST power histories in W.
        fuel_burn: Fuel-burn increments in kg.
        fuel_energy, battery_energy: Source-energy increments in J.
        phi_history: Power split history.

    Assumptions:
        ``architecture`` is fixed for optimization. Detailed cell-discharge
        depletion is intentionally excluded because FAST changes the active
        branch after SOC depletion, which is not a smooth optimizer component.
    """

    def initialize(self):
        self.options.declare("architecture", default="AC")
        self.options.declare("npoint", default=3)

    def setup(self):
        npoint = self.options["npoint"]
        nstep = npoint - 1
        self.add_input("initial_mass", val=1000.0, units="kg")
        self.add_input("distance_step", val=np.ones(nstep), units="m")
        self.add_input("time_step", val=np.ones(nstep), units="s")
        self.add_input("fuel_specific_energy", val=43200000.0, units="J/kg")
        self.add_input("battery_specific_energy", val=1000.0, units="J/kg")
        self.add_input("lift_drag", val=15.0)
        self.add_input("propulsive_efficiency", val=0.84)
        self.add_input("electric_motor_efficiency", val=0.95)
        self.add_input("electric_generator_efficiency", val=0.91)
        self.add_input("gas_turbine_efficiency", val=0.36)
        self.add_input("power_split", val=0.3)
        self.add_output("mass", val=np.ones(npoint), units="kg")
        self.add_output("fuel_power", val=np.zeros(npoint), units="W")
        self.add_output("battery_power", val=np.zeros(npoint), units="W")
        self.add_output("propulsor_power", val=np.zeros(npoint), units="W")
        self.add_output("motor_power", val=np.zeros(npoint), units="W")
        self.add_output("generator_power", val=np.zeros(npoint), units="W")
        self.add_output("required_power", val=np.zeros(npoint), units="W")
        self.add_output("fuel_burn", val=np.zeros(nstep), units="kg")
        self.add_output("fuel_energy", val=np.zeros(nstep), units="J")
        self.add_output("battery_energy", val=np.zeros(nstep), units="J")
        self.add_output("phi_history", val=np.zeros(npoint))
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = cruise_breguet_power_history_values(
            self.options["architecture"],
            self.options["npoint"],
            cruise_breguet_power_history_inputs(inputs),
        )
        for name in cruise_breguet_power_history_output_names():
            outputs[name] = values[name]

    def compute_partials(self, inputs, partials):
        data = cruise_breguet_power_history_inputs(inputs)
        jacobian = {}

        for output_name in cruise_breguet_power_history_output_names():
            output_size = (
                self.options["npoint"] - 1
                if output_name in breguet_power_history_step_output_names()
                else self.options["npoint"]
            )
            for input_name in cruise_breguet_power_history_input_names():
                input_size = np.asarray(data[input_name]).size
                jacobian[output_name, input_name] = np.zeros(
                    (output_size, input_size)
                )

        for input_name in cruise_breguet_power_history_input_names():
            size = np.asarray(data[input_name]).size
            for index in range(size):
                seeds = zero_breguet_power_history_seeds(
                    self.options["npoint"],
                    input_name,
                    index,
                )
                values = cruise_breguet_power_history_values(
                    self.options["architecture"],
                    self.options["npoint"],
                    data,
                    seeds,
                )
                for output_name in cruise_breguet_power_history_output_names():
                    column = values[f"d{output_name}"].reshape(-1)
                    jacobian[output_name, input_name][:, index] = column

        for key, block in jacobian.items():
            partials[key] = block


class CruiseBreguetDetailedBattery(om.ExplicitComponent):
    """Apply FAST CruiseBRE detailed battery discharge and depletion logic."""

    def initialize(self):
        self.options.declare("architecture", default="E")
        self.options.declare("npoint", default=3)
        self.options.declare("analysis_type", default=0)
        self.options.declare("degradation", default=0)

    def setup(self):
        npoint = self.options["npoint"]
        nstep = npoint - 1
        self.add_input("battery_power", val=np.zeros(npoint), units="W")
        self.add_input("time_step", val=np.ones(nstep), units="s")
        self.add_input("initial_soc", val=100.0)
        self.add_input("phi_history", val=np.zeros(npoint))
        self.add_input("parallel_cells", val=10.0)
        self.add_input("series_cells", val=100.0)
        self.add_input("max_cell_voltage", val=4.2)
        self.add_input("internal_resistance", val=0.01)
        self.add_input("exponential_voltage", val=0.1)
        self.add_input("exponential_capacity", val=1.0)
        self.add_input("cap_cell", val=2.4)
        self.add_input("state_of_health", val=100.0)
        self.add_output("adjusted_battery_power", val=np.zeros(npoint), units="W")
        self.add_output("soc", val=np.ones(npoint) * 100.0)
        self.add_output("adjusted_phi_history", val=np.zeros(npoint))
        self.add_output("soc_off", val=0.0)
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = cruise_breguet_detailed_battery_values(
            self.options["architecture"],
            inputs["battery_power"],
            inputs["time_step"],
            inputs["initial_soc"][0],
            inputs["phi_history"],
            inputs["parallel_cells"][0],
            inputs["series_cells"][0],
            inputs["max_cell_voltage"][0],
            inputs["internal_resistance"][0],
            inputs["exponential_voltage"][0],
            inputs["exponential_capacity"][0],
            inputs["cap_cell"][0],
            inputs["state_of_health"][0],
            self.options["analysis_type"],
            self.options["degradation"],
        )
        outputs["adjusted_battery_power"] = values["adjusted_battery_power"]
        outputs["soc"] = values["soc"]
        outputs["adjusted_phi_history"] = values["adjusted_phi_history"]
        outputs["soc_off"] = values["soc_off"]

    def compute_partials(self, inputs, partials):
        values = cruise_breguet_detailed_battery_values(
            self.options["architecture"],
            inputs["battery_power"],
            inputs["time_step"],
            inputs["initial_soc"][0],
            inputs["phi_history"],
            inputs["parallel_cells"][0],
            inputs["series_cells"][0],
            inputs["max_cell_voltage"][0],
            inputs["internal_resistance"][0],
            inputs["exponential_voltage"][0],
            inputs["exponential_capacity"][0],
            inputs["cap_cell"][0],
            inputs["state_of_health"][0],
            self.options["analysis_type"],
            self.options["degradation"],
        )

        for output in cruise_breguet_detailed_battery_output_names():
            for variable in cruise_breguet_detailed_battery_input_names():
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


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


def cruise_breguet_power_history_inputs(inputs):
    """Return scalar/vector input mapping for CruiseBRE power history."""

    return {
        "initial_mass": inputs["initial_mass"][0],
        "distance_step": np.asarray(inputs["distance_step"], dtype=float).reshape(-1),
        "time_step": np.asarray(inputs["time_step"], dtype=float).reshape(-1),
        "fuel_specific_energy": inputs["fuel_specific_energy"][0],
        "battery_specific_energy": inputs["battery_specific_energy"][0],
        "lift_drag": inputs["lift_drag"][0],
        "propulsive_efficiency": inputs["propulsive_efficiency"][0],
        "electric_motor_efficiency": inputs["electric_motor_efficiency"][0],
        "electric_generator_efficiency": inputs[
            "electric_generator_efficiency"
        ][0],
        "gas_turbine_efficiency": inputs["gas_turbine_efficiency"][0],
        "power_split": inputs["power_split"][0],
    }


def cruise_breguet_power_history_values(architecture, npoint, data, seeds=None):
    """Return FAST CruiseBRE histories and optional forward sensitivities."""

    arch = architecture.upper()
    if seeds is None:
        seeds = zero_breguet_power_history_seeds(npoint)

    values = zero_breguet_power_history_values(npoint)
    values["mass"][0] = data["initial_mass"]
    values["dmass"][0] = seeds["initial_mass"]
    values["phi_history"][:] = data["power_split"]
    values["dphi_history"][:] = seeds["power_split"]

    if arch in ("AC", "PHE", "SHE", "TE"):
        fill_combustion_breguet_history(arch, data, seeds, values)
    elif arch == "PE":
        fill_partially_electric_breguet_history(data, seeds, values)
    elif arch == "E":
        fill_electric_breguet_history(data, seeds, values)
    else:
        raise ValueError("architecture must be AC, PHE, SHE, TE, PE, or E.")

    return values


def fill_combustion_breguet_history(arch, data, seeds, values):
    """Fill AC/PHE/SHE/TE CruiseBRE histories and sensitivities."""

    eta = breguet_efficiency_sensitivity(arch, data, seeds)
    phi = data["power_split"]
    dphi = seeds["power_split"]
    split = phi / (1.0 - phi)
    dsplit = dphi / (1.0 - phi) ** 2
    fuel = data["fuel_specific_energy"]
    dfuel = seeds["fuel_specific_energy"]
    lift_drag = data["lift_drag"]
    dlift_drag = seeds["lift_drag"]
    gravity = 9.81
    term = eta["eta1"] + eta["eta2"] * split
    dterm = eta["deta1"] + eta["deta2"] * split + eta["eta2"] * dsplit
    denominator = eta["eta3"] * (fuel / gravity) * lift_drag * term
    ddenominator = denominator * (
        eta["deta3"] / eta["eta3"]
        + dfuel / fuel
        + dlift_drag / lift_drag
        + dterm / term
    )

    fill_exponential_mass_history(data, seeds, values, denominator, ddenominator)
    fill_fuel_outputs(data, seeds, values)
    values["battery_power"] = values["fuel_power"] * split
    values["dbattery_power"] = values["dfuel_power"] * split + values[
        "fuel_power"
    ] * dsplit
    values["battery_energy"] = values["battery_power"][:-1] * data["time_step"]
    values["dbattery_energy"] = (
        values["dbattery_power"][:-1] * data["time_step"]
        + values["battery_power"][:-1] * seeds["time_step"]
    )
    values["required_power"] = eta["eta3"] * (
        eta["eta1"] * values["fuel_power"]
        + eta["eta2"] * values["battery_power"]
    )
    values["drequired_power"] = eta["deta3"] * (
        eta["eta1"] * values["fuel_power"]
        + eta["eta2"] * values["battery_power"]
    ) + eta["eta3"] * (
        eta["deta1"] * values["fuel_power"]
        + eta["eta1"] * values["dfuel_power"]
        + eta["deta2"] * values["battery_power"]
        + eta["eta2"] * values["dbattery_power"]
    )
    values["propulsor_power"] = (
        values["required_power"] / data["propulsive_efficiency"]
    )
    values["dpropulsor_power"] = (
        values["drequired_power"] / data["propulsive_efficiency"]
        - values["required_power"]
        * seeds["propulsive_efficiency"]
        / data["propulsive_efficiency"] ** 2
    )

    if arch == "PHE":
        values["motor_power"] = values["battery_power"] * eta["eta2"]
        values["dmotor_power"] = (
            values["dbattery_power"] * eta["eta2"]
            + values["battery_power"] * eta["deta2"]
        )
    elif arch in ("SHE", "TE"):
        values["motor_power"] = values["propulsor_power"]
        values["dmotor_power"] = values["dpropulsor_power"]
        values["generator_power"] = (
            values["fuel_power"]
            * data["gas_turbine_efficiency"]
            * data["electric_generator_efficiency"]
        )
        values["dgenerator_power"] = values["dfuel_power"] * data[
            "gas_turbine_efficiency"
        ] * data["electric_generator_efficiency"] + values["fuel_power"] * (
            seeds["gas_turbine_efficiency"]
            * data["electric_generator_efficiency"]
            + data["gas_turbine_efficiency"]
            * seeds["electric_generator_efficiency"]
        )


def fill_partially_electric_breguet_history(data, seeds, values):
    """Fill PE CruiseBRE histories and sensitivities."""

    phi = data["power_split"]
    dphi = seeds["power_split"]
    eta_em = data["electric_motor_efficiency"]
    deta_em = seeds["electric_motor_efficiency"]
    eta_gt = data["gas_turbine_efficiency"]
    deta_gt = seeds["gas_turbine_efficiency"]
    numerator = 24.0 * eta_em * phi
    dnumerator = 24.0 * (deta_em * phi + eta_em * dphi)
    denominator = 25.0 * eta_gt + 24.0 * eta_em * phi - 25.0 * eta_gt * phi
    ddenominator = (
        25.0 * deta_gt
        + 24.0 * (deta_em * phi + eta_em * dphi)
        - 25.0 * (deta_gt * phi + eta_gt * dphi)
    )
    zeta = numerator / denominator
    dzeta = (dnumerator * denominator - numerator * ddenominator) / denominator ** 2
    eta0, deta0 = partially_electric_eta0(data, seeds, zeta, dzeta)
    breguet_denominator = (
        data["lift_drag"] * eta0 * data["fuel_specific_energy"] / 9.81
    )
    dbreguet_denominator = breguet_denominator * (
        seeds["lift_drag"] / data["lift_drag"]
        + deta0 / eta0
        + seeds["fuel_specific_energy"] / data["fuel_specific_energy"]
    )

    fill_exponential_mass_history(
        data,
        seeds,
        values,
        breguet_denominator,
        dbreguet_denominator,
    )
    fill_fuel_outputs(data, seeds, values)
    split = zeta / (1.0 - zeta)
    dsplit = dzeta / (1.0 - zeta) ** 2
    values["motor_power"] = values["fuel_power"] * split
    values["dmotor_power"] = values["dfuel_power"] * split + values[
        "fuel_power"
    ] * dsplit
    values["generator_power"] = values["motor_power"] / eta_em
    values["dgenerator_power"] = (
        values["dmotor_power"] / eta_em
        - values["motor_power"] * deta_em / eta_em ** 2
    )
    values["required_power"] = (
        values["motor_power"] * data["propulsive_efficiency"] / zeta
    )
    values["drequired_power"] = values["dmotor_power"] * data[
        "propulsive_efficiency"
    ] / zeta + values["motor_power"] * (
        seeds["propulsive_efficiency"] / zeta
        - data["propulsive_efficiency"] * dzeta / zeta ** 2
    )
    values["propulsor_power"] = (
        values["fuel_power"] * data["gas_turbine_efficiency"]
    )
    values["dpropulsor_power"] = (
        values["dfuel_power"] * data["gas_turbine_efficiency"]
        + values["fuel_power"] * seeds["gas_turbine_efficiency"]
    )


def fill_electric_breguet_history(data, seeds, values):
    """Fill E CruiseBRE histories and sensitivities."""

    distance_sum = data["distance_step"].sum()
    ddistance_sum = seeds["distance_step"].sum()
    denominator = (
        data["battery_specific_energy"]
        / 9.81
        * data["lift_drag"]
        * data["electric_motor_efficiency"]
        * data["propulsive_efficiency"]
    )
    ddenominator = denominator * (
        seeds["battery_specific_energy"] / data["battery_specific_energy"]
        + seeds["lift_drag"] / data["lift_drag"]
        + seeds["electric_motor_efficiency"] / data["electric_motor_efficiency"]
        + seeds["propulsive_efficiency"] / data["propulsive_efficiency"]
    )
    battery_weight = data["initial_mass"] * distance_sum / denominator
    dbattery_weight = (
        seeds["initial_mass"] * distance_sum
        + data["initial_mass"] * ddistance_sum
    ) / denominator - data["initial_mass"] * distance_sum * ddenominator / denominator ** 2
    values["mass"][:] = data["initial_mass"]
    values["dmass"][:] = seeds["initial_mass"]
    values["battery_energy"][:] = data["battery_specific_energy"] / battery_weight
    values["dbattery_energy"][:] = (
        seeds["battery_specific_energy"] / battery_weight
        - data["battery_specific_energy"] * dbattery_weight / battery_weight ** 2
    )
    values["battery_power"][:-1] = values["battery_energy"] / data["time_step"]
    values["dbattery_power"][:-1] = (
        values["dbattery_energy"] / data["time_step"]
        - values["battery_energy"] * seeds["time_step"] / data["time_step"] ** 2
    )
    values["motor_power"] = (
        values["battery_power"] * data["electric_motor_efficiency"]
    )
    values["dmotor_power"] = values["dbattery_power"] * data[
        "electric_motor_efficiency"
    ] + values["battery_power"] * seeds["electric_motor_efficiency"]
    values["required_power"] = (
        values["motor_power"] * data["propulsive_efficiency"]
    )
    values["drequired_power"] = values["dmotor_power"] * data[
        "propulsive_efficiency"
    ] + values["motor_power"] * seeds["propulsive_efficiency"]
    values["propulsor_power"] = values["motor_power"]
    values["dpropulsor_power"] = values["dmotor_power"]


def fill_exponential_mass_history(data, seeds, values, denominator, ddenominator):
    """Fill Breguet exponential mass recurrence and sensitivities."""

    for index in range(data["distance_step"].size):
        exponent = -data["distance_step"][index] / denominator
        dexponent = (
            -seeds["distance_step"][index] / denominator
            + data["distance_step"][index] * ddenominator / denominator ** 2
        )
        factor = np.exp(exponent)
        dfactor = factor * dexponent
        values["mass"][index + 1] = values["mass"][index] * factor
        values["dmass"][index + 1] = (
            values["dmass"][index] * factor + values["mass"][index] * dfactor
        )


def fill_fuel_outputs(data, seeds, values):
    """Fill fuel burn, fuel energy, and fuel power histories."""

    values["fuel_burn"] = -np.diff(values["mass"])
    values["dfuel_burn"] = -np.diff(values["dmass"])
    values["fuel_energy"] = values["fuel_burn"] * data["fuel_specific_energy"]
    values["dfuel_energy"] = (
        values["dfuel_burn"] * data["fuel_specific_energy"]
        + values["fuel_burn"] * seeds["fuel_specific_energy"]
    )
    values["fuel_power"][:-1] = values["fuel_energy"] / data["time_step"]
    values["dfuel_power"][:-1] = (
        values["dfuel_energy"] / data["time_step"]
        - values["fuel_energy"] * seeds["time_step"] / data["time_step"] ** 2
    )


def partially_electric_eta0(data, seeds, zeta, dzeta):
    """Return PE effective Breguet efficiency and sensitivity."""

    numerator = (
        data["gas_turbine_efficiency"]
        * data["propulsive_efficiency"]
        * data["electric_motor_efficiency"]
        * data["electric_generator_efficiency"]
    )
    dnumerator = numerator * (
        seeds["gas_turbine_efficiency"] / data["gas_turbine_efficiency"]
        + seeds["propulsive_efficiency"] / data["propulsive_efficiency"]
        + seeds["electric_motor_efficiency"] / data["electric_motor_efficiency"]
        + seeds["electric_generator_efficiency"]
        / data["electric_generator_efficiency"]
    )
    denominator = (
        (1.0 - zeta)
        * data["electric_motor_efficiency"]
        * data["electric_generator_efficiency"]
        + zeta
    )
    ddenominator = (
        -dzeta
        * data["electric_motor_efficiency"]
        * data["electric_generator_efficiency"]
        + (1.0 - zeta)
        * (
            seeds["electric_motor_efficiency"]
            * data["electric_generator_efficiency"]
            + data["electric_motor_efficiency"]
            * seeds["electric_generator_efficiency"]
        )
        + dzeta
    )
    eta0 = numerator / denominator
    deta0 = (dnumerator * denominator - numerator * ddenominator) / denominator ** 2
    return eta0, deta0


def breguet_efficiency_sensitivity(architecture, data, seeds):
    """Return CruiseBRE efficiency triplet and directional sensitivity."""

    values = cruise_breguet_efficiency_values(
        architecture,
        data["propulsive_efficiency"],
        data["electric_motor_efficiency"],
        data["electric_generator_efficiency"],
        data["gas_turbine_efficiency"],
    )
    for output_name in ("eta1", "eta2", "eta3"):
        total = 0.0
        for input_name in breguet_efficiency_input_names():
            total += values[f"d{output_name}_d{input_name}"] * seeds[input_name]
        values[f"d{output_name}"] = total

    return values


def zero_breguet_power_history_values(npoint):
    """Return zero-filled CruiseBRE history value and sensitivity arrays."""

    nstep = npoint - 1
    values = {}
    for name in cruise_breguet_power_history_output_names():
        size = nstep if name in breguet_power_history_step_output_names() else npoint
        values[name] = np.zeros(size)
        values[f"d{name}"] = np.zeros(size)
    return values


def zero_breguet_power_history_seeds(npoint, active_name=None, active_index=0):
    """Return one-hot input seed mapping for forward sensitivity propagation."""

    seeds = {}
    for name in cruise_breguet_power_history_input_names():
        if name in breguet_power_history_step_input_names():
            seeds[name] = np.zeros(npoint - 1)
        else:
            seeds[name] = 0.0

    if active_name is None:
        return seeds

    if active_name in breguet_power_history_step_input_names():
        seeds[active_name][active_index] = 1.0
    else:
        seeds[active_name] = 1.0
    return seeds


def cruise_breguet_power_history_input_names():
    """Return CruiseBRE power-history input names."""

    return (
        "initial_mass",
        "distance_step",
        "time_step",
        "fuel_specific_energy",
        "battery_specific_energy",
        "lift_drag",
        "propulsive_efficiency",
        "electric_motor_efficiency",
        "electric_generator_efficiency",
        "gas_turbine_efficiency",
        "power_split",
    )


def breguet_power_history_step_input_names():
    """Return vector step input names for CruiseBRE power history."""

    return ("distance_step", "time_step")


def cruise_breguet_power_history_output_names():
    """Return CruiseBRE power-history output names."""

    return (
        "mass",
        "fuel_power",
        "battery_power",
        "propulsor_power",
        "motor_power",
        "generator_power",
        "required_power",
        "fuel_burn",
        "fuel_energy",
        "battery_energy",
        "phi_history",
    )


def breguet_power_history_step_output_names():
    """Return vector step output names for CruiseBRE power history."""

    return ("fuel_burn", "fuel_energy", "battery_energy")


def cruise_breguet_detailed_battery_input_names():
    """Return input names for CruiseBreguetDetailedBattery derivatives."""

    return (
        "battery_power",
        "time_step",
        "initial_soc",
        "phi_history",
        "parallel_cells",
        "series_cells",
        "max_cell_voltage",
        "internal_resistance",
        "exponential_voltage",
        "exponential_capacity",
        "cap_cell",
        "state_of_health",
    )


def cruise_breguet_detailed_battery_output_names():
    """Return output names for CruiseBreguetDetailedBattery."""

    return (
        "adjusted_battery_power",
        "soc",
        "adjusted_phi_history",
        "soc_off",
    )


def cruise_breguet_detailed_battery_values(
    architecture,
    battery_power,
    time_step,
    initial_soc,
    phi_history,
    parallel_cells,
    series_cells,
    max_cell_voltage,
    internal_resistance,
    exponential_voltage,
    exponential_capacity,
    cap_cell,
    state_of_health,
    analysis_type=0,
    degradation=0,
):
    """Return detailed CruiseBRE battery discharge values and local derivatives."""

    battery_power = np.asarray(battery_power, dtype=float).reshape(-1)
    time_step = np.asarray(time_step, dtype=float).reshape(-1)
    phi_history = np.asarray(phi_history, dtype=float).reshape(-1)
    npoint = battery_power.size
    nstep = npoint - 1

    if time_step.size != nstep or phi_history.size != npoint:
        raise ValueError("CruiseBreguetDetailedBattery requires npoint histories.")

    history = battery_power_history_values(
        battery_power[:-1],
        time_step,
        initial_soc,
        parallel_cells,
        series_cells,
        max_cell_voltage,
        internal_resistance,
        exponential_voltage,
        exponential_capacity,
        cap_cell,
        state_of_health,
        True,
        False,
        analysis_type,
        degradation,
    )

    adjusted_power = battery_power.copy()
    adjusted_power[:-1] = history["output_power"]
    soc = history["soc"].copy()
    adjusted_phi = phi_history.copy()
    soc_off = 0.0
    depleted = np.where(soc < 20.0)[0]
    stop = None

    if len(depleted) > 0 and architecture != "E":
        stop = depleted[0]
        adjusted_power[stop:] = 0.0
        adjusted_phi[stop:] = 0.0
        soc[stop:] = soc[max(0, stop - 1)]
        soc_off = 1.0

    output_sizes = {
        "adjusted_battery_power": npoint,
        "soc": npoint,
        "adjusted_phi_history": npoint,
        "soc_off": 1,
    }
    input_sizes = {
        "battery_power": npoint,
        "time_step": nstep,
        "initial_soc": 1,
        "phi_history": npoint,
        "parallel_cells": 1,
        "series_cells": 1,
        "max_cell_voltage": 1,
        "internal_resistance": 1,
        "exponential_voltage": 1,
        "exponential_capacity": 1,
        "cap_cell": 1,
        "state_of_health": 1,
    }
    jacobian = {}

    for output, output_size in output_sizes.items():
        for variable, input_size in input_sizes.items():
            jacobian[output, variable] = np.zeros((output_size, input_size))

    for variable in battery_power_history_input_names_for_breguet():
        if variable == "battery_power":
            source = "requested_power"
        elif variable == "time_step":
            source = "time"
        elif variable == "initial_soc":
            source = "soc_begin"
        else:
            source = variable
        block = history["doutput_power_d%s" % source]

        if variable == "battery_power":
            jacobian["adjusted_battery_power", variable][:-1, :-1] = block
            jacobian["adjusted_battery_power", variable][-1, -1] = 1.0
        else:
            jacobian["adjusted_battery_power", variable][:-1, :] = block

        soc_block = history["dsoc_d%s" % source]

        if variable == "battery_power":
            jacobian["soc", variable][:, :-1] = soc_block
        else:
            jacobian["soc", variable] = soc_block

    jacobian["adjusted_phi_history", "phi_history"] = np.eye(npoint)

    if stop is not None:
        freeze_row = max(0, stop - 1)

        for variable in cruise_breguet_detailed_battery_input_names():
            jacobian["adjusted_battery_power", variable][stop:, :] = 0.0
            jacobian["adjusted_phi_history", variable][stop:, :] = 0.0
            jacobian["soc", variable][stop:, :] = jacobian["soc", variable][
                freeze_row,
                :,
            ]

    result = {
        "adjusted_battery_power": adjusted_power,
        "soc": soc,
        "adjusted_phi_history": adjusted_phi,
        "soc_off": soc_off,
    }

    for key, value in jacobian.items():
        result["d%s_d%s" % key] = value

    return result


def battery_power_history_input_names_for_breguet():
    """Return detailed-Breguet inputs that feed BatteryPowerHistory."""

    return (
        "battery_power",
        "time_step",
        "initial_soc",
        "parallel_cells",
        "series_cells",
        "max_cell_voltage",
        "internal_resistance",
        "exponential_voltage",
        "exponential_capacity",
        "cap_cell",
        "state_of_health",
    )


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
