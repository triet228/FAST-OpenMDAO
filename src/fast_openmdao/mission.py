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


class TakeoffSegmentKinematics(om.ExplicitComponent):
    """Compute FAST simple EvalTakeoff ground-roll kinematics.

    Inputs:
        altitude: Segment altitude used for pointwise atmosphere in m.
        target_altitude: Altitude used to convert the terminal speed in m.
        target_velocity: Terminal speed value in the selected FAST velocity type.
        mass: Segment mass in kg.

    Outputs:
        FAST simple-takeoff time, distance, TAS/EAS/Mach, density,
        acceleration, zero climb quantities, and mechanical energy histories.

    Assumptions:
        This is the smooth one-minute EvalTakeoff kinematic kernel before
        propulsion-history mutation. Power required is infinite in FAST for
        this segment, so it is intentionally left to the orchestration layer.
    """

    def initialize(self):
        self.options.declare("npoint", default=3)
        self.options.declare("target_velocity_type", default="TAS")
        self.options.declare("takeoff_time", default=60.0)
        self.options.declare("gravity", default=9.81)

    def setup(self):
        npoint = self.options["npoint"]
        self.add_input("altitude", val=0.0, units="m")
        self.add_input("target_altitude", val=0.0, units="m")
        self.add_input("target_velocity", val=100.0)
        self.add_input("mass", val=1000.0, units="kg")
        self.add_output("time", val=np.zeros(npoint), units="s")
        self.add_output("distance", val=np.zeros(npoint), units="m")
        self.add_output("true_airspeed", val=np.zeros(npoint), units="m/s")
        self.add_output("equivalent_airspeed", val=np.zeros(npoint), units="m/s")
        self.add_output("mach", val=np.zeros(npoint))
        self.add_output("density", val=np.zeros(npoint), units="kg/m**3")
        self.add_output("rate_of_climb", val=np.zeros(npoint), units="m/s")
        self.add_output("acceleration", val=np.zeros(npoint), units="m/s**2")
        self.add_output("flight_path_angle", val=np.zeros(npoint))
        self.add_output("specific_excess_power", val=np.zeros(npoint), units="m/s")
        self.add_output("potential_energy", val=np.zeros(npoint), units="J")
        self.add_output("kinetic_energy", val=np.zeros(npoint), units="J")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = takeoff_segment_kinematics_values(
            self.options["npoint"],
            self.options["target_velocity_type"],
            self.options["takeoff_time"],
            self.options["gravity"],
            inputs["altitude"][0],
            inputs["target_altitude"][0],
            inputs["target_velocity"][0],
            inputs["mass"][0],
        )

        for output_name in takeoff_segment_kinematics_output_names():
            outputs[output_name] = values[output_name]

    def compute_partials(self, inputs, partials):
        values = takeoff_segment_kinematics_values(
            self.options["npoint"],
            self.options["target_velocity_type"],
            self.options["takeoff_time"],
            self.options["gravity"],
            inputs["altitude"][0],
            inputs["target_altitude"][0],
            inputs["target_velocity"][0],
            inputs["mass"][0],
        )

        for output_name in takeoff_segment_kinematics_output_names():
            for input_name in takeoff_segment_kinematics_input_names():
                partials[output_name, input_name] = values[
                    "d%s_d%s" % (output_name, input_name)
                ]


class LandingSegmentKinematicsPower(om.ExplicitComponent):
    """Compute FAST EvalLanding kinematics and reverse-power demand.

    Inputs:
        initial_distance: Segment starting distance in m.
        initial_time: Segment starting time in s.
        altitude: Landing altitude in m.
        landing_velocity: Initial speed value in the selected FAST velocity type.
        mass: Segment mass in kg.
        available_power: Total available power history in W.

    Outputs:
        FAST landing time, distance, TAS/EAS/Mach, density, acceleration,
        reverse required power, zero climb quantities, and energy histories.
    """

    def initialize(self):
        self.options.declare("velocity_type", default="TAS")
        self.options.declare("landing_time", default=30.0)
        self.options.declare("gravity", default=9.81)

    def setup(self):
        self.add_input("initial_distance", val=0.0, units="m")
        self.add_input("initial_time", val=0.0, units="s")
        self.add_input("altitude", val=0.0, units="m")
        self.add_input("landing_velocity", val=100.0)
        self.add_input("mass", val=1000.0, units="kg")
        self.add_input("available_power", val=np.zeros(2), units="W")
        self.add_output("time", val=np.zeros(2), units="s")
        self.add_output("distance", val=np.zeros(2), units="m")
        self.add_output("true_airspeed", val=np.zeros(2), units="m/s")
        self.add_output("equivalent_airspeed", val=np.zeros(2), units="m/s")
        self.add_output("mach", val=np.zeros(2))
        self.add_output("density", val=np.zeros(2), units="kg/m**3")
        self.add_output("rate_of_climb", val=np.zeros(2), units="m/s")
        self.add_output("acceleration", val=np.zeros(2), units="m/s**2")
        self.add_output("flight_path_angle", val=np.zeros(2))
        self.add_output("required_power", val=np.zeros(2), units="W")
        self.add_output("specific_excess_power", val=np.zeros(2), units="m/s")
        self.add_output("potential_energy", val=np.zeros(2), units="J")
        self.add_output("kinetic_energy", val=np.zeros(2), units="J")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = landing_segment_kinematics_power_values(
            self.options["velocity_type"],
            self.options["landing_time"],
            self.options["gravity"],
            landing_segment_kinematics_power_inputs(inputs),
        )

        for output_name in landing_segment_kinematics_power_output_names():
            outputs[output_name] = values[output_name]

    def compute_partials(self, inputs, partials):
        values = landing_segment_kinematics_power_values(
            self.options["velocity_type"],
            self.options["landing_time"],
            self.options["gravity"],
            landing_segment_kinematics_power_inputs(inputs),
        )

        for output_name in landing_segment_kinematics_power_output_names():
            for input_name in landing_segment_kinematics_power_input_names():
                partials[output_name, input_name] = values[
                    "d%s_d%s" % (output_name, input_name)
                ]


class DetailedTakeoffSegmentKinematicsPower(om.ExplicitComponent):
    """Compute FAST detailed EvalTakeoff ground-roll physics.

    Inputs:
        altitude: Segment ground-roll altitude in m.
        target_altitude: Altitude used to convert terminal takeoff speed in m.
        target_velocity: Terminal takeoff speed in the selected FAST speed type.
        mass: Takeoff mass in kg.
        wing_loading: Sea-level-static wing loading in kg/m**2.
        available_power: Total available power history in W.

    Outputs:
        FAST detailed-takeoff velocity, density, lift-drag ratio, acceleration,
        time, distance, drag power, specific excess power, and mechanical
        energy histories.

    Assumptions:
        This ports the differentiable physics kernel before prop-analysis
        mutation. The first thrust and required-power entries are infinite in
        FAST bookkeeping; this component reports finite drag power instead.
    """

    def initialize(self):
        self.options.declare("npoint", default=3)
        self.options.declare("target_velocity_type", default="TAS")
        self.options.declare("gravity", default=9.81)

    def setup(self):
        npoint = self.options["npoint"]
        self.add_input("altitude", val=0.0, units="m")
        self.add_input("target_altitude", val=0.0, units="m")
        self.add_input("target_velocity", val=100.0)
        self.add_input("mass", val=1000.0, units="kg")
        self.add_input("wing_loading", val=100.0, units="kg/m**2")
        self.add_input("available_power", val=np.ones(npoint), units="W")
        self.add_output("time", val=np.zeros(npoint), units="s")
        self.add_output("distance", val=np.zeros(npoint), units="m")
        self.add_output("true_airspeed", val=np.zeros(npoint), units="m/s")
        self.add_output("equivalent_airspeed", val=np.zeros(npoint), units="m/s")
        self.add_output("mach", val=np.zeros(npoint))
        self.add_output("density", val=np.zeros(npoint), units="kg/m**3")
        self.add_output("lift_drag", val=np.zeros(npoint))
        self.add_output("drag_power", val=np.zeros(npoint), units="W")
        self.add_output("rate_of_climb", val=np.zeros(npoint), units="m/s")
        self.add_output("acceleration", val=np.zeros(npoint), units="m/s**2")
        self.add_output("flight_path_angle", val=np.zeros(npoint))
        self.add_output("specific_excess_power", val=np.zeros(npoint), units="m/s")
        self.add_output("potential_energy", val=np.zeros(npoint), units="J")
        self.add_output("kinetic_energy", val=np.zeros(npoint), units="J")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = detailed_takeoff_segment_kinematics_power_values(
            self.options["npoint"],
            self.options["target_velocity_type"],
            self.options["gravity"],
            detailed_takeoff_segment_kinematics_power_inputs(inputs),
        )

        for output_name in detailed_takeoff_segment_kinematics_power_output_names():
            outputs[output_name] = values[output_name]

    def compute_partials(self, inputs, partials):
        values = detailed_takeoff_segment_kinematics_power_values(
            self.options["npoint"],
            self.options["target_velocity_type"],
            self.options["gravity"],
            detailed_takeoff_segment_kinematics_power_inputs(inputs),
        )

        for output_name in detailed_takeoff_segment_kinematics_power_output_names():
            for input_name in detailed_takeoff_segment_kinematics_power_input_names():
                partials[output_name, input_name] = values[
                    "d%s_d%s" % (output_name, input_name)
                ]


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


class CruiseBreguetSourceDelta(om.ExplicitComponent):
    """Apply FAST CruiseBRE aggregate source-energy deltas to source columns.

    Inputs:
        source_energy: Current per-source used-energy matrix in J.
        source_energy_left: Current per-source remaining-energy matrix in J.
        delta: Aggregate energy delta history in J.

    Outputs:
        updated_source_energy: Source used-energy matrix after the delta.
        updated_source_energy_left: Source remaining-energy matrix after the
            delta.

    Assumptions:
        ``columns`` is a fixed set of source columns selected by source type.
        The aggregate delta is shared evenly across those columns, matching
        FAST-Python ``cruise_breguet_apply_source_delta``.
    """

    def initialize(self):
        self.options.declare("columns", default=(0,))
        self.options.declare("npoint", default=3)
        self.options.declare("nsrc", default=1)

    def setup(self):
        npoint = self.options["npoint"]
        nsrc = self.options["nsrc"]
        self.add_input("source_energy", val=np.zeros((npoint, nsrc)), units="J")
        self.add_input(
            "source_energy_left",
            val=np.zeros((npoint, nsrc)),
            units="J",
        )
        self.add_input("delta", val=np.zeros(npoint), units="J")
        self.add_output(
            "updated_source_energy",
            val=np.zeros((npoint, nsrc)),
            units="J",
        )
        self.add_output(
            "updated_source_energy_left",
            val=np.zeros((npoint, nsrc)),
            units="J",
        )
        self.declare_partials(of="updated_source_energy", wrt="*")
        self.declare_partials(of="updated_source_energy_left", wrt="*")

    def compute(self, inputs, outputs):
        values = breguet_source_delta_values(
            inputs["source_energy"],
            inputs["source_energy_left"],
            self.options["columns"],
            inputs["delta"],
        )
        outputs["updated_source_energy"] = values["updated_source_energy"]
        outputs["updated_source_energy_left"] = values["updated_source_energy_left"]

    def compute_partials(self, inputs, partials):
        values = breguet_source_delta_values(
            inputs["source_energy"],
            inputs["source_energy_left"],
            self.options["columns"],
            inputs["delta"],
        )
        for output_name in ("updated_source_energy", "updated_source_energy_left"):
            for input_name in ("source_energy", "source_energy_left", "delta"):
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


class RowMatrix(om.ExplicitComponent):
    """Repeat a fixed-size FAST scalar/vector value across history rows."""

    def initialize(self):
        self.options.declare("rows", default=1)
        self.options.declare("value_size", default=1)

    def setup(self):
        rows = self.options["rows"]
        value_size = self.options["value_size"]
        self.add_input("value", val=np.zeros(value_size))
        self.add_output("matrix", val=np.zeros((rows, value_size)))
        self.declare_partials(of="matrix", wrt="value")

    def compute(self, inputs, outputs):
        values = row_matrix_values(inputs["value"], self.options["rows"])
        outputs["matrix"] = values["matrix"]

    def compute_partials(self, inputs, partials):
        values = row_matrix_values(inputs["value"], self.options["rows"])
        partials["matrix", "value"] = values["dmatrix_dvalue"]


class HistoryVectorSlice(om.ExplicitComponent):
    """Assign a fixed FAST mission-history vector slice.

    Inputs:
        history_vector: Existing full history vector.
        values: Values written into the configured ``start:stop`` slice.

    Outputs:
        updated_history_vector: History vector after the slice assignment.

    Assumptions:
        The slice bounds are fixed OpenMDAO options. This mirrors the numerical
        behavior of FAST-Python's history-section assignment while exposing the
        overwritten and preserved entries as a linear map.
    """

    def initialize(self):
        self.options.declare("history_size", default=1)
        self.options.declare("start", default=0)
        self.options.declare("stop", default=1)

    def setup(self):
        history_size = self.options["history_size"]
        value_size = self.options["stop"] - self.options["start"]
        self.add_input("history_vector", val=np.zeros(history_size))
        self.add_input("values", val=np.zeros(value_size))
        self.add_output("updated_history_vector", val=np.zeros(history_size))
        self.declare_partials(of="updated_history_vector", wrt="history_vector")
        self.declare_partials(of="updated_history_vector", wrt="values")

    def compute(self, inputs, outputs):
        values = history_vector_slice_values(
            inputs["history_vector"],
            inputs["values"],
            self.options["start"],
            self.options["stop"],
        )
        outputs["updated_history_vector"] = values["updated_history_vector"]

    def compute_partials(self, inputs, partials):
        values = history_vector_slice_values(
            inputs["history_vector"],
            inputs["values"],
            self.options["start"],
            self.options["stop"],
        )
        partials["updated_history_vector", "history_vector"] = values[
            "dupdated_history_vector_dhistory_vector"
        ]
        partials["updated_history_vector", "values"] = values[
            "dupdated_history_vector_dvalues"
        ]


class HistoryMatrixSlice(om.ExplicitComponent):
    """Assign a fixed FAST mission-history matrix row slice."""

    def initialize(self):
        self.options.declare("num_rows", default=1)
        self.options.declare("num_cols", default=1)
        self.options.declare("start", default=0)
        self.options.declare("stop", default=1)

    def setup(self):
        num_rows = self.options["num_rows"]
        num_cols = self.options["num_cols"]
        value_rows = self.options["stop"] - self.options["start"]
        self.add_input("history_matrix", val=np.zeros((num_rows, num_cols)))
        self.add_input("values", val=np.zeros((value_rows, num_cols)))
        self.add_output("updated_history_matrix", val=np.zeros((num_rows, num_cols)))
        self.declare_partials(of="updated_history_matrix", wrt="history_matrix")
        self.declare_partials(of="updated_history_matrix", wrt="values")

    def compute(self, inputs, outputs):
        values = history_matrix_slice_values(
            inputs["history_matrix"],
            inputs["values"],
            self.options["start"],
            self.options["stop"],
        )
        outputs["updated_history_matrix"] = values["updated_history_matrix"]

    def compute_partials(self, inputs, partials):
        values = history_matrix_slice_values(
            inputs["history_matrix"],
            inputs["values"],
            self.options["start"],
            self.options["stop"],
        )
        partials["updated_history_matrix", "history_matrix"] = values[
            "dupdated_history_matrix_dhistory_matrix"
        ]
        partials["updated_history_matrix", "values"] = values[
            "dupdated_history_matrix_dvalues"
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


def takeoff_segment_kinematics_input_names():
    """Return simple takeoff segment input names."""

    return ("altitude", "target_altitude", "target_velocity", "mass")


def takeoff_segment_kinematics_output_names():
    """Return simple takeoff segment output names."""

    return (
        "time",
        "distance",
        "true_airspeed",
        "equivalent_airspeed",
        "mach",
        "density",
        "rate_of_climb",
        "acceleration",
        "flight_path_angle",
        "specific_excess_power",
        "potential_energy",
        "kinetic_energy",
    )


def takeoff_segment_kinematics_values(
    npoint,
    target_velocity_type,
    takeoff_time,
    gravity,
    altitude,
    target_altitude,
    target_velocity,
    mass,
):
    """Return simple takeoff segment outputs and dense derivatives."""

    target = flight_condition_values(
        target_altitude,
        0.0,
        target_velocity_type,
        target_velocity,
    )
    target_tas = target["tas"]
    dtarget_tas_dtarget_altitude = target["dtas_daltitude"]
    dtarget_tas_dtarget_velocity = target["dtas_dvelocity"]
    fraction = np.linspace(0.0, 1.0, npoint)
    time = np.linspace(0.0, takeoff_time, npoint)
    true_airspeed = target_tas * fraction
    dtrue_airspeed_dtarget_altitude = (
        dtarget_tas_dtarget_altitude * fraction
    )
    dtrue_airspeed_dtarget_velocity = (
        dtarget_tas_dtarget_velocity * fraction
    )
    distance = 0.5 * target_tas * takeoff_time * fraction ** 2
    ddistance_dtarget_altitude = (
        0.5 * dtarget_tas_dtarget_altitude * takeoff_time * fraction ** 2
    )
    ddistance_dtarget_velocity = (
        0.5 * dtarget_tas_dtarget_velocity * takeoff_time * fraction ** 2
    )
    acceleration = np.ones(npoint) * target_tas / takeoff_time
    dacceleration_dtarget_altitude = (
        np.ones(npoint) * dtarget_tas_dtarget_altitude / takeoff_time
    )
    dacceleration_dtarget_velocity = (
        np.ones(npoint) * dtarget_tas_dtarget_velocity / takeoff_time
    )
    equivalent_airspeed = np.zeros(npoint)
    mach = np.zeros(npoint)
    density = np.zeros(npoint)
    deas_daltitude = np.zeros(npoint)
    deas_dtarget_altitude = np.zeros(npoint)
    deas_dtarget_velocity = np.zeros(npoint)
    dmach_daltitude = np.zeros(npoint)
    dmach_dtarget_altitude = np.zeros(npoint)
    dmach_dtarget_velocity = np.zeros(npoint)
    ddensity_daltitude = np.zeros(npoint)

    for index, velocity in enumerate(true_airspeed):
        condition = flight_condition_values(altitude, 0.0, "TAS", velocity)
        equivalent_airspeed[index] = condition["eas"]
        mach[index] = condition["mach"]
        density[index] = condition["density"]
        deas_daltitude[index] = condition["deas_daltitude"]
        deas_dtarget_altitude[index] = (
            condition["deas_dvelocity"] * dtrue_airspeed_dtarget_altitude[index]
        )
        deas_dtarget_velocity[index] = (
            condition["deas_dvelocity"] * dtrue_airspeed_dtarget_velocity[index]
        )
        dmach_daltitude[index] = condition["dmach_daltitude"]
        dmach_dtarget_altitude[index] = (
            condition["dmach_dvelocity"] * dtrue_airspeed_dtarget_altitude[index]
        )
        dmach_dtarget_velocity[index] = (
            condition["dmach_dvelocity"] * dtrue_airspeed_dtarget_velocity[index]
        )
        ddensity_daltitude[index] = condition["ddensity_daltitude"]

    rate_of_climb = np.zeros(npoint)
    flight_path_angle = np.zeros(npoint)
    specific_excess_power = np.zeros(npoint)
    potential_energy = np.ones(npoint) * mass * gravity * altitude
    kinetic_energy = 0.5 * mass * true_airspeed ** 2
    zeros = np.zeros(npoint)
    values = {
        "time": time,
        "distance": distance,
        "true_airspeed": true_airspeed,
        "equivalent_airspeed": equivalent_airspeed,
        "mach": mach,
        "density": density,
        "rate_of_climb": rate_of_climb,
        "acceleration": acceleration,
        "flight_path_angle": flight_path_angle,
        "specific_excess_power": specific_excess_power,
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
    }

    for output_name in takeoff_segment_kinematics_output_names():
        for input_name in takeoff_segment_kinematics_input_names():
            values["d%s_d%s" % (output_name, input_name)] = np.zeros((npoint, 1))

    values["ddistance_dtarget_altitude"][:, 0] = ddistance_dtarget_altitude
    values["ddistance_dtarget_velocity"][:, 0] = ddistance_dtarget_velocity
    values["dtrue_airspeed_dtarget_altitude"][:, 0] = (
        dtrue_airspeed_dtarget_altitude
    )
    values["dtrue_airspeed_dtarget_velocity"][:, 0] = (
        dtrue_airspeed_dtarget_velocity
    )
    values["dequivalent_airspeed_daltitude"][:, 0] = deas_daltitude
    values["dequivalent_airspeed_dtarget_altitude"][:, 0] = deas_dtarget_altitude
    values["dequivalent_airspeed_dtarget_velocity"][:, 0] = deas_dtarget_velocity
    values["dmach_daltitude"][:, 0] = dmach_daltitude
    values["dmach_dtarget_altitude"][:, 0] = dmach_dtarget_altitude
    values["dmach_dtarget_velocity"][:, 0] = dmach_dtarget_velocity
    values["ddensity_daltitude"][:, 0] = ddensity_daltitude
    values["dacceleration_dtarget_altitude"][:, 0] = dacceleration_dtarget_altitude
    values["dacceleration_dtarget_velocity"][:, 0] = dacceleration_dtarget_velocity
    values["dpotential_energy_daltitude"][:, 0] = np.ones(npoint) * mass * gravity
    values["dpotential_energy_dmass"][:, 0] = np.ones(npoint) * gravity * altitude
    values["dkinetic_energy_dmass"][:, 0] = 0.5 * true_airspeed ** 2
    values["dkinetic_energy_dtarget_altitude"][:, 0] = (
        mass * true_airspeed * dtrue_airspeed_dtarget_altitude
    )
    values["dkinetic_energy_dtarget_velocity"][:, 0] = (
        mass * true_airspeed * dtrue_airspeed_dtarget_velocity
    )
    values["dtime_daltitude"][:, 0] = zeros

    return values


def landing_segment_kinematics_power_input_names():
    """Return landing segment kernel input names."""

    return (
        "initial_distance",
        "initial_time",
        "altitude",
        "landing_velocity",
        "mass",
        "available_power",
    )


def landing_segment_kinematics_power_output_names():
    """Return landing segment kernel output names."""

    return (
        "time",
        "distance",
        "true_airspeed",
        "equivalent_airspeed",
        "mach",
        "density",
        "rate_of_climb",
        "acceleration",
        "flight_path_angle",
        "required_power",
        "specific_excess_power",
        "potential_energy",
        "kinetic_energy",
    )


def landing_segment_kinematics_power_inputs(inputs):
    """Return numeric inputs for the landing segment kernel."""

    return {
        "initial_distance": inputs["initial_distance"][0],
        "initial_time": inputs["initial_time"][0],
        "altitude": inputs["altitude"][0],
        "landing_velocity": inputs["landing_velocity"][0],
        "mass": inputs["mass"][0],
        "available_power": np.asarray(inputs["available_power"], dtype=float).reshape(2),
    }


def landing_segment_kinematics_power_values(
    velocity_type,
    landing_time,
    gravity,
    data,
):
    """Return landing segment outputs and dense derivatives."""

    initial_distance = data["initial_distance"]
    initial_time = data["initial_time"]
    altitude = data["altitude"]
    landing_velocity = data["landing_velocity"]
    mass = data["mass"]
    available_power = data["available_power"]
    converted = flight_condition_values(altitude, 0.0, velocity_type, landing_velocity)
    landing_tas = converted["tas"]
    dlanding_tas_daltitude = converted["dtas_daltitude"]
    dlanding_tas_dlanding_velocity = converted["dtas_dvelocity"]
    time = np.asarray([initial_time, initial_time + landing_time])
    distance = np.asarray([initial_distance, initial_distance])
    true_airspeed = np.asarray([landing_tas, 0.0])
    dtrue_airspeed_daltitude = np.asarray([dlanding_tas_daltitude, 0.0])
    dtrue_airspeed_dlanding_velocity = np.asarray(
        [dlanding_tas_dlanding_velocity, 0.0]
    )
    acceleration = np.asarray([-landing_tas / landing_time, 0.0])
    dacceleration_daltitude = np.asarray(
        [-dlanding_tas_daltitude / landing_time, 0.0]
    )
    dacceleration_dlanding_velocity = np.asarray(
        [-dlanding_tas_dlanding_velocity / landing_time, 0.0]
    )
    equivalent_airspeed = np.zeros(2)
    mach = np.zeros(2)
    density = np.zeros(2)
    deas_daltitude = np.zeros(2)
    deas_dlanding_velocity = np.zeros(2)
    dmach_daltitude = np.zeros(2)
    dmach_dlanding_velocity = np.zeros(2)
    ddensity_daltitude = np.zeros(2)

    for index, velocity in enumerate(true_airspeed):
        condition = flight_condition_values(altitude, 0.0, "TAS", velocity)
        equivalent_airspeed[index] = condition["eas"]
        mach[index] = condition["mach"]
        density[index] = condition["density"]
        deas_daltitude[index] = (
            condition["deas_daltitude"]
            + condition["deas_dvelocity"] * dtrue_airspeed_daltitude[index]
        )
        deas_dlanding_velocity[index] = (
            condition["deas_dvelocity"] * dtrue_airspeed_dlanding_velocity[index]
        )
        dmach_daltitude[index] = (
            condition["dmach_daltitude"]
            + condition["dmach_dvelocity"] * dtrue_airspeed_daltitude[index]
        )
        dmach_dlanding_velocity[index] = (
            condition["dmach_dvelocity"] * dtrue_airspeed_dlanding_velocity[index]
        )
        ddensity_daltitude[index] = condition["ddensity_daltitude"]

    rate_of_climb = np.zeros(2)
    flight_path_angle = np.zeros(2)
    required_power = 0.3 * available_power.copy()
    required_power[-1] = 0.0
    specific_excess_power = np.zeros(2)
    potential_energy = np.ones(2) * mass * gravity * altitude
    kinetic_energy = 0.5 * mass * true_airspeed ** 2
    values = {
        "time": time,
        "distance": distance,
        "true_airspeed": true_airspeed,
        "equivalent_airspeed": equivalent_airspeed,
        "mach": mach,
        "density": density,
        "rate_of_climb": rate_of_climb,
        "acceleration": acceleration,
        "flight_path_angle": flight_path_angle,
        "required_power": required_power,
        "specific_excess_power": specific_excess_power,
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
    }
    input_sizes = {
        "initial_distance": 1,
        "initial_time": 1,
        "altitude": 1,
        "landing_velocity": 1,
        "mass": 1,
        "available_power": 2,
    }

    for output_name in landing_segment_kinematics_power_output_names():
        for input_name in landing_segment_kinematics_power_input_names():
            values["d%s_d%s" % (output_name, input_name)] = np.zeros(
                (2, input_sizes[input_name])
            )

    values["dtime_dinitial_time"][:, 0] = 1.0
    values["ddistance_dinitial_distance"][:, 0] = 1.0
    values["dtrue_airspeed_daltitude"][:, 0] = dtrue_airspeed_daltitude
    values["dtrue_airspeed_dlanding_velocity"][:, 0] = (
        dtrue_airspeed_dlanding_velocity
    )
    values["dequivalent_airspeed_daltitude"][:, 0] = deas_daltitude
    values["dequivalent_airspeed_dlanding_velocity"][:, 0] = deas_dlanding_velocity
    values["dmach_daltitude"][:, 0] = dmach_daltitude
    values["dmach_dlanding_velocity"][:, 0] = dmach_dlanding_velocity
    values["ddensity_daltitude"][:, 0] = ddensity_daltitude
    values["dacceleration_daltitude"][:, 0] = dacceleration_daltitude
    values["dacceleration_dlanding_velocity"][:, 0] = dacceleration_dlanding_velocity
    values["drequired_power_davailable_power"][0, 0] = 0.3
    values["dpotential_energy_daltitude"][:, 0] = np.ones(2) * mass * gravity
    values["dpotential_energy_dmass"][:, 0] = np.ones(2) * gravity * altitude
    values["dkinetic_energy_dmass"][:, 0] = 0.5 * true_airspeed ** 2
    values["dkinetic_energy_daltitude"][:, 0] = (
        mass * true_airspeed * dtrue_airspeed_daltitude
    )
    values["dkinetic_energy_dlanding_velocity"][:, 0] = (
        mass * true_airspeed * dtrue_airspeed_dlanding_velocity
    )

    return values


def detailed_takeoff_segment_kinematics_power_input_names():
    """Return detailed takeoff segment kernel input names."""

    return (
        "altitude",
        "target_altitude",
        "target_velocity",
        "mass",
        "wing_loading",
        "available_power",
    )


def detailed_takeoff_segment_kinematics_power_output_names():
    """Return detailed takeoff segment kernel output names."""

    return (
        "time",
        "distance",
        "true_airspeed",
        "equivalent_airspeed",
        "mach",
        "density",
        "lift_drag",
        "drag_power",
        "rate_of_climb",
        "acceleration",
        "flight_path_angle",
        "specific_excess_power",
        "potential_energy",
        "kinetic_energy",
    )


def detailed_takeoff_segment_kinematics_power_inputs(inputs):
    """Return numeric inputs for the detailed takeoff segment kernel."""

    return {
        "altitude": inputs["altitude"][0],
        "target_altitude": inputs["target_altitude"][0],
        "target_velocity": inputs["target_velocity"][0],
        "mass": inputs["mass"][0],
        "wing_loading": inputs["wing_loading"][0],
        "available_power": np.asarray(inputs["available_power"], dtype=float).reshape(-1),
    }


def detailed_takeoff_segment_kinematics_power_values(
    npoint,
    target_velocity_type,
    gravity,
    data,
):
    """Return detailed takeoff segment outputs and dense derivatives."""

    values = detailed_takeoff_segment_kinematics_power_seed_values(
        npoint,
        target_velocity_type,
        gravity,
        data,
        zero_detailed_takeoff_segment_kinematics_power_seeds(npoint),
    )
    input_sizes = detailed_takeoff_segment_kinematics_power_input_sizes(npoint)

    for output_name in detailed_takeoff_segment_kinematics_power_output_names():
        for input_name in detailed_takeoff_segment_kinematics_power_input_names():
            values["d%s_d%s" % (output_name, input_name)] = np.zeros(
                (npoint, input_sizes[input_name])
            )

    for input_name in detailed_takeoff_segment_kinematics_power_input_names():
        for column in range(input_sizes[input_name]):
            seeds = zero_detailed_takeoff_segment_kinematics_power_seeds(npoint)
            seeds[input_name].reshape(-1)[column] = 1.0
            derivative_values = detailed_takeoff_segment_kinematics_power_seed_values(
                npoint,
                target_velocity_type,
                gravity,
                data,
                seeds,
            )

            for output_name in detailed_takeoff_segment_kinematics_power_output_names():
                values["d%s_d%s" % (output_name, input_name)][:, column] = (
                    derivative_values["d%s" % output_name].reshape(-1)
                )

    return values


def detailed_takeoff_segment_kinematics_power_input_sizes(npoint):
    """Return input sizes for detailed takeoff derivative blocks."""

    return {
        "altitude": 1,
        "target_altitude": 1,
        "target_velocity": 1,
        "mass": 1,
        "wing_loading": 1,
        "available_power": npoint,
    }


def zero_detailed_takeoff_segment_kinematics_power_seeds(npoint):
    """Return zero derivative seeds for detailed takeoff."""

    return {
        "altitude": np.zeros(1),
        "target_altitude": np.zeros(1),
        "target_velocity": np.zeros(1),
        "mass": np.zeros(1),
        "wing_loading": np.zeros(1),
        "available_power": np.zeros(npoint),
    }


def detailed_takeoff_segment_kinematics_power_seed_values(
    npoint,
    target_velocity_type,
    gravity,
    data,
    seeds,
):
    """Return detailed takeoff values and one seeded derivative direction."""

    altitude = data["altitude"]
    target_altitude = data["target_altitude"]
    target_velocity = data["target_velocity"]
    mass = data["mass"]
    wing_loading = data["wing_loading"]
    available_power = data["available_power"]
    daltitude = seeds["altitude"][0]
    dtarget_altitude = seeds["target_altitude"][0]
    dtarget_velocity = seeds["target_velocity"][0]
    dmass = seeds["mass"][0]
    dwing_loading = seeds["wing_loading"][0]
    davailable_power = seeds["available_power"]
    target = flight_condition_values(
        target_altitude,
        0.0,
        target_velocity_type,
        target_velocity,
    )
    terminal_speed = target["tas"]
    dterminal_speed = (
        target["dtas_daltitude"] * dtarget_altitude
        + target["dtas_dvelocity"] * dtarget_velocity
    )
    takeoff_density = target["density"]
    dtakeoff_density = target["ddensity_daltitude"] * dtarget_altitude
    fraction = np.linspace(0.0, 1.0, npoint)
    true_airspeed = terminal_speed * fraction
    dtrue_airspeed = dterminal_speed * fraction
    wing_area = mass / wing_loading
    dwing_area = (
        dmass * wing_loading
        - mass * dwing_loading
    ) / wing_loading ** 2
    cl_denominator = takeoff_density * (terminal_speed / 1.1) ** 2 * wing_area
    dcl_denominator = (
        dtakeoff_density * (terminal_speed / 1.1) ** 2 * wing_area
        + takeoff_density * 2.0 * terminal_speed * dterminal_speed / 1.1 ** 2 * wing_area
        + takeoff_density * (terminal_speed / 1.1) ** 2 * dwing_area
    )
    cl_max = 2.0 * mass * gravity / cl_denominator
    dcl_max = (
        2.0 * gravity * dmass * cl_denominator
        - 2.0 * mass * gravity * dcl_denominator
    ) / cl_denominator ** 2
    cd0 = 0.0017
    k_uc = 3.16e-5
    delta_cd0 = wing_loading * k_uc * mass ** -0.215
    ddelta_cd0 = (
        dwing_loading * k_uc * mass ** -0.215
        - 0.215 * wing_loading * k_uc * mass ** -1.215 * dmass
    )
    k1 = 0.02
    k3 = 1.0 / (np.pi * 0.9 * 10.0)
    ground_effect = 0.6
    induced_factor = k1 + ground_effect * k3
    cd = cd0 + delta_cd0 + induced_factor * cl_max ** 2
    dcd = ddelta_cd0 + 2.0 * induced_factor * cl_max * dcl_max
    lift_drag_scalar = cl_max / cd
    dlift_drag_scalar = (dcl_max * cd - cl_max * dcd) / cd ** 2
    lift_drag = np.ones(npoint) * lift_drag_scalar
    dlift_drag = np.ones(npoint) * dlift_drag_scalar
    drag = np.zeros(npoint)
    ddrag = np.zeros(npoint)
    lift = np.zeros(npoint)
    dlift = np.zeros(npoint)
    friction = np.zeros(npoint)
    dfriction = np.zeros(npoint)
    acceleration = np.zeros(npoint)
    dacceleration = np.zeros(npoint)
    thrust = np.zeros(npoint)
    dthrust = np.zeros(npoint)

    for point in range(1, npoint):
        velocity = true_airspeed[point]
        dvelocity = dtrue_airspeed[point]
        thrust[point] = available_power[point] / velocity
        dthrust[point] = (
            davailable_power[point] * velocity
            - available_power[point] * dvelocity
        ) / velocity ** 2
        lift[point] = 0.5 * takeoff_density * velocity ** 2 * cl_max * wing_area
        dlift[point] = 0.5 * (
            dtakeoff_density * velocity ** 2 * cl_max * wing_area
            + takeoff_density * 2.0 * velocity * dvelocity * cl_max * wing_area
            + takeoff_density * velocity ** 2 * dcl_max * wing_area
            + takeoff_density * velocity ** 2 * cl_max * dwing_area
        )
        friction[point] = 0.02 * (mass * gravity - lift[point])
        dfriction[point] = 0.02 * (dmass * gravity - dlift[point])

        if friction[point] < 0.0:
            friction[point] = 0.0
            dfriction[point] = 0.0

        drag[point] = 0.5 * takeoff_density * velocity ** 2 * cd * wing_area
        ddrag[point] = 0.5 * (
            dtakeoff_density * velocity ** 2 * cd * wing_area
            + takeoff_density * 2.0 * velocity * dvelocity * cd * wing_area
            + takeoff_density * velocity ** 2 * dcd * wing_area
            + takeoff_density * velocity ** 2 * cd * dwing_area
        )
        acceleration[point] = (
            thrust[point] - drag[point] - friction[point]
        ) / mass
        dacceleration[point] = (
            (dthrust[point] - ddrag[point] - dfriction[point]) * mass
            - (thrust[point] - drag[point] - friction[point]) * dmass
        ) / mass ** 2

    time_step = np.zeros(npoint)
    dtime_step = np.zeros(npoint)
    distance_step = np.zeros(npoint)
    ddistance_step = np.zeros(npoint)

    for point in range(1, npoint):
        speed_delta = true_airspeed[point] - true_airspeed[point - 1]
        dspeed_delta = dtrue_airspeed[point] - dtrue_airspeed[point - 1]
        speed_square_delta = (
            true_airspeed[point] ** 2
            - true_airspeed[point - 1] ** 2
        )
        dspeed_square_delta = (
            2.0 * true_airspeed[point] * dtrue_airspeed[point]
            - 2.0 * true_airspeed[point - 1] * dtrue_airspeed[point - 1]
        )
        time_step[point] = speed_delta / acceleration[point]
        dtime_step[point] = (
            dspeed_delta * acceleration[point]
            - speed_delta * dacceleration[point]
        ) / acceleration[point] ** 2
        distance_step[point] = speed_square_delta / (2.0 * acceleration[point])
        ddistance_step[point] = (
            dspeed_square_delta * 2.0 * acceleration[point]
            - speed_square_delta * 2.0 * dacceleration[point]
        ) / (2.0 * acceleration[point]) ** 2

    time = np.cumsum(time_step)
    dtime = np.cumsum(dtime_step)
    distance = np.cumsum(distance_step)
    ddistance = np.cumsum(ddistance_step)
    drag_power = drag * true_airspeed
    ddrag_power = ddrag * true_airspeed + drag * dtrue_airspeed
    specific_excess_power = (available_power - drag_power) / (mass * gravity)
    dspecific_excess_power = (
        (davailable_power - ddrag_power) * mass * gravity
        - (available_power - drag_power) * dmass * gravity
    ) / (mass * gravity) ** 2
    equivalent_airspeed = np.zeros(npoint)
    dequivalent_airspeed = np.zeros(npoint)
    mach = np.zeros(npoint)
    dmach = np.zeros(npoint)
    density = np.zeros(npoint)
    ddensity = np.zeros(npoint)

    for point, velocity in enumerate(true_airspeed):
        condition = flight_condition_values(altitude, 0.0, "TAS", velocity)
        equivalent_airspeed[point] = condition["eas"]
        mach[point] = condition["mach"]
        density[point] = condition["density"]
        dequivalent_airspeed[point] = (
            condition["deas_daltitude"] * daltitude
            + condition["deas_dvelocity"] * dtrue_airspeed[point]
        )
        dmach[point] = (
            condition["dmach_daltitude"] * daltitude
            + condition["dmach_dvelocity"] * dtrue_airspeed[point]
        )
        ddensity[point] = condition["ddensity_daltitude"] * daltitude

    rate_of_climb = np.zeros(npoint)
    drate_of_climb = np.zeros(npoint)
    flight_path_angle = np.zeros(npoint)
    dflight_path_angle = np.zeros(npoint)
    potential_energy = np.ones(npoint) * mass * gravity * altitude
    dpotential_energy = np.ones(npoint) * gravity * (
        dmass * altitude
        + mass * daltitude
    )
    kinetic_energy = 0.5 * mass * true_airspeed ** 2
    dkinetic_energy = (
        0.5 * dmass * true_airspeed ** 2
        + mass * true_airspeed * dtrue_airspeed
    )

    return {
        "time": time,
        "distance": distance,
        "true_airspeed": true_airspeed,
        "equivalent_airspeed": equivalent_airspeed,
        "mach": mach,
        "density": density,
        "lift_drag": lift_drag,
        "drag_power": drag_power,
        "rate_of_climb": rate_of_climb,
        "acceleration": acceleration,
        "flight_path_angle": flight_path_angle,
        "specific_excess_power": specific_excess_power,
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
        "dtime": dtime,
        "ddistance": ddistance,
        "dtrue_airspeed": dtrue_airspeed,
        "dequivalent_airspeed": dequivalent_airspeed,
        "dmach": dmach,
        "ddensity": ddensity,
        "dlift_drag": dlift_drag,
        "ddrag_power": ddrag_power,
        "drate_of_climb": drate_of_climb,
        "dacceleration": dacceleration,
        "dflight_path_angle": dflight_path_angle,
        "dspecific_excess_power": dspecific_excess_power,
        "dpotential_energy": dpotential_energy,
        "dkinetic_energy": dkinetic_energy,
    }


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


def row_matrix_values(value, rows):
    """Return FAST row-matrix expansion and derivative block."""

    value = np.asarray(value, dtype=float)

    if value.ndim == 0:
        value = value.reshape(1)

    value = value.reshape(-1)
    value_size = value.size
    matrix = np.tile(value.reshape(1, -1), (rows, 1))
    derivative = np.zeros((rows * value_size, value_size))

    for row in range(rows):
        for column in range(value_size):
            derivative[row * value_size + column, column] = 1.0

    return {
        "matrix": matrix,
        "dmatrix_dvalue": derivative,
    }


def history_vector_slice_values(history_vector, values, start, stop):
    """Return FAST history-vector slice assignment and Jacobians."""

    history = np.asarray(history_vector, dtype=float).reshape(-1)
    assigned = np.asarray(values, dtype=float).reshape(-1)
    updated = history.copy()
    updated[start:stop] = assigned
    dhistory = np.eye(history.size)
    dvalues = np.zeros((history.size, assigned.size))
    dhistory[start:stop, start:stop] = 0.0

    for row, source in enumerate(range(start, stop)):
        dvalues[source, row] = 1.0

    return {
        "updated_history_vector": updated,
        "dupdated_history_vector_dhistory_vector": dhistory,
        "dupdated_history_vector_dvalues": dvalues,
    }


def history_matrix_slice_values(history_matrix, values, start, stop):
    """Return FAST history-matrix row-slice assignment and Jacobians."""

    history = np.asarray(history_matrix, dtype=float)
    assigned = np.asarray(values, dtype=float)
    updated = history.copy()
    updated[start:stop, :] = assigned
    output_size = history.size
    value_size = assigned.size
    num_cols = history.shape[1]
    dhistory = np.eye(output_size)
    dvalues = np.zeros((output_size, value_size))

    for row in range(start, stop):
        for col in range(num_cols):
            output_index = row * num_cols + col
            value_index = (row - start) * num_cols + col
            dhistory[output_index, output_index] = 0.0
            dvalues[output_index, value_index] = 1.0

    return {
        "updated_history_matrix": updated,
        "dupdated_history_matrix_dhistory_matrix": dhistory,
        "dupdated_history_matrix_dvalues": dvalues,
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


def breguet_source_delta_values(source_energy, source_energy_left, columns, delta):
    """Return Breguet source-delta outputs and derivative blocks."""

    source_energy = np.asarray(source_energy, dtype=float)
    source_energy_left = np.asarray(source_energy_left, dtype=float)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    npoint, nsrc = source_energy.shape
    updated_source_energy = source_energy.copy()
    updated_source_energy_left = source_energy_left.copy()
    columns = tuple(int(column) for column in columns)

    apply_breguet_source_delta(
        updated_source_energy,
        updated_source_energy_left,
        columns,
        delta,
    )

    source_size = npoint * nsrc
    denergy_denergy = np.eye(source_size)
    denergy_dleft = np.zeros((source_size, source_size))
    denergy_ddelta = np.zeros((source_size, npoint))
    dleft_denergy = np.zeros((source_size, source_size))
    dleft_dleft = np.eye(source_size)
    dleft_ddelta = np.zeros((source_size, npoint))

    if len(columns) > 0:
        scale = 1.0 / len(columns)
        for row in range(npoint):
            for column in columns:
                output_index = row * nsrc + column
                first_row_index = column
                denergy_denergy[output_index, :] = 0.0
                denergy_denergy[output_index, first_row_index] = 1.0
                dleft_dleft[output_index, :] = 0.0
                dleft_dleft[output_index, first_row_index] = 1.0
                denergy_ddelta[output_index, row] = scale
                dleft_ddelta[output_index, row] = -scale

    return {
        "updated_source_energy": updated_source_energy,
        "updated_source_energy_left": updated_source_energy_left,
        "dupdated_source_energy_dsource_energy": denergy_denergy,
        "dupdated_source_energy_dsource_energy_left": denergy_dleft,
        "dupdated_source_energy_ddelta": denergy_ddelta,
        "dupdated_source_energy_left_dsource_energy": dleft_denergy,
        "dupdated_source_energy_left_dsource_energy_left": dleft_dleft,
        "dupdated_source_energy_left_ddelta": dleft_ddelta,
    }


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


def cruise_segment_kinematics_power_input_names():
    """Return smooth EvalCruise kernel input names."""

    return (
        "initial_distance",
        "initial_time",
        "altitude",
        "true_airspeed",
        "mass",
        "available_power",
        "target_distance",
        "lift_drag",
    )


def cruise_segment_kinematics_power_output_names():
    """Return smooth EvalCruise kernel output names."""

    return (
        "distance",
        "time",
        "time_step",
        "rate_of_climb",
        "acceleration",
        "flight_path_angle",
        "drag_power",
        "required_power",
        "specific_excess_power",
        "potential_energy",
        "kinetic_energy",
    )


def cruise_segment_kinematics_power_inputs(inputs):
    """Return numeric inputs for the smooth EvalCruise kernel."""

    return {
        "initial_distance": inputs["initial_distance"][0],
        "initial_time": inputs["initial_time"][0],
        "altitude": np.asarray(inputs["altitude"], dtype=float).reshape(-1),
        "true_airspeed": np.asarray(inputs["true_airspeed"], dtype=float).reshape(-1),
        "mass": np.asarray(inputs["mass"], dtype=float).reshape(-1),
        "available_power": np.asarray(inputs["available_power"], dtype=float).reshape(-1),
        "target_distance": inputs["target_distance"][0],
        "lift_drag": inputs["lift_drag"][0],
    }


def cruise_segment_kinematics_power_values(npoint, gravity, data):
    """Return smooth EvalCruise kernel outputs and dense derivatives."""

    values = cruise_segment_kinematics_power_seed_values(
        npoint,
        gravity,
        data,
        zero_cruise_segment_kinematics_power_seeds(npoint),
    )
    input_sizes = cruise_segment_kinematics_power_input_sizes(npoint)
    output_sizes = cruise_segment_kinematics_power_output_sizes(npoint)

    for output_name in cruise_segment_kinematics_power_output_names():
        for input_name in cruise_segment_kinematics_power_input_names():
            values["d%s_d%s" % (output_name, input_name)] = np.zeros(
                (output_sizes[output_name], input_sizes[input_name])
            )

    for input_name in cruise_segment_kinematics_power_input_names():
        for column in range(input_sizes[input_name]):
            seeds = zero_cruise_segment_kinematics_power_seeds(npoint)
            seeds[input_name].reshape(-1)[column] = 1.0
            derivative_values = cruise_segment_kinematics_power_seed_values(
                npoint,
                gravity,
                data,
                seeds,
            )

            for output_name in cruise_segment_kinematics_power_output_names():
                values["d%s_d%s" % (output_name, input_name)][:, column] = (
                    derivative_values["d%s" % output_name].reshape(-1)
                )

    return values


def cruise_segment_kinematics_power_input_sizes(npoint):
    """Return input sizes for smooth EvalCruise dense derivative blocks."""

    return {
        "initial_distance": 1,
        "initial_time": 1,
        "altitude": npoint,
        "true_airspeed": npoint,
        "mass": npoint,
        "available_power": npoint,
        "target_distance": 1,
        "lift_drag": 1,
    }


def cruise_segment_kinematics_power_output_sizes(npoint):
    """Return output sizes for smooth EvalCruise dense derivative blocks."""

    nstep = npoint - 1
    return {
        "distance": npoint,
        "time": npoint,
        "time_step": nstep,
        "rate_of_climb": npoint,
        "acceleration": npoint,
        "flight_path_angle": npoint,
        "drag_power": npoint,
        "required_power": npoint,
        "specific_excess_power": npoint,
        "potential_energy": npoint,
        "kinetic_energy": npoint,
    }


def zero_cruise_segment_kinematics_power_seeds(npoint):
    """Return zero derivative seeds for the smooth EvalCruise kernel."""

    return {
        "initial_distance": np.zeros(1),
        "initial_time": np.zeros(1),
        "altitude": np.zeros(npoint),
        "true_airspeed": np.zeros(npoint),
        "mass": np.zeros(npoint),
        "available_power": np.zeros(npoint),
        "target_distance": np.zeros(1),
        "lift_drag": np.zeros(1),
    }


def cruise_segment_kinematics_power_seed_values(npoint, gravity, data, seeds):
    """Return smooth EvalCruise outputs and one seeded derivative direction."""

    altitude = data["altitude"]
    true_airspeed = data["true_airspeed"]
    mass = data["mass"]
    available_power = data["available_power"]
    lift_drag = data["lift_drag"]
    initial_distance = data["initial_distance"]
    initial_time = data["initial_time"]
    target_distance = data["target_distance"]
    daltitude = seeds["altitude"]
    dtrue_airspeed = seeds["true_airspeed"]
    dmass = seeds["mass"]
    davailable_power = seeds["available_power"]
    dlift_drag = seeds["lift_drag"][0]
    dinitial_distance = seeds["initial_distance"][0]
    dinitial_time = seeds["initial_time"][0]
    dtarget_distance = seeds["target_distance"][0]
    alpha = np.linspace(0.0, 1.0, npoint)
    distance = initial_distance + alpha * (target_distance - initial_distance)
    ddistance = (
        dinitial_distance
        + alpha * (dtarget_distance - dinitial_distance)
    )
    distance_step = np.diff(distance)
    ddistance_step = np.diff(ddistance)
    time_step = distance_step / true_airspeed[:-1]
    dtime_step = (
        ddistance_step * true_airspeed[:-1]
        - distance_step * dtrue_airspeed[:-1]
    ) / true_airspeed[:-1] ** 2
    time = np.zeros(npoint)
    dtime = np.zeros(npoint)
    time[0] = initial_time
    dtime[0] = dinitial_time
    time[1:] = initial_time + np.cumsum(time_step)
    dtime[1:] = dinitial_time + np.cumsum(dtime_step)
    altitude_step = np.diff(altitude)
    daltitude_step = np.diff(daltitude)
    rate_of_climb = np.zeros(npoint)
    drate_of_climb = np.zeros(npoint)
    rate_of_climb[:-1] = altitude_step / time_step
    drate_of_climb[:-1] = (
        daltitude_step * time_step
        - altitude_step * dtime_step
    ) / time_step ** 2
    speed_step = np.diff(true_airspeed)
    dspeed_step = np.diff(dtrue_airspeed)
    acceleration = np.zeros(npoint)
    dacceleration = np.zeros(npoint)
    acceleration[:-1] = speed_step / time_step
    dacceleration[:-1] = (
        dspeed_step * time_step
        - speed_step * dtime_step
    ) / time_step ** 2
    climb_ratio = rate_of_climb / true_airspeed
    dclimb_ratio = (
        drate_of_climb * true_airspeed
        - rate_of_climb * dtrue_airspeed
    ) / true_airspeed ** 2
    flight_path_angle = np.degrees(np.arcsin(climb_ratio))
    dflight_path_angle = (
        180.0
        / np.pi
        * dclimb_ratio
        / np.sqrt(1.0 - climb_ratio ** 2)
    )
    cosine_fpa = np.sqrt(1.0 - climb_ratio ** 2)
    dcosine_fpa = -climb_ratio * dclimb_ratio / cosine_fpa
    drag_power = mass * gravity * cosine_fpa * true_airspeed / lift_drag
    ddrag_power = gravity * (
        (
            dmass * cosine_fpa * true_airspeed
            + mass * dcosine_fpa * true_airspeed
            + mass * cosine_fpa * dtrue_airspeed
        )
        / lift_drag
        - mass * cosine_fpa * true_airspeed * dlift_drag / lift_drag ** 2
    )
    required_power = (
        mass * gravity * rate_of_climb
        + mass * true_airspeed * acceleration
        + drag_power
    )
    drequired_power = (
        gravity * (dmass * rate_of_climb + mass * drate_of_climb)
        + dmass * true_airspeed * acceleration
        + mass * dtrue_airspeed * acceleration
        + mass * true_airspeed * dacceleration
        + ddrag_power
    )
    specific_excess_power = (
        available_power - drag_power
    ) / (mass * gravity)
    dspecific_excess_power = (
        (davailable_power - ddrag_power) * mass * gravity
        - (available_power - drag_power) * dmass * gravity
    ) / (mass * gravity) ** 2
    potential_energy = mass * gravity * altitude
    dpotential_energy = gravity * (dmass * altitude + mass * daltitude)
    kinetic_energy = 0.5 * mass * true_airspeed ** 2
    dkinetic_energy = (
        0.5 * dmass * true_airspeed ** 2
        + mass * true_airspeed * dtrue_airspeed
    )

    return {
        "distance": distance,
        "time": time,
        "time_step": time_step,
        "rate_of_climb": rate_of_climb,
        "acceleration": acceleration,
        "flight_path_angle": flight_path_angle,
        "drag_power": drag_power,
        "required_power": required_power,
        "specific_excess_power": specific_excess_power,
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
        "ddistance": ddistance,
        "dtime": dtime,
        "dtime_step": dtime_step,
        "drate_of_climb": drate_of_climb,
        "dacceleration": dacceleration,
        "dflight_path_angle": dflight_path_angle,
        "ddrag_power": ddrag_power,
        "drequired_power": drequired_power,
        "dspecific_excess_power": dspecific_excess_power,
        "dpotential_energy": dpotential_energy,
        "dkinetic_energy": dkinetic_energy,
    }


class CruiseSegmentKinematicsPower(om.ExplicitComponent):
    """Compute FAST EvalCruise smooth-branch trajectory and required power.

    Inputs:
        initial_distance: Segment starting distance in m.
        initial_time: Segment starting time in s.
        altitude: Segment altitude control-point history in m.
        true_airspeed: Segment TAS control-point history in m/s.
        mass: Segment mass control-point history in kg.
        available_power: Total available propulsive power in W.
        target_distance: Segment ending distance in m.
        lift_drag: Cruise lift-to-drag ratio.

    Outputs:
        distance, time, time_step, rate_of_climb, acceleration, flight_path_angle,
        drag_power, required_power, specific_excess_power, potential_energy, and
        kinetic_energy histories.

    Assumptions:
        This is the smooth EvalCruise branch before propulsion-history
        mutation. It assumes the prescribed altitude and speed histories do not
        hit FAST's climb/descent-rate clipping branch.
    """

    def initialize(self):
        self.options.declare("npoint", default=3)
        self.options.declare("gravity", default=9.81)

    def setup(self):
        npoint = self.options["npoint"]
        nstep = npoint - 1
        self.add_input("initial_distance", val=0.0, units="m")
        self.add_input("initial_time", val=0.0, units="s")
        self.add_input("altitude", val=np.zeros(npoint), units="m")
        self.add_input("true_airspeed", val=np.ones(npoint), units="m/s")
        self.add_input("mass", val=np.ones(npoint), units="kg")
        self.add_input("available_power", val=np.zeros(npoint), units="W")
        self.add_input("target_distance", val=1.0, units="m")
        self.add_input("lift_drag", val=10.0)
        self.add_output("distance", val=np.zeros(npoint), units="m")
        self.add_output("time", val=np.zeros(npoint), units="s")
        self.add_output("time_step", val=np.ones(nstep), units="s")
        self.add_output("rate_of_climb", val=np.zeros(npoint), units="m/s")
        self.add_output("acceleration", val=np.zeros(npoint), units="m/s**2")
        self.add_output("flight_path_angle", val=np.zeros(npoint))
        self.add_output("drag_power", val=np.zeros(npoint), units="W")
        self.add_output("required_power", val=np.zeros(npoint), units="W")
        self.add_output("specific_excess_power", val=np.zeros(npoint), units="m/s")
        self.add_output("potential_energy", val=np.zeros(npoint), units="J")
        self.add_output("kinetic_energy", val=np.zeros(npoint), units="J")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = cruise_segment_kinematics_power_values(
            self.options["npoint"],
            self.options["gravity"],
            cruise_segment_kinematics_power_inputs(inputs),
        )

        for output_name in cruise_segment_kinematics_power_output_names():
            outputs[output_name] = values[output_name]

    def compute_partials(self, inputs, partials):
        data = cruise_segment_kinematics_power_inputs(inputs)
        values = cruise_segment_kinematics_power_values(
            self.options["npoint"],
            self.options["gravity"],
            data,
        )

        for output_name in cruise_segment_kinematics_power_output_names():
            for input_name in cruise_segment_kinematics_power_input_names():
                partials[output_name, input_name] = values[
                    "d%s_d%s" % (output_name, input_name)
                ]


class PrescribedRateSegmentKinematicsPower(om.ExplicitComponent):
    """Compute FAST prescribed-rate climb/descent kinematics and power.

    Inputs:
        initial_distance: Segment starting distance in m.
        initial_time: Segment starting time in s.
        altitude: Segment altitude control-point history in m.
        true_airspeed: Segment TAS control-point history in m/s.
        mass: Segment mass control-point history in kg.
        available_power: Total available propulsive power in W.
        rate_of_climb: Prescribed vertical-rate history in m/s.
        lift_drag: Segment lift-to-drag ratio.

    Outputs:
        Smooth branch distance, time, time_step, acceleration,
        flight_path_angle, drag_power, required_power, specific_excess_power,
        potential_energy, and kinetic_energy histories.

    Assumptions:
        This matches the FAST prescribed-rate branch before propulsion-history
        mutation and before acceleration limiting. When ``idle_floor`` is
        enabled, negative required power is clipped to FAST's descent idle
        bookkeeping value and that clipped branch has zero local derivatives.
    """

    def initialize(self):
        self.options.declare("npoint", default=3)
        self.options.declare("gravity", default=9.81)
        self.options.declare("idle_floor", default=False)
        self.options.declare("idle_power", default=0.0001)

    def setup(self):
        npoint = self.options["npoint"]
        nstep = npoint - 1
        self.add_input("initial_distance", val=0.0, units="m")
        self.add_input("initial_time", val=0.0, units="s")
        self.add_input("altitude", val=np.zeros(npoint), units="m")
        self.add_input("true_airspeed", val=np.ones(npoint), units="m/s")
        self.add_input("mass", val=np.ones(npoint), units="kg")
        self.add_input("available_power", val=np.zeros(npoint), units="W")
        self.add_input("rate_of_climb", val=np.zeros(npoint), units="m/s")
        self.add_input("lift_drag", val=10.0)
        self.add_output("distance", val=np.zeros(npoint), units="m")
        self.add_output("time", val=np.zeros(npoint), units="s")
        self.add_output("time_step", val=np.ones(nstep), units="s")
        self.add_output("acceleration", val=np.zeros(npoint), units="m/s**2")
        self.add_output("flight_path_angle", val=np.zeros(npoint))
        self.add_output("drag_power", val=np.zeros(npoint), units="W")
        self.add_output("required_power", val=np.zeros(npoint), units="W")
        self.add_output("specific_excess_power", val=np.zeros(npoint), units="m/s")
        self.add_output("potential_energy", val=np.zeros(npoint), units="J")
        self.add_output("kinetic_energy", val=np.zeros(npoint), units="J")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = prescribed_rate_segment_kinematics_power_values(
            self.options["npoint"],
            self.options["gravity"],
            self.options["idle_floor"],
            self.options["idle_power"],
            prescribed_rate_segment_kinematics_power_inputs(inputs),
        )

        for output_name in prescribed_rate_segment_kinematics_power_output_names():
            outputs[output_name] = values[output_name]

    def compute_partials(self, inputs, partials):
        values = prescribed_rate_segment_kinematics_power_values(
            self.options["npoint"],
            self.options["gravity"],
            self.options["idle_floor"],
            self.options["idle_power"],
            prescribed_rate_segment_kinematics_power_inputs(inputs),
        )

        for output_name in prescribed_rate_segment_kinematics_power_output_names():
            for input_name in prescribed_rate_segment_kinematics_power_input_names():
                partials[output_name, input_name] = values[
                    "d%s_d%s" % (output_name, input_name)
                ]


def prescribed_rate_segment_kinematics_power_input_names():
    """Return prescribed-rate segment kernel input names."""

    return (
        "initial_distance",
        "initial_time",
        "altitude",
        "true_airspeed",
        "mass",
        "available_power",
        "rate_of_climb",
        "lift_drag",
    )


def prescribed_rate_segment_kinematics_power_output_names():
    """Return prescribed-rate segment kernel output names."""

    return (
        "distance",
        "time",
        "time_step",
        "acceleration",
        "flight_path_angle",
        "drag_power",
        "required_power",
        "specific_excess_power",
        "potential_energy",
        "kinetic_energy",
    )


def prescribed_rate_segment_kinematics_power_inputs(inputs):
    """Return numeric inputs for the prescribed-rate segment kernel."""

    return {
        "initial_distance": inputs["initial_distance"][0],
        "initial_time": inputs["initial_time"][0],
        "altitude": np.asarray(inputs["altitude"], dtype=float).reshape(-1),
        "true_airspeed": np.asarray(inputs["true_airspeed"], dtype=float).reshape(-1),
        "mass": np.asarray(inputs["mass"], dtype=float).reshape(-1),
        "available_power": np.asarray(inputs["available_power"], dtype=float).reshape(-1),
        "rate_of_climb": np.asarray(inputs["rate_of_climb"], dtype=float).reshape(-1),
        "lift_drag": inputs["lift_drag"][0],
    }


def prescribed_rate_segment_kinematics_power_values(
    npoint,
    gravity,
    idle_floor,
    idle_power,
    data,
):
    """Return prescribed-rate segment outputs and dense derivatives."""

    values = prescribed_rate_segment_kinematics_power_seed_values(
        npoint,
        gravity,
        idle_floor,
        idle_power,
        data,
        zero_prescribed_rate_segment_kinematics_power_seeds(npoint),
    )
    input_sizes = prescribed_rate_segment_kinematics_power_input_sizes(npoint)
    output_sizes = prescribed_rate_segment_kinematics_power_output_sizes(npoint)

    for output_name in prescribed_rate_segment_kinematics_power_output_names():
        for input_name in prescribed_rate_segment_kinematics_power_input_names():
            values["d%s_d%s" % (output_name, input_name)] = np.zeros(
                (output_sizes[output_name], input_sizes[input_name])
            )

    for input_name in prescribed_rate_segment_kinematics_power_input_names():
        for column in range(input_sizes[input_name]):
            seeds = zero_prescribed_rate_segment_kinematics_power_seeds(npoint)
            seeds[input_name].reshape(-1)[column] = 1.0
            derivative_values = prescribed_rate_segment_kinematics_power_seed_values(
                npoint,
                gravity,
                idle_floor,
                idle_power,
                data,
                seeds,
            )

            for output_name in prescribed_rate_segment_kinematics_power_output_names():
                values["d%s_d%s" % (output_name, input_name)][:, column] = (
                    derivative_values["d%s" % output_name].reshape(-1)
                )

    return values


def prescribed_rate_segment_kinematics_power_input_sizes(npoint):
    """Return input sizes for prescribed-rate dense derivative blocks."""

    return {
        "initial_distance": 1,
        "initial_time": 1,
        "altitude": npoint,
        "true_airspeed": npoint,
        "mass": npoint,
        "available_power": npoint,
        "rate_of_climb": npoint,
        "lift_drag": 1,
    }


def prescribed_rate_segment_kinematics_power_output_sizes(npoint):
    """Return output sizes for prescribed-rate dense derivative blocks."""

    nstep = npoint - 1
    return {
        "distance": npoint,
        "time": npoint,
        "time_step": nstep,
        "acceleration": npoint,
        "flight_path_angle": npoint,
        "drag_power": npoint,
        "required_power": npoint,
        "specific_excess_power": npoint,
        "potential_energy": npoint,
        "kinetic_energy": npoint,
    }


def zero_prescribed_rate_segment_kinematics_power_seeds(npoint):
    """Return zero derivative seeds for the prescribed-rate segment kernel."""

    return {
        "initial_distance": np.zeros(1),
        "initial_time": np.zeros(1),
        "altitude": np.zeros(npoint),
        "true_airspeed": np.zeros(npoint),
        "mass": np.zeros(npoint),
        "available_power": np.zeros(npoint),
        "rate_of_climb": np.zeros(npoint),
        "lift_drag": np.zeros(1),
    }


def prescribed_rate_segment_kinematics_power_seed_values(
    npoint,
    gravity,
    idle_floor,
    idle_power,
    data,
    seeds,
):
    """Return prescribed-rate outputs and one seeded derivative direction."""

    altitude = data["altitude"]
    true_airspeed = data["true_airspeed"]
    mass = data["mass"]
    available_power = data["available_power"]
    rate_of_climb = data["rate_of_climb"]
    lift_drag = data["lift_drag"]
    initial_distance = data["initial_distance"]
    initial_time = data["initial_time"]
    daltitude = seeds["altitude"]
    dtrue_airspeed = seeds["true_airspeed"]
    dmass = seeds["mass"]
    davailable_power = seeds["available_power"]
    drate_of_climb = seeds["rate_of_climb"]
    dlift_drag = seeds["lift_drag"][0]
    dinitial_distance = seeds["initial_distance"][0]
    dinitial_time = seeds["initial_time"][0]
    altitude_step = np.diff(altitude)
    daltitude_step = np.diff(daltitude)
    time_step = altitude_step / rate_of_climb[:-1]
    dtime_step = (
        daltitude_step * rate_of_climb[:-1]
        - altitude_step * drate_of_climb[:-1]
    ) / rate_of_climb[:-1] ** 2
    time = np.zeros(npoint)
    dtime = np.zeros(npoint)
    time[0] = initial_time
    dtime[0] = dinitial_time
    time[1:] = initial_time + np.cumsum(time_step)
    dtime[1:] = dinitial_time + np.cumsum(dtime_step)
    speed_step = np.diff(true_airspeed)
    dspeed_step = np.diff(dtrue_airspeed)
    acceleration = np.zeros(npoint)
    dacceleration = np.zeros(npoint)
    acceleration[:-1] = speed_step / time_step
    dacceleration[:-1] = (
        dspeed_step * time_step
        - speed_step * dtime_step
    ) / time_step ** 2
    climb_ratio = rate_of_climb / true_airspeed
    dclimb_ratio = (
        drate_of_climb * true_airspeed
        - rate_of_climb * dtrue_airspeed
    ) / true_airspeed ** 2
    flight_path_angle = np.degrees(np.arcsin(climb_ratio))
    dflight_path_angle = (
        180.0
        / np.pi
        * dclimb_ratio
        / np.sqrt(1.0 - climb_ratio ** 2)
    )
    cosine_fpa = np.sqrt(1.0 - climb_ratio ** 2)
    dcosine_fpa = -climb_ratio * dclimb_ratio / cosine_fpa
    ground_speed = true_airspeed * cosine_fpa
    dground_speed = dtrue_airspeed * cosine_fpa + true_airspeed * dcosine_fpa
    distance = np.zeros(npoint)
    ddistance = np.zeros(npoint)
    distance[0] = initial_distance
    ddistance[0] = dinitial_distance
    distance_step = ground_speed[:-1] * time_step
    ddistance_step = dground_speed[:-1] * time_step + ground_speed[:-1] * dtime_step
    distance[1:] = initial_distance + np.cumsum(distance_step)
    ddistance[1:] = dinitial_distance + np.cumsum(ddistance_step)
    drag_power = mass * gravity * cosine_fpa * true_airspeed / lift_drag
    ddrag_power = gravity * (
        (
            dmass * cosine_fpa * true_airspeed
            + mass * dcosine_fpa * true_airspeed
            + mass * cosine_fpa * dtrue_airspeed
        )
        / lift_drag
        - mass * cosine_fpa * true_airspeed * dlift_drag / lift_drag ** 2
    )
    required_power = (
        mass * gravity * rate_of_climb
        + mass * true_airspeed * acceleration
        + drag_power
    )
    drequired_power = (
        gravity * (dmass * rate_of_climb + mass * drate_of_climb)
        + dmass * true_airspeed * acceleration
        + mass * dtrue_airspeed * acceleration
        + mass * true_airspeed * dacceleration
        + ddrag_power
    )

    if idle_floor:
        clipped = required_power < idle_power
        required_power[clipped] = idle_power
        drequired_power[clipped] = 0.0

    specific_excess_power = (
        available_power - drag_power
    ) / (mass * gravity)
    dspecific_excess_power = (
        (davailable_power - ddrag_power) * mass * gravity
        - (available_power - drag_power) * dmass * gravity
    ) / (mass * gravity) ** 2
    potential_energy = mass * gravity * altitude
    dpotential_energy = gravity * (dmass * altitude + mass * daltitude)
    kinetic_energy = 0.5 * mass * true_airspeed ** 2
    dkinetic_energy = (
        0.5 * dmass * true_airspeed ** 2
        + mass * true_airspeed * dtrue_airspeed
    )

    return {
        "distance": distance,
        "time": time,
        "time_step": time_step,
        "acceleration": acceleration,
        "flight_path_angle": flight_path_angle,
        "drag_power": drag_power,
        "required_power": required_power,
        "specific_excess_power": specific_excess_power,
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
        "ddistance": ddistance,
        "dtime": dtime,
        "dtime_step": dtime_step,
        "dacceleration": dacceleration,
        "dflight_path_angle": dflight_path_angle,
        "ddrag_power": ddrag_power,
        "drequired_power": drequired_power,
        "dspecific_excess_power": dspecific_excess_power,
        "dpotential_energy": dpotential_energy,
        "dkinetic_energy": dkinetic_energy,
    }


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

    if tas == 0.0:
        deas_daltitude = 0.0
        deas_ddisa = 0.0
    else:
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
