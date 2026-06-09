# src/fast_openmdao/propulsion.py

"""OpenMDAO components for FAST propulsion primitive equations."""

import numpy as np
import openmdao.api as om

from fast_python.atmosphere import standard_atmosphere
from fast_openmdao.regression import gaussian_process_prediction_values


RHO_SL_STD = standard_atmosphere(0.0)[2]


class EngineLapse(om.ExplicitComponent):
    """Estimate lapsed thrust or power from sea-level-static output.

    Inputs:
        sea_level_static: Sea-level-static thrust or power.
        density: Air density in kg/m**3.

    Outputs:
        lapsed_output: Lapsed thrust or power in the same units as input.

    Assumptions:
        Aircraft class is a discrete option. Turbofan output scales linearly
        with density ratio; turboprop and piston output use FAST's zero
        exponent and therefore remain equal to sea-level-static output.
    """

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")

    def setup(self):
        self.add_input("sea_level_static", val=1.0)
        self.add_input("density", val=RHO_SL_STD, units="kg/m**3")
        self.add_output("lapsed_output", val=1.0)
        self.declare_partials(of="lapsed_output", wrt="*")

    def compute(self, inputs, outputs):
        outputs["lapsed_output"] = engine_lapse_value(
            inputs["sea_level_static"][0],
            self.options["aircraft_class"],
            inputs["density"][0],
        )

    def compute_partials(self, inputs, partials):
        derivatives = engine_lapse_derivatives(
            inputs["sea_level_static"][0],
            self.options["aircraft_class"],
            inputs["density"][0],
        )
        partials["lapsed_output", "sea_level_static"] = derivatives[
            "doutput_dsea_level_static"
        ]
        partials["lapsed_output", "density"] = derivatives["doutput_ddensity"]


class SafeComponentWeight(om.ExplicitComponent):
    """Compute component weight from power and power-to-weight ratio."""

    def setup(self):
        self.add_input("power", val=1.0, units="W")
        self.add_input("power_to_weight", val=1.0)
        self.add_output("component_weight", val=1.0, units="kg")
        self.declare_partials(of="component_weight", wrt="*")

    def compute(self, inputs, outputs):
        outputs["component_weight"] = safe_component_weight_value(
            inputs["power"][0],
            inputs["power_to_weight"][0],
        )

    def compute_partials(self, inputs, partials):
        power = inputs["power"][0]
        power_to_weight = inputs["power_to_weight"][0]

        if abs(power) < 1.0e-12:
            partials["component_weight", "power"] = 0.0
            partials["component_weight", "power_to_weight"] = 0.0
            return

        partials["component_weight", "power"] = 1.0 / power_to_weight
        partials["component_weight", "power_to_weight"] = (
            -power / power_to_weight ** 2
        )


class ThrustSinkEfficiency(om.ExplicitComponent):
    """Select FAST thrust-sink fan or propeller efficiency."""

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")

    def setup(self):
        self.add_input("fan_efficiency", val=0.9)
        self.add_input("propeller_efficiency", val=0.85)
        self.add_output("thrust_sink_efficiency", val=0.9)
        self.declare_partials(of="thrust_sink_efficiency", wrt="*")

    def compute(self, inputs, outputs):
        aircraft_class = self.options["aircraft_class"].lower()

        if aircraft_class == "turbofan":
            outputs["thrust_sink_efficiency"] = inputs["fan_efficiency"][0]
        elif aircraft_class in ("turboprop", "piston"):
            outputs["thrust_sink_efficiency"] = inputs["propeller_efficiency"][0]
        else:
            raise ValueError(f"Invalid aircraft class: {self.options['aircraft_class']}")

    def compute_partials(self, inputs, partials):
        aircraft_class = self.options["aircraft_class"].lower()
        partials["thrust_sink_efficiency", "fan_efficiency"] = 0.0
        partials["thrust_sink_efficiency", "propeller_efficiency"] = 0.0

        if aircraft_class == "turbofan":
            partials["thrust_sink_efficiency", "fan_efficiency"] = 1.0
        elif aircraft_class in ("turboprop", "piston"):
            partials["thrust_sink_efficiency", "propeller_efficiency"] = 1.0
        else:
            raise ValueError(f"Invalid aircraft class: {self.options['aircraft_class']}")


class TransmitterFanEfficiency(om.ExplicitComponent):
    """Select FAST transmitter fan efficiency for supplemental power bookkeeping."""

    def initialize(self):
        self.options.declare("aircraft_class", default="Turbofan")

    def setup(self):
        self.add_input("fan_efficiency", val=0.9)
        self.add_output("transmitter_fan_efficiency", val=0.9)
        self.declare_partials(of="transmitter_fan_efficiency", wrt="fan_efficiency")

    def compute(self, inputs, outputs):
        aircraft_class = self.options["aircraft_class"].lower()

        if aircraft_class == "turbofan":
            outputs["transmitter_fan_efficiency"] = inputs["fan_efficiency"][0]
        elif aircraft_class in ("turboprop", "piston"):
            outputs["transmitter_fan_efficiency"] = 1.0
        else:
            raise ValueError(f"Invalid aircraft class: {self.options['aircraft_class']}")

    def compute_partials(self, inputs, partials):
        aircraft_class = self.options["aircraft_class"].lower()

        if aircraft_class == "turbofan":
            partials["transmitter_fan_efficiency", "fan_efficiency"] = 1.0
        elif aircraft_class in ("turboprop", "piston"):
            partials["transmitter_fan_efficiency", "fan_efficiency"] = 0.0
        else:
            raise ValueError(f"Invalid aircraft class: {self.options['aircraft_class']}")


class SimpleSourceTransmitterArchitecture(om.ExplicitComponent):
    """Build FAST's conventional or electric source-transmitter architecture."""

    def initialize(self):
        self.options.declare("num_engines", default=2)
        self.options.declare("architecture_type", default="C")

    def setup(self):
        num_engines = self.options["num_engines"]
        num_components = 2 * num_engines + 2
        num_transmitters = 2 * num_engines

        self.add_input("electric_motor_efficiency", val=0.95)
        self.add_input("thrust_sink_efficiency", val=0.85)
        self.add_output("architecture", val=np.zeros((num_components, num_components)))
        self.add_output("upstream_split", val=np.zeros((num_components, num_components)))
        self.add_output(
            "downstream_split",
            val=np.zeros((num_components, num_components)),
        )
        self.add_output(
            "upstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output(
            "downstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output("source_type", val=np.zeros(1))
        self.add_output("transmitter_type", val=np.zeros(num_transmitters))
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = simple_source_transmitter_architecture_values(
            self.options["num_engines"],
            self.options["architecture_type"],
            inputs["electric_motor_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in architecture_builder_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = simple_source_transmitter_architecture_values(
            self.options["num_engines"],
            self.options["architecture_type"],
            inputs["electric_motor_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in architecture_builder_output_names():
            for variable in simple_source_transmitter_architecture_input_names():
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


class ParallelHybridArchitecture(om.ExplicitComponent):
    """Build FAST's built-in parallel-hybrid propulsion architecture matrices."""

    def initialize(self):
        self.options.declare("num_engines", default=2)

    def setup(self):
        num_engines = self.options["num_engines"]
        num_components = 3 * num_engines + 3
        num_transmitters = 3 * num_engines

        self.add_input("power_split", val=0.5)
        self.add_input("electric_motor_efficiency", val=0.95)
        self.add_input("thrust_sink_efficiency", val=0.85)
        self.add_output("architecture", val=np.zeros((num_components, num_components)))
        self.add_output("upstream_split", val=np.zeros((num_components, num_components)))
        self.add_output(
            "downstream_split",
            val=np.zeros((num_components, num_components)),
        )
        self.add_output(
            "upstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output(
            "downstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output("source_type", val=np.zeros(2))
        self.add_output("transmitter_type", val=np.zeros(num_transmitters))
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = parallel_hybrid_architecture_values(
            self.options["num_engines"],
            inputs["power_split"][0],
            inputs["electric_motor_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in parallel_hybrid_architecture_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = parallel_hybrid_architecture_values(
            self.options["num_engines"],
            inputs["power_split"][0],
            inputs["electric_motor_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in parallel_hybrid_architecture_output_names():
            for variable in parallel_hybrid_architecture_input_names():
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


class SeriesHybridArchitecture(om.ExplicitComponent):
    """Build FAST's built-in series-hybrid propulsion architecture matrices."""

    def initialize(self):
        self.options.declare("num_engines", default=2)

    def setup(self):
        num_engines = self.options["num_engines"]
        num_components = 5 * num_engines + 3
        num_transmitters = 5 * num_engines

        self.add_input("power_split", val=0.5)
        self.add_input("electric_motor_efficiency", val=0.95)
        self.add_input("electric_generator_efficiency", val=0.95)
        self.add_input("thrust_sink_efficiency", val=0.85)
        self.add_output("architecture", val=np.zeros((num_components, num_components)))
        self.add_output("upstream_split", val=np.zeros((num_components, num_components)))
        self.add_output(
            "downstream_split",
            val=np.zeros((num_components, num_components)),
        )
        self.add_output(
            "upstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output(
            "downstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output("source_type", val=np.zeros(2))
        self.add_output("transmitter_type", val=np.zeros(num_transmitters))
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = series_hybrid_architecture_values(
            self.options["num_engines"],
            inputs["power_split"][0],
            inputs["electric_motor_efficiency"][0],
            inputs["electric_generator_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in architecture_builder_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = series_hybrid_architecture_values(
            self.options["num_engines"],
            inputs["power_split"][0],
            inputs["electric_motor_efficiency"][0],
            inputs["electric_generator_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in architecture_builder_output_names():
            for variable in series_hybrid_architecture_input_names():
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


class TurboelectricArchitecture(om.ExplicitComponent):
    """Build FAST's built-in turboelectric propulsion architecture matrices."""

    def initialize(self):
        self.options.declare("num_engines", default=2)

    def setup(self):
        num_engines = self.options["num_engines"]
        num_components = 4 * num_engines + 2
        num_transmitters = 4 * num_engines

        self.add_input("electric_motor_efficiency", val=0.95)
        self.add_input("electric_generator_efficiency", val=0.95)
        self.add_input("thrust_sink_efficiency", val=0.85)
        self.add_output("architecture", val=np.zeros((num_components, num_components)))
        self.add_output("upstream_split", val=np.zeros((num_components, num_components)))
        self.add_output(
            "downstream_split",
            val=np.zeros((num_components, num_components)),
        )
        self.add_output(
            "upstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output(
            "downstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output("source_type", val=np.zeros(1))
        self.add_output("transmitter_type", val=np.zeros(num_transmitters))
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = turboelectric_architecture_values(
            self.options["num_engines"],
            inputs["electric_motor_efficiency"][0],
            inputs["electric_generator_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in architecture_builder_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = turboelectric_architecture_values(
            self.options["num_engines"],
            inputs["electric_motor_efficiency"][0],
            inputs["electric_generator_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in architecture_builder_output_names():
            for variable in turboelectric_architecture_input_names():
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


class PartialTurboelectricArchitecture(om.ExplicitComponent):
    """Build FAST's built-in partially turboelectric architecture matrices."""

    def initialize(self):
        self.options.declare("num_engines", default=2)

    def setup(self):
        num_engines = self.options["num_engines"]
        num_components = 5 * num_engines + 2
        num_transmitters = 5 * num_engines

        self.add_input("power_split", val=0.5)
        self.add_input("electric_motor_efficiency", val=0.95)
        self.add_input("electric_generator_efficiency", val=0.95)
        self.add_input("thrust_sink_efficiency", val=0.85)
        self.add_output("architecture", val=np.zeros((num_components, num_components)))
        self.add_output("upstream_split", val=np.zeros((num_components, num_components)))
        self.add_output(
            "downstream_split",
            val=np.zeros((num_components, num_components)),
        )
        self.add_output(
            "upstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output(
            "downstream_efficiency",
            val=np.ones((num_components, num_components)),
        )
        self.add_output("source_type", val=np.zeros(1))
        self.add_output("transmitter_type", val=np.zeros(num_transmitters))
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = partial_turboelectric_architecture_values(
            self.options["num_engines"],
            inputs["power_split"][0],
            inputs["electric_motor_efficiency"][0],
            inputs["electric_generator_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in architecture_builder_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = partial_turboelectric_architecture_values(
            self.options["num_engines"],
            inputs["power_split"][0],
            inputs["electric_motor_efficiency"][0],
            inputs["electric_generator_efficiency"][0],
            inputs["thrust_sink_efficiency"][0],
        )

        for output in architecture_builder_output_names():
            for variable in partial_turboelectric_architecture_input_names():
                partials[output, variable] = values["d%s_d%s" % (output, variable)]


class PowerSupplementCheck(om.ExplicitComponent):
    """Compute FAST supplemental transmitter power for fixed architecture topology.

    Inputs:
        required_power: Required transmitter power matrix in W.
        split: Operational split matrix for the active propagation direction.
        efficiency: Efficiency matrix paired with ``split``.
        fan_efficiency: Fan or propeller efficiency for parallel assistance.

    Outputs:
        supplemental_power: Supplemental transmitter power matrix in W.

    Assumptions:
        Architecture and transmitter type vectors are discrete design choices
        fixed at setup. The component keeps FAST's matrix convention where gas
        turbines are type 1, electric transmitters are type 0, and thrust sinks
        are type 2.
    """

    def initialize(self):
        self.options.declare("num_points", default=1)
        self.options.declare("architecture")
        self.options.declare("transmitter_type")

    def setup(self):
        num_points = self.options["num_points"]
        transmitter_type = np.asarray(self.options["transmitter_type"]).reshape(-1)
        num_transmitters = len(transmitter_type)

        self.add_input(
            "required_power",
            val=np.ones((num_points, num_transmitters)),
            units="W",
        )
        self.add_input(
            "split",
            val=np.ones((num_transmitters, num_transmitters)),
        )
        self.add_input(
            "efficiency",
            val=np.ones((num_transmitters, num_transmitters)),
        )
        self.add_input("fan_efficiency", val=0.9)
        self.add_output(
            "supplemental_power",
            val=np.zeros((num_points, num_transmitters)),
            units="W",
        )
        self.declare_partials(of="supplemental_power", wrt="*")

    def compute(self, inputs, outputs):
        outputs["supplemental_power"] = power_supplement_values(
            inputs["required_power"],
            self.options["architecture"],
            inputs["split"],
            inputs["efficiency"],
            self.options["transmitter_type"],
            inputs["fan_efficiency"][0],
        )["supplemental_power"]

    def compute_partials(self, inputs, partials):
        values = power_supplement_values(
            inputs["required_power"],
            self.options["architecture"],
            inputs["split"],
            inputs["efficiency"],
            self.options["transmitter_type"],
            inputs["fan_efficiency"][0],
        )
        partials["supplemental_power", "required_power"] = values[
            "dsupplemental_drequired_power"
        ]
        partials["supplemental_power", "split"] = values["dsupplemental_dsplit"]
        partials["supplemental_power", "efficiency"] = values[
            "dsupplemental_defficiency"
        ]
        partials["supplemental_power", "fan_efficiency"] = values[
            "dsupplemental_dfan_efficiency"
        ]


class PowerFlow(om.ExplicitComponent):
    """Propagate power through a fixed FAST propulsion architecture.

    Inputs:
        initial_power: Initial component power vector in W.
        split: Operational split matrix for the selected propagation direction.
        efficiency: Efficiency matrix paired with ``split``.

    Outputs:
        propagated_power: Updated component power vector in W.

    Assumptions:
        Architecture, propagation direction, and convergence tolerance are fixed
        at setup. Analytical partials follow FAST's active-update iteration for
        the current input values, so they are exact away from tolerance-switch
        boundaries.
    """

    def initialize(self):
        self.options.declare("architecture")
        self.options.declare("direction", default=1)
        self.options.declare("tolerance", default=1.0e-6)

    def setup(self):
        architecture = np.asarray(self.options["architecture"], dtype=float)
        num_components = architecture.shape[0]
        matrix_shape = (num_components, num_components)

        self.add_input(
            "initial_power",
            val=np.zeros(num_components),
            units="W",
        )
        self.add_input("split", val=np.zeros(matrix_shape))
        self.add_input("efficiency", val=np.ones(matrix_shape))
        self.add_output(
            "propagated_power",
            val=np.zeros(num_components),
            units="W",
        )
        self.declare_partials(of="propagated_power", wrt="*")

    def compute(self, inputs, outputs):
        outputs["propagated_power"] = power_flow_values(
            inputs["initial_power"],
            self.options["architecture"],
            inputs["split"],
            inputs["efficiency"],
            self.options["direction"],
            self.options["tolerance"],
        )["propagated_power"]

    def compute_partials(self, inputs, partials):
        values = power_flow_values(
            inputs["initial_power"],
            self.options["architecture"],
            inputs["split"],
            inputs["efficiency"],
            self.options["direction"],
            self.options["tolerance"],
        )
        partials["propagated_power", "initial_power"] = values[
            "dpropagated_dinitial_power"
        ]
        partials["propagated_power", "split"] = values["dpropagated_dsplit"]
        partials["propagated_power", "efficiency"] = values[
            "dpropagated_defficiency"
        ]


class PowerAvailable(om.ExplicitComponent):
    """Compute FAST available propulsion power for fixed architecture data."""

    def initialize(self):
        self.options.declare("num_points", default=1)
        self.options.declare("architecture")
        self.options.declare("efficiency")
        self.options.declare("transmitter_type")
        self.options.declare("num_sources")
        self.options.declare("aircraft_class", default="Turbofan")

    def setup(self):
        num_points = self.options["num_points"]
        architecture = np.asarray(self.options["architecture"], dtype=float)
        transmitter_type = np.asarray(self.options["transmitter_type"]).reshape(-1)
        num_components = architecture.shape[0]
        num_transmitters = len(transmitter_type)

        self.add_input("true_airspeed", val=np.ones(num_points), units="m/s")
        self.add_input("density", val=np.ones(num_points), units="kg/m**3")
        self.add_input(
            "sea_level_static_thrust",
            val=np.ones(num_transmitters),
            units="N",
        )
        self.add_input(
            "sea_level_static_power",
            val=np.ones(num_transmitters),
            units="W",
        )
        self.add_input(
            "split",
            val=np.zeros((num_points, num_components, num_components)),
        )
        self.add_output(
            "available_power",
            val=np.zeros((num_points, num_components)),
            units="W",
        )
        self.add_output(
            "available_thrust",
            val=np.zeros((num_points, num_components)),
            units="N",
        )
        self.add_output("thrust_velocity", val=np.zeros(num_points), units="W")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = power_available_values(
            inputs["true_airspeed"],
            inputs["density"],
            inputs["sea_level_static_thrust"],
            inputs["sea_level_static_power"],
            inputs["split"],
            self.options["architecture"],
            self.options["efficiency"],
            self.options["transmitter_type"],
            self.options["num_sources"],
            self.options["aircraft_class"],
        )
        outputs["available_power"] = values["available_power"]
        outputs["available_thrust"] = values["available_thrust"]
        outputs["thrust_velocity"] = values["thrust_velocity"]

    def compute_partials(self, inputs, partials):
        values = power_available_values(
            inputs["true_airspeed"],
            inputs["density"],
            inputs["sea_level_static_thrust"],
            inputs["sea_level_static_power"],
            inputs["split"],
            self.options["architecture"],
            self.options["efficiency"],
            self.options["transmitter_type"],
            self.options["num_sources"],
            self.options["aircraft_class"],
        )
        partials["available_power", "true_airspeed"] = values["dpav_dtas"]
        partials["available_power", "density"] = values["dpav_drho"]
        partials["available_power", "sea_level_static_thrust"] = values[
            "dpav_dsls_thrust"
        ]
        partials["available_power", "sea_level_static_power"] = values[
            "dpav_dsls_power"
        ]
        partials["available_power", "split"] = values["dpav_dsplit"]
        partials["available_thrust", "true_airspeed"] = values["dtav_dtas"]
        partials["available_thrust", "density"] = values["dtav_drho"]
        partials["available_thrust", "sea_level_static_thrust"] = values[
            "dtav_dsls_thrust"
        ]
        partials["available_thrust", "sea_level_static_power"] = values[
            "dtav_dsls_power"
        ]
        partials["available_thrust", "split"] = values["dtav_dsplit"]
        partials["thrust_velocity", "true_airspeed"] = values["dtv_dtas"]
        partials["thrust_velocity", "density"] = values["dtv_drho"]
        partials["thrust_velocity", "sea_level_static_thrust"] = values[
            "dtv_dsls_thrust"
        ]
        partials["thrust_velocity", "sea_level_static_power"] = values[
            "dtv_dsls_power"
        ]
        partials["thrust_velocity", "split"] = values["dtv_dsplit"]


class EngineThrustRequirement(om.ExplicitComponent):
    """Select thrust handled by the sink connected to a fixed engine component.

    Inputs:
        thrust_output: Component thrust output matrix in N.

    Outputs:
        required_thrust: Thrust assigned to the engine component in N.

    Assumptions:
        Architecture, transmitter types, source count, and component index are
        fixed at setup. Missing or nonfinite connected sink thrust maps to zero,
        matching FAST sizing bookkeeping.
    """

    def initialize(self):
        self.options.declare("architecture")
        self.options.declare("transmitter_type")
        self.options.declare("num_sources")
        self.options.declare("component")
        self.options.declare("num_points", default=1)

    def setup(self):
        architecture = np.asarray(self.options["architecture"], dtype=float)
        num_points = self.options["num_points"]
        num_components = architecture.shape[1]

        self.add_input(
            "thrust_output",
            val=np.zeros((num_points, num_components)),
            units="N",
        )
        self.add_output(
            "required_thrust",
            val=np.zeros(num_points),
            units="N",
        )
        self.declare_partials(of="required_thrust", wrt="thrust_output")

    def compute(self, inputs, outputs):
        outputs["required_thrust"] = engine_thrust_requirement_values(
            self.options["architecture"],
            self.options["transmitter_type"],
            self.options["num_sources"],
            inputs["thrust_output"],
            self.options["component"],
        )["required_thrust"]

    def compute_partials(self, inputs, partials):
        values = engine_thrust_requirement_values(
            self.options["architecture"],
            self.options["transmitter_type"],
            self.options["num_sources"],
            inputs["thrust_output"],
            self.options["component"],
        )
        partials["required_thrust", "thrust_output"] = values[
            "drequired_dthrust_output"
        ]


class TurbopropEngineWeightForSizing(om.ExplicitComponent):
    """Compute FAST turboprop/piston engine weights from sizing power."""

    def initialize(self):
        self.options.declare("engines")
        self.options.declare("power_sls")
        self.options.declare("dry_weight")

    def setup(self):
        engines = np.asarray(self.options["engines"], dtype=bool).reshape(-1)
        num_components = len(engines) + 1
        num_engines = max(1, np.count_nonzero(engines))

        self.add_input(
            "downstream_power",
            val=np.zeros(num_components),
            units="W",
        )
        self.add_output("engine_weight", val=np.zeros(num_engines), units="kg")
        self.declare_partials(of="engine_weight", wrt="downstream_power")

    def compute(self, inputs, outputs):
        outputs["engine_weight"] = turboprop_engine_weight_for_sizing_values(
            inputs["downstream_power"],
            self.options["engines"],
            self.options["power_sls"],
            self.options["dry_weight"],
        )["engine_weight"]

    def compute_partials(self, inputs, partials):
        values = turboprop_engine_weight_for_sizing_values(
            inputs["downstream_power"],
            self.options["engines"],
            self.options["power_sls"],
            self.options["dry_weight"],
        )
        partials["engine_weight", "downstream_power"] = values[
            "dengine_weight_ddownstream_power"
        ]


class TurbofanEngineWeightForSizing(om.ExplicitComponent):
    """Compute FAST turbofan engine dry weights from sizing thrust GPR.

    Inputs:
        downstream_thrust: Downstream component thrust vector in N.

    Outputs:
        engine_weight: Per-engine dry weight estimate in kg.

    Assumptions:
        This component represents the GPR weight output from FAST
        ``engine_weights_for_sizing`` for turbofan aircraft. The separate
        ``SizedEngine`` mutation and nonlinear thermodynamic sizing remain
        orchestration responsibilities.
    """

    def initialize(self):
        self.options.declare("engines")
        self.options.declare("data_matrix")
        self.options.declare("hyperparams")
        self.options.declare("inverse_term")
        self.options.declare("prior", default=1.0)

    def setup(self):
        engines = np.asarray(self.options["engines"], dtype=bool).reshape(-1)
        num_components = len(engines) + 1
        num_engines = max(1, np.count_nonzero(engines))

        self.add_input(
            "downstream_thrust",
            val=np.zeros(num_components),
            units="N",
        )
        self.add_output("engine_weight", val=np.zeros(num_engines), units="kg")
        self.declare_partials(of="engine_weight", wrt="downstream_thrust")

    def compute(self, inputs, outputs):
        outputs["engine_weight"] = turbofan_engine_weight_for_sizing_values(
            inputs["downstream_thrust"],
            self.options["engines"],
            self.options["data_matrix"],
            self.options["hyperparams"],
            self.options["inverse_term"],
            self.options["prior"],
        )["engine_weight"]

    def compute_partials(self, inputs, partials):
        values = turbofan_engine_weight_for_sizing_values(
            inputs["downstream_thrust"],
            self.options["engines"],
            self.options["data_matrix"],
            self.options["hyperparams"],
            self.options["inverse_term"],
            self.options["prior"],
        )
        partials["engine_weight", "downstream_thrust"] = values[
            "dengine_weight_ddownstream_thrust"
        ]


class CableWeightForSizing(om.ExplicitComponent):
    """Compute FAST cable weight for fixed cable transmitter geometry.

    Inputs:
        downstream_power: Downstream component power vector in W.
        cable_power_to_weight: Cable weight coefficient from FAST specs.

    Outputs:
        cable_weight: Total cable weight contribution in kg.

    Assumptions:
        Cable transmitter mask, connection matrix, and length matrix are fixed
        architecture data. FAST ignores the final sink column in downstream
        sizing power; this component keeps the same convention.
    """

    def initialize(self):
        self.options.declare("cables")
        self.options.declare("cable_connections")
        self.options.declare("cable_lengths")

    def setup(self):
        cables = np.asarray(self.options["cables"], dtype=bool).reshape(-1)
        num_components = len(cables) + 1

        self.add_input(
            "downstream_power",
            val=np.zeros(num_components),
            units="W",
        )
        self.add_input("cable_power_to_weight", val=1.0)
        self.add_output("cable_weight", val=0.0, units="kg")
        self.declare_partials(of="cable_weight", wrt="*")

    def compute(self, inputs, outputs):
        outputs["cable_weight"] = cable_weight_for_sizing_values(
            inputs["downstream_power"],
            inputs["cable_power_to_weight"][0],
            self.options["cables"],
            self.options["cable_connections"],
            self.options["cable_lengths"],
        )["cable_weight"]

    def compute_partials(self, inputs, partials):
        values = cable_weight_for_sizing_values(
            inputs["downstream_power"],
            inputs["cable_power_to_weight"][0],
            self.options["cables"],
            self.options["cable_connections"],
            self.options["cable_lengths"],
        )
        partials["cable_weight", "downstream_power"] = values[
            "dcable_weight_ddownstream_power"
        ]
        partials["cable_weight", "cable_power_to_weight"] = values[
            "dcable_weight_dcable_power_to_weight"
        ]


class BatteryEnergyHistory(om.ExplicitComponent):
    """Accumulate FAST battery source energy for the smooth active branch.

    Inputs:
        source_power: Source power-output history in W.
        time_step: Segment time steps in s.
        initial_source_energy: Source energy used at the segment start in J.
        initial_source_energy_left: Source energy left at the segment start in J.

    Outputs:
        source_energy: Per-source used energy history in J.
        source_energy_left: Per-source remaining energy history in J.

    Assumptions:
        ``battery_sources`` is fixed source metadata. This represents the
        non-detailed battery branch of FAST ``update_battery_energy`` before
        depletion cutoff. Non-battery source columns are passed through as
        constant initial histories.
    """

    def initialize(self):
        self.options.declare("num_points", default=2)
        self.options.declare("battery_sources")

    def setup(self):
        num_points = self.options["num_points"]
        battery_sources = np.asarray(
            self.options["battery_sources"],
            dtype=bool,
        ).reshape(-1)
        num_sources = battery_sources.size
        self.add_input(
            "source_power",
            val=np.zeros((num_points, num_sources)),
            units="W",
        )
        self.add_input("time_step", val=np.ones(num_points - 1), units="s")
        self.add_input("initial_source_energy", val=np.zeros(num_sources), units="J")
        self.add_input(
            "initial_source_energy_left",
            val=np.zeros(num_sources),
            units="J",
        )
        self.add_output(
            "source_energy",
            val=np.zeros((num_points, num_sources)),
            units="J",
        )
        self.add_output(
            "source_energy_left",
            val=np.zeros((num_points, num_sources)),
            units="J",
        )
        self.declare_partials(of=["source_energy", "source_energy_left"], wrt="*")

    def compute(self, inputs, outputs):
        values = battery_energy_history_values(
            inputs["source_power"],
            inputs["time_step"],
            inputs["initial_source_energy"],
            inputs["initial_source_energy_left"],
            self.options["battery_sources"],
        )
        outputs["source_energy"] = values["source_energy"]
        outputs["source_energy_left"] = values["source_energy_left"]

    def compute_partials(self, inputs, partials):
        values = battery_energy_history_values(
            inputs["source_power"],
            inputs["time_step"],
            inputs["initial_source_energy"],
            inputs["initial_source_energy_left"],
            self.options["battery_sources"],
        )

        for output_name in ("source_energy", "source_energy_left"):
            for input_name in (
                "source_power",
                "time_step",
                "initial_source_energy",
                "initial_source_energy_left",
            ):
                partials[output_name, input_name] = values[
                    "d%s_d%s" % (output_name, input_name)
                ]


class FuelUseHistory(om.ExplicitComponent):
    """Accumulate FAST fuel-flow history into fuel burn, mass, and source energy."""

    def initialize(self):
        self.options.declare("num_steps", default=1)
        self.options.declare("num_engines", default=1)

    def setup(self):
        num_steps = self.options["num_steps"]
        num_engines = self.options["num_engines"]
        history_size = num_steps + 1
        engine_shape = (num_steps, num_engines)

        self.add_input("fuel_mass_flow", val=np.zeros(engine_shape), units="kg/s")
        self.add_input("specific_fuel_consumption", val=np.zeros(engine_shape))
        self.add_input("time_step", val=np.ones(num_steps), units="s")
        self.add_input("fuel_specific_energy", val=43.0e6, units="J/kg")
        self.add_input("initial_fuel_burn", val=0.0, units="kg")
        self.add_input("initial_mass", val=1.0, units="kg")
        self.add_input("initial_fuel_energy", val=0.0, units="J")
        self.add_input("initial_fuel_energy_left", val=0.0, units="J")
        self.add_input("mass_flow_correction", val=1.0)
        self.add_output(
            "corrected_fuel_mass_flow",
            val=np.zeros(engine_shape),
            units="kg/s",
        )
        self.add_output(
            "corrected_specific_fuel_consumption",
            val=np.zeros(engine_shape),
        )
        self.add_output("source_mass_flow", val=np.zeros(num_steps), units="kg/s")
        self.add_output("fuel_burn", val=np.zeros(history_size), units="kg")
        self.add_output("mass", val=np.ones(history_size), units="kg")
        self.add_output("fuel_energy", val=np.zeros(history_size), units="J")
        self.add_output("fuel_energy_left", val=np.zeros(history_size), units="J")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = fuel_use_history_values(
            inputs["fuel_mass_flow"],
            inputs["specific_fuel_consumption"],
            inputs["time_step"],
            inputs["fuel_specific_energy"][0],
            inputs["initial_fuel_burn"][0],
            inputs["initial_mass"][0],
            inputs["initial_fuel_energy"][0],
            inputs["initial_fuel_energy_left"][0],
            inputs["mass_flow_correction"][0],
        )

        for output in fuel_use_history_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = fuel_use_history_values(
            inputs["fuel_mass_flow"],
            inputs["specific_fuel_consumption"],
            inputs["time_step"],
            inputs["fuel_specific_energy"][0],
            inputs["initial_fuel_burn"][0],
            inputs["initial_mass"][0],
            inputs["initial_fuel_energy"][0],
            inputs["initial_fuel_energy_left"][0],
            inputs["mass_flow_correction"][0],
        )

        for output in fuel_use_history_output_names():
            for variable in fuel_use_history_input_names():
                partials[output, variable] = values[
                    "d%s_d%s" % (output, variable)
                ]


def engine_lapse_value(sea_level_static, aircraft_class, density):
    """Return scalar FAST engine-lapse value."""

    aircraft_class = aircraft_class.lower()

    if aircraft_class == "turbofan":
        return sea_level_static * density / RHO_SL_STD

    if aircraft_class in ("turboprop", "piston"):
        return sea_level_static

    raise ValueError(f"Invalid aircraft class: {aircraft_class}")


def engine_lapse_derivatives(sea_level_static, aircraft_class, density):
    """Return scalar FAST engine-lapse derivatives."""

    aircraft_class = aircraft_class.lower()

    if aircraft_class == "turbofan":
        return {
            "doutput_dsea_level_static": density / RHO_SL_STD,
            "doutput_ddensity": sea_level_static / RHO_SL_STD,
        }

    if aircraft_class in ("turboprop", "piston"):
        return {
            "doutput_dsea_level_static": 1.0,
            "doutput_ddensity": 0.0,
        }

    raise ValueError(f"Invalid aircraft class: {aircraft_class}")


def safe_component_weight_value(power, power_to_weight):
    """Return FAST safe component weight scalar value."""

    if abs(power) < 1.0e-12:
        return 0.0

    return power / power_to_weight


def simple_source_transmitter_architecture_input_names():
    """Return inputs for SimpleSourceTransmitterArchitecture derivatives."""

    return (
        "electric_motor_efficiency",
        "thrust_sink_efficiency",
    )


def parallel_hybrid_architecture_input_names():
    """Return inputs for ParallelHybridArchitecture derivatives."""

    return (
        "power_split",
        "electric_motor_efficiency",
        "thrust_sink_efficiency",
    )


def architecture_builder_output_names():
    """Return common outputs for built-in architecture builder components."""

    return (
        "architecture",
        "upstream_split",
        "downstream_split",
        "upstream_efficiency",
        "downstream_efficiency",
        "source_type",
        "transmitter_type",
    )


def parallel_hybrid_architecture_output_names():
    """Return outputs for ParallelHybridArchitecture."""

    return architecture_builder_output_names()


def series_hybrid_architecture_input_names():
    """Return inputs for SeriesHybridArchitecture derivatives."""

    return (
        "power_split",
        "electric_motor_efficiency",
        "electric_generator_efficiency",
        "thrust_sink_efficiency",
    )


def turboelectric_architecture_input_names():
    """Return inputs for TurboelectricArchitecture derivatives."""

    return (
        "electric_motor_efficiency",
        "electric_generator_efficiency",
        "thrust_sink_efficiency",
    )


def partial_turboelectric_architecture_input_names():
    """Return inputs for PartialTurboelectricArchitecture derivatives."""

    return (
        "power_split",
        "electric_motor_efficiency",
        "electric_generator_efficiency",
        "thrust_sink_efficiency",
    )


def simple_source_transmitter_architecture_values(
    num_engines,
    architecture_type,
    electric_motor_efficiency,
    thrust_sink_efficiency,
):
    """Return FAST C/E architecture matrices and dense derivatives."""

    architecture_type = architecture_type.upper()

    if architecture_type not in ("C", "E"):
        raise ValueError(f"Invalid simple architecture type: {architecture_type}")

    num_components = 2 * num_engines + 2
    architecture = np.zeros((num_components, num_components))
    upstream_split = np.zeros_like(architecture)
    downstream_split = np.zeros_like(architecture)
    upstream_efficiency = np.ones_like(architecture)
    downstream_efficiency = np.ones_like(architecture)
    source = 0
    transmitter_index = np.arange(1, 1 + num_engines)
    sink_index = np.arange(1 + num_engines, 1 + 2 * num_engines)
    final_sink = num_components - 1
    architecture[source, transmitter_index] = 1.0
    upstream_split[source, transmitter_index] = 1.0 / num_engines
    downstream_split[transmitter_index, source] = 1.0

    dupstream_efficiency_dem = np.zeros_like(architecture)
    ddownstream_efficiency_dem = np.zeros_like(architecture)
    dupstream_efficiency_dts = np.zeros_like(architecture)
    ddownstream_efficiency_dts = np.zeros_like(architecture)

    for transmitter, sink in zip(transmitter_index, sink_index):
        architecture[transmitter, sink] = 1.0
        architecture[sink, final_sink] = 1.0
        upstream_split[transmitter, sink] = 1.0
        upstream_split[sink, final_sink] = 1.0
        downstream_split[sink, transmitter] = 1.0
        downstream_split[final_sink, sink] = 1.0 / num_engines
        upstream_efficiency[transmitter, sink] = thrust_sink_efficiency
        downstream_efficiency[sink, transmitter] = thrust_sink_efficiency
        dupstream_efficiency_dts[transmitter, sink] = 1.0
        ddownstream_efficiency_dts[sink, transmitter] = 1.0

    if architecture_type == "C":
        source_type = np.asarray([1.0])
        transmitter_type = np.asarray([1.0] * num_engines + [2.0] * num_engines)
    else:
        upstream_efficiency[source, transmitter_index] = electric_motor_efficiency
        downstream_efficiency[transmitter_index, source] = electric_motor_efficiency
        dupstream_efficiency_dem[source, transmitter_index] = 1.0
        ddownstream_efficiency_dem[transmitter_index, source] = 1.0
        source_type = np.asarray([0.0])
        transmitter_type = np.asarray([0.0] * num_engines + [2.0] * num_engines)

    output_values = {
        "architecture": architecture,
        "upstream_split": upstream_split,
        "downstream_split": downstream_split,
        "upstream_efficiency": upstream_efficiency,
        "downstream_efficiency": downstream_efficiency,
        "source_type": source_type,
        "transmitter_type": transmitter_type,
    }
    derivative_maps = {
        "electric_motor_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dem,
            "downstream_efficiency": ddownstream_efficiency_dem,
        },
        "thrust_sink_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dts,
            "downstream_efficiency": ddownstream_efficiency_dts,
        },
    }
    result = dict(output_values)

    for output_name, value in output_values.items():
        output_size = np.asarray(value).size
        for input_name in simple_source_transmitter_architecture_input_names():
            derivative = derivative_maps.get(input_name, {}).get(output_name)

            if derivative is None:
                derivative = np.zeros(output_size)

            result["d%s_d%s" % (output_name, input_name)] = np.asarray(
                derivative,
                dtype=float,
            ).reshape(output_size, 1)

    return result


def parallel_hybrid_architecture_values(
    num_engines,
    power_split,
    electric_motor_efficiency,
    thrust_sink_efficiency,
):
    """Return FAST parallel-hybrid matrices and dense analytical derivatives."""

    num_components = 3 * num_engines + 3
    architecture = np.zeros((num_components, num_components))
    upstream_split = np.zeros_like(architecture)
    downstream_split = np.zeros_like(architecture)
    upstream_efficiency = np.ones_like(architecture)
    downstream_efficiency = np.ones_like(architecture)
    fuel = 0
    battery = 1
    engine_index = np.arange(2, 2 + num_engines)
    motor_index = np.arange(2 + num_engines, 2 + 2 * num_engines)
    sink_index = np.arange(2 + 2 * num_engines, 2 + 3 * num_engines)
    final_sink = num_components - 1
    architecture[fuel, engine_index] = 1.0
    architecture[battery, motor_index] = 1.0
    upstream_split[fuel, engine_index] = 1.0 / num_engines
    upstream_split[battery, motor_index] = 1.0 / num_engines
    downstream_split[engine_index, fuel] = 1.0
    downstream_split[motor_index, battery] = 1.0

    dupstream_split_dsplit = np.zeros_like(architecture)
    ddownstream_split_dsplit = np.zeros_like(architecture)
    dupstream_efficiency_dem = np.zeros_like(architecture)
    ddownstream_efficiency_dem = np.zeros_like(architecture)
    dupstream_efficiency_dts = np.zeros_like(architecture)
    ddownstream_efficiency_dts = np.zeros_like(architecture)

    for engine, motor, sink in zip(engine_index, motor_index, sink_index):
        architecture[engine, sink] = 1.0
        architecture[motor, sink] = 1.0
        architecture[sink, final_sink] = 1.0
        upstream_split[engine, sink] = 1.0
        upstream_split[motor, sink] = power_split
        upstream_split[sink, final_sink] = 1.0
        downstream_split[sink, engine] = 1.0 - power_split
        downstream_split[sink, motor] = power_split
        upstream_efficiency[battery, motor] = electric_motor_efficiency
        downstream_efficiency[motor, battery] = electric_motor_efficiency
        upstream_efficiency[engine, sink] = thrust_sink_efficiency
        upstream_efficiency[motor, sink] = thrust_sink_efficiency
        downstream_efficiency[sink, engine] = thrust_sink_efficiency
        downstream_efficiency[sink, motor] = thrust_sink_efficiency
        dupstream_split_dsplit[motor, sink] = 1.0
        ddownstream_split_dsplit[sink, engine] = -1.0
        ddownstream_split_dsplit[sink, motor] = 1.0
        dupstream_efficiency_dem[battery, motor] = 1.0
        ddownstream_efficiency_dem[motor, battery] = 1.0
        dupstream_efficiency_dts[engine, sink] = 1.0
        dupstream_efficiency_dts[motor, sink] = 1.0
        ddownstream_efficiency_dts[sink, engine] = 1.0
        ddownstream_efficiency_dts[sink, motor] = 1.0

    downstream_split[final_sink, sink_index] = 1.0 / num_engines
    source_type = np.asarray([1.0, 0.0])
    transmitter_type = np.asarray(
        [1.0] * num_engines + [0.0] * num_engines + [2.0] * num_engines
    )
    output_values = {
        "architecture": architecture,
        "upstream_split": upstream_split,
        "downstream_split": downstream_split,
        "upstream_efficiency": upstream_efficiency,
        "downstream_efficiency": downstream_efficiency,
        "source_type": source_type,
        "transmitter_type": transmitter_type,
    }
    derivative_maps = {
        "power_split": {
            "upstream_split": dupstream_split_dsplit,
            "downstream_split": ddownstream_split_dsplit,
        },
        "electric_motor_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dem,
            "downstream_efficiency": ddownstream_efficiency_dem,
        },
        "thrust_sink_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dts,
            "downstream_efficiency": ddownstream_efficiency_dts,
        },
    }
    result = dict(output_values)

    for output_name, value in output_values.items():
        output_size = np.asarray(value).size
        for input_name in parallel_hybrid_architecture_input_names():
            derivative = derivative_maps.get(input_name, {}).get(output_name)

            if derivative is None:
                derivative = np.zeros(output_size)

            result["d%s_d%s" % (output_name, input_name)] = np.asarray(
                derivative,
                dtype=float,
            ).reshape(output_size, 1)

    return result


def series_hybrid_architecture_values(
    num_engines,
    power_split,
    electric_motor_efficiency,
    electric_generator_efficiency,
    thrust_sink_efficiency,
):
    """Return FAST series-hybrid matrices and dense analytical derivatives."""

    num_components = 5 * num_engines + 3
    architecture = np.zeros((num_components, num_components))
    upstream_split = np.zeros_like(architecture)
    downstream_split = np.zeros_like(architecture)
    upstream_efficiency = np.ones_like(architecture)
    downstream_efficiency = np.ones_like(architecture)
    fuel = 0
    battery = 1
    engine_index = np.arange(2, 2 + num_engines)
    generator_index = np.arange(2 + num_engines, 2 + 2 * num_engines)
    cable_index = np.arange(2 + 2 * num_engines, 2 + 3 * num_engines)
    motor_index = np.arange(2 + 3 * num_engines, 2 + 4 * num_engines)
    sink_index = np.arange(2 + 4 * num_engines, 2 + 5 * num_engines)
    final_sink = num_components - 1
    architecture[fuel, engine_index] = 1.0
    architecture[battery, cable_index] = 1.0
    upstream_split[fuel, engine_index] = 1.0 / num_engines
    upstream_split[battery, cable_index] = 1.0 / num_engines
    downstream_split[engine_index, fuel] = 1.0
    downstream_split[cable_index, battery] = 1.0

    dupstream_split_dsplit = np.zeros_like(architecture)
    ddownstream_split_dsplit = np.zeros_like(architecture)
    dupstream_efficiency_dem = np.zeros_like(architecture)
    ddownstream_efficiency_dem = np.zeros_like(architecture)
    dupstream_efficiency_deg = np.zeros_like(architecture)
    ddownstream_efficiency_deg = np.zeros_like(architecture)
    dupstream_efficiency_dts = np.zeros_like(architecture)
    ddownstream_efficiency_dts = np.zeros_like(architecture)

    for engine, generator, cable, motor, sink in zip(
        engine_index,
        generator_index,
        cable_index,
        motor_index,
        sink_index,
    ):
        architecture[engine, generator] = 1.0
        architecture[generator, motor] = 1.0
        architecture[cable, motor] = 1.0
        architecture[motor, sink] = 1.0
        architecture[sink, final_sink] = 1.0
        upstream_split[engine, generator] = 1.0
        upstream_split[generator, motor] = 1.0
        upstream_split[cable, motor] = power_split
        upstream_split[motor, sink] = 1.0
        upstream_split[sink, final_sink] = 1.0
        downstream_split[generator, engine] = 1.0
        downstream_split[motor, generator] = 1.0 - power_split
        downstream_split[motor, cable] = power_split
        downstream_split[sink, motor] = 1.0
        upstream_efficiency[engine, generator] = electric_generator_efficiency
        upstream_efficiency[generator, motor] = electric_motor_efficiency
        upstream_efficiency[cable, motor] = electric_motor_efficiency
        upstream_efficiency[motor, sink] = thrust_sink_efficiency
        downstream_efficiency[generator, engine] = electric_generator_efficiency
        downstream_efficiency[motor, generator] = electric_motor_efficiency
        downstream_efficiency[motor, cable] = electric_motor_efficiency
        downstream_efficiency[sink, motor] = thrust_sink_efficiency
        dupstream_split_dsplit[cable, motor] = 1.0
        ddownstream_split_dsplit[motor, generator] = -1.0
        ddownstream_split_dsplit[motor, cable] = 1.0
        dupstream_efficiency_deg[engine, generator] = 1.0
        ddownstream_efficiency_deg[generator, engine] = 1.0
        dupstream_efficiency_dem[generator, motor] = 1.0
        dupstream_efficiency_dem[cable, motor] = 1.0
        ddownstream_efficiency_dem[motor, generator] = 1.0
        ddownstream_efficiency_dem[motor, cable] = 1.0
        dupstream_efficiency_dts[motor, sink] = 1.0
        ddownstream_efficiency_dts[sink, motor] = 1.0

    downstream_split[final_sink, sink_index] = 1.0 / num_engines
    source_type = np.asarray([1.0, 0.0])
    transmitter_type = np.asarray(
        [1.0] * num_engines
        + [3.0] * num_engines
        + [4.0] * num_engines
        + [0.0] * num_engines
        + [2.0] * num_engines
    )
    output_values = {
        "architecture": architecture,
        "upstream_split": upstream_split,
        "downstream_split": downstream_split,
        "upstream_efficiency": upstream_efficiency,
        "downstream_efficiency": downstream_efficiency,
        "source_type": source_type,
        "transmitter_type": transmitter_type,
    }
    derivative_maps = {
        "power_split": {
            "upstream_split": dupstream_split_dsplit,
            "downstream_split": ddownstream_split_dsplit,
        },
        "electric_motor_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dem,
            "downstream_efficiency": ddownstream_efficiency_dem,
        },
        "electric_generator_efficiency": {
            "upstream_efficiency": dupstream_efficiency_deg,
            "downstream_efficiency": ddownstream_efficiency_deg,
        },
        "thrust_sink_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dts,
            "downstream_efficiency": ddownstream_efficiency_dts,
        },
    }
    result = dict(output_values)

    for output_name, value in output_values.items():
        output_size = np.asarray(value).size
        for input_name in series_hybrid_architecture_input_names():
            derivative = derivative_maps.get(input_name, {}).get(output_name)

            if derivative is None:
                derivative = np.zeros(output_size)

            result["d%s_d%s" % (output_name, input_name)] = np.asarray(
                derivative,
                dtype=float,
            ).reshape(output_size, 1)

    return result


def turboelectric_architecture_values(
    num_engines,
    electric_motor_efficiency,
    electric_generator_efficiency,
    thrust_sink_efficiency,
):
    """Return FAST turboelectric matrices and dense analytical derivatives."""

    num_components = 4 * num_engines + 2
    architecture = np.zeros((num_components, num_components))
    upstream_split = np.zeros_like(architecture)
    downstream_split = np.zeros_like(architecture)
    upstream_efficiency = np.ones_like(architecture)
    downstream_efficiency = np.ones_like(architecture)
    fuel = 0
    engine_index = np.arange(1, 1 + num_engines)
    generator_index = np.arange(1 + num_engines, 1 + 2 * num_engines)
    motor_index = np.arange(1 + 2 * num_engines, 1 + 3 * num_engines)
    sink_index = np.arange(1 + 3 * num_engines, 1 + 4 * num_engines)
    final_sink = num_components - 1
    architecture[fuel, engine_index] = 1.0
    upstream_split[fuel, engine_index] = 1.0 / num_engines
    downstream_split[engine_index, fuel] = 1.0

    dupstream_efficiency_dem = np.zeros_like(architecture)
    ddownstream_efficiency_dem = np.zeros_like(architecture)
    dupstream_efficiency_deg = np.zeros_like(architecture)
    ddownstream_efficiency_deg = np.zeros_like(architecture)
    dupstream_efficiency_dts = np.zeros_like(architecture)
    ddownstream_efficiency_dts = np.zeros_like(architecture)

    for engine, generator, motor, sink in zip(
        engine_index,
        generator_index,
        motor_index,
        sink_index,
    ):
        architecture[engine, generator] = 1.0
        architecture[generator, motor] = 1.0
        architecture[motor, sink] = 1.0
        architecture[sink, final_sink] = 1.0
        upstream_split[engine, generator] = 1.0
        upstream_split[generator, motor] = 1.0
        upstream_split[motor, sink] = 1.0
        upstream_split[sink, final_sink] = 1.0
        downstream_split[generator, engine] = 1.0
        downstream_split[motor, generator] = 1.0
        downstream_split[sink, motor] = 1.0
        upstream_efficiency[engine, generator] = electric_generator_efficiency
        upstream_efficiency[generator, motor] = electric_motor_efficiency
        upstream_efficiency[motor, sink] = thrust_sink_efficiency
        downstream_efficiency[generator, engine] = electric_generator_efficiency
        downstream_efficiency[motor, generator] = electric_motor_efficiency
        downstream_efficiency[sink, motor] = thrust_sink_efficiency
        dupstream_efficiency_deg[engine, generator] = 1.0
        ddownstream_efficiency_deg[generator, engine] = 1.0
        dupstream_efficiency_dem[generator, motor] = 1.0
        ddownstream_efficiency_dem[motor, generator] = 1.0
        dupstream_efficiency_dts[motor, sink] = 1.0
        ddownstream_efficiency_dts[sink, motor] = 1.0

    downstream_split[final_sink, sink_index] = 1.0 / num_engines
    source_type = np.asarray([1.0])
    transmitter_type = np.asarray(
        [1.0] * num_engines
        + [3.0] * num_engines
        + [0.0] * num_engines
        + [2.0] * num_engines
    )
    output_values = {
        "architecture": architecture,
        "upstream_split": upstream_split,
        "downstream_split": downstream_split,
        "upstream_efficiency": upstream_efficiency,
        "downstream_efficiency": downstream_efficiency,
        "source_type": source_type,
        "transmitter_type": transmitter_type,
    }
    derivative_maps = {
        "electric_motor_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dem,
            "downstream_efficiency": ddownstream_efficiency_dem,
        },
        "electric_generator_efficiency": {
            "upstream_efficiency": dupstream_efficiency_deg,
            "downstream_efficiency": ddownstream_efficiency_deg,
        },
        "thrust_sink_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dts,
            "downstream_efficiency": ddownstream_efficiency_dts,
        },
    }
    result = dict(output_values)

    for output_name, value in output_values.items():
        output_size = np.asarray(value).size
        for input_name in turboelectric_architecture_input_names():
            derivative = derivative_maps.get(input_name, {}).get(output_name)

            if derivative is None:
                derivative = np.zeros(output_size)

            result["d%s_d%s" % (output_name, input_name)] = np.asarray(
                derivative,
                dtype=float,
            ).reshape(output_size, 1)

    return result


def partial_turboelectric_architecture_values(
    num_engines,
    power_split,
    electric_motor_efficiency,
    electric_generator_efficiency,
    thrust_sink_efficiency,
):
    """Return FAST partial-turboelectric matrices and dense derivatives."""

    num_components = 5 * num_engines + 2
    architecture = np.zeros((num_components, num_components))
    upstream_split = np.zeros_like(architecture)
    downstream_split = np.zeros_like(architecture)
    upstream_efficiency = np.ones_like(architecture)
    downstream_efficiency = np.ones_like(architecture)
    fuel = 0
    engine_index = np.arange(1, 1 + num_engines)
    generator_index = np.arange(1 + num_engines, 1 + 2 * num_engines)
    motor_index = np.arange(1 + 2 * num_engines, 1 + 3 * num_engines)
    inboard_index = np.arange(1 + 3 * num_engines, 1 + 4 * num_engines)
    outboard_index = np.arange(1 + 4 * num_engines, 1 + 5 * num_engines)
    final_sink = num_components - 1
    architecture[fuel, engine_index] = 1.0
    upstream_split[fuel, engine_index] = 1.0 / num_engines
    downstream_split[engine_index, fuel] = 1.0

    dupstream_split_dsplit = np.zeros_like(architecture)
    ddownstream_split_dsplit = np.zeros_like(architecture)
    dupstream_efficiency_dem = np.zeros_like(architecture)
    ddownstream_efficiency_dem = np.zeros_like(architecture)
    dupstream_efficiency_deg = np.zeros_like(architecture)
    ddownstream_efficiency_deg = np.zeros_like(architecture)
    dupstream_efficiency_dts = np.zeros_like(architecture)
    ddownstream_efficiency_dts = np.zeros_like(architecture)

    for engine, generator, motor, inboard, outboard in zip(
        engine_index,
        generator_index,
        motor_index,
        inboard_index,
        outboard_index,
    ):
        architecture[engine, generator] = 1.0
        architecture[engine, outboard] = 1.0
        architecture[generator, motor] = 1.0
        architecture[motor, inboard] = 1.0
        architecture[inboard, final_sink] = 1.0
        architecture[outboard, final_sink] = 1.0
        upstream_split[engine, generator] = 1.0 - power_split
        upstream_split[engine, outboard] = power_split
        upstream_split[generator, motor] = 1.0
        upstream_split[motor, inboard] = 1.0
        upstream_split[inboard, final_sink] = 1.0
        upstream_split[outboard, final_sink] = 1.0
        downstream_split[generator, engine] = 1.0
        downstream_split[motor, generator] = 1.0
        downstream_split[inboard, motor] = 1.0
        downstream_split[outboard, engine] = 1.0
        upstream_efficiency[engine, generator] = electric_generator_efficiency
        upstream_efficiency[engine, outboard] = thrust_sink_efficiency
        upstream_efficiency[generator, motor] = electric_motor_efficiency
        upstream_efficiency[motor, inboard] = thrust_sink_efficiency
        downstream_efficiency[generator, engine] = electric_generator_efficiency
        downstream_efficiency[motor, generator] = electric_motor_efficiency
        downstream_efficiency[inboard, motor] = thrust_sink_efficiency
        downstream_efficiency[outboard, engine] = thrust_sink_efficiency
        dupstream_split_dsplit[engine, generator] = -1.0
        dupstream_split_dsplit[engine, outboard] = 1.0
        dupstream_efficiency_deg[engine, generator] = 1.0
        ddownstream_efficiency_deg[generator, engine] = 1.0
        dupstream_efficiency_dem[generator, motor] = 1.0
        ddownstream_efficiency_dem[motor, generator] = 1.0
        dupstream_efficiency_dts[engine, outboard] = 1.0
        dupstream_efficiency_dts[motor, inboard] = 1.0
        ddownstream_efficiency_dts[inboard, motor] = 1.0
        ddownstream_efficiency_dts[outboard, engine] = 1.0

    downstream_split[final_sink, inboard_index] = (1.0 - power_split) / num_engines
    downstream_split[final_sink, outboard_index] = power_split / num_engines
    ddownstream_split_dsplit[final_sink, inboard_index] = -1.0 / num_engines
    ddownstream_split_dsplit[final_sink, outboard_index] = 1.0 / num_engines
    source_type = np.asarray([1.0])
    transmitter_type = np.asarray(
        [1.0] * num_engines
        + [3.0] * num_engines
        + [0.0] * num_engines
        + [2.0] * (2 * num_engines)
    )
    output_values = {
        "architecture": architecture,
        "upstream_split": upstream_split,
        "downstream_split": downstream_split,
        "upstream_efficiency": upstream_efficiency,
        "downstream_efficiency": downstream_efficiency,
        "source_type": source_type,
        "transmitter_type": transmitter_type,
    }
    derivative_maps = {
        "power_split": {
            "upstream_split": dupstream_split_dsplit,
            "downstream_split": ddownstream_split_dsplit,
        },
        "electric_motor_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dem,
            "downstream_efficiency": ddownstream_efficiency_dem,
        },
        "electric_generator_efficiency": {
            "upstream_efficiency": dupstream_efficiency_deg,
            "downstream_efficiency": ddownstream_efficiency_deg,
        },
        "thrust_sink_efficiency": {
            "upstream_efficiency": dupstream_efficiency_dts,
            "downstream_efficiency": ddownstream_efficiency_dts,
        },
    }
    result = dict(output_values)

    for output_name, value in output_values.items():
        output_size = np.asarray(value).size
        for input_name in partial_turboelectric_architecture_input_names():
            derivative = derivative_maps.get(input_name, {}).get(output_name)

            if derivative is None:
                derivative = np.zeros(output_size)

            result["d%s_d%s" % (output_name, input_name)] = np.asarray(
                derivative,
                dtype=float,
            ).reshape(output_size, 1)

    return result


def fuel_use_history_input_names():
    """Return FuelUseHistory input names for derivative packing."""

    return (
        "fuel_mass_flow",
        "specific_fuel_consumption",
        "time_step",
        "fuel_specific_energy",
        "initial_fuel_burn",
        "initial_mass",
        "initial_fuel_energy",
        "initial_fuel_energy_left",
        "mass_flow_correction",
    )


def fuel_use_history_output_names():
    """Return FuelUseHistory output names."""

    return (
        "corrected_fuel_mass_flow",
        "corrected_specific_fuel_consumption",
        "source_mass_flow",
        "fuel_burn",
        "mass",
        "fuel_energy",
        "fuel_energy_left",
    )


def fuel_use_history_values(
    fuel_mass_flow,
    specific_fuel_consumption,
    time_step,
    fuel_specific_energy,
    initial_fuel_burn,
    initial_mass,
    initial_fuel_energy,
    initial_fuel_energy_left,
    mass_flow_correction,
):
    """Return FAST fuel-use accumulation values and analytical derivatives."""

    fuel_mass_flow = np.asarray(fuel_mass_flow, dtype=float)
    sfc = np.asarray(specific_fuel_consumption, dtype=float)
    dt = np.asarray(time_step, dtype=float).reshape(-1)
    num_steps, num_engines = fuel_mass_flow.shape
    engine_size = fuel_mass_flow.size
    history_size = num_steps + 1

    corrected_flow = fuel_mass_flow * mass_flow_correction
    corrected_sfc = sfc * mass_flow_correction
    source_flow = np.sum(corrected_flow, axis=1)
    fuel_used = np.cumsum(source_flow * dt)
    energy_used = np.cumsum(source_flow * fuel_specific_energy * dt)
    fuel_burn = np.zeros(history_size)
    mass = np.zeros(history_size)
    fuel_energy = np.zeros(history_size)
    fuel_energy_left = np.zeros(history_size)
    fuel_burn[0] = initial_fuel_burn
    mass[0] = initial_mass
    fuel_energy[0] = initial_fuel_energy
    fuel_energy_left[0] = initial_fuel_energy_left
    fuel_burn[1:] = initial_fuel_burn + fuel_used
    mass[1:] = initial_mass - fuel_used
    fuel_energy[1:] = initial_fuel_energy + energy_used
    fuel_energy_left[1:] = initial_fuel_energy_left - energy_used
    result = {
        "corrected_fuel_mass_flow": corrected_flow,
        "corrected_specific_fuel_consumption": corrected_sfc,
        "source_mass_flow": source_flow,
        "fuel_burn": fuel_burn,
        "mass": mass,
        "fuel_energy": fuel_energy,
        "fuel_energy_left": fuel_energy_left,
    }
    zeros_engine = np.zeros((engine_size, engine_size))
    identity_engine = np.eye(engine_size)
    dcorrected_flow_dflow = identity_engine * mass_flow_correction
    dcorrected_flow_dcorrection = fuel_mass_flow.reshape(-1, 1)
    dcorrected_sfc_dsfc = identity_engine * mass_flow_correction
    dcorrected_sfc_dcorrection = sfc.reshape(-1, 1)
    dsource_dflow = np.zeros((num_steps, engine_size))
    dsource_dcorrection = np.sum(fuel_mass_flow, axis=1).reshape(-1, 1)

    for step in range(num_steps):
        start = step * num_engines
        stop = start + num_engines
        dsource_dflow[step, start:stop] = mass_flow_correction

    dfuel_burn_dflow = np.zeros((history_size, engine_size))
    dfuel_burn_ddt = np.zeros((history_size, num_steps))
    dfuel_burn_dcorrection = np.zeros((history_size, 1))
    dfuel_energy_dflow = np.zeros((history_size, engine_size))
    dfuel_energy_ddt = np.zeros((history_size, num_steps))
    dfuel_energy_defuel = np.zeros((history_size, 1))
    dfuel_energy_dcorrection = np.zeros((history_size, 1))

    for row in range(1, history_size):
        active = row
        for step in range(active):
            start = step * num_engines
            stop = start + num_engines
            dfuel_burn_dflow[row, start:stop] = dt[step] * mass_flow_correction
            dfuel_burn_ddt[row, step] = source_flow[step]
            dfuel_burn_dcorrection[row, 0] += dt[step] * np.sum(
                fuel_mass_flow[step, :]
            )
            dfuel_energy_dflow[row, start:stop] = (
                dt[step] * mass_flow_correction * fuel_specific_energy
            )
            dfuel_energy_ddt[row, step] = source_flow[step] * fuel_specific_energy
            dfuel_energy_defuel[row, 0] += source_flow[step] * dt[step]
            dfuel_energy_dcorrection[row, 0] += (
                dt[step] * np.sum(fuel_mass_flow[step, :]) * fuel_specific_energy
            )

    scalar_zero = np.zeros((history_size, 1))
    engine_time_zero = np.zeros((engine_size, num_steps))
    engine_scalar_zero = np.zeros((engine_size, 1))
    source_engine_zero = np.zeros((num_steps, engine_size))
    source_time_zero = np.zeros((num_steps, num_steps))
    source_scalar_zero = np.zeros((num_steps, 1))
    result.update(
        {
            "dcorrected_fuel_mass_flow_dfuel_mass_flow": dcorrected_flow_dflow,
            "dcorrected_fuel_mass_flow_dspecific_fuel_consumption": zeros_engine,
            "dcorrected_fuel_mass_flow_dtime_step": engine_time_zero,
            "dcorrected_fuel_mass_flow_dfuel_specific_energy": engine_scalar_zero,
            "dcorrected_fuel_mass_flow_dinitial_fuel_burn": engine_scalar_zero,
            "dcorrected_fuel_mass_flow_dinitial_mass": engine_scalar_zero,
            "dcorrected_fuel_mass_flow_dinitial_fuel_energy": engine_scalar_zero,
            "dcorrected_fuel_mass_flow_dinitial_fuel_energy_left": engine_scalar_zero,
            "dcorrected_fuel_mass_flow_dmass_flow_correction": (
                dcorrected_flow_dcorrection
            ),
            "dcorrected_specific_fuel_consumption_dfuel_mass_flow": zeros_engine,
            "dcorrected_specific_fuel_consumption_dspecific_fuel_consumption": (
                dcorrected_sfc_dsfc
            ),
            "dcorrected_specific_fuel_consumption_dtime_step": engine_time_zero,
            "dcorrected_specific_fuel_consumption_dfuel_specific_energy": (
                engine_scalar_zero
            ),
            "dcorrected_specific_fuel_consumption_dinitial_fuel_burn": (
                engine_scalar_zero
            ),
            "dcorrected_specific_fuel_consumption_dinitial_mass": (
                engine_scalar_zero
            ),
            "dcorrected_specific_fuel_consumption_dinitial_fuel_energy": (
                engine_scalar_zero
            ),
            "dcorrected_specific_fuel_consumption_dinitial_fuel_energy_left": (
                engine_scalar_zero
            ),
            "dcorrected_specific_fuel_consumption_dmass_flow_correction": (
                dcorrected_sfc_dcorrection
            ),
            "dsource_mass_flow_dfuel_mass_flow": dsource_dflow,
            "dsource_mass_flow_dspecific_fuel_consumption": source_engine_zero,
            "dsource_mass_flow_dtime_step": source_time_zero,
            "dsource_mass_flow_dfuel_specific_energy": source_scalar_zero,
            "dsource_mass_flow_dinitial_fuel_burn": source_scalar_zero,
            "dsource_mass_flow_dinitial_mass": source_scalar_zero,
            "dsource_mass_flow_dinitial_fuel_energy": source_scalar_zero,
            "dsource_mass_flow_dinitial_fuel_energy_left": source_scalar_zero,
            "dsource_mass_flow_dmass_flow_correction": dsource_dcorrection,
        }
    )
    result.update(
        fuel_history_derivative_pack(
            "fuel_burn",
            dfuel_burn_dflow,
            dfuel_burn_ddt,
            scalar_zero,
            dfuel_burn_dcorrection,
            history_size,
        )
    )
    result["dfuel_burn_dinitial_fuel_burn"] = np.ones((history_size, 1))
    result["dfuel_burn_dinitial_mass"] = scalar_zero
    result["dfuel_burn_dinitial_fuel_energy"] = scalar_zero
    result["dfuel_burn_dinitial_fuel_energy_left"] = scalar_zero
    result.update(
        fuel_history_derivative_pack(
            "mass",
            -dfuel_burn_dflow,
            -dfuel_burn_ddt,
            scalar_zero,
            -dfuel_burn_dcorrection,
            history_size,
        )
    )
    result["dmass_dinitial_fuel_burn"] = scalar_zero
    result["dmass_dinitial_mass"] = np.ones((history_size, 1))
    result["dmass_dinitial_fuel_energy"] = scalar_zero
    result["dmass_dinitial_fuel_energy_left"] = scalar_zero
    result.update(
        fuel_history_derivative_pack(
            "fuel_energy",
            dfuel_energy_dflow,
            dfuel_energy_ddt,
            dfuel_energy_defuel,
            dfuel_energy_dcorrection,
            history_size,
        )
    )
    result["dfuel_energy_dinitial_fuel_burn"] = scalar_zero
    result["dfuel_energy_dinitial_mass"] = scalar_zero
    result["dfuel_energy_dinitial_fuel_energy"] = np.ones((history_size, 1))
    result["dfuel_energy_dinitial_fuel_energy_left"] = scalar_zero
    result.update(
        fuel_history_derivative_pack(
            "fuel_energy_left",
            -dfuel_energy_dflow,
            -dfuel_energy_ddt,
            -dfuel_energy_defuel,
            -dfuel_energy_dcorrection,
            history_size,
        )
    )
    result["dfuel_energy_left_dinitial_fuel_burn"] = scalar_zero
    result["dfuel_energy_left_dinitial_mass"] = scalar_zero
    result["dfuel_energy_left_dinitial_fuel_energy"] = scalar_zero
    result["dfuel_energy_left_dinitial_fuel_energy_left"] = np.ones((history_size, 1))
    return result


def fuel_history_derivative_pack(
    output,
    dflow,
    dtime,
    dfuel_specific_energy,
    dcorrection,
    history_size,
):
    """Return common fuel history derivative entries for one output."""

    return {
        "d%s_dfuel_mass_flow" % output: dflow,
        "d%s_dspecific_fuel_consumption" % output: np.zeros(
            (history_size, dflow.shape[1])
        ),
        "d%s_dtime_step" % output: dtime,
        "d%s_dfuel_specific_energy" % output: dfuel_specific_energy,
        "d%s_dmass_flow_correction" % output: dcorrection,
    }


def power_available_values(
    true_airspeed,
    density,
    sea_level_static_thrust,
    sea_level_static_power,
    split,
    architecture,
    efficiency,
    transmitter_type,
    num_sources,
    aircraft_class,
):
    """Return FAST PowerAvailable core outputs and dense derivatives."""

    tas = np.asarray(true_airspeed, dtype=float).reshape(-1)
    rho = np.asarray(density, dtype=float).reshape(-1)
    sls_thrust = np.asarray(sea_level_static_thrust, dtype=float).reshape(-1)
    sls_power = np.asarray(sea_level_static_power, dtype=float).reshape(-1)
    split = np.asarray(split, dtype=float)
    architecture = np.asarray(architecture, dtype=float)
    efficiency = np.asarray(efficiency, dtype=float)
    transmitter_type = np.asarray(transmitter_type).reshape(-1)
    aircraft_class = aircraft_class.lower()
    num_points = len(tas)
    num_components = architecture.shape[0]
    num_transmitters = len(transmitter_type)
    nout = num_points * num_components
    nsplits = split.size
    transmitter_start = num_sources
    transmitter_stop = num_sources + num_transmitters
    transmitter_indices = np.arange(transmitter_start, transmitter_stop)
    sink_indices = np.arange(transmitter_stop, num_components)
    flow_indices = np.arange(num_sources, num_components)
    nflow = len(flow_indices)

    pav = np.zeros((num_points, num_components))
    dpav_dtas = np.zeros((nout, num_points))
    dpav_drho = np.zeros((nout, num_points))
    dpav_dsls_thrust = np.zeros((nout, num_transmitters))
    dpav_dsls_power = np.zeros((nout, num_transmitters))
    dpav_dsplit = np.zeros((nout, nsplits))

    for point in range(num_points):
        transmitter_power = sls_power.copy()
        dtrn_dtas = np.zeros((num_transmitters, num_points))
        dtrn_drho = np.zeros((num_transmitters, num_points))
        dtrn_dsls_thrust = np.zeros((num_transmitters, num_transmitters))
        dtrn_dsls_power = np.zeros((num_transmitters, num_transmitters))

        for transmitter in range(num_transmitters):
            if transmitter_type[transmitter] == 1:
                if aircraft_class == "turbofan":
                    transmitter_power[transmitter] = (
                        sls_thrust[transmitter]
                        * rho[point]
                        / RHO_SL_STD
                        * tas[point]
                    )
                    dtrn_dtas[transmitter, point] = (
                        sls_thrust[transmitter] * rho[point] / RHO_SL_STD
                    )
                    dtrn_drho[transmitter, point] = (
                        sls_thrust[transmitter] * tas[point] / RHO_SL_STD
                    )
                    dtrn_dsls_thrust[transmitter, transmitter] = (
                        rho[point] / RHO_SL_STD * tas[point]
                    )
                elif aircraft_class in ("turboprop", "piston"):
                    dtrn_dsls_power[transmitter, transmitter] = 1.0
                else:
                    raise ValueError(f"Invalid aircraft class: {aircraft_class}")
            elif transmitter_type[transmitter] in (0, 2, 3, 4):
                dtrn_dsls_power[transmitter, transmitter] = 1.0
            else:
                raise ValueError(
                    "Invalid transmitter type at position %s." % (transmitter + 1)
                )

        current_split = split[point, :, :]
        sub_split = current_split[np.ix_(transmitter_indices, transmitter_indices)]
        up_transmitters = np.where(
            (np.sum(sub_split, axis=0) > 0.0) | (transmitter_type == 2)
        )[0]
        transmitter_power[up_transmitters] = 0.0
        dtrn_dtas[up_transmitters, :] = 0.0
        dtrn_drho[up_transmitters, :] = 0.0
        dtrn_dsls_thrust[up_transmitters, :] = 0.0
        dtrn_dsls_power[up_transmitters, :] = 0.0

        initial = np.concatenate(
            [transmitter_power, np.zeros(num_components - transmitter_stop)]
        )
        flow = power_flow_values(
            initial,
            architecture[np.ix_(flow_indices, flow_indices)],
            current_split[np.ix_(flow_indices, flow_indices)],
            efficiency[np.ix_(flow_indices, flow_indices)],
            1,
            1.0e-6,
        )
        pav[point, flow_indices] = flow["propagated_power"]
        dflow_dinitial = flow["dpropagated_dinitial_power"]

        for local_output, component in enumerate(flow_indices):
            output_row = point * num_components + component
            dpav_dtas[output_row, :] = (
                dflow_dinitial[local_output, :num_transmitters] @ dtrn_dtas
            )
            dpav_drho[output_row, :] = (
                dflow_dinitial[local_output, :num_transmitters] @ dtrn_drho
            )
            dpav_dsls_thrust[output_row, :] = (
                dflow_dinitial[local_output, :num_transmitters]
                @ dtrn_dsls_thrust
            )
            dpav_dsls_power[output_row, :] = (
                dflow_dinitial[local_output, :num_transmitters] @ dtrn_dsls_power
            )

            for local_source, source in enumerate(flow_indices):
                for local_target, target in enumerate(flow_indices):
                    full_index = (
                        point * num_components * num_components
                        + source * num_components
                        + target
                    )
                    sub_index = local_source * nflow + local_target
                    dpav_dsplit[output_row, full_index] = flow[
                        "dpropagated_dsplit"
                    ][local_output, sub_index]

        for transmitter in range(num_transmitters):
            component = transmitter + num_sources

            if pav[point, component] <= sls_power[transmitter]:
                continue

            output_row = point * num_components + component
            pav[point, component] = sls_power[transmitter]
            dpav_dtas[output_row, :] = 0.0
            dpav_drho[output_row, :] = 0.0
            dpav_dsls_thrust[output_row, :] = 0.0
            dpav_dsls_power[output_row, :] = 0.0
            dpav_dsls_power[output_row, transmitter] = 1.0
            dpav_dsplit[output_row, :] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        tav = pav / tas[:, None]
    tv = np.sum(pav[:, sink_indices], axis=1)
    dtav_dtas = np.zeros_like(dpav_dtas)
    dtav_drho = np.zeros_like(dpav_drho)
    dtav_dsls_thrust = np.zeros_like(dpav_dsls_thrust)
    dtav_dsls_power = np.zeros_like(dpav_dsls_power)
    dtav_dsplit = np.zeros_like(dpav_dsplit)
    dtv_dtas = np.zeros((num_points, num_points))
    dtv_drho = np.zeros((num_points, num_points))
    dtv_dsls_thrust = np.zeros((num_points, num_transmitters))
    dtv_dsls_power = np.zeros((num_points, num_transmitters))
    dtv_dsplit = np.zeros((num_points, nsplits))

    for point in range(num_points):
        for component in range(num_components):
            row = point * num_components + component
            dtav_dtas[row, :] = dpav_dtas[row, :] / tas[point]
            dtav_dtas[row, point] -= pav[point, component] / tas[point] ** 2
            dtav_drho[row, :] = dpav_drho[row, :] / tas[point]
            dtav_dsls_thrust[row, :] = dpav_dsls_thrust[row, :] / tas[point]
            dtav_dsls_power[row, :] = dpav_dsls_power[row, :] / tas[point]
            dtav_dsplit[row, :] = dpav_dsplit[row, :] / tas[point]

        sink_rows = point * num_components + sink_indices
        dtv_dtas[point, :] = np.sum(dpav_dtas[sink_rows, :], axis=0)
        dtv_drho[point, :] = np.sum(dpav_drho[sink_rows, :], axis=0)
        dtv_dsls_thrust[point, :] = np.sum(dpav_dsls_thrust[sink_rows, :], axis=0)
        dtv_dsls_power[point, :] = np.sum(dpav_dsls_power[sink_rows, :], axis=0)
        dtv_dsplit[point, :] = np.sum(dpav_dsplit[sink_rows, :], axis=0)

    return {
        "available_power": pav,
        "available_thrust": tav,
        "thrust_velocity": tv,
        "dpav_dtas": dpav_dtas,
        "dpav_drho": dpav_drho,
        "dpav_dsls_thrust": dpav_dsls_thrust,
        "dpav_dsls_power": dpav_dsls_power,
        "dpav_dsplit": dpav_dsplit,
        "dtav_dtas": dtav_dtas,
        "dtav_drho": dtav_drho,
        "dtav_dsls_thrust": dtav_dsls_thrust,
        "dtav_dsls_power": dtav_dsls_power,
        "dtav_dsplit": dtav_dsplit,
        "dtv_dtas": dtv_dtas,
        "dtv_drho": dtv_drho,
        "dtv_dsls_thrust": dtv_dsls_thrust,
        "dtv_dsls_power": dtv_dsls_power,
        "dtv_dsplit": dtv_dsplit,
    }


def power_flow_values(power, architecture, split, efficiency, direction, tolerance):
    """Return FAST power-flow propagation and dense analytical partials."""

    pwr = np.asarray(power, dtype=float).reshape(-1).copy()
    architecture = np.asarray(architecture, dtype=float)
    split = np.asarray(split, dtype=float)
    efficiency = np.asarray(efficiency, dtype=float)
    num_components = len(pwr)
    nmatrix = split.size
    previous = np.zeros(num_components)
    dpower = np.eye(num_components)
    dsplit = np.zeros((num_components, nmatrix))
    defficiency = np.zeros((num_components, nmatrix))

    if direction == 1:
        matrix = (split * architecture * efficiency).T
    elif direction == -1:
        matrix = (split * architecture / efficiency).T
    else:
        raise ValueError("direction must be +1 for upstream or -1 for downstream.")

    iteration = 0

    while (
        np.linalg.norm(previous - pwr) > tolerance
        and iteration < num_components
    ):
        old_power = pwr.copy()
        old_dpower = dpower.copy()
        old_dsplit = dsplit.copy()
        old_defficiency = defficiency.copy()
        propagated = matrix @ old_power
        previous = old_power
        update = np.abs(propagated) > tolerance

        for output_index in np.where(update)[0]:
            pwr[output_index] = propagated[output_index]
            dpower[output_index, :] = matrix[output_index, :] @ old_dpower
            dsplit[output_index, :] = matrix[output_index, :] @ old_dsplit
            defficiency[output_index, :] = (
                matrix[output_index, :] @ old_defficiency
            )

            for input_index in range(num_components):
                arch_value = architecture[input_index, output_index]

                if abs(arch_value) < 1.0e-15:
                    continue

                parameter_index = input_index * num_components + output_index
                source_power = old_power[input_index]

                if direction == 1:
                    dmatrix_dsplit = arch_value * efficiency[
                        input_index,
                        output_index,
                    ]
                    dmatrix_defficiency = arch_value * split[
                        input_index,
                        output_index,
                    ]
                else:
                    dmatrix_dsplit = arch_value / efficiency[
                        input_index,
                        output_index,
                    ]
                    dmatrix_defficiency = (
                        -arch_value
                        * split[input_index, output_index]
                        / efficiency[input_index, output_index] ** 2
                    )

                dsplit[output_index, parameter_index] += (
                    dmatrix_dsplit * source_power
                )
                defficiency[output_index, parameter_index] += (
                    dmatrix_defficiency * source_power
                )

        iteration += 1

    return {
        "propagated_power": pwr,
        "dpropagated_dinitial_power": dpower,
        "dpropagated_dsplit": dsplit,
        "dpropagated_defficiency": defficiency,
    }


def engine_thrust_requirement_values(
    architecture,
    transmitter_type,
    num_sources,
    thrust_output,
    component,
):
    """Return FAST engine thrust requirement vector and analytical partials."""

    architecture = np.asarray(architecture, dtype=float)
    transmitter_type = np.asarray(transmitter_type).reshape(-1)
    thrust_output = np.asarray(thrust_output, dtype=float)
    num_points, num_components = thrust_output.shape
    required = np.zeros(num_points)
    derivatives = np.zeros((num_points, thrust_output.size))
    thrust_components = np.where(transmitter_type == 2)[0] + num_sources

    if len(thrust_components) == 0:
        return {
            "required_thrust": required,
            "drequired_dthrust_output": derivatives,
        }

    connected = thrust_components[
        architecture[component, thrust_components] > 0
    ]

    if len(connected) == 0:
        return {
            "required_thrust": required,
            "drequired_dthrust_output": derivatives,
        }

    thrust_component = connected[0]

    for point in range(num_points):
        thrust = thrust_output[point, thrust_component]

        if np.isfinite(thrust):
            required[point] = thrust
            derivatives[
                point,
                point * num_components + thrust_component,
            ] = 1.0

    return {
        "required_thrust": required,
        "drequired_dthrust_output": derivatives,
    }


def turboprop_engine_weight_for_sizing_values(
    downstream_power,
    engines,
    power_sls,
    dry_weight,
):
    """Return FAST turboprop/piston sizing engine weights and derivatives."""

    downstream_power = np.asarray(downstream_power, dtype=float).reshape(-1)
    engines = np.asarray(engines, dtype=bool).reshape(-1)
    power_sls = np.asarray(power_sls, dtype=float).reshape(-1)
    dry_weight = np.asarray(dry_weight, dtype=float).reshape(-1)
    valid = (~np.isnan(power_sls)) & (~np.isnan(dry_weight))

    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two valid engine database rows are required.")

    fit = np.polyfit(power_sls[valid], dry_weight[valid], 1)
    derivatives = np.zeros((max(1, np.count_nonzero(engines)), downstream_power.size))
    active_columns = np.flatnonzero(engines)

    if len(active_columns) == 0:
        return {
            "engine_weight": np.zeros(1),
            "dengine_weight_ddownstream_power": derivatives,
        }

    active_power = downstream_power[:-1][engines] / 1000.0
    engine_weight = np.polyval(fit, active_power)

    for row, column in enumerate(active_columns):
        derivatives[row, column] = fit[0] / 1000.0

    return {
        "engine_weight": engine_weight,
        "dengine_weight_ddownstream_power": derivatives,
    }


def turbofan_engine_weight_for_sizing_values(
    downstream_thrust,
    engines,
    data_matrix,
    hyperparams,
    inverse_term,
    prior,
):
    """Return FAST turbofan sizing engine weights and derivatives."""

    downstream_thrust = np.asarray(downstream_thrust, dtype=float).reshape(-1)
    engines = np.asarray(engines, dtype=bool).reshape(-1)
    active_columns = np.flatnonzero(engines)
    row_count = max(1, len(active_columns))
    engine_weight = np.zeros(row_count)
    derivatives = np.zeros((row_count, downstream_thrust.size))

    for row, column in enumerate(active_columns):
        prediction = gaussian_process_prediction_values(
            data_matrix,
            hyperparams,
            inverse_term,
            [downstream_thrust[column]],
            prior,
        )
        engine_weight[row] = prediction["posterior_mean"]
        derivatives[row, column] = prediction["dposterior_mean_dtarget"][0]

    return {
        "engine_weight": engine_weight,
        "dengine_weight_ddownstream_thrust": derivatives,
    }


def cable_weight_for_sizing_values(
    downstream_power,
    cable_power_to_weight,
    cables,
    cable_connections,
    cable_lengths,
):
    """Return FAST cable sizing weight and analytical partials."""

    downstream_power = np.asarray(downstream_power, dtype=float).reshape(-1)
    cables = np.asarray(cables, dtype=bool).reshape(-1)
    cable_connections = np.asarray(cable_connections, dtype=float)
    cable_lengths = np.asarray(cable_lengths, dtype=float)
    derivatives = np.zeros((1, downstream_power.size))

    if not np.any(cables):
        return {
            "cable_weight": 0.0,
            "dcable_weight_ddownstream_power": derivatives,
            "dcable_weight_dcable_power_to_weight": np.asarray([[0.0]]),
        }

    cable_power = downstream_power[:-1][cables]
    power_matrix = np.tile(cable_power, (cable_connections.shape[0], 1))
    weighted_geometry = cable_connections * cable_lengths
    geometry_sum = np.sum(weighted_geometry, axis=0)
    weight = float(
        np.sum(cable_power_to_weight * (power_matrix / 1.0e6) * weighted_geometry)
    )
    cable_indices = np.where(cables)[0]

    for local_index, component_index in enumerate(cable_indices):
        derivatives[0, component_index] = (
            cable_power_to_weight * geometry_sum[local_index] / 1.0e6
        )

    return {
        "cable_weight": weight,
        "dcable_weight_ddownstream_power": derivatives,
        "dcable_weight_dcable_power_to_weight": np.asarray(
            [[np.sum((power_matrix / 1.0e6) * weighted_geometry)]]
        ),
    }


def battery_energy_history_values(
    source_power,
    time_step,
    initial_source_energy,
    initial_source_energy_left,
    battery_sources,
):
    """Return smooth FAST battery-energy histories and dense partials."""

    source_power = np.asarray(source_power, dtype=float)
    time_step = np.asarray(time_step, dtype=float).reshape(-1)
    initial_source_energy = np.asarray(initial_source_energy, dtype=float).reshape(-1)
    initial_source_energy_left = np.asarray(
        initial_source_energy_left,
        dtype=float,
    ).reshape(-1)
    battery_sources = np.asarray(battery_sources, dtype=bool).reshape(-1)
    num_points, num_sources = source_power.shape
    output_size = num_points * num_sources
    source_energy = np.tile(initial_source_energy, (num_points, 1))
    source_energy_left = np.tile(initial_source_energy_left, (num_points, 1))
    denergy_dpower = np.zeros((output_size, source_power.size))
    dleft_dpower = np.zeros((output_size, source_power.size))
    denergy_dtime = np.zeros((output_size, time_step.size))
    dleft_dtime = np.zeros((output_size, time_step.size))
    denergy_dinitial = np.zeros((output_size, num_sources))
    dleft_dinitial = np.zeros((output_size, num_sources))
    denergy_dinitial_left = np.zeros((output_size, num_sources))
    dleft_dinitial_left = np.zeros((output_size, num_sources))

    for row in range(num_points):
        for source in range(num_sources):
            output_index = row * num_sources + source
            denergy_dinitial[output_index, source] = 1.0
            dleft_dinitial_left[output_index, source] = 1.0

    for source in np.where(battery_sources)[0]:
        used = np.cumsum(source_power[:-1, source] * time_step)
        source_energy[1:, source] = initial_source_energy[source] + used
        source_energy_left[1:, source] = initial_source_energy_left[source] - used

        for row in range(1, num_points):
            output_index = row * num_sources + source

            for step in range(row):
                power_index = step * num_sources + source
                denergy_dpower[output_index, power_index] = time_step[step]
                dleft_dpower[output_index, power_index] = -time_step[step]
                denergy_dtime[output_index, step] = source_power[step, source]
                dleft_dtime[output_index, step] = -source_power[step, source]

    return {
        "source_energy": source_energy,
        "source_energy_left": source_energy_left,
        "dsource_energy_dsource_power": denergy_dpower,
        "dsource_energy_dtime_step": denergy_dtime,
        "dsource_energy_dinitial_source_energy": denergy_dinitial,
        "dsource_energy_dinitial_source_energy_left": denergy_dinitial_left,
        "dsource_energy_left_dsource_power": dleft_dpower,
        "dsource_energy_left_dtime_step": dleft_dtime,
        "dsource_energy_left_dinitial_source_energy": dleft_dinitial,
        "dsource_energy_left_dinitial_source_energy_left": dleft_dinitial_left,
    }


def power_supplement_values(
    required_power,
    architecture,
    split,
    efficiency,
    transmitter_type,
    fan_efficiency,
):
    """Return FAST supplemental power and dense analytical partials."""

    required_power = np.asarray(required_power, dtype=float)
    architecture = np.asarray(architecture, dtype=float)
    split = np.asarray(split, dtype=float)
    efficiency = np.asarray(efficiency, dtype=float)
    transmitter_type = np.asarray(transmitter_type).reshape(-1)
    num_points, num_transmitters = required_power.shape
    supplemental = np.zeros((num_points, num_transmitters))
    nout = num_points * num_transmitters
    nrequired = required_power.size
    nmatrix = split.size
    drequired = np.zeros((nout, nrequired))
    dsplit = np.zeros((nout, nmatrix))
    defficiency = np.zeros((nout, nmatrix))
    dfan = np.zeros((nout, 1))

    gas_turbines = np.where(transmitter_type == 1)[0]

    for gas_turbine in gas_turbines:
        current_row = architecture[gas_turbine, :]
        connections = (current_row == 1) & (transmitter_type != 2)

        if not np.any(connections):
            continue

        connected_indices = np.where(connections)[0]
        factors = split[connections, gas_turbine] / efficiency[
            connections,
            gas_turbine,
        ]
        supplemental[:, gas_turbine] -= required_power[:, connected_indices] @ factors

        for point in range(num_points):
            output_index = point * num_transmitters + gas_turbine

            for local_index, component in enumerate(connected_indices):
                matrix_index = component * num_transmitters + gas_turbine
                required_index = point * num_transmitters + component
                required = required_power[point, component]
                factor = factors[local_index]
                drequired[output_index, required_index] -= factor
                dsplit[output_index, matrix_index] -= required / efficiency[
                    component,
                    gas_turbine,
                ]
                defficiency[output_index, matrix_index] += (
                    required
                    * split[component, gas_turbine]
                    / efficiency[component, gas_turbine] ** 2
                )

    parallel_components = np.where(np.sum(architecture, axis=0) > 1)[0]

    for component in parallel_components:
        connected = architecture[:, component] > 0
        driving = np.where(connected & (transmitter_type == 1))[0]

        if len(driving) != 1:
            continue

        helping = np.where(connected & (transmitter_type == 0))[0]

        if len(helping) == 0:
            continue

        driver = driving[0]
        supplemental[:, driver] += (
            np.sum(required_power[:, helping], axis=1) * fan_efficiency
        )

        for point in range(num_points):
            output_index = point * num_transmitters + driver
            dfan[output_index, 0] += np.sum(required_power[point, helping])

            for helper in helping:
                required_index = point * num_transmitters + helper
                drequired[output_index, required_index] += fan_efficiency

    return {
        "supplemental_power": supplemental,
        "dsupplemental_drequired_power": drequired,
        "dsupplemental_dsplit": dsplit,
        "dsupplemental_defficiency": defficiency,
        "dsupplemental_dfan_efficiency": dfan,
    }
