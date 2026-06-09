# src/fast_openmdao/propulsion.py

"""OpenMDAO components for FAST propulsion primitive equations."""

import numpy as np
import openmdao.api as om

from fast_python.atmosphere import standard_atmosphere


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
