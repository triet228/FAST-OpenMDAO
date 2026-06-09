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
