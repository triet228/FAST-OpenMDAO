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
