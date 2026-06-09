# src/fast_openmdao/battery.py

"""OpenMDAO components for FAST battery primitive equations."""

import math

import numpy as np
import openmdao.api as om


class AvailableCellCapacity(om.ExplicitComponent):
    """Return effective single-cell capacity after optional degradation scaling."""

    def initialize(self):
        self.options.declare("analysis_type", default=0)
        self.options.declare("degradation", default=0)

    def setup(self):
        self.add_input("cap_cell", val=2.4)
        self.add_input("state_of_health", val=100.0)
        self.add_output("available_cell_capacity", val=2.4)
        self.declare_partials(of="available_cell_capacity", wrt="*")

    def compute(self, inputs, outputs):
        outputs["available_cell_capacity"] = available_cell_capacity_value(
            inputs["cap_cell"][0],
            inputs["state_of_health"][0],
            self.options["analysis_type"],
            self.options["degradation"],
        )

    def compute_partials(self, inputs, partials):
        cap_cell = inputs["cap_cell"][0]

        if self.options["analysis_type"] < 0 and self.options["degradation"] == 1:
            partials["available_cell_capacity", "cap_cell"] = (
                inputs["state_of_health"][0] / 100.0
            )
            partials["available_cell_capacity", "state_of_health"] = cap_cell / 100.0
            return

        partials["available_cell_capacity", "cap_cell"] = 1.0
        partials["available_cell_capacity", "state_of_health"] = 0.0


class BatteryCurrent(om.ExplicitComponent):
    """Solve FAST's selected quadratic cell-current root."""

    def initialize(self):
        self.options.declare("is_discharge", default=True)

    def setup(self):
        self.add_input("hot_voltage", val=0.01)
        self.add_input("cold_voltage", val=4.0)
        self.add_input("requested_cell_power", val=10.0, units="W")
        self.add_output("cell_current", val=2.0, units="A")
        self.declare_partials(of="cell_current", wrt="*")

    def compute(self, inputs, outputs):
        outputs["cell_current"] = battery_current_value(
            inputs["hot_voltage"][0],
            inputs["cold_voltage"][0],
            inputs["requested_cell_power"][0],
            self.options["is_discharge"],
        )

    def compute_partials(self, inputs, partials):
        values = battery_current_derivatives(
            inputs["hot_voltage"][0],
            inputs["cold_voltage"][0],
            inputs["requested_cell_power"][0],
            self.options["is_discharge"],
        )
        partials["cell_current", "hot_voltage"] = values["dcurrent_dhot_voltage"]
        partials["cell_current", "cold_voltage"] = values["dcurrent_dcold_voltage"]
        partials["cell_current", "requested_cell_power"] = values[
            "dcurrent_drequested_cell_power"
        ]


class BatteryWeightFromEnergy(om.ExplicitComponent):
    """Size battery-source weight from final mission energy use."""

    def initialize(self):
        self.options.declare("src_type", default=(0.0,))
        self.options.declare("npoint", default=1)

    def setup(self):
        src_type = np.asarray(self.options["src_type"], dtype=float).reshape(-1)
        npoint = self.options["npoint"]
        nbatt = max(1, int(np.count_nonzero(src_type == 0.0)))
        self._src_type = src_type
        self._npoint = npoint
        self._nbatt = nbatt

        self.add_input("battery_source_energy", shape=(npoint, len(src_type)), units="J")
        self.add_input("battery_specific_energy", val=1.0, units="J/kg")
        self.add_output("battery_weight", shape=nbatt, units="kg")
        self.declare_partials(of="battery_weight", wrt="*")

    def compute(self, inputs, outputs):
        outputs["battery_weight"] = battery_weight_from_energy_values(
            self._src_type,
            inputs["battery_source_energy"],
            inputs["battery_specific_energy"][0],
        )["battery_weight"]

    def compute_partials(self, inputs, partials):
        values = battery_weight_from_energy_values(
            self._src_type,
            inputs["battery_source_energy"],
            inputs["battery_specific_energy"][0],
        )
        partials["battery_weight", "battery_source_energy"] = values[
            "dbattery_weight_dbattery_source_energy"
        ]
        partials["battery_weight", "battery_specific_energy"] = values[
            "dbattery_weight_dbattery_specific_energy"
        ]


class BatteryPowerStep(om.ExplicitComponent):
    """Run one FAST battery equivalent-circuit discharge or charge interval."""

    def initialize(self):
        self.options.declare("is_discharge", default=True)
        self.options.declare("analysis_type", default=0)
        self.options.declare("degradation", default=0)

    def setup(self):
        self.add_input("requested_power", val=1000.0, units="W")
        self.add_input("time", val=60.0, units="s")
        self.add_input("soc_begin", val=80.0)
        self.add_input("parallel_cells", val=10.0)
        self.add_input("series_cells", val=100.0)
        self.add_input("max_cell_voltage", val=4.2)
        self.add_input("internal_resistance", val=0.01)
        self.add_input("exponential_voltage", val=0.1)
        self.add_input("exponential_capacity", val=1.0)
        self.add_input("cap_cell", val=2.4)
        self.add_input("state_of_health", val=100.0)
        self.add_output("voltage", val=400.0, units="V")
        self.add_output("current", val=2.5, units="A")
        self.add_output("output_power", val=1000.0, units="W")
        self.add_output("capacity", val=20.0)
        self.add_output("soc_end", val=79.0)
        self.add_output("c_rate", val=0.1)
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = battery_power_step_values(
            inputs["requested_power"][0],
            inputs["time"][0],
            inputs["soc_begin"][0],
            inputs["parallel_cells"][0],
            inputs["series_cells"][0],
            inputs["max_cell_voltage"][0],
            inputs["internal_resistance"][0],
            inputs["exponential_voltage"][0],
            inputs["exponential_capacity"][0],
            inputs["cap_cell"][0],
            inputs["state_of_health"][0],
            self.options["is_discharge"],
            self.options["analysis_type"],
            self.options["degradation"],
        )

        for output in battery_power_step_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = battery_power_step_values(
            inputs["requested_power"][0],
            inputs["time"][0],
            inputs["soc_begin"][0],
            inputs["parallel_cells"][0],
            inputs["series_cells"][0],
            inputs["max_cell_voltage"][0],
            inputs["internal_resistance"][0],
            inputs["exponential_voltage"][0],
            inputs["exponential_capacity"][0],
            inputs["cap_cell"][0],
            inputs["state_of_health"][0],
            self.options["is_discharge"],
            self.options["analysis_type"],
            self.options["degradation"],
        )

        for output in battery_power_step_output_names():
            for variable in battery_power_step_input_names():
                partials[output, variable] = values[
                    "d%s_d%s" % (output, variable)
                ]


def available_cell_capacity_value(cap_cell, state_of_health, analysis_type, degradation):
    """Return FAST effective cell capacity scalar value."""

    if analysis_type < 0 and degradation == 1:
        return cap_cell * state_of_health / 100.0

    return cap_cell


def battery_current_value(hot_voltage, cold_voltage, requested_cell_power, is_discharge):
    """Return selected real quadratic root for FAST battery current."""

    discriminant = cold_voltage ** 2 + 4.0 * hot_voltage * requested_cell_power

    if discriminant < 0.0:
        raise ValueError("Analytical BatteryCurrent requires real roots.")

    root = math.sqrt(discriminant)

    if abs(hot_voltage) < 1.0e-14:
        current = requested_cell_power / cold_voltage
    else:
        current = (-cold_voltage + root) / (2.0 * hot_voltage)

    if is_discharge and current < 0.0:
        return 0.0

    if not is_discharge and current > 0.0:
        return 0.0

    return current


def battery_current_derivatives(
    hot_voltage,
    cold_voltage,
    requested_cell_power,
    is_discharge,
):
    """Return implicit derivatives for the selected battery-current root."""

    current = battery_current_value(
        hot_voltage,
        cold_voltage,
        requested_cell_power,
        is_discharge,
    )

    if current == 0.0:
        return {
            "dcurrent_dhot_voltage": 0.0,
            "dcurrent_dcold_voltage": 0.0,
            "dcurrent_drequested_cell_power": 0.0,
        }

    denominator = 2.0 * hot_voltage * current + cold_voltage
    return {
        "dcurrent_dhot_voltage": -(current ** 2) / denominator,
        "dcurrent_dcold_voltage": -current / denominator,
        "dcurrent_drequested_cell_power": 1.0 / denominator,
    }


def battery_power_step_input_names():
    """Return scalar input names for BatteryPowerStep derivatives."""

    return (
        "requested_power",
        "time",
        "soc_begin",
        "parallel_cells",
        "series_cells",
        "max_cell_voltage",
        "internal_resistance",
        "exponential_voltage",
        "exponential_capacity",
        "cap_cell",
        "state_of_health",
    )


def battery_power_step_output_names():
    """Return scalar output names for BatteryPowerStep."""

    return (
        "voltage",
        "current",
        "output_power",
        "capacity",
        "soc_end",
        "c_rate",
    )


def battery_power_step_values(
    requested_power,
    time,
    soc_begin,
    parallel_cells,
    series_cells,
    max_cell_voltage,
    internal_resistance,
    exponential_voltage,
    exponential_capacity,
    cap_cell,
    state_of_health,
    is_discharge,
    analysis_type,
    degradation,
):
    """Return one FAST battery equivalent-circuit step and derivatives."""

    variables = battery_power_step_input_names()
    derivatives = {}

    def seed(name):
        return {variable: 1.0 if variable == name else 0.0 for variable in variables}

    def constant():
        return {variable: 0.0 for variable in variables}

    def combine(*terms):
        result = constant()

        for scale, derivative in terms:
            for variable in variables:
                result[variable] += scale * derivative[variable]

        return result

    drequested = seed("requested_power")
    dtime = seed("time")
    dsoc = seed("soc_begin")
    dparallel = seed("parallel_cells")
    dseries = seed("series_cells")
    dmax_voltage = seed("max_cell_voltage")
    dinternal_resistance = seed("internal_resistance")
    dexp_voltage = seed("exponential_voltage")
    dexp_capacity = seed("exponential_capacity")
    dcap_cell = seed("cap_cell")
    dstate_of_health = seed("state_of_health")

    q_cell = available_cell_capacity_value(
        cap_cell,
        state_of_health,
        analysis_type,
        degradation,
    )

    if analysis_type < 0 and degradation == 1:
        dq_cell = combine(
            (state_of_health / 100.0, dcap_cell),
            (cap_cell / 100.0, dstate_of_health),
        )
    else:
        dq_cell = dcap_cell

    ncell = series_cells * parallel_cells
    dncell = combine((parallel_cells, dseries), (series_cells, dparallel))
    requested_cell_power = requested_power / ncell
    drequested_cell_power = combine(
        (1.0 / ncell, drequested),
        (-requested_power / ncell ** 2, dncell),
    )
    soc_fraction = soc_begin / 100.0
    dsoc_fraction = combine((0.01, dsoc))
    discharged_start = (1.0 - soc_fraction) * q_cell
    ddischarged_start = combine(
        (1.0 - soc_fraction, dq_cell),
        (-q_cell, dsoc_fraction),
    )
    polarized_voltage = 0.0011
    discharge_curve_slope = 0.29732

    if is_discharge:
        hot_voltage = -(polarized_voltage / soc_fraction + internal_resistance)
        dhot_voltage = combine(
            (polarized_voltage / soc_fraction ** 2, dsoc_fraction),
            (-1.0, dinternal_resistance),
        )
    else:
        charge_denominator = 1.1 - soc_fraction
        hot_voltage = polarized_voltage / charge_denominator + internal_resistance
        dhot_voltage = combine(
            (polarized_voltage / charge_denominator ** 2, dsoc_fraction),
            (1.0, dinternal_resistance),
        )

    exp_term = math.exp(-exponential_capacity * discharged_start)
    cold_voltage = (
        max_cell_voltage
        + exponential_voltage * exp_term
        - polarized_voltage * discharged_start / soc_fraction
        - discharge_curve_slope * discharged_start
    )
    dcold_voltage = combine(
        (1.0, dmax_voltage),
        (exp_term, dexp_voltage),
        (
            -exponential_voltage * exp_term * discharged_start,
            dexp_capacity,
        ),
        (
            -exponential_voltage * exp_term * exponential_capacity
            - polarized_voltage / soc_fraction
            - discharge_curve_slope,
            ddischarged_start,
        ),
        (
            polarized_voltage * discharged_start / soc_fraction ** 2,
            dsoc_fraction,
        ),
    )
    cell_current = battery_current_value(
        hot_voltage,
        cold_voltage,
        requested_cell_power,
        is_discharge,
    )
    current_partials = battery_current_derivatives(
        hot_voltage,
        cold_voltage,
        requested_cell_power,
        is_discharge,
    )
    dcell_current = combine(
        (current_partials["dcurrent_dhot_voltage"], dhot_voltage),
        (current_partials["dcurrent_dcold_voltage"], dcold_voltage),
        (
            current_partials["dcurrent_drequested_cell_power"],
            drequested_cell_power,
        ),
    )
    current = cell_current * parallel_cells
    dcurrent = combine((parallel_cells, dcell_current), (cell_current, dparallel))
    cell_voltage = cold_voltage + hot_voltage * cell_current
    dcell_voltage = combine(
        (1.0, dcold_voltage),
        (cell_current, dhot_voltage),
        (hot_voltage, dcell_current),
    )
    voltage = cell_voltage * series_cells
    dvoltage = combine((series_cells, dcell_voltage), (cell_voltage, dseries))
    time_hours = time / 3600.0
    dtime_hours = combine((1.0 / 3600.0, dtime))
    discharged_capacity = cell_current * time_hours
    ddischarged_capacity = combine(
        (time_hours, dcell_current),
        (cell_current, dtime_hours),
    )
    soc_end = soc_begin - 100.0 * discharged_capacity / q_cell
    dsoc_end = combine(
        (1.0, dsoc),
        (-100.0 / q_cell, ddischarged_capacity),
        (100.0 * discharged_capacity / q_cell ** 2, dq_cell),
    )
    capacity_unclipped = q_cell * soc_begin / 100.0 * parallel_cells
    capacity_max = q_cell * parallel_cells

    if capacity_unclipped <= capacity_max:
        capacity = capacity_unclipped
        dcapacity = combine(
            (soc_begin * parallel_cells / 100.0, dq_cell),
            (q_cell * parallel_cells / 100.0, dsoc),
            (q_cell * soc_begin / 100.0, dparallel),
        )
    else:
        capacity = capacity_max
        dcapacity = combine((parallel_cells, dq_cell), (q_cell, dparallel))

    output_power = voltage * current
    doutput_power = combine((current, dvoltage), (voltage, dcurrent))
    c_rate = current / (q_cell * parallel_cells)
    dc_rate = combine(
        (1.0 / (q_cell * parallel_cells), dcurrent),
        (-current / (q_cell ** 2 * parallel_cells), dq_cell),
        (-current / (q_cell * parallel_cells ** 2), dparallel),
    )

    values = {
        "voltage": voltage,
        "current": current,
        "output_power": output_power,
        "capacity": capacity,
        "soc_end": soc_end,
        "c_rate": c_rate,
    }
    output_derivatives = {
        "voltage": dvoltage,
        "current": dcurrent,
        "output_power": doutput_power,
        "capacity": dcapacity,
        "soc_end": dsoc_end,
        "c_rate": dc_rate,
    }

    for output in battery_power_step_output_names():
        for variable in variables:
            values["d%s_d%s" % (output, variable)] = output_derivatives[output][
                variable
            ]

    return values


def battery_weight_from_energy_values(
    src_type,
    battery_source_energy,
    battery_specific_energy,
):
    """Return FAST simple ResizeBattery weight and dense derivatives."""

    src_type = np.asarray(src_type, dtype=float).reshape(-1)
    energy = np.asarray(battery_source_energy, dtype=float)
    battery_columns = np.flatnonzero(src_type == 0.0)
    nbatt = max(1, len(battery_columns))
    weight = np.zeros(nbatt)
    dweight_denergy = np.zeros((nbatt, energy.size))
    dweight_dspecific_energy = np.zeros(nbatt)

    if len(battery_columns) == 0:
        return {
            "battery_weight": weight,
            "dbattery_weight_dbattery_source_energy": dweight_denergy,
            "dbattery_weight_dbattery_specific_energy": dweight_dspecific_energy,
        }

    battery_energy = energy[:, battery_columns]

    if np.sum(battery_energy) == 0.0:
        return {
            "battery_weight": weight,
            "dbattery_weight_dbattery_source_energy": dweight_denergy,
            "dbattery_weight_dbattery_specific_energy": dweight_dspecific_energy,
        }

    final_energy = battery_energy[-1, :]
    weight = final_energy / battery_specific_energy
    nsrc = len(src_type)

    for index, column in enumerate(battery_columns):
        dweight_denergy[index, (energy.shape[0] - 1) * nsrc + column] = (
            1.0 / battery_specific_energy
        )

    dweight_dspecific_energy = -final_energy / battery_specific_energy ** 2
    return {
        "battery_weight": weight,
        "dbattery_weight_dbattery_source_energy": dweight_denergy,
        "dbattery_weight_dbattery_specific_energy": dweight_dspecific_energy,
    }
