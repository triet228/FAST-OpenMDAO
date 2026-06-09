# src/fast_openmdao/battery.py

"""OpenMDAO components for FAST battery primitive equations."""

import math

import numpy as np
import openmdao.api as om


CYCLING_AGING_PARAMETERS = {
    1: {
        "beta": 0.001673,
        "coeff_T": 21.6745,
        "coeff_DOD": 0.022,
        "coeff_Cch": 0.2553,
        "coeff_Cdch": 0.1571,
        "coeff_mSOC": -0.0212,
        "alpha": 0.915,
        "temp_ref": 293.15,
        "mSOC_ref": 42.0,
    },
    2: {
        "beta": 0.003414,
        "coeff_T": 5.8755,
        "coeff_DOD": -0.0045,
        "coeff_Cch": 0.1038,
        "coeff_Cdch": 0.296,
        "coeff_mSOC": 0.0513,
        "alpha": 0.869,
        "temp_ref": 293.15,
        "mSOC_ref": 42.0,
    },
}


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


class BatteryChargeOCV(om.ExplicitComponent):
    """Estimate the ground-charge open-circuit voltage for one cell."""

    def initialize(self):
        self.options.declare("analysis_type", default=0)
        self.options.declare("degradation", default=0)

    def setup(self):
        self.add_input("soc_begin", val=80.0)
        self.add_input("parallel_cells", val=10.0)
        self.add_input("series_cells", val=100.0)
        self.add_input("max_cell_voltage", val=4.2)
        self.add_input("internal_resistance", val=0.01)
        self.add_input("exponential_voltage", val=0.1)
        self.add_input("exponential_capacity", val=1.0)
        self.add_input("cap_cell", val=2.4)
        self.add_input("state_of_health", val=100.0)
        self.add_output("open_circuit_voltage", val=4.0, units="V")
        self.declare_partials(of="open_circuit_voltage", wrt="*")

    def compute(self, inputs, outputs):
        values = battery_charge_ocv_values(
            inputs["soc_begin"][0],
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
        outputs["open_circuit_voltage"] = values["open_circuit_voltage"]

    def compute_partials(self, inputs, partials):
        values = battery_charge_ocv_values(
            inputs["soc_begin"][0],
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

        for variable in battery_charge_ocv_input_names():
            partials["open_circuit_voltage", variable] = values[
                "dopen_circuit_voltage_d%s" % variable
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


class BatteryNonzeroMean(om.ExplicitComponent):
    """Compute FAST's mean of nonzero battery-history values.

    Inputs:
        values: Battery history vector, such as C-rate.

    Outputs:
        nonzero_mean: Mean of entries that are nonzero, or zero if no entries
            are active.

    Assumptions:
        The nonzero mask is the active branch. Analytical derivatives are exact
        away from entries crossing zero.
    """

    def initialize(self):
        self.options.declare("vec_size", default=1)

    def setup(self):
        vec_size = self.options["vec_size"]
        self.add_input("values", val=np.ones(vec_size))
        self.add_output("nonzero_mean", val=1.0)
        self.declare_partials(of="nonzero_mean", wrt="values")

    def compute(self, inputs, outputs):
        outputs["nonzero_mean"] = battery_nonzero_mean_values(inputs["values"])[
            "nonzero_mean"
        ]

    def compute_partials(self, inputs, partials):
        values = battery_nonzero_mean_values(inputs["values"])
        partials["nonzero_mean", "values"] = values["dnonzero_mean_dvalues"]


class BatteryVector(om.ExplicitComponent):
    """Normalize FAST battery values to a one-dimensional vector.

    Inputs:
        values: Fixed-shape scalar, vector, or matrix battery data.

    Outputs:
        vector: Flattened values matching ``fast_python.battery.as_vector``
            for numeric inputs.
    """

    def initialize(self):
        self.options.declare("input_shape", default=(1,))

    def setup(self):
        input_shape = battery_shape_tuple(self.options["input_shape"])
        output_size = int(np.prod(input_shape))
        self.add_input("values", val=np.zeros(input_shape))
        self.add_output("vector", val=np.zeros(output_size))
        rows = np.arange(output_size)
        self.declare_partials(
            of="vector",
            wrt="values",
            rows=rows,
            cols=rows,
            val=np.ones(output_size),
        )

    def compute(self, inputs, outputs):
        outputs["vector"] = battery_vector_values(inputs["values"])["vector"]


class BatteryHistoryColumnMatrix(om.ExplicitComponent):
    """Normalize a FAST battery history field to rows by battery columns.

    Inputs:
        values: Fixed-shape vector or matrix battery history field.

    Outputs:
        history_matrix: Values matching
            ``fast_python.battery.as_history_matrix`` for numeric inputs.
    """

    def initialize(self):
        self.options.declare("input_shape", default=(1,))

    def setup(self):
        input_shape = battery_shape_tuple(self.options["input_shape"])
        output_shape = battery_history_column_output_shape(input_shape)
        output_size = int(np.prod(output_shape))
        self.add_input("values", val=np.zeros(input_shape))
        self.add_output("history_matrix", val=np.zeros(output_shape))
        rows = np.arange(output_size)
        self.declare_partials(
            of="history_matrix",
            wrt="values",
            rows=rows,
            cols=rows,
            val=np.ones(output_size),
        )

    def compute(self, inputs, outputs):
        outputs["history_matrix"] = battery_history_column_values(
            inputs["values"],
        )["history_matrix"]


class BatteryHistoryMatrix(om.ExplicitComponent):
    """Normalize mission history as a fixed two-dimensional battery matrix.

    Inputs:
        values: Fixed-shape vector or matrix mission history values.

    Outputs:
        history_matrix: Matrix matching ``fast_python.battery.history_matrix``.

    Assumptions:
        The expected column count is fixed at setup. Dynamic FAST-Python shape
        validation is handled by choosing an OpenMDAO layout before execution.
    """

    def initialize(self):
        self.options.declare("input_shape", default=(1,))
        self.options.declare("columns", default=1)

    def setup(self):
        input_shape = battery_shape_tuple(self.options["input_shape"])
        output_shape = battery_history_matrix_output_shape(
            input_shape,
            self.options["columns"],
        )
        output_size = int(np.prod(output_shape))
        self.add_input("values", val=np.zeros(input_shape))
        self.add_output("history_matrix", val=np.zeros(output_shape))
        rows = np.arange(output_size)
        self.declare_partials(
            of="history_matrix",
            wrt="values",
            rows=rows,
            cols=rows,
            val=np.ones(output_size),
        )

    def compute(self, inputs, outputs):
        outputs["history_matrix"] = battery_history_matrix_values(
            inputs["values"],
            self.options["columns"],
        )["history_matrix"]


class BatteryScalarOrListRestore(om.ExplicitComponent):
    """Restore FAST battery values as scalar or vector with fixed output shape."""

    def initialize(self):
        self.options.declare("value_size", default=1)

    def setup(self):
        value_size = self.options["value_size"]
        output_shape = () if value_size == 1 else (value_size,)
        self.add_input("values", val=np.ones(value_size))
        self.add_output("restored_values", val=np.ones(output_shape))

        if value_size == 1:
            self.declare_partials(
                of="restored_values",
                wrt="values",
                val=np.ones(1),
            )
        else:
            rows = np.arange(value_size)
            self.declare_partials(
                of="restored_values",
                wrt="values",
                rows=rows,
                cols=rows,
                val=np.ones(value_size),
            )

    def compute(self, inputs, outputs):
        values = np.asarray(inputs["values"], dtype=float).reshape(-1)

        if self.options["value_size"] == 1:
            outputs["restored_values"] = values[0]
        else:
            outputs["restored_values"] = values


class BatteryPowerTimeBroadcast(om.ExplicitComponent):
    """Broadcast FAST battery requested-power and time vectors to one length."""

    def initialize(self):
        self.options.declare("power_size", default=1)
        self.options.declare("time_size", default=1)

    def setup(self):
        power_size = self.options["power_size"]
        time_size = self.options["time_size"]
        output_size = battery_power_time_output_size(power_size, time_size)
        self.add_input("requested_power", val=np.zeros(power_size), units="W")
        self.add_input("time", val=np.zeros(time_size), units="s")
        self.add_output("broadcast_power", val=np.zeros(output_size), units="W")
        self.add_output("broadcast_time", val=np.zeros(output_size), units="s")
        self.declare_partials(of="broadcast_power", wrt="requested_power")
        self.declare_partials(of="broadcast_time", wrt="time")

    def compute(self, inputs, outputs):
        values = battery_power_time_broadcast_values(
            inputs["requested_power"],
            inputs["time"],
        )
        outputs["broadcast_power"] = values["broadcast_power"]
        outputs["broadcast_time"] = values["broadcast_time"]

    def compute_partials(self, inputs, partials):
        values = battery_power_time_broadcast_values(
            inputs["requested_power"],
            inputs["time"],
        )
        partials["broadcast_power", "requested_power"] = values[
            "dbroadcast_power_drequested_power"
        ]
        partials["broadcast_time", "time"] = values["dbroadcast_time_dtime"]


class BatteryInitialSOC(om.ExplicitComponent):
    """Normalize FAST battery initial state of charge default/scalar input."""

    def initialize(self):
        self.options.declare("soc_size", default=1)

    def setup(self):
        soc_size = self.options["soc_size"]

        if soc_size not in (0, 1):
            raise ValueError("BatteryInitialSOC requires soc_size 0 or 1.")

        if soc_size == 1:
            self.add_input("soc_begin", val=np.zeros(1))

        self.add_output("initial_soc", val=100.0)

        if soc_size == 1:
            self.declare_partials(of="initial_soc", wrt="soc_begin", val=1.0)

    def compute(self, inputs, outputs):
        if self.options["soc_size"] == 0:
            outputs["initial_soc"] = 100.0
            return

        outputs["initial_soc"] = inputs["soc_begin"][0]


class DetailedBatterySizing(om.ExplicitComponent):
    """Resize detailed battery parallel-cell counts and mass after a mission."""

    def initialize(self):
        self.options.declare("num_points", default=1)
        self.options.declare("num_batteries", default=1)

    def setup(self):
        num_points = self.options["num_points"]
        num_batteries = self.options["num_batteries"]
        history_shape = (num_points, num_batteries)

        self.add_input("soc", val=np.ones(history_shape) * 80.0)
        self.add_input("current", val=np.zeros(history_shape), units="A")
        self.add_input("cap_cell", val=2.4)
        self.add_input("nominal_cell_voltage", val=3.6, units="V")
        self.add_input("min_soc", val=20.0)
        self.add_input("max_c_rate", val=2.0)
        self.add_input("initial_parallel_cells", val=np.ones(num_batteries) * 10.0)
        self.add_input("series_cells", val=np.ones(num_batteries) * 100.0)
        self.add_input("battery_specific_energy", val=1.0, units="J/kg")
        self.add_output("parallel_cells", val=np.ones(num_batteries) * 10.0)
        self.add_output("battery_weight", val=np.ones(num_batteries), units="kg")
        self.add_output("c_rate", val=np.zeros(history_shape))
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = detailed_battery_sizing_values(
            inputs["soc"],
            inputs["current"],
            inputs["cap_cell"][0],
            inputs["nominal_cell_voltage"][0],
            inputs["min_soc"][0],
            inputs["max_c_rate"][0],
            inputs["initial_parallel_cells"],
            inputs["series_cells"],
            inputs["battery_specific_energy"][0],
        )
        outputs["parallel_cells"] = values["parallel_cells"]
        outputs["battery_weight"] = values["battery_weight"]
        outputs["c_rate"] = values["c_rate"]

    def compute_partials(self, inputs, partials):
        values = detailed_battery_sizing_values(
            inputs["soc"],
            inputs["current"],
            inputs["cap_cell"][0],
            inputs["nominal_cell_voltage"][0],
            inputs["min_soc"][0],
            inputs["max_c_rate"][0],
            inputs["initial_parallel_cells"],
            inputs["series_cells"],
            inputs["battery_specific_energy"][0],
        )

        for output in detailed_battery_sizing_output_names():
            for variable in detailed_battery_sizing_input_names():
                partials[output, variable] = values[
                    "d%s_d%s" % (output, variable)
                ]


class BatteryCyclingAging(om.ExplicitComponent):
    """Predict FAST cycling-aging SOH from reduced mission battery statistics."""

    def initialize(self):
        self.options.declare("chemistry", default=1)

    def setup(self):
        self.add_input("depth_of_discharge", val=30.0)
        self.add_input("discharge_c_rate", val=0.5)
        self.add_input("charge_c_rate", val=0.4)
        self.add_input("mean_soc", val=65.0)
        self.add_input("discharge_capacity_delta", val=8.0)
        self.add_input("charge_capacity_delta", val=8.0)
        self.add_input("cap_cell", val=2.4)
        self.add_input("parallel_cells", val=10.0)
        self.add_input("cumulative_fecs", val=100.0)
        self.add_input("operating_temperature", val=25.0, units="degC")
        self.add_output("state_of_health", val=99.0)
        self.add_output("full_equivalent_cycles", val=100.0)
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = battery_cycling_aging_values(
            self.options["chemistry"],
            inputs["depth_of_discharge"][0],
            inputs["discharge_c_rate"][0],
            inputs["charge_c_rate"][0],
            inputs["mean_soc"][0],
            inputs["discharge_capacity_delta"][0],
            inputs["charge_capacity_delta"][0],
            inputs["cap_cell"][0],
            inputs["parallel_cells"][0],
            inputs["cumulative_fecs"][0],
            inputs["operating_temperature"][0],
        )
        outputs["state_of_health"] = values["state_of_health"]
        outputs["full_equivalent_cycles"] = values["full_equivalent_cycles"]

    def compute_partials(self, inputs, partials):
        values = battery_cycling_aging_values(
            self.options["chemistry"],
            inputs["depth_of_discharge"][0],
            inputs["discharge_c_rate"][0],
            inputs["charge_c_rate"][0],
            inputs["mean_soc"][0],
            inputs["discharge_capacity_delta"][0],
            inputs["charge_capacity_delta"][0],
            inputs["cap_cell"][0],
            inputs["parallel_cells"][0],
            inputs["cumulative_fecs"][0],
            inputs["operating_temperature"][0],
        )

        for variable in battery_cycling_aging_input_names():
            partials["state_of_health", variable] = values[
                "dstate_of_health_d%s" % variable
            ]
            partials["full_equivalent_cycles", variable] = values[
                "dfull_equivalent_cycles_d%s" % variable
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


class BatteryPowerHistory(om.ExplicitComponent):
    """Run FAST battery equivalent-circuit dynamics over a fixed interval history."""

    def initialize(self):
        self.options.declare("num_steps", default=1)
        self.options.declare("is_discharge", default=True)
        self.options.declare("drop_initial_soc", default=False)
        self.options.declare("analysis_type", default=0)
        self.options.declare("degradation", default=0)

    def setup(self):
        num_steps = self.options["num_steps"]
        soc_size = num_steps if self.options["drop_initial_soc"] else num_steps + 1

        self.add_input("requested_power", val=np.ones(num_steps), units="W")
        self.add_input("time", val=np.ones(num_steps), units="s")
        self.add_input("soc_begin", val=100.0)
        self.add_input("parallel_cells", val=10.0)
        self.add_input("series_cells", val=100.0)
        self.add_input("max_cell_voltage", val=4.2)
        self.add_input("internal_resistance", val=0.01)
        self.add_input("exponential_voltage", val=0.1)
        self.add_input("exponential_capacity", val=1.0)
        self.add_input("cap_cell", val=2.4)
        self.add_input("state_of_health", val=100.0)
        self.add_output("voltage", val=np.zeros(num_steps), units="V")
        self.add_output("current", val=np.zeros(num_steps), units="A")
        self.add_output("output_power", val=np.zeros(num_steps), units="W")
        self.add_output("capacity", val=np.zeros(num_steps))
        self.add_output("soc", val=np.zeros(soc_size))
        self.add_output("c_rate", val=np.zeros(num_steps))
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = battery_power_history_values(
            inputs["requested_power"],
            inputs["time"],
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
            self.options["drop_initial_soc"],
            self.options["analysis_type"],
            self.options["degradation"],
        )

        for output in battery_power_history_output_names():
            outputs[output] = values[output]

    def compute_partials(self, inputs, partials):
        values = battery_power_history_values(
            inputs["requested_power"],
            inputs["time"],
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
            self.options["drop_initial_soc"],
            self.options["analysis_type"],
            self.options["degradation"],
        )

        for output in battery_power_history_output_names():
            for variable in battery_power_history_input_names():
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


def battery_charge_ocv_values(
    soc_begin,
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
    """Return FAST ground-charge cell OCV estimate and analytical derivatives."""

    values = battery_power_step_values(
        0.0,
        1.0,
        soc_begin,
        parallel_cells,
        series_cells,
        max_cell_voltage,
        internal_resistance,
        exponential_voltage,
        exponential_capacity,
        cap_cell,
        state_of_health,
        False,
        analysis_type,
        degradation,
    )
    voltage = values["voltage"]
    ocv = voltage / series_cells
    result = {"open_circuit_voltage": ocv}

    for variable in battery_charge_ocv_input_names():
        derivative = values["dvoltage_d%s" % variable] / series_cells

        if variable == "series_cells":
            derivative -= voltage / series_cells ** 2

        result["dopen_circuit_voltage_d%s" % variable] = derivative

    return result


def battery_nonzero_mean_values(values):
    """Return FAST nonzero mean and active-mask derivative."""

    array = np.asarray(values, dtype=float).reshape(-1)
    active = array != 0.0
    derivative = np.zeros(array.size)

    if not np.any(active):
        mean = 0.0
    else:
        active_count = np.count_nonzero(active)
        mean = np.mean(array[active])
        derivative[active] = 1.0 / active_count

    return {
        "nonzero_mean": mean,
        "dnonzero_mean_dvalues": derivative,
    }


def battery_shape_tuple(shape):
    """Return an OpenMDAO option shape as a tuple of integers."""

    if isinstance(shape, tuple) and len(shape) == 0:
        return ()

    array = np.asarray(shape).reshape(-1)

    if array.size == 0:
        return (1,)

    return tuple(int(value) for value in array)


def battery_vector_values(values):
    """Return FAST battery values as a one-dimensional vector."""

    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        vector = array.reshape(1)
    else:
        vector = array.reshape(-1)

    return {"vector": vector}


def battery_history_column_output_shape(input_shape):
    """Return output shape for FAST battery ``as_history_matrix``."""

    if len(input_shape) == 1:
        return (input_shape[0], 1)

    return input_shape


def battery_history_column_values(values):
    """Return a FAST battery history field as rows by battery columns."""

    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        matrix = array.reshape(-1, 1)
    else:
        matrix = array

    return {"history_matrix": matrix}


def battery_history_matrix_output_shape(input_shape, columns):
    """Return output shape for FAST battery history-matrix normalization."""

    if len(input_shape) == 1 and columns == 1:
        return (input_shape[0], 1)

    if len(input_shape) == 1:
        return (1, input_shape[0])

    return input_shape


def battery_history_matrix_values(values, columns):
    """Return FAST battery mission history as a two-dimensional matrix."""

    array = np.asarray(values, dtype=float)

    if array.ndim == 1 and columns == 1:
        matrix = array.reshape(-1, 1)
    elif array.ndim == 1:
        matrix = array.reshape(1, -1)
    else:
        matrix = array

    return {"history_matrix": matrix}


def battery_power_time_output_size(power_size, time_size):
    """Return FAST broadcast length for fixed power and time vector sizes."""

    if power_size == 1:
        return time_size

    if time_size == 1:
        return power_size

    if power_size == time_size:
        return power_size

    raise ValueError("Battery power and time sizes must match or broadcast.")


def battery_power_time_broadcast_values(requested_power, time):
    """Return FAST battery power/time broadcast outputs and derivatives."""

    requested_power = np.asarray(requested_power, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    power_size = requested_power.size
    time_size = time.size
    output_size = battery_power_time_output_size(power_size, time_size)

    if power_size == 1 and output_size > 1:
        broadcast_power = np.repeat(requested_power, output_size)
        dpower_dpower = np.ones((output_size, 1))
    else:
        broadcast_power = requested_power.copy()
        dpower_dpower = np.eye(output_size, power_size)

    if time_size == 1 and output_size > 1:
        broadcast_time = np.repeat(time, output_size)
        dtime_dtime = np.ones((output_size, 1))
    else:
        broadcast_time = time.copy()
        dtime_dtime = np.eye(output_size, time_size)

    return {
        "broadcast_power": broadcast_power,
        "broadcast_time": broadcast_time,
        "dbroadcast_power_drequested_power": dpower_dpower,
        "dbroadcast_power_dtime": np.zeros((output_size, time_size)),
        "dbroadcast_time_drequested_power": np.zeros((output_size, power_size)),
        "dbroadcast_time_dtime": dtime_dtime,
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


def battery_charge_ocv_input_names():
    """Return scalar input names for BatteryChargeOCV derivatives."""

    return (
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


def battery_cycling_aging_input_names():
    """Return scalar input names for BatteryCyclingAging derivatives."""

    return (
        "depth_of_discharge",
        "discharge_c_rate",
        "charge_c_rate",
        "mean_soc",
        "discharge_capacity_delta",
        "charge_capacity_delta",
        "cap_cell",
        "parallel_cells",
        "cumulative_fecs",
        "operating_temperature",
    )


def detailed_battery_sizing_input_names():
    """Return input names for DetailedBatterySizing derivatives."""

    return (
        "soc",
        "current",
        "cap_cell",
        "nominal_cell_voltage",
        "min_soc",
        "max_c_rate",
        "initial_parallel_cells",
        "series_cells",
        "battery_specific_energy",
    )


def detailed_battery_sizing_output_names():
    """Return output names for DetailedBatterySizing."""

    return (
        "parallel_cells",
        "battery_weight",
        "c_rate",
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


def battery_power_history_input_names():
    """Return input names for BatteryPowerHistory derivatives."""

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


def battery_power_history_output_names():
    """Return output names for BatteryPowerHistory."""

    return (
        "voltage",
        "current",
        "output_power",
        "capacity",
        "soc",
        "c_rate",
    )


def detailed_battery_sizing_values(
    soc,
    current,
    cap_cell,
    nominal_cell_voltage,
    min_soc,
    max_c_rate,
    initial_parallel_cells,
    series_cells,
    battery_specific_energy,
):
    """Return FAST detailed battery cell sizing and local analytical derivatives."""

    soc = np.asarray(soc, dtype=float)
    current = np.asarray(current, dtype=float)
    initial_parallel = np.asarray(initial_parallel_cells, dtype=float).reshape(-1)
    series = np.asarray(series_cells, dtype=float).reshape(-1)
    num_points, num_batteries = current.shape
    history_size = current.size

    if soc.shape != current.shape:
        raise ValueError("DetailedBatterySizing requires matching SOC and current.")

    if len(initial_parallel) != num_batteries or len(series) != num_batteries:
        raise ValueError("DetailedBatterySizing battery vector sizes must match.")

    delta_soc = np.max(min_soc - soc, axis=0)
    size_to = delta_soc / 100.0
    existing_capacity = cap_cell * initial_parallel
    npar_soc = np.ceil(
        np.ceil((existing_capacity + size_to * cap_cell * initial_parallel) / cap_cell)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        initial_c_rate = current / existing_capacity

    initial_c_rate[~np.isfinite(initial_c_rate)] = 0.0
    exceed_c_rate = np.abs(initial_c_rate) > max_c_rate

    if np.any(exceed_c_rate):
        max_crate = np.max(np.abs(initial_c_rate), axis=0)
        npar_crate = np.ceil(max_crate / max_c_rate) * initial_parallel
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            c_rate_soc = current / (npar_soc * cap_cell)

        exceed_c_rate_soc = np.abs(c_rate_soc) > max_c_rate

        if np.any(exceed_c_rate_soc):
            max_crate_soc = np.max(np.abs(c_rate_soc), axis=0)
            npar_crate = np.ceil(max_crate_soc / max_c_rate) * npar_soc
        else:
            npar_crate = np.zeros(num_batteries)

    parallel = np.maximum(npar_soc, npar_crate)
    battery_weight = (
        cap_cell
        * parallel
        * nominal_cell_voltage
        * series
        * 3600.0
        / battery_specific_energy
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        c_rate = current / (cap_cell * parallel)

    c_rate[~np.isfinite(c_rate)] = 0.0
    zero_battery = np.zeros((num_batteries, num_batteries))
    zero_battery_history = np.zeros((num_batteries, history_size))
    zero_battery_scalar = np.zeros((num_batteries, 1))
    zero_history_battery = np.zeros((history_size, num_batteries))
    zero_history_scalar = np.zeros((history_size, 1))
    dc_rate_dcurrent = np.zeros((history_size, history_size))
    dc_rate_dcap = np.zeros((history_size, 1))

    for point in range(num_points):
        for battery in range(num_batteries):
            row = point * num_batteries + battery
            denominator = cap_cell * parallel[battery]
            if abs(denominator) > 1.0e-14:
                dc_rate_dcurrent[row, row] = 1.0 / denominator
                dc_rate_dcap[row, 0] = -current[point, battery] / (
                    cap_cell ** 2 * parallel[battery]
                )

    dbattery_dcap = (
        parallel * nominal_cell_voltage * series * 3600.0 / battery_specific_energy
    ).reshape(-1, 1)
    dbattery_dvoltage = (
        cap_cell * parallel * series * 3600.0 / battery_specific_energy
    ).reshape(-1, 1)
    dbattery_dseries = np.diag(
        cap_cell * parallel * nominal_cell_voltage * 3600.0 / battery_specific_energy
    )
    dbattery_dspecific_energy = (
        -cap_cell
        * parallel
        * nominal_cell_voltage
        * series
        * 3600.0
        / battery_specific_energy ** 2
    ).reshape(-1, 1)

    result = {
        "parallel_cells": parallel,
        "battery_weight": battery_weight,
        "c_rate": c_rate,
        "dparallel_cells_dsoc": zero_battery_history,
        "dparallel_cells_dcurrent": zero_battery_history,
        "dparallel_cells_dcap_cell": zero_battery_scalar,
        "dparallel_cells_dnominal_cell_voltage": zero_battery_scalar,
        "dparallel_cells_dmin_soc": zero_battery_scalar,
        "dparallel_cells_dmax_c_rate": zero_battery_scalar,
        "dparallel_cells_dinitial_parallel_cells": zero_battery,
        "dparallel_cells_dseries_cells": zero_battery,
        "dparallel_cells_dbattery_specific_energy": zero_battery_scalar,
        "dbattery_weight_dsoc": zero_battery_history,
        "dbattery_weight_dcurrent": zero_battery_history,
        "dbattery_weight_dcap_cell": dbattery_dcap,
        "dbattery_weight_dnominal_cell_voltage": dbattery_dvoltage,
        "dbattery_weight_dmin_soc": zero_battery_scalar,
        "dbattery_weight_dmax_c_rate": zero_battery_scalar,
        "dbattery_weight_dinitial_parallel_cells": zero_battery,
        "dbattery_weight_dseries_cells": dbattery_dseries,
        "dbattery_weight_dbattery_specific_energy": dbattery_dspecific_energy,
        "dc_rate_dsoc": np.zeros((history_size, history_size)),
        "dc_rate_dcurrent": dc_rate_dcurrent,
        "dc_rate_dcap_cell": dc_rate_dcap,
        "dc_rate_dnominal_cell_voltage": zero_history_scalar,
        "dc_rate_dmin_soc": zero_history_scalar,
        "dc_rate_dmax_c_rate": zero_history_scalar,
        "dc_rate_dinitial_parallel_cells": zero_history_battery,
        "dc_rate_dseries_cells": zero_history_battery,
        "dc_rate_dbattery_specific_energy": zero_history_scalar,
    }
    return result


def battery_cycling_aging_values(
    chemistry,
    depth_of_discharge,
    discharge_c_rate,
    charge_c_rate,
    mean_soc,
    discharge_capacity_delta,
    charge_capacity_delta,
    cap_cell,
    parallel_cells,
    cumulative_fecs,
    operating_temperature,
):
    """Return FAST empirical cycling-aging SOH and analytical derivatives."""

    if chemistry not in CYCLING_AGING_PARAMETERS:
        raise ValueError('Invalid chemistry. Use "1" for NMC or "2" for LFP.')

    params = CYCLING_AGING_PARAMETERS[chemistry]
    capacity_sum = discharge_capacity_delta + charge_capacity_delta
    fec = capacity_sum / (2.0 * cap_cell * parallel_cells) + cumulative_fecs

    if fec <= 0.0:
        raise ValueError("BatteryCyclingAging requires positive FEC.")

    temp_actual = operating_temperature + 273.15
    theta_temp = params["coeff_T"] * (
        (temp_actual - params["temp_ref"]) / temp_actual
    )
    theta_dod = params["coeff_DOD"] * depth_of_discharge
    theta_c = params["coeff_Cch"] * charge_c_rate
    theta_c += params["coeff_Cdch"] * discharge_c_rate
    soc_shape = 1.0 + params["coeff_mSOC"] * mean_soc * (
        1.0 - mean_soc / (2.0 * params["mSOC_ref"])
    )
    exp_term = math.exp(theta_temp + theta_dod + theta_c)
    fec_term = fec ** params["alpha"]
    base_degradation = params["beta"] * exp_term * fec_term
    degradation = base_degradation * soc_shape
    state_of_health = 100.0 - degradation

    dfec = {
        "depth_of_discharge": 0.0,
        "discharge_c_rate": 0.0,
        "charge_c_rate": 0.0,
        "mean_soc": 0.0,
        "discharge_capacity_delta": 1.0 / (2.0 * cap_cell * parallel_cells),
        "charge_capacity_delta": 1.0 / (2.0 * cap_cell * parallel_cells),
        "cap_cell": -capacity_sum / (2.0 * cap_cell ** 2 * parallel_cells),
        "parallel_cells": -capacity_sum / (2.0 * cap_cell * parallel_cells ** 2),
        "cumulative_fecs": 1.0,
        "operating_temperature": 0.0,
    }
    ddegradation = {
        "depth_of_discharge": degradation * params["coeff_DOD"],
        "discharge_c_rate": degradation * params["coeff_Cdch"],
        "charge_c_rate": degradation * params["coeff_Cch"],
        "mean_soc": base_degradation
        * params["coeff_mSOC"]
        * (1.0 - mean_soc / params["mSOC_ref"]),
        "discharge_capacity_delta": degradation
        * params["alpha"]
        / fec
        * dfec["discharge_capacity_delta"],
        "charge_capacity_delta": degradation
        * params["alpha"]
        / fec
        * dfec["charge_capacity_delta"],
        "cap_cell": degradation * params["alpha"] / fec * dfec["cap_cell"],
        "parallel_cells": degradation
        * params["alpha"]
        / fec
        * dfec["parallel_cells"],
        "cumulative_fecs": degradation
        * params["alpha"]
        / fec
        * dfec["cumulative_fecs"],
        "operating_temperature": degradation
        * params["coeff_T"]
        * params["temp_ref"]
        / temp_actual ** 2,
    }

    values = {
        "state_of_health": state_of_health,
        "full_equivalent_cycles": fec,
    }

    for variable in battery_cycling_aging_input_names():
        values["dstate_of_health_d%s" % variable] = -ddegradation[variable]
        values["dfull_equivalent_cycles_d%s" % variable] = dfec[variable]

    return values


def battery_power_history_values(
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
    drop_initial_soc,
    analysis_type,
    degradation,
):
    """Return FAST battery dynamics history and dense analytical derivatives."""

    requested_power = np.asarray(requested_power, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    num_steps = len(requested_power)

    if len(time) != num_steps:
        raise ValueError("BatteryPowerHistory requires matching power and time sizes.")

    scalar_inputs = battery_power_history_input_names()[2:]
    vector_inputs = battery_power_history_input_names()[:2]
    voltage = np.zeros(num_steps)
    current = np.zeros(num_steps)
    output_power = np.zeros(num_steps)
    capacity = np.zeros(num_steps)
    c_rate = np.zeros(num_steps)
    soc_full = np.zeros(num_steps + 1)
    soc_full[0] = soc_begin
    derivatives = {}

    for output in ("voltage", "current", "output_power", "capacity", "c_rate"):
        derivatives[output] = {
            "requested_power": np.zeros((num_steps, num_steps)),
            "time": np.zeros((num_steps, num_steps)),
        }

        for variable in scalar_inputs:
            derivatives[output][variable] = np.zeros((num_steps, 1))

    dsoc_current = {
        "requested_power": np.zeros(num_steps),
        "time": np.zeros(num_steps),
    }

    for variable in scalar_inputs:
        dsoc_current[variable] = np.asarray([1.0 if variable == "soc_begin" else 0.0])

    soc_derivatives = {
        "requested_power": np.zeros((num_steps + 1, num_steps)),
        "time": np.zeros((num_steps + 1, num_steps)),
    }
    soc_derivatives["soc_begin"] = np.zeros((num_steps + 1, 1))
    soc_derivatives["soc_begin"][0, 0] = 1.0

    for variable in scalar_inputs:
        if variable == "soc_begin":
            continue

        soc_derivatives[variable] = np.zeros((num_steps + 1, 1))

    for step in range(num_steps):
        values = battery_power_step_values(
            requested_power[step],
            time[step],
            soc_full[step],
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
        )
        voltage[step] = values["voltage"]
        current[step] = values["current"]
        output_power[step] = values["output_power"]
        capacity[step] = values["capacity"]
        c_rate[step] = values["c_rate"]
        soc_full[step + 1] = values["soc_end"]

        for output in ("voltage", "current", "output_power", "capacity", "c_rate"):
            dsoc = values["d%s_dsoc_begin" % output]
            derivatives[output]["requested_power"][step, :] = (
                dsoc * dsoc_current["requested_power"]
            )
            derivatives[output]["requested_power"][step, step] += values[
                "d%s_drequested_power" % output
            ]
            derivatives[output]["time"][step, :] = dsoc * dsoc_current["time"]
            derivatives[output]["time"][step, step] += values["d%s_dtime" % output]

            for variable in scalar_inputs:
                direct = 0.0

                if variable != "soc_begin":
                    direct = values["d%s_d%s" % (output, variable)]

                derivatives[output][variable][step, 0] = (
                    direct + dsoc * dsoc_current[variable][0]
                )

        next_requested = (
            values["dsoc_end_dsoc_begin"] * dsoc_current["requested_power"]
        )
        next_requested[step] += values["dsoc_end_drequested_power"]
        next_time = values["dsoc_end_dsoc_begin"] * dsoc_current["time"]
        next_time[step] += values["dsoc_end_dtime"]
        dsoc_current["requested_power"] = next_requested
        dsoc_current["time"] = next_time
        soc_derivatives["requested_power"][step + 1, :] = next_requested
        soc_derivatives["time"][step + 1, :] = next_time

        for variable in scalar_inputs:
            direct = 0.0

            if variable != "soc_begin":
                direct = values["dsoc_end_d%s" % variable]

            dsoc_current[variable][0] = (
                direct + values["dsoc_end_dsoc_begin"] * dsoc_current[variable][0]
            )
            soc_derivatives[variable][step + 1, 0] = dsoc_current[variable][0]

    if drop_initial_soc:
        soc = soc_full[1:]
        soc_derivatives = {
            variable: value[1:, :]
            for variable, value in soc_derivatives.items()
        }
    else:
        soc = soc_full

    result = {
        "voltage": voltage,
        "current": current,
        "output_power": output_power,
        "capacity": capacity,
        "soc": soc,
        "c_rate": c_rate,
        "dsoc_drequested_power": soc_derivatives["requested_power"],
        "dsoc_dtime": soc_derivatives["time"],
    }

    for output in ("voltage", "current", "output_power", "capacity", "c_rate"):
        for variable in battery_power_history_input_names():
            result["d%s_d%s" % (output, variable)] = derivatives[output][variable]

    for variable in scalar_inputs:
        result["dsoc_d%s" % variable] = soc_derivatives[variable]

    return result


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
