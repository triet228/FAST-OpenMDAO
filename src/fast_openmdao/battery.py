# src/fast_openmdao/battery.py

"""OpenMDAO components for FAST battery primitive equations."""

import math

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
