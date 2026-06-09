# src/fast_openmdao/units.py

"""OpenMDAO components for FAST unit conversions."""

import numpy as np
import openmdao.api as om

from fast_python.units import (
    FORCE_FACTORS,
    LENGTH_FACTORS,
    MASS_FACTORS,
    TSFC_FACTORS,
    VELOCITY_FACTORS,
    convert_temperature,
)


UNIT_TABLES = {
    "mass": MASS_FACTORS,
    "length": LENGTH_FACTORS,
    "velocity": VELOCITY_FACTORS,
    "force": FORCE_FACTORS,
    "tsfc": TSFC_FACTORS,
}


class UnitConversion(om.ExplicitComponent):
    """Convert one scalar FAST value between fixed units.

    Inputs:
        value: Scalar value in the source unit.

    Outputs:
        converted_value: Scalar value in the destination unit.

    Assumptions:
        Unit labels are discrete options. The component differentiates only the
        numeric value, with affine temperature conversions handled exactly.
    """

    def initialize(self):
        self.options.declare("quantity", default="length")
        self.options.declare("oldunit", default="m")
        self.options.declare("newunit", default="ft")
        self.options.declare("input_name", default="value")
        self.options.declare("output_name", default="converted_value")

    def setup(self):
        self.add_input(self.options["input_name"], val=1.0)
        self.add_output(self.options["output_name"], val=1.0)
        self.declare_partials(
            of=self.options["output_name"],
            wrt=self.options["input_name"],
        )

    def compute(self, inputs, outputs):
        input_name = self.options["input_name"]
        output_name = self.options["output_name"]
        outputs[output_name] = convert_scalar(
            inputs[input_name][0],
            self.options["quantity"],
            self.options["oldunit"],
            self.options["newunit"],
        )

    def compute_partials(self, inputs, partials):
        partials[self.options["output_name"], self.options["input_name"]] = (
            conversion_derivative(
                self.options["quantity"],
                self.options["oldunit"],
                self.options["newunit"],
            )
        )


class UnitArrayConversion(om.ExplicitComponent):
    """Convert a fixed-shape FAST value array between fixed units.

    Inputs:
        values: Scalar, vector, or matrix values in the source unit.

    Outputs:
        converted_values: Converted values with the same fixed OpenMDAO shape.

    Assumptions:
        Unit labels and input shape are fixed during setup. FAST-Python also
        preserves nested list/tuple shape; in OpenMDAO that shape is represented
        by the component's fixed array shape.
    """

    def initialize(self):
        self.options.declare("quantity", default="length")
        self.options.declare("oldunit", default="m")
        self.options.declare("newunit", default="ft")
        self.options.declare("input_shape", default=(1,))
        self.options.declare("input_name", default="values")
        self.options.declare("output_name", default="converted_values")

    def setup(self):
        input_shape = unit_shape_tuple(self.options["input_shape"])
        size = int(np.prod(input_shape))
        self.add_input(self.options["input_name"], val=np.ones(input_shape))
        self.add_output(self.options["output_name"], val=np.ones(input_shape))
        rows = np.arange(size)
        self.declare_partials(
            of=self.options["output_name"],
            wrt=self.options["input_name"],
            rows=rows,
            cols=rows,
            val=np.ones(size)
            * conversion_derivative(
                self.options["quantity"],
                self.options["oldunit"],
                self.options["newunit"],
            ),
        )

    def compute(self, inputs, outputs):
        input_name = self.options["input_name"]
        output_name = self.options["output_name"]
        outputs[output_name] = convert_array(
            inputs[input_name],
            self.options["quantity"],
            self.options["oldunit"],
            self.options["newunit"],
        )


def convert_scalar(value, quantity, oldunit, newunit):
    """Return one scalar converted with FAST unit rules."""

    if quantity == "temperature":
        return convert_temperature(value, oldunit, newunit)

    factors = UNIT_TABLES[quantity]
    return value * factors[oldunit][newunit]


def convert_array(values, quantity, oldunit, newunit):
    """Return fixed-shape values converted with FAST unit rules."""

    values = np.asarray(values, dtype=float)
    slope = conversion_derivative(quantity, oldunit, newunit)

    if quantity == "temperature":
        offset = convert_scalar(0.0, quantity, oldunit, newunit)
        return values * slope + offset

    return values * slope


def conversion_derivative(quantity, oldunit, newunit):
    """Return derivative of converted value with respect to input value."""

    if quantity == "temperature":
        return temperature_conversion_slope(oldunit, newunit)

    factors = UNIT_TABLES[quantity]
    return factors[oldunit][newunit]


def temperature_conversion_slope(oldunit, newunit):
    """Return slope for FAST affine temperature conversion."""

    return kelvin_slope(oldunit) * from_kelvin_slope(newunit)


def kelvin_slope(unit):
    """Return derivative of Kelvin temperature with respect to source unit."""

    if unit in ("K", "C"):
        return 1.0

    if unit in ("R", "F"):
        return 1.0 / 1.8

    convert_temperature(0.0, unit, "K")
    return 0.0


def from_kelvin_slope(unit):
    """Return derivative of requested temperature with respect to Kelvin."""

    if unit in ("K", "C"):
        return 1.0

    if unit in ("R", "F"):
        return 1.8

    convert_temperature(0.0, "K", unit)
    return 0.0


def unit_shape_tuple(shape):
    """Return an OpenMDAO option shape as a tuple of integers."""

    if isinstance(shape, tuple) and len(shape) == 0:
        return ()

    array = np.asarray(shape).reshape(-1)

    if array.size == 0:
        return (1,)

    return tuple(int(value) for value in array)
