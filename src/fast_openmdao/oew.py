# src/fast_openmdao/oew.py

"""OpenMDAO components for FAST operating empty weight equations."""

import numpy as np
import openmdao.api as om


class TurbopropAirframeWeight(om.ExplicitComponent):
    """Compute turboprop airframe weight from FAST's linear MTOW fit.

    Inputs:
        mtow: Maximum takeoff weight in kg.

    Outputs:
        airframe_weight: Estimated airframe weight in kg.

    Assumptions:
        ``slope`` and ``intercept`` come from
        ``fast_python.oew.turboprop_airframe_fit`` for the selected historical
        database. The coefficients are options because the database itself is
        discrete model data, not a continuous design variable.
    """

    def initialize(self):
        self.options.declare("slope", default=0.5)
        self.options.declare("intercept", default=0.0)

    def setup(self):
        self.add_input("mtow", val=1000.0, units="kg")
        self.add_output("airframe_weight", val=500.0, units="kg")
        self.declare_partials(of="airframe_weight", wrt="mtow")

    def compute(self, inputs, outputs):
        outputs["airframe_weight"] = (
            self.options["slope"] * inputs["mtow"][0]
            + self.options["intercept"]
        )

    def compute_partials(self, inputs, partials):
        partials["airframe_weight", "mtow"] = self.options["slope"]


class NumericSum(om.ExplicitComponent):
    """Compute FAST's scalar sum for OEW numeric values."""

    def initialize(self):
        self.options.declare("vec_size", default=1)

    def setup(self):
        vec_size = self.options["vec_size"]
        self.add_input("values", val=np.ones(vec_size))
        self.add_output("numeric_sum", val=1.0)
        self.declare_partials(
            of="numeric_sum",
            wrt="values",
            val=np.ones(vec_size),
        )

    def compute(self, inputs, outputs):
        outputs["numeric_sum"] = np.sum(inputs["values"])
