# src/fast_openmdao/oew.py

"""OpenMDAO components for FAST operating empty weight equations."""

import numpy as np
import openmdao.api as om

from fast_openmdao.regression import gaussian_process_prediction_values


class TurbofanAirframeWeight(om.ExplicitComponent):
    """Compute turbofan airframe weight from FAST's preprocessed GPR."""

    def initialize(self):
        self.options.declare("data_matrix")
        self.options.declare("hyperparams")
        self.options.declare("inverse_term")
        self.options.declare("prior", default=1.0)

    def setup(self):
        self._data_matrix = np.asarray(self.options["data_matrix"], dtype=float)
        self._hyperparams = np.asarray(self.options["hyperparams"], dtype=float)
        self._inverse_term = np.asarray(self.options["inverse_term"], dtype=float)
        self.add_input("wing_area", val=100.0, units="m**2")
        self.add_input("thrust", val=100000.0, units="N")
        self.add_input("eis", val=2035.0)
        self.add_input("mtow", val=100000.0, units="kg")
        self.add_input("frame_factor", val=1.0)
        self.add_output("airframe_weight", val=50000.0, units="kg")
        self.declare_partials(of="airframe_weight", wrt="*")

    def compute(self, inputs, outputs):
        outputs["airframe_weight"] = turbofan_airframe_weight_values(
            self._data_matrix,
            self._hyperparams,
            self._inverse_term,
            self.options["prior"],
            inputs["wing_area"][0],
            inputs["thrust"][0],
            inputs["eis"][0],
            inputs["mtow"][0],
            inputs["frame_factor"][0],
        )["airframe_weight"]

    def compute_partials(self, inputs, partials):
        values = turbofan_airframe_weight_values(
            self._data_matrix,
            self._hyperparams,
            self._inverse_term,
            self.options["prior"],
            inputs["wing_area"][0],
            inputs["thrust"][0],
            inputs["eis"][0],
            inputs["mtow"][0],
            inputs["frame_factor"][0],
        )
        partials["airframe_weight", "wing_area"] = values["dweight_dwing_area"]
        partials["airframe_weight", "thrust"] = values["dweight_dthrust"]
        partials["airframe_weight", "eis"] = values["dweight_deis"]
        partials["airframe_weight", "mtow"] = values["dweight_dmtow"]
        partials["airframe_weight", "frame_factor"] = values[
            "dweight_dframe_factor"
        ]


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


def turbofan_airframe_weight_values(
    data_matrix,
    hyperparams,
    inverse_term,
    prior,
    wing_area,
    thrust,
    eis,
    mtow,
    frame_factor,
):
    """Return FAST turbofan airframe-weight GPR value and derivatives."""

    target = np.asarray([wing_area, thrust, eis, mtow])
    prediction = gaussian_process_prediction_values(
        data_matrix,
        hyperparams,
        inverse_term,
        target,
        prior,
    )
    mean = prediction["posterior_mean"]
    dmean_dtarget = prediction["dposterior_mean_dtarget"]
    return {
        "airframe_weight": mean * frame_factor,
        "dweight_dwing_area": dmean_dtarget[0] * frame_factor,
        "dweight_dthrust": dmean_dtarget[1] * frame_factor,
        "dweight_deis": dmean_dtarget[2] * frame_factor,
        "dweight_dmtow": dmean_dtarget[3] * frame_factor,
        "dweight_dframe_factor": mean,
    }
