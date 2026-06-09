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


class TurbopropOEWIterationStep(om.ExplicitComponent):
    """Compute one FAST turboprop OEW fixed-point iteration step.

    Inputs:
        *_old weights: Component weights entering the current fixed-point
            balance in kg.
        *_new weights: Propulsion sizing outputs for the current MTOW in kg.
        wing_loading: SLS wing loading in kg/m**2.
        power_loading: SLS installed power-to-weight ratio in W/kg.
        frame_factor: Airframe correction factor.

    Outputs:
        mtow_pre_sizing: MTOW estimate before propulsion resizing in kg.
        sizing_power: Installed SLS power requested before propulsion resizing
            in W.
        wing_area: Wing area implied by the pre-sizing MTOW in m**2.
        mtow: MTOW after replacing old propulsion weights with new weights in
            kg.
        airframe_weight: Updated airframe weight in kg.
        oew: Updated operating empty weight in kg.
        convergence_error: FAST fixed-point relative airframe-weight change.

    Assumptions:
        ``slope`` and ``intercept`` are the fixed linear fit returned by
        ``fast_python.oew.turboprop_airframe_fit`` for the selected database.
        The absolute-value convergence derivative is the active-branch
        derivative away from the exact nondifferentiable equality points.
    """

    def initialize(self):
        self.options.declare("slope", default=0.5)
        self.options.declare("intercept", default=0.0)
        self.options.declare("minimum_error_denominator", default=1.0e-12)

    def setup(self):
        for name in turboprop_oew_iteration_step_input_names():
            self.add_input(name, val=1.0)

        self.add_input("wing_loading", val=100.0, units="kg/m**2")
        self.add_input("power_loading", val=100.0, units="W/kg")
        self.add_input("frame_factor", val=1.0)
        self.add_output("mtow_pre_sizing", val=1.0, units="kg")
        self.add_output("sizing_power", val=1.0, units="W")
        self.add_output("wing_area", val=1.0, units="m**2")
        self.add_output("mtow", val=1.0, units="kg")
        self.add_output("airframe_weight", val=1.0, units="kg")
        self.add_output("oew", val=1.0, units="kg")
        self.add_output("convergence_error", val=1.0)
        self._declared_partials = turboprop_oew_iteration_step_dependencies()

        for output_name, input_names in self._declared_partials.items():
            self.declare_partials(of=output_name, wrt=input_names)

    def compute(self, inputs, outputs):
        values = turboprop_oew_iteration_step_values(
            self.options["slope"],
            self.options["intercept"],
            self.options["minimum_error_denominator"],
            inputs,
        )

        for name in turboprop_oew_iteration_step_output_names():
            outputs[name] = values[name]

    def compute_partials(self, inputs, partials):
        values = turboprop_oew_iteration_step_values(
            self.options["slope"],
            self.options["intercept"],
            self.options["minimum_error_denominator"],
            inputs,
        )

        for output_name, input_names in self._declared_partials.items():
            derivatives = values["partials"][output_name]

            for input_name in input_names:
                derivative = derivatives[input_name]
                partials[output_name, input_name] = derivative


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


def turboprop_oew_iteration_step_input_names():
    """Return scalar input names used by the turboprop OEW step."""

    return [
        "airframe_weight_old",
        "fuel_weight",
        "battery_weight",
        "payload_weight",
        "crew_weight",
        "motor_weight_old",
        "generator_weight_old",
        "engine_weight_old",
        "eap_weight",
        "cable_weight_old",
        "motor_weight_new",
        "generator_weight_new",
        "engine_weight_new",
        "cable_weight_new",
    ]


def turboprop_oew_iteration_step_output_names():
    """Return scalar output names produced by the turboprop OEW step."""

    return [
        "mtow_pre_sizing",
        "sizing_power",
        "wing_area",
        "mtow",
        "airframe_weight",
        "oew",
        "convergence_error",
    ]


def turboprop_oew_iteration_step_dependencies():
    """Return nonzero derivative dependencies for the turboprop OEW step."""

    old_names = [
        "airframe_weight_old",
        "fuel_weight",
        "battery_weight",
        "payload_weight",
        "crew_weight",
        "motor_weight_old",
        "generator_weight_old",
        "engine_weight_old",
        "eap_weight",
        "cable_weight_old",
    ]
    balance_names = [
        "airframe_weight_old",
        "fuel_weight",
        "battery_weight",
        "payload_weight",
        "crew_weight",
        "eap_weight",
        "motor_weight_new",
        "generator_weight_new",
        "engine_weight_new",
        "cable_weight_new",
    ]
    airframe_names = balance_names + ["frame_factor"]

    return {
        "mtow_pre_sizing": old_names,
        "sizing_power": old_names + ["power_loading"],
        "wing_area": old_names + ["wing_loading"],
        "mtow": balance_names,
        "airframe_weight": airframe_names,
        "oew": airframe_names,
        "convergence_error": airframe_names,
    }


def turboprop_oew_iteration_step_values(
    slope,
    intercept,
    minimum_error_denominator,
    inputs,
):
    """Return one turboprop OEW iteration step and exact active derivatives."""

    values = {}

    for name in turboprop_oew_iteration_step_input_names():
        values[name] = float(inputs[name][0])

    wing_loading = float(inputs["wing_loading"][0])
    power_loading = float(inputs["power_loading"][0])
    frame_factor = float(inputs["frame_factor"][0])
    old_names = [
        "airframe_weight_old",
        "fuel_weight",
        "battery_weight",
        "payload_weight",
        "crew_weight",
        "motor_weight_old",
        "generator_weight_old",
        "engine_weight_old",
        "eap_weight",
        "cable_weight_old",
    ]
    mtow_pre_sizing = sum(values[name] for name in old_names)
    mtow = (
        mtow_pre_sizing
        + values["engine_weight_new"]
        - values["engine_weight_old"]
        + values["motor_weight_new"]
        - values["motor_weight_old"]
        + values["generator_weight_new"]
        - values["generator_weight_old"]
        + values["cable_weight_new"]
        - values["cable_weight_old"]
    )
    linear_airframe = slope * mtow + intercept
    airframe_weight = linear_airframe * frame_factor
    oew = (
        airframe_weight
        + values["motor_weight_new"]
        + values["generator_weight_new"]
        + values["engine_weight_new"]
        + values["eap_weight"]
        + values["cable_weight_new"]
    )
    difference = values["airframe_weight_old"] - airframe_weight
    denominator = max(
        abs(values["airframe_weight_old"]),
        minimum_error_denominator,
    )
    convergence_error = abs(difference) / denominator
    sizing_power = mtow_pre_sizing * power_loading
    wing_area = mtow_pre_sizing / wing_loading

    derivatives = turboprop_oew_iteration_step_derivatives(
        slope,
        linear_airframe,
        frame_factor,
        values,
        mtow_pre_sizing,
        wing_loading,
        power_loading,
        airframe_weight,
        difference,
        denominator,
        minimum_error_denominator,
    )

    return {
        "mtow_pre_sizing": mtow_pre_sizing,
        "sizing_power": sizing_power,
        "wing_area": wing_area,
        "mtow": mtow,
        "airframe_weight": airframe_weight,
        "oew": oew,
        "convergence_error": convergence_error,
        "partials": derivatives,
    }


def turboprop_oew_iteration_step_derivatives(
    slope,
    linear_airframe,
    frame_factor,
    values,
    mtow_pre_sizing,
    wing_loading,
    power_loading,
    airframe_weight,
    difference,
    denominator,
    minimum_error_denominator,
):
    """Return analytical derivative maps for one turboprop OEW step."""

    input_names = turboprop_oew_iteration_step_input_names() + [
        "wing_loading",
        "power_loading",
        "frame_factor",
    ]
    dpre = dict.fromkeys(input_names, 0.0)
    dmtow = dict.fromkeys(input_names, 0.0)

    for name in [
        "airframe_weight_old",
        "fuel_weight",
        "battery_weight",
        "payload_weight",
        "crew_weight",
        "motor_weight_old",
        "generator_weight_old",
        "engine_weight_old",
        "eap_weight",
        "cable_weight_old",
    ]:
        dpre[name] = 1.0
        dmtow[name] = 1.0

    for name in [
        "engine_weight_old",
        "motor_weight_old",
        "generator_weight_old",
        "cable_weight_old",
    ]:
        dmtow[name] -= 1.0

    for name in [
        "engine_weight_new",
        "motor_weight_new",
        "generator_weight_new",
        "cable_weight_new",
    ]:
        dmtow[name] = 1.0

    dairframe = {}
    doew = {}
    derror = {}
    sign_difference = np.sign(difference)
    ddenominator = dict.fromkeys(input_names, 0.0)

    if abs(values["airframe_weight_old"]) > minimum_error_denominator:
        ddenominator["airframe_weight_old"] = np.sign(values["airframe_weight_old"])

    for name in input_names:
        dairframe[name] = slope * frame_factor * dmtow[name]

    dairframe["frame_factor"] += linear_airframe

    for name in input_names:
        doew[name] = dairframe[name]

    for name in [
        "motor_weight_new",
        "generator_weight_new",
        "engine_weight_new",
        "cable_weight_new",
        "eap_weight",
    ]:
        doew[name] += 1.0

    for name in input_names:
        ddifference = -dairframe[name]

        if name == "airframe_weight_old":
            ddifference += 1.0

        derror[name] = (
            sign_difference * ddifference / denominator
            - abs(difference) * ddenominator[name] / denominator**2
        )

    partials = {
        "mtow_pre_sizing": dict(dpre),
        "sizing_power": {
            name: dpre[name] * power_loading for name in input_names
        },
        "wing_area": {
            name: dpre[name] / wing_loading for name in input_names
        },
        "mtow": dmtow,
        "airframe_weight": dairframe,
        "oew": doew,
        "convergence_error": derror,
    }
    partials["sizing_power"]["power_loading"] = mtow_pre_sizing
    partials["wing_area"]["wing_loading"] = -mtow_pre_sizing / wing_loading**2

    return partials
