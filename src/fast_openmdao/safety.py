# src/fast_openmdao/safety.py

"""OpenMDAO components for continuous FAST safety equations."""

import numpy as np
import openmdao.api as om


class FailureModel(om.ExplicitComponent):
    """Compute FAST component failure probability for fixed exposure data.

    Inputs:
        base_rate: Component failure rates.
        exposure: Exposure time. Use scalar exposure for a shared duration or
            vector exposure for one duration per failure rate.

    Outputs:
        failure_probability: Probability ``1 - exp(-base_rate * exposure)``.
    """

    def initialize(self):
        self.options.declare("vec_size", default=1)
        self.options.declare("exposure_size", default=1)

    def setup(self):
        vec_size = self.options["vec_size"]
        exposure_size = self.options["exposure_size"]

        if exposure_size not in (1, vec_size):
            raise ValueError("FailureModel exposure_size must be 1 or vec_size.")

        self.add_input("base_rate", val=np.ones(vec_size))
        self.add_input("exposure", val=np.ones(exposure_size))
        self.add_output("failure_probability", val=np.zeros(vec_size))

        rows = np.arange(vec_size)
        self.declare_partials(
            of="failure_probability",
            wrt="base_rate",
            rows=rows,
            cols=rows,
        )

        if exposure_size == 1:
            self.declare_partials(
                of="failure_probability",
                wrt="exposure",
                rows=rows,
                cols=np.zeros(vec_size, dtype=int),
            )
        else:
            self.declare_partials(
                of="failure_probability",
                wrt="exposure",
                rows=rows,
                cols=rows,
            )

    def compute(self, inputs, outputs):
        outputs["failure_probability"] = failure_model_values(
            inputs["base_rate"],
            inputs["exposure"],
        )["failure_probability"]

    def compute_partials(self, inputs, partials):
        values = failure_model_values(inputs["base_rate"], inputs["exposure"])
        partials["failure_probability", "base_rate"] = values[
            "dfailure_probability_dbase_rate"
        ]
        partials["failure_probability", "exposure"] = values[
            "dfailure_probability_dexposure"
        ]


def failure_model_values(base_rate, exposure):
    """Return FAST failure probability and analytical derivatives."""

    base_rate = np.asarray(base_rate, dtype=float).reshape(-1)
    exposure = np.asarray(exposure, dtype=float).reshape(-1)

    if exposure.size == 1:
        exposure_values = np.ones(base_rate.size) * exposure[0]
    else:
        exposure_values = exposure

    exponential = np.exp(-base_rate * exposure_values)
    probability = 1.0 - exponential
    return {
        "failure_probability": probability,
        "dfailure_probability_dbase_rate": exposure_values * exponential,
        "dfailure_probability_dexposure": base_rate * exponential,
    }
