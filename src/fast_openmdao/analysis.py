# src/fast_openmdao/analysis.py

"""OpenMDAO components for FAST analysis helper equations."""

import numpy as np
import openmdao.api as om


class ConvergenceError(om.ExplicitComponent):
    """Compute FAST scalar relative convergence error.

    Inputs:
        delta: Difference between current and previous iteration values.
        baseline: Baseline value used for relative convergence.

    Outputs:
        convergence_error: ``abs(delta) / baseline``.

    Assumptions:
        The analytical derivative is defined away from ``delta == 0`` and
        ``baseline == 0``. FAST-Python treats NaN from invalid divisions as
        zero for array workflows; OpenMDAO optimization should avoid zero
        baselines for differentiable convergence checks.
    """

    def setup(self):
        self.add_input("delta", val=1.0)
        self.add_input("baseline", val=1.0)
        self.add_output("convergence_error", val=1.0)
        self.declare_partials(of="convergence_error", wrt="*")

    def compute(self, inputs, outputs):
        delta = inputs["delta"][0]
        baseline = inputs["baseline"][0]

        if baseline == 0.0:
            outputs["convergence_error"] = 0.0
            return

        outputs["convergence_error"] = abs(delta) / baseline

    def compute_partials(self, inputs, partials):
        delta = inputs["delta"][0]
        baseline = inputs["baseline"][0]

        if delta == 0.0 or baseline == 0.0:
            partials["convergence_error", "delta"] = 0.0
            partials["convergence_error", "baseline"] = 0.0
            return

        sign = 1.0 if delta > 0.0 else -1.0
        partials["convergence_error", "delta"] = sign / baseline
        partials["convergence_error", "baseline"] = -abs(delta) / baseline ** 2


class WeightSum(om.ExplicitComponent):
    """Compute FAST's scalar sum of source-weight values."""

    def initialize(self):
        self.options.declare("vec_size", default=1)

    def setup(self):
        vec_size = self.options["vec_size"]
        self.add_input("weight_values", val=np.ones(vec_size), units="kg")
        self.add_output("weight_sum", val=1.0, units="kg")
        self.declare_partials(
            of="weight_sum",
            wrt="weight_values",
            val=np.ones(vec_size),
        )

    def compute(self, inputs, outputs):
        outputs["weight_sum"] = np.sum(inputs["weight_values"])
