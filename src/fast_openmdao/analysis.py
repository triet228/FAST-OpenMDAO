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


class SourceWeightVector(om.ExplicitComponent):
    """Expand FAST source-weight input into a fixed OpenMDAO vector.

    Inputs:
        source_weight: Scalar source weight or one value per active source.

    Outputs:
        source_weight_vector: One weight value per active source.

    Assumptions:
        ``source_count`` comes from FAST-Python's source-type mask and is fixed
        during model setup. A single scalar weight is broadcast across active
        sources, matching FAST's multi-source initialization convention.
    """

    def initialize(self):
        self.options.declare("source_count", default=1)
        self.options.declare("input_size", default=1)

    def setup(self):
        source_count = self.options["source_count"]
        input_size = self.options["input_size"]

        if input_size not in (1, source_count):
            raise ValueError("input_size must be 1 or source_count.")

        self.add_input("source_weight", val=np.ones(input_size), units="kg")
        self.add_output("source_weight_vector", val=np.ones(source_count), units="kg")

        rows = np.arange(source_count)
        if input_size == 1:
            cols = np.zeros(source_count, dtype=int)
        else:
            cols = np.arange(source_count)

        self.declare_partials(
            of="source_weight_vector",
            wrt="source_weight",
            rows=rows,
            cols=cols,
        )

    def compute(self, inputs, outputs):
        values = np.asarray(inputs["source_weight"], dtype=float).reshape(-1)
        source_count = self.options["source_count"]

        if np.any(np.isnan(values)):
            outputs["source_weight_vector"] = np.zeros(source_count)
        elif values.size == 1 and source_count > 1:
            outputs["source_weight_vector"] = np.ones(source_count) * values[0]
        else:
            outputs["source_weight_vector"] = values

    def compute_partials(self, inputs, partials):
        values = np.asarray(inputs["source_weight"], dtype=float).reshape(-1)
        source_count = self.options["source_count"]

        if np.any(np.isnan(values)):
            partials["source_weight_vector", "source_weight"] = np.zeros(source_count)
        else:
            partials["source_weight_vector", "source_weight"] = np.ones(source_count)
