# src/fast_openmdao/regression.py

"""OpenMDAO components for FAST regression equations."""

import numpy as np
import openmdao.api as om


class SquaredExponentialKernel(om.ExplicitComponent):
    """Evaluate FAST's squared-exponential covariance kernel.

    Inputs:
        x: Target input vector.
        y: Database input vector.
        length_scales: FAST hyperparameter entries for each input dimension.
        signal_variance: FAST output hyperparameter entry.

    Outputs:
        covariance: Scalar covariance value.

    Assumptions:
        This component represents one FAST kernel row. Vectorized Gaussian
        process components can be assembled from this primitive once the
        regression matrix layout is promoted into OpenMDAO groups.
    """

    def initialize(self):
        self.options.declare("size", default=1)

    def setup(self):
        size = self.options["size"]
        self.add_input("x", val=np.ones(size))
        self.add_input("y", val=np.ones(size))
        self.add_input("length_scales", val=np.ones(size))
        self.add_input("signal_variance", val=1.0)
        self.add_output("covariance", val=1.0)
        self.declare_partials(
            of="covariance",
            wrt="x",
            rows=np.zeros(size, dtype=int),
            cols=np.arange(size),
        )
        self.declare_partials(
            of="covariance",
            wrt="y",
            rows=np.zeros(size, dtype=int),
            cols=np.arange(size),
        )
        self.declare_partials(
            of="covariance",
            wrt="length_scales",
            rows=np.zeros(size, dtype=int),
            cols=np.arange(size),
        )
        self.declare_partials(of="covariance", wrt="signal_variance")

    def compute(self, inputs, outputs):
        outputs["covariance"] = squared_exponential_kernel_value(
            inputs["x"],
            inputs["y"],
            inputs["length_scales"],
            inputs["signal_variance"][0],
        )

    def compute_partials(self, inputs, partials):
        values = squared_exponential_kernel_derivatives(
            inputs["x"],
            inputs["y"],
            inputs["length_scales"],
            inputs["signal_variance"][0],
        )
        partials["covariance", "x"] = values["dcovariance_dx"]
        partials["covariance", "y"] = values["dcovariance_dy"]
        partials["covariance", "length_scales"] = values[
            "dcovariance_dlength_scales"
        ]
        partials["covariance", "signal_variance"] = values[
            "dcovariance_dsignal_variance"
        ]


def squared_exponential_kernel_value(x_value, y_value, length_scales, signal_variance):
    """Return FAST squared-exponential kernel value."""

    delta = np.asarray(x_value, dtype=float) - np.asarray(y_value, dtype=float)
    length_scales = np.asarray(length_scales, dtype=float)
    exponent = -0.3 * np.sum(delta ** 2 / length_scales)
    return signal_variance * np.exp(exponent)


def squared_exponential_kernel_derivatives(
    x_value,
    y_value,
    length_scales,
    signal_variance,
):
    """Return analytical derivatives of FAST squared-exponential kernel."""

    x_value = np.asarray(x_value, dtype=float)
    y_value = np.asarray(y_value, dtype=float)
    length_scales = np.asarray(length_scales, dtype=float)
    delta = x_value - y_value
    covariance = squared_exponential_kernel_value(
        x_value,
        y_value,
        length_scales,
        signal_variance,
    )

    return {
        "dcovariance_dx": covariance * (-0.6 * delta / length_scales),
        "dcovariance_dy": covariance * (0.6 * delta / length_scales),
        "dcovariance_dlength_scales": covariance * (
            0.3 * delta ** 2 / length_scales ** 2
        ),
        "dcovariance_dsignal_variance": covariance / signal_variance,
    }
