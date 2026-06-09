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


class GaussianProcessPrediction(om.ExplicitComponent):
    """Evaluate FAST NLGPR posterior mean and variance for one target row."""

    def initialize(self):
        self.options.declare("data_matrix")
        self.options.declare("hyperparams")
        self.options.declare("inverse_term")

    def setup(self):
        data_matrix = np.asarray(self.options["data_matrix"], dtype=float)
        hyperparams = np.asarray(self.options["hyperparams"], dtype=float)
        inverse_term = np.asarray(self.options["inverse_term"], dtype=float)
        self._data_matrix = data_matrix
        self._hyperparams = hyperparams
        self._inverse_term = inverse_term
        self._ninput = data_matrix.shape[1] - 1

        self.add_input("target", val=np.ones(self._ninput))
        self.add_input("prior", val=1.0)
        self.add_output("posterior_mean", val=1.0)
        self.add_output("posterior_variance", val=1.0)
        rows = np.zeros(self._ninput, dtype=int)
        cols = np.arange(self._ninput)
        self.declare_partials(
            of="posterior_mean",
            wrt="target",
            rows=rows,
            cols=cols,
        )
        self.declare_partials(of="posterior_mean", wrt="prior")
        self.declare_partials(
            of="posterior_variance",
            wrt="target",
            rows=rows,
            cols=cols,
        )

    def compute(self, inputs, outputs):
        values = gaussian_process_prediction_values(
            self._data_matrix,
            self._hyperparams,
            self._inverse_term,
            inputs["target"],
            inputs["prior"][0],
        )
        outputs["posterior_mean"] = values["posterior_mean"]
        outputs["posterior_variance"] = values["posterior_variance"]

    def compute_partials(self, inputs, partials):
        values = gaussian_process_prediction_values(
            self._data_matrix,
            self._hyperparams,
            self._inverse_term,
            inputs["target"],
            inputs["prior"][0],
        )
        partials["posterior_mean", "target"] = values["dposterior_mean_dtarget"]
        partials["posterior_mean", "prior"] = values["dposterior_mean_dprior"]
        partials["posterior_variance", "target"] = values[
            "dposterior_variance_dtarget"
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


def gaussian_process_prediction_values(
    data_matrix,
    hyperparams,
    inverse_term,
    target,
    prior,
):
    """Return FAST NLGPR posterior outputs and derivatives for one target."""

    data_matrix = np.asarray(data_matrix, dtype=float)
    hyperparams = np.asarray(hyperparams, dtype=float)
    inverse_term = np.asarray(inverse_term, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    length_scales = hyperparams[:-1]
    signal_variance = hyperparams[-1]
    inputs = data_matrix[:, :-1]
    centered = data_matrix[:, -1] - prior
    delta = inputs - target
    exponent = -0.3 * np.sum(delta ** 2 / length_scales, axis=1)
    kernel = signal_variance * np.exp(exponent)
    weighted_centered = inverse_term @ centered
    posterior_mean = prior + kernel @ weighted_centered
    posterior_variance = signal_variance - kernel @ inverse_term @ kernel
    dkernel_dtarget = kernel[:, np.newaxis] * (0.6 * delta / length_scales)
    dmean_dtarget = dkernel_dtarget.T @ weighted_centered
    dmean_dprior = 1.0 - kernel @ inverse_term @ np.ones(data_matrix.shape[0])
    dvariance_dtarget = np.zeros(len(target))

    for index in range(len(target)):
        derivative = dkernel_dtarget[:, index]
        dvariance_dtarget[index] = -(
            derivative @ inverse_term @ kernel
            + kernel @ inverse_term @ derivative
        )

    return {
        "posterior_mean": posterior_mean,
        "posterior_variance": posterior_variance,
        "dposterior_mean_dtarget": dmean_dtarget,
        "dposterior_mean_dprior": dmean_dprior,
        "dposterior_variance_dtarget": dvariance_dtarget,
    }
