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


class RegressionInverseTerm(om.ExplicitComponent):
    """Build FAST regression's noise-augmented inverse covariance matrix."""

    def initialize(self):
        self.options.declare("data_matrix")
        self.options.declare("prior_size", default=1)

    def setup(self):
        data_matrix = np.asarray(self.options["data_matrix"], dtype=float)
        prior_size = self.options["prior_size"]
        row_count = data_matrix.shape[0]
        hyper_size = data_matrix.shape[1]
        output_size = row_count * row_count
        self._data_matrix = data_matrix
        self.add_input("hyperparams", val=np.ones(hyper_size))
        self.add_input("prior", val=np.ones(prior_size))
        self.add_output("inverse_term", val=np.zeros((row_count, row_count)))
        self.declare_partials(
            of="inverse_term",
            wrt="hyperparams",
            rows=np.repeat(np.arange(output_size), hyper_size),
            cols=np.tile(np.arange(hyper_size), output_size),
        )
        self.declare_partials(
            of="inverse_term",
            wrt="prior",
            rows=np.repeat(np.arange(output_size), prior_size),
            cols=np.tile(np.arange(prior_size), output_size),
        )

    def compute(self, inputs, outputs):
        outputs["inverse_term"] = regression_inverse_term_values(
            self._data_matrix,
            inputs["hyperparams"],
            inputs["prior"],
        )["inverse_term"]

    def compute_partials(self, inputs, partials):
        values = regression_inverse_term_values(
            self._data_matrix,
            inputs["hyperparams"],
            inputs["prior"],
        )
        partials["inverse_term", "hyperparams"] = values[
            "dinverse_term_dhyperparams"
        ]
        partials["inverse_term", "prior"] = values["dinverse_term_dprior"]


class RegressionVector(om.ExplicitComponent):
    """Normalize FAST regression values to a one-dimensional vector."""

    def initialize(self):
        self.options.declare("input_shape", default=(1,))

    def setup(self):
        input_shape = regression_shape_tuple(self.options["input_shape"])
        output_size = int(np.prod(input_shape))
        self.add_input("values", val=np.zeros(input_shape))
        self.add_output("vector", val=np.zeros(output_size))
        rows = np.arange(output_size)
        self.declare_partials(
            of="vector",
            wrt="values",
            rows=rows,
            cols=rows,
            val=np.ones(output_size),
        )

    def compute(self, inputs, outputs):
        outputs["vector"] = regression_vector_values(inputs["values"])["vector"]


class RegressionTwoDimensionalArray(om.ExplicitComponent):
    """Normalize FAST regression values to a two-dimensional array."""

    def initialize(self):
        self.options.declare("input_shape", default=(1,))

    def setup(self):
        input_shape = regression_shape_tuple(self.options["input_shape"])
        output_shape = regression_two_dimensional_output_shape(input_shape)
        output_size = int(np.prod(output_shape))
        self.add_input("values", val=np.zeros(input_shape))
        self.add_output("two_dimensional_values", val=np.zeros(output_shape))
        rows = np.arange(output_size)
        self.declare_partials(
            of="two_dimensional_values",
            wrt="values",
            rows=rows,
            cols=rows,
            val=np.ones(output_size),
        )

    def compute(self, inputs, outputs):
        outputs["two_dimensional_values"] = regression_two_dimensional_values(
            inputs["values"],
        )["two_dimensional_values"]


class RegressionTargetMatrix(om.ExplicitComponent):
    """Normalize FAST NLGPR target values to a two-dimensional matrix."""

    def initialize(self):
        self.options.declare("input_shape", default=(1,))

    def setup(self):
        input_shape = regression_shape_tuple(self.options["input_shape"])
        output_shape = regression_target_matrix_output_shape(input_shape)
        output_size = int(np.prod(output_shape))
        self.add_input("target_values", val=np.zeros(input_shape))
        self.add_output("target_matrix", val=np.zeros(output_shape))
        rows = np.arange(output_size)
        self.declare_partials(
            of="target_matrix",
            wrt="target_values",
            rows=rows,
            cols=rows,
            val=np.ones(output_size),
        )

    def compute(self, inputs, outputs):
        outputs["target_matrix"] = regression_target_matrix_values(
            inputs["target_values"],
        )["target_matrix"]


class RegressionSampleVariance(om.ExplicitComponent):
    """Compute FAST regression sample variance for a fixed numeric vector."""

    def initialize(self):
        self.options.declare("vec_size", default=2)

    def setup(self):
        vec_size = self.options["vec_size"]
        self.add_input("values", val=np.ones(vec_size))
        self.add_output("sample_variance", val=0.0)
        self.declare_partials(of="sample_variance", wrt="values")

    def compute(self, inputs, outputs):
        outputs["sample_variance"] = regression_sample_variance_values(
            inputs["values"],
        )["sample_variance"]

    def compute_partials(self, inputs, partials):
        values = regression_sample_variance_values(inputs["values"])
        partials["sample_variance", "values"] = values["dsample_variance_dvalues"]


class RegressionWeightedHyperparameters(om.ExplicitComponent):
    """Scale FAST regression variance hyperparameters by relevance weights."""

    def initialize(self):
        self.options.declare("num_inputs", default=1)

    def setup(self):
        num_inputs = self.options["num_inputs"]
        hyper_size = num_inputs + 1
        self.add_input("variances", val=np.ones(hyper_size))
        self.add_input("weights", val=np.ones(num_inputs))
        self.add_output("hyperparams", val=np.ones(hyper_size))
        self.declare_partials(
            of="hyperparams",
            wrt="variances",
            rows=np.arange(hyper_size),
            cols=np.arange(hyper_size),
        )
        self.declare_partials(
            of="hyperparams",
            wrt="weights",
            rows=np.repeat(np.arange(num_inputs), num_inputs),
            cols=np.tile(np.arange(num_inputs), num_inputs),
        )

    def compute(self, inputs, outputs):
        outputs["hyperparams"] = regression_weighted_hyperparameter_values(
            inputs["variances"],
            inputs["weights"],
        )["hyperparams"]

    def compute_partials(self, inputs, partials):
        values = regression_weighted_hyperparameter_values(
            inputs["variances"],
            inputs["weights"],
        )
        partials["hyperparams", "variances"] = values["dhyperparams_dvariances"]
        partials["hyperparams", "weights"] = values["dhyperparams_dweights"]


class RegressionPriorMean(om.ExplicitComponent):
    """Compute FAST regression prior mean from fixed numeric output data.

    Inputs:
        values: Output column after FAST database path lookup and numeric
            conversion. NaN entries represent missing database values.

    Outputs:
        prior_mean: Mean of the finite entries, or NaN when no finite data are
            available.
    """

    def initialize(self):
        self.options.declare("vec_size", default=1)

    def setup(self):
        vec_size = self.options["vec_size"]
        self.add_input("values", val=np.ones(vec_size))
        self.add_output("prior_mean", val=0.0)
        self.declare_partials(of="prior_mean", wrt="values")

    def compute(self, inputs, outputs):
        outputs["prior_mean"] = regression_prior_mean_values(
            inputs["values"],
        )["prior_mean"]

    def compute_partials(self, inputs, partials):
        values = regression_prior_mean_values(inputs["values"])
        partials["prior_mean", "values"] = values["dprior_mean_dvalues"]


class RegressionNumericScalar(om.ExplicitComponent):
    """Select FAST regression's first numeric scalar from fixed-shape data."""

    def initialize(self):
        self.options.declare("input_shape", default=(1,))

    def setup(self):
        input_shape = regression_shape_tuple(self.options["input_shape"])
        input_size = int(np.prod(input_shape))
        self.add_input("values", val=np.zeros(input_shape))
        self.add_output("numeric_scalar", val=0.0)
        self.declare_partials(
            of="numeric_scalar",
            wrt="values",
            rows=np.zeros(1, dtype=int),
            cols=np.zeros(1, dtype=int),
            val=np.ones(1),
        )

        if input_size < 1:
            raise ValueError("RegressionNumericScalar requires nonempty input_shape.")

    def compute(self, inputs, outputs):
        outputs["numeric_scalar"] = regression_numeric_scalar_values(
            inputs["values"],
        )["numeric_scalar"]


class RegressionNumericColumn(om.ExplicitComponent):
    """Select FAST regression numeric scalars from fixed-shape rows."""

    def initialize(self):
        self.options.declare("input_shape", default=(1, 1))

    def setup(self):
        input_shape = regression_shape_tuple(self.options["input_shape"])
        row_count = regression_numeric_column_row_count(input_shape)
        input_size = int(np.prod(input_shape))
        self.add_input("values", val=np.zeros(input_shape))
        self.add_output("numeric_column", val=np.zeros(row_count))
        rows = np.arange(row_count)

        if len(input_shape) <= 1:
            cols = rows
        else:
            cols = np.arange(row_count) * input_shape[1]

        self.declare_partials(
            of="numeric_column",
            wrt="values",
            rows=rows,
            cols=cols,
            val=np.ones(row_count),
        )

        if input_size < 1:
            raise ValueError("RegressionNumericColumn requires nonempty input_shape.")

    def compute(self, inputs, outputs):
        outputs["numeric_column"] = regression_numeric_column_values(
            inputs["values"],
        )["numeric_column"]


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


def regression_inverse_term_values(data_matrix, hyperparams, prior):
    """Return FAST reg_processing inverse covariance and derivatives."""

    data_matrix = np.asarray(data_matrix, dtype=float)
    hyperparams = np.asarray(hyperparams, dtype=float).reshape(-1)
    prior = np.asarray(prior, dtype=float).reshape(-1)
    inputs = data_matrix[:, :-1]
    length_scales = hyperparams[:-1]
    signal_variance = hyperparams[-1]
    row_count = data_matrix.shape[0]
    hyper_size = hyperparams.size
    kernel = np.zeros((row_count, row_count))
    dkernel_dhyperparams = np.zeros((hyper_size, row_count, row_count))

    for irow in range(row_count):
        for jrow in range(row_count):
            delta = inputs[irow, :] - inputs[jrow, :]
            exponent = -0.3 * np.sum(delta ** 2 / length_scales)
            kernel_value = signal_variance * np.exp(exponent)
            kernel[irow, jrow] = kernel_value
            dkernel_dhyperparams[:-1, irow, jrow] = (
                kernel_value * 0.3 * delta ** 2 / length_scales ** 2
            )
            dkernel_dhyperparams[-1, irow, jrow] = kernel_value / signal_variance

    prior_mean = np.mean(prior)
    noise_variance = (prior_mean * 5.0e-2) ** 2
    covariance = kernel + noise_variance * np.eye(row_count)
    inverse_term = np.linalg.inv(covariance)
    dinverse_dhyperparams = np.zeros((row_count * row_count, hyper_size))

    for index in range(hyper_size):
        derivative = -inverse_term @ dkernel_dhyperparams[index] @ inverse_term
        dinverse_dhyperparams[:, index] = derivative.reshape(-1)

    dnoise_dprior = 2.0 * 5.0e-2 ** 2 * prior_mean / prior.size
    prior_derivative_matrix = -dnoise_dprior * inverse_term @ inverse_term
    dinverse_dprior = np.tile(prior_derivative_matrix.reshape(-1, 1), prior.size)
    return {
        "inverse_term": inverse_term,
        "dinverse_term_dhyperparams": dinverse_dhyperparams.reshape(-1),
        "dinverse_term_dprior": dinverse_dprior.reshape(-1),
    }


def regression_shape_tuple(shape):
    """Return an OpenMDAO option shape as a tuple of integers."""

    if isinstance(shape, tuple) and len(shape) == 0:
        return ()

    array = np.asarray(shape).reshape(-1)

    if array.size == 0:
        return (1,)

    return tuple(int(value) for value in array)


def regression_vector_values(values):
    """Return FAST regression values as a one-dimensional vector."""

    return {"vector": np.asarray(values, dtype=float).reshape(-1)}


def regression_two_dimensional_output_shape(input_shape):
    """Return output shape for FAST regression ``as_2d``."""

    if len(input_shape) == 1:
        return (1, input_shape[0])

    return input_shape


def regression_two_dimensional_values(values):
    """Return FAST regression values as a two-dimensional array."""

    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        output = array.reshape(1, -1)
    else:
        output = array

    return {"two_dimensional_values": output}


def regression_target_matrix_output_shape(input_shape):
    """Return output shape for FAST regression ``target_matrix``."""

    if len(input_shape) == 0:
        return (1, 1)

    if len(input_shape) == 1:
        return (1, input_shape[0])

    return input_shape


def regression_target_matrix_values(target_values):
    """Return FAST regression target values as a two-dimensional matrix."""

    array = np.asarray(target_values, dtype=float)

    if array.ndim == 0:
        matrix = array.reshape(1, 1)
    elif array.ndim == 1:
        matrix = array.reshape(1, -1)
    else:
        matrix = array

    return {"target_matrix": matrix}


def regression_sample_variance_values(values):
    """Return MATLAB-style sample variance and analytical derivatives."""

    array = np.asarray(values, dtype=float).reshape(-1)
    size = array.size

    if size < 2:
        return {
            "sample_variance": 0.0,
            "dsample_variance_dvalues": np.zeros(size),
        }

    mean = np.mean(array)
    variance = np.sum((array - mean) ** 2) / (size - 1)
    derivative = 2.0 * (array - mean) / (size - 1)
    return {
        "sample_variance": variance,
        "dsample_variance_dvalues": derivative,
    }


def regression_weighted_hyperparameter_values(variances, weights):
    """Return FAST build_data hyperparameters and analytical derivatives."""

    variances = np.asarray(variances, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    num_inputs = weights.size
    weight_sum = np.sum(weights)
    hyperparams = np.zeros(num_inputs + 1)
    dhyperparams_dvariances = np.zeros(num_inputs + 1)
    dhyperparams_dweights = np.zeros(num_inputs * num_inputs)

    scale = weight_sum / (num_inputs * weights)
    hyperparams[:-1] = variances[:-1] * scale
    hyperparams[-1] = variances[-1]
    dhyperparams_dvariances[:-1] = scale
    dhyperparams_dvariances[-1] = 1.0

    dense_weight_partials = np.zeros((num_inputs, num_inputs))

    for irow in range(num_inputs):
        for jcol in range(num_inputs):
            if irow == jcol:
                dense_weight_partials[irow, jcol] = (
                    variances[irow]
                    * (weights[irow] - weight_sum)
                    / (num_inputs * weights[irow] ** 2)
                )
            else:
                dense_weight_partials[irow, jcol] = (
                    variances[irow] / (num_inputs * weights[irow])
                )

    dhyperparams_dweights[:] = dense_weight_partials.reshape(-1)
    return {
        "hyperparams": hyperparams,
        "dhyperparams_dvariances": dhyperparams_dvariances,
        "dhyperparams_dweights": dhyperparams_dweights,
    }


def regression_prior_mean_values(values):
    """Return FAST regression prior mean and fixed-mask derivatives."""

    array = np.asarray(values, dtype=float).reshape(-1)
    finite = ~np.isnan(array)
    count = np.count_nonzero(finite)
    derivative = np.zeros(array.size)

    if count == 0:
        return {
            "prior_mean": np.nan,
            "dprior_mean_dvalues": derivative,
        }

    derivative[finite] = 1.0 / count
    return {
        "prior_mean": np.mean(array[finite]),
        "dprior_mean_dvalues": derivative,
    }


def regression_numeric_scalar_values(values):
    """Return the first numeric scalar represented by fixed-shape values."""

    return {"numeric_scalar": float(np.asarray(values, dtype=float).reshape(-1)[0])}


def regression_numeric_column_row_count(input_shape):
    """Return output row count for FAST regression numeric-column conversion."""

    if len(input_shape) <= 1:
        return int(np.prod(input_shape))

    return input_shape[0]


def regression_numeric_column_values(values):
    """Return fixed-shape values as FAST regression numeric-column data."""

    array = np.asarray(values, dtype=float)

    if array.ndim <= 1:
        column = array.reshape(-1)
    else:
        column = array.reshape(array.shape[0], -1)[:, 0]

    return {"numeric_column": column}
