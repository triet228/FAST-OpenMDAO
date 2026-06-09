# src/fast_openmdao/specs.py

"""OpenMDAO components for FAST aircraft specification primitives."""

import numpy as np
import openmdao.api as om


class AEACustomArchitecture(om.ExplicitComponent):
    """Build the fixed AEA custom propulsion architecture matrices.

    Outputs:
        architecture: Fixed 10 by 10 component connectivity matrix.
        upstream_split: Fixed 10 by 10 upstream operation matrix.
        downstream_split: Fixed 10 by 10 downstream operation matrix.
        upstream_efficiency: Fixed 10 by 10 upstream efficiency matrix.
        downstream_efficiency: Fixed 10 by 10 downstream efficiency matrix.

    Assumptions:
        FAST-Python exposes this preset as a constant dictionary. With no
        continuous inputs, the OpenMDAO component has an empty derivative
        surface and no declared partials.
    """

    def setup(self):
        self.add_output("architecture", val=np.zeros((10, 10)))
        self.add_output("upstream_split", val=np.zeros((10, 10)))
        self.add_output("downstream_split", val=np.zeros((10, 10)))
        self.add_output("upstream_efficiency", val=np.ones((10, 10)))
        self.add_output("downstream_efficiency", val=np.ones((10, 10)))

    def compute(self, inputs, outputs):
        values = aea_custom_architecture_values()

        for name in aea_custom_architecture_output_names():
            outputs[name] = values[name]


class LM100JHybridArchitecture(om.ExplicitComponent):
    """Build LM100J_Hybrid's custom propulsion architecture matrices.

    Inputs:
        power_split: Scalar split used by the custom operation matrices.

    Outputs:
        architecture: Fixed 11 by 11 component connectivity matrix.
        upstream_split: Fixed 11 by 11 upstream operation matrix.
        downstream_split: Fixed 11 by 11 downstream operation matrix.
        upstream_efficiency: Fixed 11 by 11 upstream efficiency matrix.
        downstream_efficiency: Fixed 11 by 11 downstream efficiency matrix.
        source_type: Fixed two-source FAST source-type vector.
        transmitter_type: Fixed eight-transmitter FAST transmitter-type vector.

    Assumptions:
        The custom LM100J topology is a preset, so only the operation matrices
        are differentiable with respect to the split. Preset dictionaries and
        aircraft metadata remain FAST-Python data-factory behavior.
    """

    def setup(self):
        self.add_input("power_split", val=0.5)
        self.add_output("architecture", val=np.zeros((11, 11)))
        self.add_output("upstream_split", val=np.zeros((11, 11)))
        self.add_output("downstream_split", val=np.zeros((11, 11)))
        self.add_output("upstream_efficiency", val=np.ones((11, 11)))
        self.add_output("downstream_efficiency", val=np.ones((11, 11)))
        self.add_output("source_type", val=np.zeros(2))
        self.add_output("transmitter_type", val=np.zeros(8))
        self.declare_partials(of="*", wrt="power_split")

    def compute(self, inputs, outputs):
        values = lm100j_hybrid_architecture_values(inputs["power_split"][0])

        for name in lm100j_hybrid_architecture_output_names():
            outputs[name] = values[name]

    def compute_partials(self, inputs, partials):
        values = lm100j_hybrid_architecture_values(inputs["power_split"][0])

        for name in lm100j_hybrid_architecture_output_names():
            partials[name, "power_split"] = values[
                "d%s_dpower_split" % name
            ].reshape(-1)


class LM100JHybridOperationMatrices(om.ExplicitComponent):
    """Build LM100J_Hybrid custom operation matrices from one power split.

    Inputs:
        power_split: Scalar split used by FAST-Python's LM100J_Hybrid preset.

    Outputs:
        upstream_split: Fixed 11 by 11 upstream operation matrix.
        downstream_split: Fixed 11 by 11 downstream operation matrix.

    Assumptions:
        This component only represents the continuous split-dependent operation
        matrices from the preset. The surrounding aircraft specification
        dictionaries remain data factory behavior rather than OpenMDAO state.
    """

    def setup(self):
        self.add_input("power_split", val=0.5)
        self.add_output("upstream_split", val=np.zeros((11, 11)))
        self.add_output("downstream_split", val=np.zeros((11, 11)))
        self.declare_partials(of="*", wrt="power_split")

    def compute(self, inputs, outputs):
        values = lm100j_hybrid_operation_matrix_values(inputs["power_split"][0])
        outputs["upstream_split"] = values["upstream_split"]
        outputs["downstream_split"] = values["downstream_split"]

    def compute_partials(self, inputs, partials):
        values = lm100j_hybrid_operation_matrix_values(inputs["power_split"][0])
        partials["upstream_split", "power_split"] = values[
            "dupstream_split_dpower_split"
        ].reshape(-1)
        partials["downstream_split", "power_split"] = values[
            "ddownstream_split_dpower_split"
        ].reshape(-1)


def lm100j_hybrid_architecture_output_names():
    """Return output names for the LM100J_Hybrid architecture component."""

    return [
        "architecture",
        "upstream_split",
        "downstream_split",
        "upstream_efficiency",
        "downstream_efficiency",
        "source_type",
        "transmitter_type",
    ]


def aea_custom_architecture_output_names():
    """Return output names for the AEA custom architecture component."""

    return [
        "architecture",
        "upstream_split",
        "downstream_split",
        "upstream_efficiency",
        "downstream_efficiency",
    ]


def aea_custom_architecture_values():
    """Return the fixed AEA custom architecture matrices from FAST-Python."""

    architecture = np.asarray(
        [
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    downstream_split = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0],
        ],
        dtype=float,
    )
    upstream_efficiency = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0.661, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0.661, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0.661, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0.661, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    downstream_efficiency = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0.661, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0.661, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0.661, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0.661, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )

    return {
        "architecture": architecture,
        "upstream_split": architecture.copy(),
        "downstream_split": downstream_split,
        "upstream_efficiency": upstream_efficiency,
        "downstream_efficiency": downstream_efficiency,
    }


def lm100j_hybrid_architecture_values(power_split):
    """Return LM100J_Hybrid architecture matrices and split derivatives."""

    operation = lm100j_hybrid_operation_matrix_values(power_split)
    architecture = np.asarray(
        [
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    upstream_efficiency = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0.96, 0.96, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0.80, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0.80, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0.80, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.80, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    downstream_efficiency = np.asarray(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0.96, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0.96, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0.80, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0.80, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0.80, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0.80, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=float,
    )
    values = {
        "architecture": architecture,
        "upstream_split": operation["upstream_split"],
        "downstream_split": operation["downstream_split"],
        "upstream_efficiency": upstream_efficiency,
        "downstream_efficiency": downstream_efficiency,
        "source_type": np.asarray([1, 0], dtype=float),
        "transmitter_type": np.asarray([1, 1, 0, 0, 2, 2, 2, 2], dtype=float),
    }

    for name in lm100j_hybrid_architecture_output_names():
        values["d%s_dpower_split" % name] = np.zeros_like(values[name])

    values["dupstream_split_dpower_split"] = operation[
        "dupstream_split_dpower_split"
    ]
    values["ddownstream_split_dpower_split"] = operation[
        "ddownstream_split_dpower_split"
    ]
    return values


def lm100j_hybrid_operation_matrix_values(power_split):
    """Return LM100J_Hybrid split matrices and exact split derivatives."""

    split = np.asarray(power_split).reshape(-1)[0]
    matrix_dtype = np.result_type(split, float)
    upstream_split = np.asarray(
        [
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, split, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, split, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=matrix_dtype,
    )
    downstream_split = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0.5 - split,
                0.5 - split,
                split,
                split,
                0,
            ],
        ],
        dtype=matrix_dtype,
    )

    dupstream_split_dpower_split = np.zeros((11, 11))
    dupstream_split_dpower_split[4, 8] = 1.0
    dupstream_split_dpower_split[5, 9] = 1.0

    ddownstream_split_dpower_split = np.zeros((11, 11))
    ddownstream_split_dpower_split[10, 6] = -1.0
    ddownstream_split_dpower_split[10, 7] = -1.0
    ddownstream_split_dpower_split[10, 8] = 1.0
    ddownstream_split_dpower_split[10, 9] = 1.0

    return {
        "upstream_split": upstream_split,
        "downstream_split": downstream_split,
        "dupstream_split_dpower_split": dupstream_split_dpower_split,
        "ddownstream_split_dpower_split": ddownstream_split_dpower_split,
    }
