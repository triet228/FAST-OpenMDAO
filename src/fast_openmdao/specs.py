# src/fast_openmdao/specs.py

"""OpenMDAO components for FAST aircraft specification primitives."""

import numpy as np
import openmdao.api as om


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
