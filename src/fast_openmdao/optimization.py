# src/fast_openmdao/optimization.py

"""OpenMDAO components for FAST optimization helper equations."""

import numpy as np
import openmdao.api as om


class ElectricMotorPowerAvailable(om.ExplicitComponent):
    """Compute FAST available electric motor power for sizing constraints."""

    def setup(self):
        self.add_input("electric_motor_specific_power", val=5000.0, units="W/kg")
        self.add_input("electric_motor_weight", val=100.0, units="kg")
        self.add_output("electric_motor_power_available", val=5.0e5, units="W")
        self.declare_partials(of="electric_motor_power_available", wrt="*")

    def compute(self, inputs, outputs):
        outputs["electric_motor_power_available"] = available_product_value(
            inputs["electric_motor_specific_power"][0],
            inputs["electric_motor_weight"][0],
        )

    def compute_partials(self, inputs, partials):
        partials[
            "electric_motor_power_available",
            "electric_motor_specific_power",
        ] = inputs["electric_motor_weight"][0]
        partials[
            "electric_motor_power_available",
            "electric_motor_weight",
        ] = inputs["electric_motor_specific_power"][0]


class BatteryEnergyAvailable(om.ExplicitComponent):
    """Compute FAST available battery energy for sizing constraints."""

    def setup(self):
        self.add_input("battery_specific_energy", val=720000.0, units="J/kg")
        self.add_input("battery_weight", val=1000.0, units="kg")
        self.add_output("battery_energy_available", val=7.2e8, units="J")
        self.declare_partials(of="battery_energy_available", wrt="*")

    def compute(self, inputs, outputs):
        outputs["battery_energy_available"] = available_product_value(
            inputs["battery_specific_energy"][0],
            inputs["battery_weight"][0],
        )

    def compute_partials(self, inputs, partials):
        partials["battery_energy_available", "battery_specific_energy"] = inputs[
            "battery_weight"
        ][0]
        partials["battery_energy_available", "battery_weight"] = inputs[
            "battery_specific_energy"
        ][0]


class FeasibleStep(om.ExplicitComponent):
    """Compute FAST interior-point feasible slack step for a fixed vector size."""

    def initialize(self):
        self.options.declare("num_constraints", default=1)

    def setup(self):
        num_constraints = self.options["num_constraints"]
        self.add_input("slack", val=np.ones(num_constraints))
        self.add_input("slack_direction", val=-np.ones(num_constraints))
        self.add_output("feasible_step", val=1.0)
        self.declare_partials(of="feasible_step", wrt="*")

    def compute(self, inputs, outputs):
        outputs["feasible_step"] = feasible_step_values(
            inputs["slack"],
            inputs["slack_direction"],
            self.options["num_constraints"],
        )["feasible_step"]

    def compute_partials(self, inputs, partials):
        values = feasible_step_values(
            inputs["slack"],
            inputs["slack_direction"],
            self.options["num_constraints"],
        )
        partials["feasible_step", "slack"] = values["dfeasible_step_dslack"]
        partials["feasible_step", "slack_direction"] = values[
            "dfeasible_step_dslack_direction"
        ]


class GaussianEliminationPivot(om.ExplicitComponent):
    """Perform one FAST Gaussian-elimination pivot for a fixed pivot entry.

    Inputs:
        matrix: Tableau or linear-system matrix before elimination.

    Outputs:
        eliminated_matrix: Matrix after normalizing the pivot row and
            eliminating the pivot column from every other row.

    Assumptions:
        Pivot row and column are one-based options matching FAST-Python's
        ``gauss_elim`` arguments. The pivot entry is assumed nonzero and fixed
        away from branch/singularity changes.
    """

    def initialize(self):
        self.options.declare("num_rows", default=2)
        self.options.declare("num_cols", default=2)
        self.options.declare("pivot_row", default=1)
        self.options.declare("pivot_col", default=1)

    def setup(self):
        shape = (self.options["num_rows"], self.options["num_cols"])
        self.add_input("matrix", val=np.eye(*shape))
        self.add_output("eliminated_matrix", val=np.eye(*shape))
        self.declare_partials(of="eliminated_matrix", wrt="matrix")

    def compute(self, inputs, outputs):
        outputs["eliminated_matrix"] = gaussian_elimination_pivot_values(
            inputs["matrix"],
            self.options["pivot_row"],
            self.options["pivot_col"],
        )["eliminated_matrix"]

    def compute_partials(self, inputs, partials):
        values = gaussian_elimination_pivot_values(
            inputs["matrix"],
            self.options["pivot_row"],
            self.options["pivot_col"],
        )
        partials["eliminated_matrix", "matrix"] = values[
            "deliminated_matrix_dmatrix"
        ]


class HessianUpdate(om.ExplicitComponent):
    """Update a Hessian approximation with FAST's damped BFGS formula.

    Inputs:
        hessian: Current Hessian approximation matrix.
        step: Design-variable step vector.
        gradient_delta: Difference in objective or Lagrangian gradient.

    Outputs:
        updated_hessian: FAST damped BFGS Hessian update.

    Assumptions:
        The Powell-damping branch is fixed for the current values. Derivatives
        are exact away from the branch boundary and singular curvature terms.
    """

    def initialize(self):
        self.options.declare("size", default=2)

    def setup(self):
        size = self.options["size"]
        self.add_input("hessian", val=np.eye(size))
        self.add_input("step", val=np.ones(size))
        self.add_input("gradient_delta", val=np.ones(size))
        self.add_output("updated_hessian", val=np.eye(size))
        self.declare_partials(of="updated_hessian", wrt="*")

    def compute(self, inputs, outputs):
        outputs["updated_hessian"] = hessian_update_values(
            inputs["hessian"],
            inputs["step"],
            inputs["gradient_delta"],
        )["updated_hessian"]

    def compute_partials(self, inputs, partials):
        values = hessian_update_values(
            inputs["hessian"],
            inputs["step"],
            inputs["gradient_delta"],
        )
        partials["updated_hessian", "hessian"] = values[
            "dupdated_hessian_dhessian"
        ]
        partials["updated_hessian", "step"] = values["dupdated_hessian_dstep"]
        partials["updated_hessian", "gradient_delta"] = values[
            "dupdated_hessian_dgradient_delta"
        ]


class OneBasedHistoryValues(om.ExplicitComponent):
    """Select FAST mission-history values using fixed MATLAB one-based indices.

    Inputs:
        history_values: Flattened mission-history vector.

    Outputs:
        selected_values: Values at the configured one-based indices.

    Assumptions:
        Indices are fixed OpenMDAO options, matching FAST-Python optimization
        setup. The mapping is linear, so the analytical derivative is a
        constant selection matrix.
    """

    def initialize(self):
        self.options.declare("history_size", default=1)
        self.options.declare("indices", default=(1,))

    def setup(self):
        history_size = self.options["history_size"]
        selected_size = len(np.asarray(self.options["indices"]).reshape(-1))
        self.add_input("history_values", val=np.zeros(history_size))
        self.add_output("selected_values", val=np.zeros(selected_size))
        self.declare_partials(of="selected_values", wrt="history_values")

    def compute(self, inputs, outputs):
        values = one_based_history_values_component_values(
            inputs["history_values"],
            self.options["indices"],
        )
        outputs["selected_values"] = values["selected_values"]

    def compute_partials(self, inputs, partials):
        values = one_based_history_values_component_values(
            inputs["history_values"],
            self.options["indices"],
        )
        partials["selected_values", "history_values"] = values[
            "dselected_values_dhistory_values"
        ]


class HistoryArray(om.ExplicitComponent):
    """Select FAST mission-history values using fixed zero-based indices.

    Inputs:
        history_values: Flattened mission-history vector.

    Outputs:
        selected_values: Values at the configured zero-based indices.

    Assumptions:
        Indices are fixed options from FAST simplex setup. The mapping is
        linear, so the analytical derivative is a constant selection matrix.
    """

    def initialize(self):
        self.options.declare("history_size", default=1)
        self.options.declare("indices", default=(0,))

    def setup(self):
        history_size = self.options["history_size"]
        selected_size = len(np.asarray(self.options["indices"]).reshape(-1))
        self.add_input("history_values", val=np.zeros(history_size))
        self.add_output("selected_values", val=np.zeros(selected_size))
        self.declare_partials(of="selected_values", wrt="history_values")

    def compute(self, inputs, outputs):
        values = history_array_values(
            inputs["history_values"],
            self.options["indices"],
        )
        outputs["selected_values"] = values["selected_values"]

    def compute_partials(self, inputs, partials):
        values = history_array_values(
            inputs["history_values"],
            self.options["indices"],
        )
        partials["selected_values", "history_values"] = values[
            "dselected_values_dhistory_values"
        ]


class SplitScheduleFill(om.ExplicitComponent):
    """Fill a FAST split schedule from a flattened optimized split vector.

    Inputs:
        target_splits: Existing split schedule block.
        optimized_splits: Flattened FAST ``PowerOpt.Splits`` vector.

    Outputs:
        filled_splits: Schedule with selected segment rows replaced by the
            configured optimized split entries.

    Assumptions:
        Segment points, lambda indices, number of mission points, split count,
        and flattened split offset are fixed options from optimization setup.
        The operation is linear for fixed indexing.
    """

    def initialize(self):
        self.options.declare("num_rows", default=1)
        self.options.declare("num_splits", default=1)
        self.options.declare("num_points", default=1)
        self.options.declare("optimized_size", default=1)
        self.options.declare("lam_index", default=(1,))
        self.options.declare("segment_points", default=(0,))
        self.options.declare("offset", default=0)

    def setup(self):
        shape = (self.options["num_rows"], self.options["num_splits"])
        optimized_size = self.options["optimized_size"]
        self.add_input("target_splits", val=np.zeros(shape))
        self.add_input("optimized_splits", val=np.zeros(optimized_size))
        self.add_output("filled_splits", val=np.zeros(shape))
        self.declare_partials(of="filled_splits", wrt="*")

    def compute(self, inputs, outputs):
        values = split_schedule_fill_values(
            inputs["target_splits"],
            inputs["optimized_splits"],
            self.options["lam_index"],
            self.options["segment_points"],
            self.options["num_points"],
            self.options["num_splits"],
            self.options["offset"],
        )
        outputs["filled_splits"] = values["filled_splits"]

    def compute_partials(self, inputs, partials):
        values = split_schedule_fill_values(
            inputs["target_splits"],
            inputs["optimized_splits"],
            self.options["lam_index"],
            self.options["segment_points"],
            self.options["num_points"],
            self.options["num_splits"],
            self.options["offset"],
        )
        partials["filled_splits", "target_splits"] = values[
            "dfilled_splits_dtarget_splits"
        ]
        partials["filled_splits", "optimized_splits"] = values[
            "dfilled_splits_doptimized_splits"
        ]


class GradientBlock(om.ExplicitComponent):
    """Format FAST finite-difference gradient values as a gradient block.

    Inputs:
        gradient_values: Scalar, vector, or matrix gradient data.

    Outputs:
        gradient_block: FAST nconstraint-by-ndvars gradient matrix.

    Assumptions:
        The input shape and number of design variables are fixed options. This
        mirrors ``fast_python.optimization.as_gradient_block`` for a fixed
        setup branch and gives OpenMDAO a constant analytical Jacobian.
    """

    def initialize(self):
        self.options.declare("input_shape", default=(1,))
        self.options.declare("num_design_vars", default=1)

    def setup(self):
        input_shape = tuple(np.asarray(self.options["input_shape"]).reshape(-1))
        input_shape = tuple(int(value) for value in input_shape)
        num_design_vars = self.options["num_design_vars"]
        rows = gradient_block_output_rows(input_shape, num_design_vars)
        self.add_input("gradient_values", val=np.zeros(input_shape))
        self.add_output("gradient_block", val=np.zeros((rows, num_design_vars)))
        self.declare_partials(of="gradient_block", wrt="gradient_values")

    def compute(self, inputs, outputs):
        values = gradient_block_values(
            inputs["gradient_values"],
            self.options["num_design_vars"],
        )
        outputs["gradient_block"] = values["gradient_block"]

    def compute_partials(self, inputs, partials):
        values = gradient_block_values(
            inputs["gradient_values"],
            self.options["num_design_vars"],
        )
        partials["gradient_block", "gradient_values"] = values[
            "dgradient_block_dgradient_values"
        ]


class GradientMatrix(om.ExplicitComponent):
    """Reshape FAST constraint-gradient values into a fixed matrix.

    Inputs:
        gradient_values: Flattened FAST gradient entries.

    Outputs:
        gradient_matrix: nrow-by-ncol FAST constraint-gradient matrix.

    Assumptions:
        This represents the active nonempty branch of
        ``fast_python.optimization.gradient_matrix``. Empty or ``None``
        gradient outputs are handled by fixed OpenMDAO setup choices rather
        than runtime shape changes.
    """

    def initialize(self):
        self.options.declare("num_rows", default=1)
        self.options.declare("num_cols", default=1)

    def setup(self):
        num_rows = self.options["num_rows"]
        num_cols = self.options["num_cols"]
        size = num_rows * num_cols
        self.add_input("gradient_values", val=np.zeros(size))
        self.add_output("gradient_matrix", val=np.zeros((num_rows, num_cols)))
        rows = np.arange(size)
        self.declare_partials(
            of="gradient_matrix",
            wrt="gradient_values",
            rows=rows,
            cols=rows,
        )

    def compute(self, inputs, outputs):
        values = gradient_matrix_values(
            inputs["gradient_values"],
            self.options["num_rows"],
            self.options["num_cols"],
        )
        outputs["gradient_matrix"] = values["gradient_matrix"]

    def compute_partials(self, inputs, partials):
        values = gradient_matrix_values(
            inputs["gradient_values"],
            self.options["num_rows"],
            self.options["num_cols"],
        )
        partials["gradient_matrix", "gradient_values"] = values[
            "dgradient_matrix_dgradient_values"
        ]


class ConcatenateVectors(om.ExplicitComponent):
    """Concatenate fixed FAST constraint-vector pieces.

    Inputs:
        vector_piece_N: Nonempty one-dimensional constraint vector piece.

    Outputs:
        concatenated_vector: FAST constraint vector with empty pieces skipped.

    Assumptions:
        Piece sizes are fixed at setup, matching the shape-normalization role
        of ``fast_python.optimization.concatenate_vectors``.
    """

    def initialize(self):
        self.options.declare("piece_sizes", default=(1,))

    def setup(self):
        piece_sizes = tuple(int(size) for size in self.options["piece_sizes"])
        output_size = sum(size for size in piece_sizes if size > 0)
        self.add_output("concatenated_vector", val=np.zeros(output_size))

        start = 0
        for index, size in enumerate(piece_sizes):
            if size <= 0:
                continue

            name = f"vector_piece_{index}"
            self.add_input(name, val=np.zeros(size))
            rows = np.arange(start, start + size)
            cols = np.arange(size)
            self.declare_partials(
                of="concatenated_vector",
                wrt=name,
                rows=rows,
                cols=cols,
                val=np.ones(size),
            )
            start += size

    def compute(self, inputs, outputs):
        outputs["concatenated_vector"] = concatenate_vector_values(
            inputs,
            self.options["piece_sizes"],
        )["concatenated_vector"]


class ConcatenateMatrices(om.ExplicitComponent):
    """Stack fixed FAST gradient-matrix pieces by rows.

    Inputs:
        matrix_piece_N: Nonempty gradient matrix piece with fixed column count.

    Outputs:
        concatenated_matrix: FAST gradient matrix with empty pieces skipped.

    Assumptions:
        Piece row counts and the shared column count are fixed at setup,
        matching ``fast_python.optimization.concatenate_matrices``.
    """

    def initialize(self):
        self.options.declare("piece_rows", default=(1,))
        self.options.declare("num_cols", default=1)

    def setup(self):
        piece_rows = tuple(int(rows) for rows in self.options["piece_rows"])
        num_cols = self.options["num_cols"]
        output_rows = sum(rows for rows in piece_rows if rows > 0)
        self.add_output(
            "concatenated_matrix",
            val=np.zeros((output_rows, num_cols)),
        )

        start = 0
        for index, rows_count in enumerate(piece_rows):
            if rows_count <= 0:
                continue

            name = f"matrix_piece_{index}"
            size = rows_count * num_cols
            self.add_input(name, val=np.zeros((rows_count, num_cols)))
            rows = np.arange(start * num_cols, (start + rows_count) * num_cols)
            cols = np.arange(size)
            self.declare_partials(
                of="concatenated_matrix",
                wrt=name,
                rows=rows,
                cols=cols,
                val=np.ones(size),
            )
            start += rows_count

    def compute(self, inputs, outputs):
        outputs["concatenated_matrix"] = concatenate_matrix_values(
            inputs,
            self.options["piece_rows"],
            self.options["num_cols"],
        )["concatenated_matrix"]


class TwoDimensionalArray(om.ExplicitComponent):
    """Normalize FAST split values to explicit two-dimensional arrays.

    Inputs:
        array_values: Fixed-shape scalar/vector/matrix values.

    Outputs:
        two_dimensional_array: Values with FAST-Python's explicit 2-D shape.

    Assumptions:
        Input shape is fixed at setup. This converts
        ``fast_python.optimization.two_dimensional`` into a linear OpenMDAO
        component with an identity reshape derivative.
    """

    def initialize(self):
        self.options.declare("input_shape", default=(1,))

    def setup(self):
        input_shape = tuple(np.asarray(self.options["input_shape"]).reshape(-1))
        input_shape = tuple(int(value) for value in input_shape)
        output_shape = two_dimensional_output_shape(input_shape)
        size = int(np.prod(output_shape))
        self.add_input("array_values", val=np.zeros(input_shape))
        self.add_output("two_dimensional_array", val=np.zeros(output_shape))
        rows = np.arange(size)
        self.declare_partials(
            of="two_dimensional_array",
            wrt="array_values",
            rows=rows,
            cols=rows,
            val=np.ones(size),
        )

    def compute(self, inputs, outputs):
        outputs["two_dimensional_array"] = two_dimensional_array_values(
            inputs["array_values"]
        )["two_dimensional_array"]


class MeritFunction(om.ExplicitComponent):
    """Compute FAST interior-point line-search merit value.

    Inputs:
        objective: Scalar objective value.
        inequality_constraints: Inequality residual vector ``g``.
        equality_constraints: Equality residual vector ``h``.
        slack: Positive slack vector, when the slack branch is active.
        barrier_parameter: FAST barrier parameter ``mu``.

    Outputs:
        merit: Scalar value used by FAST's line-search objective.

    Assumptions:
        This component converts the algebraic merit expression only. The
        golden-section search and interior-point iteration remain OpenMDAO
        driver/orchestration concerns.
    """

    def initialize(self):
        self.options.declare("num_inequality", default=1)
        self.options.declare("num_equality", default=0)
        self.options.declare("use_slack", default=True)

    def setup(self):
        num_inequality = self.options["num_inequality"]
        num_equality = self.options["num_equality"]
        self.add_input("objective", val=1.0)
        self.add_input("inequality_constraints", val=np.zeros(num_inequality))
        self.add_input("equality_constraints", val=np.zeros(num_equality))

        if self.options["use_slack"]:
            self.add_input("slack", val=np.ones(num_inequality))
            self.add_input("barrier_parameter", val=1.0)

        self.add_output("merit", val=1.0)
        self.declare_partials(of="merit", wrt="objective")
        self.declare_partials(of="merit", wrt="inequality_constraints")
        self.declare_partials(of="merit", wrt="equality_constraints")

        if self.options["use_slack"]:
            self.declare_partials(of="merit", wrt="slack")
            self.declare_partials(of="merit", wrt="barrier_parameter")

    def compute(self, inputs, outputs):
        values = merit_function_values(
            inputs["objective"][0],
            inputs["inequality_constraints"],
            inputs["equality_constraints"],
            get_slack_values(inputs, self.options["num_inequality"]),
            get_barrier_parameter(inputs),
            self.options["use_slack"],
        )
        outputs["merit"] = values["merit"]

    def compute_partials(self, inputs, partials):
        values = merit_function_values(
            inputs["objective"][0],
            inputs["inequality_constraints"],
            inputs["equality_constraints"],
            get_slack_values(inputs, self.options["num_inequality"]),
            get_barrier_parameter(inputs),
            self.options["use_slack"],
        )
        partials["merit", "objective"] = values["dmerit_dobjective"]
        partials["merit", "inequality_constraints"] = values[
            "dmerit_dinequality_constraints"
        ]
        partials["merit", "equality_constraints"] = values[
            "dmerit_dequality_constraints"
        ]

        if self.options["use_slack"]:
            partials["merit", "slack"] = values["dmerit_dslack"]
            partials["merit", "barrier_parameter"] = values[
                "dmerit_dbarrier_parameter"
            ]


class PowerLimitConstraints(om.ExplicitComponent):
    """Compute FAST paired lower and upper power/energy limit constraints.

    Inputs:
        used: Used power or energy values.
        available: Available power or energy values.

    Outputs:
        lower_limit: FAST lower residual ``-used / available``.
        upper_limit: FAST upper residual ``used / available - 1``.

    Assumptions:
        This component represents the active, finite-input branch of
        ``fast_python.optimization.power_limit_constraints``. FAST's NaN/Inf
        sanitization is preserved in compute; analytical derivatives are
        meaningful for finite, nonzero available values.
    """

    def initialize(self):
        self.options.declare("vec_size", default=1)
        self.options.declare("eps", default=1.0e-6)

    def setup(self):
        vec_size = self.options["vec_size"]
        rows = np.arange(vec_size)
        self.add_input("used", val=np.ones(vec_size))
        self.add_input("available", val=np.ones(vec_size))
        self.add_output("lower_limit", val=np.zeros(vec_size))
        self.add_output("upper_limit", val=np.zeros(vec_size))
        self.declare_partials(
            of=["lower_limit", "upper_limit"],
            wrt=["used", "available"],
            rows=rows,
            cols=rows,
        )

    def compute(self, inputs, outputs):
        values = power_limit_constraint_values(
            inputs["used"],
            inputs["available"],
            self.options["eps"],
        )
        outputs["lower_limit"] = values["lower"]
        outputs["upper_limit"] = values["upper"]

    def compute_partials(self, inputs, partials):
        values = power_limit_constraint_values(
            inputs["used"],
            inputs["available"],
            self.options["eps"],
        )
        partials["lower_limit", "used"] = values["dlower_dused"]
        partials["lower_limit", "available"] = values["dlower_davailable"]
        partials["upper_limit", "used"] = values["dupper_dused"]
        partials["upper_limit", "available"] = values["dupper_davailable"]


class OperationalSplitConstraints(om.ExplicitComponent):
    """Compute FAST operational split bound residuals for one split family.

    Inputs:
        operational_splits: Flattened operational split values, grouped by
            split variable then mission point.
        design_splits: Design split limits when ``design_active`` is true.

    Outputs:
        split_constraints: FAST residuals ``operational / limit - 1``.

    Assumptions:
        This is one group from
        ``fast_python.optimization.operational_split_constraint_blocks``.
        ``design_active`` selects whether each split is bounded by a design
        variable or by the fixed ``lam_max`` option.
    """

    def initialize(self):
        self.options.declare("npoint", default=1)
        self.options.declare("nsplit", default=1)
        self.options.declare("design_active", default=True)
        self.options.declare("lam_max", default=1.0)
        self.options.declare("eps", default=1.0e-6)

    def setup(self):
        npoint = self.options["npoint"]
        nsplit = self.options["nsplit"]
        size = npoint * nsplit
        self.add_input("operational_splits", val=np.zeros(size))

        if self.options["design_active"]:
            self.add_input("design_splits", val=np.ones(nsplit))

        self.add_output("split_constraints", val=np.zeros(size))
        rows = np.arange(size)
        self.declare_partials(
            of="split_constraints",
            wrt="operational_splits",
            rows=rows,
            cols=rows,
        )

        if self.options["design_active"]:
            self.declare_partials(
                of="split_constraints",
                wrt="design_splits",
                rows=rows,
                cols=np.repeat(np.arange(nsplit), npoint),
            )

    def compute(self, inputs, outputs):
        values = operational_split_constraint_values(
            inputs["operational_splits"],
            self.options["npoint"],
            self.options["nsplit"],
            self.options["design_active"],
            get_design_split_values(inputs, self.options["nsplit"]),
            self.options["lam_max"],
            self.options["eps"],
        )
        outputs["split_constraints"] = values["constraints"]

    def compute_partials(self, inputs, partials):
        values = operational_split_constraint_values(
            inputs["operational_splits"],
            self.options["npoint"],
            self.options["nsplit"],
            self.options["design_active"],
            get_design_split_values(inputs, self.options["nsplit"]),
            self.options["lam_max"],
            self.options["eps"],
        )
        partials["split_constraints", "operational_splits"] = values[
            "dconstraints_doperational_splits"
        ]

        if self.options["design_active"]:
            partials["split_constraints", "design_splits"] = values[
                "dconstraints_ddesign_splits"
            ]


class DesignSplitBounds(om.ExplicitComponent):
    """Compute FAST design split lower and upper bound residuals.

    Inputs:
        design_splits: Design split variables from FAST ``ConSizeOpt``.

    Outputs:
        lower_bounds: FAST residual ``-design_splits``.
        upper_bounds: FAST residual ``design_splits - 1``.
    """

    def initialize(self):
        self.options.declare("vec_size", default=1)

    def setup(self):
        vec_size = self.options["vec_size"]
        rows = np.arange(vec_size)
        self.add_input("design_splits", val=np.ones(vec_size) * 0.5)
        self.add_output("lower_bounds", val=np.zeros(vec_size))
        self.add_output("upper_bounds", val=np.zeros(vec_size))
        self.declare_partials(
            of=["lower_bounds", "upper_bounds"],
            wrt="design_splits",
            rows=rows,
            cols=rows,
        )

    def compute(self, inputs, outputs):
        values = design_split_bound_values(inputs["design_splits"])
        outputs["lower_bounds"] = values["lower"]
        outputs["upper_bounds"] = values["upper"]

    def compute_partials(self, inputs, partials):
        values = design_split_bound_values(inputs["design_splits"])
        partials["lower_bounds", "design_splits"] = values["dlower_ddesign_splits"]
        partials["upper_bounds", "design_splits"] = values["dupper_ddesign_splits"]


class CruisePowerAvailableConstraint(om.ExplicitComponent):
    """Compute FAST cruise power availability sizing residuals.

    Inputs:
        cruise_power: Required cruise power from FAST ``DesCrsPow``.
        gas_turbine_power_available: Available gas-turbine power from
            ``DesPavGT``.

    Outputs:
        cruise_power_constraint: FAST residual ``cruise_power / available - 1``.

    Assumptions:
        This represents the active analysis-type-1 branch of
        ``ConSizeOpt``. FAST sanitizes NaN/Inf residuals; derivatives are exact
        for finite, nonzero available power.
    """

    def initialize(self):
        self.options.declare("vec_size", default=1)
        self.options.declare("eps", default=1.0e-6)

    def setup(self):
        vec_size = self.options["vec_size"]
        rows = np.arange(vec_size)
        self.add_input("cruise_power", val=np.ones(vec_size), units="W")
        self.add_input(
            "gas_turbine_power_available",
            val=np.ones(vec_size),
            units="W",
        )
        self.add_output("cruise_power_constraint", val=np.zeros(vec_size))
        self.declare_partials(
            of="cruise_power_constraint",
            wrt=["cruise_power", "gas_turbine_power_available"],
            rows=rows,
            cols=rows,
        )

    def compute(self, inputs, outputs):
        values = cruise_power_available_constraint_values(
            inputs["cruise_power"],
            inputs["gas_turbine_power_available"],
            self.options["eps"],
        )
        outputs["cruise_power_constraint"] = values["constraint"]

    def compute_partials(self, inputs, partials):
        values = cruise_power_available_constraint_values(
            inputs["cruise_power"],
            inputs["gas_turbine_power_available"],
            self.options["eps"],
        )
        partials["cruise_power_constraint", "cruise_power"] = values[
            "dconstraint_dcruise_power"
        ]
        partials[
            "cruise_power_constraint",
            "gas_turbine_power_available",
        ] = values["dconstraint_dgas_turbine_power_available"]


class OperationalObjective(om.ExplicitComponent):
    """Compute FAST OpsOptimize's unscaled objective selector.

    Inputs:
        fuel_burn: Final mission fuel burn.
        fuel_energy: Final fuel energy.
        battery_energy: Final battery energy.

    Outputs:
        operational_objective: Objective value selected by ``objective_type``.

    Assumptions:
        ``objective_type`` is a fixed discrete option matching FAST-Python:
        DOC, FuelBurn, or Energy.
    """

    def initialize(self):
        self.options.declare("objective_type", default="FuelBurn")

    def setup(self):
        self.add_input("fuel_burn", val=1.0, units="kg")
        self.add_input("fuel_energy", val=1.0, units="J")
        self.add_input("battery_energy", val=1.0, units="J")
        self.add_output("operational_objective", val=1.0)
        self.declare_partials(of="operational_objective", wrt="*")

    def compute(self, inputs, outputs):
        values = operational_objective_values(
            self.options["objective_type"],
            inputs["fuel_burn"][0],
            inputs["fuel_energy"][0],
            inputs["battery_energy"][0],
        )
        outputs["operational_objective"] = values["objective"]

    def compute_partials(self, inputs, partials):
        values = operational_objective_values(
            self.options["objective_type"],
            inputs["fuel_burn"][0],
            inputs["fuel_energy"][0],
            inputs["battery_energy"][0],
        )
        partials["operational_objective", "fuel_burn"] = values[
            "dobjective_dfuel_burn"
        ]
        partials["operational_objective", "fuel_energy"] = values[
            "dobjective_dfuel_energy"
        ]
        partials["operational_objective", "battery_energy"] = values[
            "dobjective_dbattery_energy"
        ]


class PowerManagementObjective(om.ExplicitComponent):
    """Compute FAST ObjPowerManagement's scaled objective selector."""

    def initialize(self):
        self.options.declare("objective_type", default="FuelBurn")

    def setup(self):
        self.add_input("fuel_burn", val=1.0, units="kg")
        self.add_input("fuel_energy", val=1.0, units="J")
        self.add_input("battery_energy", val=1.0, units="J")
        self.add_output("power_management_objective", val=1.0)
        self.declare_partials(of="power_management_objective", wrt="*")

    def compute(self, inputs, outputs):
        values = power_management_objective_values(
            self.options["objective_type"],
            inputs["fuel_burn"][0],
            inputs["fuel_energy"][0],
            inputs["battery_energy"][0],
        )
        outputs["power_management_objective"] = values["objective"]

    def compute_partials(self, inputs, partials):
        values = power_management_objective_values(
            self.options["objective_type"],
            inputs["fuel_burn"][0],
            inputs["fuel_energy"][0],
            inputs["battery_energy"][0],
        )
        partials["power_management_objective", "fuel_burn"] = values[
            "dobjective_dfuel_burn"
        ]
        partials["power_management_objective", "fuel_energy"] = values[
            "dobjective_dfuel_energy"
        ]
        partials["power_management_objective", "battery_energy"] = values[
            "dobjective_dbattery_energy"
        ]


def available_product_value(specific_capacity, installed_weight):
    """Return FAST available power or energy product."""

    return specific_capacity * installed_weight


def feasible_step_values(slack, slack_direction, num_constraints):
    """Return FAST feasible step and active-branch analytical derivatives."""

    tau = 0.005
    slack = np.asarray(slack, dtype=float).reshape(-1)
    direction = np.asarray(slack_direction, dtype=float).reshape(-1)
    limit_count = int(num_constraints)
    step = 1.0
    active = -1

    with np.errstate(divide="ignore", invalid="ignore"):
        max_step = (tau - 1.0) * slack / direction

    for index in range(limit_count):
        if max_step[index] > 0.0 and max_step[index] < step:
            step = max_step[index]
            active = index

    dslack = np.zeros(slack.size)
    ddirection = np.zeros(direction.size)

    if active >= 0:
        dslack[active] = (tau - 1.0) / direction[active]
        ddirection[active] = (
            -(tau - 1.0) * slack[active] / direction[active] ** 2
        )

    return {
        "feasible_step": step,
        "dfeasible_step_dslack": dslack,
        "dfeasible_step_dslack_direction": ddirection,
    }


def gaussian_elimination_pivot_values(matrix, pivot_row, pivot_col):
    """Return one Gaussian-elimination pivot result and dense Jacobian."""

    matrix = np.asarray(matrix, dtype=float)
    row = int(pivot_row) - 1
    col = int(pivot_col) - 1
    num_rows, num_cols = matrix.shape
    pivot = matrix[row, col]
    result = np.array(matrix, dtype=float, copy=True)
    result[row, :] = matrix[row, :] / pivot

    for irow in range(num_rows):
        if irow == row:
            continue

        result[irow, :] = matrix[irow, :] - matrix[irow, col] * result[row, :]

    jacobian = np.zeros((matrix.size, matrix.size))

    for output_row in range(num_rows):
        for output_col in range(num_cols):
            output_index = output_row * num_cols + output_col

            for input_row in range(num_rows):
                for input_col in range(num_cols):
                    input_index = input_row * num_cols + input_col
                    derivative = 0.0

                    if output_row == row:
                        if input_row == row and input_col == output_col:
                            derivative += 1.0 / pivot

                        if input_row == row and input_col == col:
                            derivative -= matrix[row, output_col] / pivot ** 2
                    else:
                        if input_row == output_row and input_col == output_col:
                            derivative += 1.0

                        if input_row == output_row and input_col == col:
                            derivative -= matrix[row, output_col] / pivot

                        if input_row == row and input_col == output_col:
                            derivative -= matrix[output_row, col] / pivot

                        if input_row == row and input_col == col:
                            derivative += (
                                matrix[output_row, col]
                                * matrix[row, output_col]
                                / pivot ** 2
                            )

                    jacobian[output_index, input_index] = derivative

    return {
        "eliminated_matrix": result,
        "deliminated_matrix_dmatrix": jacobian,
    }


def hessian_update_values(hessian, step, gradient_delta):
    """Return FAST damped BFGS Hessian update and dense Jacobians."""

    hessian = np.asarray(hessian, dtype=float)
    step = np.asarray(step, dtype=float).reshape(-1)
    gradient_delta = np.asarray(gradient_delta, dtype=float).reshape(-1)
    updated, data = hessian_update_core(hessian, step, gradient_delta)
    size = step.size
    output_size = hessian.size
    dhessian = np.zeros((output_size, output_size))
    dstep = np.zeros((output_size, size))
    dgradient = np.zeros((output_size, size))

    for index in range(output_size):
        seed = np.zeros_like(hessian)
        seed.reshape(-1)[index] = 1.0
        dhessian[:, index] = hessian_update_directional_derivative(
            data,
            seed,
            np.zeros(size),
            np.zeros(size),
        ).reshape(-1)

    for index in range(size):
        seed = np.zeros(size)
        seed[index] = 1.0
        dstep[:, index] = hessian_update_directional_derivative(
            data,
            np.zeros_like(hessian),
            seed,
            np.zeros(size),
        ).reshape(-1)
        dgradient[:, index] = hessian_update_directional_derivative(
            data,
            np.zeros_like(hessian),
            np.zeros(size),
            seed,
        ).reshape(-1)

    return {
        "updated_hessian": updated,
        "dupdated_hessian_dhessian": dhessian,
        "dupdated_hessian_dstep": dstep,
        "dupdated_hessian_dgradient_delta": dgradient,
    }


def hessian_update_core(hessian, step, gradient_delta):
    """Return BFGS update plus cached terms for linearization."""

    s_col = step.reshape(-1, 1)
    y_col = gradient_delta.reshape(-1, 1)
    sy = (s_col.T @ y_col).item()
    hs = hessian @ s_col
    qterm = (hs.T @ s_col).item()

    if sy >= 0.2 * qterm:
        theta = 1.0
        damped = False
    else:
        theta = 0.8 * qterm / (qterm - sy)
        damped = True

    residual = theta * y_col + (1.0 - theta) * hs
    residual_step = (residual.T @ s_col).item()
    updated = hessian + residual @ residual.T / residual_step
    updated = updated - hs @ hs.T / qterm
    return updated, {
        "hessian": hessian,
        "s": s_col,
        "y": y_col,
        "hs": hs,
        "sy": sy,
        "qterm": qterm,
        "theta": theta,
        "damped": damped,
        "residual": residual,
        "residual_step": residual_step,
    }


def hessian_update_directional_derivative(data, dhessian, dstep, dgradient_delta):
    """Return directional derivative of FAST's damped BFGS Hessian update."""

    dhessian = np.asarray(dhessian, dtype=float)
    ds = np.asarray(dstep, dtype=float).reshape(-1, 1)
    dy = np.asarray(dgradient_delta, dtype=float).reshape(-1, 1)
    hessian = data["hessian"]
    s_col = data["s"]
    y_col = data["y"]
    hs = data["hs"]
    sy = data["sy"]
    qterm = data["qterm"]
    theta = data["theta"]
    residual = data["residual"]
    residual_step = data["residual_step"]
    dhs = dhessian @ s_col + hessian @ ds
    dsy = (ds.T @ y_col + s_col.T @ dy).item()
    dqterm = (dhs.T @ s_col + hs.T @ ds).item()

    if data["damped"]:
        denom = qterm - sy
        dtheta = 0.8 * (qterm * dsy - sy * dqterm) / denom ** 2
    else:
        dtheta = 0.0

    dresidual = (
        theta * dy
        + (1.0 - theta) * dhs
        + dtheta * (y_col - hs)
    )
    dresidual_step = (dresidual.T @ s_col + residual.T @ ds).item()
    dfirst = (dresidual @ residual.T + residual @ dresidual.T) / residual_step
    dfirst = dfirst - residual @ residual.T * dresidual_step / residual_step ** 2
    dsecond = (dhs @ hs.T + hs @ dhs.T) / qterm
    dsecond = dsecond - hs @ hs.T * dqterm / qterm ** 2
    return dhessian + dfirst - dsecond


def one_based_history_values_component_values(history_values, indices):
    """Return fixed-index history values and the selection Jacobian."""

    history = np.asarray(history_values, dtype=float).reshape(-1)
    zero_based = np.asarray(indices, dtype=int).reshape(-1) - 1
    selected = history[zero_based]
    derivative = np.zeros((zero_based.size, history.size))

    for row, column in enumerate(zero_based):
        derivative[row, column] = 1.0

    return {
        "selected_values": selected,
        "dselected_values_dhistory_values": derivative,
    }


def history_array_values(history_values, indices):
    """Return zero-based fixed-index history values and selection Jacobian."""

    history = np.asarray(history_values, dtype=float).reshape(-1)
    zero_based = np.asarray(indices, dtype=int).reshape(-1)
    selected = history[zero_based]
    derivative = np.zeros((zero_based.size, history.size))

    for row, column in enumerate(zero_based):
        derivative[row, column] = 1.0

    return {
        "selected_values": selected,
        "dselected_values_dhistory_values": derivative,
    }


def split_schedule_fill_values(
    target_splits,
    optimized_splits,
    lam_index,
    segment_points,
    num_points,
    num_splits,
    offset,
):
    """Return FAST split-schedule fill values and constant Jacobians."""

    target = np.asarray(target_splits, dtype=float)
    optimized = np.asarray(optimized_splits, dtype=float).reshape(-1)
    lam_index = np.asarray(lam_index, dtype=int).reshape(-1)
    segment_points = np.asarray(segment_points, dtype=int).reshape(-1)
    num_points = int(num_points)
    num_splits = int(num_splits)
    offset = int(offset)
    result = np.array(target, dtype=float, copy=True)
    dtarget = np.eye(target.size)
    doptimized = np.zeros((target.size, optimized.size))

    for isplit in range(num_splits):
        for row, segment_point in enumerate(segment_points):
            split_index = (offset + isplit) * num_points
            split_index += lam_index[segment_point] - 1
            result[row, isplit] = optimized[split_index]
            output_index = row * num_splits + isplit
            dtarget[output_index, :] = 0.0
            doptimized[output_index, split_index] = 1.0

    return {
        "filled_splits": result,
        "dfilled_splits_dtarget_splits": dtarget,
        "dfilled_splits_doptimized_splits": doptimized,
    }


def gradient_block_output_rows(input_shape, num_design_vars):
    """Return FAST gradient-block row count for fixed input shape."""

    array = np.zeros(input_shape)
    shaped = reshape_gradient_block(array, num_design_vars)
    return shaped.shape[0]


def gradient_block_values(gradient_values, num_design_vars):
    """Return FAST gradient-block values and the constant shape Jacobian."""

    values = np.asarray(gradient_values, dtype=float)
    block = reshape_gradient_block(values, num_design_vars)
    derivative = np.zeros((block.size, values.size))

    for index in range(values.size):
        seed = np.zeros_like(values)
        seed.reshape(-1)[index] = 1.0
        derivative[:, index] = reshape_gradient_block(
            seed,
            num_design_vars,
        ).reshape(-1)

    return {
        "gradient_block": block,
        "dgradient_block_dgradient_values": derivative,
    }


def reshape_gradient_block(values, num_design_vars):
    """Apply FAST-Python's gradient-block reshape and column expansion rules."""

    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if len(array) == num_design_vars:
            array = array.reshape(1, num_design_vars)
        else:
            array = array.reshape(-1, 1)

    if array.shape[1] == 1 and num_design_vars != 1:
        array = np.repeat(array, num_design_vars, axis=1)

    return array.reshape(array.shape[0], num_design_vars)


def gradient_matrix_values(gradient_values, num_rows, num_cols):
    """Return FAST gradient-matrix reshape values and identity Jacobian."""

    values = np.asarray(gradient_values, dtype=float).reshape(-1)
    return {
        "gradient_matrix": values.reshape(num_rows, num_cols),
        "dgradient_matrix_dgradient_values": np.ones(values.size),
    }


def concatenate_vector_values(inputs, piece_sizes):
    """Return FAST vector concatenation for fixed nonempty input pieces."""

    pieces = []

    for index, size in enumerate(piece_sizes):
        if size <= 0:
            continue

        pieces.append(
            np.asarray(inputs[f"vector_piece_{index}"], dtype=float).reshape(-1)
        )

    if not pieces:
        return {"concatenated_vector": np.asarray([])}

    return {"concatenated_vector": np.concatenate(pieces)}


def concatenate_matrix_values(inputs, piece_rows, num_cols):
    """Return FAST matrix row stacking for fixed nonempty input pieces."""

    pieces = []

    for index, rows_count in enumerate(piece_rows):
        if rows_count <= 0:
            continue

        pieces.append(
            np.asarray(inputs[f"matrix_piece_{index}"], dtype=float).reshape(
                -1,
                num_cols,
            )
        )

    if not pieces:
        return {"concatenated_matrix": np.zeros((0, num_cols))}

    return {"concatenated_matrix": np.vstack(pieces)}


def two_dimensional_output_shape(input_shape):
    """Return FAST two-dimensional helper output shape for a fixed input."""

    if len(input_shape) == 1:
        return (input_shape[0], 1)

    return input_shape


def two_dimensional_array_values(array_values):
    """Return FAST split-array two-dimensional normalization values."""

    array = np.asarray(array_values, dtype=float).copy()

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    return {"two_dimensional_array": array}


def get_slack_values(inputs, num_inequality):
    """Return slack input when the merit component exposes it."""

    if "slack" not in inputs:
        return np.zeros(num_inequality)

    return inputs["slack"]


def get_barrier_parameter(inputs):
    """Return merit barrier parameter input when present."""

    if "barrier_parameter" not in inputs:
        return 0.0

    return inputs["barrier_parameter"][0]


def merit_function_values(
    objective,
    inequality_constraints,
    equality_constraints,
    slack,
    barrier_parameter,
    use_slack,
):
    """Return FAST merit-function value and analytical derivatives."""

    g = np.asarray(inequality_constraints, dtype=float).reshape(-1)
    h = np.asarray(equality_constraints, dtype=float).reshape(-1)
    s = np.asarray(slack, dtype=float).reshape(-1)
    mu = float(barrier_parameter)
    dmu = 0.0

    if use_slack:
        rho = 100.0 * mu
        residual = g + s
        slack_penalty = mu * np.sum(np.log(s))
        dslack = -mu / s + rho * residual
        dmu = -np.sum(np.log(s)) + 50.0 * (
            np.linalg.norm(h) ** 2 + np.linalg.norm(residual) ** 2
        )
    else:
        rho = 1.0
        residual = g
        slack_penalty = 0.0
        dslack = np.zeros_like(s)

    merit = objective - slack_penalty + 0.5 * rho * (
        np.linalg.norm(h) ** 2 + np.linalg.norm(residual) ** 2
    )

    return {
        "merit": merit,
        "dmerit_dobjective": 1.0,
        "dmerit_dinequality_constraints": rho * residual,
        "dmerit_dequality_constraints": rho * h,
        "dmerit_dslack": dslack,
        "dmerit_dbarrier_parameter": dmu,
    }


def operational_objective_values(objective_type, fuel_burn, fuel_energy, battery_energy):
    """Return OpsOptimize objective value and derivatives."""

    name = objective_type.lower()

    if name == "doc":
        return objective_value_pack(0.0, 0.0, 0.0, 0.0)

    if name == "fuelburn":
        return objective_value_pack(fuel_burn, 1.0, 0.0, 0.0)

    if name == "energy":
        return objective_value_pack(
            fuel_energy + battery_energy,
            0.0,
            1.0,
            1.0,
        )

    raise ValueError("objective_type must be DOC, FuelBurn, or Energy.")


def power_management_objective_values(
    objective_type,
    fuel_burn,
    fuel_energy,
    battery_energy,
):
    """Return ObjPowerManagement objective value and derivatives."""

    name = objective_type.lower()

    if name == "doc":
        return objective_value_pack(0.0, 0.0, 0.0, 0.0)

    if name == "fuelburn":
        return objective_value_pack(fuel_burn / 17207.0, 1.0 / 17207.0, 0.0, 0.0)

    if name == "energy":
        scale = 1.0 / 7.4335e11
        return objective_value_pack(
            (fuel_energy + battery_energy) * scale,
            0.0,
            scale,
            scale,
        )

    raise ValueError("objective_type must be DOC, FuelBurn, or Energy.")


def objective_value_pack(value, dfuel_burn, dfuel_energy, dbattery_energy):
    """Return objective value and derivative mapping."""

    return {
        "objective": value,
        "dobjective_dfuel_burn": dfuel_burn,
        "dobjective_dfuel_energy": dfuel_energy,
        "dobjective_dbattery_energy": dbattery_energy,
    }


def get_design_split_values(inputs, nsplit):
    """Return design split values when present."""

    if "design_splits" not in inputs:
        return np.ones(nsplit)

    return inputs["design_splits"]


def power_limit_constraint_values(used, available, eps):
    """Return FAST power-limit residuals and analytical derivatives."""

    used = np.asarray(used, dtype=float).reshape(-1)
    available = np.asarray(available, dtype=float).reshape(-1)
    lower_raw = -used / available
    upper_raw = used / available - 1.0
    finite = (
        np.isfinite(lower_raw)
        & np.isfinite(upper_raw)
        & (available != 0.0)
    )
    lower = sanitize_values(lower_raw, eps)
    upper = sanitize_values(upper_raw, eps)
    dlower_dused = np.zeros_like(used)
    dlower_davailable = np.zeros_like(used)
    dupper_dused = np.zeros_like(used)
    dupper_davailable = np.zeros_like(used)
    dlower_dused[finite] = -1.0 / available[finite]
    dlower_davailable[finite] = used[finite] / available[finite] ** 2
    dupper_dused[finite] = 1.0 / available[finite]
    dupper_davailable[finite] = -used[finite] / available[finite] ** 2
    return {
        "lower": lower,
        "upper": upper,
        "dlower_dused": dlower_dused,
        "dlower_davailable": dlower_davailable,
        "dupper_dused": dupper_dused,
        "dupper_davailable": dupper_davailable,
    }


def design_split_bound_values(design_splits):
    """Return FAST design split bound residuals and derivatives."""

    design_splits = np.asarray(design_splits, dtype=float).reshape(-1)
    size = design_splits.size
    return {
        "lower": -design_splits,
        "upper": design_splits - 1.0,
        "dlower_ddesign_splits": -np.ones(size),
        "dupper_ddesign_splits": np.ones(size),
    }


def cruise_power_available_constraint_values(cruise_power, available, eps):
    """Return FAST cruise power availability residual and derivatives."""

    cruise_power = np.asarray(cruise_power, dtype=float).reshape(-1)
    available = np.asarray(available, dtype=float).reshape(-1)
    raw = cruise_power / available - 1.0
    finite = np.isfinite(raw) & (available != 0.0)
    constraint = sanitize_values(raw, eps)
    dcruise = np.zeros_like(cruise_power)
    davailable = np.zeros_like(cruise_power)
    dcruise[finite] = 1.0 / available[finite]
    davailable[finite] = -cruise_power[finite] / available[finite] ** 2
    return {
        "constraint": constraint,
        "dconstraint_dcruise_power": dcruise,
        "dconstraint_dgas_turbine_power_available": davailable,
    }


def operational_split_constraint_values(
    operational_splits,
    npoint,
    nsplit,
    design_active,
    design_splits,
    lam_max,
    eps,
):
    """Return operational split residuals and analytical derivatives."""

    operational_splits = np.asarray(operational_splits, dtype=float).reshape(-1)
    design_splits = np.asarray(design_splits, dtype=float).reshape(-1)
    constraints = np.zeros(npoint * nsplit)
    doper = np.zeros(npoint * nsplit)
    ddesign = np.zeros(npoint * nsplit)

    for split_index in range(nsplit):
        start = split_index * npoint
        stop = start + npoint

        if design_active:
            limit = design_splits[split_index]
        else:
            limit = lam_max

        raw = operational_splits[start:stop] / limit - 1.0
        constraints[start:stop] = sanitize_values(raw, eps)
        finite = np.isfinite(raw) & (limit != 0.0)
        doper[start:stop][finite] = 1.0 / limit

        if design_active:
            ddesign[start:stop][finite] = (
                -operational_splits[start:stop][finite] / limit ** 2
            )

    return {
        "constraints": constraints,
        "dconstraints_doperational_splits": doper,
        "dconstraints_ddesign_splits": ddesign,
    }


def sanitize_values(values, eps):
    """Replace NaN and Inf values the way FAST optimization constraints do."""

    array = np.asarray(values, dtype=float).reshape(-1)
    array[np.isnan(array)] = 0.0
    array[np.isinf(array)] = eps
    return array
