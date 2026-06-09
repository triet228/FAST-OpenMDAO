# src/fast_openmdao/split_optimization.py

"""Utilities for optimizing arbitrary FAST propulsion split matrices."""

import numpy as np
import openmdao.api as om

from fast_openmdao.paths import get_path
from fast_openmdao.problem import make_fast_optimization_problem


class SplitGroupSums(om.ExplicitComponent):
    """Sum scalar split design variables over fixed architecture groups.

    Inputs:
        One scalar input for each optimized split entry.

    Outputs:
        split_sums: One sum per constrained split group.

    Assumptions:
        Split group membership is fixed at setup from the baseline architecture.
        The component is linear, so its analytical Jacobian is a constant
        incidence matrix. Bounds on the individual split variables are handled
        by OpenMDAO design-variable bounds.
    """

    def initialize(self):
        self.options.declare("groups", default=())

    def setup(self):
        groups = normalize_split_groups(self.options["groups"])
        input_names = unique_split_input_names(groups)

        for name in input_names:
            self.add_input(name, val=1.0)

        self.add_output("split_sums", val=np.ones(len(groups)))

        for row, group in enumerate(groups):
            for name in group:
                self.declare_partials(
                    of="split_sums",
                    wrt=name,
                    rows=np.asarray([row]),
                    cols=np.asarray([0]),
                    val=1.0,
                )

    def compute(self, inputs, outputs):
        groups = normalize_split_groups(self.options["groups"])
        sums = np.zeros(len(groups))

        for row, group in enumerate(groups):
            for name in group:
                sums[row] += inputs[name][0]

        outputs["split_sums"] = sums


def make_fast_split_optimization_problem(
    aircraft,
    mission=None,
    split_specs=(),
    input_specs=(),
    output_specs=None,
    runner=None,
    partial_derivatives=None,
    component_name="fast",
    promotes=None,
    design_vars=(),
    objective=None,
    constraints=(),
    driver=None,
    driver_options=None,
    constraint_component_name="split_constraints",
):
    """Create a FAST optimization problem with arbitrary split entries as DVs.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        mission: Optional baseline FAST mission/profile dictionary.
        split_specs: Matrix split specifications accepted by
            ``split_matrix_design_specs``.
        input_specs: Additional scalar FAST path input specs.
        output_specs: Scalar FAST result outputs.
        runner: Optional FAST-compatible runner.
        partial_derivatives: Optional analytic partial derivative mapping.
        component_name: FAST wrapper subsystem name.
        promotes: OpenMDAO promotions list for the FAST wrapper.
        design_vars: Additional OpenMDAO design-variable specs.
        objective: Objective spec. Defaults to FAST MTOW.
        constraints: Additional OpenMDAO constraint specs.
        driver: Optional OpenMDAO driver.
        driver_options: Driver options.
        constraint_component_name: Name for the split-sum constraint subsystem.

    Outputs:
        Unsetup OpenMDAO Problem with generated split design variables and
        split-sum equality constraints.

    Assumptions:
        Each optimized matrix entry already exists in the baseline FAST data.
        The generated variables are scalar path inputs because the current FAST
        bridge intentionally treats dictionary-path values as scalar leaves.
    """

    generated = split_matrix_design_specs(aircraft, mission, split_specs)
    all_input_specs = list(input_specs) + generated["input_specs"]
    all_design_vars = list(design_vars) + generated["design_vars"]
    all_constraints = list(constraints)

    problem = make_fast_optimization_problem(
        aircraft=aircraft,
        mission=mission,
        input_specs=all_input_specs,
        output_specs=output_specs,
        runner=runner,
        partial_derivatives=partial_derivatives,
        component_name=component_name,
        promotes=promotes,
        design_vars=all_design_vars,
        objective=objective,
        constraints=all_constraints,
        driver=driver,
        driver_options=driver_options,
    )

    if generated["groups"]:
        for spec in generated["input_specs"]:
            problem.model.set_input_defaults(spec["name"], val=spec["val"])

        problem.model.add_subsystem(
            constraint_component_name,
            SplitGroupSums(groups=generated["groups"]),
            promotes_inputs=generated["input_names"],
            promotes_outputs=[("split_sums", "split_sum_constraints")],
        )
        problem.model.add_constraint(
            "split_sum_constraints",
            equals=1.0,
            indices=np.arange(len(generated["groups"])),
        )

    problem.fast_split_metadata = generated
    return problem


def make_fast_auto_split_optimization_problem(
    aircraft,
    mission=None,
    input_specs=(),
    output_specs=None,
    runner=None,
    partial_derivatives=None,
    component_name="fast",
    promotes=None,
    design_vars=(),
    objective=None,
    constraints=(),
    driver=None,
    driver_options=None,
    prop_arch_path=("Specs", "Propulsion", "PropArch"),
    preferred_matrix=None,
    lower=0.0,
    upper=1.0,
    active_tol=1.0e-12,
    sum_tol=1.0e-8,
    strict=True,
    constraint_component_name="split_constraints",
):
    """Create a FAST split optimization problem by inspecting PropArch.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        mission: Optional baseline FAST mission/profile dictionary.
        input_specs: Additional scalar FAST path input specs.
        output_specs: Scalar FAST result outputs.
        runner: Optional FAST-compatible runner.
        partial_derivatives: Optional analytic partial derivative mapping.
        component_name: FAST wrapper subsystem name.
        promotes: OpenMDAO promotions list for the FAST wrapper.
        design_vars: Additional OpenMDAO design-variable specs.
        objective: Objective spec. Defaults to FAST MTOW.
        constraints: Additional OpenMDAO constraint specs.
        driver: Optional OpenMDAO driver.
        driver_options: Driver options.
        prop_arch_path: Path to the FAST propulsion architecture dictionary.
        preferred_matrix: Optional matrix name, such as ``"OperDwn"``.
        lower: Lower bound for generated split design variables.
        upper: Upper bound for generated split design variables.
        active_tol: Absolute tolerance for active architecture entries.
        sum_tol: Tolerance for recognizing normalized split groups.
        strict: If True, fail when normalization or matrix choice is ambiguous.
        constraint_component_name: Name for the split-sum constraint subsystem.

    Outputs:
        Unsetup OpenMDAO Problem with automatically generated split controls.

    Assumptions:
        Auto mode only optimizes editable matrix data already stored in
        ``PropArch``. Callable split generators and scalar split schedules are
        reported as diagnostics instead of guessed.
    """

    split_specs = infer_propulsion_split_specs(
        aircraft,
        prop_arch_path=prop_arch_path,
        preferred_matrix=preferred_matrix,
        lower=lower,
        upper=upper,
        active_tol=active_tol,
        sum_tol=sum_tol,
        strict=strict,
    )
    problem = make_fast_split_optimization_problem(
        aircraft=aircraft,
        mission=mission,
        split_specs=split_specs,
        input_specs=input_specs,
        output_specs=output_specs,
        runner=runner,
        partial_derivatives=partial_derivatives,
        component_name=component_name,
        promotes=promotes,
        design_vars=design_vars,
        objective=objective,
        constraints=constraints,
        driver=driver,
        driver_options=driver_options,
        constraint_component_name=constraint_component_name,
    )
    problem.fast_auto_split_specs = split_specs
    return problem


def infer_propulsion_split_specs(
    aircraft,
    prop_arch_path=("Specs", "Propulsion", "PropArch"),
    preferred_matrix=None,
    lower=0.0,
    upper=1.0,
    active_tol=1.0e-12,
    sum_tol=1.0e-8,
    strict=True,
):
    """Infer editable split matrix specs from a FAST PropArch dictionary.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        prop_arch_path: Path to the propulsion architecture dictionary.
        preferred_matrix: Optional matrix name to select when several exist.
        lower: Lower bound for generated split variables.
        upper: Upper bound for generated split variables.
        active_tol: Absolute tolerance for active architecture entries.
        sum_tol: Normalized group-sum tolerance.
        strict: If True, reject ambiguous or currently invalid matrices.

    Outputs:
        Tuple of split specs accepted by ``make_fast_split_optimization_problem``.
    """

    diagnostics = propulsion_split_diagnostics(
        aircraft,
        prop_arch_path=prop_arch_path,
        active_tol=active_tol,
        sum_tol=sum_tol,
    )
    valid = [item for item in diagnostics if item["usable"]]

    if preferred_matrix is not None:
        valid = [
            item
            for item in valid
            if item["matrix_name"].lower() == preferred_matrix.lower()
        ]

    if not valid:
        raise ValueError(split_diagnostic_message(diagnostics, preferred_matrix))

    if strict:
        invalid_selected = [
            item
            for item in diagnostics
            if (
                preferred_matrix is None
                or item["matrix_name"].lower() == preferred_matrix.lower()
            )
            and not item["usable"]
            and item["exists"]
        ]

        if invalid_selected and preferred_matrix is not None:
            raise ValueError(split_diagnostic_message(diagnostics, preferred_matrix))

    if len(valid) > 1 and preferred_matrix is None:
        names = ", ".join(item["matrix_name"] for item in valid)
        raise ValueError(
            "Several editable split matrices are usable (%s). Pass "
            "preferred_matrix to select the intended control." % names
        )

    return tuple(
        {
            "label": item["matrix_name"],
            "prefix": safe_name(item["matrix_name"]),
            "target": "aircraft",
            "matrix_path": item["matrix_path"],
            "architecture_path": item["architecture_path"],
            "axis": item["axis"],
            "lower": lower,
            "upper": upper,
            "active_tol": active_tol,
        }
        for item in valid
    )


def propulsion_split_diagnostics(
    aircraft,
    prop_arch_path=("Specs", "Propulsion", "PropArch"),
    active_tol=1.0e-12,
    sum_tol=1.0e-8,
):
    """Return diagnostics for known editable PropArch split matrices."""

    candidates = []
    prop_arch = get_path(aircraft, prop_arch_path)
    architecture_path = tuple(prop_arch_path) + ("Arch",)

    try:
        architecture = np.asarray(prop_arch["Arch"], dtype=float)
    except KeyError:
        architecture = None

    for matrix_name in ("OperDwn", "OperUps"):
        item = {
            "matrix_name": matrix_name,
            "matrix_path": tuple(prop_arch_path) + (matrix_name,),
            "architecture_path": architecture_path,
            "exists": matrix_name in prop_arch,
            "usable": False,
            "axis": None,
            "reason": "",
            "row_groups": 0,
            "column_groups": 0,
        }

        if matrix_name not in prop_arch:
            item["reason"] = "missing"
            candidates.append(item)
            continue

        matrix_value = prop_arch[matrix_name]

        if callable(matrix_value):
            item["reason"] = "callable split generator is not an editable matrix"
            candidates.append(item)
            continue

        try:
            matrix = np.asarray(matrix_value, dtype=float)
        except (TypeError, ValueError):
            item["reason"] = "not numeric"
            candidates.append(item)
            continue

        if matrix.ndim != 2:
            item["reason"] = "not two-dimensional"
            candidates.append(item)
            continue

        if architecture is None:
            item["reason"] = "missing architecture matrix"
            candidates.append(item)
            continue

        if architecture.shape != matrix.shape:
            item["reason"] = "architecture and split matrix shapes differ"
            candidates.append(item)
            continue

        axis = infer_split_axis(architecture, matrix, active_tol, sum_tol)
        item.update(axis)
        item["usable"] = axis["axis"] is not None
        item["reason"] = axis["reason"]
        candidates.append(item)

    return tuple(candidates)


def infer_split_axis(architecture, matrix, active_tol, sum_tol):
    """Infer whether a split matrix is normalized by rows or columns."""

    row = split_axis_score(architecture, matrix, "row", active_tol, sum_tol)
    column = split_axis_score(architecture, matrix, "column", active_tol, sum_tol)

    if row["valid"] and not column["valid"]:
        return {
            "axis": "row",
            "reason": "row-normalized branching groups",
            "row_groups": row["groups"],
            "column_groups": column["groups"],
        }

    if column["valid"] and not row["valid"]:
        return {
            "axis": "column",
            "reason": "column-normalized branching groups",
            "row_groups": row["groups"],
            "column_groups": column["groups"],
        }

    if row["valid"] and column["valid"]:
        return {
            "axis": None,
            "reason": "row and column normalization are both plausible",
            "row_groups": row["groups"],
            "column_groups": column["groups"],
        }

    if row["groups"] == 0 and column["groups"] == 0:
        reason = "no branching split groups"
    else:
        reason = "branching split groups are not normalized"

    return {
        "axis": None,
        "reason": reason,
        "row_groups": row["groups"],
        "column_groups": column["groups"],
    }


def split_axis_score(architecture, matrix, axis, active_tol, sum_tol):
    """Return whether active split groups along an axis are normalized."""

    active = np.abs(architecture) > active_tol
    group_count = 0
    valid_count = 0

    if axis == "row":
        for row in range(active.shape[0]):
            indexes = np.where(active[row, :])[0]

            if len(indexes) <= 1:
                continue

            group_count += 1
            total = np.sum(matrix[row, indexes])

            if abs(total - 1.0) <= sum_tol:
                valid_count += 1
    else:
        for col in range(active.shape[1]):
            indexes = np.where(active[:, col])[0]

            if len(indexes) <= 1:
                continue

            group_count += 1
            total = np.sum(matrix[indexes, col])

            if abs(total - 1.0) <= sum_tol:
                valid_count += 1

    return {
        "groups": group_count,
        "valid": group_count > 0 and group_count == valid_count,
    }


def split_diagnostic_message(diagnostics, preferred_matrix):
    """Return a compact error message for failed auto split inference."""

    selected = []

    for item in diagnostics:
        if (
            preferred_matrix is None
            or item["matrix_name"].lower() == preferred_matrix.lower()
        ):
            selected.append("%s: %s" % (item["matrix_name"], item["reason"]))

    if not selected:
        selected.append("no matching split matrix candidate")

    return "No unambiguous editable split matrix found. " + "; ".join(selected)


def split_matrix_design_specs(aircraft, mission=None, split_specs=()):
    """Return generated input specs, design variables, and split groups.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        mission: Optional mission/profile dictionary for mission-target splits.
        split_specs: Iterable of matrix split specifications.

    Outputs:
        Dictionary with ``input_specs``, ``design_vars``, ``groups``,
        ``input_names``, and ``entries``.
    """

    generated_inputs = []
    generated_design_vars = []
    generated_groups = []
    generated_entries = []

    for index, split_spec in enumerate(split_specs):
        normalized = normalize_split_spec(split_spec, index)
        matrix = np.asarray(
            get_split_target_value(aircraft, mission, normalized["target"], normalized["matrix_path"]),
            dtype=float,
        )
        architecture = split_architecture_matrix(
            aircraft,
            mission,
            normalized,
            matrix,
        )
        active_groups = split_active_groups(architecture, matrix, normalized)

        for group in active_groups:
            names = []

            for row, col in group:
                name = split_entry_name(normalized, row, col)
                names.append(name)
                path = tuple(normalized["matrix_path"]) + (row, col)
                value = matrix[row, col]
                generated_inputs.append(
                    {
                        "name": name,
                        "target": normalized["target"],
                        "path": path,
                        "val": value,
                        "desc": "%s split entry (%d, %d)." % (
                            normalized["label"],
                            row,
                            col,
                        ),
                    }
                )
                generated_design_vars.append(
                    split_design_var_spec(name, normalized)
                )
                generated_entries.append(
                    {
                        "name": name,
                        "target": normalized["target"],
                        "path": path,
                        "matrix_path": normalized["matrix_path"],
                        "row": row,
                        "col": col,
                        "axis": normalized["axis"],
                        "label": normalized["label"],
                    }
                )

            if len(names) > 1 or normalized["include_singletons"]:
                generated_groups.append(tuple(names))

    return {
        "input_specs": generated_inputs,
        "design_vars": generated_design_vars,
        "groups": tuple(generated_groups),
        "input_names": tuple(spec["name"] for spec in generated_inputs),
        "entries": tuple(generated_entries),
    }


def normalize_split_spec(split_spec, index):
    """Return one normalized matrix split specification."""

    if "matrix_path" not in split_spec:
        raise ValueError("Split specs require a matrix_path.")

    label = split_spec.get("label", "split_%d" % index)
    prefix = split_spec.get("prefix", safe_name(label))
    axis = split_spec.get("axis", "row")

    if axis not in ("row", "column"):
        raise ValueError("Split spec axis must be row or column.")

    return {
        "matrix_path": tuple(split_spec["matrix_path"]),
        "architecture_path": split_spec.get("architecture_path"),
        "target": split_spec.get("target", "aircraft"),
        "label": label,
        "prefix": prefix,
        "axis": axis,
        "lower": split_spec.get("lower", 0.0),
        "upper": split_spec.get("upper", 1.0),
        "ref": split_spec.get("ref"),
        "ref0": split_spec.get("ref0"),
        "adder": split_spec.get("adder"),
        "scaler": split_spec.get("scaler", 1.0),
        "active_tol": split_spec.get("active_tol", 1.0e-12),
        "include_singletons": split_spec.get("include_singletons", False),
        "active_entries": split_spec.get("active_entries"),
        "active_from": split_spec.get("active_from", "architecture"),
    }


def split_architecture_matrix(aircraft, mission, split_spec, matrix):
    """Return the architecture mask used to discover valid split entries."""

    if split_spec["architecture_path"] is None:
        return np.ones_like(matrix)

    architecture_path = tuple(split_spec["architecture_path"])
    architecture = get_split_target_value(
        aircraft,
        mission,
        split_spec["target"],
        architecture_path,
    )
    return np.asarray(architecture, dtype=float)


def split_active_groups(architecture, matrix, split_spec):
    """Return grouped active matrix indexes for one split specification."""

    if architecture.shape != matrix.shape:
        raise ValueError("Architecture and split matrices must have the same shape.")

    active = active_split_mask(architecture, matrix, split_spec)
    groups = []

    if split_spec["axis"] == "row":
        for row in range(active.shape[0]):
            cols = [col for col in range(active.shape[1]) if active[row, col]]

            if len(cols) > 1 or (cols and split_spec["include_singletons"]):
                groups.append(tuple((row, col) for col in cols))
    else:
        for col in range(active.shape[1]):
            rows = [row for row in range(active.shape[0]) if active[row, col]]

            if len(rows) > 1 or (rows and split_spec["include_singletons"]):
                groups.append(tuple((row, col) for row in rows))

    return tuple(groups)


def active_split_mask(architecture, matrix, split_spec):
    """Return boolean mask of split entries exposed as design variables."""

    if split_spec["active_entries"] is not None:
        active = np.zeros_like(matrix, dtype=bool)

        for row, col in split_spec["active_entries"]:
            active[int(row), int(col)] = True

        return active

    active_from = split_spec["active_from"]
    active_tol = split_spec["active_tol"]

    if active_from == "architecture":
        return np.abs(architecture) > active_tol

    if active_from == "matrix":
        return np.abs(matrix) > active_tol

    if active_from == "either":
        return (np.abs(architecture) > active_tol) | (np.abs(matrix) > active_tol)

    raise ValueError("active_from must be architecture, matrix, or either.")


def split_design_var_spec(name, split_spec):
    """Return one OpenMDAO design-variable spec for a generated split."""

    spec = {
        "name": name,
        "lower": split_spec["lower"],
        "upper": split_spec["upper"],
    }

    for key in ("ref", "ref0", "adder", "scaler"):
        if split_spec[key] is not None:
            spec[key] = split_spec[key]

    return spec


def split_entry_name(split_spec, row, col):
    """Return stable OpenMDAO variable name for one matrix entry."""

    return "%s_%d_%d" % (split_spec["prefix"], row, col)


def get_split_target_value(aircraft, mission, target, path):
    """Read a split-related value from the requested FAST target."""

    if target == "aircraft":
        return get_path(aircraft, path)

    if target == "mission":
        if mission is None:
            raise ValueError("Mission split specs require a mission baseline.")

        return get_path(mission, path)

    raise ValueError("Unsupported FAST split target: %s" % target)


def normalize_split_groups(groups):
    """Return split groups as tuples of input names."""

    return tuple(tuple(group) for group in groups)


def unique_split_input_names(groups):
    """Return split input names once, preserving first-use order."""

    names = []

    for group in groups:
        for name in group:
            if name not in names:
                names.append(name)

    return tuple(names)


def safe_name(value):
    """Return a conservative OpenMDAO variable-name fragment."""

    text = str(value).strip().lower()
    chars = []

    for char in text:
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")

    name = "".join(chars).strip("_")

    if not name:
        return "split"

    return name
