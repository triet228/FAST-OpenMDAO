# src/fast_openmdao/split_optimization.py

"""Utilities for optimizing arbitrary FAST propulsion split matrices."""

import inspect
from copy import deepcopy

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
    initialize_missing=True,
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
        initialize_missing: If True, create equal split matrices from ``Arch``
            when no editable split matrix is present and the control direction
            is unambiguous.
        constraint_component_name: Name for the split-sum constraint subsystem.

    Outputs:
        Unsetup OpenMDAO Problem with automatically generated split controls.

    Assumptions:
        Auto mode optimizes editable matrix data stored in ``PropArch``. Missing
        matrices can be initialized from fixed architecture topology, but
        callable split generators and scalar split schedules are reported as
        diagnostics instead of guessed.
    """

    prepared = initialize_missing_propulsion_split_matrices(
        aircraft,
        prop_arch_path=prop_arch_path,
        preferred_matrix=preferred_matrix,
        active_tol=active_tol,
        sum_tol=sum_tol,
        strict=strict,
        enabled=initialize_missing,
    )
    prepared_aircraft = prepared["aircraft"]
    split_specs = infer_propulsion_split_specs(
        prepared_aircraft,
        prop_arch_path=prop_arch_path,
        preferred_matrix=preferred_matrix,
        lower=lower,
        upper=upper,
        active_tol=active_tol,
        sum_tol=sum_tol,
        strict=strict,
        allow_multiple=not strict,
    )
    problem = make_fast_split_optimization_problem(
        aircraft=prepared_aircraft,
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
    problem.fast_auto_split_initialization = prepared
    return problem


def initialize_missing_propulsion_split_matrices(
    aircraft,
    prop_arch_path=("Specs", "Propulsion", "PropArch"),
    preferred_matrix=None,
    active_tol=1.0e-12,
    sum_tol=1.0e-8,
    strict=True,
    enabled=True,
):
    """Return an aircraft copy with inferred equal split matrices added.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        prop_arch_path: Path to the propulsion architecture dictionary.
        preferred_matrix: Optional matrix name, such as ``"OperUps"``.
        active_tol: Absolute tolerance for active architecture entries.
        sum_tol: Tolerance for recognizing normalized split groups.
        strict: If True, do not create several missing matrices at once.
        enabled: If False, return a copy without initializing missing matrices.

    Outputs:
        Dictionary containing the prepared ``aircraft``, initialization
        ``diagnostics``, and any ``initialized`` matrix records.

    Assumptions:
        Existing numeric or callable split matrices are preserved. Generated
        matrices use equal fractions over every active row or column group so
        FAST has a feasible baseline before OpenMDAO replaces branch entries
        with design variables.
    """

    prepared_aircraft = deepcopy(aircraft)
    diagnostics = propulsion_split_diagnostics(
        prepared_aircraft,
        prop_arch_path=prop_arch_path,
        active_tol=active_tol,
        sum_tol=sum_tol,
    )
    result = {
        "aircraft": prepared_aircraft,
        "diagnostics": diagnostics,
        "initialized": (),
    }

    if not enabled:
        return result

    selected = select_missing_split_initializations(
        diagnostics,
        preferred_matrix=preferred_matrix,
        strict=strict,
    )

    if not selected:
        return result

    prop_arch = get_path(prepared_aircraft, prop_arch_path)
    initialized = []

    for item in selected:
        matrix = equal_split_matrix_from_architecture(
            get_path(prepared_aircraft, item["architecture_path"]),
            item["axis"],
            active_tol=active_tol,
        )
        prop_arch[item["matrix_name"]] = matrix.tolist()
        initialized.append(
            {
                "matrix_name": item["matrix_name"],
                "matrix_path": item["matrix_path"],
                "architecture_path": item["architecture_path"],
                "axis": item["axis"],
                "groups": item["groups"],
            }
        )

    result["diagnostics"] = propulsion_split_diagnostics(
        prepared_aircraft,
        prop_arch_path=prop_arch_path,
        active_tol=active_tol,
        sum_tol=sum_tol,
    )
    result["initialized"] = tuple(initialized)
    return result


def make_fast_mission_split_schedule_optimization_problem(
    aircraft,
    mission=None,
    schedule_specs=(),
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
):
    """Create a FAST optimization problem with split schedules as DVs.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        mission: Optional baseline FAST mission/profile dictionary.
        schedule_specs: Split schedule specifications accepted by
            ``split_schedule_design_specs``.
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

    Outputs:
        Unsetup OpenMDAO Problem with one generated scalar design variable per
        selected split-schedule mission point.

    Assumptions:
        FAST schedule arrays already exist in the baseline data. This helper
        keeps the scalar FAST bridge intact by exposing each selected schedule
        leaf independently, which lets OpenMDAO optimize multiple mission
        points without requiring Dymos as a hard dependency.
    """

    generated = split_schedule_design_specs(aircraft, mission, schedule_specs)
    problem = make_fast_optimization_problem(
        aircraft=aircraft,
        mission=mission,
        input_specs=list(input_specs) + generated["input_specs"],
        output_specs=output_specs,
        runner=runner,
        partial_derivatives=partial_derivatives,
        component_name=component_name,
        promotes=promotes,
        design_vars=list(design_vars) + generated["design_vars"],
        objective=objective,
        constraints=constraints,
        driver=driver,
        driver_options=driver_options,
    )
    problem.fast_split_schedule_metadata = generated
    return problem


def make_fast_architecture_power_management_optimization_problem(
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
    power_path=("Specs", "Power"),
    split_matrices=("OperDwn", "OperUps"),
    segment_keys=None,
    lower=0.0,
    upper=1.0,
    active_tol=1.0e-12,
    sum_tol=1.0e-8,
    max_callable_args=17,
    constraint_component_name="power_split_schedule_constraints",
):
    """Create a full-mission power-management optimization problem.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary with validated PropArch.
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
        power_path: Path to the FAST power settings dictionary.
        split_matrices: PropArch matrices to convert into mission schedules.
        segment_keys: Optional FAST segment keys to optimize.
        lower: Lower bound for generated power split design variables.
        upper: Upper bound for generated power split design variables.
        active_tol: Absolute tolerance for active architecture entries.
        sum_tol: Tolerance for recognizing normalized split groups.
        max_callable_args: Maximum split arguments supported by FAST eval_split.
        constraint_component_name: Name for the split-sum constraint subsystem.

    Outputs:
        Unsetup OpenMDAO Problem with one design variable per active split
        entry at every selected mission point and equality constraints that keep
        each pointwise split group normalized.

    Assumptions:
        Numeric ``OperDwn`` and ``OperUps`` matrices describe the validated
        split topology. This helper converts those numeric entries into
        FAST-style callable split matrices and writes per-point ``LamDwn`` or
        ``LamUps`` schedules. Repeated FAST segment types share the same
        schedule because FAST stores split schedules by segment key.
    """

    prepared = prepare_architecture_power_management_schedules(
        aircraft,
        mission=mission,
        prop_arch_path=prop_arch_path,
        power_path=power_path,
        split_matrices=split_matrices,
        segment_keys=segment_keys,
        lower=lower,
        upper=upper,
        active_tol=active_tol,
        sum_tol=sum_tol,
        max_callable_args=max_callable_args,
    )
    generated = split_schedule_design_specs(
        prepared["aircraft"],
        mission,
        prepared["schedule_specs"],
    )
    problem = make_fast_optimization_problem(
        aircraft=prepared["aircraft"],
        mission=mission,
        input_specs=list(input_specs) + generated["input_specs"],
        output_specs=output_specs,
        runner=runner,
        partial_derivatives=partial_derivatives,
        component_name=component_name,
        promotes=promotes,
        design_vars=list(design_vars) + generated["design_vars"],
        objective=objective,
        constraints=constraints,
        driver=driver,
        driver_options=driver_options,
    )

    if prepared["groups"]:
        for spec in generated["input_specs"]:
            problem.model.set_input_defaults(spec["name"], val=spec["val"])

        problem.model.add_subsystem(
            constraint_component_name,
            SplitGroupSums(groups=prepared["groups"]),
            promotes_inputs=generated["input_names"],
            promotes_outputs=[("split_sums", "power_split_sum_constraints")],
        )
        problem.model.add_constraint(
            "power_split_sum_constraints",
            equals=1.0,
            indices=np.arange(len(prepared["groups"])),
        )

    problem.fast_power_management_metadata = prepared
    problem.fast_split_schedule_metadata = generated
    return problem


def prepare_architecture_power_management_schedules(
    aircraft,
    mission=None,
    prop_arch_path=("Specs", "Propulsion", "PropArch"),
    power_path=("Specs", "Power"),
    split_matrices=("OperDwn", "OperUps"),
    segment_keys=None,
    lower=0.0,
    upper=1.0,
    active_tol=1.0e-12,
    sum_tol=1.0e-8,
    max_callable_args=17,
):
    """Return an aircraft copy with full pointwise split schedules.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        mission: Optional FAST mission/profile dictionary used for segment rows.
        prop_arch_path: Path to the propulsion architecture dictionary.
        power_path: Path to the FAST power settings dictionary.
        split_matrices: Iterable of PropArch matrix names to convert.
        segment_keys: Optional FAST segment keys to optimize.
        lower: Lower bound for generated split design variables.
        upper: Upper bound for generated split design variables.
        active_tol: Absolute tolerance for active architecture entries.
        sum_tol: Tolerance for normalized matrix inference.
        max_callable_args: Maximum callable split arguments FAST can evaluate.

    Outputs:
        Dictionary containing prepared ``aircraft``, generated
        ``schedule_specs``, pointwise sum ``groups``, and metadata records.
    """

    prepared_aircraft = deepcopy(aircraft)
    prop_arch = get_path(prepared_aircraft, prop_arch_path)
    power = ensure_power_dict(prepared_aircraft, power_path)
    segments = mission_segment_schedule_records(
        prepared_aircraft,
        mission,
        segment_keys,
    )
    matrix_records = architecture_power_split_matrix_records(
        prop_arch,
        tuple(prop_arch_path),
        tuple(split_matrices),
        active_tol,
        sum_tol,
        max_callable_args,
    )
    schedule_specs = []
    schedule_records = []
    groups = []

    for matrix_record in matrix_records:
        matrix_name = matrix_record["matrix_name"]
        power_name = power_schedule_name(matrix_name)
        split_power = power.setdefault(power_name, {})
        prop_arch[matrix_name] = split_callable_from_matrix_record(matrix_record)

        for segment in segments:
            schedule = np.tile(
                np.asarray(matrix_record["initial_values"], dtype=float),
                (segment["rows"], 1),
            )
            split_power[segment["key"]] = schedule.tolist()
            prefix = safe_name("%s_%s" % (power_name, segment["key"]))
            schedule_spec = {
                "label": "%s %s" % (power_name, segment["key"]),
                "prefix": prefix,
                "target": "aircraft",
                "path": tuple(power_path) + (power_name, segment["key"]),
                "columns": tuple(range(matrix_record["entry_count"])),
                "lower": lower,
                "upper": upper,
            }
            schedule_specs.append(schedule_spec)
            schedule_records.append(
                {
                    "matrix_name": matrix_name,
                    "power_name": power_name,
                    "segment_key": segment["key"],
                    "rows": segment["rows"],
                    "prefix": prefix,
                    "path": schedule_spec["path"],
                    "entries": matrix_record["entries"],
                }
            )

            for point in range(segment["rows"]):
                for group in matrix_record["groups"]:
                    names = tuple(
                        split_schedule_entry_name(schedule_spec, point, column)
                        for column in group
                    )

                    if len(names) > 1:
                        groups.append(names)

    if not schedule_specs:
        raise ValueError(
            "No numeric branching PropArch split matrices were available for "
            "full-mission power-management optimization."
        )

    return {
        "aircraft": prepared_aircraft,
        "schedule_specs": tuple(schedule_specs),
        "groups": tuple(groups),
        "matrices": tuple(matrix_records),
        "segments": tuple(segments),
        "schedules": tuple(schedule_records),
    }


def architecture_power_split_matrix_records(
    prop_arch,
    prop_arch_path,
    split_matrices,
    active_tol,
    sum_tol,
    max_callable_args,
):
    """Return matrix records for numeric PropArch split matrices."""

    if "Arch" not in prop_arch:
        raise ValueError("PropArch must contain Arch for power-management setup.")

    architecture = np.asarray(prop_arch["Arch"], dtype=float)
    records = []

    for matrix_name in split_matrices:
        if matrix_name not in prop_arch:
            continue

        matrix_value = prop_arch[matrix_name]

        if callable(matrix_value):
            continue

        matrix = np.asarray(matrix_value, dtype=float)
        axis = infer_split_axis(architecture, matrix, active_tol, sum_tol)

        if axis["axis"] is None:
            continue

        split_spec = {
            "matrix_path": tuple(prop_arch_path) + (matrix_name,),
            "architecture_path": tuple(prop_arch_path) + ("Arch",),
            "axis": axis["axis"],
            "active_tol": active_tol,
            "include_singletons": False,
            "active_entries": None,
            "active_from": "architecture",
        }
        active_groups = split_active_groups(architecture, matrix, split_spec)
        entries = []
        groups = []

        for group in active_groups:
            group_indexes = []

            for row, col in group:
                group_indexes.append(len(entries))
                entries.append((row, col))

            groups.append(tuple(group_indexes))

        if not entries:
            continue

        if len(entries) > max_callable_args:
            raise ValueError(
                "%s needs %d split arguments, but FAST eval_split supports at "
                "most %d." % (matrix_name, len(entries), max_callable_args)
            )

        records.append(
            {
                "matrix_name": matrix_name,
                "axis": axis["axis"],
                "matrix": matrix.copy(),
                "entries": tuple(entries),
                "entry_count": len(entries),
                "groups": tuple(groups),
                "initial_values": tuple(matrix[row, col] for row, col in entries),
                "reason": axis["reason"],
            }
        )

    return tuple(records)


def split_callable_from_matrix_record(record):
    """Return a FAST split callable backed by one matrix record."""

    matrix = np.asarray(record["matrix"], dtype=float).copy()
    entries = tuple(record["entries"])

    def split_matrix(*values):
        result = matrix.copy()

        for index, (row, col) in enumerate(entries):
            result[row, col] = values[index]

        return result.tolist()

    parameters = [
        inspect.Parameter(
            "split_%d" % index,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for index in range(len(entries))
    ]
    split_matrix.__signature__ = inspect.Signature(parameters)
    return split_matrix


def ensure_power_dict(aircraft, power_path):
    """Return the mutable FAST power dictionary, creating it when missing."""

    current = aircraft

    for key in power_path:
        current = current.setdefault(key, {})

    return current


def power_schedule_name(matrix_name):
    """Return the FAST power schedule name for a PropArch split matrix."""

    if matrix_name == "OperDwn":
        return "LamDwn"

    if matrix_name == "OperUps":
        return "LamUps"

    return "Lam%s" % matrix_name


def mission_segment_schedule_records(aircraft, mission, segment_keys):
    """Return FAST segment keys and point counts for split schedules."""

    selected = normalize_segment_key_selection(segment_keys)
    profile = mission_profile(aircraft, mission)
    records = {}

    if profile is not None and "Segs" in profile:
        segments = list(profile["Segs"])
        segment_points = profile.get("SegPts")

        for index, segment_name in enumerate(segments):
            key = segment_name_to_split_key(segment_name)

            if key is None:
                continue

            if selected is not None and key not in selected:
                continue

            if segment_points is not None:
                rows = int(np.asarray(segment_points).reshape(-1)[index])
            else:
                rows = segment_default_point_count(aircraft, key)

            if key not in records:
                records[key] = {
                    "key": key,
                    "rows": rows,
                    "segments": [segment_name],
                }
            else:
                records[key]["rows"] = max(records[key]["rows"], rows)
                records[key]["segments"].append(segment_name)

    if records:
        ordered = [records[key] for key in FAST_FLIGHT_SEGMENT_KEYS if key in records]
        return tuple(ordered)

    if selected is None:
        raise ValueError(
            "Full-mission power-management optimization needs a mission profile "
            "or explicit segment_keys."
        )

    return tuple(
        {
            "key": key,
            "rows": segment_default_point_count(aircraft, key),
            "segments": [],
        }
        for key in selected
    )


def normalize_segment_key_selection(segment_keys):
    """Return selected FAST segment keys or None for profile-driven setup."""

    if segment_keys is None:
        return None

    if isinstance(segment_keys, str):
        segment_keys = (segment_keys,)

    return tuple(segment_name_to_split_key(value) or str(value) for value in segment_keys)


def mission_profile(aircraft, mission):
    """Return the FAST mission profile dictionary if one is available."""

    if isinstance(mission, dict):
        if isinstance(mission.get("Profile"), dict):
            return mission["Profile"]

        return mission

    try:
        return aircraft["Mission"]["Profile"]
    except KeyError:
        return None


FAST_FLIGHT_SEGMENT_KEYS = ("Tko", "Clb", "Crs", "Des", "Lnd")


def segment_name_to_split_key(name):
    """Return FAST split key for a mission segment name."""

    mapping = {
        "takeoff": "Tko",
        "detailedtakeoff": "Tko",
        "climb": "Clb",
        "cruise": "Crs",
        "cruisebre": "Crs",
        "descent": "Des",
        "landing": "Lnd",
        "tko": "Tko",
        "clb": "Clb",
        "crs": "Crs",
        "des": "Des",
        "lnd": "Lnd",
    }
    return mapping.get(str(name).replace("_", "").replace(" ", "").lower())


def segment_default_point_count(aircraft, key):
    """Return FAST default point count for one split segment key."""

    setting_names = {
        "Tko": "TkoPoints",
        "Clb": "ClbPoints",
        "Crs": "CrsPoints",
        "Des": "DesPoints",
    }

    if key == "Lnd":
        return 2

    setting = setting_names.get(key)
    value = None

    if setting is not None:
        value = aircraft.get("Settings", {}).get(setting)

    if value is None or is_nan_like(value):
        return 10

    return int(value)


def is_nan_like(value):
    """Return whether a FAST setting value represents NaN."""

    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return str(value).lower() == "nan"


def infer_propulsion_split_specs(
    aircraft,
    prop_arch_path=("Specs", "Propulsion", "PropArch"),
    preferred_matrix=None,
    lower=0.0,
    upper=1.0,
    active_tol=1.0e-12,
    sum_tol=1.0e-8,
    strict=True,
    allow_multiple=False,
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
        allow_multiple: If True, return every usable matrix instead of requiring
            ``preferred_matrix`` when several matrices are valid.

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

    if len(valid) > 1 and preferred_matrix is None and not allow_multiple:
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
            "can_initialize": False,
            "initialization_axis": None,
            "initialization_groups": 0,
        }

        if matrix_name not in prop_arch:
            item["reason"] = "missing"

            if architecture is not None and architecture.ndim == 2:
                initialization = missing_split_initialization_candidate(
                    matrix_name,
                    architecture,
                    item["matrix_path"],
                    architecture_path,
                    active_tol,
                )

                if initialization is not None:
                    item.update(initialization)

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


def missing_split_initialization_candidate(
    matrix_name,
    architecture,
    matrix_path,
    architecture_path,
    active_tol,
):
    """Return initialization metadata for a missing split matrix candidate."""

    if matrix_name == "OperDwn":
        axis = "row"
        groups = split_branch_group_count(architecture, axis, active_tol)
    elif matrix_name == "OperUps":
        axis = "column"
        groups = split_branch_group_count(architecture, axis, active_tol)
    else:
        return None

    if groups == 0:
        return None

    return {
        "can_initialize": True,
        "initialization_axis": axis,
        "initialization_groups": groups,
        "axis": axis,
        "groups": groups,
        "matrix_path": matrix_path,
        "architecture_path": architecture_path,
    }


def select_missing_split_initializations(
    diagnostics,
    preferred_matrix=None,
    strict=True,
):
    """Return missing split matrices that should be initialized from ``Arch``."""

    existing_usable = [item for item in diagnostics if item["usable"]]
    candidates = [
        item
        for item in diagnostics
        if item.get("can_initialize")
        and (
            preferred_matrix is None
            or item["matrix_name"].lower() == preferred_matrix.lower()
        )
    ]

    if existing_usable and strict and preferred_matrix is None:
        return ()

    if not candidates:
        return ()

    if preferred_matrix is not None:
        return tuple(candidates[:1])

    if len(candidates) == 1:
        return tuple(candidates)

    if strict:
        names = ", ".join(item["matrix_name"] for item in candidates)
        raise ValueError(
            "Architecture supports several missing split matrix conventions "
            "(%s). Pass preferred_matrix to choose one or set strict=False to "
            "initialize all inferred split controls." % names
        )

    return tuple(candidates)


def equal_split_matrix_from_architecture(architecture, axis, active_tol=1.0e-12):
    """Return a feasible equal split matrix for active architecture entries."""

    architecture = np.asarray(architecture, dtype=float)
    active = np.abs(architecture) > active_tol
    matrix = np.zeros_like(architecture, dtype=float)

    if axis == "row":
        for row in range(active.shape[0]):
            indexes = np.where(active[row, :])[0]

            if len(indexes) > 0:
                matrix[row, indexes] = 1.0 / len(indexes)

        return matrix

    if axis == "column":
        for col in range(active.shape[1]):
            indexes = np.where(active[:, col])[0]

            if len(indexes) > 0:
                matrix[indexes, col] = 1.0 / len(indexes)

        return matrix

    raise ValueError("Split matrix initialization axis must be row or column.")


def split_branch_group_count(architecture, axis, active_tol):
    """Return number of active branching groups along one matrix axis."""

    architecture = np.asarray(architecture, dtype=float)
    active = np.abs(architecture) > active_tol
    groups = 0

    if axis == "row":
        for row in range(active.shape[0]):
            if np.count_nonzero(active[row, :]) > 1:
                groups += 1

        return groups

    if axis == "column":
        for col in range(active.shape[1]):
            if np.count_nonzero(active[:, col]) > 1:
                groups += 1

        return groups

    raise ValueError("Split branch group axis must be row or column.")


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


def split_schedule_design_specs(aircraft, mission=None, schedule_specs=()):
    """Return input and design-variable specs for split schedules.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        mission: Optional mission/profile dictionary for mission-target splits.
        schedule_specs: Iterable of schedule split specifications.

    Outputs:
        Dictionary with ``input_specs``, ``design_vars``, ``input_names``, and
        ``entries``. Each entry maps one OpenMDAO scalar design variable to one
        mission-point leaf in the FAST schedule.
    """

    generated_inputs = []
    generated_design_vars = []
    generated_entries = []

    for index, schedule_spec in enumerate(schedule_specs):
        normalized = normalize_split_schedule_spec(schedule_spec, index)
        schedule = np.asarray(
            get_split_target_value(
                aircraft,
                mission,
                normalized["target"],
                normalized["path"],
            ),
            dtype=float,
        )
        entries = split_schedule_entries(schedule, normalized)

        for point, column, value in entries:
            name = split_schedule_entry_name(normalized, point, column)
            path = split_schedule_entry_path(normalized["path"], point, column)
            generated_inputs.append(
                {
                    "name": name,
                    "target": normalized["target"],
                    "path": path,
                    "val": value,
                    "desc": "%s split schedule point %s." % (
                        normalized["label"],
                        split_schedule_entry_label(point, column),
                    ),
                }
            )
            generated_design_vars.append(
                split_schedule_design_var_spec(name, normalized)
            )
            generated_entries.append(
                {
                    "name": name,
                    "target": normalized["target"],
                    "path": path,
                    "schedule_path": normalized["path"],
                    "point": point,
                    "column": column,
                    "label": normalized["label"],
                }
            )

    return {
        "input_specs": generated_inputs,
        "design_vars": generated_design_vars,
        "input_names": tuple(spec["name"] for spec in generated_inputs),
        "entries": tuple(generated_entries),
    }


def normalize_split_schedule_spec(schedule_spec, index):
    """Return one normalized mission split schedule specification."""

    if "path" not in schedule_spec:
        raise ValueError("Split schedule specs require a path.")

    label = schedule_spec.get("label", "schedule_%d" % index)
    prefix = schedule_spec.get("prefix", safe_name(label))

    return {
        "path": tuple(schedule_spec["path"]),
        "target": schedule_spec.get("target", "aircraft"),
        "label": label,
        "prefix": prefix,
        "points": schedule_spec.get("points"),
        "columns": schedule_spec.get("columns"),
        "lower": schedule_spec.get("lower", 0.0),
        "upper": schedule_spec.get("upper", 1.0),
        "ref": schedule_spec.get("ref"),
        "ref0": schedule_spec.get("ref0"),
        "adder": schedule_spec.get("adder"),
        "scaler": schedule_spec.get("scaler", 1.0),
    }


def split_schedule_entries(schedule, schedule_spec):
    """Return selected schedule entries as point, column, value tuples."""

    array = np.asarray(schedule, dtype=float)

    if array.ndim == 0:
        raise ValueError("Split schedule optimization requires multiple points.")

    points = split_schedule_indexes(schedule_spec["points"], array.shape[0], "points")

    if array.ndim == 1:
        if schedule_spec["columns"] is not None:
            raise ValueError("Columns are only valid for 2D split schedules.")

        return tuple(
            (point, None, array[point])
            for point in points
        )

    if array.ndim == 2:
        columns = split_schedule_indexes(
            schedule_spec["columns"],
            array.shape[1],
            "columns",
        )
        return tuple(
            (point, column, array[point, column])
            for point in points
            for column in columns
        )

    raise ValueError("Split schedule optimization supports 1D or 2D schedules.")


def split_schedule_indexes(indexes, size, label):
    """Return selected schedule indexes, defaulting to the full axis."""

    if indexes is None:
        return tuple(range(size))

    selected = tuple(int(value) for value in np.asarray(indexes).reshape(-1))

    for value in selected:
        if value < 0 or value >= size:
            raise ValueError(
                "%s index %d is outside schedule size %d." % (
                    label,
                    value,
                    size,
                )
            )

    return selected


def split_schedule_entry_name(schedule_spec, point, column):
    """Return stable OpenMDAO variable name for one schedule entry."""

    if column is None:
        return "%s_%d" % (schedule_spec["prefix"], point)

    return "%s_%d_%d" % (schedule_spec["prefix"], point, column)


def split_schedule_entry_path(path, point, column):
    """Return FAST path for one schedule point or point-column leaf."""

    if column is None:
        return tuple(path) + (point,)

    return tuple(path) + (point, column)


def split_schedule_entry_label(point, column):
    """Return a compact human-readable schedule entry label."""

    if column is None:
        return "%d" % point

    return "%d,%d" % (point, column)


def split_schedule_design_var_spec(name, schedule_spec):
    """Return one OpenMDAO design-variable spec for a schedule entry."""

    spec = {
        "name": name,
        "lower": schedule_spec["lower"],
        "upper": schedule_spec["upper"],
    }

    for key in ("ref", "ref0", "adder", "scaler"):
        if schedule_spec[key] is not None:
            spec[key] = schedule_spec[key]

    return spec


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
