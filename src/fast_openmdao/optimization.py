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
