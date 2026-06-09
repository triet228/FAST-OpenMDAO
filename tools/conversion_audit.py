# tools/conversion_audit.py

"""Audit FAST-Python functions against FAST-OpenMDAO conversion evidence.

The audit is intentionally conservative: direct OpenMDAO function/class name
matches are treated as converted, while remaining functions are grouped by the
module-level rationale in docs/conversion_inventory.md. It is a review aid, not
proof that a component is correct; parity and derivative tests remain the
authoritative validation for converted equations.
"""

import argparse
import ast
from pathlib import Path


SUPPORT_STATUSES = {
    "bridge",
    "complete-equation-scope",
    "converted",
    "support",
}

COVERAGE_ALIASES = {
    "aea_architecture_matrices": {"AEACustomArchitecture"},
    "a_astar": {"FlowArea"},
    "as_history_matrix": {"BatteryHistoryColumnMatrix"},
    "as_vector": {"BatteryVector", "EngineVector"},
    "astar_a": {"ChokedArea"},
    "battery_power_dynamics": {"BatteryPowerHistory", "BatteryPowerStep"},
    "battery_specific_energy": {"BatterySpecificEnergyProjection"},
    "bms_cost_fraction": {"BMSCostFraction"},
    "burner": {"BurnerFlow"},
    "charging": {"BatteryPowerHistory", "BatteryPowerStep"},
    "comp_stage": {"CompressorStageFlow"},
    "convert_force": {"UnitArrayConversion", "UnitConversion"},
    "convert_length": {"UnitArrayConversion", "UnitConversion"},
    "convert_mass": {"UnitArrayConversion", "UnitConversion"},
    "convert_spec_units": {"SpecPowerUnitConversion"},
    "convert_temperature": {"UnitArrayConversion", "UnitConversion"},
    "convert_tsfc": {"UnitArrayConversion", "UnitConversion"},
    "convert_velocity": {"UnitArrayConversion", "UnitConversion"},
    "convert_with_factors": {"UnitArrayConversion", "UnitConversion"},
    "cp_air": {"AirIntegratedHeat", "AirSpecificHeat"},
    "cp_jeta": {"JetAIntegratedHeat"},
    "create_prop_arch": {
        "ParallelHybridArchitecture",
        "PartialTurboelectricArchitecture",
        "SeriesHybridArchitecture",
        "SimpleSourceTransmitterArchitecture",
        "TurboelectricArchitecture",
    },
    "cv_air": {"AirSpecificHeatVolume"},
    "cycling_aging": {"BatteryCyclingAging"},
    "cycling_aging_parameters": {"BatteryCyclingAging"},
    "diffuser": {"DiffuserFlow"},
    "discharging": {"BatteryPowerHistory", "BatteryPowerStep"},
    "electric_motor_specific_power": {"ElectricMotorSpecificPowerProjection"},
    "engine_weights_for_sizing": {
        "TurbofanEngineWeightForSizing",
        "TurbopropEngineWeightForSizing",
    },
    "estimate_charge_ocv": {"BatteryChargeOCV"},
    "estimate_fuel_use": {"FuelUseHistory"},
    "fan_system": {"FanFlowSplit"},
    "far25_climb": {"FAR25ClimbConstraint"},
    "far25_engine_gradient": {"FAR25EngineGradient"},
    "detailed_battery_enabled": {"DetailedBatteryFlag"},
    "get_thrust_sink_efficiency": {"ThrustSinkEfficiency"},
    "nlgpr": {"GaussianProcessPrediction"},
    "history_matrix": {"BatteryHistoryMatrix"},
    "initial_source_weight": {"SourceWeightVector"},
    "jet25_111": {"JetFAR25NamedClimbConstraint"},
    "jet25_119": {"JetFAR25NamedClimbConstraint"},
    "jet25_121a": {"JetFAR25NamedClimbConstraint"},
    "jet25_121b": {"JetFAR25NamedClimbConstraint"},
    "jet25_121c": {"JetFAR25NamedClimbConstraint"},
    "jet25_121d": {"JetFAR25NamedClimbConstraint"},
    "jet_aeo_climb": {"JetAEOClimbConstraint"},
    "jet_app": {"JetApproachConstraint"},
    "jet_ceil": {"JetCeilingConstraint"},
    "jet_crs": {"JetCruiseConstraint"},
    "jet_cruise_like": {"JetCruiseConstraint"},
    "jet_div": {"JetCruiseConstraint"},
    "jet_lfl": {"JetLandingFieldLengthConstraint"},
    "jet_tofl": {"JetTakeoffFieldLengthConstraint"},
    "kpp_projection": {"KPPProjection"},
    "lm100j_hybrid_architecture": {"LM100JHybridArchitecture"},
    "lm100j_hybrid_oper_dwn": {"LM100JHybridOperationMatrices"},
    "lm100j_hybrid_oper_ups": {"LM100JHybridOperationMatrices"},
    "new_gamma": {"ThermalPerfectGamma"},
    "newton_raphson_tt1": {"AirTemperatureFromHeatAdded"},
    "newton_raphson_tt3": {"AirTemperatureFromHeatRemoved"},
    "nonzero_mean": {"BatteryNonzeroMean"},
    "normalize_split_values": {"SplitValuesVector"},
    "numeric_column": {"RegressionNumericColumn"},
    "numeric_scalar": {"RegressionNumericScalar"},
    "off_design_nozzle": {"OffDesignNozzleMach"},
    "oei_multiplier": {"OEIMultiplier"},
    "perf_ex_nozzle": {"PerfectExpansionNozzleFlow"},
    "prepare_initial_soc": {"BatteryInitialSOC"},
    "prepare_power_time": {"BatteryPowerTimeBroadcast"},
    "prior_calculation": {"RegressionPriorMean"},
    "ps_pt": {"StaticPressure"},
    "pt_ps": {"TotalPressure"},
    "reg_processing": {"RegressionInverseTerm"},
    "regression_airframe_weight": {"TurbofanAirframeWeight"},
    "restore_source_weight": {"SourceWeightRestore"},
    "restore_scalar_or_list": {
        "BatteryScalarOrListRestore",
        "EngineScalarOrListRestore",
    },
    "resize_battery": {"BatteryWeightFromEnergy", "DetailedBatterySizing"},
    "resize_detailed_battery": {"DetailedBatterySizing"},
    "rhos_rhot": {"StaticDensity"},
    "sample_variance": {"RegressionSampleVariance"},
    "simple_off_design": {"SimpleOffDesignTurbofan"},
    "solve_battery_current": {"BatteryCurrent"},
    "square_exp_kernel": {"SquaredExponentialKernel"},
    "split_scalar": {"SplitScalar"},
    "sum_weight": {"WeightSum"},
    "target_matrix": {"RegressionTargetMatrix"},
    "temperature_from_kelvin": {"UnitArrayConversion", "UnitConversion"},
    "temperature_to_kelvin": {"UnitArrayConversion", "UnitConversion"},
    "ts_tt": {"StaticTemperature"},
    "tt_ts": {"TotalTemperature"},
    "turb_stage": {"TurbineStageFlow"},
    "turboprop_airframe_fit": {"TurbopropAirframeWeight"},
    "update_battery_energy": {"BatteryEnergyCutoff", "BatteryEnergyHistory"},
    "zero_segment_splits": {"ZeroSegmentSplits"},
}


def main():
    parser = argparse.ArgumentParser(
        description="Audit FAST-Python function coverage in FAST-OpenMDAO."
    )
    parser.add_argument(
        "--fast-python-root",
        default="../FAST-Python/src/fast_python",
        help="Path to FAST-Python src/fast_python.",
    )
    parser.add_argument(
        "--openmdao-root",
        default="src/fast_openmdao",
        help="Path to FAST-OpenMDAO source package.",
    )
    parser.add_argument(
        "--inventory",
        default="docs/conversion_inventory.md",
        help="Path to the conversion inventory markdown file.",
    )
    parser.add_argument(
        "--show-remaining",
        action="store_true",
        help="List functions classified by inventory rationale.",
    )
    parser.add_argument(
        "--show-covered",
        action="store_true",
        help="Also list functions with direct OpenMDAO function/class matches.",
    )
    args = parser.parse_args()

    fast_python_root = Path(args.fast_python_root).resolve()
    openmdao_root = Path(args.openmdao_root).resolve()
    inventory_path = Path(args.inventory).resolve()
    inventory = parse_inventory(inventory_path)
    openmdao_symbols = collect_openmdao_symbols(openmdao_root)
    rows = audit_functions(fast_python_root, openmdao_symbols, inventory)

    printed = 0
    for row in rows:
        if row["coverage"] == "documented-remaining" and not args.show_remaining:
            continue

        if row["coverage"] == "covered" and not args.show_covered:
            continue

        printed += 1
        print(
            "{module}:{line}:{name} | {coverage} | {status} | {target}".format(
                **row
            )
        )

    uncovered = [row for row in rows if row["coverage"] != "covered"]
    unsupported = [row for row in uncovered if row["status"] not in SUPPORT_STATUSES]
    print(
        "SUMMARY modules={modules} functions={functions} covered={covered} "
        "documented_remaining={documented} unsupported={unsupported} printed={printed}".format(
            modules=len({row["module"] for row in rows}),
            functions=len(rows),
            covered=len(rows) - len(uncovered),
            documented=len(uncovered) - len(unsupported),
            unsupported=len(unsupported),
            printed=printed,
        )
    )

    return 1 if unsupported else 0


def parse_inventory(inventory_path):
    text = inventory_path.read_text()
    modules = {}

    for line in text.splitlines():
        if not line.startswith("| `"):
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]

        if len(cells) != 3:
            continue

        module = cells[0].strip("`")

        if not module.endswith(".py"):
            continue

        modules[module] = {
            "status": cells[1].strip("`"),
            "target": cells[2],
        }

    return modules


def collect_openmdao_symbols(openmdao_root):
    symbols = set()

    for path in openmdao_root.glob("*.py"):
        tree = ast.parse(path.read_text())

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                symbols.add(node.name)

    return symbols


def audit_functions(fast_python_root, openmdao_symbols, inventory):
    rows = []

    for path in sorted(fast_python_root.glob("*.py")):
        if path.name == "__init__.py":
            continue

        module_info = inventory.get(
            path.name,
            {
                "status": "missing-inventory",
                "target": "No conversion inventory entry.",
            },
        )
        tree = ast.parse(path.read_text())

        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue

            camel_name = snake_to_camel(node.name)
            coverage = "covered"

            aliases = COVERAGE_ALIASES.get(node.name, set())

            if (
                node.name not in openmdao_symbols
                and camel_name not in openmdao_symbols
                and not aliases.intersection(openmdao_symbols)
            ):
                coverage = "documented-remaining"

            rows.append(
                {
                    "module": path.name,
                    "line": node.lineno,
                    "name": node.name,
                    "coverage": coverage,
                    "status": module_info["status"],
                    "target": module_info["target"],
                }
            )

    return rows


def snake_to_camel(name):
    return "".join(part.capitalize() for part in name.split("_"))


if __name__ == "__main__":
    raise SystemExit(main())
