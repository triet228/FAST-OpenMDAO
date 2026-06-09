# src/fast_openmdao/examples/compact_electric.py

"""Compact all-electric FAST-Python optimization example."""

import argparse
import os
from copy import deepcopy

os.environ.setdefault("OPENMDAO_REPORTS", "0")

from fast_openmdao.problem import make_fast_optimization_problem


def make_compact_energy_problem(
    range_initial=20000.0,
    range_lower=10000.0,
    range_upper=40000.0,
):
    """Return a real FAST-Python optimization problem for mission energy.

    Inputs:
        range_initial: Initial mission range target in meters.
        range_lower: Lower range bound in meters.
        range_upper: Upper range bound in meters.

    Outputs:
        Unsetup OpenMDAO problem that minimizes stored-source energy over
        mission range for a compact all-electric aircraft.

    Assumptions:
        This is a fast smoke example for OpenMDAO plumbing. It is intentionally
        small and off-design, not a meaningful aircraft design objective.
    """

    aircraft = make_compact_aircraft()
    mission = aircraft.pop("Mission")["Profile"]
    return make_fast_optimization_problem(
        aircraft=aircraft,
        mission=mission,
        input_specs=[
            {
                "name": "mission_range",
                "target": "mission",
                "path": ("Target", "Valu", 0),
                "val": range_initial,
                "units": "m",
                "desc": "Mission range target.",
            },
        ],
        output_specs=[
            {
                "name": "mtow",
                "path": ("mtow",),
                "units": "kg",
                "desc": "Maximum takeoff weight.",
            },
            {
                "name": "energy_used",
                "path": (
                    "aircraft",
                    "Mission",
                    "History",
                    "SI",
                    "Energy",
                    "E_ES",
                    -1,
                    0,
                ),
                "units": "J",
                "desc": "Final stored-source mission energy.",
            },
        ],
        design_vars=[
            {
                "name": "mission_range",
                "lower": range_lower,
                "upper": range_upper,
                "units": "m",
                "scaler": 1.0e-4,
            },
        ],
        objective={
            "name": "energy_used",
            "scaler": 1.0e-7,
        },
        driver_options={
            "maxiter": 20,
            "tol": 1.0e-9,
        },
    )


def make_compact_lift_to_drag_problem(
    lift_to_drag_initial=10.0,
    lift_to_drag_lower=5.0,
    lift_to_drag_upper=25.0,
):
    """Return a FAST-Python optimization problem for cruise L/D.

    Inputs:
        lift_to_drag_initial: Initial cruise lift-to-drag ratio.
        lift_to_drag_lower: Lower cruise lift-to-drag bound.
        lift_to_drag_upper: Upper cruise lift-to-drag bound.

    Outputs:
        Unsetup OpenMDAO problem that minimizes stored-source mission energy
        over cruise lift-to-drag ratio for the compact all-electric aircraft.

    Assumptions:
        This validates a second system-level design variable through the full
        FAST-Python backend. It is still a compact smoke model, not a real
        aircraft design recommendation.
    """

    aircraft = make_compact_aircraft()
    mission = aircraft.pop("Mission")["Profile"]
    return make_fast_optimization_problem(
        aircraft=aircraft,
        mission=mission,
        input_specs=[
            {
                "name": "cruise_lift_to_drag",
                "target": "aircraft",
                "path": ("Specs", "Aero", "L_D", "Crs"),
                "val": lift_to_drag_initial,
                "desc": "Cruise lift-to-drag ratio.",
            },
        ],
        output_specs=[
            {
                "name": "mtow",
                "path": ("mtow",),
                "units": "kg",
                "desc": "Maximum takeoff weight.",
            },
            {
                "name": "energy_used",
                "path": (
                    "aircraft",
                    "Mission",
                    "History",
                    "SI",
                    "Energy",
                    "E_ES",
                    -1,
                    0,
                ),
                "units": "J",
                "desc": "Final stored-source mission energy.",
            },
        ],
        design_vars=[
            {
                "name": "cruise_lift_to_drag",
                "lower": lift_to_drag_lower,
                "upper": lift_to_drag_upper,
                "scaler": 0.1,
            },
        ],
        objective={
            "name": "energy_used",
            "scaler": 1.0e-7,
        },
        driver_options={
            "maxiter": 20,
            "tol": 1.0e-9,
        },
    )


def make_compact_hybrid_power_split_problem(
    split_initial=0.4,
    split_lower=0.0,
    split_upper=0.75,
):
    """Return a FAST-Python optimization problem for hybrid climb split.

    Inputs:
        split_initial: Initial climb downstream split.
        split_lower: Lower split bound.
        split_upper: Upper split bound.

    Outputs:
        Unsetup OpenMDAO problem that minimizes battery source energy over
        climb power split for a compact series-hybrid aircraft.

    Assumptions:
        The compact hybrid aircraft is a deliberately small smoke model. Its
        fuel side is not a calibrated aircraft model; the validation objective
        is battery-source energy, which remains finite and split-sensitive over
        the tested bounds.
    """

    aircraft = make_compact_hybrid_aircraft()
    mission = aircraft.pop("Mission")["Profile"]
    return make_fast_optimization_problem(
        aircraft=aircraft,
        mission=mission,
        input_specs=[
            {
                "name": "climb_power_split",
                "target": "aircraft",
                "path": ("Specs", "Power", "LamDwn", "Clb"),
                "val": split_initial,
                "desc": "Climb downstream power split.",
            },
        ],
        output_specs=[
            {
                "name": "battery_energy_used",
                "path": (
                    "aircraft",
                    "Mission",
                    "History",
                    "SI",
                    "Energy",
                    "E_ES",
                    -1,
                    1,
                ),
                "units": "J",
                "desc": "Final battery-source mission energy.",
            },
        ],
        design_vars=[
            {
                "name": "climb_power_split",
                "lower": split_lower,
                "upper": split_upper,
                "scaler": 1.0,
            },
        ],
        objective={
            "name": "battery_energy_used",
            "scaler": 1.0e-7,
        },
        driver_options={
            "maxiter": 20,
            "tol": 1.0e-9,
        },
    )


def run_demo(range_initial=20000.0, range_lower=10000.0, range_upper=40000.0):
    """Run the compact electric optimization demo.

    Inputs:
        range_initial: Initial mission range target in meters.
        range_lower: Lower range bound in meters.
        range_upper: Upper range bound in meters.

    Outputs:
        Summary dictionary with driver success, optimized range, MTOW, and
        final stored-source energy.

    Side effects:
        Runs the OpenMDAO driver. OpenMDAO reports are disabled by default for
        this CLI-style smoke example to avoid generated report directories.
    """

    problem = make_compact_energy_problem(
        range_initial=range_initial,
        range_lower=range_lower,
        range_upper=range_upper,
    )
    problem.setup()
    result = problem.run_driver()

    return {
        "success": bool(result.success),
        "mission_range": float(problem.get_val("mission_range", units="m")[0]),
        "mtow": float(problem.get_val("mtow", units="kg")[0]),
        "energy_used": float(problem.get_val("energy_used", units="J")[0]),
    }


def run_lift_to_drag_demo(
    lift_to_drag_initial=10.0,
    lift_to_drag_lower=5.0,
    lift_to_drag_upper=25.0,
):
    """Run the compact electric cruise-L/D optimization demo.

    Inputs:
        lift_to_drag_initial: Initial cruise lift-to-drag ratio.
        lift_to_drag_lower: Lower cruise lift-to-drag bound.
        lift_to_drag_upper: Upper cruise lift-to-drag bound.

    Outputs:
        Summary dictionary with driver success, optimized cruise L/D, MTOW, and
        final stored-source energy.
    """

    problem = make_compact_lift_to_drag_problem(
        lift_to_drag_initial=lift_to_drag_initial,
        lift_to_drag_lower=lift_to_drag_lower,
        lift_to_drag_upper=lift_to_drag_upper,
    )
    problem.setup()
    result = problem.run_driver()

    return {
        "success": bool(result.success),
        "cruise_lift_to_drag": float(problem.get_val("cruise_lift_to_drag")[0]),
        "mtow": float(problem.get_val("mtow", units="kg")[0]),
        "energy_used": float(problem.get_val("energy_used", units="J")[0]),
    }


def run_hybrid_power_split_demo(
    split_initial=0.4,
    split_lower=0.0,
    split_upper=0.75,
):
    """Run the compact hybrid climb split optimization demo."""

    problem = make_compact_hybrid_power_split_problem(
        split_initial=split_initial,
        split_lower=split_lower,
        split_upper=split_upper,
    )
    problem.setup()
    result = problem.run_driver()

    return {
        "success": bool(result.success),
        "climb_power_split": float(problem.get_val("climb_power_split")[0]),
        "battery_energy_used": float(
            problem.get_val("battery_energy_used", units="J")[0]
        ),
    }


def evaluate_fast_python_point(mission_range):
    """Evaluate the compact FAST-Python case at one mission range.

    Inputs:
        mission_range: Mission distance target in meters.

    Outputs:
        Summary dictionary with mission range, MTOW, and stored-source energy.

    Assumptions:
        This function is the one-point oracle used to validate OpenMDAO
        optimization results against repeated FAST-Python evaluations.
    """

    from fast_python import run

    aircraft = make_compact_aircraft()
    mission = deepcopy(aircraft.pop("Mission")["Profile"])
    mission["Target"]["Valu"][0] = mission_range
    result = run(aircraft, mission)
    history = result["aircraft"]["Mission"]["History"]["SI"]

    return {
        "mission_range": float(mission_range),
        "mtow": float(result["mtow"]),
        "energy_used": float(history["Energy"]["E_ES"][-1][0]),
    }


def evaluate_fast_python_lift_to_drag_point(lift_to_drag):
    """Evaluate the compact FAST-Python case at one cruise L/D value.

    Inputs:
        lift_to_drag: Cruise lift-to-drag ratio.

    Outputs:
        Summary dictionary with cruise L/D, MTOW, and stored-source energy.
    """

    from fast_python import run

    aircraft = make_compact_aircraft()
    aircraft["Specs"]["Aero"]["L_D"]["Crs"] = lift_to_drag
    mission = deepcopy(aircraft.pop("Mission")["Profile"])
    result = run(aircraft, mission)
    history = result["aircraft"]["Mission"]["History"]["SI"]

    return {
        "cruise_lift_to_drag": float(lift_to_drag),
        "mtow": float(result["mtow"]),
        "energy_used": float(history["Energy"]["E_ES"][-1][0]),
    }


def evaluate_fast_python_hybrid_power_split_point(split):
    """Evaluate the compact hybrid FAST-Python case at one split value.

    Inputs:
        split: Climb downstream split.

    Outputs:
        Summary dictionary with split and final battery-source energy.
    """

    from fast_python import run

    aircraft = make_compact_hybrid_aircraft()
    aircraft["Specs"]["Power"]["LamDwn"]["Clb"] = split
    mission = deepcopy(aircraft.pop("Mission")["Profile"])
    result = run(aircraft, mission)
    history = result["aircraft"]["Mission"]["History"]["SI"]

    return {
        "climb_power_split": float(split),
        "battery_energy_used": float(history["Energy"]["E_ES"][-1][1]),
    }


def sample_fast_python_range_sweep(
    range_lower=10000.0,
    range_upper=40000.0,
    sample_count=9,
):
    """Evaluate FAST-Python repeatedly across the compact range design space.

    Inputs:
        range_lower: Lower mission range bound in meters.
        range_upper: Upper mission range bound in meters.
        sample_count: Number of evenly spaced FAST-Python evaluations.

    Outputs:
        List of point-evaluation summaries sorted by increasing mission range.
    """

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    step = (range_upper - range_lower) / (sample_count - 1)
    return [
        evaluate_fast_python_point(range_lower + index * step)
        for index in range(sample_count)
    ]


def sample_fast_python_lift_to_drag_sweep(
    lift_to_drag_lower=5.0,
    lift_to_drag_upper=25.0,
    sample_count=9,
):
    """Evaluate FAST-Python repeatedly across the cruise-L/D design space.

    Inputs:
        lift_to_drag_lower: Lower cruise lift-to-drag bound.
        lift_to_drag_upper: Upper cruise lift-to-drag bound.
        sample_count: Number of evenly spaced FAST-Python evaluations.

    Outputs:
        List of point-evaluation summaries sorted by increasing cruise L/D.
    """

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    step = (lift_to_drag_upper - lift_to_drag_lower) / (sample_count - 1)
    return [
        evaluate_fast_python_lift_to_drag_point(
            lift_to_drag_lower + index * step
        )
        for index in range(sample_count)
    ]


def sample_fast_python_hybrid_power_split_sweep(
    split_lower=0.0,
    split_upper=0.75,
    sample_count=7,
):
    """Evaluate FAST-Python repeatedly across hybrid split design space."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    step = (split_upper - split_lower) / (sample_count - 1)
    return [
        evaluate_fast_python_hybrid_power_split_point(
            split_lower + index * step
        )
        for index in range(sample_count)
    ]


def best_sampled_point(samples):
    """Return the sampled FAST-Python point with minimum stored-source energy."""

    return min(samples, key=lambda sample: sample["energy_used"])


def validate_demo_against_fast_python_samples(
    range_initial=20000.0,
    range_lower=10000.0,
    range_upper=40000.0,
    sample_count=9,
    range_tolerance=1.0e-4,
    energy_tolerance=1.0e-4,
):
    """Compare OpenMDAO optimization against repeated FAST-Python evaluations.

    Inputs:
        range_initial: Initial OpenMDAO mission range target in meters.
        range_lower: Lower mission range bound in meters.
        range_upper: Upper mission range bound in meters.
        sample_count: Number of FAST-Python point evaluations.
        range_tolerance: Allowed optimized-range mismatch in meters.
        energy_tolerance: Allowed objective mismatch in joules.

    Outputs:
        Dictionary containing the OpenMDAO optimum, sampled FAST-Python points,
        best sampled point, and boolean agreement flags.
    """

    openmdao_summary = run_demo(
        range_initial=range_initial,
        range_lower=range_lower,
        range_upper=range_upper,
    )
    samples = sample_fast_python_range_sweep(
        range_lower=range_lower,
        range_upper=range_upper,
        sample_count=sample_count,
    )
    best_sample = best_sampled_point(samples)
    range_error = abs(openmdao_summary["mission_range"] - best_sample["mission_range"])
    energy_error = abs(openmdao_summary["energy_used"] - best_sample["energy_used"])

    return {
        "success": openmdao_summary["success"],
        "agrees_with_samples": (
            openmdao_summary["success"]
            and range_error <= range_tolerance
            and energy_error <= energy_tolerance
        ),
        "openmdao": openmdao_summary,
        "best_fast_python_sample": best_sample,
        "fast_python_samples": samples,
        "range_error": range_error,
        "energy_error": energy_error,
    }


def validate_hybrid_power_split_demo_against_fast_python_samples(
    split_initial=0.4,
    split_lower=0.0,
    split_upper=0.75,
    sample_count=7,
    split_tolerance=1.0e-4,
    energy_tolerance=1.0e-4,
):
    """Compare hybrid split optimization against FAST-Python samples.

    Inputs:
        split_initial: Initial OpenMDAO climb split.
        split_lower: Lower split bound.
        split_upper: Upper split bound.
        sample_count: Number of FAST-Python point evaluations.
        split_tolerance: Allowed optimized-split mismatch.
        energy_tolerance: Allowed objective mismatch in joules.

    Outputs:
        Dictionary containing the OpenMDAO optimum, sampled FAST-Python points,
        best sampled point, and boolean agreement flags.
    """

    openmdao_summary = run_hybrid_power_split_demo(
        split_initial=split_initial,
        split_lower=split_lower,
        split_upper=split_upper,
    )
    samples = sample_fast_python_hybrid_power_split_sweep(
        split_lower=split_lower,
        split_upper=split_upper,
        sample_count=sample_count,
    )
    best_sample = min(samples, key=lambda sample: sample["battery_energy_used"])
    split_error = abs(
        openmdao_summary["climb_power_split"]
        - best_sample["climb_power_split"]
    )
    energy_error = abs(
        openmdao_summary["battery_energy_used"]
        - best_sample["battery_energy_used"]
    )

    return {
        "success": openmdao_summary["success"],
        "agrees_with_samples": (
            openmdao_summary["success"]
            and split_error <= split_tolerance
            and energy_error <= energy_tolerance
        ),
        "openmdao": openmdao_summary,
        "best_fast_python_sample": best_sample,
        "fast_python_samples": samples,
        "split_error": split_error,
        "energy_error": energy_error,
    }


def validate_hybrid_power_split_multistart_against_fast_python_samples(
    split_initials=None,
    split_lower=0.0,
    split_upper=0.75,
    sample_count=7,
    split_tolerance=1.0e-4,
    energy_tolerance=1.0e-4,
):
    """Compare several hybrid split optimizations against one FAST-Python sweep.

    Inputs:
        split_initials: Iterable of OpenMDAO starting splits. Defaults to low,
            middle, and high values inside the tested bounds.
        split_lower: Lower split bound.
        split_upper: Upper split bound.
        sample_count: Number of FAST-Python point evaluations.
        split_tolerance: Allowed optimized-split mismatch.
        energy_tolerance: Allowed objective mismatch in joules.

    Outputs:
        Dictionary containing each OpenMDAO run, the shared FAST-Python sweep,
        the best sampled point, and aggregate agreement.

    Assumptions:
        The FAST-Python sweep is independent of optimizer starting point, so it
        is evaluated once and reused to check every OpenMDAO optimum.
    """

    if split_initials is None:
        split_initials = (0.15, 0.4, 0.7)

    samples = sample_fast_python_hybrid_power_split_sweep(
        split_lower=split_lower,
        split_upper=split_upper,
        sample_count=sample_count,
    )
    best_sample = min(samples, key=lambda sample: sample["battery_energy_used"])
    runs = []

    for split_initial in split_initials:
        openmdao_summary = run_hybrid_power_split_demo(
            split_initial=split_initial,
            split_lower=split_lower,
            split_upper=split_upper,
        )
        split_error = abs(
            openmdao_summary["climb_power_split"]
            - best_sample["climb_power_split"]
        )
        energy_error = abs(
            openmdao_summary["battery_energy_used"]
            - best_sample["battery_energy_used"]
        )
        runs.append(
            {
                "split_initial": split_initial,
                "success": openmdao_summary["success"],
                "agrees_with_samples": (
                    openmdao_summary["success"]
                    and split_error <= split_tolerance
                    and energy_error <= energy_tolerance
                ),
                "openmdao": openmdao_summary,
                "split_error": split_error,
                "energy_error": energy_error,
            }
        )

    return {
        "success": all(run["success"] for run in runs),
        "agrees_with_samples": all(run["agrees_with_samples"] for run in runs),
        "runs": runs,
        "best_fast_python_sample": best_sample,
        "fast_python_samples": samples,
    }


def validate_lift_to_drag_demo_against_fast_python_samples(
    lift_to_drag_initial=10.0,
    lift_to_drag_lower=5.0,
    lift_to_drag_upper=25.0,
    sample_count=9,
    lift_to_drag_tolerance=1.0e-4,
    energy_tolerance=1.0e-4,
):
    """Compare cruise-L/D OpenMDAO optimization against FAST-Python samples.

    Inputs:
        lift_to_drag_initial: Initial OpenMDAO cruise lift-to-drag ratio.
        lift_to_drag_lower: Lower cruise lift-to-drag bound.
        lift_to_drag_upper: Upper cruise lift-to-drag bound.
        sample_count: Number of FAST-Python point evaluations.
        lift_to_drag_tolerance: Allowed optimized-L/D mismatch.
        energy_tolerance: Allowed objective mismatch in joules.

    Outputs:
        Dictionary containing the OpenMDAO optimum, sampled FAST-Python points,
        best sampled point, and boolean agreement flags.
    """

    openmdao_summary = run_lift_to_drag_demo(
        lift_to_drag_initial=lift_to_drag_initial,
        lift_to_drag_lower=lift_to_drag_lower,
        lift_to_drag_upper=lift_to_drag_upper,
    )
    samples = sample_fast_python_lift_to_drag_sweep(
        lift_to_drag_lower=lift_to_drag_lower,
        lift_to_drag_upper=lift_to_drag_upper,
        sample_count=sample_count,
    )
    best_sample = best_sampled_point(samples)
    lift_to_drag_error = abs(
        openmdao_summary["cruise_lift_to_drag"]
        - best_sample["cruise_lift_to_drag"]
    )
    energy_error = abs(openmdao_summary["energy_used"] - best_sample["energy_used"])

    return {
        "success": openmdao_summary["success"],
        "agrees_with_samples": (
            openmdao_summary["success"]
            and lift_to_drag_error <= lift_to_drag_tolerance
            and energy_error <= energy_tolerance
        ),
        "openmdao": openmdao_summary,
        "best_fast_python_sample": best_sample,
        "fast_python_samples": samples,
        "lift_to_drag_error": lift_to_drag_error,
        "energy_error": energy_error,
    }


def cli():
    """Parse command-line arguments and run the compact example."""

    parser = argparse.ArgumentParser(
        description="Run a compact FAST-Python OpenMDAO optimization smoke example."
    )
    parser.add_argument("--range-initial", type=float, default=20000.0)
    parser.add_argument("--range-lower", type=float, default=10000.0)
    parser.add_argument("--range-upper", type=float, default=40000.0)
    parser.add_argument("--validate-samples", type=int, default=0)
    args = parser.parse_args()

    if args.validate_samples:
        summary = validate_demo_against_fast_python_samples(
            range_initial=args.range_initial,
            range_lower=args.range_lower,
            range_upper=args.range_upper,
            sample_count=args.validate_samples,
        )
        openmdao_summary = summary["openmdao"]
        best_sample = summary["best_fast_python_sample"]
        print(f"Success: {summary['success']}")
        print(f"Agrees with FAST-Python samples: {summary['agrees_with_samples']}")
        print(f"OpenMDAO mission range: {openmdao_summary['mission_range']:.6f} m")
        print(f"OpenMDAO energy used: {openmdao_summary['energy_used']:.6f} J")
        print(f"Best sampled mission range: {best_sample['mission_range']:.6f} m")
        print(f"Best sampled energy used: {best_sample['energy_used']:.6f} J")
        return

    summary = run_demo(
        range_initial=args.range_initial,
        range_lower=args.range_lower,
        range_upper=args.range_upper,
    )

    print(f"Success: {summary['success']}")
    print(f"Mission range: {summary['mission_range']:.6f} m")
    print(f"MTOW: {summary['mtow']:.6f} kg")
    print(f"Energy used: {summary['energy_used']:.6f} J")


def make_compact_aircraft():
    """Return a compact all-electric aircraft for FAST-Python smoke runs."""

    return {
        "Settings": {
            "ClbPoints": 3,
            "CrsPoints": 3,
            "DesPoints": 3,
            "Analysis": {
                "Type": -2,
                "MaxIter": 3,
            },
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
                "MaxPax": 0,
            },
            "Performance": {
                "Vels": {
                    "Tko": 100,
                },
                "RCMax": 1000,
                "Range": 20000,
            },
            "Aero": {
                "L_D": {
                    "Clb": 10,
                    "Crs": 10,
                    "Des": 10,
                },
                "W_S": {
                    "SLS": 10,
                },
            },
            "Weight": {
                "MTOW": 1000,
                "OEW": 800,
                "Crew": 0,
                "Payload": 0,
                "Fuel": 0,
                "Batt": 200,
            },
            "Power": {
                "SLS": 1000000,
                "SpecEnergy": {
                    "Fuel": 43200000,
                    "Batt": 1000000,
                },
                "Eta": {
                    "EM": 1,
                    "EG": 1,
                    "Propeller": 1,
                },
                "P_W": {
                    "SLS": 1,
                    "EM": 1000000,
                    "EG": 1000000,
                },
                "LamDwn": {
                    "SLS": 0,
                    "Clb": 0,
                    "Crs": 0,
                    "Des": 0,
                },
                "LamUps": {
                    "SLS": 0,
                    "Clb": 0,
                    "Crs": 0,
                    "Des": 0,
                },
                "Battery": {
                    "SerCells": float("nan"),
                    "ParCells": float("nan"),
                },
            },
            "Propulsion": {
                "NumEngines": 1,
                "SLSPower": [1000000, 1000000],
                "SLSThrust": [0, 0],
                "PropArch": {
                    "Type": "E",
                },
            },
        },
        "Mission": {
            "Profile": make_compact_mission(),
        },
    }


def make_compact_hybrid_aircraft():
    """Return a compact series-hybrid aircraft for split smoke runs."""

    aircraft = make_compact_aircraft()
    aircraft["Specs"]["Propulsion"]["PropArch"]["Type"] = "SHE"
    aircraft["Specs"]["Propulsion"]["NumEngines"] = 2
    aircraft["Specs"]["Propulsion"]["Engine"] = compact_turboprop_engine_spec()
    aircraft["Specs"]["Weight"]["Fuel"] = 200.0
    aircraft["Specs"]["Weight"]["Batt"] = 200.0
    aircraft["Specs"]["Power"]["Eta"]["GT"] = 0.4
    aircraft["Specs"]["Power"]["Eta"]["EM"] = 0.95
    aircraft["Specs"]["Power"]["Eta"]["EG"] = 0.92
    aircraft["Specs"]["Power"]["Eta"]["Propeller"] = 0.85
    aircraft["Specs"]["Power"]["LamDwn"]["SLS"] = 0.0
    aircraft["Specs"]["Power"]["LamDwn"]["Clb"] = 0.0
    aircraft["Specs"]["Power"]["LamDwn"]["Crs"] = 0.0
    aircraft["Specs"]["Power"]["LamDwn"]["Des"] = 0.0
    aircraft["HistData"] = {
        "Eng": {
            "E1": {
                "Power_SLS": 100000.0,
                "DryWeight": 10.0,
            },
            "E2": {
                "Power_SLS": 500000.0,
                "DryWeight": 50.0,
            },
            "E3": {
                "Power_SLS": 1000000.0,
                "DryWeight": 100.0,
            },
        },
    }
    return aircraft


def compact_turboprop_engine_spec():
    """Return a compact turboprop engine spec for hybrid smoke runs."""

    return {
        "Mach": 0.05,
        "Alt": 0,
        "OPR": 15,
        "Tt4Max": 1200,
        "ReqPower": 3.0e6,
        "NPR": 1.3,
        "NoSpools": 2,
        "RPMs": [15000, 12000],
        "EtaPoly": {
            "Inlet": 0.99,
            "Diffusers": 0.99,
            "Compressors": 0.9,
            "Combustor": 0.98,
            "Turbines": 0.9,
            "Nozzles": 0.985,
        },
    }


def make_compact_mission():
    """Return a compact mission profile that runs quickly in FAST-Python."""

    return {
        "Segs": ["Climb", "Cruise", "Descent"],
        "ID": [1, 1, 1],
        "Target": {
            "Valu": [20000],
            "Type": ["Dist"],
        },
        "AltBeg": [0, 1000, 1000],
        "AltEnd": [1000, 1000, 0],
        "VelBeg": [100, 100, 100],
        "VelEnd": [100, 100, 100],
        "TypeBeg": ["TAS", "TAS", "TAS"],
        "TypeEnd": ["TAS", "TAS", "TAS"],
        "ClbRate": [10, float("nan"), -10],
    }


if __name__ == "__main__":
    cli()
