# src/fast_openmdao/examples/compact_electric.py

"""Compact all-electric FAST-Python optimization example."""

import argparse
import os

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


def cli():
    """Parse command-line arguments and run the compact example."""

    parser = argparse.ArgumentParser(
        description="Run a compact FAST-Python OpenMDAO optimization smoke example."
    )
    parser.add_argument("--range-initial", type=float, default=20000.0)
    parser.add_argument("--range-lower", type=float, default=10000.0)
    parser.add_argument("--range-upper", type=float, default=40000.0)
    args = parser.parse_args()
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
