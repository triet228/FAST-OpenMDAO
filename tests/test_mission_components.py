# tests/test_mission_components.py

"""Tests for OpenMDAO FAST mission primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    CruiseBreguetDetailedBattery,
    CruiseBreguetEfficiencyTriplet,
    CruiseBreguetPowerHistory,
    CruiseBreguetPowerSplit,
    CruiseBreguetPropulsiveEfficiency,
    CruiseBreguetSourceDelta,
    CruiseBreguetSourceEnergy,
    CruiseSegmentKinematicsPower,
    CruiseTimeTargetDistance,
    DetailedTakeoffSegmentKinematicsPower,
    FlightConditions,
    HistoryMatrixSlice,
    HistoryVectorSlice,
    InitialEnergyRemaining,
    LandingSegmentKinematicsPower,
    PrescribedRateSegmentKinematicsPower,
    RowMatrix,
    TakeoffSegmentKinematics,
)
from fast_python.data_struct import init_mission_history
from fast_python.mission import (
    assign_history_matrix,
    assign_history_vector,
    compute_flight_conditions,
    cruise_breguet_apply_source_delta,
    cruise_breguet_discharge_battery,
    cruise_breguet_efficiency_triplet,
    cruise_breguet_power_history,
    cruise_breguet_power_split,
    cruise_breguet_propulsive_efficiency,
    cruise_breguet_source_energy,
    cruise_time_target_to_distance,
    eval_climb,
    eval_cruise,
    eval_descent,
    eval_detailed_takeoff,
    eval_landing,
    eval_takeoff,
    initial_energy_remaining,
    row_matrix,
)


def test_flight_conditions_match_fast_python_for_velocity_types():
    """Check flight-condition parity for TAS, EAS, and Mach inputs."""

    cases = [
        ("TAS", 100.0),
        ("EAS", 95.0),
        ("Mach", 0.35),
    ]

    for velocity_type, velocity in cases:
        problem = om.Problem()
        problem.model.add_subsystem(
            "conditions",
            FlightConditions(velocity_type=velocity_type),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("altitude", 1000.0, units="m")
        problem.set_val("disa", 5.0, units="K")
        problem.set_val("velocity", velocity)
        problem.run_model()

        expected = compute_flight_conditions(1000.0, 5.0, velocity_type, velocity)

        for name, expected_value in zip(output_names(), expected):
            assert np.isclose(problem.get_val(name)[0], expected_value)


def test_flight_conditions_declares_analytic_partials():
    """Check flight-condition analytical partials against finite difference."""

    for velocity_type, velocity in [("TAS", 100.0), ("EAS", 95.0), ("Mach", 0.35)]:
        problem = om.Problem()
        problem.model.add_subsystem(
            "conditions",
            FlightConditions(velocity_type=velocity_type),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("altitude", 1000.0, units="m")
        problem.set_val("disa", 5.0, units="K")
        problem.set_val("velocity", velocity)
        problem.run_model()

        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-4,
        )

        for partial_data in partials["conditions"].values():
            assert partial_data["abs error"].forward < 1.0e-5


def test_cruise_time_target_distance_matches_fast_python():
    """Check cruise time target conversion parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "target",
        CruiseTimeTargetDistance(type_begin="TAS", type_end="EAS"),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("altitude_begin", 1000.0, units="m")
    problem.set_val("altitude_end", 2000.0, units="m")
    problem.set_val("velocity_begin", 100.0)
    problem.set_val("velocity_end", 90.0)
    problem.set_val("target_minutes", 10.0, units="min")
    problem.run_model()

    mission = make_cruise_target_mission()

    assert np.isclose(
        problem.get_val("distance", units="m")[0],
        cruise_time_target_to_distance(mission, 0, 10.0),
    )


def test_cruise_time_target_distance_declares_analytic_partials():
    """Check cruise time target conversion partials against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "target",
        CruiseTimeTargetDistance(type_begin="TAS", type_end="EAS"),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("altitude_begin", 1000.0, units="m")
    problem.set_val("altitude_end", 2000.0, units="m")
    problem.set_val("velocity_begin", 100.0)
    problem.set_val("velocity_end", 90.0)
    problem.set_val("target_minutes", 10.0, units="min")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-5,
    )

    for partial_data in partials["target"].values():
        assert partial_data["abs error"].forward < 1.0e-5


def test_takeoff_segment_kinematics_matches_fast_python_eval_takeoff():
    """Check simple EvalTakeoff kinematic kernel parity with FAST-Python."""

    result = eval_takeoff(init_mission_history(make_takeoff_aircraft()))
    history = result["Mission"]["History"]["SI"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "takeoff",
        TakeoffSegmentKinematics(npoint=3, target_velocity_type="TAS"),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("altitude", 0.0, units="m")
    problem.set_val("target_altitude", 0.0, units="m")
    problem.set_val("target_velocity", 100.0)
    problem.set_val("mass", 1000.0, units="kg")
    problem.run_model()

    assert np.allclose(problem.get_val("time", units="s"), history["Performance"]["Time"])
    assert np.allclose(problem.get_val("distance", units="m"), history["Performance"]["Dist"])
    assert np.allclose(problem.get_val("true_airspeed", units="m/s"), history["Performance"]["TAS"])
    assert np.allclose(problem.get_val("equivalent_airspeed", units="m/s"), history["Performance"]["EAS"])
    assert np.allclose(problem.get_val("mach"), history["Performance"]["Mach"])
    assert np.allclose(problem.get_val("density", units="kg/m**3"), history["Performance"]["Rho"])
    assert np.allclose(problem.get_val("rate_of_climb", units="m/s"), history["Performance"]["RC"])
    assert np.allclose(problem.get_val("acceleration", units="m/s**2"), history["Performance"]["Acc"])
    assert np.allclose(problem.get_val("flight_path_angle"), history["Performance"]["FPA"])
    assert np.allclose(problem.get_val("specific_excess_power", units="m/s"), history["Performance"]["Ps"])
    assert np.allclose(problem.get_val("potential_energy", units="J"), history["Energy"]["PE"])
    assert np.allclose(problem.get_val("kinetic_energy", units="J"), history["Energy"]["KE"])


def test_landing_segment_kinematics_power_matches_fast_python_eval_landing():
    """Check EvalLanding kinematic and reverse-power kernel parity."""

    result = eval_landing(init_mission_history(make_landing_aircraft()))
    history = result["Mission"]["History"]["SI"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "landing",
        LandingSegmentKinematicsPower(velocity_type="TAS"),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("initial_distance", 0.0, units="m")
    problem.set_val("initial_time", 0.0, units="s")
    problem.set_val("altitude", 0.0, units="m")
    problem.set_val("landing_velocity", 100.0)
    problem.set_val("mass", 1000.0, units="kg")
    problem.set_val("available_power", [1000.0, 1000.0], units="W")
    problem.run_model()

    assert np.allclose(problem.get_val("time", units="s"), history["Performance"]["Time"])
    assert np.allclose(problem.get_val("distance", units="m"), history["Performance"]["Dist"])
    assert np.allclose(problem.get_val("true_airspeed", units="m/s"), history["Performance"]["TAS"])
    assert np.allclose(problem.get_val("equivalent_airspeed", units="m/s"), history["Performance"]["EAS"])
    assert np.allclose(problem.get_val("mach"), history["Performance"]["Mach"])
    assert np.allclose(problem.get_val("density", units="kg/m**3"), history["Performance"]["Rho"])
    assert np.allclose(problem.get_val("rate_of_climb", units="m/s"), history["Performance"]["RC"])
    assert np.allclose(problem.get_val("acceleration", units="m/s**2"), history["Performance"]["Acc"])
    assert np.allclose(problem.get_val("flight_path_angle"), history["Performance"]["FPA"])
    assert np.allclose(problem.get_val("required_power", units="W"), history["Power"]["Req"])
    assert np.allclose(problem.get_val("specific_excess_power", units="m/s"), history["Performance"]["Ps"])
    assert np.allclose(problem.get_val("potential_energy", units="J"), history["Energy"]["PE"])
    assert np.allclose(problem.get_val("kinetic_energy", units="J"), history["Energy"]["KE"])


def test_detailed_takeoff_segment_kinematics_power_matches_fast_python():
    """Check detailed EvalTakeoff physics kernel parity with FAST-Python."""

    result = eval_detailed_takeoff(init_mission_history(make_detailed_takeoff_aircraft()))
    history = result["Mission"]["History"]["SI"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "takeoff",
        DetailedTakeoffSegmentKinematicsPower(npoint=3, target_velocity_type="TAS"),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("altitude", 0.0, units="m")
    problem.set_val("target_altitude", 0.0, units="m")
    problem.set_val("target_velocity", 100.0)
    problem.set_val("mass", 1000.0, units="kg")
    problem.set_val("wing_loading", 100.0, units="kg/m**2")
    problem.set_val("available_power", [1000000.0, 1000000.0, 1000000.0], units="W")
    problem.run_model()

    assert np.allclose(problem.get_val("time", units="s"), history["Performance"]["Time"])
    assert np.allclose(problem.get_val("distance", units="m"), history["Performance"]["Dist"])
    assert np.allclose(problem.get_val("true_airspeed", units="m/s"), history["Performance"]["TAS"])
    assert np.allclose(problem.get_val("equivalent_airspeed", units="m/s"), history["Performance"]["EAS"])
    assert np.allclose(problem.get_val("mach"), history["Performance"]["Mach"])
    assert np.allclose(problem.get_val("density", units="kg/m**3"), history["Performance"]["Rho"])
    assert np.allclose(problem.get_val("lift_drag"), history["Performance"]["LD"])
    assert np.allclose(problem.get_val("rate_of_climb", units="m/s"), history["Performance"]["RC"])
    assert np.allclose(problem.get_val("acceleration", units="m/s**2"), history["Performance"]["Acc"])
    assert np.allclose(problem.get_val("flight_path_angle"), history["Performance"]["FPA"])
    assert np.allclose(problem.get_val("specific_excess_power", units="m/s"), history["Performance"]["Ps"])
    assert np.allclose(problem.get_val("potential_energy", units="J"), history["Energy"]["PE"])
    assert np.allclose(problem.get_val("kinetic_energy", units="J"), history["Energy"]["KE"])


def test_ground_segment_kernels_declare_analytic_partials():
    """Check takeoff and landing ground-segment derivatives."""

    takeoff = om.Problem()
    takeoff.model.add_subsystem(
        "takeoff",
        TakeoffSegmentKinematics(npoint=4, target_velocity_type="EAS"),
        promotes=["*"],
    )
    takeoff.setup()
    takeoff.set_val("altitude", 100.0, units="m")
    takeoff.set_val("target_altitude", 200.0, units="m")
    takeoff.set_val("target_velocity", 95.0)
    takeoff.set_val("mass", 980.0, units="kg")
    takeoff.run_model()

    landing = om.Problem()
    landing.model.add_subsystem(
        "landing",
        LandingSegmentKinematicsPower(velocity_type="EAS"),
        promotes=["*"],
    )
    landing.setup()
    landing.set_val("initial_distance", 500.0, units="m")
    landing.set_val("initial_time", 80.0, units="s")
    landing.set_val("altitude", 120.0, units="m")
    landing.set_val("landing_velocity", 92.0)
    landing.set_val("mass", 970.0, units="kg")
    landing.set_val("available_power", [1400.0, 900.0], units="W")
    landing.run_model()
    detailed = om.Problem()
    detailed.model.add_subsystem(
        "detailed",
        DetailedTakeoffSegmentKinematicsPower(npoint=4, target_velocity_type="EAS"),
        promotes=["*"],
    )
    detailed.setup()
    detailed.set_val("altitude", 90.0, units="m")
    detailed.set_val("target_altitude", 110.0, units="m")
    detailed.set_val("target_velocity", 88.0)
    detailed.set_val("mass", 980.0, units="kg")
    detailed.set_val("wing_loading", 95.0, units="kg/m**2")
    detailed.set_val(
        "available_power",
        [1000000.0, 980000.0, 960000.0, 940000.0],
        units="W",
    )
    detailed.run_model()

    for problem, subsystem in (
        (takeoff, "takeoff"),
        (landing, "landing"),
        (detailed, "detailed"),
    ):
        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-5,
        )

        for key, partial_data in partials[subsystem].items():
            absolute_error = partial_data["abs error"].forward
            relative_error = partial_data["rel error"].forward
            assert absolute_error < 1.0e-4 or relative_error < 1.0e-5, (
                key,
                absolute_error,
                relative_error,
            )


def test_cruise_breguet_efficiency_triplet_matches_fast_python():
    """Check CruiseBRE efficiency triplet parity for each architecture."""

    eta_prop = 0.84
    eta_em = 0.95
    eta_eg = 0.91
    eta_gt = 0.36

    for architecture in ("AC", "PHE", "SHE", "TE"):
        problem = om.Problem()
        problem.model.add_subsystem(
            "efficiency",
            CruiseBreguetEfficiencyTriplet(architecture=architecture),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("propulsive_efficiency", eta_prop)
        problem.set_val("electric_motor_efficiency", eta_em)
        problem.set_val("electric_generator_efficiency", eta_eg)
        problem.set_val("gas_turbine_efficiency", eta_gt)
        problem.run_model()

        expected = cruise_breguet_efficiency_triplet(
            architecture,
            eta_prop,
            eta_em,
            eta_eg,
            eta_gt,
        )

        for name, expected_value in zip(("eta1", "eta2", "eta3"), expected):
            assert np.isclose(problem.get_val(name)[0], expected_value)


def test_cruise_breguet_efficiency_triplet_declares_analytic_partials():
    """Check CruiseBRE efficiency triplet derivatives by architecture."""

    for architecture in ("AC", "PHE", "SHE", "TE"):
        problem = om.Problem()
        problem.model.add_subsystem(
            "efficiency",
            CruiseBreguetEfficiencyTriplet(architecture=architecture),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("propulsive_efficiency", 0.84)
        problem.set_val("electric_motor_efficiency", 0.95)
        problem.set_val("electric_generator_efficiency", 0.91)
        problem.set_val("gas_turbine_efficiency", 0.36)
        problem.run_model()

        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-6,
        )

        for partial_data in partials["efficiency"].values():
            assert partial_data["abs error"].forward < 1.0e-8


def test_cruise_breguet_selectors_match_fast_python():
    """Check CruiseBRE propulsive efficiency and power split selectors."""

    efficiency_problem = om.Problem()
    efficiency_problem.model.add_subsystem(
        "selector",
        CruiseBreguetPropulsiveEfficiency(source="propulsion"),
        promotes=["*"],
    )
    efficiency_problem.setup()
    efficiency_problem.set_val("propulsion_propulsive_efficiency", 0.84)
    efficiency_problem.set_val("power_propeller_efficiency", 0.82)
    efficiency_problem.run_model()

    assert np.isclose(
        efficiency_problem.get_val("propulsive_efficiency")[0],
        cruise_breguet_propulsive_efficiency(
            {
                "Propulsion": {"Eta": {"Prop": 0.84}},
                "Power": {"Eta": {"Propeller": 0.82}},
            }
        ),
    )

    split_problem = om.Problem()
    split_problem.model.add_subsystem(
        "selector",
        CruiseBreguetPowerSplit(source="phi"),
        promotes=["*"],
    )
    split_problem.setup()
    split_problem.set_val("phi_cruise", 0.3)
    split_problem.set_val("lambda_down_cruise", 0.2)
    split_problem.run_model()

    assert np.isclose(
        split_problem.get_val("power_split")[0],
        cruise_breguet_power_split(
            {
                "Power": {
                    "Phi": {"Crs": 0.3},
                    "LamDwn": {"Crs": 0.2},
                },
            }
        ),
    )


def test_cruise_breguet_selectors_declare_analytic_partials():
    """Check CruiseBRE selector derivatives by fixed source path."""

    cases = [
        (
            "propulsion_efficiency",
            CruiseBreguetPropulsiveEfficiency(source="propulsion"),
            {
                "propulsion_propulsive_efficiency": 0.84,
                "power_propeller_efficiency": 0.82,
            },
        ),
        (
            "power_efficiency",
            CruiseBreguetPropulsiveEfficiency(source="power"),
            {
                "propulsion_propulsive_efficiency": 0.84,
                "power_propeller_efficiency": 0.82,
            },
        ),
        (
            "phi_split",
            CruiseBreguetPowerSplit(source="phi"),
            {
                "phi_cruise": 0.3,
                "lambda_down_cruise": 0.2,
            },
        ),
        (
            "lambda_split",
            CruiseBreguetPowerSplit(source="lambda_down"),
            {
                "phi_cruise": 0.3,
                "lambda_down_cruise": 0.2,
            },
        ),
    ]

    for name, component, values in cases:
        problem = om.Problem()
        problem.model.add_subsystem(name, component, promotes=["*"])
        problem.setup()

        for variable, value in values.items():
            problem.set_val(variable, value)

        problem.run_model()
        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-6,
        )

        for partial_data in partials[name].values():
            assert partial_data["abs error"].forward < 1.0e-10


def test_cruise_breguet_power_history_matches_fast_python():
    """Check non-detailed CruiseBRE power-history parity by architecture."""

    values = make_breguet_power_history_values()

    for architecture in ("AC", "PHE", "SHE", "TE", "PE", "E"):
        problem = om.Problem()
        problem.model.add_subsystem(
            "history",
            CruiseBreguetPowerHistory(
                architecture=architecture,
                npoint=values["npoint"],
            ),
            promotes=["*"],
        )
        problem.setup()

        for name, value in values["inputs"].items():
            problem.set_val(name, value)

        problem.run_model()
        expected = fast_python_breguet_power_history(architecture, values)

        for output_name, expected_value in expected.items():
            assert np.allclose(problem.get_val(output_name), expected_value)


def test_cruise_breguet_power_history_declares_analytic_partials():
    """Check CruiseBRE power-history derivatives for smooth branches."""

    values = make_breguet_power_history_values()

    for architecture in ("AC", "PHE", "SHE", "TE", "PE", "E"):
        problem = om.Problem()
        problem.model.add_subsystem(
            "history",
            CruiseBreguetPowerHistory(
                architecture=architecture,
                npoint=values["npoint"],
            ),
            promotes=["*"],
        )
        problem.setup()

        for name, value in values["inputs"].items():
            problem.set_val(name, value)

        problem.run_model()
        partials = problem.check_partials(
            out_stream=None,
            method="fd",
            form="central",
            step=1.0e-6,
            step_calc="rel",
        )

        for key, partial_data in partials["history"].items():
            absolute_error = partial_data["abs error"].forward
            relative_error = partial_data["rel error"].forward
            assert absolute_error < 2.0e-2 or relative_error < 1.0e-2, (
                architecture,
                key,
                absolute_error,
                relative_error,
            )


def test_cruise_segment_kinematics_power_matches_fast_python_eval_cruise():
    """Check smooth EvalCruise kernel parity with FAST-Python."""

    aircraft = init_mission_history(make_smooth_cruise_aircraft())
    result = eval_cruise(aircraft)
    history = result["Mission"]["History"]["SI"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "cruise",
        CruiseSegmentKinematicsPower(npoint=3),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("initial_distance", 0.0, units="m")
    problem.set_val("initial_time", 0.0, units="s")
    problem.set_val("altitude", [0.0, 0.0, 0.0], units="m")
    problem.set_val("true_airspeed", [100.0, 100.0, 100.0], units="m/s")
    problem.set_val("mass", [1000.0, 1000.0, 1000.0], units="kg")
    problem.set_val("available_power", [200000.0, 200000.0, 200000.0], units="W")
    problem.set_val("target_distance", 1000.0, units="m")
    problem.set_val("lift_drag", 10.0)
    problem.run_model()

    assert np.allclose(problem.get_val("distance", units="m"), history["Performance"]["Dist"])
    assert np.allclose(problem.get_val("time", units="s"), history["Performance"]["Time"])
    assert np.allclose(problem.get_val("rate_of_climb", units="m/s"), history["Performance"]["RC"])
    assert np.allclose(problem.get_val("acceleration", units="m/s**2"), history["Performance"]["Acc"])
    assert np.allclose(problem.get_val("flight_path_angle"), history["Performance"]["FPA"])
    assert np.allclose(problem.get_val("required_power", units="W"), history["Power"]["Req"])
    assert np.allclose(problem.get_val("specific_excess_power", units="m/s"), history["Performance"]["Ps"])
    assert np.allclose(problem.get_val("potential_energy", units="J"), history["Energy"]["PE"])
    assert np.allclose(problem.get_val("kinetic_energy", units="J"), history["Energy"]["KE"])
    assert np.allclose(problem.get_val("time_step", units="s"), [5.0, 5.0])
    assert np.allclose(problem.get_val("drag_power", units="W"), history["Power"]["Req"])


def test_cruise_segment_kinematics_power_declares_analytic_partials():
    """Check smooth EvalCruise kernel derivatives against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "cruise",
        CruiseSegmentKinematicsPower(npoint=4),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("initial_distance", 10.0, units="m")
    problem.set_val("initial_time", 20.0, units="s")
    problem.set_val("altitude", [100.0, 140.0, 170.0, 210.0], units="m")
    problem.set_val("true_airspeed", [90.0, 95.0, 100.0, 105.0], units="m/s")
    problem.set_val("mass", [1000.0, 990.0, 980.0, 970.0], units="kg")
    problem.set_val(
        "available_power",
        [200000.0, 202000.0, 204000.0, 206000.0],
        units="W",
    )
    problem.set_val("target_distance", 1010.0, units="m")
    problem.set_val("lift_drag", 12.0)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-5,
    )

    for key, partial_data in partials["cruise"].items():
        absolute_error = partial_data["abs error"].forward
        relative_error = partial_data["rel error"].forward
        assert absolute_error < 1.0e-4 or relative_error < 1.0e-5, (
            key,
            absolute_error,
            relative_error,
        )


def test_prescribed_rate_segment_kinematics_power_matches_fast_python():
    """Check prescribed-rate EvalClimb and EvalDescent kernel parity."""

    cases = (
        (
            "climb",
            make_prescribed_climb_aircraft(),
            eval_climb,
            False,
            [0.0, 500.0, 1000.0],
            [10.0, 10.0, 0.0],
            [1000000.0, 1000000.0, 1000000.0],
        ),
        (
            "descent",
            make_prescribed_descent_aircraft(),
            eval_descent,
            True,
            [1000.0, 500.0, 0.0],
            [-10.0, -10.0, 0.0],
            [1000000.0, 1000000.0, 1000000.0],
        ),
    )

    for name, aircraft, evaluator, idle_floor, altitude, rate, available_power in cases:
        result = evaluator(init_mission_history(aircraft))
        history = result["Mission"]["History"]["SI"]
        problem = om.Problem()
        problem.model.add_subsystem(
            name,
            PrescribedRateSegmentKinematicsPower(npoint=3, idle_floor=idle_floor),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("initial_distance", 0.0, units="m")
        problem.set_val("initial_time", 0.0, units="s")
        problem.set_val("altitude", altitude, units="m")
        problem.set_val("true_airspeed", [100.0, 100.0, 100.0], units="m/s")
        problem.set_val("mass", [1000.0, 1000.0, 1000.0], units="kg")
        problem.set_val("available_power", available_power, units="W")
        problem.set_val("rate_of_climb", rate, units="m/s")
        problem.set_val("lift_drag", 10.0)
        problem.run_model()

        assert np.allclose(problem.get_val("distance", units="m"), history["Performance"]["Dist"])
        assert np.allclose(problem.get_val("time", units="s"), history["Performance"]["Time"])
        assert np.allclose(problem.get_val("acceleration", units="m/s**2"), history["Performance"]["Acc"])
        assert np.allclose(problem.get_val("flight_path_angle"), history["Performance"]["FPA"])
        assert np.allclose(problem.get_val("required_power", units="W"), history["Power"]["Req"])
        assert np.allclose(problem.get_val("specific_excess_power", units="m/s"), history["Performance"]["Ps"])
        assert np.allclose(problem.get_val("potential_energy", units="J"), history["Energy"]["PE"])
        assert np.allclose(problem.get_val("kinetic_energy", units="J"), history["Energy"]["KE"])
        assert np.allclose(problem.get_val("time_step", units="s"), [50.0, 50.0])
        assert np.allclose(problem.get_val("drag_power", units="W"), expected_drag_power(rate))


def test_prescribed_rate_segment_kinematics_power_declares_analytic_partials():
    """Check prescribed-rate segment derivatives against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "segment",
        PrescribedRateSegmentKinematicsPower(npoint=4, idle_floor=False),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("initial_distance", 10.0, units="m")
    problem.set_val("initial_time", 20.0, units="s")
    problem.set_val("altitude", [100.0, 160.0, 230.0, 310.0], units="m")
    problem.set_val("true_airspeed", [90.0, 94.0, 98.0, 103.0], units="m/s")
    problem.set_val("mass", [1000.0, 990.0, 980.0, 970.0], units="kg")
    problem.set_val(
        "available_power",
        [220000.0, 222000.0, 224000.0, 226000.0],
        units="W",
    )
    problem.set_val("rate_of_climb", [8.0, 9.0, 10.0, 0.0], units="m/s")
    problem.set_val("lift_drag", 12.0)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-5,
    )

    for key, partial_data in partials["segment"].items():
        absolute_error = partial_data["abs error"].forward
        relative_error = partial_data["rel error"].forward
        assert absolute_error < 1.0e-4 or relative_error < 1.0e-5, (
            key,
            absolute_error,
            relative_error,
        )


def test_cruise_breguet_detailed_battery_matches_fast_python():
    """Check detailed CruiseBRE battery discharge parity."""

    for architecture, values in (
        ("E", make_breguet_detailed_battery_values(False)),
        ("PHE", make_breguet_detailed_battery_values(True)),
    ):
        problem = om.Problem()
        problem.model.add_subsystem(
            "battery",
            CruiseBreguetDetailedBattery(
                architecture=architecture,
                npoint=values["npoint"],
            ),
            promotes=["*"],
        )
        problem.setup()
        set_breguet_detailed_battery_values(problem, values)
        problem.run_model()

        expected = fast_python_breguet_detailed_battery(architecture, values)
        assert np.allclose(
            problem.get_val("adjusted_battery_power", units="W"),
            expected["adjusted_battery_power"],
        )
        assert np.allclose(problem.get_val("soc"), expected["soc"])
        assert np.allclose(
            problem.get_val("adjusted_phi_history"),
            expected["adjusted_phi_history"],
        )
        assert np.isclose(problem.get_val("soc_off")[0], expected["soc_off"])


def test_cruise_breguet_detailed_battery_declares_analytic_partials():
    """Check detailed CruiseBRE battery derivatives on a smooth branch."""

    values = make_breguet_detailed_battery_values(False)
    problem = om.Problem()
    problem.model.add_subsystem(
        "battery",
        CruiseBreguetDetailedBattery(
            architecture="E",
            npoint=values["npoint"],
        ),
        promotes=["*"],
    )
    problem.setup()
    set_breguet_detailed_battery_values(problem, values)
    problem.run_model()
    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for key, partial_data in partials["battery"].items():
        absolute_error = partial_data["abs error"].forward
        relative_error = partial_data["rel error"].forward
        assert absolute_error < 1.0e-3 or relative_error < 1.0e-5, (
            key,
            absolute_error,
            relative_error,
        )


def test_cruise_breguet_source_energy_matches_fast_python():
    """Check CruiseBRE source-energy allocation parity with FAST-Python."""

    values = make_breguet_source_energy_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "source",
        CruiseBreguetSourceEnergy(src_type=values["src_type"], npoint=4),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("initial_source_energy", values["initial_energy"], units="J")
    problem.set_val(
        "initial_source_energy_left",
        values["initial_energy_left"],
        units="J",
    )
    problem.set_val("fuel_energy", values["fuel_energy"], units="J")
    problem.set_val("battery_energy", values["battery_energy"], units="J")
    problem.run_model()

    expected_energy, expected_left = cruise_breguet_source_energy(
        values["specs"],
        values["history"],
        1,
        4,
        values["fuel_energy"],
        values["battery_energy"],
    )

    assert np.allclose(problem.get_val("source_energy", units="J"), expected_energy)
    assert np.allclose(problem.get_val("source_energy_left", units="J"), expected_left)


def test_cruise_breguet_source_energy_declares_analytic_partials():
    """Check CruiseBRE source-energy allocation derivatives."""

    values = make_breguet_source_energy_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "source",
        CruiseBreguetSourceEnergy(src_type=values["src_type"], npoint=4),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("initial_source_energy", values["initial_energy"], units="J")
    problem.set_val(
        "initial_source_energy_left",
        values["initial_energy_left"],
        units="J",
    )
    problem.set_val("fuel_energy", values["fuel_energy"], units="J")
    problem.set_val("battery_energy", values["battery_energy"], units="J")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-3,
    )

    for partial_data in partials["source"].values():
        assert partial_data["abs error"].forward < 1.0e-5


def test_cruise_breguet_source_delta_matches_fast_python():
    """Check aggregate source-energy delta parity with FAST-Python."""

    source_energy = np.asarray(
        [
            [100.0, 20.0, 40.0],
            [105.0, 25.0, 45.0],
            [110.0, 30.0, 50.0],
            [115.0, 35.0, 55.0],
        ],
    )
    source_energy_left = np.asarray(
        [
            [900.0, 500.0, 700.0],
            [895.0, 495.0, 695.0],
            [890.0, 490.0, 690.0],
            [885.0, 485.0, 685.0],
        ],
    )
    delta = np.asarray([0.0, 60.0, 120.0, 180.0])
    expected_energy = source_energy.copy()
    expected_left = source_energy_left.copy()
    cruise_breguet_apply_source_delta(expected_energy, expected_left, [1, 2], delta)

    problem = om.Problem()
    problem.model.add_subsystem(
        "delta",
        CruiseBreguetSourceDelta(columns=(1, 2), npoint=4, nsrc=3),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("source_energy", source_energy, units="J")
    problem.set_val("source_energy_left", source_energy_left, units="J")
    problem.set_val("delta", delta, units="J")
    problem.run_model()

    assert np.allclose(
        problem.get_val("updated_source_energy", units="J"),
        expected_energy,
    )
    assert np.allclose(
        problem.get_val("updated_source_energy_left", units="J"),
        expected_left,
    )


def test_cruise_breguet_source_delta_declares_analytic_partials():
    """Check aggregate source-energy delta derivatives."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "delta",
        CruiseBreguetSourceDelta(columns=(1, 2), npoint=4, nsrc=3),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val(
        "source_energy",
        np.arange(12.0).reshape(4, 3) + 100.0,
        units="J",
    )
    problem.set_val(
        "source_energy_left",
        np.arange(12.0).reshape(4, 3) + 500.0,
        units="J",
    )
    problem.set_val("delta", np.asarray([0.0, 60.0, 120.0, 180.0]), units="J")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-3,
    )

    for partial_data in partials["delta"].values():
        assert partial_data["abs error"].forward < 1.0e-5


def test_initial_energy_remaining_matches_fast_python():
    """Check first-segment source energy initialization parity."""

    values = make_initial_energy_remaining_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "initial_energy",
        InitialEnergyRemaining(
            src_type=values["src_type"],
            npoint=values["npoint"],
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val(
        "fuel_specific_energy",
        values["fuel_specific_energy"],
        units="J/kg",
    )
    problem.set_val(
        "battery_specific_energy",
        values["battery_specific_energy"],
        units="J/kg",
    )
    problem.set_val("fuel_weight", values["fuel_weight"], units="kg")
    problem.set_val("battery_weight", values["battery_weight"], units="kg")
    problem.run_model()

    expected = initial_energy_remaining(values["specs"], values["npoint"])

    assert np.allclose(problem.get_val("source_energy_left", units="J"), expected)


def test_initial_energy_remaining_declares_analytic_partials():
    """Check source energy initialization derivatives."""

    values = make_initial_energy_remaining_values()
    problem = om.Problem()
    problem.model.add_subsystem(
        "initial_energy",
        InitialEnergyRemaining(
            src_type=values["src_type"],
            npoint=values["npoint"],
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val(
        "fuel_specific_energy",
        values["fuel_specific_energy"],
        units="J/kg",
    )
    problem.set_val(
        "battery_specific_energy",
        values["battery_specific_energy"],
        units="J/kg",
    )
    problem.set_val("fuel_weight", values["fuel_weight"], units="kg")
    problem.set_val("battery_weight", values["battery_weight"], units="kg")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-5,
    )

    for partial_data in partials["initial_energy"].values():
        assert partial_data["abs error"].forward < 1.0e-5


def test_row_matrix_matches_fast_python():
    """Check row-wise history expansion parity with FAST-Python."""

    value = np.asarray([0.25, 0.5, 0.75])
    problem = om.Problem()
    problem.model.add_subsystem(
        "row_matrix",
        RowMatrix(rows=4, value_size=3),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("value", value)
    problem.run_model()

    assert np.allclose(problem.get_val("matrix"), row_matrix(value, 4))


def test_row_matrix_declares_analytic_partials():
    """Check row-wise history expansion derivatives."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "row_matrix",
        RowMatrix(rows=4, value_size=3),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("value", np.asarray([0.25, 0.5, 0.75]))
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["row_matrix"].values():
        assert partial_data["abs error"].forward < 1.0e-8


def test_history_vector_slice_matches_fast_python():
    """Check fixed history-vector slice assignment parity with FAST-Python."""

    history = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    values = np.asarray([20.0, 30.0])
    start = 2
    stop = 4

    problem = om.Problem()
    problem.model.add_subsystem(
        "history_slice",
        HistoryVectorSlice(history_size=5, start=start, stop=stop),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("history_vector", history)
    problem.set_val("values", values)
    problem.run_model()

    section = {"Power": history.tolist()}
    assign_history_vector(section, "Power", values, start, stop)

    assert np.allclose(
        problem.get_val("updated_history_vector"),
        np.asarray(section["Power"]),
    )


def test_history_vector_slice_declares_analytic_partials():
    """Check fixed history-vector slice assignment derivatives."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "history_slice",
        HistoryVectorSlice(history_size=5, start=2, stop=4),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("history_vector", np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]))
    problem.set_val("values", np.asarray([20.0, 30.0]))
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["history_slice"].values():
        assert partial_data["abs error"].forward < 1.0e-9


def test_history_matrix_slice_matches_fast_python():
    """Check fixed history-matrix row slice assignment parity with FAST-Python."""

    history = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    values = np.asarray(
        [
            [20.0, 30.0],
            [40.0, 50.0],
        ]
    )
    start = 1
    stop = 3

    problem = om.Problem()
    problem.model.add_subsystem(
        "history_slice",
        HistoryMatrixSlice(num_rows=4, num_cols=2, start=start, stop=stop),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("history_matrix", history)
    problem.set_val("values", values)
    problem.run_model()

    section = {"Power": history.tolist()}
    assign_history_matrix(section, "Power", values, start, stop)

    assert np.allclose(
        problem.get_val("updated_history_matrix"),
        np.asarray(section["Power"]),
    )


def test_history_matrix_slice_declares_analytic_partials():
    """Check fixed history-matrix row slice assignment derivatives."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "history_slice",
        HistoryMatrixSlice(num_rows=4, num_cols=2, start=1, stop=3),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val(
        "history_matrix",
        np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        ),
    )
    problem.set_val("values", np.asarray([[20.0, 30.0], [40.0, 50.0]]))
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-6,
    )

    for partial_data in partials["history_slice"].values():
        assert partial_data["abs error"].forward < 1.0e-9


def output_names():
    """Return output names in FAST-Python flight-condition order."""

    return (
        "eas",
        "tas",
        "mach",
        "temperature",
        "pressure",
        "density",
        "viscosity",
    )


def make_cruise_target_mission():
    """Return minimal mission profile for cruise time-target conversion."""

    return {
        "AltBeg": [1000.0],
        "AltEnd": [2000.0],
        "VelBeg": [100.0],
        "VelEnd": [90.0],
        "TypeBeg": ["TAS"],
        "TypeEnd": ["EAS"],
    }


def make_breguet_source_energy_values():
    """Return source-energy allocation values and FAST-shaped inputs."""

    src_type = np.asarray([1.0, 0.0, 0.0])
    initial_energy = np.asarray([100.0, 20.0, 40.0])
    initial_energy_left = np.asarray([900.0, 500.0, 700.0])
    fuel_energy = np.asarray([10.0, 70.0, 130.0, 190.0])
    battery_energy = np.asarray([30.0, 90.0, 150.0, 210.0])

    return {
        "src_type": src_type,
        "initial_energy": initial_energy,
        "initial_energy_left": initial_energy_left,
        "fuel_energy": fuel_energy,
        "battery_energy": battery_energy,
        "specs": {
            "Propulsion": {
                "PropArch": {
                    "SrcType": src_type,
                },
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 1000.0,
                    "Batt": 1000.0,
                },
            },
            "Weight": {
                "Fuel": 1.0,
                "Batt": [1.0, 1.0],
            },
        },
        "history": {
            "Energy": {
                "E_ES": [
                    [0.0, 0.0, 0.0],
                    initial_energy,
                ],
                "Eleft_ES": [
                    [0.0, 0.0, 0.0],
                    initial_energy_left,
                ],
            },
        },
    }


def make_breguet_power_history_values():
    """Return FAST-shaped CruiseBRE power-history values."""

    return {
        "npoint": 4,
        "inputs": {
            "initial_mass": 18500.0,
            "distance_step": np.asarray([8200.0, 9100.0, 10400.0]),
            "time_step": np.asarray([70.0, 76.0, 83.0]),
            "fuel_specific_energy": 43.2e6,
            "battery_specific_energy": 900000.0,
            "lift_drag": 16.5,
            "propulsive_efficiency": 0.84,
            "electric_motor_efficiency": 0.95,
            "electric_generator_efficiency": 0.91,
            "gas_turbine_efficiency": 0.36,
            "power_split": 0.32,
        },
    }


def make_takeoff_aircraft():
    """Return a minimal all-electric aircraft for simple EvalTakeoff parity."""

    aircraft = make_smooth_cruise_aircraft()
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "Tko": 0,
    }
    aircraft["Specs"]["Power"]["LamUps"] = {
        "Tko": 0,
    }
    aircraft["Specs"]["Propulsion"]["SLSPower"] = [1000]
    aircraft["Specs"]["Weight"]["Batt"] = 100
    aircraft["Mission"]["Profile"] = {
        "SegsID": 1,
        "MissID": 1,
        "SegPts": [3],
        "SegBeg": [1],
        "SegEnd": [3],
        "AltEnd": [0],
        "VelEnd": [100],
        "TypeEnd": ["TAS"],
    }
    return aircraft


def make_landing_aircraft():
    """Return a minimal all-electric aircraft for EvalLanding parity."""

    aircraft = make_takeoff_aircraft()
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "Lnd": 0,
    }
    aircraft["Specs"]["Power"]["LamUps"] = {
        "Lnd": 0,
    }
    aircraft["Mission"]["Profile"] = {
        "SegsID": 1,
        "MissID": 1,
        "SegPts": [2],
        "SegBeg": [1],
        "SegEnd": [2],
        "AltBeg": [0],
        "VelBeg": [100],
        "TypeBeg": ["TAS"],
    }
    return aircraft


def make_detailed_takeoff_aircraft():
    """Return a minimal all-electric aircraft for detailed EvalTakeoff parity."""

    aircraft = make_takeoff_aircraft()
    aircraft["Specs"]["Aero"] = {
        "W_S": {
            "SLS": 100,
        },
    }
    aircraft["Specs"]["Propulsion"]["SLSPower"] = [1000000]
    return aircraft


def make_smooth_cruise_aircraft():
    """Return a minimal all-electric aircraft for smooth EvalCruise parity."""

    return {
        "Settings": {
            "nargOperUps": 0,
            "nargOperDwn": 0,
            "Analysis": {
                "Type": 1,
            },
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
            },
            "Aero": {
                "L_D": {
                    "Crs": 10,
                },
            },
            "Performance": {
                "Alts": {
                    "Tko": 0,
                },
                "RCMax": 1000,
            },
            "Weight": {
                "MTOW": 1000,
                "Batt": 2000,
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": 43200000,
                    "Batt": 1000,
                },
                "LamDwn": {
                    "Crs": 0,
                },
                "LamUps": {
                    "Crs": 0,
                },
                "Battery": {
                    "SerCells": float("nan"),
                    "ParCells": float("nan"),
                },
            },
            "Propulsion": {
                "SLSPower": [200000],
                "SLSThrust": [0],
                "PropArch": {
                    "Type": "E",
                    "Arch": [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    "OperUps": [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    "OperDwn": [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                    ],
                    "EtaUps": [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    "EtaDwn": [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    "SrcType": [0],
                    "TrnType": [0],
                    "ParConns": [[]],
                },
            },
        },
        "Mission": {
            "Profile": {
                "SegsID": 1,
                "MissID": 1,
                "SegPts": [3],
                "SegBeg": [1],
                "SegEnd": [3],
                "AltBeg": [0],
                "AltEnd": [0],
                "VelBeg": [100],
                "VelEnd": [100],
                "TypeBeg": ["TAS"],
                "TypeEnd": ["TAS"],
                "CrsTarget": 1000,
            },
        },
    }


def make_prescribed_climb_aircraft():
    """Return a minimal all-electric aircraft for prescribed EvalClimb parity."""

    aircraft = make_smooth_cruise_aircraft()
    aircraft["Specs"]["Aero"] = {
        "L_D": {
            "Clb": 10,
        },
    }
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "Clb": 0,
    }
    aircraft["Specs"]["Power"]["LamUps"] = {
        "Clb": 0,
    }
    aircraft["Specs"]["Propulsion"]["SLSPower"] = [1000000]
    aircraft["Specs"]["Weight"]["Batt"] = 20000
    aircraft["Mission"]["Profile"] = {
        "SegsID": 1,
        "MissID": 1,
        "SegPts": [3],
        "SegBeg": [1],
        "SegEnd": [3],
        "AltBeg": [0],
        "AltEnd": [1000],
        "VelBeg": [100],
        "VelEnd": [100],
        "TypeBeg": ["TAS"],
        "TypeEnd": ["TAS"],
        "ClbRate": [10],
    }
    return aircraft


def make_prescribed_descent_aircraft():
    """Return a minimal all-electric aircraft for prescribed EvalDescent parity."""

    aircraft = make_smooth_cruise_aircraft()
    aircraft["Specs"]["Aero"] = {
        "L_D": {
            "Des": 10,
        },
    }
    aircraft["Specs"]["Power"]["LamDwn"] = {
        "Des": 0,
    }
    aircraft["Specs"]["Power"]["LamUps"] = {
        "Des": 0,
    }
    aircraft["Specs"]["Propulsion"]["SLSPower"] = [1000000]
    aircraft["Specs"]["Weight"]["Batt"] = 20000
    aircraft["Mission"]["Profile"] = {
        "SegsID": 1,
        "MissID": 1,
        "SegPts": [3],
        "SegBeg": [1],
        "SegEnd": [3],
        "AltBeg": [1000],
        "AltEnd": [0],
        "VelBeg": [100],
        "VelEnd": [100],
        "TypeBeg": ["TAS"],
        "TypeEnd": ["TAS"],
        "ClbRate": [-10],
    }
    return aircraft


def expected_drag_power(rate_of_climb):
    """Return prescribed segment drag power for the fixed parity fixtures."""

    rate_of_climb = np.asarray(rate_of_climb, dtype=float)
    flight_path_angle = np.degrees(np.arcsin(rate_of_climb / 100.0))
    return 1000.0 * 9.81 * np.cos(np.radians(flight_path_angle)) * 100.0 / 10.0


def fast_python_breguet_power_history(architecture, values):
    """Return FAST-Python CruiseBRE outputs in OpenMDAO names."""

    inputs = values["inputs"]
    mass = np.zeros(values["npoint"])
    mass[0] = inputs["initial_mass"]
    soc = np.ones(values["npoint"]) * 100.0
    aircraft = {
        "Specs": {
            "Power": {
                "Battery": {
                    "SerCells": np.nan,
                    "ParCells": np.nan,
                },
            },
        },
    }
    result = cruise_breguet_power_history(
        aircraft,
        architecture,
        mass,
        inputs["distance_step"],
        inputs["time_step"],
        np.ones(values["npoint"]) * 120.0,
        9.81,
        inputs["fuel_specific_energy"],
        inputs["battery_specific_energy"],
        inputs["lift_drag"],
        inputs["propulsive_efficiency"],
        inputs["electric_motor_efficiency"],
        inputs["electric_generator_efficiency"],
        inputs["gas_turbine_efficiency"],
        inputs["power_split"],
        soc,
    )

    return {
        "fuel_power": result[0],
        "battery_power": result[1],
        "propulsor_power": result[2],
        "motor_power": result[3],
        "generator_power": result[4],
        "required_power": result[5],
        "fuel_burn": result[6],
        "fuel_energy": result[7],
        "battery_energy": result[8],
        "phi_history": result[9],
        "mass": mass,
    }


def make_breguet_detailed_battery_values(trigger_depletion):
    """Return FAST-shaped values for detailed CruiseBRE battery discharge."""

    if trigger_depletion:
        battery_power = np.asarray([15000.0, 16000.0, 17000.0, 18000.0])
        initial_soc = 22.0
    else:
        battery_power = np.asarray([900.0, 760.0, 610.0, 480.0])
        initial_soc = 88.0

    return {
        "npoint": 4,
        "battery_power": battery_power,
        "time_step": np.asarray([35.0, 40.0, 45.0]),
        "initial_soc": initial_soc,
        "phi_history": np.asarray([0.35, 0.32, 0.28, 0.25]),
        "parallel_cells": 9.0,
        "series_cells": 84.0,
        "max_cell_voltage": 4.2,
        "internal_resistance": 0.01,
        "exponential_voltage": 0.1,
        "exponential_capacity": 1.0,
        "cap_cell": 2.4,
        "state_of_health": 100.0,
    }


def fast_python_breguet_detailed_battery(architecture, values):
    """Return FAST-Python detailed CruiseBRE battery outputs."""

    aircraft = {
        "Specs": {
            "Battery": {
                "MaxExtVolCell": values["max_cell_voltage"],
                "IntResist": values["internal_resistance"],
                "ExpVol": values["exponential_voltage"],
                "ExpCap": values["exponential_capacity"],
                "CapCell": values["cap_cell"],
            },
            "Power": {
                "Battery": {
                    "ParCells": values["parallel_cells"],
                    "SerCells": values["series_cells"],
                },
            },
        },
        "Mission": {
            "Profile": {
                "MissID": 1,
            },
            "History": {
                "Flags": {
                    "SOCOff": [0],
                },
            },
        },
    }
    soc = np.ones(values["npoint"]) * values["initial_soc"]
    result = cruise_breguet_discharge_battery(
        aircraft,
        values["battery_power"].copy(),
        values["time_step"],
        soc,
        values["phi_history"].copy(),
        architecture,
    )
    return {
        "adjusted_battery_power": result[0],
        "soc": result[1],
        "adjusted_phi_history": result[2],
        "soc_off": aircraft["Mission"]["History"]["Flags"]["SOCOff"][0],
    }


def set_breguet_detailed_battery_values(problem, values):
    """Set OpenMDAO detailed CruiseBRE battery inputs."""

    problem.set_val("battery_power", values["battery_power"], units="W")
    problem.set_val("time_step", values["time_step"], units="s")
    problem.set_val("initial_soc", values["initial_soc"])
    problem.set_val("phi_history", values["phi_history"])
    problem.set_val("parallel_cells", values["parallel_cells"])
    problem.set_val("series_cells", values["series_cells"])
    problem.set_val("max_cell_voltage", values["max_cell_voltage"])
    problem.set_val("internal_resistance", values["internal_resistance"])
    problem.set_val("exponential_voltage", values["exponential_voltage"])
    problem.set_val("exponential_capacity", values["exponential_capacity"])
    problem.set_val("cap_cell", values["cap_cell"])
    problem.set_val("state_of_health", values["state_of_health"])


def make_initial_energy_remaining_values():
    """Return FAST-shaped source energy initialization values."""

    src_type = np.asarray([1.0, 0.0, 0.0])
    fuel_specific_energy = 1200.0
    battery_specific_energy = 900.0
    fuel_weight = np.asarray([12.0])
    battery_weight = np.asarray([10.0, 15.0])
    return {
        "src_type": src_type,
        "npoint": 4,
        "fuel_specific_energy": fuel_specific_energy,
        "battery_specific_energy": battery_specific_energy,
        "fuel_weight": fuel_weight,
        "battery_weight": battery_weight,
        "specs": {
            "Propulsion": {
                "PropArch": {
                    "SrcType": src_type,
                },
            },
            "Power": {
                "SpecEnergy": {
                    "Fuel": fuel_specific_energy,
                    "Batt": battery_specific_energy,
                },
            },
            "Weight": {
                "Fuel": fuel_weight,
                "Batt": battery_weight,
            },
        },
    }
