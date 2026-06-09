# tests/test_constraint_components.py

"""Tests for OpenMDAO FAST constraint primitive components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    CruiseDynamicPressure,
    FAR25ClimbConstraint,
    FAR25EngineGradient,
    JetAEOClimbConstraint,
    JetApproachConstraint,
    JetCruiseConstraint,
    JetFAR25NamedClimbConstraint,
    JetLandingFieldLengthConstraint,
    JetTakeoffFieldLengthConstraint,
    OEIMultiplier,
    PsLossSigmoid,
)
from fast_python.constraint import (
    cruise_dynamic_pressure,
    far25_engine_gradient,
    jet_app,
    jet25_111,
    jet25_119,
    jet25_121a,
    jet25_121b,
    jet25_121c,
    jet25_121d,
    jet_aeo_climb,
    jet_crs,
    jet_div,
    jet_lfl,
    jet_tofl,
    oei_multiplier,
    sigmoid,
)


def test_ps_loss_sigmoid_matches_fast_python():
    """Check PsLoss sigmoid parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "sigmoid",
        PsLossSigmoid(a=10.0, b=2.0, c=0.2, d=1.0),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("ps_loss", 0.3)
    problem.run_model()

    aircraft = make_constraint_aircraft(constraint_type=1, ps_loss=0.3)

    assert np.isclose(
        problem.get_val("sigmoid")[0],
        sigmoid(aircraft, 10.0, 2.0, 0.2, 1.0),
    )


def test_oei_multiplier_matches_fast_python():
    """Check OEI multiplier parity for both FAST modes."""

    engine_problem = om.Problem()
    engine_problem.model.add_subsystem(
        "oei",
        OEIMultiplier(constraint_type=0),
        promotes=["*"],
    )
    engine_problem.setup()
    engine_problem.set_val("num_engines", 4.0)
    engine_problem.run_model()

    assert np.isclose(
        engine_problem.get_val("oei_multiplier")[0],
        oei_multiplier(make_constraint_aircraft(constraint_type=0, num_engines=4)),
    )

    ps_problem = om.Problem()
    ps_problem.model.add_subsystem(
        "oei",
        OEIMultiplier(constraint_type=1),
        promotes=["*"],
    )
    ps_problem.setup()
    ps_problem.set_val("ps_loss", 0.3)
    ps_problem.run_model()

    assert np.isclose(
        ps_problem.get_val("oei_multiplier")[0],
        oei_multiplier(make_constraint_aircraft(constraint_type=1, ps_loss=0.3)),
    )


def test_far25_engine_gradient_matches_fast_python():
    """Check FAR 25 engine-gradient selector parity."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "gradient",
        FAR25EngineGradient(num_engines=3, constraint_type=0),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("two_engine", 0.024)
    problem.set_val("three_engine", 0.027)
    problem.set_val("four_engine", 0.030)
    problem.run_model()

    aircraft = make_constraint_aircraft(constraint_type=0, num_engines=3)

    assert np.isclose(
        problem.get_val("engine_gradient")[0],
        far25_engine_gradient(aircraft, 0.024, 0.027, 0.030),
    )


def test_cruise_dynamic_pressure_matches_fast_python():
    """Check cruise dynamic pressure parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem("cruise", CruiseDynamicPressure(), promotes=["*"])
    problem.setup()
    problem.set_val("altitude", 9000.0, units="m")
    problem.set_val("mach", 0.58)
    problem.run_model()

    expected = cruise_dynamic_pressure(9000.0, 0.58)

    assert np.isclose(problem.get_val("dynamic_pressure")[0], expected[0])
    assert np.isclose(problem.get_val("density_ratio")[0], expected[1])
    assert np.isclose(problem.get_val("velocity", units="ft/s")[0], expected[2])


def test_jet_field_residual_components_match_fast_python():
    """Check JetApp, JetTOFL, and JetLFL component parity."""

    aircraft = make_full_constraint_aircraft()

    approach = om.Problem()
    approach.model.add_subsystem(
        "constraint",
        JetApproachConstraint(
            req_type=1,
            cl_landing=2.4,
            wland_mtow=0.85,
            approach_velocity=70.0,
        ),
        promotes=["*"],
    )
    approach.setup()
    approach.set_val("wing_loading", 400.0, units="kg/m**2")
    approach.set_val("thrust_loading", 0.3)
    approach.run_model()

    assert np.isclose(
        approach.get_val("approach_residual")[0],
        jet_app(400.0, 0.3, aircraft),
    )

    takeoff = om.Problem()
    takeoff.model.add_subsystem(
        "constraint",
        JetTakeoffFieldLengthConstraint(
            aircraft_class="Turbofan",
            cl_takeoff=2.0,
            balanced_field_length=1800.0,
            stall_velocity=60.0,
        ),
        promotes=["*"],
    )
    takeoff.setup()
    takeoff.set_val("wing_loading", 400.0, units="kg/m**2")
    takeoff.set_val("thrust_loading", 0.3)
    takeoff.run_model()

    assert np.isclose(
        takeoff.get_val("takeoff_field_length_residual")[0],
        jet_tofl(400.0, 0.3, aircraft),
    )

    landing = om.Problem()
    landing.model.add_subsystem(
        "constraint",
        JetLandingFieldLengthConstraint(
            req_type=1,
            cl_landing=2.4,
            landing_field_length=1500.0,
            obstacle_length=15.0,
            wland_mtow=0.85,
        ),
        promotes=["*"],
    )
    landing.setup()
    landing.set_val("wing_loading", 400.0, units="kg/m**2")
    landing.set_val("thrust_loading", 0.3)
    landing.run_model()

    assert np.isclose(
        landing.get_val("landing_field_length_residual")[0],
        jet_lfl(400.0, 0.3, aircraft),
    )


def test_jet_cruise_residual_components_match_fast_python():
    """Check JetCrs and JetDiv component parity."""

    aircraft = make_full_constraint_aircraft()
    cases = [
        (
            "cruise",
            JetCruiseConstraint(
                aircraft_class="Turbofan",
                req_type=1,
                cd0=0.02,
                aspect_ratio=9.0,
                oswald=0.8,
                altitude=10000.0,
                mach=0.78,
                lapse_exp=0.6,
                devries_exp=0.1,
            ),
            jet_crs,
        ),
        (
            "diversion",
            JetCruiseConstraint(
                aircraft_class="Turbofan",
                req_type=1,
                cd0=0.02,
                aspect_ratio=9.0,
                oswald=0.8,
                altitude=8000.0,
                mach=0.65,
                lapse_exp=0.6,
                devries_exp=0.2,
            ),
            jet_div,
        ),
    ]

    for name, component, fast_function in cases:
        problem = om.Problem()
        problem.model.add_subsystem(name, component, promotes=["*"])
        problem.setup()
        problem.set_val("wing_loading", 400.0, units="kg/m**2")
        problem.set_val("thrust_loading", 0.3)
        problem.run_model()

        assert np.isclose(
            problem.get_val("cruise_residual")[0],
            fast_function(400.0, 0.3, aircraft),
        )


def test_far25_climb_component_matches_fast_python_jet25_111():
    """Check shared FAR 25 climb residual parity through Jet25_111."""

    aircraft = make_full_constraint_aircraft()
    problem = om.Problem()
    problem.model.add_subsystem(
        "constraint",
        FAR25ClimbConstraint(
            aircraft_class="Turbofan",
            req_type=1,
            cl=2.0,
            cd0=0.025,
            aspect_ratio=9.0,
            oswald=0.75,
            correction=2.2,
            gradient=0.012,
            ks=1.2,
            stall_velocity=60.0,
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("wing_loading", 400.0, units="kg/m**2")
    problem.set_val("thrust_loading", 0.3)
    problem.run_model()

    assert np.isclose(
        problem.get_val("far25_climb_residual")[0],
        jet25_111(400.0, 0.3, aircraft),
    )


def test_named_far25_climb_components_match_fast_python():
    """Check named FAR 25 climb wrappers match FAST-Python functions."""

    aircraft = make_full_constraint_aircraft()
    fast_functions = {
        "jet25_111": jet25_111,
        "jet25_119": jet25_119,
        "jet25_121a": jet25_121a,
        "jet25_121b": jet25_121b,
        "jet25_121c": jet25_121c,
        "jet25_121d": jet25_121d,
    }

    for name, fast_function in fast_functions.items():
        problem = om.Problem()
        problem.model.add_subsystem(
            "constraint",
            make_named_far25_component(name, aircraft),
            promotes=["*"],
        )
        problem.setup()
        problem.set_val("wing_loading", 400.0, units="kg/m**2")
        problem.set_val("thrust_loading", 0.3)
        problem.run_model()

        assert np.isclose(
            problem.get_val("far25_named_residual")[0],
            fast_function(400.0, 0.3, aircraft),
        )


def test_jet_aeo_climb_component_matches_fast_python():
    """Check all-engines-operative climb residual parity."""

    aircraft = make_full_constraint_aircraft()
    problem = om.Problem()
    problem.model.add_subsystem(
        "constraint",
        make_aeo_climb_component(aircraft),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("wing_loading", 400.0, units="kg/m**2")
    problem.set_val("thrust_loading", 0.3)
    problem.run_model()

    assert np.isclose(
        problem.get_val("aeo_climb_residual")[0],
        jet_aeo_climb(400.0, 0.3, aircraft),
    )


def test_constraint_primitives_declare_analytic_partials():
    """Check constraint primitive derivatives against finite difference."""

    cases = [
        ("sigmoid", PsLossSigmoid(), {"ps_loss": 0.3}),
        ("oei", OEIMultiplier(constraint_type=1), {"num_engines": 2.0, "ps_loss": 0.3}),
        ("gradient", FAR25EngineGradient(num_engines=2), {"two_engine": 0.024, "three_engine": 0.027, "four_engine": 0.030}),
        ("cruise", CruiseDynamicPressure(), {"altitude": 9000.0, "mach": 0.58}),
        (
            "approach",
            JetApproachConstraint(
                req_type=1,
                cl_landing=2.4,
                wland_mtow=0.85,
                approach_velocity=70.0,
            ),
            {"wing_loading": 400.0, "thrust_loading": 0.3},
        ),
        (
            "takeoff",
            JetTakeoffFieldLengthConstraint(
                aircraft_class="Turbofan",
                cl_takeoff=2.0,
                balanced_field_length=1800.0,
                stall_velocity=60.0,
            ),
            {"wing_loading": 400.0, "thrust_loading": 0.3},
        ),
        (
            "landing",
            JetLandingFieldLengthConstraint(
                req_type=1,
                cl_landing=2.4,
                landing_field_length=1500.0,
                obstacle_length=15.0,
                wland_mtow=0.85,
            ),
            {"wing_loading": 400.0, "thrust_loading": 0.3},
        ),
        (
            "cruise_residual",
            JetCruiseConstraint(
                aircraft_class="Turbofan",
                req_type=1,
                cd0=0.02,
                aspect_ratio=9.0,
                oswald=0.8,
                altitude=10000.0,
                mach=0.78,
                lapse_exp=0.6,
                devries_exp=0.1,
            ),
            {"wing_loading": 400.0, "thrust_loading": 0.3},
        ),
        (
            "far25",
            FAR25ClimbConstraint(
                aircraft_class="Turbofan",
                req_type=1,
                cl=2.0,
                cd0=0.025,
                aspect_ratio=9.0,
                oswald=0.75,
                correction=2.2,
                gradient=0.012,
                ks=1.2,
                stall_velocity=60.0,
            ),
            {"wing_loading": 400.0, "thrust_loading": 0.3},
        ),
        (
            "named_far25",
            make_named_far25_component("jet25_121d", make_full_constraint_aircraft()),
            {"wing_loading": 400.0, "thrust_loading": 0.3},
        ),
        (
            "aeo_climb",
            make_aeo_climb_component(make_full_constraint_aircraft()),
            {"wing_loading": 400.0, "thrust_loading": 0.3},
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
            step=1.0e-5,
        )

        for partial_data in partials[name].values():
            assert partial_data["abs error"].forward < 1.0e-5


def test_constraint_residual_optimizations_match_fast_python_zero_points():
    """Run several OpenMDAO optimizations and compare to FAST-Python roots."""

    aircraft = make_full_constraint_aircraft()

    approach_problem = make_zero_residual_problem(
        JetApproachConstraint(
            req_type=1,
            cl_landing=2.4,
            wland_mtow=0.85,
            approach_velocity=70.0,
        ),
        "approach_residual",
        design_var="wing_loading",
        initial=500.0,
        lower=100.0,
        upper=1000.0,
    )
    approach_problem.run_driver()
    approach_root = approach_problem.get_val("wing_loading", units="kg/m**2")[0]
    assert abs(jet_app(approach_root, 0.3, aircraft)) < 1.0e-7

    takeoff_problem = make_zero_residual_problem(
        JetTakeoffFieldLengthConstraint(
            aircraft_class="Turbofan",
            cl_takeoff=2.0,
            balanced_field_length=1800.0,
            stall_velocity=60.0,
        ),
        "takeoff_field_length_residual",
        design_var="thrust_loading",
        initial=0.3,
        lower=0.05,
        upper=1.0,
    )
    takeoff_problem.set_val("wing_loading", 400.0, units="kg/m**2")
    takeoff_problem.run_driver()
    takeoff_root = takeoff_problem.get_val("thrust_loading")[0]
    assert abs(jet_tofl(400.0, takeoff_root, aircraft)) < 1.0e-7

    landing_problem = make_zero_residual_problem(
        JetLandingFieldLengthConstraint(
            req_type=1,
            cl_landing=2.4,
            landing_field_length=1500.0,
            obstacle_length=15.0,
            wland_mtow=0.85,
        ),
        "landing_field_length_residual",
        design_var="wing_loading",
        initial=500.0,
        lower=100.0,
        upper=1000.0,
    )
    landing_problem.run_driver()
    landing_root = landing_problem.get_val("wing_loading", units="kg/m**2")[0]
    assert abs(jet_lfl(landing_root, 0.3, aircraft)) < 1.0e-7

    cruise_problem = make_zero_residual_problem(
        JetCruiseConstraint(
            aircraft_class="Turbofan",
            req_type=1,
            cd0=0.02,
            aspect_ratio=9.0,
            oswald=0.8,
            altitude=10000.0,
            mach=0.78,
            lapse_exp=0.6,
            devries_exp=0.1,
        ),
        "cruise_residual",
        design_var="thrust_loading",
        initial=0.3,
        lower=0.01,
        upper=1.0,
    )
    cruise_problem.set_val("wing_loading", 400.0, units="kg/m**2")
    cruise_problem.run_driver()
    cruise_root = cruise_problem.get_val("thrust_loading")[0]
    assert abs(jet_crs(400.0, cruise_root, aircraft)) < 1.0e-7

    far25_problem = make_zero_residual_problem(
        FAR25ClimbConstraint(
            aircraft_class="Turbofan",
            req_type=1,
            cl=2.0,
            cd0=0.025,
            aspect_ratio=9.0,
            oswald=0.75,
            correction=2.2,
            gradient=0.012,
            ks=1.2,
            stall_velocity=60.0,
        ),
        "far25_climb_residual",
        design_var="thrust_loading",
        initial=0.3,
        lower=0.01,
        upper=1.0,
    )
    far25_problem.set_val("wing_loading", 400.0, units="kg/m**2")
    far25_problem.run_driver()
    far25_root = far25_problem.get_val("thrust_loading")[0]
    assert abs(jet25_111(400.0, far25_root, aircraft)) < 1.0e-7


def make_zero_residual_problem(component, residual_name, design_var, initial, lower, upper):
    """Return an OpenMDAO optimization that drives one residual to zero."""

    problem = om.Problem()
    problem.model.add_subsystem("constraint", component, promotes=["*"])
    problem.model.add_subsystem(
        "objective",
        om.ExecComp(
            f"objective = {residual_name} ** 2",
            **{
                residual_name: {
                    "val": 0.0,
                },
                "objective": {
                    "val": 0.0,
                },
            },
        ),
        promotes=["*"],
    )
    problem.driver = om.ScipyOptimizeDriver()
    problem.driver.options["optimizer"] = "SLSQP"
    problem.driver.options["disp"] = False
    problem.driver.options["tol"] = 1.0e-12
    problem.model.add_design_var(design_var, lower=lower, upper=upper)
    problem.model.add_objective("objective")
    problem.setup()
    problem.set_val(design_var, initial)
    return problem


def make_named_far25_component(name, aircraft):
    """Return a named FAR 25 OpenMDAO component from FAST aircraft fields."""

    specs = aircraft["Specs"]
    performance = specs["Performance"]
    aero = specs["Aero"]
    return JetFAR25NamedClimbConstraint(
        name=name,
        aircraft_class=specs["TLAR"]["Class"],
        constraint_type=aircraft["Settings"]["ConstraintType"],
        req_type=specs["TLAR"]["ReqType"],
        num_engines=specs["Propulsion"]["NumEngines"],
        ps_loss=performance["PsLoss"],
        cl_takeoff=aero["CL"]["Tko"],
        cl_landing=aero["CL"]["Lnd"],
        cl_cruise=aero["CL"]["Crs"],
        cd0_takeoff=aero["CD0"]["Tko"],
        cd0_landing=aero["CD0"]["Lnd"],
        cd0_cruise=aero["CD0"]["Crs"],
        aspect_ratio=aero["AR"],
        oswald_takeoff=aero["e"]["Tko"],
        oswald_landing=aero["e"]["Lnd"],
        oswald_cruise=aero["e"]["Crs"],
        temperature_correction=performance["TempInc"],
        wland_mtow=performance["Wland_MTOW"],
        max_continuous=performance["MaxCont"],
        stall_velocity=performance["Vels"]["Stl"],
    )


def make_aeo_climb_component(aircraft):
    """Return a JetAEOClimb OpenMDAO component from FAST aircraft fields."""

    specs = aircraft["Specs"]
    performance = specs["Performance"]
    aero = specs["Aero"]
    return JetAEOClimbConstraint(
        aircraft_class=specs["TLAR"]["Class"],
        req_type=specs["TLAR"]["ReqType"],
        cl_cruise=aero["CL"]["Crs"],
        cd0_cruise=aero["CD0"]["Crs"],
        aspect_ratio=aero["AR"],
        oswald_takeoff=aero["e"]["Tko"],
        stall_velocity=performance["Vels"]["Stl"],
        extra_gradient=performance["ExtraGrad"],
        constraint_type=aircraft["Settings"]["ConstraintType"],
        num_engines=specs["Propulsion"]["NumEngines"],
        ps_loss=performance["PsLoss"],
    )


def make_constraint_aircraft(constraint_type, ps_loss=0.2, num_engines=2):
    """Return minimal aircraft dictionary for FAST-Python constraint helpers."""

    return {
        "Settings": {
            "ConstraintType": constraint_type,
        },
        "Specs": {
            "Performance": {
                "PsLoss": ps_loss,
            },
            "Propulsion": {
                "NumEngines": num_engines,
            },
        },
    }


def make_full_constraint_aircraft():
    """Return FAST-Python constraint fixture used for residual parity tests."""

    aircraft = make_constraint_aircraft(0, ps_loss=0.3, num_engines=2)
    aircraft["Specs"]["TLAR"] = {
        "Class": "Turbofan",
        "CFRPart": 25,
        "ReqType": 1,
    }
    aircraft["Specs"]["Performance"].update(
        {
            "TempInc": 1.1,
            "MaxCont": 0.92,
            "ExtraGrad": 0.018,
            "Wland_MTOW": 0.85,
            "Vels": {
                "App": 70,
                "Crs": 0.78,
                "Div": 0.65,
                "Stl": 60,
            },
            "Alts": {
                "Crs": 10000,
                "Div": 8000,
                "Srv": 11000,
            },
            "TOFL": 1800,
            "LFL": 1500,
            "ObstLen": 15,
        }
    )
    aircraft["Specs"]["Aero"] = {
        "AR": 9,
        "CD0": {
            "Tko": 0.05,
            "Crs": 0.02,
            "Lnd": 0.08,
        },
        "CL": {
            "Crs": 1.5,
            "Lnd": 2.4,
            "Tko": 2.0,
        },
        "e": {
            "Tko": 0.75,
            "Crs": 0.8,
            "Lnd": 0.7,
        },
    }
    return aircraft
