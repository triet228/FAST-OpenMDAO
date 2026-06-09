# tests/test_database_components.py

"""Tests for OpenMDAO FAST database-derived equation components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import MacLiftDragEstimate, TurbopropCruiseLiftDragEstimate
from fast_python.atmosphere import standard_atmosphere
from fast_python.database import calc_prop_vals, mac_ld


def test_mac_lift_drag_estimate_matches_fast_python():
    """Check FAST MAC L/D estimate parity."""

    aspect_ratio = 9.5
    reynolds = 1.25e7
    problem = om.Problem()
    problem.model.add_subsystem("mac", MacLiftDragEstimate(), promotes=["*"])
    problem.setup()
    problem.set_val("aspect_ratio", aspect_ratio)
    problem.set_val("reynolds", reynolds)
    problem.run_model()

    assert np.isclose(
        problem.get_val("lift_drag")[0],
        mac_ld(aspect_ratio, reynolds),
    )


def test_mac_lift_drag_estimate_declares_analytic_partials():
    """Check MAC L/D analytical derivatives against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem("mac", MacLiftDragEstimate(), promotes=["*"])
    problem.setup()
    problem.set_val("aspect_ratio", 9.5)
    problem.set_val("reynolds", 1.25e7)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-4,
    )

    for partial_data in partials["mac"].values():
        assert (
            partial_data["abs error"].forward < 1.0e-6
            or partial_data["rel error"].forward < 1.0e-6
        )


def test_turboprop_cruise_lift_drag_estimate_matches_fast_python():
    """Check FAST CalcPropVals turboprop cruise L/D parity."""

    plane = make_prop_plane()
    expected = calc_prop_vals(plane, "Vals")["Specs"]["Aero"]["L_D"]["Crs"]
    cruise_power = plane["Specs"]["Power"]["Crs"] * 1000 * plane["Specs"][
        "Propulsion"
    ]["NumEngines"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "lift_drag",
        TurbopropCruiseLiftDragEstimate(
            temperature=standard_atmosphere(7500.0)[0],
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("mtow", plane["Specs"]["Weight"]["MTOW"], units="kg")
    problem.set_val("cruise_power", cruise_power, units="W")
    problem.set_val("cruise_mach", plane["Specs"]["Performance"]["Vels"]["Crs"])
    problem.run_model()

    assert np.isclose(problem.get_val("lift_drag")[0], expected)


def test_turboprop_cruise_lift_drag_estimate_declares_analytic_partials():
    """Check turboprop cruise L/D derivatives against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "lift_drag",
        TurbopropCruiseLiftDragEstimate(
            temperature=standard_atmosphere(7500.0)[0],
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("mtow", 18500.0, units="kg")
    problem.set_val("cruise_power", 1.6e6, units="W")
    problem.set_val("cruise_mach", 0.42)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-4,
    )

    for partial_data in partials["lift_drag"].values():
        assert (
            partial_data["abs error"].forward < 1.0e-6
            or partial_data["rel error"].forward < 1.0e-6
        )


def make_prop_plane():
    """Return minimal turboprop database aircraft for CalcPropVals."""

    return {
        "Overview": {
            "KeyWords": "baseline passenger",
            "PayloadType": "P",
            "Monikers": np.nan,
            "AlternateDesignation": np.nan,
        },
        "Specs": {
            "TLAR": {
                "MaxPax": 40,
            },
            "Weight": {
                "MTOW": 18500.0,
                "OEW": 11000.0,
                "Fuel": 2500.0,
                "Cargo": 500.0,
                "MaxPayload": np.nan,
            },
            "Propulsion": {
                "NumEngines": 2,
                "Engine": {
                    "DryWeight": 450.0,
                    "Power_SLS": 1200.0,
                    "Power_SLS_Eq": np.nan,
                    "Power_Cont_Eq": 1000.0,
                },
            },
            "Power": {
                "SLS": np.nan,
                "Cont": np.nan,
                "Clb": 900.0,
                "Crs": 800.0,
            },
            "Aero": {
                "S": 55.0,
                "Span": 24.0,
                "Height": 7.0,
                "TipChord": 1.5,
                "RootChord": 3.2,
            },
            "Performance": {
                "Vels": {
                    "Crs": 0.42,
                    "Tko": 150.0,
                },
            },
        },
    }
