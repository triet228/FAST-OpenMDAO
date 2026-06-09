# tests/test_database_components.py

"""Tests for OpenMDAO FAST database-derived equation components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    DatabaseFanThrustNormalization,
    DatabaseGeometryLoads,
    DatabasePropPowerNormalization,
    DatabaseWeightFractions,
    MacLiftDragEstimate,
    TurbofanCruiseLiftDragEstimate,
    TurbopropCruiseLiftDragEstimate,
)
from fast_python.atmosphere import standard_atmosphere
from fast_python.database import calc_fan_vals, calc_prop_vals, mac_ld


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


def test_turbofan_cruise_lift_drag_estimate_matches_fast_python():
    """Check FAST CalcFanVals turbofan cruise L/D parity."""

    plane = make_fan_plane()
    expected = calc_fan_vals(plane, "Vals")["Specs"]["Aero"]["L_D"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "lift_drag",
        TurbofanCruiseLiftDragEstimate(
            num_engines=plane["Specs"]["Propulsion"]["NumEngines"],
        ),
        promotes=["*"],
    )
    problem.setup()
    set_turbofan_cruise_lift_drag_values(problem, plane)
    problem.run_model()

    assert np.isclose(problem.get_val("breguet_lift_drag")[0], expected["CrsBRE"])
    assert np.isclose(problem.get_val("thrust_lift_drag")[0], expected["Crs"])


def test_turbofan_cruise_lift_drag_estimate_declares_analytic_partials():
    """Check turbofan cruise L/D derivatives against finite difference."""

    plane = make_fan_plane()
    problem = om.Problem()
    problem.model.add_subsystem(
        "lift_drag",
        TurbofanCruiseLiftDragEstimate(
            num_engines=plane["Specs"]["Propulsion"]["NumEngines"],
        ),
        promotes=["*"],
    )
    problem.setup()
    set_turbofan_cruise_lift_drag_values(problem, plane)
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


def test_database_weight_fractions_match_fast_python_calc_prop_vals():
    """Check FAST database weight-derived value parity."""

    plane = make_prop_plane()
    expected = calc_prop_vals(plane, "Vals")["Specs"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "weights",
        DatabaseWeightFractions(
            num_engines=plane["Specs"]["Propulsion"]["NumEngines"],
        ),
        promotes=["*"],
    )
    problem.setup()
    set_database_weight_fraction_values(problem, plane)
    problem.run_model()

    assert np.isclose(
        problem.get_val("airframe_weight", units="kg")[0],
        expected["Weight"]["Airframe"],
    )
    assert np.isclose(problem.get_val("oew_mtow")[0], expected["Weight"]["OEW_MTOW"])
    assert np.isclose(
        problem.get_val("engine_fraction")[0],
        expected["Weight"]["EngineFrac"],
    )
    assert np.isclose(
        problem.get_val("fuel_fraction")[0],
        expected["Weight"]["FuelFrac"],
    )
    assert np.isclose(
        problem.get_val("wing_loading", units="kg/m**2")[0],
        expected["Aero"]["W_S"]["SLS"],
    )


def test_database_weight_fractions_declares_analytic_partials():
    """Check database weight-fraction derivatives against finite difference."""

    plane = make_prop_plane()
    problem = om.Problem()
    problem.model.add_subsystem(
        "weights",
        DatabaseWeightFractions(
            num_engines=plane["Specs"]["Propulsion"]["NumEngines"],
        ),
        promotes=["*"],
    )
    problem.setup()
    set_database_weight_fraction_values(problem, plane)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-4,
    )

    for partial_data in partials["weights"].values():
        assert (
            partial_data["abs error"].forward < 1.0e-6
            or partial_data["rel error"].forward < 1.0e-6
        )


def test_database_geometry_loads_match_fast_python_calc_fan_vals():
    """Check FAST turbofan database geometry/load preprocessing parity."""

    plane = make_fan_plane()
    expected = calc_fan_vals(plane, "Vals")["Specs"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "loads",
        DatabaseGeometryLoads(
            num_engines=plane["Specs"]["Propulsion"]["NumEngines"],
            payload_source="pax_cargo",
        ),
        promotes=["*"],
    )
    problem.setup()
    set_database_geometry_load_values(problem, plane, max_payload=0.0)
    problem.run_model()

    assert np.isclose(problem.get_val("taper_ratio")[0], expected["Aero"]["TaperRatio"])
    assert np.isclose(problem.get_val("aspect_ratio")[0], expected["Aero"]["AR"])
    assert np.isclose(
        problem.get_val("payload", units="kg")[0],
        expected["Weight"]["Payload"],
    )
    assert np.isclose(
        problem.get_val("burden", units="kg")[0],
        expected["Weight"]["Burden"],
    )
    assert np.isclose(
        problem.get_val("structure_burden")[0],
        expected["Weight"]["Structure_Burden"],
    )
    assert np.isclose(problem.get_val("mzfw_mtow")[0], expected["Weight"]["MZFW_MTOW"])


def test_database_geometry_loads_match_fast_python_calc_prop_vals():
    """Check FAST turboprop database geometry/load preprocessing parity."""

    plane = make_prop_plane()
    plane["Specs"]["Weight"]["MaxPayload"] = 4200.0
    expected = calc_prop_vals(plane, "Vals")["Specs"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "loads",
        DatabaseGeometryLoads(
            num_engines=plane["Specs"]["Propulsion"]["NumEngines"],
            payload_source="max_payload",
        ),
        promotes=["*"],
    )
    problem.setup()
    set_database_geometry_load_values(problem, plane, max_payload=4200.0)
    problem.run_model()

    assert np.isclose(problem.get_val("taper_ratio")[0], expected["Aero"]["TaperRatio"])
    assert np.isclose(problem.get_val("aspect_ratio")[0], expected["Aero"]["AR"])
    assert np.isclose(
        problem.get_val("payload", units="kg")[0],
        expected["Weight"]["Payload"],
    )


def test_database_geometry_loads_declares_analytic_partials():
    """Check database geometry/load derivatives against finite difference."""

    plane = make_fan_plane()
    problem = om.Problem()
    problem.model.add_subsystem(
        "loads",
        DatabaseGeometryLoads(
            num_engines=plane["Specs"]["Propulsion"]["NumEngines"],
            payload_source="pax_cargo",
        ),
        promotes=["*"],
    )
    problem.setup()
    set_database_geometry_load_values(problem, plane, max_payload=0.0)
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-4,
    )

    for partial_data in partials["loads"].values():
        assert (
            partial_data["abs error"].forward < 1.0e-6
            or partial_data["rel error"].forward < 1.0e-6
        )


def test_database_fan_thrust_normalization_matches_fast_python_calc_fan_vals():
    """Check FAST turbofan database thrust normalization parity."""

    plane = make_fan_plane()
    expected = calc_fan_vals(plane, "Vals")["Specs"]["Propulsion"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "thrust",
        DatabaseFanThrustNormalization(
            num_engines=plane["Specs"]["Propulsion"]["NumEngines"],
            thrust_source="engine",
        ),
        promotes=["*"],
    )
    problem.setup()
    set_database_fan_thrust_values(problem, plane)
    problem.run_model()

    assert np.isclose(
        problem.get_val("thrust_loading_sls")[0],
        expected["T_W"]["SLS"],
    )
    assert np.isclose(problem.get_val("thrust_sls", units="N")[0], expected["Thrust"]["SLS"])
    assert np.isclose(problem.get_val("thrust_max", units="N")[0], expected["Thrust"]["Max"])
    assert np.isclose(
        problem.get_val("thrust_cruise", units="N")[0],
        expected["Thrust"]["Crs"],
    )


def test_database_prop_power_normalization_matches_fast_python_calc_prop_vals():
    """Check FAST turboprop database power normalization parity."""

    plane = make_prop_plane()
    expected = calc_prop_vals(plane, "Vals")["Specs"]["Power"]
    problem = om.Problem()
    problem.model.add_subsystem(
        "power",
        DatabasePropPowerNormalization(
            num_engines=plane["Specs"]["Propulsion"]["NumEngines"],
            sls_power_source="engine",
            continuous_power_source="engine_equivalent",
        ),
        promotes=["*"],
    )
    problem.setup()
    set_database_prop_power_values(problem, plane)
    problem.run_model()

    assert np.isclose(problem.get_val("sea_level_power", units="W")[0], expected["SLS"])
    assert np.isclose(problem.get_val("continuous_power", units="W")[0], expected["Cont"])
    assert np.isclose(problem.get_val("climb_power_total", units="W")[0], expected["Clb"])
    assert np.isclose(problem.get_val("cruise_power_total", units="W")[0], expected["Crs"])
    assert np.isclose(
        problem.get_val("sea_level_power_loading", units="kW/kg")[0],
        expected["P_W"]["SLS"],
    )


def test_database_propulsion_normalization_declares_analytic_partials():
    """Check database propulsion-normalization derivatives."""

    fan_plane = make_fan_plane()
    prop_plane = make_prop_plane()
    problem = om.Problem()
    problem.model.add_subsystem(
        "thrust",
        DatabaseFanThrustNormalization(
            num_engines=fan_plane["Specs"]["Propulsion"]["NumEngines"],
            thrust_source="engine",
        ),
    )
    problem.model.add_subsystem(
        "power",
        DatabasePropPowerNormalization(
            num_engines=prop_plane["Specs"]["Propulsion"]["NumEngines"],
            sls_power_source="engine",
            continuous_power_source="engine_equivalent",
        ),
    )
    problem.setup()
    set_database_fan_thrust_values(problem, fan_plane, prefix="thrust.")
    set_database_prop_power_values(problem, prop_plane, prefix="power.")
    problem.run_model()

    partials = problem.check_partials(
        out_stream=None,
        method="fd",
        form="central",
        step=1.0e-4,
    )

    for component_name in ("thrust", "power"):
        for partial_data in partials[component_name].values():
            assert (
                partial_data["abs error"].forward < 1.0e-6
                or partial_data["rel error"].forward < 1.0e-6
            )


def set_database_weight_fraction_values(problem, plane):
    """Set OpenMDAO inputs from the shared turboprop database fixture."""

    specs = plane["Specs"]
    problem.set_val("mtow", specs["Weight"]["MTOW"], units="kg")
    problem.set_val("oew", specs["Weight"]["OEW"], units="kg")
    problem.set_val("fuel_weight", specs["Weight"]["Fuel"], units="kg")
    problem.set_val(
        "engine_dry_weight",
        specs["Propulsion"]["Engine"]["DryWeight"],
        units="kg",
    )
    problem.set_val("wing_area", specs["Aero"]["S"], units="m**2")


def set_turbofan_cruise_lift_drag_values(problem, plane):
    """Set OpenMDAO inputs from the shared turbofan database fixture."""

    specs = plane["Specs"]
    problem.set_val("mtow", specs["Weight"]["MTOW"], units="kg")
    problem.set_val("fuel_weight", specs["Weight"]["Fuel"], units="kg")
    problem.set_val("range", specs["Performance"]["Range"] * 1000.0, units="m")
    problem.set_val(
        "tsfc_cruise",
        specs["Propulsion"]["Engine"]["TSFC_Crs"],
    )
    problem.set_val("cruise_mach", specs["Performance"]["Vels"]["Crs"])
    problem.set_val(
        "temperature",
        standard_atmosphere(specs["Performance"]["Alts"]["Crs"])[0],
        units="K",
    )
    problem.set_val(
        "engine_thrust_cruise",
        specs["Propulsion"]["Engine"]["Thrust_Crs"],
        units="N",
    )


def set_database_geometry_load_values(problem, plane, max_payload):
    """Set OpenMDAO inputs for database geometry/load preprocessing."""

    specs = plane["Specs"]
    problem.set_val("mtow", specs["Weight"]["MTOW"], units="kg")
    problem.set_val("mzfw", specs["Weight"].get("MZFW", 0.0), units="kg")
    problem.set_val("fuel_weight", specs["Weight"]["Fuel"], units="kg")
    problem.set_val(
        "engine_dry_weight",
        specs["Propulsion"]["Engine"]["DryWeight"],
        units="kg",
    )
    cargo_weight = specs["Weight"].get("Cargo", 0.0)

    if np.isnan(cargo_weight):
        cargo_weight = 0.0

    problem.set_val("cargo_weight", cargo_weight, units="kg")
    problem.set_val("max_payload", max_payload, units="kg")
    problem.set_val("max_passengers", specs["TLAR"]["MaxPax"])
    problem.set_val("wing_span", specs["Aero"]["Span"], units="m")
    problem.set_val("wing_area", specs["Aero"]["S"], units="m**2")
    problem.set_val("tip_chord", specs["Aero"]["TipChord"], units="m")
    problem.set_val("root_chord", specs["Aero"]["RootChord"], units="m")


def set_database_fan_thrust_values(problem, plane, prefix=""):
    """Set OpenMDAO inputs for database fan thrust normalization."""

    specs = plane["Specs"]
    engine = specs["Propulsion"]["Engine"]
    thrust = specs["Propulsion"]["Thrust"]
    problem.set_val(f"{prefix}mtow", specs["Weight"]["MTOW"], units="kg")
    problem.set_val(f"{prefix}engine_thrust_sls", engine["Thrust_SLS"], units="N")
    problem.set_val(f"{prefix}engine_thrust_max", engine["Thrust_Max"], units="N")
    problem.set_val(f"{prefix}engine_thrust_cruise", engine["Thrust_Crs"], units="N")
    problem.set_val(f"{prefix}specified_thrust_sls", 0.0, units="N")
    problem.set_val(f"{prefix}specified_thrust_max", 0.0, units="N")

    if not np.isnan(thrust["SLS"]):
        problem.set_val(f"{prefix}specified_thrust_sls", thrust["SLS"], units="N")

    if not np.isnan(thrust["Max"]):
        problem.set_val(f"{prefix}specified_thrust_max", thrust["Max"], units="N")


def set_database_prop_power_values(problem, plane, prefix=""):
    """Set OpenMDAO inputs for database prop power normalization."""

    specs = plane["Specs"]
    engine = specs["Propulsion"]["Engine"]
    power = specs["Power"]
    problem.set_val(f"{prefix}mtow", specs["Weight"]["MTOW"], units="kg")
    problem.set_val(f"{prefix}engine_power_sls", engine["Power_SLS"], units="kW")
    problem.set_val(
        f"{prefix}engine_power_sls_equivalent",
        0.0,
        units="kW",
    )

    if not np.isnan(engine["Power_SLS_Eq"]):
        problem.set_val(
            f"{prefix}engine_power_sls_equivalent",
            engine["Power_SLS_Eq"],
            units="kW",
        )

    problem.set_val(
        f"{prefix}engine_power_continuous_equivalent",
        engine["Power_Cont_Eq"],
        units="kW",
    )
    problem.set_val(f"{prefix}specified_power_sls", 0.0, units="kW")
    problem.set_val(f"{prefix}specified_power_continuous", 0.0, units="kW")
    problem.set_val(f"{prefix}climb_power", power["Clb"], units="kW")
    problem.set_val(f"{prefix}cruise_power", power["Crs"], units="kW")

    if not np.isnan(power["SLS"]):
        problem.set_val(f"{prefix}specified_power_sls", power["SLS"], units="kW")

    if not np.isnan(power["Cont"]):
        problem.set_val(
            f"{prefix}specified_power_continuous",
            power["Cont"],
            units="kW",
        )


def make_fan_plane():
    """Return minimal turbofan database aircraft for CalcFanVals."""

    return {
        "Overview": {
            "KeyWords": "baseline single long business",
            "PayloadType": "P",
            "Monikers": np.nan,
            "AlternateDesignation": np.nan,
        },
        "Specs": {
            "Weight": {
                "MTOW": 70000.0,
                "Fuel": 15000.0,
                "OEW": 42000.0,
                "MZFW": 55000.0,
                "Cargo": np.nan,
            },
            "TLAR": {
                "MaxPax": 150,
            },
            "Propulsion": {
                "Engine": {
                    "TSFC_Crs": 0.6,
                    "Thrust_Crs": 60000.0,
                    "Thrust_Max": 120000.0,
                    "Thrust_SLS": 100000.0,
                    "DryWeight": 2000.0,
                },
                "NumEngines": 2,
                "Thrust": {
                    "SLS": np.nan,
                    "Max": np.nan,
                },
            },
            "Performance": {
                "Range": 3000.0,
                "Alts": {
                    "Crs": 10000.0,
                },
                "Vels": {
                    "Crs": 0.78,
                    "Tko": 140.0,
                },
            },
            "Aero": {
                "Span": 35.0,
                "S": 120.0,
                "MAC": 4.0,
                "WingtipDevice": "Winglets",
                "TipChord": 2.0,
                "RootChord": 5.0,
                "Height": 12.0,
            },
        },
    }


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
