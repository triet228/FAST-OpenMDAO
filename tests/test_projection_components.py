# tests/test_projection_components.py

"""Tests for OpenMDAO FAST projection components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import (
    BatterySpecificEnergyProjection,
    ElectricMotorSpecificPowerProjection,
    KPPProjection,
)
from fast_python.projection import kpp_projection


def test_kpp_projection_matches_fast_python():
    """Check configurable KPP projection parity with FAST-Python."""

    problem = om.Problem()
    problem.model.add_subsystem(
        "projection",
        KPPProjection(
            aircraft_class="Turboprop",
            kpp="M(L/D)",
            output_name="mach_lift_drag",
        ),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("year", 2025.0)
    problem.run_model()

    assert np.isclose(
        problem.get_val("mach_lift_drag")[0],
        kpp_projection("Turboprop", 2025.0, "M(L/D)"),
    )


def test_projection_components_declare_analytic_partials():
    """Check projection derivatives against finite difference."""

    cases = [
        KPPProjection(
            aircraft_class="Turbofan",
            kpp="Cruise SFC",
            output_name="cruise_sfc",
        ),
        KPPProjection(
            aircraft_class="Turboprop",
            kpp="Total Takeoff T/ MTOW",
            output_name="takeoff_thrust_weight",
        ),
        BatterySpecificEnergyProjection(),
        ElectricMotorSpecificPowerProjection(),
    ]

    for component in cases:
        problem = om.Problem()
        problem.model.add_subsystem("projection", component, promotes=["*"])
        problem.setup()
        problem.set_val("year", 2030.0)
        problem.run_model()
        partials = problem.check_partials(out_stream=None, method="fd")

        for partial_data in partials["projection"].values():
            assert partial_data["abs error"].forward < 1.0e-5
