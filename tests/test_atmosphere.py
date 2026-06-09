# tests/test_atmosphere.py

"""Tests for derivative-native FAST atmosphere OpenMDAO components."""

import os

os.environ.setdefault("OPENMDAO_REPORTS", "0")

import numpy as np
import openmdao.api as om

from fast_openmdao import Gravity, StandardAtmosphere
from fast_python.atmosphere import gravity, standard_atmosphere


def test_gravity_matches_fast_python_and_declares_partials():
    """Check gravity values and analytic derivatives against finite difference."""

    problem = om.Problem()
    problem.model.add_subsystem("gravity", Gravity(), promotes=["*"])
    problem.setup()
    problem.set_val("altitude", 10000.0, units="m")
    problem.run_model()

    assert np.isclose(problem.get_val("gravity", units="m/s**2")[0], gravity(10000.0))

    partials = problem.check_partials(out_stream=None, method="fd")
    assert partials["gravity"][("gravity", "altitude")]["abs error"].forward < 1.0e-8


def test_standard_atmosphere_matches_fast_python_across_layers():
    """Check atmosphere outputs preserve FAST-Python layer equations."""

    for altitude in [0.0, 10000.0, 15000.0, 25000.0, 40000.0, 49000.0, 60000.0, 80000.0, 90000.0]:
        problem = om.Problem()
        problem.model.add_subsystem("atmosphere", StandardAtmosphere(), promotes=["*"])
        problem.setup()
        problem.set_val("altitude", altitude, units="m")
        problem.run_model()

        temperature, pressure, density = standard_atmosphere(altitude)

        assert np.isclose(problem.get_val("temperature", units="K")[0], temperature)
        assert np.isclose(problem.get_val("pressure", units="Pa")[0], pressure)
        assert np.isclose(problem.get_val("density", units="kg/m**3")[0], density)


def test_standard_atmosphere_declares_analytic_partials():
    """Check atmosphere analytic derivatives against finite difference."""

    for altitude in [10000.0, 15000.0, 25000.0, 40000.0, 49000.0, 60000.0, 80000.0, 90000.0]:
        problem = om.Problem()
        problem.model.add_subsystem("atmosphere", StandardAtmosphere(), promotes=["*"])
        problem.setup()
        problem.set_val("altitude", altitude, units="m")
        problem.run_model()

        partials = problem.check_partials(out_stream=None, method="fd")

        for partial_data in partials["atmosphere"].values():
            assert partial_data["abs error"].forward < 5.0e-5
