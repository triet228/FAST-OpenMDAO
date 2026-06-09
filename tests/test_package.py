# tests/test_package.py

"""Smoke tests for the FAST OpenMDAO package scaffold."""

import fast_openmdao


def test_package_exposes_version():
    """Check that the package imports from the local src layout."""

    assert fast_openmdao.__version__ == "0.1.0"

