# src/fast_openmdao/__init__.py

"""OpenMDAO integration layer for FAST-Python."""

from fast_openmdao.components import FastPythonComponent
from fast_openmdao.problem import make_fast_optimization_problem, make_fast_problem

__version__ = "0.1.0"

__all__ = [
    "FastPythonComponent",
    "__version__",
    "make_fast_optimization_problem",
    "make_fast_problem",
]
