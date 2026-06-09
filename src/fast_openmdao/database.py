# src/fast_openmdao/database.py

"""OpenMDAO components for FAST database-derived numerical equations."""

import numpy as np
import openmdao.api as om


class MacLiftDragEstimate(om.ExplicitComponent):
    """Compute FAST's Korn-style cruise lift-to-drag estimate.

    Inputs:
        aspect_ratio: Wing aspect ratio.
        reynolds: Reynolds number based on mean aerodynamic chord.

    Outputs:
        lift_drag: Cruise lift-to-drag estimate used by FAST database
            preprocessing.

    Assumptions:
        This converts the smooth scalar equation from ``fast_python.database``
        ``mac_ld``. The surrounding database field mutation remains support
        code rather than a differentiable OpenMDAO component.
    """

    def setup(self):
        self.add_input("aspect_ratio", val=9.0)
        self.add_input("reynolds", val=1.0e7)
        self.add_output("lift_drag", val=18.0)
        self.declare_partials(of="lift_drag", wrt="*")

    def compute(self, inputs, outputs):
        outputs["lift_drag"] = mac_lift_drag_values(
            inputs["aspect_ratio"][0],
            inputs["reynolds"][0],
        )["lift_drag"]

    def compute_partials(self, inputs, partials):
        values = mac_lift_drag_values(
            inputs["aspect_ratio"][0],
            inputs["reynolds"][0],
        )
        partials["lift_drag", "aspect_ratio"] = values["dlift_drag_daspect_ratio"]
        partials["lift_drag", "reynolds"] = values["dlift_drag_dreynolds"]


def mac_lift_drag_values(aspect_ratio, reynolds):
    """Return FAST MAC L/D estimate and analytical derivatives."""

    aspect_ratio = float(aspect_ratio)
    reynolds = float(reynolds)
    base = aspect_ratio ** 2 * reynolds
    correction = 1.0 + 3.6 * aspect_ratio ** (-9.0 / 4.0)
    lift_drag = 0.321 * base ** (3.0 / 16.0) * correction ** -0.5
    dlog_daspect_ratio = (
        3.0 / (8.0 * aspect_ratio)
        + (3.6 * 9.0 / 8.0)
        * aspect_ratio ** (-13.0 / 4.0)
        / correction
    )
    dlog_dreynolds = 3.0 / (16.0 * reynolds)
    return {
        "lift_drag": lift_drag,
        "dlift_drag_daspect_ratio": lift_drag * dlog_daspect_ratio,
        "dlift_drag_dreynolds": lift_drag * dlog_dreynolds,
    }
