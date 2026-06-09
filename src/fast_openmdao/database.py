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


class TurbopropCruiseLiftDragEstimate(om.ExplicitComponent):
    """Compute FAST turboprop database cruise lift-to-drag estimate.

    Inputs:
        mtow: Maximum takeoff weight in kg.
        cruise_power: Total cruise shaft power in W.
        cruise_mach: Cruise Mach number.

    Outputs:
        lift_drag: Cruise lift-to-drag estimate used by ``CalcPropVals``.

    Assumptions:
        FAST database preprocessing evaluates this at the fixed 7500 m
        standard-atmosphere temperature. That temperature is an option here
        because it is discrete preprocessing context, not a design variable in
        FAST ``calc_prop_values``.
    """

    def initialize(self):
        self.options.declare("temperature", default=239.4403228917732)

    def setup(self):
        self.add_input("mtow", val=10000.0, units="kg")
        self.add_input("cruise_power", val=1.0e6, units="W")
        self.add_input("cruise_mach", val=0.4)
        self.add_output("lift_drag", val=12.0)
        self.declare_partials(of="lift_drag", wrt="*")

    def compute(self, inputs, outputs):
        outputs["lift_drag"] = turboprop_cruise_lift_drag_values(
            inputs["mtow"][0],
            inputs["cruise_power"][0],
            inputs["cruise_mach"][0],
            self.options["temperature"],
        )["lift_drag"]

    def compute_partials(self, inputs, partials):
        values = turboprop_cruise_lift_drag_values(
            inputs["mtow"][0],
            inputs["cruise_power"][0],
            inputs["cruise_mach"][0],
            self.options["temperature"],
        )
        partials["lift_drag", "mtow"] = values["dlift_drag_dmtow"]
        partials["lift_drag", "cruise_power"] = values["dlift_drag_dcruise_power"]
        partials["lift_drag", "cruise_mach"] = values["dlift_drag_dcruise_mach"]


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


def turboprop_cruise_lift_drag_values(mtow, cruise_power, cruise_mach, temperature):
    """Return FAST turboprop cruise L/D estimate and derivatives."""

    mtow = float(mtow)
    cruise_power = float(cruise_power)
    cruise_mach = float(cruise_mach)
    speed_factor = np.sqrt(1.4 * 287.0 * temperature)
    numerator = mtow * 0.995 * 0.985 * 9.81 * cruise_mach * speed_factor
    lift_drag = numerator / cruise_power
    return {
        "lift_drag": lift_drag,
        "dlift_drag_dmtow": lift_drag / mtow,
        "dlift_drag_dcruise_power": -lift_drag / cruise_power,
        "dlift_drag_dcruise_mach": lift_drag / cruise_mach,
    }
