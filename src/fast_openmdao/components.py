# src/fast_openmdao/components.py

"""OpenMDAO components that wrap FAST-Python workflows."""

from copy import deepcopy

import openmdao.api as om

from fast_openmdao.paths import get_path, set_path, scalar_value


class FastPythonComponent(om.ExplicitComponent):
    """Run a FAST-Python workflow from OpenMDAO scalar variables.

    Inputs:
        aircraft: Baseline FAST aircraft dictionary.
        mission: Optional baseline FAST mission/profile dictionary.
        input_specs: Dictionaries with name, target, path, val, units, and desc.
        output_specs: Dictionaries with name, path, units, and desc.
        runner: Optional callable accepting aircraft and mission dictionaries.

    Outputs:
        OpenMDAO scalar outputs extracted from the FAST result dictionary.

    Assumptions:
        The first derivative bridge uses OpenMDAO finite difference partials.
        This keeps optimization plumbing available while FAST-Python internals
        are prepared for analytic or complex-step derivatives.
    """

    def initialize(self):
        self.options.declare("aircraft")
        self.options.declare("mission", default=None)
        self.options.declare("input_specs", default=())
        self.options.declare("output_specs", default=None)
        self.options.declare("runner", default=None)
        self._last_result = None

    def setup(self):
        for spec in self.input_specs():
            self.add_input(
                spec["name"],
                val=spec.get("val", 1.0),
                units=spec.get("units"),
                desc=spec.get("desc", ""),
            )

        for spec in self.output_specs():
            self.add_output(
                spec["name"],
                val=spec.get("val", 1.0),
                units=spec.get("units"),
                desc=spec.get("desc", ""),
            )

        if self.input_specs() and self.output_specs():
            self.declare_partials(of="*", wrt="*", method="fd")

    @property
    def last_result(self):
        """Return the most recent FAST result dictionary."""

        return self._last_result

    def compute(self, inputs, outputs):
        aircraft = deepcopy(self.options["aircraft"])
        mission = deepcopy(self.options["mission"])

        for spec in self.input_specs():
            target = spec.get("target", "aircraft")

            if target == "aircraft":
                set_path(aircraft, spec["path"], inputs[spec["name"]])
            elif target == "mission":
                if mission is None:
                    raise ValueError("Mission input specs require a mission baseline.")

                set_path(mission, spec["path"], inputs[spec["name"]])
            else:
                raise ValueError(f"Unsupported FAST input target: {target}")

        result = self._run_fast(aircraft, mission)
        self._last_result = result

        for spec in self.output_specs():
            outputs[spec["name"]] = scalar_value(get_path(result, spec["path"]))

    def _run_fast(self, aircraft, mission):
        runner = self.options["runner"]

        if runner is None:
            return default_runner(aircraft, mission)

        return runner(aircraft, mission)

    def input_specs(self):
        """Return normalized input specs."""

        return self.options["input_specs"] or ()

    def output_specs(self):
        """Return normalized output specs."""

        return self.options["output_specs"] or (default_mtow_output(),)


def default_runner(aircraft, mission):
    """Run the default FAST-Python native backend."""

    from fast_python import run

    return run(aircraft, mission)


def default_mtow_output():
    """Return the default MTOW output spec."""

    return {
        "name": "mtow",
        "path": ("mtow",),
        "units": "kg",
        "desc": "FAST-Python maximum takeoff weight.",
    }
