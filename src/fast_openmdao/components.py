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
        partial_derivatives: Optional mapping of (output, input) names to
            analytic scalar derivatives or callables.

    Outputs:
        OpenMDAO scalar outputs extracted from the FAST result dictionary.

    Assumptions:
        Analytic partials are used whenever supplied. Missing partials fall
        back to OpenMDAO finite difference because arbitrary FAST-Python runs
        are still black-box calculations until their internals expose
        derivative-native equations.
    """

    def initialize(self):
        self.options.declare("aircraft")
        self.options.declare("mission", default=None)
        self.options.declare("input_specs", default=())
        self.options.declare("output_specs", default=None)
        self.options.declare("runner", default=None)
        self.options.declare("partial_derivatives", default=None)
        self._last_result = None
        self._last_input_values = None

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

        if not self.input_specs() or not self.output_specs():
            return

        if self.partial_derivatives():
            self.declare_configured_partials()
        else:
            self.declare_partials(of="*", wrt="*", method="fd")

    @property
    def last_result(self):
        """Return the most recent FAST result dictionary."""

        return self._last_result

    def compute(self, inputs, outputs):
        aircraft, mission = self.build_run_inputs(inputs)
        result = self._run_fast(aircraft, mission)
        self._last_result = result
        self._last_input_values = self.input_values(inputs)

        for spec in self.output_specs():
            outputs[spec["name"]] = scalar_value(get_path(result, spec["path"]))

    def compute_partials(self, inputs, partials):
        partial_derivatives = self.partial_derivatives()

        if not partial_derivatives:
            return

        aircraft, mission = self.build_run_inputs(inputs)
        input_values = self.input_values(inputs)

        if self._last_result is None or input_values != self._last_input_values:
            self._last_result = self._run_fast(aircraft, mission)
            self._last_input_values = input_values

        for key, derivative in partial_derivatives.items():
            of, wrt = key
            partials[of, wrt] = scalar_value(
                evaluate_derivative(
                    derivative,
                    inputs,
                    self._last_result,
                    aircraft,
                    mission,
                )
            )

    def build_run_inputs(self, inputs):
        """Return aircraft and mission copies with OpenMDAO values applied."""

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

        return aircraft, mission

    def declare_configured_partials(self):
        """Declare analytic partials where provided and FD fallback elsewhere."""

        partial_derivatives = self.partial_derivatives()

        for output_spec in self.output_specs():
            for input_spec in self.input_specs():
                of = output_spec["name"]
                wrt = input_spec["name"]

                if (of, wrt) in partial_derivatives:
                    self.declare_partials(of=of, wrt=wrt)
                else:
                    self.declare_partials(of=of, wrt=wrt, method="fd")

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

    def partial_derivatives(self):
        """Return normalized analytic partial derivative specs."""

        configured = self.options["partial_derivatives"] or {}
        return {
            normalize_partial_key(key): derivative
            for key, derivative in configured.items()
        }

    def input_values(self, inputs):
        """Return current scalar input values for stale-result detection."""

        return tuple(
            (spec["name"], scalar_value(inputs[spec["name"]]))
            for spec in self.input_specs()
        )


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


def normalize_partial_key(key):
    """Return an (output, input) tuple for a partial derivative key."""

    if isinstance(key, str):
        if "," in key:
            items = key.split(",", 1)
        elif "|" in key:
            items = key.split("|", 1)
        else:
            items = key.split(":", 1)

        if len(items) != 2:
            raise ValueError(
                "Partial derivative keys must identify output and input names."
            )

        return tuple(item.strip() for item in items)

    if len(key) != 2:
        raise ValueError("Partial derivative keys must have two entries.")

    return tuple(key)


def evaluate_derivative(derivative, inputs, result, aircraft, mission):
    """Return an analytic derivative value from a scalar or callable spec."""

    if callable(derivative):
        return derivative(inputs, result, aircraft, mission)

    return derivative
