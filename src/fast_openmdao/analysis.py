# src/fast_openmdao/analysis.py

"""OpenMDAO components for FAST analysis helper equations."""

import numpy as np
import openmdao.api as om


class ConvergenceError(om.ExplicitComponent):
    """Compute FAST scalar relative convergence error.

    Inputs:
        delta: Difference between current and previous iteration values.
        baseline: Baseline value used for relative convergence.

    Outputs:
        convergence_error: ``abs(delta) / baseline``.

    Assumptions:
        The analytical derivative is defined away from ``delta == 0`` and
        ``baseline == 0``. FAST-Python treats NaN from invalid divisions as
        zero for array workflows; OpenMDAO optimization should avoid zero
        baselines for differentiable convergence checks.
    """

    def setup(self):
        self.add_input("delta", val=1.0)
        self.add_input("baseline", val=1.0)
        self.add_output("convergence_error", val=1.0)
        self.declare_partials(of="convergence_error", wrt="*")

    def compute(self, inputs, outputs):
        delta = inputs["delta"][0]
        baseline = inputs["baseline"][0]

        if baseline == 0.0:
            outputs["convergence_error"] = 0.0
            return

        outputs["convergence_error"] = abs(delta) / baseline

    def compute_partials(self, inputs, partials):
        delta = inputs["delta"][0]
        baseline = inputs["baseline"][0]

        if delta == 0.0 or baseline == 0.0:
            partials["convergence_error", "delta"] = 0.0
            partials["convergence_error", "baseline"] = 0.0
            return

        sign = 1.0 if delta > 0.0 else -1.0
        partials["convergence_error", "delta"] = sign / baseline
        partials["convergence_error", "baseline"] = -abs(delta) / baseline ** 2


class WeightSum(om.ExplicitComponent):
    """Compute FAST's scalar sum of source-weight values."""

    def initialize(self):
        self.options.declare("vec_size", default=1)

    def setup(self):
        vec_size = self.options["vec_size"]
        self.add_input("weight_values", val=np.ones(vec_size), units="kg")
        self.add_output("weight_sum", val=1.0, units="kg")
        self.declare_partials(
            of="weight_sum",
            wrt="weight_values",
            val=np.ones(vec_size),
        )

    def compute(self, inputs, outputs):
        outputs["weight_sum"] = np.sum(inputs["weight_values"])


class WingAreaFromLoading(om.ExplicitComponent):
    """Compute FAST analysis setup wing area from MTOW and wing loading."""

    def setup(self):
        self.add_input("mtow", val=1000.0, units="kg")
        self.add_input("wing_loading", val=100.0, units="kg/m**2")
        self.add_output("wing_area", val=10.0, units="m**2")
        self.declare_partials(of="wing_area", wrt="*")

    def compute(self, inputs, outputs):
        outputs["wing_area"] = wing_area_from_loading_values(
            inputs["mtow"][0],
            inputs["wing_loading"][0],
        )["wing_area"]

    def compute_partials(self, inputs, partials):
        values = wing_area_from_loading_values(
            inputs["mtow"][0],
            inputs["wing_loading"][0],
        )
        partials["wing_area", "mtow"] = values["dwing_area_dmtow"]
        partials["wing_area", "wing_loading"] = values[
            "dwing_area_dwing_loading"
        ]


class DetailedBatteryFlag(om.ExplicitComponent):
    """Return FAST analysis detailed-battery flag from prescribed cell counts."""

    def setup(self):
        self.add_input("series_cells", val=100.0)
        self.add_input("parallel_cells", val=10.0)
        self.add_output("detailed_battery_enabled", val=1.0)

    def compute(self, inputs, outputs):
        outputs["detailed_battery_enabled"] = detailed_battery_flag_values(
            inputs["series_cells"][0],
            inputs["parallel_cells"][0],
        )["detailed_battery_enabled"]


class SourceWeightVector(om.ExplicitComponent):
    """Expand FAST source-weight input into a fixed OpenMDAO vector.

    Inputs:
        source_weight: Scalar source weight or one value per active source.

    Outputs:
        source_weight_vector: One weight value per active source.

    Assumptions:
        ``source_count`` comes from FAST-Python's source-type mask and is fixed
        during model setup. A single scalar weight is broadcast across active
        sources, matching FAST's multi-source initialization convention.
    """

    def initialize(self):
        self.options.declare("source_count", default=1)
        self.options.declare("input_size", default=1)

    def setup(self):
        source_count = self.options["source_count"]
        input_size = self.options["input_size"]

        if input_size not in (1, source_count):
            raise ValueError("input_size must be 1 or source_count.")

        self.add_input("source_weight", val=np.ones(input_size), units="kg")
        self.add_output("source_weight_vector", val=np.ones(source_count), units="kg")

        rows = np.arange(source_count)
        if input_size == 1:
            cols = np.zeros(source_count, dtype=int)
        else:
            cols = np.arange(source_count)

        self.declare_partials(
            of="source_weight_vector",
            wrt="source_weight",
            rows=rows,
            cols=cols,
        )

    def compute(self, inputs, outputs):
        values = np.asarray(inputs["source_weight"], dtype=float).reshape(-1)
        source_count = self.options["source_count"]

        if np.any(np.isnan(values)):
            outputs["source_weight_vector"] = np.zeros(source_count)
        elif values.size == 1 and source_count > 1:
            outputs["source_weight_vector"] = np.ones(source_count) * values[0]
        else:
            outputs["source_weight_vector"] = values

    def compute_partials(self, inputs, partials):
        values = np.asarray(inputs["source_weight"], dtype=float).reshape(-1)
        source_count = self.options["source_count"]

        if np.any(np.isnan(values)):
            partials["source_weight_vector", "source_weight"] = np.zeros(source_count)
        else:
            partials["source_weight_vector", "source_weight"] = np.ones(source_count)


class AnalysisWeightUpdate(om.ExplicitComponent):
    """Update FAST analysis-loop MTOW, source weights, and OEW.

    Inputs:
        mtow: Current maximum takeoff weight in kg.
        fuel_weight: Previous per-source fuel weights in kg.
        battery_weight: Previous per-source battery weights in kg.
        fuel_burn: Mission fuel burn or final fuel source weights in kg.
        resized_battery_weight: Battery weights after mission resizing in kg.
        payload_weight: Payload weight in kg.
        crew_weight: Crew weight in kg.
        oew: Current OEW in kg, passed through when OEW updating is disabled.

    Outputs:
        mtow_new: Updated MTOW in kg.
        fuel_weight_new: Updated fuel source weights in kg.
        battery_weight_new: Updated battery source weights in kg.
        oew_new: Updated or passed-through OEW in kg.

    Assumptions:
        ``update_oew`` is the discrete FAST analysis-type branch
        ``analysis_type > -2``. FAST treats ``fuel_burn`` as a vector-compatible
        value and computes deltas against the previous source weights.
    """

    def initialize(self):
        self.options.declare("num_fuel_sources", default=1)
        self.options.declare("num_battery_sources", default=1)
        self.options.declare("update_oew", default=True)

    def setup(self):
        nfuel = self.options["num_fuel_sources"]
        nbatt = self.options["num_battery_sources"]
        self.add_input("mtow", val=1000.0, units="kg")
        self.add_input("fuel_weight", val=np.ones(nfuel), units="kg")
        self.add_input("battery_weight", val=np.ones(nbatt), units="kg")
        self.add_input("fuel_burn", val=np.ones(nfuel), units="kg")
        self.add_input("resized_battery_weight", val=np.ones(nbatt), units="kg")
        self.add_input("payload_weight", val=100.0, units="kg")
        self.add_input("crew_weight", val=50.0, units="kg")
        self.add_input("oew", val=500.0, units="kg")
        self.add_output("mtow_new", val=1000.0, units="kg")
        self.add_output("fuel_weight_new", val=np.ones(nfuel), units="kg")
        self.add_output("battery_weight_new", val=np.ones(nbatt), units="kg")
        self.add_output("oew_new", val=500.0, units="kg")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = analysis_weight_update_values(
            inputs["mtow"][0],
            inputs["fuel_weight"],
            inputs["battery_weight"],
            inputs["fuel_burn"],
            inputs["resized_battery_weight"],
            inputs["payload_weight"][0],
            inputs["crew_weight"][0],
            inputs["oew"][0],
            self.options["update_oew"],
        )
        outputs["mtow_new"] = values["mtow_new"]
        outputs["fuel_weight_new"] = values["fuel_weight_new"]
        outputs["battery_weight_new"] = values["battery_weight_new"]
        outputs["oew_new"] = values["oew_new"]

    def compute_partials(self, inputs, partials):
        values = analysis_weight_update_values(
            inputs["mtow"][0],
            inputs["fuel_weight"],
            inputs["battery_weight"],
            inputs["fuel_burn"],
            inputs["resized_battery_weight"],
            inputs["payload_weight"][0],
            inputs["crew_weight"][0],
            inputs["oew"][0],
            self.options["update_oew"],
        )

        for output_name in analysis_weight_update_output_names():
            for input_name in analysis_weight_update_input_names():
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


def analysis_weight_update_input_names():
    """Return analysis weight-update component input names."""

    return [
        "mtow",
        "fuel_weight",
        "battery_weight",
        "fuel_burn",
        "resized_battery_weight",
        "payload_weight",
        "crew_weight",
        "oew",
    ]


def analysis_weight_update_output_names():
    """Return analysis weight-update component output names."""

    return [
        "mtow_new",
        "fuel_weight_new",
        "battery_weight_new",
        "oew_new",
    ]


def analysis_weight_update_values(
    mtow,
    fuel_weight,
    battery_weight,
    fuel_burn,
    resized_battery_weight,
    payload_weight,
    crew_weight,
    oew,
    update_oew,
):
    """Return FAST analysis-loop weight update values and derivatives."""

    mtow = float(mtow)
    fuel_weight = np.asarray(fuel_weight, dtype=float).reshape(-1)
    battery_weight = np.asarray(battery_weight, dtype=float).reshape(-1)
    fuel_burn = np.asarray(fuel_burn, dtype=float).reshape(-1)
    resized_battery_weight = np.asarray(resized_battery_weight, dtype=float).reshape(-1)
    payload_weight = float(payload_weight)
    crew_weight = float(crew_weight)
    oew = float(oew)
    fuel_delta = fuel_burn - fuel_weight
    battery_delta = resized_battery_weight - battery_weight
    mtow_new = mtow + np.sum(fuel_delta) + np.sum(battery_delta)
    fuel_weight_new = fuel_weight + fuel_delta
    battery_weight_new = battery_weight + battery_delta

    if update_oew:
        oew_new = (
            mtow_new
            - np.sum(fuel_weight_new)
            - np.sum(battery_weight_new)
            - payload_weight
            - crew_weight
        )
    else:
        oew_new = oew

    values = {
        "mtow_new": mtow_new,
        "fuel_weight_new": fuel_weight_new,
        "battery_weight_new": battery_weight_new,
        "oew_new": oew_new,
    }
    nfuel = len(fuel_weight)
    nbatt = len(battery_weight)

    for output_name in analysis_weight_update_output_names():
        for input_name in analysis_weight_update_input_names():
            if output_name == "fuel_weight_new":
                output_size = nfuel
            elif output_name == "battery_weight_new":
                output_size = nbatt
            else:
                output_size = 1

            if input_name in ("fuel_weight", "fuel_burn"):
                input_size = nfuel
            elif input_name in ("battery_weight", "resized_battery_weight"):
                input_size = nbatt
            else:
                input_size = 1

            values[f"d{output_name}_d{input_name}"] = np.zeros(
                (output_size, input_size)
            )

    values["dmtow_new_dmtow"] = np.asarray([[1.0]])
    values["dmtow_new_dfuel_weight"] = -np.ones((1, nfuel))
    values["dmtow_new_dbattery_weight"] = -np.ones((1, nbatt))
    values["dmtow_new_dfuel_burn"] = np.ones((1, nfuel))
    values["dmtow_new_dresized_battery_weight"] = np.ones((1, nbatt))
    values["dfuel_weight_new_dfuel_burn"] = np.eye(nfuel)
    values["dfuel_weight_new_dfuel_weight"] = np.zeros((nfuel, nfuel))
    values["dbattery_weight_new_dresized_battery_weight"] = np.eye(nbatt)
    values["dbattery_weight_new_dbattery_weight"] = np.zeros((nbatt, nbatt))

    if update_oew:
        values["doew_new_dmtow"] = np.asarray([[1.0]])
        values["doew_new_dfuel_weight"] = -np.ones((1, nfuel))
        values["doew_new_dbattery_weight"] = -np.ones((1, nbatt))
        values["doew_new_dfuel_burn"] = np.zeros((1, nfuel))
        values["doew_new_dresized_battery_weight"] = np.zeros((1, nbatt))
        values["doew_new_dpayload_weight"] = np.asarray([[-1.0]])
        values["doew_new_dcrew_weight"] = np.asarray([[-1.0]])
        values["doew_new_doew"] = np.asarray([[0.0]])
    else:
        values["doew_new_doew"] = np.asarray([[1.0]])

    return values


def wing_area_from_loading_values(mtow, wing_loading):
    """Return FAST analysis setup wing area and derivatives."""

    mtow = float(mtow)
    wing_loading = float(wing_loading)
    wing_area = mtow / wing_loading
    return {
        "wing_area": wing_area,
        "dwing_area_dmtow": 1.0 / wing_loading,
        "dwing_area_dwing_loading": -mtow / wing_loading ** 2,
    }


def detailed_battery_flag_values(series_cells, parallel_cells):
    """Return FAST detailed-battery flag for fixed scalar inputs."""

    if np.isnan(series_cells) or np.isnan(parallel_cells):
        value = 0.0
    else:
        value = 1.0

    return {"detailed_battery_enabled": value}
