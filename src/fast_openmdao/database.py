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


class DatabaseWeightFractions(om.ExplicitComponent):
    """Compute FAST database weight fractions and wing loading.

    Inputs:
        mtow: Maximum takeoff weight in kg.
        oew: Operating empty weight in kg.
        fuel_weight: Fuel weight in kg.
        engine_dry_weight: Dry weight per engine in kg.
        wing_area: Wing area in m**2.

    Outputs:
        airframe_weight: OEW minus installed engine dry weight in kg.
        oew_mtow: OEW to MTOW ratio.
        engine_fraction: Installed engine dry weight to MTOW ratio.
        fuel_fraction: Fuel weight to MTOW ratio.
        wing_loading: MTOW divided by wing area in kg/m**2.

    Assumptions:
        ``num_engines`` is discrete database metadata and is therefore an
        option. These equations appear in FAST ``CalcFanVals`` and
        ``CalcPropVals`` preprocessing.
    """

    def initialize(self):
        self.options.declare("num_engines", default=1)

    def setup(self):
        self.add_input("mtow", val=10000.0, units="kg")
        self.add_input("oew", val=6000.0, units="kg")
        self.add_input("fuel_weight", val=2000.0, units="kg")
        self.add_input("engine_dry_weight", val=500.0, units="kg")
        self.add_input("wing_area", val=50.0, units="m**2")
        self.add_output("airframe_weight", val=5000.0, units="kg")
        self.add_output("oew_mtow", val=0.6)
        self.add_output("engine_fraction", val=0.05)
        self.add_output("fuel_fraction", val=0.2)
        self.add_output("wing_loading", val=200.0, units="kg/m**2")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = database_weight_fraction_values(
            inputs["mtow"][0],
            inputs["oew"][0],
            inputs["fuel_weight"][0],
            inputs["engine_dry_weight"][0],
            inputs["wing_area"][0],
            self.options["num_engines"],
        )

        for name in database_weight_fraction_output_names():
            outputs[name] = values[name]

    def compute_partials(self, inputs, partials):
        values = database_weight_fraction_values(
            inputs["mtow"][0],
            inputs["oew"][0],
            inputs["fuel_weight"][0],
            inputs["engine_dry_weight"][0],
            inputs["wing_area"][0],
            self.options["num_engines"],
        )

        for output_name in database_weight_fraction_output_names():
            for input_name in database_weight_fraction_input_names():
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


class DatabaseGeometryLoads(om.ExplicitComponent):
    """Compute FAST database geometry and load preprocessing values.

    Inputs:
        mtow: Maximum takeoff weight in kg.
        mzfw: Maximum zero-fuel weight in kg.
        fuel_weight: Fuel weight in kg.
        engine_dry_weight: Dry weight per engine in kg.
        cargo_weight: Cargo weight in kg.
        max_payload: Direct payload value in kg when selected by option.
        max_passengers: Passenger count used by FAST's 95 kg/passenger rule.
        wing_span: Wing span in m.
        wing_area: Wing area in m**2.
        tip_chord: Wing tip chord in m.
        root_chord: Wing root chord in m.

    Outputs:
        taper_ratio: Tip chord divided by root chord.
        aspect_ratio: Span squared divided by wing area.
        payload: FAST database payload estimate in kg.
        burden: Payload plus fuel plus installed engine dry weight in kg.
        structure_burden: Ratio of non-burden weight to burden weight.
        mzfw_mtow: MZFW to MTOW ratio.

    Assumptions:
        FAST switches between passenger/cargo payload and direct max-payload
        based on missing database fields. That discrete choice is an option so
        the component remains smooth for optimization.
    """

    def initialize(self):
        self.options.declare("num_engines", default=1)
        self.options.declare("payload_source", default="pax_cargo")

    def setup(self):
        self.add_input("mtow", val=10000.0, units="kg")
        self.add_input("mzfw", val=8000.0, units="kg")
        self.add_input("fuel_weight", val=2000.0, units="kg")
        self.add_input("engine_dry_weight", val=500.0, units="kg")
        self.add_input("cargo_weight", val=0.0, units="kg")
        self.add_input("max_payload", val=3000.0, units="kg")
        self.add_input("max_passengers", val=40.0)
        self.add_input("wing_span", val=25.0, units="m")
        self.add_input("wing_area", val=60.0, units="m**2")
        self.add_input("tip_chord", val=1.5, units="m")
        self.add_input("root_chord", val=3.0, units="m")
        self.add_output("taper_ratio", val=0.5)
        self.add_output("aspect_ratio", val=10.0)
        self.add_output("payload", val=3000.0, units="kg")
        self.add_output("burden", val=5000.0, units="kg")
        self.add_output("structure_burden", val=1.0)
        self.add_output("mzfw_mtow", val=0.8)
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = database_geometry_load_values(
            inputs["mtow"][0],
            inputs["mzfw"][0],
            inputs["fuel_weight"][0],
            inputs["engine_dry_weight"][0],
            inputs["cargo_weight"][0],
            inputs["max_payload"][0],
            inputs["max_passengers"][0],
            inputs["wing_span"][0],
            inputs["wing_area"][0],
            inputs["tip_chord"][0],
            inputs["root_chord"][0],
            self.options["num_engines"],
            self.options["payload_source"],
        )

        for name in database_geometry_load_output_names():
            outputs[name] = values[name]

    def compute_partials(self, inputs, partials):
        values = database_geometry_load_values(
            inputs["mtow"][0],
            inputs["mzfw"][0],
            inputs["fuel_weight"][0],
            inputs["engine_dry_weight"][0],
            inputs["cargo_weight"][0],
            inputs["max_payload"][0],
            inputs["max_passengers"][0],
            inputs["wing_span"][0],
            inputs["wing_area"][0],
            inputs["tip_chord"][0],
            inputs["root_chord"][0],
            self.options["num_engines"],
            self.options["payload_source"],
        )

        for output_name in database_geometry_load_output_names():
            for input_name in database_geometry_load_input_names():
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


class DatabaseFanThrustNormalization(om.ExplicitComponent):
    """Compute FAST turbofan database thrust totals and thrust loading."""

    def initialize(self):
        self.options.declare("num_engines", default=1)
        self.options.declare("thrust_source", default="engine")

    def setup(self):
        self.add_input("mtow", val=70000.0, units="kg")
        self.add_input("engine_thrust_sls", val=100000.0, units="N")
        self.add_input("engine_thrust_max", val=120000.0, units="N")
        self.add_input("engine_thrust_cruise", val=60000.0, units="N")
        self.add_input("specified_thrust_sls", val=100000.0, units="N")
        self.add_input("specified_thrust_max", val=120000.0, units="N")
        self.add_output("thrust_loading_sls", val=0.3)
        self.add_output("thrust_sls", val=200000.0, units="N")
        self.add_output("thrust_max", val=240000.0, units="N")
        self.add_output("thrust_cruise", val=120000.0, units="N")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = database_fan_thrust_normalization_values(
            inputs["mtow"][0],
            inputs["engine_thrust_sls"][0],
            inputs["engine_thrust_max"][0],
            inputs["engine_thrust_cruise"][0],
            inputs["specified_thrust_sls"][0],
            inputs["specified_thrust_max"][0],
            self.options["num_engines"],
            self.options["thrust_source"],
        )

        for name in database_fan_thrust_output_names():
            outputs[name] = values[name]

    def compute_partials(self, inputs, partials):
        values = database_fan_thrust_normalization_values(
            inputs["mtow"][0],
            inputs["engine_thrust_sls"][0],
            inputs["engine_thrust_max"][0],
            inputs["engine_thrust_cruise"][0],
            inputs["specified_thrust_sls"][0],
            inputs["specified_thrust_max"][0],
            self.options["num_engines"],
            self.options["thrust_source"],
        )

        for output_name in database_fan_thrust_output_names():
            for input_name in database_fan_thrust_input_names():
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


class DatabasePropPowerNormalization(om.ExplicitComponent):
    """Compute FAST turboprop database power totals and SLS power loading."""

    def initialize(self):
        self.options.declare("num_engines", default=1)
        self.options.declare("sls_power_source", default="engine")
        self.options.declare("continuous_power_source", default="engine_equivalent")

    def setup(self):
        self.add_input("mtow", val=18500.0, units="kg")
        self.add_input("engine_power_sls", val=1200.0, units="kW")
        self.add_input("engine_power_sls_equivalent", val=1200.0, units="kW")
        self.add_input("engine_power_continuous_equivalent", val=1000.0, units="kW")
        self.add_input("specified_power_sls", val=1200.0, units="kW")
        self.add_input("specified_power_continuous", val=1000.0, units="kW")
        self.add_input("climb_power", val=900.0, units="kW")
        self.add_input("cruise_power", val=800.0, units="kW")
        self.add_output("sea_level_power", val=2400000.0, units="W")
        self.add_output("continuous_power", val=2000000.0, units="W")
        self.add_output("climb_power_total", val=1800000.0, units="W")
        self.add_output("cruise_power_total", val=1600000.0, units="W")
        self.add_output("sea_level_power_loading", val=0.13, units="kW/kg")
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = database_prop_power_normalization_values(
            inputs["mtow"][0],
            inputs["engine_power_sls"][0],
            inputs["engine_power_sls_equivalent"][0],
            inputs["engine_power_continuous_equivalent"][0],
            inputs["specified_power_sls"][0],
            inputs["specified_power_continuous"][0],
            inputs["climb_power"][0],
            inputs["cruise_power"][0],
            self.options["num_engines"],
            self.options["sls_power_source"],
            self.options["continuous_power_source"],
        )

        for name in database_prop_power_output_names():
            outputs[name] = values[name]

    def compute_partials(self, inputs, partials):
        values = database_prop_power_normalization_values(
            inputs["mtow"][0],
            inputs["engine_power_sls"][0],
            inputs["engine_power_sls_equivalent"][0],
            inputs["engine_power_continuous_equivalent"][0],
            inputs["specified_power_sls"][0],
            inputs["specified_power_continuous"][0],
            inputs["climb_power"][0],
            inputs["cruise_power"][0],
            self.options["num_engines"],
            self.options["sls_power_source"],
            self.options["continuous_power_source"],
        )

        for output_name in database_prop_power_output_names():
            for input_name in database_prop_power_input_names():
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


class TurbofanCruiseLiftDragEstimate(om.ExplicitComponent):
    """Compute FAST turbofan database cruise lift-to-drag estimates."""

    def initialize(self):
        self.options.declare("num_engines", default=1)

    def setup(self):
        self.add_input("mtow", val=70000.0, units="kg")
        self.add_input("fuel_weight", val=15000.0, units="kg")
        self.add_input("range", val=3000000.0, units="m")
        self.add_input("tsfc_cruise", val=0.6)
        self.add_input("cruise_mach", val=0.78)
        self.add_input("temperature", val=223.15, units="K")
        self.add_input("engine_thrust_cruise", val=60000.0, units="N")
        self.add_output("breguet_lift_drag", val=9.0)
        self.add_output("thrust_lift_drag", val=6.0)
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        values = turbofan_cruise_lift_drag_values(
            inputs["mtow"][0],
            inputs["fuel_weight"][0],
            inputs["range"][0],
            inputs["tsfc_cruise"][0],
            inputs["cruise_mach"][0],
            inputs["temperature"][0],
            inputs["engine_thrust_cruise"][0],
            self.options["num_engines"],
        )

        for name in turbofan_cruise_lift_drag_output_names():
            outputs[name] = values[name]

    def compute_partials(self, inputs, partials):
        values = turbofan_cruise_lift_drag_values(
            inputs["mtow"][0],
            inputs["fuel_weight"][0],
            inputs["range"][0],
            inputs["tsfc_cruise"][0],
            inputs["cruise_mach"][0],
            inputs["temperature"][0],
            inputs["engine_thrust_cruise"][0],
            self.options["num_engines"],
        )

        for output_name in turbofan_cruise_lift_drag_output_names():
            for input_name in turbofan_cruise_lift_drag_input_names():
                partials[output_name, input_name] = values[
                    f"d{output_name}_d{input_name}"
                ]


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


def turbofan_cruise_lift_drag_input_names():
    """Return turbofan cruise L/D component input names."""

    return [
        "mtow",
        "fuel_weight",
        "range",
        "tsfc_cruise",
        "cruise_mach",
        "temperature",
        "engine_thrust_cruise",
    ]


def turbofan_cruise_lift_drag_output_names():
    """Return turbofan cruise L/D component output names."""

    return [
        "breguet_lift_drag",
        "thrust_lift_drag",
    ]


def turbofan_cruise_lift_drag_values(
    mtow,
    fuel_weight,
    mission_range,
    tsfc_cruise,
    cruise_mach,
    temperature,
    engine_thrust_cruise,
    num_engines,
):
    """Return FAST turbofan cruise L/D estimates and derivatives."""

    mtow = float(mtow)
    fuel_weight = float(fuel_weight)
    mission_range = float(mission_range)
    tsfc_cruise = float(tsfc_cruise)
    cruise_mach = float(cruise_mach)
    temperature = float(temperature)
    engine_thrust_cruise = float(engine_thrust_cruise)
    num_engines = float(num_engines)
    tsfc_si = tsfc_cruise * 0.453592 / 4.44822 / 3600.0 * 9.81
    speed = cruise_mach * np.sqrt(1.4 * 287.0 * temperature)
    weight_ratio = np.log(mtow / (mtow - fuel_weight))
    breguet_lift_drag = mission_range * tsfc_si / speed / weight_ratio
    thrust_lift_drag = (
        mtow * 0.995 * 0.985 * 9.81 / (engine_thrust_cruise * num_engines)
    )
    values = {
        "breguet_lift_drag": breguet_lift_drag,
        "thrust_lift_drag": thrust_lift_drag,
    }

    for output_name in turbofan_cruise_lift_drag_output_names():
        for input_name in turbofan_cruise_lift_drag_input_names():
            values[f"d{output_name}_d{input_name}"] = 0.0

    dweight_ratio_dmtow = 1.0 / mtow - 1.0 / (mtow - fuel_weight)
    dweight_ratio_dfuel = 1.0 / (mtow - fuel_weight)
    values["dbreguet_lift_drag_drange"] = breguet_lift_drag / mission_range
    values["dbreguet_lift_drag_dtsfc_cruise"] = breguet_lift_drag / tsfc_cruise
    values["dbreguet_lift_drag_dcruise_mach"] = -breguet_lift_drag / cruise_mach
    values["dbreguet_lift_drag_dtemperature"] = -0.5 * breguet_lift_drag / temperature
    values["dbreguet_lift_drag_dmtow"] = (
        -breguet_lift_drag / weight_ratio * dweight_ratio_dmtow
    )
    values["dbreguet_lift_drag_dfuel_weight"] = (
        -breguet_lift_drag / weight_ratio * dweight_ratio_dfuel
    )
    values["dthrust_lift_drag_dmtow"] = thrust_lift_drag / mtow
    values["dthrust_lift_drag_dengine_thrust_cruise"] = (
        -thrust_lift_drag / engine_thrust_cruise
    )
    return values


def database_weight_fraction_input_names():
    """Return database weight-fraction component input names."""

    return [
        "mtow",
        "oew",
        "fuel_weight",
        "engine_dry_weight",
        "wing_area",
    ]


def database_weight_fraction_output_names():
    """Return database weight-fraction component output names."""

    return [
        "airframe_weight",
        "oew_mtow",
        "engine_fraction",
        "fuel_fraction",
        "wing_loading",
    ]


def database_weight_fraction_values(
    mtow,
    oew,
    fuel_weight,
    engine_dry_weight,
    wing_area,
    num_engines,
):
    """Return FAST database weight-derived values and derivatives."""

    mtow = float(mtow)
    oew = float(oew)
    fuel_weight = float(fuel_weight)
    engine_dry_weight = float(engine_dry_weight)
    wing_area = float(wing_area)
    num_engines = float(num_engines)
    installed_engine_weight = engine_dry_weight * num_engines
    values = {
        "airframe_weight": oew - installed_engine_weight,
        "oew_mtow": oew / mtow,
        "engine_fraction": installed_engine_weight / mtow,
        "fuel_fraction": fuel_weight / mtow,
        "wing_loading": mtow / wing_area,
    }

    for output_name in database_weight_fraction_output_names():
        for input_name in database_weight_fraction_input_names():
            values[f"d{output_name}_d{input_name}"] = 0.0

    values["dairframe_weight_doew"] = 1.0
    values["dairframe_weight_dengine_dry_weight"] = -num_engines
    values["doew_mtow_doew"] = 1.0 / mtow
    values["doew_mtow_dmtow"] = -oew / mtow ** 2
    values["dengine_fraction_dengine_dry_weight"] = num_engines / mtow
    values["dengine_fraction_dmtow"] = -installed_engine_weight / mtow ** 2
    values["dfuel_fraction_dfuel_weight"] = 1.0 / mtow
    values["dfuel_fraction_dmtow"] = -fuel_weight / mtow ** 2
    values["dwing_loading_dmtow"] = 1.0 / wing_area
    values["dwing_loading_dwing_area"] = -mtow / wing_area ** 2
    return values


def database_geometry_load_input_names():
    """Return database geometry/load component input names."""

    return [
        "mtow",
        "mzfw",
        "fuel_weight",
        "engine_dry_weight",
        "cargo_weight",
        "max_payload",
        "max_passengers",
        "wing_span",
        "wing_area",
        "tip_chord",
        "root_chord",
    ]


def database_geometry_load_output_names():
    """Return database geometry/load component output names."""

    return [
        "taper_ratio",
        "aspect_ratio",
        "payload",
        "burden",
        "structure_burden",
        "mzfw_mtow",
    ]


def database_geometry_load_values(
    mtow,
    mzfw,
    fuel_weight,
    engine_dry_weight,
    cargo_weight,
    max_payload,
    max_passengers,
    wing_span,
    wing_area,
    tip_chord,
    root_chord,
    num_engines,
    payload_source,
):
    """Return FAST database geometry/load values and derivatives."""

    mtow = float(mtow)
    mzfw = float(mzfw)
    fuel_weight = float(fuel_weight)
    engine_dry_weight = float(engine_dry_weight)
    cargo_weight = float(cargo_weight)
    max_payload = float(max_payload)
    max_passengers = float(max_passengers)
    wing_span = float(wing_span)
    wing_area = float(wing_area)
    tip_chord = float(tip_chord)
    root_chord = float(root_chord)
    num_engines = float(num_engines)

    if payload_source == "pax_cargo":
        payload = cargo_weight + max_passengers * 95.0
        dpayload_dcargo_weight = 1.0
        dpayload_dmax_payload = 0.0
        dpayload_dmax_passengers = 95.0
    elif payload_source == "max_payload":
        payload = max_payload
        dpayload_dcargo_weight = 0.0
        dpayload_dmax_payload = 1.0
        dpayload_dmax_passengers = 0.0
    else:
        raise ValueError("payload_source must be 'pax_cargo' or 'max_payload'")

    installed_engine_weight = engine_dry_weight * num_engines
    burden = payload + fuel_weight + installed_engine_weight
    structure_burden = (mtow - burden) / burden
    values = {
        "taper_ratio": tip_chord / root_chord,
        "aspect_ratio": wing_span ** 2 / wing_area,
        "payload": payload,
        "burden": burden,
        "structure_burden": structure_burden,
        "mzfw_mtow": mzfw / mtow,
    }

    for output_name in database_geometry_load_output_names():
        for input_name in database_geometry_load_input_names():
            values[f"d{output_name}_d{input_name}"] = 0.0

    values["dtaper_ratio_dtip_chord"] = 1.0 / root_chord
    values["dtaper_ratio_droot_chord"] = -tip_chord / root_chord ** 2
    values["daspect_ratio_dwing_span"] = 2.0 * wing_span / wing_area
    values["daspect_ratio_dwing_area"] = -wing_span ** 2 / wing_area ** 2
    values["dpayload_dcargo_weight"] = dpayload_dcargo_weight
    values["dpayload_dmax_payload"] = dpayload_dmax_payload
    values["dpayload_dmax_passengers"] = dpayload_dmax_passengers
    values["dburden_dfuel_weight"] = 1.0
    values["dburden_dengine_dry_weight"] = num_engines
    values["dburden_dcargo_weight"] = dpayload_dcargo_weight
    values["dburden_dmax_payload"] = dpayload_dmax_payload
    values["dburden_dmax_passengers"] = dpayload_dmax_passengers
    values["dstructure_burden_dmtow"] = 1.0 / burden
    dstructure_dburden = -mtow / burden ** 2
    values["dstructure_burden_dfuel_weight"] = dstructure_dburden
    values["dstructure_burden_dengine_dry_weight"] = (
        dstructure_dburden * num_engines
    )
    values["dstructure_burden_dcargo_weight"] = (
        dstructure_dburden * dpayload_dcargo_weight
    )
    values["dstructure_burden_dmax_payload"] = (
        dstructure_dburden * dpayload_dmax_payload
    )
    values["dstructure_burden_dmax_passengers"] = (
        dstructure_dburden * dpayload_dmax_passengers
    )
    values["dmzfw_mtow_dmzfw"] = 1.0 / mtow
    values["dmzfw_mtow_dmtow"] = -mzfw / mtow ** 2
    return values


def database_fan_thrust_input_names():
    """Return turbofan thrust-normalization input names."""

    return [
        "mtow",
        "engine_thrust_sls",
        "engine_thrust_max",
        "engine_thrust_cruise",
        "specified_thrust_sls",
        "specified_thrust_max",
    ]


def database_fan_thrust_output_names():
    """Return turbofan thrust-normalization output names."""

    return [
        "thrust_loading_sls",
        "thrust_sls",
        "thrust_max",
        "thrust_cruise",
    ]


def database_fan_thrust_normalization_values(
    mtow,
    engine_thrust_sls,
    engine_thrust_max,
    engine_thrust_cruise,
    specified_thrust_sls,
    specified_thrust_max,
    num_engines,
    thrust_source,
):
    """Return FAST turbofan thrust totals/loading and derivatives."""

    mtow = float(mtow)
    engine_thrust_sls = float(engine_thrust_sls)
    engine_thrust_max = float(engine_thrust_max)
    engine_thrust_cruise = float(engine_thrust_cruise)
    specified_thrust_sls = float(specified_thrust_sls)
    specified_thrust_max = float(specified_thrust_max)
    num_engines = float(num_engines)

    if thrust_source == "engine":
        loading_reference = engine_thrust_max
        thrust_sls_reference = engine_thrust_sls
        thrust_max_reference = engine_thrust_max
        dloading_reference_dengine_thrust_max = 1.0
        dloading_reference_dspecified_thrust_max = 0.0
        dsls_reference_dengine_thrust_sls = 1.0
        dsls_reference_dspecified_thrust_sls = 0.0
        dmax_reference_dengine_thrust_max = 1.0
        dmax_reference_dspecified_thrust_max = 0.0
    elif thrust_source == "specified":
        loading_reference = specified_thrust_max
        thrust_sls_reference = specified_thrust_sls
        thrust_max_reference = specified_thrust_max
        dloading_reference_dengine_thrust_max = 0.0
        dloading_reference_dspecified_thrust_max = 1.0
        dsls_reference_dengine_thrust_sls = 0.0
        dsls_reference_dspecified_thrust_sls = 1.0
        dmax_reference_dengine_thrust_max = 0.0
        dmax_reference_dspecified_thrust_max = 1.0
    else:
        raise ValueError("thrust_source must be 'engine' or 'specified'")

    thrust_loading_sls = loading_reference * num_engines / mtow / 9.81
    values = {
        "thrust_loading_sls": thrust_loading_sls,
        "thrust_sls": thrust_sls_reference * num_engines,
        "thrust_max": thrust_max_reference * num_engines,
        "thrust_cruise": engine_thrust_cruise * num_engines,
    }

    for output_name in database_fan_thrust_output_names():
        for input_name in database_fan_thrust_input_names():
            values[f"d{output_name}_d{input_name}"] = 0.0

    loading_scale = num_engines / mtow / 9.81
    values["dthrust_loading_sls_dmtow"] = -thrust_loading_sls / mtow
    values["dthrust_loading_sls_dengine_thrust_max"] = (
        dloading_reference_dengine_thrust_max * loading_scale
    )
    values["dthrust_loading_sls_dspecified_thrust_max"] = (
        dloading_reference_dspecified_thrust_max * loading_scale
    )
    values["dthrust_sls_dengine_thrust_sls"] = (
        dsls_reference_dengine_thrust_sls * num_engines
    )
    values["dthrust_sls_dspecified_thrust_sls"] = (
        dsls_reference_dspecified_thrust_sls * num_engines
    )
    values["dthrust_max_dengine_thrust_max"] = (
        dmax_reference_dengine_thrust_max * num_engines
    )
    values["dthrust_max_dspecified_thrust_max"] = (
        dmax_reference_dspecified_thrust_max * num_engines
    )
    values["dthrust_cruise_dengine_thrust_cruise"] = num_engines
    return values


def database_prop_power_input_names():
    """Return turboprop power-normalization input names."""

    return [
        "mtow",
        "engine_power_sls",
        "engine_power_sls_equivalent",
        "engine_power_continuous_equivalent",
        "specified_power_sls",
        "specified_power_continuous",
        "climb_power",
        "cruise_power",
    ]


def database_prop_power_output_names():
    """Return turboprop power-normalization output names."""

    return [
        "sea_level_power",
        "continuous_power",
        "climb_power_total",
        "cruise_power_total",
        "sea_level_power_loading",
    ]


def database_prop_power_normalization_values(
    mtow,
    engine_power_sls,
    engine_power_sls_equivalent,
    engine_power_continuous_equivalent,
    specified_power_sls,
    specified_power_continuous,
    climb_power,
    cruise_power,
    num_engines,
    sls_power_source,
    continuous_power_source,
):
    """Return FAST turboprop power totals/loading and derivatives."""

    mtow = float(mtow)
    engine_power_sls = float(engine_power_sls)
    engine_power_sls_equivalent = float(engine_power_sls_equivalent)
    engine_power_continuous_equivalent = float(engine_power_continuous_equivalent)
    specified_power_sls = float(specified_power_sls)
    specified_power_continuous = float(specified_power_continuous)
    climb_power = float(climb_power)
    cruise_power = float(cruise_power)
    num_engines = float(num_engines)

    if sls_power_source == "engine":
        sls_reference = engine_power_sls
        dsls_dengine_power_sls = 1.0
        dsls_dengine_power_sls_equivalent = 0.0
        dsls_dspecified_power_sls = 0.0
    elif sls_power_source == "engine_equivalent":
        sls_reference = engine_power_sls_equivalent
        dsls_dengine_power_sls = 0.0
        dsls_dengine_power_sls_equivalent = 1.0
        dsls_dspecified_power_sls = 0.0
    elif sls_power_source == "specified":
        sls_reference = specified_power_sls
        dsls_dengine_power_sls = 0.0
        dsls_dengine_power_sls_equivalent = 0.0
        dsls_dspecified_power_sls = 1.0
    else:
        raise ValueError(
            "sls_power_source must be 'engine', 'engine_equivalent', or 'specified'"
        )

    if continuous_power_source == "engine_equivalent":
        continuous_reference = engine_power_continuous_equivalent
        dcont_dengine_power_continuous_equivalent = 1.0
        dcont_dspecified_power_continuous = 0.0
    elif continuous_power_source == "specified":
        continuous_reference = specified_power_continuous
        dcont_dengine_power_continuous_equivalent = 0.0
        dcont_dspecified_power_continuous = 1.0
    else:
        raise ValueError(
            "continuous_power_source must be 'engine_equivalent' or 'specified'"
        )

    total_scale = 1000.0 * num_engines
    sea_level_power = sls_reference * total_scale
    values = {
        "sea_level_power": sea_level_power,
        "continuous_power": continuous_reference * total_scale,
        "climb_power_total": climb_power * total_scale,
        "cruise_power_total": cruise_power * total_scale,
        "sea_level_power_loading": sea_level_power / mtow / 1000.0,
    }

    for output_name in database_prop_power_output_names():
        for input_name in database_prop_power_input_names():
            values[f"d{output_name}_d{input_name}"] = 0.0

    values["dsea_level_power_dengine_power_sls"] = (
        dsls_dengine_power_sls * total_scale
    )
    values["dsea_level_power_dengine_power_sls_equivalent"] = (
        dsls_dengine_power_sls_equivalent * total_scale
    )
    values["dsea_level_power_dspecified_power_sls"] = (
        dsls_dspecified_power_sls * total_scale
    )
    values["dcontinuous_power_dengine_power_continuous_equivalent"] = (
        dcont_dengine_power_continuous_equivalent * total_scale
    )
    values["dcontinuous_power_dspecified_power_continuous"] = (
        dcont_dspecified_power_continuous * total_scale
    )
    values["dclimb_power_total_dclimb_power"] = total_scale
    values["dcruise_power_total_dcruise_power"] = total_scale
    values["dsea_level_power_loading_dmtow"] = (
        -values["sea_level_power_loading"] / mtow
    )
    values["dsea_level_power_loading_dengine_power_sls"] = (
        dsls_dengine_power_sls * num_engines / mtow
    )
    values["dsea_level_power_loading_dengine_power_sls_equivalent"] = (
        dsls_dengine_power_sls_equivalent * num_engines / mtow
    )
    values["dsea_level_power_loading_dspecified_power_sls"] = (
        dsls_dspecified_power_sls * num_engines / mtow
    )
    return values
