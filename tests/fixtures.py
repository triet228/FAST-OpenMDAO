# tests/fixtures.py

"""Shared FAST input fixtures for OpenMDAO tests."""


def make_compact_aircraft():
    """Return a compact all-electric aircraft for fast native smoke tests."""

    return {
        "Settings": {
            "ClbPoints": 3,
            "CrsPoints": 3,
            "DesPoints": 3,
            "Analysis": {
                "Type": -2,
                "MaxIter": 3,
            },
        },
        "Specs": {
            "TLAR": {
                "Class": "Turboprop",
                "MaxPax": 0,
            },
            "Performance": {
                "Vels": {
                    "Tko": 100,
                },
                "RCMax": 1000,
                "Range": 20000,
            },
            "Aero": {
                "L_D": {
                    "Clb": 10,
                    "Crs": 10,
                    "Des": 10,
                },
                "W_S": {
                    "SLS": 10,
                },
            },
            "Weight": {
                "MTOW": 1000,
                "OEW": 800,
                "Crew": 0,
                "Payload": 0,
                "Fuel": 0,
                "Batt": 200,
            },
            "Power": {
                "SLS": 1000000,
                "SpecEnergy": {
                    "Fuel": 43200000,
                    "Batt": 1000000,
                },
                "Eta": {
                    "EM": 1,
                    "EG": 1,
                    "Propeller": 1,
                },
                "P_W": {
                    "SLS": 1,
                    "EM": 1000000,
                    "EG": 1000000,
                },
                "LamDwn": {
                    "SLS": 0,
                    "Clb": 0,
                    "Crs": 0,
                    "Des": 0,
                },
                "LamUps": {
                    "SLS": 0,
                    "Clb": 0,
                    "Crs": 0,
                    "Des": 0,
                },
                "Battery": {
                    "SerCells": float("nan"),
                    "ParCells": float("nan"),
                },
            },
            "Propulsion": {
                "NumEngines": 1,
                "SLSPower": [1000000, 1000000],
                "SLSThrust": [0, 0],
                "PropArch": {
                    "Type": "E",
                },
            },
        },
        "Mission": {
            "Profile": make_compact_mission(),
        },
    }


def make_compact_mission():
    """Return a compact mission profile that runs quickly in FAST-Python."""

    return {
        "Segs": ["Climb", "Cruise", "Descent"],
        "ID": [1, 1, 1],
        "Target": {
            "Valu": [20000],
            "Type": ["Dist"],
        },
        "AltBeg": [0, 1000, 1000],
        "AltEnd": [1000, 1000, 0],
        "VelBeg": [100, 100, 100],
        "VelEnd": [100, 100, 100],
        "TypeBeg": ["TAS", "TAS", "TAS"],
        "TypeEnd": ["TAS", "TAS", "TAS"],
        "ClbRate": [10, float("nan"), -10],
    }

