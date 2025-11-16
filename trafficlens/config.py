"""
Configuration and constants for TrafficLens.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_COLUMNS = [
    "VehicleType",
    "DetectionTime_O",
    "GantryID_O",
    "DetectionTime_D",
    "GantryID_D",
    "TripLength",
    "TripEnd",
    "TripInformation",
]

TASKS = [
    ("VehicleType", None),
    ("DetectionTime_O", "time"),
    ("GantryID_O", "code"),
    ("DetectionTime_D", "time"),
    ("GantryID_D", "code"),
    ("TripLength", None),
    ("TripEnd", "code"),
    ("TripInformation", "text"),
]



