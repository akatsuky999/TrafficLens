"""
Data statistics utilities for TrafficLens.

This module contains backend logic for computing basic statistics on
traffic data tables. GUI 代码只负责调用这些函数并展示结果。
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .config import DEFAULT_COLUMNS


def overview_stats(df: pd.DataFrame) -> Dict[str, int]:
    """
    Return high-level statistics for the given dataframe.

    Keys:
    - total_rows
    - total_cols
    - distinct_vehicle_types
    """
    if df is None or df.empty:
        return {"total_rows": 0, "total_cols": 0, "distinct_vehicle_types": 0}

    total_rows = int(len(df))
    total_cols = int(len(df.columns))
    distinct_vehicle = int(
        df["VehicleType"].nunique(dropna=True) if "VehicleType" in df.columns else 0
    )
    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "distinct_vehicle_types": distinct_vehicle,
    }


def vehicle_type_counts(df: pd.DataFrame) -> pd.Series:
    """
    Return frequency counts of VehicleType.

    Index: vehicle type; Values: counts
    """
    if df is None or df.empty or "VehicleType" not in df.columns:
        return pd.Series([], dtype="int64")

    return df["VehicleType"].value_counts(dropna=False).sort_index()


def trip_length_stats(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Compute basic statistics for TripLength.

    Returns dict with keys: count, min, mean, max
    or None if no valid numeric values are present.
    """
    if df is None or df.empty or "TripLength" not in df.columns:
        return None

    ser = pd.to_numeric(df["TripLength"], errors="coerce").dropna()
    if ser.empty:
        return None

    return {
        "count": float(ser.count()),
        "min": float(ser.min()),
        "mean": float(ser.mean()),
        "max": float(ser.max()),
    }


