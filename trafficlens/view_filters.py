from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd



FIELD_KINDS: Dict[str, str] = {
    "DetectionTime_O": "time",
    "DetectionTime_D": "time",
    "VehicleType": "category",
    "TripLength": "numeric",
}


def get_field_kind(col_name: str) -> Optional[str]:
    """Return logical kind for a column."""
    return FIELD_KINDS.get(col_name)


def build_view_hint(
    df: pd.DataFrame, col_name: str
) -> Tuple[str, str, Optional[List[str]]]:
    kind = get_field_kind(col_name)
    if df is None or df.empty or col_name not in df.columns or kind is None:
        return kind or "", f"{col_name}: no data available.", None

    series = df[col_name]

    if kind == "category":
        cats = (
            series.dropna()
            .astype(str)
            .value_counts()
            .sort_values(ascending=False)
            .index.tolist()
        )
        hint = f"{col_name}: {len(cats)} classes available."
        return kind, hint, cats

    if kind == "time":
        ser = pd.to_datetime(series, errors="coerce").dropna()
        if ser.empty:
            return kind, f"{col_name}: no valid datetime values.", None
        vmin, vmax = ser.min(), ser.max()
        hint = f"{col_name} range: {vmin} ~ {vmax}"
        return kind, hint, None

    if kind == "numeric":
        ser = pd.to_numeric(series, errors="coerce").dropna()
        if ser.empty:
            return kind, f"{col_name}: no valid numeric values.", None
        vmin, vmax = ser.min(), ser.max()
        hint = f"{col_name} range: {vmin:.3f} ~ {vmax:.3f}"
        return kind, hint, None

    return kind, f"{col_name}: unsupported type.", None


def apply_view_filter(
    df: pd.DataFrame,
    col_name: str,
    kind: str,
    value: str,
    from_str: str,
    to_str: str,
) -> pd.DataFrame:
    if df is None or df.empty or col_name not in df.columns:
        return df.iloc[0:0].copy()

    series = df[col_name]

    if kind == "category":
        if not value:
            return df.copy()
        mask = series.astype(str) == value
        return df[mask].copy()

    if kind == "time":
        ser = pd.to_datetime(series, errors="coerce")
        mask = ser.notna()
        if from_str:
            start = pd.to_datetime(from_str, errors="coerce")
            if pd.isna(start):
                raise ValueError("Invalid start datetime.")
            mask &= ser >= start
        if to_str:
            end = pd.to_datetime(to_str, errors="coerce")
            if pd.isna(end):
                raise ValueError("Invalid end datetime.")
            mask &= ser <= end
        return df[mask].copy()

    if kind == "numeric":
        ser = pd.to_numeric(series, errors="coerce")
        mask = ser.notna()
        if from_str:
            try:
                start_val = float(from_str)
            except ValueError:
                raise ValueError("Start value must be numeric.")
            mask &= ser >= start_val
        if to_str:
            try:
                end_val = float(to_str)
            except ValueError:
                raise ValueError("End value must be numeric.")
            mask &= ser <= end_val
        return df[mask].copy()

    return df.copy()


