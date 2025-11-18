from __future__ import annotations

import numpy as np
import pandas as pd


def generate_spatiotemporal(
    df: pd.DataFrame,
    time_col: str,
    node_col: str,
    freq_min: int,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    if time_col not in df.columns or node_col not in df.columns:
        raise ValueError("time_col or node_col not found in dataframe.")
    if freq_min <= 0:
        raise ValueError("freq_min must be > 0.")

    time_series = pd.to_datetime(df[time_col], errors="coerce")
    if time_series.isna().all():
        raise ValueError(f"Column {time_col} cannot be parsed as datetime.")

    valid_mask = time_series.notna()
    df_valid = df.loc[valid_mask].copy()
    time_series = time_series.loc[valid_mask]

    freq = f"{int(freq_min)}min"
    bins = time_series.dt.floor(freq)

    df_grouped = (
        df_valid.groupby([bins, df_valid[node_col].astype(str)])
        .size()
        .unstack(fill_value=0)
    )

    df_grouped = df_grouped.sort_index()
    df_grouped = df_grouped.reindex(sorted(df_grouped.columns), axis=1)

    if time_col == "DetectionTime_D" and "DetectionTime_O" in df.columns:
        time_o = pd.to_datetime(df["DetectionTime_O"], errors="coerce")
        mask_o = time_o.notna()
        if mask_o.any():
            bins_o = time_o.loc[mask_o].dt.floor(freq)
            unique_o = pd.Index(sorted(bins_o.unique()))
            n_o = len(unique_o)
            if n_o and len(df_grouped) > n_o:
                df_grouped = df_grouped.iloc[:n_o]

    df_grouped.index.name = "time_bin"

    return df_grouped


def generate_spatiotemporal_from_tripinfo(
    df: pd.DataFrame,
    trip_col: str,
    freq_min: int,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    if trip_col not in df.columns:
        raise ValueError("trip_col not found in dataframe.")
    if freq_min <= 0:
        raise ValueError("freq_min must be > 0.")

    times = []
    nodes = []
    append_time = times.append
    append_node = nodes.append

    for value in df[trip_col]:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        parts = text.split(";")
        for p in parts:
            segment = p.strip()
            if not segment:
                continue
            if "+" not in segment:
                continue
            time_str, node_str = segment.split("+", 1)
            time_str = time_str.strip()
            node_str = node_str.strip()
            if not time_str or not node_str:
                continue
            append_time(time_str)
            append_node(node_str)

    if not times:
        raise ValueError("No valid trajectory points found in TripInformation.")

    tmp = pd.DataFrame({"time": times, "node": nodes})
    tmp["time"] = pd.to_datetime(tmp["time"], errors="coerce")
    tmp = tmp.dropna(subset=["time"])
    if tmp.empty:
        raise ValueError("All trajectory times are invalid or cannot be parsed.")

    st_df = generate_spatiotemporal(tmp, "time", "node", freq_min)

    if "DetectionTime_O" in df.columns:
        time_o = pd.to_datetime(df["DetectionTime_O"], errors="coerce")
        mask_o = time_o.notna()
        if mask_o.any():
            freq = f"{int(freq_min)}min"
            bins_o = time_o.loc[mask_o].dt.floor(freq)
            unique_o = pd.Index(sorted(bins_o.unique()))
            n_o = len(unique_o)
            if n_o and len(st_df) > n_o:
                st_df = st_df.iloc[:n_o]

    return st_df


