"""
Spatio-temporal dataset generator for TrafficLens.

Given a tabular traffic dataset, this module can aggregate traffic flow
into a time x node matrix, where:
  - rows are time bins (e.g. every 5 minutes)
  - columns are node IDs (e.g. gantry codes)
  - cell values are traffic counts (flows) in that bin at that node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def generate_spatiotemporal(
    df: pd.DataFrame,
    time_col: str,
    node_col: str,
    freq_min: int,
) -> pd.DataFrame:
    """
    Aggregate a flat table into a time x node flow matrix.

    Parameters
    ----------
    df : DataFrame
        Source data (will not be modified).
    time_col : str
        Name of the time column (string or datetime-like values).
    node_col : str
        Name of the node column (e.g. gantry ID).
    freq_min : int
        Aggregation frequency in minutes (must be > 0).

    Returns
    -------
    DataFrame
        Index: time bins (datetime)
        Columns: sorted unique nodes (str)
        Values: counts (int) = number of records in each bin/node.
    """
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
    df_grouped.index.name = "time_bin"

    return df_grouped


