"""
Data statistics utilities for TrafficLens.

This module contains backend logic for computing statistics and creating
publication-quality figures. The GUI is responsible only for calling
these functions and embedding the returned matplotlib Figure.
"""

from __future__ import annotations

from typing import Dict, Optional

from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Global plotting style (journal-like)
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)

# Try to use a custom font from the project Fonts directory (if available)
try:
    project_root = Path(__file__).resolve().parent.parent
    fonts_dir = project_root / "Fonts"
    custom_font: Optional[Path] = None

    # Prefer JetBrainsMono-Regular if present, otherwise first TTF
    preferred = fonts_dir / "JetBrainsMono-Regular.ttf"
    if preferred.exists():
        custom_font = preferred
    else:
        for p in fonts_dir.glob("*.ttf"):
            custom_font = p
            break

    if custom_font is not None:
        font_manager.fontManager.addfont(str(custom_font))
        prop = font_manager.FontProperties(fname=str(custom_font))
        plt.rcParams["font.family"] = prop.get_name()
except Exception:
    # Fallback to default matplotlib font if anything goes wrong
    pass


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


# ---------------------------------------------------------------------------
# Categorical: VehicleType
# ---------------------------------------------------------------------------

def vehicle_type_pie(df: pd.DataFrame) -> Dict[str, object]:
    """
    Build a pie chart for VehicleType and return statistics + Figure.

    Returns dict with:
    - stats: { total, n_unique, top_counts }
    - figure: matplotlib Figure
    """
    if df is None or df.empty or "VehicleType" not in df.columns:
        raise ValueError("VehicleType column not found or dataframe is empty.")

    counts = df["VehicleType"].value_counts(dropna=False).sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No valid data in VehicleType column.")

    # Pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    counts.plot(
        kind="pie",
        ax=ax,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        pctdistance=0.8,
    )
    ax.set_ylabel("")
    ax.set_title("VehicleType distribution")

    # Stats box
    total = int(counts.sum())
    n_unique = int(counts.size)
    top_counts = counts.head(5).to_dict()
    text_lines = [f"Total: {total}", f"Unique: {n_unique}", "Top classes:"]
    text_lines += [f"{k}: {v}" for k, v in top_counts.items()]
    ax.text(
        1.05,
        0.5,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="center",
        fontsize=9,
    )

    fig.tight_layout()

    return {
        "stats": {
            "total": total,
            "n_unique": n_unique,
            "top_counts": top_counts,
        },
        "figure": fig,
    }


# ---------------------------------------------------------------------------
# Time-like: DetectionTime_O / DetectionTime_D
# ---------------------------------------------------------------------------

def time_hist_5min(df: pd.DataFrame, col: str) -> Dict[str, object]:
    """
    Group a time-like column in 5-minute intervals and plot a bar chart.

    Returns dict with:
    - stats: { count, start, end }
    - figure: matplotlib Figure
    """
    if df is None or df.empty or col not in df.columns:
        raise ValueError(f"Column {col} not found or dataframe is empty.")

    dt = pd.to_datetime(df[col], errors="coerce")
    dt = dt.dropna()
    if dt.empty:
        raise ValueError(f"No valid datetime values in column {col}.")

    # Group by 5-minute intervals
    bins = dt.dt.floor("5min")
    counts = bins.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype("datetime64[ns]"), counts.values, width=4 / (24 * 60))
    ax.set_title(f"{col} frequency per 5-minute interval")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    fig.autofmt_xdate()
    # Stats box
    stats = {
        "count": int(len(dt)),
        "start": dt.min(),
        "end": dt.max(),
    }
    text_lines = [
        f"Count: {stats['count']}",
        f"Start: {stats['start']}",
        f"End:   {stats['end']}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    return {
        "stats": stats,
        "figure": fig,
    }


# ---------------------------------------------------------------------------
# Numeric: TripLength
# ---------------------------------------------------------------------------

def trip_length_hist(df: pd.DataFrame, bins: int = 30) -> Optional[Dict[str, object]]:
    """
    Plot a histogram for TripLength and return statistics + Figure.

    Returns dict with:
    - stats: { count, min, mean, std, max }
    - figure: matplotlib Figure
    or None if no valid numeric values are present.
    """
    if df is None or df.empty or "TripLength" not in df.columns:
        return None

    ser = pd.to_numeric(df["TripLength"], errors="coerce").dropna()
    if ser.empty:
        return None

    # finer bins for higher frequency resolution
    bins = max(bins, 60)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ser.values, bins=bins, edgecolor="black", linewidth=0.5)
    ax.set_title("TripLength distribution")
    ax.set_xlabel("TripLength")
    ax.set_ylabel("Frequency")

    stats = {
        "count": float(ser.count()),
        "min": float(ser.min()),
        "mean": float(ser.mean()),
        "std": float(ser.std(ddof=1)) if ser.count() > 1 else 0.0,
        "max": float(ser.max()),
    }

    text_lines = [
        f"n = {stats['count']:.0f}",
        f"min = {stats['min']:.2f}",
        f"mean = {stats['mean']:.2f}",
        f"std = {stats['std']:.2f}",
        f"max = {stats['max']:.2f}",
    ]
    ax.text(
        0.98,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    return {
        "stats": stats,
        "figure": fig,
    }


# ---------------------------------------------------------------------------
# Code-like: GantryID_O / GantryID_D
# ---------------------------------------------------------------------------

def gantry_code_bar(df: pd.DataFrame, col: str, top_n: int = 30) -> Dict[str, object]:
    """
    Plot a bar chart for gantry codes (code-like features).

    x-axis: unique labels; y-axis: frequency.
    Only top_n most frequent labels are shown for readability.

    Returns dict with:
    - stats: { n_unique, top_counts }
    - figure: matplotlib Figure
    """
    if df is None or df.empty or col not in df.columns:
        raise ValueError(f"Column {col} not found or dataframe is empty.")

    counts = df[col].astype(str).value_counts(dropna=False)
    if counts.empty:
        raise ValueError(f"No valid data in column {col}.")

    top_counts = counts.head(top_n)

    # Make the figure slightly smaller so it fits comfortably in the GUI
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(top_counts.index, top_counts.values, edgecolor="black", linewidth=0.6)
    ax.set_title(f"{col} frequency (top {top_n})")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=60)

    stats = {
        "n_unique": int(counts.size),
        "top_counts": top_counts.to_dict(),
    }

    text_lines = [
        f"Unique codes: {stats['n_unique']}",
        f"Top shown: {len(top_counts)}",
    ]
    ax.text(
        0.98,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()

    return {
        "stats": stats,
        "figure": fig,
    }



